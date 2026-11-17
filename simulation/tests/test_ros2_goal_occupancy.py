# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Goal occupancy and departure priority retain actual arrival and body clearance."""

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from pedstrain_simulation.simulation.ros2_navigation.goal_occupancy import GoalOccupancyCoordinator
from pedstrain_simulation.simulation.ros2_navigation.planner import (
    OccupancyGrid,
    convex_footprint,
    segment_polygon_clear,
)
from pedstrain_simulation.simulation.ros2_navigation.solver import NavigationEngine


def _floor():
    return OccupancyGrid(np.zeros((120, 160), dtype=np.int8), 0.05, (-4.0, -3.0))


def _body(name, x, y=0.0):
    return {
        "name": name,
        "position": [x, y, 0.0],
        "velocity": [0.0, 0.0, 0.0],
        "yaw": 0.0,
        "radius_m": 0.25,
        "height_m": 1.8,
    }


def _intention(target, *, phase="walking", next_target=None, context=(0, 0)):
    return {
        "target": list(target),
        "active": phase == "walking",
        "phase": phase,
        "next_target": next_target,
        "route_context": context,
        "path": [],
        "speed_mps": 0.9,
        "arrival_tolerance_m": 0.2,
    }


def _pair(*, goal=(0.0, 0.0, 0.0), next_target=(-2.0, 0.0, 0.0), phase="dwelling"):
    actors = [_body("holder", 0.0), _body("visitor", 1.8)]
    intentions = {
        "holder": _intention((0.0, 0.0, 0.0), phase=phase, next_target=next_target),
        "visitor": _intention(goal),
    }
    return actors, intentions


def test_a_new_peer_blocks_the_remaining_exit_without_sending_the_owner_to_passed_vertices():
    actors = [_body("holder", 1.5), _body("visitor", 3.0, -0.9), _body("new_peer", 4.0, -2.0)]
    intentions = {
        "holder": _intention((1.5, 0.0, 0.0), phase="dwelling", next_target=(3.5, 0.0, 0.0)),
        "visitor": _intention((1.5, 0.0, 0.0)),
        "new_peer": _intention((4.0, -2.0, 0.0), phase="waiting"),
    }
    intentions["holder"]["next_path"] = [[1.5, 0.0], [-1.0, 1.0], [0.0, 2.0], [3.5, 0.0]]
    original = copy.deepcopy(intentions)
    coordinator = GoalOccupancyCoordinator(_floor(), {"map_clearance_m": 0.03, "pair_clearance_m": 0.25})
    coordinator.update(0.0, actors, intentions)
    intentions["holder"].update(phase="walking", active=True, target=[3.5, 0.0, 0.0], route_context=(1, 0))
    first = coordinator.update(1.0, actors, intentions)
    assert first["objectives"]["holder"]["target"] == [3.5, 0.0]
    actors[0]["position"] = [1.515, 0.0, 0.0]
    actors[2]["position"] = [2.5, 0.0, 0.0]
    result = coordinator.update(2.0, actors, intentions)
    directive = result["objectives"].get("holder")
    if directive is not None:
        assert directive["target"][0] >= actors[0]["position"][0]
    actors[2]["position"] = [2.5, -2.0, 0.0]
    resumed = coordinator.update(3.0, actors, intentions)["objectives"]["holder"]
    assert resumed["phase"] == "goal_egress" and resumed["target"][0] > actors[0]["position"][0]
    assert intentions["visitor"] == original["visitor"]


def test_blocked_egress_chord_replans_to_the_same_exit_before_releasing_waiter():
    actors = [_body("holder", 0.0), _body("visitor", -1.8), _body("third", 0.575, 1.5)]
    intentions = {
        "holder": _intention((0.0, 0.0, 0.0), phase="dwelling", next_target=(3.0, 0.0, 0.0)),
        "visitor": _intention((0.0, 0.0, 0.0)),
        "third": _intention((3.0, 0.0, 0.0), phase="waiting"),
    }
    coordinator = GoalOccupancyCoordinator(_floor(), {"map_clearance_m": 0.03, "pair_clearance_m": 0.25})
    coordinator.update(0.0, actors, intentions)
    intentions["holder"] = _intention((3.0, 0.0, 0.0), context=(1, 0))
    original = copy.deepcopy(intentions)
    departure = coordinator.update(1.0, actors, intentions)["objectives"]["holder"]
    exit_point = departure["target"]
    assert departure["phase"] == "goal_egress" and exit_point[1] == 0.0

    actors[2]["position"][1] = 0.58
    blocked = coordinator.update(2.0, actors, intentions)["objectives"]
    assert blocked["visitor"]["phase"] == "goal_wait"
    assert blocked["holder"]["phase"] == "goal_egress"
    assert blocked["holder"]["target"][0] > 0.0 and blocked["holder"]["target"][1] < 0.0
    assert blocked["holder"]["certified_ray"] is True

    actors[0]["position"] = [*exit_point, 0.0]
    released = coordinator.update(3.0, actors, intentions)
    assert all(hold["name"] != "holder" for hold in released["status"]["holders"])
    assert released["objectives"].get("visitor", {}).get("phase") != "goal_wait"
    assert intentions == original


@pytest.mark.parametrize(
    ("cache", "visitor_phase"),
    [("disjoint", "goal_approach"), ("crossing", "goal_wait"), ("stale", "goal_wait")],
)
def test_cached_departure_route_preserves_the_blocked_exit_reservation(cache, visitor_phase):
    next_target = (3.0, 0.0, 0.0)
    actors, intentions = _pair(next_target=next_target)
    holder, visitor = actors
    visitor["position"] = [0.0, 0.9, 0.0]
    blocker = _body("blocker", 4.0, -2.0)
    blocker_origin = blocker["position"]
    actors.append(blocker)
    intentions["blocker"] = _intention(blocker_origin, phase="waiting")
    settings = {"map_clearance_m": 0.03, "pair_clearance_m": 0.25}
    grid = _floor()
    coordinator = GoalOccupancyCoordinator(grid, settings)
    assert coordinator.update(0.0, actors, intentions)["objectives"]["visitor"]["phase"] == "goal_wait"

    intentions["holder"] = _intention(next_target, context=(1, 0))
    departure = coordinator.update(1.0, actors, intentions)
    original_egress = departure["objectives"]["holder"]
    assert original_egress["phase"] == "goal_egress"

    # A stationary peer occupies the exact reserved endpoint after the owner sidesteps.
    blocker["position"] = [*original_egress["target"], 0.0]
    holder["position"] = [0.5, -1.2, 0.0]
    path = [holder["position"][:2], [2.0, -1.3], list(next_target[:2])]
    if cache == "crossing":
        path[1:1] = [[0.0, -0.4], [-0.85, -1.3]]
    if cache == "stale":
        path[-1][0] += 2 * intentions["holder"]["arrival_tolerance_m"]
    intentions["holder"]["path"] = path
    original = copy.deepcopy(intentions)

    fallback = coordinator.update(2.0, actors, intentions)
    assert fallback["status"]["holders"] == departure["status"]["holders"]
    assert fallback["status"]["queue"][0]["holder"] == holder["name"]
    assert fallback["objectives"]["visitor"]["phase"] == visitor_phase
    if cache == "stale":
        assert "holder" not in fallback["objectives"]
    else:
        guide = fallback["objectives"]["holder"]
        assert guide["phase"] == "goal_egress" and not guide["certified_ray"]
        assert guide["target"] != original_egress["target"]
        start = np.asarray(holder["position"][:2])
        end = np.asarray(guide["target"])
        assert grid.segment_clear(start, end, holder["radius_m"] + settings["map_clearance_m"])
        chord = end - start
        for peer in (visitor, blocker):
            relative = start - peer["position"][:2]
            closest = np.clip(-np.dot(relative, chord) / np.dot(chord, chord), 0, 1)
            required = holder["radius_m"] + peer["radius_m"] + settings["pair_clearance_m"]
            assert np.linalg.norm(relative + closest * chord) >= required
    assert intentions == original

    blocker["position"] = blocker_origin
    restored = coordinator.update(3.0, actors, intentions)
    assert restored["objectives"]["holder"] == original_egress
    assert restored["status"]["holders"] == departure["status"]["holders"]
    holder["position"] = [*original_egress["target"], 0.0]
    released = coordinator.update(4.0, actors, intentions)
    assert released["status"]["holders"] == []
    assert released["objectives"]["visitor"]["phase"] == "goal_approach"
    assert intentions == original


@pytest.mark.parametrize(("lateral_m", "certified"), [(-0.00005, False), (0.0, True), (0.00005, True)])
def test_goal_egress_heading_lock_requires_the_measured_gap(monkeypatch, lateral_m, certified):
    actors, intentions = _pair()
    coordinator = GoalOccupancyCoordinator(_floor(), {"map_clearance_m": 0.03, "pair_clearance_m": 0.25})
    coordinator.update(0.0, actors, intentions)
    intentions["holder"] = _intention((-2.0, 0.0, 0.0), context=(1, 0))
    coordinator.update(1.0, actors, intentions)

    actors[0]["position"] = [1.8, 0.75, 0.0]
    point = [2.1, 0.75 + lateral_m]
    monkeypatch.setattr(coordinator, "_egress_guide", lambda *args: point)
    objectives = coordinator.update(2.0, actors, intentions)["objectives"]
    assert objectives["visitor"]["phase"] == "goal_wait"
    assert objectives["holder"]["phase"] == "goal_egress"
    assert objectives["holder"]["target"] == point
    assert objectives["holder"]["certified_ray"] is certified


def test_nearby_but_compatible_goals_keep_normal_rvo_guidance_and_do_not_move_the_holder():
    actors, intentions = _pair(goal=(0.95, 0.0, 0.0))
    original = copy.deepcopy((actors, intentions))
    coordinator = GoalOccupancyCoordinator(_floor(), {"map_clearance_m": 0.03, "pair_clearance_m": 0.25})
    result = coordinator.update(0.0, actors, intentions)
    assert result["objectives"] == {} and result["status"]["queue"] == []
    assert (actors, intentions) == original


def test_static_goal_projection_does_not_trigger_unrelated_goal_occupancy_guidance():
    cells = np.zeros((120, 160), dtype=np.int8)
    cells[66:, :94] = 100
    grid = OccupancyGrid(cells, 0.05, (-4.0, -3.0))
    actors, intentions = _pair(goal=(0.9, 0.6, 0.0))
    actors[1]["position"][1] = 0.6
    intentions["visitor"]["path"] = [[2.0, 0.6], [1.0, 0.6]]
    coordinator = GoalOccupancyCoordinator(grid, {"map_clearance_m": 0.03, "pair_clearance_m": 0.25})
    result = coordinator.update(0.0, actors, intentions)
    assert result["objectives"] == {} and result["status"]["queue"] == []


@pytest.mark.parametrize("goal_x", [0.6])
def test_original_arrival_ball_keeps_its_legal_tangent_or_far_side_region(goal_x):
    actors, intentions = _pair(goal=(goal_x, 0.0, 0.0), next_target=None, phase="complete")
    coordinator = GoalOccupancyCoordinator(_floor(), {"map_clearance_m": 0.03, "pair_clearance_m": 0.25})
    result = coordinator.update(0.0, actors, intentions)
    assert result["status"]["queue"] == []
    target = result["objectives"]["visitor"]["target"]
    assert math.dist(target, actors[0]["position"][:2]) >= 0.75 - 1e-12
    assert math.dist(target, intentions["visitor"]["target"][:2]) <= 0.2 + 1e-12
    assert "holder" not in result["objectives"]


def test_permanently_occupied_goal_is_reported_without_moving_the_arrived_person_or_faking_arrival():
    actors, intentions = _pair(next_target=None, phase="complete")
    coordinator = GoalOccupancyCoordinator(_floor(), {"map_clearance_m": 0.03, "pair_clearance_m": 0.25})
    for time_s in (0.0, 2.0, 20.0):
        result = coordinator.update(time_s, actors, intentions)
        assert result["objectives"]["visitor"]["target"] is None
        assert result["objectives"]["visitor"]["phase"] == "goal_wait"
        assert "holder" not in result["objectives"]
        assert result["status"]["queue"][0]["reason"] == "permanent_goal_occupancy"
    assert actors[0]["position"] == [0.0, 0.0, 0.0]
    assert intentions["visitor"]["target"] == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("cached_path", [None, [[0.0, 0.0]]])
def test_departure_priority_survives_the_original_route_index_change_until_measured_exit(cached_path):
    actors, intentions = _pair()
    intentions["holder"]["next_path"] = cached_path
    coordinator = GoalOccupancyCoordinator(_floor(), {"map_clearance_m": 0.03, "pair_clearance_m": 0.25})
    waiting = coordinator.update(0.0, actors, intentions)
    assert waiting["objectives"]["visitor"]["phase"] == "goal_wait"
    intentions["holder"] = _intention((-2.0, 0.0, 0.0), context=(1, 0))
    departing = coordinator.update(1.0, actors, intentions)
    assert departing["objectives"]["visitor"]["phase"] == "goal_wait"
    assert departing["objectives"]["holder"]["phase"] == "goal_egress"
    actors[0]["position"][0] = -1.2
    released = coordinator.update(2.0, actors, intentions)
    assert all(hold["name"] != "holder" for hold in released["status"]["holders"])
    assert released["objectives"].get("visitor", {}).get("phase") != "goal_wait"


@pytest.mark.parametrize(
    ("next_target", "measured_pose", "door_safety", "released"),
    [
        ((-2.0, 0.0, 0.0), (-1.98, 0.1, 0.0), False, True),
        ((-2.0, 0.0, 0.0), (-1.98, 0.1, 0.21), False, False),
        ((-0.95, 0.0, 0.0), (-0.90, 0.1, 0.0), False, False),
        ((-2.0, 0.0, 0.0), (-1.98, 0.1, 0.0), True, False),
    ],
    ids=("actual_arrival", "outside_3d_tolerance", "near_old_goal", "door_safety_owned"),
)
def test_an_off_line_arrival_releases_only_a_cleared_departure_reservation(
    next_target, measured_pose, door_safety, released
):
    actors, intentions = _pair(next_target=next_target)
    coordinator = GoalOccupancyCoordinator(_floor(), {"map_clearance_m": 0.03, "pair_clearance_m": 0.25})
    waiting = coordinator.update(0.0, actors, intentions)
    assert waiting["objectives"]["visitor"]["phase"] == "goal_wait"
    intentions["holder"] = _intention(next_target, context=(1, 0))
    departing = coordinator.update(1.0, actors, intentions)
    assert departing["objectives"]["holder"]["phase"] == "goal_egress"

    # A door or other live obstacle can guide the owner to the original ball
    # without visiting the exact reserved exit line.
    actors[0]["position"] = list(measured_pose)
    priority = {"holder": {"phase": "door_close_return", "target": list(next_target[:2])}} if door_safety else {}
    original = copy.deepcopy((actors, intentions))
    result = coordinator.update(2.0, actors, intentions, priority_objectives=priority)
    assert ("holder" not in {item["name"] for item in result["status"]["holders"]}) == released
    if released:
        assert "holder" not in result["objectives"]
        assert result["objectives"]["visitor"]["phase"] == "goal_approach"
    assert (actors, intentions) == original


def test_single_goal_waiter_recovers_before_exit_release_when_approach_is_disjoint():
    actors, intentions = _pair(next_target=(-3.0, 0.0, 0.0))
    actors[1]["position"] = [0.0, 0.9, 0.0]
    coordinator = GoalOccupancyCoordinator(_floor(), {"map_clearance_m": 0.03, "pair_clearance_m": 0.25})
    waiting = coordinator.update(0.0, actors, intentions)
    assert waiting["objectives"]["visitor"]["phase"] == "goal_wait"
    assert "holder" not in waiting["objectives"]

    intentions["holder"] = _intention((-3.0, 0.0, 0.0), context=(1, 0))
    original = copy.deepcopy(intentions)
    first = coordinator.update(1.0, actors, intentions)["objectives"]
    assert first["visitor"]["phase"] == "goal_wait"
    assert first["holder"]["phase"] == "goal_egress"
    actors[0]["position"][0] = -0.9
    actors[0]["velocity"][0] = -0.9
    resumed = coordinator.update(2.0, actors, intentions)
    assert resumed["status"]["holders"][0]["phase"] == "departing"
    assert resumed["objectives"]["holder"] == first["holder"]
    assert resumed["objectives"]["visitor"]["phase"] == "goal_approach"
    assert resumed["objectives"]["visitor"]["target"] == [0.0, 0.0]
    assert intentions == original

    # A transient third body blocks the approach without obstructing the
    # certified exit. Keep the holder's guide and reuse the route once clear.
    actors.append(_body("blocker", 0.0, 0.12))
    intentions["blocker"] = _intention((0.0, 0.12, 0.0), phase="waiting")
    blocked = coordinator.update(2.1, actors, intentions)["objectives"]
    assert blocked["visitor"]["phase"] == "goal_wait"
    assert blocked["visitor"]["target"] is None
    assert blocked["holder"] == first["holder"]
    actors[-1]["position"][1] = 2.5
    clear = coordinator.update(2.2, actors, intentions)["objectives"]
    assert clear["visitor"] == resumed["objectives"]["visitor"]
    assert clear["holder"] == first["holder"]


def test_goal_waiter_advances_one_bounded_segment_before_goal_becomes_reachable():
    actors = [_body("holder", 1.375, 1.273), _body("visitor", 0.0, 1.25)]
    goal = (1.435, 1.313, 0.0)
    exit_goal = (-2.0, 0.4, 0.0)
    intentions = {
        "holder": _intention(actors[0]["position"], phase="dwelling", next_target=exit_goal),
        "visitor": _intention(goal),
    }
    intentions["holder"]["next_path"] = [[1.375, 1.273], [0.24, 0.53], [-2.0, 0.4]]
    settings = {"map_clearance_m": 0.0275, "pair_clearance_m": 0.225, "max_yield_distance_m": 0.4}
    coordinator = GoalOccupancyCoordinator(_floor(), settings)
    assert coordinator.update(0.0, actors, intentions)["objectives"]["visitor"]["phase"] == "goal_wait"

    intentions["holder"] = _intention(exit_goal, context=(1, 0))
    original = copy.deepcopy(intentions)
    departing = coordinator.update(1.0, actors, intentions)
    owner = departing["objectives"]["holder"]
    visitor = departing["objectives"]["visitor"]
    origin = actors[1]["position"][:2]
    assert owner["phase"] == "goal_egress"
    assert visitor["phase"] == "goal_yield" and visitor["certified_ray"]
    assert 0.05 < math.dist(origin, visitor["target"]) <= settings["max_yield_distance_m"]
    assert math.dist(visitor["target"], goal[:2]) < math.dist(origin, goal[:2])
    assert not departing["status"]["queue"][0]["admitted"]

    actors[1]["position"] = [*visitor["target"], 0.0]
    waiting = coordinator.update(2.0, actors, intentions)
    assert waiting["objectives"]["visitor"]["phase"] == "goal_wait"
    assert waiting["objectives"]["holder"] == owner
    assert not waiting["status"]["queue"][0]["admitted"]
    assert intentions == original


@pytest.mark.parametrize("situation", ["standing", "moving", "blocked_prefix"])
def test_interrupted_prefix_can_resume_while_the_departing_owner_is_stationary(situation):
    holder = _body("holder", 1.375, 1.273)
    visitor = _body("visitor", 0.0, 1.25)
    next_target = (-2.0, 0.4, 0.0)
    intentions = {
        "holder": _intention(holder["position"], phase="dwelling", next_target=next_target),
        "visitor": _intention((1.435, 1.313, 0.0)),
    }
    intentions["holder"]["next_path"] = [[1.375, 1.273], [0.24, 0.53], [-2.0, 0.4]]
    settings = {"map_clearance_m": 0.0275, "pair_clearance_m": 0.225, "max_yield_distance_m": 0.4}
    coordinator = GoalOccupancyCoordinator(_floor(), settings)
    actors = [holder, visitor]
    coordinator.update(0.0, actors, intentions)
    intentions["holder"] = _intention(next_target, context=(1, 0))
    departure = coordinator.update(1.0, actors, intentions)
    prefix = departure["objectives"]["visitor"]
    assert prefix["phase"] == "goal_yield"
    origin = np.asarray(visitor["position"][:2])
    target = np.asarray(prefix["target"])
    visitor["position"] = [*(origin + 0.25 * (target - origin)), 0.0]

    blocker = _body("exit_blocker", -0.4, 0.55)
    actors.append(blocker)
    intentions["exit_blocker"] = _intention(blocker["position"], phase="waiting")
    if situation == "moving":
        holder["velocity"] = [-0.09, 0.0, 0.0]
    if situation == "blocked_prefix":
        direction = (target - origin) / np.linalg.norm(target - origin)
        required = 2 * visitor["radius_m"] + settings["pair_clearance_m"]
        x, y = target + (required - 0.10) * direction
        peer = _body("prefix_blocker", x, y)
        assert all(math.dist(peer["position"], actor["position"]) > required for actor in actors)
        actors.append(peer)
        intentions["prefix_blocker"] = _intention(peer["position"], phase="waiting")
    original = copy.deepcopy(intentions)

    blocked = coordinator.update(2.0, actors, intentions)
    assert blocked["status"]["holders"] == departure["status"]["holders"]
    assert blocked["status"]["queue"][0]["holder"] == holder["name"]
    assert not blocked["status"]["queue"][0]["admitted"]
    if situation == "standing":
        assert blocked["objectives"]["visitor"] == prefix
        assert blocked["objectives"]["holder"]["phase"] == "goal_wait"
        assert blocked["objectives"]["holder"]["target"] is None
    else:
        assert blocked["objectives"]["visitor"]["phase"] == "goal_wait"
        assert "holder" not in blocked["objectives"]
    assert intentions == original

    resumed = coordinator.update(3.0, [holder, visitor], intentions)
    assert resumed["objectives"]["holder"] == departure["objectives"]["holder"]
    assert resumed["objectives"]["visitor"] == prefix
    assert resumed["status"]["holders"] == departure["status"]["holders"]
    assert intentions == original


@pytest.mark.parametrize("extra_clearance_m", [0.0, 0.0015])
def test_parallel_goal_waiter_aligns_without_drifting_into_owner_exit(extra_clearance_m):
    cells = np.zeros((100, 300), dtype=np.int8)
    # A low wall corner forces the waiter to use the upper side of the exit.
    cells[32:36, 172:182] = 100
    grid = OccupancyGrid(cells, 0.02, (-4.0, -0.28))
    actors = [_body("holder", 0.0), _body("visitor", -1.15)]
    settings = {"map_clearance_m": 0.0275, "pair_clearance_m": 0.225}
    required_gap = sum(actor["radius_m"] for actor in actors) + settings["pair_clearance_m"]
    actors[1]["position"][1] = required_gap + extra_clearance_m
    plans = [
        {
            "name": actor["name"],
            "position": actor["position"].copy(),
            "behavior": "walk",
            "speed": 0.9,
            "arrival_tolerance": 0.2,
            "waypoints": [{"position": [0.0, 0.0, 0.0], "wait": 0.0}],
        }
        for actor in actors
    ]
    plans[0]["waypoints"].append({"position": [-3.0, 0.0, 0.0], "wait": 0.0})
    engine = NavigationEngine(plans, grid, settings, actors)
    try:
        intentions = {
            "holder": _intention((0.0, 0.0, 0.0), phase="dwelling", next_target=(-3.0, 0.0, 0.0)),
            "visitor": _intention((0.0, 0.0, 0.0)),
        }
        engine._goals.update(0.0, actors, intentions)
        actors[0].update(position=[-0.8, 0.0, 0.0], yaw=math.pi)
        actors[1]["yaw"] = -math.pi / 2
        intentions["holder"] = _intention((-3.0, 0.0, 0.0), context=(1, 0))
        admitted = engine._goals.update(2.0, actors, intentions)["objectives"]
        assert admitted["visitor"]["phase"] == "goal_approach"
        actors[1]["velocity"] = [0.0, -0.51, 0.0]
        engine.routes["holder"].index = 1
        engine.routes["holder"].phase = engine.routes["visitor"].phase = "walking"

        for tick in range(1, 361):
            output = engine.step(2.0 + tick / 60, 1 / 60, actors)["agents"]
            assert all(actor["error"] is None for actor in output)
            assert math.dist(*(actor["position"] for actor in output)) >= required_gap - 1e-8
            if tick == 1:
                assert output[1]["phase"] == "goal_approach"
                assert output[1]["position"] == actors[1]["position"]
            actors = [{**before, **after} for before, after in zip(actors, output)]
        assert all(actor["phase"] == "complete" for actor in output)
    finally:
        engine.close()


def test_a_skipped_forecast_target_invalidates_the_old_exit_certificate():
    actors, intentions = _pair(next_target=(0.0, 0.0, 0.0))
    coordinator = GoalOccupancyCoordinator(_floor(), {"map_clearance_m": 0.03, "pair_clearance_m": 0.25})
    coordinator.update(0.0, actors, intentions)
    intentions["holder"] = _intention((0.0, -2.0, 0.0), context=(2, 0))
    result = coordinator.update(1.0, actors, intentions)
    assert result["objectives"]["holder"]["phase"] == "goal_egress"
    assert result["objectives"]["holder"]["target"][1] < 0
    assert result["status"]["holders"][0]["next_target_m"] == [0.0, -2.0, 0.0]


def test_only_continuous_unowned_admission_blockage_times_out(monkeypatch):
    actors, intentions = _pair()
    coordinator = GoalOccupancyCoordinator(_floor(), {"map_clearance_m": 0.03, "pair_clearance_m": 0.25})
    coordinator.update(0.0, actors, intentions)
    actors[0]["position"][0] = -1.2
    intentions["holder"] = _intention((-2.0, 0.0, 0.0), context=(1, 0))
    monkeypatch.setattr(coordinator, "_plan", lambda *a, **k: None)
    for tick in range(30):
        result = coordinator.update(1.0 + tick / 60, actors, intentions)
        assert result["objectives"]["visitor"]["phase"] == "goal_wait"
    with pytest.raises(RuntimeError, match="admission progress for 12 simulation seconds"):
        coordinator.update(13.1, actors, intentions)


def test_admission_recovery_clears_its_unowned_blocked_timer(monkeypatch):
    actors, intentions = _pair()
    coordinator = GoalOccupancyCoordinator(_floor(), {"map_clearance_m": 0.03, "pair_clearance_m": 0.25})
    coordinator.update(0.0, actors, intentions)
    actors[0]["position"][0] = -1.2
    intentions["holder"] = _intention((-2.0, 0.0, 0.0), context=(1, 0))
    plan = coordinator._plan
    monkeypatch.setattr(coordinator, "_plan", lambda *a, **k: None)
    coordinator.update(1.0, actors, intentions)
    monkeypatch.setattr(coordinator, "_plan", plan)
    assert coordinator.update(1.1, actors, intentions)["objectives"]["visitor"]["phase"] == "goal_wait"
    recovered = coordinator.update(1.6, actors, intentions)
    assert recovered["objectives"]["visitor"]["phase"] == "goal_approach"
    assert recovered["status"]["queue"][0]["admission_blocked_s"] is None
    assert coordinator.update(20.0, actors, intentions)["objectives"]["visitor"]["phase"] == "goal_approach"


def test_a_new_observed_holder_context_does_not_count_as_unowned_admission_blockage():
    actors, intentions = _pair()
    coordinator = GoalOccupancyCoordinator(_floor(), {"map_clearance_m": 0.03, "pair_clearance_m": 0.25})
    coordinator.update(0.0, actors, intentions)
    intentions["holder"] = _intention((0.0, 0.0, 0.0), phase="dwelling", next_target=(-2.0, 0.0, 0.0), context=(1, 0))
    for time_s in (1.0, 15.0, 30.0):
        result = coordinator.update(time_s, actors, intentions)
        assert result["objectives"]["visitor"]["phase"] == "goal_wait"
        assert result["status"]["queue"][0]["admission_blocked_s"] is None


def test_a_geometrically_compatible_inside_goal_still_waits_for_real_door_admission():
    actors, intentions = _pair(goal=(0.6, 0.0, 0.0), next_target=None, phase="complete")
    coordinator = GoalOccupancyCoordinator(_floor(), {"map_clearance_m": 0.03, "pair_clearance_m": 0.25})
    intentions["visitor"]["queued_door_entry"] = True
    waiting = coordinator.update(0.0, actors, intentions)
    assert waiting["objectives"]["visitor"]["phase"] == "goal_wait"
    assert waiting["status"]["queue"][0]["reason"] == "door_admission_queue"
    intentions["visitor"]["queued_door_entry"] = False
    assert coordinator.update(1.0, actors, intentions)["objectives"]["visitor"]["phase"] == "goal_approach"


def test_an_existing_door_or_passing_owner_is_never_overwritten_by_a_goal_queue():
    actors, intentions = _pair()
    coordinator = GoalOccupancyCoordinator(_floor(), {"map_clearance_m": 0.03, "pair_clearance_m": 0.25})
    first = coordinator.update(0.0, actors, intentions)
    priority = {"visitor": {"target": [2.0, 0.2], "phase": "door_yield", "token": ("door",)}}
    suspended = coordinator.update(0.5, actors, intentions, priority_objectives=priority)
    assert "visitor" not in suspended["objectives"]
    assert suspended["status"]["queue"][0]["origin_xy_m"] == first["status"]["queue"][0]["origin_xy_m"]


@pytest.mark.parametrize("business_phase", ["turning", "dwelling", "complete"])
def test_an_existing_holder_keeps_its_original_occupancy_during_explicit_door_safety_motion(business_phase):
    actors, intentions = _pair(phase=business_phase)
    coordinator = GoalOccupancyCoordinator(_floor(), {"map_clearance_m": 0.03, "pair_clearance_m": 0.25})
    first = coordinator.update(0.0, actors, intentions)
    hold = coordinator._holders["holder"]
    saved = (hold.context, hold.origin.copy(), hold.next_target.copy(), hold.started)
    for index, safety_phase in enumerate(("door_close_yield", "door_close_wait", "door_close_return", "door_yield"), 1):
        actors[0]["position"][0] = 0.3
        priority = {
            "holder": {
                "phase": safety_phase,
                "target": [0.35, 0.0] if safety_phase != "door_close_wait" else None,
                "origin": [0.0, 0.0],
                "token": ("door", safety_phase),
            }
        }
        result = coordinator.update(float(index), actors, intentions, priority_objectives=priority)
        assert "holder" not in result["objectives"]
        assert result["objectives"]["visitor"]["phase"] == "goal_wait"
        assert coordinator._holders["holder"] is hold
        assert hold.context == saved[0] and hold.started == saved[3] and not hold.departing
        np.testing.assert_array_equal(hold.origin, saved[1])
        np.testing.assert_array_equal(hold.next_target, saved[2])
        assert result["status"]["holders"][0]["phase"] == "held"
        assert result["status"]["queue"][0]["origin_xy_m"] == first["status"]["queue"][0]["origin_xy_m"]
    actors[0]["position"][0] = 0.0
    returned = coordinator.update(5.0, actors, intentions)
    assert returned["objectives"]["visitor"]["phase"] == "goal_wait"
    assert not hold.departing


@pytest.mark.parametrize("safety_phase", [None, "door_wait", "passing_wait"])
def test_other_priority_labels_do_not_excuse_an_unobserved_arrival(safety_phase):
    actors, intentions = _pair()
    coordinator = GoalOccupancyCoordinator(_floor(), {"map_clearance_m": 0.03, "pair_clearance_m": 0.25})
    coordinator.update(0.0, actors, intentions)
    actors[0]["position"][0] = 0.3
    priority = {} if safety_phase is None else {"holder": {"phase": safety_phase}}
    with pytest.raises(ValueError, match="outside its original arrival tolerance"):
        coordinator.update(1.0, actors, intentions, priority_objectives=priority)


def test_safety_priority_cannot_create_a_new_holder_that_never_arrived():
    actors, intentions = _pair()
    actors[0]["position"][0] = 0.3
    coordinator = GoalOccupancyCoordinator(_floor(), {"map_clearance_m": 0.03, "pair_clearance_m": 0.25})
    with pytest.raises(ValueError, match="outside its original arrival tolerance"):
        coordinator.update(0.0, actors, intentions, priority_objectives={"holder": {"phase": "door_close_yield"}})


def test_a_busy_holders_original_goal_and_return_channel_stay_reserved_for_new_approachers():
    actors, intentions = _pair(goal=(0.2, 0.54, 0.0))
    intentions["visitor"]["active"] = False
    coordinator = GoalOccupancyCoordinator(_floor(), {"map_clearance_m": 0.03, "pair_clearance_m": 0.25})
    coordinator.update(0.0, actors, intentions)
    priority = {"holder": {"phase": "door_close_yield", "target": [0.4, 0.0], "origin": [0.0, 0.0]}}
    coordinator.update(0.1, actors, intentions, priority_objectives=priority)
    actors[0]["position"][0] = 0.4
    intentions["visitor"]["active"] = True
    priority["holder"].update(phase="door_close_wait", target=None)
    waiting = coordinator.update(1.0, actors, intentions, priority_objectives=priority)
    assert waiting["objectives"]["visitor"]["phase"] == "goal_wait"
    assert "holder" not in waiting["objectives"]
    actors[0]["position"][0] = 0.0
    available = coordinator.update(2.0, actors, intentions)
    assert available["objectives"]["visitor"]["phase"] == "goal_approach"
    assert available["status"]["queue"][0]["admitted"]


def _resource_cycle():
    actors, intentions = _pair(next_target=(0.0, -2.0, 0.0))
    actors[0]["position"][1] = 0.18
    intentions["holder"]["queued_door_entry"] = True
    intentions["visitor"]["door_outgoing"] = True
    coordinator = GoalOccupancyCoordinator(_floor(), {"map_clearance_m": 0.03, "pair_clearance_m": 0.25})
    coordinator.update(0.0, actors, intentions)
    return coordinator, actors, intentions


def test_a_door_goal_resource_cycle_moves_only_the_owner_after_its_original_dwell_ended():
    coordinator, actors, intentions = _resource_cycle()
    original_holder = actors[0]["position"].copy()
    dwelling = coordinator.update(0.5, actors, intentions)
    assert "holder" not in dwelling["objectives"]
    assert actors[0]["position"] == original_holder
    intentions["holder"] = _intention((0.0, -2.0, 0.0), context=(1, 0))
    intentions["holder"]["queued_door_entry"] = True
    priority = {"visitor": {"phase": "door_exit", "target": [1.0, 0.0]}}
    yielding = coordinator.update(1.0, actors, intentions, priority_objectives=priority)
    directive = yielding["objectives"]["holder"]
    assert directive["phase"] == "goal_yield" and directive["certified_ray"]
    assert "visitor" not in yielding["objectives"]
    assert math.dist(directive["target"], directive["origin"]) <= 0.4
    actors[0]["position"] = [*directive["target"], 0.0]
    ready = coordinator.update(2.0, actors, intentions)
    assert ready["objectives"]["holder"]["phase"] == "goal_wait"
    assert ready["objectives"]["visitor"]["phase"] == "goal_approach"
    actors[1]["position"] = [*ready["objectives"]["visitor"]["target"], 0.0]
    assert math.dist(actors[1]["position"], [0.0, 0.0, 0.0]) <= 0.2
    intentions["visitor"]["door_outgoing"] = False
    released = coordinator.update(3.0, actors, intentions)
    assert released["objectives"]["holder"]["phase"] == "goal_egress"
    assert coordinator._holders["holder"].departure_yield is None


def test_a_departure_yield_cannot_add_another_point_four_meters_after_a_door_refuge():
    coordinator, actors, intentions = _resource_cycle()
    intentions["holder"] = _intention((0.0, -2.0, 0.0), context=(1, 0))
    intentions["holder"].update(queued_door_entry=True, yield_origin_xy_m=[0.0, 0.0])
    original = copy.deepcopy((actors, intentions))
    with pytest.raises(RuntimeError, match="No certified departure-yield refuge within 0.4 m"):
        coordinator.update(1.0, actors, intentions)
    assert (actors, intentions) == original


def test_waiters_are_admitted_fifo_and_rebind_to_the_actual_new_holder():
    actors, intentions = _pair()
    actors[1]["name"] = "first"
    intentions["first"] = intentions.pop("visitor")
    actors.append(_body("second", 1.5, 1.0))
    intentions["second"] = _intention((0.0, 0.0, 0.0))
    intentions["second"]["active"] = False
    coordinator = GoalOccupancyCoordinator(_floor(), {"map_clearance_m": 0.03, "pair_clearance_m": 0.25})
    coordinator.update(0.0, actors, intentions)
    intentions["second"]["active"] = True
    coordinator.update(0.1, actors, intentions)
    actors[0]["position"][0] = -1.2
    intentions["holder"] = _intention((-2.0, 0.0, 0.0), context=(1, 0))
    released = coordinator.update(1.0, actors, intentions)
    assert released["objectives"]["first"]["phase"] == "goal_approach"
    assert released["objectives"]["second"]["phase"] == "goal_wait"
    actors[1]["position"] = [0.0, 0.0, 0.0]
    intentions["first"] = _intention((0.0, 0.0, 0.0), phase="dwelling", next_target=(0.0, 2.0, 0.0))
    occupied = coordinator.update(2.0, actors, intentions)
    assert occupied["objectives"]["second"]["phase"] == "goal_wait"
    assert occupied["status"]["queue"][0]["holder"] == "first"


def test_the_reserved_exit_is_certified_against_all_waiters_together():
    actors = [_body("holder", 0.0), _body("first", 0.9, 0.4), _body("second", 0.9, -0.4)]
    intentions = {state["name"]: _intention((0.0, 0.0, 0.0)) for state in actors}
    intentions["holder"] = _intention((0.0, 0.0, 0.0), phase="dwelling", next_target=(3.0, 0.0, 0.0))
    coordinator = GoalOccupancyCoordinator(_floor(), {"map_clearance_m": 0.03, "pair_clearance_m": 0.25})
    result = coordinator.update(0.0, actors, intentions)
    path = coordinator._holders["holder"].exit_path
    for state in actors[1:]:
        directive = result["objectives"][state["name"]]
        point = directive["target"] if directive["phase"] == "goal_yield" else state["position"][:2]
        for a, b in zip(path, path[1:]):
            assert coordinator._clear("holder", a, b, {s["name"]: s for s in actors}, (), [(np.asarray(point), 0.75)])


def _bay():
    cells = np.full((100, 160), 100, dtype=np.int8)
    cells[20:44, :] = 0
    cells[10:60, 96:] = cells[10:60, :32] = 0
    return OccupancyGrid(cells, 0.05, (-4.0, -2.5))


def test_only_the_approacher_yields_within_a_fixed_point_four_meter_origin_when_it_blocks_the_exit():
    actors = [_body("holder", 0.4, -0.65), _body("visitor", 1.1, -1.0)]
    intentions = {
        "holder": _intention((0.4, -0.65, 0.0), phase="dwelling", next_target=[2.5, -0.9, 0.0]),
        "visitor": _intention((0.4, -0.65, 0.0)),
    }
    original = copy.deepcopy((actors, intentions))
    coordinator = GoalOccupancyCoordinator(_bay(), {"map_clearance_m": 0.03, "pair_clearance_m": 0.25})
    result = coordinator.update(0.0, actors, intentions)
    directive = result["objectives"]["visitor"]
    assert directive["phase"] == "goal_yield" and directive["certified_ray"]
    assert math.dist(directive["target"], directive["origin"]) <= 0.4
    assert "holder" not in result["objectives"]
    repeated = coordinator.update(0.1, actors, intentions)
    assert repeated["objectives"]["visitor"]["origin"] == directive["origin"]
    assert repeated["objectives"]["visitor"]["target"] == directive["target"]
    assert (actors, intentions) == original


@pytest.mark.parametrize("scheduled_departure", [None, "deadline", "no_owner_refuge"])
def test_no_room_for_a_bounded_refuge_fails_explicitly_without_displacing_either_person(scheduled_departure):
    cells = np.full((80, 120), 100, dtype=np.int8)
    cells[32:48, :] = 0
    grid = OccupancyGrid(cells, 0.05, (-2.0, -2.0))
    actors, intentions = _pair(next_target=(2.0, 0.0, 0.0))
    actors[1]["position"][0] = 0.9
    original = copy.deepcopy((actors, intentions))
    coordinator = GoalOccupancyCoordinator(grid, {"map_clearance_m": 0.03, "pair_clearance_m": 0.25})
    if scheduled_departure is None:
        with pytest.raises(RuntimeError, match="No certified goal waiting refuge within 0.4 m"):
            coordinator.update(0.0, actors, intentions)
        assert (actors, intentions) == original
        return
    intentions["holder"]["scheduled_departure_s"] = 2.0
    waiting = coordinator.update(0.0, actors, intentions)
    assert waiting["objectives"]["visitor"]["phase"] == "goal_wait"
    assert "holder" not in waiting["objectives"]
    queue = waiting["status"]["queue"][0]
    assert queue["reason"] == "awaiting_holder_departure"
    assert queue["origin_xy_m"] == actors[1]["position"][:2]
    assert actors == original[0]
    if scheduled_departure == "deadline":
        intentions["holder"]["scheduled_departure_s"] = 1000.0
        repeated = coordinator.update(1.0, actors, intentions)
        assert repeated["status"]["queue"][0]["waiting_deadline_s"] == queue["waiting_deadline_s"]
        with pytest.raises(RuntimeError, match="Scheduled goal departure waiting timed out"):
            coordinator.update(queue["waiting_deadline_s"] + 1e-3, actors, intentions)
    else:
        intentions["holder"] = _intention((2.0, 0.0, 0.0), context=(1, 0))
        with pytest.raises(RuntimeError, match="No certified departure-yield refuge within 0.4 m"):
            coordinator.update(2.0, actors, intentions)
    assert actors == original[0]


@pytest.mark.parametrize("boundary", ["no_renew", "no_witness", "safety_priority", "unsafe", "no_next"])
def test_an_already_departing_owner_uses_one_bounded_recovery_window_or_fails_explicitly(boundary):
    cells = np.full((80, 120), 100, dtype=np.int8)
    cells[32:48, :] = 0
    grid = OccupancyGrid(cells, 0.05, (-2.0, -2.0))
    actors, intentions = _pair(next_target=(2.0, 0.0, 0.0))
    actors[1]["position"][0] = 0.9
    coordinator = GoalOccupancyCoordinator(grid, {"map_clearance_m": 0.03, "pair_clearance_m": 0.25})
    if boundary == "no_next":
        intentions["holder"] = _intention((0.0, 0.0, 0.0), phase="complete")
        result = coordinator.update(0.0, actors, intentions)
        assert result["objectives"]["visitor"]["phase"] == "goal_wait"
        assert result["status"]["queue"][0]["permanent"] is True
        assert result["status"]["queue"][0]["scheduled_departure_s"] is None
        return
    coordinator.update(-1.0, actors[:1], {"holder": intentions["holder"]})
    intentions["holder"] = _intention((2.0, 0.0, 0.0), context=(1, 0))
    intentions["holder"]["scheduled_departure_s"] = 1000.0
    priority = {"holder": {"phase": "door_close_wait", "target": None, "origin": [0.0, 0.0]}}
    if boundary == "safety_priority":
        with pytest.raises(RuntimeError, match="No certified goal waiting refuge within 0.4 m"):
            coordinator.update(0.0, actors, intentions, priority_objectives=priority)
        return
    if boundary == "unsafe":
        obstacles = NavigationEngine._dynamic_obstacles(
            [{"position": actors[1]["position"], "velocity": [0.0, 0.0, 0.0], "velocity_valid": True, "radius_m": 0.1}]
        )
        with pytest.raises(RuntimeError, match="No certified"):
            coordinator.update(0.0, actors, intentions, obstacles)
        return
    first = coordinator.update(0.0, actors, intentions)
    queue = first["status"]["queue"][0]
    assert first["objectives"]["visitor"]["phase"] == "goal_wait"
    assert queue["scheduled_departure_s"] == 0.0
    assert queue["origin_xy_m"] == actors[1]["position"][:2]
    if boundary == "no_witness":
        with pytest.raises(RuntimeError, match="No certified departure-yield refuge within 0.4 m"):
            coordinator.update(0.1, actors, intentions)
    else:
        intentions["holder"]["scheduled_departure_s"] = 2000.0
        owned = {"holder": {"phase": "passing_owned"}}
        repeated = coordinator.update(1.0, actors, intentions, priority_objectives=owned)
        assert repeated["status"]["queue"][0]["waiting_deadline_s"] == queue["waiting_deadline_s"]
        with pytest.raises(RuntimeError, match="Scheduled goal departure waiting timed out"):
            coordinator.update(queue["waiting_deadline_s"] + 1e-3, actors, intentions, priority_objectives=owned)


@pytest.mark.parametrize("coordinated", [False, True])
def test_a_waiting_goal_keeps_its_narrow_exit_available_in_a_real_sfm_rvo_closed_loop(monkeypatch, coordinated):
    cells = np.full((120, 160), 100, dtype=np.int8)
    cells[52:68, :] = 0
    cells[30:90, 48:] = 0
    grid = OccupancyGrid(cells, 0.05, (-1.0, -3.0))
    actors = [_body("holder", 0.0), _body("visitor", 2.0, 1.0)]
    plans = [
        {
            "name": "holder",
            "position": actors[0]["position"],
            "behavior": "walk",
            "speed": 0.9,
            "arrival_tolerance": 0.2,
            "waypoints": [
                {"position": [0.0, 0.0, 0.0], "wait": 4.0, "yaw": 0.0},
                {"position": [3.0, 0.0, 0.0], "wait": 0.0, "yaw": None},
            ],
        },
        {
            "name": "visitor",
            "position": actors[1]["position"],
            "behavior": "walk",
            "speed": 0.9,
            "arrival_tolerance": 0.2,
            "waypoints": [{"position": [0.0, 0.0, 0.0], "wait": 0.0, "yaw": None}],
        },
    ]
    original = copy.deepcopy(plans)
    settings = {"map_clearance_m": 0.03, "pair_clearance_m": 0.25, "obstacle_time_horizon_s": 0.5}
    engine = NavigationEngine(plans, grid, settings, actors)
    if not coordinated:
        monkeypatch.setattr(engine._goals, "update", lambda *args, **kwargs: {"objectives": {}, "status": {}})
    states = actors
    error = None
    phases = set()
    for tick in range(1800):
        try:
            result = engine.step(tick / 60, 1 / 60, states)
        except RuntimeError as failure:
            error = failure
            break
        next_states = result["agents"]
        for before, after in zip(states, next_states):
            assert not after["error"]
            assert grid.segment_clear(before["position"], after["position"], 0.28)
            assert math.dist(before["position"], after["position"]) <= 0.9 / 60 + 1e-7
            if before["name"] == "holder" and after["phase"] in ("turning", "dwelling"):
                assert after["position"] == actors[0]["position"]
            phases.add(after["phase"])
        relative = np.asarray(states[0]["position"][:2]) - states[1]["position"][:2]
        change = (
            np.asarray(next_states[0]["position"][:2])
            - states[0]["position"][:2]
            - np.asarray(next_states[1]["position"][:2])
            + states[1]["position"][:2]
        )
        assert engine._chord_distance(relative, change) >= 0.75 - 1e-6
        states = next_states
        if all(state["phase"] == "complete" for state in states):
            break
    engine.close()
    if coordinated:
        assert error is None, str(error)
        assert all(state["phase"] == "complete" for state in states)
        assert "goal_wait" in phases and "goal_egress" in phases
    else:
        assert error is not None or not all(state["phase"] == "complete" for state in states)
    assert plans == original


def test_a_compatible_sliver_of_the_original_arrival_region_remains_reachable_in_closed_loop(monkeypatch):
    actors, _ = _pair(goal=(0.6, 0.0, 0.0), next_target=None, phase="complete")
    plans = [
        {
            "name": state["name"],
            "position": state["position"],
            "behavior": "walk",
            "speed": 0.9,
            "arrival_tolerance": 0.2,
            "waypoints": [{"position": [0.0 if i == 0 else 0.6, 0.0, 0.0], "wait": 0.0, "yaw": None}],
        }
        for i, state in enumerate(actors)
    ]
    settings = {"map_clearance_m": 0.03, "pair_clearance_m": 0.25, "obstacle_time_horizon_s": 0.5}
    engine = NavigationEngine(plans, _floor(), settings, actors)
    states = actors
    for tick in range(1200):
        states = engine.step(tick / 60, 1 / 60, states)["agents"]
        assert all(state["error"] is None for state in states)
        assert states[0]["position"] == actors[0]["position"]
        if all(state["phase"] == "complete" for state in states):
            break
    engine.close()
    assert all(state["phase"] == "complete" for state in states)
    assert math.dist(states[1]["position"], [0.6, 0.0, 0.0]) <= 0.2
    assert math.dist(states[0]["position"], states[1]["position"]) >= 0.75


def test_a_zero_width_goal_region_waits_explicitly_without_creep_or_route_stall():
    actors = [_body("holder", 0.0), _body("visitor", 2.0)]
    plans = [
        {
            "name": state["name"],
            "position": state["position"],
            "behavior": "walk",
            "speed": 0.9,
            "arrival_tolerance": 0.2,
            "waypoints": [{"position": [0.0 if i == 0 else 0.55, 0.0, 0.0], "wait": 0.0, "yaw": None}],
        }
        for i, state in enumerate(actors)
    ]
    settings = {"map_clearance_m": 0.03, "pair_clearance_m": 0.25, "obstacle_time_horizon_s": 0.5}
    engine = NavigationEngine(plans, _floor(), settings, actors)
    states = actors
    stopped = None
    for tick in range(1500):
        result = engine.step(tick / 60, 1 / 60, states)
        states = result["agents"]
        assert all(state["error"] is None for state in states)
        assert states[0]["position"] == actors[0]["position"]
        if states[1]["phase"] == "goal_wait":
            if stopped is None:
                stopped = states[1]["position"]
            assert states[1]["position"] == stopped
    assert stopped is not None and states[1]["phase"] == "goal_wait"
    status = result["goal_occupancy"]
    assert status["queue"][0]["reason"] == "zero_width_goal_occupancy"
    assert status["queue"][0]["permanent"] is True
    engine.close()


def _portal_floor():
    resolution = 0.025
    cells = np.zeros((160, 160), dtype=np.int8)
    rows, cols = np.indices(cells.shape)
    x, y = -2 + (cols + 0.5) * resolution, -2 + (rows + 0.5) * resolution
    cells[(y < 0) & ((x < -1.3) | (x > 0.4) | (y < -1.5))] = 100
    cells[(abs(y) < 0.04) & ((x < -0.7) | (x > 0.3))] = 100
    return OccupancyGrid(cells, resolution, (-2.0, -2.0))


@pytest.mark.parametrize("obstructed", [False, True])
def test_goal_queue_clears_an_exclusive_portal_before_waiting(obstructed):
    from test_ros2_door_interactions import _case

    _, spec, _, _, settings = _case()
    spec["exclusive_entry"] = True
    grid = _portal_floor()
    holder, visitor = _body("holder", -0.2, 1.4), _body("visitor", -0.2, 0.36)
    visitor["yaw"] = math.pi / 2
    actors = [holder, visitor]
    intentions = {
        "holder": _intention(holder["position"], phase="dwelling", next_target=(1.4, 1.4, 0.0)),
        "visitor": _intention(holder["position"]),
    }
    original_intentions = copy.deepcopy(intentions)
    coordinator = GoalOccupancyCoordinator(grid, settings)
    directive = coordinator.update(0.0, actors, intentions)["objectives"]["visitor"]
    target = directive["target"]
    assert directive["phase"] == "goal_yield" and directive["certified_ray"]
    assert target[1] >= visitor["position"][1] + 0.1
    assert math.dist(target, visitor["position"][:2]) <= 0.4
    assert grid.segment_clear(visitor["position"], target, visitor["radius_m"] + 0.03)
    assert math.dist(target, holder["position"][:2]) >= 0.75

    if obstructed:
        # Only the optional staging segment is obstructed; standing and the
        # holder's exit remain safe. Removing the obstacle cannot renew it.
        obstacles = NavigationEngine._dynamic_obstacles(
            [
                {
                    "prim_path": "/cart",
                    "position": [-0.2, 0.78, 0.0],
                    "radius_m": 0.08,
                    "velocity": [0.0, 0.0, 0.0],
                    "velocity_valid": True,
                },
            ]
        )
    else:
        visitor["position"] = [*target, 0.0]
        obstacles = []
    assert coordinator.update(0.5, actors, intentions, obstacles)["objectives"]["visitor"]["phase"] == "goal_wait"
    assert coordinator.update(1.0, actors, intentions)["objectives"]["visitor"]["target"] is None
    assert intentions == original_intentions


def test_a_cached_arrival_endpoint_improves_when_the_original_goal_region_becomes_visible():
    from test_ros2_door_interactions import _case, _leaf

    _, _, _, _, settings = _case()
    grid = _portal_floor()
    actors = [_body("holder", -0.455393477731074, 0.7862301123018068), _body("visitor", -0.9, -0.5)]
    for actor in actors:
        actor["radius_m"] = 0.15
    goal, tolerance = [-0.65, 0.45, 0.0], 0.2
    intentions = {
        "holder": _intention(actors[0]["position"], phase="complete"),
        "visitor": _intention(goal),
    }
    obstacles = NavigationEngine._dynamic_obstacles([_leaf(math.pi / 2)])
    coordinator = GoalOccupancyCoordinator(grid, settings)
    coordinator.update(0.0, actors, intentions, obstacles)
    actors[1]["position"] = [-0.5954306877354002, 0.19664517064870857, 0.0]
    original = copy.deepcopy((actors, intentions))
    result = coordinator.update(0.1, actors, intentions, obstacles)
    endpoint = result["objectives"]["visitor"]["target"]
    required = sum(actor["radius_m"] for actor in actors) + 0.25
    assert math.dist([*endpoint, 0.0], goal) < tolerance
    assert math.dist(endpoint, actors[0]["position"][:2]) > required
    assert grid.segment_clear(actors[1]["position"], endpoint, actors[1]["radius_m"] + 0.03)
    assert segment_polygon_clear(actors[1]["position"], endpoint, obstacles[0].footprint, actors[1]["radius_m"] + 0.03)
    assert (actors, intentions) == original


def test_a_completed_visible_vertex_does_not_block_approach_to_its_controlled_door():
    from test_ros2_door_interactions import _case, _leaf

    _, _, _, _, settings = _case()
    grid = _portal_floor()
    actors = [
        _body("holder", -0.6618578083357228, 0.2532927024871117),
        _body("visitor", -0.455393477731074, 0.7862301123018068),
    ]
    for actor in actors:
        actor["radius_m"] = 0.15
    intentions = {
        "holder": _intention(actors[0]["position"], phase="complete"),
        "visitor": _intention([-0.65, -0.5, 0.0]),
    }
    coordinator = GoalOccupancyCoordinator(grid, settings)
    obstacles = NavigationEngine._dynamic_obstacles([_leaf(0.0)])
    first = coordinator.update(0.0, actors, intentions, obstacles)
    actors[1]["position"] = [*first["objectives"]["visitor"]["target"], 0.0]
    result = coordinator.update(0.1, actors, intentions, obstacles)
    target = result["objectives"]["visitor"]["target"]
    assert target is not None and target != actors[1]["position"][:2]
    assert grid.segment_clear(actors[1]["position"], target, actors[1]["radius_m"] + 0.03)
    # Guidance reaches the gate; the door controller and actual swept solve own admission.
    assert target[1] < settings["door_specs"][0]["approach_distance_m"]


def test_exclusive_door_goal_exchange_clears_the_resource_cycle_after_the_original_dwell():
    from test_ros2_door_interactions import _case, _leaf

    _, spec, _, _, settings = _case(start=0.0)
    spec.update(
        exclusive_entry=True,
        closed_footprint_xy_m=[[-0.74, -0.04], [0.34, -0.04], [0.34, 0.04], [-0.74, 0.04]],
        z_min_m=0.0,
        z_max_m=2.3,
    )
    grid = _portal_floor()
    actors = [_body("exiter", -0.65, -0.5), _body("holder", -0.65, 0.45)]
    for actor in actors:
        actor["radius_m"] = 0.15
    actors[0]["yaw"] = math.pi
    plans = [
        {
            "name": "exiter",
            "position": actors[0]["position"],
            "behavior": "walk",
            "speed": 0.6,
            "arrival_tolerance": 0.2,
            "waypoints": [{"position": actors[1]["position"], "wait": 0.0, "yaw": None}],
        },
        {
            "name": "holder",
            "position": actors[1]["position"],
            "behavior": "walk",
            "speed": 0.6,
            "arrival_tolerance": 0.2,
            "waypoints": [
                {"position": actors[1]["position"], "wait": 1.0, "yaw": None},
                {"position": actors[0]["position"], "wait": 1.0, "yaw": None},
            ],
        },
    ]
    original = copy.deepcopy(plans)
    engine = NavigationEngine(plans, grid, settings, actors)
    states = actors
    angle = previous_angle = 0.0
    required = sum(actor["radius_m"] for actor in actors) + settings.get("pair_clearance_m", 0.25)
    yielded = False
    for tick in range(1800):
        observation = [
            {
                "name": "door",
                "valid": True,
                "fully_open": angle == 90 and previous_angle == angle,
                "fully_closed": angle == 0 and previous_angle == angle,
                "actual_angle_deg": angle,
                "angular_speed_degps": (angle - previous_angle) * 60,
            }
        ]
        result = engine.step(tick / 60, 1 / 60, states, [_leaf(math.radians(angle))], observation)
        next_states = result["agents"]
        assert all(state["error"] is None for state in next_states)
        for before, after in zip(states, next_states):
            assert grid.segment_clear(before["position"], after["position"], before["radius_m"] + 0.03)
            assert math.dist(before["position"], after["position"]) <= 0.6 / 60 + 1e-7
        relative = np.asarray(states[0]["position"][:2]) - states[1]["position"][:2]
        change = (
            np.asarray(next_states[0]["position"][:2])
            - states[0]["position"][:2]
            - np.asarray(next_states[1]["position"][:2])
            + states[1]["position"][:2]
        )
        assert engine._chord_distance(relative, change) >= required - 1e-6
        assert not all(
            -1.3 <= state["position"][0] <= 0.4 and -1.5 <= state["position"][1] <= 0 for state in next_states
        )
        if tick / 60 < 1.0:
            assert next_states[1]["position"] == actors[1]["position"]
        for hold in result["goal_occupancy"]["holders"]:
            transaction = hold.get("departure_yield")
            if transaction is not None:
                yielded = True
                owner = next(state for state in next_states if state["name"] == hold["name"])
                assert math.dist(owner["position"][:2], transaction["origin_xy_m"]) <= 0.4 + 1e-7
        command = result["door_commands"][0]
        previous_angle = angle
        if not command["hold"]:
            angle = min(90, angle + 3) if command["request_open"] else max(0, angle - 3)
        states = [{**before, **after} for before, after in zip(states, next_states)]
        if all(state["phase"] == "complete" for state in states):
            break
    engine.close()
    assert yielded
    assert all(state["phase"] == "complete" for state in states)
    assert math.dist(states[0]["position"], plans[0]["waypoints"][0]["position"]) <= 0.2
    assert math.dist(states[1]["position"], plans[1]["waypoints"][1]["position"]) <= 0.2
    assert plans == original


def _native_goal_queue_snapshot():
    # Native v4 first-wait observed poses and original immutable route inputs.
    with np.load(Path(__file__).parent / "fixtures/native_goal_queue_20261104.npz", allow_pickle=False) as fixture:
        grid = OccupancyGrid(fixture["occupancy"], float(fixture["resolution_m"]), tuple(fixture["origin_xy_m"]))
        observation = json.loads(str(fixture["observation_json"]))
    plans = [
        {
            "name": "walker_02",
            "position": [-0.24079, -0.61706, 0.0],
            "yaw": 0.0,
            "behavior": "walk",
            "speed": 0.9,
            "start_time": 3.0,
            "repeat": True,
            "waypoints": [
                {"position": [-2.90847, 0.24044, 0.0], "wait": 4.0, "yaw": 90.0},
                {"position": [-5.25953, -1.03968, 0.0], "wait": 2.0, "yaw": 90.0},
            ],
            "arrival_tolerance": 0.2,
            "asset_path": None,
            "gender": "male",
        },
        {
            "name": "walker_03",
            "position": [3.634, -0.50313, 0.0],
            "yaw": 6.0,
            "behavior": "walk",
            "speed": 0.9,
            "start_time": 5.0,
            "repeat": True,
            "waypoints": [
                {"position": [3.46515, 1.31306, 0.0], "wait": 0.0, "yaw": 180.0},
                {"position": [-2.20991, 1.34166, 0.0], "wait": 0.0, "yaw": 180.0},
                {"position": [-6.25, -1.45, 0.0], "wait": 8.0, "yaw": 180.0},
            ],
            "arrival_tolerance": 0.2,
            "asset_path": None,
            "gender": "male",
        },
    ]
    spawn_agents = [
        {
            "name": "walker_02",
            "position": [-0.24079, -0.61706, 0.0],
            "yaw": -1.5707963267948963,
            "velocity": [0.0, 0.0, 0.0],
            "radius_m": 0.25,
            "height_m": 1.8,
        },
        {
            "name": "walker_03",
            "position": [3.634, -0.50313, 0.0],
            "yaw": -1.466076571675237,
            "velocity": [0.0, 0.0, 0.0],
            "radius_m": 0.25,
            "height_m": 1.8,
        },
    ]
    states = [
        {
            "name": "walker_02",
            "position": [-5.430285375862988, -1.1350378809220356, 0.0],
            "yaw": 1.5707963267948966,
            "velocity": [0.0, 0.0, 0.0],
            "radius_m": 0.25,
            "height_m": 1.8,
        },
        {
            "name": "walker_03",
            "position": [-3.6004798961850573, -1.6435861452366665, 0.0],
            "yaw": -2.8336278609657812,
            "velocity": [-0.838457107543945, -0.266700565814972, 0.0],
            "radius_m": 0.25,
            "height_m": 1.8,
        },
    ]
    settings = {
        "map_clearance_m": 0.03,
        "pair_clearance_m": 0.25,
        "time_horizon_s": 2.5,
        "obstacle_time_horizon_s": 0.5,
        "neighbor_distance_m": 4.0,
        "relaxation_time_s": 0.4,
        "max_turn_rate_radps": 2.5,
        "seed": 42,
    }
    settings["door_specs"] = observation["door_specs"]
    engine = NavigationEngine(plans, grid, settings, spawn_agents)
    engine.routes["walker_02"].index = 1
    engine.routes["walker_02"].phase = "dwelling"
    engine.routes["walker_02"].deadline = 25.44999999999995
    engine.routes["walker_03"].index = 2
    engine.routes["walker_03"].phase = "walking"
    return engine, states, plans, 25.066666666666524, observation


def test_native_goal_queue_admission_waits_until_the_departing_body_has_actually_passed():
    engine, states, plans, start, observation = _native_goal_queue_snapshot()
    axis = np.asarray(plans[0]["waypoints"][0]["position"][:2]) - states[0]["position"][:2]
    axis /= np.linalg.norm(axis)
    try:
        for tick in range(900):
            result = engine.step(
                start + tick / 60, 1 / 60, states, observation["obstacles"], observation["door_states"]
            )
            next_states = result["agents"]
            assert all(state["error"] is None for state in next_states)
            queue = next((item for item in result["goal_occupancy"]["queue"] if item["name"] == "walker_03"), None)
            if queue is not None and queue["admitted"]:
                relative = np.asarray(next_states[0]["position"][:2]) - next_states[1]["position"][:2]
                assert float(np.dot(relative, axis)) > 0
                break
            states = [{**before, **after} for before, after in zip(states, next_states)]
        else:
            pytest.fail("The original native queue never admitted its waiting pedestrian")
    finally:
        engine.close()


def test_native_goal_queue_snapshot_reaches_both_original_pending_goals_without_losing_its_detour():
    engine, states, plans, start, observation = _native_goal_queue_snapshot()
    original = copy.deepcopy(plans)
    arrived = set()
    goals = {"walker_02": plans[0]["waypoints"][0]["position"], "walker_03": plans[1]["waypoints"][2]["position"]}
    try:
        for tick in range(1500):
            result = engine.step(
                start + tick / 60, 1 / 60, states, observation["obstacles"], observation["door_states"]
            )
            next_states = result["agents"]
            for before, after in zip(states, next_states):
                assert not after["error"]
                assert engine.grid.segment_clear(
                    before["position"], after["position"], before["radius_m"] + engine.margin
                )
                assert math.dist(before["position"], after["position"]) <= 0.9 / 60 + 1e-7
                for obstacle in observation["obstacles"]:
                    assert segment_polygon_clear(
                        before["position"],
                        after["position"],
                        convex_footprint(obstacle["footprint_xy_m"]),
                        before["radius_m"] + engine.margin,
                    )
                if after["waypoint_index"] == 2 and after["route_phase"] in ("turning", "dwelling"):
                    assert math.dist(after["position"], goals[after["name"]]) <= 0.2
                    arrived.add(after["name"])
            relative = np.asarray(states[0]["position"][:2]) - states[1]["position"][:2]
            movement = (
                np.asarray(next_states[0]["position"][:2])
                - states[0]["position"][:2]
                - np.asarray(next_states[1]["position"][:2])
                + states[1]["position"][:2]
            )
            assert engine._chord_distance(relative, movement) >= 0.75 - 1e-7
            states = [{**before, **after} for before, after in zip(states, next_states)]
            if len(arrived) == 2:
                break
        assert arrived == set(goals)
        assert plans == original
    finally:
        engine.close()


def test_native_three_person_observation_keeps_the_uncontrolled_door_in_its_exit_certificate():
    engine, _, _, start, observation = _native_goal_queue_snapshot()
    coordinator = GoalOccupancyCoordinator(engine.grid, engine.settings)
    actors = copy.deepcopy(observation["observed"])
    intentions = copy.deepcopy(observation["intentions"])
    obstacles = NavigationEngine._dynamic_obstacles(observation["obstacles"])
    first = coordinator.update(start, actors, intentions, obstacles)
    visitor = next(actor for actor in actors if actor["name"] == "walker_03")
    directive = first["objectives"]["walker_03"]
    if directive["phase"] == "goal_yield":
        assert math.dist(directive["target"], directive["origin"]) <= 0.4
        visitor["position"] = [*directive["target"], visitor["position"][2]]
    intentions["walker_02"].update(
        phase="walking",
        active=True,
        route_context=[2, 0],
        target=[-2.90847, 0.24044, 0.0],
        next_target=[-0.24079, -0.61706, 0.0],
    )
    result = coordinator.update(25.483333333333167, actors, intentions, obstacles)
    assert result["objectives"]["walker_02"]["phase"] == "goal_egress"
    owner = next(actor for actor in actors if actor["name"] == "walker_02")
    target = result["objectives"]["walker_02"]["target"]
    for obstacle in observation["obstacles"]:
        assert segment_polygon_clear(
            owner["position"], target, convex_footprint(obstacle["footprint_xy_m"]), owner["radius_m"] + engine.margin
        )
    engine.close()


def _native_return_snapshot():
    with np.load(Path(__file__).parent / "fixtures/native_goal_queue_20261104.npz", allow_pickle=False) as fixture:
        grid = OccupancyGrid(fixture["occupancy"], float(fixture["resolution_m"]), tuple(fixture["origin_xy_m"]))
        snapshot = json.loads(str(fixture["return_observation_json"]))
    engine = NavigationEngine(snapshot["plans"], grid, snapshot["settings"], snapshot["spawn_agents"])
    for name, intention in snapshot["intentions"].items():
        route = engine.routes[name]
        route.index, route.cycles = intention["route_context"]
        route.phase = intention["phase"]
        route.path = copy.deepcopy(intention["path"])
    old = snapshot["holder"]
    seed = next(copy.deepcopy(state) for state in snapshot["states"] if state["name"] == old["name"])
    seed["position"] = [*old["origin_xy_m"], seed["position"][2]]
    intention = _intention(
        old["target_m"], phase="dwelling", next_target=old["next_target_m"], context=tuple(old["route_context"])
    )
    engine._goals.update(
        old["started_s"], [seed], {old["name"]: intention}, engine._dynamic_obstacles(snapshot["obstacles"])
    )
    return engine, snapshot


def test_native_return_traffic_can_use_a_small_inner_refuge_without_weakening_any_certificate():
    engine, snapshot = _native_return_snapshot()
    try:
        result = engine.step(
            snapshot["time_s"], 1 / 60, snapshot["states"], snapshot["obstacles"], snapshot["door_states"]
        )
        queue = next(item for item in result["goal_occupancy"]["queue"] if item["name"] == "walker_03")
        assert 0 < math.dist(queue["origin_xy_m"], queue["refuge_xy_m"]) <= 0.2
        before = next(state for state in snapshot["states"] if state["name"] == "walker_03")
        target = queue["refuge_xy_m"]
        assert engine.grid.segment_clear(before["position"], target, before["radius_m"] + engine.margin)
        for other in snapshot["states"]:
            if other["name"] == before["name"]:
                continue
            relative = np.asarray(before["position"][:2]) - other["position"][:2]
            motion = np.asarray(target) - before["position"][:2]
            closest = np.clip(-np.dot(relative, motion) / np.dot(motion, motion), 0.0, 1.0)
            assert np.linalg.norm(relative + closest * motion) >= (
                before["radius_m"] + other["radius_m"] + engine.pair_clearance - 1e-7
            )
        for obstacle in snapshot["obstacles"]:
            assert segment_polygon_clear(
                before["position"],
                target,
                convex_footprint(obstacle["footprint_xy_m"]),
                before["radius_m"] + engine.margin,
            )
    finally:
        engine.close()


def test_native_return_queue_clears_and_both_approachers_resume_their_original_routes():
    engine, snapshot = _native_return_snapshot()
    states = copy.deepcopy(snapshot["states"])
    original_plans = copy.deepcopy(snapshot["plans"])
    admitted = set()
    admitted_origins = {}
    progressed = set()
    saw_egress = False
    try:
        for tick in range(840):
            result = engine.step(
                snapshot["time_s"] + tick / 60, 1 / 60, states, snapshot["obstacles"], snapshot["door_states"]
            )
            next_states = result["agents"]
            assert all(state["error"] is None for state in next_states)
            for before, after in zip(states, next_states):
                assert engine.grid.segment_clear(
                    before["position"], after["position"], before["radius_m"] + engine.margin
                )
                assert math.dist(before["position"], after["position"]) <= 0.9 / 60 + 1e-7
                for obstacle in snapshot["obstacles"]:
                    assert segment_polygon_clear(
                        before["position"],
                        after["position"],
                        convex_footprint(obstacle["footprint_xy_m"]),
                        before["radius_m"] + engine.margin,
                    )
            for i, first in enumerate(next_states):
                for j, second in enumerate(next_states[:i]):
                    relative = np.asarray(states[i]["position"][:2]) - states[j]["position"][:2]
                    motion = (
                        np.asarray(first["position"][:2])
                        - states[i]["position"][:2]
                        - np.asarray(second["position"][:2])
                        + states[j]["position"][:2]
                    )
                    squared = np.dot(motion, motion)
                    closest = np.clip(-np.dot(relative, motion) / squared, 0.0, 1.0) if squared else 0.0
                    assert np.linalg.norm(relative + closest * motion) >= 0.75 - 1e-7
            admitted.update(item["name"] for item in result["goal_occupancy"]["queue"] if item["admitted"])
            for state in next_states:
                if state["name"] == "walker_01" and state["phase"] == "goal_egress":
                    saw_egress = True
                if state["name"] in admitted and state["phase"] in ("goal_approach", "walking"):
                    origin = admitted_origins.setdefault(state["name"], state["position"])
                    if math.dist(origin, state["position"]) >= 0.05:
                        progressed.add(state["name"])
            states = [{**before, **after} for before, after in zip(states, next_states)]
            if progressed == {"walker_02", "walker_03"}:
                break
        assert progressed == {"walker_02", "walker_03"}
        assert saw_egress
        assert snapshot["plans"] == original_plans
    finally:
        engine.close()


def test_a_native_already_departing_owner_recovers_without_adding_displacement_or_wait():
    directory = Path(__file__).parent / "fixtures"
    snapshot = json.loads((directory / "native_already_departing_goal_20261106.json").read_text())
    with np.load(directory / snapshot["provenance"]["map_fixture"], allow_pickle=False) as fixture:
        grid = OccupancyGrid(fixture["occupancy"], float(fixture["resolution_m"]), tuple(fixture["origin_xy_m"]))
    original = copy.deepcopy(snapshot["plans"])
    engine = NavigationEngine(snapshot["plans"], grid, snapshot["settings"], snapshot["spawn_agents"])
    for name, fields in snapshot["routes"].items():
        for field, value in fields.items():
            if field in ("door_token", "point_target") and value is not None:
                value = tuple(value)
            setattr(engine.routes[name], field, copy.deepcopy(value))
    request, hold = snapshot["request"], snapshot["hold"]
    owner = next(copy.deepcopy(state) for state in request["agents"] if state["name"] == hold["name"])
    owner["position"] = [*hold["origin"], owner["position"][2]]
    engine._goals.update(
        hold["started"],
        [owner],
        {
            owner["name"]: _intention(
                hold["target"], phase="dwelling", next_target=hold["next_target"], context=tuple(hold["context"][0])
            )
        },
        engine._dynamic_obstacles(request["obstacles"]),
    )
    goals = {name: engine._current_target(route)["position"] for name, route in engine.routes.items()}
    states, reached = request["agents"], set()
    initial_distance = math.dist(owner["position"], goals[owner["name"]])
    newcomer = request["agents"][0]["name"]
    saw_transaction = owner_resumed = False
    try:
        for tick in range(1500):
            time_s = request["time_s"] + tick / 60
            result = engine.step(time_s, 1 / 60, states, request["obstacles"], request["doors"])
            after = result["agents"]
            assert all(state["error"] is None for state in after)
            active_transaction = False
            for record in result["goal_occupancy"]["holders"]:
                transaction = record["departure_yield"]
                if record["name"] == owner["name"] and transaction is not None:
                    saw_transaction = True
                    active_transaction = True
                    assert transaction["origin_xy_m"] == hold["origin"]
                    assert transaction["refuge_xy_m"] == hold["origin"]
            for before, current in zip(states, after):
                assert grid.segment_clear(before["position"], current["position"], before["radius_m"] + engine.margin)
                assert math.dist(before["position"], current["position"]) <= 0.9 / 60 + 1e-7
                assert abs(math.remainder(current["yaw"] - before["yaw"], 2 * math.pi)) <= engine.turn_rate / 60 + 1e-6
                for obstacle in request["obstacles"]:
                    assert segment_polygon_clear(
                        before["position"], current["position"], obstacle["footprint_xy_m"], 0.28
                    )
                if math.dist(current["position"], goals[current["name"]]) <= 0.2:
                    reached.add(current["name"])
                if current["name"] == owner["name"]:
                    if active_transaction:
                        assert current["position"] == owner["position"]
                    elif saw_transaction and current["phase"] in ("walking", "goal_egress"):
                        owner_resumed |= initial_distance - math.dist(
                            current["position"], goals[owner["name"]]
                        ) >= current.get("radius_m", 0.25)
            for i, first in enumerate(after):
                for j, second in enumerate(after[:i]):
                    relative = np.asarray(states[i]["position"][:2]) - states[j]["position"][:2]
                    change = np.asarray(first["position"][:2]) - states[i]["position"][:2]
                    change -= np.asarray(second["position"][:2]) - states[j]["position"][:2]
                    squared = float(np.dot(change, change))
                    closest = np.clip(-np.dot(relative, change) / squared, 0, 1) if squared else 0.0
                    assert np.linalg.norm(relative + closest * change) >= 0.75 - 1e-7
            states = [{**old, **new} for old, new in zip(states, after)]
            if newcomer in reached and owner_resumed:
                break
        assert saw_transaction and newcomer in reached and owner_resumed
        assert snapshot["plans"] == original
    finally:
        engine.close()


def test_a_native_goal_resource_cycle_yields_the_owner_only_after_its_original_turn_and_dwell():
    fixture_dir = Path(__file__).parent / "fixtures"
    snapshot = json.loads((fixture_dir / "native_goal_departure_swap_20261105.json").read_text())
    with np.load(fixture_dir / snapshot["provenance"]["map_fixture"], allow_pickle=False) as fixture:
        grid = OccupancyGrid(fixture["occupancy"], float(fixture["resolution_m"]), tuple(fixture["origin_xy_m"]))
    original_plans = copy.deepcopy(snapshot["plans"])
    engine = NavigationEngine(snapshot["plans"], grid, snapshot["settings"], snapshot["spawn_agents"])
    for name, recorded in snapshot["routes"].items():
        for field, value in recorded.items():
            setattr(engine.routes[name], field, copy.deepcopy(value))
    request = snapshot["request"]
    states = copy.deepcopy(request["agents"])
    owner_origin = states[1]["position"].copy()
    visitor_origin = states[2]["position"].copy()
    visitor_goal = snapshot["plans"][2]["waypoints"][2]["position"]
    original_wait = snapshot["plans"][1]["waypoints"][1]["wait"]
    dwell_started = None
    yielded = visitor_arrived = owner_resumed = False
    try:
        for tick in range(1200):
            time_s = request["time_s"] + tick / 60
            result = engine.step(time_s, 1 / 60, states, request["obstacles"], request["doors"])
            next_states = result["agents"]
            assert all(state["error"] is None for state in next_states)
            owner, visitor = next_states[1:]
            if owner["route_phase"] == "dwelling" and dwell_started is None:
                dwell_started = time_s
            if owner["route_phase"] in ("turning", "dwelling") and owner["waypoint_index"] == 1:
                assert owner["position"] == owner_origin
            elif owner["position"] != owner_origin:
                assert dwell_started is not None and time_s >= dwell_started + original_wait - 1e-9
            for hold in result["goal_occupancy"]["holders"]:
                transaction = hold.get("departure_yield")
                if transaction is not None and hold["name"] == owner["name"]:
                    yielded = True
                    assert transaction["origin_xy_m"] == owner_origin[:2]
                    assert math.dist(owner["position"][:2], transaction["origin_xy_m"]) <= 0.4 + 1e-7
            if not yielded:
                assert visitor["position"] == visitor_origin
            visitor_arrived |= math.dist(visitor["position"], visitor_goal) <= 0.2
            owner_resumed |= (
                yielded
                and owner["phase"] in ("walking", "goal_egress")
                and math.dist(owner["position"], owner_origin) > 0.2
            )
            for before, after in zip(states, next_states):
                assert grid.segment_clear(before["position"], after["position"], before["radius_m"] + engine.margin)
                assert math.dist(before["position"], after["position"]) <= 0.9 / 60 + 1e-7
                assert abs(math.remainder(after["yaw"] - before["yaw"], 2 * math.pi)) <= engine.turn_rate / 60 + 1e-6
                for obstacle in request["obstacles"]:
                    assert segment_polygon_clear(
                        before["position"],
                        after["position"],
                        convex_footprint(obstacle["footprint_xy_m"]),
                        before["radius_m"] + engine.margin,
                    )
            for i, first in enumerate(next_states):
                for j, second in enumerate(next_states[:i]):
                    relative = np.asarray(states[i]["position"][:2]) - states[j]["position"][:2]
                    motion = (
                        np.asarray(first["position"][:2])
                        - states[i]["position"][:2]
                        - np.asarray(second["position"][:2])
                        + states[j]["position"][:2]
                    )
                    squared = float(np.dot(motion, motion))
                    closest = np.clip(-np.dot(relative, motion) / squared, 0, 1) if squared else 0
                    assert np.linalg.norm(relative + closest * motion) >= 0.75 - 1e-7
            states = [{**before, **after} for before, after in zip(states, next_states)]
            if visitor_arrived and owner_resumed:
                break
        assert yielded and visitor_arrived and owner_resumed
        assert snapshot["plans"] == original_plans
    finally:
        engine.close()
