# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measured door-clearance transactions preserve the original pedestrian route."""

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from pedstrain_simulation.simulation.ros2_navigation.interactions import (
    DoorCoordinator,
    _NoCertifiedRefuge,
    _polygon_clearance,
)
from pedstrain_simulation.simulation.ros2_navigation.planner import (
    OccupancyGrid,
    convex_footprint,
    segment_polygon_clear,
)
from pedstrain_simulation.simulation.ros2_navigation.solver import NavigationEngine


def _case(start=1.0):
    grid = OccupancyGrid(np.zeros((160, 160), dtype=np.int8), 0.025, (-2.0, -2.0))
    spec = {
        "name": "door",
        "body_prim_path": "/leaf",
        "enabled": True,
        "open_angle_deg": 90.0,
        "closed_angle_deg": 0.0,
        "target_ramp_duration_s": 0.4,
        "interior_polygon_xy_m": [[-1.3, -1.5], [0.4, -1.5], [0.4, 0.0], [-1.3, 0.0]],
        "swept_footprint_xy_m": [[-0.72, -1.05], [0.35, -1.05], [0.35, 0.05], [-0.72, 0.05]],
        "doorway_segment_xy_m": [[-0.7, 0.0], [0.3, 0.0]],
        "approach_distance_m": 0.55,
    }
    agents = [{"name": "walker", "position": [-0.65, -0.65, 0.0], "yaw": 0.0, "radius_m": 0.15, "height_m": 1.8}]
    plans = [
        {
            "name": "walker",
            "position": [-0.65, -0.65, 0.0],
            "behavior": "walk",
            "speed": 0.6,
            "start_time": start,
            "arrival_tolerance": 0.2,
            "repeat": False,
            "waypoints": [{"position": [-0.2, 0.8, 0.0], "wait": 0.2, "yaw": None}],
        }
    ]
    settings = {"map_clearance_m": 0.03, "door_specs": [spec]}
    return grid, spec, agents, plans, settings


def test_admission_events_keep_the_initial_proximity_evidence_without_reissuing_a_lease():
    grid, _, agents, _, settings = _case(start=0.0)
    actors = [{**agents[0], "position": [-0.2, 0.45, 0.0]}]
    intentions = {"walker": {"active": True, "target": [-0.2, -1.0, 0.0], "path": [[-0.2, 0.45], [-0.2, -1.0]]}}
    observed = [{"name": "door", "valid": True, "fully_open": False, "angular_speed_degps": 0.0}]
    coordinator = DoorCoordinator(grid, settings)
    first = coordinator.update(0.0, actors, intentions, [], observed)
    assert len(first["events"]) == 1
    assert first["events"][0]["distance_m"] == pytest.approx(0.45)
    assert first["events"][0]["approach_distance_m"] == 0.55
    assert coordinator.update(1 / 60, actors, intentions, [], observed)["events"] == []
    actors[0]["position"][1] = 0.9
    assert coordinator.update(2 / 60, actors, intentions, [], observed)["events"] == []


@pytest.mark.parametrize("angle", [0.0, 0.77, -1.32])
@pytest.mark.parametrize("entering", [False, True])
def test_nearby_actor_must_face_its_portal_before_requesting_open(angle, entering):
    grid, spec, agents, _, settings = _case(start=0.0)
    spec["facing_half_angle_deg"] = 45.0
    # Keep the stationary test poses out of the sweep and rotate the portal
    # together with its room, start and original route.
    spec["swept_footprint_xy_m"] = [[-0.65, -0.08], [0.25, -0.08], [0.25, 0.05], [-0.65, 0.05]]
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    for key in ("swept_footprint_xy_m", "interior_polygon_xy_m", "doorway_segment_xy_m"):
        spec[key] = (np.asarray(spec[key]) @ rotation.T).tolist()
    side = 1 if entering else -1
    start = [*(rotation @ [-0.2, side * 0.45]), 0.0]
    target = [*(rotation @ [-0.2, -side]), 0.0]
    toward = angle - side * math.pi / 2
    actor = {**agents[0], "position": start, "yaw": toward + math.pi}
    intention = {"active": True, "target": target, "path": [start[:2], target[:2]]}
    observed = [{"name": "door", "valid": True, "fully_open": False, "angular_speed_degps": 0.0}]
    coordinator = DoorCoordinator(grid, settings)

    inactive = coordinator.update(0.0, [actor], {"walker": {**intention, "active": False}}, [], observed)
    assert not inactive["door_commands"][0]["request_open"] and not inactive["objectives"]
    for tick, offset in ((1, math.pi), (2, math.pi / 2)):
        actor["yaw"] = toward + offset
        waiting = coordinator.update(tick / 60, [actor], {"walker": intention}, [], observed)
        assert not waiting["events"] and not waiting["door_commands"][0]["request_open"]
        assert waiting["objectives"]["walker"]["phase"] == "door_face"
        assert waiting["objectives"]["walker"]["target"] is None
        assert math.remainder(waiting["objectives"]["walker"]["yaw"] - toward, math.tau) == pytest.approx(0.0)
    actor["yaw"] = toward + math.pi / 6
    admitted = coordinator.update(3 / 60, [actor], {"walker": intention}, [], observed)
    assert admitted["door_commands"][0]["request_open"]
    assert admitted["events"][0]["facing_error_deg"] == pytest.approx(30.0)
    assert admitted["events"][0]["facing_half_angle_deg"] == 45.0
    assert admitted["door_commands"][0]["opening_authorization"] == {
        "actor": "walker",
        "signed_facing_error_deg": pytest.approx(30.0),
        "facing_half_angle_deg": 45.0,
    }


def test_actor_turning_away_during_refuge_must_reface_before_the_first_opening_command():
    grid, spec, agents, _, settings = _case(start=0.0)
    spec["facing_half_angle_deg"] = 45.0
    spec["swept_footprint_xy_m"] = [[-0.65, -0.38], [0.25, -0.38], [0.25, 0.05], [-0.65, 0.05]]
    actor = {**agents[0], "position": [-0.2, -0.28, 0.0], "yaw": math.pi / 2}
    intentions = {"walker": {"active": True, "target": [-0.2, 0.8, 0.0], "path": [[-0.2, -0.28], [-0.2, 0.8]]}}
    observed = [{"name": "door", "valid": True, "fully_open": False, "angular_speed_degps": 0.0}]
    coordinator = DoorCoordinator(grid, settings)

    first = coordinator.update(0.0, [actor], intentions, [], observed)
    assert first["events"] and not first["door_commands"][0]["request_open"]
    refuge = first["objectives"]["walker"]["target"]
    actor.update(position=[*refuge, 0.0], yaw=-math.pi / 2)
    coordinator.update(1 / 60, [actor], intentions, [], observed)
    waiting = coordinator.update(2 / 60, [actor], intentions, [], observed)
    assert not waiting["door_commands"][0]["request_open"] and not waiting["events"]
    assert waiting["objectives"]["walker"]["phase"] == "door_face"
    actor["yaw"] = waiting["objectives"]["walker"]["yaw"]
    first_open = coordinator.update(3 / 60, [actor], intentions, [], observed)
    assert first_open["door_commands"][0]["request_open"]
    assert first_open["door_commands"][0]["opening_authorization"]["signed_facing_error_deg"] == pytest.approx(0)
    actor["yaw"] += math.pi
    continuing = coordinator.update(4 / 60, [actor], intentions, [], observed)["door_commands"][0]
    assert continuing["request_open"] and "opening_authorization" not in continuing


def test_oblique_forward_ray_missing_the_portal_cannot_start_opening():
    grid, spec, actors, _, settings = _case(start=0.0)
    spec.update(
        facing_half_angle_deg=45.0,
        doorway_segment_xy_m=[[0.0, 0.0], [0.64, 0.0]],
        swept_footprint_xy_m=[[0, -0.08], [0.64, -0.08], [0.64, 0.05], [0, 0.05]],
        interior_polygon_xy_m=[[-0.5, -1.5], [0.9, -1.5], [0.9, 0], [-0.5, 0]],
    )
    actors[0].update(position=[-0.41, 0.35, 0.0], yaw=0.01)
    intentions = {"walker": {"active": True, "target": [0.6, -0.6, 0.0]}}
    observed = [{"name": "door", "valid": True, "fully_open": False, "angular_speed_degps": 0.0}]
    coordinator = DoorCoordinator(grid, settings)
    coordinator.update(0.0, actors, intentions, [], observed)
    waiting = coordinator.update(1 / 60, actors, intentions, [], observed)
    assert not waiting["door_commands"][0]["request_open"] and not waiting["events"]
    aim = waiting["objectives"]["walker"]
    assert aim["phase"] == "door_face"
    assert aim["yaw"] == pytest.approx(math.atan2(-0.35, 0.73))
    actors[0]["yaw"] = aim["yaw"]
    assert coordinator.update(2 / 60, actors, intentions, [], observed)["door_commands"][0]["request_open"]


def test_strict_entry_repositions_an_offside_actor_before_it_can_open():
    grid, spec, actors, _, settings = _case(start=0.0)
    spec.update(
        require_front_approach=True,
        facing_half_angle_deg=30.0,
        doorway_segment_xy_m=[[-0.4, 0.0], [0.4, 0.0]],
        swept_footprint_xy_m=[[-0.4, -0.08], [0.4, -0.08], [0.4, 0.05], [-0.4, 0.05]],
    )
    actors[0].update(position=[-0.51, 0.4, 0.0], yaw=math.atan2(-0.4, 0.51))
    intentions = {"walker": {"active": True, "target": [0.0, -0.9, 0.0]}}
    observed = [{"name": "door", "valid": True, "fully_open": False, "angular_speed_degps": 0.0}]
    coordinator = DoorCoordinator(grid, settings)
    coordinator.update(0.0, actors, intentions, [], observed)

    offside = coordinator.update(1 / 60, actors, intentions, [], observed)
    assert not offside["door_commands"][0]["request_open"] and not offside["events"]
    reposition = offside["objectives"]["walker"]
    assert reposition["phase"] == "door_position"
    x, y = reposition["target"]
    assert -0.4 < x < 0.4 and 0.05 + actors[0]["radius_m"] + settings["map_clearance_m"] < y <= 0.55

    actors[0].update(position=[x, y, 0.0], yaw=math.pi / 2)
    turned_away = coordinator.update(2 / 60, actors, intentions, [], observed)
    assert not turned_away["door_commands"][0]["request_open"]
    assert turned_away["objectives"]["walker"]["phase"] == "door_face"
    actors[0]["yaw"] = turned_away["objectives"]["walker"]["yaw"]
    opened = coordinator.update(3 / 60, actors, intentions, [], observed)
    assert opened["door_commands"][0]["request_open"]


@pytest.mark.parametrize(("depth", "opens"), [(0.68, True), (0.75, False)])
def test_strict_exit_uses_measured_refuge_range_and_body_frontage(depth, opens):
    grid, spec, actors, _, settings = _case(start=0.0)
    spec.update(
        require_front_approach=True,
        exit_approach_distance_m=0.72,
        facing_half_angle_deg=30.0,
        doorway_segment_xy_m=[[-0.4, 0.0], [0.4, 0.0]],
    )
    actors[0].update(position=[-0.35, -0.45, 0.0], yaw=math.atan2(0.45, 0.35))
    intentions = {"walker": {"active": True, "target": [0.0, 0.9, 0.0]}}
    observed = [{"name": "door", "valid": True, "fully_open": False, "angular_speed_degps": 0.0}]
    # Give the synthetic leaf a short swept region so the staged actor can
    # clear the leaf before the actual pose is checked for first opening.
    spec["swept_footprint_xy_m"] = [[-0.4, -0.5], [0.4, -0.5], [0.4, 0.05], [-0.4, 0.05]]
    coordinator = DoorCoordinator(grid, settings)
    initial = coordinator.update(0.0, actors, intentions, [], observed)
    assert not initial["events"] and initial["objectives"]["walker"]["phase"] == "door_position"
    actors[0].update(position=[-0.45, -depth, 0.0], yaw=math.atan2(depth, 0.45))
    coordinator.update(1 / 60, actors, intentions, [], observed)
    actual = coordinator.update(2 / 60, actors, intentions, [], observed)
    assert actual["door_commands"][0]["request_open"] is opens


def test_strict_exit_can_admit_after_repositioning_beyond_entry_range():
    grid, spec, actors, _, settings = _case(start=0.0)
    spec.update(
        require_front_approach=True,
        exit_approach_distance_m=0.72,
        facing_half_angle_deg=30.0,
        doorway_segment_xy_m=[[-0.4, 0.0], [0.4, 0.0]],
        swept_footprint_xy_m=[[-0.8, -0.38], [0.4, -0.38], [0.4, 0.05], [-0.8, 0.05]],
    )
    actors[0].update(position=[-0.60, -0.49, 0.0], yaw=math.atan2(0.49, 0.60))
    intentions = {"walker": {"active": True, "target": [0.1, 0.9, 0.0]}}
    observed = [{"name": "door", "valid": True, "fully_open": False, "angular_speed_degps": 0.0}]
    coordinator = DoorCoordinator(grid, settings)
    first = coordinator.update(0.0, actors, intentions, [], observed)
    assert not first["events"] and not first["door_commands"][0]["request_open"]
    approach = first["objectives"]["walker"]
    assert approach["phase"] == "door_position"
    x, y = approach["target"]

    actors[0].update(position=[x, y, 0.0], yaw=math.atan2(-y, -x))
    admitted = coordinator.update(1 / 60, actors, intentions, [], observed)
    evidence = admitted["events"][0]
    assert 0.55 < evidence["distance_m"] < evidence["approach_distance_m"] == 0.72
    assert coordinator.update(2 / 60, actors, intentions, [], observed)["door_commands"][0]["request_open"]


def test_strict_exit_stages_before_admission_when_sweep_clearance_requires_more_than_a_yield():
    grid, spec, actors, _, settings = _case(start=0.0)
    spec.update(
        require_front_approach=True,
        exit_approach_distance_m=0.86,
        facing_half_angle_deg=30.0,
        doorway_segment_xy_m=[[-0.4, 0.0], [0.4, 0.0]],
        swept_footprint_xy_m=[[-0.4, -0.5], [0.4, -0.5], [0.4, 0.05], [-0.4, 0.05]],
    )
    actors[0].update(position=[0.0, -0.3, 0.0], yaw=math.pi / 2, radius_m=0.25)
    intentions = {"walker": {"active": True, "target": [0.0, 0.9, 0.0]}}
    observed = [{"name": "door", "valid": True, "fully_open": False, "angular_speed_degps": 0.0}]
    coordinator = DoorCoordinator(grid, settings)

    first = coordinator.update(0.0, actors, intentions, [], observed)
    assert not first["events"] and not first["door_commands"][0]["request_open"]
    approach = first["objectives"]["walker"]
    assert approach["phase"] == "door_position" and approach["certified_ray"]
    x, y = approach["target"]
    assert math.dist(actors[0]["position"][:2], (x, y)) > 0.4
    assert _polygon_clearance((x, y), np.asarray(spec["swept_footprint_xy_m"])) > 0.28

    actors[0].update(position=[x, y, 0.0], yaw=-math.pi / 2)
    facing = coordinator.update(1 / 60, actors, intentions, [], observed)
    assert not facing["events"] and not facing["door_commands"][0]["request_open"]
    actors[0]["yaw"] = facing["objectives"]["walker"]["yaw"]
    opened = coordinator.update(2 / 60, actors, intentions, [], observed)
    assert opened["events"] and opened["door_commands"][0]["request_open"]


def test_facing_actor_cannot_use_an_unstarted_lease_after_leaving_approach_range():
    grid, spec, actors, _, settings = _case(start=0.0)
    spec["facing_half_angle_deg"] = 45.0
    actors[0].update(position=[-0.2, 0.45, 0.0], yaw=-math.pi / 2)
    intentions = {"walker": {"active": True, "target": [-0.2, -1.3, 0.0]}}
    observed = [{"name": "door", "valid": True, "fully_open": False, "angular_speed_degps": 0.0}]
    coordinator = DoorCoordinator(grid, settings)
    admitted = coordinator.update(0.0, actors, intentions, [], observed)
    assert admitted["events"] and not admitted["door_commands"][0]["request_open"]
    actors[0]["position"][1] = 0.65
    assert not coordinator.update(1 / 60, actors, intentions, [], observed)["door_commands"][0]["request_open"]
    assert not coordinator.update(2 / 60, actors, intentions, [], observed)["objectives"]


def test_a_fully_open_door_does_not_renew_an_uncompleted_access_lease_forever():
    grid, _, agents, _, settings = _case(start=0.0)
    actors = [{**agents[0], "position": [-0.2, 0.45, 0.0]}]
    intentions = {"walker": {"active": True, "target": [-0.2, -1.0, 0.0], "path": [[-0.2, 0.45], [-0.2, -1.0]]}}
    observed = [{"name": "door", "valid": True, "fully_open": False, "angular_speed_degps": 0.0}]
    coordinator = DoorCoordinator(grid, settings)
    coordinator.update(0.0, actors, intentions, [], observed)
    coordinator.update(1 / 60, actors, intentions, [], observed)
    actors[0]["position"][1] = 0.9
    observed[0]["fully_open"] = True
    with pytest.raises(RuntimeError, match="Door access timed out"):
        for tick in range(2, 1200):
            result = coordinator.update(tick / 60, actors, intentions, [], observed)
            assert result["door_commands"][0]["request_open"] and not result["events"]


def test_committed_exit_clears_the_open_door_before_waiting_for_an_occupied_goal():
    grid, spec, actors, plans, settings = _case(start=0.0)
    spec.update(
        exclusive_entry=True,
        facing_half_angle_deg=45.0,
        swept_footprint_xy_m=[[-0.65, -0.08], [0.25, -0.08], [0.25, 0.05], [-0.65, 0.05]],
    )
    actors[0].update(position=[-0.2, -0.45, 0.0], yaw=math.pi / 2)
    occupied = [-0.2, 0.95, 0.0]
    actors.append({**actors[0], "name": "holder", "position": occupied})
    plans[0].update(position=actors[0]["position"], waypoints=[{"position": occupied, "wait": 0.2}])
    plans.append(
        {**plans[0], "name": "holder", "position": occupied, "waypoints": [{"position": occupied, "wait": 20.0}]}
    )
    original = copy.deepcopy(plans)
    engine = NavigationEngine(plans, grid, settings, actors)
    dt = 1 / 60
    states = actors
    try:
        for tick in range(540):
            opened = tick >= 2
            doors = [
                {
                    "name": "door",
                    "valid": True,
                    "fully_open": opened,
                    "actual_angle_deg": 90.0 if opened else 0.0,
                    "angular_speed_degps": 0.0,
                }
            ]
            command = engine.step(tick * dt, dt, states, [], doors)
            walker, holder = command["agents"]
            assert walker["error"] is None and holder["error"] is None
            assert holder["phase"] == "dwelling" and holder["position"] == occupied
            assert (
                engine._chord_distance(
                    np.asarray(states[0]["position"]) - occupied,
                    np.asarray(walker["position"]) - states[0]["position"],
                )
                >= 0.55 - 1e-8
            )
            if walker["phase"] == "goal_wait":
                assert walker["position"][1] > 0.05 + actors[0]["radius_m"] + settings["map_clearance_m"]
                assert not command["door_commands"][0]["request_open"]
                break
            states = [{**state, **updated} for state, updated in zip(states, command["agents"])]
        else:
            pytest.fail("The departing actor never cleared the door and joined the goal queue.")
        assert plans == original
    finally:
        engine.close()


@pytest.mark.parametrize("invalidated", ["context", "inactive"])
def test_an_invalidated_route_releases_access_without_reusing_its_old_deadline(invalidated):
    grid, _, agents, _, settings = _case(start=0.0)
    actors = [{**agents[0], "position": [-0.2, 0.45, 0.0]}]
    intentions = {"walker": {"active": True, "target": [-0.2, -1.0, 0.0], "path": [[-0.2, 0.45], [-0.2, -1.0]]}}
    observed = [{"name": "door", "valid": True, "fully_open": False, "angular_speed_degps": 0.0}]
    coordinator = DoorCoordinator(grid, settings)
    coordinator.update(0.0, actors, intentions, [], observed)
    coordinator.update(1 / 60, actors, intentions, [], observed)
    actors[0]["position"][1] = 0.9
    observed[0]["fully_open"] = True
    coordinator.update(2 / 60, actors, intentions, [], observed)
    if invalidated == "context":
        intentions["walker"].update(target=[-0.2, 1.2, 0.0], path=[[-0.2, 0.9], [-0.2, 1.2]], route_context=(1, 0))
    else:
        intentions["walker"]["active"] = False
    result = coordinator.update(120.0, actors, intentions, [], observed)
    assert not result["door_commands"][0]["request_open"] and not result["objectives"] and not result["events"]
    coordinator.reset()
    assert not coordinator.doors["door"].requesters and not coordinator.doors["door"].logged_admissions


@pytest.mark.parametrize("entering", [False, True])
def test_room_membership_does_not_open_a_distant_portal_before_the_configured_approach(entering):
    grid, spec, agents, _, settings = _case(start=0.0)
    spec["interior_polygon_xy_m"] = [[-1.3, -8.0], [0.4, -8.0], [0.4, 0.0], [-1.3, 0.0]]
    actors = [{**agents[0], "position": [-0.2, 1.0 if entering else -6.0, 0.0]}]
    goal = [-0.2, -6.0 if entering else 2.0, 0.0]
    intentions = {"walker": {"active": True, "target": goal, "path": [actors[0]["position"][:2], goal[:2]]}}
    coordinator = DoorCoordinator(grid, settings)
    observation = [
        {"name": "door", "valid": True, "fully_open": False, "actual_angle_deg": 0.0, "angular_speed_degps": 0.0}
    ]
    coordinator.update(0.0, actors, intentions, [], observation)
    result = coordinator.update(1 / 60, actors, intentions, [], observation)
    assert not result["door_commands"][0]["request_open"]
    assert not result["objectives"]
    actors[0]["position"][1] = 0.45
    intentions["walker"]["target"] = [-0.2, -1.0, 0.0]
    intentions["walker"]["path"] = [[-0.2, 0.45], [-0.2, -1.0]]
    result = coordinator.update(2 / 60, actors, intentions, [], observation)
    assert result["door_commands"][0]["request_open"]
    # Once authorized, a physical opening is completed even if avoidance
    # subsequently takes its requester outside the initial trigger region.
    actors[0]["position"][1] = 0.9
    result = coordinator.update(3 / 60, actors, intentions, [], observation)
    assert result["door_commands"][0]["request_open"]
    assert result["objectives"]["walker"]["phase"] == "door_wait"
    observation[0]["fully_open"] = True
    for tick in (4, 5):
        result = coordinator.update(tick / 60, actors, intentions, [], observation)
        assert not result["objectives"]
        assert result["door_commands"][0]["hold"] and result["door_commands"][0]["request_open"]
    actors[0]["position"][1] = -0.2
    result = coordinator.update(6 / 60, actors, intentions, [], observation)
    assert result["door_commands"][0]["hold"] and result["door_commands"][0]["request_open"]
    actors[0]["position"][1] = -1.3
    coordinator.update(7 / 60, actors, intentions, [], observation)
    result = coordinator.update(8 / 60, actors, intentions, [], observation)
    assert not result["door_commands"][0]["request_open"]


def test_adjacent_portals_share_no_open_request_unless_the_planned_route_crosses_them():
    grid, spec, agents, _, settings = _case(start=0.0)
    cells = np.zeros((100, 100), dtype=np.int8)
    cells[60, :] = 100
    cells[60, 33:47] = cells[60, 49:63] = 0
    grid = OccupancyGrid(cells, 0.05, (-2.0, -3.0))
    spec["interior_polygon_xy_m"] = [[-1.0, -2.5], [1.5, -2.5], [1.5, 0.0], [-1.0, 0.0]]
    spec["doorway_segment_xy_m"] = [[-0.32, 0.0], [0.32, 0.0]]
    spec["swept_footprint_xy_m"] = [[-0.35, -0.4], [0.35, -0.4], [0.35, 0.03], [-0.35, 0.03]]
    other = copy.deepcopy(spec)
    other["name"], other["body_prim_path"] = "neighbor", "/neighbor_leaf"
    for field in ("doorway_segment_xy_m", "swept_footprint_xy_m"):
        other[field] = [[x + 0.8, y] for x, y in other[field]]
    settings["door_specs"].append(other)
    actors = [{**agents[0], "position": [0.4, 0.4, 0.0], "radius_m": 0.1}]
    intentions = {"walker": {"active": True, "target": [0.8, -2.0, 0.0], "path": [[0.4, 0.4], [0.8, 0.2], [0.8, -2.0]]}}
    observation = [
        {"name": name, "valid": True, "fully_open": False, "actual_angle_deg": 0.0, "angular_speed_degps": 0.0}
        for name in ("door", "neighbor")
    ]
    coordinator = DoorCoordinator(grid, settings)
    coordinator.update(0.0, actors, intentions, [], observation)
    result = coordinator.update(1 / 60, actors, intentions, [], observation)
    commands = {item["name"]: item for item in result["door_commands"]}
    assert not commands["door"]["request_open"] and commands["neighbor"]["request_open"]


def test_same_room_motion_and_other_floor_routes_do_not_request_an_unneeded_door():
    grid, spec, agents, _, settings = _case(start=0.0)
    spec["z_min_m"], spec["z_max_m"] = 0.0, 2.3
    coordinator = DoorCoordinator(grid, settings)
    observation = [
        {"name": "door", "valid": True, "fully_open": False, "actual_angle_deg": 0.0, "angular_speed_degps": 0.0}
    ]
    actors = [{**agents[0], "position": [-0.2, -0.45, 0.0]}]
    intentions = {"walker": {"active": True, "target": [-0.1, -0.3, 0.0], "path": [[-0.2, -0.45], [-0.1, -0.3]]}}
    for tick in range(2):
        result = coordinator.update(tick / 60, actors, intentions, [], observation)
        assert not result["door_commands"][0]["request_open"]
        assert not result["objectives"]
    # Passing parallel to a closed doorway does not request entry merely
    # because the body's padding is close to the leaf.
    actors[0].update(position=[-0.2, 0.25, 0.0], radius_m=0.25)
    intentions["walker"].update(target=[0.2, 0.25, 0.0], path=[[-0.2, 0.25], [0.2, 0.25]])
    result = coordinator.update(2 / 60, actors, intentions, [], observation)
    assert not result["door_commands"][0]["request_open"] and not result["objectives"]
    actors[0]["position"] = [-0.2, 0.45, 5.0]
    intentions["walker"].update(target=[-0.2, -1.0, 5.0], path=[[-0.2, 0.45], [-0.2, -1.0]])
    for tick in (3, 4):
        result = coordinator.update(tick / 60, actors, intentions, [], observation)
        assert not result["door_commands"][0]["request_open"]
    actors[0]["position"][2] = 0.0
    intentions["walker"]["arrival_tolerance_m"] = 0.2
    result = coordinator.update(5 / 60, actors, intentions, [], observation)
    assert not result["door_commands"][0]["request_open"] and not result["objectives"]


@pytest.mark.parametrize("actor_z, door_z", [(3.0, (0.0, 2.3)), (0.0, (3.0, 5.3))])
def test_a_closing_door_on_another_floor_does_not_block_navigation(actor_z, door_z):
    grid, spec, agents, plans, settings = _case(start=0.0)
    spec.update(z_min_m=door_z[0], z_max_m=door_z[1], physical_obstacle=True)
    settings["floor_height_m"] = actor_z
    agents[0].update(position=[-0.2, -0.6, actor_z], yaw=math.pi / 2, radius_m=0.25)
    plans[0].update(position=agents[0]["position"], speed=0.9)
    plans[0]["waypoints"] = [{"position": [-0.2, 0.8, actor_z], "wait": 0.0, "yaw": None}]
    observation = [
        {"name": "door", "valid": True, "fully_open": True, "actual_angle_deg": 90.0, "angular_speed_degps": 0.0}
    ]
    engine = NavigationEngine(plans, grid, settings, agents)
    states = agents
    try:
        for tick in range(2):
            states = engine.step(tick / 60, 1 / 60, states, door_states=observation)["agents"]
            assert states[0]["phase"] == "walking" and states[0]["error"] is None
        assert states[0]["position"][1] > agents[0]["position"][1]
        assert states[0]["position"][2] == actor_z
    finally:
        engine.close()


def _closing_sweep(height_fields):
    grid, spec, agents, _, settings = _case(start=0.0)
    spec.update(height_fields)
    actors = [{**agents[0], "position": [1.0, 1.0, 0.0], "radius_m": 0.25}]
    intentions = {"walker": {"active": False, "target": None}}
    observation = [
        {"name": "door", "valid": True, "fully_open": True, "actual_angle_deg": 90.0, "angular_speed_degps": 0.0}
    ]
    coordinator = DoorCoordinator(grid, settings)
    coordinator.update(0.0, actors, intentions, [], observation)
    result = coordinator.update(1 / 60, actors, intentions, [], observation)
    assert result["door_commands"] == [{"name": "door", "hold": False, "request_open": False}]
    assert len(result["obstacles"]) == 1
    return grid, result["obstacles"][0]


@pytest.mark.parametrize(
    "height_fields, center_z",
    [
        ({"z_min_m": 0.0, "z_max_m": 2.3}, 1.15),
        ({}, 0.0),
        ({"z_min_m": 0.0}, 0.0),
        ({"z_max_m": 2.3}, 0.0),
        ({"z_min_m": -math.inf, "z_max_m": math.inf}, 0.0),
    ],
)
def test_closing_sweep_preserves_declared_height_without_guessing_unknown_bounds(height_fields, center_z):
    _, obstacle = _closing_sweep(height_fields)
    assert {key: obstacle[key] for key in ("z_min_m", "z_max_m") if key in obstacle} == height_fields
    assert obstacle["position"][2] == center_z


@pytest.mark.parametrize("height_fields, actor_z", [({"z_min_m": 0.0, "z_max_m": 2.3}, 0.0), ({}, 3.0)])
def test_same_floor_and_unknown_height_closing_sweeps_still_block_an_intersecting_body(height_fields, actor_z):
    grid, obstacle = _closing_sweep(height_fields)
    agent = {
        "name": "walker",
        "position": [-0.2, -0.6, actor_z],
        "yaw": math.pi / 2,
        "radius_m": 0.25,
        "height_m": 1.8,
    }
    plan = {
        "name": "walker",
        "position": agent["position"],
        "behavior": "walk",
        "speed": 0.9,
        "arrival_tolerance": 0.2,
        "waypoints": [{"position": [-0.2, 0.8, actor_z]}],
    }
    engine = NavigationEngine([plan], grid, {"map_clearance_m": 0.03}, [agent])
    try:
        with pytest.raises(RuntimeError, match="Observed dynamic motion has no certified next step"):
            engine.step(0.0, 1 / 60, [agent], obstacles=[obstacle])
    finally:
        engine.close()


def test_a_nearby_body_skirt_around_the_jamb_still_requests_the_blocking_leaf():
    grid, _, agents, _, settings = _case(start=0.0)
    actors = [{**agents[0], "position": [-0.75, 0.3, 0.0]}]
    intentions = {"walker": {"active": True, "target": [-0.75, -1.0, 0.0], "path": [[-0.75, 0.3], [-0.75, -1.0]]}}
    observation = [
        {"name": "door", "valid": True, "fully_open": False, "actual_angle_deg": 0.0, "angular_speed_degps": 0.0}
    ]
    coordinator = DoorCoordinator(grid, settings)
    coordinator.update(0.0, actors, intentions, [], observation)
    result = coordinator.update(1 / 60, actors, intentions, [], observation)
    assert result["door_commands"][0]["request_open"]


def _leaf(angle):
    points = np.array([[-1.0, -0.02], [0.0, -0.02], [0.0, 0.02], [-1.0, 0.02]])
    c, s = math.cos(angle), math.sin(angle)
    points = points @ np.array([[c, s], [-s, c]]) + [0.3, 0.0]
    return {
        "prim_path": "/leaf",
        "position": [*np.mean(points, axis=0), 0.0],
        "radius_m": 0.51,
        "velocity": [0.0, 0.0, 0.0],
        "velocity_valid": True,
        "footprint_xy_m": points.tolist(),
    }


@pytest.mark.parametrize("transformed", [False, True])
def test_arrived_body_temporarily_clears_closing_sweep_and_returns_to_its_original_pose(transformed):
    grid, spec, actors, _, settings = _case(start=0.0)
    grid = OccupancyGrid(np.zeros((640, 640), dtype=np.int8), 0.025, (-8.0, -8.0))
    spec["closed_footprint_xy_m"] = [[-0.74, -0.04], [0.34, -0.04], [0.34, 0.04], [-0.74, 0.04]]
    rotation = np.array([[math.cos(0.37), -math.sin(0.37)], [math.sin(0.37), math.cos(0.37)]])
    offset = np.array([5.0, -1.0])

    def transform(points):
        return np.asarray(points) @ rotation.T + offset if transformed else np.asarray(points)

    for field in ("interior_polygon_xy_m", "swept_footprint_xy_m", "closed_footprint_xy_m", "doorway_segment_xy_m"):
        spec[field] = transform(spec[field]).tolist()
    origin = transform([actors[0]["position"][:2]])[0]
    actors[0]["position"] = [*origin, 0.0]
    intentions = {
        "walker": {
            "active": False,
            "can_door_refuge": True,
            "target": [*origin, 0.0],
            "arrival_tolerance_m": 0.1,
            "speed_mps": 0.6,
            "route_context": (0, 0),
            "phase": "dwelling",
        }
    }
    original = copy.deepcopy(intentions)
    coordinator = DoorCoordinator(grid, settings)
    angle = previous_angle = 90.0
    phases = set()
    closed_before_return = False
    for tick in range(900):
        now = tick / 60
        angular_speed = (angle - previous_angle) * 60
        observation = [
            {
                "name": "door",
                "valid": True,
                "fully_open": angle == 90 and abs(angular_speed) < 1,
                "fully_closed": angle == 0 and abs(angular_speed) < 1,
                "actual_angle_deg": angle,
                "angular_speed_degps": angular_speed,
            }
        ]
        leaf = _leaf(math.radians(angle))
        leaf["footprint_xy_m"] = transform(leaf["footprint_xy_m"]).tolist()
        leaf["position"] = [*np.mean(leaf["footprint_xy_m"], axis=0), 0.0]
        result = coordinator.update(now, actors, intentions, [leaf], observation)
        directive = result["objectives"].get("walker")
        before = np.asarray(actors[0]["position"][:2])
        if directive:
            phases.add(directive["phase"])
            assert np.linalg.norm(before - origin) <= 0.4 + 1e-8
            if directive["phase"] == "door_close_return":
                closed_before_return |= angle == 0 and observation[0]["fully_closed"]
                assert closed_before_return
            target = directive["target"]
            if target is not None:
                delta = np.asarray(target) - before
                distance = np.linalg.norm(delta)
                after = before + delta * min(1.0, (0.6 / 60) / distance) if distance else before
                assert grid.segment_clear(before, after, 0.18)
                actors[0]["position"] = [*after, 0.0]
        command = result["door_commands"][0]
        previous_angle = angle
        if not command["hold"]:
            assert not command["request_open"]
            assert _polygon_clearance(before, coordinator.doors["door"].sweep) > 0.18
            angle = max(0.0, angle - 180 / 60)
        if closed_before_return and not directive:
            break
    assert {"door_close_yield", "door_close_wait", "door_close_return"} <= phases
    assert closed_before_return and np.linalg.norm(np.asarray(actors[0]["position"][:2]) - origin) <= 1e-4
    assert intentions == original


@pytest.mark.parametrize("goal_yaw", [None, 90.0])
def test_engine_closing_refuge_preserves_original_heading_and_full_dwell(goal_yaw):
    grid, spec, actors, plans, settings = _case(start=0.0)
    spec["closed_footprint_xy_m"] = [[-0.74, -0.04], [0.34, -0.04], [0.34, 0.04], [-0.74, 0.04]]
    origin = list(actors[0]["position"])
    plans[0]["waypoints"] = [{"position": origin, "wait": 2.0, "yaw": goal_yaw}]
    original = copy.deepcopy(plans)
    engine = NavigationEngine(plans, grid, settings, actors)
    angle = previous_angle = 90.0
    phases = set()
    dwell_s = 0.0
    closed_before_return = False
    returned_yaw = None
    states = actors
    try:
        for tick in range(1200):
            observation = [
                {
                    "name": "door",
                    "valid": True,
                    "fully_open": angle == 90 and angle == previous_angle,
                    "fully_closed": angle == 0 and angle == previous_angle,
                    "actual_angle_deg": angle,
                    "angular_speed_degps": (angle - previous_angle) * 60,
                }
            ]
            result = engine.step(tick / 60, 1 / 60, states, [_leaf(math.radians(angle))], observation)
            state = result["agents"][0]
            assert state["error"] is None
            assert math.dist(state["position"], origin) <= 0.4 + 1e-7
            assert math.dist(state["position"], states[0]["position"]) <= 0.6 / 60 + 1e-7
            assert abs(engine._angle(state["yaw"] - states[0]["yaw"])) <= engine.turn_rate / 60 + 1e-7
            phases.add(state["phase"])
            if state["phase"] == "door_close_return":
                assert observation[0]["fully_closed"]
                closed_before_return = True
                returned_yaw = state["yaw"]
            if state["phase"] == "dwelling":
                assert math.dist(state["position"], origin) <= 1e-4
                expected_yaw = (
                    math.radians(goal_yaw)
                    if goal_yaw is not None
                    else actors[0]["yaw"]
                    if returned_yaw is None
                    else returned_yaw
                )
                assert abs(engine._angle(state["yaw"] - expected_yaw)) <= 1e-7
                dwell_s += 1 / 60
            command = result["door_commands"][0]
            previous_angle = angle
            if not command["hold"]:
                angle = min(90.0, angle + 3.0) if command["request_open"] else max(0.0, angle - 3.0)
            states = [{**actors[0], **state}]
            if state["phase"] == "complete":
                break
    finally:
        engine.close()
    assert closed_before_return
    assert {"door_close_yield", "door_close_wait", "door_close_return"} <= phases
    if goal_yaw is None:
        assert "door_close_restore" not in phases
        assert returned_yaw is not None and abs(engine._angle(returned_yaw - actors[0]["yaw"])) > 1e-7
        assert abs(engine._angle(states[0]["yaw"] - returned_yaw)) <= 1e-7
    else:
        assert "door_close_restore" in phases
        assert abs(engine._angle(states[0]["yaw"] - math.radians(goal_yaw))) <= 1e-7
    assert states[0]["phase"] == "complete" and dwell_s >= 2.0 - 1e-7
    assert plans == original


@pytest.mark.parametrize("sweep_clear_before_arrival", [False, True])
@pytest.mark.parametrize("wait_s", [0.0, 1.2])
def test_entered_room_observes_closed_leaf_before_authored_dwell_or_departure(sweep_clear_before_arrival, wait_s):
    grid, spec, actors, plans, settings = _case(start=0.0)
    spec["approach_distance_m"] = 0.9
    spec["closed_footprint_xy_m"] = [[-0.74, -0.04], [0.34, -0.04], [0.34, 0.04], [-0.74, 0.04]]
    actors[0].update(position=[-0.2, 0.45, 0.0], yaw=-math.pi / 2)
    plans[0].update(
        position=actors[0]["position"],
        waypoints=[
            {"position": [-0.2, -1.48 if sweep_clear_before_arrival else -1.14, 0.0], "wait": wait_s, "yaw": None},
            {"position": [-0.2, 0.8, 0.0], "wait": 0.0, "yaw": None},
        ],
    )
    original = copy.deepcopy(plans)
    engine = NavigationEngine(plans, grid, settings, actors)
    angle = previous_angle = 0.0
    closed_inside = False
    dwell_s = 0.0
    phases = set()
    state = actors[0]
    try:
        for tick in range(1200):
            observation = {
                "name": "door",
                "valid": True,
                "fully_open": angle == 90 and angle == previous_angle,
                "fully_closed": angle == 0 and angle == previous_angle,
                "actual_angle_deg": angle,
                "angular_speed_degps": (angle - previous_angle) * 60,
            }
            result = engine.step(tick / 60, 1 / 60, [state], [_leaf(math.radians(angle))], [observation])
            updated = result["agents"][0]
            assert updated["error"] is None
            if state.get("waypoint_index", 0) == 0:
                closed_inside |= observation["fully_closed"] and state["position"][1] < -0.6
            if updated["waypoint_index"] == 0:
                phases.add(updated["phase"])
                if updated["phase"] == "dwelling":
                    assert closed_inside and observation["fully_closed"]
                    dwell_s += 1 / 60
            else:
                assert closed_inside
            command = result["door_commands"][0]
            previous_angle = angle
            if not command["hold"]:
                angle = min(90.0, angle + 3.0) if command["request_open"] else max(0.0, angle - 3.0)
            state = {**state, **updated}
            if state["phase"] == "complete":
                break
    finally:
        engine.close()
    assert closed_inside and state["phase"] == "complete"
    assert "door_close_wait" in phases
    assert ("door_close_yield" in phases) is not sweep_clear_before_arrival
    assert ("door_close_return" in phases) is not sweep_clear_before_arrival
    assert dwell_s == pytest.approx(wait_s)
    assert plans == original


def test_a_queued_route_around_a_jamb_does_not_alternate_waiting_with_unauthorized_steps(monkeypatch):
    _, spec, actors, plans, settings = _case(start=0.0)
    spec.update(exclusive_entry=True, z_min_m=0.0, z_max_m=2.3)
    cells = np.zeros((160, 160), dtype=np.int8)
    rows, cols = np.indices(cells.shape)
    x, y = -2 + (cols + 0.5) * 0.025, -2 + (rows + 0.5) * 0.025
    cells[(y < 0) & ((x < -1.3) | (x > 0.4) | (y < -1.5))] = 100
    cells[(abs(y) < 0.04) & ((x < -0.7) | (x > 0.3))] = 100
    grid = OccupancyGrid(cells, 0.025, (-2.0, -2.0))
    actors[0]["position"] = [-0.65, 0.45, 0.0]
    actors.append({**actors[0], "name": "resident", "position": [-0.65, -0.9, 0.0]})
    plans[0].update(
        position=actors[0]["position"],
        waypoints=[
            {"position": actors[0]["position"], "wait": 0.1, "yaw": None},
            {"position": [-0.65, -0.5, 0.0], "wait": 0.0, "yaw": None},
        ],
    )
    plans.append({"name": "resident", "position": actors[1]["position"], "behavior": "stand"})
    engine = NavigationEngine(plans, grid, settings, actors)
    monkeypatch.setattr(engine._goals, "update", lambda *args, **kwargs: {"objectives": {}, "status": {}})
    states, anchor = actors, None
    observation = [
        {
            "name": "door",
            "valid": True,
            "fully_open": False,
            "fully_closed": True,
            "actual_angle_deg": 0.0,
            "angular_speed_degps": 0.0,
        }
    ]
    try:
        for tick in range(180):
            result = engine.step(tick / 60, 1 / 60, states, [_leaf(0.0)], observation)
            state = result["agents"][0]
            assert not state["error"] and not result["door_commands"][0]["request_open"]
            if state["phase"] == "door_wait" and anchor is None:
                anchor = list(state["position"])
            if anchor is not None:
                assert state["phase"] == "door_wait"
                assert state["position"] == anchor and state["velocity"] == [0.0, 0.0, 0.0]
            states = [{**before, **after} for before, after in zip(states, result["agents"])]
    finally:
        engine.close()
    assert anchor is not None


def test_a_goal_overlapping_the_closed_leaf_never_authorizes_forced_closing():
    grid, spec, actors, _, settings = _case(start=0.0)
    spec["closed_footprint_xy_m"] = [[-0.74, -0.04], [0.34, -0.04], [0.34, 0.04], [-0.74, 0.04]]
    actors[0]["position"] = [-0.3, -0.03, 0.0]
    intentions = {
        "walker": {
            "active": False,
            "can_door_refuge": True,
            "target": [-0.3, -0.03, 0.0],
            "arrival_tolerance_m": 0.1,
            "speed_mps": 0.6,
            "route_context": (0, 0),
        }
    }
    observed = [
        {"name": "door", "valid": True, "fully_open": True, "actual_angle_deg": 90.0, "angular_speed_degps": 0.0}
    ]
    coordinator = DoorCoordinator(grid, settings)
    for tick in range(3):
        result = coordinator.update(tick / 60, actors, intentions, [], observed)
        assert result["door_commands"][0]["hold"]
        assert not result["objectives"]
        assert "closed" in result["closure_status"]["blocked"]["walker"]["reason"]


def test_optional_closing_holds_open_when_the_original_arrival_pose_has_no_bounded_refuge(monkeypatch):
    grid, spec, actors, _, settings = _case(start=0.0)
    spec["closed_footprint_xy_m"] = [[-0.74, -0.04], [0.34, -0.04], [0.34, 0.04], [-0.74, 0.04]]
    actors[0]["position"] = [-0.457554, -0.462758, 0.0]
    intentions = {
        "walker": {
            "active": False,
            "can_door_refuge": True,
            "target": [-0.65, -0.5, 0.0],
            "arrival_tolerance_m": 0.2,
            "speed_mps": 0.6,
        }
    }
    observation = [
        {"name": "door", "valid": True, "fully_open": True, "actual_angle_deg": 90.0, "angular_speed_degps": 0.0}
    ]
    coordinator = DoorCoordinator(grid, settings)
    original = copy.deepcopy(actors)
    original_intentions = copy.deepcopy(intentions)
    search_calls = []

    def forbidden_search(*args, **kwargs):
        search_calls.append(1)
        raise AssertionError("The proved geometric lower bound should skip candidate search.")

    monkeypatch.setattr(coordinator, "_refuge", forbidden_search)
    for tick in range(60):
        result = coordinator.update(tick / 60, actors, intentions, [_leaf(math.pi / 2)], observation)
        assert not result["objectives"] and result["door_commands"][0]["hold"]
        assert not result["closure_status"]["transactions"]
        blocked = result["closure_status"]["blocked"]["walker"]
        assert blocked["reason"] == "no_refuge" and blocked["minimum_displacement_m"] > 0.4424
    assert not search_calls and actors == original and intentions == original_intentions


def test_a_mandatory_opening_refuge_still_fails_when_no_bounded_candidate_exists():
    grid, _, actors, _, settings = _case(start=0.0)
    actors[0]["position"] = [0.0, -0.5, 0.0]
    intentions = {"walker": {"active": True, "target": [0.0, 0.8, 0.0], "arrival_tolerance_m": 0.1, "speed_mps": 0.6}}
    observed = [
        {"name": "door", "valid": True, "fully_open": False, "actual_angle_deg": 0.0, "angular_speed_degps": 0.0}
    ]
    coordinator = DoorCoordinator(grid, settings)
    with pytest.raises(RuntimeError, match="No certified continuous door refuge within 0.4 m"):
        coordinator.update(0.0, actors, intentions, [_leaf(0)], observed)


def test_walk_continues_inside_the_arrival_ball_when_its_first_point_cannot_clear_the_closing_sweep():
    grid, spec, actors, plans, settings = _case(start=0.0)
    spec["closed_footprint_xy_m"] = [[-0.74, -0.04], [0.34, -0.04], [0.34, 0.04], [-0.74, 0.04]]
    first = [-0.457554, -0.462758, 0.0]
    goal = [-0.65, -0.5, 0.0]
    actors[0]["position"] = first.copy()
    plans[0].update(position=first.copy(), waypoints=[{"position": goal, "wait": 2.0, "yaw": 90.0}])
    original = copy.deepcopy(plans)
    engine = NavigationEngine(plans, grid, settings, actors)
    sweep = engine._doors.doors["door"].sweep
    radius = actors[0]["radius_m"] + settings["map_clearance_m"]
    assert math.dist(first, goal) < plans[0]["arrival_tolerance"]
    assert radius - _polygon_clearance(first, sweep) > engine._doors.max_retreat_m

    angle = previous_angle = 90.0
    states = actors
    closure_started = False
    dwell_ticks = 0
    try:
        for tick in range(600):
            observation = [
                {
                    "name": "door",
                    "valid": True,
                    "fully_open": angle == 90 and angle == previous_angle,
                    "fully_closed": angle == 0 and angle == previous_angle,
                    "actual_angle_deg": angle,
                    "angular_speed_degps": (angle - previous_angle) * 60,
                }
            ]
            result = engine.step(tick / 60, 1 / 60, states, [_leaf(math.radians(angle))], observation)
            state = result["agents"][0]
            assert state["error"] is None
            assert state["waypoint_index"] == 0
            if state["phase"] == "dwelling":
                assert observation[0]["fully_closed"]
                assert math.dist(state["position"], goal) <= 1e-4
                assert state["yaw"] == pytest.approx(math.pi / 2, abs=0.025)
                dwell_ticks += 1
            command = result["door_commands"][0]
            previous_angle = angle
            if not command["hold"] and not command["request_open"]:
                assert _polygon_clearance(states[0]["position"], sweep) > radius
                closure_started = True
                angle = max(0.0, angle - 3.0)
            states = [{**actors[0], **state}]
            if state["phase"] == "complete":
                break
    finally:
        engine.close()
    assert closure_started and dwell_ticks >= 120
    assert state["phase"] == "complete" and plans == original


def test_optional_closing_never_hides_a_colliding_actual_pose_behind_the_geometric_bound():
    coordinator, actors, intentions, observed = _concurrent_closure_case()
    actors[0]["position"] = [0.0, -0.5, 0.0]
    intentions["walker"]["target"] = [0.0, -0.5, 0.0]
    obstacle = {"position": [0.0, -0.5, 0.0], "radius_m": 0.1}
    with pytest.raises(RuntimeError, match="actual body pose"):
        coordinator.update(0.0, actors, intentions, [obstacle], observed)


def test_optional_candidate_exhaustion_is_throttled_without_starting_a_wait_transaction(monkeypatch):
    coordinator, actors, intentions, observed = _concurrent_closure_case()
    calls = []

    def unavailable(*args, **kwargs):
        calls.append(1)
        raise _NoCertifiedRefuge("No certified candidate under the current dynamic blockers.")

    monkeypatch.setattr(coordinator, "_refuge", unavailable)
    for now in (0.0, 1 / 60, 0.25, 0.49):
        result = coordinator.update(now, actors, intentions, [], observed)
        assert not result["objectives"] and not result["closure_status"]["transactions"]
        assert result["closure_status"]["blocked"]["walker"]["reason"] == "no_refuge"
    assert len(calls) == 1
    coordinator.update(0.5, actors, intentions, [], observed)
    assert len(calls) == 2


def test_context_invalidation_and_reset_cannot_reuse_an_old_closing_objective():
    grid, spec, actors, _, settings = _case(start=0.0)
    spec["closed_footprint_xy_m"] = [[-0.74, -0.04], [0.34, -0.04], [0.34, 0.04], [-0.74, 0.04]]
    intentions = {
        "walker": {
            "active": False,
            "can_door_refuge": True,
            "target": list(actors[0]["position"]),
            "arrival_tolerance_m": 0.1,
            "speed_mps": 0.6,
            "route_context": (0, 0),
        }
    }
    observed = [
        {"name": "door", "valid": True, "fully_open": True, "actual_angle_deg": 90.0, "angular_speed_degps": 0.0}
    ]
    coordinator = DoorCoordinator(grid, settings)
    assert (
        coordinator.update(0.0, actors, intentions, [], observed)["objectives"]["walker"]["phase"] == "door_close_yield"
    )
    intentions["walker"]["can_door_refuge"] = False
    result = coordinator.update(1 / 60, actors, intentions, [], observed)
    assert not result["objectives"] and not result["closure_status"]["transactions"]
    intentions["walker"]["can_door_refuge"] = True
    assert coordinator.update(2 / 60, actors, intentions, [], observed)["objectives"]
    coordinator.reset()
    intentions["walker"].update(target=[-0.65, -1.0, 0.0], route_context=(1, 0))
    result = coordinator.update(0.0, actors, intentions, [], observed)
    assert not result["objectives"] and not result["closure_status"]["transactions"]


def _concurrent_closure_case(*, outer_swing=False):
    grid, spec, actors, _, settings = _case(start=0.0)
    spec["closed_footprint_xy_m"] = [[-0.74, -0.04], [0.34, -0.04], [0.34, 0.04], [-0.74, 0.04]]
    if outer_swing:
        spec["swept_footprint_xy_m"][2][1] = spec["swept_footprint_xy_m"][3][1] = 0.6
    intentions = {
        "walker": {
            "active": False,
            "can_door_refuge": True,
            "target": list(actors[0]["position"]),
            "arrival_tolerance_m": 0.1,
            "speed_mps": 0.6,
            "route_context": (0, 0),
        }
    }
    observed = [
        {"name": "door", "valid": True, "fully_open": True, "actual_angle_deg": 90.0, "angular_speed_degps": 0.0}
    ]
    return DoorCoordinator(grid, settings), actors, intentions, observed


def test_closing_owner_queues_a_new_entry_until_the_bounded_transaction_releases():
    coordinator, actors, intentions, observed = _concurrent_closure_case()
    actors.append({**actors[0], "name": "entrant", "position": [-0.2, 0.45, 0.0]})
    intentions["entrant"] = {
        "active": True,
        "target": [-0.2, -1.2, 0.0],
        "route_context": (0, 0),
        "arrival_tolerance_m": 0.1,
        "speed_mps": 0.6,
    }
    queued_position = list(actors[1]["position"])
    first = coordinator.update(0.0, actors, intentions, [], observed)
    assert first["objectives"]["entrant"]["phase"] == "door_wait"
    assert first["objectives"]["entrant"]["target"] is None
    assert not first["door_commands"][0]["request_open"]
    actors[0]["position"] = [*first["objectives"]["walker"]["target"], 0.0]
    for now in (0.5, 0.6):
        result = coordinator.update(now, actors, intentions, [], observed)
        assert result["objectives"]["entrant"]["phase"] == "door_wait"
        assert not result["door_commands"][0]["request_open"]
    assert not result["door_commands"][0]["hold"]
    observed[0].update(fully_open=False, fully_closed=True, actual_angle_deg=0.0)
    result = coordinator.update(1.2, actors, intentions, [], observed)
    assert result["objectives"]["walker"]["phase"] == "door_close_return"
    actors[0]["position"] = list(intentions["walker"]["target"])
    result = coordinator.update(1.8, actors, intentions, [], observed)
    assert not result["closure_status"]["transactions"]
    assert actors[1]["position"] == queued_position
    # The closure gate has a definite end; physical occupancy can still hold
    # the leaf, but the next entrant is again eligible for ordinary admission.
    assert "entrant" in coordinator.doors["door"].requesters


def test_a_genuine_exit_defers_the_closing_owners_return_into_the_door_sweep():
    coordinator, actors, intentions, observed = _concurrent_closure_case()
    first = coordinator.update(0.0, actors, intentions, [], observed)
    refuge = first["objectives"]["walker"]["target"]
    actors[0]["position"] = [*refuge, 0.0]
    coordinator.update(0.5, actors, intentions, [], observed)
    coordinator.update(0.6, actors, intentions, [], observed)
    observed[0].update(fully_open=False, fully_closed=True, actual_angle_deg=0.0)
    assert (
        coordinator.update(1.2, actors, intentions, [], observed)["objectives"]["walker"]["phase"]
        == "door_close_return"
    )
    actors.append({**actors[0], "name": "exiter", "position": [0.2, -0.25, 0.0]})
    intentions["exiter"] = {
        "active": True,
        "target": [0.05, 0.8, 0.0],
        "route_context": (0, 0),
        "arrival_tolerance_m": 0.1,
        "speed_mps": 0.6,
    }
    result = coordinator.update(1.3, actors, intentions, [], observed)
    assert result["objectives"]["walker"]["phase"] in ("door_close_yield", "door_close_wait")
    assert "exiter" in coordinator.doors["door"].requesters


def test_existing_outgoing_access_keeps_priority_after_retreating_beyond_the_initial_approach():
    coordinator, actors, intentions, observed = _concurrent_closure_case()
    intentions["walker"]["can_door_refuge"] = False
    actors.append({**actors[0], "name": "exiter", "position": [0.05, -0.45, 0.0]})
    intentions["exiter"] = {
        "active": True,
        "target": [0.05, 0.8, 0.0],
        "route_context": (0, 0),
        "arrival_tolerance_m": 0.1,
        "speed_mps": 0.6,
    }
    coordinator.update(0.0, actors, intentions, [], observed)
    actors[1]["position"] = [0.05, -0.7, 0.0]
    intentions["walker"]["can_door_refuge"] = True
    result = coordinator.update(0.5, actors, intentions, [], observed)
    assert result["door_commands"][0]["request_open"]
    assert result["objectives"].get("exiter", {}).get("phase") != "door_wait"


def test_queued_entrant_keeps_clearing_an_outer_sweep_after_crossing_the_approach_distance():
    coordinator, actors, intentions, observed = _concurrent_closure_case(outer_swing=True)
    actors.append({**actors[0], "name": "entrant", "position": [-0.2, 0.5, 0.0]})
    intentions["entrant"] = {
        "active": True,
        "target": [-0.2, -1.2, 0.0],
        "route_context": (0, 0),
        "arrival_tolerance_m": 0.1,
        "speed_mps": 0.6,
    }
    first = coordinator.update(0.0, actors, intentions, [], observed)
    assert first["objectives"]["entrant"]["phase"] == "door_yield"
    assert first["objectives"]["entrant"]["target"][1] > 0.78
    actors[1]["position"] = [-0.2, 0.56, 0.0]
    result = coordinator.update(0.1, actors, intentions, [], observed)
    assert result["objectives"]["entrant"]["phase"] == "door_yield"
    assert result["objectives"]["entrant"]["target"] == first["objectives"]["entrant"]["target"]
    assert not result["door_commands"][0]["request_open"]


@pytest.mark.parametrize("exclusive", [False, True])
def test_declared_exclusive_entry_blocks_new_entry_while_a_resident_dwells(exclusive):
    grid, spec, actors, _, settings = _case(start=0.0)
    spec.update(exclusive_entry=exclusive, z_min_m=0.0, z_max_m=2.3)
    actors[0]["position"] = [-0.65, -1.3, 0.0]
    actors.append({**actors[0], "name": "entrant", "position": [0.05, 0.45, 0.0]})
    intentions = {
        "walker": {"active": False, "phase": "dwelling", "target": list(actors[0]["position"])},
        "entrant": {
            "active": True,
            "target": [0.05, -1.1, 0.0],
            "arrival_tolerance_m": 0.1,
            "speed_mps": 0.6,
            "route_context": (0, 0),
        },
    }
    observed = [
        {
            "name": "door",
            "valid": True,
            "fully_open": False,
            "fully_closed": True,
            "actual_angle_deg": 0.0,
            "angular_speed_degps": 0.0,
        }
    ]
    coordinator = DoorCoordinator(grid, settings)
    for tick in range(3):
        result = coordinator.update(tick / 60, actors, intentions, [], observed)
    assert result["door_commands"][0]["request_open"] is not exclusive
    if exclusive:
        assert result["objectives"]["entrant"]["phase"] == "door_wait"
        assert result["objectives"]["entrant"]["safety_critical"] is False
        assert result["objectives"]["entrant"]["queued_door_entry"] is True
        assert result["objectives"]["entrant"]["yield_origin_xy_m"] == actors[1]["position"][:2]
        assert not coordinator.doors["door"].requesters
        requests = coordinator.admission_intentions(actors, intentions, ["entrant"])
        assert requests["entrant"]["active"] is False
        deferred = coordinator.update(3 / 60, actors, requests, [], observed)
        assert deferred["objectives"]["entrant"]["queued_door_entry"]
        assert not deferred["objectives"]["entrant"]["safety_critical"]
        coordinator.reset()
        actors[1]["position"][1] = 0.9
        distant = coordinator.update(0.0, actors, intentions, [], observed)
        assert not distant["objectives"] and not distant["door_commands"][0]["request_open"]
        actors[0]["position"] = [-1.7, 1.5, 0.0]
        actors[1]["position"][1] = 0.45
        coordinator.reset()
        for tick in range(2):
            cleared = coordinator.update(tick / 60, actors, intentions, [], observed)
        assert cleared["door_commands"][0]["request_open"]


@pytest.mark.parametrize("other_floor", [False, True])
def test_an_existing_resident_never_suppresses_a_true_exit_and_other_floors_do_not_occupy(other_floor):
    grid, spec, actors, _, settings = _case(start=0.0)
    spec.update(exclusive_entry=True, z_min_m=0.0, z_max_m=2.3)
    actors[0]["position"] = [-0.65, -1.3, 5.0 if other_floor else 0.0]
    actors.append({**actors[0], "name": "exiter", "position": [0.05, -0.45, 0.0]})
    intentions = {
        "walker": {"active": False, "phase": "dwelling", "target": list(actors[0]["position"])},
        "exiter": {
            "active": True,
            "target": [0.05, 0.8, 0.0],
            "arrival_tolerance_m": 0.1,
            "speed_mps": 0.6,
            "route_context": (0, 0),
        },
    }
    observed = [
        {"name": "door", "valid": True, "fully_open": True, "actual_angle_deg": 90.0, "angular_speed_degps": 0.0}
    ]
    coordinator = DoorCoordinator(grid, settings)
    result = coordinator.update(0.0, actors, intentions, [], observed)
    assert result["door_commands"][0]["request_open"] and not result["objectives"]
    if other_floor:
        actors[1]["position"] = [0.05, 0.45, 0.0]
        intentions["exiter"]["target"] = [0.05, -1.1, 0.0]
        intentions["exiter"]["route_context"] = (1, 0)
        result = coordinator.update(1 / 60, actors, intentions, [], observed)
        assert result["door_commands"][0]["request_open"] and not result["objectives"]


def test_goal_defer_query_preserves_real_far_exits_without_opening_a_far_portal():
    grid, spec, actors, _, settings = _case(start=0.0)
    spec.update(exclusive_entry=True, z_min_m=0.0, z_max_m=2.3)
    actors[0]["position"] = [-0.65, -1.3, 0.0]
    intentions = {
        "walker": {
            "active": True,
            "target": [-0.2, 0.8, 0.0],
            "route_context": (0, 0),
            "arrival_tolerance_m": 0.1,
            "speed_mps": 0.6,
        }
    }
    original = copy.deepcopy(intentions)
    coordinator = DoorCoordinator(grid, settings)
    requests = coordinator.admission_intentions(actors, intentions, ["walker"])
    assert requests["walker"]["active"] and intentions == original
    observation = [
        {"name": "door", "valid": True, "fully_open": False, "actual_angle_deg": 0.0, "angular_speed_degps": 0.0}
    ]
    result = coordinator.update(0.0, actors, requests, [], observation)
    assert result["outgoing_actors"] == ["walker"]
    assert not result["door_commands"][0]["request_open"] and not result["objectives"]
    actors[0]["position"][2] = 5.0
    intentions["walker"]["target"][2] = 5.0
    requests = coordinator.admission_intentions(actors, intentions, ["walker"])
    result = coordinator.update(1 / 60, actors, requests, [], observation)
    assert not result["outgoing_actors"] and not result["door_commands"][0]["request_open"]


def test_goal_defer_query_respects_a_valid_outgoing_lease_and_rejects_its_stale_context():
    grid, _, actors, _, settings = _case(start=0.0)
    actors[0]["position"] = [-0.65, -0.45, 0.0]
    intentions = {
        "walker": {
            "active": True,
            "target": [-0.2, 1.5, 0.0],
            "route_context": (0, 0),
            "arrival_tolerance_m": 0.1,
            "speed_mps": 0.6,
        }
    }
    observed = [
        {"name": "door", "valid": True, "fully_open": True, "actual_angle_deg": 90.0, "angular_speed_degps": 0.0}
    ]
    coordinator = DoorCoordinator(grid, settings)
    coordinator.update(0.0, actors, intentions, [], observed)
    actors[0]["position"] = [-0.2, 0.9, 0.0]
    requests = coordinator.admission_intentions(actors, intentions, ["walker"])
    assert requests["walker"]["active"]
    result = coordinator.update(1.0, actors, requests, [], observed)
    assert "walker" in result["outgoing_actors"]
    intentions["walker"].update(target=[-0.2, -1.0, 0.0], route_context=(1, 0))
    requests = coordinator.admission_intentions(actors, intentions, ["walker"])
    assert not requests["walker"]["active"]
    result = coordinator.update(2.0, actors, requests, [], observed)
    assert not result["outgoing_actors"]


def test_an_admitted_clear_opening_wait_is_critical_before_and_during_motion():
    grid, _, actors, _, settings = _case(start=0.0)
    actors[0]["position"][1] = -0.45
    intentions = {
        "walker": {
            "active": True,
            "target": [-0.2, 0.8, 0.0],
            "route_context": (0, 0),
            "arrival_tolerance_m": 0.1,
            "speed_mps": 0.6,
        }
    }
    observed = [
        {"name": "door", "valid": True, "fully_open": False, "actual_angle_deg": 0.0, "angular_speed_degps": 0.0}
    ]
    coordinator = DoorCoordinator(grid, settings)
    first = coordinator.update(0.0, actors, intentions, [], observed)
    actors[0]["position"] = [*first["objectives"]["walker"]["target"], 0.0]
    for now in (0.5, 0.6):
        result = coordinator.update(now, actors, intentions, [], observed)
        assert result["objectives"]["walker"]["phase"] == "door_wait"
        assert result["objectives"]["walker"]["safety_critical"] is True
        assert not result["objectives"]["walker"].get("queued_door_entry", False)


def test_a_cleared_retreat_reenters_the_measured_sweep_without_renewing_its_budget():
    grid, _, actors, _, settings = _case(start=0.0)
    actors[0]["position"] = [-0.2, 0.21, 0.0]
    intentions = {
        "walker": {
            "active": True,
            "target": [-0.2, -1.0, 0.0],
            "route_context": (0, 0),
            "arrival_tolerance_m": 0.1,
            "speed_mps": 0.6,
        }
    }
    observed = [
        {"name": "door", "valid": True, "fully_open": False, "actual_angle_deg": 0.0, "angular_speed_degps": 0.0}
    ]
    coordinator = DoorCoordinator(grid, settings)
    first = coordinator.update(0.0, actors, intentions, [_leaf(0)], observed)
    initial = first["objectives"]["walker"]
    retreat = coordinator.retreats["walker"]
    deadline = retreat.started + retreat.timeout
    actors[0]["position"] = [*initial["target"], 0.0]
    cleared = coordinator.update(0.1, actors, intentions, [_leaf(0)], observed)
    assert cleared["objectives"]["walker"]["phase"] == "door_wait"
    actors[0]["position"] = [-0.2, 0.2287, 0.0]
    reentered = coordinator.update(0.2, actors, intentions, [_leaf(0)], observed)
    assert reentered["objectives"]["walker"]["phase"] == "door_yield"
    assert reentered["objectives"]["walker"]["origin"] == initial["origin"]
    assert reentered["objectives"]["walker"]["target"] == initial["target"]
    assert coordinator.retreats["walker"] is retreat
    assert reentered["door_commands"][0]["hold"]
    with pytest.raises(RuntimeError, match="Door refuge timed out"):
        coordinator.update(deadline + 0.001, actors, intentions, [_leaf(0)], observed)


@pytest.mark.parametrize("obstacle_horizon", [None, 0.5])
def test_exit_retreat_holds_the_door_until_two_measured_clear_frames_then_resumes_original_route(obstacle_horizon):
    grid, spec, agents, plans, settings = _case()
    if obstacle_horizon is not None:
        settings["obstacle_time_horizon_s"] = obstacle_horizon
    original = copy.deepcopy(plans)
    engine = NavigationEngine(plans, grid, settings, agents)
    states = agents
    angle = previous_angle = 0.0
    previous_clear = False
    saw_yield = saw_open = saw_small_temporary_remaining = False
    reached_at = None
    for tick in range(1200):
        now = tick / 60
        angular_speed = math.degrees((angle - previous_angle) * 60)
        full = abs(angle - math.pi / 2) < 1e-8 and abs(angular_speed) < 5.0
        observation = {
            "name": "door",
            "valid": True,
            "fully_open": full,
            "actual_angle_deg": math.degrees(angle),
            "angular_speed_degps": angular_speed,
        }
        before = copy.deepcopy(states[0])
        result = engine.step(now, 1 / 60, states, [_leaf(angle)], [observation])
        state = result["agents"][0]
        command = result["door_commands"][0]
        assert state["error"] is None
        if now < 1.0:
            assert state["position"] == agents[0]["position"] and state["yaw"] == 0.0
            assert command["hold"] and not command["request_open"]
        actual_clear = _polygon_clearance(before["position"], engine._doors.doors["door"].sweep) > 0.18 + 1e-6
        if command["request_open"] and not full:
            assert actual_clear and previous_clear and not command["hold"]
        previous_clear = actual_clear
        if state["phase"] == "door_yield":
            saw_yield = True
            assert state["waypoint_index"] == 0 and state["completed_cycles"] == 0
            assert math.dist(state["position"], agents[0]["position"]) <= 0.4 + 1e-7
            refuge = engine._doors.retreats["walker"].target
            if math.dist(state["position"][:2], refuge) < 0.2 and not actual_clear:
                saw_small_temporary_remaining = True
                assert command["hold"] and not command["request_open"]
        if full:
            saw_open = True
        if math.dist(state["position"], [-0.2, 0.8, 0.0]) <= 0.2 and reached_at is None:
            reached_at = now
        if state["phase"] == "complete":
            assert reached_at is not None and now - reached_at >= 0.2 - 1 / 60 - 1e-8
            break
        previous_angle = angle
        if not command["hold"]:
            wanted = math.pi / 2 if command["request_open"] else 0.0
            angle += float(np.clip(wanted - angle, -math.pi / 0.8 / 60, math.pi / 0.8 / 60))
        states = [state]
    else:
        pytest.fail("Door transaction never resumed and completed the original route")
    assert saw_yield and saw_open and saw_small_temporary_remaining
    assert plans == original and engine.routes["walker"].index == 0


def test_no_refuge_inside_the_bounded_displacement_fails_without_moving_the_body():
    grid, spec, agents, plans, settings = _case(start=0.0)
    spec["swept_footprint_xy_m"] = [[-1.5, -1.5], [1.5, -1.5], [1.5, 1.5], [-1.5, 1.5]]
    agents[0]["position"] = [0.0, -0.4, 0.0]
    plans[0]["position"] = agents[0]["position"]
    engine = NavigationEngine(plans, grid, settings, agents)
    original = copy.deepcopy(agents)
    with pytest.raises(RuntimeError, match="refuge within 0.4"):
        engine.step(
            0.0,
            1 / 60,
            agents,
            [_leaf(0.0)],
            [{"name": "door", "valid": True, "fully_open": False, "angular_speed_degps": 0.0, "actual_angle_deg": 0.0}],
        )
    assert agents == original and engine.routes["walker"].index == 0


def test_an_unresponsive_door_has_a_bounded_observed_open_timeout():
    grid, spec, agents, plans, settings = _case(start=0.0)
    agents[0]["position"] = [-0.2, 0.45, 0.0]
    plans[0]["position"] = agents[0]["position"]
    plans[0]["waypoints"][0]["position"] = [-0.2, -1.2, 0.0]
    engine = NavigationEngine(plans, grid, settings, agents)
    states = agents
    with pytest.raises(RuntimeError, match="observed opening endpoint"):
        for tick in range(400):
            state = engine.step(
                tick / 60,
                1 / 60,
                states,
                [_leaf(0.0)],
                [
                    {
                        "name": "door",
                        "valid": True,
                        "fully_open": False,
                        "angular_speed_degps": 0.0,
                        "actual_angle_deg": 0.0,
                    }
                ],
            )["agents"][0]
            assert state["position"] == agents[0]["position"]
            states = [state]


def test_float32_collapsed_polygon_edge_does_not_create_a_nonfinite_rvo_boundary():
    grid = OccupancyGrid(np.zeros((100, 100), dtype=np.int8), 0.1, (-5.0, -5.0))
    agents = [{"name": "walker", "position": [0.0, 0.0, 0.0], "yaw": math.pi / 2, "radius_m": 0.2, "height_m": 1.8}]
    plans = [
        {
            "name": "walker",
            "position": [0.0, 0.0, 0.0],
            "behavior": "walk",
            "speed": 0.6,
            "waypoints": [{"position": [0.0, 2.0, 0.0]}],
        }
    ]
    polygon = [[1.0, -1.0], [2.0 - 3.8e-9, -1.0], [2.0, -1.0 + 3.8e-9], [2.0, 1.0], [1.0, 1.0]]
    obstacle = {
        "position": [1.5, 0.0, 0.0],
        "radius_m": 1.2,
        "velocity": [0.0, 0.0, 0.0],
        "velocity_valid": True,
        "footprint_xy_m": polygon,
    }
    engine = NavigationEngine(plans, grid, {}, agents)
    state = engine.step(0.0, 1 / 60, agents, [obstacle])["agents"][0]
    assert np.isfinite([*state["position"], *state["velocity"], state["yaw"]]).all()
    assert state["error"] is None and state["position"][1] > 0


def _opening_refuge_case():
    grid = OccupancyGrid(np.zeros((100, 100), dtype=np.int8), 0.05, (-2.5, -2.5))
    spec = {
        "name": "door",
        "body_prim_path": "/leaf",
        "enabled": True,
        "open_angle_deg": 90.0,
        "closed_angle_deg": 0.0,
        "target_ramp_duration_s": 0.4,
        "approach_distance_m": 0.55,
        "interior_polygon_xy_m": [[-1.3, -2.0], [0.4, -2.0], [0.4, -0.5], [-1.3, -0.5]],
        "doorway_segment_xy_m": [[-0.7, -0.5], [0.3, -0.5]],
        "swept_footprint_xy_m": [[-0.72, -1.55], [0.35, -1.55], [0.35, 0.1], [-0.72, 0.1]],
        "closed_footprint_xy_m": [[-0.7, -0.52], [0.3, -0.52], [0.3, -0.48], [-0.7, -0.48]],
    }
    actor = {
        "name": "walker",
        "position": [0.0, 0.0, 0.0],
        "yaw": 0.0,
        "velocity": [0.0, 0.0, 0.0],
        "radius_m": 0.25,
        "height_m": 1.8,
    }
    plan = {
        "name": "walker",
        "position": actor["position"].copy(),
        "behavior": "walk",
        "speed": 0.9,
        "arrival_tolerance": 0.005,
        "waypoints": [{"position": [0.0, -1.0, 0.0], "wait": 0.0, "yaw": None}],
    }
    obstacles = [
        {
            "prim_path": "/leaf",
            "position": [-0.2, -0.5, 0.0],
            "velocity": [0.0, 0.0, 0.0],
            "velocity_valid": True,
            "radius_m": 0.51,
            "footprint_xy_m": spec["closed_footprint_xy_m"],
        }
    ]
    doors = [
        {
            "name": "door",
            "valid": True,
            "fully_open": False,
            "fully_closed": True,
            "actual_angle_deg": 0.0,
            "angular_speed_degps": 0.0,
        }
    ]
    settings = {"map_clearance_m": 0.03, "door_specs": [spec]}
    intentions = {"walker": {"active": True, "target": [0.0, -1.0, 0.0], "speed_mps": plan["speed"]}}
    return grid, [actor], [plan], settings, intentions, obstacles, doors


def test_opening_refuge_yaw_is_monotone_until_two_observed_clear_frames_authorize_the_door():
    fixture_dir = Path(__file__).with_name("fixtures")
    snapshot = json.loads((fixture_dir / "native_opening_refuge_20261106.json").read_text())
    with np.load(fixture_dir / snapshot["provenance"]["map_fixture"], allow_pickle=False) as data:
        grid = OccupancyGrid(data["occupancy"], float(data["resolution_m"]), tuple(data["origin_xy_m"]))
    plans, settings = snapshot["plans"], snapshot["settings"]
    original = copy.deepcopy(plans)
    states = copy.deepcopy(snapshot["actors"])
    focus = snapshot["focus"]
    origin = next(state["position"] for state in states if state["name"] == focus)
    sweeps = {
        spec["name"]: convex_footprint(spec["swept_footprint_xy_m"])
        for spec in settings["door_specs"]
        if spec.get("enabled", True)
    }
    focus_state = next(state for state in states if state["name"] == focus)
    clearance = focus_state["radius_m"] + settings["map_clearance_m"]
    relevant = {
        name for name, polygon in sweeps.items() if not segment_polygon_clear(origin, origin, polygon, clearance)
    }
    previous = None
    signed = variation = 0.0
    targets = set()
    dt = 1 / 60
    engine = NavigationEngine(plans, grid, settings, states)
    try:
        for tick in range(360):
            result = engine.step(
                snapshot["time_s"] + tick * dt, dt, states, snapshot["obstacles"], snapshot["door_states"]
            )
            before = {state["name"]: state for state in states}
            after = {state["name"]: state for state in result["agents"]}
            for name, state in after.items():
                start, end = np.asarray(before[name]["position"][:2]), np.asarray(state["position"][:2])
                body_clearance = before[name]["radius_m"] + settings["map_clearance_m"]
                assert grid.segment_clear(start, end, body_clearance)
                assert (
                    math.dist(start, end) <= next(plan["speed"] for plan in plans if plan["name"] == name) * dt + 1e-7
                )
                assert abs(math.remainder(state["yaw"] - before[name]["yaw"], 2 * math.pi)) <= 2.5 * dt + 1e-7
                for obstacle in snapshot["obstacles"]:
                    relative_end = end - np.asarray(obstacle["velocity"][:2]) * dt
                    assert segment_polygon_clear(
                        start, relative_end, convex_footprint(obstacle["footprint_xy_m"]), body_clearance
                    )
                for peer in before:
                    if peer <= name:
                        continue
                    relative = start - np.asarray(before[peer]["position"][:2])
                    change = end - start - (np.asarray(after[peer]["position"][:2]) - before[peer]["position"][:2])
                    fraction = (
                        np.clip(-np.dot(relative, change) / np.dot(change, change), 0, 1)
                        if np.dot(change, change)
                        else 0
                    )
                    required = before[name]["radius_m"] + before[peer]["radius_m"] + settings["pair_clearance_m"]
                    assert np.linalg.norm(relative + fraction * change) >= required - 1e-7
            if result["guidance"][focus]["phase"] == "door_yield":
                change = math.remainder(after[focus]["yaw"] - before[focus]["yaw"], 2 * math.pi)
                signed += change
                variation += abs(change)
                targets.add(tuple(result["guidance"][focus]["target"]))
                assert math.dist(after[focus]["position"], origin) <= 0.4 + 1e-7
            commands = [
                command
                for command in result["door_commands"]
                if command["name"] in relevant and command["request_open"]
            ]
            if commands:
                assert previous is not None
                for command in commands:
                    for observed in (before, previous):
                        assert all(
                            segment_polygon_clear(
                                state["position"],
                                state["position"],
                                sweeps[command["name"]],
                                state["radius_m"] + settings["map_clearance_m"],
                            )
                            for state in observed.values()
                        )
                break
            previous = before
            states = [{**before[state["name"]], **state} for state in result["agents"]]
        else:
            pytest.fail("The measured-clear opening request did not complete within the original refuge budget")
    finally:
        engine.close()
    assert len(targets) == 1 and variation > 0
    assert variation - abs(signed) <= settings["max_turn_rate_radps"] * dt
    assert plans == original


@pytest.mark.parametrize(
    ("kind", "velocity", "known", "ray"),
    [
        ("leaf", [0.0, 0.05, 0.0], True, False),
        ("leaf", [0.0, -0.05, 0.0], True, True),
        ("disc", [-0.1, 0.0, 0.0], True, False),
        ("disc", [0.1, 0.0, 0.0], True, True),
        ("disc", [0.0, 0.0, 0.0], False, False),
    ],
)
def test_opening_ray_requires_the_entire_remaining_external_sweep_including_its_own_leaf(kind, velocity, known, ray):
    grid, actors, _, settings, intentions, obstacles, doors = _opening_refuge_case()
    if kind == "leaf":
        obstacles[0].update(velocity=velocity, velocity_valid=known)
    else:
        obstacles.append(
            {
                "prim_path": "/external",
                "position": [0.8, 0.2, 0.0],
                "radius_m": 0.1,
                "velocity": velocity,
                "velocity_valid": known,
            }
        )
    result = DoorCoordinator(grid, settings).update(0.0, actors, intentions, obstacles, doors)
    objective = result["objectives"]["walker"]
    assert objective["phase"] == "door_yield"
    assert objective.get("certified_ray", False) is ray
    assert not result["door_commands"][0]["request_open"]


def test_opening_ray_does_not_renew_its_deadline_at_zero_remaining_time():
    grid, actors, _, settings, intentions, obstacles, doors = _opening_refuge_case()
    coordinator = DoorCoordinator(grid, settings)
    first = coordinator.update(0.0, actors, intentions, obstacles, doors)["objectives"]["walker"]
    assert first.get("certified_ray") is True
    deadline = 1.6 + 0.4 / intentions["walker"]["speed_mps"] + 4.0
    last = coordinator.update(deadline, actors, intentions, obstacles, doors)["objectives"]["walker"]
    assert last["target"] == first["target"] and last["origin"] == first["origin"]
    assert not last.get("certified_ray", False)
    with pytest.raises(RuntimeError, match="Door refuge timed out"):
        coordinator.update(deadline + 1 / 60, actors, intentions, obstacles, doors)


def test_opening_refuge_preserves_rvo_escape_from_an_approaching_external_wall():
    grid, actors, plans, settings, _, obstacles, doors = _opening_refuge_case()
    moving = {
        "prim_path": "/moving_wall",
        "position": [0.0, -0.3051, 0.0],
        "radius_m": 1.01,
        "footprint_xy_m": [[-1.0, -0.3301], [1.0, -0.3301], [1.0, -0.2801], [-1.0, -0.2801]],
        "velocity": [0.0, 0.018, 0.0],
        "velocity_valid": True,
    }
    obstacles.append(moving)
    engine = NavigationEngine(plans, grid, settings, actors)
    try:
        result = engine.step(0.0, 1 / 60, actors, obstacles, doors)
    finally:
        engine.close()
    state = result["agents"][0]
    assert result["guidance"]["walker"]["phase"] == "door_yield"
    start, end = np.asarray(actors[0]["position"][:2]), np.asarray(state["position"][:2])
    assert np.linalg.norm(end - start) > 0
    assert np.linalg.norm(state["velocity"][:2]) <= plans[0]["speed"] + 1e-7
    assert abs(math.remainder(state["yaw"] - actors[0]["yaw"], 2 * math.pi)) <= 2.5 / 60 + 1e-7
    assert np.linalg.norm(end - start) <= 0.4
    relative_end = end - np.asarray(moving["velocity"][:2]) / 60
    assert segment_polygon_clear(start, relative_end, convex_footprint(moving["footprint_xy_m"]), 0.28)
