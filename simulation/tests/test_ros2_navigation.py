# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent occupancy navigation and reciprocal crowd execution contracts."""

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pedstrain_simulation.simulation.ros2_navigation.planner import (
    OccupancyGrid,
    convex_footprint,
    segment_polygon_clear,
)
from pedstrain_simulation.simulation.ros2_navigation.solver import NavigationEngine, _RvoSolver


def test_occupancy_uses_ros_rows_and_routes_around_inflated_obstacles():
    cells = np.zeros((60, 80), dtype=np.int8)
    cells[10:50, 38:42] = 100
    grid = OccupancyGrid(cells, 0.1, (-4.0, -3.0))
    assert grid.world_to_cell((-3.85, -2.75)) == (2, 1)
    path = grid.plan((-3.0, 0.0), (3.0, 0.0), 0.2, 0.15)
    assert math.dist(path[-1], (3.0, 0.0)) <= 0.15
    assert all(grid.segment_clear(a, b, 0.25) for a, b in zip(path, path[1:]))
    assert any(abs(point[1]) > 2.0 for point in path)


@pytest.mark.parametrize("narrow", [False, True])
def test_path_clearance_preference_survives_lookahead_and_yields_to_narrow_geometry(narrow):
    cells = np.zeros((60, 80), dtype=np.int8)
    if narrow:
        cells[:27] = cells[33:] = 100
    else:
        cells[13:47, 38:42] = 100
    grid = OccupancyGrid(cells, 0.1, (-4.0, -3.0))
    start, goal = [-3.0, 0.0, 0.0], [3.0, 0.0, 0.0]
    agent = {"name": "walker", "position": start, "yaw": 0.0, "radius_m": 0.2, "height_m": 1.8}
    plan = {
        "name": "walker",
        "position": start,
        "behavior": "walk",
        "speed": 0.9,
        "arrival_tolerance": 0.1,
        "waypoints": [{"position": goal}],
    }
    engine = NavigationEngine([plan], grid, {"map_clearance_m": 0.03, "preferred_path_clearance_m": 0.1}, [agent])
    try:
        command = engine.step(0.0, 1 / 60, [agent])
        guide = command["guidance"]["walker"]["target"]
        assert guide is not None and grid.segment_clear(start, guide, 0.23)
        if narrow:
            assert guide == goal and not grid.segment_clear(start, guide, 0.33)
        else:
            hard_path = grid.plan(start, goal, 0.2, 0.1, 0.03)
            assert not grid.segment_clear(start, hard_path[1], 0.33)
            assert grid.segment_clear(start, guide, 0.33)
        assert command["agents"][0]["error"] is None
    finally:
        engine.close()


def test_fine_grid_detour_bounds_full_swept_certificate_work(monkeypatch):
    cells = np.zeros((100, 160), dtype=np.int8)
    cells[16:84, 78:82] = 100
    grid = OccupancyGrid(cells, 0.05, (-4.0, -2.5))
    certify = grid.segment_clear
    calls = 0

    def count_certificates(*args):
        nonlocal calls
        calls += 1
        return certify(*args)

    monkeypatch.setattr(grid, "segment_clear", count_certificates)
    path = grid.plan((-3.0, 0.0), (3.0, 0.0), 0.2, 0.2)
    assert math.dist(path[-1], (3.0, 0.0)) <= 0.2
    assert all(certify(a, b, 0.25) for a, b in zip(path, path[1:]))
    # Full geometric certificates should scale with the final route, rather
    # than every explored neighbor in this fine raster.
    assert calls < 500


def _two_body_swept_spacing(before, after):
    start = np.asarray(before[0]["position"][:2]) - before[1]["position"][:2]
    end = np.asarray(after[0]["position"][:2]) - after[1]["position"][:2]
    change = end - start
    squared = float(np.dot(change, change))
    at = float(np.clip(-np.dot(start, change) / squared, 0, 1)) if squared else 0.0
    return np.linalg.norm(start + at * change)


def test_scheduled_parked_peer_is_passed_before_its_authored_release():
    path = Path(__file__).with_name("fixtures") / "native_parked_follower_20261109.npz"
    with np.load(path, allow_pickle=False) as fixture:
        grid = OccupancyGrid(fixture["occupancy"], float(fixture["resolution_m"]), tuple(fixture["origin_xy_m"]))
        agents = json.loads(str(fixture["agents"]))
        plans = json.loads(str(fixture["plans"]))
        settings = json.loads(str(fixture["settings"]))
    settings.update(map_clearance_m=0.0275, pair_clearance_m=0.225)
    original = copy.deepcopy(plans)
    engine = NavigationEngine(plans, grid, settings, agents)
    states, dt = agents, 1 / 60
    peer_origin = np.asarray(agents[1]["position"][:2])
    axis = np.asarray(plans[0]["waypoints"][0]["position"][:2]) - agents[0]["position"][:2]
    axis /= np.linalg.norm(axis)
    try:
        for tick in range(round(plans[1]["start_time"] / dt)):
            next_states = engine.step(tick * dt, dt, states)["agents"]
            follower, peer = next_states
            assert follower["error"] is None and peer["error"] is None
            assert peer["position"] == agents[1]["position"]
            speed = math.hypot(*follower["velocity"][:2])
            assert math.dist(states[0]["position"], follower["position"]) <= plans[0]["speed"] * dt + 1e-6
            assert (
                abs(math.remainder(follower["yaw"] - states[0]["yaw"], math.tau))
                <= settings["max_turn_rate_radps"] * dt + 1e-6
            )
            if tick * dt < plans[0]["start_time"]:
                assert follower["position"] == agents[0]["position"] and speed == 0.0
            if follower["phase"] == "following_wait":
                assert follower["position"] == states[0]["position"] and speed == 0.0
            clearance = agents[0]["radius_m"] + settings["map_clearance_m"]
            assert grid.segment_clear(states[0]["position"], follower["position"], clearance)
            required = agents[0]["radius_m"] + agents[1]["radius_m"] + settings["pair_clearance_m"]
            assert _two_body_swept_spacing(states, next_states) >= required - 1e-9
            states = [{**state, **command} for state, command in zip(states, next_states)]
        # Forward clearance past the stationary body, rather than just a brief speed burst.
        assert np.dot(np.asarray(follower["position"][:2]) - peer_origin, axis) > agents[0]["radius_m"]
        assert plans == original
    finally:
        engine.close()


def test_scheduled_parked_peer_in_single_file_corridor_is_not_pushed():
    cells = np.full((50, 160), 100, dtype=np.int8)
    cells[18:32, :] = 0
    grid = OccupancyGrid(cells, 0.05, (-4.0, -1.25))
    actors = [
        {"name": name, "position": [x, 0.0, 0.0], "yaw": 0.0, "radius_m": 0.25, "height_m": 1.8}
        for name, x in (("follower", -2.0), ("peer", -1.0))
    ]
    plans = [
        {
            "name": actor["name"],
            "position": actor["position"],
            "behavior": "walk",
            "speed": 0.9,
            "start_time": start_time,
            "waypoints": [{"position": [goal, 0.0, 0.0]}],
        }
        for actor, start_time, goal in zip(actors, (0.0, 3.0), (1.5, 3.0))
    ]
    settings = {"map_clearance_m": 0.0275, "pair_clearance_m": 0.225}
    required = sum(actor["radius_m"] for actor in actors) + settings["pair_clearance_m"]
    engine = NavigationEngine(plans, grid, settings, actors)
    states, dt = actors, 1 / 60
    before_release_x = actors[0]["position"][0]
    try:
        for tick in range(round(6.0 / dt)):
            next_states = engine.step(tick * dt, dt, states)["agents"]
            follower, peer = next_states
            assert follower["error"] is None and peer["error"] is None
            if tick * dt < plans[1]["start_time"]:
                assert peer["position"] == actors[1]["position"]
                assert follower["position"][0] < peer["position"][0]
                before_release_x = follower["position"][0]
            assert all(
                grid.segment_clear(
                    before["position"], after["position"], actor["radius_m"] + settings["map_clearance_m"]
                )
                for before, after, actor in zip(states, next_states, actors)
            )
            assert _two_body_swept_spacing(states, next_states) >= required - 1e-9
            states = [{**state, **command} for state, command in zip(states, next_states)]
        assert follower["position"][0] > before_release_x + actors[0]["radius_m"]
    finally:
        engine.close()


def test_random_raster_routes_retain_exact_swept_disk_certificates():
    rng = np.random.default_rng(20261103)
    reached = 0
    for _ in range(40):
        cells = np.zeros((32, 42), dtype=np.int8)
        for _ in range(8):
            row, column = int(rng.integers(2, 27)), int(rng.integers(2, 37))
            cells[row : row + int(rng.integers(1, 5)), column : column + int(rng.integers(1, 5))] = rng.choice(
                [-1, 100]
            )
        grid = OccupancyGrid(cells, 0.1, (-2.1, -1.6))
        radius = float(rng.choice([0.02, 0.09, 0.18]))
        clearance = radius + 0.03
        free = np.argwhere(grid.free_mask(clearance))
        selected = rng.choice(len(free), size=2, replace=False)
        start, goal = [grid.cell_center(tuple(free[index])) for index in selected]
        try:
            path = grid.plan(start, goal, radius, 0.15, margin_m=0.03)
        except ValueError:
            continue
        reached += 1
        assert math.dist(path[-1], goal) <= 0.15
        assert all(grid.segment_clear(a, b, clearance) for a, b in zip(path, path[1:]))
    assert reached >= 30


def test_disconnected_map_does_not_project_a_goal_outside_its_original_tolerance():
    cells = np.zeros((40, 60), dtype=np.int8)
    cells[:, 29:31] = 100
    grid = OccupancyGrid(cells, 0.1, (-3.0, -2.0))
    with pytest.raises(ValueError, match="reachable"):
        grid.plan((-2.0, 0.0), (2.0, 0.0), 0.2, 0.2)


def test_goal_projection_limit_caps_astar_endpoint_without_changing_the_original_arrival_ball():
    cells = np.zeros((50, 50), dtype=np.int8)
    cells[:, 20:25] = 100
    start, goal = (2.05, 0.05), (0.15, 0.05)
    grid = OccupancyGrid(cells, 0.1, (-2.5, -2.5), goal_projection_tolerance_m=0.16)
    path = grid.plan(start, goal, 0.25, 0.3, 0.03)
    assert math.dist(path[-1], goal) <= 0.16
    strict = OccupancyGrid(cells, 0.1, (-2.5, -2.5), goal_projection_tolerance_m=0.1)
    with pytest.raises(ValueError, match="configured goal projection tolerance"):
        strict.plan(start, goal, 0.25, 0.3, 0.03)
    assert strict.plan(path[-1], goal, 0.25, 0.3, 0.03) == [path[-1], path[-1]]


def test_goal_projection_uses_continuous_free_space_inside_a_subcell_arrival_ball():
    grid = OccupancyGrid(np.zeros((10, 10), dtype=np.int8), 0.1, (0.0, 0.0))
    path = grid.plan((0.5, 0.5), (0.05, 0.5), 0.1, 0.09, margin_m=0.0)
    assert math.dist(path[-1], (0.05, 0.5)) <= 0.09
    assert all(grid.segment_clear(a, b, 0.1) for a, b in zip(path, path[1:]))


def test_alternating_boundary_goals_reuse_certificates_and_evict_least_recent_goal(monkeypatch):
    grid = OccupancyGrid(np.zeros((20, 30), dtype=np.int8), 0.05, (0.0, 0.0))
    start, first, second = (1.0, 0.5), (0.01, 0.3), (0.01, 0.7)
    radius, tolerance = 0.08, 0.16
    visits = {first: 0, second: 0}
    near_cells = grid._near_cells

    def count_endpoint_search(point, distance, mask):
        if point in visits:
            visits[point] += 1
        return near_cells(point, distance, mask)

    monkeypatch.setattr(grid, "_near_cells", count_endpoint_search)
    first_route = grid.plan(start, first, radius, tolerance, margin_m=0.0)
    second_route = grid.plan(start, second, radius, tolerance, margin_m=0.0)
    assert grid.plan(start, first, radius, tolerance, margin_m=0.0) == first_route
    assert visits == {first: 1, second: 1}

    for index in range(15):
        grid.plan(start, (0.02, 0.21 + index * 0.037), radius, tolerance, margin_m=0.0)

    assert grid.plan(start, second, radius, tolerance, margin_m=0.0) == second_route
    assert grid.plan(start, first, radius, tolerance, margin_m=0.0) == first_route
    assert visits == {first: 2, second: 2}


def test_repeated_boundary_goal_plans_match_fresh_grids_when_targets_or_clearance_change():
    cells = np.zeros((60, 80), dtype=np.int8)
    grid = OccupancyGrid(cells, 0.02, (0.0, 0.0))
    queries = [
        ((1.0, 0.5), (0.01, 0.5), 0.05, 0.2, 0.01),
        ((0.9, 0.6), (0.01, 0.5), 0.05, 0.2, 0.01),
        ((1.0, 0.5), (0.02, 0.55), 0.05, 0.16, 0.01),
        ((1.0, 0.5), (0.01, 0.5), 0.05, 0.2, 0.03),
        ((1.0, 0.5), (0.01, 0.5), 0.05, 0.2, 0.01),
    ]
    for start, goal, radius, tolerance, margin in queries:
        expected = OccupancyGrid(cells, 0.02, (0.0, 0.0)).plan(start, goal, radius, tolerance, margin)
        actual = grid.plan(start, goal, radius, tolerance, margin)
        assert actual == expected
        assert math.dist(actual[-1], goal) <= tolerance
        assert all(grid.segment_clear(a, b, radius + margin) for a, b in zip(actual, actual[1:]))


def test_goal_endpoint_may_occupy_a_safe_part_of_a_blocked_center_cell():
    grid = OccupancyGrid(np.zeros((30, 40), dtype=np.int8), 0.05, (0.0, 0.0))
    goal = (0.09968, 0.7)
    path = grid.plan((1.0, 0.7), goal, 0.25, 0.2, margin_m=0.03)
    assert math.dist(path[-1], goal) <= 0.2
    assert all(grid.segment_clear(a, b, 0.28) for a, b in zip(path, path[1:]))
    assert not grid.free_mask(0.28)[grid.world_to_cell(path[-1])]


def test_independent_obstacle_horizon_avoids_a_long_creep_into_the_original_arrival_ball():
    cells = np.zeros((100, 200), dtype=np.int8)
    cells[:, :20] = 100
    grid = OccupancyGrid(cells, 0.02, (-2.0, -1.0))
    agents = [{"name": "walker", "position": [-0.8, 0.0, 0.0], "yaw": math.pi, "radius_m": 0.25, "height_m": 1.8}]
    goal = [-1.515, 0.0, 0.0]
    plans = [
        {
            "name": "walker",
            "position": agents[0]["position"],
            "behavior": "walk",
            "speed": 0.9,
            "arrival_tolerance": 0.2,
            "waypoints": [{"position": goal}],
        }
    ]
    engine = NavigationEngine(
        plans, grid, {"map_clearance_m": 0.03, "time_horizon_s": 2.5, "obstacle_time_horizon_s": 0.5}, agents
    )
    states = agents
    longest = current = 0.0
    for tick in range(1200):
        state = engine.step(tick / 60, 1 / 60, states)["agents"][0]
        assert state["error"] is None
        assert grid.segment_clear(states[0]["position"], state["position"], 0.28)
        assert math.dist(states[0]["position"], state["position"]) <= 0.9 / 60 + 1e-8
        current = current + 1 / 60 if state["phase"] == "walking" and math.hypot(*state["velocity"][:2]) < 0.03 else 0.0
        longest = max(longest, current)
        states = [state]
        if state["phase"] == "complete":
            break
    assert state["phase"] == "complete" and math.dist(state["position"], goal) <= 0.2
    assert longest <= 1.0


@pytest.mark.parametrize(
    "distance,heading_deg,speed,turn_rate,hz,bystander",
    [
        (0.03, 60.0, 0.9, 2.5, 60, False),
        (0.03, 90.0, 0.9, 2.5, 60, False),
        (0.03, 180.0, 0.9, 2.5, 60, False),
        (0.1, 90.0, 0.9, 1.5, 30, False),
        (0.1, 180.0, 0.3, 4.0, 120, False),
        (0.3, 60.0, 0.2, 1.2, 60, False),
        (0.1, 90.0, 0.9, 2.5, 60, True),
        (2.0, 0.0, 0.9, 2.5, 60, False),
        (0.03, 0.0, 0.9, 2.5, 60, (0.78, 0.0)),
        (0.03, 90.0, 0.9, 2.5, 60, (0.78, 0.02)),
    ],
)
def test_near_goal_arrival_does_not_circle_while_preserving_motion_limits(
    distance, heading_deg, speed, turn_rate, hz, bystander
):
    grid = OccupancyGrid(np.zeros((120, 120), dtype=np.int8), 0.05, (-3.0, -3.0))
    radius, margin, body_gap, tolerance = 0.25, 0.03, 0.25, 0.005
    actor = {
        "name": "walker",
        "position": [0.0, 0.0, 0.0],
        "yaw": math.radians(heading_deg),
        "velocity": [0.0, 0.0, 0.0],
        "radius_m": radius,
        "height_m": 1.8,
    }
    goal = [distance, 0.0, 0.0]
    original_goal = tuple(goal)
    plan = {
        "name": actor["name"],
        "position": actor["position"],
        "behavior": "walk",
        "speed": speed,
        "arrival_tolerance": tolerance,
        "waypoints": [{"position": goal, "wait": 0.0, "yaw": None}],
    }
    agents, plans = [actor], [plan]
    if bystander:
        peer_xy = bystander if isinstance(bystander, tuple) else (0.0, 1.25)
        agents.append({**actor, "name": "standing", "position": [*peer_xy, 0.0], "yaw": 0.0})
        plans.append({"name": "standing", "position": agents[1]["position"], "behavior": "stand"})
    engine = NavigationEngine(
        plans, grid, {"map_clearance_m": margin, "pair_clearance_m": body_gap, "max_turn_rate_radps": turn_rate}, agents
    )
    states, dt, total_turn = agents, 1 / hz, 0.0
    initial_deviation = abs(math.remainder(actor["yaw"], 2 * math.pi))
    try:
        for tick in range(20 * hz):
            next_states = engine.step(tick * dt, dt, states)["agents"]
            for before, after in zip(states, next_states):
                assert after["error"] is None
                assert math.hypot(*after["velocity"][:2]) <= speed + 1e-6
                assert math.dist(before["position"], after["position"]) <= speed * dt + 1e-6
                assert grid.segment_clear(before["position"], after["position"], radius + margin)
                turn = math.remainder(after["yaw"] - before["yaw"], 2 * math.pi)
                assert abs(turn) <= turn_rate * dt + 1e-5
                if math.hypot(*after["velocity"][:2]) > 0.0:
                    motion_heading = math.atan2(after["velocity"][1], after["velocity"][0])
                    assert abs(math.remainder(after["yaw"] - motion_heading, 2 * math.pi)) < 1e-6
                if after["name"] == actor["name"]:
                    total_turn += abs(turn)
            if bystander:
                assert next_states[1]["position"] == agents[1]["position"]
                relative = np.asarray(states[0]["position"][:2]) - states[1]["position"][:2]
                change = np.asarray(next_states[0]["position"][:2]) - states[0]["position"][:2]
                squared_length = float(np.dot(change, change))
                fraction = float(np.clip(-np.dot(relative, change) / squared_length, 0, 1)) if squared_length else 0.0
                assert np.linalg.norm(relative + fraction * change) - 2 * radius >= body_gap - 1e-5
            if heading_deg == 0.0 and not bystander:
                assert abs(next_states[0]["position"][1]) <= 1e-5
                assert abs(math.remainder(next_states[0]["yaw"], 2 * math.pi)) <= 1e-4
            states = next_states
            if states[0]["phase"] == "complete":
                break
        else:
            pytest.fail("The original near goal was not reached within 20 simulation seconds")
        assert math.dist(states[0]["position"], original_goal) <= tolerance
        assert tuple(plan["waypoints"][0]["position"]) == original_goal and plan["arrival_tolerance"] == tolerance
        assert total_turn <= initial_deviation + math.pi / 2 + 1e-5
    finally:
        engine.close()


def test_arrival_with_forward_momentum_does_not_accumulate_repeated_heading_corrections():
    grid = OccupancyGrid(np.zeros((120, 120), dtype=np.int8), 0.05, (-3.0, -3.0))
    distance, bearing = 0.3, math.radians(20.0)
    radius, margin, gap, speed, turn_rate, dt = 0.25, 0.03, 0.25, 0.9, 2.5, 1 / 60
    goal = [distance * math.cos(bearing), distance * math.sin(bearing), 0.0]
    peer = [goal[0] + 2 * radius + gap, goal[1] + 0.02, 0.0]
    actors = [
        {
            "name": "walker",
            "position": [0.0, 0.0, 0.0],
            "velocity": [0.2, 0.0, 0.0],
            "yaw": 0.0,
            "radius_m": radius,
            "height_m": 1.8,
        },
        {
            "name": "standing",
            "position": peer,
            "velocity": [0.0, 0.0, 0.0],
            "yaw": 0.0,
            "radius_m": radius,
            "height_m": 1.8,
        },
    ]
    plans = [
        {
            "name": "walker",
            "position": actors[0]["position"],
            "behavior": "walk",
            "speed": speed,
            "arrival_tolerance": 0.005,
            "waypoints": [{"position": goal, "wait": 0.0, "yaw": None}],
        },
        {"name": "standing", "position": peer, "behavior": "stand"},
    ]
    original = copy.deepcopy(plans)
    engine = NavigationEngine(plans, grid, {"map_clearance_m": margin, "pair_clearance_m": gap}, actors)
    states, total_turn = actors, 0.0
    try:
        for tick in range(1200):
            next_states = engine.step(tick * dt, dt, states)["agents"]
            for before, after in zip(states, next_states):
                assert after["error"] is None
                velocity = math.hypot(*after["velocity"][:2])
                assert velocity <= speed + 1e-6
                assert math.dist(before["position"], after["position"]) <= speed * dt + 1e-6
                assert grid.segment_clear(before["position"], after["position"], radius + margin)
                turn = abs(math.remainder(after["yaw"] - before["yaw"], 2 * math.pi))
                assert turn <= turn_rate * dt + 1e-5
                if velocity > 0.0:
                    heading = math.atan2(after["velocity"][1], after["velocity"][0])
                    assert abs(math.remainder(after["yaw"] - heading, 2 * math.pi)) < 1e-6
                if after["name"] == "walker":
                    total_turn += turn
            assert next_states[1]["position"] == peer
            relative = np.asarray(states[0]["position"][:2]) - peer[:2]
            motion = np.asarray(next_states[0]["position"][:2]) - states[0]["position"][:2]
            squared = float(np.dot(motion, motion))
            closest = float(np.clip(-np.dot(relative, motion) / squared, 0, 1)) if squared else 0.0
            assert np.linalg.norm(relative + closest * motion) >= 2 * radius + gap - 1e-12
            states = [{**before, **after} for before, after in zip(states, next_states)]
            if states[0]["phase"] == "complete":
                break
        assert states[0]["phase"] == "complete"
        assert math.dist(states[0]["position"], goal) <= plans[0]["arrival_tolerance"]
        assert total_turn < 2 * math.pi
        assert plans == original
    finally:
        engine.close()


@pytest.mark.parametrize("quarter_turns", [0, 1, 2, 3])
def test_clear_departure_next_to_a_wall_does_not_choose_the_long_turn(quarter_turns):
    cells = np.zeros((120, 120), dtype=np.int8)
    cells[:54] = 100
    grid = OccupancyGrid(np.rot90(cells, -quarter_turns), 0.05, (-3.0, -3.0))
    angle = quarter_turns * math.pi / 2
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    goal = [*(rotation @ [1.0, 0.4]), 0.0]
    actor = {
        "name": "walker",
        "position": [0.0, 0.0, 0.0],
        "yaw": math.radians(-143.0) + angle,
        "radius_m": 0.25,
        "height_m": 1.8,
    }
    plan = {
        "name": actor["name"],
        "position": actor["position"],
        "behavior": "walk",
        "speed": 0.9,
        "arrival_tolerance": 0.05,
        "waypoints": [{"position": goal}],
    }
    original = copy.deepcopy(plan)
    assert grid.segment_clear(actor["position"], goal, 0.28)
    initial_turn = abs(math.remainder(math.atan2(goal[1], goal[0]) - actor["yaw"], 2 * math.pi))
    engine = NavigationEngine([plan], grid, {"map_clearance_m": 0.03}, [actor])
    states, total_turn, dt = [actor], 0.0, 1 / 60
    try:
        for tick in range(600):
            state = engine.step(tick * dt, dt, states)["agents"][0]
            assert state["error"] is None
            assert grid.segment_clear(states[0]["position"], state["position"], 0.28)
            assert math.dist(states[0]["position"], state["position"]) <= plan["speed"] * dt + 1e-6
            turn = abs(math.remainder(state["yaw"] - states[0]["yaw"], 2 * math.pi))
            assert turn <= engine.turn_rate * dt + 1e-5
            total_turn += turn
            if math.hypot(*state["velocity"][:2]) > 0:
                heading = math.atan2(state["velocity"][1], state["velocity"][0])
                assert abs(math.remainder(state["yaw"] - heading, 2 * math.pi)) < 1e-6
            states = [{**actor, **state}]
            if state["phase"] == "complete":
                break
        assert state["phase"] == "complete"
        assert math.dist(state["position"], goal) <= plan["arrival_tolerance"]
        assert total_turn <= initial_turn + math.pi / 4
        assert plan == original
    finally:
        engine.close()


def test_far_parking_turn_finishes_before_first_translation():
    grid = OccupancyGrid(np.zeros((120, 120), dtype=np.int8), 0.05, (-3.0, -3.0))
    actor = {"name": "walker", "position": [0.0, 0.0, 0.0], "yaw": math.pi, "radius_m": 0.25, "height_m": 1.8}
    goal = (2.0, 0.0, 0.0)
    plan = {
        "name": actor["name"],
        "position": actor["position"],
        "behavior": "walk",
        "speed": 0.9,
        "arrival_tolerance": 0.05,
        "waypoints": [{"position": list(goal), "yaw": None}],
    }
    engine = NavigationEngine([plan], grid, {"map_clearance_m": 0.03}, [actor])
    dt = 1 / 60
    try:
        first = engine.step(0.0, dt, [actor])["agents"][0]
        assert first["error"] is None and first["position"] == actor["position"]
        assert first["yaw"] != actor["yaw"]
        assert abs(math.remainder(first["yaw"] - actor["yaw"], 2 * math.pi)) <= 2.5 * dt + 1e-5
        previous = first
        for tick in range(1, 240):
            state = engine.step(tick * dt, dt, [previous])["agents"][0]
            assert state["error"] is None
            assert abs(math.remainder(state["yaw"] - previous["yaw"], 2 * math.pi)) <= 2.5 * dt + 1e-5
            assert math.dist(previous["position"], state["position"]) <= 0.9 * dt + 1e-6
            assert grid.segment_clear(previous["position"], state["position"], 0.28)
            if math.dist(state["position"], actor["position"]) > 1e-7:
                assert abs(math.remainder(state["yaw"], 2 * math.pi)) <= 1e-6
                assert math.hypot(*state["velocity"][:2]) <= 0.9 + 1e-6
                break
            assert state["position"] == actor["position"]
            previous = state
        else:
            pytest.fail("The already-started parking turn never completed and resumed translation")
        assert tuple(plan["waypoints"][0]["position"]) == goal
    finally:
        engine.close()


def test_committed_parking_turn_keeps_a_safe_escape_from_an_approaching_wall():
    grid = OccupancyGrid(np.zeros((100, 100), dtype=np.int8), 0.05, (-2.5, -2.5))
    actor = {
        "name": "walker",
        "position": [0.0, 0.0, 0.0],
        "yaw": math.pi / 2,
        "velocity": [0.0, 0.0, 0.0],
        "radius_m": 0.25,
        "height_m": 1.8,
    }
    plan = {
        "name": actor["name"],
        "position": actor["position"],
        "behavior": "walk",
        "speed": 0.9,
        "arrival_tolerance": 0.005,
        "waypoints": [{"position": [0.1, 0.0, 0.0], "yaw": None}],
    }
    wall = {
        "position": [0.0, -0.306, 0.0],
        "velocity": [0.0, 0.12, 0.0],
        "velocity_valid": True,
        "radius_m": 1.01,
        "footprint_xy_m": [[-1.0, -0.331], [1.0, -0.331], [1.0, -0.281], [-1.0, -0.281]],
    }
    engine = NavigationEngine([plan], grid, {"map_clearance_m": 0.03}, [actor])
    # Restore a turn already owned before this new wall observation arrived.
    engine.routes[actor["name"]].travel_heading = 0.0
    dt, clearance = 1 / 60, actor["radius_m"] + 0.03
    try:
        top = max(point[1] for point in wall["footprint_xy_m"])
        assert actor["position"][1] - (top + wall["velocity"][1] * dt) < clearance
        state = engine.step(0.0, dt, [actor], [wall])["agents"][0]
        assert state["error"] is None and state["position"][1] > actor["position"][1]
        assert math.dist(actor["position"], state["position"]) <= 0.9 * dt + 1e-6
        assert abs(math.remainder(state["yaw"] - actor["yaw"], 2 * math.pi)) <= 2.5 * dt + 1e-5
        assert grid.segment_clear(actor["position"], state["position"], clearance)
        relative_end = np.asarray(state["position"][:2]) - np.asarray(wall["velocity"][:2]) * dt
        assert segment_polygon_clear(
            actor["position"], relative_end, np.asarray(wall["footprint_xy_m"]), clearance - 1e-6
        )
        # The full relative chord stays over this horizontal face; its closest height is an endpoint.
        assert (
            min(actor["position"][1] - top, state["position"][1] - top - wall["velocity"][1] * dt) >= clearance - 1e-6
        )
    finally:
        engine.close()


def test_obstacle_horizon_preserves_the_independent_reciprocal_agent_horizon():
    grid = OccupancyGrid(np.zeros((80, 100), dtype=np.int8), 0.1, (-5.0, -4.0))
    agents = [
        {"name": "east", "position": [-0.6, 0.0, 0.0], "yaw": 0.0, "velocity": [0.9, 0.0, 0.0], "radius_m": 0.25},
        {"name": "west", "position": [0.6, 0.0, 0.0], "yaw": math.pi, "velocity": [-0.9, 0.0, 0.0], "radius_m": 0.25},
    ]
    plans = [
        {
            "name": actor["name"],
            "position": actor["position"],
            "behavior": "walk",
            "speed": 0.9,
            "waypoints": [{"position": [-3 * actor["position"][0], 0.0, 0.0]}],
        }
        for actor in agents
    ]
    outputs = []
    for extra in ({}, {"obstacle_time_horizon_s": 2.5}, {"obstacle_time_horizon_s": 0.5}):
        engine = NavigationEngine(plans, grid, {"time_horizon_s": 2.5, **extra}, agents)
        outputs.append(engine.step(0.0, 1 / 60, agents)["agents"])
        engine.close()
    assert outputs[0] == outputs[1] == outputs[2]
    assert max(math.hypot(*actor["velocity"][:2]) for actor in outputs[0]) < 0.2


def test_an_old_solver_abi_is_rejected_before_using_its_changed_argument_layout(monkeypatch):
    monkeypatch.setattr(
        "pedstrain_simulation.simulation.ros2_navigation.solver.ctypes.CDLL",
        lambda _: SimpleNamespace(pedestrian_rvo_abi_version=lambda: 2),
    )
    with pytest.raises(RuntimeError, match="ABI mismatch.*rebuild"):
        _RvoSolver([], "old_solver.so")


@pytest.mark.parametrize("speed", [0.0, 0.9])
def test_submicron_heading_projection_respects_zero_speed_and_the_turn_sector(speed):
    grid = OccupancyGrid(np.zeros((20, 20), dtype=np.int8), 0.1, (-1.0, -1.0))
    agents = [{"name": "walker", "position": [0.0, 0.0, 0.0], "yaw": 0.0, "radius_m": 0.25}]
    plans = [{"name": "walker", "position": agents[0]["position"], "behavior": "stand"}]
    engine = NavigationEngine(plans, grid, {}, agents)
    try:
        engine.step(0.0, 1 / 60, agents)
        velocity = engine._solver.heading_project(
            np.array([[0.0, 5e-8]]), np.array([0.0]), 2.5 / 60, np.array([speed])
        )[0]
        if speed == 0:
            np.testing.assert_array_equal(velocity, [0.0, 0.0])
        elif np.linalg.norm(velocity) > 0:
            assert abs(math.atan2(velocity[1], velocity[0])) <= 2.5 / 60 + 1e-6
    finally:
        engine.close()


@pytest.mark.parametrize("indices", [[1], [2, 0], [2, 0, 1]])
def test_heading_subset_uses_the_selected_agents_constraints(indices):
    grid = OccupancyGrid(np.zeros((80, 100), dtype=np.int8), 0.1, (-5.0, -4.0))
    agents = [
        {"name": name, "position": [*point, 0.0], "yaw": yaw, "radius_m": 0.25}
        for name, point, yaw in (
            ("east", (-0.4, 0.0), 0.0),
            ("west", (0.4, 0.0), math.pi),
            ("south", (0.0, 2.0), -math.pi / 2),
        )
    ]
    plans = [
        {
            "name": actor["name"],
            "position": actor["position"],
            "behavior": "walk",
            "speed": 0.6,
            "waypoints": [{"position": goal}],
        }
        for actor, goal in zip(agents, ([3.0, 0.0, 0.0], [-3.0, 0.0, 0.0], [0.0, -3.0, 0.0]))
    ]
    horizon, gap = 2.5, 0.25
    engine = NavigationEngine(plans, grid, {"time_horizon_s": horizon, "pair_clearance_m": gap}, agents)
    try:
        engine.step(0.0, 1 / 60, agents)
        preferred = np.array([[0.5, 0.1], [-0.5, -0.1], [0.1, -0.5]])
        yaws = np.arctan2(preferred[:, 1], preferred[:, 0])
        speeds = np.full(3, 0.6)
        full = engine._solver.heading_project(preferred, yaws, math.pi, speeds)
        selected = engine._solver.heading_project(preferred[indices], yaws[indices], math.pi, speeds[indices], indices)
        np.testing.assert_array_equal(selected, full[indices])
        # Reciprocal responsibility splits the remaining frontal distance equally.
        required = sum(actor["radius_m"] for actor in agents[:2]) + gap
        frontal_bound = (math.dist(agents[0]["position"], agents[1]["position"]) - required) / (2 * horizon)
        assert full[0, 0] <= frontal_bound + 1e-6
        assert full[1, 0] >= -frontal_bound - 1e-6
        wrong_agent = engine._solver.heading_project(preferred[1:2], yaws[1:2], math.pi, speeds[1:2])[0]
        assert wrong_agent[0] < -frontal_bound - 0.1
    finally:
        engine.close()


@pytest.mark.parametrize("blocked_value", [100, -1])
def test_occupied_or_unknown_cells_prevent_diagonal_corner_cutting(blocked_value):
    cells = np.zeros((3, 3), dtype=np.int8)
    cells[0, 1] = cells[1, 0] = blocked_value
    grid = OccupancyGrid(cells, 0.1, (0.0, 0.0))
    with pytest.raises(ValueError, match="reachable"):
        grid.plan((0.05, 0.05), (0.15, 0.15), 0.001, 0.001, margin_m=0.0)


def test_engine_tracks_a_global_detour_without_crossing_static_cells():
    cells = np.zeros((60, 80), dtype=np.int8)
    cells[10:50, 38:42] = 100
    grid = OccupancyGrid(cells, 0.1, (-4.0, -3.0))
    agents = [{"name": "walker", "position": [-3.0, 0.0, 0.0], "yaw": 0.0, "radius_m": 0.2, "height_m": 1.8}]
    plan = {
        "name": "walker",
        "position": [-3.0, 0.0, 0.0],
        "behavior": "walk",
        "speed": 0.8,
        "arrival_tolerance": 0.15,
        "waypoints": [{"position": [3.0, 0.0, 0.0]}],
    }
    engine = NavigationEngine([plan], grid, {}, agents)
    states = agents
    for tick in range(1500):
        state = engine.step(tick / 60, 1 / 60, states)["agents"][0]
        assert state["error"] is None
        assert grid.segment_clear(states[0]["position"], state["position"], 0.25)
        if state["phase"] == "complete":
            assert math.dist(state["position"], [3.0, 0.0, 0.0]) <= 0.15
            break
        states = [state]
    else:
        pytest.fail("The free global detour never reached its original goal")


def test_cached_route_reuses_hard_clearance_when_measured_pose_leaves_preferred_margin(monkeypatch):
    cells = np.zeros((80, 100), dtype=np.int8)
    cells[:40] = 100
    grid = OccupancyGrid(cells, 0.05, (-2.5, -2.0))
    start, goal = [-1.5, 0.36, 0.0], [1.5, 0.36, 0.0]
    agent = {"name": "walker", "position": start, "yaw": 0.0, "radius_m": 0.25, "height_m": 1.8}
    plan = {"name": "walker", "position": start, "behavior": "walk", "waypoints": [{"position": goal}]}
    engine = NavigationEngine([plan], grid, {"map_clearance_m": 0.03, "preferred_path_clearance_m": 0.05}, [agent])
    try:
        engine.step(0.0, 1 / 60, [agent])
        actual = {**agent, "position": [-1.49, 0.30, 0.0], "velocity": [0.0, 0.0, 0.0]}
        assert not grid.segment_clear(actual["position"], goal, 0.33)
        assert grid.segment_clear(actual["position"], goal, 0.28)

        def unnecessary_replan(*args, **kwargs):
            pytest.fail("A hard-clearance cached chord should be reused")

        monkeypatch.setattr(grid, "plan", unnecessary_replan)
        result = engine.step(1 / 60, 1 / 60, [actual])
        assert result["agents"][0]["error"] is None
        assert result["guidance"]["walker"]["target"] == goal
    finally:
        engine.close()


def test_cached_route_reconnects_from_actual_feedback_inside_a_connected_side_bay():
    cells = np.full((80, 100), 100, dtype=np.int8)
    cells[35:45, 10:90] = 0
    cells[35:75, 45:55] = 0
    grid = OccupancyGrid(cells, 0.1, (-5.0, -4.0))
    agents = [{"name": "walker", "position": [0.0, 0.0, 0.0], "yaw": 0.0, "radius_m": 0.2, "height_m": 1.8}]
    plan = {
        "name": "walker",
        "position": [0.0, 0.0, 0.0],
        "behavior": "walk",
        "speed": 0.6,
        "arrival_tolerance": 0.15,
        "waypoints": [{"position": [3.0, 0.0, 0.0]}],
    }
    engine = NavigationEngine([plan], grid, {}, agents)
    engine.step(0.0, 1 / 60, agents)
    actual = {**agents[0], "position": [0.0, 0.35, 0.0], "velocity": [0.0, 0.0, 0.0]}
    states = [actual]
    for tick in range(1, 601):
        state = engine.step(tick / 60, 1 / 60, states)["agents"][0]
        assert state["error"] is None
        assert math.dist(state["position"], states[0]["position"]) <= 0.6 / 60 + 1e-6
        assert grid.segment_clear(states[0]["position"], state["position"], 0.25)
        states = [state]
        if state["phase"] == "complete":
            break
    assert state["phase"] == "complete" and math.dist(state["position"], [3.0, 0.0, 0.0]) <= 0.15


def test_head_on_shared_route_reaches_both_original_goals_without_overlap():
    grid = OccupancyGrid(np.zeros((80, 140), dtype=np.int8), 0.1, (-7.0, -4.0))
    agents = [
        {"name": "east", "position": [-3.0, 0.0, 0.0], "yaw": 0.0, "radius_m": 0.25, "height_m": 1.8},
        {"name": "west", "position": [3.0, 0.0, 0.0], "yaw": math.pi, "radius_m": 0.25, "height_m": 1.8},
    ]
    plans = [
        {
            "name": agent["name"],
            "position": agent["position"],
            "behavior": "walk",
            "speed": 0.9,
            "start_time": 1.0,
            "repeat": False,
            "arrival_tolerance": 0.2,
            "waypoints": [{"position": [-agent["position"][0], 0.0, 0.0], "wait": 0.0, "yaw": None}],
        }
        for agent in agents
    ]
    engine = NavigationEngine(plans, grid, {"pair_clearance_m": 0.25}, agents)
    states = agents
    longest = [0.0, 0.0]
    current = [0.0, 0.0]
    minimum_gap = math.inf
    for tick in range(1200):
        result = engine.step(tick / 60.0, 1 / 60.0, states)
        assert not result["passing"]["transactions"]
        next_states = result["agents"]
        for index, state in enumerate(next_states):
            assert state["error"] is None
            speed = math.hypot(*state["velocity"][:2])
            assert speed <= 0.9 + 1e-6
            heading_step = math.atan2(
                math.sin(state["yaw"] - states[index]["yaw"]), math.cos(state["yaw"] - states[index]["yaw"])
            )
            assert abs(heading_step) <= 2.5 / 60 + 1e-5
            if speed > 1e-8:
                assert math.atan2(state["velocity"][1], state["velocity"][0]) == pytest.approx(state["yaw"])
            current[index] = current[index] + 1 / 60 if state["phase"] == "walking" and speed < 0.03 else 0.0
            longest[index] = max(longest[index], current[index])
        rel = np.array(states[0]["position"][:2]) - states[1]["position"][:2]
        change = (
            np.array(next_states[0]["position"][:2])
            - states[0]["position"][:2]
            - np.array(next_states[1]["position"][:2])
            + states[1]["position"][:2]
        )
        fraction = float(np.clip(-np.dot(rel, change) / np.dot(change, change), 0, 1)) if np.dot(change, change) else 0
        gap = float(np.linalg.norm(rel + fraction * change)) - 0.5
        minimum_gap = min(minimum_gap, gap)
        assert gap >= -1e-6
        states = next_states
    assert [state["phase"] for state in states] == ["complete", "complete"]
    assert max(longest) <= 1.0
    assert minimum_gap >= 0.25 - 1e-5


@pytest.mark.parametrize("rotation", [0.0, 1.07])
def test_slow_close_head_on_avoidance_keeps_facing_forward_and_reaches_both_goals(rotation):
    grid = OccupancyGrid(np.zeros((160, 240), dtype=np.int8), 0.05, (-6.0, -4.0))
    axis = np.array([math.cos(rotation), math.sin(rotation)])
    speed, radius, body_gap, dt = 0.2, 0.25, 0.25, 1 / 60
    agents = [
        {
            "name": name,
            "position": [*(sign * 0.4 * axis), 0.0],
            "yaw": rotation + (math.pi if sign > 0 else 0.0),
            "radius_m": radius,
            "height_m": 1.8,
        }
        for name, sign in (("east", -1), ("west", 1))
    ]
    plans = [
        {
            "name": agent["name"],
            "position": agent["position"],
            "behavior": "walk",
            "speed": speed,
            "arrival_tolerance": 0.2,
            "waypoints": [{"position": [*(-sign * 4.0 * axis), 0.0], "wait": 0.0, "yaw": None}],
        }
        for agent, sign in zip(agents, (-1, 1))
    ]
    engine = NavigationEngine(plans, grid, {"pair_clearance_m": body_gap, "map_clearance_m": 0.03}, agents)
    states = agents
    try:
        for tick in range(2400):
            result = engine.step(tick * dt, dt, states)
            next_states = result["agents"]
            for index, state in enumerate(next_states):
                assert state["error"] is None
                velocity = np.asarray(state["velocity"][:2])
                assert np.linalg.norm(velocity) <= speed + 1e-6
                assert math.dist(state["position"], states[index]["position"]) <= speed * dt + 1e-6
                turn = math.atan2(
                    math.sin(state["yaw"] - states[index]["yaw"]),
                    math.cos(state["yaw"] - states[index]["yaw"]),
                )
                assert abs(turn) <= 2.5 * dt + 1e-5
                forward = axis if index == 0 else -axis
                facing = np.array([math.cos(state["yaw"]), math.sin(state["yaw"])])
                assert np.dot(facing, forward) >= -1e-6
            relative = np.asarray(states[0]["position"][:2]) - states[1]["position"][:2]
            change = (
                np.asarray(next_states[0]["position"][:2])
                - states[0]["position"][:2]
                - np.asarray(next_states[1]["position"][:2])
                + states[1]["position"][:2]
            )
            denominator = float(np.dot(change, change))
            fraction = float(np.clip(-np.dot(relative, change) / denominator, 0, 1)) if denominator else 0.0
            assert np.linalg.norm(relative + fraction * change) - 2 * radius >= body_gap - 1e-5
            states = next_states
            if all(state["phase"] == "complete" for state in states):
                break
        else:
            pytest.fail("Slow opposing pedestrians did not reach their original goals within 40 simulation seconds")
        for state, plan in zip(states, plans):
            assert math.dist(state["position"], plan["waypoints"][0]["position"]) <= plan["arrival_tolerance"]
    finally:
        engine.close()


@pytest.mark.parametrize("snapshot_field", ["snapshot_json", "late_snapshot_json"], ids=["native44", "native48"])
def test_real_wall_and_peer_snapshot_escapes_tangent_creep_without_relaxing_clearance(snapshot_field):
    fixture_path = Path(__file__).with_name("fixtures") / "native_heading_tangent_20261105.npz"
    with np.load(fixture_path, allow_pickle=False) as fixture:
        snapshot = json.loads(str(fixture[snapshot_field]))
        grid = OccupancyGrid(fixture["occupancy"], float(fixture["resolution_m"]), tuple(fixture["origin_xy_m"]))
    agents, settings = snapshot["agents"], snapshot["settings"]
    walking_name, goal = snapshot["walking_name"], snapshot["local_goal_m"]
    plans = [
        {
            "name": actor["name"],
            "position": actor["position"],
            "behavior": "walk" if actor["name"] == walking_name else "stand",
            "speed": 0.9,
            "arrival_tolerance": 0.2,
            "waypoints": [{"position": goal, "wait": 0.0, "yaw": None}] if actor["name"] == walking_name else [],
        }
        for actor in agents
    ]
    engine = NavigationEngine(plans, grid, settings, agents)
    states = agents
    walking_index = next(i for i, actor in enumerate(agents) if actor["name"] == walking_name)
    initial_distance = math.dist(agents[walking_index]["position"], goal)
    dt = 1 / settings["control_hz"]
    footprints = [convex_footprint(obstacle["footprint_xy_m"]) for obstacle in snapshot["obstacles"]]
    try:
        for tick in range(round(3.0 / dt)):
            next_states = engine.step(tick * dt, dt, states, obstacles=snapshot["obstacles"])["agents"]
            for i, (before, after) in enumerate(zip(states, next_states)):
                assert after["error"] is None
                speed = math.hypot(*after["velocity"][:2])
                assert speed <= plans[i]["speed"] + 1e-6
                assert math.dist(before["position"], after["position"]) <= plans[i]["speed"] * dt + 1e-6
                if speed > 1e-8:
                    motion_yaw = math.atan2(after["velocity"][1], after["velocity"][0])
                    assert abs(math.remainder(after["yaw"] - motion_yaw, 2 * math.pi)) < 1e-6
                turn = math.remainder(after["yaw"] - before["yaw"], 2 * math.pi)
                assert abs(turn) <= settings["max_turn_rate_radps"] * dt + 1e-5
                if after["name"] != walking_name:
                    assert after["position"] == agents[i]["position"]
                clearance = agents[i]["radius_m"] + settings["map_clearance_m"]
                assert grid.segment_clear(before["position"], after["position"], clearance)
                for footprint in footprints:
                    assert segment_polygon_clear(before["position"][:2], after["position"][:2], footprint, clearance)
            for i, after in enumerate(next_states):
                for j in range(i):
                    relative = np.asarray(states[i]["position"][:2]) - states[j]["position"][:2]
                    change = (
                        np.asarray(after["position"][:2])
                        - states[i]["position"][:2]
                        - np.asarray(next_states[j]["position"][:2])
                        + states[j]["position"][:2]
                    )
                    denominator = float(np.dot(change, change))
                    fraction = float(np.clip(-np.dot(relative, change) / denominator, 0, 1)) if denominator else 0.0
                    body_gap = np.linalg.norm(relative + fraction * change) - sum(agents[k]["radius_m"] for k in (i, j))
                    assert body_gap >= settings["pair_clearance_m"] - 1e-12
            assert next_states[walking_index]["phase"] in ("walking", "complete")
            states = next_states
        # The old solution retained nonzero sideways micromotion while making
        # no approach to this visible guide for the entire observation window.
        assert (
            initial_distance - math.dist(states[walking_index]["position"], goal) >= agents[walking_index]["radius_m"]
        )
    finally:
        engine.close()


def test_far_wall_parking_uses_a_certified_side_step_without_repeated_large_reversals():
    directory = Path(__file__).parent / "fixtures"
    snapshot = json.loads((directory / "native_parking_side_step_20261106.json").read_text())
    with np.load(directory / snapshot["provenance"]["map_fixture"], allow_pickle=False) as fixture:
        grid = OccupancyGrid(fixture["occupancy"], float(fixture["resolution_m"]), tuple(fixture["origin_xy_m"]))
    actors, settings, targets = snapshot["actors"], snapshot["settings"], snapshot["targets"]
    plans = [
        {
            "name": actor["name"],
            "position": actor["position"],
            "behavior": "walk",
            "speed": 0.9,
            "arrival_tolerance": 0.2,
            "waypoints": [targets[actor["name"]]],
        }
        for actor in actors
    ]
    original = copy.deepcopy(plans)
    index = next(i for i, actor in enumerate(actors) if actor["name"] == "walker_03")
    goal = targets[actors[index]["name"]]["position"]
    delta = np.asarray(goal[:2]) - actors[index]["position"][:2]
    initial_turn = abs(math.remainder(math.atan2(delta[1], delta[0]) - actors[index]["yaw"], 2 * math.pi))
    initial_distance = math.dist(actors[index]["position"], goal)
    footprints = [convex_footprint(obstacle["footprint_xy_m"]) for obstacle in snapshot["obstacles"]]
    states, total_turn = actors, 0.0
    dt = 1 / settings["control_hz"]
    engine = NavigationEngine(plans, grid, settings, actors)
    try:
        for tick in range(round(3.0 / dt)):
            result = engine.step(
                snapshot["time_s"] + tick * dt, dt, states, snapshot["obstacles"], snapshot["door_states"]
            )
            next_states = result["agents"]
            for i, (before, after) in enumerate(zip(states, next_states)):
                assert after["error"] is None
                speed = math.hypot(*after["velocity"][:2])
                assert speed <= plans[i]["speed"] + 1e-6
                assert math.dist(before["position"], after["position"]) <= plans[i]["speed"] * dt + 1e-6
                turn = abs(math.remainder(after["yaw"] - before["yaw"], 2 * math.pi))
                assert turn <= settings["max_turn_rate_radps"] * dt + 1e-5
                if speed > 0.0:
                    motion_yaw = math.atan2(after["velocity"][1], after["velocity"][0])
                    assert abs(math.remainder(motion_yaw - after["yaw"], 2 * math.pi)) < 1e-6
                if i == index:
                    total_turn += turn
                clearance = actors[i]["radius_m"] + settings["map_clearance_m"]
                assert grid.segment_clear(before["position"], after["position"], clearance)
                for footprint in footprints:
                    assert segment_polygon_clear(before["position"], after["position"], footprint, clearance)
                for j in range(i):
                    relative = np.asarray(states[i]["position"][:2]) - states[j]["position"][:2]
                    motion = np.asarray(after["position"][:2]) - states[i]["position"][:2]
                    motion -= np.asarray(next_states[j]["position"][:2]) - states[j]["position"][:2]
                    squared = float(np.dot(motion, motion))
                    fraction = float(np.clip(-np.dot(relative, motion) / squared, 0, 1)) if squared else 0.0
                    required = sum(actors[k]["radius_m"] for k in (i, j)) + settings["pair_clearance_m"]
                    assert np.linalg.norm(relative + fraction * motion) >= required - 1e-12
            states = [{**before, **after} for before, after in zip(states, next_states)]
        assert initial_distance - math.dist(states[index]["position"], goal) >= actors[index]["radius_m"]
        assert total_turn < initial_turn + math.pi
        assert plans == original
    finally:
        engine.close()


def test_already_separated_opposing_routes_do_not_make_an_unnecessary_passing_detour():
    grid = OccupancyGrid(np.zeros((80, 140), dtype=np.int8), 0.1, (-7.0, -4.0))
    radius, pair_gap, lateral_spacing = 0.25, 0.25, 1.0
    agents = [
        {"name": name, "position": [x, y, 0.0], "yaw": yaw, "radius_m": radius, "height_m": 1.8}
        for name, x, y, yaw in (("east", -3.0, -0.5, 0.0), ("west", 3.0, 0.5, math.pi))
    ]
    plans = [
        {
            "name": agent["name"],
            "position": agent["position"],
            "behavior": "walk",
            "speed": 0.9,
            "arrival_tolerance": 0.2,
            "waypoints": [{"position": [-agent["position"][0], agent["position"][1], 0.0]}],
        }
        for agent in agents
    ]
    engine = NavigationEngine(plans, grid, {"pair_clearance_m": pair_gap, "map_clearance_m": 0.03}, agents)
    states = agents
    deviation = 0.0
    for tick in range(720):
        result = engine.step(tick / 60, 1 / 60, states)
        states = result["agents"]
        assert not result["passing"]["transactions"]
        assert all(state["error"] is None for state in states)
        deviation = max(
            deviation, *(abs(state["position"][1] - agent["position"][1]) for state, agent in zip(states, agents))
        )
    engine.close()
    assert all(state["phase"] == "complete" for state in states)
    assert deviation < (lateral_spacing - 2 * radius - pair_gap) / 2


@pytest.mark.parametrize("advance_ticks", [0, 60])
def test_live_obstacle_replans_a_cached_route_without_returning_to_a_passed_vertex(advance_ticks):
    grid = OccupancyGrid(np.zeros((80, 120), dtype=np.int8), 0.05, (-1.0, -2.0))
    agents = [{"name": "walker", "position": [1.5, 0.5, 0.0], "yaw": 0.0, "radius_m": 0.25}]
    plans = [
        {
            "name": "walker",
            "position": [0.0, 0.0, 0.0],
            "behavior": "walk",
            "speed": 0.9,
            "arrival_tolerance": 0.2,
            "waypoints": [{"position": [3.5, 0.0, 0.0]}],
        }
    ]
    engine = NavigationEngine(plans, grid, {"map_clearance_m": 0.03}, agents)
    engine.routes["walker"].path = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.5, 0.0]]
    try:
        states = agents
        for tick in range(advance_ticks):
            states = engine.step(tick / 60, 1 / 60, states)["agents"]
        obstacle = {
            "path": "/ObservedBox",
            "position": [states[0]["position"][0] + 0.65, states[0]["position"][1], 0.0],
            "radius_m": 0.3,
            "velocity": [0.0, 0.0, 0.0],
            "velocity_valid": True,
        }
        result = engine.step(advance_ticks / 60, 1 / 60, states, [obstacle])
        assert result["agents"][0]["error"] is None
        target = result["guidance"]["walker"]["target"]
        assert target[0] > states[0]["position"][0]
        assert grid.segment_clear(states[0]["position"], target, 0.28)
    finally:
        engine.close()


def _chord_guard_engine(grid, positions):
    agents = [
        {"name": f"person_{i}", "position": [*point, 0.0], "yaw": 0.0, "radius_m": 0.25}
        for i, point in enumerate(positions)
    ]
    plans = [{"name": actor["name"], "position": actor["position"], "behavior": "stand"} for actor in agents]
    return NavigationEngine(plans, grid, {"pair_clearance_m": 0.25, "map_clearance_m": 0.03}, agents)


@pytest.mark.parametrize(
    "speed,spacing,stopped,following",
    [(0.9, 0.75000005, True, False), (1e-7, 3.0, False, False), (0.9, 0.7500000005, True, True)],
)
def test_execution_discards_float_solver_residuals_but_preserves_small_requested_speeds(
    speed, spacing, stopped, following
):
    positions = np.array([[0.0, 0.0], [spacing, 0.0], [-2.0, 2.0]])
    grid = OccupancyGrid(np.zeros((80, 120), dtype=np.int8), 0.1, (-6.0, -4.0))
    engine = _chord_guard_engine(grid, positions)
    velocities = np.zeros((3, 2))
    preferred = np.array([[speed, 0.0], [0.0, 0.0], [0.0, 0.0]])
    radii = np.full(3, 0.25)
    caps = np.array([speed, 0.0, 0.0])
    yaws = np.zeros(3)
    if following:
        velocities[0, 0] = speed
        preferred[1, 0] = caps[1] = speed
        yaws[1] = math.pi
    try:
        actual, _ = engine._solve_crowd(positions, velocities, preferred, radii, caps, yaws, 1 / 60, [])
        if stopped:
            np.testing.assert_array_equal(actual[0], [0.0, 0.0])
        else:
            assert actual[0, 0] > 0.99 * speed
        assert np.linalg.norm(actual[0]) <= speed + 1e-12
        assert engine._chord_distance(positions[0] - positions[1], actual[0] / 60) >= 0.75
        np.testing.assert_array_equal(actual[1:], np.zeros((2, 2)))
    finally:
        engine.close()


def test_heading_stop_preserves_the_executed_peer_gap_without_stopping_unrelated_motion():
    grid = OccupancyGrid(np.zeros((100, 120), dtype=np.int8), 0.1, (-5.0, -5.0))
    positions = np.array([[0.0, 0.0], [0.76, 0.0], [3.0, 2.0]])
    engine = _chord_guard_engine(grid, positions)
    velocities = np.array([[0.9, 0.0], [0.0, 0.0], [0.4, 0.1]])
    unrelated = velocities[2].copy()
    engine._guard_chords(positions, velocities, np.full(3, 0.25), 1 / 60)
    assert engine._chord_distance(positions[0] - positions[1], (velocities[0] - velocities[1]) / 60) >= 0.75 - 1e-10
    assert 0 < velocities[0, 0] < 0.9
    np.testing.assert_array_equal(velocities[2], unrelated)
    engine.close()


@pytest.mark.parametrize(
    ("wallward_rad", "stall_s", "should_move"), [(0.0004, 1.0, True), (0.0004, 0.5, False), (0.015, 1.0, False)]
)
def test_wall_guard_preserves_a_near_tangent_orca_step_at_hard_clearance(wallward_rad, stall_s, should_move):
    cells = np.zeros((80, 80), dtype=np.int8)
    cells[:40, 40:] = 100  # The corner of the occupied quadrant is at the origin.
    grid = OccupancyGrid(cells, 0.05, (-2.0, -2.0))
    radius, margin, tangent, speed, dt = 0.25, 0.03, 0.55, 0.034, 1 / 60
    clearance = radius + margin
    position = (clearance + 2e-12) * np.array([-math.sin(tangent), math.cos(tangent)])
    positions = position[None, :]
    engine = _chord_guard_engine(grid, positions)
    engine._last_time = stall_s
    engine.routes[engine.names[0]].stalled_since = 0.0
    radii, caps = np.array([radius]), np.array([0.9])
    yaws, angles = np.array([tangent]), np.array([engine.turn_rate * dt])
    inward = tangent - wallward_rad
    velocity = speed * np.array([[math.cos(inward), math.sin(inward)]])
    try:
        # Initialize real ORCA constraints before feeding the guard a known wallward projection.
        engine._solver.solve(
            positions,
            np.zeros_like(positions),
            velocity,
            radii,
            caps,
            np.array([False]),
            dt,
            engine.neighbor_distance,
            engine.horizon,
            engine.pair_clearance / 2,
            margin,
            engine.obstacle_horizon,
        )
        assert not grid.segment_clear(position, position + velocity[0] * dt, clearance)
        engine._guard_chords(positions, velocity, radii, dt, (yaws, angles, caps), [True])
        assert grid.segment_clear(position, position + velocity[0] * dt, clearance)
        if should_move:
            assert math.hypot(*velocity[0]) >= 0.95 * speed
            assert abs(math.atan2(velocity[0, 1], velocity[0, 0]) - tangent) <= angles[0]
        else:
            np.testing.assert_array_equal(velocity[0], [0.0, 0.0])
    finally:
        engine.close()


@pytest.mark.parametrize("spacing", [0.748, 0.75])
def test_peer_guard_keeps_a_separating_leader_and_does_not_worsen_an_existing_comfort_gap(spacing):
    grid = OccupancyGrid(np.zeros((80, 120), dtype=np.int8), 0.1, (-6.0, -4.0))
    positions = np.array([[spacing, 0.0], [0.0, 0.0]])
    engine = _chord_guard_engine(grid, positions)
    velocities = np.array([[0.1, 0.0], [0.2, 0.0]])
    engine._guard_chords(positions, velocities, np.full(2, 0.25), 1 / 60)
    assert engine._chord_distance(positions[0] - positions[1], (velocities[0] - velocities[1]) / 60) >= spacing - 1e-12
    assert velocities[0, 0] == pytest.approx(0.1)
    assert 0 < velocities[1, 0] <= velocities[0, 0] + 1e-10
    outgoing = np.array([[0.1, 0.0], [0.0, 0.0]])
    engine._guard_chords(positions, outgoing, np.full(2, 0.25), 1 / 60)
    np.testing.assert_array_equal(outgoing, [[0.1, 0.0], [0.0, 0.0]])
    engine.close()


def test_peer_guard_rejects_actual_body_overlap_instead_of_hiding_it_with_zero_velocity():
    grid = OccupancyGrid(np.zeros((80, 120), dtype=np.int8), 0.1, (-6.0, -4.0))
    positions = np.array([[0.0, 0.0], [0.49, 0.0]])
    engine = _chord_guard_engine(grid, positions)
    with pytest.raises(RuntimeError, match="bodies overlap"):
        engine._guard_chords(positions, np.zeros((2, 2)), np.full(2, 0.25), 1 / 60)
    engine.close()


def test_peer_guard_rechecks_an_entire_following_chain_without_forcing_all_actors_to_stop():
    grid = OccupancyGrid(np.zeros((80, 120), dtype=np.int8), 0.1, (-6.0, -4.0))
    positions = np.array([[0.0, 0.0], [0.75, 0.0], [1.5, 0.0]])
    engine = _chord_guard_engine(grid, positions)
    velocities = np.array([[0.3, 0.0], [0.2, 0.0], [0.1, 0.0]])
    engine._guard_chords(positions, velocities, np.full(3, 0.25), 1 / 60)
    for i in range(3):
        assert velocities[i, 0] > 0
        for j in range(i):
            assert (
                engine._chord_distance(positions[i] - positions[j], (velocities[i] - velocities[j]) / 60)
                >= 0.75 - 1e-10
            )
    engine.close()


def test_wall_guard_keeps_positive_swept_prefix_of_a_heading_proposal(monkeypatch):
    cells = np.zeros((80, 80), dtype=np.int8)
    cells[40:] = 100
    grid = OccupancyGrid(cells, 0.05, (-2.0, -2.0))
    ray = np.array([0.48, 0.36])
    start = [0.0, -0.284, 0.0]
    goal = [1.0, -0.284, 0.0]
    dt = 1 / 60
    walker = {"name": "walker", "position": start, "yaw": math.atan2(*ray[::-1]), "radius_m": 0.25, "height_m": 1.8}
    peer = {"name": "peer", "position": [-0.82, -0.284, 0.0], "yaw": 0.0, "radius_m": 0.25, "height_m": 1.8}
    plans = [
        {
            "name": "walker",
            "position": start,
            "behavior": "walk",
            "speed": 0.9,
            "arrival_tolerance": 0.1,
            "waypoints": [{"position": goal}],
        },
        {"name": "peer", "position": peer["position"], "behavior": "stand"},
    ]
    engine = NavigationEngine(plans, grid, {"map_clearance_m": 0.03}, [walker, peer])

    def heading_proposal(preferred, yaws, angles, speeds, agent_indices=None):
        result = np.zeros_like(preferred)
        if agent_indices is None:
            result[0] = ray
        return result

    monkeypatch.setattr(engine._solver, "heading_project", heading_proposal)
    try:
        assert not grid.segment_clear(start, np.asarray(start[:2]) + ray * dt, 0.28)
        moving, holding = engine.step(0.0, dt, [walker, peer])["agents"]
        assert moving["error"] is None and holding["error"] is None
        assert holding["position"] == peer["position"]
        assert grid.segment_clear(start, moving["position"], 0.28)
        assert moving["velocity"][:2] == pytest.approx(ray * (2 / 3), abs=2e-4)
        assert moving["yaw"] == pytest.approx(walker["yaw"])
    finally:
        engine.close()


def test_peer_guard_keeps_positive_certified_prefixes_when_full_chords_cross_midstep():
    grid = OccupancyGrid(np.zeros((80, 120), dtype=np.int8), 0.1, (-6.0, -4.0))
    positions = np.array([[-1.0, 0.0], [1.0, 0.0]])
    engine = _chord_guard_engine(grid, positions)
    velocities = np.array([[2.0, 0.0], [-2.0, 0.0]])
    engine._guard_chords(positions, velocities, np.full(2, 0.25), 1.0)
    assert engine._chord_distance(positions[0] - positions[1], velocities[0] - velocities[1]) >= 0.75 - 1e-10
    assert np.linalg.norm(velocities, axis=1).min() > 0
    engine.close()


@pytest.mark.parametrize("dynamic_stop", [False, True])
def test_static_or_dynamic_stops_cannot_discard_the_executed_pair_clearance(monkeypatch, dynamic_stop):
    cells = np.zeros((80, 120), dtype=np.int8)
    if not dynamic_stop:
        cells[:, 81:] = 100
    grid = OccupancyGrid(cells, 0.05, (-3.0, -2.0))
    positions = np.array([[0.0, 0.0], [0.76, 0.0]])
    engine = _chord_guard_engine(grid, positions)
    candidates = np.array([[0.9, 0.0], [0.9, 0.0]])
    if dynamic_stop:
        monkeypatch.setattr(engine._solver, "solve", lambda *a, **k: candidates.copy())
        monkeypatch.setattr(engine._solver, "heading_project", lambda velocity, *a: velocity.copy())
        monkeypatch.setattr(
            engine, "_dynamic_clear", lambda point, velocity, *a: point[0] < 0.5 or not np.any(velocity)
        )
        obstacle = SimpleNamespace(footprint=None, position=np.array([2.0, 2.0]), velocity=np.zeros(2), radius=0.1)
        velocities, _ = engine._solve_crowd(
            positions, candidates, candidates, np.full(2, 0.25), np.full(2, 0.9), np.zeros(2), 1 / 60, [obstacle]
        )
        assert velocities[1, 0] == 0
    else:
        velocities = candidates.copy()
        engine._guard_chords(positions, velocities, np.full(2, 0.25), 1 / 60)
        assert velocities[1, 0] == 0
    assert engine._chord_distance(positions[0] - positions[1], (velocities[0] - velocities[1]) / 60) >= 0.75 - 1e-10
    engine.close()


@pytest.mark.parametrize("dt", [1 / 60, 1 / 120])
def test_side_velocity_preference_is_integrated_through_relaxation_once(dt):
    grid = OccupancyGrid(np.zeros((80, 120), dtype=np.int8), 0.1, (-6.0, -4.0))
    agents = [
        {"name": "east", "position": [-1.5, 0.0, 0.0], "yaw": 0.0, "radius_m": 0.25, "height_m": 1.8},
        {"name": "west", "position": [1.5, 0.0, 0.0], "yaw": math.pi, "radius_m": 0.25, "height_m": 1.8},
    ]
    plans = [
        {
            "name": a["name"],
            "position": a["position"],
            "behavior": "walk",
            "speed": 0.9,
            "waypoints": [{"position": [-a["position"][0], 0.0, 0.0]}],
        }
        for a in agents
    ]
    engine = NavigationEngine(plans, grid, {"relaxation_time_s": 0.4}, agents)
    captured = []
    original = engine._solver.solve

    def observe(*args, **kwargs):
        captured.append(np.asarray(args[2]).copy())
        return original(*args, **kwargs)

    engine._solver.solve = observe
    engine.step(0.0, dt, agents)
    assert captured[0][:, 1] == pytest.approx([-0.15 * dt / 0.4, 0.15 * dt / 0.4], abs=1e-10)


@pytest.mark.parametrize("speed", [0.2, 0.9])
def test_certified_clear_portal_guidance_is_not_reversed_by_soft_wall_force(speed):
    cells = np.zeros((100, 100), dtype=np.int8)
    cells[48:52, :] = 100
    cells[48:52, 43:57] = 0
    grid = OccupancyGrid(cells, 0.05, (-2.5, -2.5))
    agents = [{"name": "walker", "position": [0.0, -0.45, 0.0], "yaw": math.pi / 2, "radius_m": 0.25, "height_m": 1.8}]
    plans = [
        {
            "name": "walker",
            "position": [0.0, -0.45, 0.0],
            "behavior": "walk",
            "speed": speed,
            "arrival_tolerance": 0.15,
            "waypoints": [{"position": [0.0, 1.2, 0.0]}],
        }
    ]
    assert grid.segment_clear([0.0, -0.45], [0.0, 1.2], 0.28)
    engine = NavigationEngine(plans, grid, {"map_clearance_m": 0.03, "obstacle_time_horizon_s": 0.5}, agents)
    states = agents
    for tick in range(1200):
        state = engine.step(tick / 60, 1 / 60, states)["agents"][0]
        assert state["error"] is None
        assert grid.segment_clear(states[0]["position"], state["position"], 0.28)
        states = [state]
        if state["phase"] == "complete":
            break
    assert state["phase"] == "complete" and math.dist(state["position"], [0.0, 1.2, 0.0]) <= 0.15


@pytest.mark.parametrize("repeat", [False, True])
def test_closed_route_preserves_final_heading_and_dwell_before_completing_or_repeating(repeat):
    grid = OccupancyGrid(np.zeros((80, 80), dtype=np.int8), 0.1, (-4.0, -4.0))
    agents = [{"name": "walker", "position": [0.0, 0.0, 0.0], "yaw": 0.0, "radius_m": 0.2, "height_m": 1.8}]
    plan = {
        "name": "walker",
        "position": [0.0, 0.0, 0.0],
        "behavior": "walk",
        "speed": 0.6,
        "start_time": 0.2,
        "repeat": repeat,
        "arrival_tolerance": 0.1,
        "waypoints": [
            {"position": [1.0, 0.0, 0.0], "wait": 0.3, "yaw": 90.0},
            {"position": [0.0, 0.0, 0.0], "wait": 0.4, "yaw": 180.0},
        ],
    }
    engine = NavigationEngine([plan], grid, {}, agents)
    states = agents
    final_dwell = None
    for tick in range(1800):
        now = tick / 60
        state = engine.step(now, 1 / 60, states)["agents"][0]
        assert state["error"] is None
        difference = math.atan2(math.sin(state["yaw"] - states[0]["yaw"]), math.cos(state["yaw"] - states[0]["yaw"]))
        assert abs(difference) <= 2.5 / 60 + 1e-5
        if state["phase"] == "dwelling" and state["waypoint_index"] == 1 and final_dwell is None:
            final_dwell = now
            assert math.dist(state["position"], [0, 0, 0]) <= 0.1
            assert abs(math.atan2(math.sin(state["yaw"] - math.pi), math.cos(state["yaw"] - math.pi))) <= 0.025
        if state["completed_cycles"]:
            assert final_dwell is not None and now - final_dwell >= 0.4 - 1e-9
            assert state["phase"] != "complete" if repeat else state["phase"] == "complete"
            break
        states = [state]
    else:
        pytest.fail("Closed route never completed its original final turn and wait")


@pytest.mark.parametrize(
    ("start_time", "wait_s", "expected_ticks"), [(0.0, 8.0, 480), (30.8, 8.0, 480), (0.0, 8.005, 481)]
)
def test_authored_start_and_dwell_round_up_to_the_next_actual_tick(start_time, wait_s, expected_ticks):
    grid = OccupancyGrid(np.zeros((40, 40), dtype=np.int8), 0.1, (-2.0, -2.0))
    agent = {"name": "walker", "position": [0.0, 0.0, 0.0], "yaw": 0.0, "radius_m": 0.2, "height_m": 1.8}
    plan = {
        "name": "walker",
        "position": agent["position"],
        "behavior": "walk",
        "start_time": start_time,
        "waypoints": [{"position": [0.05, 0.0, 0.0], "wait": wait_s}],
    }
    engine = NavigationEngine([plan], grid, {}, [agent])
    dt = 1 / 60
    # Repeated additions reproduce the native timeline's tiny negative roundoff.
    now = sum([dt] * 1847)
    if start_time:
        assert engine.step(now, dt, [agent])["agents"][0]["phase"] == "waiting"
    now += dt
    for tick in range(expected_ticks + 1):
        state = engine.step(now, dt, [agent])["agents"][0]
        assert state["phase"] == ("dwelling" if tick < expected_ticks else "complete")
        assert state["position"] == agent["position"]
        now += dt


@pytest.mark.parametrize(
    "policy,closed,spawn_adjusted",
    [
        (None, False, False),
        ("preserve", False, False),
        ("follow_route", False, False),
        ("follow_route", True, False),
        ("follow_route", False, True),
        ("follow_route", True, True),
    ],
)
def test_auto_return_heading_policy_keeps_explicit_yaw_and_every_original_wait(policy, closed, spawn_adjusted):
    grid = OccupancyGrid(np.zeros((100, 120), dtype=np.int8), 0.1, (-5.0, -5.0))
    actor = {"name": "walker", "position": [0.0, 0.0, 0.0], "yaw": 0.0, "radius_m": 0.25, "height_m": 1.8}
    waypoints = [
        {"position": [1.0, 0.0, 0.0], "wait": 0.2, "yaw": 90.0},
        {"position": [2.0, 0.0, 0.0], "wait": 0.25, "yaw": -90.0},
    ]
    if closed:
        waypoints.append({"position": [0.0, 0.0, 0.0], "wait": 0.3, "yaw": 45.0})
    if spawn_adjusted:
        actor["position"] = [-0.2, 0.0, 0.0] if closed else waypoints[-1]["position"]
    plan = {
        "name": actor["name"],
        "position": actor["position"],
        "behavior": "walk",
        "speed": 0.6,
        "repeat": True,
        "arrival_tolerance": 0.05,
        "waypoints": waypoints,
    }
    if spawn_adjusted:
        plan["authored_closed_route"] = closed
    original_plan = json.dumps(plan, sort_keys=True)
    expected = [(tuple(point["position"]), point["wait"], point["yaw"]) for point in waypoints]
    if not closed:
        expected.append(
            (
                tuple(waypoints[0]["position"]),
                waypoints[0]["wait"],
                None if policy == "follow_route" else waypoints[0]["yaw"],
            )
        )
    settings = {} if policy is None else {"auto_return_heading": policy}
    engine = NavigationEngine([plan], grid, settings, [actor])
    states, dt, first_dwell, dwell_finished, turning_seen = [actor], 1 / 60, {}, set(), set()
    return_heading = None
    try:
        for tick in range(2400):
            now = tick * dt
            previous = states[0]
            state = engine.step(now, dt, states)["agents"][0]
            assert state["error"] is None
            assert abs(math.remainder(state["yaw"] - previous["yaw"], 2 * math.pi)) <= 2.5 * dt + 1e-5
            assert math.dist(previous["position"], state["position"]) <= 0.6 * dt + 1e-6
            if previous.get("phase") == "dwelling" and (previous["waypoint_index"], previous["completed_cycles"]) != (
                state["waypoint_index"],
                state["completed_cycles"],
            ):
                index = previous["waypoint_index"]
                if index < len(expected):
                    assert now - first_dwell[index] >= expected[index][1] - 1e-9
                    dwell_finished.add(index)
            index = state["waypoint_index"]
            if closed:
                assert index < len(waypoints)
            if state["completed_cycles"]:
                break
            if state["phase"] == "turning":
                turning_seen.add(index)
                assert closed or policy != "follow_route" or index != 2
            if state["phase"] == "dwelling" and index < len(expected):
                position, wait, heading = expected[index]
                assert math.dist(state["position"], position) <= plan["arrival_tolerance"]
                if index not in first_dwell:
                    first_dwell[index] = now
                    if not closed and policy == "follow_route" and index == 2:
                        return_heading = previous["yaw"]
                if heading is not None:
                    assert abs(math.remainder(state["yaw"] - math.radians(heading), 2 * math.pi)) <= 0.025 + 1e-6
                else:
                    assert return_heading is not None
                    assert abs(math.remainder(state["yaw"] - return_heading, 2 * math.pi)) <= 1e-6
            states = [state]
        else:
            pytest.fail("Return policy did not finish the original route, yaw and waits within 40 simulation seconds")
        assert dwell_finished == set(range(len(expected)))
        assert {0, 1} <= turning_seen
        if closed or policy != "follow_route":
            assert 2 in turning_seen
        assert json.dumps(plan, sort_keys=True) == original_plan
    finally:
        engine.close()


@pytest.mark.parametrize(
    "speed, advances, offset, yaw",
    [
        (0.002, True, (2.0, 0.0), 0.0),
        (1e-8, True, (2.0, 0.0), 0.0),
        (0.002, True, (0.0, 2.0), -0.7),
        (1e-8, True, (0.0, 2.0), 0.0),
        (1e-8, True, (0.0, 2.0), -0.7),
        (0.9, False, (2.0, 0.0), 0.0),
        (0.002, False, (2.0, 0.0), 0.0),
        (1e-8, False, (2.0, 0.0), 0.0),
    ],
)
def test_route_progress_budget_respects_nominal_speed_and_actual_feedback(speed, advances, offset, yaw):
    grid = OccupancyGrid(np.zeros((100, 120), dtype=np.int8), 0.1, (-6.0, -5.0))
    actor = {"name": "walker", "position": [-1.0, 0.0, 0.0], "yaw": yaw, "radius_m": 0.25, "height_m": 1.8}
    goal = [actor["position"][0] + offset[0], actor["position"][1] + offset[1], 0.0]
    plan = {
        "name": actor["name"],
        "position": actor["position"],
        "behavior": "walk",
        "speed": speed,
        "arrival_tolerance": 0.2,
        "waypoints": [{"position": goal, "wait": 0.0, "yaw": None}],
    }
    engine = NavigationEngine([plan], grid, {"map_clearance_m": 0.03}, [actor])
    states, dt = [actor], 1 / 60
    try:
        for tick in range(1201):
            result = engine.step(tick * dt, dt, states)["agents"][0]
            if advances:
                assert result["error"] is None
                assert grid.segment_clear(states[0]["position"], result["position"], 0.28)
                turn = math.remainder(result["yaw"] - states[0]["yaw"], 2 * math.pi)
                assert abs(turn) <= 2.5 * dt + 1e-5
                if math.hypot(*result["velocity"][:2]) > 0.0:
                    motion_yaw = math.atan2(result["velocity"][1], result["velocity"][0])
                    assert abs(math.remainder(result["yaw"] - motion_yaw, 2 * math.pi)) < 1e-6
                states = [result]
            elif result["error"] is not None:
                assert "No safe route progress" in result["error"]
                assert 12.0 < tick * dt <= 12.0 + dt + 1e-9
                assert result["position"] == actor["position"]
                break
        else:
            assert advances, "Stationary actual feedback never reached the original progress timeout"
        if advances:
            assert result["phase"] == "walking"
            assert math.dist(actor["position"], goal) - math.dist(result["position"], goal) >= 0.8 * speed * tick * dt
    finally:
        engine.close()


def test_unreachable_engine_retains_actual_pose_and_reports_a_route_error():
    cells = np.zeros((40, 60), dtype=np.int8)
    cells[:, 29:31] = 100
    grid = OccupancyGrid(cells, 0.1, (-3.0, -2.0))
    agents = [{"name": "walker", "position": [-2.0, 0.0, 0.0], "yaw": 0.0, "radius_m": 0.2, "height_m": 1.8}]
    plans = [
        {
            "name": "walker",
            "position": [-2.0, 0.0, 0.0],
            "behavior": "walk",
            "speed": 0.8,
            "waypoints": [{"position": [2.0, 0.0, 0.0]}],
            "arrival_tolerance": 0.2,
        }
    ]
    state = NavigationEngine(plans, grid, {}, agents).step(0.0, 1 / 60, agents)["agents"][0]
    assert state["phase"] == "failed" and "reachable" in state["error"]
    assert state["position"] == agents[0]["position"] and state["velocity"] == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("reachable", [True, False])
def test_wander_uses_its_local_candidate_budget_and_rejects_disconnected_destinations(reachable):
    if reachable:
        grid = OccupancyGrid(np.zeros((500, 500), dtype=np.int8), 0.1, (-25.0, -25.0))
        origin = [-24.5, -24.5, 0.0]
    else:
        cells = np.zeros((80, 80), dtype=np.int8)
        cells[30:50, 30:50] = 100
        cells[36:44, 36:44] = 0
        grid = OccupancyGrid(cells, 0.1, (-4.0, -4.0))
        origin = [0.0, 0.0, 0.0]
    actor = {"name": "wanderer", "position": origin, "yaw": 0.0, "radius_m": 0.25, "height_m": 1.8}
    plan = {"name": actor["name"], "position": origin, "behavior": "wander", "speed": 0.9}
    known_goal = [origin[0] + 2.0, origin[1], 0.0]
    assert grid.segment_clear(known_goal, known_goal, 0.28)
    assert grid.segment_clear(origin, known_goal, 0.28) is reachable
    engine = NavigationEngine([plan], grid, {"seed": 42, "map_clearance_m": 0.03}, [actor])
    try:
        result = engine.step(0.0, 1 / 60, [actor])
        state = result["agents"][0]
        if reachable:
            assert state["error"] is None
            assert state["phase"] == "wandering"
            target = result["guidance"][actor["name"]]["target"]
            assert 1.0 <= math.dist(origin, target) <= 4.0
            assert grid.segment_clear(origin, target, 0.28)
        else:
            assert state["phase"] == "failed" and "No reachable wander destination" in state["error"]
            assert state["position"] == origin and state["velocity"] == [0.0, 0.0, 0.0]
    finally:
        engine.close()


def test_wander_finds_its_small_reachable_local_component_among_disconnected_free_rooms():
    cells = np.zeros((200, 200), dtype=np.int8)
    cells[80:120, 80:180] = 100
    cells[94:107, 86:170] = 0
    grid = OccupancyGrid(cells, 0.05, (-5.0, -5.0))
    actor = {"name": "wanderer", "position": [0.0, 0.0, 0.0], "yaw": 0.0, "radius_m": 0.25, "height_m": 1.8}
    plan = {"name": actor["name"], "behavior": "wander", "speed": 0.9}
    assert grid.segment_clear(actor["position"], (2.0, 0.0), 0.28)
    engine = NavigationEngine([plan], grid, {"seed": 42, "map_clearance_m": 0.03}, [actor])
    try:
        result = engine.step(0.0, 1 / 60, [actor])
        state = result["agents"][0]
        assert state["error"] is None
        assert state["phase"] == "wandering"
        target = result["guidance"][actor["name"]]["target"]
        assert 1.0 <= math.dist(actor["position"], target) <= 4.0
        assert grid.segment_clear(actor["position"], target, 0.28)
    finally:
        engine.close()


@pytest.mark.parametrize("speed", [0.02, 0.7])
def test_wander_moves_inside_disk_clear_space_and_standing_stays_fixed(speed):
    grid = OccupancyGrid(np.zeros((100, 100), dtype=np.int8), 0.1, (-5.0, -5.0))
    agents = [
        {"name": "wanderer", "position": [-2.0, 0.0, 0.0], "yaw": 0.0, "radius_m": 0.2, "height_m": 1.8},
        {"name": "standing", "position": [2.0, 0.0, 0.0], "yaw": 1.0, "radius_m": 0.2, "height_m": 1.8},
    ]
    plans = [{"name": "wanderer", "behavior": "wander", "speed": speed}, {"name": "standing", "behavior": "stand"}]
    engine = NavigationEngine(plans, grid, {"seed": 19}, agents)
    distance = 0.0
    states = agents
    for tick in range(600):
        result = engine.step(tick / 60, 1 / 60, states)["agents"]
        assert all(state["error"] is None for state in result)
        assert result[1]["position"] == agents[1]["position"] and result[1]["yaw"] == 1.0
        assert grid.segment_clear(states[0]["position"], result[0]["position"], 0.25)
        distance += math.dist(states[0]["position"], result[0]["position"])
        states = result
    assert distance > speed * 2.0


@pytest.mark.parametrize("obstacle_speed", [0.0, 0.6])
def test_measured_passive_obstacles_are_avoided_without_being_integrated(obstacle_speed):
    grid = OccupancyGrid(np.zeros((100, 120), dtype=np.int8), 0.1, (-6.0, -5.0))
    agents = [{"name": "walker", "position": [-3.0, 0.0, 0.0], "yaw": 0.0, "radius_m": 0.2, "height_m": 1.8}]
    plans = [
        {
            "name": "walker",
            "behavior": "walk",
            "position": [-3.0, 0.0, 0.0],
            "speed": 0.8,
            "arrival_tolerance": 0.15,
            "waypoints": [{"position": [3.0, 0.0, 0.0]}],
        }
    ]
    engine = NavigationEngine(plans, grid, {}, agents)
    states = agents
    for tick in range(1200):
        now = tick / 60
        location = [0.0, -2.0 + obstacle_speed * now if obstacle_speed else 0.0, 0.0]
        obstacle = {
            "position": location,
            "radius_m": 0.3,
            "velocity": [0.0, obstacle_speed, 0.0],
            "velocity_valid": True,
            "path": "/moving_obstacle",
        }
        obstacles = [obstacle] if tick < 900 else []
        state = engine.step(now, 1 / 60, states, obstacles)["agents"][0]
        assert state["error"] is None
        assert obstacle["position"] == location
        if obstacles:
            relative = np.asarray(states[0]["position"][:2]) - location[:2]
            motion = (
                np.asarray(state["position"][:2]) - states[0]["position"][:2] - np.array([0.0, obstacle_speed / 60])
            )
            fraction = (
                float(np.clip(-np.dot(relative, motion) / np.dot(motion, motion), 0, 1))
                if np.dot(motion, motion)
                else 0.0
            )
            assert np.linalg.norm(relative + fraction * motion) >= 0.5 - 1e-6
        states = [state]
    assert state["phase"] == "complete"


@pytest.mark.parametrize("polygon", [False, True])
def test_zero_velocity_is_not_accepted_as_a_measurement_when_validity_is_false(polygon):
    grid = OccupancyGrid(np.zeros((60, 60), dtype=np.int8), 0.1, (-3.0, -3.0))
    agents = [{"name": "standing", "position": [0.0, 0.0, 0.0], "yaw": 0.0, "radius_m": 0.2, "height_m": 1.8}]
    engine = NavigationEngine([{"name": "standing", "behavior": "stand"}], grid, {}, agents)
    obstacle = {"position": [2.0, 0.0, 0.0], "radius_m": 0.2, "velocity": [0.0, 0.0, 0.0], "velocity_valid": False}
    if polygon:
        obstacle["footprint_xy_m"] = [[1.8, -0.1], [2.2, -0.1], [2.2, 0.1], [1.8, 0.1]]
    with pytest.raises(ValueError, match="velocity is not observed"):
        engine.step(0.0, 1 / 60, agents, [obstacle])
    obstacle["velocity_valid"] = True
    result = engine.step(0.0, 1 / 60, agents, [obstacle])
    assert result["agents"][0]["position"] == [0.0, 0.0, 0.0]


def test_symmetric_narrow_corridor_does_not_create_lateral_force_switching():
    cells = np.zeros((80, 100), dtype=np.int8)
    cells[:36, :] = cells[44:, :] = 100
    grid = OccupancyGrid(cells, 0.1, (-5.0, -4.0))
    agents = [{"name": "walker", "position": [-3.0, 0.0, 0.0], "yaw": 0.0, "radius_m": 0.2, "height_m": 1.8}]
    plans = [
        {
            "name": "walker",
            "position": [-3.0, 0.0, 0.0],
            "behavior": "walk",
            "speed": 0.8,
            "arrival_tolerance": 0.15,
            "waypoints": [{"position": [3.0, 0.0, 0.0]}],
        }
    ]
    engine = NavigationEngine(plans, grid, {}, agents)
    states = agents
    lateral = []
    headings = []
    for tick in range(900):
        state = engine.step(tick / 60, 1 / 60, states)["agents"][0]
        assert state["error"] is None
        lateral.append(abs(state["position"][1]))
        headings.append(abs(state["yaw"]))
        states = [state]
    assert state["phase"] == "complete"
    assert max(lateral) < 1e-5 and max(headings) < 1e-4


def test_single_walker_makes_continuous_progress_through_an_l_corridor():
    cells = np.full((100, 100), 100, dtype=np.int8)
    cells[45:55, 10:55] = 0
    cells[45:90, 45:55] = 0
    grid = OccupancyGrid(cells, 0.1, (-5.0, -5.0))
    agents = [{"name": "walker", "position": [-3.0, 0.0, 0.0], "yaw": 0.0, "radius_m": 0.2, "height_m": 1.8}]
    plans = [
        {
            "name": "walker",
            "position": [-3.0, 0.0, 0.0],
            "behavior": "walk",
            "speed": 0.8,
            "arrival_tolerance": 0.15,
            "waypoints": [{"position": [0.0, 3.0, 0.0]}],
        }
    ]
    engine = NavigationEngine(plans, grid, {}, agents)
    states = agents
    low = longest = 0.0
    for tick in range(1200):
        state = engine.step(tick / 60, 1 / 60, states)["agents"][0]
        assert state["error"] is None
        low = low + 1 / 60 if state["phase"] == "walking" and math.hypot(*state["velocity"][:2]) < 0.03 else 0.0
        longest = max(longest, low)
        assert grid.segment_clear(states[0]["position"], state["position"], 0.25)
        states = [state]
    assert state["phase"] == "complete" and longest < 1.5


@pytest.mark.parametrize("quarter_turns", [0, 1, 2, 3])
def test_walker_keeps_speed_through_a_swept_clear_corner_preview(quarter_turns):
    cells = np.full((100, 100), 100, dtype=np.int8)
    cells[44:56, 10:56] = 0
    cells[44:90, 44:56] = 0
    grid = OccupancyGrid(np.rot90(cells, -quarter_turns), 0.1, (-5.0, -5.0))
    angle = quarter_turns * math.pi / 2
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    start, goal = [rotation @ point for point in ([-3.0, 0.0], [0.0, 3.0])]
    actor = {"name": "walker", "position": [*start, 0.0], "yaw": angle, "radius_m": 0.2, "height_m": 1.8}
    plan = {
        "name": actor["name"],
        "position": actor["position"],
        "behavior": "walk",
        "speed": 0.8,
        "arrival_tolerance": 0.15,
        "waypoints": [{"position": [*goal, 0.0]}],
    }
    original = copy.deepcopy(plan)
    engine = NavigationEngine([plan], grid, {"obstacle_time_horizon_s": 0.5}, [actor])
    state, dt = actor, 1 / 60
    corner_speeds = []
    try:
        for tick in range(900):
            before = state
            state = engine.step(tick * dt, dt, [before])["agents"][0]
            assert state["error"] is None
            assert grid.segment_clear(before["position"], state["position"], 0.25)
            speed = math.hypot(*state["velocity"][:2])
            assert speed <= plan["speed"] + 1e-6
            turn = math.remainder(state["yaw"] - before["yaw"], math.tau)
            assert abs(turn) <= engine.turn_rate * dt + 1e-5
            if math.hypot(*state["position"][:2]) < 0.65:
                corner_speeds.append(speed)
            if state["phase"] == "complete":
                break
        assert state["phase"] == "complete"
        assert math.dist(state["position"], [*goal, 0.0]) <= plan["arrival_tolerance"]
        assert corner_speeds and np.mean(corner_speeds) > 0.69
        assert plan == original
    finally:
        engine.close()


@pytest.mark.parametrize("quarter_turns", [0, 1, 2, 3])
def test_raster_corner_does_not_require_a_tiny_sideways_arrival_before_continuing(quarter_turns):
    cells = np.full((100, 100), 100, dtype=np.int8)
    cells[10:85, 45:59] = 0
    cells[70:85, 10:59] = 0
    grid = OccupancyGrid(np.rot90(cells, -quarter_turns), 0.05, (-2.5, -2.5))
    angle = quarter_turns * math.pi / 2
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    start, goal = [rotation @ point for point in ([-1.5, 1.32], [0.05, -1.5])]
    actor = {
        "name": "walker",
        "position": [*start, 0.0],
        "yaw": angle,
        "radius_m": 0.25,
        "height_m": 1.8,
    }
    plan = {
        "name": actor["name"],
        "position": actor["position"],
        "behavior": "walk",
        "speed": 0.9,
        "arrival_tolerance": 0.05,
        "waypoints": [{"position": [*goal, 0.0]}],
    }
    original = copy.deepcopy(plan)
    engine = NavigationEngine([plan], grid, {"map_clearance_m": 0.03}, [actor])
    states, total_turn, net_turn, dt = [actor], 0.0, 0.0, 1 / 60
    try:
        for tick in range(1200):
            state = engine.step(tick * dt, dt, states)["agents"][0]
            assert state["error"] is None
            assert grid.segment_clear(states[0]["position"], state["position"], 0.28)
            assert math.dist(states[0]["position"], state["position"]) <= plan["speed"] * dt + 1e-6
            turn = math.remainder(state["yaw"] - states[0]["yaw"], 2 * math.pi)
            assert abs(turn) <= engine.turn_rate * dt + 1e-5
            total_turn += abs(turn)
            net_turn += turn
            if math.hypot(*state["velocity"][:2]) > 0:
                heading = math.atan2(state["velocity"][1], state["velocity"][0])
                assert abs(math.remainder(state["yaw"] - heading, 2 * math.pi)) < 1e-6
            states = [{**actor, **state}]
            if state["phase"] == "complete":
                break
        assert state["phase"] == "complete"
        assert math.dist(state["position"], [*goal, 0.0]) <= plan["arrival_tolerance"]
        # A single elbow must not add a large turn-and-undo near its raster vertex.
        assert total_turn <= abs(net_turn) + math.pi / 2
        assert plan == original
    finally:
        engine.close()


def test_walker_turns_away_before_following_its_velocity_into_a_corner():
    cells = np.zeros((120, 160), dtype=np.int8)
    cells[15:60, 80:125] = 100
    grid = OccupancyGrid(cells, 0.05, (-4.0, -3.0))
    actor = {
        "name": "walker",
        "position": [-0.65, 0.08, 0.0],
        "velocity": [0.9, 0.0, 0.0],
        "yaw": 0.0,
        "radius_m": 0.25,
        "height_m": 1.8,
    }
    goal = [1.4, 0.75, 0.0]
    plan = {
        "name": actor["name"],
        "position": actor["position"],
        "behavior": "walk",
        "speed": 0.9,
        "arrival_tolerance": 0.15,
        "waypoints": [{"position": goal}],
    }
    settings = {"map_clearance_m": 0.0275, "preferred_path_clearance_m": 0.05, "obstacle_time_horizon_s": 0.5}
    state, dt = actor, 1 / 60
    engine = NavigationEngine([plan], grid, settings, [actor])
    try:
        for tick in range(240):
            before = state
            state = engine.step(tick * dt, dt, [before])["agents"][0]
            assert state["error"] is None
            # The open side has room for a route with the preferred wall clearance.
            assert grid.segment_clear(before["position"], state["position"], 0.32)
            assert abs(math.remainder(state["yaw"] - before["yaw"], math.tau)) <= engine.turn_rate * dt + 1e-6
            if state["phase"] == "complete":
                break
        assert state["phase"] == "complete"
        assert math.dist(state["position"], goal) <= plan["arrival_tolerance"]
    finally:
        engine.close()


def test_opposing_walkers_keep_room_from_the_wall_on_a_straight_route():
    cells = np.zeros((84, 200), dtype=np.int8)
    cells[64:] = 100
    grid = OccupancyGrid(cells, 0.05, (-5.0, -1.6))
    states = []
    plans = []
    for name, start, goal, yaw_deg in (
        ("upper", [1.15, 0.87, 0.0], [-2.5, 0.90, 0.0], 156.0),
        ("lower", [-0.65, 0.65, 0.0], [2.5, 0.45, 0.0], -9.0),
    ):
        yaw = math.radians(yaw_deg)
        states.append(
            {
                "name": name,
                "position": start,
                "radius_m": 0.25,
                "height_m": 1.8,
                "yaw": yaw,
                "velocity": [0.86 * math.cos(yaw), 0.86 * math.sin(yaw), 0.0],
            }
        )
        plans.append(
            {
                "name": name,
                "position": start,
                "behavior": "walk",
                "speed": 0.9,
                "arrival_tolerance": 0.1,
                "waypoints": [{"position": goal}],
            }
        )
    settings = {
        "obstacle_time_horizon_s": 0.5,
        "map_clearance_m": 0.0275,
        "pair_clearance_m": 0.225,
        "preferred_path_clearance_m": 0.05,
        "neighbor_distance_m": 4.8,
    }
    engine = NavigationEngine(plans, grid, settings, states)
    dt = 1 / 60
    try:
        for tick in range(360):
            before = states
            states = engine.step(tick * dt, dt, before)["agents"]
            assert all(state["error"] is None for state in states)
            # A head-on sidestep has room to leave more than the hard wall margin.
            assert grid.segment_clear(before[0]["position"], states[0]["position"], 0.36)
            assert grid.segment_clear(before[1]["position"], states[1]["position"], 0.2775)
            relative = np.asarray(before[0]["position"][:2]) - before[1]["position"][:2]
            end = np.asarray(states[0]["position"][:2]) - states[1]["position"][:2]
            assert engine._chord_distance(relative, end - relative) >= 0.725 - 1e-9
            if all(state["phase"] == "complete" for state in states):
                break
        assert all(state["phase"] == "complete" for state in states)
    finally:
        engine.close()


def test_native_cached_corner_refresh_keeps_progress_at_open_portal():
    fixture_path = Path(__file__).with_name("fixtures") / "native_cached_corner_20261109.npz"
    with np.load(fixture_path, allow_pickle=False) as fixture:
        snapshot = json.loads(str(fixture["snapshot_json"]))
        grid = OccupancyGrid(fixture["occupancy"], float(fixture["resolution_m"]), tuple(fixture["origin_xy_m"]))
    settings = snapshot["settings"]
    states = snapshot["observed"]
    plans = snapshot["plans"]
    original = copy.deepcopy(plans)
    engine = NavigationEngine(plans, grid, settings, states)
    for name, previous in snapshot["routes"].items():
        vars(engine.routes[name]).update(previous)
    initial = next(actor for actor in states if actor["name"] == "walker_02")
    dt = 1 / settings["control_hz"]
    footprints = [convex_footprint(item["footprint_xy_m"]) for item in snapshot["obstacles"]]
    try:
        for tick in range(round(0.5 / dt)):
            result = engine.step(snapshot["time_s"] + tick * dt, dt, states, snapshot["obstacles"], snapshot["doors"])
            commands = {actor["name"]: actor for actor in result["agents"]}
            for index, before in enumerate(states):
                after = commands[before["name"]]
                assert after["error"] is None
                clearance = before["radius_m"] + settings["map_clearance_m"]
                assert grid.segment_clear(before["position"], after["position"], clearance)
                assert all(
                    segment_polygon_clear(before["position"][:2], after["position"][:2], polygon, clearance)
                    for polygon in footprints
                )
                turn = abs(math.remainder(after["yaw"] - before["yaw"], math.tau))
                assert turn <= settings["max_turn_rate_radps"] * dt + 1e-6
                for peer in states[:index]:
                    start = np.asarray(before["position"][:2]) - peer["position"][:2]
                    end = np.asarray(after["position"][:2]) - commands[peer["name"]]["position"][:2]
                    delta = end - start
                    squared = float(np.dot(delta, delta))
                    fraction = float(np.clip(-np.dot(start, delta) / squared, 0, 1)) if squared else 0.0
                    required = before["radius_m"] + peer["radius_m"] + settings["pair_clearance_m"]
                    assert np.linalg.norm(start + fraction * delta) >= required - 1e-9
            states = [{**actor, **commands[actor["name"]]} for actor in states]
        final = next(actor for actor in states if actor["name"] == initial["name"])
        # A static-clear cached chord used to stop this moving actor at the jamb.
        assert math.dist(final["position"], initial["position"]) > initial["radius_m"]
        assert plans == original
    finally:
        engine.close()


def test_thin_dynamic_door_footprint_does_not_falsely_block_a_clear_corridor():
    cells = np.full((40, 100), 100, dtype=np.int8)
    cells[15:27, :] = 0
    grid = OccupancyGrid(cells, 0.05, (-2.5, -0.5))
    agents = [{"name": "walker", "position": [-1.5, 0.55, 0.0], "yaw": 0.0, "radius_m": 0.2, "height_m": 1.8}]
    plans = [
        {
            "name": "walker",
            "position": [-1.5, 0.55, 0.0],
            "behavior": "walk",
            "speed": 0.6,
            "arrival_tolerance": 0.15,
            "waypoints": [{"position": [1.5, 0.55, 0.0]}],
        }
    ]
    door = {
        "position": [0.0, 0.0, 0.0],
        "radius_m": 0.35,
        "velocity": [0.0, 0.0, 0.0],
        "velocity_valid": True,
        "footprint_xy_m": [[-0.35, -0.02], [0.35, -0.02], [0.35, 0.02], [-0.35, 0.02]],
    }
    engine = NavigationEngine(plans, grid, {"map_clearance_m": 0.03, "obstacle_time_horizon_s": 0.5}, agents)
    states = agents
    for tick in range(1200):
        state = engine.step(tick / 60, 1 / 60, states, [door])["agents"][0]
        assert state["error"] is None
        states = [state]
        if state["phase"] == "complete":
            break
    assert state["phase"] == "complete"


def test_global_route_goes_around_an_observed_solid_barrier_missing_from_the_static_map():
    grid = OccupancyGrid(np.zeros((200, 200), dtype=np.int8), 0.05, (-5.0, -5.0))
    actor = {"name": "walker", "position": [-2.0, 0.0, 0.0], "yaw": 0.0, "radius_m": 0.25, "height_m": 1.8}
    plan = {
        "name": "walker",
        "position": actor["position"],
        "behavior": "walk",
        "speed": 0.9,
        "arrival_tolerance": 0.2,
        "waypoints": [{"position": [2.0, 0.0, 0.0], "wait": 0.0, "yaw": None}],
    }
    barrier = {
        "prim_path": "/uncontrolled/barrier",
        "position": [0.0, 0.0, 0.0],
        "radius_m": 1.6,
        "velocity": [0.0, 0.0, 0.0],
        "velocity_valid": True,
        "footprint_xy_m": [[-0.15, -1.5], [0.15, -1.5], [0.15, 1.5], [-0.15, 1.5]],
    }
    engine = NavigationEngine([plan], grid, {"map_clearance_m": 0.03, "obstacle_time_horizon_s": 0.5}, [actor])
    states, furthest = [actor], 0.0
    try:
        for tick in range(1500):
            result = engine.step(tick / 60, 1 / 60, states, [barrier])
            state = result["agents"][0]
            assert state["error"] is None
            assert segment_polygon_clear(states[0]["position"], state["position"], barrier["footprint_xy_m"], 0.28)
            furthest = max(furthest, abs(state["position"][1]))
            states = [state]
            if state["phase"] == "complete":
                break
    finally:
        engine.close()
    assert state["phase"] == "complete" and math.dist(state["position"], plan["waypoints"][0]["position"]) <= 0.2
    assert furthest > 1.5 + actor["radius_m"]


@pytest.mark.parametrize("enabled", [False, True])
def test_only_a_declared_controlled_door_is_deferred_to_the_door_interlock(enabled):
    grid = OccupancyGrid(np.zeros((200, 200), dtype=np.int8), 0.05, (-5.0, -5.0))
    actor = {"name": "walker", "position": [-2.0, 0.0, 0.0], "yaw": 0.0, "radius_m": 0.25}
    plan = {
        "name": "walker",
        "position": actor["position"],
        "behavior": "walk",
        "speed": 0.9,
        "waypoints": [{"position": [2.0, 0.0, 0.0]}],
    }
    footprint = [[-0.15, -1.5], [0.15, -1.5], [0.15, 1.5], [-0.15, 1.5]]
    spec = {
        "name": "door",
        "body_prim_path": "/barrier",
        "enabled": enabled,
        "open_angle_deg": 90.0,
        "closed_angle_deg": 0.0,
        "approach_distance_m": 0.55,
        "interior_polygon_xy_m": [[0.0, -3.0], [4.0, -3.0], [4.0, 3.0], [0.0, 3.0]],
        "swept_footprint_xy_m": footprint,
        "doorway_segment_xy_m": [[0.0, -1.5], [0.0, 1.5]],
    }
    obstacle = {
        "prim_path": "/barrier/physical_shape",
        "position": [0.0, 0.0, 0.0],
        "radius_m": 1.6,
        "footprint_xy_m": footprint,
        "velocity": [0.0, 0.0, 0.0],
        "velocity_valid": True,
    }
    engine = NavigationEngine([plan], grid, {"door_specs": [spec], "map_clearance_m": 0.03}, [actor])
    try:
        result = engine.step(0.0, 1 / 60, [actor], [obstacle])
        assert not result["agents"][0]["error"]
        target = result["guidance"]["walker"]["target"]
        assert (abs(target[1]) < 1e-9) is enabled
        assert result["agents"][0]["position"][0] < -1.9
    finally:
        engine.close()


def test_a_fast_moving_polygon_crossing_is_rejected_even_when_both_endpoints_are_clear():
    grid = OccupancyGrid(np.zeros((100, 100), dtype=np.int8), 0.1, (-5.0, -5.0))
    agents = [{"name": "standing", "position": [0.0, 0.0, 0.0], "yaw": 0.0, "radius_m": 0.2, "height_m": 1.8}]
    engine = NavigationEngine(
        [{"name": "standing", "behavior": "stand"}], grid, {"obstacle_time_horizon_s": 0.5}, agents
    )
    obstacle = {
        "position": [0.0, -0.95, 0.0],
        "radius_m": 0.41,
        "velocity": [0.0, 120.0, 0.0],
        "velocity_valid": True,
        "footprint_xy_m": [[-0.4, -1.0], [0.4, -1.0], [0.4, -0.9], [-0.4, -0.9]],
    }
    with pytest.raises(RuntimeError, match="no certified next step"):
        engine.step(0.0, 1 / 60, agents, [obstacle])


def test_real_near_peer_snapshot_keeps_heading_steady_while_reaching_its_visible_target():
    fixture_dir = Path(__file__).parent / "fixtures"
    snapshot = json.loads((fixture_dir / "native_near_peer_heading_20261106.json").read_text())
    with np.load(fixture_dir / snapshot["provenance"]["map_fixture"], allow_pickle=False) as fixture:
        grid = OccupancyGrid(fixture["occupancy"], float(fixture["resolution_m"]), tuple(fixture["origin_xy_m"]))
    agents, plans, settings = snapshot["agents"], snapshot["plans"], snapshot["settings"]
    original_plans = copy.deepcopy(plans)
    focus = next(index for index, actor in enumerate(agents) if actor["name"] == snapshot["focus_agent"])
    goal = plans[focus]["waypoints"][0]["position"]
    dt = 1 / settings["control_hz"]
    engine = NavigationEngine(plans, grid, settings, agents)
    states, total_turn, net_turn = agents, 0.0, 0.0
    try:
        for tick in range(round(snapshot["duration_s"] / dt)):
            result = engine.step(tick * dt, dt, states, snapshot["obstacles"])
            next_states = result["agents"]
            for index, (before, after) in enumerate(zip(states, next_states)):
                radius = agents[index]["radius_m"]
                assert after["error"] is None
                assert math.hypot(*after["velocity"][:2]) <= plans[index]["speed"] + 1e-6
                assert math.dist(before["position"], after["position"]) <= plans[index]["speed"] * dt + 1e-6
                turn = math.remainder(after["yaw"] - before["yaw"], 2 * math.pi)
                assert abs(turn) <= settings["max_turn_rate_radps"] * dt + 1e-6
                if math.hypot(*after["velocity"][:2]) > 0.0:
                    movement_heading = math.atan2(after["velocity"][1], after["velocity"][0])
                    assert abs(math.remainder(after["yaw"] - movement_heading, 2 * math.pi)) < 1e-6
                assert grid.segment_clear(before["position"], after["position"], radius + settings["map_clearance_m"])
                for obstacle in snapshot["obstacles"]:
                    relative_end = np.asarray(after["position"][:2]) - np.asarray(obstacle["velocity"][:2]) * dt
                    assert segment_polygon_clear(
                        before["position"],
                        relative_end,
                        convex_footprint(obstacle["footprint_xy_m"]),
                        radius + settings["map_clearance_m"],
                    )
                if index == focus:
                    total_turn += abs(turn)
                    net_turn += turn
                for peer, other in enumerate(next_states[:index]):
                    relative = np.asarray(before["position"][:2]) - states[peer]["position"][:2]
                    change = (
                        np.asarray(after["position"][:2])
                        - before["position"][:2]
                        - np.asarray(other["position"][:2])
                        + states[peer]["position"][:2]
                    )
                    squared = float(np.dot(change, change))
                    fraction = float(np.clip(-np.dot(relative, change) / squared, 0, 1)) if squared else 0.0
                    assert np.linalg.norm(relative + fraction * change) >= (
                        radius + agents[peer]["radius_m"] + settings["pair_clearance_m"] - 1e-12
                    )
            target = result["guidance"][snapshot["focus_agent"]]["target"]
            if target is not None:
                assert math.dist(target, goal) < 1e-9
            states = next_states
            if states[focus]["phase"] == "complete":
                break
        assert states[focus]["phase"] == "complete"
        assert math.dist(states[focus]["position"], goal) <= plans[focus]["arrival_tolerance"]
        assert total_turn - abs(net_turn) <= math.radians(snapshot["reverse_excess_limit_deg"])
        assert plans == original_plans
    finally:
        engine.close()
