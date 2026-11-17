# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded, measured passing transactions retain the original route contract."""

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pedstrain_simulation.simulation.ros2_navigation.goal_occupancy import _ReservationGrid
from pedstrain_simulation.simulation.ros2_navigation.passing import PassingCoordinator, _OverlayGrid, _Transaction
from pedstrain_simulation.simulation.ros2_navigation.planner import OccupancyGrid, segment_polygon_clear
from pedstrain_simulation.simulation.ros2_navigation.solver import NavigationEngine


class _FullEndpointCalculation(_OverlayGrid):
    def _goal_endpoints(self, goal, tolerance, mask, clearance):
        return OccupancyGrid._goal_endpoints(self, goal, tolerance, mask, clearance)


@pytest.mark.parametrize("value", [-1, 100])
def test_actual_start_component_keeps_diagonally_touching_rooms_disconnected(value):
    grid = OccupancyGrid(np.array([[0, value], [value, 0]], dtype=np.int8), 1.0, (0.0, 0.0))
    component = grid._reachable_mask((0.5, 0.5), 0.28)
    assert component is not None and component.shape == (2, 2)
    assert np.array_equal(component, [[True, False], [False, False]])
    assert not component.flags.writeable


@pytest.mark.parametrize("constraint", ["peers", "reservations"])
def test_actual_start_component_uses_every_certified_subcell_seed(constraint):
    base = OccupancyGrid(np.zeros((120, 200), dtype=np.int8), 0.02, (-2.0, -1.2))
    centers = [np.array([0.0, 0.751]), np.array([0.0, -0.751])]
    grid = (
        _OverlayGrid(base, 0.25, 0.03, [(center, 0.75) for center in centers], [])
        if constraint == "peers"
        else _ReservationGrid(base, 0.25, 0.03, [], [], [(center, center, 0.75) for center in centers])
    )
    component = grid._reachable_mask((0.0, 0.0), 0.28)
    assert component is not None
    for target in ((-1.0, 0.0), (1.0, 0.0)):
        assert grid.segment_clear((0.0, 0.0), target, 0.28)
        assert component[grid.world_to_cell(target)]


def test_missing_raster_seeds_do_not_claim_a_clear_continuous_route_is_unreachable():
    base = OccupancyGrid(np.zeros((120, 200), dtype=np.int8), 0.02, (-2.0, -1.2))
    grid = _OverlayGrid(base, 0.25, 0.03, [(np.array([0.0, y]), 0.75) for y in (0.750001, -0.750001)], [])
    start, goal = (0.0, 0.0), (1.0, 0.0)
    assert grid._reachable_mask(start, 0.28) is None
    assert grid.plan(start, goal, 0.25, 0.2, 0.03) == [start, goal]


def test_an_empty_exact_grid_component_retains_its_disk_safe_boundary():
    grid = OccupancyGrid(np.zeros((100, 100), dtype=np.int8), 0.02, (0.0, 0.0))
    actual = grid._reachable_mask((1.0, 1.0), 0.28)
    assert actual is not None and np.array_equal(actual, grid.free_mask(0.28))
    assert not actual[0, 0] and actual[50, 50] and not actual.flags.writeable


@pytest.mark.parametrize("constraint", ["peers", "reservations"])
@pytest.mark.parametrize("sealed", [False, True])
def test_a_safe_subcell_start_connects_beyond_the_local_seed_window_without_crossing_a_sealed_wall(constraint, sealed):
    cells = np.zeros((200, 200), dtype=np.int8)
    if sealed:
        cells[:, 125:130] = 100
    base = OccupancyGrid(cells, 0.02, (-2.0, -2.0))
    radius, margin, gap, tolerance = 0.25, 0.03, 0.25, 0.2
    required = 2 * radius + gap
    centers = [np.array([0.0, required + 0.001]), np.array([0.0, -required - 0.001])]
    grid = (
        _OverlayGrid(base, radius, margin, [(center, required) for center in centers], [])
        if constraint == "peers"
        else _ReservationGrid(base, radius, margin, [], [], [(center, center, required) for center in centers])
    )
    start, goal = (0.0, 0.0), (1.0, 0.5)
    assert grid.segment_clear(start, start, radius + margin)
    if sealed:
        with pytest.raises(ValueError, match="No reachable path"):
            grid.plan(start, goal, radius, tolerance, margin)
        return
    witness = [start, (0.21, 0.01), (0.8, 0.0), goal]
    assert all(grid.segment_clear(a, b, radius + margin) for a, b in zip(witness, witness[1:]))
    path = grid.plan(start, goal, radius, tolerance, margin)
    assert path[0] == start and math.dist(path[-1], goal) <= tolerance
    for a, b in zip(path, path[1:]):
        assert grid.segment_clear(a, b, radius + margin)
        delta = np.asarray(b) - a
        squared = np.dot(delta, delta)
        for center in centers:
            fraction = np.clip(np.dot(center - a, delta) / squared, 0.0, 1.0) if squared else 0.0
            assert np.linalg.norm(center - a - fraction * delta) >= required - 1e-9


@pytest.mark.parametrize("constraint", ["peers", "reservations"])
def test_an_already_arrived_safe_pose_does_not_require_a_grid_center_in_its_goal_ball(constraint):
    base = OccupancyGrid(np.zeros((200, 200), dtype=np.int8), 0.02, (-2.0, -2.0))
    radius, margin, required = 0.25, 0.03, 0.75
    centers = [np.array([0.0, 0.751]), np.array([0.0, -0.751])]
    grid = (
        _OverlayGrid(base, radius, margin, [(center, required) for center in centers], [])
        if constraint == "peers"
        else _ReservationGrid(base, radius, margin, [], [], [(center, center, required) for center in centers])
    )
    start, goal, tolerance = (0.0, 0.0), (0.0, 0.2), 0.2
    assert grid.segment_clear(start, start, radius + margin)
    path = grid.plan(start, goal, radius, tolerance, margin)
    assert path and all(math.dist(point, start) == 0 for point in path)
    assert math.dist(path[-1], goal) <= tolerance


def test_already_arrived_actors_do_not_start_a_new_opposing_passing_transaction():
    grid = OccupancyGrid(np.zeros((120, 160), dtype=np.int8), 0.05, (-4.0, -3.0))
    states = [
        {"name": name, "position": [x, 0.0, 0.0], "velocity": [0.0, 0.0, 0.0], "radius_m": 0.25, "yaw": 0.0}
        for name, x in (("arrived", 0.0), ("approaching", 0.8))
    ]
    intentions = {
        name: {
            "active": True,
            "target": [x, 0.0, 0.0],
            "arrival_tolerance_m": 0.2,
            "speed_mps": 0.9,
            "route_context": (0, 0),
        }
        for name, x in (("arrived", 0.1), ("approaching", -1.0))
    }
    original = copy.deepcopy((states, intentions))
    coordinator = PassingCoordinator(grid, {"map_clearance_m": 0.03, "pair_clearance_m": 0.25})
    for now in (0.0, 1.5, 2.25):
        result = coordinator.update(now, states, intentions, [], {})
        assert result["status"]["transactions"] == []
    assert (states, intentions) == original


@pytest.mark.parametrize(
    ("lane_distance", "holder_speed", "wall_ahead", "needs_retreat"),
    [(0.754, 0.8, False, True), (0.81, 0.8, False, False), (0.754, 0.065, False, True), (0.754, 0.065, True, False)],
)
def test_passing_uses_the_measured_holding_pose_only_with_clearance_reserve(
    lane_distance, holder_speed, wall_ahead, needs_retreat
):
    cells = np.zeros((100, 150), dtype=np.int8)
    if wall_ahead:
        cells[60:78, 92:96] = 100
    grid = OccupancyGrid(cells, 0.05, (-4.0, -2.5))
    states = [
        {
            "name": "holder",
            "position": [0.0, lane_distance, 0.0],
            "velocity": [holder_speed, 0.0, 0.0],
            "radius_m": 0.25,
            "yaw": 0.0,
        },
        {
            "name": "passer",
            "position": [1.2, 0.0, 0.0],
            "velocity": [0.0, 0.0, 0.0],
            "radius_m": 0.25,
            "yaw": math.pi / 2 + 0.07,
        },
    ]
    intentions = {
        name: {
            "active": True,
            "target": target,
            "arrival_tolerance_m": 0.2,
            "speed_mps": 0.9,
            "route_context": (0, 0),
        }
        for name, target in (("holder", [1.8, -0.1, 0.0]), ("passer", [-1.8, 0.0, 0.0]))
    }
    original = copy.deepcopy((states, intentions))
    coordinator = PassingCoordinator(grid, {"map_clearance_m": 0.0275, "pair_clearance_m": 0.225})
    coordinator.update(0.0, states, intentions, [], {})
    result = coordinator.update(1.5, states, intentions, [], {})
    transaction = result["status"]["transactions"][0]
    assert transaction["yielder"] == "holder"
    assert transaction["passer"] == "passer"
    if needs_retreat:
        assert transaction["phase"] == "retreat"
        assert result["objectives"]["holder"]["phase"] == "passing_yield"
        expected = intentions["passer"]["target"][:2] if holder_speed >= 0.12 else None
        assert result["objectives"]["passer"]["target"] == expected
    else:
        assert transaction["phase"] == "passing"
        assert transaction["refuge_xy_m"] == states[0]["position"][:2]
        assert result["objectives"]["holder"]["target"] is None
        assert result["objectives"]["passer"]["target"] == intentions["passer"]["target"][:2]
    assert (states, intentions) == original


@pytest.mark.parametrize("unsafe_hold", [False, True], ids=["blocked_chord", "unsafe_hold"])
def test_passing_exchanges_order_locally_when_the_retreat_would_approach_a_wall(unsafe_hold):
    cells = np.zeros((100, 100), dtype=np.int8)
    cells[76:] = 100  # Upper wall at y = 1.3 m.
    cells[56, :51] = 100  # Partition ends at x = 0.05 m between the two lanes.
    grid = OccupancyGrid(cells, 0.05, (-2.5, -2.5))
    radius, margin, gap = 0.25, 0.03, 0.25
    states = [
        {"name": name, "position": pose, "velocity": velocity, "radius_m": radius, "yaw": yaw}
        for name, pose, velocity, yaw in (
            ("holder", [0.0, 0.83, 0.0], [0.0, 0.0, 0.0], 3 * math.pi / 4),
            ("passer", [-0.8, 0.0, 0.0], [0.025, 0.0, 0.0], 0.0),
        )
    ]
    tolerance = 0.05
    intentions = {
        name: {
            "active": True,
            "target": target,
            "arrival_tolerance_m": tolerance,
            "speed_mps": 0.9,
            "route_context": (0, 0),
        }
        for name, target in (("holder", [-1.5, 0.83, 0.0]), ("passer", [0.4, 0.28, 0.0]))
    }
    original = copy.deepcopy(intentions)
    coordinator = PassingCoordinator(
        grid, {"map_clearance_m": margin, "preferred_path_clearance_m": 0.05, "pair_clearance_m": gap}
    )
    holder, passer = states
    # The holder blocks every safe arrival at the goal while standing here.
    required = 2 * radius + gap
    assert math.dist(holder["position"], intentions["passer"]["target"]) + tolerance < required
    coordinator.update(0.0, states, intentions, [], {})
    result = coordinator.update(1.5, states, intentions, [], {})
    [transaction] = result["status"]["transactions"]
    assert transaction["phase"] == "passing"
    assert transaction["yielder"] == "holder" and transaction["passer"] == "passer"
    assert transaction["refuge_xy_m"] == holder["position"][:2]
    assert result["objectives"]["holder"]["target"] is None
    assert result["objectives"]["passer"]["phase"] == "passing_go"

    origin, start = (np.asarray(state["position"][:2]) for state in states)
    point = np.asarray(result["objectives"]["passer"]["target"])
    axis = (point - start) / math.dist(point, start)
    nearest = start + np.dot(origin - start, axis) * axis
    assert np.dot(origin - start, axis) > 0 and np.dot(point - origin, axis) > 0.15
    assert math.dist(origin, nearest) >= required + 0.05
    assert grid.segment_clear(start, point, radius + margin)
    assert math.dist(point, intentions["passer"]["target"][:2]) > tolerance

    if unsafe_hold:
        # The obstacle will reach the holder in the next control tick and stays clear of the passer.
        obstacle_radius = 0.1
        obstacle = SimpleNamespace(
            name="obstacle",
            position=origin + [radius + obstacle_radius + 0.01, 0.0],
            velocity=np.array([-1.0, 0.0]),
            radius=obstacle_radius,
            footprint=None,
        )
        assert coordinator.update(2.0, states, intentions, [obstacle], {}) == {
            "objectives": {},
            "status": {"transactions": []},
        }
        assert intentions == original
        return

    obstacle = SimpleNamespace(name="obstacle", position=point, velocity=np.zeros(2), radius=0.1, footprint=None)
    assert all(math.dist(state["position"][:2], point) > radius + obstacle.radius for state in states)
    blocked = coordinator.update(2.0, states, intentions, [obstacle], {})
    [waiting] = blocked["status"]["transactions"]
    assert waiting["deadline_s"] == transaction["deadline_s"]
    assert {objective["phase"] for objective in blocked["objectives"].values()} == {"passing_wait"}
    assert all(blocked["objectives"][name]["target"] is None for name in intentions)
    clear = coordinator.update(2.1, states, intentions, [], {})
    assert clear["status"]["transactions"][0]["deadline_s"] == transaction["deadline_s"]
    assert clear["objectives"]["passer"]["target"] == point.tolist()

    passer["position"] = [*nearest, 0.0]
    assert coordinator.update(3.0, states, intentions, [], {})["status"]["transactions"]
    passer["position"] = [*point, 0.0]
    assert coordinator.update(3.5, states, intentions, [], {}) == {"objectives": {}, "status": {"transactions": []}}
    assert intentions == original


def test_refuge_prefers_room_from_a_stopped_third_walker():
    cells = np.zeros((160, 250), dtype=np.int8)
    cells[:60, 100:210] = 100
    grid = OccupancyGrid(cells, 0.05, (-5.0, -3.0))
    settings = {"map_clearance_m": 0.0275, "pair_clearance_m": 0.225, "neighbor_distance_m": 4.8}
    coordinator = PassingCoordinator(grid, settings)
    actors, intentions = [], {}
    for name, position, yaw, velocity, goal in (
        ("west", (-0.7457, 0.046), -161, (-0.13, -0.045), (-4.4, -1.3)),
        ("east", (-0.69794, -0.70541), 46, (0.0, 0.0), (2.5, 1.3)),
        ("parked", (-1.80247, 0.07117), 90, (0.0, 0.0), None),
        ("nearby", (-1.802, 0.826), -90, (0.0, -0.01), None),
    ):
        actors.append(
            {
                "name": name,
                "position": [*position, 0.0],
                "yaw": math.radians(yaw),
                "velocity": [*velocity, 0.0],
                "radius_m": 0.25,
            }
        )
        intentions[name] = {
            "active": goal is not None,
            "target": [*goal, 0.0] if goal else None,
            "arrival_tolerance_m": 0.12,
            "speed_mps": 0.9,
            "route_context": (0, 0),
            "path": [],
        }
    original = copy.deepcopy((actors, intentions))
    coordinator.update(0.0, actors, intentions, [], {})
    coordinator.update(1.51, actors, intentions, [], {})
    result = coordinator.update(2.28, actors, intentions, [], {})
    transaction = result["status"]["transactions"][0]
    refuge = transaction["refuge_xy_m"]
    assert transaction["phase"] == "retreat"
    assert math.dist(refuge, actors[2]["position"][:2]) >= 0.775
    assert math.dist(transaction["origin_xy_m"], refuge) <= 0.4
    assert "parked" not in result["objectives"]
    assert (actors, intentions) == original


@pytest.mark.parametrize("spacing,goal_z,finished", [(0.76, 0.0, True), (0.74, 0.0, False), (0.76, 0.3, False)])
def test_existing_passing_finishes_only_for_actual_three_dimensional_arrival_with_required_gap(
    spacing, goal_z, finished
):
    grid = OccupancyGrid(np.zeros((120, 160), dtype=np.int8), 0.05, (-4.0, -3.0))
    states = [
        {"name": name, "position": [x, 0.0, 0.0], "velocity": [0.0, 0.0, 0.0], "radius_m": 0.25, "yaw": 0.0}
        for name, x in (("holder", 0.0), ("passer", spacing))
    ]
    intentions = {
        name: {"active": True, "target": target, "arrival_tolerance_m": 0.2, "speed_mps": 0.9, "route_context": (0, 0)}
        for name, target in (("holder", [-2.0, 0.0, 0.0]), ("passer", [spacing, 0.0, goal_z]))
    }
    original = copy.deepcopy((states, intentions))
    coordinator = PassingCoordinator(grid, {"map_clearance_m": 0.03, "pair_clearance_m": 0.25})
    coordinator.transactions[("holder", "passer")] = _Transaction(
        "holder",
        "passer",
        np.array([0.0, 0.0]),
        np.array([-0.398, 0.0]),
        tuple(coordinator._context(intentions[name]) for name in ("holder", "passer")),
        0.0,
        4.0,
        [[spacing, 0.0], [spacing, 0.0]],
        np.array([1.0, 0.0]),
    )
    result = coordinator.update(1.0, states, intentions, [], {})
    assert bool(result["status"]["transactions"]) is not finished
    if finished:
        assert result["objectives"] == {}
    else:
        assert result["objectives"]["passer"]["phase"] == "passing_wait"
    assert (states, intentions) == original


@pytest.mark.parametrize("constraint", ["peer", "disc", "polygon"])
@pytest.mark.parametrize("near", [False, True])
def test_overlay_goal_regions_retain_full_calculation_with_near_and_far_observed_constraints(constraint, near):
    cells = np.zeros((100, 100), dtype=np.int8)
    base = OccupancyGrid(cells, 0.02, (0.0, 0.0))
    start, target = (1.5, 0.5), (0.01, 0.5)
    radius, margin, tolerance = 0.05, 0.01, 0.2
    base.plan(start, target, radius, tolerance, margin)
    center = np.array([0.15, 0.5] if near else [1.5, 1.5])
    peers = [(center, 0.15)] if constraint == "peer" else []
    obstacles = []
    if constraint != "peer":
        polygon = None
        if constraint == "polygon":
            polygon = center + np.array([[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05]])
        obstacles = [SimpleNamespace(position=center, velocity=np.zeros(2), radius=0.08, footprint=polygon)]
    actual_grid = _OverlayGrid(base, radius, margin, peers, obstacles)
    reference = _FullEndpointCalculation(base, radius, margin, peers, obstacles)
    try:
        expected = reference.plan(start, target, radius, tolerance, margin)
    except ValueError:
        with pytest.raises(ValueError):
            actual_grid.plan(start, target, radius, tolerance, margin)
        return
    actual = actual_grid.plan(start, target, radius, tolerance, margin)
    assert actual == expected
    assert math.dist(actual[-1], target) <= tolerance
    assert all(reference.segment_clear(a, b, radius + margin) for a, b in zip(actual, actual[1:]))


@pytest.mark.parametrize("overlay", [False, True])
def test_unknown_grid_geometry_keeps_its_subcell_goal_boundary(overlay):
    cells = np.zeros((60, 80), dtype=np.int8)
    physical = cells.copy()
    physical[:, :2] = 100
    extra = OccupancyGrid(physical, 0.02, (0.0, 0.0))
    base = OccupancyGrid(cells, 0.02, (0.0, 0.0))
    start, target = (1.0, 0.5), (0.04, 0.5)
    radius, tolerance = 0.1, 0.12
    base.plan(start, target, radius, tolerance, 0.0)

    class AdditionalGeometry(_OverlayGrid if overlay else OccupancyGrid):
        def free_mask(self, clearance):
            return extra.free_mask(clearance)

        def segment_clear(self, a, b, clearance):
            return extra.segment_clear(a, b, clearance)

    grid = AdditionalGeometry(base, radius, 0.0, [], []) if overlay else AdditionalGeometry(cells, 0.02, (0.0, 0.0))
    actual = grid.plan(start, target, radius, tolerance, 0.0)
    assert actual == extra.plan(start, target, radius, tolerance, 0.0)
    assert all(extra.segment_clear(a, b, radius) for a, b in zip(actual, actual[1:]))


def test_overlay_fixed_radius_clearance_keeps_its_original_goal_calculation():
    base = OccupancyGrid(np.zeros((60, 80), dtype=np.int8), 0.02, (0.0, 0.0))
    start, target = (1.0, 0.5), (0.01, 0.5)
    radius, margin, tolerance = 0.05, 0.01, 0.2
    base.plan(start, target, radius, tolerance, margin)
    overlay = _OverlayGrid(base, 0.08, margin, [], [])
    reference = _FullEndpointCalculation(base, 0.08, margin, [], [])
    actual = overlay.plan(start, target, radius, tolerance, margin)
    assert actual == reference.plan(start, target, radius, tolerance, margin)
    assert all(reference.segment_clear(a, b, radius + margin) for a, b in zip(actual, actual[1:]))


def test_passing_replans_around_a_new_peer_instead_of_returning_to_a_passed_vertex():
    grid = OccupancyGrid(np.zeros((120, 160), dtype=np.int8), 0.05, (-4.0, -3.0))
    actors = [
        {
            "name": name,
            "position": [x, y, 0.0],
            "velocity": [0.0, 0.0, 0.0],
            "radius_m": 0.25,
            "height_m": 1.8,
        }
        for name, x, y in (("holder", 3.0, 2.0), ("walker", 0.0, 0.0), ("new_peer", 1.5, -1.0))
    ]
    intentions = {
        actor["name"]: {
            "active": actor["name"] != "new_peer",
            "target": [3.5, 0.0, 0.0],
            "arrival_tolerance_m": 0.2,
            "speed_mps": 0.9,
            "route_context": (0, 0),
        }
        for actor in actors
    }
    original = copy.deepcopy(intentions)
    coordinator = PassingCoordinator(grid, {"map_clearance_m": 0.03, "pair_clearance_m": 0.25})
    coordinator.transactions[("holder", "walker")] = _Transaction(
        "holder",
        "walker",
        np.array([3.0, 2.0]),
        np.array([3.0, 2.0]),
        (((0, 0), (3.5, 0.0, 0.0)),) * 2,
        0.0,
        50.0,
        [[0.0, 0.0], [-1.0, 1.0], [0.0, 2.0], [3.5, 0.0]],
        np.array([1.0, 0.0]),
        "passing",
    )
    first = coordinator.update(2.0, actors, intentions, [], {})
    assert first["objectives"]["walker"]["target"] == [3.5, 0.0]
    actors[1]["position"] = [0.015, 0.0, 0.0]
    actors[2]["position"] = [1.5, 0.0, 0.0]
    result = coordinator.update(3.2, actors, intentions, [], {})
    directive = result["objectives"]["walker"]
    assert directive["phase"] == "passing_go"
    assert directive["target"][0] > actors[1]["position"][0]
    assert grid.segment_clear(actors[1]["position"], directive["target"], 0.28)
    for peer in (actors[0], actors[2]):
        start = np.asarray(actors[1]["position"][:2]) - peer["position"][:2]
        delta = np.asarray(directive["target"]) - actors[1]["position"][:2]
        fraction = np.clip(-np.dot(start, delta) / np.dot(delta, delta), 0.0, 1.0)
        assert np.linalg.norm(start + fraction * delta) >= 0.75
    assert intentions == original


@pytest.mark.parametrize("safe_handoff", [True, False])
def test_passing_releases_an_obsolete_route_when_a_peer_occupies_the_goal(safe_handoff):
    grid = OccupancyGrid(np.zeros((120, 160), dtype=np.int8), 0.05, (-4.0, -3.0))
    actors = [
        {"name": name, "position": [x, y, 0.0], "velocity": [0.0, 0.0, 0.0], "radius_m": 0.25}
        for name, x, y in (("holder", -0.5, 0.9), ("passer", -1.0, 0.0), ("blocker", 2.0, 0.0))
    ]
    intentions = {
        name: {
            "active": name != "blocker",
            "target": target,
            "arrival_tolerance_m": 0.2,
            "speed_mps": 0.9,
            "route_context": (0, 0),
        }
        for name, target in (("holder", [-3.0, 0.9, 0.0]), ("passer", [2.0, 0.0, 0.0]), ("blocker", [2.0, 0.0, 0.0]))
    }
    original = copy.deepcopy((actors, intentions))
    coordinator = PassingCoordinator(grid, {"map_clearance_m": 0.03, "pair_clearance_m": 0.25})
    key = ("holder", "passer")
    refuge = np.asarray(actors[0]["position"][:2])
    coordinator.transactions[key] = _Transaction(
        *key,
        refuge.copy(),
        refuge,
        tuple(coordinator._context(intentions[name]) for name in key),
        0.0,
        4.0,
        [actors[1]["position"][:2], intentions["passer"]["target"][:2]],
        np.array([1.0, 0.0]),
        "passing",
    )
    if safe_handoff:
        result = coordinator.update(1.0, actors, intentions, [], {})
        assert result["status"]["transactions"] == []
        assert result["objectives"] == {}
    else:
        # The moving footprint makes holding the passer's current pose unsafe.
        obstacle = SimpleNamespace(
            name="cart", position=np.array([-1.56, 0.0]), velocity=np.array([0.9, 0.0]), radius=0.3, footprint=None
        )
        with pytest.raises(ValueError, match="original arrival tolerance"):
            coordinator.update(1.0, actors, intentions, [obstacle], {})
    assert (actors, intentions) == original


def _bottleneck(with_bay=True):
    cells = np.full((100, 160), 100, dtype=np.int8)
    cells[20:44, :] = 0
    if with_bay:
        cells[10:60, 96:] = cells[10:60, :32] = 0
    grid = OccupancyGrid(cells, 0.05, (-4.0, -2.5))
    positions = [(0.4, -0.65), (1.1, -1.0)] if with_bay else [(0.0, -0.7), (0.7, -1.0)]
    agents = [
        {"name": name, "position": [*point, 0.0], "yaw": yaw, "radius_m": 0.25, "height_m": 1.8}
        for name, point, yaw in zip(("east", "west"), positions, (0.0, math.pi))
    ]
    plans = [
        {
            "name": actor["name"],
            "position": actor["position"],
            "behavior": "walk",
            "speed": 0.9,
            "arrival_tolerance": 0.2,
            "waypoints": [{"position": [2.5 if i == 0 else -3.0, -0.9, 0.0], "wait": 0.4, "yaw": 90.0}],
        }
        for i, actor in enumerate(agents)
    ]
    settings = {"map_clearance_m": 0.03, "pair_clearance_m": 0.25, "obstacle_time_horizon_s": 0.5}
    return grid, agents, plans, settings


def _native_passing_ready_snapshot():
    with np.load(Path(__file__).parent / "fixtures/native_passing_ready_20261105.npz", allow_pickle=False) as fixture:
        grid = OccupancyGrid(fixture["occupancy"], float(fixture["resolution_m"]), tuple(fixture["origin_xy_m"]))
        snapshot = json.loads(str(fixture["observation_json"]))
    coordinator = PassingCoordinator(grid, snapshot["settings"])
    old = snapshot["transaction"]
    participants = (old["yielder"], old["passer"])
    coordinator.transactions[participants] = _Transaction(
        *participants,
        np.asarray(old["origin_xy_m"]),
        np.asarray(old["refuge_xy_m"]),
        tuple(coordinator._context(snapshot["intentions"][name]) for name in participants),
        old["started_s"],
        old["deadline_s"],
        old["path"],
        np.asarray(old["axis"]),
    )
    return coordinator, snapshot, NavigationEngine._dynamic_obstacles(snapshot["obstacles"])


def test_native_passing_waits_for_actual_clearance_before_releasing_the_passer():
    coordinator, snapshot, obstacles = _native_passing_ready_snapshot()
    original = copy.deepcopy(snapshot["intentions"])
    old = snapshot["transaction"]
    states = snapshot["states"]
    result = coordinator.update(snapshot["time_s"], states, snapshot["intentions"], obstacles, {})
    transaction = result["status"]["transactions"][0]
    assert transaction["phase"] == "retreat" and transaction["deadline_s"] == old["deadline_s"]
    assert result["objectives"][old["yielder"]]["target"] == old["refuge_xy_m"]
    assert result["objectives"][old["passer"]]["phase"] == "passing_wait"
    assert transaction["readiness_error"]
    held = next(state for state in states if state["name"] == old["yielder"])
    held["position"] = [*old["refuge_xy_m"], held["position"][2]]
    held["velocity"] = [0.0, 0.0, 0.0]
    ready = coordinator.update(snapshot["time_s"] + 0.1, states, snapshot["intentions"], obstacles, {})
    assert ready["status"]["transactions"][0]["phase"] == "passing"
    assert ready["status"]["transactions"][0]["readiness_error"] is None
    assert ready["objectives"][old["passer"]]["phase"] == "passing_go"
    assert math.dist(held["position"][:2], old["origin_xy_m"]) <= 0.4
    assert snapshot["intentions"] == original


def test_native_passing_prefers_a_certified_forward_refuge_with_a_small_body_turn():
    with np.load(Path(__file__).parent / "fixtures/native_passing_ready_20261105.npz", allow_pickle=False) as fixture:
        grid = OccupancyGrid(fixture["occupancy"], float(fixture["resolution_m"]), tuple(fixture["origin_xy_m"]))
        snapshot = json.loads(str(fixture["choose_observation_json"]))
    coordinator = PassingCoordinator(grid, snapshot["settings"])
    actors = snapshot["actors"]
    states = list(actors.values())
    intentions = snapshot["intentions"]
    original = copy.deepcopy((states, intentions))
    obstacles = NavigationEngine._dynamic_obstacles(snapshot["obstacles"])
    for now in (snapshot["time_s"] - 2.5, snapshot["time_s"] - 0.75, snapshot["time_s"]):
        result = coordinator.update(now, states, intentions, obstacles, {})
    selected = result["status"]["transactions"][0]
    held = actors[selected["yielder"]]
    movement = np.asarray(selected["refuge_xy_m"]) - selected["origin_xy_m"]
    turn = abs(math.remainder(math.atan2(movement[1], movement[0]) - held["yaw"], 2 * math.pi))
    assert turn < math.pi / 2
    # The actual preceding native command selected this visible route shortcut.
    direction = np.asarray([2.43, 0.15]) - held["position"][:2]
    assert np.dot(movement, direction) >= 0
    assert np.linalg.norm(movement) <= 0.4
    assert (states, intentions) == original


@pytest.mark.parametrize("release_blocker", [False, True])
def test_an_occupied_goal_after_exact_refuge_arrival_keeps_the_original_bounded_deadline(release_blocker):
    coordinator, snapshot, obstacles = _native_passing_ready_snapshot()
    old, intentions, states = snapshot["transaction"], snapshot["intentions"], snapshot["states"]
    held = next(state for state in states if state["name"] == old["yielder"])
    held["position"] = [*old["refuge_xy_m"], held["position"][2]]
    blocker = next(state for state in states if state["name"] not in (old["yielder"], old["passer"]))
    free_position = list(blocker["position"])
    blocker["position"] = list(intentions[old["passer"]]["target"])
    for state in states:
        state["velocity"] = [0.0, 0.0, 0.0]
    original = copy.deepcopy((states, intentions))
    for now in (snapshot["time_s"], snapshot["time_s"] + 0.1):
        result = coordinator.update(now, states, intentions, obstacles, {})
        transaction = result["status"]["transactions"][0]
        assert transaction["phase"] == "retreat" and transaction["deadline_s"] == old["deadline_s"]
        assert transaction["readiness_error"]
        assert all(result["objectives"][name]["phase"] == "passing_wait" for name in (old["yielder"], old["passer"]))
    if release_blocker:
        blocker["position"] = free_position
        expected = copy.deepcopy((states, intentions))
        ready = coordinator.update(snapshot["time_s"] + 0.2, states, intentions, obstacles, {})
        assert ready["status"]["transactions"][0]["phase"] == "passing"
        assert ready["objectives"][old["passer"]]["phase"] == "passing_go"
        assert (states, intentions) == expected
        return
    pending = coordinator.update(old["deadline_s"] - 1e-5, states, intentions, obstacles, {})
    assert pending["status"]["transactions"][0]["deadline_s"] == old["deadline_s"]
    with pytest.raises(RuntimeError, match="bounded passing transaction timed out"):
        coordinator.update(old["deadline_s"] + 1e-5, states, intentions, obstacles, {})
    assert (states, intentions) == original


@pytest.mark.parametrize("blocked", [False, True])
def test_a_late_retreat_uses_the_measured_refuge_only_with_a_complete_safe_exit(blocked):
    grid = OccupancyGrid(np.zeros((120, 160), dtype=np.int8), 0.05, (-4.0, -3.0))
    states = [
        {"name": name, "position": position, "velocity": [0.0, 0.0, 0.0], "radius_m": 0.25, "yaw": 0.0}
        for name, position in (("held", [0.0, 0.8, 0.0]), ("passer", [-1.0, 0.0, 0.0]))
    ]
    intentions = {
        name: {"active": True, "target": target, "arrival_tolerance_m": 0.2, "speed_mps": 0.9, "route_context": (0, 0)}
        for name, target in (("held", [-2.0, 0.8, 0.0]), ("passer", [2.0, 0.0, 0.0]))
    }
    if blocked:
        states.append({**states[1], "name": "blocker", "position": list(intentions["passer"]["target"])})
        intentions["blocker"] = {"active": False}
    coordinator = PassingCoordinator(grid, {"map_clearance_m": 0.03, "pair_clearance_m": 0.25})
    original = copy.deepcopy((states, intentions))
    deadline = 4.0
    key = ("held", "passer")
    coordinator.transactions[key] = _Transaction(
        *key,
        np.array([0.0, 0.6]),
        np.array([0.0, 0.85]),
        tuple(coordinator._context(intentions[name]) for name in key),
        0.0,
        deadline,
        [states[1]["position"][:2], intentions["passer"]["target"][:2]],
        np.array([1.0, 0.0]),
    )

    ordinary = coordinator.update(3.5, states, intentions, [], {})
    assert ordinary["objectives"]["passer"]["phase"] == "passing_wait"
    assert ordinary["objectives"]["held"]["target"] == [0.0, 0.85]

    late = coordinator.update(3.8, states, intentions, [], {})
    [transaction] = late["status"]["transactions"]
    if blocked:
        assert transaction["phase"] == "retreat" and transaction["deadline_s"] == deadline
        assert late["objectives"]["passer"]["phase"] == "passing_wait"
        with pytest.raises(RuntimeError, match="bounded passing transaction timed out"):
            coordinator.update(deadline + 0.01, states, intentions, [], {})
    else:
        assert transaction["phase"] == "passing"
        assert transaction["refuge_xy_m"] == states[0]["position"][:2]
        assert late["objectives"]["held"]["target"] is None
        assert late["objectives"]["passer"]["phase"] == "passing_go"
        assert late["objectives"]["passer"]["target"] == intentions["passer"]["target"][:2]
    assert (states, intentions) == original


@pytest.mark.parametrize("reflected", [False, True])
def test_bounded_refuge_unblocks_opposing_routes_in_a_single_file_throat(reflected):
    grid, agents, plans, settings = _bottleneck()
    if reflected:
        grid = OccupancyGrid(grid.occupancy[:, ::-1], grid.resolution_m, grid.origin_xy_m)
        for actor, plan in zip(agents, plans):
            actor["position"] = [-actor["position"][0], *actor["position"][1:]]
            actor["yaw"] = math.pi - actor["yaw"]
            plan["position"] = actor["position"]
            plan["waypoints"][0]["position"][0] *= -1
    original = copy.deepcopy(plans)
    engine = NavigationEngine(plans, grid, settings, agents)
    states = agents
    seen = set()
    for tick in range(1200):
        result = engine.step(tick / 60, 1 / 60, states)
        next_states = result["agents"]
        for i, state in enumerate(next_states):
            assert state["error"] is None
            assert grid.segment_clear(states[i]["position"], state["position"], 0.28)
            assert math.dist(states[i]["position"], state["position"]) <= 0.9 / 60 + 1e-7
            assert abs(engine._angle(state["yaw"] - states[i]["yaw"])) <= 2.5 / 60 + 1e-6
            seen.add(state["phase"])
        relative = np.asarray(states[0]["position"][:2]) - states[1]["position"][:2]
        motion = (
            np.asarray(next_states[0]["position"][:2])
            - states[0]["position"][:2]
            - np.asarray(next_states[1]["position"][:2])
            + states[1]["position"][:2]
        )
        assert engine._chord_distance(relative, motion) >= 0.75 - 1e-6
        for transaction in result.get("passing", {}).get("transactions", []):
            held = next(a for a in next_states if a["name"] == transaction["yielder"])
            assert math.dist(held["position"][:2], transaction["origin_xy_m"]) <= 0.4 + 1e-7
            assert held["waypoint_index"] == 0 and held["completed_cycles"] == 0
        states = next_states
        if all(state["phase"] == "complete" for state in states):
            break
    assert all(state["phase"] == "complete" for state in states)
    assert {"passing_yield", "passing_wait", "passing_go"} <= seen
    assert plans == original


@pytest.mark.parametrize("resident_start_s", [0.5, 2.0])
def test_a_nearby_goal_queue_preserves_through_traffic_and_existing_bounded_passing(resident_start_s):
    grid, agents, plans, settings = _bottleneck()
    resident = {
        "name": "resident",
        "position": [-3.0, -1.6, 0.0],
        "yaw": math.pi / 2,
        "radius_m": 0.25,
        "height_m": 1.8,
    }
    agents.append(resident)
    plans.append(
        {
            "name": resident["name"],
            "position": resident["position"],
            "behavior": "walk",
            "speed": 0.9,
            "arrival_tolerance": 0.2,
            "start_time": resident_start_s,
            "waypoints": [
                {"position": [-3.0, -0.9, 0.0], "wait": 8.0, "yaw": None},
                {"position": [-3.7, -1.6, 0.0], "wait": 0.0, "yaw": None},
            ],
        }
    )
    original = copy.deepcopy(plans)
    engine = NavigationEngine(plans, grid, settings, agents)
    states = agents
    resident_dwell_s = 0.0
    seen_passing = False
    try:
        for tick in range(1800):
            result = engine.step(tick / 60, 1 / 60, states)
            next_states = result["agents"]
            by_name = {state["name"]: state for state in next_states}
            for before, after in zip(states, next_states):
                assert after["error"] is None
                assert grid.segment_clear(before["position"], after["position"], 0.28)
                assert math.dist(before["position"], after["position"]) <= 0.9 / 60 + 1e-7
                assert abs(engine._angle(after["yaw"] - before["yaw"])) <= 2.5 / 60 + 1e-6
            for i in range(len(states)):
                for j in range(i):
                    relative = np.asarray(states[i]["position"][:2]) - states[j]["position"][:2]
                    motion = (
                        np.asarray(next_states[i]["position"][:2])
                        - states[i]["position"][:2]
                        - np.asarray(next_states[j]["position"][:2])
                        + states[j]["position"][:2]
                    )
                    assert engine._chord_distance(relative, motion) >= 0.75 - 1e-6
            for transaction in result["passing"]["transactions"]:
                seen_passing = True
                held = by_name[transaction["yielder"]]
                assert math.dist(held["position"][:2], transaction["origin_xy_m"]) <= 0.4 + 1e-7
            for queued in result["goal_occupancy"]["queue"]:
                if by_name[queued["name"]]["phase"] == "goal_yield":
                    assert math.dist(by_name[queued["name"]]["position"][:2], queued["origin_xy_m"]) <= 0.4 + 1e-7
            held = by_name["resident"]
            if held["phase"] == "dwelling" and held["waypoint_index"] == 0:
                assert math.dist(held["position"], plans[-1]["waypoints"][0]["position"]) <= 0.2
                assert math.dist(held["position"], states[-1]["position"]) <= 1e-10
                resident_dwell_s += 1 / 60
            states = next_states
            if all(state["phase"] == "complete" for state in states):
                break
    finally:
        engine.close()
    assert seen_passing
    assert all(state["phase"] == "complete" for state in states)
    assert resident_dwell_s >= 8.0 - 1e-7
    assert plans == original


def _owned_passage(case="crossing"):
    cells = np.zeros((120, 160), dtype=np.int8)
    if case == "uncached_wall":
        cells[78, 80] = 100
    grid = OccupancyGrid(cells, 0.05, (-4.0, -3.0))
    states = [
        {"name": name, "position": [*point, 0.0], "velocity": [0.0, 0.0, 0.0], "yaw": 0.0, "radius_m": 0.25}
        for name, point in (("held", (0.0, -1.0)), ("passer", (-2.0, 0.0)), ("third", (0.0, 1.4)))
    ]
    if case == "inside":
        states[-1]["position"][1] = 0.7
    if case == "distant":
        states[-1]["position"][0] = 2.0
    intentions = {
        name: {"active": True, "target": target, "arrival_tolerance_m": 0.2, "speed_mps": 0.9, "route_context": (0, 0)}
        for name, target in (("held", [-3.0, -1.0, 0.0]), ("passer", [2.0, 0.0, 0.0]), ("third", [0.0, -2.0, 0.0]))
    }
    if case == "parallel":
        intentions["third"]["target"] = [2.0, 1.4, 0.0]
    if case == "distant":
        intentions["third"]["target"][0] = 2.0
    for actor in states:
        intentions[actor["name"]]["path"] = [actor["position"][:2], intentions[actor["name"]]["target"][:2]]
    if case in ("uncached", "uncached_wall"):
        intentions["third"]["path"] = []
    coordinator = PassingCoordinator(grid, {"map_clearance_m": 0.03, "pair_clearance_m": 0.25})
    coordinator.transactions[("held", "passer")] = _Transaction(
        "held",
        "passer",
        np.array([0.0, -1.0]),
        np.array([0.2, -1.0]),
        tuple(coordinator._context(intentions[name]) for name in ("held", "passer")),
        0.0,
        20.0,
        [[-2.0, 0.0], [2.0, 0.0]],
        np.array([1.0, 0.0]),
    )
    obstacles = []
    if case == "moving_wall":
        obstacles = [
            SimpleNamespace(
                position=np.array([0.0, 1.094]),
                velocity=np.array([0.0, 0.12]),
                radius=1.01,
                footprint=np.array([[-1.0, 1.069], [1.0, 1.069], [1.0, 1.119], [-1.0, 1.119]]),
                name="/moving_wall",
            )
        ]
    priority = {"third": {"phase": "goal_wait", "target": None}} if case == "priority" else {}
    return coordinator, states, intentions, obstacles, priority


@pytest.mark.parametrize(
    "case", ["crossing", "parallel", "uncached", "uncached_wall", "priority", "inside", "moving_wall", "distant"]
)
def test_owned_passage_queues_only_a_safe_conflicting_route_with_no_higher_priority(case):
    coordinator, states, intentions, obstacles, priority = _owned_passage(case)
    before = copy.deepcopy((states, intentions))
    result = coordinator.update(0.0, states, intentions, obstacles, priority)
    waiting = case in ("crossing", "uncached")
    assert ("third" in result["objectives"]) is waiting
    assert coordinator.waiting_actors == (("third",) if waiting else ())
    assert [record["name"] for record in result["status"].get("queued", [])] == (["third"] if waiting else [])
    assert result["status"]["transactions"][0]["deadline_s"] == 20.0
    if waiting:
        assert result["objectives"]["third"]["target"] is None
        assert result["objectives"]["third"]["phase"] == "passing_wait"
    assert (states, intentions) == before
    coordinator.reset()
    assert coordinator.waiting_actors == ()


def test_owned_passage_leaves_room_to_approach_before_queueing():
    coordinator, states, intentions, obstacles, _ = _owned_passage()
    third = states[2]
    third["position"][1] = 2.0
    assert "third" not in coordinator.update(0.0, states, intentions, obstacles, {})["objectives"]

    third["position"][1] = 1.4
    result = coordinator.update(0.8, states, intentions, obstacles, {})
    assert result["objectives"]["third"]["phase"] == "passing_wait"
    assert result["status"]["queued"][0]["origin_xy_m"] == third["position"][:2]


def test_owned_passage_queue_keeps_its_parent_deadline_and_releases_after_actual_passing_or_priority():
    coordinator, states, intentions, obstacles, _ = _owned_passage()
    first = coordinator.update(0.0, states, intentions, obstacles, {})
    original_point = first["status"]["queued"][0]["origin_xy_m"]
    states[0]["position"] = [0.2, -1.0, 0.0]
    ready = coordinator.update(1.0, states, intentions, obstacles, {})
    deadline = ready["status"]["transactions"][0]["deadline_s"]
    repeated = coordinator.update(1.1, states, intentions, obstacles, {})
    assert repeated["status"]["transactions"][0]["deadline_s"] == deadline
    assert repeated["status"]["queued"][0]["origin_xy_m"] == original_point
    priority = {"third": {"phase": "goal_wait", "target": None}}
    claimed = coordinator.update(1.2, states, intentions, obstacles, priority)
    assert "third" not in claimed["objectives"] and coordinator.waiting_actors == ()
    assert coordinator.update(1.3, states, intentions, obstacles, {})["status"]["queued"]
    states[1]["position"] = [1.0, 0.0, 0.0]
    released = coordinator.update(2.0, states, intentions, obstacles, {})
    assert released["status"]["transactions"] == [] and released["status"].get("queued", []) == []
    assert "third" not in released["objectives"] and coordinator.waiting_actors == ()


def test_released_outsider_can_resume_before_starting_another_early_passage():
    coordinator, states, intentions, obstacles, _ = _owned_passage()
    assert coordinator.update(0.0, states, intentions, obstacles, {})["status"]["queued"]
    states[0]["position"] = [0.2, -1.0, 0.0]
    assert coordinator.update(1.0, states, intentions, obstacles, {})["status"]["queued"]

    passer, outsider = states[1:]
    passer.update(position=[1.0, 0.0, 0.0], velocity=[0.05, 0.0, 0.0])
    outsider.update(position=[2.4, 0.0, 0.0], velocity=[-0.05, 0.0, 0.0], yaw=math.pi)
    intentions["third"].update(target=[-2.0, 0.0, 0.0], path=[[2.4, 0.0], [-2.0, 0.0]])
    released = coordinator.update(2.0, states, intentions, obstacles, {})
    assert released == {"objectives": {}, "status": {"transactions": []}}

    _observe_motion(states[1:], 1.5)
    recovered = coordinator.update(3.5, states, intentions, obstacles, {})
    assert len(recovered["status"]["transactions"]) == 1
    assert {recovered["status"]["transactions"][0][role] for role in ("yielder", "passer")} == {"third", "passer"}


def test_a_native_third_arrival_waits_outside_the_owned_passage_and_all_original_goals_resume():
    directory = Path(__file__).parent / "fixtures"
    snapshot = json.loads((directory / "native_passing_third_arrival_20261105.json").read_text())
    with np.load(directory / snapshot["provenance"]["map_fixture"], allow_pickle=False) as fixture:
        grid = OccupancyGrid(fixture["occupancy"], float(fixture["resolution_m"]), tuple(fixture["origin_xy_m"]))
    original = copy.deepcopy(snapshot["plans"])
    engine = NavigationEngine(snapshot["plans"], grid, snapshot["settings"], snapshot["spawn_agents"])
    for name, fields in snapshot["routes"].items():
        for key, value in fields.items():
            if key in ("door_token", "point_target") and value is not None:
                value = tuple(value)
            setattr(engine.routes[name], key, copy.deepcopy(value))
    for saved in snapshot["transactions"]:
        saved = copy.deepcopy(saved)
        for key in ("origin", "refuge", "axis"):
            saved[key] = np.asarray(saved[key])
        saved["contexts"] = tuple((tuple(context), tuple(target)) for context, target in saved["contexts"])
        transaction = _Transaction(**saved)
        engine._passing.transactions[(transaction.yielder, transaction.passer)] = transaction
    goals = {name: engine._current_target(route)["position"] for name, route in engine.routes.items()}
    states = snapshot["states"]
    reached, saw_queue, saw_release = set(), False, False
    parents = {(saved["yielder"], saved["passer"]) for saved in snapshot["transactions"]}
    passing_deadlines, pending_deadlines = {}, {}
    try:
        for tick in range(1500):
            time_s = snapshot["time_s"] + tick / 60
            result = engine.step(time_s, 1 / 60, states, snapshot["obstacles"], snapshot["door_states"])
            next_states = result["agents"]
            assert all(state["error"] is None for state in next_states)
            queued = {record["name"] for record in result["passing"].get("queued", [])}
            if "walker_01" in queued:
                saw_queue = True
                wait = next(record for record in result["passing"]["queued"] if record["name"] == "walker_01")
                assert next_states[0]["position"] == states[0]["position"]
                assert next_states[0]["position"][:2] == wait["origin_xy_m"]
                assert next_states[0]["phase"] == "passing_wait"
            if saw_queue and not any(
                (transaction["yielder"], transaction["passer"]) in parents
                for transaction in result["passing"]["transactions"]
            ):
                saw_release = True
                assert not any(
                    record["name"] == "walker_01" and (record["yielder"], record["passer"]) in parents
                    for record in result["passing"].get("queued", [])
                )
                if "walker_01" in engine._passing.waiting_actors:
                    pending = [
                        record
                        for record in result["passing"].get("pending", [])
                        if "walker_01" in record["actors"]
                        and all(set(record["actors"]) != set(parent) for parent in parents)
                    ]
                    assert len(pending) == 1
                    identity = tuple(pending[0]["actors"])
                    pending_deadlines.setdefault(identity, pending[0]["deadline_s"])
                    assert time_s <= pending[0]["deadline_s"] == pending_deadlines[identity]
                    assert next_states[0]["position"] == states[0]["position"]
            for transaction in result["passing"]["transactions"]:
                held = next(state for state in next_states if state["name"] == transaction["yielder"])
                assert math.dist(held["position"][:2], transaction["origin_xy_m"]) <= 0.4 + 1e-7
                if transaction["phase"] == "passing":
                    identity = (transaction["yielder"], transaction["passer"], tuple(transaction["origin_xy_m"]))
                    passing_deadlines.setdefault(identity, transaction["deadline_s"])
                    assert transaction["deadline_s"] == passing_deadlines[identity]
            for before, after in zip(states, next_states):
                assert grid.segment_clear(before["position"], after["position"], before["radius_m"] + 0.03)
                assert math.dist(before["position"], after["position"]) <= 0.9 / 60 + 1e-7
                assert abs(math.remainder(after["yaw"] - before["yaw"], 2 * math.pi)) <= 2.5 / 60 + 1e-6
                for obstacle in snapshot["obstacles"]:
                    assert segment_polygon_clear(
                        before["position"], after["position"], obstacle["footprint_xy_m"], 0.28
                    )
                if math.dist(after["position"], goals[after["name"]]) <= 0.2:
                    reached.add(after["name"])
            for i, first in enumerate(next_states):
                for j, second in enumerate(next_states[:i]):
                    relative = np.asarray(states[i]["position"][:2]) - states[j]["position"][:2]
                    motion = np.asarray(first["position"][:2]) - states[i]["position"][:2]
                    motion -= np.asarray(second["position"][:2]) - states[j]["position"][:2]
                    squared = float(np.dot(motion, motion))
                    closest = np.clip(-np.dot(relative, motion) / squared, 0, 1) if squared else 0.0
                    assert np.linalg.norm(relative + closest * motion) >= 0.75 - 1e-7
            states = [{**before, **after} for before, after in zip(states, next_states)]
            if len(reached) == 3 and saw_release:
                break
        assert saw_queue and saw_release and reached == set(goals)
        assert time_s > snapshot["provenance"]["failure_time_s"]
        assert snapshot["plans"] == original
    finally:
        engine.close()


def test_a_passing_transaction_cannot_renew_its_budget_when_actual_motion_never_occurs():
    grid, agents, plans, settings = _bottleneck()
    engine = NavigationEngine(plans, grid, settings, agents)
    states = agents
    for tick in range(300):
        result = engine.step(tick / 60, 1 / 60, states)
        states = result["agents"]
        if result["passing"]["transactions"]:
            transaction = result["passing"]["transactions"][0]
            break
    else:
        pytest.fail("The actual opposing blockage did not start a passing transaction")
    frozen = [{**state, "velocity": [0.0, 0.0, 0.0]} for state in states]
    with pytest.raises(RuntimeError, match="bounded passing transaction timed out"):
        for tick in range(tick + 1, tick + 600):
            result = engine.step(tick / 60, 1 / 60, frozen)
            assert result["passing"]["transactions"][0]["deadline_s"] == transaction["deadline_s"]


def test_door_priority_preserves_an_existing_passing_origin_and_deadline(monkeypatch):
    grid, agents, plans, settings = _bottleneck()
    engine = NavigationEngine(plans, grid, settings, agents)
    states = agents
    for tick in range(300):
        result = engine.step(tick / 60, 1 / 60, states)
        states = result["agents"]
        if result["passing"]["transactions"]:
            before = result["passing"]["transactions"][0]
            break
    else:
        pytest.fail("No passing transaction was available for the door-priority boundary")
    held = before["yielder"]
    monkeypatch.setattr(
        engine._doors,
        "update",
        lambda *args: {
            "objectives": {held: {"target": None, "phase": "door_wait", "token": ("door",)}},
            "obstacles": [],
            "door_commands": [],
        },
    )
    result = engine.step((tick + 1) / 60, 1 / 60, states)
    after = result["passing"]["transactions"][0]
    assert after["phase"] == "door_priority"
    assert after["origin_xy_m"] == before["origin_xy_m"] and after["deadline_s"] == before["deadline_s"]
    assert next(a for a in result["agents"] if a["name"] == held)["phase"] == "door_wait"
    engine._passing.reset()
    assert not engine._passing.transactions


def test_a_sealed_single_file_corridor_fails_explicitly_without_unbounded_retreat():
    grid, agents, plans, settings = _bottleneck(with_bay=False)
    engine = NavigationEngine(plans, grid, settings, agents)
    states = agents
    with pytest.raises(RuntimeError, match="bounded passing"):
        for tick in range(600):
            states = engine.step(tick / 60, 1 / 60, states)["agents"]
            assert all(state["error"] is None for state in states)


@pytest.mark.parametrize("diverging", [False, True])
def test_close_slow_pedestrians_without_a_predicted_conflict_do_not_start_passing(diverging):
    grid = OccupancyGrid(np.zeros((80, 120), dtype=np.int8), 0.1, (-6.0, -4.0))
    actors = [
        {"name": name, "position": [x, y, 0.0], "yaw": yaw, "radius_m": 0.25, "height_m": 1.8}
        for name, x, y, yaw in (
            (("left", -0.4, 0.0, math.pi), ("right", 0.4, 0.0, 0.0))
            if diverging
            else (("left", -0.05, -0.4, 0.0), ("right", 0.05, 0.4, math.pi))
        )
    ]
    intentions = {
        actor["name"]: {
            "target": [
                math.copysign(3.0, actor["position"][0]) * (1 if diverging else -1),
                actor["position"][1],
                0.0,
            ],
            "active": True,
            "speed_mps": 0.9,
            "arrival_tolerance_m": 0.2,
            "route_context": (0, 0),
            "path": [],
        }
        for actor in actors
    }
    coordinator = PassingCoordinator(grid, {"map_clearance_m": 0.03, "pair_clearance_m": 0.25})
    for time_s in (0.0, 1.5, 2.25, 3.0):
        result = coordinator.update(time_s, actors, intentions, [], {})
        assert result == {"objectives": {}, "status": {"transactions": []}}


@pytest.mark.parametrize("advancing", [None, "upper", "lower"])
def test_close_crossing_routes_require_both_bodies_to_remain_slow_before_bounded_yield(advancing):
    grid = OccupancyGrid(np.zeros((80, 120), dtype=np.int8), 0.1, (-6.0, -4.0))
    states = [
        {
            "name": name,
            "position": [0.0, sign * 0.38, 0.0],
            "velocity": [0.0, 0.0, 0.0],
            "yaw": -sign * math.pi / 4,
            "radius_m": 0.25,
        }
        for name, sign in (("upper", 1), ("lower", -1))
    ]
    intentions = {
        state["name"]: {
            "target": [3.0, -math.copysign(2.62, state["position"][1]), 0.0],
            "active": True,
            "speed_mps": 0.9,
            "arrival_tolerance_m": 0.2,
            "route_context": (0, 0),
            "path": [],
        }
        for state in states
    }
    original = copy.deepcopy(intentions)
    coordinator = PassingCoordinator(grid, {"map_clearance_m": 0.03, "pair_clearance_m": 0.25})
    empty = {"objectives": {}, "status": {"transactions": []}}
    for now in (0.0, 1.5, 2.0):
        assert coordinator.update(now, states, intentions, [], {}) == empty
    now = 2.3
    if advancing is not None:
        moving = next(state for state in states if state["name"] == advancing)
        moving.update(position=[0.02, *moving["position"][1:]], velocity=[0.05, 0.0, 0.0], yaw=0.0)
        assert coordinator.update(now, states, intentions, [], {}) == empty
        moving["velocity"] = [0.0, 0.0, 0.0]
        for now in (2.4, 3.0):
            assert coordinator.update(now, states, intentions, [], {}) == empty
        now = 3.2
    result = coordinator.update(now, states, intentions, [], {})
    assert len(result["status"]["transactions"]) == 1
    transaction = result["status"]["transactions"][0]
    assert {transaction["passer"], transaction["yielder"]} == {"upper", "lower"}
    assert result["objectives"][transaction["yielder"]]["phase"] == "passing_yield"
    assert 0 < math.dist(transaction["origin_xy_m"], transaction["refuge_xy_m"]) <= 0.4
    assert intentions == original


def _external_blocker_case():
    fixture_dir = Path(__file__).with_name("fixtures")
    fixture = json.loads((fixture_dir / "native_passing_external_blocker_20261106.json").read_text())
    with np.load(fixture_dir / fixture["provenance"]["map_fixture"], allow_pickle=False) as data:
        grid = OccupancyGrid(data["occupancy"], float(data["resolution_m"]), tuple(data["origin_xy_m"]))
    coordinator = PassingCoordinator(grid, fixture["settings"])
    obstacles = NavigationEngine._dynamic_obstacles(fixture["obstacles"])
    return coordinator, fixture["actors"], fixture["intentions"], obstacles


def _begin_external_certificate_wait(coordinator, states, intentions, obstacles):
    for time_s in (0.0, 1.5, 2.25):
        result = coordinator.update(time_s, states, intentions, obstacles, {})
    pending = result["status"]["pending"][0]
    assert result["status"]["transactions"] == []
    assert coordinator.waiting_actors == tuple(pending["actors"])
    assert all(
        result["objectives"][name]["phase"] == "passing_wait" and result["objectives"][name]["target"] is None
        for name in pending["actors"]
    )
    assert not set(pending["outside_actors"]) & set(result["objectives"])
    return result, pending


def test_a_moving_third_body_can_clear_the_original_goal_route_before_bounded_passing_begins():
    coordinator, states, intentions, obstacles = _external_blocker_case()
    original = copy.deepcopy((states, intentions))
    _, pending = _begin_external_certificate_wait(coordinator, states, intentions, obstacles)
    for state in states:
        if state["name"] in pending["outside_actors"]:
            state["position"] = (np.asarray(state["position"]) + np.asarray(state["velocity"]) * 0.5).tolist()
    result = coordinator.update(2.75, states, intentions, obstacles, {})
    assert not result["status"].get("pending")
    transaction = result["status"]["transactions"][0]
    actual = next(state for state in states if state["name"] == transaction["yielder"])
    assert transaction["origin_xy_m"] == actual["position"][:2]
    assert math.dist(transaction["refuge_xy_m"], actual["position"][:2]) <= 0.4
    assert coordinator.waiting_actors == ()
    assert intentions == original[1]
    assert all(
        state == original[0][index]
        for index, state in enumerate(states)
        if state["name"] not in pending["outside_actors"]
    )


def test_an_external_certificate_wait_keeps_its_first_deadline_and_fails_if_the_blocker_does_not_clear():
    coordinator, states, intentions, obstacles = _external_blocker_case()
    _, pending = _begin_external_certificate_wait(coordinator, states, intentions, obstacles)
    unchanged = copy.deepcopy(states)
    expected = 1.5 + 4.0 + 0.4 / min(intentions[name]["speed_mps"] for name in pending["actors"])
    assert pending["deadline_s"] == pytest.approx(expected)
    for time_s in (2.75, 3.25, 4.25, expected):
        result = coordinator.update(time_s, states, intentions, obstacles, {})
        assert result["status"]["pending"][0]["deadline_s"] == pending["deadline_s"]
        assert not result["status"]["transactions"]
    with pytest.raises(RuntimeError, match="bounded passing certificate wait timed out"):
        coordinator.update(expected + 1 / 60, states, intentions, obstacles, {})
    assert states == unchanged


@pytest.mark.parametrize("change", ["context", "priority", "reset", "unsafe"])
def test_an_external_certificate_wait_releases_ownership_when_its_standing_certificate_is_invalidated(change):
    coordinator, states, intentions, obstacles = _external_blocker_case()
    _, pending = _begin_external_certificate_wait(coordinator, states, intentions, obstacles)
    controlled = pending["actors"][0]
    priority = {}
    if change == "context":
        intentions[controlled]["route_context"] = (99, 0)
    elif change == "priority":
        priority[controlled] = {"target": None, "phase": "door_wait"}
    elif change == "reset":
        coordinator.reset()
    else:
        peer = next(state for state in states if state["name"] in pending["outside_actors"])
        position = next(state["position"] for state in states if state["name"] == controlled)
        peer["position"] = [position[0] + 0.749, position[1], position[2]]
    result = coordinator.update(2.25 + 1 / 60, states, intentions, obstacles, priority)
    assert not result["status"].get("pending") and not result["status"]["transactions"]
    assert not coordinator.waiting_actors and controlled not in result["objectives"]


@pytest.mark.parametrize("external_velocity", [[0.9, 0.0, 0.0], None])
def test_a_remote_third_body_does_not_hide_a_pair_that_has_no_static_bounded_refuge(external_velocity):
    grid, actors, plans, settings = _bottleneck(with_bay=False)
    actors.append(
        {
            "name": "remote",
            "position": [-2.0, -0.9, 0.0],
            "velocity": external_velocity,
            "yaw": 0.0,
            "radius_m": 0.25,
            "height_m": 1.8,
        }
    )
    if external_velocity is None:
        actors[-1].pop("velocity")
    intentions = {
        actor["name"]: {
            "active": True,
            "target": plan["waypoints"][0]["position"],
            "arrival_tolerance_m": plan["arrival_tolerance"],
            "speed_mps": plan["speed"],
            "route_context": (0, 0),
            "path": [],
        }
        for actor, plan in zip(
            actors, [*plans, {"waypoints": [{"position": [-3.0, -0.9, 0.0]}], "arrival_tolerance": 0.2, "speed": 0.9}]
        )
    }
    coordinator = PassingCoordinator(grid, settings)
    coordinator.update(0.0, actors, intentions, [], {})
    coordinator.update(1.5, actors, intentions, [], {})
    with pytest.raises(RuntimeError, match="No certified bounded passing refuge"):
        coordinator.update(2.25, actors, intentions, [], {})


def test_external_blockage_retry_resumes_all_three_unchanged_pending_goals_in_a_public_closed_loop():
    fixture_dir = Path(__file__).with_name("fixtures")
    snapshot = json.loads((fixture_dir / "native_passing_external_blocker_20261106.json").read_text())
    with np.load(fixture_dir / snapshot["provenance"]["map_fixture"], allow_pickle=False) as data:
        grid = OccupancyGrid(data["occupancy"], float(data["resolution_m"]), tuple(data["origin_xy_m"]))
    plans = snapshot["plans"]
    original = copy.deepcopy(plans)
    states = copy.deepcopy(snapshot["actors"])
    goals = {plan["name"]: plan["waypoints"][0]["position"] for plan in plans}
    dt = 1 / snapshot["settings"]["control_hz"]
    reached, phases = set(), set()
    pending_deadline = None
    engine = NavigationEngine(plans, grid, snapshot["settings"], states)
    try:
        # Establish the original conflict persistence from measured poses,
        # without injecting route, planner, or coordinator internals.
        for time_s in (0.0, 1.5, 2.25):
            result = engine.step(time_s, dt, states, snapshot["obstacles"], snapshot["door_states"])
        for tick in range(1801):
            if tick:
                result = engine.step(2.25 + tick * dt, dt, states, snapshot["obstacles"], snapshot["door_states"])
            before = {state["name"]: state for state in states}
            after = {state["name"]: state for state in result["agents"]}
            for name, state in after.items():
                assert state["error"] is None
                clearance = before[name]["radius_m"] + snapshot["settings"]["map_clearance_m"]
                start, end = np.asarray(before[name]["position"][:2]), np.asarray(state["position"][:2])
                assert grid.segment_clear(start, end, clearance)
                assert (
                    np.linalg.norm(end - start)
                    <= next(plan["speed"] for plan in plans if plan["name"] == name) * dt + 1e-7
                )
                assert (
                    abs(math.remainder(state["yaw"] - before[name]["yaw"], 2 * math.pi))
                    <= snapshot["settings"]["max_turn_rate_radps"] * dt + 1e-7
                )
                for obstacle in snapshot["obstacles"]:
                    assert segment_polygon_clear(
                        start,
                        end - np.asarray(obstacle["velocity"][:2]) * dt,
                        np.asarray(obstacle["footprint_xy_m"]),
                        clearance,
                    )
                for peer in before:
                    if peer <= name:
                        continue
                    relative = start - before[peer]["position"][:2]
                    delta = end - start - (np.asarray(after[peer]["position"][:2]) - before[peer]["position"][:2])
                    squared = np.dot(delta, delta)
                    fraction = np.clip(-np.dot(relative, delta) / squared, 0, 1) if squared else 0
                    required = (
                        before[name]["radius_m"] + before[peer]["radius_m"] + snapshot["settings"]["pair_clearance_m"]
                    )
                    assert np.linalg.norm(relative + fraction * delta) >= required - 1e-7
                tolerance = next(plan["arrival_tolerance"] for plan in plans if plan["name"] == name)
                if math.dist(state["position"], goals[name]) <= tolerance:
                    reached.add(name)
            for pending in result["passing"].get("pending", []):
                phases.add("pending")
                if pending_deadline is None:
                    pending_deadline = pending["deadline_s"]
                assert pending["deadline_s"] == pending_deadline
                assert all(after[name]["position"] == before[name]["position"] for name in pending["actors"])
            for transaction in result["passing"]["transactions"]:
                phases.add(transaction["phase"])
                assert (
                    math.dist(after[transaction["yielder"]]["position"][:2], transaction["origin_xy_m"]) <= 0.4 + 1e-7
                )
            states = [{**before[state["name"]], **state} for state in result["agents"]]
            if reached == set(goals) and not result["passing"]["transactions"]:
                break
        else:
            pytest.fail("A temporary external blockage did not release the original passing routes")
    finally:
        engine.close()
    assert phases >= {"pending", "retreat", "passing"}
    assert plans == original


@pytest.mark.parametrize("observation", ["unknown", "stationary", "priority"])
def test_an_unproven_external_release_does_not_defer_the_original_refuge_failure(observation):
    coordinator, states, intentions, obstacles = _external_blocker_case()
    original = copy.deepcopy(states)
    outsider = next(state for state in states if state["name"] == "walker_03")
    priority = {}
    if observation == "unknown":
        outsider.pop("velocity")
    elif observation == "stationary":
        outsider["velocity"] = [0.0, 0.0, 0.0]
    else:
        priority[outsider["name"]] = {"target": None, "phase": "door_wait"}
    coordinator.update(0.0, states, intentions, obstacles, priority)
    coordinator.update(1.5, states, intentions, obstacles, priority)
    with pytest.raises(RuntimeError, match="No certified bounded passing refuge"):
        coordinator.update(2.25, states, intentions, obstacles, priority)
    assert not coordinator.waiting_actors
    assert all(state["position"] == original[index]["position"] for index, state in enumerate(states))


def test_passing_looks_through_a_nearby_planning_corner_with_a_complete_observed_segment_certificate():
    grid = OccupancyGrid(np.zeros((120, 160), dtype=np.int8), 0.05, (-4.0, -3.0))
    settings = {"map_clearance_m": 0.03, "pair_clearance_m": 0.25, "control_hz": 60}
    states = [
        {"name": name, "position": [*position, 0.0], "velocity": [0.0, 0.0, 0.0], "yaw": 0.0, "radius_m": 0.25}
        for name, position in (("held", (0.2, 1.0)), ("passer", (0.0, -0.002)))
    ]
    intentions = {
        name: {"active": True, "target": target, "arrival_tolerance_m": 0.2, "speed_mps": 0.9, "route_context": (0, 0)}
        for name, target in (("held", [-1.0, 1.0, 0.0]), ("passer", [1.0, 0.0, 0.0]))
    }
    polygon = np.array([[0.2, -1.0], [1.5, -1.0], [1.5, -0.280001], [0.2, -0.280001]])
    obstacles = [
        SimpleNamespace(position=np.array([0.85, -0.64]), velocity=np.zeros(2), radius=0.75, footprint=polygon)
    ]
    start = np.asarray(states[1]["position"][:2])
    clearance = states[1]["radius_m"] + settings["map_clearance_m"]
    assert not segment_polygon_clear(start, [1.0, 0.0], polygon, clearance)
    assert segment_polygon_clear(start, [0.15, 0.0], polygon, clearance)
    coordinator = PassingCoordinator(grid, settings)
    coordinator.transactions[("held", "passer")] = _Transaction(
        "held",
        "passer",
        np.array([0.2, 1.0]),
        np.array([0.2, 1.0]),
        tuple(coordinator._context(intentions[name]) for name in ("held", "passer")),
        0.0,
        10.0,
        [start.tolist(), [0.0, 0.0], [1.0, 0.0]],
        np.array([1.0, 0.0]),
        phase="passing",
    )
    original = copy.deepcopy(intentions)
    result = coordinator.update(1.0, states, intentions, obstacles, {})
    target = np.asarray(result["objectives"]["passer"]["target"])
    assert result["objectives"]["passer"]["phase"] == "passing_go"
    assert target[0] > 0 and target[1] == 0
    assert math.dist(start, target) > intentions["passer"]["speed_mps"] / settings["control_hz"]
    duration = math.dist(start, target) / intentions["passer"]["speed_mps"]
    assert segment_polygon_clear(start, target - obstacles[0].velocity * duration, polygon, clearance)
    assert (
        math.dist(target, states[0]["position"][:2])
        >= sum(state["radius_m"] for state in states) + settings["pair_clearance_m"]
    )
    assert result["status"]["transactions"][0]["deadline_s"] == 10.0
    assert intentions == original


@pytest.mark.parametrize("controlled", [False, True])
def test_passing_predicts_the_executed_cached_route_around_live_geometry_owned_by_the_correct_layer(controlled):
    grid = OccupancyGrid(np.zeros((80, 120), dtype=np.int8), 0.1, (-6.0, -4.0))
    states = [
        {"name": name, "position": [*position, 0.0], "velocity": [0.0, 0.0, 0.0], "yaw": 0.0, "radius_m": 0.25}
        for name, position in (("upper", (0.0, 0.0)), ("lower", (-0.56, -0.56)))
    ]
    intentions = {
        "upper": {
            "active": True,
            "target": [2.0, 2.0, 0.0],
            "arrival_tolerance_m": 0.2,
            "speed_mps": 0.9,
            "route_context": (0, 0),
            "path": [[0.0, 0.0], [-0.2, -0.1], [-0.7, -0.1], [-0.7, 1.5], [2.0, 2.0]],
        },
        "lower": {
            "active": True,
            "target": [0.6, 0.0, 0.0],
            "arrival_tolerance_m": 0.2,
            "speed_mps": 0.9,
            "route_context": (0, 0),
            "path": [[-0.56, -0.56], [0.6, 0.0]],
        },
    }
    polygon = np.array([[-0.4, 0.3], [0.4, 0.3], [0.4, 1.1], [-0.4, 1.1]])
    obstacle = SimpleNamespace(
        name="/box", position=np.array([0.0, 0.7]), velocity=np.zeros(2), radius=0.6, footprint=polygon
    )
    settings = {"map_clearance_m": 0.03, "pair_clearance_m": 0.25}
    if controlled:
        settings["door_specs"] = [{"name": "door", "body_prim_path": "/box", "enabled": True}]
    original = copy.deepcopy((states, intentions))
    assert grid.segment_clear(states[0]["position"], intentions["upper"]["target"], 0.28)
    assert not segment_polygon_clear(states[0]["position"], intentions["upper"]["target"], polygon, 0.28)
    assert segment_polygon_clear(states[0]["position"], intentions["upper"]["path"][2], polygon, 0.28)
    coordinator = PassingCoordinator(grid, settings)
    for time_s in (0.0, 1.5, 2.25):
        result = coordinator.update(time_s, states, intentions, [obstacle], {})
    if controlled:
        assert result["objectives"] == {} and not result["status"]["transactions"]
    else:
        assert len(result["status"]["transactions"]) == 1
        transaction = result["status"]["transactions"][0]
        assert transaction["phase"] == "retreat"
        held = next(state for state in states if state["name"] == transaction["yielder"])
        passer = next(state for state in states if state["name"] == transaction["passer"])
        origin, refuge = np.asarray(transaction["origin_xy_m"]), np.asarray(transaction["refuge_xy_m"])
        assert math.dist(origin, refuge) <= 0.4
        assert grid.segment_clear(origin, refuge, held["radius_m"] + settings["map_clearance_m"])
        assert segment_polygon_clear(origin, refuge, polygon, held["radius_m"] + settings["map_clearance_m"])
        relative = origin - passer["position"][:2]
        movement = refuge - origin
        fraction = np.clip(-np.dot(relative, movement) / np.dot(movement, movement), 0, 1)
        assert (
            np.linalg.norm(relative + fraction * movement)
            >= held["radius_m"] + passer["radius_m"] + settings["pair_clearance_m"]
        )
        assert transaction["deadline_s"] == pytest.approx(2.25 + 4.0 + 0.4 / intentions[held["name"]]["speed_mps"])
    assert (states, intentions) == original


def test_passing_uses_the_certified_continuation_instead_of_an_oppositely_pointing_near_corner():
    coordinator, _, _, obstacles = _external_blocker_case()
    states = [
        {
            "name": "east",
            "position": [-4.633358599433307, -2.28978047877669, 0.0],
            "velocity": [0.0, 0.0, 0.0],
            "yaw": 0.2640678740211594,
            "radius_m": 0.25,
        },
        {
            "name": "west",
            "position": [-4.183712792100318, -1.6123239773187672, 0.0],
            "velocity": [-0.007198775187156468, -0.004847495350990548, 0.0],
            "yaw": -2.548958265268027,
            "radius_m": 0.25,
        },
    ]
    intentions = {
        "east": {
            "active": True,
            "target": [2.39829, -0.6456, 0.0],
            "arrival_tolerance_m": 0.2,
            "speed_mps": 0.9,
            "route_context": (0, 1),
            "path": [
                states[0]["position"][:2],
                [-1.09, 0.31],
                [0.49, 0.31],
                [2.27, 0.27],
                [2.43, 0.15],
                [2.47, -0.03],
                [2.39829, -0.6456],
            ],
        },
        "west": {
            "active": True,
            "target": [-6.25, -1.45, 0.0],
            "arrival_tolerance_m": 0.2,
            "speed_mps": 0.9,
            "route_context": (2, 1),
            "path": [
                states[1]["position"][:2],
                [-4.17, -1.63],
                [-4.35, -1.67],
                [-5.91, -1.61],
                [-6.059999997615814, -1.5018181824684145],
            ],
        },
    }
    original = copy.deepcopy((states, intentions))
    for time_s in (0.0, 1.5, 2.25):
        result = coordinator.update(time_s, states, intentions, obstacles, {})
    assert len(result["status"]["transactions"]) == 1
    transaction = result["status"]["transactions"][0]
    assert transaction["yielder"] == "east" and transaction["passer"] == "west"
    assert result["objectives"]["west"]["phase"] == "passing_wait"
    assert math.dist(transaction["origin_xy_m"], transaction["refuge_xy_m"]) <= 0.4
    assert transaction["deadline_s"] == pytest.approx(2.25 + 4.0 + 0.4 / intentions["east"]["speed_mps"])
    assert (states, intentions) == original


def _early_passage_case(blocked=False, yield_distance=0.4):
    grid = OccupancyGrid(np.zeros((80, 120), dtype=np.int8), 0.1, (-6.0, -4.0))
    states = [
        {"name": name, "position": [x, 0.0, 0.0], "velocity": [velocity, 0.0, 0.0], "yaw": yaw, "radius_m": 0.25}
        for name, x, velocity, yaw in (("east", -0.775, 0.05, 0.0), ("west", 0.775, -0.05, math.pi))
    ]
    intentions = {
        state["name"]: {
            "active": True,
            "target": [goal, 0.0, 0.0],
            "arrival_tolerance_m": 0.2,
            "speed_mps": 0.9,
            "route_context": (0, 0),
            "path": [state["position"][:2], [goal, 0.0]],
        }
        for state, goal in zip(states, (3.0, -3.0))
    }
    obstacles = []
    if blocked:
        for name, low, high in (("upper", 0.4, 4.0), ("lower", -4.0, -0.4)):
            obstacles.append(
                SimpleNamespace(
                    name=name,
                    position=np.array([0.0, (low + high) / 2]),
                    velocity=np.zeros(2),
                    radius=6.3,
                    footprint=np.array([[-6.0, low], [6.0, low], [6.0, high], [-6.0, high]]),
                )
            )
    settings = {"map_clearance_m": 0.03, "pair_clearance_m": 0.25, "max_yield_distance_m": yield_distance}
    return PassingCoordinator(grid, settings), states, intentions, obstacles


def _observe_motion(states, elapsed):
    for state in states:
        state["position"] = (np.asarray(state["position"]) + np.asarray(state["velocity"]) * elapsed).tolist()


@pytest.mark.parametrize("yield_distance", [0.4, 0.55])
def test_a_warm_slow_encounter_starts_only_after_a_complete_early_passing_certificate(yield_distance):
    coordinator, states, intentions, obstacles = _early_passage_case(yield_distance=yield_distance)
    original = copy.deepcopy(intentions)
    assert coordinator.update(0.0, states, intentions, obstacles, {}) == {
        "objectives": {},
        "status": {"transactions": []},
    }
    _observe_motion(states, 1.5)
    separation = math.dist(states[0]["position"], states[1]["position"])
    assert 0.83 < separation <= 0.75 + 2 * yield_distance
    result = coordinator.update(1.5, states, intentions, obstacles, {})
    assert len(result["status"]["transactions"]) == 1
    transaction = result["status"]["transactions"][0]
    assert transaction["phase"] == "retreat"
    assert any(objective["target"] is not None for objective in result["objectives"].values())
    assert math.dist(transaction["origin_xy_m"], transaction["refuge_xy_m"]) <= yield_distance
    assert transaction["deadline_s"] == pytest.approx(1.5 + 4.0 + yield_distance / 0.9)
    assert not result["status"].get("pending") and not coordinator.waiting_actors
    assert intentions == original


def test_a_stopped_walker_can_advance_along_a_certified_prefix_during_the_peers_retreat():
    coordinator, states, intentions, obstacles = _early_passage_case()
    holder, passer = states
    holder["velocity"] = [0.2, 0.0, 0.0]
    passer["velocity"] = [0.0, 0.0, 0.0]
    passer["yaw"] = -2.9
    assert coordinator.update(0.0, states, intentions, obstacles, {})["objectives"] == {}
    _observe_motion(states, 1.5)
    original = copy.deepcopy((states, intentions))

    result = coordinator.update(1.5, states, intentions, obstacles, {})
    transaction = result["status"]["transactions"][0]
    assert transaction["phase"] == "retreat"
    assert transaction["yielder"] == holder["name"]
    assert transaction["origin_xy_m"] == holder["position"][:2]
    assert result["objectives"][holder["name"]]["target"] == transaction["refuge_xy_m"]
    assert result["objectives"][passer["name"]]["phase"] == "passing_go"
    target = result["objectives"][passer["name"]]["target"]
    assert math.dist(passer["position"][:2], target) > 0.05

    # The retreat certificate assumes the passer follows this chord. A passer
    # facing slightly off it must turn toward it before advancing.
    aligning_states = copy.deepcopy(states)
    delta = np.asarray(target) - passer["position"][:2]
    bearing = math.atan2(delta[1], delta[0])
    aligning_states[1]["yaw"] = bearing + 0.17
    plans = [
        {
            "name": state["name"],
            "position": state["position"],
            "behavior": "walk",
            "speed": intentions[state["name"]]["speed_mps"],
            "arrival_tolerance": intentions[state["name"]]["arrival_tolerance_m"],
            "waypoints": [{"position": intentions[state["name"]]["target"]}],
        }
        for state in aligning_states
    ]
    engine = NavigationEngine(
        plans, coordinator.grid, {"map_clearance_m": 0.03, "pair_clearance_m": 0.25}, aligning_states
    )
    engine._passing = copy.deepcopy(coordinator)
    dt = 1.0 / 60.0
    try:
        turning = engine.step(1.5 + dt, dt, aligning_states)["agents"][1]
        assert turning["position"] == passer["position"]
        assert abs(math.remainder(turning["yaw"] - bearing, math.tau)) < 0.17
    finally:
        engine.close()

    obstacle = SimpleNamespace(
        name="new_obstacle", position=np.asarray(target), velocity=np.zeros(2), radius=0.35, footprint=None
    )
    blocked = coordinator.update(1.51, states, intentions, [obstacle], {})
    assert blocked["status"]["transactions"][0]["deadline_s"] == transaction["deadline_s"]
    assert blocked["objectives"][passer["name"]]["phase"] == "passing_wait"
    assert blocked["objectives"][passer["name"]]["target"] is None
    assert (states, intentions) == original


def test_zero_startup_does_not_claim_early_ownership_or_consume_the_first_moving_opportunity():
    coordinator, states, intentions, obstacles = _early_passage_case()
    for state in states:
        state["velocity"] = [0.0, 0.0, 0.0]
    for time_s in (0.0, 1.5):
        result = coordinator.update(time_s, states, intentions, obstacles, {})
        assert result == {"objectives": {}, "status": {"transactions": []}}
    for state, speed in zip(states, (0.05, -0.05)):
        state["velocity"] = [speed, 0.0, 0.0]
    _observe_motion(states, 0.1)
    result = coordinator.update(1.6, states, intentions, obstacles, {})
    assert len(result["status"]["transactions"]) == 1
    assert result["status"]["transactions"][0]["elapsed_s"] == 0


def test_failed_early_certification_leaves_motion_free_and_preserves_the_original_close_range_timer():
    coordinator, states, intentions, obstacles = _early_passage_case(blocked=True)
    coordinator.update(0.0, states, intentions, obstacles, {})
    _observe_motion(states, 1.5)
    first = coordinator.update(1.5, states, intentions, obstacles, {})
    assert first == {"objectives": {}, "status": {"transactions": []}}
    assert coordinator.waiting_actors == ()
    obstacles = obstacles[:1]
    _observe_motion(states, 0.1)
    # A new free route does not cause a second early attempt without enough measured motion and time.
    assert coordinator.update(1.6, states, intentions, obstacles, {}) == first
    for state, x in zip(states, (-0.4, 0.4)):
        state["position"][0] = x
        state["velocity"] = [0.0, 0.0, 0.0]
    assert coordinator.update(2.6, states, intentions, obstacles, {}) == first
    assert coordinator.update(2.6 + 0.75 - 1 / 60, states, intentions, obstacles, {}) == first
    result = coordinator.update(2.6 + 0.75, states, intentions, obstacles, {})
    assert len(result["status"]["transactions"]) == 1
    assert result["status"]["transactions"][0]["deadline_s"] == pytest.approx(2.6 + 0.75 + 4.0 + 0.4 / 0.9)


@pytest.mark.parametrize("boundary", ["reset", "context", "separated"])
def test_a_new_encounter_clears_a_failed_early_attempt(boundary):
    coordinator, states, intentions, obstacles = _early_passage_case(blocked=True)
    coordinator.update(0.0, states, intentions, obstacles, {})
    _observe_motion(states, 1.5)
    empty = coordinator.update(1.5, states, intentions, obstacles, {})
    assert empty == {"objectives": {}, "status": {"transactions": []}}
    obstacles = obstacles[:1]
    _observe_motion(states, 0.1)
    assert coordinator.update(1.6, states, intentions, obstacles, {}) == empty
    if boundary == "separated":
        for state, x in zip(states, (-2.5, 2.5)):
            state["position"][0] = x
        assert coordinator.update(4.7, states, intentions, obstacles, {}) == empty
        for state, x in zip(states, (-0.7, 0.7)):
            state["position"][0] = x
        now = 6.1
    else:
        if boundary == "reset":
            coordinator.reset()
        else:
            intentions["east"]["route_context"] = (1, 0)
        changed = coordinator.update(1.7, states, intentions, obstacles, {})
        if boundary == "context":
            assert len(changed["status"]["transactions"]) == 1
            assert changed["status"]["transactions"][0]["elapsed_s"] == 0
            return
        assert changed == empty
        _observe_motion(states, 1.5 + 1 / 60)
        now = 1.7 + 1.5 + 1 / 60
    result = coordinator.update(now, states, intentions, obstacles, {})
    assert len(result["status"]["transactions"]) == 1


def test_a_failed_early_certificate_can_recover_after_measured_motion_before_close_range():
    coordinator, states, intentions, obstacles = _early_passage_case(blocked=True)
    coordinator.update(0.0, states, intentions, obstacles, {})
    _observe_motion(states, 1.5)
    empty = coordinator.update(1.5, states, intentions, obstacles, {})
    assert empty == {"objectives": {}, "status": {"transactions": []}}

    obstacles = obstacles[:1]
    _observe_motion(states, 0.8)
    assert coordinator.update(2.3, states, intentions, obstacles, {}) == empty
    _observe_motion(states, 0.4)
    assert math.dist(states[0]["position"], states[1]["position"]) > 0.83
    result = coordinator.update(2.7, states, intentions, obstacles, {})
    assert len(result["status"]["transactions"]) == 1
    assert result["status"]["transactions"][0]["elapsed_s"] == 0
    assert not result["status"].get("pending")


@pytest.mark.parametrize("boundary", ["wide", "priority"])
def test_early_certification_does_not_claim_a_wide_or_higher_priority_encounter(boundary):
    coordinator, states, intentions, obstacles = _early_passage_case()
    priority = {}
    if boundary == "wide":
        for state, x in zip(states, (-2.5, 2.5)):
            state["position"][0] = x
    else:
        priority["east"] = {"target": None, "phase": "door_wait"}
    for time_s in (0.0, 1.5, 2.25):
        result = coordinator.update(time_s, states, intentions, obstacles, priority)
        assert result == {"objectives": {}, "status": {"transactions": []}}
    assert not coordinator.waiting_actors


@pytest.mark.parametrize("legacy_close", [False, True])
def test_one_failed_early_certificate_does_not_spend_the_next_pairs_future_update_budget(legacy_close):
    grid = OccupancyGrid(np.zeros((80, 120), dtype=np.int8), 0.1, (-6.0, -4.0))
    states = [
        {"name": name, "position": [x, y, 0.0], "velocity": [speed, 0.0, 0.0], "yaw": yaw, "radius_m": 0.25}
        for name, x, y, speed, yaw in (
            ("a", -0.775, -2.0, 0.05, 0.0),
            ("b", 0.775, -2.0, -0.05, math.pi),
            ("c", -0.775, 2.0, 0.05, 0.0),
            ("d", 0.775, 2.0, -0.05, math.pi),
        )
    ]
    if legacy_close:
        for state in states:
            state["velocity"] = [0.0, 0.0, 0.0]
        for state, x in zip(states[2:], (-0.4, 0.4)):
            state["position"][0] = x
    intentions = {
        state["name"]: {
            "active": True,
            "target": [goal, state["position"][1], 0.0],
            "arrival_tolerance_m": 0.2,
            "speed_mps": 0.9,
            "route_context": (0, 0),
            "path": [],
        }
        for state, goal in zip(states, (3.0, -3.0, 3.0, -3.0))
    }
    obstacles = [
        SimpleNamespace(
            name=str(low),
            position=np.array([0.0, (low + high) / 2]),
            velocity=np.zeros(2),
            radius=6.1,
            footprint=np.array([[-6.0, low], [6.0, low], [6.0, high], [-6.0, high]]),
        )
        for low, high in ((-1.6, -1.2), (-2.8, -2.4))
    ]
    coordinator = PassingCoordinator(grid, {"map_clearance_m": 0.03, "pair_clearance_m": 0.25})
    coordinator.update(0.0, states, intentions, obstacles, {})
    _observe_motion(states, 1.5)
    result = coordinator.update(1.5, states, intentions, obstacles, {})
    assert result == {"objectives": {}, "status": {"transactions": []}}
    if legacy_close:
        for state, speed in zip(states[:2], (0.05, -0.05)):
            state["velocity"][0] = speed
        _observe_motion(states, 0.75)
        now = 1.5 + 0.75
    else:
        _observe_motion(states, 1 / 60)
        now = 1.5 + 1 / 60
    result = coordinator.update(now, states, intentions, obstacles, {})
    assert len(result["status"]["transactions"]) == 1
    transaction = result["status"]["transactions"][0]
    assert {transaction["yielder"], transaction["passer"]} == {"c", "d"}
    assert not {"a", "b"} & set(result["objectives"])


def _continuous_goal_case():
    coordinator, states, intentions, obstacles = _early_passage_case()
    states[0]["position"] = [-2.3, 0.0, 0.0]
    states[1].update(position=[-0.1, 0.0, 0.0], velocity=[0.5, 0.0, 0.0], yaw=0.0)
    intentions["west"].update(target=[0.7, 0.0, 0.0], path=[], phase="walking")
    return coordinator, states, intentions, obstacles


def test_continuous_navigation_stays_warm_across_an_arrived_zero_wait_goal_and_new_route_context():
    coordinator, states, intentions, obstacles = _continuous_goal_case()
    coordinator.update(0.0, states, intentions, obstacles, {})
    states[0]["position"][0] += 0.05 * 1.5
    states[1].update(position=[0.7, 0.0, 0.0], velocity=[0.05, 0.0, 0.0])
    assert coordinator.update(1.5, states, intentions, obstacles, {})["status"]["transactions"] == []
    intentions["west"].update(target=[-3.0, 0.0, 0.0], route_context=(1, 0))
    _observe_motion(states, 1 / 60)
    started = 1.5 + 1 / 60
    assert coordinator.update(started, states, intentions, obstacles, {})["status"]["transactions"] == []
    _observe_motion(states, 0.4)
    result = coordinator.update(started + 0.4, states, intentions, obstacles, {})
    assert len(result["status"]["transactions"]) == 1
    assert result["status"]["transactions"][0]["elapsed_s"] == 0


def test_temporary_goal_ownership_keeps_navigation_warm_without_claiming_the_owned_actor():
    coordinator, states, intentions, obstacles = _early_passage_case()
    states[0]["velocity"] = [0.0, 0.0, 0.0]
    priority = {"east": {"target": None, "phase": "goal_wait"}}
    coordinator.update(0.0, states, intentions, obstacles, priority)
    _observe_motion(states, 1.5)
    result = coordinator.update(1.5, states, intentions, obstacles, priority)
    assert result == {"objectives": {}, "status": {"transactions": []}}
    _observe_motion(states, 0.1)
    result = coordinator.update(1.6, states, intentions, obstacles, {})
    assert len(result["status"]["transactions"]) == 1


def test_an_original_dwell_resets_warm_time_before_navigation_resumes():
    coordinator, states, intentions, obstacles = _continuous_goal_case()
    coordinator.update(0.0, states, intentions, obstacles, {})
    states[0]["position"][0] += 0.05 * 1.5
    states[1].update(position=[0.7, 0.0, 0.0], velocity=[0.0, 0.0, 0.0])
    intentions["west"].update(active=False, phase="dwelling")
    assert coordinator.update(1.5, states, intentions, obstacles, {})["status"]["transactions"] == []
    _observe_motion(states, 1.0)
    coordinator.update(2.5, states, intentions, obstacles, {})
    intentions["west"].update(active=True, phase="walking", target=[-3.0, 0.0, 0.0], route_context=(1, 0))
    assert coordinator.update(2.6, states, intentions, obstacles, {})["status"]["transactions"] == []
    states[1].update(velocity=[-0.05, 0.0, 0.0], yaw=math.pi)
    _observe_motion(states, 1.5 + 1 / 60)
    started = 2.6 + 1.5 + 1 / 60
    assert coordinator.update(started, states, intentions, obstacles, {})["status"]["transactions"] == []
    _observe_motion(states, 0.4)
    result = coordinator.update(started + 0.4, states, intentions, obstacles, {})
    assert len(result["status"]["transactions"]) == 1


def test_a_new_context_keeps_active_warm_time_but_restarts_close_conflict_qualification():
    coordinator, states, intentions, obstacles = _early_passage_case()
    for state, x in zip(states, (-0.4, 0.4)):
        state.update(position=[x, 0.0, 0.0], velocity=[0.0, 0.0, 0.0])
    empty = {"objectives": {}, "status": {"transactions": []}}
    assert coordinator.update(0.0, states, intentions, obstacles, {}) == empty
    assert coordinator.update(1.5, states, intentions, obstacles, {}) == empty
    intentions["east"]["route_context"] = (1, 0)
    assert coordinator.update(1.7, states, intentions, obstacles, {}) == empty
    assert coordinator.update(1.5 + 0.75, states, intentions, obstacles, {}) == empty
    result = coordinator.update(1.7 + 0.75, states, intentions, obstacles, {})
    assert len(result["status"]["transactions"]) == 1
    assert result["status"]["transactions"][0]["deadline_s"] == pytest.approx(1.7 + 0.75 + 4.0 + 0.4 / 0.9)


@pytest.mark.parametrize("occlusion", ["static", "observed"])
def test_distant_certification_waits_for_current_centerline_visibility_without_poisoning_its_opportunity(occlusion):
    coordinator, states, intentions, _ = _early_passage_case()
    polygon = np.array([[-0.1, -0.3], [0.1, -0.3], [0.1, 0.3], [-0.1, 0.3]])
    obstacles = []
    if occlusion == "static":
        cells = np.zeros((80, 120), dtype=np.int8)
        cells[37:43, 59:61] = 100
        grid = OccupancyGrid(cells, 0.1, (-6.0, -4.0))
        coordinator = PassingCoordinator(grid, {"map_clearance_m": 0.03, "pair_clearance_m": 0.25})
    else:
        obstacles = [
            SimpleNamespace(name="/box", position=np.zeros(2), velocity=np.zeros(2), radius=0.32, footprint=polygon)
        ]
    for state, x in zip(states, (-1.5, 1.5)):
        state["position"][0] = x
    coordinator.update(0.0, states, intentions, obstacles, {})
    _observe_motion(states, 1.5)
    empty = {"objectives": {}, "status": {"transactions": []}}
    assert coordinator.update(1.5, states, intentions, obstacles, {}) == empty
    for state in states:
        state["position"][1] = 1.0
    assert coordinator.update(2.0, states, intentions, obstacles, {}) == empty
    # Brief visibility before another occlusion does not qualify the encounter.
    for state in states:
        state["position"][1] = 0.0
    assert coordinator.update(2.1, states, intentions, obstacles, {}) == empty
    for state in states:
        state["position"][1] = 1.0
    assert coordinator.update(2.5, states, intentions, obstacles, {}) == empty
    _observe_motion(states, 0.4)
    result = coordinator.update(2.5 + 0.4, states, intentions, obstacles, {})
    assert len(result["status"]["transactions"]) == 1


def test_failed_distant_certification_does_not_consume_the_later_nearby_certificate():
    coordinator, states, intentions, obstacles = _early_passage_case(blocked=True)
    for state, x in zip(states, (-1.5, 1.5)):
        state["position"][0] = x
    coordinator.update(0.0, states, intentions, obstacles, {})
    _observe_motion(states, 1.5)
    empty = {"objectives": {}, "status": {"transactions": []}}
    assert coordinator.update(1.5, states, intentions, obstacles, {}) == empty
    _observe_motion(states, 0.4)
    assert coordinator.update(1.5 + 0.4, states, intentions, obstacles, {}) == empty
    obstacles = obstacles[:1]
    _observe_motion(states, 0.1)
    assert coordinator.update(2.0, states, intentions, obstacles, {}) == empty
    for state, x in zip(states, (-0.7, 0.7)):
        state["position"][0] = x
    result = coordinator.update(2.6, states, intentions, obstacles, {})
    assert len(result["status"]["transactions"]) == 1
    assert result["status"]["transactions"][0]["elapsed_s"] == 0
