# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU tests for the frozen DRL-VO front-scan synthesizer."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from pedstrain_simulation.simulation.ros2_navigation.drl.scan import (
    FRONT_ANGLE_INCREMENT_RAD,
    FRONT_ANGLE_MIN_RAD,
    front_beam_angles,
    overlay_dynamic_obstacles,
    synthesize_front_scan,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures/native_goal_queue_20261104.npz"


def _wall_grid() -> tuple[np.ndarray, float, tuple[float, float]]:
    occupancy = np.zeros((100, 100), dtype=np.int8)
    resolution = 0.05
    origin = (-2.5, -2.5)
    occupancy[:, 80] = 100
    return occupancy, resolution, origin


def test_front_angles_match_legacy_contract() -> None:
    angles = front_beam_angles()
    assert angles.shape == (720,)
    assert math.isclose(angles[0], -math.pi / 2)
    assert math.isclose(angles[360], 0.0)
    assert math.isclose(angles[719], -math.pi / 2 + 719 * math.pi / 720)
    assert math.isclose(angles[1] - angles[0], FRONT_ANGLE_INCREMENT_RAD)
    assert math.isclose(FRONT_ANGLE_MIN_RAD, -math.pi / 2)


def test_center_beam_hits_wall_and_sides_do_not() -> None:
    occupancy, resolution, origin = _wall_grid()
    scan, stats = synthesize_front_scan(occupancy, resolution, origin, (0.0, 0.0), 0.0)
    assert scan.shape == (720,)
    assert scan.dtype == np.float32
    assert math.isclose(float(scan[360]), 1.5, abs_tol=0.01)
    assert float(scan[0]) == 0.0
    assert stats["valid_beams"] > 0
    assert math.isclose(stats["sample_step_m"], resolution / 2)


def test_yaw_rotates_front_direction() -> None:
    occupancy = np.zeros((100, 100), dtype=np.int8)
    occupancy[80, :] = 100
    scan, _ = synthesize_front_scan(occupancy, 0.05, (-2.5, -2.5), (0.0, 0.0), math.pi / 2)
    assert math.isclose(float(scan[360]), 1.5, abs_tol=0.01)


def test_unknown_cells_do_not_block_and_out_of_grid_is_no_return() -> None:
    occupancy = np.full((100, 100), -1, dtype=np.int8)
    scan, stats = synthesize_front_scan(occupancy, 0.05, (-2.5, -2.5), (0.0, 0.0), 0.0)
    assert stats["valid_beams"] == 0
    assert np.all(scan == 0.0)
    far, _ = synthesize_front_scan(occupancy, 0.05, (-2.5, -2.5), (100.0, 100.0), 0.0)
    assert np.all(far == 0.0)


def test_fixture_map_spawn_has_valid_returns() -> None:
    assert FIXTURE.is_file()
    with np.load(FIXTURE, allow_pickle=False) as data:
        occupancy = data["occupancy"]
        resolution = float(data["resolution_m"])
        origin = tuple(float(value) for value in data["origin_xy_m"])
    scan, stats = synthesize_front_scan(occupancy, resolution, origin, (-2.3, -2.0), 0.0)
    assert scan.shape == (720,)
    assert stats["valid_beams"] > 100
    valid = scan[scan > 0]
    assert np.all(valid >= 0.1) and np.all(valid <= 8.0)


def test_dynamic_overlay_marks_pedestrians_and_excludes_robot() -> None:
    occupancy = np.zeros((100, 100), dtype=np.int8)
    pedestrians = [{"position": [0.0, 0.0, 0.0], "radius_m": 0.25, "name": "p"}]
    obstacles = [
        {"prim_path": "/World/EVT_3/base_link", "position": [1.5, -1.0, 0.0], "radius_m": 1.0},
        {"prim_path": "/World/AutomaticDoors/Door1", "position": [1.0, 0.5, 0.0], "radius_m": 0.2},
    ]
    overlaid = overlay_dynamic_obstacles(
        occupancy, 0.05, (-2.5, -2.5), pedestrians, obstacles, exclude_prefix="/World/EVT_3"
    )
    assert overlaid[50, 50] == 100
    assert overlaid[60, 70] == 100
    assert overlaid[30, 80] == 0
    assert occupancy[50, 50] == 0
    scan, stats = synthesize_front_scan(overlaid, 0.05, (-2.5, -2.5), (0.0, 0.0), 0.0)
    assert stats["minimum_m"] is not None
    assert float(scan[360]) < 0.5
