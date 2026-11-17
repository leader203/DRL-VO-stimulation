# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Synthesize the frozen DRL-VO 720-beam front scan from a static occupancy grid."""

from __future__ import annotations

import math

import numpy as np

FRONT_BEAM_COUNT = 720
FRONT_ANGLE_MIN_RAD = -math.pi / 2
FRONT_ANGLE_INCREMENT_RAD = math.pi / 720


def front_beam_angles(
    count: int = FRONT_BEAM_COUNT,
    angle_min: float = FRONT_ANGLE_MIN_RAD,
    angle_increment: float = FRONT_ANGLE_INCREMENT_RAD,
) -> np.ndarray:
    """Return the model's front-FOV beam angles relative to the robot base [rad]."""
    if type(count) is not int or count <= 0:
        raise ValueError("count must be a positive integer.")
    if not all(math.isfinite(value) for value in (angle_min, angle_increment)) or angle_increment <= 0:
        raise ValueError("angle_min and angle_increment must be finite with a positive increment.")
    return angle_min + np.arange(count, dtype=np.float64) * angle_increment


def overlay_dynamic_obstacles(
    occupancy: np.ndarray,
    resolution_m: float,
    origin_xy_m: tuple[float, float],
    pedestrian_states: list[dict[str, object]] = (),
    obstacles: list[dict[str, object]] = (),
    *,
    exclude_prefix: str | None = None,
    margin_m: float = 0.03,
) -> np.ndarray:
    """Return a static grid with pedestrian and dynamic obstacle disks marked occupied."""
    grid = np.array(occupancy, dtype=np.int8, copy=True)
    if not math.isfinite(resolution_m) or resolution_m <= 0:
        raise ValueError("resolution_m must be finite and positive.")
    origin_x, origin_y = (float(value) for value in origin_xy_m)
    height, width = grid.shape
    if not math.isfinite(margin_m) or margin_m < 0:
        raise ValueError("margin_m must be finite and nonnegative.")

    def paint(x: float, y: float, radius: float) -> None:
        if not all(math.isfinite(value) for value in (x, y, radius)) or radius <= 0:
            return
        row_low = max(0, math.floor((y - origin_y - radius) / resolution_m))
        row_high = min(height - 1, math.ceil((y - origin_y + radius) / resolution_m))
        col_low = max(0, math.floor((x - origin_x - radius) / resolution_m))
        col_high = min(width - 1, math.ceil((x - origin_x + radius) / resolution_m))
        if row_low > row_high or col_low > col_high:
            return
        rows = np.arange(row_low, row_high + 1)
        cols = np.arange(col_low, col_high + 1)
        centers_x = origin_x + (cols + 0.5) * resolution_m
        centers_y = origin_y + (rows + 0.5) * resolution_m
        mask = (centers_x[None, :] - x) ** 2 + (centers_y[:, None] - y) ** 2 <= radius * radius
        block = grid[np.ix_(rows, cols)]
        block[mask] = 100
        grid[np.ix_(rows, cols)] = block

    for state in pedestrian_states:
        position = state.get("position")
        if position is None:
            continue
        paint(float(position[0]), float(position[1]), float(state["radius_m"]) + margin_m)
    for obstacle in obstacles:
        path = str(obstacle.get("prim_path", ""))
        if exclude_prefix is not None and (path == exclude_prefix or path.startswith(exclude_prefix + "/")):
            continue
        position = obstacle.get("position")
        radius = float(obstacle.get("radius_m", 0.0))
        if position is not None:
            paint(float(position[0]), float(position[1]), radius + margin_m)
    return grid


def synthesize_front_scan(
    occupancy: np.ndarray,
    resolution_m: float,
    origin_xy_m: tuple[float, float],
    robot_xy_m: tuple[float, float],
    yaw_rad: float,
    *,
    min_range_m: float = 0.1,
    max_range_m: float = 8.0,
    sample_step_m: float | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Cast the model's 720 front beams against occupied grid cells.

    The frozen policy consumes the legacy 180-degree front scan: angles
    ``-pi/2 + j * pi/720`` for ``j`` in ``0..719``, ordered counter-clockwise
    from the robot's right through forward to its left.  A value of zero means
    no return; unknown and free cells do not block a ray.
    """
    grid = np.asarray(occupancy)
    if grid.ndim != 2 or not np.issubdtype(grid.dtype, np.integer):
        raise ValueError("occupancy must be a two-dimensional integer grid.")
    if not math.isfinite(resolution_m) or resolution_m <= 0:
        raise ValueError("resolution_m must be finite and positive.")
    origin_x, origin_y = (float(value) for value in origin_xy_m)
    robot_x, robot_y = (float(value) for value in robot_xy_m)
    yaw = float(yaw_rad)
    if not all(math.isfinite(value) for value in (origin_x, origin_y, robot_x, robot_y, yaw)):
        raise ValueError("Map origin, robot pose and yaw must be finite.")
    if not (math.isfinite(min_range_m) and math.isfinite(max_range_m) and 0 < min_range_m < max_range_m):
        raise ValueError("Scan ranges must be finite and satisfy 0 < min_range_m < max_range_m.")
    step = resolution_m / 2.0 if sample_step_m is None else float(sample_step_m)
    if not math.isfinite(step) or step <= 0:
        raise ValueError("sample_step_m must be finite and positive.")

    distances = np.arange(min_range_m, max_range_m + step * 0.5, step, dtype=np.float64)
    angles = yaw + front_beam_angles()
    x = robot_x + np.cos(angles)[:, None] * distances[None, :]
    y = robot_y + np.sin(angles)[:, None] * distances[None, :]
    rows = np.floor((y - origin_y) / resolution_m).astype(np.int64)
    cols = np.floor((x - origin_x) / resolution_m).astype(np.int64)
    height, width = grid.shape
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    rows = np.clip(rows, 0, height - 1)
    cols = np.clip(cols, 0, width - 1)
    blocked = inside & (grid[rows, cols] == 100)
    hit = blocked.any(axis=1)
    first = np.argmax(blocked, axis=1)
    ranges = np.where(hit, distances[first], 0.0).astype(np.float32)
    ranges[(ranges < min_range_m) | (ranges > max_range_m)] = 0.0
    valid = ranges > 0
    stats = {
        "valid_beams": int(valid.sum()),
        "beam_count": int(FRONT_BEAM_COUNT),
        "coverage": float(valid.mean()),
        "minimum_m": float(ranges[valid].min()) if valid.any() else None,
        "maximum_m": float(ranges[valid].max()) if valid.any() else None,
        "sample_step_m": step,
        "source": "occupancy_grid",
    }
    return ranges, stats
