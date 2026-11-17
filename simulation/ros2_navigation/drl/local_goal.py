# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NumPy-2-safe port of the DRL-VO pure-pursuit local goal."""

from __future__ import annotations

import math

import numpy as np


def _closest_on_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, float]:
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        return start.copy(), float(np.linalg.norm(point - start))
    direction = vector / length
    projected = float(np.dot(point - start, direction))
    if projected < 0.0:
        closest = start
    elif projected > length:
        closest = end
    else:
        closest = start + projected * direction
    return closest.copy(), float(np.linalg.norm(closest - point))


def _find_closest_point(path: np.ndarray, point: np.ndarray, segment: int = -1) -> tuple[np.ndarray, float, int]:
    closest = np.array([np.nan, np.nan])
    distance = math.inf
    selected = -1
    if segment == -1:
        for index in range(len(path) - 1):
            candidate, candidate_distance, candidate_segment = _find_closest_point(path, point, index)
            if candidate_distance < distance:
                closest, distance, selected = candidate, candidate_distance, candidate_segment
        return closest, distance, selected
    return (*_closest_on_segment(point, path[segment], path[segment + 1]), segment)


def _find_goal(
    path: np.ndarray, point: np.ndarray, closest: np.ndarray, distance: float, segment: int, lookahead: float
) -> np.ndarray:
    if distance > lookahead:
        return closest.copy()
    segment_max = len(path) - 2
    end = path[segment + 1]
    end_distance = float(np.linalg.norm(point - end))
    while end_distance < lookahead and segment < segment_max:
        segment += 1
        end = path[segment + 1]
        end_distance = float(np.linalg.norm(point - end))
    if end_distance < lookahead:
        return path[segment_max + 1].copy()
    closest, _, segment = _find_closest_point(path, point, segment)
    start, end = path[segment], path[segment + 1]
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        return end.copy()
    direction = vector / length
    relative = point - closest
    along = float(np.dot(relative, direction))
    lateral = abs(float(relative[0] * direction[1] - relative[1] * direction[0]))
    radicand = max(0.0, lookahead * lookahead - lateral * lateral)
    return closest + (math.sqrt(radicand) + along) * direction


def local_goal_base(
    path_xy: np.ndarray | list[tuple[float, float]] | list[list[float]],
    robot_pose_xyyaw: tuple[float, float, float],
    lookahead_m: float = 0.5,
) -> np.ndarray:
    """Return the pure-pursuit target in the robot base frame [m]."""
    path = np.asarray(path_xy, dtype=float)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) == 0 or not np.isfinite(path).all():
        raise ValueError("path_xy must be a finite nonempty (N, 2) array.")
    pose = np.asarray(robot_pose_xyyaw, dtype=float)
    if pose.shape != (3,) or not np.isfinite(pose).all():
        raise ValueError("robot_pose_xyyaw must contain three finite values.")
    if not math.isfinite(lookahead_m) or lookahead_m <= 0:
        raise ValueError("lookahead_m must be finite and positive.")
    keep = np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-10]
    path = path[keep]
    if len(path) < 2:
        target = path[-1]
    else:
        closest, distance, segment = _find_closest_point(path, pose[:2])
        target = _find_goal(path, pose[:2], closest, distance, segment, float(lookahead_m))
    delta = np.asarray(target, dtype=float) - pose[:2]
    yaw = float(pose[2])
    c, s = math.cos(yaw), math.sin(yaw)
    local = np.array([c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]], dtype=float)
    if not np.isfinite(local).all():
        raise ValueError("Pure pursuit produced a non-finite local goal.")
    return local
