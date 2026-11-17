# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Conservative safety guards around the frozen DRL-VO action."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...configuration.drl import DrlPolicySettings
from ..planner import OccupancyGrid
from .robot import RobotState


@dataclass(frozen=True)
class GuardResult:
    """One guarded planar command and the reason for any override."""

    linear_m_s: float
    angular_rad_s: float
    override: bool
    reason: str | None
    details: dict[str, object]


def _pedestrian_clearance(
    robot: RobotState, states: list[dict[str, object]], horizon_s: float, robot_radius_m: float
) -> tuple[float, str | None]:
    minimum = math.inf
    source = None
    robot_position = np.asarray(robot.pose_xyyaw[:2], dtype=float)
    robot_velocity = np.asarray(robot.linear_velocity_ros_mps[:2], dtype=float)
    for state in states:
        position = np.asarray(state["position"][:2], dtype=float)
        velocity = np.asarray(state["velocity"][:2], dtype=float)
        radius = float(state["radius_m"])
        relative = position - robot_position
        relative_velocity = velocity - robot_velocity
        current = float(np.linalg.norm(relative)) - radius - robot_radius_m
        predicted = current
        speed_squared = float(np.dot(relative_velocity, relative_velocity))
        if speed_squared > 1e-12:
            time = max(0.0, min(horizon_s, -float(np.dot(relative, relative_velocity)) / speed_squared))
            predicted = float(np.linalg.norm(relative + relative_velocity * time)) - radius - robot_radius_m
        clearance = min(current, predicted)
        if clearance < minimum:
            minimum = clearance
            source = str(state.get("name"))
    return minimum, source


def apply_safety_guards(
    *,
    commanded_linear_m_s: float,
    commanded_angular_rad_s: float,
    robot: RobotState,
    states: list[dict[str, object]],
    scan: np.ndarray,
    grid: OccupancyGrid,
    settings: DrlPolicySettings,
    stopped: bool,
    map_clearance_m: float,
) -> GuardResult:
    """Clamp and veto a policy command; all vetoes publish zero."""
    linear = min(max(float(commanded_linear_m_s), 0.0), settings.max_linear_m_s)
    angular = min(max(float(commanded_angular_rad_s), -settings.max_angular_rad_s), settings.max_angular_rad_s)
    reasons: list[str] = []
    details: dict[str, object] = {}

    ranges = np.asarray(scan, dtype=float)
    valid = ranges[ranges > 0]
    minimum_scan = float(valid.min()) if valid.size else None
    details["minimum_scan_m"] = minimum_scan
    if stopped:
        scan_blocked = minimum_scan is not None and minimum_scan < settings.min_scan_release_m
    else:
        scan_blocked = minimum_scan is not None and minimum_scan < settings.min_scan_stop_m
    pedestrian, pedestrian_name = _pedestrian_clearance(
        robot, states, settings.pedestrian_horizon_s, settings.robot_clearance_radius_m
    )
    details["pedestrian_clearance_m"] = pedestrian if math.isfinite(pedestrian) else None
    details["pedestrian_name"] = pedestrian_name
    if stopped:
        pedestrian_blocked = pedestrian < settings.pedestrian_release_m
    else:
        pedestrian_blocked = pedestrian < settings.min_pedestrian_clearance_m
    if scan_blocked:
        reasons.append("scan_clearance")
    if pedestrian_blocked:
        reasons.append("pedestrian_clearance")
    if linear > 0:
        heading = np.array([math.cos(robot.yaw_rad), math.sin(robot.yaw_rad)], dtype=float)
        start = np.asarray(robot.pose_xyyaw[:2], dtype=float)
        end = start + heading * (settings.forward_guard_distance_m + settings.robot_length_m / 2)
        clearance = settings.robot_width_m / 2 + map_clearance_m
        if not grid.segment_clear(tuple(start), tuple(end), clearance):
            reasons.append("forward_clearance")
            linear = 0.0
    if reasons:
        return GuardResult(0.0, 0.0, True, ",".join(dict.fromkeys(reasons)), details)
    return GuardResult(linear, angular, False, None, details)
