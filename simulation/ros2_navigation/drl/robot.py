# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read the live EVT robot pose from PhysX and expose ROS map motion."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..frames import StageFrame


@dataclass(frozen=True)
class RobotState:
    """One measured robot sample in ROS map coordinates."""

    position_ros_m: tuple[float, float, float]
    yaw_rad: float
    pose_xyyaw: tuple[float, float, float]
    linear_velocity_ros_mps: tuple[float, float, float]
    yaw_rate_radps: float
    forward_speed_mps: float


def _rotate_quaternion(rotation: tuple[float, float, float, float], vector: tuple[float, float, float]):
    """Rotate a vector by an xyzw quaternion without importing pxr."""
    x, y, z, w = (float(value) for value in rotation)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        raise RuntimeError("Robot PhysX rotation is degenerate.")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


class PhysxRobotObserver:
    """Sample one enabled rigid body from PhysX on the Kit thread."""

    def __init__(self, rigid_body_path: str, frame: StageFrame) -> None:
        if not isinstance(rigid_body_path, str) or not rigid_body_path.startswith("/"):
            raise ValueError("rigid_body_path must be an absolute prim path.")
        self.rigid_body_path = rigid_body_path
        self.frame = frame
        self._last_time: float | None = None
        self._last_position: np.ndarray | None = None
        self._last_yaw: float | None = None

    def read(self, now_s: float) -> RobotState:
        import omni.physx

        pose = omni.physx.get_physx_interface().get_rigidbody_transformation(self.rigid_body_path)
        if not isinstance(pose, dict) or not pose.get("ret_val"):
            raise RuntimeError(f"PhysX has no live pose for {self.rigid_body_path}.")
        position = np.asarray(tuple(float(value) for value in pose["position"]), dtype=float)
        rotation = tuple(float(value) for value in pose["rotation"])
        if position.shape != (3,) or len(rotation) != 4 or not np.isfinite(position).all():
            raise RuntimeError(f"PhysX returned an invalid pose for {self.rigid_body_path}.")
        position_ros = np.asarray(self.frame.stage_to_ros(tuple(position)), dtype=float)
        forward = _rotate_quaternion(rotation, (1.0, 0.0, 0.0))
        forward_ros = np.asarray(self.frame.stage_vector_to_ros(forward), dtype=float)
        yaw = math.atan2(float(forward_ros[1]), float(forward_ros[0]))
        linear = np.zeros(3, dtype=float)
        yaw_rate = 0.0
        now = float(now_s)
        if self._last_time is not None and now > self._last_time:
            delta = max(now - self._last_time, 1e-9)
            linear = (position_ros - self._last_position) / delta
            yaw_error = math.atan2(math.sin(yaw - self._last_yaw), math.cos(yaw - self._last_yaw))
            yaw_rate = yaw_error / delta
        forward_ros = forward_ros / max(float(np.linalg.norm(forward_ros[:2])), 1e-12)
        forward_speed = float(np.dot(linear[:2], forward_ros[:2]))
        self._last_time = now
        self._last_position = position_ros.copy()
        self._last_yaw = yaw
        return RobotState(
            position_ros_m=(float(position_ros[0]), float(position_ros[1]), float(position_ros[2])),
            yaw_rad=float(yaw),
            pose_xyyaw=(float(position_ros[0]), float(position_ros[1]), float(yaw)),
            linear_velocity_ros_mps=(float(linear[0]), float(linear[1]), float(linear[2])),
            yaw_rate_radps=float(yaw_rate),
            forward_speed_mps=forward_speed,
        )
