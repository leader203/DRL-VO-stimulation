# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit right-handed conversions between USD world and ROS map frames."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real


def _point(value: Sequence[float], name: str) -> tuple[float, float, float]:
    if len(value) != 3 or any(isinstance(component, bool) or not isinstance(component, Real) for component in value):
        raise ValueError(f"{name} must contain three finite numbers.")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{name} must contain three finite numbers.")
    return result


@dataclass(frozen=True)
class StageFrame:
    """Convert stage-world lengths to a right-handed ROS map frame.

    Attributes:
        meters_per_unit: Meters represented by one USD length unit [m].
        up_axis: USD stage up axis, ``Y`` or ``Z``.
        origin_m: ROS map origin in canonical Z-up world coordinates [m].
        yaw_rad: Map-axis yaw relative to the canonical world [rad].

    Canonical world coordinates use the ``world_m`` input convention. For Y-up USD,
    canonical coordinates are ``(stage_x, -stage_z, stage_y) * meters_per_unit``.
    Parents must be composed before passing a point to :meth:`stage_to_ros`.
    """

    meters_per_unit: float
    up_axis: str
    origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_rad: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.meters_per_unit, bool)
            or not isinstance(self.meters_per_unit, Real)
            or not math.isfinite(self.meters_per_unit)
            or self.meters_per_unit <= 0
        ):
            raise ValueError("meters_per_unit must be finite and positive [m].")
        if self.up_axis not in ("Y", "Z"):
            raise ValueError("up_axis must be Y or Z.")
        if isinstance(self.yaw_rad, bool) or not isinstance(self.yaw_rad, Real) or not math.isfinite(self.yaw_rad):
            raise ValueError("yaw_rad must be finite [rad].")
        object.__setattr__(self, "origin_m", _point(self.origin_m, "origin_m"))

    def world_to_ros(self, point: Sequence[float]) -> tuple[float, float, float]:
        """Convert canonical Z-up world position [m] into ROS map position [m]."""
        x, y, z = (value - origin for value, origin in zip(_point(point, "point"), self.origin_m))
        cosine, sine = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        return _point((cosine * x + sine * y, -sine * x + cosine * y, z), "converted ROS position")

    def ros_to_world(self, point: Sequence[float]) -> tuple[float, float, float]:
        """Convert ROS map position [m] into canonical Z-up world position [m]."""
        x, y, z = _point(point, "point")
        cosine, sine = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        return _point(
            (
                cosine * x - sine * y + self.origin_m[0],
                sine * x + cosine * y + self.origin_m[1],
                z + self.origin_m[2],
            ),
            "converted world position",
        )

    def stage_to_ros(self, point: Sequence[float]) -> tuple[float, float, float]:
        """Convert already-composed USD world position [stage unit] to ROS [m]."""
        return self.world_to_ros(self.stage_to_world(point))

    def ros_vector_to_world(self, vector: Sequence[float]) -> tuple[float, float, float]:
        """Rotate a ROS map linear vector [m/s] into canonical Z-up world axes."""
        x, y, z = _point(vector, "vector")
        cosine, sine = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        return _point((cosine * x - sine * y, sine * x + cosine * y, z), "converted world vector")

    def ros_to_stage(self, point: Sequence[float]) -> tuple[float, float, float]:
        """Convert ROS map position [m] to USD world position [stage unit]."""
        return self.world_to_stage(self.ros_to_world(point))

    def stage_to_world(self, point: Sequence[float]) -> tuple[float, float, float]:
        """Convert composed USD world position [stage unit] to canonical Z-up [m].

        This conversion is independent of the ROS map origin and heading.
        """
        x, y, z = (value * self.meters_per_unit for value in _point(point, "point"))
        return _point((x, -z, y) if self.up_axis == "Y" else (x, y, z), "converted world position")

    def world_to_stage(self, point: Sequence[float]) -> tuple[float, float, float]:
        """Convert canonical Z-up world position [m] to USD world [stage unit]."""
        x, y, z = _point(point, "point")
        world = (x, z, -y) if self.up_axis == "Y" else (x, y, z)
        return _point(tuple(value / self.meters_per_unit for value in world), "converted stage position")

    def stage_vector_to_ros(self, vector: Sequence[float]) -> tuple[float, float, float]:
        """Convert a world linear vector [stage unit/s] to ROS [m/s], without translation."""
        x, y, z = (value * self.meters_per_unit for value in _point(vector, "vector"))
        if self.up_axis == "Y":
            x, y, z = x, -z, y
        cosine, sine = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        return _point((cosine * x + sine * y, -sine * x + cosine * y, z), "converted ROS vector")

    def ros_yaw_to_stage_quaternion(self, yaw: float) -> tuple[float, float, float, float]:
        """Return stage-world XYZW orientation for a canonical +X-forward, +Z-up body.

        Args:
            yaw: Body heading in ROS map coordinates [rad].

        The orientation includes the Y-up basis conversion when required. Asset
        visual-forward corrections belong on a child, not in this quaternion.
        """
        if isinstance(yaw, bool) or not isinstance(yaw, Real) or not math.isfinite(yaw):
            raise ValueError("yaw must be finite [rad].")
        half = (self.yaw_rad + yaw) / 2
        sine, cosine = math.sin(half), math.cos(half)
        if self.up_axis == "Z":
            return 0.0, 0.0, sine, cosine
        scale = math.sqrt(0.5)
        return -scale * cosine, scale * sine, scale * sine, scale * cosine
