# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configure ROS 2 navigation and validate its SI settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal

from ._validation import _finite_number
from .paths import ROS2_NAVIGATION_SETTINGS_PATH


@dataclass(frozen=True)
class NavigationConfig:
    """Navigation selection and the explicit input pedestrian position frame.

    ``world_m`` retains canonical Z-up world positions [m]. ROS 2 callers may
    select ``stage_world`` for composed Isaac scene-world positions [stage unit],
    using the loaded stage's units and up axis. This affects initial/waypoint
    positions only; speeds [m/s], arrival tolerances [m] and times [s] stay SI.
    Waypoint headings [deg] use the selected frame's up axis and +X zero
    direction. Initial yaw retains its asset-relative facing convention.
    """

    backend: Literal["ros2"] = "ros2"
    settings_path: Path | None = None
    waypoint_frame: Literal["world_m", "stage_world"] = "world_m"

    def __post_init__(self) -> None:
        if self.backend != "ros2":
            raise ValueError("Only navigation.backend=ros2 is supported. Migrate IRA profiles to ROS 2.")
        if self.settings_path is not None and not isinstance(self.settings_path, Path):
            raise ValueError("navigation.settings_path must be a local Path.")
        if self.waypoint_frame not in ("world_m", "stage_world"):
            raise ValueError("navigation.waypoint_frame must be world_m or stage_world.")


@dataclass(frozen=True)
class Ros2NavigationSettings:
    """Occupancy mapping, human dimensions [m], ROS timing [s] and angular rate [rad/s].

    ``origin_m`` is the ROS map origin in canonical Z-up world coordinates [m];
    ``yaw_rad`` rotates the map axes relative to that world [rad]. ``bounds_m``
    and ``floor_height_m`` use ROS map coordinates [m], independently of the
    input waypoint frame. Scene unit changes never rescale these SI settings.
    ``auto_return_heading`` preserves copied waypoint headings by default;
    ``follow_route`` lets generated return legs face their direction of travel.
    """

    resolution_m: float = 0.05
    bounds_m: tuple[float, float, float, float] | None = None
    floor_height_m: float = 0.0
    floor_tolerance_m: float = 0.12
    floor_clearance_m: float = 0.08
    max_slope_deg: float = 15.0
    body_radius_m: float = 0.25
    body_height_m: float = 1.8
    map_clearance_m: float = 0.03
    pair_clearance_m: float = 0.25
    max_cells: int = 250000
    origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_rad: float = 0.0
    dynamic_roots: tuple[str, ...] = ()
    control_hz: int = 60
    max_turn_rate_radps: float = 2.5
    time_horizon_s: float = 2.5
    obstacle_time_horizon_s: float = 0.5
    neighbor_distance_m: float = 4.0
    relaxation_time_s: float = 0.4
    asset_root: str | None = None
    ros_setup_path: str = "/opt/ros/jazzy/setup.bash"
    ros_namespace: str = "/pedestrian_sim"
    ros_domain_id: int = 37
    request_timeout_s: float = 5.0
    auto_return_heading: Literal["preserve", "follow_route"] = "preserve"
    spawn_tolerance_m: float = 0.0
    goal_projection_tolerance_m: float | None = None
    max_yield_distance_m: float = 0.4
    preferred_path_clearance_m: float = 0.0

    def __post_init__(self) -> None:
        positive = (
            "resolution_m",
            "floor_tolerance_m",
            "body_radius_m",
            "body_height_m",
            "max_turn_rate_radps",
            "time_horizon_s",
            "obstacle_time_horizon_s",
            "neighbor_distance_m",
            "relaxation_time_s",
            "request_timeout_s",
            "max_yield_distance_m",
        )
        for name in positive:
            value = getattr(self, name)
            if not _finite_number(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in (
            "floor_height_m",
            "yaw_rad",
            "floor_clearance_m",
            "map_clearance_m",
            "pair_clearance_m",
            "max_slope_deg",
        ):
            value = getattr(self, name)
            if not _finite_number(value):
                raise ValueError(f"{name} must be finite.")
        if min(self.floor_clearance_m, self.map_clearance_m, self.pair_clearance_m) < 0:
            raise ValueError("Map, floor and pair clearances must be nonnegative [m].")
        if not _finite_number(self.spawn_tolerance_m) or self.spawn_tolerance_m < 0:
            raise ValueError("spawn_tolerance_m must be finite and nonnegative [m].")
        if not _finite_number(self.preferred_path_clearance_m) or self.preferred_path_clearance_m < 0:
            raise ValueError("preferred_path_clearance_m must be finite and nonnegative [m].")
        if self.goal_projection_tolerance_m is not None and (
            not _finite_number(self.goal_projection_tolerance_m) or self.goal_projection_tolerance_m <= 0
        ):
            raise ValueError("goal_projection_tolerance_m must be null or finite and positive [m].")
        if self.floor_clearance_m >= self.body_height_m or not 0 <= self.max_slope_deg < 90:
            raise ValueError("Invalid vertical mapping band or maximum slope.")
        if self.body_height_m < 2 * self.body_radius_m:
            raise ValueError("body_height_m must be at least twice body_radius_m for the physical capsule.")
        if not _finite_number(2 * self.body_radius_m + self.pair_clearance_m):
            raise ValueError("The body diameter plus pair_clearance_m must be finite [m].")
        if not _finite_number(self.body_radius_m + self.map_clearance_m + self.preferred_path_clearance_m):
            raise ValueError("The body radius plus map_clearance_m and preferred_path_clearance_m must be finite [m].")
        for name in ("control_hz", "max_cells"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if type(self.ros_domain_id) is not int or not 0 <= self.ros_domain_id <= 232:
            raise ValueError("ros_domain_id must be an integer in [0, 232].")
        for name, length in (("origin_m", 3), ("bounds_m", 4)):
            value = getattr(self, name)
            if value is None and name == "bounds_m":
                continue
            if (
                not isinstance(value, (tuple, list))
                or len(value) != length
                or not all(_finite_number(x) for x in value)
            ):
                raise ValueError(f"{name} must contain {length} finite coordinates [m].")
            object.__setattr__(self, name, tuple(value))
        if self.bounds_m is not None and (self.bounds_m[0] >= self.bounds_m[2] or self.bounds_m[1] >= self.bounds_m[3]):
            raise ValueError("bounds_m must be [xmin, ymin, xmax, ymax] with positive extent.")
        if not isinstance(self.dynamic_roots, (tuple, list)) or any(
            not isinstance(root, str)
            or not root.startswith("/")
            or (root != "/" and not all(part.isidentifier() for part in root[1:].split("/")))
            for root in self.dynamic_roots
        ):
            raise ValueError("dynamic_roots must contain absolute USD prim paths.")
        object.__setattr__(self, "dynamic_roots", tuple(self.dynamic_roots))
        if (
            not isinstance(self.ros_namespace, str)
            or not self.ros_namespace.startswith("/")
            or self.ros_namespace != "/"
            and any(not p.isascii() or not p.isidentifier() for p in self.ros_namespace.rstrip("/")[1:].split("/"))
        ):
            raise ValueError("ros_namespace must be an absolute ROS 2 namespace with valid name tokens.")
        if not isinstance(self.ros_setup_path, str) or not Path(self.ros_setup_path).is_absolute():
            raise ValueError("ros_setup_path must be an absolute local setup script path.")
        if self.asset_root is not None and (not isinstance(self.asset_root, str) or not self.asset_root.strip()):
            raise ValueError("asset_root must be a nonempty path or URI.")
        if type(self.auto_return_heading) is not str or self.auto_return_heading not in ("preserve", "follow_route"):
            raise ValueError("auto_return_heading must be preserve or follow_route.")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible configuration with SI physical quantities."""
        return asdict(self)


def load_ros2_navigation_settings(path: Path | None = None) -> Ros2NavigationSettings:
    """Read strict ROS 2 navigation YAML without importing ROS or the simulator."""
    from .loading import load_yaml_mapping

    source = path or ROS2_NAVIGATION_SETTINGS_PATH
    source = source.expanduser().resolve()
    data = load_yaml_mapping(source)
    unknown = data.keys() - {f.name for f in fields(Ros2NavigationSettings)}
    if unknown:
        raise ValueError(f"{source}: Unknown ROS 2 settings: {', '.join(sorted(unknown))}")
    try:
        return Ros2NavigationSettings(**data)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source}: {error}") from error
