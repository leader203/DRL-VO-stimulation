# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Immutable input plans; position coordinates follow navigation.waypoint_frame."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PedestrianGender = Literal["any", "male", "female"]


@dataclass(frozen=True)
class Waypoint:
    """A navigation target and the body pose to hold after arrival.

    Attributes:
        position: World target [m] for world_m, or [stage unit] for stage_world.
        wait: Dwell time after arrival and any requested turn [s].
        yaw: Optional body heading around the selected world up axis [deg].
            None preserves the arrival heading.
    """

    position: tuple[float, float, float]
    wait: float = 0.0
    yaw: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _point(self.position, "waypoint.position"))
        _number(self.wait, "waypoint.wait", zero=True)
        if self.yaw is not None:
            _number(self.yaw, "waypoint.yaw", negative=True)


@dataclass(frozen=True)
class PedestrianPlan:
    """One pedestrian's schedule, with an explicit input position convention.

    Attributes:
        name: Unique actor/group identifier.
        position: Initial world position [m or stage unit, depending on
            navigation.waypoint_frame], or None for safe backend sampling.
        yaw: Initial rotation around the selected up axis, relative to the asset's authored facing [deg].
        behavior: ``walk``, ``stand`` or ``wander``.
        speed: Requested walking speed [m/s].
        start_time: Execution start on the simulation timeline [s].
        repeat: Repeat a closed route, or return along an open route to its initial position.
        waypoints: Navigation targets with independent dwell times [s] and
            optional arrival body headings [deg].
        arrival_tolerance: Maximum target position error for arrival [m].
        asset_path: Optional local character file/directory overriding the shared pool,
            normalized to an absolute path when the plan is constructed.
        gender: Optional asset-label filter; ``any`` preserves unrestricted selection.

    Open repeated routes reverse their waypoints on the return journey. A route
    ending at its declared initial position is already closed and repeats only
    its explicit targets, preserving the final heading and dwell. Scene setup
    converts the declared position frame to canonical metric coordinates.
    """

    name: str
    position: tuple[float, float, float] | None = None
    yaw: float = 0.0
    behavior: str = "wander"
    speed: float = 1.0
    start_time: float = 0.0
    repeat: bool = False
    waypoints: tuple[Waypoint, ...] = ()
    arrival_tolerance: float = 0.15
    asset_path: Path | None = None
    gender: PedestrianGender = "any"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isascii() or not self.name.isidentifier():
            raise ValueError("Pedestrian name must be an ASCII identifier, such as walker_01.")
        _gender(self.gender, f"{self.name}.gender")
        if self.asset_path is not None and not isinstance(self.asset_path, Path):
            raise ValueError(f"{self.name}.asset_path must be a Path or None.")
        if self.asset_path is not None:
            object.__setattr__(self, "asset_path", self.asset_path.expanduser().resolve())
        if self.position is not None:
            object.__setattr__(self, "position", _point(self.position, f"{self.name}.position"))
        _number(self.yaw, f"{self.name}.yaw", negative=True)
        if self.behavior not in ("walk", "stand", "wander"):
            raise ValueError(f"{self.name}.behavior must be walk, stand or wander.")
        _number(self.speed, f"{self.name}.speed")
        _number(self.start_time, f"{self.name}.start_time", zero=True)
        _number(self.arrival_tolerance, f"{self.name}.arrival_tolerance")
        if type(self.repeat) is not bool:
            raise ValueError(f"{self.name}.repeat must be a boolean.")
        if not isinstance(self.waypoints, (tuple, list)) or not all(
            isinstance(point, Waypoint) for point in self.waypoints
        ):
            raise ValueError(f"{self.name}.waypoints must contain Waypoint objects.")
        object.__setattr__(self, "waypoints", tuple(self.waypoints))
        if self.behavior == "walk":
            if self.position is None or not self.waypoints:
                raise ValueError(f"{self.name}: walk requires an initial position and at least one waypoint.")
        elif self.waypoints or self.repeat:
            raise ValueError(f"{self.name}: waypoints and repeat are only supported for walk behavior.")
        positions = [point.position for point in self.waypoints]
        if len(set(positions)) != len(positions):
            raise ValueError(f"{self.name}.waypoints contains duplicate positions.")
        if (
            self.position is not None
            and self.position in positions
            and (len(positions) < 2 or self.position in positions[:-1])
        ):
            raise ValueError(f"{self.name}: the initial position may only be the final waypoint after another target.")

    def to_dict(self) -> dict[str, object]:
        """Return an independent JSON-compatible copy of the requested plan.

        Positions are [m], speed is [m/s], times are [s], and headings are [deg].
        A null position awaits NavMesh sampling; a null asset path inherits the
        shared pool. These are configuration values, not observed runtime state.
        """
        return {
            "name": self.name,
            "position": list(self.position) if self.position is not None else None,
            "yaw": self.yaw,
            "behavior": self.behavior,
            "speed": self.speed,
            "start_time": self.start_time,
            "repeat": self.repeat,
            "waypoints": [
                {"position": list(point.position), "wait": point.wait, "yaw": point.yaw} for point in self.waypoints
            ],
            "arrival_tolerance": self.arrival_tolerance,
            "asset_path": str(self.asset_path) if self.asset_path is not None else None,
            "gender": self.gender,
        }


def _gender(value: object, name: str) -> None:
    if not isinstance(value, str) or value not in ("any", "male", "female"):
        raise ValueError(f"{name} must be any, male or female.")


def _point(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain three finite world coordinates [m].")
    for coordinate in value:
        _number(coordinate, name, negative=True)
    return tuple(float(coordinate) for coordinate in value)


def _number(value: object, name: str, *, zero: bool = False, negative: bool = False) -> None:
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite number.")
    if not negative and (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be {'non-negative' if zero else 'positive'}.")
