# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Validated video configuration, independent of Kit and recording implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ._validation import _finite_number


@dataclass(frozen=True)
class RecordingSettings:
    """Optional simulation recording; paths are relative to the declaring YAML/TOML.

    Attributes:
        enabled: Observe the selected independent simulation and encode its rendered frames.
        output_path: New MP4 output; existing videos, metadata and frames are preserved.
        fps: Video sampling and playback frequency [Hz]. Actual simulation updates
            must be at least this frequent; ROS capture frequency must divide its
            control frequency exactly. USD time-code rate is not a capture limit.
        warmup_s: Unrecorded simulation settling interval [s].
        camera_eye_m: Optional camera position in canonical Z-up world coordinates [m].
        camera_target_m: Optional world look-at point [m], in the same convention
            as camera_eye_m. Neither ROS field is expressed in the map frame.
        max_stationary_s: Optional maximum continuous low-speed time [s], or None.
        camera_view: Automatic camera direction. ``overhead`` cannot be combined
            with explicit camera coordinates.
    """

    enabled: bool = False
    output_path: Path | None = None
    fps: int = 12
    warmup_s: float = 0.0
    camera_eye_m: tuple[float, float, float] | None = None
    camera_target_m: tuple[float, float, float] | None = None
    max_stationary_s: float | None = None
    camera_view: Literal["oblique", "overhead"] = "oblique"

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("recording.enabled must be a boolean.")
        if self.output_path is not None and (
            not isinstance(self.output_path, Path) or self.output_path.suffix.lower() != ".mp4"
        ):
            raise ValueError("recording.output_path must be an MP4 filesystem Path.")
        if self.enabled and self.output_path is None:
            raise ValueError("recording.output_path is required when recording is enabled.")
        if type(self.fps) is not int or not 1 <= self.fps <= 60:
            raise ValueError("recording.fps must be an integer from 1 to 60 [Hz].")
        if not _finite_number(self.warmup_s) or self.warmup_s < 0:
            raise ValueError("recording.warmup_s must be finite and nonnegative [s].")
        if self.max_stationary_s is not None and (
            not _finite_number(self.max_stationary_s) or self.max_stationary_s <= 0
        ):
            raise ValueError("recording.max_stationary_s must be positive and finite [s].")
        if (self.camera_eye_m is None) != (self.camera_target_m is None):
            raise ValueError("Provide both recording camera_eye_m and camera_target_m, or neither.")
        if self.camera_view not in ("oblique", "overhead"):
            raise ValueError("recording.camera_view must be oblique or overhead.")
        if self.camera_view == "overhead" and self.camera_eye_m is not None:
            raise ValueError(
                "recording.camera_view=overhead uses automatic framing without explicit camera coordinates."
            )
        for name in ("camera_eye_m", "camera_target_m"):
            point = getattr(self, name)
            if point is not None:
                if (
                    not isinstance(point, (tuple, list))
                    or len(point) != 3
                    or not all(_finite_number(value) for value in point)
                ):
                    raise ValueError(f"recording.{name} must contain three finite world coordinates [m].")
                object.__setattr__(self, name, tuple(point))
        if self.camera_eye_m is not None and self.camera_eye_m == self.camera_target_m:
            raise ValueError("Recording camera position and target must be distinct.")
