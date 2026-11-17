# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate fresh recording destinations and encode captured frames."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from ..configuration._validation import _finite_number
from ..configuration.navigation import Ros2NavigationSettings, load_ros2_navigation_settings
from ..configuration.paths import (
    ROS2_NAVIGATION_SETTINGS_PATH,
    resolve_config_path,
    resolve_output_path,
)
from ..configuration.population import resolve_population
from ..configuration.resources import resolve_ros2_motion_library
from ..configuration.scenarios.assets import character_catalog_files

if TYPE_CHECKING:
    from ..configuration.models import SimulationConfig


def validate_recording_runtime(config: SimulationConfig) -> None:
    """Check recording inputs and tools without creating files or starting the SDK."""
    if not config.recording.enabled:
        return
    if config.recording.output_path is None:
        raise ValueError("Enabled recording requires recording.output_path.")
    if not _finite_number(config.settings.duration) or config.settings.duration <= 0:
        raise ValueError("Recording requires a positive finite simulation duration [s].")
    if not config.settings.render_viewport:
        raise ValueError("Recording requires render_viewport=true.")
    if config.settings.start_paused or config.settings.console:
        raise ValueError("Recording requires automatic playback without console controls.")
    navigation = (
        load_ros2_navigation_settings(config.navigation.settings_path)
        if config.pedestrians.enabled
        else Ros2NavigationSettings()
    )
    if config.recording.fps > navigation.control_hz or navigation.control_hz % config.recording.fps:
        raise ValueError(
            f"ROS 2 recording.fps ({config.recording.fps} Hz) must not exceed control_hz "
            f"({navigation.control_hz} Hz) and must divide it exactly."
        )
    frame_count = config.settings.duration * config.recording.fps
    if not _finite_number(frame_count):
        raise ValueError("Recording duration and fps must produce a finite frame count.")
    if round(frame_count) < 1:
        raise ValueError("Recording duration and fps must produce at least one frame.")
    if config.settings.width % 2 or config.settings.height % 2:
        raise ValueError("Recording width and height must be even integers.")
    output, frames, metadata = _recording_paths(config)
    if output.suffix.lower() != ".mp4":
        raise ValueError("Recording output_path must end in .mp4.")
    for parent in output.parents:
        if parent.exists() and not parent.is_dir():
            raise ValueError(f"Recording output parent is not a directory: {parent}")
    existing = [str(path) for path in (output, frames, metadata) if path.exists() or path.is_symlink()]
    if existing:
        raise ValueError(
            "Recording output, frames and metadata must use a fresh name; files are never overwritten. "
            f"Existing paths: {', '.join(existing)}"
        )
    if shutil.which("ffmpeg") is None:
        raise ValueError("Recording requires an existing ffmpeg executable; dependencies are not changed.")
    if shutil.which("ffprobe") is None:
        raise ValueError("ROS 2 recording requires an existing ffprobe executable to verify the encoded video.")


def _recording_paths(config: SimulationConfig) -> tuple[Path, Path, Path]:
    output = resolve_output_path(config.recording.output_path)
    return output, output.parent / (output.stem + "_frames"), output.with_suffix(".json")


def _validate_report_path(config: SimulationConfig, report_path: Path | None) -> None:
    """Reject final status destinations that would overwrite inputs or recording data."""
    if report_path is None:
        return
    supplied = resolve_output_path(report_path)
    # Keep the final entry for output checks and resolve aliases that write_text
    # would follow, including a report symlink to a not-yet-created recording.
    reports = {supplied, resolve_config_path(supplied)}
    inputs = [
        config.scene.asset_path,
        *config.scene.layers,
        config.pedestrians.config_path,
        config.navigation.settings_path or ROS2_NAVIGATION_SETTINGS_PATH,
        *config._source_paths,
    ]
    inputs.extend(robot.asset_path or robot.layer_path for robot in config.robots)
    if _overwrites_inputs(reports, inputs):
        raise ValueError("Simulation report_path must not overwrite an input configuration or asset file.")
    if config.recording.enabled and config.recording.output_path is not None:
        if any(
            report.is_relative_to(path) or path.is_relative_to(report)
            for report in reports
            for path in _recording_paths(config)
        ):
            raise ValueError("Simulation report_path must be separate from recording outputs and their path hierarchy.")
    if _overwrites_inputs(reports, _population_report_inputs(config)):
        raise ValueError("Simulation report_path must not overwrite an input configuration or asset file.")
    for report in reports:
        if report.is_dir():
            raise ValueError(f"Simulation report_path must name a file: {report}")
        for parent in report.parents:
            if parent.exists() and not parent.is_dir():
                raise ValueError(f"Simulation report_path parent is not a directory: {parent}")


def _overwrites_inputs(reports: set[Path], inputs: Iterable[Path | None]) -> bool:
    for path in inputs:
        if path is None:
            continue
        target = path.expanduser().resolve()
        for report in reports:
            if report == target or (target.is_dir() and report.is_relative_to(target)):
                return True
            if report.exists() and target.exists() and report.samefile(target):
                return True
    return False


def _population_report_inputs(config: SimulationConfig) -> list[Path]:
    """Protect the effective ROS population inputs without opening USD."""
    population = resolve_population(config).config
    if population is None:
        return []
    navigation = load_ros2_navigation_settings(config.navigation.settings_path)
    sources = dict.fromkeys(asset for _, asset in population.pedestrian_assets)
    return [
        *population._source_paths,
        *sources,
        *(file for source in sources for file in character_catalog_files(source, navigation.asset_root)),
        resolve_ros2_motion_library(population.motion_library, navigation.asset_root),
    ]


def _encode_video(config: SimulationConfig, frames: Path, output: Path) -> None:
    subprocess.run(
        [
            shutil.which("ffmpeg"),
            "-nostdin",
            "-n",
            "-loglevel",
            "error",
            "-framerate",
            str(config.recording.fps),
            "-i",
            str(frames / "frame_%06d.png"),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
        timeout=120,
    )
