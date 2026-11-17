# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Immutable scene, pedestrian, robot and session configuration types."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from ._validation import _finite_number
from .drl import DrlPolicySettings
from .navigation import NavigationConfig
from .recording import RecordingSettings as RecordingSettings
from .resources import DEFAULT_MOTION_LIBRARY
from .scenarios.models import PedestrianPlan


@dataclass(frozen=True)
class SceneConfig:
    """Optional base USD override and layers expressed in the destination frame.

    Authored length/mass units and up-axis in additional layers must match the
    environment. Overlays without authored metrics use its coordinate frame.
    """

    asset_path: Path | None = None
    layers: tuple[Path, ...] = ()


@dataclass(frozen=True)
class PedestrianConfig:
    """Select actors for the ROS 2 session.

    config_path replaces characters using a project YAML scenario or TOML wrapper.
    """

    enabled: bool = True
    config_path: Path | None = None

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("pedestrians.enabled must be a boolean.")


@dataclass(frozen=True)
class RobotConfig:
    """Load a robot reference or a composed robot layer.

    Attributes:
        asset_path: Robot USD with the destination's length/mass units and up-axis;
            its default prim is referenced unless overridden.
        layer_path: Alternative USD layer with the robot already placed at prim_path.
        prim_path: Unique destination prim in the simulation.
        source_prim_path: Optional model prim inside asset_path.
        position: Optional initial translation [m] in the destination parent's frame.
        yaw_degrees: Optional initial heading [deg] about the scene up-axis.
    """

    asset_path: Path | None = None
    layer_path: Path | None = None
    prim_path: str = "/World/Robot"
    source_prim_path: str | None = None
    position: tuple[float, float, float] | None = None
    yaw_degrees: float | None = None

    def __post_init__(self) -> None:
        if (self.asset_path is None) == (self.layer_path is None):
            raise ValueError("A robot requires exactly one of asset_path or layer_path.")
        _prim_path(self.prim_path)
        if self.source_prim_path is not None:
            _prim_path(self.source_prim_path)
            if self.asset_path is None:
                raise ValueError("source_prim_path requires asset_path.")
        if self.position is not None:
            if (
                not isinstance(self.position, (tuple, list))
                or len(self.position) != 3
                or not all(_finite_number(value) for value in self.position)
            ):
                raise ValueError("Robot position must contain three finite coordinates [m].")
            object.__setattr__(self, "position", tuple(self.position))
        if self.yaw_degrees is not None and not _finite_number(self.yaw_degrees):
            raise ValueError("Robot yaw_degrees must be finite [deg].")


@dataclass(frozen=True)
class SimulationSettings:
    """Application, playback, and optional physics overrides.

    Attributes:
        headless: Whether to hide the window.
        duration: Auto-exit threshold on the simulation timeline [s], not wall time;
            zero waits for exit.
        start_paused: Whether to wait for a play command after setup.
        width: Render width in pixels.
        height: Render height in pixels.
        physics_hz: Optional physics step frequency [Hz]; None preserves the scene.
        solver: Optional PhysX solver, PGS or TGS; None preserves the scene.
        seed: Optional random seed for character selection and navigation.
        extensions: Extra SDK extensions to enable before loading the scene.
        console: Enable terminal playback controls.
        report_path: Optional final status JSON output; TOML paths are file-relative.
        render_viewport: Update the main viewport. False requires headless mode
            and recording disabled; native simulation continues to advance.
    """

    headless: bool = False
    duration: float = 0.0
    start_paused: bool = False
    width: int = 1280
    height: int = 720
    physics_hz: int | None = None
    solver: str | None = None
    seed: int | None = None
    extensions: tuple[str, ...] = ()
    console: bool = False
    report_path: Path | None = None
    render_viewport: bool = True

    def __post_init__(self) -> None:
        if any(
            type(value) is not bool for value in (self.headless, self.start_paused, self.console, self.render_viewport)
        ):
            raise ValueError("headless, start_paused, console and render_viewport must be booleans.")
        if not self.render_viewport and not self.headless:
            raise ValueError("render_viewport=false requires headless=true.")
        if self.report_path is not None and not isinstance(self.report_path, Path):
            raise ValueError("report_path must be a filesystem Path.")
        if not _finite_number(self.duration) or self.duration < 0:
            raise ValueError("duration must be finite and non-negative [s].")
        for name in ("width", "height", "physics_hz"):
            value = getattr(self, name)
            if (name != "physics_hz" or value is not None) and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer.")
        if self.solver not in (None, "PGS", "TGS"):
            raise ValueError("solver must be PGS or TGS.")
        if self.seed is not None and (type(self.seed) is not int or not 0 <= self.seed <= 0xFFFFFFFF):
            raise ValueError("seed must be a 32-bit unsigned integer.")
        if not isinstance(self.extensions, (tuple, list)) or not all(
            isinstance(name, str) and name for name in self.extensions
        ):
            raise ValueError("extensions must be a list of extension names.")
        object.__setattr__(self, "extensions", tuple(self.extensions))


@dataclass(frozen=True)
class SimulationConfig:
    """Complete ROS 2 session with explicitly selected local scene and actor inputs."""

    scene: SceneConfig = field(default_factory=SceneConfig)
    pedestrians: PedestrianConfig = field(default_factory=PedestrianConfig)
    robots: tuple[RobotConfig, ...] = ()
    settings: SimulationSettings = field(default_factory=SimulationSettings)
    recording: RecordingSettings = field(default_factory=RecordingSettings)
    pedestrian_plans: tuple[PedestrianPlan, ...] = ()
    pedestrian_assets: tuple[tuple[str, Path], ...] = ()
    motion_library: str = DEFAULT_MOTION_LIBRARY
    # Loader provenance survives replace() so status output cannot overwrite a
    # consumed YAML or wrapper whose contents are now held only in memory.
    _source_paths: tuple[Path, ...] = field(default=(), repr=False, compare=False, kw_only=True)
    navigation: NavigationConfig = field(default_factory=NavigationConfig)
    drl: DrlPolicySettings | None = None

    def validate(self) -> None:
        """Reject missing effective inputs and overlapping robot destinations before starting Kit."""
        if self.scene.asset_path is None:
            raise ValueError("Provide an existing local scene.asset_path.")
        if self.drl is not None and self.drl.enabled:
            if len(self.robots) != 1:
                raise ValueError("DRL policy requires exactly one robot.")
            if self.drl.goal_xy_m is None:
                raise ValueError("DRL policy requires goal_xy_m.")
            self.drl.validate_files()
        if self.recording.enabled:
            if not self.settings.render_viewport:
                raise ValueError("Recording requires render_viewport=true.")
            frame_count = self.settings.duration * self.recording.fps
            if not _finite_number(frame_count):
                raise ValueError("Recording duration and fps must produce a finite frame count.")
            if round(frame_count) < 1:
                raise ValueError("Recording requires a positive duration producing at least one frame.")
            if self.settings.width % 2 or self.settings.height % 2:
                raise ValueError("MP4 recording width and height must be even integers.")
            if self.settings.start_paused or self.settings.console:
                raise ValueError("Recording requires automatic playback without console controls.")
        paths = [self.scene.asset_path, *self.scene.layers]
        paths.extend(robot.asset_path or robot.layer_path for robot in self.robots)
        for path in paths:
            if path is not None and not path.is_file():
                raise ValueError(f"Input file not found: {path}")
        from .population import resolve_population

        population = resolve_population(self)
        if population.config is not None:
            from .navigation import load_ros2_navigation_settings
            from .scenarios.compilation import _validate_scenario_resources

            navigation = load_ros2_navigation_settings(self.navigation.settings_path)
            _validate_scenario_resources(population.config, asset_root=navigation.asset_root)
        destinations = [robot.prim_path for robot in self.robots]
        for index, path in enumerate(destinations):
            if any(
                path == other or path.startswith(other + "/") or other.startswith(path + "/")
                for other in destinations[:index]
            ):
                raise ValueError(f"Overlapping robot prim_path: {path}")


def _with_ros2_navigation_settings(config: SimulationConfig, path: Path) -> SimulationConfig:
    """Select a launch-local ROS settings file before validating effective inputs."""
    from .paths import resolve_config_path

    navigation = replace(config.navigation, settings_path=resolve_config_path(path))
    return replace(config, navigation=navigation)


def _prim_path(value: str) -> None:
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or not all(part.isidentifier() for part in value[1:].split("/"))
    ):
        raise ValueError(f"Invalid absolute prim_path: {value!r}")
