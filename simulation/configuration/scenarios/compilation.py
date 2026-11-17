# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resolve ROS 2 actor resources and retain their execution plans."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ..models import RobotConfig, SceneConfig, SimulationConfig, SimulationSettings
from ..paths import resolve_config_path
from ..recording import RecordingSettings
from ..resources import DEFAULT_MOTION_LIBRARY as DEFAULT_MOTION_LIBRARY
from .assets import find_character_assets
from .models import PedestrianPlan


def compile_scenario_config(
    *,
    scene: SceneConfig,
    plans: tuple[PedestrianPlan, ...],
    assets: Path | None = None,
    motion_library: Path | str = DEFAULT_MOTION_LIBRARY,
    settings: SimulationSettings | None = None,
    robots: tuple[RobotConfig, ...] = (),
    recording: RecordingSettings | None = None,
) -> SimulationConfig:
    """Build a ROS 2 actor population and retain its explicit execution plans.

    Args:
        scene: Existing scene asset and optional USD layers.
        plans: One validated plan per actor; actor names must be unique.
        assets: Explicit shared character USD, directory, or catalog for plans without
            asset_path. None requires an asset_path on every plan.
        motion_library: Local motion library or the navigation asset-root default token.
        settings: Simulation playback and application configuration.
        robots: Optional robot assets/layers.
        recording: Optional recording configuration.

    Returns:
        Session configuration with resolved actor inputs and no SDK imports.
        Consumed local input paths are expanded and made absolute at compilation,
        without modifying the supplied objects. The default motion-library token is
        retained only when passed as that exact string.

    Raises:
        ValueError: If selected local inputs are missing or a requested gender
            has no matching character asset.
    """
    config = _build_scenario_config(
        scene=scene,
        plans=plans,
        assets=assets,
        motion_library=motion_library,
        settings=settings,
        robots=robots,
        recording=recording,
    )
    config.validate()
    return config


def _build_scenario_config(
    *,
    scene: SceneConfig,
    plans: tuple[PedestrianPlan, ...],
    assets: Path | None = None,
    motion_library: Path | str = DEFAULT_MOTION_LIBRARY,
    settings: SimulationSettings | None = None,
    robots: tuple[RobotConfig, ...] = (),
    recording: RecordingSettings | None = None,
) -> SimulationConfig:
    """Compile authored plans before final profile overrides select their inputs."""
    if scene.asset_path is None:
        raise ValueError("A pedestrian scenario requires scene.asset_path.")
    if not plans or not all(isinstance(plan, PedestrianPlan) for plan in plans):
        raise ValueError("A pedestrian scenario requires at least one PedestrianPlan.")
    if len({plan.name for plan in plans}) != len(plans):
        raise ValueError("Pedestrian names must be unique.")
    positions = [plan.position for plan in plans if plan.position is not None]
    if len(set(positions)) != len(positions):
        raise ValueError("Pedestrians must not have identical explicit initial positions.")
    scene = replace(
        scene,
        asset_path=resolve_config_path(scene.asset_path),
        layers=tuple(resolve_config_path(path) for path in scene.layers),
    )
    robots = tuple(
        replace(
            robot,
            asset_path=resolve_config_path(robot.asset_path) if robot.asset_path is not None else None,
            layer_path=resolve_config_path(robot.layer_path) if robot.layer_path is not None else None,
        )
        for robot in robots
    )
    motion_library = (
        DEFAULT_MOTION_LIBRARY
        if isinstance(motion_library, str) and motion_library == DEFAULT_MOTION_LIBRARY
        else resolve_config_path(Path(motion_library))
    )
    selected = []
    for plan in plans:
        source = plan.asset_path or assets
        if source is None:
            raise ValueError(f"pedestrians.assets is required when {plan.name} has no explicit asset_path.")
        selected.append((plan.name, resolve_config_path(source)))
    return SimulationConfig(
        scene=scene,
        robots=robots,
        settings=settings or SimulationSettings(),
        recording=recording or RecordingSettings(),
        pedestrian_plans=tuple(plans),
        pedestrian_assets=tuple(selected),
        motion_library=str(motion_library),
    )


def _validate_scenario_resources(config: SimulationConfig, *, asset_root: str | None) -> None:
    """Check the selected project population without validating unused scene inputs."""
    if not config.pedestrian_plans:
        raise ValueError("Enabled pedestrians require at least one pedestrian plan.")
    assets = dict(config.pedestrian_assets)
    names = {plan.name for plan in config.pedestrian_plans}
    if (
        len(names) != len(config.pedestrian_plans)
        or len(assets) != len(config.pedestrian_assets)
        or assets.keys() != names
    ):
        raise ValueError("Every unique pedestrian plan requires exactly one asset source.")
    motion_library = config.motion_library
    if motion_library != DEFAULT_MOTION_LIBRARY and not Path(motion_library).expanduser().is_file():
        raise ValueError(f"Resource not found for pedestrians.motion_library: {motion_library}")
    checked_assets = set()
    for plan in config.pedestrian_plans:
        source = assets[plan.name]
        key = (source, plan.gender)
        if key in checked_assets:
            continue
        if not (source.is_file() or source.is_dir()):
            field = (
                f"pedestrians.actors.{plan.name}.asset_path" if plan.asset_path is not None else "pedestrians.assets"
            )
            raise ValueError(f"Resource not found for {field}: {source}")
        try:
            find_character_assets(source, plan.gender, asset_root=asset_root)
        except ValueError as error:
            field = (
                f"pedestrians.actors.{plan.name}.gender={plan.gender}"
                if plan.gender != "any"
                else f"pedestrians.actors.{plan.name}.asset_path"
                if plan.asset_path is not None
                else "pedestrians.assets"
            )
            raise ValueError(f"{field}: {error}") from error
        checked_assets.add(key)
