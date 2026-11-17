# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Create editable project-schema scenes without starting Isaac Sim."""

import json
import os
import shutil
from pathlib import Path
from string import Template
from typing import Literal

from ..configuration.navigation import NavigationConfig, load_ros2_navigation_settings
from ..configuration.paths import CONFIG_ROOT, resolve_config_path
from ..configuration.scenarios.assets import find_character_assets


def create_simulation_project(
    directory: Path,
    *,
    scene_asset: Path,
    pedestrian_count: int = 5,
    behavior: Literal["wander", "standing"] = "wander",
    character_assets: Path | None = None,
    configuration_format: Literal["yaml"] = "yaml",
    navigation_backend: Literal["ros2"] = "ros2",
    navigation_settings: Path | None = None,
    waypoint_frame: Literal["world_m", "stage_world"] = "world_m",
) -> Path:
    """Create an editable ROS 2 simulation project.

    Args:
        directory: New project directory; existing directories are never overwritten.
        scene_asset: Existing environment USD with solid collision geometry and floor support.
        pedestrian_count: Number of pedestrians to spawn.
        behavior: Initial behavior; users edit the project YAML afterward.
        character_assets: Existing USD/USDA, character directory, or YAML asset catalog.
        configuration_format: Only ``yaml`` is supported. The former ``native`` value
            is rejected with migration guidance.
        navigation_backend: Supported controller, ROS 2.
        navigation_settings: ROS 2 map/controller YAML. Omission creates a local copy.
        waypoint_frame: Input positions use canonical world [m] or composed USD
            world [stage unit]. Speeds and arrival tolerances remain SI.

    Returns:
        Absolute path to the generated simulation.yaml profile.
        Scene, character and explicit settings references are file-relative
        when possible; inputs on different drives retain absolute paths.

    Raises:
        ValueError: If the count, preset or source assets are invalid.
        FileExistsError: If the destination already exists.
    """
    import yaml

    directory = resolve_config_path(directory)
    scene_asset = resolve_config_path(scene_asset)
    if character_assets is None:
        raise ValueError("Provide explicit character_assets; no repository asset pool is inferred.")
    character_assets = resolve_config_path(character_assets)
    if type(pedestrian_count) is not int or pedestrian_count <= 0:
        raise ValueError("pedestrian_count must be a positive integer.")
    NavigationConfig(backend=navigation_backend, waypoint_frame=waypoint_frame)
    if behavior not in ("wander", "standing"):
        raise ValueError("behavior must be wander or standing.")
    if configuration_format != "yaml":
        raise ValueError(
            "Only project YAML is supported; use configuration_format='yaml' "
            "and schema_version: 1 instead of native IRA/Actor SDG configuration."
        )
    if not scene_asset.is_file() or not character_assets.exists():
        raise ValueError("Provide an existing scene asset and character asset source.")
    navigation_settings = resolve_config_path(navigation_settings) if navigation_settings is not None else None
    ros2_settings = load_ros2_navigation_settings(navigation_settings)
    try:
        find_character_assets(character_assets, asset_root=ros2_settings.asset_root)
    except ValueError as error:
        raise ValueError(f"character_assets must contain a discoverable USD/USDA entry: {error}") from error
    values = {
        "scene_asset": json.dumps(_project_asset_reference(scene_asset, directory), ensure_ascii=False),
        "characters": json.dumps(_project_asset_reference(character_assets, directory), ensure_ascii=False),
        "navigation": (
            f"navigation:\n  backend: ros2\n  waypoint_frame: {waypoint_frame}\n  settings_path: "
            + json.dumps(
                _project_asset_reference(navigation_settings, directory)
                if navigation_settings is not None
                else "navigation_settings.yaml",
                ensure_ascii=False,
            )
        ),
        "count": str(pedestrian_count),
        "pedestrian_defaults": "  defaults:\n    behavior: stand" if behavior == "standing" else "",
    }
    templates = CONFIG_ROOT / "templates"
    profile_name = "simulation.yaml"
    files = {profile_name: Template((templates / "scenario.yaml.in").read_text()).substitute(values)}
    if navigation_settings is None:
        files["navigation_settings.yaml"] = yaml.safe_dump(ros2_settings.to_dict(), sort_keys=False)
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()
    try:
        for name, contents in files.items():
            (directory / name).write_text(contents)
    except OSError:
        shutil.rmtree(directory)
        raise
    return directory / profile_name


def _project_asset_reference(asset: Path, directory: Path) -> str:
    """Prefer relocatable references, retaining absolute paths across drives."""
    try:
        return os.path.relpath(asset, directory)
    except ValueError:
        return str(asset)
