# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load launcher profiles without importing Isaac Sim."""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import tomllib

from .models import PedestrianConfig, RobotConfig, SceneConfig, SimulationConfig, SimulationSettings
from .navigation import NavigationConfig
from .paths import resolve_config_reference, resolve_simulation_profile
from .recording import RecordingSettings


def load_yaml_mapping(path: Path) -> dict[str, object]:
    """Read a safe YAML mapping, rejecting duplicate keys with source locations.

    Args:
        path: User configuration file.

    Returns:
        Mapping with string keys, without importing the simulator.

    Raises:
        ValueError: If YAML is invalid, contains duplicate keys or is not a mapping.
    """
    import yaml

    class UniqueKeyLoader(yaml.SafeLoader):
        def construct_mapping(self, node, deep=False):
            keys = set()
            for key_node, _ in node.value:
                key = self.construct_object(key_node, deep=deep)
                if not isinstance(key, str):
                    raise yaml.constructor.ConstructorError(
                        None, None, "Configuration keys must be strings", key_node.start_mark
                    )
                if key in keys:
                    raise yaml.constructor.ConstructorError(
                        None, None, f"Duplicate configuration key: {key}", key_node.start_mark
                    )
                keys.add(key)
            return super().construct_mapping(node, deep=deep)

    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in {path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return data


def load_simulation_config(path: Path) -> SimulationConfig:
    """Read a project YAML scenario or a TOML wrapper selecting project YAML.

    Args:
        path: YAML/TOML file or a project directory with exactly one standard
            simulation.yaml/simulation.toml; resources remain file-relative.

    Returns:
        Validated session configuration without importing or starting Isaac Sim.
    """
    path = resolve_simulation_profile(path)
    try:
        config = _load_simulation_config(path)
        config.validate()
        return config
    except ValueError as error:
        if str(error).startswith(f"{path}:"):
            raise
        raise ValueError(f"{path}: {error}") from error


def _load_simulation_config(path: Path) -> SimulationConfig:
    """Parse source data without validating a population that may be overridden."""
    path = resolve_simulation_profile(path)
    if path.suffix.lower() in (".yaml", ".yml"):
        data = load_yaml_mapping(path)
        if "isaacsim.replicator.agent" in data:
            raise ValueError(
                f"{path}: Native IRA/Actor SDG input is not supported. "
                "Use project YAML with schema_version: 1, scene and pedestrians; "
                "see simulation/docs/custom_scenes.md for migration."
            )
        from .scenarios.loading import _load_scenario_mapping

        return _load_scenario_mapping(path, data)
    try:
        return _load_toml_profile(path)
    except ValueError as error:
        if str(error).startswith(f"{path}:"):
            raise
        raise ValueError(f"{path}: {error}") from error


def _load_toml_profile(path: Path) -> SimulationConfig:
    """Parse profile tables and resolve their file-local references."""
    data = tomllib.loads(path.read_text())
    if "ira_config" in data:
        raise ValueError('ira_config was retired; use scenario = "simulation.yaml" with a ROS 2 project YAML.')
    if "native_ira_settings" in data:
        raise ValueError("native_ira_settings was retired; ROS 2 uses the packaged Kit experience.")
    if "simulation_config" in data:
        raise ValueError("The IRA training profile was retired. Use a ROS 2 simulation profile.")

    def check_keys(values: dict, allowed: set[str], name: str) -> None:
        if not isinstance(values, dict):
            raise ValueError(f"{name} must be a TOML table.")
        if unknown := values.keys() - allowed:
            raise ValueError(f"Unknown {name} fields: {', '.join(sorted(unknown))}")

    def local_path(value: object, name: str, *, output: bool = False) -> Path:
        return resolve_config_reference(value, source=path, field=name, output=output)

    def table(name: str, cls: type) -> dict:
        value = data.get(name, {})
        check_keys(value, {item.name for item in fields(cls)}, name)
        return value

    check_keys(
        data,
        {
            "scenario",
            "navigation",
            "scene",
            "pedestrians",
            "robots",
            "simulation",
            "recording",
        },
        "configuration",
    )
    scene = dict(table("scene", SceneConfig))
    if "asset_path" in scene:
        scene["asset_path"] = local_path(scene["asset_path"], "scene.asset_path")
    if "layers" in scene:
        if not isinstance(scene["layers"], list):
            raise ValueError("scene.layers must be a TOML array.")
        scene["layers"] = tuple(
            local_path(value, f"scene.layers[{index}]") for index, value in enumerate(scene["layers"])
        )
    if isinstance(data.get("pedestrians"), dict) and "settings_path" in data["pedestrians"]:
        raise ValueError("pedestrians.settings_path was retired; use navigation.settings_path for ROS 2 settings.")
    pedestrians = dict(table("pedestrians", PedestrianConfig))
    if "config_path" in pedestrians:
        pedestrians["config_path"] = local_path(pedestrians["config_path"], "pedestrians.config_path")
    robots = []
    robot_values = data.get("robots", [])
    if not isinstance(robot_values, list):
        raise ValueError("robots must be a TOML array of tables.")
    for index, robot in enumerate(robot_values):
        check_keys(robot, {item.name for item in fields(RobotConfig)}, f"robots[{index}]")
        robot = dict(robot)
        for key in ("asset_path", "layer_path"):
            if key in robot:
                robot[key] = local_path(robot[key], f"robots[{index}].{key}")
        robots.append(RobotConfig(**robot))
    simulation = dict(table("simulation", SimulationSettings))
    if "report_path" in simulation:
        simulation["report_path"] = local_path(simulation["report_path"], "simulation.report_path", output=True)
    recording = dict(table("recording", RecordingSettings))
    if "output_path" in recording:
        recording["output_path"] = local_path(recording["output_path"], "recording.output_path", output=True)
    if "scenario" not in data:
        raise ValueError(
            'TOML profiles require scenario = "simulation.yaml"; no historical scene or population is inferred.'
        )
    source_path = local_path(data["scenario"], "scenario")
    if source_path.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError("scenario must select ROS 2 project YAML with schema_version: 1.")
    source = _load_simulation_config(source_path)
    # Wrapper settings override only explicit fields; population data stays in the YAML.
    if "layers" in scene:
        scene["layers"] = (*source.scene.layers, *scene["layers"])
    navigation = table("navigation", NavigationConfig)
    if "settings_path" in navigation:
        navigation["settings_path"] = local_path(navigation["settings_path"], "navigation.settings_path")
    config = replace(
        source,
        scene=replace(source.scene, **scene),
        pedestrians=replace(source.pedestrians, **pedestrians),
        robots=tuple(robots) if "robots" in data else source.robots,
        settings=replace(source.settings, **simulation),
        recording=replace(source.recording, **recording),
        _source_paths=(*source._source_paths, path),
        navigation=replace(source.navigation, **navigation),
    )
    return config
