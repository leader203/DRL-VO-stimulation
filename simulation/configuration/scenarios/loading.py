# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Strict, SDK-free loading of readable pedestrian scenario YAML files."""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

from ..loading import load_yaml_mapping
from ..models import RobotConfig, SceneConfig, SimulationConfig, SimulationSettings
from ..navigation import NavigationConfig
from ..paths import resolve_config_path, resolve_config_reference
from ..recording import RecordingSettings
from .compilation import DEFAULT_MOTION_LIBRARY, _build_scenario_config
from .models import PedestrianPlan, Waypoint, _gender, _number, _point


def load_scenario_config(path: Path) -> SimulationConfig:
    """Read a complete pedestrian YAML scenario, resolving file-relative resources.

    Args:
        path: Local ``schema_version: 1`` YAML document. Position coordinates
            follow navigation.waypoint_frame ([m] or [stage unit]); headings
            use [deg], speeds [m/s], tolerances [m] and start/dwell times [s].
            ROS setup reads stage units and up axis before converting positions.

    Returns:
        Validated session configuration and one immutable plan per pedestrian.

    Raises:
        ValueError: For unknown/duplicate fields, invalid values, missing local
            resources, conflicting counts or ambiguous repeated route positions.

    Remote resources are rejected so preflight cannot mistake an inaccessible
    URL for a validated local simulation asset. Only the documented default
    motion-library asset-root token is resolved against the ROS 2 asset root.
    """
    path = resolve_config_path(path)
    config = _load_scenario_mapping(path, load_yaml_mapping(path))
    try:
        config.validate()
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
    return config


def _load_scenario_mapping(path: Path, data: dict[str, object]) -> SimulationConfig:
    """Compile an already-read scenario mapping using its resolved source path."""
    try:
        return _parse_scenario_mapping(path, data)
    except ValueError as error:
        if str(error).startswith(f"{path}:"):
            raise
        raise ValueError(f"{path}: {error}") from error


def _parse_scenario_mapping(path: Path, data: dict[str, object]) -> SimulationConfig:
    """Resolve resources and normalize concise inputs before compilation."""
    if "native_ira_settings" in data:
        raise ValueError("native_ira_settings was retired; ROS 2 uses the packaged Kit experience.")
    _keys(
        data,
        {
            "schema_version",
            "scene",
            "simulation",
            "navigation",
            "robots",
            "recording",
            "pedestrians",
        },
        "scenario",
    )
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("schema_version must be the integer 1.")

    def local(value: object, name: str, *, output: bool = False) -> Path:
        # Existence belongs to final validation, after profile and CLI overrides.
        return resolve_config_reference(value, source=path, field=name, output=output)

    scene_values = _mapping(data.get("scene"), "scene")
    _keys(scene_values, {"asset_path", "layers"}, "scene")
    scene_asset = local(scene_values.get("asset_path"), "scene.asset_path")
    layer_values = _sequence(scene_values.get("layers", []), "scene.layers")
    scene = SceneConfig(
        asset_path=scene_asset,
        layers=tuple(local(value, f"scene.layers[{index}]") for index, value in enumerate(layer_values)),
    )
    simulation = dict(_mapping(data.get("simulation", {}), "simulation"))
    _keys(simulation, {item.name for item in fields(SimulationSettings)}, "simulation")
    if "report_path" in simulation:
        simulation["report_path"] = local(simulation["report_path"], "simulation.report_path", output=True)
    settings = SimulationSettings(**simulation)
    recording = dict(_mapping(data.get("recording", {}), "recording"))
    _keys(recording, {item.name for item in fields(RecordingSettings)}, "recording")
    if "output_path" in recording:
        recording["output_path"] = local(recording["output_path"], "recording.output_path", output=True)
    recording_settings = RecordingSettings(**recording)

    robots = []
    for index, values in enumerate(_sequence(data.get("robots", []), "robots")):
        name = f"robots[{index}]"
        values = dict(_mapping(values, name))
        _keys(values, {item.name for item in fields(RobotConfig)}, name)
        for key in ("asset_path", "layer_path"):
            if key in values:
                values[key] = local(values[key], f"{name}.{key}")
        for key in ("prim_path", "source_prim_path"):
            if key in values and not isinstance(values[key], str):
                raise ValueError(f"{name}.{key} must be an absolute USD prim path.")
        if "position" in values:
            values["position"] = _point(values["position"], f"{name}.position")
        if "yaw_degrees" in values:
            _number(values["yaw_degrees"], f"{name}.yaw_degrees", negative=True)
        robots.append(RobotConfig(**values))

    pedestrian_values = _mapping(data.get("pedestrians"), "pedestrians")
    if "settings_path" in pedestrian_values:
        raise ValueError("pedestrians.settings_path was retired; use navigation.settings_path for ROS 2 settings.")
    if "root_prim_path" in pedestrian_values:
        raise ValueError("pedestrians.root_prim_path was retired; ROS 2 allocates its own character namespace.")
    _keys(
        pedestrian_values,
        {"count", "assets", "motion_library", "defaults", "actors"},
        "pedestrians",
    )
    actors = _mapping(pedestrian_values["actors"], "pedestrians.actors") if "actors" in pedestrian_values else None
    count = pedestrian_values.get("count", len(actors) if actors is not None else None)
    if type(count) is not int or count < 1:
        raise ValueError("pedestrians.count must be a positive integer.")
    assets = local(pedestrian_values["assets"], "pedestrians.assets") if "assets" in pedestrian_values else None
    motion_library = pedestrian_values.get("motion_library", DEFAULT_MOTION_LIBRARY)
    if motion_library != DEFAULT_MOTION_LIBRARY:
        motion_library = local(motion_library, "pedestrians.motion_library")
    defaults = _mapping(pedestrian_values.get("defaults", {}), "pedestrians.defaults")
    _keys(
        defaults,
        {"behavior", "speed", "start_time", "repeat", "arrival_tolerance", "wait", "yaw", "gender"},
        "pedestrians.defaults",
    )
    if "behavior" in defaults and defaults["behavior"] not in ("walk", "stand", "wander"):
        raise ValueError("pedestrians.defaults.behavior must be walk, stand or wander.")
    if "gender" in defaults:
        _gender(defaults["gender"], "pedestrians.defaults.gender")
    for name in ("speed", "arrival_tolerance", "start_time"):
        if name in defaults:
            _number(defaults[name], f"pedestrians.defaults.{name}", zero=name == "start_time")
    if "yaw" in defaults:
        _number(defaults["yaw"], "pedestrians.defaults.yaw", negative=True)
    if "repeat" in defaults and type(defaults["repeat"]) is not bool:
        raise ValueError("pedestrians.defaults.repeat must be a boolean.")
    default_wait = defaults.get("wait", 0.0)
    _number(default_wait, "pedestrians.defaults.wait", zero=True)
    if actors is None:
        actors = {f"walker_{index + 1:02d}": {} for index in range(count)}
    if len(actors) != count:
        raise ValueError(f"pedestrians.count is {count}, but pedestrians.actors defines {len(actors)} actors.")
    plans = tuple(_load_pedestrian_plan(name, actor, defaults, source=path) for name, actor in actors.items())

    config = _build_scenario_config(
        scene=scene,
        plans=plans,
        assets=assets,
        motion_library=motion_library,
        settings=settings,
        robots=tuple(robots),
        recording=recording_settings,
    )
    navigation = dict(_mapping(data.get("navigation", {}), "navigation"))
    _keys(navigation, {item.name for item in fields(NavigationConfig)}, "navigation")
    if "settings_path" in navigation:
        navigation["settings_path"] = local(navigation["settings_path"], "navigation.settings_path")
    return replace(config, _source_paths=(path,), navigation=NavigationConfig(**navigation))


def _load_pedestrian_plan(name: str, actor: object, defaults: dict, *, source: Path) -> PedestrianPlan:
    """Apply actor overrides and attach actor/waypoint locations to validation errors."""
    actor_name = f"pedestrians.actors.{name}"
    try:
        return _parse_pedestrian_plan(name, _mapping(actor, "actor"), defaults, source=source)
    except ValueError as error:
        if str(error).startswith(f"{source}:"):
            raise
        raise ValueError(f"{actor_name}: {error}") from error


def _parse_pedestrian_plan(name: str, actor: dict, defaults: dict, *, source: Path) -> PedestrianPlan:
    _keys(
        actor,
        {
            "position",
            "yaw",
            "behavior",
            "speed",
            "start_time",
            "repeat",
            "waypoints",
            "arrival_tolerance",
            "wait",
            "asset_path",
            "gender",
        },
        "actor",
    )
    values = {**defaults, **actor}
    wait = values.pop("wait", 0.0)
    _number(wait, "wait", zero=True)
    waypoints = []
    for index, point in enumerate(_sequence(values.pop("waypoints", []), "waypoints")):
        point_name = f"waypoints[{index}]"
        try:
            if isinstance(point, dict):
                _keys(point, {"position", "wait", "yaw"}, point_name)
                waypoints.append(
                    Waypoint(_point(point.get("position"), point_name), point.get("wait", wait), point.get("yaw"))
                )
            else:
                waypoints.append(Waypoint(_point(point, point_name), wait))
        except ValueError as error:
            raise ValueError(f"{point_name}: {error}") from error
    if waypoints and "behavior" not in values:
        values["behavior"] = "walk"
    if "wait" in actor and wait != 0 and values.get("behavior", "wander") != "walk":
        raise ValueError("wait is only supported for walk behavior.")
    if "position" in values:
        values["position"] = _point(values["position"], "position")
    if "asset_path" in values:
        field = f"pedestrians.actors.{name}.asset_path"
        target = resolve_config_reference(values["asset_path"], source=source, field=field)
        values["asset_path"] = target
    return PedestrianPlan(name=name, waypoints=tuple(waypoints), **values)


def _mapping(value: object, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping.")
    return value


def _sequence(value: object, name: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a YAML list.")
    return value


def _keys(values: dict, allowed: set[str], name: str) -> None:
    if any(not isinstance(key, str) for key in values):
        raise ValueError(f"{name} field names must be strings.")
    if unknown := values.keys() - allowed:
        raise ValueError(f"Unknown {name} fields: {', '.join(sorted(unknown))}")
