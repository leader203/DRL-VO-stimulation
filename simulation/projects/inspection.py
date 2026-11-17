# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect local configuration references without importing the simulator."""

import math
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import cast

from ..configuration.loading import load_simulation_config
from ..configuration.models import SimulationConfig
from ..configuration.navigation import Ros2NavigationSettings, load_ros2_navigation_settings
from ..configuration.paths import ROS2_KIT_EXPERIENCE_PATH, ROS2_NAVIGATION_SETTINGS_PATH, resolve_simulation_profile
from ..configuration.population import resolve_population
from ..configuration.resources import resolve_ros2_motion_library
from ..configuration.scenarios.assets import character_catalog_files
from ..configuration.scenarios.models import PedestrianPlan
from ..recording.output import _validate_report_path, validate_recording_runtime
from ..ros2_navigation.frames import StageFrame


def inspect_simulation_config(path: Path) -> dict[str, object]:
    """Report effective scene assets, pedestrian plans, and ROS 2 navigation settings.

    Args:
        path: Local YAML profile, TOML wrapper, or project directory.

    Returns:
        SDK-free preview with fixed-spawn distance checks [m] when available.
        USD composition, map extraction and route feasibility require runtime checks.
    """
    path = resolve_simulation_profile(path)
    return _inspect_simulation_config(load_simulation_config(path), path, check_placement=True)


def inspect_config_dependencies(path: Path) -> dict[str, object]:
    """Report declared local inputs without checking output paths or starting Kit."""
    path = resolve_simulation_profile(path)
    summary = _inspect_simulation_config(load_simulation_config(path), path, validate_outputs=False)
    return cast(dict[str, object], summary["dependencies"])


def _inspect_simulation_config(
    config: SimulationConfig, path: Path, *, validate_outputs: bool = True, check_placement: bool = False
) -> dict[str, object]:
    """Inspect final effective settings, including CLI overrides."""
    config.validate()
    selection = resolve_population(config)
    population = selection.config
    plans = population.pedestrian_plans if population is not None else ()
    population_source = config.pedestrians.config_path or path
    assets = [str(config.scene.asset_path), *(str(value) for value in config.scene.layers)]
    assets.extend(str(robot.asset_path or robot.layer_path) for robot in config.robots)
    sources = [path, *config._source_paths, ROS2_KIT_EXPERIENCE_PATH]
    navigation = None
    navigation_path = None
    parameters = None
    effective_avoidance = None
    minimum_distance = None
    clearance = None
    if population is not None:
        navigation_path = config.navigation.settings_path or ROS2_NAVIGATION_SETTINGS_PATH
        navigation = load_ros2_navigation_settings(navigation_path)
        try:
            motion = resolve_ros2_motion_library(population.motion_library, navigation.asset_root)
        except ValueError as error:
            raise ValueError(f"{population_source}: {error}") from error
        assets.append(str(motion))
        for asset in dict.fromkeys(asset for _, asset in population.pedestrian_assets):
            assets.append(str(asset))
            assets.extend(str(file) for file in character_catalog_files(asset, navigation.asset_root))
        sources.extend([population_source, *population._source_paths, navigation_path])
        parameters = navigation.to_dict()
        effective_avoidance = {"ros2": True, "ira": False, "navmesh": False}
        diameter = 2 * navigation.body_radius_m
        planning_radius = navigation.body_radius_m + navigation.map_clearance_m
        minimum_distance = diameter + navigation.pair_clearance_m
        clearance = {
            "planning_radius_m": planning_radius,
            "preferred_path_radius_m": planning_radius + navigation.preferred_path_clearance_m,
            "fixed_spawn_body_center_distance_m": diameter,
            "pedestrian_target_center_distance_m": minimum_distance,
        }
    drl = None
    if config.drl is not None and config.drl.enabled:
        drl = {
            "enabled": True,
            "package_root": str(config.drl.package_root) if config.drl.package_root else None,
            "checkpoint": str(config.drl.checkpoint) if config.drl.checkpoint else None,
            "expected_sha256": config.drl.expected_sha256,
            "device": config.drl.device,
            "goal_xy_m": list(config.drl.goal_xy_m) if config.drl.goal_xy_m else None,
            "dry_run": config.drl.dry_run,
            "max_linear_m_s": config.drl.max_linear_m_s,
            "max_angular_rad_s": config.drl.max_angular_rad_s,
        }
        if config.drl.package_root is not None:
            sources.append(config.drl.package_root)
        if config.drl.checkpoint is not None:
            sources.append(config.drl.checkpoint)
    if validate_outputs:
        validate_recording_runtime(config)
        _validate_report_path(config, config.settings.report_path)
    simulation = asdict(config.settings)
    simulation["report_path"] = str(config.settings.report_path) if config.settings.report_path else None
    scene = str(config.scene.asset_path)
    dependencies = {
        "entrypoint": str(path),
        "scope": "known_configurations_and_declared_assets",
        "outputs_checked": validate_outputs,
        "configuration_files": list(dict.fromkeys(str(source) for source in sources)),
        "runtime_settings": {
            "path": str(ROS2_KIT_EXPERIENCE_PATH),
            "selection": "ros2_runtime",
            "asset_root": navigation.asset_root if navigation is not None else None,
        },
        "pedestrian_settings": {
            "path": str(navigation_path) if navigation_path is not None else None,
            "selection": (
                "disabled"
                if navigation is None
                else "explicit"
                if config.navigation.settings_path
                else "package_default"
            ),
        },
        "scene": {"path": scene, "selection": "scene_override"},
        "population": {
            "path": str(population_source) if population is not None else None,
            "selection": selection.source,
        },
        "assets": list(dict.fromkeys(assets)),
        "unverified_remote_assets": [],
        "unverified_sdk_assets": [],
    }
    return {
        "simulation_config": str(path),
        "navigation": {
            "backend": "ros2",
            "waypoint_frame": config.navigation.waypoint_frame,
            "settings_path": str(config.navigation.settings_path)
            if config.navigation.settings_path and population is not None
            else None,
        },
        "pedestrian_controller": "ros2_sfm_orca" if population is not None else "disabled",
        "pedestrian_effective_avoidance": effective_avoidance,
        "drl_vo": drl,
        "simulation": simulation,
        "dependencies": dependencies,
        "scene": scene,
        "configuration_format": "pedestrian_scenario_v1",
        "pedestrians_config": str(population_source),
        "pedestrian_parameters": parameters,
        "random_spawn_minimum_center_distance_m": minimum_distance,
        "ros2_clearance": clearance,
        "fixed_spawn_check": _fixed_spawn_check(plans, navigation, config.navigation.waypoint_frame, scene)
        if navigation is not None and check_placement
        else None,
        "pedestrians": len(plans),
        "groups": {plan.name: 1 for plan in plans},
        "pedestrian_plans": [plan.to_dict() for plan in plans],
        "pedestrian_plans_frame": config.navigation.waypoint_frame,
        "schedules": {
            plan.name: {
                "behavior": plan.behavior,
                "start_time_s": plan.start_time,
                "repeat": plan.repeat,
                "waypoints": len(plan.waypoints),
            }
            for plan in plans
        },
        "robots": [robot.prim_path for robot in config.robots],
        "unverified_remote_assets": [],
        "unverified_sdk_assets": [],
        "recording": {
            "enabled": config.recording.enabled,
            "output_path": str(config.recording.output_path) if config.recording.output_path else None,
            "fps": config.recording.fps,
            "duration_s": config.settings.duration,
            "warmup_s": config.recording.warmup_s,
            "camera_view": config.recording.camera_view,
            "camera_eye_m": list(config.recording.camera_eye_m) if config.recording.camera_eye_m else None,
            "camera_target_m": list(config.recording.camera_target_m) if config.recording.camera_target_m else None,
            "max_stationary_s": config.recording.max_stationary_s,
        },
    }


def _fixed_spawn_check(
    plans: tuple[PedestrianPlan, ...], navigation: Ros2NavigationSettings, waypoint_frame: str, scene: str
) -> dict[str, object]:
    """Report fixed-body pair distances [m] without composing a stage or sampling spawns."""
    fixed = [(plan.name, plan.position) for plan in plans if plan.position is not None]
    diameter = 2 * navigation.body_radius_m
    overlaps = []
    report = {
        "status": "not_checked",
        "fixed_count": len(fixed),
        "minimum_center_distance_m": diameter,
        "overlaps": overlaps,
    }
    if len(fixed) < 2:
        return {**report, "reason": "fewer_than_two_fixed_spawns"}
    if waypoint_frame == "stage_world":
        frame = _authored_stage_frame(scene)
        if frame is None:
            return {**report, "reason": "base_stage_metrics_unavailable"}
        try:
            fixed = [(name, frame.stage_to_world(point)) for name, point in fixed]
        except ValueError:
            return {**report, "reason": "invalid_stage_world_coordinates"}
    for (left, a), (right, b) in combinations(fixed, 2):
        distance = math.dist(a[:2], b[:2])
        if distance < diameter - 1e-9:
            overlaps.append({"actors": [left, right], "center_distance_m": distance})
    report["status"] = "overlap" if overlaps else "no_body_overlap"
    return report


def _authored_stage_frame(scene: str) -> StageFrame | None:
    """Read local root-layer metrics only, avoiding scene composition and remote dependencies."""
    if not Path(scene).is_file():
        return None
    try:
        from pxr import Sdf, Tf, UsdGeom
    except ImportError:
        return None
    try:
        layer = Sdf.Layer.OpenAsAnonymous(scene, metadataOnly=True)
        if layer is not None:
            root = layer.pseudoRoot
            units_key, axis_key = UsdGeom.Tokens.metersPerUnit, UsdGeom.Tokens.upAxis
            if root.HasInfo(units_key) and root.HasInfo(axis_key):
                return StageFrame(root.GetInfo(units_key), str(root.GetInfo(axis_key)))
    except (Tf.ErrorException, ValueError):
        return None
    return None
