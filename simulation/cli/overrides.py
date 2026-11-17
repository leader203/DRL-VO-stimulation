# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resolve CLI selections and overrides into an immutable session configuration."""

from __future__ import annotations

import argparse
from dataclasses import replace

from ..configuration.drl import load_drl_policy_settings
from ..configuration.loading import _load_simulation_config
from ..configuration.models import RobotConfig, SimulationConfig, _with_ros2_navigation_settings
from ..configuration.paths import resolve_config_path


def _simulation_from_args(args: argparse.Namespace) -> SimulationConfig:
    """Apply CLI overrides to a profile without mutating its frozen settings."""
    profile = args.simulation_config or args.config
    population_override = args.pedestrians_config
    if profile is None:
        if population_override is not None:
            # A standalone profile retains its own population selection.
            profile, population_override = population_override, None
        elif args.scene is not None:
            raise ValueError(
                "--scene requires an explicit pedestrian configuration: use --config SCENARIO.yaml "
                "or init PROJECT --scene USD --characters DIR --count N, then run PROJECT."
            )
        else:
            raise ValueError("Select an explicit ROS 2 project with run PROJECT or --config SCENARIO.yaml.")
    config = _load_simulation_config(profile)
    if args.config is not None and args.simulation_config is not None:
        replacement = _load_simulation_config(args.config)
        config = replace(
            config,
            scene=replacement.scene,
            pedestrians=replace(config.pedestrians, config_path=replacement.pedestrians.config_path),
            pedestrian_plans=replacement.pedestrian_plans,
            pedestrian_assets=replacement.pedestrian_assets,
            motion_library=replacement.motion_library,
            _source_paths=tuple(dict.fromkeys((*config._source_paths, *replacement._source_paths))),
        )
    if args.scene is not None:
        config = replace(config, scene=replace(config.scene, asset_path=resolve_config_path(args.scene)))
    if population_override is not None or args.no_pedestrians:
        pedestrians = replace(
            config.pedestrians,
            enabled=False if args.no_pedestrians else config.pedestrians.enabled,
            config_path=resolve_config_path(population_override)
            if population_override is not None
            else config.pedestrians.config_path,
        )
        config = replace(config, pedestrians=pedestrians)
    if args.navigation_settings is not None:
        config = _with_ros2_navigation_settings(config, args.navigation_settings)
    if args.robot_usd is not None or args.robot_layer is not None:
        if args.robot_layer is not None and args.robot_prim_path is None:
            raise ValueError("--robot-layer requires --robot-prim-path.")
        robot = RobotConfig(
            asset_path=resolve_config_path(args.robot_usd) if args.robot_usd else None,
            layer_path=resolve_config_path(args.robot_layer) if args.robot_layer else None,
            prim_path=args.robot_prim_path or "/World/Robot",
            source_prim_path=args.robot_source_prim,
            position=args.robot_position,
            yaw_degrees=args.robot_yaw,
        )
        config = replace(config, robots=(*config.robots, robot))
    elif any(
        value is not None
        for value in (args.robot_position, args.robot_yaw, args.robot_source_prim, args.robot_prim_path)
    ):
        raise ValueError("Robot pose/source options require --robot-usd or --robot-layer.")
    drl_config = getattr(args, "drl_policy_config", None)
    drl_package_root = getattr(args, "drl_package_root", None)
    drl_goal = getattr(args, "drl_goal", None)
    drl_device = getattr(args, "drl_device", None)
    drl_dry_run = getattr(args, "drl_dry_run", None)
    if drl_config is not None:
        drl = load_drl_policy_settings(resolve_config_path(drl_config))
        changes = {}
        if drl_package_root is not None:
            root = resolve_config_path(drl_package_root)
            changes["package_root"] = root
            changes["checkpoint"] = root / (drl.checkpoint.name if drl.checkpoint else "frozen_policy.pt")
        if drl_goal is not None:
            changes["goal_xy_m"] = tuple(drl_goal)
        if drl_device is not None:
            changes["device"] = drl_device
        if drl_dry_run is not None:
            changes["dry_run"] = drl_dry_run
        if changes:
            drl = replace(drl, **changes)
        drl.validate_files()
        config = replace(config, drl=drl)
    elif any(value is not None for value in (drl_package_root, drl_goal, drl_device, drl_dry_run)):
        raise ValueError("DRL package/goal/device/dry-run options require --drl-policy-config.")
    overrides = {
        name: getattr(args, name)
        for name in (
            "headless",
            "render_viewport",
            "start_paused",
            "duration",
            "width",
            "height",
            "physics_hz",
            "solver",
            "seed",
            "console",
        )
        if getattr(args, name) is not None
    }
    if args.no_report:
        overrides["report_path"] = None
    elif args.report is not None:
        overrides["report_path"] = resolve_config_path(args.report)
    config = replace(config, settings=replace(config.settings, **overrides))
    config.validate()
    return config
