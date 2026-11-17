# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Project creation and local configuration checks, without SDK startup."""

import argparse
import json
import shlex
import sys
from pathlib import Path

from ..configuration.paths import resolve_config_path
from ..projects import create_simulation_project, inspect_config_dependencies, inspect_simulation_config
from .arguments import _reject_retired_options

# Preserve the former private entry while keeping dependency inspection in projects.
_dependency_summary = inspect_config_dependencies


def main(argv: list[str]) -> None:
    """Handle init/check commands before the simulation launcher is invoked."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("init", help="Create a project YAML scene in a new directory.")
    create.add_argument("directory", type=Path)
    create.add_argument(
        "--scene", type=Path, required=True, help="Environment USD with collision geometry and floor support."
    )
    create.add_argument("--count", type=int, default=5, help="Number of pedestrians.")
    create.add_argument("--behavior", choices=("wander", "standing"), default="wander")
    create.add_argument("--navigation-backend", choices=("ros2",), default="ros2")
    create.add_argument("--navigation-settings", type=Path, help="ROS 2 map, coordinate and crowd settings YAML.")
    create.add_argument(
        "--waypoint-frame",
        choices=("stage_world", "world_m"),
        help="Position coordinates: composed Isaac stage world or canonical meters; ROS default: stage_world.",
    )
    create.add_argument(
        "--characters", type=Path, required=True, help="Local character USD, directory, or YAML catalog."
    )
    create.add_argument(
        "--format", choices=("yaml",), default="yaml", help="Project YAML (the only supported pedestrian format)."
    )
    check = commands.add_parser("check", help="Check local profile/resources/counts; does not bake or validate routes.")
    check.add_argument(
        "profile",
        nargs="?",
        type=Path,
        default=Path("."),
        metavar="PROJECT",
        help="Project directory or YAML/TOML; omitted selects the current directory.",
    )
    check.add_argument(
        "--dependencies", action="store_true", help="Describe configuration references only; no output/tool checks."
    )
    _reject_retired_options(parser, argv)
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            profile = create_simulation_project(
                args.directory,
                scene_asset=args.scene,
                pedestrian_count=args.count,
                behavior=args.behavior,
                character_assets=args.characters,
                configuration_format=args.format,
                navigation_backend=args.navigation_backend,
                navigation_settings=args.navigation_settings,
                waypoint_frame=args.waypoint_frame or "stage_world",
            )
            print(json.dumps({"simulation_config": str(profile)}, indent=2, ensure_ascii=False))
            launcher = Path(__file__).resolve().parents[1] / "run.py"
            command = [sys.executable, str(launcher)]
            edit = str(profile)
            navigation_source = (
                resolve_config_path(args.navigation_settings)
                if args.navigation_settings
                else profile.with_name("navigation_settings.yaml")
            )
            print(
                f"Edit: {edit}\n"
                f"ROS 2 navigation: {navigation_source}\n"
                f"Run: {shlex.join([*command, 'run', str(profile.parent)])}\n"
                f"Check: {shlex.join([*command, 'check', str(profile.parent)])}",
                file=sys.stderr,
            )
        else:
            summary = (
                _dependency_summary(args.profile) if args.dependencies else inspect_simulation_config(args.profile)
            )
            print(json.dumps(summary, indent=2, ensure_ascii=False))
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
