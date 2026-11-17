# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configure and launch ROS 2 pedestrian simulation without eager SDK imports."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

_RETIRED_OPTIONS = frozenset(
    {
        "--agent-config",
        "--check-agent-config",
        "--copy-runtime-settings",
        "--defer-pedestrian-placement",
        "--legacy-example",
        "--model-profile",
        "--native-ira-settings",
        "--pedestrian-settings",
        "--scene-config",
    }
)


def _reject_retired_options(parser: argparse.ArgumentParser, argv: Sequence[str]) -> None:
    for argument in argv:
        option = argument.partition("=")[0]
        if option in _RETIRED_OPTIONS:
            parser.error(
                f"{option} was removed. Select a ROS 2 project with run PROJECT and use "
                "--navigation-settings for crowd, map, and motion configuration."
            )


def create_parser() -> argparse.ArgumentParser:
    """Build the ROS 2 launcher parser without importing a simulator."""
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=__doc__,
        epilog=(
            "Project workflow: init DIRECTORY --scene USD --characters DIR --count N; edit YAML; run [PROJECT]. "
            "Optional preview: check [PROJECT] or run [PROJECT] --dry-run."
        ),
    )
    scene = parser.add_argument_group("Scene and pedestrians")
    scene.add_argument(
        "--simulation-config", type=Path, help="ROS 2 scenario YAML or launcher TOML; file-relative paths."
    )
    scene.add_argument(
        "--config",
        type=Path,
        help="Explicit ROS 2 scenario YAML or launcher TOML; no scene is selected implicitly.",
    )
    scene.add_argument("--scene", type=Path, help="Override the base environment USD.")
    scene.add_argument(
        "--pedestrians-config",
        type=Path,
        help="Project YAML or TOML. Replaces the population of a selected session or supplies the whole session alone.",
    )
    scene.add_argument("--navigation-settings", type=Path, help="Use this ROS 2 navigation YAML for this launch only.")
    scene.add_argument("--no-pedestrians", action="store_true", help="Disable pedestrian spawning and navigation.")
    robot = parser.add_argument_group("Additional robot")
    source = robot.add_mutually_exclusive_group()
    source.add_argument("--robot-usd", type=Path, help="Reference one additional robot USD model root.")
    source.add_argument("--robot-layer", type=Path, help="Compose one additional preconfigured robot USD layer.")
    robot.add_argument(
        "--robot-prim-path",
        help="Destination prim (default: /World/Robot for --robot-usd); required for a robot layer.",
    )
    robot.add_argument("--robot-source-prim", help="Model prim inside --robot-usd (default: asset default prim).")
    robot.add_argument(
        "--robot-position", type=float, nargs=3, metavar=("X", "Y", "Z"), help="Initial translation [m]."
    )
    robot.add_argument("--robot-yaw", type=float, help="Initial heading [deg] about the scene up-axis.")
    runtime = parser.add_argument_group("Simulation and control")
    runtime.add_argument("--headless", action=argparse.BooleanOptionalAction, default=None)
    runtime.add_argument(
        "--render-viewport",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Update the main viewport; disabling requires --headless and recording off.",
    )
    runtime.add_argument("--start-paused", action=argparse.BooleanOptionalAction, default=None)
    runtime.add_argument("--duration", type=float, help="Auto-exit at timeline time [s]; 0 waits for exit.")
    runtime.add_argument("--width", type=int, help="Render width in pixels.")
    runtime.add_argument("--height", type=int, help="Render height in pixels.")
    runtime.add_argument("--physics-hz", type=int, help="Override physics frequency [Hz]; omitted preserves the scene.")
    runtime.add_argument("--solver", choices=("PGS", "TGS"), help="Override the PhysX solver.")
    runtime.add_argument("--seed", type=int, help="Override the simulation random seed.")
    runtime.add_argument(
        "--console",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override terminal controls: play, pause, status, and quit.",
    )
    report = runtime.add_mutually_exclusive_group()
    report.add_argument("--report", type=Path, help="Override configured final session status JSON output.")
    report.add_argument("--no-report", action="store_true", help="Disable the configured status report.")
    drl = parser.add_argument_group("DRL-VO policy")
    drl.add_argument(
        "--drl-policy-config",
        type=Path,
        help="Enable the frozen DRL-VO policy with this YAML; file-relative package/checkpoint paths.",
    )
    drl.add_argument("--drl-package-root", type=Path, help="Override the DRL model package directory.")
    drl.add_argument("--drl-goal", type=float, nargs=2, metavar=("X", "Y"), help="DRL goal in ROS map metres.")
    drl.add_argument("--drl-device", help="Torch device for DRL inference, such as cpu or cuda:0.")
    drl.add_argument(
        "--drl-dry-run",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run observation and inference without publishing /cmd_vel.",
    )
    return parser
