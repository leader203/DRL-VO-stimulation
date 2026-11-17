# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configure and launch ROS 2 pedestrian simulation without eager SDK imports."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

from ..configuration.paths import resolve_simulation_profile
from ..projects.inspection import _inspect_simulation_config
from ..runtime.runner import run_simulation
from .arguments import _reject_retired_options
from .arguments import create_parser as create_parser
from .overrides import _simulation_from_args


def main(argv: list[str] | None = None, *, runner: Callable[..., None] | None = None) -> None:
    """Resolve launcher inputs, then invoke the shared simulation lifecycle.

    Args:
        argv: Explicit arguments, or None to read the process command line.
        runner: Optional lifecycle callable for embedding; defaults to run_simulation.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "monitor":
        from ..ros2_navigation.monitor import main as monitor_main

        monitor_main(argv[1:])
        return
    if argv and argv[0] == "workflow":
        from .workflow import main as workflow_main

        workflow_main(argv[1:])
        return
    if argv and argv[0] in ("init", "check"):
        from .projects import main as project_main

        project_main(argv)
        return
    parser = create_parser()
    project_run = bool(argv and argv[0] == "run")
    if project_run:
        parser.prog += " run"
        parser.add_argument(
            "project",
            nargs="?",
            type=Path,
            default=Path("."),
            metavar="PROJECT",
            help="Project directory or YAML/TOML; omitted selects the current directory.",
        )
        parser.add_argument("--dry-run", action="store_true", help="Check effective settings and print JSON; no SDK.")
    arguments = argv[1:] if project_run else argv
    _reject_retired_options(parser, arguments)
    args, kit_args = parser.parse_known_args(arguments)
    try:
        if project_run:
            if args.simulation_config is not None or args.config is not None:
                raise ValueError("run PROJECT cannot be combined with --simulation-config or --config.")
            args.simulation_config = resolve_simulation_profile(args.project)
        config = _simulation_from_args(args)
        if project_run:
            summary = _inspect_simulation_config(config, args.simulation_config, check_placement=args.dry_run)
            if args.dry_run:
                print(json.dumps(summary, indent=2, ensure_ascii=False))
                return
            print(
                f"Preflight: profile={summary['simulation_config']}; "
                f"{summary['pedestrians']} pedestrians; scene={summary['scene']}. "
                "Local inputs checked; ROS 2/PhysX occupancy and animation validation follows during setup.",
                file=sys.stderr,
            )
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    execute = run_simulation if runner is None else runner
    execute(
        config,
        extra_args=kit_args,
        console=config.settings.console,
        report_path=config.settings.report_path,
    )
