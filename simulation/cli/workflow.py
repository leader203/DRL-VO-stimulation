# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One-command configuration, execution, coordinate and trajectory workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
import uuid
from pathlib import Path

from ..configuration.paths import resolve_simulation_profile
from ..projects.workflow import run_simulation_workflow


def main(argv: list[str] | None = None) -> None:
    """Run an isolated workflow without importing the native SDK in this process."""
    parser = argparse.ArgumentParser(prog="run.py workflow", description=__doc__)
    parser.add_argument("project", type=Path, nargs="?", default=Path("."))
    parser.add_argument(
        "--output", type=Path, help="New run directory; omission creates a fresh project/output/workflow_*."
    )
    parser.add_argument("--navigation-settings", type=Path, help="Use a ROS 2 navigation YAML for this run.")
    parser.add_argument(
        "--record",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override video selection; retain configured FPS, warmup and camera.",
    )
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--require-route-goals", action="store_true", help="Require all original goals within the user duration."
    )
    parser.add_argument("--timeout-s", type=float, help="Owned native process-group wall limit [s].")
    args = parser.parse_args(argv)
    try:
        project = resolve_simulation_profile(args.project)
        output = args.output or project.parent / "output" / ("workflow_" + uuid.uuid4().hex[:8])
        result = run_simulation_workflow(
            project,
            output,
            record=args.record,
            headless=args.headless,
            require_route_goals=args.require_route_goals,
            timeout_s=args.timeout_s,
            navigation_settings=args.navigation_settings,
        )
    except (OSError, ValueError, TypeError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    print(
        "WORKFLOW_RESULT "
        + json.dumps(
            {
                "passed": result["passed"],
                "report": str(Path(result["output_directory"]) / "workflow_report.json"),
                "video": result.get("video"),
                "errors": result["errors"],
            }
        ),
        flush=True,
    )
    if not result["passed"]:
        raise SystemExit(1)
