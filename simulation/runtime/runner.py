# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Public entry point for the ROS 2 simulation owner loop."""

from pathlib import Path

from ..configuration.models import SimulationConfig


def run_simulation(
    config: SimulationConfig,
    *,
    extra_args: list[str] | None = None,
    console: bool | None = None,
    report_path: Path | None = None,
    agent_config: object | None = None,
) -> None:
    """Run the scene and ROS 2 pedestrian controller to the configured duration [s].

    Args:
        config: ROS 2 scene and population settings.
        extra_args: Additional Kit command-line options.
        console: Override the playback console setting.
        report_path: Override the final status JSON destination.
        agent_config: Obsolete external director option, which must be omitted.
    """
    if agent_config is not None:
        raise ValueError("External IRA directors were removed. Use a ROS 2 scene with run PROJECT.")
    config.validate()
    from ..ros2_navigation.runtime import run_ros2_simulation

    run_ros2_simulation(config, extra_args=extra_args, console=console, report_path=report_path)
