# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ROS 2 pedestrian simulation command-line entry point."""


def main(argv: list[str] | None = None) -> None:
    """Run the ROS 2 launcher without eagerly importing it."""
    from .launcher import main as launch

    launch(argv)
