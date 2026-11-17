# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility import for robot rendering helpers; use ``robots.visuals`` for new code."""

from .robots.visuals import RobotVisuals

__all__ = ["RobotVisuals"]
