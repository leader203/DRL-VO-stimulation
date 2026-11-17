# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integer-tick rate clocks for the fixed 60 Hz ROS owner loop."""

from __future__ import annotations

import math


class RateClock:
    """Fire every ``control_hz / rate_hz`` control ticks without sim-time drift."""

    def __init__(self, rate_hz: float, control_hz: float) -> None:
        if not math.isfinite(rate_hz) or not math.isfinite(control_hz) or rate_hz <= 0 or control_hz <= 0:
            raise ValueError("rate_hz and control_hz must be finite and positive.")
        ratio = control_hz / rate_hz
        every = round(ratio)
        if every < 1 or not math.isclose(ratio, every, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"{control_hz:g} Hz control cannot schedule {rate_hz:g} Hz exactly.")
        self.rate_hz = float(rate_hz)
        self.control_hz = float(control_hz)
        self.every = int(every)
        self.ticks = 0

    def due(self) -> bool:
        return self.ticks % self.every == 0

    def commit(self) -> None:
        self.ticks += 1

    def reset(self) -> None:
        self.ticks = 0
