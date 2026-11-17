# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Summarize planar trajectories and waiting intervals."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

_STATIONARY_SPEED_MPS = 0.05


class _TrajectoryMonitor:
    def __init__(self, *, up_axis: str, stationary_limit_s: float | None = None) -> None:
        self._axes = (0, 1) if up_axis == "Z" else (0, 2)
        self._stationary_limit_s = stationary_limit_s
        self._origin_s: float | None = None
        self._last_s: float | None = None
        self._actors: dict[str, _Motion] = {}

    def observe(self, now: float, positions: dict[str, tuple[float, ...]]) -> None:
        if not math.isfinite(now) or any(
            not all(math.isfinite(value) for value in point) for point in positions.values()
        ):
            raise ValueError("Recorded positions and simulation time must be finite.")
        if self._last_s is not None and (now <= self._last_s or set(positions) != set(self._actors)):
            raise RuntimeError("Recorded time must increase and the participant set must remain unchanged.")
        if self._origin_s is None:
            self._origin_s = now
        elapsed = now - self._last_s if self._last_s is not None else 0
        for actor_id, position in positions.items():
            motion = self._actors.setdefault(actor_id, _Motion(position, position))
            distance = math.hypot(*(position[i] - motion.last[i] for i in self._axes))
            motion.distance_m += distance
            motion.maximum_displacement_m = max(motion.maximum_displacement_m, math.dist(position, motion.origin))
            if elapsed > 0:
                if distance / elapsed < _STATIONARY_SPEED_MPS:
                    motion.stationary_s += elapsed
                    motion.longest_stationary_s = max(motion.longest_stationary_s, motion.stationary_s)
                else:
                    motion.stationary_s = 0
                    motion.moving_s += elapsed
                start, end = self._last_s - self._origin_s, now - self._origin_s
                while start < end - 1e-9:
                    bucket = int((start + 1e-8) // 10)
                    until = min(end, (bucket + 1) * 10)
                    motion.windows[bucket] = motion.windows.get(bucket, 0) + distance * (until - start) / elapsed
                    start = until
            motion.last = position
        self._last_s = now

    def stalled(self) -> list[str]:
        return [
            actor_id
            for actor_id, motion in self._actors.items()
            if self._stationary_limit_s is not None and motion.stationary_s > self._stationary_limit_s
        ]

    def status(self) -> dict[str, object]:
        return {
            "elapsed_s": self._last_s - self._origin_s if self._last_s is not None else 0,
            "stationary_speed_threshold_mps": _STATIONARY_SPEED_MPS,
            "stationary_limit_s": self._stationary_limit_s,
            "window_duration_s": 10,
            "actors": {
                actor_id: {
                    "distance_m": m.distance_m,
                    "maximum_displacement_m": m.maximum_displacement_m,
                    "moving_s": m.moving_s,
                    "longest_stationary_s": m.longest_stationary_s,
                    "last_position_m": m.last,
                    "window_distance_m": [m.windows.get(i, 0) for i in range(max(m.windows, default=-1) + 1)],
                }
                for actor_id, m in self._actors.items()
            },
        }


@dataclass
class _Motion:
    origin: tuple[float, ...]
    last: tuple[float, ...]
    distance_m: float = 0.0
    maximum_displacement_m: float = 0.0
    moving_s: float = 0.0
    stationary_s: float = 0.0
    longest_stationary_s: float = 0.0
    windows: dict[int, float] = field(default_factory=dict)
