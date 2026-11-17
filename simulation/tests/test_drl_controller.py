# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU tests for the DRL-VO controller state machine and scheduling."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pedstrain_simulation.simulation.configuration.drl import DrlPolicySettings
from pedstrain_simulation.simulation.ros2_navigation.drl.controller import DrlVoController
from pedstrain_simulation.simulation.ros2_navigation.drl.robot import RobotState
from pedstrain_simulation.simulation.ros2_navigation.frames import StageFrame
from pedstrain_simulation.simulation.ros2_navigation.planner import OccupancyGrid


class FakeObserver:
    rigid_body_path = "/World/EVT_3/base_link"

    def __init__(self, pose: tuple[float, float, float] = (0.5, 0.5, 0.0)) -> None:
        self.pose = pose

    def read(self, now: float) -> RobotState:
        return RobotState(self.pose, self.pose[2], self.pose, (0.0, 0.0, 0.0), 0.0, 0.0)


class FakeAdapter:
    def __init__(self) -> None:
        self.frames = 0

    def append_scan(self, scan, timestamp_ns) -> None:
        pass

    def sample_history(self, timestamp_ns) -> bool:
        self.frames += 1
        return True

    @property
    def history_ready(self) -> bool:
        return self.frames >= 10

    def build_observation(self, states, pose, local_goal) -> tuple[np.ndarray, dict[str, object]]:
        return np.zeros(19202, dtype=np.float32), {"pedestrian_count": len(states)}


class FakePolicy:
    checkpoint_sha256 = "test"
    parameter_count = 1

    def __init__(self, action: np.ndarray | None = None, error: bool = False) -> None:
        self.action = np.array([0.2, 0.0]) if action is None else action
        self.error = error
        self.calls = 0

    def infer(self, observation) -> tuple[np.ndarray, dict[str, object]]:
        self.calls += 1
        if self.error:
            raise ValueError("bad observation")
        return self.action, {"value": 0.0}

    def close(self) -> None:
        pass


def _controller(tmp_path: Path, *, goal=(2.0, 2.0), dry_run=True, policy=None, observer=None, goal_settle=0.5, max_invalid=3):
    settings = DrlPolicySettings(
        goal_xy_m=goal,
        dry_run=dry_run,
        goal_settle_s=goal_settle,
        goal_timeout_s=0.0,
        max_invalid_observations=max_invalid,
    )
    grid = OccupancyGrid(np.zeros((100, 100), dtype=np.int8), 0.1, (0.0, 0.0))
    navigation = SimpleNamespace(control_hz=60.0, map_clearance_m=0.02)
    return DrlVoController(
        settings,
        grid=grid,
        frame=StageFrame(1.0, "Z", (0.0, 0.0, 0.0), 0.0),
        transport=None,
        output=tmp_path,
        robots=(SimpleNamespace(prim_path="/World/EVT_3"),),
        navigation=navigation,
        observer=observer or FakeObserver(),
        policy=policy or FakePolicy(),
        adapter=FakeAdapter(),
    )


def test_warmup_then_dry_run(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    controller.start()
    for tick in range(6):
        controller.step((tick + 1) / 60.0, 1 / 60.0, [])
    assert controller.status()["phase"] == "WARMUP"
    assert controller.status()["policy_steps"] == 0
    controller._alignment.done = True
    for tick in range(6, 15):
        controller.step((tick + 1) / 60.0, 1 / 60.0, [])
    assert controller.status()["phase"] == "DRY_RUN"
    assert controller.status()["policy_steps"] >= 1
    assert (tmp_path / "drl_manifest.json").is_file()
    assert (tmp_path / "drl_policy.jsonl").is_file()
    controller.close()


def test_goal_reached_zeroes_and_latches(tmp_path: Path) -> None:
    observer = FakeObserver((2.0, 2.0, 0.0))
    controller = _controller(tmp_path, goal=(2.0, 2.0), observer=observer, goal_settle=0.0)
    controller.start()
    controller._alignment.done = True
    for tick in range(15):
        controller.step((tick + 1) / 60.0, 1 / 60.0, [])
    assert controller.status()["phase"] == "GOAL_REACHED"
    assert controller.status()["goal"]["reached"] is True
    controller.close()


def test_invalid_observation_latches_failed(tmp_path: Path) -> None:
    policy = FakePolicy(error=True)
    controller = _controller(tmp_path, policy=policy, max_invalid=2)
    controller.start()
    controller._alignment.done = True
    for tick in range(15):
        controller.step((tick + 1) / 60.0, 1 / 60.0, [])
    assert controller.status()["phase"] == "FAILED"
    assert controller.status()["invalid_observations"] >= 2
    controller.close()


def test_pause_and_close_are_safe_without_transport(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    controller.start()
    controller.pause()
    assert controller.status()["phase"] == "PAUSED"
    controller.close()
