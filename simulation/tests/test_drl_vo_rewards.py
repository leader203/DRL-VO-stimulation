# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parity between the migrated Isaac Lab reward terms and the DRL-VO reference.

The reference is ``DRL-VO/code/reward_logic.py`` — the harness copy of
``DRLNavEnv._compute_reward``/``_is_done`` — imported directly so the migrated
terms are checked against the author's source rather than a re-derivation.
"""

from __future__ import annotations

import importlib.util
import math
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pedstrain_simulation.simulation.rl import rewards as terms

REFERENCE_PATH = Path(__file__).resolve().parents[2] / "DRL-VO" / "code" / "reward_logic.py"


def _reference_module():
    spec = importlib.util.spec_from_file_location("_drl_vo_reward_reference", REFERENCE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _RecordingRandom:
    """Mersenne Twister that records the headings the reference draws."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self.draws: list[float] = []

    def uniform(self, low: float, high: float) -> float:
        value = self._rng.uniform(low, high)
        self.draws.append(value)
        return value

    def getstate(self) -> object:
        return self._rng.getstate()

    def setstate(self, state: object) -> None:
        self._rng.setstate(state)


class _Goal:
    """Minimal stand-in for :class:`..rl.reference.DrlVoReference`."""

    def __init__(self, distance: float, local: tuple[float, float]) -> None:
        self.distance_m = torch.tensor([distance], dtype=torch.float32)
        self.local_goal_base_xy = torch.tensor([local], dtype=torch.float32)

    def update(self, env: object) -> None:
        del env

    def reset(self, env: object = None, env_ids: object = None) -> None:
        del env, env_ids

    def set(self, distance: float, local: tuple[float, float]) -> None:
        self.distance_m = self.distance_m.new_tensor([distance])
        self.local_goal_base_xy = self.local_goal_base_xy.new_tensor([local])


def _quaternion(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def _environment(
    *,
    distance: float = 5.0,
    local: tuple[float, float] = (1.0, 0.0),
    scan: list[float] | None = None,
    pedestrians: list[tuple[float, float, float, float]] = (),
    yaw: float = 0.0,
    robot: tuple[float, float] = (0.0, 0.0),
    forward: float = 0.0,
    yaw_rate: float = 0.0,
) -> SimpleNamespace:
    """Build a stub environment; pedestrian tuples are robot-frame ``x, y, vx, vy``."""
    count = len(pedestrians)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    world = [
        [cosine * px - sine * py + robot[0], sine * px + cosine * py + robot[1], 0.0]
        for px, py, _, _ in pedestrians
    ]
    velocities = [
        [cosine * vx - sine * vy, sine * vx + cosine * vy, 0.0] for _, _, vx, vy in pedestrians
    ]
    state = SimpleNamespace(
        positions_w=torch.tensor(world, dtype=torch.float64).reshape(1, count, 3),
        velocities_w=torch.tensor(velocities, dtype=torch.float64).reshape(1, count, 3),
        radii=torch.full((1, count), 0.25, dtype=torch.float64),
        valid=torch.ones((1, count), dtype=torch.bool),
    )
    data = SimpleNamespace(
        root_pos_w=torch.tensor([[robot[0], robot[1], 0.0]], dtype=torch.float64),
        root_quat_w=torch.tensor([_quaternion(yaw)], dtype=torch.float64),
        root_lin_vel_b=torch.tensor([[forward, 0.0, 0.0]], dtype=torch.float64),
        root_ang_vel_b=torch.tensor([[0.0, 0.0, yaw_rate]], dtype=torch.float64),
    )
    bridge = SimpleNamespace(state=state, config=SimpleNamespace(max_pedestrians=count))
    ranges = torch.tensor([scan if scan is not None else [4.0] * 720], dtype=torch.float64)
    return SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        episode_length_buf=torch.zeros(1, dtype=torch.long),
        scene={"robot": SimpleNamespace(data=data)},
        pedestrian_bridge=bridge,
        drl_vo_scan=ranges,
        drl_vo_goal=_Goal(distance, local),
    )


def _scan_minimum(scan: list[float]) -> float:
    hits = [value for value in scan if value > 0.0]
    return min(hits) if hits else 8.0


def test_reward_terms_reproduce_the_reference_over_a_trajectory():
    reference = _reference_module().OriginalReward(seed=1337)
    reference.rng = _RecordingRandom(1337)
    env = _environment()
    generator = torch.Generator().manual_seed(7)
    distance = 6.0
    for step in range(48):
        # Move towards the goal with a wandering bearing and a crowd that drifts.
        distance = max(0.05, distance - 0.06 - float(torch.rand((), generator=generator)) * 0.02)
        local_angle = float(torch.rand((), generator=generator) - 0.5) * 1.2
        local = (math.cos(local_angle), math.sin(local_angle))
        close = 0.15 if step in (20, 21, 22, 23) else 4.5
        scan = [0.0] * 719 + [close]
        pedestrians = [
            (1.5 + 0.1 * step, 0.6, -0.4, 0.0),
            (4.0, -1.2 + 0.05 * step, 0.0, 0.3),
        ]
        yaw = 0.3 + 0.05 * step
        forward = 0.25
        yaw_rate = 1.4 if step % 7 == 0 else 0.2
        env.scene["robot"].data.root_quat_w = torch.tensor([_quaternion(yaw)], dtype=torch.float64)
        env.scene["robot"].data.root_pos_w = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float64)
        env.scene["robot"].data.root_lin_vel_b = torch.tensor([[forward, 0.0, 0.0]], dtype=torch.float64)
        env.scene["robot"].data.root_ang_vel_b = torch.tensor([[0.0, 0.0, yaw_rate]], dtype=torch.float64)
        env.pedestrian_bridge.state.positions_w = torch.tensor(
            [
                [
                    [
                        math.cos(yaw) * px - math.sin(yaw) * py + 2.0,
                        math.sin(yaw) * px + math.cos(yaw) * py + 1.0,
                        0.0,
                    ]
                    for px, py, _, _ in pedestrians
                ]
            ],
            dtype=torch.float64,
        )
        env.pedestrian_bridge.state.velocities_w = torch.tensor(
            [
                [
                    [math.cos(yaw) * vx - math.sin(yaw) * vy, math.sin(yaw) * vx + math.cos(yaw) * vy, 0.0]
                    for _, _, vx, vy in pedestrians
                ]
            ],
            dtype=torch.float64,
        )
        env.pedestrian_bridge.state.valid = torch.ones((1, len(pedestrians)), dtype=torch.bool)
        env.pedestrian_bridge.state.radii = torch.full((1, len(pedestrians)), 0.25, dtype=torch.float64)
        env.pedestrian_bridge.config.max_pedestrians = len(pedestrians)
        env.drl_vo_scan = torch.tensor([scan], dtype=torch.float64)
        env.drl_vo_goal.set(distance, local)
        env.episode_length_buf = torch.tensor([step + 1], dtype=torch.long)

        recorder = reference.rng
        recorder.draws = []
        expected, done, reason, detail = reference.step(
            distance=distance,
            minimum_scan=_scan_minimum(scan),
            actual_v=forward,
            actual_w=yaw_rate,
            goal_base_xy=local,
            pedestrians_base_xy_vxy=pedestrians,
        )
        env.drl_vo_angle_source = lambda count, draws=tuple(recorder.draws): torch.tensor(
            list(draws) + [0.0] * (count - len(draws)), dtype=torch.float64
        )

        goal = float(terms.goal_reward(env))
        obstacle = float(terms.obstacle_reward(env))
        angular = float(terms.angular_reward(env))
        heading = float(terms.heading_reward(env))
        assert goal == pytest.approx(detail["goal"], abs=1e-5), step
        assert obstacle == pytest.approx(detail["obstacle"], abs=1e-5), step
        assert angular == pytest.approx(detail["angular"], abs=1e-5), step
        assert heading == pytest.approx(detail["vo"], abs=1e-5), step
        assert goal + obstacle + angular + heading == pytest.approx(expected, abs=1e-4), step

        arrived = bool(terms.goal_reached(env)[0])
        collision = bool(terms.collision_budget_exceeded(env)[0])
        budget = bool(terms.iteration_budget_exhausted(env)[0])
        observed = "goal" if arrived else "collision" if collision else "timeout" if budget else "running"
        assert observed == reason, (step, observed, reason)
        assert done == (reason != "running")


def test_iteration_budget_matches_the_source_boundary():
    reference = _reference_module().OriginalReward(seed=3)
    reference.rng = _RecordingRandom(3)
    env = _environment(distance=5.0)
    env.drl_vo_angle_source = lambda count: torch.zeros(count, dtype=torch.float64)
    reasons = {}
    for step in range(513):
        env.episode_length_buf = torch.tensor([step + 1], dtype=torch.long)
        _, _, reason, detail = reference.step(
            distance=5.0,
            minimum_scan=4.0,
            actual_v=0.0,
            actual_w=0.0,
            goal_base_xy=(1.0, 0.0),
            pedestrians_base_xy_vxy=[],
        )
        goal = float(terms.goal_reward(env))
        assert goal == pytest.approx(detail["goal"], abs=1e-6), step
        terminal = bool(terms.iteration_budget_exhausted(env)[0])
        reasons[step] = (reason, terminal, goal)
        if reason != "running":
            break
    last = max(reasons)
    assert reasons[last][0] == "timeout"
    assert reasons[last][1] is True
    assert reasons[last][2] == pytest.approx(-20.0)
    assert reasons[512][2] == pytest.approx(-20.0)


@pytest.mark.parametrize(
    ("minimum", "expected"),
    [
        (0.02, -20.0),
        (0.15, -20.0),
        (0.3, -20.0),
        (0.3001, pytest.approx(-0.2 * (0.9 - 0.3001))),
        (0.6, pytest.approx(-0.2 * 0.3)),
        (0.8999, pytest.approx(-0.2 * 0.0001)),
        (0.9, 0.0),
        (5.0, 0.0),
    ],
)
def test_obstacle_penalty_thresholds_match_the_source(minimum, expected):
    env = _environment(scan=[4.0] * 719 + [minimum])
    value = float(terms.obstacle_reward(env))
    if isinstance(expected, float):
        assert value == pytest.approx(expected, abs=1e-6)
    else:
        assert value == expected


def test_obstacle_penalty_ignores_misses_and_reports_the_cap_without_returns():
    env = _environment(scan=[0.0] * 720)
    assert float(terms.obstacle_reward(env)) == pytest.approx(0.0)
    env.drl_vo_scan = torch.tensor([[4.0] * 719 + [float("inf")]])
    assert float(terms.obstacle_reward(env)) == 0.0


def test_obstacle_penalty_scales_with_the_robot_radius():
    env = _environment(scan=[4.0] * 719 + [0.6])
    source = float(terms.obstacle_reward(env, robot_radius_m=0.3))
    migrated = float(terms.obstacle_reward(env, robot_radius_m=0.37))
    assert source == pytest.approx(-0.2 * (0.9 - 0.6))
    assert migrated == pytest.approx(-0.2 * (1.11 - 0.6))
    assert float(terms.obstacle_reward(_environment(scan=[4.0] * 719 + [0.37]), robot_radius_m=0.37)) == -20.0
    assert float(terms.obstacle_reward(_environment(scan=[4.0] * 719 + [0.45]), robot_radius_m=0.37)) < 0.0


@pytest.mark.parametrize(
    ("yaw_rate", "expected"),
    [(0.0, 0.0), (0.999, 0.0), (1.0, 0.0), (1.5, -0.15), (-2.0, -0.2)],
)
def test_angular_penalty_matches_the_source(yaw_rate, expected):
    env = _environment(yaw_rate=yaw_rate)
    assert float(terms.angular_reward(env)) == pytest.approx(expected)


def test_heading_reward_without_a_crowd_uses_the_goal_bearing():
    env = _environment(local=(math.cos(0.4), math.sin(0.4)))
    assert float(terms.heading_reward(env)) == pytest.approx(0.6 * (math.pi / 6 - 0.4))


def test_heading_reward_leaves_a_crowded_cone():
    # A pedestrian dead ahead blocks every heading inside its cone, so the
    # chosen heading must leave the goal bearing and lose part of the margin.
    env = _environment(local=(1.0, 0.0), pedestrians=[(2.0, 0.0, 0.0, 0.0)], forward=0.4)
    env.drl_vo_angle_source = lambda count: (torch.arange(count, dtype=torch.float64) / count * 2 - 1) * math.pi
    crowded = float(terms.heading_reward(env))
    clear = float(terms.heading_reward(_environment(local=(1.0, 0.0), forward=0.4)))
    assert clear == pytest.approx(0.6 * math.pi / 6)
    assert crowded < clear
    assert crowded > -0.6 * math.pi


def test_progress_uses_the_ten_step_ring():
    env = _environment(distance=5.0)
    env.drl_vo_angle_source = lambda count: torch.zeros(count, dtype=torch.float64)
    env.episode_length_buf = torch.tensor([1], dtype=torch.long)
    assert float(terms.goal_reward(env)) == pytest.approx(0.0)
    for step in range(2, 11):
        env.drl_vo_goal.set(5.0 - 0.1 * (step - 1), (1.0, 0.0))
        env.episode_length_buf = torch.tensor([step], dtype=torch.long)
        terms.goal_reward(env)
    # Iteration 10 compares against the sample from iteration 0, not iteration 9.
    env.drl_vo_goal.set(4.0, (1.0, 0.0))
    env.episode_length_buf = torch.tensor([11], dtype=torch.long)
    assert float(terms.goal_reward(env)) == pytest.approx(3.2 * (5.0 - 4.0))


def test_arrival_pays_the_arrival_constant_and_terminates():
    env = _environment(distance=0.29)
    assert float(terms.goal_reward(env)) == pytest.approx(20.0)
    assert bool(terms.goal_reached(env)[0]) is True
    env.drl_vo_goal.set(0.31, (1.0, 0.0))
    assert bool(terms.goal_reached(env)[0]) is False


def test_collision_counter_survives_a_goal_boundary_like_the_source():
    env = _environment(scan=[4.0] * 719 + [0.2])
    env.drl_vo_goal.set(6.0, (1.0, 0.0))
    assert bool(terms.collision_budget_exceeded(env)[0]) is False
    assert int(terms.episode_state(env).collisions[0]) == 1
    # A reset clears the distance ring only; the source keeps bump_num.
    terms.reset_episode_state(env)
    assert int(terms.episode_state(env).collisions[0]) == 1
    assert bool(terms.collision_budget_exceeded(env)[0]) is False
    assert bool(terms.collision_budget_exceeded(env)[0]) is True
    assert int(terms.episode_state(env).collisions[0]) == 0


def test_arrival_does_not_count_a_collision():
    env = _environment(scan=[4.0] * 719 + [0.2], distance=0.1)
    assert bool(terms.collision_budget_exceeded(env)[0]) is False
    assert int(terms.episode_state(env).collisions[0]) == 0
