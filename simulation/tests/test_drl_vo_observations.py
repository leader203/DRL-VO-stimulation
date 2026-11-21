# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Layout and normalization checks for the frozen DRL-VO observation."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from pedstrain_simulation.simulation.rl import drl_observations as obs


class _Goal:
    def __init__(self, local: tuple[float, float]) -> None:
        self.local_goal_base_xy = torch.tensor([local], dtype=torch.float32)
        self.distance_m = torch.tensor([2.0], dtype=torch.float32)

    def update(self, env: object) -> None:
        del env

    def reset(self, env: object = None, env_ids: object = None) -> None:
        del env, env_ids


def _environment(pedestrians=(), scan: float = 15.0, local: tuple[float, float] = (0.5, 0.0)):
    count = len(pedestrians)
    positions = [[px, py, 0.0] for (px, py), _ in pedestrians]
    velocities = [[vx, vy, 0.0] for _, (vx, vy) in pedestrians]
    state = SimpleNamespace(
        positions_w=torch.tensor(positions, dtype=torch.float32).reshape(1, count, 3),
        velocities_w=torch.tensor(velocities, dtype=torch.float32).reshape(1, count, 3),
        radii=torch.full((1, count), 0.25, dtype=torch.float32),
        valid=torch.ones((1, count), dtype=torch.bool),
    )
    data = SimpleNamespace(
        root_pos_w=torch.zeros(1, 3),
        root_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
    )
    return SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        common_step_counter=0,
        episode_length_buf=torch.zeros(1, dtype=torch.long),
        scene={"robot": SimpleNamespace(data=data)},
        pedestrian_bridge=SimpleNamespace(state=state, config=SimpleNamespace(max_pedestrians=count)),
        drl_vo_scan=torch.full((1, obs.SCAN_BEAMS), scan),
        drl_vo_goal=_Goal(local),
    )


def _prime_history(env: object, value: float = 15.0) -> None:
    """Fill the scan history and mark it current so no push happens this step."""
    state = obs.observation_state(env)
    state.scans.fill_(float(value))
    state.filled = obs.SCAN_HISTORY
    state.stamp = (int(env.common_step_counter), int(env.episode_length_buf[0]))


def test_observation_matches_the_frozen_layout_and_normalization():
    env = _environment(pedestrians=[((2.5, 0.0), (1.0, 0.0))])
    _prime_history(env, 15.0)
    observation = obs.drl_vo_observations(env)
    assert tuple(observation.shape) == (1, obs.OBSERVATION_SIZE) == (1, 19202)

    # Pedestrian map: 2 x 80 x 80, x forward at 0.25 m per cell, y to the left.
    # Channel 0 marks the cell, channel 1 carries the walker speed; both are
    # scaled by 0.5 on the way out.
    row = int(2.5 / 0.25)
    column = int((10.0 - 0.0) / 0.25)
    index = row * 80 + column
    assert observation[0, index].item() == pytest.approx(1.0 / 2.0)
    assert observation[0, 6400 + index].item() == pytest.approx(1.0 / 2.0)

    # ``_prime_history`` writes the ring directly, so this raw 15 m reading skips
    # the miss substitution and packs at 2 * 15 / 8 - 1 = 2.75.
    assert torch.allclose(observation[0, 12800:19200], torch.full((6400,), 2.75))

    # The local goal (0.5, 0) normalizes to 2 * (0.5 + 2) / 4 - 1 = 0.25.
    assert observation[0, 19200].item() == pytest.approx(0.25)
    assert observation[0, 19201].item() == pytest.approx(0.0)


def test_a_miss_packs_as_the_maximum_range():
    """A beam with no return must not read as an obstacle against the bumper."""
    env = _environment(scan=0.0)
    env.common_step_counter = 3
    observation = obs.drl_vo_observations(env)
    # A 0 m return means nothing is inside range; it packs as the 8 m maximum:
    # 2 * 8 / 8 - 1 = 1.
    assert torch.allclose(observation[0, 12800:19200], torch.ones(6400))


def test_scan_block_uses_the_min_and_mean_of_nine_beam_groups():
    env = _environment(scan=3.0)
    _prime_history(env, 3.0)
    observation = obs.drl_vo_observations(env)
    expected = 2.0 * 3.0 / 8.0 - 1.0
    assert torch.allclose(observation[0, 12800:19200], torch.full((6400,), expected))

    state = obs.observation_state(env)
    state.scans[0] = torch.arange(obs.SCAN_BEAMS, dtype=torch.float32)
    observation = obs.drl_vo_observations(env)
    block = observation[0, 12800:19200].reshape(4, 1600)
    assert torch.equal(block[0], block[3])
    # Group zero holds beams 0..8: minimum 0 m, mean 4 m. A raw 0 written
    # straight into the ring skips the miss substitution, so the minimum still
    # packs as -1; only the scale moved from 30 m to 8 m.
    assert block[0, 0].item() == pytest.approx(-1.0)
    assert block[0, 80].item() == pytest.approx(2.0 * 4.0 / 8.0 - 1.0)


def test_scan_history_keeps_the_ten_latest_frames():
    env = _environment(scan=1.0)
    state = obs.observation_state(env)
    for frame in range(1, 12):
        env.drl_vo_scan = torch.full((1, 720), float(frame))
        env.common_step_counter = frame
        obs.drl_vo_observations(env)
    assert state.filled == obs.SCAN_HISTORY
    # Frames 1 and 2 were pushed out; the oldest retained frame is 2 m.
    assert float(state.scans[0, 0, 0]) == pytest.approx(2.0)
    assert float(state.scans[0, -1, 0]) == pytest.approx(11.0)


def test_observation_state_resets_with_the_episode():
    env = _environment(scan=4.0)
    obs.drl_vo_observations(env)
    obs.reset_observation_state(env)
    state = obs.observation_state(env)
    assert state.filled == 0
    assert float(state.scans.abs().sum()) == 0.0


def test_pedestrians_outside_the_map_are_dropped():
    env = _environment(pedestrians=[((-1.0, 0.0), (1.0, 0.0)), ((30.0, 0.0), (1.0, 0.0))])
    _prime_history(env, 15.0)
    observation = obs.drl_vo_observations(env)
    assert float(observation[0, :12800].abs().sum()) == 0.0


def test_scan_angles_match_the_legacy_front_fan():
    angles = obs.scan_angles()
    assert tuple(angles.shape) == (1, 720)
    assert angles[0, 0].item() == pytest.approx(-math.pi / 2)
    assert angles[0, 359].item() == pytest.approx(-math.pi / 720, abs=1e-6)
    assert float(angles[0, -1]) == pytest.approx(math.pi / 2 - math.pi / 720)
