# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Algorithm checks for the DRL-VO PPO port and its network."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pedstrain_simulation.simulation.rl import ppo as drl_ppo

CHECKPOINT = Path(__file__).resolve().parents[2] / "DRL-VO" / "frozen_policy.pt"


def test_gae_matches_a_hand_computed_three_step_case():
    buffer = drl_ppo.RolloutBuffer(3, 4, torch.device("cpu"), gamma=0.5, gae_lambda=0.5)
    values = [1.0, 2.0, 3.0]
    for index, value in enumerate(values):
        buffer.add(torch.zeros(4), torch.zeros(2), 1.0, index == 2, 0.0, value)
    buffer.compute_returns_and_advantages(last_value=0.0, last_done=True)
    assert buffer.advantages.tolist() == pytest.approx([1.0, 0.0, -2.0])
    assert buffer.returns.tolist() == pytest.approx([2.0, 2.0, 1.0])


def test_gae_bootstraps_when_the_rollout_ends_mid_episode():
    buffer = drl_ppo.RolloutBuffer(2, 4, torch.device("cpu"), gamma=0.5, gae_lambda=0.5)
    buffer.add(torch.zeros(4), torch.zeros(2), 1.0, False, 0.0, 1.0)
    buffer.add(torch.zeros(4), torch.zeros(2), 1.0, False, 0.0, 2.0)
    buffer.compute_returns_and_advantages(last_value=4.0, last_done=False)
    # t=1: 1 + 0.5 * 4 - 2 = 1 ; t=0: 1 + 0.5 * 2 - 1 + 0.25 * 1 = 1.25
    assert buffer.advantages.tolist() == pytest.approx([1.25, 1.0])
    assert buffer.returns.tolist() == pytest.approx([2.25, 3.0])


def test_policy_shapes_match_the_published_architecture():
    policy = drl_ppo.DrlVoPolicy()
    mean, value, std = policy(torch.zeros(1, drl_ppo.OBSERVATION_SIZE))
    assert tuple(mean.shape) == (1, 2)
    assert tuple(value.shape) == (1,)
    assert tuple(std.shape) == (1, 2)
    assert torch.allclose(std, torch.ones(1, 2))
    names = dict(policy.named_parameters())
    assert "features_extractor.conv1.weight" in names
    assert "mlp_extractor.policy_net.0.weight" in names
    assert "mlp_extractor.value_net.0.weight" in names
    assert "action_net.weight" in names
    assert "value_net.weight" in names
    assert "log_std" in names


def test_frozen_checkpoint_loads_without_key_mismatch():
    if not CHECKPOINT.is_file() or CHECKPOINT.stat().st_size < 1_000_000:
        pytest.skip("DRL-VO/frozen_policy.pt is not available in this checkout")
    policy = drl_ppo.DrlVoPolicy()
    report = drl_ppo.load_frozen_policy(CHECKPOINT, policy)
    assert report["missing"] == []
    assert report["unexpected"] == []


def test_update_requires_a_full_rollout():
    config = drl_ppo.PpoConfig(n_steps=2, batch_size=1, n_epochs=1)
    trainer = drl_ppo.DrlVoPPO(drl_ppo.DrlVoPolicy(), config, device=torch.device("cpu"))
    buffer = drl_ppo.RolloutBuffer(2, drl_ppo.OBSERVATION_SIZE, torch.device("cpu"), gamma=0.99, gae_lambda=0.95)
    with pytest.raises(RuntimeError):
        trainer.update(buffer)


def test_update_produces_finite_metrics_and_moves_the_actor():
    torch.manual_seed(0)
    config = drl_ppo.PpoConfig(n_steps=2, batch_size=2, n_epochs=2)
    policy = drl_ppo.DrlVoPolicy()
    trainer = drl_ppo.DrlVoPPO(policy, config, device=torch.device("cpu"), seed=1)
    buffer = drl_ppo.RolloutBuffer(2, drl_ppo.OBSERVATION_SIZE, torch.device("cpu"), gamma=0.99, gae_lambda=0.95)
    before = policy.action_net.weight.detach().clone()
    for index in range(2):
        buffer.add(torch.randn(drl_ppo.OBSERVATION_SIZE), torch.zeros(2), 1.0, index == 1, 0.0, 0.5)
    buffer.compute_returns_and_advantages(last_value=0.0, last_done=True)
    statistics = trainer.update(buffer)
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"):
        assert torch.isfinite(torch.tensor(statistics[key])), key
    assert not torch.equal(before, policy.action_net.weight.detach())
    assert trainer.steps_done == 2
    assert trainer.iteration == 1


def test_configuration_rejects_inconsistent_sizes():
    with pytest.raises(ValueError):
        drl_ppo.PpoConfig(n_steps=8, batch_size=16).validate()
    with pytest.raises(ValueError):
        drl_ppo.PpoConfig(gamma=1.5).validate()
    drl_ppo.PpoConfig().validate()


def test_migrated_defaults_match_the_drl_vo_training_config():
    config = drl_ppo.PpoConfig()
    assert config.n_steps == 512
    assert config.batch_size == 128
    assert config.n_epochs == 10
    assert config.learning_rate == 1e-3
    assert config.gamma == 0.99
    assert config.gae_lambda == 0.95
    assert config.clip_range == 0.2
    assert config.entropy_coefficient == 0.0
    assert config.value_coefficient == 0.5
    assert config.max_grad_norm == 0.5
    assert config.adam_epsilon == 1e-5
