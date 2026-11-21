# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO with the DRL-VO hyper-parameters and feature extractor.

``DRL-VO/TRAINING_CONFIG.json`` pins the algorithm to Stable-Baselines3 0.10.0
PPO with ``CustomCNN(features_dim=256)``, ``net_arch=[dict(pi=[256], vf=[128])]``,
``n_steps=512``, ``batch_size=128``, ``n_epochs=10``, ``learning_rate=1e-3``,
``gamma=0.99``, ``gae_lambda=0.95``, ``clip_range=0.2``, ``entropy=0``,
``value_coefficient=0.5``, ``max_grad_norm=0.5`` and ``Adam(eps=1e-5)``.

That engine cannot be installed next to Isaac Lab, so this module reimplements
the same algorithm and keeps the module layout identical to SB3's
``ActorCriticCnnPolicy``. Identical attribute names let
:func:`load_checkpoint` lift ``frozen_policy.pt`` weights straight into the
Isaac Lab trainer for warm starts, and :func:`save_checkpoint` writes a
``state_dict`` that maps back the same way.
"""

from __future__ import annotations

import math
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as functional

OBSERVATION_SIZE = 19202
ACTION_SIZE = 2
FEATURES_DIM = 256


# ---------------------------------------------------------------------------
# Network: a direct port of DRL-VO/reference/custom_cnn_full.py
# ---------------------------------------------------------------------------


def _conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups,
                     bias=False, dilation=dilation)


def _conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    """The DRL-VO residual block with ``expansion = 2``."""

    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = _conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = _conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = _conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class CustomCNN(nn.Module):
    """The published DRL-VO fusion network (pedestrian map + scan + goal)."""

    def __init__(self, features_dim: int = FEATURES_DIM) -> None:
        super().__init__()
        block, layers = Bottleneck, [2, 1, 1]
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.features_dim = features_dim
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(1, 1)), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(1, 1)), nn.BatchNorm2d(256),
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2)), nn.BatchNorm2d(256)
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(1, 1)), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=(1, 1)), nn.BatchNorm2d(512),
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(64, 512, kernel_size=(1, 1), stride=(4, 4)), nn.BatchNorm2d(512)
        )
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_fc = nn.Sequential(nn.Linear(256 * block.expansion + 2, features_dim), nn.ReLU())
        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion)
            )
        layers = [block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, self.dilation,
                        norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                                dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
        # Zero-initialize the last BN in each residual branch, as in the source.
        for module in self.modules():
            if isinstance(module, Bottleneck):
                nn.init.constant_(module.bn3.weight, 0)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        ped_pos = observations[:, :12800]
        scan = observations[:, 12800:19200]
        goal = observations[:, 19200:]
        fusion = torch.cat((scan.reshape(-1, 1, 80, 80), ped_pos.reshape(-1, 2, 80, 80)), dim=1)
        x = self.maxpool(self.relu(self.bn1(self.conv1(fusion))))
        identity3 = self.downsample3(x)
        x = self.layer1(x)
        identity2 = self.downsample2(x)
        x = self.layer2(x)
        x = self.relu2(self.conv2_2(x) + identity2)
        x = self.layer3(x)
        x = self.relu3(self.conv3_2(x) + identity3)
        fusion_out = torch.flatten(self.avgpool(x), 1)
        return self.linear_fc(torch.cat((fusion_out, goal.reshape(-1, 2)), dim=1))


class MlpExtractor(nn.Module):
    """``MlpExtractor`` clone with SB3's ``policy_net``/``value_net`` names."""

    def __init__(self, feature_dim: int = FEATURES_DIM, pi_layers=(256,), vf_layers=(128,)) -> None:
        super().__init__()
        self.latent_dim_pi = pi_layers[-1]
        self.latent_dim_vf = vf_layers[-1]
        self.policy_net = self._build(feature_dim, pi_layers)
        self.value_net = self._build(feature_dim, vf_layers)

    @staticmethod
    def _build(feature_dim: int, layers) -> nn.Sequential:
        modules, last = [], feature_dim
        for width in layers:
            modules.extend((nn.Linear(last, width), nn.Tanh()))
            last = width
        return nn.Sequential(*modules)


class DrlVoPolicy(nn.Module):
    """Actor-critic with SB3 ``ActorCriticCnnPolicy`` module names."""

    def __init__(self, features_dim: int = FEATURES_DIM, pi_layers=(256,), vf_layers=(128,)) -> None:
        super().__init__()
        self.features_extractor = CustomCNN(features_dim)
        self.mlp_extractor = MlpExtractor(features_dim, pi_layers, vf_layers)
        self.action_net = nn.Linear(self.mlp_extractor.latent_dim_pi, ACTION_SIZE)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        self.log_std = nn.Parameter(torch.zeros(ACTION_SIZE))
        # Bounded noise. Unbounded, sigma walked from 1.0 to 1.6 within
        # eighteen iterations, which clamps most sampled actions to +-1 while
        # evaluation runs the mean - the agent then trains on a distribution
        # it never acts on. The trainer lowers the ceiling over time.
        self.std_low = 0.08
        self.std_high = 0.90
        self._range = (-1.0, 1.0)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(mean, value, std)`` for a batch of observations."""
        features = self.features_extractor(observations)
        mean = self.action_net(self.mlp_extractor.policy_net(features))
        value = self.value_net(self.mlp_extractor.value_net(features)).squeeze(-1)
        std = self.log_std.clamp(math.log(self.std_low), math.log(self.std_high)).exp()
        return mean, value, std.expand_as(mean)

    def set_std_bounds(self, low: float, high: float) -> None:
        """Clamp the exploration noise to ``[low, high]`` standard deviations."""
        if not 0.0 < low <= high:
            raise ValueError(f"std bounds must satisfy 0 < low <= high, got {low}, {high}")
        self.std_low = float(low)
        self.std_high = float(high)

    def distribution(self, observations: torch.Tensor) -> tuple[torch.distributions.Normal, torch.Tensor]:
        """Return the diagonal Gaussian and the state value."""
        mean, value, std = self(observations)
        return torch.distributions.Normal(mean, std), value


# ---------------------------------------------------------------------------
# Rollout and optimisation
# ---------------------------------------------------------------------------


@dataclass
class PpoConfig:
    """PPO settings copied from ``DRL-VO/TRAINING_CONFIG.json``."""

    n_steps: int = 512
    batch_size: int = 128
    n_epochs: int = 10
    learning_rate: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coefficient: float = 0.0
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    adam_epsilon: float = 1e-5
    normalize_advantage: bool = True
    target_kl: float | None = None
    # Transitions one rollout holds. ``None`` keeps the single-environment reading
    # (``n_steps``); a vectorised rollout sets ``n_steps * num_envs`` so a minibatch
    # may span environments.
    sample_count: int | None = None

    def validate(self) -> None:
        if self.n_steps < 1 or self.batch_size < 1 or self.n_epochs < 1:
            raise ValueError("n_steps, batch_size and n_epochs must be positive")
        limit = self.n_steps if self.sample_count is None else int(self.sample_count)
        if self.batch_size > limit:
            raise ValueError("batch_size cannot exceed the rollout sample count")
        if not 0.0 <= self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must be in [0, 1]")


class RolloutBuffer:
    """Fixed-size on-policy storage for one PPO iteration."""

    def __init__(self, size: int, observation_size: int, device: torch.device, *, gamma: float, gae_lambda: float) -> None:
        self.size = size
        self.device = device
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.observations = torch.zeros((size, observation_size), dtype=torch.float32, device=device)
        self.actions = torch.zeros((size, ACTION_SIZE), dtype=torch.float32, device=device)
        self.log_probabilities = torch.zeros((size,), dtype=torch.float32, device=device)
        self.values = torch.zeros((size,), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((size,), dtype=torch.float32, device=device)
        self.dones = torch.zeros((size,), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((size,), dtype=torch.float32, device=device)
        self.returns = torch.zeros((size,), dtype=torch.float32, device=device)
        self.position = 0

    def __len__(self) -> int:
        return self.position

    @property
    def full(self) -> bool:
        return self.position >= self.size

    def reset(self) -> None:
        self.position = 0

    def add(self, observation, action, reward, done, log_probability, value) -> None:
        """Append one transition; ``done`` marks the end of the episode at this step."""
        index = self.position
        self.observations[index] = observation
        self.actions[index] = action
        self.rewards[index] = float(reward)
        self.dones[index] = float(bool(done))
        self.log_probabilities[index] = float(log_probability)
        self.values[index] = float(value)
        self.position += 1

    def compute_returns_and_advantages(self, last_value: float, last_done: bool) -> None:
        """Fill ``advantages``/``returns`` with GAE(lambda) over the stored steps.

        Args:
            last_value: Value of the observation that follows the rollout.
            last_done: Whether that following observation starts a new episode.
        """
        advantage = 0.0
        for index in reversed(range(self.position)):
            if index == self.position - 1:
                next_value, terminal = float(last_value), 1.0 - float(bool(last_done))
            else:
                next_value, terminal = float(self.values[index + 1]), 1.0 - float(self.dones[index])
            delta = float(self.rewards[index]) + self.gamma * next_value * terminal - float(self.values[index])
            advantage = delta + self.gamma * self.gae_lambda * terminal * advantage
            self.advantages[index] = advantage
            self.returns[index] = advantage + float(self.values[index])


class DrlVoPPO:
    """PPO trainer matching the DRL-VO training configuration.

    Args:
        policy: Actor-critic to optimise.
        config: PPO settings.
        device: Torch device for the policy and the rollout.
        seed: Seed for the minibatch permutations.
        tensorboard: Unused placeholder kept for call-site symmetry.
    """

    def __init__(self, policy: DrlVoPolicy, config: PpoConfig, *, device: torch.device, seed: int = 0) -> None:
        config.validate()
        self.policy = policy.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.learning_rate, eps=config.adam_epsilon
        )
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.steps_done = 0
        self.iteration = 0

    # -- inference ------------------------------------------------------

    def set_std_bounds(self, low: float, high: float) -> None:
        """Forward the exploration-noise bounds to the policy."""
        self.policy.set_std_bounds(low, high)

    @torch.no_grad()
    def act(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an unclipped action with its value and log-probability."""
        self.policy.eval()
        distribution, value = self.policy.distribution(observation)
        action = distribution.sample()
        return action, value, distribution.log_prob(action).sum(dim=-1)

    @torch.no_grad()
    def value(self, observation: torch.Tensor) -> torch.Tensor:
        self.policy.eval()
        return self.policy.distribution(observation)[1]

    # -- optimisation ---------------------------------------------------

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Run ``n_epochs`` of clipped-surrogate updates over the rollout.

        Args:
            buffer: A full rollout with advantages already computed.

        Returns:
            Scalar diagnostics for this iteration.
        """
        if not buffer.full:
            raise RuntimeError("PPO update requires a full rollout buffer")
        self.policy.train()
        count = buffer.position
        total_policy = total_value = total_entropy = total_kl = total_clip = 0.0
        batches = 0
        stopped_early = False
        target_kl = self.config.target_kl
        for _ in range(self.config.n_epochs):
            permutation = torch.randperm(count, generator=self.generator)
            for start in range(0, count, self.config.batch_size):
                index = permutation[start : start + self.config.batch_size].to(self.device)
                observations = buffer.observations[index]
                actions = buffer.actions[index]
                old_log_probability = buffer.log_probabilities[index]
                old_values = buffer.values[index]
                advantages = buffer.advantages[index]
                if self.config.normalize_advantage and len(index) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                distribution, values = self.policy.distribution(observations)
                log_probability = distribution.log_prob(actions).sum(dim=-1)
                entropy = distribution.entropy().sum(dim=-1).mean()
                ratio = (log_probability - old_log_probability).exp()
                surrogate = torch.max(
                    -advantages * ratio,
                    -advantages * ratio.clamp(1.0 - self.config.clip_range, 1.0 + self.config.clip_range),
                ).mean()
                value_loss = functional.mse_loss(values, buffer.returns[index])
                loss = surrogate + self.config.value_coefficient * value_loss - (
                    self.config.entropy_coefficient * entropy
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.config.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    approximate_kl = ((ratio - 1.0) - (log_probability - old_log_probability)).mean()
                    clipped = ((ratio - 1.0).abs() > self.config.clip_range).float().mean()
                total_policy += float(surrogate)
                total_value += float(value_loss)
                total_entropy += float(entropy)
                total_kl += float(approximate_kl)
                total_clip += float(clipped)
                batches += 1
                # A run whose KL drifts from 0.005 to 0.25 and clips six of every
                # ten samples is not learning, it is thrashing. Standard PPO
                # guard: once the policy has moved 1.5x past the budget, stop
                # spending epochs on stale off-policy data.
                if target_kl is not None and float(approximate_kl) > 1.5 * float(target_kl):
                    stopped_early = True
                    break
            del old_values
            if stopped_early:
                break
        self.iteration += 1
        self.steps_done += count
        return {
            "policy_loss": total_policy / max(batches, 1),
            "value_loss": total_value / max(batches, 1),
            "entropy": total_entropy / max(batches, 1),
            "approx_kl": total_kl / max(batches, 1),
            "clip_fraction": total_clip / max(batches, 1),
            "transitions": float(count),
            "epochs": float(self.config.n_epochs),
            "batches": float(batches),
            "early_stop": float(stopped_early),
        }

    # -- persistence ----------------------------------------------------

    def state_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
            "iteration": self.iteration,
            "config": self.config.__dict__.copy(),
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), target)
        return target

    def load(self, path: str | Path, *, load_optimizer: bool = True) -> None:
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.policy.load_state_dict(payload["policy"])
        if load_optimizer and "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.steps_done = int(payload.get("steps_done", 0))
        self.iteration = int(payload.get("iteration", 0))


def _load_torch_object(source: Path, device: torch.device) -> object:
    """Read a checkpoint that is either a Stable-Baselines3 zip or a ``torch.save`` dump."""
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            if "policy.pth" in archive.namelist():  # Stable-Baselines3 layout
                import io

                with io.BytesIO(archive.read("policy.pth")) as buffer:
                    return _torch_load(buffer, device)
    return _torch_load(source, device)


def _torch_load(target: object, device: torch.device) -> object:
    """``torch.load`` preferring the restricted unpickler, falling back for old dumps."""
    try:
        return torch.load(target, map_location=device, weights_only=True)
    except Exception:  # noqa: BLE001 - checkpoints written by older torch/pickle versions
        if hasattr(target, "seek"):
            target.seek(0)  # type: ignore[union-attr]
        return torch.load(target, map_location=device, weights_only=False)


def _unwrap_state_dict(payload: object) -> dict[str, torch.Tensor]:
    """Extract a ``{name: tensor}`` mapping from a checkpoint payload."""
    if isinstance(payload, torch.nn.Module):
        return dict(payload.state_dict())
    if not isinstance(payload, dict):
        raise RuntimeError(f"unsupported checkpoint payload: {type(payload).__name__}")
    if payload and all(torch.is_tensor(value) for value in payload.values()):
        return dict(payload)  # type: ignore[arg-type]
    for key in ("policy", "state_dict", "model", "actor_critic"):
        inner = payload.get(key)
        if isinstance(inner, dict) and inner and all(torch.is_tensor(value) for value in inner.values()):
            return dict(inner)  # type: ignore[arg-type]
    tensors = {key: value for key, value in payload.items() if torch.is_tensor(value)}
    if tensors:
        return tensors
    raise RuntimeError("checkpoint payload does not contain a state dict")


def load_frozen_policy(
    path: str | Path, policy: DrlVoPolicy, *, device: torch.device = torch.device("cpu")
) -> dict[str, list[str]]:
    """Load the published ``frozen_policy.pt`` weights into ``policy``.

    The published checkpoint is a plain ``torch.save`` of the actor-critic
    ``state_dict`` written by the reference ``CustomCNN``; a Stable-Baselines3
    ``.zip`` holding ``policy.pth`` is accepted as well. Both keep the original
    module names, so the tensors map one to one onto :class:`DrlVoPolicy`.

    Args:
        path: Checkpoint file, normally ``DRL-VO/frozen_policy.pt``.
        policy: Target actor-critic.
        device: Device used while reading the tensors.

    Returns:
        The ``missing`` and ``unexpected`` key lists reported by
        ``load_state_dict`` (both empty for a matching architecture) and how many
        tensors the checkpoint carried.
    """
    source = Path(path)
    weights = _unwrap_state_dict(_load_torch_object(source, device))
    report = policy.load_state_dict(weights, strict=False)
    return {
        "missing": list(report.missing_keys),
        "unexpected": list(report.unexpected_keys),
        "loaded": len(weights),
    }


def explained_variance(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Return ``1 - Var(target - prediction) / Var(target)``."""
    if predictions.numel() < 2:
        return float("nan")
    residual = torch.var(targets - predictions)
    total = torch.var(targets)
    if float(total) == 0.0:
        return float("nan")
    return float(1.0 - residual / total)
