# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-environment PPO training loop for the DRL-VO navigation task.

The loop is deliberately thin: the environment owns the simulator, the policy
owns the network, and this module owns only bookkeeping — rollout collection,
GAE, the PPO update call, episode statistics and checkpointing. It reuses the
DRL-VO reward and termination semantics through
:mod:`.rewards`, so a change of reward never touches the loop.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .ppo import DrlVoPolicy, DrlVoPPO, PpoConfig, RolloutBuffer, explained_variance


@dataclass
class TrainingLog:
    """Running episode statistics for one log record."""

    returns: list[float] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    terminated: int = 0
    truncated: int = 0
    current_return: float = 0.0
    current_length: int = 0

    def reset(self) -> None:
        """Drop every counter, including the episode in progress."""
        self.reset_window()
        self.current_return = 0.0
        self.current_length = 0

    def reset_window(self) -> None:
        """Drop the finished-episode window but keep the episode in progress."""
        self.returns.clear()
        self.lengths.clear()
        self.terminated = 0
        self.truncated = 0

    def summary(self) -> dict[str, object]:
        return {
            "episode_return": sum(self.returns) / len(self.returns) if self.returns else None,
            "episode_length": sum(self.lengths) / len(self.lengths) if self.lengths else None,
            "episodes": len(self.returns),
            "terminated": self.terminated,
            "truncated": self.truncated,
        }


def policy_observations(observation: object, device: torch.device) -> torch.Tensor:
    """Flatten a manager observation into a ``[1, N]`` float tensor."""
    value = observation["policy"] if isinstance(observation, dict) else observation
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return tensor.reshape(1, -1).to(device=device, dtype=torch.float32)


def train(
    env: object,
    *,
    total_steps: int,
    output: Path,
    config: PpoConfig | None = None,
    seed: int = 1337,
    device: torch.device | None = None,
    warm_start: Path | None = None,
    learning_rate_decay: tuple[float, float] | None = None,
    log_every_iterations: int = 1,
    save_every_iterations: int = 0,
    tag: str = "drl_vo",
) -> dict[str, object]:
    """Run PPO on ``env`` and write metrics plus checkpoints under ``output``.

    Args:
        env: A ``ManagerBasedRLEnv`` whose observation group is ``policy``.
        total_steps: Environment steps to collect, rounded down to a multiple
            of ``config.n_steps``.
        output: Directory for ``metrics.jsonl``, ``run.json`` and checkpoints.
        config: PPO settings; the DRL-VO defaults are used when omitted.
        seed: Seed for torch, the environment and the minibatch order.
        device: Torch device; the environment device by default.
        warm_start: Optional checkpoint (``frozen_policy.pt`` or a file written
            by :meth:`DrlVoPPO.save`) to initialise the policy from.
        learning_rate_decay: Optional ``(start, end)`` linear learning-rate
            schedule over the run.
        log_every_iterations: Write one metrics record every N PPO iterations.
        save_every_iterations: Write a checkpoint every N iterations; 0 saves
            only at the end.
        tag: Prefix for checkpoint file names.

    Returns:
        A summary dictionary, also written to ``output/run.json``.
    """
    config = config or PpoConfig()
    config.validate()
    torch.manual_seed(int(seed))
    device = device or torch.device(str(getattr(env, "device", "cpu")))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    policy = DrlVoPolicy()
    trainer = DrlVoPPO(policy, config, device=device, seed=seed)
    warm_start_report: dict[str, object] | None = None
    if warm_start is not None:
        payload = torch.load(Path(warm_start), map_location=device, weights_only=False)
        if "policy" in payload:
            trainer.load(warm_start)
            warm_start_report = {"source": str(warm_start), "format": "trainer-checkpoint"}
        else:
            from .ppo import load_frozen_policy

            report = load_frozen_policy(warm_start, policy, device=device)
            if report["missing"] or report["unexpected"]:
                raise RuntimeError(f"Warm start key mismatch: {report}")
            warm_start_report = {
                "source": str(warm_start),
                "format": "frozen-policy",
                "loaded": int(report.get("loaded", 0)),
            }

    buffer = RolloutBuffer(
        config.n_steps, 19202, device, gamma=config.gamma, gae_lambda=config.gae_lambda
    )
    steps = max(config.n_steps, (int(total_steps) // config.n_steps) * config.n_steps)
    iterations = steps // config.n_steps
    base_learning_rate = config.learning_rate

    log = TrainingLog()
    metrics = (output / "metrics.jsonl").open("w", buffering=1)
    started = time.monotonic()
    observation = policy_observations(env.reset()[0], device)
    last_record: dict[str, object] = {}
    try:
        for iteration in range(1, iterations + 1):
            if learning_rate_decay is not None:
                start, end = learning_rate_decay
                fraction = (iteration - 1) / max(iterations - 1, 1)
                for group in trainer.optimizer.param_groups:
                    group["lr"] = start + (end - start) * fraction
            buffer.reset()
            for _ in range(config.n_steps):
                action, value, log_probability = trainer.act(observation)
                applied = torch.clamp(action, -1.0, 1.0)
                next_observation, reward, terminated, truncated, _ = env.step(applied)
                done = bool(terminated[0]) or bool(truncated[0])
                buffer.add(
                    observation[0].detach(), action[0].detach(), float(reward[0]), done,
                    float(log_probability[0]), float(value[0]),
                )
                log.current_return += float(reward[0])
                log.current_length += 1
                if done:
                    log.returns.append(log.current_return)
                    log.lengths.append(log.current_length)
                    log.terminated += int(bool(terminated[0]))
                    log.truncated += int(bool(truncated[0]))
                    log.current_return, log.current_length = 0.0, 0
                observation = policy_observations(next_observation, device)
                if trainer.steps_done + buffer.position >= steps:
                    break

            with torch.no_grad():
                last_value = trainer.value(observation)
            buffer.compute_returns_and_advantages(
                float(last_value[0]), last_done=bool(done) if len(buffer) else False
            )
            statistics = trainer.update(buffer)
            statistics["iteration"] = iteration
            statistics["step"] = trainer.steps_done
            statistics["elapsed_s"] = time.monotonic() - started
            statistics["steps_per_s"] = trainer.steps_done / max(statistics["elapsed_s"], 1e-9)
            statistics["explained_variance"] = explained_variance(buffer.values, buffer.returns)
            statistics["learning_rate"] = trainer.optimizer.param_groups[0]["lr"]
            statistics.update(log.summary())
            log.reset_window()
            last_record = statistics
            if iteration % max(log_every_iterations, 1) == 0:
                metrics.write(json.dumps(statistics) + "\n")
            if save_every_iterations and iteration % save_every_iterations == 0:
                trainer.save(output / f"{tag}_iter{iteration:05d}.pt")
            if trainer.steps_done >= steps:
                break
    finally:
        metrics.close()

    checkpoint = trainer.save(output / f"{tag}_final.pt")
    summary = {
        "tag": tag,
        "steps": trainer.steps_done,
        "iterations": trainer.iteration,
        "checkpoint": str(checkpoint),
        "base_learning_rate": base_learning_rate,
        "warm_start": warm_start_report,
        "config": config.__dict__.copy(),
        "last_record": last_record,
    }
    (output / "run.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def evaluate(env: object, policy: DrlVoPolicy, *, episodes: int, device: torch.device) -> dict[str, float]:
    """Roll out the deterministic policy and summarise the episode returns.

    Args:
        env: The navigation environment.
        policy: Trained actor-critic; the distribution mean is applied.
        episodes: Number of episode boundaries to collect.
        device: Torch device of the policy.

    Returns:
        Mean return, mean length and the arrival rate. Arrival is inferred from
        the final step reward: the arrival constant is 20 while the other three
        terms stay below one, so half of it is a safe discriminator.
    """
    policy.eval()
    observation = policy_observations(env.reset()[0], device)
    returns, lengths, successes = [], [], 0
    total, length = 0.0, 0
    finished = 0
    while finished < episodes:
        with torch.no_grad():
            mean, _, _ = policy(observation)
        observation, reward, terminated, truncated, _ = env.step(torch.clamp(mean, -1.0, 1.0))
        last_reward = float(reward[0])
        total += last_reward
        length += 1
        if bool(terminated[0]) or bool(truncated[0]):
            returns.append(total)
            lengths.append(length)
            successes += int(float(last_reward) >= 0.5 * 20.0)
            total, length = 0.0, 0
            finished += 1
        observation = policy_observations(observation, device)
    return {
        "return": sum(returns) / len(returns),
        "length": sum(lengths) / len(lengths),
        "success_rate": successes / max(len(returns), 1),
    }
