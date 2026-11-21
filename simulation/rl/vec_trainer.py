# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vectorised PPO training and protocol evaluation for the cloned-room task.

:mod:`.trainer` steps one environment and stores one rollout row per step. Here
``num_envs`` robots advance together, so the rollout is a ``[n_steps, num_envs]``
grid flattened to ``n_steps * num_envs`` transitions for the same
:class:`.ppo.DrlVoPPO` update — the algorithm, the hyper-parameters and the network
are untouched, only the sample layout changes.

Episodes are scored with the published protocol rather than with the training
signal: an episode succeeds when the robot arrives inside the goal tolerance
without ever touching an obstacle, within the time limit. Every finished episode
is classified as arrival, collision or timeout, and the success rate is the share
of the first kind.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .ppo import DrlVoPolicy, DrlVoPPO, PpoConfig, explained_variance

OBSERVATION_SIZE = 19202
TERM_ARRIVAL = "arrival"
TERM_COLLISION = "collision"
TERM_PEDESTRIAN = "pedestrian"
TERM_TIMEOUT = "budget"


def observations_tensor(observation: object, device: torch.device) -> torch.Tensor:
    """Return the policy observation group as a ``[num_envs, 19202]`` float tensor."""
    value = observation["policy"] if isinstance(observation, dict) else observation
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return tensor.reshape(int(tensor.shape[0]) if tensor.dim() > 1 else 1, -1).to(device=device, dtype=torch.float32)


def termination_flags(env: object) -> dict[str, torch.Tensor]:
    """Return the ``arrival``/``collision``/``timeout`` terms of the last step.

    Args:
        env: A manager-based environment.

    Returns:
        Mapping from term name to a ``[num_envs]`` boolean tensor; a missing manager
        yields all-false flags so a caller without terminations still runs.
    """
    manager = getattr(env, "termination_manager", None)
    flags: dict[str, torch.Tensor] = {}
    for name in (TERM_ARRIVAL, TERM_COLLISION, TERM_PEDESTRIAN, TERM_TIMEOUT):
        if manager is None:
            flags[name] = torch.zeros(int(env.num_envs), dtype=torch.bool)
            continue
        try:
            term = manager.get_term(name)
        except (KeyError, ValueError):
            # Rooms without pedestrians simply have no such term.
            flags[name] = torch.zeros(int(env.num_envs), dtype=torch.bool)
            continue
        flags[name] = term if isinstance(term, torch.Tensor) else torch.as_tensor(term)
    return flags


@dataclass
class EpisodeOutcome:
    """One finished episode, in protocol terms."""

    length: int
    time_s: float
    episode_return: float
    arrived: bool
    collided: bool
    timed_out: bool
    pedestrian: bool = False

    @property
    def success(self) -> bool:
        """Arrival without any contact, within the time limit."""
        return bool(self.arrived and not self.collided)


@dataclass
class EpisodeTracker:
    """Accumulate episodes across all environments and summarise a window."""

    num_envs: int
    step_dt_s: float
    records: list[EpisodeOutcome] = field(default_factory=list)
    current_return: list[float] = field(default_factory=list)
    current_length: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.current_return:
            self.current_return = [0.0] * self.num_envs
            self.current_length = [0] * self.num_envs

    def update(self, reward: torch.Tensor, done: torch.Tensor, flags: dict[str, torch.Tensor]) -> None:
        """Fold one step of every environment into the running episodes."""
        rewards = reward.reshape(-1).detach().cpu().tolist()
        finished = done.reshape(-1).detach().cpu().tolist()
        arrived = flags[TERM_ARRIVAL].reshape(-1).detach().cpu().tolist()
        touched = flags[TERM_COLLISION].reshape(-1).detach().cpu().tolist()
        pedestrian = flags[TERM_PEDESTRIAN].reshape(-1).detach().cpu().tolist()
        timed_out = flags[TERM_TIMEOUT].reshape(-1).detach().cpu().tolist()
        for index in range(self.num_envs):
            self.current_return[index] += float(rewards[index])
            self.current_length[index] += 1
            if not finished[index]:
                continue
            length = int(self.current_length[index])
            self.records.append(
                EpisodeOutcome(
                    length=length,
                    time_s=length * self.step_dt_s,
                    episode_return=float(self.current_return[index]),
                    arrived=bool(arrived[index]),
                    # The protocol fails an episode on any physical contact, so a
                    # pedestrian footprint counts exactly like a wall.
                    collided=bool(touched[index] or pedestrian[index]),
                    timed_out=bool(timed_out[index]),
                    pedestrian=bool(pedestrian[index]),
                )
            )
            self.current_return[index] = 0.0
            self.current_length[index] = 0

    def summary(self) -> dict[str, object]:
        """Aggregate the finished episodes of the window."""
        return summarise(self.records)

    def reset_window(self) -> None:
        """Drop the finished episodes but keep the episodes in progress."""
        self.records.clear()


def summarise(records: list[EpisodeOutcome]) -> dict[str, object]:
    """Turn finished episodes into the reported protocol statistics.

    Args:
        records: Episodes collected so far.

    Returns:
        Counts, rates, means and the per-episode outcome list.
    """
    count = len(records)
    if not count:
        return {
            "episodes": 0,
            "success_rate": None,
            "collision_rate": None,
            "timeout_rate": None,
            "arrival_rate": None,
            "mean_length": None,
            "mean_time_s": None,
            "mean_return": None,
            "outcomes": [],
        }
    successes = sum(record.success for record in records)
    collisions = sum(record.collided for record in records)
    timeouts = sum(record.timed_out and not record.collided for record in records)
    arrivals = sum(record.arrived for record in records)
    pedestrian_hits = sum(record.pedestrian for record in records)
    return {
        "episodes": count,
        "success_rate": successes / count,
        "collision_rate": collisions / count,
        "timeout_rate": timeouts / count,
        "arrival_rate": arrivals / count,
        "pedestrian_collision_rate": pedestrian_hits / count,
        "mean_length": sum(record.length for record in records) / count,
        "mean_time_s": sum(record.time_s for record in records) / count,
        "mean_return": sum(record.episode_return for record in records) / count,
        "outcomes": [
            {
                "length": record.length,
                "time_s": round(record.time_s, 3),
                "return": round(record.episode_return, 4),
                "arrived": record.arrived,
                "collided": record.collided,
                "pedestrian": record.pedestrian,
                "timed_out": record.timed_out,
                "success": record.success,
            }
            for record in records
        ],
    }


class VecRolloutBuffer:
    """On-policy storage for ``n_steps`` steps of ``num_envs`` environments."""

    def __init__(
        self,
        steps: int,
        num_envs: int,
        observation_size: int,
        device: torch.device,
        *,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        self.size = int(steps)
        self.num_envs = int(num_envs)
        self.device = device
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.observations = torch.zeros((steps, num_envs, observation_size), dtype=torch.float32, device=device)
        self.actions = torch.zeros((steps, num_envs, 2), dtype=torch.float32, device=device)
        self.log_probabilities = torch.zeros((steps, num_envs), dtype=torch.float32, device=device)
        self.values = torch.zeros((steps, num_envs), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((steps, num_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((steps, num_envs), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((steps, num_envs), dtype=torch.float32, device=device)
        self.returns = torch.zeros((steps, num_envs), dtype=torch.float32, device=device)
        self.position = 0

    @property
    def full(self) -> bool:
        """Whether every step of the rollout has been written."""
        return self.position >= self.size

    def reset(self) -> None:
        """Start a new rollout."""
        self.position = 0

    def add(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        log_probability: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Append one step of every environment."""
        index = self.position
        self.observations[index] = observation.detach()
        self.actions[index] = action.detach()
        self.rewards[index] = reward.detach().reshape(-1)
        self.dones[index] = done.detach().reshape(-1).float()
        self.log_probabilities[index] = log_probability.detach().reshape(-1)
        self.values[index] = value.detach().reshape(-1)
        self.position += 1

    def compute_returns_and_advantages(self, last_value: torch.Tensor, last_done: torch.Tensor) -> None:
        """Fill GAE(lambda) per environment over the stored steps.

        Args:
            last_value: Values of the observation following the rollout, ``[num_envs]``.
            last_done: Whether that observation starts a new episode, ``[num_envs]``.
        """
        advantage = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        next_value = last_value.detach().reshape(-1).to(self.device)
        next_done = last_done.detach().reshape(-1).float().to(self.device)
        for index in reversed(range(self.position)):
            terminal = 1.0 - next_done
            delta = self.rewards[index] + self.gamma * next_value * terminal - self.values[index]
            advantage = delta + self.gamma * self.gae_lambda * terminal * advantage
            self.advantages[index] = advantage
            self.returns[index] = advantage + self.values[index]
            next_value = self.values[index]
            next_done = self.dones[index]

    def flat(self) -> "FlatRollout":
        """Return the rollout as ``n_steps * num_envs`` transitions for the PPO update."""
        return FlatRollout(self)


class FlatRollout:
    """View of a :class:`VecRolloutBuffer` with one row per transition.

    The attribute names match :class:`~.ppo.RolloutBuffer`, so
    :meth:`~.ppo.DrlVoPPO.update` runs unchanged on a vectorised rollout.
    """

    def __init__(self, buffer: VecRolloutBuffer) -> None:
        self.observations = buffer.observations.view(buffer.position * buffer.num_envs, -1)
        self.actions = buffer.actions.view(buffer.position * buffer.num_envs, -1)
        self.log_probabilities = buffer.log_probabilities.reshape(-1)
        self.values = buffer.values.reshape(-1)
        self.rewards = buffer.rewards.reshape(-1)
        self.dones = buffer.dones.reshape(-1)
        self.advantages = buffer.advantages.reshape(-1)
        self.returns = buffer.returns.reshape(-1)
        self.position = buffer.position * buffer.num_envs
        self.size = buffer.size * buffer.num_envs

    @property
    def full(self) -> bool:
        """Whether the source rollout was complete."""
        return self.position >= self.size


def train_vec(
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
    on_iteration: object = None,
) -> dict[str, object]:
    """Run PPO over ``num_envs`` cloned robots and write metrics plus checkpoints.

    Args:
        env: Environment whose observation group is ``policy``.
        total_steps: Transitions to collect, counted across all environments and
            rounded down to whole iterations of ``config.n_steps``.
        output: Directory for ``metrics.jsonl``, ``run.json`` and checkpoints.
        config: PPO settings; the DRL-VO defaults are used when omitted.
        seed: Seed for torch, the environment and the minibatch order.
        device: Torch device of the policy.
        warm_start: Optional checkpoint that initialises the policy.
        learning_rate_decay: Optional ``(start, end)`` linear schedule.
        log_every_iterations: Write one metrics record every N iterations.
        save_every_iterations: Checkpoint every N iterations; 0 saves only at the end.
        tag: Prefix for checkpoint file names.
        on_iteration: Optional ``(iteration, policy, statistics) -> bool`` hook called
            after each update.  Returning ``True`` tells the loop that the hook
            drove the robots, so the rollout, the tracker and the observation are
            rebuilt before training continues.

    Returns:
        The summary dictionary, also written to ``output/run.json``.
    """
    config = config or PpoConfig()
    config.sample_count = int(config.n_steps) * int(env.num_envs)
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
        if isinstance(payload, dict) and "policy" in payload:
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

    num_envs = int(env.num_envs)
    buffer = VecRolloutBuffer(
        config.n_steps, num_envs, OBSERVATION_SIZE, device,
        gamma=config.gamma, gae_lambda=config.gae_lambda,
    )
    per_iteration = config.n_steps * num_envs
    iterations = max(1, int(total_steps) // per_iteration)
    base_learning_rate = config.learning_rate
    tracker = EpisodeTracker(num_envs=num_envs, step_dt_s=float(getattr(env, "step_dt", 0.05)))

    metrics = (output / "metrics.jsonl").open("w", buffering=1)
    started = time.monotonic()
    observation = observations_tensor(env.reset()[0], device)
    last_record: dict[str, object] = {}
    done = torch.zeros(num_envs, dtype=torch.bool, device=device)
    try:
        for iteration in range(1, iterations + 1):
            # Exploration schedule. Wide early so the policy can discover that
            # moving forward pays at all; narrow late so the sampled actions it
            # trains on look like the deterministic mean it is scored on.
            progress = (iteration - 1) / max(iterations - 1, 1)
            if progress < 0.10:
                noise_ceiling = 0.90
            elif progress < 0.55:
                noise_ceiling = 0.90 + (0.35 - 0.90) * ((progress - 0.10) / 0.45)
            else:
                noise_ceiling = 0.35 + (0.22 - 0.35) * ((progress - 0.55) / 0.45)
            trainer.set_std_bounds(0.08, noise_ceiling)
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
                done = (terminated | truncated).reshape(-1)
                flags = termination_flags(env)
                buffer.add(observation, action, reward, done, log_probability, value)
                tracker.update(reward, done, flags)
                observation = observations_tensor(next_observation, device)

            with torch.no_grad():
                last_value = trainer.value(observation)
            buffer.compute_returns_and_advantages(last_value, done.float())
            statistics = trainer.update(buffer.flat())
            statistics["iteration"] = iteration
            statistics["step"] = trainer.steps_done
            statistics["elapsed_s"] = time.monotonic() - started
            statistics["steps_per_s"] = trainer.steps_done / max(statistics["elapsed_s"], 1e-9)
            statistics["explained_variance"] = explained_variance(buffer.values.reshape(-1), buffer.returns.reshape(-1))
            statistics["learning_rate"] = trainer.optimizer.param_groups[0]["lr"]
            statistics.update(tracker.summary())
            tracker.reset_window()
            last_record = statistics
            if iteration % max(log_every_iterations, 1) == 0:
                metrics.write(json.dumps(statistics) + "\n")
                print(
                    f"[drl-vo-vec] iter {iteration}/{iterations} "
                    f"steps {trainer.steps_done} {statistics['steps_per_s']:.1f}/s "
                    f"lr {statistics['learning_rate']:.1e} "
                    f"ev {statistics['explained_variance']:+.2f} "
                    f"eps {statistics.get('episodes', 0)} "
                    f"success {float(statistics.get('success_rate') or 0.0):.2f} "
                    f"collision {float(statistics.get('collision_rate') or 0.0):.2f} "
                    f"timeout {float(statistics.get('timeout_rate') or 0.0):.2f} "
                    f"len {float(statistics.get('mean_length') or 0.0):.0f}",
                    flush=True,
                )
            if save_every_iterations and iteration % save_every_iterations == 0:
                trainer.save(output / f"{tag}_iter{iteration:05d}.pt")
            if on_iteration is not None and on_iteration(iteration, policy, statistics):
                # The hook drove the robots on its own, so the rollout state, the
                # episode windows and the observation row no longer describe the
                # training stream; rebuild all three before the next iteration.
                observation = observations_tensor(env.reset()[0], device)
                done = torch.zeros(num_envs, dtype=torch.bool, device=device)
                tracker = EpisodeTracker(num_envs=num_envs, step_dt_s=float(getattr(env, "step_dt", 0.05)))
    finally:
        metrics.close()

    checkpoint = trainer.save(output / f"{tag}_final.pt")
    summary = {
        "tag": tag,
        "steps": trainer.steps_done,
        "iterations": trainer.iteration,
        "num_envs": num_envs,
        "transitions_per_iteration": per_iteration,
        "checkpoint": str(checkpoint),
        "base_learning_rate": base_learning_rate,
        "warm_start": warm_start_report,
        "config": config.__dict__.copy(),
        "last_record": last_record,
    }
    (output / "run.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def evaluate_vec(
    env: object,
    policy: DrlVoPolicy,
    *,
    episodes: int,
    device: torch.device,
    max_steps: int | None = None,
) -> dict[str, object]:
    """Roll out the deterministic policy and score every episode by the protocol.

    Args:
        env: The navigation environment.
        policy: Trained actor-critic; the distribution mean is applied.
        episodes: Episodes to finish before returning.
        device: Torch device of the policy.
        max_steps: Optional hard stop on policy steps, as a safety valve.

    Returns:
        The :func:`summarise` statistics plus the steps taken.
    """
    policy.eval()
    tracker = EpisodeTracker(num_envs=int(env.num_envs), step_dt_s=float(getattr(env, "step_dt", 0.05)))
    observation = observations_tensor(env.reset()[0], device)
    done = torch.zeros(int(env.num_envs), dtype=torch.bool, device=device)
    steps = 0
    while len(tracker.records) < int(episodes):
        with torch.no_grad():
            mean, _, _ = policy(observation)
        observation, reward, terminated, truncated, _ = env.step(torch.clamp(mean, -1.0, 1.0))
        done = (terminated | truncated).reshape(-1)
        tracker.update(reward, done, termination_flags(env))
        observation = observations_tensor(observation, device)
        steps += 1
        if max_steps is not None and steps >= int(max_steps):
            break
    summary = summarise(tracker.records[: int(episodes)])
    summary["steps"] = steps
    return summary
