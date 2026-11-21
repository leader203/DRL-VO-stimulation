# Data-parallel (federated averaging) PPO loop for the DRL-VO navigation task.
#
# Several machines share one NFS directory but each owns its own GPU and its
# own native ROS2 environment instance. Every worker samples + updates on its
# own environment, writes its policy weights into a per-round directory, and
# worker 0 averages all weights into a single global policy. All workers then
# reload the global policy and continue. This is synchronous FedAvg: observed
# data is parallel, but there is a single shared policy.
#
# Synchronisation uses only NFS files (no network sockets), because the boxes
# cannot reach each other over ICMP/TCP but do share /workspace over NFS.

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch

from .ppo import DrlVoPolicy, DrlVoPPO, PpoConfig, RolloutBuffer, explained_variance
from .trainer import TrainingLog, evaluate, policy_observations


def _atomic_torch_save(payload: object, path: Path) -> None:
    """Write a torch checkpoint atomically so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _policy_weights(trainer: DrlVoPPO) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in trainer.policy.state_dict().items()}


def _average_weights(weight_files: list[Path]) -> dict[str, torch.Tensor]:
    """Element-wise mean of several policy state_dicts."""
    avg: dict[str, torch.Tensor] | None = None
    count = 0
    for f in weight_files:
        payload = torch.load(f, map_location="cpu", weights_only=False)
        weights = payload["policy"] if "policy" in payload else payload
        if avg is None:
            avg = {k: v.float().clone() for k, v in weights.items()}
        else:
            for k in avg:
                avg[k] += weights[k].float()
        count += 1
    assert avg is not None and count > 0
    for k in avg:
        avg[k] /= count
    return avg


def train_fedavg(
    env: object,
    *,
    worker_id: int,
    num_workers: int,
    shared_dir: Path,
    total_steps: int,
    output: Path,
    config: PpoConfig | None = None,
    seed: int = 1337,
    device: torch.device | None = None,
    warm_start: Path | None = None,
    save_every_iterations: int = 1,
    eval_every_iterations: int = 1,
    eval_episodes: int = 2,
    poll_s: float = 1.0,
    wait_s: float = 1800.0,
) -> dict[str, object]:
    """Run synchronous FedAvg PPO across ``num_workers`` NFS-sharing workers."""
    config = config or PpoConfig()
    config.validate()
    # Decorrelate exploration across workers while keeping the SAME global init.
    torch.manual_seed(int(seed) + worker_id * 10007)
    device = device or torch.device(str(getattr(env, "device", "cpu")))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    shared_dir = Path(shared_dir)
    shared_dir.mkdir(parents=True, exist_ok=True)

    policy = DrlVoPolicy()
    trainer = DrlVoPPO(policy, config, device=device, seed=seed + worker_id * 10007)
    buffer = RolloutBuffer(config.n_steps, 19202, device, gamma=config.gamma, gae_lambda=config.gae_lambda)

    steps = max(config.n_steps, (int(total_steps) // config.n_steps) * config.n_steps)
    iterations = steps // config.n_steps
    log = TrainingLog()
    metrics = (output / f"metrics_w{worker_id}.jsonl").open("w", buffering=1)
    started = time.monotonic()
    evaluation_records: list[dict[str, object]] = []
    last_record: dict[str, object] = {}

    def wait_until(predicate, message: str) -> None:
        deadline = time.monotonic() + wait_s
        while not predicate():
            if time.monotonic() > deadline:
                raise TimeoutError(f"worker {worker_id} timed out: {message}")
            time.sleep(poll_s)

    def phase(message: str) -> None:
        print(f"[fedavg w{worker_id}] {message} (+{time.monotonic()-started:.1f}s)", flush=True)

    try:
        # ---- round 1: common initial weights from warm_start -----------------
        r1 = shared_dir / "round_00001"
        r1.mkdir(parents=True, exist_ok=True)
        if worker_id == 0:
            if warm_start is not None:
                payload = torch.load(Path(warm_start), map_location="cpu", weights_only=False)
                weights = payload["policy"] if "policy" in payload else payload
            else:
                weights = policy.state_dict()
            _atomic_torch_save({"policy": weights}, r1 / "global.pt")
            _atomic_text("ok", r1 / "global.ready")
            phase("published global init")
        else:
            wait_until(lambda: (r1 / "global.ready").exists(), "round 1 global init")

        for iteration in range(1, iterations + 1):
            rdir = shared_dir / f"round_{iteration:05d}"
            # ---- load the global policy for this round --------------------
            gpath = rdir / "global.pt"
            wait_until(lambda: gpath.exists(), f"round {iteration} global.pt")
            payload = torch.load(gpath, map_location="cpu", weights_only=False)
            global_weights = payload["policy"] if "policy" in payload else payload
            trainer.policy.load_state_dict(global_weights)
            phase(f"round {iteration} loaded global")

            # ---- sample n_steps + one PPO update --------------------------
            buffer.reset()
            observation = policy_observations(env.reset()[0], device)
            done = False
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

            with torch.no_grad():
                last_value = trainer.value(observation)
            buffer.compute_returns_and_advantages(
                float(last_value[0]), last_done=bool(done) if len(buffer) else False
            )
            statistics = trainer.update(buffer)
            statistics["iteration"] = iteration
            statistics["step"] = iteration * config.n_steps
            statistics["worker"] = worker_id
            statistics["elapsed_s"] = time.monotonic() - started
            statistics["explained_variance"] = explained_variance(buffer.values, buffer.returns)
            statistics.update(log.summary())
            log.reset_window()
            last_record = statistics
            metrics.write(json.dumps(statistics) + "\n")
            phase(f"round {iteration} updated")

            # ---- publish this worker's weights ----------------------------
            _atomic_torch_save({"policy": _policy_weights(trainer)}, rdir / f"worker_{worker_id}.pt")
            _atomic_text("ok", rdir / f"worker_{worker_id}.done")
            phase(f"round {iteration} published weights")

            # ---- aggregate (worker 0) -------------------------------------
            if worker_id == 0:
                wfiles = [rdir / f"worker_{j}.pt" for j in range(num_workers)]
                wait_until(
                    lambda: all((rdir / f"worker_{j}.done").exists() for j in range(num_workers)),
                    f"round {iteration} waiting for all workers",
                )
                avg = _average_weights(wfiles)
                _atomic_torch_save({"policy": avg}, shared_dir / f"global_iter{iteration:05d}.pt")
                phase(f"round {iteration} aggregated {num_workers} workers")

                # ---- evaluation with the aggregated global policy ---------
                if iteration % max(eval_every_iterations, 1) == 0:
                    eval_policy = DrlVoPolicy()
                    eval_policy.load_state_dict(avg)
                    ev = evaluate(env, eval_policy, episodes=eval_episodes, device=device)
                    ev["iteration"] = iteration
                    evaluation_records.append(ev)
                    _atomic_text(json.dumps(ev) + "\n", output / f"eval_iter{iteration:05d}.json")
                    phase(f"round {iteration} eval success_rate={ev['success_rate']:.3f}")

                # ---- publish next round (or finish) -----------------------
                if iteration < iterations:
                    nxt = shared_dir / f"round_{iteration+1:05d}"
                    nxt.mkdir(parents=True, exist_ok=True)
                    _atomic_torch_save({"policy": avg}, nxt / "global.pt")
                    _atomic_text("ok", nxt / "global.ready")
            # ---- wait for aggregation, then loop --------------------------
            if worker_id != 0:
                if iteration < iterations:
                    nxt_ready = shared_dir / f"round_{iteration+1:05d}" / "global.ready"
                    wait_until(lambda: nxt_ready.exists(), f"round {iteration+1} global ready")

            if save_every_iterations and iteration % save_every_iterations == 0 and worker_id == 0:
                _atomic_torch_save(
                    {"policy": torch.load(shared_dir / f"global_iter{iteration:05d}.pt", map_location="cpu")["policy"]},
                    output / f"drl_vo_iter{iteration:05d}.pt",
                )

        # ---- final global checkpoint (worker 0) ---------------------------
        if worker_id == 0:
            final = shared_dir / f"global_iter{iterations:05d}.pt"
            payload = torch.load(final, map_location="cpu", weights_only=False)
            _atomic_torch_save({"policy": payload["policy"]}, output / "drl_vo_final.pt")
    finally:
        metrics.close()

    summary = {
        "worker": worker_id,
        "num_workers": num_workers,
        "steps": iterations * config.n_steps,
        "iterations": iterations,
        "last_record": last_record,
        "evaluations": evaluation_records,
        "elapsed_s": time.monotonic() - started,
    }
    _atomic_text(json.dumps(summary, indent=2) + "\n", output / f"run_w{worker_id}.json")
    return summary
