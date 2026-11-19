"""Score several policies on one protocol, on one environment build.

The four training machines ran independent PPO copies, so their checkpoints
have diverged. This script puts them on the same footing:

* one environment build, so geometry, walkers and raster are identical;
* the same seed sequence for every candidate, so each batch starts from the
  same ``num_envs`` start and goal pairs and the comparison is paired;
* the same success rule as the cluster evaluation: reach the goal inside the
  time limit without touching anything.

Besides the individual checkpoints it also scores a parameter average (model
soup) and an action ensemble, which usually beat the best single member of a
same-origin population.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

_SIMULATION = Path(__file__).resolve().parent
sys.path.insert(0, str(_SIMULATION.parents[1]))

DEFAULT_RUNS = Path("/workspace/wujunlin/runs")
TAG = "ped_train_%s_1794995086"
FINAL = "drl_vo_final.pt"


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", nargs="*", default=None, help="Checkpoints to compare; defaults to the four runs.")
    parser.add_argument("--output", type=Path, required=True, help="Directory for the comparison report.")
    parser.add_argument("--num-envs", type=int, default=16, help="Cloned robots scored together.")
    parser.add_argument("--episodes", type=int, default=64, help="Episodes per candidate, spread over batches.")
    parser.add_argument("--seed", type=int, default=20260101, help="First batch seed; later batches step by one.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sim-device", default="cpu")
    parser.add_argument("--character-gender", default="male")
    parser.add_argument("--character-offset", type=int, default=0)
    parser.add_argument("--visual-stride", type=int, default=2)
    parser.add_argument("--episode-steps", type=int, default=500, help="Policy steps per episode (500 = 25 s).")
    parser.add_argument("--room-usd", type=Path, default=_SIMULATION / "rl" / "assets" / "third_room.usd")
    parser.add_argument("--robot-usd", type=Path, default=_SIMULATION / "rl" / "assets" / "evt_robot.usd")
    parser.add_argument("--env-spacing", type=float, default=16.0)
    parser.add_argument("--robot-radius", type=float, default=0.37)
    parser.add_argument("--goal-radius", type=float, default=0.3)
    parser.add_argument("--no-soup", action="store_true", help="Skip the parameter average.")
    parser.add_argument("--no-ensemble", action="store_true", help="Skip the action ensemble.")
    return parser.parse_args()


def _default_checkpoints() -> list[Path]:
    """Return the final checkpoint of every finished training run."""
    found = []
    for host in ("a", "b", "d", "e"):
        path = DEFAULT_RUNS / (TAG % host) / FINAL
        if path.exists():
            found.append(path)
    if not found:
        raise SystemExit(f"No checkpoints under {DEFAULT_RUNS}")
    return found


def _environment_args(args: argparse.Namespace) -> argparse.Namespace:
    """Build the namespace ``_attach_reference`` and ``_attach_pedestrians`` expect."""
    return argparse.Namespace(
        num_envs=args.num_envs,
        episode_steps=args.episode_steps,
        seed=args.seed,
        device=args.device,
        sim_device=args.sim_device,
        room_usd=args.room_usd,
        robot_usd=args.robot_usd,
        env_spacing=args.env_spacing,
        robot_radius=args.robot_radius,
        goal_radius=args.goal_radius,
        collision_budget=1,
        replan_steps=10,
        lookahead=0.5,
        map_cache=_SIMULATION / "rl" / "assets" / "third_room_map.npz",
        rebuild_map=False,
        scenario=_SIMULATION / "configs" / "third_3" / "simulation.yaml",
        pedestrians=True,
        asset_root=None,
        character_gender=args.character_gender,
        character_offset=args.character_offset,
        visual_stride=args.visual_stride,
        no_fabric=False,
    )


def _average(payloads: list[dict]) -> dict:
    """Average every floating-point tensor of the given policy state dicts."""
    import torch

    averaged = {}
    for key, first in payloads[0].items():
        if first.is_floating_point():
            stacked = torch.stack([state[key].detach().float() for state in payloads])
            averaged[key] = stacked.mean(dim=0).to(first.dtype)
        else:
            averaged[key] = first
    return averaged


class _ActionEnsemble:
    """Average the deterministic actions of several policies."""

    def __init__(self, policies: list[object]) -> None:
        self.policies = list(policies)

    def eval(self) -> "_ActionEnsemble":
        for policy in self.policies:
            policy.eval()
        return self

    def __call__(self, observation):
        import torch

        with torch.no_grad():
            means = [policy(observation)[0] for policy in self.policies]
        return torch.stack(means, dim=0).mean(dim=0), None, None


def _merge(batches: list[dict]) -> dict:
    """Pool per-batch outcome lists into one protocol summary."""
    outcomes = [outcome for batch in batches for outcome in batch["outcomes"]]
    count = len(outcomes)
    if not count:
        return {"episodes": 0}
    successes = sum(1 for item in outcomes if item["success"])
    collisions = sum(1 for item in outcomes if item["collided"])
    pedestrian = sum(1 for item in outcomes if item["pedestrian"])
    timeouts = sum(1 for item in outcomes if item["timed_out"] and not item["collided"])
    arrivals = sum(1 for item in outcomes if item["arrived"])
    return {
        "episodes": count,
        "successes": successes,
        "success_rate": successes / count,
        "collision_rate": collisions / count,
        "pedestrian_collision_rate": pedestrian / count,
        "timeout_rate": timeouts / count,
        "arrival_rate": arrivals / count,
        "mean_length": sum(item["length"] for item in outcomes) / count,
        "mean_time_s": sum(item["time_s"] for item in outcomes) / count,
        "outcomes": outcomes,
    }


def _score(env, reference, policy, *, seeds: list[int], per_batch: int, device) -> dict:
    """Roll ``policy`` over identical seed batches and pool the outcomes."""
    from pedstrain_simulation.simulation.rl.vec_trainer import evaluate_vec

    batches = []
    started = time.perf_counter()
    for seed in seeds:
        # Replaying the sampler makes every candidate meet the same start and
        # goal pairs for the first batch of each seed.
        import numpy as np

        reference._random = np.random.RandomState(int(seed))
        batches.append(evaluate_vec(env, policy, episodes=per_batch, device=device))
    summary = _merge(batches)
    summary["seconds"] = round(time.perf_counter() - started, 1)
    return summary


def _run(args: argparse.Namespace, output: Path, report: dict) -> None:
    import numpy as np
    import torch

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sim import SimulationContext

    from pedstrain_simulation.simulation.rl.ppo import DrlVoPolicy
    from pedstrain_simulation.simulation.rl.vec_task import drl_vo_vec_cfg
    from pedstrain_simulation.simulation.train_drl_vo_vec import (
        _attach_pedestrians,
        _attach_reference,
        _launch,
        _load_policy,
        _sim_device,
    )

    clock = [time.perf_counter()]

    def phase(message: str) -> None:
        now = time.perf_counter()
        print(f"[compare] {message} (+{now - clock[0]:.1f}s)", flush=True)
        clock[0] = now

    environment = _environment_args(args)
    cfg = drl_vo_vec_cfg(
        room_usd=str(environment.room_usd),
        robot_usd=str(environment.robot_usd),
        num_envs=environment.num_envs,
        device=_sim_device(environment),
        seed=environment.seed,
        episode_steps=environment.episode_steps,
        goal_radius_m=environment.goal_radius,
        robot_radius_m=environment.robot_radius,
        env_spacing_m=environment.env_spacing,
        collision_budget=1,
        use_fabric=True,
    )
    SimulationContext._app_control_on_stop_handle_fn = lambda *a, **k: None
    env = ManagerBasedRLEnv(cfg)
    phase(f"environment built: {env.num_envs} clones")

    reference, grid = _attach_reference(env, environment)
    phase(f"occupancy ready: {grid.width}x{grid.height}")
    crowd, stage = _attach_pedestrians(env, environment)
    phase(f"pedestrians ready: {crowd.count} per room")
    env.reset()
    phase("first reset")
    # Fail loudly if a clone was placed outside its room, rather than reporting
    # twenty-five seconds of falling as a policy failure.
    for _ in range(10):
        env.step(torch.zeros((int(env.num_envs), int(env.action_manager.total_action_dim))))
    heights = env.scene["robot"].data.root_pos_w.torch[:, 2].detach().cpu()
    if bool((heights < -0.5).any()):
        raise RuntimeError(f"The robots fell through the room floor: {heights.tolist()}")

    device = torch.device(args.device)
    checkpoints = [Path(item) for item in (args.checkpoints or _default_checkpoints())]
    report["checkpoints"] = [str(path) for path in checkpoints]
    report["map"] = {"width": int(grid.width), "height": int(grid.height), "resolution_m": float(grid.resolution_m)}
    report["pedestrians"] = {"count": int(crowd.count), "characters": [path.name for path in stage.characters]}

    policies: dict[str, object] = {}
    payloads: list[dict] = []
    for path in checkpoints:
        name = path.parent.name.replace("ped_train_", "").split("_")[0]
        policies[name] = _load_policy(path, device)
        blob = torch.load(path, map_location=device, weights_only=False)
        payloads.append(blob["policy"] if isinstance(blob, dict) and "policy" in blob else blob)
        phase(f"loaded {name} from {path}")

    if not args.no_soup and len(payloads) > 1:
        soup = DrlVoPolicy()
        soup.load_state_dict(_average(payloads))
        policies["soup"] = soup.to(device)
        phase("built the parameter average")
    if not args.no_ensemble and len(payloads) > 1:
        # The ensemble sits beside the singles, never instead of them.
        members = [_load_policy(path, device) for path in checkpoints]
        policies["ensemble"] = _ActionEnsemble(members)
        phase(f"built an ensemble of {len(members)} policies")

    seeds = [int(args.seed) + index for index in range(max(1, int(np.ceil(args.episodes / args.num_envs))))]
    per_batch = int(args.num_envs)
    report["protocol"] = {
        "seeds": seeds,
        "episodes_per_candidate": len(seeds) * per_batch,
        "time_limit_s": args.episode_steps / 20.0,
        "goal_radius_m": args.goal_radius,
        "contact_radius_m": args.robot_radius,
        "success": "arrival within the time limit without any contact",
    }

    results = {}
    for name, policy in policies.items():
        phase(f"scoring {name} over {len(seeds) * per_batch} episodes")
        summary = _score(env, reference, policy, seeds=seeds, per_batch=per_batch, device=device)
        # Keep the report compact: the pooled outcome list is large.
        results[name] = {key: value for key, value in summary.items() if key != "outcomes"}
        results[name]["outcomes"] = summary["outcomes"]
        phase(
            f"{name}: success {summary['success_rate']:.3f} "
            f"collision {summary['collision_rate']:.3f} "
            f"pedestrian {summary['pedestrian_collision_rate']:.3f} "
            f"timeout {summary['timeout_rate']:.3f} "
            f"len {summary['mean_length']:.0f}"
        )
        env.reset()

    report["results"] = results
    best = max(results, key=lambda key: results[key]["success_rate"])
    report["best"] = best
    phase(f"best candidate: {best} at {results[best]['success_rate']:.3f}")
    env.close()


def main() -> None:
    args = _parse()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"passed": False, "mode": "compare"}
    app = None
    try:
        from pedstrain_simulation.simulation.train_drl_vo_vec import _launch

        app = _launch(args)
        _run(args, output, report)
        report["passed"] = True
    except BaseException:
        report["error"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        (output / "compare.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"COMPARE_RESULT {output / 'compare.json'}", flush=True)
        if app is not None:
            try:
                app.close(exit_code=0 if report.get("passed") else 1)
            except TypeError:
                app.close()
    if not report.get("passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
