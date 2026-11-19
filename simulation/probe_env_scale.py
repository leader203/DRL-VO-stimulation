"""Measure how the step cost scales with the number of cloned rooms.

Training is the only consumer of this number, and the loop hides it behind PPO
updates, so the probe drives zero actions for a fixed number of steps and reports
the wall time per step. Zero actions keep every robot parked, which removes
terminations from the measurement while still exercising physics, the scan, the
A* replans and the reward terms.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_SIMULATION = Path(__file__).resolve().parent
sys.path.insert(0, str(_SIMULATION.parents[1]))


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-envs", type=int, required=True, help="Cloned rooms to build.")
    parser.add_argument("--steps", type=int, default=40, help="Steps to time after a short warm-up.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--map-cache", type=Path, default=None, help="Defaults to a per-size cache file.")
    parser.add_argument("--pedestrians", dest="pedestrians", action="store_true", default=True)
    parser.add_argument("--no-pedestrians", dest="pedestrians", action="store_false")
    parser.add_argument("--scenario", type=Path, default=_SIMULATION / "configs" / "third_3" / "simulation.yaml")
    parser.add_argument("--character-gender", default="male")
    parser.add_argument("--visual-stride", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = _parse()
    # Kit has to be up before any simulation module is imported.
    from pedstrain_simulation.simulation.train_drl_vo_vec import _launch

    _launch(args)

    import torch

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sim import SimulationContext

    from pedstrain_simulation.simulation.rl.vec_task import drl_vo_vec_cfg
    from pedstrain_simulation.simulation.train_drl_vo_vec import (
        DEFAULT_ROBOT,
        DEFAULT_ROOM,
        _attach_pedestrians,
        _attach_reference,
    )

    cache = args.map_cache or _SIMULATION / "rl" / "assets" / f"third_room_map_{int(args.num_envs)}.npz"
    environment = argparse.Namespace(
        num_envs=args.num_envs,
        episode_steps=args.episode_steps,
        seed=args.seed,
        device=args.device,
        sim_device="cpu",
        room_usd=DEFAULT_ROOM,
        robot_usd=DEFAULT_ROBOT,
        env_spacing=16.0,
        robot_radius=0.37,
        goal_radius=0.3,
        collision_budget=1,
        replan_steps=10,
        lookahead=0.5,
        map_cache=cache,
        rebuild_map=False,
        scenario=args.scenario,
        pedestrians=args.pedestrians,
        asset_root=None,
        character_gender=args.character_gender,
        character_offset=0,
        visual_stride=args.visual_stride,
        no_fabric=False,
    )

    cfg = drl_vo_vec_cfg(
        room_usd=str(DEFAULT_ROOM),
        robot_usd=str(DEFAULT_ROBOT),
        num_envs=args.num_envs,
        device="cpu",
        seed=args.seed,
        episode_steps=args.episode_steps,
        goal_radius_m=0.3,
        robot_radius_m=0.37,
        env_spacing_m=16.0,
        collision_budget=1,
        use_fabric=True,
    )
    SimulationContext._app_control_on_stop_handle_fn = lambda *a, **k: None
    started = time.perf_counter()
    env = ManagerBasedRLEnv(cfg)
    print(f"[scale] built {env.num_envs} clones in {time.perf_counter() - started:.1f}s", flush=True)

    reference, grid = _attach_reference(env, environment)
    print(f"[scale] occupancy {grid.width}x{grid.height} at {time.perf_counter() - started:.1f}s", flush=True)
    if args.pedestrians:
        crowd, _ = _attach_pedestrians(env, environment)
        print(f"[scale] {crowd.count} walkers per room", flush=True)
    env.reset()

    action = torch.zeros((int(env.num_envs), int(env.action_manager.total_action_dim)))
    for _ in range(int(args.warmup)):
        env.step(action)
    started = time.perf_counter()
    for _ in range(int(args.steps)):
        env.step(action)
    elapsed = time.perf_counter() - started
    per_step = elapsed / max(int(args.steps), 1)
    print(
        f"[scale] RESULT num_envs={env.num_envs} step_ms={per_step * 1000:.0f} "
        f"transitions_per_s={env.num_envs / per_step:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
