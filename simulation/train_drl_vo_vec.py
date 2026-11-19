# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train or evaluate the DRL-VO policy on 16 cloned rooms with PPO.

The single-robot entry point (:mod:`simulation.train_drl_vo`) steps one composed ROS
stage; this one clones the same rest room and the EVT robot ``--num-envs`` times
inside one Isaac Lab scene, so ``num_envs`` episodes run in lockstep on one GPU::

    OMNI_KIT_ACCEPT_EULA=YES python simulation/train_drl_vo_vec.py \
        --num-envs 16 --total-steps 480000 --warm-start DRL-VO/frozen_policy.pt \
        --output runs/vec_ppo

Every episode starts at a random free pose and gets a random goal, and is scored by
the published protocol:

* **success** — the robot reaches the goal within the time limit (25 s at 20 Hz,
  ``--episode-steps 500``) and never comes within ``--robot-radius-m`` of an
  obstacle; a contact ends the episode immediately;
* **success rate** — successful episodes over all episodes, which is what
  ``report.json`` records and what the distributed evaluation aggregates.

``--mode eval`` loads a checkpoint and only measures, which is the entry point the
six-GPU launcher uses.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

_SIMULATION = Path(__file__).resolve().parent
sys.path.insert(0, str(_SIMULATION.parents[1]))

DEFAULT_ROOM = _SIMULATION / "rl" / "assets" / "third_room.usd"
DEFAULT_ROBOT = _SIMULATION / "rl" / "assets" / "evt_robot.usd"


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("train", "eval", "train-eval"), default="train-eval")
    parser.add_argument("--output", type=Path, help="Run directory; a timestamped one is created by default.")
    parser.add_argument("--num-envs", type=int, default=16, help="Cloned robots stepped together.")
    parser.add_argument("--episode-steps", type=int, default=500, help="Policy steps per episode (500 = 25 s).")
    parser.add_argument("--total-steps", type=int, default=480_000, help="Transitions collected across all envs.")
    parser.add_argument("--n-steps", type=int, default=512, help="Policy steps per rollout (per environment).")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help=(
            "PPO minibatch size in transitions. The source trained with 128; the CNN holds "
            "batch x 512 x 80 x 80 activations, so 2048 needs far more than 48 GB."
        ),
    )
    parser.add_argument("--epochs", type=int, default=10, help="PPO epochs per rollout.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--lr-final", type=float, default=None, help="Final learning rate of a linear schedule.")
    parser.add_argument("--warm-start", type=Path, help="Checkpoint that initialises the policy.")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint to evaluate in --mode eval.")
    parser.add_argument("--episodes", type=int, default=64, help="Episodes to score in the evaluation.")
    parser.add_argument("--seed", type=int, default=1337, help="Environment, torch and sampling seed.")
    parser.add_argument("--device", default="cuda:0", help="GPU that runs physics and the policy.")
    parser.add_argument(
        "--sim-device",
        default="cpu",
        help="Device of the environment's torch buffers; auto keeps it equal to --device.",
    )
    parser.add_argument("--room-usd", type=Path, default=DEFAULT_ROOM, help="Cloneable room asset.")
    parser.add_argument("--robot-usd", type=Path, default=DEFAULT_ROBOT, help="Cloneable robot asset.")
    parser.add_argument("--env-spacing", type=float, default=16.0, help="Distance between clone origins [m].")
    parser.add_argument("--robot-radius", type=float, default=0.37, help="Footprint radius [m].")
    parser.add_argument("--goal-radius", type=float, default=0.3, help="Arrival tolerance [m].")
    parser.add_argument("--collision-budget", type=int, default=1, help="Contact steps that end an episode.")
    parser.add_argument("--replan-steps", type=int, default=10, help="Policy steps between A* replans.")
    parser.add_argument("--lookahead", type=float, default=0.5, help="Pure-pursuit lookahead [m].")
    parser.add_argument("--save-every", type=int, default=10, help="Checkpoint every N iterations; 0 disables.")
    parser.add_argument("--no-fabric", action="store_true", help="Read state through the PhysX tensor API.")
    parser.add_argument("--watchdog-s", type=int, default=0, help="Dump thread stacks after this many seconds.")
    parser.add_argument("--tag", default="drl_vo", help="Checkpoint file prefix.")
    parser.add_argument(
        "--map-cache",
        type=Path,
        default=_SIMULATION / "rl" / "assets" / "third_room_map.npz",
        help="Where the rasterised room is cached; rebuilding it costs two minutes.",
    )
    parser.add_argument("--rebuild-map", action="store_true", help="Ignore the cached occupancy raster.")
    parser.add_argument(
        "--scenario",
        type=Path,
        default=_SIMULATION / "configs" / "third_3" / "simulation.yaml",
        help="Scenario YAML whose pedestrians block supplies the walker routes.",
    )
    parser.add_argument(
        "--pedestrians",
        dest="pedestrians",
        action="store_true",
        default=True,
        help="Replay the scenario's walkers and spawn their characters in every room.",
    )
    parser.add_argument("--no-pedestrians", dest="pedestrians", action="store_false", help="Empty rooms.")
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=None,
        help="Directory holding Isaac/People/...; defaults to the repository's Character-assets.",
    )
    parser.add_argument("--character-gender", default="male", help="male, female or any, as characters.yaml labels.")
    parser.add_argument("--character-offset", type=int, default=0, help="Rotates the character pool per machine.")
    parser.add_argument(
        "--visual-stride",
        type=int,
        default=2,
        help="Policy steps between character transform writes; 0 freezes the characters.",
    )
    parser.add_argument(
        "--video-every",
        type=int,
        default=0,
        help="Every N iterations, score the policy and record one clip; 0 disables both.",
    )
    parser.add_argument("--eval-episodes", type=int, default=8, help="Episodes scored at each probe; 0 skips the score.")
    parser.add_argument("--video-steps", type=int, default=320, help="Policy steps in one recorded clip.")
    parser.add_argument("--video-stride", type=int, default=2, help="Capture one frame every N policy steps.")
    parser.add_argument("--video-width", type=int, default=640, help="Clip width [px].")
    parser.add_argument("--video-height", type=int, default=400, help="Clip height [px].")
    return parser.parse_args()


def _launch(args: argparse.Namespace):
    """Start Kit headless before any simulation module is imported."""
    import os

    os.environ.setdefault("OMNI_CRASHREPORTER_ENABLED", "0")
    from isaaclab.app import AppLauncher

    return AppLauncher(
        launcher_args={
            "headless": True,
            "device": args.device,
            "kit_args": "--/crashreporter/enabled=false --/crashreporter/skipOldDumpUpload=true",
        }
    ).app


def _sim_device(args: argparse.Namespace) -> str:
    return args.device if args.sim_device == "auto" else args.sim_device


def _room_exclusions(env: object, scene_roots: list[str]) -> tuple[str, ...]:
    """Prim paths the occupancy raster must ignore: every clone but the first, and every robot.

    The map is built once, from ``env_0``'s room, and every other environment is
    measured in that room's coordinates, so the other fifteen copies have to be
    invisible to the ray caster or they would land in the raster as obstacles in
    the wrong place.

    Args:
        env: The multi-environment task.
        scene_roots: The scene's environment root prim paths.

    Returns:
        Absolute prim paths to exclude.
    """
    excluded = list(scene_roots[1:])
    for root in scene_roots:
        excluded.append(f"{root}/Robot")
    return tuple(excluded)


def _build_grid(env: object, args: argparse.Namespace):
    """Rasterise ``env_0``'s room into the ROS occupancy grid the policy plans on."""
    from pedstrain_simulation.simulation import load_ros2_navigation_settings, load_simulation_config
    from pedstrain_simulation.simulation.ros2_navigation.frames import StageFrame
    from pedstrain_simulation.simulation.ros2_navigation.mapping import build_occupancy_grid
    from pedstrain_simulation.simulation.ros2_navigation.planner import OccupancyGrid

    simulation = load_simulation_config(_SIMULATION / "configs" / "third_3" / "no_crowd.toml")
    navigation = load_ros2_navigation_settings(simulation.navigation.settings_path)
    frame = StageFrame(1.0, "Z", navigation.origin_m, navigation.yaw_rad)
    settings = navigation.to_dict()
    settings.update(seed=int(args.seed), plans=[])

    import numpy as np
    import torch

    origins = env.scene.env_origins
    origins = origins if isinstance(origins, torch.Tensor) else origins.torch
    offset = tuple(float(value) for value in origins[0, :2].detach().cpu())

    cache = Path(args.map_cache)
    if cache.exists() and not args.rebuild_map:
        with np.load(cache) as stored:
            data = {
                "occupancy": stored["occupancy"],
                "resolution_m": float(stored["resolution_m"]),
                "origin_xy_m": tuple(float(value) for value in stored["origin_xy_m"]),
            }
            stored_offset = tuple(float(value) for value in stored["offset_xy"])
            tiles = int(stored["tiles"])
        # The raster is a snapshot of env_0's room in world coordinates, and the
        # clone layout moves env_0 whenever num_envs changes, so a cache built by
        # a differently sized run would put the map somewhere else entirely.
        fresh = max(abs(a - b) for a, b in zip(stored_offset, offset)) < 1e-6
        if fresh:
            print(
                f"[drl-vo-vec] occupancy restored from {cache} (built for {tiles} clones, offset {stored_offset})",
                flush=True,
            )
            return (
                OccupancyGrid(
                    data["occupancy"],
                    data["resolution_m"],
                    data["origin_xy_m"],
                    goal_projection_tolerance_m=navigation.goal_projection_tolerance_m,
                ),
                frame,
                offset,
            )
        print(
            f"[drl-vo-vec] cache was rasterised with env_0 at {stored_offset} but this run puts env_0 at {offset}; "
            f"rebuilding for {int(env.num_envs)} clones",
            flush=True,
        )

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("The multi-environment stage is not open")
    roots = [f"/World/envs/env_{index}" for index in range(int(env.num_envs))]
    data = build_occupancy_grid(stage, settings, frame, excluded_roots=_room_exclusions(env, roots))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache,
        occupancy=np.asarray(data["occupancy"], dtype=np.int8),
        resolution_m=float(data["resolution_m"]),
        origin_xy_m=np.asarray(data["origin_xy_m"], dtype=np.float64),
        offset_xy=np.asarray(offset, dtype=np.float64),
        tiles=np.int64(int(env.num_envs)),
    )
    return (
        OccupancyGrid(
            data["occupancy"],
            data["resolution_m"],
            data["origin_xy_m"],
            goal_projection_tolerance_m=navigation.goal_projection_tolerance_m,
        ),
        frame,
        offset,
    )


def _attach_reference(env: object, args: argparse.Namespace):
    """Attach the vectorised DRL-VO reference the reward, termination and observation read."""
    from pedstrain_simulation.simulation.rl.vec_reference import VecDrlVoReference

    grid, frame, offset = _build_grid(env, args)
    reference = VecDrlVoReference(
        grid=grid,
        frame=frame,
        num_envs=int(env.num_envs),
        robot_radius_m=args.robot_radius,
        lookahead_m=args.lookahead,
        replan_period_steps=args.replan_steps,
        arrival_tolerance_m=args.goal_radius,
        seed=args.seed,
        device=env.device,
        map_offset=offset,
    )
    env.drl_vo_goal = reference
    reference.prepare(env)
    return reference, grid


def _attach_pedestrians(env: object, args: argparse.Namespace):
    """Replay the scenario's walkers in every room and place their character USDs."""
    from pedstrain_simulation.simulation.rl.vec_pedestrians import (
        VecCharacterStage,
        VecPedestrians,
        load_character_pool,
        load_routes,
        pick_characters,
    )

    routes = load_routes(args.scenario)
    crowd = VecPedestrians(
        routes,
        num_envs=int(env.num_envs),
        device=env.device,
        dt=float(env.step_dt),
    )
    # ``pedestrian_bridge`` is the key the migrated reward and observation read;
    # ``drl_vo_crowd`` is the one the reset event and the stage sync advance.
    env.drl_vo_crowd = crowd
    env.pedestrian_bridge = crowd
    pool = load_character_pool(args.asset_root, gender=args.character_gender)
    characters = pick_characters(pool, len(routes), offset=args.character_offset)
    stage = VecCharacterStage(characters, num_envs=int(env.num_envs), visual_stride=args.visual_stride)
    stage.attach(env, crowd)
    env.drl_vo_characters = stage
    return crowd, stage


def _load_policy(path: Path, device):
    """Return the trained actor-critic stored in ``path``."""
    import torch

    from pedstrain_simulation.simulation.rl.ppo import DrlVoPolicy

    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload["policy"] if isinstance(payload, dict) and "policy" in payload else payload
    policy = DrlVoPolicy()
    policy.load_state_dict(state)
    return policy.to(device)


def _render_replay(path: Path, grid, track_robot, track_crowd, goals, *, fps: int, outcome: str) -> dict:
    """Draw an overhead replay of one episode and write it as an mp4.

    The training machines are headless and their ``--enable_cameras`` path blocks
    while starting the renderer, so the clip is drawn from the state trace
    instead. That also puts the whole episode on screen at once: the occupancy
    raster, the goal, the route taken, every walker and the moment of contact.
    """
    import matplotlib

    matplotlib.use("Agg")
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Circle, Polygon

    resolution = float(grid.resolution_m)
    width, height = int(grid.width), int(grid.height)
    x0, y0 = float(grid.origin_xy_m[0]), float(grid.origin_xy_m[1])
    extent = (x0, x0 + width * resolution, y0, y0 + height * resolution)

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=int(fps), macro_block_size=None)
    figure, axes = plt.subplots(figsize=(8.6, 5.0), dpi=110)
    try:
        for index in range(len(track_robot)):
            axes.clear()
            axes.imshow(
                np.asarray(grid.occupancy),
                cmap="Greys",
                origin="lower",
                extent=extent,
                vmin=0,
                vmax=100,
                interpolation="nearest",
            )
            trail = np.asarray(track_robot[: index + 1])
            axes.plot(trail[:, 0], trail[:, 1], color="#1f6feb", linewidth=1.8, alpha=0.85, label="robot path")
            if track_crowd:
                positions = np.asarray(track_crowd[index])
                axes.scatter(positions[:, 0], positions[:, 1], s=70, c="#d1242f", zorder=4, label="pedestrians")
            goal = goals[index]
            axes.scatter([goal[0]], [goal[1]], s=190, marker="*", c="#1a7f37", zorder=5, label="goal")
            pose = track_robot[index]
            heading = pose[2]
            nose = np.array([np.cos(heading), np.sin(heading)]) * 0.45
            side = np.array([-np.sin(heading), np.cos(heading)]) * 0.24
            axes.add_patch(
                Polygon(
                    [pose[:2] + nose, pose[:2] - nose * 0.35 + side, pose[:2] - nose * 0.35 - side],
                    closed=True,
                    color="#0b3d91",
                    zorder=6,
                )
            )
            axes.add_patch(Circle(pose[:2], 0.37, fill=False, color="#0b3d91", linewidth=1.0, zorder=6))
            axes.set_xlim(extent[0], extent[1])
            axes.set_ylim(extent[2], extent[3])
            axes.set_aspect("equal")
            axes.set_xlabel("x [m]")
            axes.set_ylabel("y [m]")
            axes.set_title(
                f"DRL-VO evaluation replay  |  t = {index / max(fps, 1):.1f} s  |  {outcome}",
                fontsize=10,
            )
            if index == 0:
                axes.legend(loc="upper right", fontsize=7, framealpha=0.85)
            figure.tight_layout()
            figure.canvas.draw()
            writer.append_data(np.asarray(figure.canvas.buffer_rgba())[:, :, :3])
    finally:
        plt.close(figure)
        writer.close()
    return {"frames": len(track_robot), "fps": int(fps), "path": str(path)}


def _make_probe(env, reference, grid, crowd, args: argparse.Namespace, output: Path, device):
    """Build the per-iteration hook that scores the policy and records a replay."""
    from pedstrain_simulation.simulation.rl.vec_trainer import evaluate_vec, observations_tensor, termination_flags

    every = max(int(args.video_every), 1)
    stream = (output / "evals.jsonl").open("w", buffering=1)

    def probe(iteration: int, policy, statistics: dict) -> bool:
        if int(iteration) % every:
            return False
        record = {
            "iteration": int(iteration),
            "env_step": int(statistics.get("step", 0)),
            "train_success_rate": statistics.get("success_rate"),
            "train_collision_rate": statistics.get("collision_rate"),
        }
        if int(args.eval_episodes) > 0:
            scored = evaluate_vec(env, policy, episodes=int(args.eval_episodes), device=device)
            record.update({key: value for key, value in scored.items() if key != "outcomes"})
            print(
                f"[drl-vo-vec] probe it{iteration}: eval success {float(scored['success_rate'] or 0.0):.3f} "
                f"collision {float(scored['collision_rate'] or 0.0):.3f} "
                f"pedestrian {float(scored.get('pedestrian_collision_rate') or 0.0):.3f} "
                f"timeout {float(scored['timeout_rate'] or 0.0):.3f}",
                flush=True,
            )
        if int(args.video_steps) > 0:
            import numpy as np
            import torch

            stride = max(int(args.video_stride), 1)
            track_robot: list = []
            track_crowd: list = []
            goals: list = []
            outcome = "still running"
            observation = observations_tensor(env.reset()[0], device)
            for step in range(int(args.video_steps)):
                with torch.no_grad():
                    mean, _, _ = policy(observation)
                observation, _, terminated, truncated, _ = env.step(torch.clamp(mean, -1.0, 1.0))
                if step % stride == 0:
                    track_robot.append(reference.poses_ros(env)[0].copy())
                    goals.append(reference.goals_ros_xy[0].copy())
                    if crowd is not None:
                        track_crowd.append(
                            np.asarray(
                                [reference.local_to_ros(float(x), float(y)) for x, y in crowd.room_xy()[0]]
                            )
                        )
                if bool((terminated | truncated).reshape(-1)[0]):
                    flags = termination_flags(env)
                    if bool(flags["arrival"].reshape(-1)[0]):
                        outcome = f"arrived in {step * float(env.step_dt):.1f} s"
                    elif bool(flags["collision"].reshape(-1)[0]) or bool(
                        flags.get("pedestrian", torch.zeros(1, dtype=torch.bool)).reshape(-1)[0]
                    ):
                        outcome = f"contact at {step * float(env.step_dt):.1f} s"
                    else:
                        outcome = f"timeout at {step * float(env.step_dt):.1f} s"
                    break
                observation = observations_tensor(observation, device)
            fps = max(1, int(round(1.0 / (float(env.step_dt) * stride))))
            clip = _render_replay(
                output / "videos" / f"iter_{int(iteration):04d}.mp4",
                grid,
                track_robot,
                track_crowd,
                goals,
                fps=fps,
                outcome=outcome,
            )
            clip["outcome"] = outcome
            record["video"] = clip
            print(f"[drl-vo-vec] probe it{iteration}: replay {clip}", flush=True)
        stream.write(json.dumps(record) + "\n")
        return True

    return probe


def _run(app, args: argparse.Namespace, output: Path, report: dict) -> None:
    import torch

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sim import SimulationContext

    from pedstrain_simulation.simulation.rl.ppo import PpoConfig
    from pedstrain_simulation.simulation.rl.vec_task import drl_vo_vec_cfg
    from pedstrain_simulation.simulation.rl.vec_trainer import evaluate_vec, train_vec

    clock = [time.perf_counter()]

    def phase(message: str) -> None:
        now = time.perf_counter()
        print(f"[drl-vo-vec] {message} (+{now - clock[0]:.1f}s)", flush=True)
        clock[0] = now

    cfg = drl_vo_vec_cfg(
        room_usd=str(args.room_usd),
        robot_usd=str(args.robot_usd),
        num_envs=args.num_envs,
        device=_sim_device(args),
        seed=args.seed,
        episode_steps=args.episode_steps,
        goal_radius_m=args.goal_radius,
        robot_radius_m=args.robot_radius,
        env_spacing_m=args.env_spacing,
        collision_budget=args.collision_budget,
        use_fabric=not args.no_fabric,
    )
    # A standalone Kit app subscribes a timeline-stop callback that renders in a
    # loop until the timeline plays again; with no window that loop never ends.
    SimulationContext._app_control_on_stop_handle_fn = lambda *a, **k: None
    env = ManagerBasedRLEnv(cfg)
    phase(f"environment built: {env.num_envs} clones, device={env.device}")

    reference, grid = _attach_reference(env, args)
    phase(f"occupancy ready: {grid.width}x{grid.height} cells")
    crowd = stage = None
    if args.pedestrians:
        crowd, stage = _attach_pedestrians(env, args)
        phase(f"pedestrians ready: {crowd.count} per room, characters {[path.stem for path in stage.characters]}")
    env.reset()
    phase("first reset with random start poses")
    # A clone whose room is misplaced drops straight through the floor and then
    # times out for twenty-five seconds per episode, which looks like a policy
    # failure; fail loudly instead.
    import torch as _torch

    for _ in range(10):
        env.step(_torch.zeros((int(env.num_envs), int(env.action_manager.total_action_dim))))
    heights = (env.scene["robot"].data.root_pos_w.torch[:, 2]).detach().cpu()
    report["height_check"] = [round(float(value), 3) for value in heights]
    if bool((heights < -0.5).any()):
        raise RuntimeError(f"The robots fell through the room floor: heights {report['height_check']}")
    phase(f"robot heights {[round(float(v), 3) for v in heights[:4]]}")

    report["observation_size"] = int(env.observation_manager.group_obs_dim["policy"][0])
    report["action_dim"] = int(env.action_manager.total_action_dim)
    report["num_envs"] = int(env.num_envs)
    report["physics_dt_s"] = float(env.physics_dt)
    report["step_dt_s"] = float(env.step_dt)
    report["episode_steps"] = int(args.episode_steps)
    report["episode_length_s"] = float(cfg.episode_length_s)
    report["physics_device"] = str(env.sim.device)
    report["sim_device"] = str(env.device)
    report["policy_device"] = str(args.device)
    report["map"] = {"width": int(grid.width), "height": int(grid.height), "resolution_m": float(grid.resolution_m)}
    report["start_poses"] = reference.sample_start_poses(min(int(env.num_envs), 4)).tolist()
    report["pedestrians"] = {
        "enabled": bool(args.pedestrians),
        "count": int(crowd.count) if crowd is not None else 0,
        "routes": [route.name for route in crowd.routes] if crowd is not None else [],
        "characters": [path.name for path in stage.characters] if stage is not None else [],
        "visual_stride": int(args.visual_stride),
    }

    device = torch.device(args.device)
    if args.mode in ("train", "train-eval"):
        ppo = PpoConfig(n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.epochs, learning_rate=args.lr)
        phase("starting PPO")
        probe = (
            _make_probe(env, reference, grid, crowd, args, output, device)
            if int(args.video_every) > 0
            else None
        )
        summary = train_vec(
            env,
            total_steps=args.total_steps,
            output=output,
            config=ppo,
            seed=args.seed,
            device=device,
            warm_start=args.warm_start,
            learning_rate_decay=(args.lr, args.lr_final) if args.lr_final is not None else None,
            save_every_iterations=args.save_every,
            tag=args.tag,
            on_iteration=probe,
        )
        report["training"] = summary
        report["warm_start"] = summary.get("warm_start")
        checkpoint = Path(summary["checkpoint"])
    else:
        if args.checkpoint is None:
            raise SystemExit("--mode eval needs --checkpoint")
        checkpoint = args.checkpoint

    if args.mode in ("eval", "train-eval") and args.episodes > 0:
        policy = _load_policy(checkpoint, device)
        phase("evaluating the protocol")
        statistics = evaluate_vec(env, policy, episodes=args.episodes, device=device)
        report["evaluation"] = statistics
        phase(f"success rate {statistics['success_rate']}")
    report["passed"] = True
    env.close()


def main() -> None:
    """Compose the cloned room, build the task and train or evaluate."""
    args = _parse()
    if args.watchdog_s > 0:
        import faulthandler

        faulthandler.dump_traceback_later(args.watchdog_s, repeat=True)
    output = (args.output or Path("runs") / ("drl_vo_vec_" + datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "passed": False,
        "mode": args.mode,
        "output": str(output),
        "seed": args.seed,
        "num_envs": args.num_envs,
        "protocol": {
            "time_limit_s": args.episode_steps / 20.0,
            "goal_radius_m": args.goal_radius,
            "contact_radius_m": args.robot_radius,
            "collision_budget": args.collision_budget,
            "success": "arrival within the time limit without any contact",
        },
    }
    app = None
    try:
        app = _launch(args)
        _run(app, args, output, report)
    except BaseException:
        report["error"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"DRL_VO_VEC_RESULT {output / 'report.json'}", flush=True)
        if app is not None:
            try:
                app.close(exit_code=0 if report.get("passed") else 1)
            except TypeError:
                app.close()
    if not report.get("passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
