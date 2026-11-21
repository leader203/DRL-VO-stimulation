# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the DRL-VO navigation policy with PPO inside Isaac Lab.

Run from the repository root with the Isaac Lab environment active:

.. code-block:: bash

    OMNI_KIT_ACCEPT_EULA=YES python simulation/train_drl_vo.py \
        --task-config simulation/configs/third_3/simulation.yaml \
        --total-steps 480000 --output runs/drl_vo_ppo

The environment is the current ROS 2 crowd stage: the composed USD scene, the
EVT robot and the ROS-driven pedestrians. Nothing is respawned or duplicated.
Rewards and terminations come from :mod:`simulation.rl.rewards`, the observation
from :mod:`simulation.rl.drl_observations`, and PPO from
:mod:`simulation.rl.ppo` with the DRL-VO hyper-parameters.

Without a ROS 2 installation the crowd cannot start, but the task still runs on
the composed static scene if the task config declares no pedestrians:
``simulation/configs/third_3/no_crowd.toml`` is that profile, and it is the
supported way to smoke-test the pipeline on a workstation. ``--launcher`` selects
the Kit experience: ``isaaclab`` (default, Isaac Sim 4.5) or ``project`` for the
6.2 experience that also enables the ROS 2 crowd extensions.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from uuid import uuid4

_SIM = Path(__file__).resolve().parent
sys.path.insert(0, str(_SIM.parents[1]))


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--task-config",
        type=Path,
        default=_SIM / "configs" / "third_3" / "simulation.yaml",
        help="Simulation profile that composes the training stage.",
    )
    parser.add_argument("--asset-root", type=Path, help="Override the configured Isaac Sim asset root.")
    parser.add_argument("--output", type=Path, help="Run directory; a timestamped one is created by default.")
    parser.add_argument("--total-steps", type=int, default=480000, help="Environment steps to collect.")
    parser.add_argument("--ros-domain-id", type=int, default=97, help="DDS domain for the crowding process.")
    parser.add_argument(
        "--device",
        default="cuda:0",
        help=(
            "GPU that runs the PhysX simulation and the policy network; pass cpu to run both "
            "on the host."
        ),
    )
    parser.add_argument(
        "--sim-device",
        default=None,
        help=(
            "Device of the environment's torch buffers. Isaac Sim 4.5 hands back host tensors "
            "from the PhysX tensor API, so this defaults to cpu whenever the simulation runs on "
            "a GPU; pass auto to keep it equal to --device."
        ),
    )
    parser.add_argument(
        "--launcher",
        choices=("isaaclab", "project"),
        default="isaaclab",
        help=(
            "Kit experience to start: Isaac Lab's AppLauncher (Isaac Sim 4.5) or the project's "
            "ros2_app.kit, which needs Isaac Sim 6.2 and the ROS 2 crowd extensions."
        ),
    )
    parser.add_argument("--seed", type=int, default=1337, help="Environment, torch and minibatch seed.")
    parser.add_argument("--no-headless", action="store_true", help="Open the Kit window.")
    parser.add_argument("--episode-steps", type=int, default=None, help="Rewarded steps per episode.")
    parser.add_argument("--robot-prim-path", default="/World/EVT_3", help="Articulation root in the stage.")
    parser.add_argument("--robot-radius-m", type=float, default=0.37, help="Footprint radius for reward thresholds.")
    parser.add_argument("--wheel-radius-m", type=float, default=0.05, help="Drive wheel radius [m].")
    parser.add_argument("--wheel-track-m", type=float, default=0.35, help="Wheel separation [m].")
    parser.add_argument("--goal-ros", type=float, nargs=2, metavar=("X", "Y"), help="Fixed goal in the ROS map frame.")
    parser.add_argument("--sample-goals", action="store_true", help="Resample the goal every episode.")
    parser.add_argument("--lookahead-m", type=float, default=0.5, help="Pure-pursuit lookahead for the local goal.")
    parser.add_argument("--replan-steps", type=int, default=10, help="Policy steps between A* replans.")
    parser.add_argument("--max-pedestrians", type=int, default=None, help="Observation capacity for the crowd.")
    parser.add_argument(
        "--no-fabric",
        action="store_true",
        help="Read articulation state from the PhysX tensor API instead of the fabric.",
    )
    parser.add_argument(
        "--kit-arg",
        action="append",
        default=[],
        metavar="SETTING",
        help="Extra Kit argument for the launch, e.g. --kit-arg=--/physics/suppressReadback=true.",
    )
    parser.add_argument("--warm-start", type=Path, help="Checkpoint to initialise the policy from.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--lr-final", type=float, default=None, help="Final learning rate of a linear schedule.")
    parser.add_argument("--n-steps", type=int, default=512, help="Rollout length in policy steps.")
    parser.add_argument("--batch-size", type=int, default=128, help="PPO minibatch size.")
    parser.add_argument("--epochs", type=int, default=10, help="PPO epochs per rollout.")
    parser.add_argument("--save-every", type=int, default=20, help="Checkpoint every N iterations; 0 disables.")
    parser.add_argument("--eval-episodes", type=int, default=0, help="Deterministic episodes to run after training.")
    parser.add_argument(
        "--watchdog-s",
        type=int,
        default=0,
        help="Dump the stacks of every thread after this many seconds, repeating; 0 disables.",
    )
    parser.add_argument("--worker-id", type=int, default=0, help="FedAvg worker index.")
    parser.add_argument("--num-workers", type=int, default=4, help="Total FedAvg workers.")
    parser.add_argument("--shared-dir", type=Path, help="NFS-shared directory for FedAvg synchronisation.")
    parser.add_argument("--eval-every", type=int, default=1, help="Evaluate the global policy every N rounds.")
    args = parser.parse_args()
    if not 0 <= args.ros_domain_id <= 232:
        parser.error("--ros-domain-id must be between 0 and 232")
    if args.total_steps < args.n_steps:
        parser.error("--total-steps must be at least --n-steps")
    return args


def _launch_app(config, navigation, args: argparse.Namespace):
    """Start Kit before anything imports ``pxr``/``isaaclab`` simulation modules.

    ``isaaclab`` is the Isaac Lab entry point and pairs with the installed Isaac
    Sim 4.5; ``project`` loads ``simulation/configs/runtime/ros2_app.kit``, which
    is the Isaac Sim 6.2 experience the ROS 2 crowd needs.
    """
    import os

    os.environ.setdefault("OMNI_CRASHREPORTER_ENABLED", "0")
    os.environ.setdefault("OMNI_CRASHREPORTER_SKIPOLDDUMPUPLOAD", "1")
    kit_settings = ["--/crashreporter/enabled=false", "--/crashreporter/skipOldDumpUpload=true"]
    kit_settings.extend(args.kit_arg)
    if navigation.asset_root:
        kit_settings.append(f"--/persistent/isaac/asset_root/default={navigation.asset_root}")
    if args.launcher == "project":
        from pedstrain_simulation.simulation.ros2_navigation.runtime import _create_app

        return _create_app(
            config,
            navigation,
            [
                "--enable",
                "isaacsim.core.version",
                "--/exts/isaacsim.core.simulation_manager/enable_default_callbacks=false",
            ],
        )
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(
        launcher_args={
            "headless": config.settings.headless,
            "device": args.device,
            "kit_args": " ".join(kit_settings),
        }
    )
    return launcher.app


def _run_coroutine(app, coroutine):
    """Complete the setup coroutine, which no longer suspends on this runtime.

    Kit cannot drive a nested coroutine from a standalone script (``run_coroutine``
    returns without executing), so :meth:`Ros2TrainingScene.setup` opens its stage
    with the synchronous USD call and this helper only has to unwrap the result.
    """
    del app
    import asyncio

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coroutine)
    finally:
        loop.close()


def _ignore(*args: object, **kwargs: object) -> None:
    """Replacement for Isaac Lab's interactive timeline-stop keep-alive callback."""
    del args, kwargs


def _attach_physx(stage) -> None:
    """Attach ``omni.physx`` to the composed stage.

    Isaac Sim 4.5 does not attach physics on stage open, on app updates or on
    timeline play, so Isaac Lab would build its simulation view without a physics
    scene. Attaching first is what makes ``physics_sim_view`` and the map queries
    work; on 6.2 the interface reports the stage already attached.
    """
    import omni.physx
    from pxr import UsdUtils

    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    interface = omni.physx.get_physx_simulation_interface()
    if interface.get_attached_stage() != stage_id:
        interface.attach_stage(stage_id)
    if interface.get_attached_stage() != stage_id:
        raise RuntimeError(f"Cannot attach PhysX to {stage.GetRootLayer().realPath}")


def _sim_device(args: argparse.Namespace) -> str:
    """Resolve the device of the environment's torch buffers.

    The PhysX tensor API of Isaac Sim 4.5 materialises articulation state as host
    tensors — ``set_root_transforms`` even rejects a CUDA index tensor — so an
    environment built on ``cuda`` has buffers and a simulation device that disagree
    as soon as anything writes to the view. The simulation itself still runs on the
    GPU because that follows the launcher's ``physics_gpu``, not this setting.
    """
    if args.sim_device is None:
        return "cpu" if str(args.device).startswith("cuda") else str(args.device)
    if args.sim_device == "auto":
        return str(args.device)
    return str(args.sim_device)


def _run(app, config, args: argparse.Namespace, output: Path, namespace: str, report: dict) -> None:
    import omni.usd
    import torch

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sim import SimulationContext, use_stage

    from pedstrain_simulation.simulation.rl import DrlVoReference, Ros2TrainingScene
    from pedstrain_simulation.simulation.rl.ppo import PpoConfig
    from pedstrain_simulation.simulation.rl.task import drl_vo_task_cfg
    from pedstrain_simulation.simulation.rl.trainer import evaluate
    from pedstrain_simulation.simulation.rl.distributed import train_fedavg

    clock = [time.perf_counter()]

    def phase(message: str) -> None:
        """Report one construction phase with its duration; flush for nohup logs."""
        now = time.perf_counter()
        print(f"[drl-vo] {message} (+{now - clock[0]:.1f}s)", flush=True)
        clock[0] = now

    context = omni.usd.get_context()
    capacity = args.max_pedestrians
    scene_args = {"output": output / "runtime"}
    if capacity is not None:
        scene_args["max_pedestrians"] = capacity
    env = None
    with Ros2TrainingScene(config, **scene_args) as prepared:
        try:
            _run_coroutine(app, prepared.setup())
            phase(f"stage ready: {prepared.physics_prim_path}")
            # Isaac Sim 4.5 attaches omni.physx to a freshly opened stage only from the
            # app update loop, so the pump and the explicit attach are required there.
            # Isaac Sim 6.2 attaches on its own and rejects a second attach ("Stage ...
            # already attached"), which leaves the physics view with a broken CUDA
            # context, so the workaround is applied only where PhysxCfg still exists.
            import isaaclab.sim as sim_utils

            if getattr(sim_utils, "PhysxCfg", None) is not None:
                import omni.kit.app

                for _ in range(3):
                    omni.kit.app.get_app().update()
                _attach_physx(prepared.stage)
                phase("physx attached")
            with use_stage(prepared.stage):
                cfg = drl_vo_task_cfg(
                    robot_prim_path=args.robot_prim_path,
                    robot_radius_m=args.robot_radius_m,
                    wheel_radius_m=args.wheel_radius_m,
                    wheel_track_m=args.wheel_track_m,
                    physics_prim_path=prepared.physics_prim_path,
                    device=_sim_device(args),
                    seed=args.seed,
                    pedestrian_action=prepared.has_crowd,
                    use_fabric=not args.no_fabric,
                    **({} if args.episode_steps is None else {"episode_steps": args.episode_steps}),
                )
                # A standalone Kit app subscribes a timeline-stop callback that renders
                # in a loop until the timeline plays again; with no window and no player
                # that loop never ends. Physics setup does not need it, so drop the body.
                SimulationContext._app_control_on_stop_handle_fn = _ignore
                with prepared.bind():
                    env = ManagerBasedRLEnv(cfg)
            phase("environment built")
            grid = prepared.occupancy()
            phase(f"occupancy ready: {grid.width}x{grid.height} cells")
            reference = DrlVoReference(
                grid=grid,
                frame=prepared.frame,
                robot_radius_m=args.robot_radius_m,
                lookahead_m=args.lookahead_m,
                replan_period_steps=args.replan_steps,
                goal_ros_xy=tuple(args.goal_ros) if args.goal_ros else None,
                sample_goals=args.sample_goals or args.goal_ros is None,
                seed=args.seed,
            )
            env.drl_vo_goal = reference
            phase("goal reference attached")
            if env.sim.physics_sim_view is None:
                raise RuntimeError("Isaac Lab built no PhysX simulation view for the composed stage.")
            report["observation_size"] = int(env.observation_manager.group_obs_dim["policy"][0])
            report["action_dim"] = int(env.action_manager.total_action_dim)
            report["physics_dt_s"] = float(env.physics_dt)
            report["step_dt_s"] = float(env.step_dt)
            report["decimation"] = int(cfg.decimation)
            report["physics_device"] = str(env.sim.device)
            report["sim_device"] = str(env.device)
            report["policy_device"] = str(args.device)

            ppo = PpoConfig(
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.epochs,
                learning_rate=args.lr,
            )
            phase("starting PPO")
            summary = train_fedavg(
                env,
                worker_id=args.worker_id,
                num_workers=args.num_workers,
                shared_dir=args.shared_dir,
                total_steps=args.total_steps,
                output=output,
                config=ppo,
                seed=args.seed,
                device=torch.device(args.device),
                warm_start=args.warm_start,
                save_every_iterations=args.save_every,
                eval_every_iterations=args.eval_every,
                eval_episodes=args.eval_episodes,
            )
            report["training"] = summary
            report["warm_start"] = summary.get("warm_start")
            if args.eval_episodes and args.worker_id == 0 and summary.get("evaluations"):
                report["evaluation"] = summary["evaluations"][-1]
            report["passed"] = True
        finally:
            try:
                if env is not None:
                    try:
                        if getattr(env, "pedestrian_bridge", None) is not None:
                            env.pedestrian_bridge.close()
                    finally:
                        try:
                            env.close()
                        except AttributeError as error:  # manager teardown before a full build
                            print(f"[drl-vo] environment close skipped: {error}", flush=True)
            finally:
                if context.get_stage() is not None:
                    context.close_stage()


def main() -> None:
    """Compose the stage, build the task and run PPO."""
    args = _parse()
    if args.watchdog_s > 0:
        import faulthandler

        faulthandler.dump_traceback_later(args.watchdog_s, repeat=True)
    output = (args.output or Path("runs") / ("drl_vo_ppo_" + datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    output.mkdir(parents=True, exist_ok=True)
    namespace = f"/drl_vo_training/run_{uuid4().hex[:8]}"
    report = {
        "passed": False,
        "output": str(output),
        "task_config": str(args.task_config),
        "ros_domain_id": args.ros_domain_id,
        "ros_namespace": namespace,
        "total_steps": args.total_steps,
        "seed": args.seed,
    }
    app = None
    try:
        from pedstrain_simulation.simulation import load_ros2_navigation_settings, load_simulation_config

        config = load_simulation_config(args.task_config)
        if args.no_headless:
            object.__setattr__(config.settings, "headless", False)
        navigation = load_ros2_navigation_settings(config.navigation.settings_path)
        if args.asset_root is not None:
            object.__setattr__(navigation, "asset_root", str(args.asset_root))
        app = _launch_app(config, navigation, args)
        _run(app, config, args, output, namespace, report)
    except BaseException:
        report["error"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"DRL_VO_TRAINING_RESULT {output / 'report.json'}", flush=True)
        if app is not None:
            try:
                app.close(exit_code=0 if report.get("passed") else 1)
            except TypeError:  # the Isaac Lab launcher exposes close() without an exit code
                app.close()
    if not report.get("passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
