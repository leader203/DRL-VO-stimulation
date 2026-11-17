# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise the ROS 2 crowd adapter in a real single-world Isaac Lab manager environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

_SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SIM.parents[1]))

_CONTROL_HZ = 60
_DECIMATION = 2
_PHYSICS_DT = 1 / (_CONTROL_HZ * _DECIMATION)
_ROBOT_RADIUS_M = 0.2
_ACTOR_START_M = (-2.0, 1.4, 0.0)
_ACTOR_SPEED_MPS = 0.9
_TIMEOUT_STEPS = 15


def _write_inputs(output: Path, asset_root: Path | None, domain_id: int, namespace: str) -> Path:
    import yaml

    from pedstrain_simulation.simulation import load_ros2_navigation_settings

    root = asset_root or load_ros2_navigation_settings().asset_root
    if root is None:
        raise ValueError("Supply the Isaac Sim asset root with --asset-root.")
    floor = output / "floor.usda"
    floor.write_text(
        """#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{
    def Cube "Floor" (prepend apiSchemas = ["PhysicsCollisionAPI"])
    {
        double size = 2
        double3 xformOp:scale = (5, 4, 0.1)
        double3 xformOp:translate = (0, 0, -0.1)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }
}
""",
        encoding="utf-8",
    )
    settings = output / "navigation_settings.yaml"
    settings.write_text(
        yaml.safe_dump(
            {
                "asset_root": str(Path(root).resolve()),
                "ros_domain_id": domain_id,
                "ros_namespace": namespace,
                "control_hz": _CONTROL_HZ,
                "origin_m": [0.2, -0.3, 0.0],
                "yaw_rad": 0.15,
                "resolution_m": 0.1,
                "bounds_m": [-4.0, -3.0, 4.0, 3.0],
            }
        ),
        encoding="utf-8",
    )
    profile = output / "simulation.yaml"
    profile.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "navigation": {"backend": "ros2", "waypoint_frame": "world_m", "settings_path": settings.name},
                "scene": {"asset_path": floor.name},
                "simulation": {"headless": True, "render_viewport": False, "seed": 42},
                "pedestrians": {
                    "count": 1,
                    "assets": str(_SIM / "configs" / "runtime" / "characters.yaml"),
                    "actors": {
                        "walker": {
                            "behavior": "walk",
                            "position": list(_ACTOR_START_M),
                            "yaw": 90.0,
                            "speed": _ACTOR_SPEED_MPS,
                            "waypoints": [{"position": [2.0, 1.4, 0.0]}],
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return profile


def _environment_config():
    import torch
    from isaaclab_physx.physics import PhysxCfg

    from isaaclab import sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
    from isaaclab.managers import (
        ActionTerm,
        ActionTermCfg,
        EventTermCfg,
        ObservationGroupCfg,
        ObservationTermCfg,
        RewardTermCfg,
        SceneEntityCfg,
        TerminationTermCfg,
    )
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils import configclass

    from pedstrain_simulation.simulation.rl import (
        Ros2PedestrianActionCfg,
        pedestrian_collision,
        pedestrian_observations,
        pedestrian_proximity_cost,
    )

    class PlanarForceAction(ActionTerm):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self._raw = torch.zeros((env.num_envs, 2), device=env.device)
            self._forces = torch.zeros((env.num_envs, 1, 3), device=env.device)
            self._torques = torch.zeros_like(self._forces)

        @property
        def action_dim(self) -> int:
            return 2

        @property
        def raw_actions(self):
            return self._raw

        @property
        def processed_actions(self):
            return self._forces[:, 0, :2]

        def process_actions(self, actions):
            self._raw.copy_(actions)
            self._forces[:, 0, :2] = actions.clamp(-1.0, 1.0) * 2.0

        def apply_actions(self):
            self._asset.permanent_wrench_composer.set_forces_and_torques_index(
                forces=self._forces,
                torques=self._torques,
                positions=self._asset.data.body_com_pos_w.torch,
                body_ids=[0],
                is_global=True,
            )

        def reset(self, env_ids=None):
            selection = slice(None) if env_ids is None else env_ids
            self._raw[selection] = 0.0
            self._forces[selection] = 0.0

    @configclass
    class PlanarForceActionCfg(ActionTermCfg):
        class_type: type = PlanarForceAction
        asset_name: str = "robot"

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        robot = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.SphereCfg(
                radius=_ROBOT_RADIUS_M,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -1.4, 0.22)),
        )

    @configclass
    class ActionsCfg:
        robot_force = PlanarForceActionCfg()
        pedestrians = Ros2PedestrianActionCfg(asset_name="robot")

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        pedestrians = ObservationTermCfg(func=pedestrian_observations, params={"robot_radius_m": _ROBOT_RADIUS_M})
        concatenate_terms = True
        enable_corruption = False

    @configclass
    class ObservationsCfg:
        policy = PolicyCfg()

    @configclass
    class RewardsCfg:
        clearance = RewardTermCfg(
            func=pedestrian_proximity_cost, weight=-1.0, params={"robot_radius_m": _ROBOT_RADIUS_M}
        )

    @configclass
    class TerminationsCfg:
        overlap = TerminationTermCfg(func=pedestrian_collision, params={"robot_radius_m": _ROBOT_RADIUS_M})
        time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)

    @configclass
    class EventsCfg:
        reset_robot = EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot"), "pose_range": {}, "velocity_range": {}},
        )

    return ManagerBasedRLEnvCfg(
        scene=SceneCfg(num_envs=1, env_spacing=10.0, replicate_physics=False, filter_collisions=False),
        sim=sim_utils.SimulationCfg(
            dt=_PHYSICS_DT,
            render_interval=_DECIMATION,
            physics=PhysxCfg(),
            device="cuda:0",
            create_stage_in_memory=False,
            enable_scene_query_support=True,
        ),
        actions=ActionsCfg(),
        observations=ObservationsCfg(),
        rewards=RewardsCfg(),
        terminations=TerminationsCfg(),
        events=EventsCfg(),
        decimation=_DECIMATION,
        episode_length_s=_TIMEOUT_STEPS * (_DECIMATION * _PHYSICS_DT),
        seed=42,
        ui_window_class_type=None,
    )


@contextmanager
def _ros_observer(domain_id):
    import rclpy
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node

    context = Context()
    rclpy.init(context=context, domain_id=domain_id)
    node = Node("pedestrian_rl_smoke_observer", context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    try:
        yield node, executor
    finally:
        executor.remove_node(node)
        executor.shutdown()
        node.destroy_node()
        context.try_shutdown()


def _wait_for_graph(observer, predicate):
    node, executor = observer
    deadline = time.monotonic() + 10.0
    while not predicate(node):
        if time.monotonic() >= deadline:
            raise TimeoutError("The ROS 2 graph did not reflect the expected crowd lifecycle.")
        executor.spin_once(timeout_sec=0.05)


def _crowd_topics(node, namespace):
    return {name for name, _ in node.get_topic_names_and_types() if name.startswith(namespace + "/")}


def _verify_clock_isolation(observer, namespace):
    def discovered(node):
        return any(
            topic.endswith("/state") and node.count_publishers(topic) for topic in _crowd_topics(node, namespace)
        )

    _wait_for_graph(observer, discovered)
    node, _ = observer
    publishers = node.get_publishers_info_by_topic("/clock")
    assert not any(
        info.node_namespace == namespace or info.node_namespace.startswith(namespace + "/") for info in publishers
    ), "The training adapter claimed the global ROS /clock topic."
    return sorted(_crowd_topics(node, namespace))


def _physical_walker(stage):
    from pxr import UsdPhysics

    actors = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.GetName() == "walker" and UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get()
    ]
    assert len(actors) == 1, "The expected kinematic walker was not created."
    return actors[0]


def _robot_obstacle_probe(robot, tensor_sim, physics, bridge, report, step):
    """Check the exact robot footprint passed to DDS against same-tick CUDA physics."""
    (path,) = robot.root_view.prim_paths
    tensor_robot = tensor_sim.create_rigid_body_view([path])
    assert tuple(tensor_robot.prim_paths) == (path,)

    def checked_step(time_s, dt, states, *, obstacles, door_states):
        matched = [obstacle for obstacle in obstacles if obstacle["prim_path"] == path]
        assert len(matched) == 1, "The robot must reach ROS as one physical obstacle."
        obstacle = matched[0]
        physical = tensor_robot.get_transforms()[0, :3].tolist()
        direct = physics.get_rigidbody_transformation(path)
        center = bridge._frame.ros_to_world(obstacle["position"])
        footprint = obstacle["footprint_xy_m"]
        footprint_center = [(min(axis) + max(axis)) / 2 for axis in zip(*footprint, strict=True)]
        position_error = math.dist(physical, center)
        footprint_error = math.dist(bridge._frame.world_to_ros(physical)[:2], footprint_center)
        report["last_robot_obstacle"] = {
            "time_s": time_s,
            "path": path,
            "physx_tensor_world_m": physical,
            "physx_direct_world_m": [float(value) for value in direct["position"]] if direct.get("ret_val") else None,
            "dds_center_world_m": center,
            "dds_footprint_center_map_m": footprint_center,
            "position_error_m": position_error,
            "footprint_center_error_m": footprint_error,
        }
        report["max_robot_obstacle_error_m"] = max(report.get("max_robot_obstacle_error_m", 0.0), position_error)
        report["robot_obstacle_samples"] = report.get("robot_obstacle_samples", 0) + 1
        assert position_error <= 2e-4 and footprint_error <= 2e-4, report["last_robot_obstacle"]
        return step(time_s, dt, states, obstacles=obstacles, door_states=door_states)

    return checked_step


def _exercise(env, stage, steps, report):
    import torch

    import omni.physics.tensors
    import omni.physx
    from pxr import UsdGeom

    from isaaclab.sim.utils.stage import get_current_stage_id

    bridge = env.pedestrian_bridge
    actor_path = _physical_walker(stage)
    physics = omni.physx.get_physx_interface()
    tensor_sim = omni.physics.tensors.create_simulation_view("torch", stage_id=get_current_stage_id())
    tensor_walker = tensor_sim.create_rigid_body_view([actor_path])
    assert tensor_walker.count == 1 and tuple(tensor_walker.prim_paths) == (actor_path,)
    physics_steps = []
    subscription = physics.subscribe_physics_step_events(physics_steps.append)
    action = torch.tensor([[1.0, 0.0]], device=env.device)
    robot = env.scene["robot"]
    robot_data = robot.data
    transport = bridge._transport
    transport_step = transport.step
    transport.step = _robot_obstacle_probe(robot, tensor_sim, physics, bridge, report, transport_step)
    try:
        obs, _ = env.reset(seed=42)
        assert obs["policy"].shape == (1, 14) and env.action_manager.total_action_dim == 2
        assert torch.isfinite(obs["policy"]).all() and not torch.count_nonzero(obs["policy"][0, 7:])
        assert bridge.state.valid.tolist() == [[True, False]]
        position = bridge.state.positions_w[0, 0].tolist()
        assert math.dist(position, _ACTOR_START_M) < 1e-5, "ROS map pose did not convert to metric world."
        robot_origin = robot_data.root_pos_w.torch[0, :2].clone()
        start_controls, start_physics = bridge.control_steps, len(physics_steps)
        env.sim.render()
        assert (bridge.control_steps, len(physics_steps)) == (start_controls, start_physics)
        report["render_without_extra_physics_or_control"] = True
        max_robot_distance = 0.0
        max_actor_speed = 0.0
        max_actor_pose_error = 0.0
        max_actor_heading_error = 0.0
        previous_physical = None
        physical_travel = 0.0
        timeout_resets = 0
        report["action_shape"] = [env.num_envs, env.action_manager.total_action_dim]
        report["observation_shape"] = list(obs["policy"].shape)
        for tick in range(steps):
            if tick == _TIMEOUT_STEPS // 2:
                before = len(physics_steps), bridge.control_steps, bridge.state.positions_w.clone()
                env.reset()
                assert (len(physics_steps), bridge.control_steps) == before[:2]
                torch.testing.assert_close(bridge.state.positions_w, before[2], rtol=0, atol=1e-6)
                report["manual_reset_without_crowd_advance"] = True

            obs, rewards, terminated, truncated, _ = env.step(action)
            assert len(physics_steps) - start_physics == (tick + 1) * _DECIMATION
            assert bridge.control_steps - start_controls == tick + 1
            assert obs["policy"].shape == (1, 14) and torch.isfinite(obs["policy"]).all()
            assert rewards.shape == terminated.shape == truncated.shape == (1,)
            assert torch.isfinite(rewards).all() and not terminated.item()
            if truncated.item():
                assert env.episode_length_buf.item() == 0
                timeout_resets += 1

            transform = physics.get_rigidbody_transformation(actor_path)
            tensor_pose = tensor_walker.get_transforms()[0].tolist()
            physical, physical_quaternion = tensor_pose[:3], tensor_pose[3:]
            observed = bridge.state.positions_w[0, 0].tolist()
            position_error = math.dist(physical, observed)
            root = stage.GetPrimAtPath(actor_path)
            rotation = UsdGeom.XformCache().GetLocalToWorldTransform(root).ExtractRotationQuat()
            authored_quaternion = (*rotation.GetImaginary(), rotation.GetReal())
            dot = sum(a * b for a, b in zip(physical_quaternion, authored_quaternion, strict=True))
            norm = math.hypot(*physical_quaternion) * math.hypot(*authored_quaternion)
            heading_error = 2 * math.acos(min(1.0, abs(dot) / norm))
            report["last_step"] = {
                "policy_step": tick + 1,
                "physx_tensor_actor_world_m": physical,
                "physx_direct_actor_world_m": (
                    [float(value) for value in transform["position"]] if transform.get("ret_val") else None
                ),
                "observed_actor_world_m": observed,
                "actor_position_error_m": position_error,
                "actor_rotation_error_rad": heading_error,
                "physics_steps": len(physics_steps) - start_physics,
                "ros_control_steps": bridge.control_steps - start_controls,
            }
            assert position_error <= 2e-4 and heading_error <= 1e-4, report["last_step"]
            max_actor_pose_error = max(max_actor_pose_error, position_error)
            max_actor_heading_error = max(max_actor_heading_error, heading_error)
            if previous_physical is not None:
                displacement = math.dist(physical, previous_physical)
                assert displacement <= _ACTOR_SPEED_MPS / _CONTROL_HZ + 1e-4, (
                    "A reset displaced the continuously walking actor."
                )
                physical_travel += displacement
            previous_physical = physical
            max_actor_speed = max(max_actor_speed, float(bridge.state.velocities_w[0, 0].norm()))
            robot_distance = float((robot_data.root_pos_w.torch[0, :2] - robot_origin).norm())
            max_robot_distance = max(max_robot_distance, robot_distance)
        assert timeout_resets > 0, "No automatic manager timeout reset ran."
        assert report["robot_obstacle_samples"] == steps, "The robot footprint must reach every ROS control tick."
        assert max_robot_distance > 0.005, "The force action did not move the physical robot."
        assert physical_travel > 0.1 and max_actor_speed > 0.05, "The actor showed no physical navigation."
        physics_elapsed = math.fsum(physics_steps[start_physics:])
        assert math.isclose(physics_elapsed, steps / _CONTROL_HZ, rel_tol=1e-6, abs_tol=1e-8)
        report.update(
            actor_path=actor_path,
            physics_steps=len(physics_steps) - start_physics,
            physics_elapsed_s=physics_elapsed,
            ros_control_steps=bridge.control_steps - start_controls,
            timeout_resets=timeout_resets,
            robot_max_distance_m=max_robot_distance,
            physical_actor_travel_m=physical_travel,
            max_actor_position_error_m=max_actor_pose_error,
            max_actor_rotation_error_rad=max_actor_heading_error,
            actor_max_speed_mps=max_actor_speed,
        )
    finally:
        transport.step = transport_step
        del subscription


def _run_native(app, config, output, domain_id, namespace, steps, report):
    import omni.usd
    from pxr import UsdPhysics

    from isaaclab.app.settings_manager import initialize_carb_settings
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sim import use_stage

    from pedstrain_simulation.simulation.rl import Ros2TrainingScene

    initialize_carb_settings()
    context = omni.usd.get_context()
    with _ros_observer(domain_id) as observer:
        with Ros2TrainingScene(config, max_pedestrians=2, output=output / "runtime") as prepared:
            env = None
            try:
                app.run_coroutine(prepared.setup())
                temporary_world = Path(prepared.stage.GetRootLayer().realPath)
                with use_stage(prepared.stage):
                    cfg = _environment_config()
                    cfg.sim.physics_prim_path = prepared.physics_prim_path
                    with prepared.bind():
                        env = ManagerBasedRLEnv(cfg)
                    process = env.pedestrian_bridge._transport._process
                    scenes = [str(prim.GetPath()) for prim in prepared.stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
                    assert scenes == [prepared.physics_prim_path]
                    device = str(env.scene["robot"].data.root_link_pose_w.warp.device)
                    assert device == "cuda:0"
                    report["active_physics_scenes"] = scenes
                    report["physics_device"] = device
                    _exercise(env, prepared.stage, steps, report)
                    report["crowd_topics"] = _verify_clock_isolation(observer, namespace)
                    report["global_clock_isolated"] = True
            finally:
                try:
                    if env is not None:
                        try:
                            env.pedestrian_bridge.close()
                        finally:
                            env.close()
                finally:
                    if context.get_stage() is not None:
                        app.run_coroutine(context.close_stage_async())
        assert context.get_stage() is None, "The training USD is still attached."
        assert not temporary_world.parent.exists(), "Temporary ROS training USD remains after close."
        assert process.poll() is not None, "The ROS navigation subprocess is still running."
        _wait_for_graph(observer, lambda node: not _crowd_topics(node, namespace))
        report["stage_closed"] = True
        report["temporary_composition_removed"] = True
        report["ros_process_exit_code"] = process.returncode
        report["ros_graph_detached"] = True


def _source_hashes():
    paths = [Path(__file__).resolve()]
    for package in ("rl", "ros2_navigation"):
        paths.extend((_SIM / package).glob("*.py"))
    return {str(path.relative_to(_SIM)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def main() -> None:
    """Run real CUDA PhysX and ROS 2 DDS, saving a receipt in a fresh directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory for smoke output.")
    parser.add_argument("--asset-root", type=Path, help="Override the configured local Isaac Sim asset root.")
    parser.add_argument("--ros-domain-id", type=int, default=96, help="DDS domain for this test run.")
    parser.add_argument("--steps", type=int, default=45, help="Policy steps at 60 Hz.")
    args = parser.parse_args()
    if args.steps < 2 * _TIMEOUT_STEPS:
        parser.error(f"steps must be at least {2 * _TIMEOUT_STEPS} to exercise both reset paths.")
    if not 0 <= args.ros_domain_id <= 232:
        parser.error("ros-domain-id must be between 0 and 232.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("ROS_LOG_DIR", str(output / "ros_logs"))

    from pedstrain_simulation.simulation import load_ros2_navigation_settings, load_simulation_config
    from pedstrain_simulation.simulation.ros2_navigation.runtime import _create_app

    namespace = f"/pedestrian_rl_smoke/run_{uuid4().hex[:8]}"
    report = {
        "passed": False,
        "ros_namespace_prefix": namespace,
        "ros_domain_id": args.ros_domain_id,
        "policy_steps": args.steps,
        "physics_dt_s": _PHYSICS_DT,
        "control_hz": _CONTROL_HZ,
        "source_sha256": _source_hashes(),
    }
    app = None
    try:
        config = load_simulation_config(_write_inputs(output, args.asset_root, args.ros_domain_id, namespace))
        navigation = load_ros2_navigation_settings(config.navigation.settings_path)
        kit_args = [
            "--enable",
            "isaacsim.core.version",
            "--/exts/isaacsim.core.simulation_manager/enable_default_callbacks=false",
        ]
        app = _create_app(config, navigation, kit_args)
        _run_native(app, config, output, args.ros_domain_id, namespace, args.steps, report)
        report["source_unchanged"] = report["source_sha256"] == _source_hashes()
        assert report["source_unchanged"], "The adapter source changed during the smoke run."
        report["passed"] = True
    except BaseException:
        report["error"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        path = output / "result.json"
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"ROS2_RL_SMOKE_RESULT {path}", flush=True)
        if app is not None:
            app.close(exit_code=0 if report["passed"] else 1)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
