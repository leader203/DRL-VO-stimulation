# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based Isaac Lab task for DRL-VO navigation on the current stage.

The task keeps the DRL-VO agent interface — a 2-D ``[-1, 1]`` twist command and
the 19,202-element observation — and replaces everything the source read from
Gazebo and ROS 1:

======================  =========================================
source                  task
======================  =========================================
``/odom`` twist         articulation ``root_lin_vel_b``/``root_ang_vel_b``
``/cmd_vel``            :class:`.drive.DifferentialWheelVelocityAction`
``/cnn_data`` scan      :class:`.reference.DrlVoReference` (occupancy ray cast)
``/move_base/...``      :class:`.reference.DrlVoReference` (A* + pure pursuit)
``/track_ped``          ``env.pedestrian_bridge`` (ROS 2 pedestrians)
reward/done             :mod:`.rewards` (four terms, three terminations)
======================  =========================================

Timing matches the source's 20 Hz policy step: 120 Hz physics, 60 Hz ROS
pedestrian control, so ``decimation = 6`` gives six physics substeps, three ROS
crowd updates and one policy action per step, and the 513-step episode limit
becomes ``episode_length_s = 513 / 20``.
"""

from __future__ import annotations

from isaaclab import sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .drive import DifferentialWheelVelocityActionCfg
from .drl_observations import drl_vo_observations, reset_observation_state
from .isaaclab import Ros2PedestrianActionCfg
from .rewards import (
    ANGLE_THRESHOLD_RAD,
    ANGULAR_SPEED_THRESHOLD_RAD_S,
    COLLISION_BUDGET,
    COLLISION_LOWER_M,
    GOAL_RADIUS_M,
    MAX_ITERATIONS,
    R_ANGLE,
    R_ARRIVAL,
    R_COLLISION,
    R_ROTATION,
    R_SCAN_DISTANCE,
    R_WAYPOINT,
    angular_reward,
    collision_budget_exceeded,
    goal_reached,
    goal_reward,
    heading_reward,
    iteration_budget_exhausted,
    obstacle_reward,
    reset_episode_state,
)

POLICY_HZ = 20
PHYSICS_HZ = 120
ROS_CONTROL_HZ = 60
EPISODE_STEPS = MAX_ITERATIONS + 1
ROBOT_RADIUS_M = 0.37
WHEEL_RADIUS_M = 0.05
WHEEL_TRACK_M = 0.35
ROBOT_PRIM_PATH = "/World/EVT_3"
WHEEL_JOINTS = ("wheel_fl_joint", "wheel_fr_joint", "wheel_rl_joint", "wheel_rr_joint")
ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")


def drl_vo_task_cfg(
    *,
    robot_prim_path: str = ROBOT_PRIM_PATH,
    robot_radius_m: float = ROBOT_RADIUS_M,
    wheel_radius_m: float = WHEEL_RADIUS_M,
    wheel_track_m: float = WHEEL_TRACK_M,
    physics_prim_path: str | None = None,
    device: str = "cuda:0",
    seed: int = 1337,
    episode_steps: int = EPISODE_STEPS,
    goal_radius_m: float = GOAL_RADIUS_M,
    episode_length_s: float | None = None,
    pedestrian_action: bool = True,
    use_fabric: bool = True,
) -> ManagerBasedRLEnvCfg:
    """Build the DRL-VO navigation task configuration.

    The caller must construct the environment on a prepared ROS stage and then
    attach the measurement provider with
    ``env.drl_vo_goal = DrlVoReference(...)``; rewards, terminations and
    observations all read it.

    Args:
        robot_prim_path: Existing articulation root in the composed stage.
        robot_radius_m: Footprint radius used by the reward thresholds [m]; the
            source tuned them for a 0.3 m turtlebot.
        wheel_radius_m: Drive wheel radius, used to convert m/s to rad/s [m].
        wheel_track_m: Left/right wheel separation [m].
        physics_prim_path: PhysX scene prim path from the prepared stage.
        device: Simulation device.
        seed: Environment seed.
        episode_steps: Rewarded steps per episode; 513 matches the source.
        goal_radius_m: Arrival tolerance [m].
        episode_length_s: Override for the manager's episode length [s].
        pedestrian_action: Add the zero-dimensional ROS crowd clock. Disable it on
            a stage without a crowd, where nothing needs advancing each substep.
        use_fabric: Read articulation state through the PhysX fabric. Disabling it
            makes Isaac Lab read the PhysX tensor API directly, which is the path
            that can keep buffers on the simulation device.

    Returns:
        A ready-to-instantiate ``ManagerBasedRLEnvCfg``.
    """
    if not robot_prim_path.startswith("/"):
        raise ValueError("robot_prim_path must be an absolute prim path")
    if episode_steps < 2:
        raise ValueError("episode_steps must be at least 2")
    if robot_radius_m <= 0 or wheel_radius_m <= 0 or wheel_track_m <= 0 or goal_radius_m <= 0:
        raise ValueError("Robot geometry and the goal radius must be positive [m]")
    if PHYSICS_HZ % POLICY_HZ or PHYSICS_HZ % ROS_CONTROL_HZ:
        raise ValueError("PHYSICS_HZ must be an integer multiple of both POLICY_HZ and ROS_CONTROL_HZ")
    decimation = PHYSICS_HZ // POLICY_HZ
    length = episode_length_s if episode_length_s is not None else episode_steps / POLICY_HZ

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        """The composed ROS stage supplies the robot; nothing is spawned."""

        robot = ArticulationCfg(
            prim_path=robot_prim_path,
            spawn=None,
            actuators={
                "wheels": ImplicitActuatorCfg(
                    joint_names_expr=list(WHEEL_JOINTS),
                    stiffness=0.0,
                    damping=2.0,
                    velocity_limit_sim=40.0,
                ),
                "arm": ImplicitActuatorCfg(
                    joint_names_expr=list(ARM_JOINTS),
                    stiffness=400.0,
                    damping=40.0,
                ),
            },
        )

    if pedestrian_action:

        @configclass
        class ActionsCfg:
            """Policy twist plus the zero-dimensional ROS crowd clock."""

            robot_drive = DifferentialWheelVelocityActionCfg(
                asset_name="robot", wheel_radius_m=wheel_radius_m, wheel_track_m=wheel_track_m
            )
            pedestrians = Ros2PedestrianActionCfg(asset_name="robot")

    else:

        @configclass
        class ActionsCfg:
            """Policy twist alone: the stage carries no ROS crowd to advance."""

            robot_drive = DifferentialWheelVelocityActionCfg(
                asset_name="robot", wheel_radius_m=wheel_radius_m, wheel_track_m=wheel_track_m
            )

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """The frozen DRL-VO observation as one concatenated 19,202-element term."""

        drl_vo = ObservationTermCfg(func=drl_vo_observations, params={"robot_name": "robot"})
        concatenate_terms = True
        enable_corruption = False

    @configclass
    class ObservationsCfg:
        policy = PolicyCfg()

    @configclass
    class RewardsCfg:
        """The four migrated DRL-VO terms; the weights are the source constants."""

        goal = RewardTermCfg(
            func=goal_reward,
            weight=1.0,
            params={"r_arrival": R_ARRIVAL, "r_waypoint": R_WAYPOINT, "goal_radius_m": goal_radius_m},
        )
        obstacle = RewardTermCfg(
            func=obstacle_reward,
            weight=1.0,
            params={
                "robot_radius_m": robot_radius_m,
                "r_collision": R_COLLISION,
                "r_scan": R_SCAN_DISTANCE,
                "collision_lower_m": COLLISION_LOWER_M,
            },
        )
        angular = RewardTermCfg(
            func=angular_reward,
            weight=1.0,
            params={"r_rotation": R_ROTATION, "angular_speed_threshold_rad_s": ANGULAR_SPEED_THRESHOLD_RAD_S},
        )
        heading = RewardTermCfg(
            func=heading_reward,
            weight=1.0,
            params={
                "robot_radius_m": robot_radius_m,
                "r_angle": R_ANGLE,
                "angle_threshold_rad": ANGLE_THRESHOLD_RAD,
            },
        )

    @configclass
    class TerminationsCfg:
        """Goal, collision budget and iteration budget, all terminal like the source."""

        arrival = TerminationTermCfg(func=goal_reached, params={"goal_radius_m": goal_radius_m})
        collision = TerminationTermCfg(
            func=collision_budget_exceeded,
            params={
                "robot_radius_m": robot_radius_m,
                "collision_lower_m": COLLISION_LOWER_M,
                "budget": COLLISION_BUDGET,
            },
        )
        budget = TerminationTermCfg(func=iteration_budget_exhausted, params={"max_iterations": episode_steps - 1})

    @configclass
    class EventsCfg:
        """Remember the staged spawn, return the robot to it, and clear the episode memory."""

        capture_spawn = EventTermCfg(func=capture_spawn_state, mode="startup", params={})
        reset_robot = EventTermCfg(func=reset_robot_default, mode="reset", params={})
        reset_reward = EventTermCfg(func=reset_drl_vo_episode, mode="reset", params={})

    # Isaac Lab 3.0 dropped ``physx=`` in favour of a pluggable ``physics`` preset and
    # no longer exposes ``PhysxCfg`` from ``sim_utils`` at all; leaving ``physics`` at
    # its default selects the Isaac Sim PhysX backend. Isaac Lab 2.x still needs the
    # explicit config, so the keyword is supplied only where the symbol exists.
    physics_kwargs = {}
    physx_cfg = getattr(sim_utils, "PhysxCfg", None)
    if physx_cfg is not None:
        physics_kwargs["physx"] = physx_cfg()
    simulation = sim_utils.SimulationCfg(
        dt=1.0 / PHYSICS_HZ,
        render_interval=decimation,
        device=device,
        use_fabric=use_fabric,
        create_stage_in_memory=False,
        enable_scene_query_support=True,
        **physics_kwargs,
    )
    # Isaac Lab 2.x reuses the physics scene authored on the composed stage, so the
    # path is handed over. 3.0 instead deletes every PhysicsScene on the stage and
    # recreates one *at* ``physics_prim_path``, which makes an authored path collide
    # with itself ("A prim already exists at path ..."), so 3.0 keeps its default.
    if physics_prim_path is not None and physx_cfg is not None:
        simulation.physics_prim_path = physics_prim_path

    return ManagerBasedRLEnvCfg(
        scene=SceneCfg(num_envs=1, env_spacing=10.0, replicate_physics=False, filter_collisions=False),
        sim=simulation,
        actions=ActionsCfg(),
        observations=ObservationsCfg(),
        rewards=RewardsCfg(),
        terminations=TerminationsCfg(),
        events=EventsCfg(),
        decimation=decimation,
        episode_length_s=length,
        seed=seed,
        ui_window_class_type=None,
    )


def _torch_view(value: object) -> "torch.Tensor":
    """Return the torch tensor behind an Isaac Lab buffer.

    Isaac Lab 3.0 wraps articulation buffers in ``ProxyArray``; 2.x handed back
    tensors directly.
    """
    import torch

    return value if isinstance(value, torch.Tensor) else value.torch


def _selection(env: object, env_ids: object, device: object) -> "torch.Tensor":
    """Normalise the manager's environment selection to an index tensor.

    Isaac Lab 3.0 passes ``slice`` for a full reset and an index tensor for a partial
    one, so downstream code receives one consistent, device-resident tensor.

    Args:
        env: The navigation environment.
        env_ids: ``None``, a ``slice`` or an index tensor from the term manager.
        device: Device of the buffer the indices will address.

    Returns:
        A one-dimensional ``torch.long`` tensor on ``device``.
    """
    import torch

    all_ids = torch.arange(int(env.num_envs), device=device, dtype=torch.long)
    if env_ids is None:
        return all_ids
    if isinstance(env_ids, slice):
        return all_ids[env_ids]
    return torch.as_tensor(env_ids, dtype=torch.long, device=device).reshape(-1)


def _root_state(env: object) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Return the robot root pose and velocity buffers as tensors.

    The pair is named ``root_link_pose_w``/``root_link_vel_w`` in Isaac Lab 3.0 and
    ``root_pose_w``/``root_vel_w`` (or the combined ``root_state_w``) in 2.x.
    """
    data = env.scene["robot"].data
    pose = getattr(data, "root_link_pose_w", None)
    velocity = getattr(data, "root_link_vel_w", None)
    if pose is None or velocity is None:
        pose = getattr(data, "root_pose_w", pose)
        velocity = getattr(data, "root_vel_w", velocity)
    if pose is None or velocity is None:
        state = _torch_view(data.root_state_w)
        return state[:, :7].clone(), state[:, 7:].clone()
    return _torch_view(pose).clone(), _torch_view(velocity).clone()


def capture_spawn_state(env: object, env_ids: object) -> None:
    """Record the pose the composed stage authored for the robot.

    Isaac Lab derives ``default_root_pose`` from ``cfg.scene.robot.init_state``, and
    this task spawns nothing, so that default is the identity while the stage has
    placed the robot somewhere else entirely. Resetting to the default therefore
    teleports the robot to the world origin — inside geometry, which trips the
    collision budget after a few steps. The stage is authoritative, so its pose is
    captured once at startup, before any reset overwrites it.

    ``env_ids`` is declared without a default for the reason given in
    :func:`reset_robot_default`, and is unused: the capture always covers every
    environment.

    Args:
        env: The navigation environment.
        env_ids: Environments the term was applied to, ignored here.
    """
    del env_ids
    pose, velocity = _root_state(env)
    env.drl_vo_spawn_pose = pose
    env.drl_vo_spawn_velocity = velocity


def reset_robot_default(env: object, env_ids: object) -> None:
    """Restore the pose the composed stage spawned the robot at.

    The task has no spawn randomization, so the captured state is written verbatim
    instead of going through ``mdp.reset_root_state_uniform``. The target comes from
    :func:`capture_spawn_state`; ``default_root_pose`` is only the fallback for a
    caller that built the environment without that startup event.

    ``env_ids`` carries no default on purpose. The 3.0 manager copies every parameter
    that has one into the term params and still passes the ids positionally, so a
    defaulted ``env_ids`` is supplied twice and raises ``got multiple values for
    argument 'env_ids'``.

    Args:
        env: The navigation environment.
        env_ids: Environments being reset, supplied positionally by the manager.
    """
    import torch

    asset = env.scene["robot"]
    pose = getattr(env, "drl_vo_spawn_pose", None)
    velocity = getattr(env, "drl_vo_spawn_velocity", None)
    if pose is None or velocity is None:
        data = asset.data
        pose_source = getattr(data, "default_root_pose", None)
        velocity_source = getattr(data, "default_root_vel", None)
        if pose_source is not None and velocity_source is not None:
            pose, velocity = _torch_view(pose_source), _torch_view(velocity_source)
        else:
            state = _torch_view(data.default_root_state)
            pose, velocity = state[:, :7], state[:, 7:]
    # Write in the buffers' own dtype/device: the PhysX view owns where the data lives.
    pose = pose.to(dtype=torch.float32)
    velocity = velocity.to(dtype=torch.float32)
    ids = _selection(env, env_ids, pose.device)
    asset.write_root_pose_to_sim(pose[ids].clone(), env_ids=ids)
    asset.write_root_velocity_to_sim(velocity[ids].clone(), env_ids=ids)


def reset_drl_vo_episode(env: object, env_ids: object) -> None:
    """Clear the reward ring, the scan history and the goal reference.

    ``env_ids`` carries no default for the same reason as in
    :func:`reset_robot_default`.

    Args:
        env: The navigation environment.
        env_ids: Environments being reset, supplied positionally by the manager.
    """
    ids = _selection(env, env_ids, env.device)
    reset_episode_state(env, ids)
    reset_observation_state(env, ids)
    reference = getattr(env, "drl_vo_goal", None)
    if reference is not None:
        reference.reset(env, ids)
        env.drl_vo_scan = None
