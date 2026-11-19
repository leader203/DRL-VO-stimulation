# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-environment DRL-VO navigation task: 16 robots, random starts, random goals.

The single-robot task in :mod:`.task` steps one composed ROS stage. This task
instead *clones* one room and one robot ``num_envs`` times inside a single Isaac
Lab scene, which is what makes 16 simultaneous rollouts affordable on one GPU:

``{ENV_REGEX_NS}/Room``   ``simulation/assets/THIRD_EXP_1.usd``, the same rest room
                          the ROS scenario composes, spawned per environment;
``{ENV_REGEX_NS}/Robot``  ``simulation/rl/assets/evt_robot.usd`` (baked by
                          ``simulation/tools/export_robot_usd.py``), the EVT base
                          without its physics scene, ground plane or ROS graph.

Episode semantics follow the published evaluation protocol rather than the
training schedule of ``DRLNavEnv``:

* every reset teleports the robot to a random free pose and samples a new goal;
* an episode succeeds only if the robot reaches the goal within
  ``episode_length_s`` **and** never comes within ``robot_radius_m`` of an
  obstacle; a contact ends the episode immediately (``COLLISION_BUDGET = 1``);
* the reward stays the four migrated DRL-VO terms, so the optimisation target is
  still the published one, while success is judged by the stricter protocol.
"""

from __future__ import annotations

import math

import torch

from isaaclab import sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
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
from .rewards import (
    ANGLE_THRESHOLD_RAD,
    ANGULAR_SPEED_THRESHOLD_RAD_S,
    COLLISION_LOWER_M,
    GOAL_RADIUS_M,
    R_ANGLE,
    R_ARRIVAL,
    R_COLLISION,
    R_ROTATION,
    R_SCAN_DISTANCE,
    R_WAYPOINT,
    _planar_clearances,
    angular_reward,
    episode_state,
    goal_reached,
    goal_reward,
    heading_reward,
    iteration_budget_exhausted,
    obstacle_reward,
    reset_episode_state,
)

POLICY_HZ = 20
PHYSICS_HZ = 120
EPISODE_STEPS = 500  # 25 s at the source's 20 Hz policy rate
ROBOT_RADIUS_M = 0.37
WHEEL_RADIUS_M = 0.05
WHEEL_TRACK_M = 0.35
ROBOT_SPAWN_XY = (-2.3, -2.0)
ROBOT_SPAWN_Z = 0.02
WHEEL_JOINTS = ("wheel_fl_joint", "wheel_fr_joint", "wheel_rl_joint", "wheel_rr_joint")
ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
# One room is roughly 10 m across; 16 m keeps the clones apart while staying in a
# compact PhysX scene.
DEFAULT_ENV_SPACING_M = 16.0


def drl_vo_vec_cfg(
    *,
    room_usd: str,
    robot_usd: str,
    num_envs: int = 16,
    device: str = "cuda:0",
    seed: int = 1337,
    episode_steps: int = EPISODE_STEPS,
    goal_radius_m: float = GOAL_RADIUS_M,
    robot_radius_m: float = ROBOT_RADIUS_M,
    wheel_radius_m: float = WHEEL_RADIUS_M,
    wheel_track_m: float = WHEEL_TRACK_M,
    env_spacing_m: float = DEFAULT_ENV_SPACING_M,
    collision_budget: int = 1,
    use_fabric: bool = True,
    filter_collisions: bool = True,
) -> ManagerBasedRLEnvCfg:
    """Build the cloned-room navigation task.

    Args:
        room_usd: Room asset spawned once per environment.
        robot_usd: Robot asset spawned once per environment.
        num_envs: Number of cloned robots.
        device: Device of the environment's buffers.
        seed: Environment seed.
        episode_steps: Policy steps before the time limit ends the episode (500 = 25 s).
        goal_radius_m: Arrival tolerance [m].
        robot_radius_m: Footprint radius [m]; also the contact threshold.
        wheel_radius_m: Drive wheel radius [m].
        wheel_track_m: Left/right wheel separation [m].
        env_spacing_m: Distance between clone origins [m].
        collision_budget: Contact steps that end the episode; 1 is the protocol.
        use_fabric: Read articulation state through the PhysX fabric.
        filter_collisions: Filter collisions between clones. Without it every robot is
            broadphase-tested against all sixteen rooms, which dominates the step.

    Returns:
        A ready-to-instantiate ``ManagerBasedRLEnvCfg``; attach the reference with
        ``env.drl_vo_goal = VecDrlVoReference(...)`` before the first ``reset()``.
    """
    if int(num_envs) < 1:
        raise ValueError("num_envs must be positive")
    if episode_steps < 2:
        raise ValueError("episode_steps must be at least 2")
    if collision_budget < 1:
        raise ValueError("collision_budget must be positive")
    decimation = PHYSICS_HZ // POLICY_HZ

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        """One room and one robot per environment, cloned by the scene."""

        room = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Room", spawn=sim_utils.UsdFileCfg(usd_path=room_usd))
        robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=robot_usd),
            init_state=ArticulationCfg.InitialStateCfg(pos=(ROBOT_SPAWN_XY[0], ROBOT_SPAWN_XY[1], ROBOT_SPAWN_Z)),
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

    @configclass
    class ActionsCfg:
        """The policy twist mapped onto the four wheel joints."""

        robot_drive = DifferentialWheelVelocityActionCfg(
            asset_name="robot", wheel_radius_m=wheel_radius_m, wheel_track_m=wheel_track_m
        )

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """The frozen 19,202-element DRL-VO observation, one row per environment."""

        drl_vo = ObservationTermCfg(func=drl_vo_observations, params={"robot_name": "robot"})
        concatenate_terms = True
        enable_corruption = False

    @configclass
    class ObservationsCfg:
        policy = PolicyCfg()

    @configclass
    class RewardsCfg:
        """The four migrated DRL-VO terms; weights are the source constants."""

        goal = RewardTermCfg(
            func=goal_reward,
            weight=1.0,
            params={
                "r_arrival": R_ARRIVAL,
                "r_waypoint": R_WAYPOINT,
                "goal_radius_m": goal_radius_m,
                "max_iterations": episode_steps - 1,
            },
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
        """Arrival, contact and the 25 s time limit."""

        arrival = TerminationTermCfg(func=goal_reached, params={"goal_radius_m": goal_radius_m})
        collision = TerminationTermCfg(
            func=clearance_contact,
            params={"robot_radius_m": robot_radius_m, "budget": collision_budget},
        )
        pedestrian = TerminationTermCfg(
            func=pedestrian_contact,
            params={"robot_name": "robot", "robot_radius_m": robot_radius_m, "budget": collision_budget},
        )
        budget = TerminationTermCfg(func=iteration_budget_exhausted, params={"max_iterations": episode_steps - 1})

    @configclass
    class EventsCfg:
        """Respawn at a random free pose with a fresh goal, then clear the episode memory."""

        reset_robot = EventTermCfg(func=reset_robot_random, mode="reset", params={})
        reset_episode = EventTermCfg(func=reset_episode_memory, mode="reset", params={})

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

    return ManagerBasedRLEnvCfg(
        scene=SceneCfg(
            num_envs=int(num_envs),
            env_spacing=env_spacing_m,
            replicate_physics=True,
            filter_collisions=bool(filter_collisions),
        ),
        sim=simulation,
        actions=ActionsCfg(),
        observations=ObservationsCfg(),
        rewards=RewardsCfg(),
        terminations=TerminationsCfg(),
        events=EventsCfg(),
        decimation=decimation,
        episode_length_s=episode_steps / POLICY_HZ,
        seed=seed,
        ui_window_class_type=None,
    )


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def _selection(env: object, env_ids: object, device: torch.device) -> torch.Tensor:
    """Normalise the manager's selection to an index tensor on ``device``."""
    all_ids = torch.arange(int(env.num_envs), device=device, dtype=torch.long)
    if env_ids is None:
        return all_ids
    if isinstance(env_ids, slice):
        return all_ids[env_ids]
    return torch.as_tensor(env_ids, dtype=torch.long, device=device).reshape(-1)


def _root_buffers(env: object) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the robot root pose and velocity buffers as tensors."""
    data = env.scene["robot"].data
    pose = getattr(data, "root_link_pose_w", None)
    velocity = getattr(data, "root_link_vel_w", None)
    if pose is None or velocity is None:
        pose = getattr(data, "root_pose_w", pose)
        velocity = getattr(data, "root_vel_w", velocity)
    if pose is None or velocity is None:
        state = data.root_state_w
        state = state if isinstance(state, torch.Tensor) else state.torch
        return state[:, :7], state[:, 7:]
    pose = pose if isinstance(pose, torch.Tensor) else pose.torch
    velocity = velocity if isinstance(velocity, torch.Tensor) else velocity.torch
    return pose, velocity


def reset_robot_random(env: object, env_ids: object) -> None:
    """Teleport the selected robots to random free poses inside their own room.

    The reference owns the free-space sampling because it owns the raster; when it
    has not been attached yet (the environment's very first construction reset) the
    robots keep the pose configured in ``init_state``, which is a valid spot.

    ``env_ids`` carries no default on purpose: the 3.0 manager copies every
    parameter that has one into the term params *and* passes the ids positionally.

    Args:
        env: The navigation environment.
        env_ids: Environments being reset, supplied positionally by the manager.
    """
    reference = getattr(env, "drl_vo_goal", None)
    if reference is None or not hasattr(reference, "sample_start_poses"):
        return
    asset = env.scene["robot"]
    pose_buffer, velocity_buffer = _root_buffers(env)
    ids = _selection(env, env_ids, pose_buffer.device)
    reference.prepare(env)
    # Walkers restart at their scripted spawn every episode, so those disks are the
    # only places a robot must not appear in, or it fails the episode at step one.
    crowd = getattr(env, "drl_vo_crowd", None)
    avoid = None
    if crowd is not None:
        from .vec_pedestrians import PEDESTRIAN_RADIUS_M

        keep_out = PEDESTRIAN_RADIUS_M + float(ROBOT_RADIUS_M) + 0.2
        avoid = [(route.start_xy[0], route.start_xy[1], keep_out) for route in crowd.routes]
    sampled = reference.sample_start_poses(len(ids), avoid=avoid)
    origins = env.scene.env_origins
    origins = origins if isinstance(origins, torch.Tensor) else origins.torch

    pose = pose_buffer[ids].clone().to(dtype=torch.float32)
    pose[:, 0] = torch.as_tensor(sampled[:, 0], dtype=torch.float32, device=pose.device) + origins[ids, 0]
    pose[:, 1] = torch.as_tensor(sampled[:, 1], dtype=torch.float32, device=pose.device) + origins[ids, 1]
    pose[:, 2] = float(ROBOT_SPAWN_Z)
    half = torch.as_tensor(sampled[:, 2] * 0.5, dtype=torch.float32, device=pose.device)
    pose[:, 3] = 0.0
    pose[:, 4] = 0.0
    pose[:, 5] = torch.sin(half)
    pose[:, 6] = torch.cos(half)

    velocity = velocity_buffer[ids].clone().to(dtype=torch.float32)
    velocity[:] = 0.0
    asset.write_root_pose_to_sim(pose, env_ids=ids)
    asset.write_root_velocity_to_sim(velocity, env_ids=ids)
    # First step of the episode: the reward ring and the scan history start empty.
    reset_episode_state(env, ids)
    reset_observation_state(env, ids)


def reset_episode_memory(env: object, env_ids: object) -> None:
    """Resample the goal and clear the reward/observation memory of the selection.

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
    crowd = getattr(env, "drl_vo_crowd", None)
    if crowd is not None:
        crowd.reset(env, ids)
        hits = getattr(env, "drl_vo_pedestrian_hits", None)
        if hits is not None:
            hits[ids] = 0
        stage = getattr(env, "drl_vo_characters", None)
        if stage is not None:
            stage.sync(crowd, force=True)


# ---------------------------------------------------------------------------
# Termination
# ---------------------------------------------------------------------------


def clearance_contact(
    env: object,
    robot_radius_m: float = ROBOT_RADIUS_M,
    budget: int = 1,
) -> torch.Tensor:
    """Terminate once the robot has been inside the obstacle envelope ``budget`` steps.

    The source judged a bump from the 90° front scan and allowed three strikes. The
    evaluation protocol is stricter — any physical contact fails the episode — so
    the omnidirectional clearance of :class:`ClearanceMap` is used instead: it sees
    a wall the robot reversed into, which a front scan cannot.

    Args:
        env: Environment with an attached ``drl_vo_goal`` reference.
        robot_radius_m: Footprint radius [m].
        budget: Contact steps that end the episode.
    """
    reference = getattr(env, "drl_vo_goal", None)
    if reference is None:
        raise RuntimeError("Attach the DRL-VO reference as env.drl_vo_goal before stepping the environment.")
    reference.update(env)
    clearance = reference.clearance_m.to(device=env.device, dtype=torch.float32)
    if clearance.shape != (int(env.num_envs),):
        raise RuntimeError("The reference must publish clearance_m with shape [num_envs]")
    state = episode_state(env)
    state.collisions = state.collisions + (clearance <= float(robot_radius_m)).long()
    exceeded = state.collisions >= int(budget)
    state.collisions = torch.where(exceeded, torch.zeros_like(state.collisions), state.collisions)
    return exceeded


def pedestrian_contact(
    env: object,
    robot_name: str = "robot",
    robot_radius_m: float = ROBOT_RADIUS_M,
    budget: int = 1,
) -> torch.Tensor:
    """Terminate when the robot touches a pedestrian's capsule.

    The protocol fails an episode on any physical contact with a person, so this
    compares the planar distance between the two footprints with the sum of their
    radii — the same quantity the deployed keep-out disk uses. Without an attached
    crowd every environment is clear.

    Args:
        env: Environment that may own ``env.pedestrian_bridge``.
        robot_name: Scene key of the robot.
        robot_radius_m: Robot footprint radius [m].
        budget: Contact steps that end the episode.

    Returns:
        Boolean mask ``[num_envs]`` of the environments that touched a pedestrian.
    """
    if getattr(env, "pedestrian_bridge", None) is None:
        return torch.zeros(int(env.num_envs), dtype=torch.bool, device=env.device)
    clearance, valid = _planar_clearances(env, robot_name, robot_radius_m)
    if clearance.shape[1] == 0:
        return torch.zeros(int(env.num_envs), dtype=torch.bool, device=env.device)
    nearest = torch.where(valid, clearance, torch.full_like(clearance, float("inf"))).amin(dim=1)
    hit = nearest <= 0.0
    if int(budget) <= 1:
        return hit
    counter = getattr(env, "drl_vo_pedestrian_hits", None)
    if counter is None or tuple(counter.shape) != tuple(hit.shape):
        counter = torch.zeros_like(hit, dtype=torch.long)
        env.drl_vo_pedestrian_hits = counter
    counter += hit.long()
    exceeded = counter >= int(budget)
    counter[exceeded] = 0
    return exceeded


def spawn_pose(env: object) -> tuple[float, float, float]:
    """Return the configured spawn pose in clone-local coordinates [m, rad]."""
    return float(ROBOT_SPAWN_XY[0]), float(ROBOT_SPAWN_XY[1]), 0.0


def yaw_quaternion(yaw_rad: float) -> tuple[float, float, float, float]:
    """Return the XYZW quaternion of a planar yaw."""
    return 0.0, 0.0, math.sin(yaw_rad * 0.5), math.cos(yaw_rad * 0.5)
