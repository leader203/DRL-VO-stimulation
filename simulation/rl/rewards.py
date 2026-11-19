# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DRL-VO reward terms migrated from the Gazebo turtlebot trainer.

The source reward lives in ``DRL-VO/reference/drl_nav_env.py``
(``DRLNavEnv._compute_reward``) and is reproduced statement by statement by
``DRL-VO/code/reward_logic.py``. This module re-expresses the same four sums as
Isaac Lab manager terms on the current ROS 2 / PhysX stage:

============  ==========================================  =====================
term          source reading                              Isaac Lab reading
============  ==========================================  =====================
goal          ``_goal_reached_reward``                    goal distance [m]
obstacle      ``_obstacle_collision_punish``              ``min_scan`` [m]
angular       ``_angular_velocity_punish``                physical yaw rate
heading       ``_theta_reward``                           VO cone over crowd
============  ==========================================  =====================

Everything the source measured through ROS 1 topics is read from Isaac Lab
state instead: the robot pose/velocity from ``scene[robot_name].data``, the
crowd from ``env.pedestrian_bridge.state``, the goal reference from
``env.drl_vo_goal`` (see :mod:`.goal`), and the front scan from a ray-caster
sensor or a pre-computed ``env.drl_vo_scan`` tensor.

Fidelity notes, all deliberate:

* The distance register is a 10-slot ring, so the progress term compares
  against the sample from ten iterations ago, not the previous sample.
* Collision counting is cumulative and *not* reset by a goal or timeout
  episode boundary (``reward_logic.OriginalReward.reset_task`` does not clear
  ``bump_num``); :func:`reset_episode_state` therefore leaves it alone.
* ``max_iterations`` is the last *rewarded* iteration. The source increments
  its counter after computing the reward, so Isaac Lab
  ``episode_length_buf - 1`` is the matching index.
* Thresholds that were expressed in turtlebot body radii (0.3 m) scale with
  ``robot_radius_m``; the goal tolerance does not, because it is a task
  tolerance rather than a footprint measurement.
* A scan with no return inside range is treated as ``max_range_m``; the source
  raised on an all-zero scan, which is not a usable training signal.

The functions are plain ``func(env, **params)`` reward terms: each returns one
value per environment with shape ``[num_envs]`` and does not touch the
simulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .observations import _floating_tensor, _geometry, _nonnegative_parameter, _planar_clearances

# Source constants from DRLNavEnv._compute_reward, TRAINING_CONFIG.json "reward".
R_ARRIVAL = 20.0
R_WAYPOINT = 3.2
R_COLLISION = -20.0
R_SCAN_DISTANCE = -0.2
R_ANGLE = 0.6
R_ROTATION = -0.1
GOAL_RADIUS_M = 0.3
COLLISION_LOWER_M = 0.02
DISTANCE_HISTORY = 10
MAX_ITERATIONS = 512
ANGLE_THRESHOLD_RAD = math.pi / 6
ANGULAR_SPEED_THRESHOLD_RAD_S = 1.0
VO_SAMPLES = 60
VO_PEDESTRIAN_RANGE_M = 7.0
COLLISION_BUDGET = 3
# The turtlebot body radius the source thresholds were tuned for [m].
SOURCE_ROBOT_RADIUS_M = 0.3


@dataclass
class DrlVoEpisodeState:
    """Per-environment reward memory that the simulator does not own.

    Attributes:
        distances: Ring of the last ``distance_history`` goal distances [m],
            shape ``[num_envs, distance_history]``.
        collisions: Cumulative close-scan counter, shape ``[num_envs]``,
            carried across episode boundaries exactly like the source.
    """

    distances: torch.Tensor
    collisions: torch.Tensor


def episode_state(env: object) -> DrlVoEpisodeState:
    """Return this environment's DRL-VO reward memory, allocating it once.

    Args:
        env: Environment with ``num_envs`` and ``device``.

    Returns:
        The state stored on ``env.drl_vo_reward_state``.
    """
    state = getattr(env, "drl_vo_reward_state", None)
    if state is None:
        count = int(env.num_envs)
        if count <= 0:
            raise ValueError("num_envs must be positive")
        device = env.device
        state = DrlVoEpisodeState(
            distances=torch.zeros((count, DISTANCE_HISTORY), dtype=torch.float32, device=device),
            collisions=torch.zeros((count,), dtype=torch.long, device=device),
        )
        env.drl_vo_reward_state = state
    return state


def reset_episode_state(env: object, env_ids: torch.Tensor | slice | None = None) -> None:
    """Clear the distance ring at a robot reset while keeping the counters.

    Use this as a ``mode="reset"`` event term next to the robot reset, so the
    progress term restarts from the new spawn while the source's collision
    carry-over is preserved.

    Args:
        env: Environment with ``num_envs`` and ``device``.
        env_ids: Environment indices to reset; all of them when ``None``.
    """
    state = episode_state(env)
    selection = slice(None) if env_ids is None else env_ids
    state.distances[selection] = 0.0


def iteration_index(env: object) -> torch.Tensor:
    """Return the source-equivalent iteration index, shape ``[num_envs]``.

    ``ManagerBasedRLEnv`` increments ``episode_length_buf`` before computing
    rewards, whereas ``DRLNavEnv`` computes the reward before incrementing its
    own counter. Subtracting one aligns the two so the first reward of an
    episode seeds the distance ring (iteration 0) and the last reward of the
    budget sees iteration ``max_iterations``.
    """
    lengths = getattr(env, "episode_length_buf", None)
    if not isinstance(lengths, torch.Tensor):
        raise RuntimeError("The DRL-VO reward terms require ManagerBasedRLEnv.episode_length_buf")
    return (lengths.to(dtype=torch.long) - 1).clamp_min(0)


# ---------------------------------------------------------------------------
# Measurements shared with the task
# ---------------------------------------------------------------------------


def goal_distance(env: object) -> torch.Tensor:
    """Return the distance to the global goal [m], shape ``[num_envs]``.

    The reference is published once per step by the attached goal provider, so
    every term and the observation see the same target.
    """
    provider = getattr(env, "drl_vo_goal", None)
    if provider is None:
        raise RuntimeError("Attach a DRL-VO goal provider as env.drl_vo_goal before computing rewards.")
    provider.update(env)
    distance = provider.distance_m
    if not isinstance(distance, torch.Tensor) or distance.shape != (int(env.num_envs),):
        raise RuntimeError("The DRL-VO goal provider must expose distance_m with shape [num_envs]")
    return distance


def local_goal_base(env: object) -> torch.Tensor:
    """Return the pure-pursuit target in the robot frame [m], shape ``[num_envs, 2]``."""
    provider = getattr(env, "drl_vo_goal", None)
    if provider is None:
        raise RuntimeError("Attach a DRL-VO goal provider as env.drl_vo_goal before computing rewards.")
    provider.update(env)
    local = provider.local_goal_base_xy
    if not isinstance(local, torch.Tensor) or local.shape != (int(env.num_envs), 2):
        raise RuntimeError("The DRL-VO goal provider must expose local_goal_base_xy with shape [num_envs, 2]")
    return local


def robot_velocity(env: object, robot_name: str = "robot") -> tuple[torch.Tensor, torch.Tensor]:
    """Return physical ``(forward_speed, yaw_rate)`` [m/s, rad/s], each ``[num_envs]``.

    The source read ``Twist.linear.x`` and ``Twist.angular.z`` from ``/odom``.
    Body-frame root velocities are the direct replacement; the yaw rate is the
    body z rate and the forward speed is the body x rate.
    """
    data = _robot_data(env, robot_name)
    linear = getattr(data, "root_lin_vel_b", None)
    angular = getattr(data, "root_ang_vel_b", None)
    forward = _floating_tensor(linear, "root_lin_vel_b", (int(env.num_envs), 3))[:, 0]
    yaw_rate = _floating_tensor(angular, "root_ang_vel_b", (int(env.num_envs), 3))[:, 2]
    return forward, yaw_rate


def minimum_scan(env: object, sensor_name: str = "scan", max_range_m: float = 8.0) -> torch.Tensor:
    """Return ``min_scan`` in the source's sense, shape ``[num_envs]``.

    ``min_scan`` is the closest front return that is greater than zero, exactly
    like ``np.amin(cnn_data.scan[scan != 0])``. Range sensors report a miss as
    zero (the legacy scan) or infinity (an Isaac Lab ray caster); both are
    excluded here. A scan without any return means nothing is within
    ``max_range_m`` and is returned as that distance, which the source could
    only express as an exception.

    Args:
        env: Environment exposing ``drl_vo_scan`` or ``scene[sensor_name]``.
        sensor_name: Scene key of a ray-caster sensor, used when no explicit
            ``drl_vo_scan`` tensor is attached.
        max_range_m: Finite no-return distance [m].

    Returns:
        Minimum front range [m] per environment.
    """
    limit = _nonnegative_parameter(max_range_m, "max_range_m", positive=True)
    ranges = getattr(env, "drl_vo_scan", None)
    if ranges is None:
        provider = getattr(env, "drl_vo_goal", None)
        if provider is not None:
            provider.update(env)
            ranges = getattr(env, "drl_vo_scan", None)
    if ranges is None:
        ranges = _sensor_ranges(env, sensor_name)
    if not isinstance(ranges, torch.Tensor) or not ranges.is_floating_point():
        raise RuntimeError("The front scan must be a floating tensor of ranges [m]")
    if ranges.dim() != 2 or ranges.shape[0] != int(env.num_envs):
        raise RuntimeError(f"The front scan must have shape [num_envs, beams], got {tuple(ranges.shape)}")
    hit = torch.isfinite(ranges) & (ranges > 0.0)
    return torch.where(hit, ranges, ranges.new_full((), limit)).amin(dim=-1)


def _sensor_ranges(env: object, sensor_name: str) -> torch.Tensor:
    scene = getattr(env, "scene", None)
    try:
        sensor = scene[sensor_name] if scene is not None else None
    except KeyError:
        sensor = None
    if sensor is None:
        raise RuntimeError(
            "Attach a front range sensor as env.drl_vo_scan or as a ray caster in scene['%s']" % sensor_name
        )
    data = sensor.data
    hits = _torch_view(data.ray_hits_w, "ray_hits_w")
    hits = _floating_tensor(hits, "ray_hits_w", tuple(hits.shape))
    origin = _floating_tensor(_torch_view(data.pos_w, "pos_w"), "pos_w", (int(env.num_envs), 3), hits)
    return torch.linalg.vector_norm(hits - origin.unsqueeze(1), dim=-1)


def _torch_view(value: object, name: str) -> object:
    # Isaac Lab sensors may expose torch tensors or Warp arrays with a .torch view.
    return value if isinstance(value, torch.Tensor) else getattr(value, "torch", value)



def _robot_data(env: object, robot_name: str) -> object:
    if not isinstance(robot_name, str) or not robot_name.strip():
        raise ValueError("robot_name must be a nonempty scene asset key")
    return env.scene[robot_name].data


# ---------------------------------------------------------------------------
# Reward terms
# ---------------------------------------------------------------------------


def goal_reward(
    env: object,
    robot_name: str = "robot",
    goal_radius_m: float = GOAL_RADIUS_M,
    r_arrival: float = R_ARRIVAL,
    r_waypoint: float = R_WAYPOINT,
    max_iterations: int = MAX_ITERATIONS,
    distance_history: int = DISTANCE_HISTORY,
) -> torch.Tensor:
    """Reward goal arrival and progress, shape ``[num_envs]``.

    Mirrors ``DRLNavEnv._goal_reached_reward``: arrival inside
    ``goal_radius_m`` pays ``r_arrival``, running out of iterations pays
    ``-r_arrival``, and every other step pays ``r_waypoint`` times the change
    against the distance register. Iteration 0 seeds the whole register, so the
    first reward of an episode is zero.

    Args:
        env: Environment with an attached ``drl_vo_goal`` provider.
        robot_name: Unused scene key kept for signature symmetry.
        goal_radius_m: Arrival tolerance [m], nonnegative.
        r_arrival: Arrival constant, also negated for a timeout.
        r_waypoint: Progress gain per metre.
        max_iterations: Last rewarded iteration of the episode.
        distance_history: Ring length used by the progress difference.
    """
    radius = _nonnegative_parameter(goal_radius_m, "goal_radius_m")
    if type(max_iterations) is not int or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if type(distance_history) is not int or distance_history < 1:
        raise ValueError("distance_history must be a positive integer")
    state = episode_state(env)
    if state.distances.shape[1] != distance_history:
        state.distances = state.distances.new_zeros((state.distances.shape[0], distance_history))

    distance = goal_distance(env)
    iteration = iteration_index(env)
    index = iteration % distance_history
    seed = iteration == 0
    if bool(seed.any()):
        state.distances[seed] = distance[seed].unsqueeze(-1)
    previous = state.distances.gather(1, index.unsqueeze(-1)).squeeze(-1)
    arrived = distance <= radius
    timed_out = iteration >= max_iterations
    reward = torch.where(
        arrived,
        torch.full_like(distance, r_arrival),
        torch.where(timed_out, torch.full_like(distance, -r_arrival), r_waypoint * (previous - distance)),
    )
    state.distances.scatter_(1, index.unsqueeze(-1), distance.unsqueeze(-1))
    return reward


def obstacle_reward(
    env: object,
    robot_radius_m: float = SOURCE_ROBOT_RADIUS_M,
    r_collision: float = R_COLLISION,
    r_scan: float = R_SCAN_DISTANCE,
    collision_lower_m: float = COLLISION_LOWER_M,
    sensor_name: str = "scan",
    max_range_m: float = 8.0,
) -> torch.Tensor:
    """Penalize collisions and proximity from the front scan, shape ``[num_envs]``.

    Mirrors ``DRLNavEnv._obstacle_collision_punish``: a return inside
    ``[collision_lower_m, robot_radius_m]`` pays ``r_collision``, a return
    inside ``3 * robot_radius_m`` pays ``r_scan`` per metre of missing
    clearance, and anything farther is free. A return closer than
    ``collision_lower_m`` falls into the proximity branch, exactly like the
    source.

    Args:
        env: Environment exposing a front scan.
        robot_radius_m: Robot body radius [m]; the source used the turtlebot's
            0.3 m, so pass the footprint radius of the migrated robot.
        r_collision: Collision constant.
        r_scan: Proximity gain per metre.
        collision_lower_m: Smallest return still considered a collision [m].
        sensor_name: Scene key of the fallback ray caster.
        max_range_m: Finite no-return distance [m].
    """
    radius = _nonnegative_parameter(robot_radius_m, "robot_radius_m")
    lower = _nonnegative_parameter(collision_lower_m, "collision_lower_m")
    scan = minimum_scan(env, sensor_name, max_range_m)
    contact = (scan >= lower) & (scan <= radius)
    proximity = (scan < 3.0 * radius) & ~contact
    return torch.where(
        contact,
        torch.full_like(scan, r_collision),
        torch.where(proximity, r_scan * (3.0 * radius - scan), torch.zeros_like(scan)),
    )


def angular_reward(
    env: object,
    robot_name: str = "robot",
    r_rotation: float = R_ROTATION,
    angular_speed_threshold_rad_s: float = ANGULAR_SPEED_THRESHOLD_RAD_S,
) -> torch.Tensor:
    """Penalize turning faster than the source threshold, shape ``[num_envs]``.

    Mirrors ``DRLNavEnv._angular_velocity_punish``: ``r_rotation`` times the
    absolute yaw rate, but only above ``angular_speed_threshold_rad_s``.

    Args:
        env: Environment with the robot articulation.
        robot_name: Scene key of the robot.
        r_rotation: Non-positive gain applied to the yaw rate.
        angular_speed_threshold_rad_s: Dead band around zero [rad/s].
    """
    threshold = _nonnegative_parameter(angular_speed_threshold_rad_s, "angular_speed_threshold_rad_s")
    _, yaw_rate = robot_velocity(env, robot_name)
    excess = yaw_rate.abs() > threshold
    return torch.where(excess, r_rotation * yaw_rate.abs(), torch.zeros_like(yaw_rate))


def heading_reward(
    env: object,
    robot_name: str = "robot",
    robot_radius_m: float = SOURCE_ROBOT_RADIUS_M,
    r_angle: float = R_ANGLE,
    angle_threshold_rad: float = ANGLE_THRESHOLD_RAD,
    vo_samples: int = VO_SAMPLES,
    pedestrian_range_m: float = VO_PEDESTRIAN_RANGE_M,
) -> torch.Tensor:
    """Reward a heading that keeps the goal bearing outside pedestrian cones.

    Mirrors ``DRLNavEnv._theta_reward``. With no pedestrian the preferred goal
    bearing is used directly; otherwise ``vo_samples`` candidate headings are
    drawn uniformly from ``[-pi, pi)``, headings whose velocity obstacle cone
    against any pedestrian within ``pedestrian_range_m`` is blocked are
    discarded, and the free heading closest to the goal bearing is scored as
    ``r_angle * (angle_threshold_rad - |heading|)``.

    Args:
        env: Environment with the robot and ``pedestrian_bridge``.
        robot_name: Scene key of the robot.
        robot_radius_m: Robot body radius [m]; the excluded cone half-width is
            ``3 * robot_radius_m``, as in the source.
        r_angle: Gain on the angular margin.
        angle_threshold_rad: Angular tolerance around the chosen heading [rad].
        vo_samples: Candidate headings per step.
        pedestrian_range_m: Pedestrians farther than this are ignored [m].
    """
    radius = _nonnegative_parameter(robot_radius_m, "robot_radius_m")
    if type(vo_samples) is not int or vo_samples < 1:
        raise ValueError("vo_samples must be a positive integer")
    reach = _nonnegative_parameter(pedestrian_range_m, "pedestrian_range_m", positive=True)

    goal = local_goal_base(env)
    preferred = torch.atan2(goal[:, 1], goal[:, 0])

    forward, _ = robot_velocity(env, robot_name)
    positions, velocities, valid = _pedestrians_base(env, robot_name)
    angles = _sample_angles(env, vo_samples, positions.dtype, positions.device)
    direction = _velocity_obstacle_heading(
        positions, velocities, valid, angles, preferred, forward, 3.0 * radius, reach
    )
    return r_angle * (angle_threshold_rad - direction.abs())


def _pedestrians_base(
    env: object, robot_name: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return pedestrian planar state in the robot yaw frame.

    Returns:
        Positions [m] ``[num_envs, count, 2]``, velocities [m/s]
        ``[num_envs, count, 2]`` and the validity mask ``[num_envs, count]``.
        Positions are measured from the robot origin, matching the source's
        ``/track_ped`` base-frame poses.
    """
    bridge = getattr(env, "pedestrian_bridge", None)
    if bridge is None:
        count = 0
        zeros = torch.zeros((int(env.num_envs), count, 2), dtype=torch.float32, device=env.device)
        return zeros, zeros.clone(), torch.zeros((int(env.num_envs), count), dtype=torch.bool, device=env.device)
    state, positions, _, valid, robot_data, robot_position = _geometry(env, robot_name)
    velocity = _floating_tensor(state.velocities_w, "velocities_w", positions.shape, positions)
    yaw = _yaw(robot_data, positions)
    cosine, sine = yaw.cos().unsqueeze(-1), yaw.sin().unsqueeze(-1)
    relative = positions[..., :2] - robot_position[:, None, :2]
    rotated_position = torch.stack(
        (cosine * relative[..., 0] + sine * relative[..., 1], -sine * relative[..., 0] + cosine * relative[..., 1]),
        dim=-1,
    )
    rotated_velocity = torch.stack(
        (
            cosine * velocity[..., 0] + sine * velocity[..., 1],
            -sine * velocity[..., 0] + cosine * velocity[..., 1],
        ),
        dim=-1,
    )
    return torch.where(valid.unsqueeze(-1), rotated_position, 0.0), torch.where(
        valid.unsqueeze(-1), rotated_velocity, 0.0
    ), valid


def _yaw(robot_data: object, reference: torch.Tensor) -> torch.Tensor:
    quaternion = _floating_tensor(robot_data.root_quat_w, "root_quat_w", (reference.shape[0], 4), reference)
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if bool((norm <= torch.finfo(quaternion.dtype).eps).any()):
        raise ValueError("root_quat_w must contain a nonzero quaternion in XYZW order")
    quaternion = quaternion / norm
    x, y, z, w = quaternion.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _sample_angles(env: object, count: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Draw ``count`` candidate headings from ``[-pi, pi)``."""
    source = getattr(env, "drl_vo_angle_source", None)
    if source is None:
        return (torch.rand((count,), dtype=dtype, device=device) * 2.0 - 1.0) * math.pi
    angles = torch.as_tensor(source(count), dtype=dtype, device=device)
    if angles.shape != (count,):
        raise ValueError("drl_vo_angle_source must return one heading per sample")
    return angles


def _velocity_obstacle_heading(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    valid: torch.Tensor,
    angles: torch.Tensor,
    preferred: torch.Tensor,
    forward: torch.Tensor,
    cone_half_width_m: float,
    reach_m: float,
) -> torch.Tensor:
    """Pick the free candidate heading closest to the preferred bearing.

    Args:
        positions: Pedestrian positions in the robot frame [m], ``[num_envs, count, 2]``.
        velocities: Pedestrian velocities in the robot frame [m/s], same shape.
        valid: Pedestrian validity ``[num_envs, count]``.
        angles: Candidate headings [rad], ``[samples]``.
        preferred: Goal bearing [rad], ``[num_envs]``.
        forward: Robot forward speed [m/s], ``[num_envs]``.
        cone_half_width_m: Excluded half-width of the collision cone [m].
        reach_m: Pedestrian range gate [m].

    Returns:
        Chosen heading [rad], shape ``[num_envs]``. Without a pedestrian the
        goal bearing is returned; with one but no free candidate the source's
        ``pi/2`` starting value is kept.
    """
    count = positions.shape[1]
    if count == 0:
        return preferred
    distance = torch.linalg.vector_norm(positions, dim=-1)
    near = valid & (distance <= reach_m)
    bearing = torch.atan2(positions[..., 1], positions[..., 0])
    radicand = (distance * distance - cone_half_width_m * cone_half_width_m).clamp_min(0.0)
    half_width = torch.atan2(torch.full_like(radicand, cone_half_width_m), radicand.sqrt())
    sine, cosine = angles.sin()[None, :, None], angles.cos()[None, :, None]
    relative = torch.atan2(
        forward[:, None, None] * sine - velocities[..., 1].unsqueeze(1),
        forward[:, None, None] * cosine - velocities[..., 0].unsqueeze(1),
    )
    inside = (relative >= (bearing - half_width)[:, None, :]) & (relative <= (bearing + half_width)[:, None, :])
    blocked = (inside & near[:, None, :]).any(dim=-1)
    error = (angles[None, :] - preferred[:, None]).pow(2)
    free_error = torch.where(blocked, torch.full_like(error, math.inf), error)
    best = free_error.argmin(dim=-1)
    chosen = angles[best]
    # The source initialised the search at pi/2 and kept it when no heading was free.
    none_free = ~torch.isfinite(free_error.gather(1, best.unsqueeze(-1)).squeeze(-1))
    searched = torch.where(none_free, torch.full_like(chosen, math.pi / 2), chosen)
    # The source always searched when the tracker reported any track, even a distant one.
    return torch.where(valid.any(dim=-1), searched, preferred)


def drl_vo_reward(env: object, **params: object) -> torch.Tensor:
    """Return all four migrated terms summed, shape ``[num_envs]``.

    Convenience for parity tests and for logging the unweighted total. The task
    configuration uses the individual terms as reward terms instead, so the
    manager reports and resets each one.

    Args:
        env: Environment with a goal provider, scan and crowd.

    Returns:
        The scalar reward the source ``_compute_reward`` returned.
    """
    goal_params = _subset(
        params, "robot_name", "goal_radius_m", "r_arrival", "r_waypoint", "max_iterations", "distance_history"
    )
    obstacle_params = _subset(
        params, "robot_radius_m", "r_collision", "r_scan", "collision_lower_m", "sensor_name", "max_range_m"
    )
    angular_params = _subset(params, "robot_name", "r_rotation", "angular_speed_threshold_rad_s")
    heading_params = _subset(
        params,
        "robot_name",
        "robot_radius_m",
        "r_angle",
        "angle_threshold_rad",
        "vo_samples",
        "pedestrian_range_m",
    )
    return (
        goal_reward(env, **goal_params)
        + obstacle_reward(env, **obstacle_params)
        + angular_reward(env, **angular_params)
        + heading_reward(env, **heading_params)
    )


def _subset(params: dict, *names: str) -> dict:
    return {name: params[name] for name in names if name in params}


# ---------------------------------------------------------------------------
# Episode terminations
# ---------------------------------------------------------------------------


def goal_reached(env: object, goal_radius_m: float = GOAL_RADIUS_M) -> torch.Tensor:
    """Terminate when the robot is inside the goal ball, shape ``[num_envs]`` bool.

    Matches the first branch of ``DRLNavEnv._is_done``.

    Args:
        env: Environment with an attached ``drl_vo_goal`` provider.
        goal_radius_m: Arrival tolerance [m], nonnegative.
    """
    radius = _nonnegative_parameter(goal_radius_m, "goal_radius_m")
    return goal_distance(env) <= radius


def collision_budget_exceeded(
    env: object,
    robot_radius_m: float = SOURCE_ROBOT_RADIUS_M,
    collision_lower_m: float = COLLISION_LOWER_M,
    budget: int = COLLISION_BUDGET,
    sensor_name: str = "scan",
    max_range_m: float = 8.0,
    goal_radius_m: float = GOAL_RADIUS_M,
) -> torch.Tensor:
    """Terminate after ``budget`` close-scan steps, shape ``[num_envs]`` bool.

    Matches the second branch of ``DRLNavEnv._is_done``: every step whose
    minimum return falls inside ``[collision_lower_m, robot_radius_m]``
    increments a cumulative counter, and reaching ``budget`` terminates the
    episode and clears the counter. The counter is deliberately *not* cleared by
    a goal or timeout boundary, reproducing the source's carry-over.

    Args:
        env: Environment exposing a front scan.
        robot_radius_m: Robot body radius [m].
        collision_lower_m: Smallest return still counted as a collision [m].
        budget: Number of counted steps that ends the episode.
        sensor_name: Scene key of the fallback ray caster.
        max_range_m: Finite no-return distance [m].
        goal_radius_m: Arrival tolerance; an arrival step does not count [m].
    """
    radius = _nonnegative_parameter(robot_radius_m, "robot_radius_m")
    lower = _nonnegative_parameter(collision_lower_m, "collision_lower_m")
    if type(budget) is not int or budget < 1:
        raise ValueError("budget must be a positive integer")
    state = episode_state(env)
    scan = minimum_scan(env, sensor_name, max_range_m)
    # The source only counts a bump on a step that did not reach the goal.
    arrived = goal_distance(env) <= _nonnegative_parameter(goal_radius_m, "goal_radius_m")
    counted = (scan >= lower) & (scan <= radius) & ~arrived
    state.collisions = state.collisions + counted.long()
    exceeded = state.collisions >= budget
    state.collisions = torch.where(exceeded, torch.zeros_like(state.collisions), state.collisions)
    return exceeded


def iteration_budget_exhausted(env: object, max_iterations: int = MAX_ITERATIONS) -> torch.Tensor:
    """Terminate once the source's iteration budget is spent, shape ``[num_envs]`` bool.

    The source incremented its counter and terminated on ``counter > 512``, with
    the last rewarded iteration being 512. Here ``episode_length_buf`` already
    carries that increment, so the test is ``episode_length_buf > 512``.

    Args:
        env: Environment with ``episode_length_buf``.
        max_iterations: Last rewarded iteration.
    """
    if type(max_iterations) is not int or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    return env.episode_length_buf > max_iterations
