# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tensor observations from the latest ROS 2 pedestrian snapshot."""

from __future__ import annotations

import math
from numbers import Real

import torch


def pedestrian_observations(env: object, robot_name: str = "robot", robot_radius_m: float = 0.35) -> torch.Tensor:
    """Return fixed-size robot-relative pedestrian features, shape ``[1, N * 7]``.

    Each pedestrian contributes ``[x, y, vx, vy, radius, clearance, valid]``:
    position [m] and relative linear velocity [m/s] use the robot's yaw frame;
    radius and planar circular clearance use [m]. Invalid slots are entirely
    zero. Circular clearance approximates geometry, not PhysX contact reports.

    The robot quaternion follows Isaac Lab's ``(x, y, z, w)`` convention.
    Relative linear velocity subtracts the velocity of the robot's root link
    and rotates the difference into the yaw frame. The observation reads the
    bridge's cached snapshot in the canonical Z-up world frame.

    Args:
        env: Environment with ``pedestrian_bridge`` and ``scene[robot_name]``.
        robot_name: Robot asset key in the scene.
        robot_radius_m: Approximate robot footprint radius [m], nonnegative.

    Returns:
        Features with the bridge position tensor's floating dtype and device.

    Raises:
        ValueError: If configuration, state shape, or numeric values are invalid.
    """
    state, positions, radii, valid, robot_data, robot_position = _geometry(env, robot_name)
    robot_radius = _nonnegative_parameter(robot_radius_m, "robot_radius_m")
    velocity = _floating_tensor(state.velocities_w, "velocities_w", positions.shape, positions)
    # In Isaac Lab root_lin_vel_w aliases the center of mass, while root_pos_w
    # uses the link origin. Prefer velocity measured at the same point.
    robot_linear_velocity = getattr(robot_data, "root_link_lin_vel_w", None)
    if robot_linear_velocity is None:
        robot_linear_velocity = robot_data.root_lin_vel_w
    robot_velocity = _floating_tensor(robot_linear_velocity, "robot root linear velocity", (1, 3), positions)
    quaternion = _floating_tensor(robot_data.root_quat_w, "root_quat_w", (1, 4), positions)
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if bool((norm <= torch.finfo(positions.dtype).eps).any()):
        raise ValueError("root_quat_w must contain a nonzero quaternion in XYZW order")
    quaternion = quaternion / norm
    x, y, z, w = quaternion.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cosine, sine = yaw.cos().unsqueeze(-1), yaw.sin().unsqueeze(-1)

    def rotate_xy(vector: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (cosine * vector[..., 0] + sine * vector[..., 1], -sine * vector[..., 0] + cosine * vector[..., 1]),
            dim=-1,
        )

    relative_position = positions[..., :2] - robot_position[:, None, :2]
    relative_velocity = velocity[..., :2] - robot_velocity[:, None, :2]
    clearance = torch.linalg.vector_norm(relative_position, dim=-1) - radii - robot_radius
    features = torch.cat(
        (
            rotate_xy(relative_position),
            rotate_xy(relative_velocity),
            radii.unsqueeze(-1),
            clearance.unsqueeze(-1),
            valid.to(dtype=positions.dtype).unsqueeze(-1),
        ),
        dim=-1,
    )
    return torch.where(valid.unsqueeze(-1), features, 0.0).reshape(1, -1)


def _nonnegative_parameter(value: float, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0.0 or (positive and value == 0.0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return value


def _floating_tensor(
    value: object, name: str, shape: tuple[int, ...], reference: torch.Tensor | None = None
) -> torch.Tensor:
    # Isaac Lab ProxyArray exposes a zero-copy torch view without importing Warp.
    tensor = value if isinstance(value, torch.Tensor) else getattr(value, "torch", None)
    if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
        raise ValueError(f"{name} must be a floating tensor or a ProxyArray with a .torch view")
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}")
    if reference is not None:
        tensor = tensor.to(device=reference.device, dtype=reference.dtype)
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def _geometry(
    env: object, robot_name: str
) -> tuple[object, torch.Tensor, torch.Tensor, torch.Tensor, object, torch.Tensor]:
    if not isinstance(robot_name, str) or not robot_name.strip():
        raise ValueError("robot_name must be a nonempty scene asset key")
    bridge = env.pedestrian_bridge
    count = bridge.config.max_pedestrians
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("max_pedestrians must be a nonnegative integer")
    # The ROS runtime drove a single world; the cloned-room task publishes one row
    # per robot, so the leading axis follows ``env.num_envs`` instead of a literal 1.
    worlds = int(getattr(env, "num_envs", 1))
    if worlds < 1:
        raise ValueError("num_envs must be positive")
    state = bridge.state
    positions = _floating_tensor(state.positions_w, "positions_w", (worlds, count, 3))
    radii = _floating_tensor(state.radii, "radii", (worlds, count), positions)
    valid = state.valid
    if not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool or tuple(valid.shape) != (worlds, count):
        raise ValueError(f"valid must be a bool tensor of shape [{worlds}, {count}]")
    valid = valid.to(device=positions.device)
    if bool(((radii < 0.0) | (valid & (radii == 0.0))).any()):
        raise ValueError("Pedestrian radii must be positive in valid slots and nonnegative elsewhere [m]")
    robot_data = env.scene[robot_name].data
    robot_position = _floating_tensor(robot_data.root_pos_w, "root_pos_w", (worlds, 3), positions)
    return state, positions, radii, valid, robot_data, robot_position


def _planar_clearances(env: object, robot_name: str, robot_radius_m: float) -> tuple[torch.Tensor, torch.Tensor]:
    radius = _nonnegative_parameter(robot_radius_m, "robot_radius_m")
    _, positions, radii, valid, _, robot_position = _geometry(env, robot_name)
    distances = torch.linalg.vector_norm(positions[..., :2] - robot_position[:, None, :2], dim=-1)
    return distances - radii - radius, valid
