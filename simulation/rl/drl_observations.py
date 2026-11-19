# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The frozen DRL-VO 19,202-element observation, built from Isaac Lab state.

The layout and the arithmetic are a direct port of
``DRL-VO/reference/custom_cnn_full.py`` (``CustomCNN.forward``) and
``DRL-VO/code/drlvo_observation_adapter.py``:

======  ==============  =======================================================
block   size            content
======  ==============  =======================================================
0       12,800          ``2 x 80 x 80`` pedestrian velocity map, robot frame,
                        20 m x 20 m at 0.25 m per cell, cells hold the
                        pedestrian's absolute velocity rotated into the frame
12,800   6,400          scan history ``10 x 720`` reduced to ``10 x 2 x 80`` by
                        taking the min and the mean of each run of nine beams,
                        flattened and tiled four times to fill ``80 x 80``
19,200       2          pure-pursuit local goal in the robot frame [m]
======  ==============  =======================================================

Normalization follows the ROS 1 source: ``velocity / 2`` for the map,
``2 * range / 30 - 1`` for the scan and ``2 * (goal + 2) / 4 - 1`` for the goal.
Nothing is clipped, so a pedestrian faster than 2 m/s exceeds the nominal
``[-1, 1]`` box exactly as in training.

Everything is batched over the environment axis: the term returns
``[num_envs, 19202]`` and keeps one scan ring per robot, so the same code serves the
single-robot ROS task and the 16-robot cloned room task.
"""

from __future__ import annotations

import math

import torch

from .rewards import _pedestrians_base

PEDESTRIAN_MAP_CELLS = 80
PEDESTRIAN_MAP_CELL_M = 0.25
PEDESTRIAN_MAP_FORWARD_M = PEDESTRIAN_MAP_CELLS * PEDESTRIAN_MAP_CELL_M
PEDESTRIAN_MAP_HALF_WIDTH_M = PEDESTRIAN_MAP_FORWARD_M / 2.0
SCAN_HISTORY = 10
SCAN_BEAMS = 720
SCAN_GROUP = SCAN_BEAMS // PEDESTRIAN_MAP_CELLS
COMPRESSED_SCAN = SCAN_HISTORY * 2 * PEDESTRIAN_MAP_CELLS
OBSERVATION_SIZE = 2 * PEDESTRIAN_MAP_CELLS**2 + PEDESTRIAN_MAP_CELLS**2 + 2


class DrlVoObservationState:
    """Ring of the last ``SCAN_HISTORY`` front scans, shape ``[num_envs, 10, 720]``.

    Args:
        device: Device of the ring.
        dtype: Element type of the ring.
        num_envs: Number of robots sharing the ring.
    """

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32, num_envs: int = 1) -> None:
        self.num_envs = max(int(num_envs), 1)
        self.scans = torch.zeros((self.num_envs, SCAN_HISTORY, SCAN_BEAMS), dtype=dtype, device=device)
        self.filled = 0
        self.stamp: tuple[int, int] | None = None
        # A zero range normalizes to -1, i.e. "an obstacle pressed against every
        # beam". An empty ring would therefore tell the policy that the robot
        # spawns boxed in, so a reset environment is primed with its first real
        # scan instead of being left at zero.
        self.needs_prime = torch.ones(self.num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids: object = None) -> None:
        """Clear the history at an episode boundary.

        Args:
            env_ids: Environment indices to clear; every robot when ``None``.
        """
        if env_ids is None:
            self.scans.zero_()
            self.needs_prime.fill_(True)
            self.filled = 0
        else:
            self.scans[env_ids] = 0.0
            self.needs_prime[env_ids] = True
        self.stamp = None

    def push(self, scan: torch.Tensor) -> None:
        """Append one front scan per environment, dropping the oldest frame.

        Args:
            scan: Ranges [m], shape ``[num_envs, 720]``.
        """
        if tuple(scan.shape) != (self.num_envs, SCAN_BEAMS):
            raise RuntimeError(
                f"A scan push needs shape [{self.num_envs}, {SCAN_BEAMS}], got {tuple(scan.shape)}"
            )
        self.scans[:, :-1] = self.scans[:, 1:].clone()
        self.scans[:, -1] = scan
        primed = self.needs_prime.to(device=scan.device)
        if bool(primed.any()):
            # Fill the whole window so the first observation already looks like a
            # steady state instead of nine frames of contact.
            self.scans[primed] = scan[primed].unsqueeze(1).expand(-1, SCAN_HISTORY, -1)
            self.needs_prime[primed] = False
        self.filled = min(self.filled + 1, SCAN_HISTORY)


def observation_state(env: object) -> DrlVoObservationState:
    """Return this environment's DRL-VO observation buffer, allocating it once."""
    state = getattr(env, "drl_vo_observation_state", None)
    if state is None:
        state = DrlVoObservationState(env.device, num_envs=int(getattr(env, "num_envs", 1)))
        env.drl_vo_observation_state = state
    elif state.num_envs != int(getattr(env, "num_envs", 1)):
        state = DrlVoObservationState(env.device, num_envs=int(getattr(env, "num_envs", 1)))
        env.drl_vo_observation_state = state
    return state


def reset_observation_state(env: object, env_ids: object = None) -> None:
    """Clear the scan history at a reset event.

    Args:
        env: Environment owning the ring.
        env_ids: Environment indices to clear; every robot when ``None``.
    """
    observation_state(env).reset(env_ids)


def drl_vo_observations(
    env: object,
    robot_name: str = "robot",
    robot_radius_m: float = 0.37,
    scan_scale_m: float = 30.0,
) -> torch.Tensor:
    """Return the frozen DRL-VO observation, shape ``[num_envs, 19202]``.

    Args:
        env: Environment with an attached ``drl_vo_goal`` reference, the robot
            articulation and ``drl_vo_scan`` (published by the reference).
        robot_name: Scene key of the robot.
        robot_radius_m: Unused footprint radius kept for parameter symmetry.
        scan_scale_m: Legacy scan divisor [m].

    Returns:
        Normalized observation on the environment device.
    """
    del robot_radius_m
    count = int(env.num_envs)
    provider = getattr(env, "drl_vo_goal", None)
    if provider is None:
        # The manager asks for the term once while the environment is still being
        # built, before the caller can attach the reference: that reference needs a
        # live PhysX scene, which only exists once the environment is constructed.
        if int(getattr(env, "common_step_counter", 0)) == 0:
            return torch.zeros((count, OBSERVATION_SIZE), dtype=torch.float32, device=env.device)
        raise RuntimeError("Attach a DRL-VO reference as env.drl_vo_goal before computing observations.")
    provider.update(env)
    _advance_crowd(env)
    device, dtype = env.device, torch.float32

    positions, velocities, valid = _pedestrians_base(env, robot_name)
    velocity_map = _pedestrian_velocity_map(positions, velocities, valid, count, device, dtype)

    scan = getattr(env, "drl_vo_scan", None)
    if not isinstance(scan, torch.Tensor) or scan.shape != (count, SCAN_BEAMS):
        raise RuntimeError(
            f"The DRL-VO reference must publish env.drl_vo_scan with shape [{count}, {SCAN_BEAMS}]"
        )
    state = observation_state(env)
    stamp = (int(getattr(env, "common_step_counter", 0)), int(env.episode_length_buf[0]))
    if state.stamp != stamp:
        state.push(scan.detach().to(device=device, dtype=dtype))
        state.stamp = stamp

    goal = provider.local_goal_base_xy.to(dtype=dtype)

    observation = torch.cat(
        (
            velocity_map.reshape(count, -1) * 0.5,
            _compress_scan(state.scans) * (2.0 / scan_scale_m) - 1.0,
            (goal + 2.0) * 0.5 - 1.0,
        ),
        dim=1,
    )
    if observation.shape != (count, OBSERVATION_SIZE):
        raise RuntimeError(f"DRL-VO observation has shape {tuple(observation.shape)}, expected {(count, OBSERVATION_SIZE)}")
    return observation


def _advance_crowd(env: object) -> None:
    """Move the scripted pedestrians once per policy step.

    The observation is the first term the manager evaluates after the physics step,
    so this is the point where the walkers, the reward, the pedestrian-contact
    termination and the rendered characters all agree on one instant.

    Args:
        env: Environment that may own ``env.drl_vo_crowd``.
    """
    crowd = getattr(env, "drl_vo_crowd", None)
    if crowd is None:
        return
    crowd.advance(env)
    stage = getattr(env, "drl_vo_characters", None)
    if stage is not None:
        stage.sync(crowd, step=int(getattr(env, "common_step_counter", 0)))


def _pedestrian_velocity_map(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    valid: torch.Tensor,
    count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Rasterize robot-frame pedestrian velocities into ``[num_envs, 2, 80, 80]``.

    Cells hold the pedestrian's absolute velocity in the robot frame; a cell
    written twice keeps the last pedestrian, matching the legacy Python loop.
    """
    result = torch.zeros((count, 2, PEDESTRIAN_MAP_CELLS, PEDESTRIAN_MAP_CELLS), dtype=dtype, device=device)
    if positions.shape[1] == 0:
        return result
    x, y = positions[..., 0], positions[..., 1]
    inside = valid & (x >= 0.0) & (x <= PEDESTRIAN_MAP_FORWARD_M) & (y.abs() <= PEDESTRIAN_MAP_HALF_WIDTH_M)
    if not bool(inside.any()):
        return result
    row = (x / PEDESTRIAN_MAP_CELL_M).floor().clamp(max=PEDESTRIAN_MAP_CELLS - 1).long()
    column = ((PEDESTRIAN_MAP_HALF_WIDTH_M - y) / PEDESTRIAN_MAP_CELL_M).floor().clamp(
        max=PEDESTRIAN_MAP_CELLS - 1
    ).long()
    for index in range(count):
        selected = inside[index]
        if not bool(selected.any()):
            continue
        flat = (row[index][selected] * PEDESTRIAN_MAP_CELLS + column[index][selected]).reshape(1, -1)
        value = velocities[index][selected].to(dtype=dtype)
        result[index, 0].view(-1).scatter_(0, flat.reshape(-1), value[:, 0])
        result[index, 1].view(-1).scatter_(0, flat.reshape(-1), value[:, 1])
    return result


def _compress_scan(history: torch.Tensor) -> torch.Tensor:
    """Reduce ``[num_envs, 10, 720]`` to the legacy per-robot 1,600-element block.

    Args:
        history: Scan history [m], shape ``[num_envs, 10, 720]``.

    Returns:
        Tensor of shape ``[num_envs, 1600]`` holding the group minima and the group
        means, each tiled four times to fill the ``80 x 80`` block.
    """
    if history.dim() != 3 or tuple(history.shape[1:]) != (SCAN_HISTORY, SCAN_BEAMS):
        raise RuntimeError(f"Scan history must have shape [num_envs, {SCAN_HISTORY}, {SCAN_BEAMS}]")
    count = history.shape[0]
    groups = history.reshape(count, SCAN_HISTORY, PEDESTRIAN_MAP_CELLS, SCAN_GROUP)
    reduced = torch.stack((groups.amin(dim=3), groups.mean(dim=3)), dim=2)
    return reduced.reshape(count, -1).repeat(1, 4)


def scan_angles(device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Return the legacy front beam angles [rad] in the robot frame."""
    step = math.pi / SCAN_BEAMS
    return (torch.arange(SCAN_BEAMS, dtype=torch.float32, device=device) * step - math.pi / 2).reshape(1, -1)
