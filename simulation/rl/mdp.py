# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable manager terms based on approximate circular pedestrian footprints."""

from __future__ import annotations

import torch

from .observations import _nonnegative_parameter, _planar_clearances


def pedestrian_clearance(
    env: object,
    robot_name: str = "robot",
    robot_radius_m: float = 0.35,
    max_clearance_m: float = 100.0,
) -> torch.Tensor:
    """Return nearest planar footprint clearance [m], shape ``[1]``.

    Distances are capped at ``max_clearance_m``; an empty scene returns that
    finite cap. Negative clearance means overlapping circular approximations,
    not an observed PhysX contact. This term reads the bridge's cached state.

    Args:
        env: Environment with ``pedestrian_bridge`` and ``scene[robot_name]``.
        robot_name: Robot asset key in the scene.
        robot_radius_m: Approximate robot footprint radius [m], nonnegative.
        max_clearance_m: Finite positive distance cap and empty-scene value [m].

    Returns:
        Clearances with the bridge position tensor's floating dtype and device.
    """
    maximum = _nonnegative_parameter(max_clearance_m, "max_clearance_m", positive=True)
    clearances, valid = _planar_clearances(env, robot_name, robot_radius_m)
    if clearances.shape[1] == 0:
        return clearances.new_full((1,), maximum)
    return torch.where(valid, clearances, maximum).amin(dim=-1).clamp_max(maximum)


def pedestrian_collision(
    env: object,
    robot_name: str = "robot",
    robot_radius_m: float = 0.35,
    clearance_threshold_m: float = 0.0,
) -> torch.Tensor:
    """Flag circular footprint contact or proximity, shape ``[1]`` and bool dtype.

    Contact includes touching at the threshold. The test is a Z-up planar
    approximation, not a PhysX contact query; different-height bodies may be
    flagged. Padding is ignored, and no pedestrians always returns ``False``.

    Args:
        env: Environment with ``pedestrian_bridge`` and ``scene[robot_name]``.
        robot_name: Robot asset key in the scene.
        robot_radius_m: Approximate robot footprint radius [m], nonnegative.
        clearance_threshold_m: Nonnegative extra clearance triggering contact [m].

    Returns:
        Contact flags on the bridge position tensor's device.
    """
    threshold = _nonnegative_parameter(clearance_threshold_m, "clearance_threshold_m")
    clearances, valid = _planar_clearances(env, robot_name, robot_radius_m)
    return ((clearances <= threshold) & valid).any(dim=-1)


def pedestrian_proximity_cost(
    env: object,
    robot_name: str = "robot",
    robot_radius_m: float = 0.35,
    desired_clearance_m: float = 0.5,
) -> torch.Tensor:
    """Return a bounded circular-footprint proximity cost, shape ``[1]``.

    The cost increases linearly from zero at ``desired_clearance_m`` to one
    at contact, and stays one for overlap. With no pedestrians it is zero.
    Use a negative reward weight to penalize this positive cost. The geometry
    is a planar approximation, not a PhysX contact measurement.

    Args:
        env: Environment with ``pedestrian_bridge`` and ``scene[robot_name]``.
        robot_name: Robot asset key in the scene.
        robot_radius_m: Approximate robot footprint radius [m], nonnegative.
        desired_clearance_m: Positive clearance at which the cost reaches zero [m].

    Returns:
        Costs in [0, 1], with the bridge position tensor's floating dtype and device.
    """
    desired = _nonnegative_parameter(desired_clearance_m, "desired_clearance_m", positive=True)
    clearance = pedestrian_clearance(env, robot_name, robot_radius_m, max_clearance_m=desired)
    return ((desired - clearance) / desired).clamp(0.0, 1.0)
