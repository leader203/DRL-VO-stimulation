# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU contracts for robot-relative ROS 2 pedestrian observations and MDP terms."""

import math
from types import SimpleNamespace

import pytest
import torch

from pedstrain_simulation.simulation.rl.mdp import pedestrian_clearance, pedestrian_collision, pedestrian_proximity_cost
from pedstrain_simulation.simulation.rl.observations import pedestrian_observations


def _env(positions, *, velocities=None, radii=None, valid=None):
    count = len(positions)
    state = SimpleNamespace(
        positions_w=torch.tensor(positions, dtype=torch.float64).reshape(1, count, 3),
        velocities_w=torch.tensor(
            velocities if velocities is not None else [[0.0, 0.0, 0.0]] * count, dtype=torch.float64
        ).reshape(1, count, 3),
        radii=torch.tensor(radii if radii is not None else [0.25] * count, dtype=torch.float64).reshape(1, count),
        valid=torch.tensor(valid if valid is not None else [True] * count, dtype=torch.bool).reshape(1, count),
    )
    robot = SimpleNamespace(
        root_pos_w=torch.zeros(1, 3),
        root_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        root_link_lin_vel_w=torch.zeros(1, 3),
    )
    bridge = SimpleNamespace(state=state, config=SimpleNamespace(max_pedestrians=count))
    return SimpleNamespace(pedestrian_bridge=bridge, scene={"robot": SimpleNamespace(data=robot)})


def test_observations_rotate_world_deltas_and_use_robot_link_velocity():
    env = _env(
        [[10, 22, 1], [50, 50, 0]],
        velocities=[[1, 2, 0], [10, 10, 0]],
        valid=[True, False],
    )
    robot = env.scene["robot"].data
    # Exercise the Isaac Lab ProxyArray interface without importing Kit or Warp.
    robot.root_pos_w = SimpleNamespace(torch=torch.tensor([[10.0, 20.0, 3.0]]))
    robot.root_quat_w = SimpleNamespace(torch=torch.tensor([[0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]]))
    robot.root_link_lin_vel_w = SimpleNamespace(torch=torch.tensor([[1.0, 0.0, 0.0]]))
    robot.root_lin_vel_w = SimpleNamespace(torch=torch.tensor([[5.0, 10.0, 0.0]]))

    observation = pedestrian_observations(env, robot_radius_m=0.5)
    expected = torch.tensor([[2.0, 0.0, 2.0, 0.0, 0.25, 1.25, 1.0] + [0.0] * 7], dtype=torch.float64)
    torch.testing.assert_close(observation, expected)
    assert observation.device == env.pedestrian_bridge.state.positions_w.device


@pytest.mark.parametrize(
    ("distance", "clearance", "contact", "cost"),
    [(0.5, -0.25, True, 1.0), (0.75, 0.0, True, 1.0), (1.0, 0.25, False, 0.5), (1.25, 0.5, False, 0.0)],
)
def test_footprint_terms_preserve_contact_boundaries_and_ignore_padding(distance, clearance, contact, cost):
    env = _env([[distance, 0.0, 0.0], [0.0, 0.0, 0.0]], valid=[True, False])
    assert pedestrian_clearance(env, robot_radius_m=0.5).tolist() == [clearance]
    result = pedestrian_collision(env, robot_radius_m=0.5)
    assert result.dtype == torch.bool and result.tolist() == [contact]
    assert pedestrian_proximity_cost(env, robot_radius_m=0.5).tolist() == [cost]


@pytest.mark.parametrize("capacity", [0, 2])
def test_empty_crowd_yields_finite_terms_without_contact(capacity):
    env = _env([[0.0, 0.0, 0.0]] * capacity, radii=[0.0] * capacity, valid=[False] * capacity)
    assert pedestrian_observations(env).shape == (1, capacity * 7)
    assert torch.count_nonzero(pedestrian_observations(env)) == 0
    assert pedestrian_clearance(env, max_clearance_m=8.0).tolist() == [8.0]
    assert pedestrian_collision(env, clearance_threshold_m=100.0).tolist() == [False]
    assert pedestrian_proximity_cost(env).tolist() == [0.0]


def test_invalid_snapshot_rejects_wrong_capacity_and_degenerate_heading():
    env = _env([[1.0, 0.0, 0.0]])
    env.scene["robot"].data.root_quat_w.zero_()
    with pytest.raises(ValueError, match="nonzero quaternion"):
        pedestrian_observations(env)
    env.pedestrian_bridge.config.max_pedestrians = 2
    with pytest.raises(ValueError, match="positions_w must have shape"):
        pedestrian_clearance(env)
