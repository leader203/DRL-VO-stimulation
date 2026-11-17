# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU tests for the NumPy-2-safe DRL-VO pure-pursuit local goal."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pedstrain_simulation.simulation.ros2_navigation.drl.local_goal import local_goal_base


def test_straight_lookahead() -> None:
    goal = local_goal_base(np.array([[0.0, 0.0], [3.0, 0.0]]), (0.0, 0.0, 0.0), 0.5)
    assert np.allclose(goal, [0.5, 0.0])


def test_behind_path_targets_closest_point() -> None:
    goal = local_goal_base(np.array([[2.0, 0.0], [4.0, 0.0]]), (0.0, 0.0, 0.0), 0.5)
    assert np.allclose(goal, [2.0, 0.0])


def test_single_point_and_duplicate_points() -> None:
    assert np.allclose(local_goal_base([[1.0, 2.0]], (0.0, 0.0, 0.0), 0.5), [1.0, 2.0])
    assert np.allclose(local_goal_base([[0.0, 0.0], [0.0, 0.0], [3.0, 0.0]], (0.0, 0.0, 0.0), 0.5), [0.5, 0.0])


def test_base_frame_rotation() -> None:
    goal = local_goal_base(np.array([[0.0, 0.0], [3.0, 0.0]]), (0.0, 0.0, math.pi / 2), 0.5)
    assert np.allclose(goal, [0.0, -0.5], atol=1e-9)


def test_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        local_goal_base(np.array([[0.0, 0.0]]), (0.0, 0.0), 0.5)
    with pytest.raises(ValueError):
        local_goal_base(np.array([[0.0, 0.0], [1.0, 0.0]]), (0.0, 0.0, 0.0), 0.0)
    with pytest.raises(ValueError):
        local_goal_base(np.array([[0.0, 0.0], [np.nan, 0.0]]), (0.0, 0.0, 0.0), 0.5)
