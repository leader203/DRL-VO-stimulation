# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Soft wall field continuity and strength, separate from collision safety."""

import math

import numpy as np
import pytest

from pedstrain_simulation.simulation.ros2_navigation.wall_force import compute_wall_force


def _ring_walls(count):
    angles = np.arange(count) * (2 * math.pi / count)
    centers = np.column_stack((np.cos(angles), np.sin(angles))) * 0.6
    tangents = np.column_stack((-np.sin(angles), np.cos(angles))) * 0.05
    return np.hstack((centers - tangents, centers + tangents))


def test_eighth_neighbor_rank_exchange_has_no_finite_force_jump():
    walls = _ring_walls(9)
    first = compute_wall_force(np.array([-1e-7, 0.0]), 0.25, 0.03, walls)
    second = compute_wall_force(np.array([1e-7, 0.0]), 0.25, 0.03, walls)
    assert np.linalg.norm(second - first) < 1e-4


def test_equal_distance_ring_cancels_independently_of_input_order():
    walls = _ring_walls(17)
    for segments in (walls, walls[::-1], np.roll(walls, 5, axis=0)):
        np.testing.assert_allclose(compute_wall_force(np.zeros(2), 0.25, 0.03, segments), 0, atol=1e-12)


def test_original_wall_range_has_a_continuous_zero_boundary():
    walls = np.array([[0.0, -2.0, 0.0, 2.0]])
    inner = compute_wall_force(np.array([1.25 - 1e-6, 0.0]), 0.25, 0.03, walls)
    outer = compute_wall_force(np.array([1.25 + 1e-6, 0.0]), 0.25, 0.03, walls)
    assert 0 < inner[0] < 1e-8
    np.testing.assert_array_equal(outer, np.zeros(2))


def test_single_wall_retains_original_nearby_strength_and_decay():
    walls = np.array([[0.0, -2.0, 0.0, 2.0]])
    for distance in (0.3, 0.6, 0.9):
        expected = 3 * math.exp((0.25 + 0.03 - distance) / 0.2)
        np.testing.assert_allclose(compute_wall_force(np.array([distance, 0.0]), 0.25, 0.03, walls), [expected, 0])


def test_symmetric_corridor_centers_an_offset_body_without_selection_bias():
    walls = np.array([[-2.0, -0.6, 2.0, -0.6], [-2.0, 0.6, 2.0, 0.6]])
    np.testing.assert_allclose(compute_wall_force(np.zeros(2), 0.25, 0.03, walls), 0, atol=1e-12)
    above = compute_wall_force(np.array([0.0, 0.05]), 0.25, 0.03, walls)
    below = compute_wall_force(np.array([0.0, -0.05]), 0.25, 0.03, walls)
    assert above[1] < 0
    np.testing.assert_allclose(above, -below, atol=1e-12)


def test_many_identical_boundaries_retain_at_most_eight_wall_weights():
    walls = np.tile([0.0, -2.0, 0.0, 2.0], (64, 1))
    expected = 8 * 3 * math.exp((0.28 - 0.6) / 0.2)
    np.testing.assert_allclose(compute_wall_force(np.array([0.6, 0.0]), 0.25, 0.03, walls), [expected, 0])


def test_wall_field_is_covariant_under_rigid_coordinate_changes():
    walls = _ring_walls(13)
    position = np.array([0.08, -0.03])
    angle = 0.63
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    offset = np.array([8.0, -4.0])
    transformed = (walls.reshape(-1, 2) @ rotation.T + offset).reshape(-1, 4)
    expected = compute_wall_force(position, 0.25, 0.03, walls) @ rotation.T
    np.testing.assert_allclose(
        compute_wall_force(position @ rotation.T + offset, 0.25, 0.03, transformed), expected, atol=1e-12
    )


def test_empty_coincident_and_degenerate_boundaries_remain_finite():
    np.testing.assert_array_equal(compute_wall_force(np.zeros(2), 0.25, 0.03, np.empty((0, 4))), np.zeros(2))
    for wall in ([0.0, -1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]):
        assert np.isfinite(compute_wall_force(np.zeros(2), 0.25, 0.03, np.array([wall]))).all()


@pytest.mark.parametrize(
    ("position", "radius", "margin", "walls"),
    [
        ([math.nan, 0.0], 0.25, 0.03, [[0.0, -1.0, 0.0, 1.0]]),
        ([0.0, 0.0], -0.25, 0.03, [[0.0, -1.0, 0.0, 1.0]]),
        ([0.0, 0.0], 0.25, -0.03, [[0.0, -1.0, 0.0, 1.0]]),
        ([0.0, 0.0], 0.25, 0.03, [[0.0, -1.0, math.inf, 1.0]]),
    ],
)
def test_invalid_wall_fields_are_rejected(position, radius, margin, walls):
    with pytest.raises(ValueError):
        compute_wall_force(np.array(position), radius, margin, np.array(walls))
