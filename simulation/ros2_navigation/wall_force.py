# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded soft wall acceleration, independent of hard collision certificates."""

import math

import numpy as np


def compute_wall_force(position: np.ndarray, radius: float, margin: float, walls: np.ndarray) -> np.ndarray:
    """Return continuous planar wall repulsion [m/s²].

    Args:
        position: Body center in the wall coordinate frame [m], shape [2].
        radius: Positive body radius [m].
        margin: Nonnegative soft clearance beyond the body [m].
        walls: Segment endpoints [m], shape [N, 4], ordered as x0, y0, x1, y1.

    Returns:
        Planar acceleration [m/s²], shape [2]. Nearby walls retain gain 3 and
        exponential falloff 0.2 [m]. Continuous weights sum to at most eight;
        a 0.2 [m] taper reaches zero at the original radius + 1 [m] range.

    Raises:
        ValueError: Inputs have invalid shapes or nonfinite physical values.
    """
    position = np.asarray(position, dtype=float)
    walls = np.asarray(walls, dtype=float)
    if position.shape != (2,) or walls.ndim != 2 or walls.shape[1] != 4:
        raise ValueError("Wall force requires position [2] and wall segments [N, 4].")
    if (
        not np.isfinite(position).all()
        or not np.isfinite(walls).all()
        or not math.isfinite(radius)
        or not math.isfinite(margin)
        or radius <= 0
        or margin < 0
    ):
        raise ValueError("Wall force requires finite geometry, positive radius and nonnegative margin [m].")
    if not len(walls):
        return np.zeros(2)
    start = walls[:, :2]
    delta = walls[:, 2:] - start
    lengths_sq = np.einsum("ij,ij->i", delta, delta)
    fractions = np.clip(
        np.divide(
            np.einsum("ij,ij->i", position - start, delta),
            lengths_sq,
            out=np.zeros_like(lengths_sq),
            where=lengths_sq > 0,
        ),
        0,
        1,
    )
    vectors = position - (start + fractions[:, None] * delta)
    distances = np.sqrt(np.einsum("ij,ij->i", vectors, vectors))
    cutoff = radius + 1.0
    if len(walls) > 8:
        eighth_distance = np.partition(distances, 7)[7]
        # Order statistics are continuous even when equidistant walls exchange
        # rank; a hard nearest-eight subset is not.
        cutoff = min(cutoff, eighth_distance + 0.2)
    fraction = np.clip((cutoff - distances) / 0.2, 0, 1)
    weights = fraction * fraction * (3 - 2 * fraction)
    weights /= max(1.0, float(np.sum(weights)) / 8.0)
    magnitude = 3.0 * np.exp(np.minimum(5.0, (radius + margin - distances) / 0.2))
    normals = vectors / np.maximum(distances[:, None], 1e-9)
    return np.sum((weights * magnitude)[:, None] * normals, axis=0)
