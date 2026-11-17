# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU tests for the frozen DRL-VO 19,202-element observation adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pedstrain_simulation.simulation.configuration.drl import load_drl_policy_settings
from pedstrain_simulation.simulation.ros2_navigation.drl.observation import DrlObservationAdapter
from pedstrain_simulation.simulation.ros2_navigation.drl.package import DrlPackage

CONFIG = Path(__file__).resolve().parents[1] / "configs/runtime/drl_vo.yaml"


def _api():
    settings = load_drl_policy_settings(CONFIG)
    if settings.package_root is None or not settings.package_root.is_dir():
        pytest.skip("external DRL package is not available")
    return DrlPackage.from_settings(settings).load_observation_api()


def _states() -> list[dict[str, object]]:
    return [
        {"name": f"walker_{index}", "position": [float(index), 0.0, 0.0], "velocity": [0.5, 0.0, 0.0],
         "radius_m": 0.25, "height_m": 1.8}
        for index in range(4)
    ]


def _adapter_with_history(api) -> DrlObservationAdapter:
    adapter = DrlObservationAdapter(api)
    scan = np.full(720, 2.0, dtype=np.float32)
    for tick in range(60):
        now = int(round((tick + 1) * 1e9 / 60))
        if tick % 4 == 0:
            adapter.append_scan(scan, now)
        adapter.sample_history(now)
    return adapter


def test_observation_shape_and_direct_equivalence() -> None:
    api = _api()
    adapter = _adapter_with_history(api)
    assert adapter.history_ready
    observation, metadata = adapter.build_observation(_states(), (0.0, 0.0, 0.0), np.array([0.5, 0.0]))
    assert observation.shape == (19202,)
    assert observation.dtype == np.float32
    assert np.isfinite(observation).all()
    assert metadata["pedestrian_count"] == 4
    ped = api.oracle_pedestrian_velocity_map(
        np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32),
        np.array([[0.5, 0.0], [0.5, 0.0], [0.5, 0.0], [0.5, 0.0]], dtype=np.float32),
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
    )
    expected = api.build_observation(ped, adapter.history_array(), np.array([0.5, 0.0]), clip_pedestrian=False)
    assert np.array_equal(observation, np.asarray(expected, dtype=np.float32))


def test_pedestrian_map_orientation_and_absolute_velocity() -> None:
    api = _api()
    positions = np.array([[1.0, 0.0]], dtype=np.float32)
    velocities = np.array([[0.5, 0.0]], dtype=np.float32)
    forward = api.oracle_pedestrian_velocity_map(positions, velocities, np.array([0.0, 0.0, 0.0]))
    assert forward.shape == (2, 80, 80)
    assert np.isclose(forward[0, 4, 40], 0.5)
    assert np.isclose(forward[1, 4, 40], 0.0)
    rotated = api.oracle_pedestrian_velocity_map(
        np.array([[2.0, 1.0]], dtype=np.float32),
        velocities,
        np.array([0.0, 0.0, np.pi / 2]),
    )
    assert np.isclose(rotated[0, 3, 48], 0.0, atol=1e-6)
    assert np.isclose(rotated[1, 3, 48], -0.5, atol=1e-6)


def test_scan_history_requires_ten_causal_frames() -> None:
    api = _api()
    adapter = DrlObservationAdapter(api)
    scan = np.ones(720, dtype=np.float32)
    for tick in range(36):
        now = int(round((tick + 1) * 1e9 / 60))
        if tick % 4 == 0:
            adapter.append_scan(scan, now)
        adapter.sample_history(now)
    assert adapter.history_ready
    assert adapter.history_array().shape == (10, 720)
    with pytest.raises(ValueError):
        adapter.append_scan(scan, 0)
