# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional torch test for loading the frozen DRL-VO checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from pedstrain_simulation.simulation.configuration.drl import load_drl_policy_settings
from pedstrain_simulation.simulation.ros2_navigation.drl.package import DrlPackage
from pedstrain_simulation.simulation.ros2_navigation.drl.policy import FrozenDrlVoPolicy

CONFIG = Path(__file__).resolve().parents[1] / "configs/runtime/drl_vo.yaml"


def test_frozen_policy_loads_and_is_deterministic() -> None:
    settings = load_drl_policy_settings(CONFIG)
    if settings.package_root is None or not settings.package_root.is_dir():
        pytest.skip("external DRL package is not available")
    package = DrlPackage.from_settings(settings)
    policy = FrozenDrlVoPolicy(package, device="cpu", expected_sha256=settings.expected_sha256)
    assert policy.checkpoint_sha256 == "bef18b93c76f5eef245942b0aa1bede03faaf14efcc7a0fbc38e972a989996ca"
    assert policy.parameter_count > 2_600_000
    observation = np.zeros(19202, dtype=np.float32)
    first, _ = policy.infer(observation)
    second, _ = policy.infer(observation)
    assert first.shape == (2,)
    assert np.array_equal(first, second)
    assert np.allclose(first, [3.3135927, -0.8180083], atol=1e-4)
    policy.close()
