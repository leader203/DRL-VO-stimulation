# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU tests for the optional DRL policy configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from pedstrain_simulation.simulation.cli.arguments import create_parser
from pedstrain_simulation.simulation.configuration.drl import DrlPolicySettings, load_drl_policy_settings


def _package(tmp_path: Path) -> Path:
    root = tmp_path / "package"
    (root / "code").mkdir(parents=True)
    (root / "reference/python").mkdir(parents=True)
    for name in ("native_ppo.py", "drlvo_observation_adapter.py", "first_subgoal_alignment.py"):
        (root / "code" / name).write_text("")
    (root / "frozen_policy.pt").write_bytes(b"stub")
    return root


def test_example_config_loads() -> None:
    settings = load_drl_policy_settings(Path(__file__).resolve().parents[1] / "configs/runtime/drl_vo.yaml")
    assert settings.enabled
    assert settings.device == "cpu"
    assert settings.goal_xy_m == (3.53016, 1.4301)
    assert settings.policy_hz == 20.0
    assert settings.scan_hz == 15.0
    assert settings.validate_files is not None


def test_relative_package_and_checkpoint(tmp_path: Path) -> None:
    root = _package(tmp_path)
    config = tmp_path / "drl.yaml"
    config.write_text(
        "schema_version: 1\n"
        "package_root: package\n"
        "checkpoint: frozen_policy.pt\n"
        "goal_xy_m: [1.0, 2.0]\n"
        "dry_run: true\n"
    )
    settings = load_drl_policy_settings(config)
    assert settings.package_root == root.resolve()
    assert settings.checkpoint == (root / "frozen_policy.pt").resolve()


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    root = _package(tmp_path)
    config = tmp_path / "drl.yaml"
    config.write_text(f"package_root: {root}\nbogus: 1\n")
    with pytest.raises(ValueError, match="Unknown DRL policy fields"):
        load_drl_policy_settings(config)


def test_missing_checkpoint_is_rejected(tmp_path: Path) -> None:
    root = _package(tmp_path)
    (root / "frozen_policy.pt").unlink()
    config = tmp_path / "drl.yaml"
    config.write_text(f"package_root: {root}\n")
    with pytest.raises(ValueError, match="checkpoint not found"):
        load_drl_policy_settings(config)


def test_invalid_contract_values() -> None:
    with pytest.raises(ValueError):
        DrlPolicySettings(policy_hz=10.0)
    with pytest.raises(ValueError):
        DrlPolicySettings(scan_hz=10.0)
    with pytest.raises(ValueError):
        DrlPolicySettings(max_linear_m_s=1.0)
    with pytest.raises(ValueError):
        DrlPolicySettings(min_scan_stop_m=0.5, min_scan_release_m=0.4)


def test_parser_exposes_drl_flags() -> None:
    args = create_parser().parse_args(
        ["--drl-policy-config", "drl.yaml", "--drl-goal", "1", "2", "--drl-device", "cpu"]
    )
    assert args.drl_policy_config == Path("drl.yaml")
    assert args.drl_goal == [1.0, 2.0]
    assert args.drl_device == "cpu"
