# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional frozen DRL-VO policy configuration; safe to import before Isaac starts."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, fields
from pathlib import Path

from .paths import resolve_config_path

_DEVICE = re.compile(r"^(cpu|cuda(:\d+)?)$")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


def _finite(value: object, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    value = float(value)
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive.")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return value


@dataclass(frozen=True)
class DrlPolicySettings:
    """Frozen DRL-VO observation/action and safety contract for one robot."""

    schema_version: int = 1
    enabled: bool = True
    package_root: Path | None = None
    checkpoint: Path | None = None
    expected_sha256: str | None = None
    device: str = "cpu"
    goal_xy_m: tuple[float, float] | None = None
    goal_tolerance_m: float = 0.3
    goal_settle_s: float = 0.5
    goal_timeout_s: float = 120.0
    replan_period_s: float = 0.5
    path_deviation_m: float = 0.3
    lookahead_m: float = 0.5
    robot_rigid_body_path: str | None = None
    robot_footprint_radius_m: float = 0.37
    robot_clearance_radius_m: float = 0.51
    robot_length_m: float = 0.742
    robot_width_m: float = 0.447
    escape_angular_rad_s: float = 0.6
    escape_reverse_m_s: float = 0.2
    scan_source: str = "occupancy_grid"
    scan_min_range_m: float = 0.1
    scan_max_range_m: float = 8.0
    scan_sample_step_m: float | None = None
    min_scan_stop_m: float = 0.25
    min_scan_release_m: float = 0.35
    max_linear_m_s: float = 0.5
    max_angular_rad_s: float = 2.0
    min_pedestrian_clearance_m: float = 0.15
    pedestrian_release_m: float = 0.25
    pedestrian_horizon_s: float = 0.8
    forward_guard_distance_m: float = 0.25
    max_invalid_observations: int = 3
    policy_hz: float = 20.0
    scan_hz: float = 15.0
    scan_history: int = 10
    alignment_tolerance_rad: float = math.radians(5.0)
    alignment_max_w_rad_s: float = 0.8
    alignment_settle_s: float = 0.15
    alignment_stopped_w_rad_s: float = 0.1
    alignment_kp: float = 2.0
    # enforce: veto/recover after the policy (default).
    # observe: record vetoes but publish the raw policy command.
    # off: publish the raw policy command and only log the guard result.
    guard_mode: str = "enforce"
    alignment_enabled: bool = True
    dry_run: bool = True
    save_observations: bool = False
    log_every_n: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("schema_version must be 1.")
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a boolean.")
        if self.package_root is not None and not isinstance(self.package_root, Path):
            raise ValueError("package_root must be a filesystem Path.")
        if self.checkpoint is not None and not isinstance(self.checkpoint, Path):
            raise ValueError("checkpoint must be a filesystem Path.")
        if not isinstance(self.device, str) or not _DEVICE.match(self.device):
            raise ValueError("device must be cpu, cuda or cuda:N.")
        if self.expected_sha256 is not None:
            if not isinstance(self.expected_sha256, str) or not _SHA256.match(self.expected_sha256):
                raise ValueError("expected_sha256 must be a 64-character hexadecimal digest.")
        if self.goal_xy_m is not None:
            if (
                not isinstance(self.goal_xy_m, (tuple, list))
                or len(self.goal_xy_m) != 2
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in self.goal_xy_m)
            ):
                raise ValueError("goal_xy_m must contain two finite coordinates [m].")
            goal = tuple(float(value) for value in self.goal_xy_m)
            if not all(math.isfinite(value) for value in goal):
                raise ValueError("goal_xy_m must contain two finite coordinates [m].")
            object.__setattr__(self, "goal_xy_m", goal)
        for name in (
            "goal_tolerance_m",
            "replan_period_s",
            "lookahead_m",
            "robot_footprint_radius_m",
            "robot_clearance_radius_m",
            "robot_length_m",
            "robot_width_m",
            "escape_angular_rad_s",
            "escape_reverse_m_s",
            "scan_min_range_m",
            "scan_max_range_m",
            "min_scan_stop_m",
            "min_scan_release_m",
            "max_linear_m_s",
            "max_angular_rad_s",
            "min_pedestrian_clearance_m",
            "pedestrian_release_m",
            "pedestrian_horizon_s",
            "forward_guard_distance_m",
            "policy_hz",
            "scan_hz",
            "alignment_tolerance_rad",
            "alignment_max_w_rad_s",
            "alignment_settle_s",
            "alignment_stopped_w_rad_s",
            "alignment_kp",
        ):
            _finite(getattr(self, name), name, positive=True)
        for name in ("goal_settle_s", "goal_timeout_s", "path_deviation_m"):
            _finite(getattr(self, name), name, nonnegative=True)
        if self.scan_sample_step_m is not None:
            _finite(self.scan_sample_step_m, "scan_sample_step_m", positive=True)
        if not (self.scan_min_range_m < self.scan_max_range_m):
            raise ValueError("scan_min_range_m must be below scan_max_range_m.")
        if self.min_scan_stop_m >= self.min_scan_release_m:
            raise ValueError("min_scan_stop_m must be below min_scan_release_m.")
        if self.min_pedestrian_clearance_m >= self.pedestrian_release_m:
            raise ValueError("min_pedestrian_clearance_m must be below pedestrian_release_m.")
        if not 0 < self.max_linear_m_s <= 0.5:
            raise ValueError("max_linear_m_s must be in (0, 0.5].")
        if not 0 < self.max_angular_rad_s <= 2.0:
            raise ValueError("max_angular_rad_s must be in (0, 2.0].")
        if not math.isclose(self.policy_hz, 20.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("policy_hz must be 20 for the frozen policy contract.")
        if not math.isclose(self.scan_hz, 15.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("scan_hz must be 15 for the frozen policy contract.")
        if type(self.scan_history) is not int or self.scan_history != 10:
            raise ValueError("scan_history must be 10 for the frozen policy contract.")
        if self.scan_source != "occupancy_grid":
            raise ValueError("scan_source must be occupancy_grid.")
        if self.robot_rigid_body_path is not None and (
            not isinstance(self.robot_rigid_body_path, str) or not self.robot_rigid_body_path.startswith("/")
        ):
            raise ValueError("robot_rigid_body_path must be an absolute prim path.")
        for name in ("max_invalid_observations", "log_every_n"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if type(self.dry_run) is not bool or type(self.save_observations) is not bool:
            raise ValueError("dry_run and save_observations must be booleans.")
        if self.guard_mode not in ("enforce", "observe", "off"):
            raise ValueError("guard_mode must be enforce, observe or off.")
        if type(self.alignment_enabled) is not bool:
            raise ValueError("alignment_enabled must be a boolean.")

    def validate_files(self) -> None:
        """Fail early when the external package or frozen checkpoint is missing."""
        if not self.enabled:
            return
        if self.package_root is None or not self.package_root.is_dir():
            raise ValueError(f"DRL package_root is not an existing directory: {self.package_root}")
        checkpoint = self.checkpoint or self.package_root / "frozen_policy.pt"
        if not checkpoint.is_file():
            raise ValueError(f"DRL checkpoint not found: {checkpoint}")
        for relative in ("code/native_ppo.py", "code/drlvo_observation_adapter.py", "reference/python"):
            if not (self.package_root / relative).exists():
                raise ValueError(f"DRL package is missing {relative}: {self.package_root}")


def load_drl_policy_settings(path: Path) -> DrlPolicySettings:
    """Read a strict DRL YAML with file-relative package and checkpoint paths."""
    from .loading import load_yaml_mapping

    source = resolve_config_path(path)
    values = load_yaml_mapping(source)
    allowed = {item.name for item in fields(DrlPolicySettings)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown DRL policy fields: {', '.join(unknown)}")
    values = dict(values)
    if "goal_xy_m" in values and values["goal_xy_m"] is not None:
        goal = values["goal_xy_m"]
        if not isinstance(goal, (list, tuple)) or len(goal) != 2:
            raise ValueError("goal_xy_m must contain two coordinates [m].")
        values["goal_xy_m"] = tuple(goal)
    base = source.parent
    package_root = values.get("package_root")
    if package_root is not None:
        package_root = Path(package_root).expanduser()
        package_root = (base / package_root).resolve() if not package_root.is_absolute() else package_root.resolve()
        values["package_root"] = package_root
    checkpoint = values.get("checkpoint")
    if checkpoint is not None:
        checkpoint = Path(checkpoint).expanduser()
        if not checkpoint.is_absolute():
            if package_root is None:
                raise ValueError("A relative checkpoint requires package_root.")
            checkpoint = package_root / checkpoint
        values["checkpoint"] = checkpoint.resolve()
    elif package_root is not None:
        values["checkpoint"] = (package_root / "frozen_policy.pt").resolve()
    settings = DrlPolicySettings(**values)
    settings.validate_files()
    return settings
