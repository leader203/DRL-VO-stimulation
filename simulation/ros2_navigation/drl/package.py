# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Locate and dynamically import the external frozen DRL-VO package."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from types import ModuleType
from pathlib import Path

from ...configuration.drl import DrlPolicySettings


class DrlPackage:
    """One external model package and checkpoint pair."""

    def __init__(self, root: Path, checkpoint: Path) -> None:
        self.root = Path(root)
        self.checkpoint = Path(checkpoint)

    @classmethod
    def from_settings(cls, settings: DrlPolicySettings) -> "DrlPackage":
        if settings.package_root is None:
            raise ValueError("DRL package_root is required.")
        package = cls(settings.package_root, settings.checkpoint or settings.package_root / "frozen_policy.pt")
        package.require()
        return package

    def require(self) -> None:
        if not self.root.is_dir():
            raise ValueError(f"DRL package_root is not an existing directory: {self.root}")
        if not self.checkpoint.is_file():
            raise ValueError(f"DRL checkpoint not found: {self.checkpoint}")
        for relative in ("code/native_ppo.py", "code/drlvo_observation_adapter.py", "code/first_subgoal_alignment.py"):
            if not (self.root / relative).is_file():
                raise ValueError(f"DRL package is missing {relative}: {self.root}")
        if not (self.root / "reference/python").is_dir():
            raise ValueError(f"DRL package is missing reference/python: {self.root}")

    @property
    def checkpoint_sha256(self) -> str:
        digest = hashlib.sha256()
        with self.checkpoint.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _ensure_import_paths(self) -> None:
        code = str(self.root / "code")
        reference = str(self.root / "reference/python")
        for path in (reference, code):
            if path not in sys.path:
                sys.path.insert(0, path)

    def _load(self, relative: str, name: str) -> ModuleType:
        path = self.root / relative
        if not path.is_file():
            raise ValueError(f"DRL package module not found: {path}")
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load DRL package module: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    def load_observation_api(self) -> ModuleType:
        return self._load("code/drlvo_observation_adapter.py", "_pedstrain_drlvo_observation_api")

    def load_alignment_api(self) -> ModuleType:
        return self._load("code/first_subgoal_alignment.py", "_pedstrain_drlvo_alignment_api")

    def load_native_ppo(self) -> ModuleType:
        self._ensure_import_paths()
        return self._load("code/native_ppo.py", "_pedstrain_drlvo_native_ppo")
