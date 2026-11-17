# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Frozen DRL-VO checkpoint loading and deterministic inference."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from .package import DrlPackage


class FrozenDrlVoPolicy:
    """A frozen SB3 policy rebuilt from the package's exact CNN and state dict."""

    def __init__(self, package: DrlPackage, *, device: str = "cpu", expected_sha256: str | None = None) -> None:
        self.package = package
        self.device = str(device)
        package.require()
        self._sha256 = package.checkpoint_sha256
        if expected_sha256 is not None and self._sha256.lower() != expected_sha256.lower():
            raise RuntimeError(
                f"DRL checkpoint SHA256 mismatch: expected {expected_sha256}, got {self._sha256} ({package.checkpoint})"
            )
        torch = importlib.import_module("torch")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("DRL policy requested CUDA but torch.cuda.is_available() is false.")
        native = package.load_native_ppo()
        agent = native.load_native_agent(
            None, device=self.device, seed=1337, rollout_steps=512, batch_size=128, epochs=10
        )
        state = torch.load(str(package.checkpoint), map_location=self.device, weights_only=True)
        if not isinstance(state, dict):
            raise RuntimeError("DRL checkpoint is not a state_dict.")
        agent.policy.load_state_dict(state, strict=True)
        agent.policy.eval()
        for parameter in agent.policy.parameters():
            parameter.requires_grad_(False)
        self._torch = torch
        self._policy = agent.policy
        self._device = torch.device(self.device)
        self.parameter_count = sum(parameter.numel() for parameter in self._policy.parameters())

    @property
    def checkpoint_sha256(self) -> str:
        return self._sha256

    def infer(self, observation: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        """Run one deterministic mean action for a finite 19,202-element observation."""
        value = np.asarray(observation, dtype=np.float32)
        if value.shape != (19202,):
            raise ValueError(f"DRL observation must have shape (19202,), got {value.shape}.")
        if not np.isfinite(value).all():
            raise ValueError("DRL observation contains non-finite values.")
        tensor = self._torch.as_tensor(value[None], dtype=self._torch.float32, device=self._device)
        with self._torch.inference_mode():
            action, value_tensor, log_probability = self._policy.forward(tensor, deterministic=True)
        raw = action[0].detach().cpu().numpy().astype(np.float64)
        if raw.shape != (2,) or not np.isfinite(raw).all():
            raise RuntimeError("DRL policy returned an invalid action.")
        metadata: dict[str, object] = {
            "value": float(value_tensor.item()) if value_tensor is not None else None,
            "log_probability": float(log_probability.item()) if log_probability is not None else None,
            "device": self.device,
        }
        return raw, metadata

    def close(self) -> None:
        self._policy = None

    def __del__(self) -> None:
        self._policy = None
