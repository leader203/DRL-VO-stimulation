# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Differential wheel-velocity action for the EVT base.

``DRLNavEnv._take_action`` mapped the policy's ``[-1, 1]`` pair onto a turtlebot
twist::

    v = a0 * 0.9               # -0.9 .. 0.9 m/s, 0 stops
    w = a1 * 3                 # -3 .. 3 rad/s

The current robot has four driven wheels (``wheel_fl_joint`` .. ``wheel_rr_joint``)
and a ``/cmd_vel`` graph that turns a twist into wheel speeds. Isaac Lab owns the
articulation directly, so this term performs the same mapping and applies the
wheel velocity targets the graph would have produced.
"""

from __future__ import annotations

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

# The crowd table peaks at 0.9 m/s. A 0.5 m/s ceiling cannot clear a doorway
# against a walker coming the other way, so the base gets the same top speed
# as the people it has to avoid. 0.9 / 3.0 keeps the turn radius at 0.30 m,
# which is what the 0.76 m stall doors allow.
MAX_LINEAR_M_S = 0.9
MAX_ANGULAR_RAD_S = 3.0


class DifferentialWheelVelocityAction(ActionTerm):
    """Apply ``(v, w)`` as left/right wheel velocity targets.

    Args:
        cfg: Term configuration.
        env: Environment owning the articulation.
    """

    cfg: DifferentialWheelVelocityActionCfg

    def __init__(self, cfg: DifferentialWheelVelocityActionCfg, env: object) -> None:
        super().__init__(cfg, env)
        if env.num_envs < 1:
            raise ValueError("num_envs must be positive")
        if cfg.wheel_radius_m <= 0 or cfg.wheel_track_m <= 0:
            raise ValueError("wheel_radius_m and wheel_track_m must be positive [m].")
        names = self._asset.joint_names
        indices = {name: index for index, name in enumerate(names)}
        missing = [name for name in cfg.wheel_joint_names if name not in indices]
        if missing:
            raise ValueError(f"The robot has no wheel joints {missing}; joints are {names}")
        self._left_ids, self._right_ids = [], []
        for name in cfg.wheel_joint_names:
            lowered = name.lower()
            if "_fl" in lowered or "_rl" in lowered:
                self._left_ids.append(indices[name])
            elif "_fr" in lowered or "_rr" in lowered:
                self._right_ids.append(indices[name])
            else:
                raise ValueError(f"Cannot classify wheel joint {name} as left or right")
        if not self._left_ids or not self._right_ids:
            raise ValueError("At least one left and one right wheel joint are required")
        left_slots = [slot for slot, name in enumerate(cfg.wheel_joint_names) if indices[name] in self._left_ids]
        right_slots = [slot for slot, name in enumerate(cfg.wheel_joint_names) if indices[name] in self._right_ids]
        self._raw = torch.zeros((env.num_envs, 2), dtype=torch.float32, device=env.device)
        self._targets = torch.zeros(
            (env.num_envs, len(cfg.wheel_joint_names)), dtype=torch.float32, device=env.device
        )
        self._left_slots = torch.tensor(left_slots, dtype=torch.long, device=env.device)
        self._right_slots = torch.tensor(right_slots, dtype=torch.long, device=env.device)
        self._joint_ids = torch.tensor(
            [indices[name] for name in cfg.wheel_joint_names], dtype=torch.long, device=env.device
        )

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        """The clipped policy action actually applied, shape ``[num_envs, 2]``."""
        return self._raw

    @property
    def wheel_velocities(self) -> torch.Tensor:
        """The wheel velocity targets [rad/s], shape ``[num_envs, len(wheel_joint_names)]``."""
        return self._targets

    def process_actions(self, actions: torch.Tensor) -> None:
        # The policy may live on the GPU while the environment keeps its buffers on
        # the device the PhysX tensor API hands back, so the boundary is explicit.
        clipped = actions.reshape(-1, 2).clamp(-1.0, 1.0).to(device=self._raw.device, dtype=torch.float32)
        self._raw.copy_(clipped)
        # Signed, so a blocked robot can back out instead of holding the
        # throttle until the episode ends. The forward-only mapping
        # ``(a + 1) * 0.45`` has no inverse once a wall is reached, and the
        # washroom is small enough that every episode reaches one.
        linear = clipped[:, 0] * MAX_LINEAR_M_S
        angular = clipped[:, 1] * MAX_ANGULAR_RAD_S
        half_track = self.cfg.wheel_track_m / 2.0
        radius = self.cfg.wheel_radius_m
        left = (linear - angular * half_track) / radius
        right = (linear + angular * half_track) / radius
        self._targets[:, self._left_slots] = left.unsqueeze(1)
        self._targets[:, self._right_slots] = right.unsqueeze(1)

    def apply_actions(self) -> None:
        self._asset.set_joint_velocity_target(self._targets, joint_ids=self._joint_ids)

    def reset(self, env_ids: object = None) -> None:
        selection = slice(None) if env_ids is None else env_ids
        self._raw[selection] = 0.0
        self._targets[selection] = 0.0


@configclass
class DifferentialWheelVelocityActionCfg(ActionTermCfg):
    """Configure the differential wheel action for the EVT base."""

    class_type: type[ActionTerm] = DifferentialWheelVelocityAction
    asset_name: str = "robot"
    wheel_joint_names: tuple[str, ...] = ("wheel_fl_joint", "wheel_fr_joint", "wheel_rl_joint", "wheel_rr_joint")
    wheel_radius_m: float = 0.05
    wheel_track_m: float = 0.35
