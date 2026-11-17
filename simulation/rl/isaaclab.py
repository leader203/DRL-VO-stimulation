# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A zero-dimensional Isaac Lab action term for ROS 2 pedestrians."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class Ros2PedestrianAction(ActionTerm):
    """Advance the prepared ROS 2 crowd before each Isaac Lab physics substep.

    Add the term alongside the task's robot action in a single environment.
    Construct the environment inside the prepared training scene's bind scope.
    The prepared scene owns bridge cleanup.
    """

    def __init__(self, cfg: Ros2PedestrianActionCfg, env: ManagerBasedEnv) -> None:
        from isaaclab_physx.physics import PhysxManager

        from isaaclab.utils.string import string_to_callable

        from .scene import _bound_training_scene

        if env.num_envs != 1:
            raise ValueError("ROS 2 pedestrian training requires num_envs=1.")
        physics_manager = env.sim.physics_manager
        if isinstance(physics_manager, str):
            physics_manager = string_to_callable(str(physics_manager))
        if physics_manager is not PhysxManager:
            raise ValueError("ROS 2 pedestrian training requires Isaac Sim PhysX.")
        if physics_manager.handles_decimation():
            raise ValueError("ROS 2 pedestrian training requires one action application per physics substep.")
        if env.sim.cfg.create_stage_in_memory:
            raise ValueError("Reuse the prepared ROS 2 stage with sim.create_stage_in_memory=False.")
        if hasattr(env, "pedestrian_bridge"):
            raise ValueError("Only one ROS 2 pedestrian action term may own the environment's crowd.")

        prepared = _bound_training_scene(env.sim.stage)
        super().__init__(cfg, env)
        self._empty_actions = torch.empty((1, 0), device=env.device)
        self._export_IO_descriptor = False
        self.bridge = prepared.create_bridge(physics_dt_s=env.physics_dt, device=env.device)
        env.pedestrian_bridge = self.bridge

    @property
    def action_dim(self) -> int:
        """The pedestrian term contributes zero policy action dimensions."""
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        """Empty policy action tensor with shape [1, 0]."""
        return self._empty_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Empty processed action tensor with shape [1, 0]."""
        return self._empty_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        """Validate the zero-dimensional slice supplied by ActionManager."""
        if actions.shape != self._empty_actions.shape:
            raise ValueError("The ROS 2 pedestrian term accepts an empty policy action tensor of shape [1, 0].")

    def apply_actions(self) -> None:
        """Advance the crowd by one physics interval [s] before physics runs."""
        self.bridge.advance(self._env.physics_dt)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Handle a robot episode boundary while preserving crowd progression."""
        self.bridge.reset(env_ids)


@configclass
class Ros2PedestrianActionCfg(ActionTermCfg):
    """Configure the pedestrian clock as a zero-dimensional action term."""

    class_type: type[ActionTerm] = Ros2PedestrianAction
    asset_name: str = "robot"
