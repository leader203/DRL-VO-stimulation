# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check the ROS 2 pedestrian action contract without starting Kit or DDS."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def action_contract(monkeypatch):
    from isaaclab.utils.string import ResolvableString

    class PhysxManager:
        folded = False

        @classmethod
        def handles_decimation(cls):
            return cls.folded

    class ActionTerm:
        def __init__(self, cfg, env):
            self.cfg, self._env = cfg, env
            self._asset = env.scene[cfg.asset_name]

    managers = ModuleType("isaaclab.managers")
    managers.ActionTerm, managers.ActionTermCfg = ActionTerm, type("ActionTermCfg", (), {})
    physics = ModuleType("isaaclab_physx.physics")
    physics.PhysxManager = PhysxManager
    monkeypatch.setitem(sys.modules, managers.__name__, managers)
    monkeypatch.setitem(sys.modules, physics.__name__, physics)

    calls = []
    stage = object()
    bridge = SimpleNamespace(
        advance=lambda dt: calls.append(("advance", dt)), reset=lambda ids: calls.append(("reset", ids))
    )

    def create_bridge(*, physics_dt_s, device):
        calls.append(("create", physics_dt_s, device))
        return bridge

    def bound_training_scene(current_stage):
        if current_stage is not stage:
            raise RuntimeError("Prepared ROS 2 stage must be bound before environment construction.")
        return SimpleNamespace(create_bridge=create_bridge)

    scene = ModuleType("pedstrain_simulation.simulation.rl.scene")
    scene._bound_training_scene = bound_training_scene
    monkeypatch.setitem(sys.modules, scene.__name__, scene)

    name = "pedstrain_simulation.simulation.rl._isaaclab_action_test"
    path = Path(__file__).resolve().parents[1] / "rl/isaaclab.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        physics_dt=1.0 / 60.0,
        scene={"robot": object()},
        sim=SimpleNamespace(
            physics_manager=ResolvableString("isaaclab_physx.physics:PhysxManager"),
            cfg=SimpleNamespace(create_stage_in_memory=False),
            stage=stage,
        ),
    )
    return module, env, calls, bridge, PhysxManager


def test_ros2_pedestrian_action_does_not_add_policy_dimensions(action_contract):
    module, env, calls, bridge, _ = action_contract
    cfg = module.Ros2PedestrianActionCfg()
    term = cfg.class_type(cfg, env)

    assert env.pedestrian_bridge is term.bridge is bridge
    assert term.action_dim == 0
    assert term.raw_actions.shape == term.processed_actions.shape == (1, 0)
    assert term.raw_actions.device == torch.device(env.device)
    actions = torch.ones((1, 2))
    term.process_actions(actions[:, 2:])
    with pytest.raises(ValueError, match="empty policy action"):
        term.process_actions(actions)
    assert calls == [("create", env.physics_dt, env.device)]

    for _ in range(2):
        term.apply_actions()
    env_ids = torch.tensor([0], dtype=torch.int32)
    term.reset(env_ids)
    assert calls[1:3] == [("advance", env.physics_dt)] * 2
    assert calls[3][0] == "reset" and calls[3][1] is env_ids
    with pytest.raises(ValueError, match="Only one"):
        cfg.class_type(cfg, env)


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        ("multiple", "num_envs=1"),
        ("backend", "requires Isaac Sim PhysX"),
        ("folded", "per physics substep"),
        ("memory", "create_stage_in_memory=False"),
        ("stage", "must be bound"),
    ],
)
def test_ros2_pedestrian_action_rejects_incompatible_ownership(action_contract, invalid, message):
    module, env, calls, _, manager = action_contract
    if invalid == "multiple":
        env.num_envs = 2
    elif invalid == "backend":
        env.sim.physics_manager = object()
    elif invalid == "folded":
        manager.folded = True
    elif invalid == "memory":
        env.sim.cfg.create_stage_in_memory = True
    else:
        env.sim.stage = object()

    cfg = module.Ros2PedestrianActionCfg()
    with pytest.raises((RuntimeError, ValueError), match=message):
        cfg.class_type(cfg, env)
    assert calls == []
    assert not hasattr(env, "pedestrian_bridge")
