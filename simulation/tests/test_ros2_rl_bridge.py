# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify the external physics clock, reset boundary, and failure behavior."""

import math
from types import SimpleNamespace

import pytest

from pedstrain_simulation.simulation.configuration.navigation import Ros2NavigationSettings
from pedstrain_simulation.simulation.rl.bridge import Ros2PedestrianBridge, _physics_ratio, _TrainingPhysicsBodies
from pedstrain_simulation.simulation.rl.scene import Ros2TrainingScene, _bound_training_scene
from pedstrain_simulation.simulation.ros2_navigation.frames import StageFrame


class _Actors:
    def __init__(self):
        self.body = {"name": "walker", "position": [0, 0, 0], "velocity": [0, 0, 0], "radius_m": 0.25}

    def states(self):
        return [self.body]

    def apply(self, commands, dt):
        self.body = commands[0]


class _Transport:
    def __init__(self):
        self.calls = []
        self.failure = None
        self.closed = False

    def step(self, time_s, dt, agents, *, obstacles, door_states):
        self.calls.append((time_s, dt, obstacles))
        if self.failure is not None:
            raise self.failure
        walker = agents[0]
        x, y, z = walker["position"]
        return {"agents": [{**walker, "position": [x + dt, y, z], "velocity": [1, 0, 0]}]}

    def close(self):
        self.closed = True


@pytest.fixture
def prepared_bridge():
    actors, transport = _Actors(), _Transport()
    robot = {"prim_path": "/Robot", "position": [0.0, 0.0, 0.2]}
    doors = SimpleNamespace(
        observations=lambda now: [],
        set_navigation_intents=lambda _: None,
        update=lambda *args: None,
        close=lambda: None,
    )
    bridge = Ros2PedestrianBridge(
        frame=StageFrame(1, "Z", (10, -2, 0), math.pi / 2),
        navigation=Ros2NavigationSettings(control_hz=50),
        actors=actors,
        obstacles=SimpleNamespace(sample=lambda: [robot], close=lambda: None),
        doors=doors,
        transport=transport,
        kinematics=SimpleNamespace(apply=lambda *args: None, close=lambda: None),
        capacity=2,
        device="cpu",
        physics_dt_s=0.01,
    )
    try:
        yield bridge, transport, robot
    finally:
        bridge.close()


def test_physics_substeps_and_resets_preserve_the_ros_clock(prepared_bridge):
    bridge, transport, robot = prepared_bridge
    bridge.advance(0.01)
    assert transport.calls == []
    bridge.advance(0.01)
    assert bridge.control_steps == 1
    assert bridge.state.positions_w.tolist()[0][0] == pytest.approx([10, -1.98, 0])
    assert bridge.state.velocities_w.tolist()[0][0] == pytest.approx([0, 1, 0], abs=1e-7)
    assert bridge.state.valid.tolist() == [[True, False]]
    robot["position"] = [0.04, 0.0, 0.2]
    bridge.reset(slice(0, 0))
    bridge.advance(0.01)
    bridge.advance(0.01)
    assert transport.calls[1][2][0]["velocity"] == pytest.approx([2, 0, 0])
    snapshot = bridge.state
    bridge.reset([0])
    assert bridge.state is snapshot
    assert bridge.control_steps == 2 and len(transport.calls) == 2
    robot["position"] = [1.0, 0.0, 0.2]
    bridge.advance(0.01)
    bridge.advance(0.01)
    assert [call[0] for call in transport.calls] == pytest.approx([0.02, 0.04, 0.06])
    assert transport.calls[2][2][0]["velocity"] == [0, 0, 0]
    assert bridge.state.positions_w.tolist()[0][0] == pytest.approx([10, -1.94, 0])
    with pytest.raises(ValueError, match="ID 0"):
        bridge.reset([True])


def test_transport_failure_invalidates_observations_and_cannot_be_retried(prepared_bridge):
    bridge, transport, _ = prepared_bridge
    transport.failure = TimeoutError("lost ROS reply")
    bridge.advance(0.01)
    with pytest.raises(TimeoutError, match="lost ROS reply"):
        bridge.advance(0.01)
    with pytest.raises(RuntimeError, match="rebuild"):
        _ = bridge.state
    with pytest.raises(RuntimeError, match="rebuild"):
        bridge.reset()
    with pytest.raises(RuntimeError, match="rebuild"):
        bridge.advance(0.01)
    assert len(transport.calls) == 1 and bridge.control_steps == 0
    bridge.close()
    assert transport.closed


@pytest.mark.parametrize("dt", [False, 0, 0.015, 1 / 30, math.nan])
def test_training_rejects_non_integral_or_invalid_physics_clock(dt):
    with pytest.raises(ValueError, match="physics substep|Physics dt"):
        _physics_ratio(dt, 60)


def test_training_binding_is_scoped_to_the_prepared_stage():
    prepared = Ros2TrainingScene.__new__(Ros2TrainingScene)
    prepared._stage = object()
    with prepared.bind():
        assert _bound_training_scene(prepared.stage) is prepared
        with pytest.raises(RuntimeError, match="prepared ROS stage"):
            _bound_training_scene(object())
    with pytest.raises(RuntimeError, match="prepared ROS stage"):
        _bound_training_scene(prepared.stage)


def test_kinematic_targets_use_physx_row_order_and_stage_frame():
    import torch

    writes = []
    targets = _TrainingPhysicsBodies.__new__(_TrainingPhysicsBodies)
    targets._names = ("second", "first")
    targets._targets = torch.zeros((2, 7))
    targets._indices = torch.arange(2, dtype=torch.int32)
    targets._view = SimpleNamespace(set_kinematic_targets=lambda pose, ids: writes.append((pose.clone(), ids.tolist())))
    frame = StageFrame(1, "Z", (10, 20, 0), math.pi / 2)
    targets.apply(
        [
            {"name": "first", "position": [0, 1, 0], "yaw": -math.pi / 2},
            {"name": "second", "position": [2, 0, 0], "yaw": 0.0},
        ],
        frame,
    )
    pose, ids = writes[0]
    assert ids == [0, 1]
    assert pose[0].tolist() == pytest.approx([10, 22, 0, 0, 0, math.sqrt(0.5), math.sqrt(0.5)])
    assert pose[1].tolist() == pytest.approx([9, 20, 0, 0, 0, 0, 1])


def test_rigid_pose_reader_refreshes_transforms_without_rebuilding_an_unchanged_view():
    import torch

    builds = []
    data = torch.tensor([[1, 0, 0, 0, 0, 0, 1], [2, 0, 0, 0, 0, 0, 1]], dtype=torch.float32)
    bodies = _TrainingPhysicsBodies.__new__(_TrainingPhysicsBodies)
    bodies._read_views = {}

    def create(paths):
        builds.append(paths)
        return SimpleNamespace(prim_paths=tuple(reversed(paths)), get_transforms=lambda: data)

    bodies._simulation = SimpleNamespace(create_rigid_body_view=create)
    assert bodies.read_poses(("/A", "/B"), group="obstacles")["/B"]["position"] == [1, 0, 0]
    data[0, 0] = 4
    assert bodies.read_poses(("/A", "/B"), group="obstacles")["/B"]["position"] == [4, 0, 0]
    assert builds == [["/A", "/B"]]
