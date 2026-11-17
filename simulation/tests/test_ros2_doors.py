# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise metadata-driven environment doors without running PhysX or an agent SDK."""

import copy
import math
import sys
from types import ModuleType, SimpleNamespace

import pytest

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from pedstrain_simulation.simulation.ros2_navigation.doors import Ros2Doors
from pedstrain_simulation.simulation.ros2_navigation.frames import StageFrame


def _fixture(*, transformed=False):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    parent = UsdGeom.Xform.Define(stage, "/Campus")
    if transformed:
        parent.AddTranslateOp().Set((10, -3, 2))
        parent.AddRotateZOp().Set(90)
        parent.AddScaleOp().Set((2, 1, 1))
    controller = UsdGeom.Xform.Define(stage, "/Campus/Gates").GetPrim()
    for name, value in (("approachDistance", 0.6), ("activationHalfWidth", 0.32), ("closeDelay", 0.5), ("speed", 60)):
        controller.CreateAttribute("automaticDoor:" + name, Sdf.ValueTypeNames.Double).Set(value)
    config = UsdGeom.Xform.Define(stage, "/Campus/Gates/ArbitraryLabel").GetPrim()
    config.CreateAttribute("automaticDoor:center", Sdf.ValueTypeNames.Double3).Set((0, 0, 0))
    config.CreateAttribute("automaticDoor:interiorMin", Sdf.ValueTypeNames.Double2).Set((-0.5, -1.5))
    config.CreateAttribute("automaticDoor:interiorMax", Sdf.ValueTypeNames.Double2).Set((0.5, -0.2))
    config.CreateAttribute("automaticDoor:openAngle", Sdf.ValueTypeNames.Double).Set(80)
    leaf = UsdGeom.Cube.Define(stage, "/Campus/Panel")
    leaf.CreateSizeAttr(1)
    leaf.AddTranslateOp().Set((0, -0.03, 1))
    leaf.AddScaleOp().Set((0.8, 0.06, 2))
    UsdPhysics.CollisionAPI.Apply(leaf.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.RigidBodyAPI.Apply(leaf.GetPrim())
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Campus/Hinge")
    joint.CreateAxisAttr("Z")
    joint.CreateBody1Rel().SetTargets([leaf.GetPath()])
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTargetPositionAttr(0)
    drive.CreateStiffnessAttr(100)
    config.CreateRelationship("automaticDoor:joint").SetTargets([joint.GetPath()])
    frame = StageFrame(1, "Z", (1, 2, 0), 0.3)
    return stage, frame, parent, config, drive


def _state(frame, parent, point=(0, 0.5, 0), *, yaw=-math.pi / 2, velocity=(0, 0, 0), name="person"):
    matrix = parent.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    world = matrix.Transform(Gf.Vec3d(*point))
    position = frame.stage_to_ros(world)
    forward = matrix.TransformDir(Gf.Vec3d(math.cos(yaw), math.sin(yaw), 0))
    tip = frame.stage_to_ros(world + forward)
    world_velocity = matrix.TransformDir(Gf.Vec3d(*velocity))
    return {
        "name": name,
        "position": list(position),
        "yaw": math.atan2(tip[1] - position[1], tip[0] - position[0]),
        "velocity": frame.stage_vector_to_ros(world_velocity),
        "radius_m": 0.2,
        "height_m": 1.8,
    }


@pytest.mark.parametrize("transformed", [False, True])
def test_external_approach_uses_full_parent_transform_and_degree_per_second_ramp(transformed):
    stage, frame, parent, config, drive = _fixture(transformed=transformed)
    before = stage.GetRootLayer().ExportToString()
    state = _state(frame, parent)
    original = copy.deepcopy(state)
    doors = Ros2Doors(stage, frame)
    doors.update([state], 0.0, 0.1)
    assert drive.GetTargetPositionAttr().Get() == pytest.approx(6.0)
    assert state == original
    assert stage.GetPrimAtPath("/Campus/Panel").GetAttribute("physics:collisionEnabled").Get() is True
    assert stage.GetRootLayer().ExportToString() == before
    assert doors.status()["doors"][str(config.GetPath())]["requested_open"]
    doors.close()
    assert drive.GetTargetPositionAttr().Get() == 0


def test_parent_scale_does_not_expand_the_physical_approach_width():
    stage, frame, parent, _, drive = _fixture(transformed=True)
    doors = Ros2Doors(stage, frame)
    doors.update([_state(frame, parent, (0.2, 0.5, 0))], 0, 0.1)
    assert drive.GetTargetPositionAttr().Get() == 0
    doors.close()


def test_inside_occupancy_blocks_entry_but_not_an_occupants_outward_departure():
    stage, frame, parent, config, drive = _fixture()
    doors = Ros2Doors(stage, frame)
    visitor = _state(frame, parent, name="visitor")
    occupant = _state(frame, parent, (0, -1.2, 0), name="occupant")
    doors.update([visitor, occupant], 0, 0.1)
    assert drive.GetTargetPositionAttr().Get() == 0
    assert doors.status()["doors"][str(config.GetPath())]["interior_occupied"]
    departing = _state(frame, parent, (0, -0.4, 0), yaw=math.pi / 2, velocity=(0, 0.3, 0), name="occupant")
    doors.update([departing], 0.1, 0.1)
    assert drive.GetTargetPositionAttr().Get() == pytest.approx(6)
    doors.close()


def test_threshold_and_swing_occupancy_hold_closing_without_disabling_collision():
    stage, frame, parent, config, drive = _fixture()
    drive.GetTargetPositionAttr().Set(80)
    doors = Ros2Doors(stage, frame)
    opened = drive.GetTargetPositionAttr().Get()
    # Wrong-facing threshold occupant still protects a physically open door.
    doors.update([_state(frame, parent, (0, -0.3, 0), yaw=-math.pi / 2)], 1.0, 0.1)
    assert drive.GetTargetPositionAttr().Get() == opened
    assert doors.status()["doors"][str(config.GetPath())]["closing_protected"]
    doors.update([], 1.3, 0.1)
    held = drive.GetTargetPositionAttr().Get()
    assert held >= opened
    doors.update([], 1.6, 0.1)
    assert drive.GetTargetPositionAttr().Get() == pytest.approx(held - 6)
    assert stage.GetPrimAtPath("/Campus/Panel").GetAttribute("physics:collisionEnabled").Get() is True
    doors.close()


def test_authorized_departure_finishes_opening_after_local_avoidance_stops_the_body():
    stage, frame, parent, config, drive = _fixture()
    controller = config.GetParent()
    # Actual automaticDoor metadata: 180 deg/s, 88 deg, 0.25 s hold, 0.55 m approach.
    for name, value in (("speed", 180.0), ("closeDelay", 0.25), ("approachDistance", 0.55)):
        controller.GetAttribute("automaticDoor:" + name).Set(value)
    config.GetAttribute("automaticDoor:openAngle").Set(88.0)
    doors = Ros2Doors(stage, frame)
    # Reported stopped state relative to the authored doorway center; all distances [m].
    moving = _state(frame, parent, (-0.0656, -0.4905, 0), yaw=math.pi / 2, velocity=(0, 0.3, 0))
    moving["radius_m"] = 0.25
    stopped = dict(moving, velocity=[0.0, 0.0, 0.0])
    dt = 1 / 60
    for index in range(4):
        doors.update([moving], index * dt, dt)
    assert drive.GetTargetPositionAttr().Get() == pytest.approx(12)
    targets = []
    for index in range(4, 64):
        doors.update([stopped], index * dt, dt)
        targets.append(drive.GetTargetPositionAttr().Get())
    assert targets[-1] == pytest.approx(88)
    assert all(0 <= b - a <= 180 * dt + 1e-6 for a, b in zip(targets, targets[1:]))
    assert not doors.status()["doors"][str(config.GetPath())]["requested_open"]
    assert stage.GetPrimAtPath("/Campus/Panel").GetAttribute("physics:collisionEnabled").Get() is True
    doors.close()


def test_doorway_occupancy_authorizes_opening_without_speed_or_facing():
    stage, frame, parent, _, drive = _fixture()
    doors = Ros2Doors(stage, frame)
    state = _state(frame, parent, (0, 0.1, 0), yaw=math.pi / 2, velocity=(0, 0, 0))
    doors.update([state], 0, 0.1)
    assert drive.GetTargetPositionAttr().Get() == pytest.approx(6)
    doors.close()


def test_other_floor_and_zero_dt_do_not_move_the_door():
    stage, frame, parent, _, drive = _fixture()
    doors = Ros2Doors(stage, frame)
    doors.update([_state(frame, parent, (0, 0.4, 5))], 0.0, 0.1)
    assert drive.GetTargetPositionAttr().Get() == 0
    doors.update([_state(frame, parent)], 0.1, 0.0)
    assert drive.GetTargetPositionAttr().Get() == 0
    doors.close()


def test_close_removes_only_its_layer_and_preserves_newer_external_target():
    stage, frame, parent, _, drive = _fixture()
    unrelated = Sdf.Layer.CreateAnonymous("unrelated.usda")
    stage.GetSessionLayer().subLayerPaths.append(unrelated.identifier)
    doors = Ros2Doors(stage, frame)
    doors.update([_state(frame, parent)], 0.0, 0.1)
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        drive.GetTargetPositionAttr().Set(13)
    doors.close()
    doors.close()
    assert list(stage.GetSessionLayer().subLayerPaths) == [unrelated.identifier]
    assert drive.GetTargetPositionAttr().Get() == 13
    assert doors.status()["closed"]


def test_no_metadata_is_a_noop_and_invalid_metadata_does_not_leave_a_layer():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/AnObjectNamedDoor")
    doors = Ros2Doors(stage, StageFrame(0.01, "Y"))
    doors.update([], 0, 0.1)
    assert doors.status()["count"] == 0 and not stage.GetSessionLayer().subLayerPaths
    doors.close()
    stage, frame, _, config, _ = _fixture()
    config.GetRelationship("automaticDoor:joint").SetTargets(["/MissingJoint"])
    with pytest.raises(ValueError, match="revolute joint"):
        Ros2Doors(stage, frame)
    assert not stage.GetSessionLayer().subLayerPaths


def test_unvalidated_metadata_axis_convention_is_rejected_explicitly():
    stage, _, _, _, _ = _fixture()
    UsdGeom.SetStageUpAxis(stage, "Y")
    with pytest.raises(ValueError, match="meter-based Z-up"):
        Ros2Doors(stage, StageFrame(1, "Y"))
    assert not stage.GetSessionLayer().subLayerPaths


@pytest.mark.parametrize("stage_unit, frame_unit", [(0.01, 0.01), (1.0, 0.01)])
def test_door_metadata_never_guesses_units_for_non_meter_or_mismatched_frames(stage_unit, frame_unit):
    stage, _, _, _, _ = _fixture()
    UsdGeom.SetStageMetersPerUnit(stage, stage_unit)
    with pytest.raises(ValueError, match="meter-based Z-up"):
        Ros2Doors(stage, StageFrame(frame_unit, "Z"), navigation_interlock=True)
    assert not stage.GetSessionLayer().subLayerPaths


def _interlock_fixture(monkeypatch, *, transformed=False):
    stage, frame, parent, config, drive = _fixture(transformed=transformed)
    joint = UsdPhysics.RevoluteJoint.Get(stage, config.GetRelationship("automaticDoor:joint").GetTargets()[0])
    body = stage.GetPrimAtPath(joint.GetBody1Rel().GetTargets()[0])
    transform = Gf.Transform(UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    rotation = transform.GetRotation().GetQuat()
    joint.CreateLocalPos0Attr(Gf.Vec3f(*transform.GetTranslation()))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0))
    joint.CreateLocalRot0Attr(Gf.Quatf(rotation.GetReal(), Gf.Vec3f(*rotation.GetImaginary())))
    joint.CreateLocalRot1Attr(Gf.Quatf(1))
    joint.CreateLowerLimitAttr(0)
    joint.CreateUpperLimitAttr(80)
    physical = {"angle": 0.0, "valid": True}

    def read(path):
        q = rotation * Gf.Rotation(Gf.Vec3d(0, 0, 1), physical["angle"]).GetQuat()
        return {
            "ret_val": physical["valid"],
            "position": tuple(transform.GetTranslation()),
            "rotation": (*q.GetImaginary(), q.GetReal()),
        }

    omni, physx = ModuleType("omni"), ModuleType("omni.physx")
    physx.get_physx_interface = lambda: SimpleNamespace(get_rigidbody_transformation=read)
    omni.physx = physx
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.physx", physx)
    return stage, frame, parent, config, drive, physical


def test_navigation_hold_wins_over_open_request_and_clear_sweep_allows_opening(monkeypatch):
    stage, frame, parent, config, drive, physical = _interlock_fixture(monkeypatch)
    drive.GetTargetPositionAttr().Set(80)
    physical["angle"] = 20
    doors = Ros2Doors(stage, frame, navigation_interlock=True)
    name = str(config.GetPath())
    # Omitted intents hold the true pose, not a stale 80-degree target.
    doors.update([], 0, 0.1)
    assert drive.GetTargetPositionAttr().Get() == pytest.approx(20)
    doors.set_navigation_intents([{"name": name, "hold": True, "request_open": True}])
    physical["angle"] = 22
    doors.update([], 0.1, 0.1)
    assert drive.GetTargetPositionAttr().Get() == pytest.approx(22)
    doors.set_navigation_intents([{"name": name, "hold": False, "request_open": True}])
    doors.update([_state(frame, parent, (0, -0.2, 0))], 0.2, 0.1)
    assert drive.GetTargetPositionAttr().Get() == pytest.approx(22)
    doors.update([], 0.3, 0.1)
    assert drive.GetTargetPositionAttr().Get() == pytest.approx(28)
    assert stage.GetPrimAtPath("/Campus/Panel").GetAttribute("physics:collisionEnabled").Get() is True
    doors.close()


def test_fully_open_uses_physical_angle_and_low_speed_and_same_tick_cannot_fake_settling(monkeypatch):
    stage, frame, _, _, drive, physical = _interlock_fixture(monkeypatch)
    drive.GetTargetPositionAttr().Set(80)
    doors = Ros2Doors(stage, frame, navigation_interlock=True)
    first = doors.observations(0)[0]
    assert first["valid"] and not first["fully_open"] and first["actual_angle_deg"] == pytest.approx(0)
    physical["angle"] = 80
    fast = doors.observations(0.1)[0]
    assert fast["angular_speed_degps"] == pytest.approx(800) and not fast["fully_open"]
    assert doors.observations(0.1)[0] == fast
    doors.update([], 0.1, 0.1)
    assert doors.observations(0.1)[0] == fast
    settled = doors.observations(0.2)[0]
    assert settled["fully_open"] and settled["angular_speed_degps"] == pytest.approx(0)
    physical["valid"] = False
    invalid = doors.observations(0.3)[0]
    assert not invalid["valid"] and not invalid["fully_open"]
    doors.close()


def test_fully_closed_requires_a_real_closed_angle_and_a_settled_physical_sample(monkeypatch):
    stage, frame, _, config, _, physical = _interlock_fixture(monkeypatch)
    doors = Ros2Doors(stage, frame, navigation_interlock=True)
    assert not doors.observations(0.0)[0]["fully_closed"]
    assert doors.observations(0.1)[0]["fully_closed"]
    physical["angle"] = 20
    assert not doors.observations(0.2)[0]["fully_closed"]
    physical["angle"] = 0
    assert not doors.observations(0.3)[0]["fully_closed"]
    assert doors.observations(0.4)[0]["fully_closed"]
    assert doors.status()["doors"][str(config.GetPath())]["fully_closed"]
    doors.close()


@pytest.mark.parametrize(
    "controller_value,config_value,exclusive", [(None, None, True), (False, None, False), (False, True, True)]
)
def test_declared_entry_exclusivity_keeps_legacy_default_and_per_door_override(
    monkeypatch, controller_value, config_value, exclusive
):
    stage, frame, parent, config, drive, _ = _interlock_fixture(monkeypatch)
    if controller_value is not None:
        config.GetParent().CreateAttribute("automaticDoor:exclusiveEntry", Sdf.ValueTypeNames.Bool).Set(
            controller_value
        )
    if config_value is not None:
        config.CreateAttribute("automaticDoor:exclusiveEntry", Sdf.ValueTypeNames.Bool).Set(config_value)
    source = stage.GetRootLayer().ExportToString()
    doors = Ros2Doors(stage, frame)
    assert doors.descriptors()[0]["exclusive_entry"] is exclusive
    entrant = _state(frame, parent, (0.0, 0.5, 0.0), name="entrant")
    resident = _state(frame, parent, (0.0, -1.3, 0.0), name="resident", yaw=0.0)
    doors.update([entrant, resident], 0.0, 0.1)
    assert drive.GetTargetPositionAttr().Get() == pytest.approx(0.0 if exclusive else 6.0)
    assert stage.GetRootLayer().ExportToString() == source
    doors.close()


@pytest.mark.parametrize("override", [False, True])
def test_door_front_policy_uses_controller_settings_and_per_door_overrides(override):
    stage, frame, _, config, _ = _fixture()
    doors = Ros2Doors(stage, frame, navigation_interlock=True)
    descriptor = doors.descriptors()[0]
    assert descriptor["require_front_approach"] is False
    assert descriptor["exit_approach_distance_m"] == descriptor["approach_distance_m"]
    doors.close()

    controller = config.GetParent()
    controller.CreateAttribute("automaticDoor:requireFrontApproach", Sdf.ValueTypeNames.Bool).Set(True)
    controller.CreateAttribute("automaticDoor:exitApproachDistance", Sdf.ValueTypeNames.Double).Set(0.72)
    if override:
        config.CreateAttribute("automaticDoor:requireFrontApproach", Sdf.ValueTypeNames.Bool).Set(False)
        config.CreateAttribute("automaticDoor:exitApproachDistance", Sdf.ValueTypeNames.Double).Set(0.69)
    doors = Ros2Doors(stage, frame, navigation_interlock=True)
    descriptor = doors.descriptors()[0]
    assert descriptor["require_front_approach"] is not override
    assert descriptor["approach_distance_m"] == pytest.approx(0.6)
    assert descriptor["exit_approach_distance_m"] == pytest.approx(0.69 if override else 0.72)
    doors.close()


@pytest.mark.parametrize("transformed", [False, True])
def test_closed_guard_footprint_keeps_settled_angle_tolerance_without_filling_the_whole_swing(monkeypatch, transformed):
    stage, frame, _, _, _, _ = _interlock_fixture(monkeypatch, transformed=transformed)
    doors = Ros2Doors(stage, frame, navigation_interlock=True)
    descriptor = doors.descriptors()[0]

    def area(polygon):
        return abs(sum(a[0] * b[1] - a[1] * b[0] for a, b in zip(polygon, (*polygon[1:], polygon[0])))) / 2

    assert area(descriptor["closed_footprint_xy_m"]) < area(descriptor["swept_footprint_xy_m"]) / 3
    body = stage.GetPrimAtPath(descriptor["body_prim_path"])
    bound = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"], False, True).ComputeWorldBound(body)
    extent = bound.GetRange()
    authored = Gf.Transform(UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    joint = UsdPhysics.RevoluteJoint.Get(stage, descriptor["joint_prim_path"])
    pivot = Gf.Vec3d(*joint.GetLocalPos0Attr().Get())
    for angle in (-1.0, -0.5, 0.0, 0.5, 1.0):
        rotation = Gf.Rotation(
            Gf.Rotation(joint.GetLocalRot0Attr().Get()).GetQuat()
            * Gf.Rotation(Gf.Vec3d(0, 0, 1), angle).GetQuat()
            * Gf.Rotation(joint.GetLocalRot1Attr().Get()).GetInverse().GetQuat()
        )
        for x in (extent.GetMin()[0], extent.GetMax()[0]):
            for y in (extent.GetMin()[1], extent.GetMax()[1]):
                for z in (extent.GetMin()[2], extent.GetMax()[2]):
                    corner = bound.GetMatrix().Transform(Gf.Vec3d(x, y, z))
                    local = authored.GetRotation().GetInverse().TransformDir(corner - authored.GetTranslation())
                    point = frame.stage_to_ros(pivot + rotation.TransformDir(local))
                    polygon = descriptor["closed_footprint_xy_m"]
                    assert (
                        min(
                            (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
                            for a, b in zip(polygon, (*polygon[1:], polygon[0]))
                        )
                        >= -1e-10
                    )
    closed = doors.descriptors()[0]["closed_footprint_xy_m"]
    closed.clear()
    assert doors.descriptors()[0]["closed_footprint_xy_m"]
    doors.close()


def test_interlocked_open_door_cannot_close_on_a_crossing_body(monkeypatch):
    stage, frame, parent, config, drive, physical = _interlock_fixture(monkeypatch)
    doors = Ros2Doors(stage, frame, navigation_interlock=True)
    physical["angle"] = 80
    doors.observations(0)
    assert doors.observations(0.1)[0]["fully_open"]
    doors.set_navigation_intents([{"name": str(config.GetPath()), "hold": False, "request_open": False}])
    doors.update([_state(frame, parent, (0, -0.2, 0))], 0.1, 0.1)
    assert drive.GetTargetPositionAttr().Get() == pytest.approx(80)
    doors.update([], 0.3, 0.1)
    assert drive.GetTargetPositionAttr().Get() == pytest.approx(80)
    doors.update([], 0.7, 0.1)
    assert drive.GetTargetPositionAttr().Get() == pytest.approx(74)
    doors.close()


@pytest.mark.parametrize("transformed", [False, True])
def test_continuous_door_envelope_contains_between_sample_rotations_and_ros_room_polygon(monkeypatch, transformed):
    stage, frame, parent, config, _, _ = _interlock_fixture(monkeypatch, transformed=transformed)
    if transformed:
        config.CreateAttribute("automaticDoor:approachDistance", Sdf.ValueTypeNames.Double).Set(0.43)
    doors = Ros2Doors(stage, frame, navigation_interlock=True)
    spec = doors.descriptors()[0]
    assert spec["enabled"] and spec["name"] == str(config.GetPath())
    assert spec["approach_distance_m"] == pytest.approx(0.43 if transformed else 0.6)
    a, b = spec["doorway_segment_xy_m"]
    assert math.dist(a, b) == pytest.approx(0.64)
    center = frame.stage_to_ros(parent.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).Transform(Gf.Vec3d(0)))
    assert [(x + y) / 2 for x, y in zip(a, b)] == pytest.approx(center[:2])
    body = stage.GetPrimAtPath(spec["body_prim_path"])
    bounds = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"], False, True
    ).ComputeWorldBound(body)
    extent = bounds.GetRange()
    corners = [
        bounds.GetMatrix().Transform(Gf.Vec3d(x, y, z))
        for x in (extent.GetMin()[0], extent.GetMax()[0])
        for y in (extent.GetMin()[1], extent.GetMax()[1])
        for z in (extent.GetMin()[2], extent.GetMax()[2])
    ]
    authored = Gf.Transform(UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    joint = UsdPhysics.RevoluteJoint.Get(stage, spec["joint_prim_path"])
    pivot = Gf.Vec3d(*joint.GetLocalPos0Attr().Get())
    joint0 = Gf.Rotation(joint.GetLocalRot0Attr().Get())
    joint1_inverse = Gf.Rotation(joint.GetLocalRot1Attr().Get()).GetInverse()
    polygon = spec["swept_footprint_xy_m"]
    for i in range(403):
        angle = -1 + 82 * i / 402
        rotation = Gf.Rotation(
            joint0.GetQuat() * Gf.Rotation(Gf.Vec3d(0, 0, 1), angle).GetQuat() * joint1_inverse.GetQuat()
        )
        for corner in corners:
            physical_local = authored.GetRotation().GetInverse().TransformDir(corner - authored.GetTranslation())
            point = frame.stage_to_ros(pivot + rotation.TransformDir(physical_local))
            crosses = [
                (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
                for a, b in zip(polygon, (*polygon[1:], polygon[0]))
            ]
            assert min(crosses) >= -1e-10
    room = spec["interior_polygon_xy_m"]
    area = abs(sum(a[0] * b[1] - a[1] * b[0] for a, b in zip(room, (*room[1:], room[0])))) / 2
    assert area == pytest.approx(2.6 if transformed else 1.3)
    doors.descriptors()[0]["swept_footprint_xy_m"].clear()
    assert doors.descriptors()[0]["swept_footprint_xy_m"]
    doors.close()


def test_navigation_interlock_requires_explicit_room_metadata_but_default_api_does_not():
    stage, frame, _, config, _ = _fixture()
    config.RemoveProperty("automaticDoor:interiorMin")
    config.RemoveProperty("automaticDoor:interiorMax")
    with pytest.raises(ValueError, match="interiorMin/interiorMax"):
        Ros2Doors(stage, frame, navigation_interlock=True)
    assert not stage.GetSessionLayer().subLayerPaths
    doors = Ros2Doors(stage, frame)
    doors.close()
