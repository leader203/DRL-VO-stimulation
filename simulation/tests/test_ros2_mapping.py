# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check map contracts with real USD geometry and controlled PhysX boundary queries."""

import math
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import yaml

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdUtils

from pedstrain_simulation.simulation.ros2_navigation.frames import StageFrame
from pedstrain_simulation.simulation.ros2_navigation.mapping import (
    build_occupancy_grid,
    query_dynamic_obstacles,
    save_occupancy_grid,
)


@pytest.mark.parametrize("axis", ["Y", "Z"])
@pytest.mark.parametrize("units", [1.0, 0.01])
def test_frames_use_ros_origin_heading_and_the_full_right_handed_basis(axis, units):
    frame = StageFrame(units, axis, (3.0, -2.0, 1.0), math.pi / 2)
    world = (4.0, 0.0, 1.5)
    stage = tuple(value / units for value in ((4.0, 1.5, 0.0) if axis == "Y" else world))
    assert frame.stage_to_ros(stage) == pytest.approx((2.0, -1.0, 0.5))
    assert frame.world_to_ros(world) == pytest.approx((2.0, -1.0, 0.5))
    assert frame.ros_to_stage((2.0, -1.0, 0.5)) == pytest.approx(stage)
    assert frame.ros_to_world((2.0, -1.0, 0.5)) == pytest.approx(world)
    assert frame.stage_vector_to_ros((1 / units, 0, 0)) == pytest.approx((0, -1, 0))
    x, y, z, w = frame.ros_yaw_to_stage_quaternion(-math.pi / 2)
    rotation = Gf.Rotation(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    assert tuple(rotation.TransformDir(Gf.Vec3d(1, 0, 0))) == pytest.approx((1, 0, 0))
    assert tuple(rotation.TransformDir(Gf.Vec3d(0, 0, 1))) == pytest.approx((0, 1, 0) if axis == "Y" else (0, 0, 1))


@pytest.mark.parametrize("axis", ["Y", "Z"])
@pytest.mark.parametrize("units", [1.0, 0.01])
def test_stage_world_points_have_an_explicit_map_independent_metric_conversion(axis, units):
    frame = StageFrame(units, axis, (3.0, -2.0, 1.0), 0.43)
    canonical = (1.25, -2.5, 0.75)
    stage = tuple(value / units for value in ((1.25, 0.75, 2.5) if axis == "Y" else canonical))
    assert frame.stage_to_world(stage) == pytest.approx(canonical)
    assert frame.world_to_stage(canonical) == pytest.approx(stage)
    assert frame.world_to_ros(frame.stage_to_world(stage)) == pytest.approx(frame.stage_to_ros(stage))
    assert frame.world_to_stage(frame.ros_to_world(frame.stage_to_ros(stage))) == pytest.approx(stage)


@pytest.mark.parametrize(
    "kwargs", [{"meters_per_unit": 0}, {"up_axis": "X"}, {"yaw_rad": math.inf}, {"origin_m": (0, math.nan, 0)}]
)
def test_frames_reject_invalid_coordinate_contracts(kwargs):
    values = {"meters_per_unit": 1.0, "up_axis": "Z", **kwargs}
    with pytest.raises(ValueError):
        StageFrame(**values)


def _stage(frame):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, frame.meters_per_unit)
    UsdGeom.SetStageUpAxis(stage, frame.up_axis)
    floor = UsdGeom.Cube.Define(stage, "/Scene/Floor")
    floor.CreateSizeAttr(2.0 / frame.meters_per_unit)
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    for path, dynamic in (("/Scene/Wall", False), ("/Scene/Crowd/Body", True)):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(0.1 / frame.meters_per_unit)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        if dynamic:
            UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim()).CreateKinematicEnabledAttr(True)
    identifier = UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
    return stage, identifier


def _collision_mesh(stage, path, low, high, presentation):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([(x, y, z) for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr([0, 2, 6, 4, 1, 5, 7, 3, 0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6])
    mesh.CreateExtentAttr([low, high])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
    if presentation in ("guide", "both"):
        mesh.CreatePurposeAttr("guide")
    if presentation in ("invisible", "both"):
        mesh.CreateVisibilityAttr("invisible")
    return mesh


class _Queries:
    def __init__(self, frame):
        self.frame = frame
        self.overlaps = []
        self.hole = lambda x, y: x > 0 and y < 0
        self.empty = False
        self.extra_hit = None
        self.proto_index = 0xFFFFFFFF
        self.wall_path = "/Scene/Wall"
        self.body_hit = lambda x, y: True

    def overlap_box(self, half, center, quaternion, callback, any_hit):
        assert not any_hit
        self.overlaps.append((half, center, quaternion))
        x, y, _ = self.frame.stage_to_ros(center)
        # A dynamic collider preceding a wall must not stop traversal.
        if self.body_hit(x, y):
            callback(SimpleNamespace(collision="/Scene/Crowd/Body", protoIndex=self.proto_index))
        if self.extra_hit is not None:
            callback(SimpleNamespace(collision=self.extra_hit, protoIndex=self.proto_index))
        if x < 0 and y > 0:
            callback(SimpleNamespace(collision=self.wall_path, protoIndex=self.proto_index))
        return 1

    def raycast_all(self, origin, direction, distance, callback, both_sides):
        x, y, _ = self.frame.stage_to_ros(origin)
        assert math.sqrt(sum(value * value for value in direction)) == pytest.approx(1.0)
        if not self.empty and not self.hole(x, y):
            normal = (0, 1, 0) if self.frame.up_axis == "Y" else (0, 0, 1)
            callback(
                SimpleNamespace(collision="/Scene/Floor", position=self.frame.ros_to_stage((x, y, 0)), normal=normal)
            )
            return True
        return False


def _physics(monkeypatch, queries, identifier, *, attached=None):
    omni = ModuleType("omni")
    physx = ModuleType("omni.physx")
    physx.get_physx_scene_query_interface = lambda: queries
    physx.get_physx_simulation_interface = lambda: SimpleNamespace(
        get_attached_stage=lambda: identifier if attached is None else attached
    )
    physx.get_physx_interface = lambda: SimpleNamespace(
        get_rigidbody_transformation=lambda path: {
            "ret_val": True,
            "position": (9.0, 10.0, 1.0),
            "rotation": (0.0, 0.0, 0.0, 1.0),
        }
    )
    omni.physx = physx
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.physx", physx)


@pytest.mark.parametrize("axis,units", [("Z", 1.0), ("Z", 0.01), ("Y", 0.01)])
def test_mapping_requires_floor_support_and_continues_past_dynamic_hits(monkeypatch, axis, units):
    frame = StageFrame(units, axis)
    stage, identifier = _stage(frame)
    queries = _Queries(frame)
    _physics(monkeypatch, queries, identifier)
    data = build_occupancy_grid(stage, {"resolution_m": 1.0, "bounds_m": [-1, -1, 1, 1]}, frame)
    np.testing.assert_array_equal(data["occupancy"], [[0, -1], [100, 0]])
    assert data["occupancy"].dtype == np.int8
    assert data["origin_xy_m"] == (-1.0, -1.0)
    half, center, quaternion = queries.overlaps[0]
    assert half == pytest.approx((0.5 / units, 0.5 / units, 0.835 / units))
    assert frame.stage_to_ros(center) == pytest.approx((-0.5, -0.5, 0.915))
    assert quaternion == pytest.approx(frame.ros_yaw_to_stage_quaternion(0))


def test_missing_corner_support_stays_unknown_even_when_the_cell_center_is_supported(monkeypatch):
    frame = StageFrame(1.0, "Z")
    stage, identifier = _stage(frame)
    queries = _Queries(frame)
    queries.hole = lambda x, y: x > 0.8 and y > 0.8
    _physics(monkeypatch, queries, identifier)
    data = build_occupancy_grid(stage, {"resolution_m": 1.0, "bounds_m": [-1, -1, 1, 1]}, frame)
    assert data["occupancy"][1, 1] == -1


@pytest.mark.parametrize("kind", ["cube", "reference_instance"])
@pytest.mark.parametrize("proto_index", [0, 1])
def test_non_pointinstancer_collisions_keep_their_classification_regardless_of_query_proto_index(
    monkeypatch, kind, proto_index
):
    frame = StageFrame(1.0, "Z")
    stage, identifier = _stage(frame)
    queries = _Queries(frame)
    queries.proto_index = proto_index
    if kind == "reference_instance":
        source = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(source, "/Model")
        shape = UsdGeom.Cube.Define(source, "/Model/Collider")
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        instance = UsdGeom.Xform.Define(stage, "/OrdinaryInstance").GetPrim()
        instance.GetReferences().AddReference(source.GetRootLayer().identifier, "/Model")
        instance.SetInstanceable(True)
        queries.wall_path = "/OrdinaryInstance/Collider"
        assert stage.GetPrimAtPath(queries.wall_path).IsInstanceProxy()
    _physics(monkeypatch, queries, identifier)
    data = build_occupancy_grid(stage, {"resolution_m": 1.0, "bounds_m": [-1, -1, 1, 1]}, frame)
    np.testing.assert_array_equal(data["occupancy"], [[0, -1], [100, 0]])


@pytest.mark.parametrize("prototype_path", ["/Points/Prototype", "/SeparatePrototype"])
def test_real_pointinstancer_colliders_are_rejected_from_usd_prototype_relationships(monkeypatch, prototype_path):
    frame = StageFrame(1.0, "Z")
    stage, identifier = _stage(frame)
    instancer = UsdGeom.PointInstancer.Define(stage, "/Points")
    shape = UsdGeom.Cube.Define(stage, prototype_path)
    UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
    instancer.CreatePrototypesRel().SetTargets([shape.GetPath()])
    instancer.CreateProtoIndicesAttr([0])
    instancer.CreatePositionsAttr([(0, 0, 0)])
    _physics(monkeypatch, _Queries(frame), identifier)
    with pytest.raises(RuntimeError, match="PointInstancer collision"):
        build_occupancy_grid(stage, {"resolution_m": 1.0, "bounds_m": [-1, -1, 1, 1]}, frame)


def _fixed_joint(stage, path, body, anchor=None):
    joint = UsdPhysics.FixedJoint.Define(stage, path)
    joint.CreateBody1Rel().SetTargets([body])
    if anchor is not None:
        joint.CreateBody0Rel().SetTargets([anchor])
    return joint


@pytest.mark.parametrize("anchor", [None, "/StaticAnchor"])
def test_world_anchored_unbreakable_fixed_bodies_are_baked_as_actual_static_colliders(monkeypatch, anchor):
    frame = StageFrame(1.0, "Z")
    stage, identifier = _stage(frame)
    if anchor is not None:
        UsdGeom.Xform.Define(stage, anchor)
    _fixed_joint(stage, "/Constraints/Fixed", "/Scene/Crowd/Body", anchor)
    queries = _Queries(frame)
    queries.body_hit = lambda x, y: x > 0 and y > 0
    _physics(monkeypatch, queries, identifier)
    data = build_occupancy_grid(stage, {"resolution_m": 1.0, "bounds_m": [-1, -1, 1, 1]}, frame)
    np.testing.assert_array_equal(data["occupancy"], [[0, -1], [100, 100]])
    assert query_dynamic_obstacles(stage, frame, ()) == []


def test_transitive_fixed_chain_proves_only_its_rigid_body_component_static(monkeypatch):
    frame = StageFrame(1.0, "Z")
    stage, identifier = _stage(frame)
    base = UsdGeom.Xform.Define(stage, "/AnchoredBase").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(base)
    _fixed_joint(stage, "/Constraints/Anchor", "/AnchoredBase")
    _fixed_joint(stage, "/Constraints/Link", "/Scene/Crowd/Body", "/AnchoredBase")
    queries = _Queries(frame)
    queries.body_hit = lambda x, y: x > 0 and y > 0
    _physics(monkeypatch, queries, identifier)
    data = build_occupancy_grid(stage, {"resolution_m": 1.0, "bounds_m": [-1, -1, 1, 1]}, frame)
    assert data["occupancy"][1, 1] == 100
    assert query_dynamic_obstacles(stage, frame, ()) == []


@pytest.mark.parametrize("condition", ["disabled", "break_force", "break_torque"])
def test_disabled_or_breakable_fixed_constraints_do_not_hide_movable_obstacles(monkeypatch, condition):
    frame = StageFrame(1.0, "Z")
    stage, identifier = _stage(frame)
    joint = _fixed_joint(stage, "/Constraints/Fixed", "/Scene/Crowd/Body")
    if condition == "disabled":
        joint.CreateJointEnabledAttr(False)
    elif condition == "break_force":
        joint.CreateBreakForceAttr(1.0)
    else:
        joint.CreateBreakTorqueAttr(1.0)
    queries = _Queries(frame)
    queries.body_hit = lambda x, y: x > 0 and y > 0
    _physics(monkeypatch, queries, identifier)
    data = build_occupancy_grid(stage, {"resolution_m": 1.0, "bounds_m": [-1, -1, 1, 1]}, frame)
    assert data["occupancy"][1, 1] == 0
    assert [disc["prim_path"] for disc in query_dynamic_obstacles(stage, frame, ())] == ["/Scene/Crowd/Body"]


def test_explicit_actor_and_robot_roots_are_excluded_even_without_rigid_body_api(monkeypatch):
    frame = StageFrame(1.0, "Z")
    stage, identifier = _stage(frame)
    body = UsdGeom.Cube.Define(stage, "/Scene/Robot/Body")
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    queries = _Queries(frame)
    queries.extra_hit = "/Scene/Robot/Body"
    _physics(monkeypatch, queries, identifier)
    data = build_occupancy_grid(stage, {"resolution_m": 1.0, "bounds_m": [-1, -1, 1, 1]}, frame, ("/Scene/Robot",))
    assert data["occupancy"][0, 0] == 0


def _composed_robot(monkeypatch, tmp_path, *, world_fixed):
    from pedstrain_simulation.simulation.configuration.models import RobotConfig, SceneConfig
    from pedstrain_simulation.simulation.ros2_navigation.runtime import _compose_world
    from pedstrain_simulation.simulation.scene.physics import validate_robots

    frame = StageFrame(1.0, "Z")
    base_path, asset_path = tmp_path / "base.usda", tmp_path / "robot.usda"
    base = Usd.Stage.CreateNew(str(base_path))
    asset = Usd.Stage.CreateNew(str(asset_path))
    for stage in (base, asset):
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, "Z")
    floor = UsdGeom.Cube.Define(base, "/Scene/Floor")
    floor.CreateSizeAttr(2.0)
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    asset.SetDefaultPrim(UsdGeom.Xform.Define(asset, "/Robot").GetPrim())
    body = UsdGeom.Xform.Define(asset, "/Robot/Base")
    body.AddTranslateOp().Set((0.0, 0.0, 0.2))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    collider = UsdGeom.Cube.Define(asset, "/Robot/Base/Collision")
    collider.CreateSizeAttr(0.4)
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    if world_fixed:
        _fixed_joint(asset, "/Robot/WorldAnchor", "/Robot/Base")
    base.GetRootLayer().Save()
    asset.GetRootLayer().Save()
    original = asset_path.read_bytes()
    config = SimpleNamespace(
        scene=SceneConfig(asset_path=base_path),
        robots=(RobotConfig(asset_path=asset_path, prim_path="/Scene/Robot"),),
        pedestrians=SimpleNamespace(enabled=False),
    )
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    world, _ = _compose_world(config, runtime)
    stage = Usd.Stage.Open(str(world))
    identifier = UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
    queries = _Queries(frame)
    queries.hole = lambda x, y: False

    def overlap(half, center, quaternion, callback, any_hit):
        if abs(center[0]) <= 0.5 and abs(center[1]) <= 0.5:
            callback(SimpleNamespace(collision="/Scene/Robot/Base/Collision", protoIndex=0xFFFFFFFF))
        return 1

    queries.overlap_box = overlap
    _physics(monkeypatch, queries, identifier)
    usd = ModuleType("omni.usd")
    usd.get_context = lambda: SimpleNamespace(get_stage=lambda: stage)
    monkeypatch.setitem(sys.modules, "omni.usd", usd)
    monkeypatch.setattr(sys.modules["omni"], "usd", usd, raising=False)
    validate_robots(config)
    assert not stage.GetPrimAtPath("/Scene/Robot").HasAPI(UsdPhysics.RigidBodyAPI)
    assert stage.GetPrimAtPath("/Scene/Robot/Base").HasAPI(UsdPhysics.RigidBodyAPI)
    assert asset_path.read_bytes() == original
    return stage, frame


def test_a_composed_world_fixed_robot_is_baked_as_a_static_obstacle_by_default(monkeypatch, tmp_path):
    stage, frame = _composed_robot(monkeypatch, tmp_path, world_fixed=True)
    data = build_occupancy_grid(stage, {"resolution_m": 0.5, "bounds_m": [-1, -1, 1, 1]}, frame)
    expected = np.zeros((4, 4), dtype=np.int8)
    expected[1:3, 1:3] = 100
    np.testing.assert_array_equal(data["occupancy"], expected)
    assert query_dynamic_obstacles(stage, frame, ()) == []


@pytest.mark.parametrize("world_fixed, dynamic_root", [(False, "/Scene/Robot"), (True, "/Scene/Robot/Base/Collision")])
def test_explicit_robot_subtrees_read_the_actual_child_rigid_pose(monkeypatch, tmp_path, world_fixed, dynamic_root):
    from pedstrain_simulation.simulation.ros2_navigation.mapping import DynamicObstacleObserver

    stage, frame = _composed_robot(monkeypatch, tmp_path, world_fixed=world_fixed)
    live = {"x": 1.0}
    calls = []

    def pose(path):
        calls.append(path)
        return {"ret_val": True, "position": (live["x"], 0.0, 0.2), "rotation": (0.0, 0.0, 0.0, 1.0)}

    monkeypatch.setattr(
        sys.modules["omni.physx"], "get_physx_interface", lambda: SimpleNamespace(get_rigidbody_transformation=pose)
    )
    observer = DynamicObstacleObserver(stage, frame, (), dynamic_roots=(dynamic_root,))
    try:
        for x in (1.0, 2.0):
            live["x"] = x
            obstacles = observer.sample()
            assert len(obstacles) == 1
            assert obstacles[0]["position"] == pytest.approx((x, 0.0, 0.2))
            assert obstacles[0]["prim_path"] == "/Scene/Robot/Base"
            assert calls[-1] == "/Scene/Robot/Base"
    finally:
        observer.close()


def test_an_explicit_script_root_without_a_rigid_body_follows_its_usd_transform(monkeypatch):
    from pedstrain_simulation.simulation.ros2_navigation.mapping import DynamicObstacleObserver

    frame = StageFrame(1.0, "Z")
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    root = UsdGeom.Xform.Define(stage, "/ScriptRoot")
    translate = root.AddTranslateOp()
    collider = UsdGeom.Cube.Define(stage, "/ScriptRoot/Collision")
    collider.CreateSizeAttr(0.4)
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    identifier = UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
    _physics(monkeypatch, object(), identifier)

    def unexpected_physics_pose(path):
        pytest.fail(f"A script-owned nonrigid root must not query a rigid pose: {path}")

    monkeypatch.setattr(
        sys.modules["omni.physx"],
        "get_physx_interface",
        lambda: SimpleNamespace(get_rigidbody_transformation=unexpected_physics_pose),
    )
    observer = DynamicObstacleObserver(stage, frame, (), dynamic_roots=("/ScriptRoot",))
    try:
        for x in (1.0, 2.0):
            translate.Set((x, 0.0, 0.2))
            obstacles = observer.sample()
            assert len(obstacles) == 1 and obstacles[0]["prim_path"] == "/ScriptRoot"
            assert obstacles[0]["position"] == pytest.approx((x, 0.0, 0.2))
    finally:
        observer.close()


def test_paused_stageupdate_queries_do_not_require_a_manual_simulation_attachment(monkeypatch):
    frame = StageFrame(1.0, "Z")
    stage, identifier = _stage(frame)
    _physics(monkeypatch, _Queries(frame), identifier, attached=0)
    usd, timeline = ModuleType("omni.usd"), ModuleType("omni.timeline")
    usd.get_context = lambda: SimpleNamespace(get_stage_id=lambda: identifier, get_stage=lambda: stage)
    timeline.get_timeline_interface = lambda: SimpleNamespace(is_stopped=lambda: False, get_current_time=lambda: 0.5)
    for name, module in (("omni.usd", usd), ("omni.timeline", timeline)):
        monkeypatch.setitem(sys.modules, name, module)
        setattr(sys.modules["omni"], name.rsplit(".", 1)[1], module)
    data = build_occupancy_grid(stage, {"resolution_m": 1.0, "bounds_m": [-1, -1, 1, 1]}, frame)
    assert data["diagnostics"]["supported_cells"] > 0


@pytest.mark.parametrize("problem", ["detached", "no_support", "unknown_shape", "budget"])
def test_unready_missing_or_unclassified_physics_never_becomes_a_free_map(monkeypatch, problem):
    frame = StageFrame(1.0, "Z")
    stage, identifier = _stage(frame)
    queries = _Queries(frame)
    queries.empty = problem == "no_support"
    queries.extra_hit = "/Unknown/PhysicsShape" if problem == "unknown_shape" else None
    _physics(monkeypatch, queries, identifier, attached=0 if problem == "detached" else None)
    settings = {"resolution_m": 1.0, "bounds_m": [-1, -1, 1, 1], "max_cells": 1 if problem == "budget" else 10}
    with pytest.raises(
        (RuntimeError, ValueError),
        match={
            "detached": "not ready",
            "no_support": "No supported",
            "unknown_shape": "unclassified",
            "budget": "max_cells",
        }[problem],
    ):
        build_occupancy_grid(stage, settings, frame)


def test_auto_bounds_include_plan_points_but_ignore_visual_only_geometry(monkeypatch):
    frame = StageFrame(1.0, "Z", origin_m=(10.0, 0.0, 0.0))
    stage, identifier = _stage(frame)
    UsdGeom.Cube.Define(stage, "/SkyVisual").CreateSizeAttr(10000.0)
    queries = _Queries(frame)
    queries.hole = lambda x, y: False
    _physics(monkeypatch, queries, identifier)
    data = build_occupancy_grid(stage, {"resolution_m": 1.0, "plan_points_m": [(14.0, 0.0, 0.0)]}, frame)
    xmin, ymin = data["origin_xy_m"]
    assert xmin <= -11.0 and xmin + data["occupancy"].shape[1] >= 4.0
    assert data["occupancy"].size < 100


@pytest.mark.parametrize("presentation", ["guide", "invisible", "both"])
def test_hidden_guide_collision_meshes_contribute_finite_auto_bounds_and_occupied_cells(monkeypatch, presentation):
    frame = StageFrame(1.0, "Z")
    stage, identifier = _stage(frame)
    floor = UsdGeom.Cube(stage.GetPrimAtPath("/Scene/Floor"))
    floor.CreateSizeAttr(1)
    floor.AddTranslateOp().Set((2, 0, -0.05))
    floor.AddScaleOp().Set((6, 2, 0.1))
    mesh = _collision_mesh(stage, "/Scene/PhysicalOnly", (3, 0, 0), (4, 1, 1.8), presentation)

    class Queries(_Queries):
        def overlap_box(self, half, center, quaternion, callback, any_hit):
            x, y, _ = frame.stage_to_ros(center)
            if 3 <= x <= 4 and 0 <= y <= 1:
                callback(SimpleNamespace(collision=str(mesh.GetPath())))
            return 1

    queries = Queries(frame)
    queries.hole = lambda x, y: not (-1 <= x <= 5 and -1 <= y <= 1)
    _physics(monkeypatch, queries, identifier)
    data = build_occupancy_grid(stage, {"resolution_m": 1.0}, frame)
    xmin, ymin = data["origin_xy_m"]
    row, column = math.floor(0.5 - ymin), math.floor(3.5 - xmin)
    assert data["occupancy"][row, column] == 100
    row, column = math.floor(-0.5 - ymin), math.floor(1.5 - xmin)
    assert data["occupancy"][row, column] == 0
    assert mesh.GetVisibilityAttr().Get() == ("invisible" if presentation in ("invisible", "both") else "inherited")


@pytest.mark.parametrize("presentation", ["guide", "invisible", "both"])
def test_hidden_guide_collision_meshes_are_included_in_live_dynamic_footprints(monkeypatch, presentation):
    frame = StageFrame(1.0, "Z")
    stage, identifier = _stage(frame)
    _collision_mesh(stage, "/Scene/Crowd/Body/PhysicalOnly", (2, -1, 0), (3, 1, 1), presentation)
    _physics(monkeypatch, _Queries(frame), identifier)
    disc = query_dynamic_obstacles(stage, frame, ())[0]
    assert max(point[0] for point in disc["footprint_xy_m"]) == pytest.approx(12)
    assert disc["z_max_m"] == pytest.approx(2)


def test_export_preserves_ros_bottom_left_origin_and_flips_image_rows(tmp_path):
    data = {
        "occupancy": np.array([[0, -1], [100, 0]], dtype=np.int8),
        "resolution_m": 0.1,
        "origin_xy_m": (-2.3, -4.5),
        "floor_height_m": 1.2,
    }
    paths = save_occupancy_grid(data, tmp_path / "map")
    assert paths["pgm"].read_bytes() == b"P5\n2 2\n255\n" + bytes([0, 254, 254, 205])
    metadata = yaml.safe_load(paths["yaml"].read_text())
    assert metadata["origin"] == [-2.3, -4.5, 0.0] and metadata["resolution"] == 0.1
    with np.load(paths["npz"], allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["occupancy"], data["occupancy"])
        assert archive["floor_height_m"] == 1.2
    before = paths["pgm"].read_bytes()
    with pytest.raises(FileExistsError):
        save_occupancy_grid(data, tmp_path / "map")
    assert paths["pgm"].read_bytes() == before


def test_dynamic_discs_use_live_physics_pose_and_keep_parent_scale(monkeypatch):
    frame = StageFrame(1.0, "Z")
    stage, identifier = _stage(frame)
    parent = UsdGeom.Xform.Define(stage, "/Scene/Crowd")
    parent.AddTranslateOp().Set((5, 2, 0))
    parent.AddRotateZOp().Set(90)
    parent.AddScaleOp().Set((2, 2, 2))
    cube = UsdGeom.Cube(stage.GetPrimAtPath("/Scene/Crowd/Body"))
    cube.CreateSizeAttr(1)
    cube.AddTranslateOp().Set((1, 0, 0))
    _physics(monkeypatch, _Queries(frame), identifier)
    discs = query_dynamic_obstacles(stage, frame, ())
    assert len(discs) == 1
    assert discs[0]["position"] == pytest.approx((9, 10, 1))
    assert discs[0]["radius_m"] == pytest.approx(math.sqrt(2))
    np.testing.assert_allclose(discs[0]["footprint_xy_m"], [[8, 9], [10, 9], [10, 11], [8, 11]])
    assert discs[0]["z_min_m"] == pytest.approx(0)
    assert discs[0]["z_max_m"] == pytest.approx(2)
    assert not discs[0]["velocity_valid"]
    assert query_dynamic_obstacles(stage, frame, ("/Scene/Crowd",)) == []


def _thin_rotated_door(monkeypatch, frame, angle=math.pi / 4):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, frame.meters_per_unit)
    UsdGeom.SetStageUpAxis(stage, frame.up_axis)
    root = UsdGeom.Xform.Define(stage, "/Door")
    root.AddTranslateOp().Set(frame.ros_to_stage((3.0, 4.0, 0.9)))
    q = frame.ros_yaw_to_stage_quaternion(angle)
    root.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(q[3], *q[:3]))
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateKinematicEnabledAttr(True)
    panel = UsdGeom.Cube.Define(stage, "/Door/Panel")
    panel.CreateSizeAttr(1.0)
    panel.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(
        tuple(value / frame.meters_per_unit for value in (2.0, 0.1, 1.8))
    )
    UsdPhysics.CollisionAPI.Apply(panel.GetPrim())
    identifier = UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
    _physics(monkeypatch, object(), identifier)
    sys.modules["omni.physx"].get_physx_interface = lambda: SimpleNamespace(
        get_rigidbody_transformation=lambda path: {
            "ret_val": True,
            "position": frame.ros_to_stage((3.0, 4.0, 0.9)),
            "rotation": q,
        }
    )
    return stage, root, panel


def _polygon_area(points):
    return abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, (*points[1:], points[0])))) / 2


@pytest.mark.parametrize("up,units", [("Z", 1.0), ("Y", 0.01)])
def test_authored_live_door_rotation_keeps_projected_area_and_direction(monkeypatch, up, units):
    frame = StageFrame(units, up)
    stage, _, _ = _thin_rotated_door(monkeypatch, frame)
    obstacle = query_dynamic_obstacles(stage, frame, ())[0]
    polygon = np.asarray(obstacle["footprint_xy_m"])
    assert _polygon_area(polygon.tolist()) == pytest.approx(2.0 * 0.1, abs=1e-10)
    edges = np.roll(polygon, -1, axis=0) - polygon
    edge = max(edges, key=np.linalg.norm)
    direction = edge / np.linalg.norm(edge)
    assert abs(float(np.dot(direction, [math.sqrt(0.5), math.sqrt(0.5)]))) == pytest.approx(1.0)
    assert obstacle["radius_m"] == pytest.approx(math.hypot(1.0, 0.05))


def test_empty_world_aabb_corner_does_not_block_a_disk_clear_of_the_rotated_door(monkeypatch):
    from pedstrain_simulation.simulation.ros2_navigation.planner import segment_polygon_clear

    frame = StageFrame(1.0, "Z")
    stage, _, _ = _thin_rotated_door(monkeypatch, frame)
    obstacle = query_dynamic_obstacles(stage, frame, ())[0]
    point = (3.65, 3.35)
    assert segment_polygon_clear(point, point, obstacle["footprint_xy_m"], 0.25 + 0.05)


@pytest.mark.parametrize(
    "start,end,clearance,expected",
    [
        ((-2.0, 0.0), (2.0, 0.0), 0.0, False),
        ((-2.0, 0.6), (2.0, 0.6), 0.5, False),
        ((-2.0, 0.600001), (2.0, 0.600001), 0.5, True),
        ((-2.0, 0.1000005), (2.0, 0.1000005), 0.0, False),
        ((-2.0, 0.1000015), (2.0, 0.1000015), 0.0, True),
        ((0.0, 0.0), (0.0, 0.0), 0.2, False),
        ((3.0, 4.0), (3.0, 4.0), 0.2, True),
    ],
)
def test_polygon_chords_retain_crossings_tangency_and_zero_clearance_tolerance(start, end, clearance, expected):
    from pedstrain_simulation.simulation.ros2_navigation.planner import segment_polygon_clear

    polygon = ((-0.5, -0.1), (0.5, -0.1), (0.5, 0.1), (-0.5, 0.1))
    assert segment_polygon_clear(start, end, polygon, clearance) is expected


def test_polygon_chords_match_independent_scalar_distance_for_rotated_rectangles():
    from pedstrain_simulation.simulation.ros2_navigation.planner import segment_polygon_clear

    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]

    def difference(a, b):
        return a[0] - b[0], a[1] - b[1]

    def point_distance(point, a, b):
        dx, dy = difference(b, a)
        length = dx * dx + dy * dy
        fraction = min(1.0, max(0.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length)) if length else 0.0
        return math.hypot(point[0] - a[0] - fraction * dx, point[1] - a[1] - fraction * dy)

    def reference(start, end, polygon, clearance):
        sides = list(zip(polygon, (*polygon[1:], polygon[0])))
        if any(
            all(cross(difference(b, a), difference(point, a)) >= -1e-12 for a, b in sides) for point in (start, end)
        ):
            return False
        delta = difference(end, start)
        distance = math.inf
        for a, b in sides:
            edge = difference(b, a)
            denominator = cross(delta, edge)
            if abs(denominator) > 1e-14:
                t = cross(difference(a, start), edge) / denominator
                u = cross(difference(a, start), delta) / denominator
                if 0 <= t <= 1 and 0 <= u <= 1:
                    return False
            distance = min(
                distance,
                point_distance(start, a, b),
                point_distance(end, a, b),
                point_distance(a, start, end),
                point_distance(b, start, end),
            )
        return distance > math.hypot(clearance, 1e-6)

    generator = np.random.default_rng(20261104)
    for index in range(300):
        angle = generator.uniform(-math.pi, math.pi)
        cosine, sine = math.cos(angle), math.sin(angle)
        center = generator.uniform(-4.0, 4.0, 2)
        half_x, half_y = generator.uniform(0.01, 1.0, 2)
        polygon = [
            (center[0] + cosine * x - sine * y, center[1] + sine * x + cosine * y)
            for x, y in ((-half_x, -half_y), (half_x, -half_y), (half_x, half_y), (-half_x, half_y))
        ]
        start, end = generator.uniform(-6.0, 6.0, (2, 2))
        if index % 3 == 0:
            end = start
        clearance = float(generator.choice((0.0, 1e-6, 0.03, 0.28, 0.6)))
        assert segment_polygon_clear(start, end, polygon, clearance) == reference(start, end, polygon, clearance)


def test_oriented_corners_retain_parent_nonuniform_scale_and_apply_the_live_rigid_pose(monkeypatch):
    frame = StageFrame(1.0, "Z")
    stage, _, _ = _thin_rotated_door(monkeypatch, frame, angle=0.0)
    stage.RemovePrim("/Door")
    parent = UsdGeom.Xform.Define(stage, "/Parent")
    parent.AddTranslateOp().Set((3.0, 4.0, 0.0))
    parent.AddRotateZOp().Set(30.0)
    parent.AddScaleOp().Set((2.0, 0.5, 1.5))
    root = UsdGeom.Xform.Define(stage, "/Parent/Door")
    root.AddTranslateOp().Set((1.0, 0.2, 0.9))
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateKinematicEnabledAttr(True)
    panel = UsdGeom.Cube.Define(stage, "/Parent/Door/Panel")
    panel.CreateSizeAttr(1.0)
    panel.AddTranslateOp().Set((0.3, 0.1, 0.0))
    panel.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set((2.0, 0.1, 1.8))
    UsdPhysics.CollisionAPI.Apply(panel.GetPrim())
    angle, position = math.radians(-25.0), (10.0, -2.0, 1.35)
    sys.modules["omni.physx"].get_physx_interface = lambda: SimpleNamespace(
        get_rigidbody_transformation=lambda path: {
            "ret_val": True,
            "position": position,
            "rotation": (0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)),
        }
    )
    obstacle = query_dynamic_obstacles(stage, frame, ())[0]
    cosine, sine = math.cos(angle), math.sin(angle)
    # Scale the offset and the physical half extents, then rotate into the live pose.
    center = (position[0] + cosine * 0.6 - sine * 0.05, position[1] + sine * 0.6 + cosine * 0.05)
    expected = [
        (center[0] + cosine * x - sine * y, center[1] + sine * x + cosine * y)
        for x, y in ((-2.0, -0.025), (2.0, -0.025), (2.0, 0.025), (-2.0, 0.025))
    ]
    np.testing.assert_allclose(sorted(obstacle["footprint_xy_m"]), sorted(expected), atol=1e-10)
    assert _polygon_area(obstacle["footprint_xy_m"]) == pytest.approx(0.2)
    assert obstacle["position"] == pytest.approx((*center, position[2]))
    assert obstacle["z_min_m"] == pytest.approx(0.0) and obstacle["z_max_m"] == pytest.approx(2.7)


def test_dynamic_observer_caches_topology_but_follows_fresh_pose_and_releases_its_notice(monkeypatch):
    from pxr import UsdSkel

    from pedstrain_simulation.simulation.ros2_navigation import mapping

    observer_type = mapping.DynamicObstacleObserver
    frame = StageFrame(1.0, "Z")
    stage, identifier = _stage(frame)
    translation = UsdGeom.Xformable(stage.GetPrimAtPath("/Scene/Crowd/Body")).AddTranslateOp()
    translation.Set((0, 0, 0))
    pose = UsdSkel.Animation.Define(stage, "/Poses/Idle").CreateTranslationsAttr()
    pose.Set([(0, 0, 0)])
    _physics(monkeypatch, _Queries(frame), identifier)
    live = {"position": (9.0, 10.0, 1.0)}
    sys.modules["omni.physx"].get_physx_interface = lambda: SimpleNamespace(
        get_rigidbody_transformation=lambda path: {"ret_val": True, **live, "rotation": (0, 0, 0, 1)}
    )
    inventories, notices = [], []
    original_inventory, original_notice = mapping._inventory, observer_type._on_objects_changed

    def inventory(*args):
        inventories.append(1)
        return original_inventory(*args)

    def changed(self, notice, sender):
        notices.append(1)
        return original_notice(self, notice, sender)

    monkeypatch.setattr(mapping, "_inventory", inventory)
    monkeypatch.setattr(observer_type, "_on_objects_changed", changed)
    observer = observer_type(stage, frame, ())
    assert observer.sample()[0]["position"] == pytest.approx((9, 10, 1))
    live["position"] = (11.0, 12.0, 1.0)
    translation.Set((0.5, 0.25, 0))
    pose.Set([(0.1, 0, 0)])
    assert observer.sample()[0]["position"] == pytest.approx((11, 12, 1))
    observer.sample()
    assert len(inventories) == 1
    observer.close()
    observer.close()
    after_close = len(notices)
    translation.Set((1, 0, 0))
    assert len(notices) == after_close
    with pytest.raises(RuntimeError, match="closed"):
        observer.sample()


@pytest.mark.parametrize(
    "change", ["add_static", "remove_static", "disable_static", "static_to_dynamic", "dynamic_to_static"]
)
def test_dynamic_observer_rejects_static_membership_changes_instead_of_using_a_stale_map(monkeypatch, change):
    from pedstrain_simulation.simulation.ros2_navigation.mapping import DynamicObstacleObserver

    frame = StageFrame(1.0, "Z")
    stage, identifier = _stage(frame)
    _physics(monkeypatch, _Queries(frame), identifier)
    observer = DynamicObstacleObserver(stage, frame, ())
    try:
        assert len(observer.sample()) == 1
        if change == "add_static":
            UsdPhysics.CollisionAPI.Apply(UsdGeom.Cube.Define(stage, "/NewStatic").GetPrim())
        elif change == "remove_static":
            stage.RemovePrim("/Scene/Wall")
        elif change == "disable_static":
            UsdPhysics.CollisionAPI(stage.GetPrimAtPath("/Scene/Wall")).GetCollisionEnabledAttr().Set(False)
        elif change == "static_to_dynamic":
            UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath("/Scene/Wall"))
        else:
            _fixed_joint(stage, "/Constraints/Anchor", "/Scene/Crowd/Body")
        with pytest.raises(RuntimeError, match="Static.*rebuild.*map.*session"):
            observer.sample()
    finally:
        observer.close()


def test_declared_dynamic_roots_remain_live_when_their_body_is_world_anchored(monkeypatch):
    from pedstrain_simulation.simulation.ros2_navigation.mapping import DynamicObstacleObserver

    frame = StageFrame(1.0, "Z")
    stage, identifier = _stage(frame)
    _physics(monkeypatch, _Queries(frame), identifier)
    observer = DynamicObstacleObserver(stage, frame, (), dynamic_roots=("/Scene/Crowd/Body",))
    try:
        assert len(observer.sample()) == 1
        _fixed_joint(stage, "/Constraints/Anchor", "/Scene/Crowd/Body")
        assert [o["prim_path"] for o in observer.sample()] == ["/Scene/Crowd/Body"]
    finally:
        observer.close()


def test_dynamic_observer_invalidates_collision_additions_and_fixed_constraint_changes(monkeypatch):
    from pedstrain_simulation.simulation.ros2_navigation.mapping import DynamicObstacleObserver

    frame = StageFrame(1.0, "Z")
    stage, identifier = _stage(frame)
    _physics(monkeypatch, _Queries(frame), identifier)
    observer = DynamicObstacleObserver(stage, frame, ())
    assert len(observer.sample()) == 1
    enabled = UsdPhysics.CollisionAPI(stage.GetPrimAtPath("/Scene/Crowd/Body")).GetCollisionEnabledAttr()
    enabled.Set(False)
    assert observer.sample() == []
    enabled.Set(True)
    assert len(observer.sample()) == 1
    body = UsdGeom.Cube.Define(stage, "/NewBody").GetPrim()
    UsdPhysics.CollisionAPI.Apply(body)
    UsdPhysics.RigidBodyAPI.Apply(body)
    assert len(observer.sample()) == 2
    joint = _fixed_joint(stage, "/Constraints/Fixed", "/Scene/Crowd/Body")
    joint.CreateBreakForceAttr(1)
    assert len(observer.sample()) == 2
    joint.CreateJointEnabledAttr(False)
    assert len(observer.sample()) == 2
    joint.CreateJointEnabledAttr(True)
    assert len(observer.sample()) == 2
    joint.GetBreakForceAttr().Set(math.inf)
    with pytest.raises(RuntimeError, match="Static.*rebuild.*map.*session"):
        observer.sample()
    observer.close()
