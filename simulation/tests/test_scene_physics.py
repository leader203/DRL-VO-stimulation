# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Verify stage ownership, physics settings and robot visuals."""

import ast
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize("default_root", [None, "/Campus", "/Depot/Interior"])
def test_runtime_namespace_needs_no_world_and_preserves_transformed_roots(default_root):
    from pxr import Usd, UsdGeom

    from pedstrain_simulation.simulation.scene.namespaces import select_runtime_prim_path

    stage = Usd.Stage.CreateInMemory()
    if default_root:
        root = UsdGeom.Xform.Define(stage, default_root)
        root.AddTranslateOp().Set((14.0, -3.0, 0.0))
        stage.SetDefaultPrim(root.GetPrim())
    before = stage.GetRootLayer().ExportToString()
    selected = select_runtime_prim_path(stage, "Control", unique=True)
    assert str(selected) == (default_root or "") + "/Control"
    assert stage.GetRootLayer().ExportToString() == before
    stage.DefinePrim(selected)
    assert str(select_runtime_prim_path(stage, "Control", unique=True)) == str(selected) + "_1"
    assert not stage.GetPrimAtPath("/World")


def _behavior_class(relative_path, name, **namespace):
    source = ast.parse((Path(__file__).resolve().parents[1] / relative_path).read_text())
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == name)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    scope = {"math": math, **namespace}
    exec(compile(module, relative_path, "exec"), scope)
    return scope[name]


def _scene_physics_boundary(monkeypatch, stage):
    import sys

    import pxr
    from pxr import Sdf

    modules = {name: ModuleType(name) for name in ("omni", "omni.usd")}
    modules["omni"].usd = modules["omni.usd"]
    modules["omni.usd"].get_context = lambda: SimpleNamespace(get_stage=lambda: stage)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    def apply(prim):
        return SimpleNamespace(
            CreateTimeStepsPerSecondAttr=lambda value: prim.CreateAttribute(
                "physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.Int
            ).Set(value),
            CreateSolverTypeAttr=lambda value: prim.CreateAttribute(
                "physxScene:solverType", Sdf.ValueTypeNames.Token
            ).Set(value),
        )

    monkeypatch.setattr(pxr, "PhysxSchema", SimpleNamespace(PhysxSceneAPI=SimpleNamespace(Apply=apply)), raising=False)


def _instance_physics_stage(tmp_path):
    from pxr import Usd, UsdGeom, UsdPhysics

    source = Usd.Stage.CreateNew(str(tmp_path / "physics_instance.usda"))
    root = UsdGeom.Xform.Define(source, "/Asset").GetPrim()
    source.SetDefaultPrim(root)
    UsdPhysics.Scene.Define(source, "/Asset/Dynamics")
    source.GetRootLayer().Save()
    stage = Usd.Stage.CreateInMemory()
    campus = UsdGeom.Xform.Define(stage, "/Campus").GetPrim()
    stage.SetDefaultPrim(campus)
    instance = stage.DefinePrim("/Campus/Imported")
    instance.GetReferences().AddReference(str(tmp_path / "physics_instance.usda"))
    instance.SetInstanceable(True)
    return stage


@pytest.mark.parametrize("override", [False, True])
def test_scene_physics_counts_scenes_inside_instances_before_mutation(tmp_path, monkeypatch, override):
    from pxr import UsdPhysics

    from pedstrain_simulation.simulation.configuration.models import SimulationSettings
    from pedstrain_simulation.simulation.scene.physics import configure_physics

    stage = _instance_physics_stage(tmp_path)
    UsdPhysics.Scene.Define(stage, "/Campus/MainDynamics")
    _scene_physics_boundary(monkeypatch, stage)
    before = stage.GetSessionLayer().ExportToString()
    with pytest.raises(ValueError, match="Multiple physics scenes"):
        configure_physics(SimulationSettings(physics_hz=60 if override else None))
    assert stage.GetSessionLayer().ExportToString() == before


def test_scene_physics_preserves_instanced_scene_and_rejects_uneditable_override(tmp_path, monkeypatch):
    from pedstrain_simulation.simulation.configuration.models import SimulationSettings
    from pedstrain_simulation.simulation.scene.physics import configure_physics

    stage = _instance_physics_stage(tmp_path)
    _scene_physics_boundary(monkeypatch, stage)
    before = stage.GetSessionLayer().ExportToString()
    source = (tmp_path / "physics_instance.usda").read_bytes()
    configure_physics(SimulationSettings())
    assert stage.GetSessionLayer().ExportToString() == before
    with pytest.raises(ValueError, match="instance"):
        configure_physics(SimulationSettings(physics_hz=60, solver="TGS"))
    assert stage.GetSessionLayer().ExportToString() == before
    assert (tmp_path / "physics_instance.usda").read_bytes() == source


def test_scene_physics_override_uses_the_existing_editable_scene_session_layer(monkeypatch):
    from pxr import Usd, UsdPhysics

    from pedstrain_simulation.simulation.configuration.models import SimulationSettings
    from pedstrain_simulation.simulation.scene.physics import configure_physics

    stage = Usd.Stage.CreateInMemory()
    physics = UsdPhysics.Scene.Define(stage, "/Campus/ExistingDynamics").GetPrim()
    _scene_physics_boundary(monkeypatch, stage)
    original = stage.GetRootLayer().ExportToString()
    configure_physics(SimulationSettings(physics_hz=90, solver="TGS"))
    assert physics.GetAttribute("physxScene:timeStepsPerSecond").Get() == 90
    assert physics.GetAttribute("physxScene:solverType").Get() == "TGS"
    assert stage.GetRootLayer().ExportToString() == original


@pytest.mark.parametrize("units", [1.0, 0.01])
@pytest.mark.parametrize("scale", [(0.01, 0.01, 0.01), (1, 1, 1), (2, 2, 2), (2, 3, 4)])
def test_robot_visuals_preserve_world_size_and_offsets_when_physics_pose_changes(units, scale):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, units)
    UsdGeom.SetStageUpAxis(stage, "Z")
    root = UsdGeom.Xform.Define(stage, "/Robot")
    root.AddTranslateOp().Set((5 / units, -3 / units, 0))
    root.AddRotateZOp().Set(90)
    body = UsdGeom.Xform.Define(stage, "/Robot/Body")
    body.AddTranslateOp().Set((2 / units, 1 / units, 0.3 / units))
    body.AddRotateZOp().Set(30)
    body.AddScaleOp().Set(scale)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    corners = [(x, y, z) for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)]
    mesh = UsdGeom.Mesh.Define(stage, "/Robot/Body/Visual")
    mesh.CreatePointsAttr([tuple(point[i] / (units * scale[i]) for i in range(3)) for point in corners])
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr([0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.AddTranslateOp().Set((0.25 / (units * scale[0]), 0, 0))
    settings = SimpleNamespace(
        GetAttribute=lambda name: SimpleNamespace(Get=lambda: True),
        GetRelationship=lambda name: SimpleNamespace(GetTargets=lambda: [body.GetPath()]),
    )
    native_pose = {}
    physics = SimpleNamespace(
        get_rigidbody_transformation=lambda path: {"ret_val": True, **native_pose},
        subscribe_physics_on_step_events=lambda *args: SimpleNamespace(unsubscribe=lambda: None),
    )
    cls = _behavior_class(
        "behaviors/robots/visuals.py",
        "RobotVisuals",
        Gf=Gf,
        Sdf=Sdf,
        Usd=Usd,
        UsdGeom=UsdGeom,
        UsdPhysics=UsdPhysics,
        UsdShade=UsdShade,
        _Visual=lambda source, session_type, body, local_transform, transform: SimpleNamespace(
            source=source, session_type=session_type, body=body, local_transform=local_transform, transform=transform
        ),
        omni=SimpleNamespace(physx=SimpleNamespace(get_physx_interface=lambda: physics)),
    )
    original = stage.GetRootLayer().ExportToString()
    helper = cls(stage, settings)
    copy = UsdGeom.Mesh.Get(stage, "/Robot/RenderVisuals/mesh_0")
    try:
        for angle, position in ((120.0, (4.0, -1.0, 0.3)), (210.0, (6.0, 1.0, 0.8))):
            yaw = math.radians(angle)
            native_pose.update(
                position=tuple(value / units for value in position),
                rotation=(0, 0, math.sin(yaw / 2), math.cos(yaw / 2)),
            )
            helper._on_physics_step(1 / 60)
            matrix = copy.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            measured = [matrix.Transform(Gf.Vec3d(*point)) * units for point in copy.GetPointsAttr().Get()]
            for actual, (x, y, z) in zip(measured, corners, strict=True):
                expected = (
                    position[0] + math.cos(yaw) * (x + 0.25) - math.sin(yaw) * y,
                    position[1] + math.sin(yaw) * (x + 0.25) + math.cos(yaw) * y,
                    position[2] + z,
                )
                assert tuple(actual) == pytest.approx(expected, abs=2e-6)
    finally:
        helper.close()
    assert stage.GetRootLayer().ExportToString() == original
    assert stage.GetPrimAtPath("/Robot/Body/Visual").IsA(UsdGeom.Mesh)
