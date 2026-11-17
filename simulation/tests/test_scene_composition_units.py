# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical unit and frame contracts at USD composition boundaries."""

import math
from types import SimpleNamespace

import pytest

from pxr import Usd, UsdGeom

from pedstrain_simulation.simulation.configuration.models import RobotConfig, SceneConfig
from pedstrain_simulation.simulation.scene.composition import create_runtime_layer


def _scene(path, units=1.0, axis="Z"):
    stage = Usd.Stage.CreateNew(str(path))
    if units is not None:
        UsdGeom.SetStageMetersPerUnit(stage, units)
    if axis is not None:
        UsdGeom.SetStageUpAxis(stage, axis)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Campus").GetPrim())
    stage.GetRootLayer().Save()
    return stage


@pytest.mark.parametrize("units,axis", [(0.01, "Z"), (1.0, "Y"), (math.nan, "Z")])
@pytest.mark.parametrize("kind", ["asset", "robot_layer", "scene_layer"])
def test_composition_rejects_incompatible_authored_metrics_before_writing(tmp_path, units, axis, kind):
    base = _scene(tmp_path / "base.usda")
    source_path = tmp_path / "source.usda"
    source = _scene(source_path, units, axis)
    robot = UsdGeom.Xform.Define(source, "/Campus/Robot").GetPrim()
    source.SetDefaultPrim(robot)
    UsdGeom.Cube.Define(source, "/Campus/Robot/Body").CreateSizeAttr(100 if units == 0.01 else 1)
    source.GetRootLayer().Save()
    original = source_path.read_bytes()
    config = SimpleNamespace(
        scene=SceneConfig(layers=(source_path,) if kind == "scene_layer" else ()),
        robots=()
        if kind == "scene_layer"
        else (
            RobotConfig(**{"asset_path" if kind == "asset" else "layer_path": source_path}, prim_path="/Campus/Robot"),
        ),
    )
    output = tmp_path / "runtime"
    output.mkdir()
    with pytest.raises(ValueError, match="units|metersPerUnit|upAxis"):
        create_runtime_layer(config, base.GetRootLayer().realPath, output)
    assert source_path.read_bytes() == original
    assert not (output / "simulation.usda").exists()


def test_overlay_without_authored_metrics_uses_the_destination_frame(tmp_path):
    base = _scene(tmp_path / "base.usda")
    overlay_path = tmp_path / "overlay.usda"
    overlay = _scene(overlay_path, None, None)
    UsdGeom.Cube.Define(overlay, "/Campus/Body").CreateSizeAttr(1)
    overlay.GetRootLayer().Save()
    config = SimpleNamespace(scene=SceneConfig(layers=(overlay_path,)), robots=())
    result = Usd.Stage.Open(str(create_runtime_layer(config, base.GetRootLayer().realPath, tmp_path)))
    assert UsdGeom.GetStageMetersPerUnit(result) == 1
    assert UsdGeom.GetStageUpAxis(result) == "Z"
    assert result.GetPrimAtPath("/Campus/Body")


@pytest.mark.parametrize("kind", ["asset", "robot_layer", "scene_layer"])
def test_composition_rejects_different_mass_units_before_writing(tmp_path, kind):
    from pxr import UsdPhysics

    base = _scene(tmp_path / "base.usda")
    source_path = tmp_path / "grams.usda"
    source = _scene(source_path)
    UsdPhysics.SetStageKilogramsPerUnit(source, 0.001)
    source.SetDefaultPrim(UsdGeom.Xform.Define(source, "/Campus/Robot").GetPrim())
    source.GetRootLayer().Save()
    config = SimpleNamespace(
        scene=SceneConfig(layers=(source_path,) if kind == "scene_layer" else ()),
        robots=()
        if kind == "scene_layer"
        else (
            RobotConfig(**{"asset_path" if kind == "asset" else "layer_path": source_path}, prim_path="/Campus/Robot"),
        ),
    )
    with pytest.raises(ValueError, match="kilogramsPerUnit"):
        create_runtime_layer(config, base.GetRootLayer().realPath, tmp_path)
    assert not (tmp_path / "simulation.usda").exists()


def test_runtime_composition_retains_matching_mass_units(tmp_path):
    from pxr import UsdPhysics

    base = _scene(tmp_path / "base.usda")
    source_path = tmp_path / "robot.usda"
    source = _scene(source_path)
    for stage in (base, source):
        UsdPhysics.SetStageKilogramsPerUnit(stage, 0.001)
    body = UsdGeom.Cube.Define(source, "/Campus/Robot").GetPrim()
    UsdPhysics.MassAPI.Apply(body).CreateMassAttr(1000)
    source.SetDefaultPrim(body)
    base.GetRootLayer().Save()
    source.GetRootLayer().Save()
    config = SimpleNamespace(
        scene=SceneConfig(), robots=(RobotConfig(asset_path=source_path, prim_path="/Campus/Robot"),)
    )
    result = Usd.Stage.Open(str(create_runtime_layer(config, base.GetRootLayer().realPath, tmp_path)))
    mass = UsdPhysics.MassAPI(result.GetPrimAtPath("/Campus/Robot")).GetMassAttr().Get()
    assert mass * UsdPhysics.GetStageKilogramsPerUnit(result) == pytest.approx(1.0)


@pytest.mark.parametrize("units,axis", [(1.0, "Z"), (0.01, "Z"), (0.001, "Y")])
def test_matching_robot_metrics_preserve_asset_scale_and_pose_units(tmp_path, units, axis):
    from pxr import Gf

    base = _scene(tmp_path / "base.usda", units, axis)
    source_path = tmp_path / "robot.usda"
    source = _scene(source_path, units, axis)
    robot = UsdGeom.Xform.Define(source, "/Campus/Robot")
    source.SetDefaultPrim(robot.GetPrim())
    robot.AddScaleOp().Set((0.01, 0.01, 0.01))
    UsdGeom.Cube.Define(source, "/Campus/Robot/Body").CreateSizeAttr(1 / units / 0.01)
    source.GetRootLayer().Save()
    original = source_path.read_bytes()
    config = SimpleNamespace(
        scene=SceneConfig(),
        robots=(RobotConfig(asset_path=source_path, prim_path="/Campus/Robot", position=(2, 3, 4), yaw_degrees=90),),
    )
    result = Usd.Stage.Open(str(create_runtime_layer(config, base.GetRootLayer().realPath, tmp_path)))
    root = UsdGeom.Xformable(result.GetPrimAtPath("/Campus/Robot"))
    matrix = root.GetLocalTransformation()
    assert tuple(v * units for v in matrix.ExtractTranslation()) == pytest.approx((2, 3, 4))
    assert tuple(Gf.Transform(matrix).GetScale()) == pytest.approx((0.01, 0.01, 0.01))
    forward = matrix.TransformDir(Gf.Vec3d(1, 0, 0)).GetNormalized()
    assert tuple(forward) == pytest.approx((0, 1, 0) if axis == "Z" else (0, 0, -1), abs=1e-8)
    bounds = (
        UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
        .ComputeWorldBound(result.GetPrimAtPath("/Campus/Robot/Body"))
        .ComputeAlignedRange()
    )
    assert tuple(v * units for v in bounds.GetSize()) == pytest.approx((1, 1, 1))
    assert source_path.read_bytes() == original
