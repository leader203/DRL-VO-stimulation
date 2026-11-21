# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent character pose/animation contracts without starting Kit or ROS."""

import math
from pathlib import Path

import pytest

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdSkel

from pedstrain_simulation.simulation.configuration.scenarios.models import PedestrianPlan
from pedstrain_simulation.simulation.ros2_navigation import actors
from pedstrain_simulation.simulation.ros2_navigation.frames import StageFrame


class _Rig:
    joints = ("root", "root/knee")
    units = 1.0
    authored_yaw = -math.pi / 2
    alignment = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), 90.0))
    arc = (0.0, 0.5, 1.0)
    idle_times = (0.0, 1.0)

    def __init__(self, asset, motion):
        base = ((Gf.Vec3f(0.0), Gf.Vec3f(0.0, 0.0, 0.8)), (Gf.Quatf(1.0), Gf.Quatf(1.0)))
        bent = (base[0], (Gf.Quatf(1.0), Gf.Quatf(Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), 60.0).GetQuat())))
        self.walk, self.idle, self.closed = (base, bent, base), (base, base), False

    def close(self):
        self.closed = True


def _asset(path):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    root = UsdSkel.Root.Define(stage, "/Visual")
    stage.SetDefaultPrim(root.GetPrim())
    skeleton = UsdSkel.Skeleton.Define(stage, "/Visual/Skeleton")
    skeleton.CreateJointsAttr(_Rig.joints)
    skeleton.CreateRestTransformsAttr([Gf.Matrix4d(1.0), Gf.Matrix4d().SetTranslate(Gf.Vec3d(0.0, 0.0, 0.8))])
    skeleton.CreateBindTransformsAttr(skeleton.GetRestTransformsAttr().Get())
    stage.GetRootLayer().Save()
    return path


def _population(tmp_path, monkeypatch, *, up="Z", units=1.0, origin=(0.0, 0.0, 0.0), yaw=0.0, count=2):
    monkeypatch.setattr(actors, "_Motion", lambda path: object())
    monkeypatch.setattr(actors, "_Rig", _Rig)
    asset = _asset(tmp_path / "actor.usda")
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, units)
    UsdGeom.SetStageUpAxis(stage, up)
    UsdGeom.Xform.Define(stage, "/Shifted").AddTranslateOp().Set((100.0, -200.0, 300.0))
    plans = tuple(
        PedestrianPlan(f"p{i}", position=(i * 2.0, 1.0, 0.0), yaw=30.0, behavior="stand") for i in range(count)
    )
    frame = StageFrame(units, up, origin, yaw)
    population = actors.Ros2Actors(
        stage, plans, {p.name: asset for p in plans}, Path("motion.usd"), "/Shifted/Actors", frame, 9
    )
    return population, stage, frame, plans


@pytest.mark.parametrize("up", ["Y", "Z"])
@pytest.mark.parametrize("units", [0.01, 1.0])
def test_actor_world_pose_facing_capsule_and_layer_cleanup_are_independent_of_scene(tmp_path, monkeypatch, up, units):
    population, stage, frame, plans = _population(
        tmp_path, monkeypatch, up=up, units=units, origin=(4.0, -3.0, 0.2), yaw=0.4
    )
    parent = UsdGeom.Xformable(stage.GetPrimAtPath("/Shifted"))
    parent.AddRotateXYZOp().Set((15.0, -20.0, 30.0))
    parent.AddScaleOp().Set((2.0, 3.0, 4.0))
    states = population.states()
    for plan, state in zip(plans, states):
        assert state["position"] == pytest.approx(frame.world_to_ros(plan.position))
        assert math.remainder(
            state["yaw"] - (math.radians(plan.yaw) - math.pi / 2 - 0.4), 2 * math.pi
        ) == pytest.approx(0.0, abs=1e-6)
        body = UsdGeom.Capsule.Get(stage, f"/Shifted/Actors/{plan.name}/BodyCollider")
        assert body.GetRadiusAttr().Get() * units == pytest.approx(0.25)
        assert (body.GetHeightAttr().Get() + 2 * body.GetRadiusAttr().Get()) * units == pytest.approx(1.75)
        assert body.GetVisibilityAttr().Get() == "invisible"
        assert UsdPhysics.CollisionAPI(body.GetPrim()).GetCollisionEnabledAttr().Get()
        root = stage.GetPrimAtPath(f"/Shifted/Actors/{plan.name}")
        assert UsdPhysics.RigidBodyAPI(root).GetKinematicEnabledAttr().Get()
        assert not any("BehaviorAgent" in str(p.GetMetadata("apiSchemas")) for p in Usd.PrimRange(root))
    rigs = list(population._rigs.values())
    population.close()
    population.close()
    assert not stage.GetPrimAtPath("/Shifted/Actors")
    assert stage.GetPrimAtPath("/Shifted") and all(r.closed for r in rigs)


@pytest.mark.parametrize(
    "invalid", ["missing", "duplicate", "nan", "yaw", "vertical", "vertical_position", "unknown", "bool"]
)
def test_invalid_batch_cannot_partially_move_another_actor_or_advance_animation(tmp_path, monkeypatch, invalid):
    population, stage, _, _ = _population(tmp_path, monkeypatch)
    commands = population.states()
    commands[0]["position"][0] += 0.1
    if invalid == "missing":
        commands.pop()
    elif invalid == "duplicate":
        commands[1]["name"] = commands[0]["name"]
    elif invalid == "nan":
        commands[1]["position"][1] = math.nan
    elif invalid == "yaw":
        commands[1]["yaw"] = math.inf
    elif invalid == "vertical":
        commands[1]["velocity"][2] = 0.1
    elif invalid == "vertical_position":
        commands[1]["position"][2] += 0.3
    elif invalid == "unknown":
        commands[1]["name"] = "foreign"
    else:
        commands[1]["position"][0] = True
    before, status = stage.GetSessionLayer().ExportToString(), population.status()
    owned = population._layer.ExportToString()
    with pytest.raises(ValueError):
        population.apply(commands, 1 / 60)
    assert stage.GetSessionLayer().ExportToString() == before
    assert population._layer.ExportToString() == owned and population.status() == status
    population.close()


def test_gait_phase_uses_actual_root_travel_and_freezes_when_velocity_claims_motion(tmp_path, monkeypatch):
    population, stage, _, _ = _population(tmp_path, monkeypatch, count=1)
    start = population.status()["actors"]["p0"]
    commands = population.states()
    commands[0]["velocity"] = [8.0, 0.0, 0.0]
    population.apply(commands, 0.1)
    still = population.status()["actors"]["p0"]
    assert still["walk_phase"] == start["walk_phase"] and still["walk_weight"] == 0.0
    assert population.states()[0]["velocity"] == [0.0, 0.0, 0.0]
    commands[0]["position"][0] += 0.25
    population.apply(commands, 0.5)
    moving = population.status()["actors"]["p0"]
    assert moving["distance_m"] == pytest.approx(0.25)
    assert (moving["walk_phase"] - start["walk_phase"]) % 1.0 == pytest.approx(0.25)
    assert population.states()[0]["velocity"] == pytest.approx([0.5, 0.0, 0.0])
    animation = UsdSkel.Animation.Get(stage, "/Shifted/Actors/p0/Pose")
    assert len(animation.GetRotationsAttr().Get()) == len(_Rig.joints)
    for _ in range(20):
        population.apply(population.states(), 0.1)
    stopped = population.status()["actors"]["p0"]
    assert stopped["walk_phase"] == moving["walk_phase"] and stopped["walk_weight"] < 1e-6
    population.close()


def test_slow_translation_uses_shorter_visible_steps_for_the_same_travel(tmp_path, monkeypatch):
    population, _, _, _ = _population(tmp_path, monkeypatch)
    start = population.status()["actors"]
    for index in range(30):
        commands = population.states()
        commands[0]["position"][0] += 0.12 / 60
        if index < 6:
            commands[1]["position"][0] += 0.6 / 60
        population.apply(commands, 1 / 60)

    result = population.status()["actors"]
    assert [result[name]["distance_m"] for name in ("p0", "p1")] == pytest.approx([0.06, 0.06])
    slow_phase, normal_phase = ((result[name]["walk_phase"] - start[name]["walk_phase"]) % 1.0 for name in ("p0", "p1"))
    assert slow_phase >= 2 * normal_phase
    assert result["p0"]["walk_weight"] > 0.2
    population.close()


@pytest.mark.parametrize("up", ["Y", "Z"])
@pytest.mark.parametrize("writer", ["position", "orientation", "scale", "roll", "pitch", "shear"])
def test_other_root_writer_is_detected_instead_of_silently_fighting_ros(tmp_path, monkeypatch, writer, up):
    population, stage, _, _ = _population(tmp_path, monkeypatch, count=1, up=up)
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        root = stage.GetPrimAtPath("/Shifted/Actors/p0")
        if writer == "position":
            root.GetAttribute("xformOp:translate").Set((9.0, 0.0, 0.0))
        elif writer == "orientation":
            root.GetAttribute("xformOp:orient").Set(Gf.Quatd(1.0))
        elif writer == "scale":
            UsdGeom.Xformable(root).AddScaleOp().Set((1.2, 1.2, 1.2))
        elif writer in ("roll", "pitch"):
            axis = Gf.Vec3d(1.0, 0.0, 0.0) if writer == "roll" else Gf.Vec3d(0.0, 1.0, 0.0)
            orientation = root.GetAttribute("xformOp:orient")
            orientation.Set(orientation.Get() * Gf.Rotation(axis, 15.0).GetQuat())
        else:
            shear = Gf.Matrix4d(1.0)
            shear.SetRow(1, Gf.Vec4d(0.2, 1.0, 0.0, 0.0))
            UsdGeom.Xformable(root).AddTransformOp(opSuffix="externalShear").Set(shear)
    before = stage.GetSessionLayer().ExportToString(), population._layer.ExportToString()
    with pytest.raises(RuntimeError, match="Another transform writer") as failure:
        population.states()
    assert "expected_position_m=" in str(failure.value) and "actual_position_m=" in str(failure.value)
    assert "position_error_m=" in str(failure.value) and "yaw_error_rad=" in str(failure.value)
    assert (stage.GetSessionLayer().ExportToString(), population._layer.ExportToString()) == before
    population.close()


@pytest.mark.parametrize("up", ["Y", "Z"])
@pytest.mark.parametrize("units", [0.01, 1.0])
@pytest.mark.parametrize("map_yaw", [0.0, 0.4])
def test_near_identity_world_rotations_do_not_report_a_second_pose_writer(tmp_path, monkeypatch, up, units, map_yaw):
    population, _, _, _ = _population(tmp_path, monkeypatch, count=1, up=up, units=units, yaw=map_yaw)
    # The first two angles reproduce the recorded 5.25s turn and its next tick.
    for world_yaw in (0.03602866060060651, -0.005639096063488816, 0.0, 1e-4, -1e-4, 0.001, -0.001):
        commands = population.states()
        expected = world_yaw - map_yaw
        commands[0]["yaw"] = expected
        population.apply(commands, 1 / 60)
        actual = population.states()[0]
        assert abs(math.remainder(actual["yaw"] - expected, 2 * math.pi)) < 1e-5
    population.close()


@pytest.mark.parametrize("dt", [0.0, -0.01, math.inf, math.nan, True])
def test_actor_update_requires_a_real_positive_timestep(tmp_path, monkeypatch, dt):
    population, _, _, _ = _population(tmp_path, monkeypatch, count=1)
    with pytest.raises(ValueError):
        population.apply(population.states(), dt)
    population.close()


def test_retarget_aims_every_bone_and_reproduces_the_rest_pose(tmp_path):
    import types

    from pxr import Sdf, UsdUtils

    def skeleton_asset(path, count, units, up):
        stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageMetersPerUnit(stage, units)
        UsdGeom.SetStageUpAxis(stage, up)
        root = UsdSkel.Root.Define(stage, "/Visual")
        stage.SetDefaultPrim(root.GetPrim())
        skeleton = UsdSkel.Skeleton.Define(stage, "/Visual/Skeleton")
        joints = ["Root", *(f"Root/j{i}" for i in range(count - 1))]
        skeleton.CreateJointsAttr(joints)
        transforms = [Gf.Matrix4d(1.0) for _ in joints]
        tags = [
            "",
            "Hips",
            "LeftLeg",
            "LeftShin",
            "LeftFoot",
            "RightLeg",
            "RightShin",
            "RightFoot",
            *([""] * (count - 8)),
        ]
        for index in (2, 3, 4, 5, 6, 7):
            point = Gf.Vec3d(0.0)
            point["XYZ".index(up)] = (0.8 - 0.35 * ((index - 2) % 3)) / units
            transforms[index].SetTranslate(point)
        skeleton.CreateRestTransformsAttr(transforms)
        skeleton.CreateBindTransformsAttr(transforms)
        prim = skeleton.GetPrim()
        prim.CreateAttribute("controlRig:retargetTags", Sdf.ValueTypeNames.TokenArray).Set(tags)
        prim.CreateAttribute("controlRig:retargetTransforms", Sdf.ValueTypeNames.Matrix4dArray).Set(transforms)
        prim.CreateAttribute("controlRig:forwardAxis", Sdf.ValueTypeNames.Token).Set("Z" if up == "Y" else "MINUS Y")
        prim.CreateAttribute("controlRig:upAxis", Sdf.ValueTypeNames.Token).Set(up)
        stage.GetRootLayer().Save()
        return stage, skeleton, tags

    source_path, target_path = tmp_path / "source.usda", tmp_path / "target.usda"
    source, skeleton, tags = skeleton_asset(source_path, 81, 0.01, "Y")
    skeleton_asset(target_path, 101, 1.0, "Z")
    animation = UsdSkel.Animation.Define(source, "/Visual/SourceMotion")
    animation.CreateJointsAttr(skeleton.GetJointsAttr().Get())
    for time in (0.0, 1.0):
        # An unanimated clip: every joint holds its rest rotation, so a faithful
        # retarget must hand back the target's own rest pose.
        animation.CreateTranslationsAttr().Set([Gf.Vec3f(5.0, 6.0, 7.0) for _ in range(81)], time)
        animation.CreateRotationsAttr().Set([Gf.Quatf(1.0) for _ in range(81)], time)
    motion = types.SimpleNamespace(
        path=source_path,
        skeleton=skeleton,
        joints=tuple(skeleton.GetJointsAttr().Get()),
        tags=tags,
        units=0.01,
        tcps=30.0,
        arc=(0.0, 1.0),
        walk=animation,
        idle=animation,
        walk_times=(0.0, 1.0),
        idle_times=(0.0, 1.0),
        travel_direction=lambda: Gf.Vec3d(0.0, 0.0, 1.0),
    )

    def parts(quaternion):
        return (quaternion.GetReal(), *tuple(quaternion.GetImaginary()))

    def distance(first, second):
        a, b = parts(first), parts(second)
        return abs(abs(sum(x * y for x, y in zip(a, b))) - 1.0)

    rig = actors._Rig(target_path, motion)
    assert rig.controller is None
    assert len(rig.walk) == 2 and len(rig.idle) == 2
    for translations, rotations in (*rig.walk, *rig.idle):
        assert len(translations) == len(rotations) == 101
        assert all(abs(sum(v * v for v in parts(q)) - 1.0) < 1e-5 for q in rotations)
        for index, matrix in enumerate(rig.rest):
            assert distance(rotations[index], matrix.ExtractRotationQuat()) < 1e-4
            assert Gf.Vec3f(translations[index]) == Gf.Vec3f(matrix.ExtractTranslation())
    identifier = rig.cache_id
    rig.close()
    assert not UsdUtils.StageCache.Get().Find(identifier)
