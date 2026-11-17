# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD characters whose sole world-pose writer is the ROS2 execution backend.

Animation uses the independent SDK retargeter and UsdSkel. It never creates an
IRA agent, navigation query, animation graph, or root-motion controller.
"""

from __future__ import annotations

import bisect
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..configuration.scenarios.models import PedestrianPlan

if TYPE_CHECKING:
    from pxr import Usd

    from .frames import StageFrame


def _finite_vector(value, size: int, label: str) -> tuple[float, ...]:
    try:
        parts = tuple(value)
        if isinstance(value, (str, bytes)) or any(isinstance(part, bool) for part in parts):
            raise ValueError(label)
        result = tuple(float(part) for part in parts)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} requires {size} finite numbers.") from error
    if len(result) != size or not all(math.isfinite(part) for part in result):
        raise ValueError(f"{label} requires {size} finite numbers.")
    return result


def _axis(token: str) -> tuple[float, float, float]:
    axes = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}
    negative = token.startswith("MINUS ")
    key = token.removeprefix("MINUS ")
    if key not in axes:
        raise ValueError(f"Unsupported authored rig axis {token!r}.")
    return tuple((-1.0 if negative else 1.0) * part for part in axes[key])


def _mix(first, second, fraction):
    from pxr import Gf

    translations = tuple((1.0 - fraction) * a + fraction * b for a, b in zip(first[0], second[0]))
    rotations = tuple(Gf.Slerp(fraction, a, b) for a, b in zip(first[1], second[1]))
    return translations, rotations


def _sample(poses, coordinates, value):
    index = min(len(coordinates) - 2, max(0, bisect.bisect_right(coordinates, value) - 1))
    width = coordinates[index + 1] - coordinates[index]
    fraction = min(1.0, max(0.0, (value - coordinates[index]) / width)) if width > 0.0 else 0.0
    return _mix(poses[index], poses[index + 1], fraction)


class _Motion:
    def __init__(self, path: Path):
        from pxr import Usd, UsdGeom, UsdSkel

        self.path = path
        self.stage = Usd.Stage.Open(str(path))
        if self.stage is None:
            raise ValueError(f"Cannot read motion library: {path}")
        skeletons = [UsdSkel.Skeleton(p) for p in self.stage.Traverse() if p.IsA(UsdSkel.Skeleton)]
        if len(skeletons) != 1:
            raise ValueError("A motion library must provide exactly one tagged source skeleton.")
        self.skeleton = skeletons[0]
        self.joints = tuple(self.skeleton.GetJointsAttr().Get())
        self.units = UsdGeom.GetStageMetersPerUnit(self.stage)
        self.up = "XYZ".index(str(UsdGeom.GetStageUpAxis(self.stage)))
        self.tcps = self.stage.GetTimeCodesPerSecond()
        self.tags = tuple(self.skeleton.GetPrim().GetAttribute("controlRig:retargetTags").Get() or ())
        if len(self.tags) != len(self.joints) or not all(tag in self.tags for tag in ("RightFoot", "LeftFoot")):
            raise ValueError("Motion skeleton lacks a complete Human retarget setup.")
        root = self.stage.GetDefaultPrim().GetPath()
        self.walk = UsdSkel.Animation.Get(self.stage, root.AppendPath("BuiltinActions/MoveWalk/WalkForward"))
        self.idle = UsdSkel.Animation.Get(self.stage, root.AppendPath("BuiltinActions/Idle/Idle"))
        if not self.walk or not self.idle:
            raise ValueError("Motion library requires native WalkForward and Idle SkelAnimation clips.")
        self.walk_times, self.arc = self._cycle()
        self.idle_times = self.idle.GetTranslationsAttr().GetTimeSamples()
        if len(self.idle_times) < 2:
            raise ValueError("Idle clip must contain at least two poses.")

    def _cycle(self):
        from pxr import UsdSkel

        times = self.walk.GetTranslationsAttr().GetTimeSamples()
        topology = UsdSkel.Topology(self.joints)
        if tuple(self.walk.GetJointsAttr().Get()) != self.joints or len(times) < 4:
            raise ValueError("Walk clip must use its source skeleton joint order and contain a gait cycle.")
        feet, arc = [], [0.0]
        root = [self.walk.GetTranslationsAttr().Get(t)[0] for t in times]
        for a, b in zip(root, root[1:]):
            arc.append(arc[-1] + self.units * math.sqrt(sum((b[k] - a[k]) ** 2 for k in range(3) if k != self.up)))
        foot = self.tags.index("RightFoot")
        for t in times:
            pose = UsdSkel.ConcatJointTransforms(topology, self.walk.GetTransforms(t))
            feet.append(float(pose[foot].ExtractTranslation()[self.up]))
        contacts = [i for i in range(2, len(times) - 2) if feet[i] == min(feet[i - 2 : i + 3])]
        candidates = []
        for a, b in zip(contacts, contacts[1:]):
            duration, length = (times[b] - times[a]) / self.tcps, arc[b] - arc[a]
            if not 0.55 <= duration <= 1.6 or not 0.45 <= length <= 2.5:
                continue
            qa, qb = self.walk.GetRotationsAttr().Get(times[a]), self.walk.GetRotationsAttr().Get(times[b])
            error = sum(
                1.0 - abs(float(x.GetReal() * y.GetReal() + x.GetImaginary() * y.GetImaginary()))
                for x, y in zip(qa[1:], qb[1:])
            )
            candidates.append((error, a, b))
        if not candidates:
            raise ValueError("WalkForward contains no verifiable steady same-foot gait cycle.")
        _, low, high = min(candidates)
        return tuple(times[low : high + 1]), tuple(value - arc[low] for value in arc[low : high + 1])


class _Rig:
    def __init__(self, asset: Path, motion: _Motion):
        from omni.anim.retarget.core import RetargetController
        from pxr import Gf, Usd, UsdGeom, UsdSkel, UsdUtils

        source = Usd.Stage.Open(str(asset))
        if source is None:
            raise ValueError(f"Cannot read character asset: {asset}")
        self.units, self.up = UsdGeom.GetStageMetersPerUnit(source), str(UsdGeom.GetStageUpAxis(source))
        self.stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(self.stage, self.units)
        UsdGeom.SetStageUpAxis(self.stage, self.up)
        UsdGeom.Xform.Define(self.stage, "/Visual").GetPrim().GetReferences().AddReference(str(asset))
        skeletons = [UsdSkel.Skeleton(p) for p in self.stage.Traverse() if p.IsA(UsdSkel.Skeleton)]
        if len(skeletons) != 1:
            raise ValueError(f"Character must contain exactly one skinned skeleton: {asset}")
        skeleton = self.skeleton = skeletons[0]
        self.joints = tuple(skeleton.GetJointsAttr().Get())
        tags = tuple(skeleton.GetPrim().GetAttribute("controlRig:retargetTags").Get() or ())
        if len(tags) != len(self.joints) or not all(tag in tags for tag in ("Hips", "LeftFoot", "RightFoot")):
            raise ValueError(f"Character lacks Human retarget tags: {asset}")
        forward = _axis(str(skeleton.GetPrim().GetAttribute("controlRig:forwardAxis").Get()))
        up = _axis(str(skeleton.GetPrim().GetAttribute("controlRig:upAxis").Get()))
        f, u = Gf.Vec3d(*forward), Gf.Vec3d(*up)
        if abs(Gf.Dot(f, u)) > 1e-6:
            raise ValueError("Character forward and up axes must be perpendicular.")
        right = Gf.Cross(f, u)
        # Asset forward becomes ROS +X; asset up becomes ROS +Z.
        self.alignment = Gf.Matrix4d(
            f[0], -right[0], u[0], 0.0, f[1], -right[1], u[1], 0.0, f[2], -right[2], u[2], 0.0, 0.0, 0.0, 0.0, 1.0
        )
        canonical_forward = (forward[0], -forward[2], forward[1]) if self.up == "Y" else forward
        self.authored_yaw = math.atan2(canonical_forward[1], canonical_forward[0])
        self.cache_id = UsdUtils.StageCache.Get().Insert(self.stage)
        try:
            self.controller = RetargetController(
                str(motion.path), str(motion.skeleton.GetPath()), self.cache_id.ToLongInt(), str(skeleton.GetPath())
            )
            if not self.controller:
                raise RuntimeError(f"SDK could not initialize independent animation retargeting: {asset}")
            self.scale = self._leg_length(skeleton, tags, self.units) / self._leg_length(
                motion.skeleton, motion.tags, motion.units
            )
            self.arc = tuple(distance * self.scale for distance in motion.arc)
            self.walk = tuple(self._retarget(motion.walk, t) for t in motion.walk_times)
            self.idle = tuple(self._retarget(motion.idle, t) for t in motion.idle_times)
            self.walk = (*self.walk[:-1], self.walk[0])
            self.idle = (*self.idle[:-1], self.idle[0])
            self.idle_times = tuple((t - motion.idle_times[0]) / motion.tcps for t in motion.idle_times)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _leg_length(skeleton, tags, units):
        from pxr import UsdSkel

        transforms = skeleton.GetPrim().GetAttribute("controlRig:retargetTransforms").Get()
        if not transforms:
            transforms = skeleton.GetRestTransformsAttr().Get()
        pose = UsdSkel.ConcatJointTransforms(UsdSkel.Topology(skeleton.GetJointsAttr().Get()), transforms)
        lengths = []
        for side in ("Left", "Right"):
            indices = [tags.index(side + suffix) for suffix in ("Leg", "Shin", "Foot")]
            lengths.append(
                sum(
                    (pose[a].ExtractTranslation() - pose[b].ExtractTranslation()).GetLength()
                    for a, b in zip(indices, indices[1:])
                )
                * units
            )
        value = sum(lengths) / 2
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("Tagged retarget leg lengths must be finite and positive [m].")
        return value

    def _retarget(self, animation, time):
        import carb
        from pxr import Gf

        translations = [carb.Float3(*v) for v in animation.GetTranslationsAttr().Get(time)]
        rotations = [carb.Float4(*q.GetImaginary(), q.GetReal()) for q in animation.GetRotationsAttr().Get(time)]
        translations[0], rotations[0] = carb.Float3(0.0, 0.0, 0.0), carb.Float4(0.0, 0.0, 0.0, 1.0)
        values, quaternions = self.controller.retarget(translations, rotations)
        if len(values) != len(self.joints) or len(quaternions) != len(self.joints):
            raise RuntimeError("SDK retargeter returned an incomplete target pose.")
        position = tuple(Gf.Vec3f(*_finite_vector(v, 3, "retarget translation")) for v in values)
        checked = tuple(_finite_vector(q, 4, "retarget rotation") for q in quaternions)
        if any(sum(v * v for v in q) < 1e-12 for q in checked):
            raise RuntimeError("SDK retargeter returned a zero rotation quaternion.")
        rotation = tuple(Gf.Quatf(q[3], q[0], q[1], q[2]).GetNormalized() for q in checked)
        # The bridge owns the world root; sampled animation contains only body motion.
        return (Gf.Vec3f(0.0, 0.0, 0.0), *position[1:]), (Gf.Quatf(1.0), *rotation[1:])

    def close(self):
        from pxr import UsdUtils

        self.controller = None
        if getattr(self, "cache_id", None) is not None:
            UsdUtils.StageCache.Get().Erase(self.cache_id)
            self.cache_id = None


@dataclass
class _Actor:
    root: object
    animation: object
    translation: object
    orientation: object
    rig: _Rig
    position: tuple[float, ...]
    yaw: float
    velocity: tuple[float, ...] = (0.0, 0.0, 0.0)
    distance: float = 0.0
    idle_time: float = 0.0
    walk_position: float = 0.0
    weight: float = 0.0


class Ros2Actors:
    """Own referenced visuals, independent skeleton animation and kinematic bodies."""

    def __init__(
        self,
        stage: Usd.Stage,
        plans: tuple[PedestrianPlan, ...],
        asset_sources: dict[str, Path],
        motion_library: Path,
        root_prim_path: str,
        frame: StageFrame,
        seed: int,
        radius_m: float = 0.25,
        height_m: float = 1.75,
    ) -> None:
        """Load actors at plan positions [m]; body radius/total height use [m].

        Initial plan yaw [deg] rotates the asset's authored facing. ROS state yaw
        [rad] instead describes the body's actual +X-forward heading.
        """
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdSkel

        if not plans or len({p.name for p in plans}) != len(plans) or set(asset_sources) != {p.name for p in plans}:
            raise ValueError("Actor plans and asset sources require identical unique names.")
        if (
            not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), frame.meters_per_unit, rel_tol=1e-9)
            or str(UsdGeom.GetStageUpAxis(stage)) != frame.up_axis
        ):
            raise ValueError("Actor frame metrics must match the actual USD stage.")
        if not all(math.isfinite(v) and v > 0.0 for v in (radius_m, height_m)) or height_m < 2 * radius_m:
            raise ValueError("Body dimensions must be finite, positive, and height >= diameter [m].")
        path = Sdf.Path(root_prim_path)
        if not path.IsAbsoluteRootOrPrimPath() or path == Sdf.Path.absoluteRootPath or stage.GetPrimAtPath(path):
            raise ValueError("Actor root must be an unused absolute USD prim path.")
        self.stage, self.frame, self.root_path = stage, frame, root_prim_path
        self.radius_m, self.height_m, self.closed = radius_m, height_m, False
        self._actors, self._rigs = {}, {}
        self._layer = Sdf.Layer.CreateAnonymous("ros2-actors.usda")
        self._session = stage.GetSessionLayer()
        self._session.subLayerPaths.insert(0, self._layer.identifier)
        generator = random.Random(seed)
        try:
            motion = _Motion(Path(motion_library).resolve())
            with Usd.EditContext(stage, self._layer):
                UsdGeom.Xform.Define(stage, path).SetResetXformStack(True)
                for plan in sorted(plans, key=lambda p: p.name):
                    position = _finite_vector(plan.position, 3, f"{plan.name}.position")
                    asset = Path(asset_sources[plan.name]).resolve()
                    rig = self._rigs.get(asset)
                    if rig is None:
                        rig = self._rigs[asset] = _Rig(asset, motion)
                    actor_path = path.AppendChild(plan.name)
                    root = UsdGeom.Xform.Define(stage, actor_path)
                    translation = root.AddTranslateOp()
                    orientation = root.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
                    ros_position = (
                        tuple(frame.world_to_ros(position))
                        if hasattr(frame, "world_to_ros")
                        else tuple(frame.stage_to_ros(tuple(v / frame.meters_per_unit for v in position)))
                    )
                    yaw = math.radians(plan.yaw) + rig.authored_yaw - getattr(frame, "yaw_rad", 0.0)
                    visual = UsdGeom.Xform.Define(stage, actor_path.AppendChild("Visual"))
                    visual.GetPrim().GetReferences().AddReference(str(asset))
                    visual.AddTransformOp(opSuffix="rosFacing").Set(rig.alignment)
                    visual.AddScaleOp(opSuffix="assetUnits").Set(Gf.Vec3d(*([rig.units / frame.meters_per_unit] * 3)))
                    for prim in Usd.PrimRange(visual.GetPrim()):
                        schemas = str(prim.GetMetadata("apiSchemas"))
                        if "BehaviorAgent" in schemas or "Scripting" in schemas or prim.GetTypeName() == "OmniGraph":
                            raise ValueError(f"Actor asset contains a runtime controller at {prim.GetPath()}.")
                    skeleton = next(
                        UsdSkel.Skeleton(p) for p in Usd.PrimRange(visual.GetPrim()) if p.IsA(UsdSkel.Skeleton)
                    )
                    animation = UsdSkel.Animation.Define(stage, actor_path.AppendChild("Pose"))
                    animation.CreateJointsAttr(rig.joints)
                    UsdSkel.BindingAPI(skeleton).CreateAnimationSourceRel().SetTargets([animation.GetPath()])
                    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateKinematicEnabledAttr(True)
                    body = UsdGeom.Capsule.Define(stage, actor_path.AppendChild("BodyCollider"))
                    body.CreateAxisAttr("Z")
                    body.CreateRadiusAttr(radius_m / frame.meters_per_unit)
                    body.CreateHeightAttr((height_m - 2 * radius_m) / frame.meters_per_unit)
                    body.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, height_m / (2 * frame.meters_per_unit)))
                    body.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
                    UsdPhysics.CollisionAPI.Apply(body.GetPrim())
                    actor = _Actor(
                        root,
                        animation,
                        translation,
                        orientation,
                        rig,
                        ros_position,
                        yaw,
                        idle_time=generator.random() * rig.idle_times[-1],
                        walk_position=generator.random() * rig.arc[-1],
                    )
                    self._actors[plan.name] = actor
                    self._write(actor, ros_position, yaw, self._pose(actor, 0.0, 0.0, 0.0)[0])
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:
                error.add_note(f"Actor cleanup also failed: {cleanup}")
            raise

    @staticmethod
    def _pose(actor, travel, speed, dt):
        rig = actor.rig
        distance = actor.distance + travel
        idle_time = actor.idle_time + dt
        alpha = 1.0 - math.exp(-dt / 0.12)
        # Shorten the steps at low speed and traverse more of the clip per
        # meter, so the reduced leg excursion still follows the root travel.
        stride = min(1.0, max(0.25, speed / 0.5))
        target_weight = stride * min(1.0, speed / 0.03)
        weight = actor.weight + alpha * (target_weight - actor.weight)
        walk_position = (actor.walk_position + travel / stride) % rig.arc[-1]
        walk = _sample(rig.walk, rig.arc, walk_position)
        idle = _sample(rig.idle, rig.idle_times, idle_time % rig.idle_times[-1])
        return _mix(idle, walk, weight), distance, idle_time, weight, walk_position

    def _write(self, actor, position, yaw, pose):
        from pxr import Gf, Vt

        quaternion = self.frame.ros_yaw_to_stage_quaternion(yaw)
        actor.translation.Set(Gf.Vec3d(*self.frame.ros_to_stage(position)))
        actor.orientation.Set(Gf.Quatd(quaternion[3], *quaternion[:3]))
        actor.animation.CreateTranslationsAttr().Set(Vt.Vec3fArray(pose[0]))
        actor.animation.CreateRotationsAttr().Set(Vt.QuatfArray(pose[1]))
        actor.animation.CreateScalesAttr().Set(Vt.Vec3hArray([Gf.Vec3h(1.0)] * len(actor.rig.joints)))

    def states(self) -> list[dict[str, object]]:
        """Read actor-root ROS state [m, rad, m/s], rejecting external root-basis changes."""
        from pxr import Gf, UsdGeom

        if self.closed:
            raise RuntimeError("Actor resources are closed.")
        cache, result = UsdGeom.XformCache(), []
        for name, actor in self._actors.items():
            matrix = cache.GetLocalToWorldTransform(actor.root.GetPrim())
            origin = matrix.ExtractTranslation()
            point = tuple(self.frame.stage_to_ros(origin))
            tip = self.frame.stage_to_ros(origin + matrix.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0)))
            yaw = math.atan2(tip[1] - point[1], tip[0] - point[0])
            position_error = math.dist(point, actor.position)
            yaw_error = math.remainder(yaw - actor.yaw, 2 * math.pi)
            quaternion = self.frame.ros_yaw_to_stage_quaternion(actor.yaw)
            expected_basis = Gf.Matrix3d(1.0).SetRotate(Gf.Quatd(quaternion[3], *quaternion[:3]))
            basis_error = max(
                abs(matrix[row][column] - expected_basis[row][column]) for row in range(3) for column in range(3)
            )
            if position_error > 1e-5 or abs(yaw_error) > 1e-5 or basis_error > 1e-5:
                raise RuntimeError(
                    f"Another transform writer displaced ROS actor {name}. "
                    f"[expected_position_m={actor.position}, actual_position_m={point}, "
                    f"position_error_m={position_error}, expected_yaw_rad={actor.yaw}, "
                    f"actual_yaw_rad={yaw}, yaw_error_rad={yaw_error}, basis_error={basis_error}]"
                )
            result.append(
                dict(
                    name=name,
                    position=list(point),
                    yaw=yaw,
                    velocity=list(actor.velocity),
                    radius_m=self.radius_m,
                    height_m=self.height_m,
                )
            )
        return result

    def apply(self, commands: list[dict[str, object]], dt: float) -> None:
        """Validate a complete pose batch, then write roots and distance-driven animation.

        Positions use ROS [m], yaw [rad], velocities [m/s], and dt [s]. Invalid
        batches leave all roots and animation phases untouched. Position height
        must remain on each actor's prepared ROS plane.
        """
        from pxr import Usd

        if type(dt) not in (int, float) or not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("Actor update dt must be finite and positive [s].")
        self.states()
        if not isinstance(commands, list) or len(commands) != len(self._actors):
            raise ValueError("Every actor must receive exactly one command in each batch.")
        prepared, seen = [], set()
        for command in commands:
            if (
                not isinstance(command, dict)
                or not isinstance(command.get("name"), str)
                or command.get("name") not in self._actors
                or command["name"] in seen
            ):
                raise ValueError("Command names must match the actor population exactly once.")
            name = command["name"]
            seen.add(name)
            position = _finite_vector(command.get("position"), 3, f"{name}.position")
            velocity = _finite_vector(command.get("velocity"), 3, f"{name}.velocity")
            yaw = _finite_vector((command.get("yaw"),), 1, f"{name}.yaw")[0]
            if abs(velocity[2]) > 1e-9:
                raise ValueError("ROS planar actor velocity must have zero vertical component [m/s].")
            actor = self._actors[name]
            if abs(position[2] - actor.position[2]) > 1e-9:
                raise ValueError("ROS planar actor position must retain its prepared height [m].")
            _finite_vector(self.frame.ros_to_stage(position), 3, "stage position")
            _finite_vector(self.frame.ros_yaw_to_stage_quaternion(yaw), 4, "stage orientation")
            travel = math.dist(position[:2], actor.position[:2])
            actual_velocity = tuple((position[i] - actor.position[i]) / dt if i < 2 else 0.0 for i in range(3))
            pose = self._pose(actor, travel, travel / dt, dt)
            prepared.append((actor, position, yaw, actual_velocity, pose))
        with Usd.EditContext(self.stage, self._layer):
            for actor, position, yaw, velocity, (pose, distance, idle_time, weight, walk_position) in prepared:
                self._write(actor, position, yaw, pose)
                actor.position, actor.yaw, actor.velocity = position, yaw, velocity
                actor.distance, actor.idle_time, actor.weight = distance, idle_time, weight
                actor.walk_position = walk_position

    def status(self) -> dict[str, object]:
        """Report animation phase, traveled distance [m] and root ownership checks."""
        states = self.states()
        return {
            "backend": "independent_usd_skel_retarget",
            "root_prim_path": self.root_path,
            "root_pose_checks_passed": len(states) == len(self._actors),
            "actors": {
                name: {
                    "distance_m": a.distance,
                    "walk_phase": a.walk_position / a.rig.arc[-1],
                    "walk_weight": a.weight,
                    "target_joints": len(a.rig.joints),
                }
                for name, a in self._actors.items()
            },
        }

    def close(self) -> None:
        """Remove only owned layer opinions and release cached retarget resources."""
        if self.closed:
            return
        errors = []
        try:
            # Detaching a Kit stage can expire its anonymous Sdf handles.
            if self._session and self._layer and self._layer.identifier in self._session.subLayerPaths:
                self._session.subLayerPaths.remove(self._layer.identifier)
        except BaseException as error:
            errors.append(error)
        for rig in self._rigs.values():
            try:
                rig.close()
            except BaseException as error:
                errors.append(error)
        if errors:
            for error in errors[1:]:
                errors[0].add_note(f"Additional actor cleanup failed: {error}")
            raise errors[0]
        self.closed = True
        self._actors.clear()
        self._rigs.clear()
