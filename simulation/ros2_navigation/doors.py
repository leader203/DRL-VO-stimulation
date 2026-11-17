# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drive metadata-defined environment doors from ROS observations, without agent SDKs."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from itertools import product
from typing import TYPE_CHECKING

from .frames import StageFrame

if TYPE_CHECKING:
    from pxr import Sdf, Usd


def _finite(value: object, name: str, *, minimum: float | None = None) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"{name} must be finite" + (f" and >= {minimum}." if minimum is not None else "."))
    return float(value)


def _vector(value: Sequence[float], length: int, name: str) -> tuple[float, ...]:
    if value is None or len(value) != length:
        raise ValueError(f"{name} requires {length} finite coordinates.")
    return tuple(_finite(component, name) for component in value)


def _hull_xy(points) -> list[list[float]]:
    vertices = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(vertices) < 3:
        return []

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    low, high = [], []
    for point in vertices:
        while len(low) > 1 and cross(low[-2], low[-1], point) <= 0:
            low.pop()
        low.append(point)
    for point in reversed(vertices):
        while len(high) > 1 and cross(high[-2], high[-1], point) <= 0:
            high.pop()
        high.append(point)
    return [list(point) for point in (*low[:-1], *high[:-1])]


def _sweep_outer(points, pivot, minimum: float, maximum: float) -> list[list[float]]:
    """Intersect continuous-rotation supporting halfplanes; never use an inscribed angle-sampling hull."""
    if maximum - minimum >= 2 * math.pi - 1e-10:
        minimum, maximum = -math.pi, math.pi
    normals, support = [], []
    for index in range(128):
        direction = index * 2 * math.pi / 128
        nx, ny = math.cos(direction), math.sin(direction)
        best = -math.inf
        for point in points:
            x, y = point[0] - pivot[0], point[1] - pivot[1]
            optimum = direction - math.atan2(y, x)
            optimum += math.ceil((minimum - optimum) / (2 * math.pi)) * 2 * math.pi
            angles = (minimum, maximum, optimum) if optimum <= maximum else (minimum, maximum)
            best = max(
                best,
                *(nx * (x * math.cos(a) - y * math.sin(a)) + ny * (x * math.sin(a) + y * math.cos(a)) for a in angles),
            )
        normals.append((nx, ny))
        support.append(best + nx * pivot[0] + ny * pivot[1])
    corners = []
    for index, (nx, ny) in enumerate(normals):
        next_index = (index + 1) % len(normals)
        mx, my = normals[next_index]
        determinant = nx * my - ny * mx
        corners.append(
            (
                (support[index] * my - ny * support[next_index]) / determinant,
                (nx * support[next_index] - support[index] * mx) / determinant,
            )
        )
    return _hull_xy(corners)


def _circle_intersects(point, radius: float, polygon) -> bool:
    if not polygon:
        return False
    signs = []
    nearest = math.inf
    for a, b in zip(polygon, (*polygon[1:], polygon[0])):
        dx, dy = b[0] - a[0], b[1] - a[1]
        signs.append(dx * (point[1] - a[1]) - dy * (point[0] - a[0]))
        length = dx * dx + dy * dy
        if length <= 1e-24:
            continue
        t = max(0.0, min(1.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length))
        nearest = min(nearest, math.hypot(point[0] - a[0] - t * dx, point[1] - a[1] - t * dy))
    return min(signs) >= -1e-10 or nearest <= radius + 1e-8


@dataclass
class _Door:
    config: Usd.Prim
    parent: Usd.Prim
    joint: Usd.Prim
    target: float
    values: dict[str, object]
    half_width: float
    swing_depth: float
    vertical_bounds: tuple[float, float]
    hold_until: float = 0.0
    closing_hold_until: float = 0.0
    requested: bool = False
    occupied: bool = False
    protected: bool = False


class Ros2Doors:
    """Control revolute doors declared by ``automaticDoor:*`` metadata.

    The existing schema expresses center/interior coordinates in the controller's
    parent frame, in [m], with local +Y outside and -Y inside. This adapter accepts
    meter-based Z-up stages and upright, orthogonal parent transforms; arbitrary
    Y-up or differently authored metadata schemas are explicitly unsupported.
    Targets use angular Drive degrees, ramped by speed [deg/s] and simulation dt [s].
    Door colliders and pedestrian poses are never modified. An authorized opening
    lease covers the remaining target ramp plus closeDelay; proximity braking does
    not revoke it. Occupied threshold/swing regions inhibit closing only.
    navigation_interlock=True instead requires complete interiorMin/interiorMax
    metadata for each colliding door, external hold/open intents and real PhysX
    readback. Missing door metadata is a no-op; room identity is never guessed.
    exclusiveEntry defaults to True, preserving occupied-room entry exclusion;
    an explicitly public door may set it to False without changing exit safety.
    requireFrontApproach selects front-of-portal entry in the navigation interlock.
    exitApproachDistance optionally sets the interior approach limit [m].
    """

    def __init__(
        self, stage: Usd.Stage, frame: StageFrame, *, navigation_interlock: bool = False, rigid_pose_reader=None
    ) -> None:
        """Discover configured joints and own only an anonymous drive-target layer."""
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics

        if type(navigation_interlock) is not bool:
            raise ValueError("navigation_interlock must be boolean.")
        self._stage, self.frame = stage, frame
        self._rigid_pose_reader = rigid_pose_reader
        self.navigation_interlock = navigation_interlock
        self._geometry: dict[str, dict[str, object]] = {}
        self._intents: dict[str, dict[str, bool]] = {}
        self._observation_history: dict[str, tuple[float, float]] = {}
        self._observation_cache: dict[str, dict[str, object]] = {}
        self._navigation_status: dict[str, dict[str, object]] = {}
        self._doors: list[_Door] = []
        self._layer: Sdf.Layer | None = None
        self._last_time: float | None = None
        self._previous: dict[str, tuple[float, tuple[float, ...]]] = {}
        self.closed = False
        configurations = [prim for prim in stage.Traverse() if prim.GetRelationship("automaticDoor:joint")]
        if not configurations:
            return
        if (
            frame.up_axis != "Z"
            or str(UsdGeom.GetStageUpAxis(stage)) != "Z"
            or not math.isclose(frame.meters_per_unit, 1.0, rel_tol=1e-9)
            or not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0, rel_tol=1e-9)
        ):
            raise ValueError("automaticDoor metadata currently requires a meter-based Z-up stage.")
        cache = UsdGeom.XformCache()
        for config in configurations:
            targets = config.GetRelationship("automaticDoor:joint").GetTargets()
            if len(targets) != 1:
                raise ValueError(f"automaticDoor:joint requires exactly one revolute joint: {config.GetPath()}")
            joint = UsdPhysics.RevoluteJoint.Get(stage, targets[0])
            if not joint or joint.GetAxisAttr().Get() != "Z" or joint.GetJointEnabledAttr().Get() is False:
                raise ValueError(f"Configured door requires an enabled Z-axis revolute joint: {targets[0]}")
            if not joint.GetPrim().HasAPI(UsdPhysics.DriveAPI, "angular"):
                raise ValueError(f"Configured door requires an existing angular Drive: {targets[0]}")
            bodies = joint.GetBody1Rel().GetTargets()
            body = stage.GetPrimAtPath(bodies[0]) if len(bodies) == 1 else None
            if not body:
                raise ValueError(f"Configured door requires a valid body1: {targets[0]}")
            controller = config.GetParent()
            parent = controller.GetParent()
            values = self._settings(config, controller)
            matrix, inverse, scales = self._parent_frame(parent, cache)
            bounds = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"], False, True
            ).ComputeWorldBound(body)
            extent = bounds.GetRange()
            if extent.IsEmpty():
                raise ValueError(f"Configured door body has no geometry bounds: {body.GetPath()}")
            corners = [
                inverse.Transform(bounds.GetMatrix().Transform(point))
                for point in product(*zip(extent.GetMin(), extent.GetMax()))
            ]
            low = [min(point[i] for point in corners) * scales[i] for i in range(3)]
            high = [max(point[i] for point in corners) * scales[i] for i in range(3)]
            if not all(math.isfinite(value) for value in (*low, *high)) or high[2] <= low[2]:
                raise ValueError(f"Configured door body has invalid geometry bounds: {body.GetPath()}")
            half_width = max(float(values["activationHalfWidth"]), (high[0] - low[0]) / 2)
            swing_depth = math.hypot(high[0] - low[0], high[1] - low[1])
            target = _finite(UsdPhysics.DriveAPI(joint, "angular").GetTargetPositionAttr().Get(), "drive target [deg]")
            self._doors.append(
                _Door(config, parent, joint.GetPrim(), target, values, half_width, swing_depth, (low[2], high[2]))
            )
        if navigation_interlock:
            self.descriptors()
        self._layer = Sdf.Layer.CreateAnonymous("ros2_environment_doors.usda")
        stage.GetSessionLayer().subLayerPaths.insert(0, self._layer.identifier)

    @staticmethod
    def _settings(config: Usd.Prim, controller: Usd.Prim) -> dict[str, object]:
        def value(name: str, default=None):
            result = config.GetAttribute("automaticDoor:" + name).Get()
            if result is None:
                result = controller.GetAttribute("automaticDoor:" + name).Get()
            return default if result is None else result

        result = {
            name: _finite(value(name, default), name, minimum=0.0)
            for name, default in (
                ("approachDistance", 0.55),
                ("activationHalfWidth", 0.32),
                ("closeDelay", 0.25),
                ("speed", 180.0),
                ("thresholdMargin", 0.04),
                ("doorwayDepth", 0.05),
                ("openTolerance", 1.0),
                ("settleSpeed", 5.0),
            )
        }
        if min(result["approachDistance"], result["activationHalfWidth"], result["speed"]) <= 0:
            raise ValueError("Door approach distance, half width and speed must be positive.")
        result["exitApproachDistance"] = _finite(
            value("exitApproachDistance", result["approachDistance"]), "automaticDoor:exitApproachDistance [m]"
        )
        if result["exitApproachDistance"] <= 0:
            raise ValueError("automaticDoor:exitApproachDistance must be positive [m].")
        result["requireFrontApproach"] = value("requireFrontApproach", False)
        if type(result["requireFrontApproach"]) is not bool:
            raise ValueError("automaticDoor:requireFrontApproach must be boolean.")
        result["openAngle"] = _finite(value("openAngle", 88.0), "openAngle [deg]")
        result["closedAngle"] = _finite(value("closedAngle", 0.0), "closedAngle [deg]")
        result["facingHalfAngle"] = _finite(value("facingHalfAngle", 45.0), "facingHalfAngle [deg]", minimum=0)
        if result["facingHalfAngle"] > 180 or result["openAngle"] == result["closedAngle"]:
            raise ValueError("Door facing angle must be <= 180 degrees and open/closed angles must differ.")
        result["center"] = _vector(value("center"), 3, "automaticDoor:center [m]")
        lo, hi = value("interiorMin"), value("interiorMax")
        if (lo is None) != (hi is None):
            raise ValueError("Provide both automaticDoor interiorMin/interiorMax or neither.")
        result["interiorMin"] = _vector(lo, 2, "interiorMin [m]") if lo is not None else None
        result["interiorMax"] = _vector(hi, 2, "interiorMax [m]") if hi is not None else None
        if lo is not None and any(a >= b for a, b in zip(result["interiorMin"], result["interiorMax"])):
            raise ValueError("Door interior bounds must have positive area.")
        result["enabled"] = value("enabled", True)
        if type(result["enabled"]) is not bool:
            raise ValueError("automaticDoor:enabled must be boolean.")
        result["exclusiveEntry"] = value("exclusiveEntry", True)
        if type(result["exclusiveEntry"]) is not bool:
            raise ValueError("automaticDoor:exclusiveEntry must be boolean.")
        return result

    @staticmethod
    def _parent_frame(parent: Usd.Prim, cache):
        from pxr import Gf

        matrix = cache.GetLocalToWorldTransform(parent)
        axes = [matrix.TransformDir(Gf.Vec3d(*(1.0 if i == axis else 0.0 for i in range(3)))) for axis in range(3)]
        scales = tuple(axis.GetLength() for axis in axes)
        if not all(math.isfinite(value) and value > 1e-12 for value in scales):
            raise ValueError("Door metadata parent has a singular/nonfinite transform.")
        axes = [axis / scale for axis, scale in zip(axes, scales)]
        if (
            any(abs(Gf.Dot(axes[i], axes[j])) > 1e-6 for i in range(3) for j in range(i))
            or Gf.Dot(Gf.Cross(axes[0], axes[1]), axes[2]) < 1 - 1e-6
            or Gf.Dot(axes[2], Gf.Vec3d(0, 0, 1)) < 1 - 1e-6
        ):
            raise ValueError("Door metadata requires an upright right-handed parent without shear.")
        return matrix, matrix.GetInverse(), scales

    def _geometry_for(self, door: _Door) -> dict[str, object]:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        name = str(door.config.GetPath())
        if name in self._geometry:
            return self._geometry[name]
        joint = UsdPhysics.RevoluteJoint(door.joint)
        body1 = self._stage.GetPrimAtPath(joint.GetBody1Rel().GetForwardedTargets()[0])
        body0_paths = joint.GetBody0Rel().GetForwardedTargets()
        if len(body0_paths) > 1:
            raise ValueError(f"Door body0 requires zero or one target: {door.joint.GetPath()}")
        body0 = self._stage.GetPrimAtPath(body0_paths[0]) if body0_paths else None
        if body0_paths and not body0:
            raise ValueError(f"Door body0 target is missing: {body0_paths[0]}")
        m0 = UsdGeom.Xformable(body0).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) if body0 else Gf.Matrix4d(1)
        m1 = UsdGeom.Xformable(body1).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t0, t1 = Gf.Transform(m0), Gf.Transform(m1)
        rotation0, rotation1 = t0.GetRotation(), t1.GetRotation()
        local_rotation0 = Gf.Rotation(joint.GetLocalRot0Attr().Get())
        local_rotation1 = Gf.Rotation(joint.GetLocalRot1Attr().Get())
        offset0 = rotation0.GetInverse().TransformDir(
            m0.Transform(Gf.Vec3d(*joint.GetLocalPos0Attr().Get())) - t0.GetTranslation()
        )
        offset1 = rotation1.GetInverse().TransformDir(m1.TransformDir(Gf.Vec3d(*joint.GetLocalPos1Attr().Get())))
        pivot = t0.GetTranslation() + rotation0.TransformDir(offset0)
        joint_rotation = Gf.Rotation(rotation0.GetQuat() * local_rotation0.GetQuat())
        zero_rotation = Gf.Rotation(joint_rotation.GetQuat() * local_rotation1.GetInverse().GetQuat())
        zero_position = pivot - zero_rotation.TransformDir(offset1)
        axis = Gf.Vec3d(0, 0, 1)
        world_axis = joint_rotation.TransformDir(axis)
        if abs(world_axis[2]) < 1 - 1e-6:
            raise ValueError(f"Navigation door envelope requires a vertical hinge: {door.joint.GetPath()}")
        points, collider_paths = [], []
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"], False, True)
        for prim in Usd.PrimRange(body1, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
                continue
            if any(str(schema).split(":")[0] == "PhysxTriggerAPI" for schema in prim.GetAppliedSchemas()):
                continue
            owner = prim
            while owner and owner != body1 and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
                owner = owner.GetParent()
            if owner != body1:
                continue
            bounds = cache.ComputeWorldBound(prim)
            extent = bounds.GetRange()
            if extent.IsEmpty():
                raise ValueError(f"Door physical collider has no bounds: {prim.GetPath()}")
            for corner in product(*zip(extent.GetMin(), extent.GetMax())):
                world = bounds.GetMatrix().Transform(Gf.Vec3d(*corner))
                physical_local = rotation1.GetInverse().TransformDir(world - t1.GetTranslation())
                points.append(self.frame.stage_to_ros(zero_position + zero_rotation.TransformDir(physical_local)))
            collider_paths.append(str(prim.GetPath()))
        closed, opened = float(door.values["closedAngle"]), float(door.values["openAngle"])
        low, high = float(joint.GetLowerLimitAttr().Get()), float(joint.GetUpperLimitAttr().Get())
        if low > high or min(closed, opened) < low or max(closed, opened) > high:
            raise ValueError(f"Door targets must lie inside the revolute limits: {door.joint.GetPath()}")
        minimum, maximum = (
            (low, high) if math.isfinite(low) and math.isfinite(high) and high - low < 360 else (-180, 180)
        )
        guard = float(door.values["openTolerance"])
        minimum, maximum = minimum - guard, maximum + guard
        sign = 1 if world_axis[2] >= 0 else -1
        angles = sorted((sign * math.radians(minimum), sign * math.radians(maximum)))
        pivot_ros = self.frame.stage_to_ros(pivot)
        parent, _, scales = self._parent_frame(door.parent, UsdGeom.XformCache())
        center = door.values["center"]
        half_width = float(door.values["activationHalfWidth"]) / scales[0]
        doorway = [
            list(
                self.frame.stage_to_ros(parent.Transform(Gf.Vec3d(center[0] + sign * half_width, center[1], center[2])))
            )[:2]
            for sign in (-1, 1)
        ]
        interior = []
        if door.values["interiorMin"] is not None:
            lo, hi = door.values["interiorMin"], door.values["interiorMax"]
            interior = _hull_xy(
                self.frame.stage_to_ros(parent.Transform(Gf.Vec3d(x, y, door.values["center"][2])))
                for x in (lo[0], hi[0])
                for y in (lo[1], hi[1])
            )
        if self.navigation_interlock and points and door.values["enabled"] and not interior:
            raise ValueError(f"Door navigation interlock requires interiorMin/interiorMax metadata: {name}")
        descriptor = {
            "name": name,
            "body_prim_path": str(body1.GetPath()),
            "joint_prim_path": str(door.joint.GetPath()),
            "interior_polygon_xy_m": interior,
            "doorway_segment_xy_m": doorway,
            "approach_distance_m": door.values["approachDistance"],
            "exit_approach_distance_m": door.values["exitApproachDistance"],
            "require_front_approach": door.values["requireFrontApproach"],
            "facing_half_angle_deg": door.values["facingHalfAngle"],
            "exclusive_entry": door.values["exclusiveEntry"],
            "swept_footprint_xy_m": _sweep_outer(points, pivot_ros, *angles) if points else [],
            "closed_footprint_xy_m": (
                _sweep_outer(
                    points,
                    pivot_ros,
                    *sorted((sign * math.radians(closed - guard), sign * math.radians(closed + guard))),
                )
                if points
                else []
            ),
            "enabled": bool(door.values["enabled"] and points),
            "physical_obstacle": bool(points),
            "z_min_m": min(point[2] for point in points) if points else None,
            "z_max_m": max(point[2] for point in points) if points else None,
            "hinge_position_m": list(pivot_ros),
            "open_angle_deg": opened,
            "closed_angle_deg": closed,
            "angle_interval_deg": [minimum, maximum],
            "speed_degps": door.values["speed"],
            "target_ramp_duration_s": abs(opened - closed) / door.values["speed"],
            "opening_duration_s": abs(opened - closed) / door.values["speed"],
            "open_tolerance_deg": guard,
            "settle_speed_degps": door.values["settleSpeed"],
            "sweep_support_halfplanes": 128,
            "sweep_angle_guard_deg": guard,
            "collider_prim_paths": collider_paths,
        }
        geometry = {
            "descriptor": descriptor,
            "body0": str(body0.GetPath()) if body0 else None,
            "offset0": offset0,
            "offset1": offset1,
            "local_rotation0": local_rotation0,
            "local_rotation1": local_rotation1,
            "pivot": pivot,
            "rotation0": joint_rotation,
            "axis": axis,
        }
        self._geometry[name] = geometry
        return geometry

    def descriptors(self) -> list[dict[str, object]]:
        """Return JSON-ready ROS geometry [m] covering every allowed rigid hinge rotation [deg].

        ``doorway_segment_xy_m`` spans the metadata activation half width in
        physical meters, irrespective of parent scale. ``approach_distance_m``
        bounds initial center-to-segment triggering. When ``require_front_approach``
        is enabled, ``exit_approach_distance_m`` sets the interior approach limit.
        ``facing_half_angle_deg`` bounds the measured heading error to the
        portal midpoint at admission and before the initial joint motion request.
        The measured forward ray must also intersect the finite portal.
        Analytic angular support maxima give an outer envelope, not an inscribed
        angle-sampling hull. opening_duration_s is nominal target-ramp time; it
        does not bound PhysX settling or contact-delayed movement.
        closed_footprint_xy_m includes the closed endpoint's angular tolerance;
        it certifies a return pose, not safe passage through a moving leaf.
        """
        if self.closed:
            raise RuntimeError("Door adapter is closed.")
        return [copy.deepcopy(self._geometry_for(door)["descriptor"]) for door in self._doors]

    def set_navigation_intents(self, intents: list[dict[str, object]]) -> None:
        """Replace this tick's intents; omitted doors hold, and hold wins over request_open."""
        if self.closed:
            raise RuntimeError("Door adapter is closed.")
        if not self.navigation_interlock and self._doors:
            raise RuntimeError("Navigation intents require navigation_interlock=True.")
        known = {str(door.config.GetPath()) for door in self._doors}
        validated = {}
        for intent in intents:
            name = intent.get("name")
            if name not in known or name in validated:
                raise ValueError("Door intents require unique configured names.")
            if any(type(intent.get(key)) is not bool for key in ("hold", "request_open")):
                raise ValueError("Door hold/request_open intents must be booleans.")
            validated[name] = {"hold": intent["hold"], "request_open": intent["request_open"]}
        self._intents = validated

    def observations(self, time_s: float) -> list[dict[str, object]]:
        """Read real PhysX joint angles [deg] and measured angular rates [deg/s].

        fully_open requires angular proximity and two time-separated physical
        observations showing low speed. First/invalid samples never imply open.
        Same-tick repeated reads preserve the previous measured rate only when
        the real angle is unchanged; they cannot reset a moving door to zero speed.
        """
        from pxr import Gf, UsdGeom, UsdPhysics

        if self.closed:
            raise RuntimeError("Door adapter is closed.")
        now = _finite(time_s, "time_s [s]", minimum=0)
        result = []

        def has_rigid_body(path):
            prim = self._stage.GetPrimAtPath(path)
            return prim.HasAPI(UsdPhysics.RigidBodyAPI) and (
                UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is not False
            )

        entries = [(door, self._geometry_for(door)) for door in self._doors]
        poses = None
        if self._rigid_pose_reader is not None:
            paths = set()
            for _, geometry in entries:
                if not geometry["descriptor"]["enabled"]:
                    continue
                paths.add(geometry["descriptor"]["body_prim_path"])
                if geometry["body0"] and has_rigid_body(geometry["body0"]):
                    paths.add(geometry["body0"])
            poses = self._rigid_pose_reader(tuple(sorted(paths)))

        def pose(path, *, require_physics=False):
            prim = self._stage.GetPrimAtPath(path)
            if require_physics or has_rigid_body(path):
                state = (
                    import_module("omni.physx").get_physx_interface().get_rigidbody_transformation(path)
                    if poses is None
                    else poses[path]
                )
                if not state.get("ret_val"):
                    raise RuntimeError(f"No live PhysX pose for door body: {path}")
                position = Gf.Vec3d(*_vector(state.get("position"), 3, "physical door position"))
                q = _vector(state.get("rotation"), 4, "physical door quaternion XYZW")
                norm = math.sqrt(sum(value * value for value in q))
                if norm < 1e-12:
                    raise RuntimeError("Physical door quaternion is degenerate.")
                return position, Gf.Rotation(Gf.Quatd(q[3] / norm, Gf.Vec3d(*q[:3]) / norm))
            transform = Gf.Transform(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            return transform.GetTranslation(), transform.GetRotation()

        for door, geometry in entries:
            name = str(door.config.GetPath())
            descriptor = geometry["descriptor"]
            observation = {
                "name": name,
                "time_s": now,
                "enabled": descriptor["enabled"],
                "valid": False,
                "fully_open": False,
                "fully_closed": False,
                "actual_angle_deg": None,
                "angular_speed_degps": None,
            }
            if not descriptor["enabled"]:
                result.append(observation)
                continue
            try:
                p0, r0 = (
                    pose(geometry["body0"]) if geometry["body0"] else (Gf.Vec3d(0), Gf.Rotation(Gf.Vec3d(0, 0, 1), 0.0))
                )
                pivot = p0 + r0.TransformDir(geometry["offset0"])
                joint0 = Gf.Rotation(r0.GetQuat() * geometry["local_rotation0"].GetQuat())
                if (pivot - geometry["pivot"]).GetLength() > 1e-4:
                    raise RuntimeError("Door hinge moved; rebuild its navigation envelope.")
                reference, current = geometry["rotation0"].GetQuat(), joint0.GetQuat()
                dot = abs(
                    reference.GetReal() * current.GetReal() + Gf.Dot(reference.GetImaginary(), current.GetImaginary())
                )
                if dot < 1 - 1e-10:
                    raise RuntimeError("Door hinge frame rotated; rebuild its navigation envelope.")
                p1, r1 = pose(descriptor["body_prim_path"], require_physics=True)
                relative = joint0.GetInverse().GetQuat() * r1.GetQuat() * geometry["local_rotation1"].GetQuat()
                component = Gf.Dot(relative.GetImaginary(), geometry["axis"])
                if math.hypot(component, relative.GetReal()) < 1e-10:
                    raise RuntimeError("Physical door has no valid revolute twist rotation.")
                if (relative.GetImaginary() - geometry["axis"] * component).GetLength() > 1e-3:
                    raise RuntimeError("Physical door rotation left its revolute axis.")
                angle = math.degrees(math.remainder(2 * math.atan2(component, relative.GetReal()), 2 * math.pi))
                previous = self._observation_history.get(name)
                if previous and now == previous[0]:
                    if abs(math.remainder(angle - previous[1], 360)) > 1e-5:
                        raise RuntimeError("Physical door angle changed within the same navigation tick.")
                    result.append(copy.deepcopy(self._observation_cache[name]))
                    continue
                speed = (
                    math.remainder(angle - previous[1], 360) / (now - previous[0])
                    if previous and now > previous[0]
                    else None
                )
                error = abs(math.remainder(angle - descriptor["open_angle_deg"], 360))
                closed_error = abs(math.remainder(angle - descriptor["closed_angle_deg"], 360))
                settled = speed is not None and abs(speed) <= descriptor["settle_speed_degps"]
                observation.update(
                    valid=True,
                    actual_angle_deg=angle,
                    angular_speed_degps=speed,
                    fully_open=bool(settled and error <= descriptor["open_tolerance_deg"]),
                    fully_closed=bool(settled and closed_error <= descriptor["open_tolerance_deg"]),
                    settled=settled,
                    angle_error_deg=error,
                    anchor_error_m=(p1 + r1.TransformDir(geometry["offset1"]) - pivot).GetLength(),
                )
                self._observation_history[name] = now, angle
            except (ImportError, AttributeError, RuntimeError, ValueError) as error:
                observation["error"] = str(error)
            self._observation_cache[name] = observation
            result.append(copy.deepcopy(observation))
        return result

    def _update_interlocked(self, bodies, now: float, dt: float) -> None:
        from pxr import Usd, UsdPhysics

        observed = {item["name"]: item for item in self.observations(now)}
        commands = []
        for door in self._doors:
            name = str(door.config.GetPath())
            descriptor = self._geometry_for(door)["descriptor"]
            if not descriptor["enabled"]:
                continue
            state = observed[name]
            if not state["valid"]:
                raise RuntimeError(f"Door interlock needs a valid physical observation: {name}: {state.get('error')}")
            blocked = any(
                position[2] <= descriptor["z_max_m"]
                and position[2] + height >= descriptor["z_min_m"]
                and _circle_intersects(position, radius, descriptor["swept_footprint_xy_m"])
                for _, position, _, _, _, radius, height in bodies
            )
            intent = self._intents.get(name, {"hold": True, "request_open": False})
            actual = state["actual_angle_deg"]
            hold = intent["hold"] or (blocked and not state["fully_open"])
            if blocked:
                door.closing_hold_until = max(door.closing_hold_until, now + door.values["closeDelay"])
            if not intent["request_open"] and (blocked or now < door.closing_hold_until):
                hold = True
            if hold:
                target = actual
            else:
                wanted = door.values["openAngle"] if intent["request_open"] else door.values["closedAngle"]
                limit = door.values["speed"] * dt
                target = door.target + max(-limit, min(limit, wanted - door.target))
            commands.append((door, target, hold, intent, blocked))
        with Usd.EditContext(self._stage, self._layer):
            for door, target, hold, intent, blocked in commands:
                attr = UsdPhysics.DriveAPI(door.joint, "angular").GetTargetPositionAttr()
                if abs(float(attr.Get()) - target) > 1e-9:
                    attr.Set(target)
                    if not math.isclose(float(attr.Get()), target, abs_tol=1e-5):
                        raise RuntimeError(f"Another layer overrides the ROS door Drive: {door.joint.GetPath()}")
                door.target, door.requested = target, intent["request_open"]
                door.protected = blocked and not intent["request_open"]
                self._navigation_status[str(door.config.GetPath())] = {
                    "navigation_hold": hold,
                    "sweep_blocked": blocked,
                }

    def update(self, states: list[dict[str, object]], time_s: float, dt: float) -> None:
        """Read complete ROS body snapshots and ramp door targets by speed [deg/s] * dt [s].

        Each body supplies name, position [m], yaw [rad], radius_m [m], height_m [m]
        and optionally world velocity [m/s]. Missing velocity uses observed position
        history. No body state or planner state is changed.
        """
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        if self.closed:
            raise RuntimeError("Door adapter is closed.")
        now = _finite(time_s, "time_s [s]", minimum=0)
        step = _finite(dt, "dt [s]", minimum=0)
        if not self._doors or step == 0:
            return
        bodies = []
        names = set()
        for state in states:
            name = state.get("name")
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("Door observations require unique nonempty body names.")
            names.add(name)
            position = _vector(state.get("position"), 3, "body position [m]")
            yaw = _finite(state.get("yaw"), "body yaw [rad]")
            radius = _finite(state.get("radius_m"), "body radius [m]", minimum=0)
            height = _finite(state.get("height_m"), "body height [m]", minimum=0)
            old = self._previous.get(name)
            velocity = state.get("velocity")
            if velocity is None:
                velocity = (
                    tuple((p - q) / (now - old[0]) for p, q in zip(position, old[1]))
                    if old and now > old[0]
                    else (0, 0, 0)
                )
            velocity = _vector(velocity, 3, "body velocity [m/s]")
            world = Gf.Vec3d(*self.frame.ros_to_stage(position))
            tip = Gf.Vec3d(
                *self.frame.ros_to_stage((position[0] + math.cos(yaw), position[1] + math.sin(yaw), position[2]))
            )
            future = Gf.Vec3d(*self.frame.ros_to_stage(tuple(p + v for p, v in zip(position, velocity))))
            bodies.append((name, position, world, tip - world, future - world, radius, height))
        if self.navigation_interlock:
            self._update_interlocked(bodies, now, step)
        if self._last_time is not None and now < self._last_time:
            for door in self._doors:
                door.hold_until = now
                door.closing_hold_until = now
        cache = UsdGeom.XformCache(Usd.TimeCode(now * self._stage.GetTimeCodesPerSecond()))
        changes = []
        for door in self._doors:
            values = door.values
            if not values["enabled"]:
                continue
            if self.navigation_interlock and self._geometry_for(door)["descriptor"]["physical_obstacle"]:
                continue
            _, inverse, scales = self._parent_frame(door.parent, cache)
            center = values["center"]
            entry, leaving, doorway, inside, swing = False, False, False, False, False
            for _, _, world, direction, velocity, radius, height in bodies:
                local = inverse.Transform(world)
                if (
                    local[2] * scales[2] > door.vertical_bounds[1]
                    or (local[2] * scales[2] + height) < door.vertical_bounds[0]
                ):
                    continue
                x, y = ((local[i] - center[i]) * scales[i] for i in (0, 1))
                lo, hi = values["interiorMin"], values["interiorMax"]
                inside = inside or (lo is not None and all(a <= p <= b for a, p, b in zip(lo, local, hi)))
                aligned = abs(x) <= values["activationHalfWidth"]
                local_direction = inverse.TransformDir(direction)
                fx, fy = local_direction[0] * scales[0], local_direction[1] * scales[1]
                length = math.hypot(fx, fy)
                toward = (-fy if y > 0 else fy) / length if length > 1e-12 else -1
                facing = toward >= math.cos(math.radians(values["facingHalfAngle"]))
                vy = inverse.TransformDir(velocity)[1] * scales[1]
                entry = entry or (aligned and facing and 0 <= y <= values["approachDistance"])
                leaving = leaving or (aligned and facing and vy > 0 and -values["approachDistance"] <= y < 0)
                doorway = doorway or (
                    abs(x) <= door.half_width + radius
                    and abs(y) <= values["doorwayDepth"] + radius + values["thresholdMargin"]
                )
                swing = swing or (
                    abs(x) <= door.half_width + radius
                    and -door.swing_depth - radius <= y <= radius + values["doorwayDepth"]
                )
            request = doorway or leaving or (entry and (not inside or not values["exclusiveEntry"]))
            opening_time = abs(values["openAngle"] - door.target) / values["speed"]
            until = max(door.hold_until, now + opening_time + values["closeDelay"]) if request else door.hold_until
            closing_until = (
                max(door.closing_hold_until, now + values["closeDelay"]) if swing else door.closing_hold_until
            )
            wanted = values["openAngle"] if now < until or request else values["closedAngle"]
            closing = abs(wanted - values["closedAngle"]) < abs(door.target - values["closedAngle"]) - 1e-9
            protected = closing and (swing or now < closing_until)
            if protected:
                wanted = door.target
            limit = values["speed"] * step
            target = door.target + max(-limit, min(limit, wanted - door.target))
            changes.append((door, target, until, closing_until, request, inside, protected))
        with Usd.EditContext(self._stage, self._layer):
            for door, target, until, closing_until, request, inside, protected in changes:
                attr = UsdPhysics.DriveAPI(door.joint, "angular").GetTargetPositionAttr()
                if abs(target - door.target) > 1e-9:
                    attr.Set(target)
                    if not math.isclose(float(attr.Get()), target, abs_tol=1e-5):
                        raise RuntimeError(f"Another layer overrides the ROS door Drive: {door.joint.GetPath()}")
                door.target, door.hold_until = target, until
                door.closing_hold_until = closing_until
                door.requested, door.occupied, door.protected = request, inside, protected
        self._previous = {name: (now, position) for name, position, *_ in bodies}
        self._last_time = now

    def status(self) -> dict[str, object]:
        """Return door targets [deg], simulation hold deadlines [s] and occupancy decisions."""
        return {
            "closed": self.closed,
            "navigation_interlock": self.navigation_interlock,
            "count": len(self._doors),
            "doors": {
                str(door.config.GetPath()): {
                    "joint": str(door.joint.GetPath()),
                    "target_degrees": door.target,
                    "hold_until_s": max(door.hold_until, door.closing_hold_until),
                    "opening_lease_until_s": door.hold_until,
                    "closing_hold_until_s": door.closing_hold_until,
                    "requested_open": door.requested,
                    "interior_occupied": door.occupied,
                    "closing_protected": door.protected,
                    "actual_angle_deg": self._observation_cache.get(str(door.config.GetPath()), {}).get(
                        "actual_angle_deg"
                    ),
                    "fully_open": self._observation_cache.get(str(door.config.GetPath()), {}).get("fully_open", False),
                    "fully_closed": self._observation_cache.get(str(door.config.GetPath()), {}).get(
                        "fully_closed", False
                    ),
                    **self._navigation_status.get(str(door.config.GetPath()), {}),
                }
                for door in self._doors
            },
        }

    def close(self) -> None:
        """Remove only the adapter-owned layer, revealing original or newer external targets."""
        if self.closed:
            return
        if self._layer is not None and self._stage:
            layers = self._stage.GetSessionLayer().subLayerPaths
            if self._layer.identifier in layers:
                layers.remove(self._layer.identifier)
        self.closed = True
