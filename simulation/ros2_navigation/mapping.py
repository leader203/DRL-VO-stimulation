# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Map real PhysX collision geometry without IRA, navigation meshes or visual projections."""

from __future__ import annotations

import io
import json
import math
from collections import deque
from collections.abc import Mapping, Sequence
from importlib import import_module
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .frames import StageFrame

if TYPE_CHECKING:
    from pxr import Usd


def _number(settings: Mapping[str, object], name: str, default: float, *, positive: bool = True) -> float:
    value = settings.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite [m or deg, depending on field].")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive.")
    return float(value)


def _roots(values: Sequence[str]) -> tuple[str, ...]:
    from pxr import Sdf

    result = []
    for value in values:
        if not isinstance(value, str) or not Sdf.Path(value).IsAbsoluteRootOrPrimPath():
            raise ValueError("Excluded and dynamic roots must be absolute USD prim paths.")
        result.append(value.rstrip("/") or "/")
    return tuple(result)


def _under(path: str, roots: tuple[str, ...]) -> bool:
    return any(root == "/" or path == root or path.startswith(root + "/") for root in roots)


def _dynamic_root(prim: Usd.Prim, roots: tuple[str, ...]) -> str | None:
    from pxr import UsdGeom, UsdPhysics

    explicit = next((root for root in roots if _under(str(prim.GetPath()), (root,))), None)
    varying = None
    while prim and not prim.IsPseudoRoot():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            if UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is not False:
                return str(prim.GetPath())
        xform = UsdGeom.Xformable(prim)
        if varying is None and xform and xform.TransformMightBeTimeVarying():
            varying = str(prim.GetPath())
        prim = prim.GetParent()
    return explicit or varying


def _world_fixed_roots(stage: Usd.Stage, joints: Sequence[Usd.Prim], dynamic: tuple[str, ...]) -> set[str]:
    """Prove world-fixed body components using enabled, unbreakable fixed joints only."""
    from pxr import UsdGeom, UsdPhysics

    def endpoint(paths) -> tuple[bool, str | None]:
        if not paths:
            return True, None
        if len(paths) != 1:
            return False, None
        prim = stage.GetPrimAtPath(paths[0])
        if not prim or not prim.IsActive() or not prim.IsLoaded():
            return False, None
        body, found = None, False
        while prim and not prim.IsPseudoRoot():
            if _under(str(prim.GetPath()), dynamic):
                return False, None
            xform = UsdGeom.Xformable(prim)
            if xform and xform.TransformMightBeTimeVarying():
                return False, None
            if not found and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                found = True
                if UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is not False:
                    body = str(prim.GetPath())
            prim = prim.GetParent()
        return True, body

    edges: dict[str, set[str]] = {}
    anchored: set[str] = set()
    for prim in joints:
        joint = UsdPhysics.FixedJoint(prim)
        if joint.GetJointEnabledAttr().Get() is False:
            continue
        breaking = (joint.GetBreakForceAttr().Get(), joint.GetBreakTorqueAttr().Get())
        if not all(isinstance(value, (int, float)) and value > 0 and math.isinf(value) for value in breaking):
            continue
        if any(
            attribute.GetNumTimeSamples()
            for attribute in (
                joint.GetLocalPos0Attr(),
                joint.GetLocalPos1Attr(),
                joint.GetLocalRot0Attr(),
                joint.GetLocalRot1Attr(),
            )
        ):
            continue
        valid_a, a = endpoint(joint.GetBody0Rel().GetForwardedTargets())
        valid_b, b = endpoint(joint.GetBody1Rel().GetForwardedTargets())
        if not valid_a or not valid_b:
            continue
        if a is not None and b is not None:
            edges.setdefault(a, set()).add(b)
            edges.setdefault(b, set()).add(a)
        elif a is not None or b is not None:
            anchored.add(a if a is not None else b)
    queue = deque(anchored)
    while queue:
        for body in edges.get(queue.popleft(), ()):
            if body not in anchored:
                anchored.add(body)
                queue.append(body)
    return anchored


def _inventory(stage: Usd.Stage, excluded: tuple[str, ...], dynamic: tuple[str, ...]) -> dict[str, str | None]:
    from pxr import Usd, UsdGeom, UsdPhysics

    colliders = {}
    point_instancer_geometry = []
    fixed_joints = []
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdPhysics.FixedJoint):
            fixed_joints.append(prim)
        if prim.IsA(UsdGeom.PointInstancer):
            point_instancer_geometry.append(str(prim.GetPath()))
            point_instancer_geometry.extend(
                str(path) for path in UsdGeom.PointInstancer(prim).GetPrototypesRel().GetForwardedTargets()
            )
        if not prim.IsLoaded() or prim.IsAbstract() or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        path = str(prim.GetPath())
        if _under(path, excluded):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        if any(str(schema).split(":")[0] == "PhysxTriggerAPI" for schema in prim.GetAppliedSchemas()):
            continue
        colliders[path] = _dynamic_root(prim, dynamic)
    point_roots = tuple(point_instancer_geometry)
    for path in colliders:
        if _under(path, point_roots):
            raise RuntimeError(f"PointInstancer collision mapping is not supported: {path}")
    anchored = _world_fixed_roots(stage, fixed_joints, dynamic)
    for path, body in colliders.items():
        if body in anchored and not _under(path, dynamic):
            colliders[path] = None
    return colliders


def _queries(stage: Usd.Stage, frame: StageFrame, physics_scenes: Sequence[Usd.Prim] | None = None):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils

    if not stage or not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), frame.meters_per_unit, rel_tol=1e-9):
        raise ValueError("StageFrame length units must match the loaded USD stage.")
    if str(UsdGeom.GetStageUpAxis(stage)) != frame.up_axis:
        raise ValueError("StageFrame up axis must match the loaded USD stage.")
    prims = stage.Traverse(Usd.TraverseInstanceProxies()) if physics_scenes is None else physics_scenes
    for prim in prims:
        if prim.IsA(UsdPhysics.Scene) and prim.GetAttribute("physxScene:enableSceneQuerySupport").Get() is False:
            raise RuntimeError(f"PhysX scene queries are disabled: {prim.GetPath()}")
    module = import_module("omni.physx")
    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    attached = module.get_physx_simulation_interface().get_attached_stage()
    ready = stage_id > 0 and attached == stage_id
    if stage_id > 0 and attached == 0:
        # Timeline stageupdate owns its scene separately from manual IPhysxSimulation.attach_stage.
        try:
            context = import_module("omni.usd").get_context()
            timeline = import_module("omni.timeline").get_timeline_interface()
            ready = context.get_stage_id() == stage_id and context.get_stage() == stage and not timeline.is_stopped()
        except (ImportError, AttributeError):
            ready = False
    if not ready:
        raise RuntimeError("PhysX scene queries are not ready for this stage; initialize physics before mapping.")
    return module.get_physx_scene_query_interface(), stage_id


def _time_code(stage: Usd.Stage):
    from pxr import Usd

    try:
        timeline = import_module("omni.timeline").get_timeline_interface()
    except ImportError:
        return Usd.TimeCode.Default()
    return Usd.TimeCode(timeline.get_current_time() * stage.GetTimeCodesPerSecond())


def _world_corners(cache, prim: Usd.Prim):
    from pxr import Gf

    bound = cache.ComputeWorldBound(prim)
    bounds, matrix = bound.GetRange(), bound.GetMatrix()
    if bounds.IsEmpty():
        raise ValueError(f"Collision geometry has no finite bounds: {prim.GetPath()}")
    low, high = bounds.GetMin(), bounds.GetMax()
    if not all(math.isfinite(float(value)) for value in (*low, *high)) or not all(
        math.isfinite(float(value)) for row in matrix for value in row
    ):
        raise ValueError(f"Collision geometry has nonfinite bounds: {prim.GetPath()}")
    # Preserve the bound's orientation before applying the live rigid-pose delta.
    return [matrix.Transform(Gf.Vec3d(*point)) for point in product(*zip(low, high))]


def _plan_points(settings: Mapping[str, object], frame: StageFrame) -> list[tuple[float, float, float]]:
    points = list(settings.get("plan_points_m", ()))
    for plan in settings.get("plans", ()):
        position = plan.get("position") if isinstance(plan, Mapping) else plan.position
        if position is not None:
            points.append(position)
        waypoints = plan.get("waypoints", ()) if isinstance(plan, Mapping) else plan.waypoints
        for waypoint in waypoints:
            points.append(waypoint.get("position") if isinstance(waypoint, Mapping) else waypoint.position)
    return [frame.world_to_ros(point) for point in points]


def _bounds(stage: Usd.Stage, settings: Mapping[str, object], frame: StageFrame, colliders):
    from pxr import UsdGeom

    supplied = settings.get("bounds_m")
    if supplied is not None:
        if len(supplied) != 4 or any(isinstance(value, bool) for value in supplied):
            raise ValueError("bounds_m must contain [xmin, ymin, xmax, ymax] in ROS map meters.")
        bounds = tuple(float(value) for value in supplied)
    else:
        # Collision proxies may be hidden guide geometry; rendering visibility is not a physics switch.
        cache = UsdGeom.BBoxCache(
            _time_code(stage), ["default", "render", "proxy", "guide"], useExtentsHint=False, ignoreVisibility=True
        )
        points = _plan_points(settings, frame)
        for path, dynamic in colliders.items():
            if dynamic is None:
                points.extend(frame.stage_to_ros(point) for point in _world_corners(cache, stage.GetPrimAtPath(path)))
        if not points:
            raise ValueError("Automatic map bounds require static collision geometry or plan_points_m.")
        margin = _number(settings, "bounds_margin_m", 0.5)
        bounds = (
            min(point[0] for point in points) - margin,
            min(point[1] for point in points) - margin,
            max(point[0] for point in points) + margin,
            max(point[1] for point in points) + margin,
        )
    if not all(math.isfinite(value) for value in bounds) or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError("bounds_m must be finite and have positive width and height.")
    return bounds


def _hit_value(hit, name: str):
    return hit[name] if isinstance(hit, Mapping) else getattr(hit, name)


def _solid_hit(hit, stage: Usd.Stage, colliders, excluded: tuple[str, ...]) -> bool:
    path = str(_hit_value(hit, "collision"))
    if _under(path, excluded):
        return False
    # PhysX protoIndex is not a USD type test; ordinary meshes and instance proxies can carry it too.
    if path in colliders:
        return colliders[path] is None
    # Known disabled colliders/triggers are not solid. Unknown query paths are an error.
    prim = stage.GetPrimAtPath(path)
    if prim and not _inventory_for_prim(prim):
        return False
    raise RuntimeError(f"PhysX reported unclassified collision geometry: {path}")


def _inventory_for_prim(prim: Usd.Prim) -> bool:
    from pxr import UsdPhysics

    return bool(
        prim.HasAPI(UsdPhysics.CollisionAPI)
        and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
        and not any(str(schema).split(":")[0] == "PhysxTriggerAPI" for schema in prim.GetAppliedSchemas())
    )


def build_occupancy_grid(
    stage: Usd.Stage, settings: Mapping[str, object], frame: StageFrame, excluded_roots: Sequence[str] = ()
) -> dict[str, object]:
    """Sample static solid collisions and floor support into a ROS occupancy grid.

    Args:
        stage: PhysX-attached USD stage, frozen for the duration of the query.
        settings: Bounds/floor/resolution [m] in the ROS map frame, body height [m],
            support tolerance [m], slope [deg] and cell budget. ``plan_points_m``
            and ``plans`` use canonical Z-up world coordinates [m] for automatic
            bounds. The session normalizes ``stage_world`` input plans before
            mapping; this API does not interpret raw scene-world input values.
        frame: Explicit stage/world/map coordinate conversion.
        excluded_roots: Actor/robot subtrees excluded from static mapping.

    Returns:
        Int8 ``occupancy[height, width]`` with -1/0/100 unknown/free/occupied;
        row zero is minimum ROS Y. Metadata lengths are [m]. Five support samples
        per cell are a finite-resolution test, not a certificate for subcell holes.

    Raises:
        RuntimeError: Physics is not attached, returned invalid geometry, or no
            floor support was found. No visual-geometry fallback is used.
        ValueError: Units, bounds or mapping settings are invalid.
    """
    resolution = _number(settings, "resolution_m", 0.05)
    floor = _number(settings, "floor_height_m", 0.0, positive=False)
    tolerance = _number(settings, "floor_tolerance_m", 0.12)
    height = _number(settings, "body_height_m", 1.75)
    clearance = _number(settings, "floor_clearance_m", 0.08, positive=False)
    slope = _number(settings, "max_slope_deg", 15.0, positive=False)
    maximum = settings.get("max_cells", 250000)
    if clearance < 0 or height <= clearance or not 0 <= slope < 90:
        raise ValueError("Require body_height_m > floor_clearance_m and max_slope_deg in [0,90).")
    if type(maximum) is not int or maximum < 1:
        raise ValueError("max_cells must be a positive integer.")
    excluded = _roots(excluded_roots)
    colliders = _inventory(stage, excluded, _roots(settings.get("dynamic_roots", ())))
    if not any(value is None for value in colliders.values()):
        raise RuntimeError("Mapping requires enabled static solid collision geometry, including a floor.")
    bounds = _bounds(stage, settings, frame, colliders)
    xmin, ymin = (math.floor(value / resolution) * resolution for value in bounds[:2])
    dimensions = tuple((high - low) / resolution for low, high in zip((xmin, ymin), bounds[2:]))
    if any(not math.isfinite(value) or value > maximum for value in dimensions):
        raise ValueError(f"Map dimensions exceed max_cells={maximum}.")
    width, rows = (math.ceil(value) for value in dimensions)
    if width * rows > maximum:
        raise ValueError(f"Map requires {width * rows} cells, exceeding max_cells={maximum}.")
    queries, stage_id = _queries(stage, frame)
    result = np.full((rows, width), -1, dtype=np.int8)
    inverse_units = 1.0 / frame.meters_per_unit
    rotation = frame.ros_yaw_to_stage_quaternion(0.0)
    half = (resolution / 2 * inverse_units, resolution / 2 * inverse_units, (height - clearance) / 2 * inverse_units)
    zero, below = frame.ros_to_stage((0.0, 0.0, 0.0)), frame.ros_to_stage((0.0, 0.0, -1.0))
    direction = tuple((value - origin) * frame.meters_per_unit for value, origin in zip(below, zero))
    support_samples, supported_cells = 0, 0
    errors = []
    minimum_cosine = math.cos(math.radians(slope))
    offset = resolution * 0.499
    offsets = ((0.0, 0.0), (-offset, -offset), (-offset, offset), (offset, -offset), (offset, offset))

    def supported(x: float, y: float) -> bool:
        nonlocal support_samples
        support_samples += 1
        accepted = False

        def report(hit):
            nonlocal accepted
            try:
                if _solid_hit(hit, stage, colliders, excluded):
                    point = frame.stage_to_ros(_hit_value(hit, "position"))
                    normal = frame.stage_vector_to_ros(_hit_value(hit, "normal"))
                    length = math.sqrt(sum(value * value for value in normal))
                    if length <= 0:
                        raise RuntimeError("PhysX returned a degenerate floor normal.")
                    if abs(point[2] - floor) <= tolerance + 1e-8 and normal[2] / length >= minimum_cosine:
                        accepted = True
            except Exception as error:
                errors.append(error)
                return False
            return True

        queries.raycast_all(
            frame.ros_to_stage((x, y, floor + tolerance)), direction, 2 * tolerance * inverse_units, report, False
        )
        if errors:
            raise RuntimeError(f"Floor support query failed: {errors[-1]}") from errors[-1]
        return accepted

    for row in range(rows):
        y = ymin + (row + 0.5) * resolution
        for column in range(width):
            x = xmin + (column + 0.5) * resolution
            blocked = False

            def report(hit):
                nonlocal blocked
                try:
                    blocked = _solid_hit(hit, stage, colliders, excluded)
                except Exception as error:
                    errors.append(error)
                    return False
                return not blocked

            queries.overlap_box(
                half, frame.ros_to_stage((x, y, floor + (height + clearance) / 2)), rotation, report, False
            )
            if errors:
                raise RuntimeError(f"Static collision query failed: {errors[-1]}") from errors[-1]
            if blocked:
                result[row, column] = 100
            elif all(supported(x + dx, y + dy) for dx, dy in offsets):
                result[row, column] = 0
                supported_cells += 1
    if supported_cells == 0:
        raise RuntimeError(
            "No supported free cells found; verify floor height, collision cooking and PhysX query readiness."
        )
    return {
        "occupancy": result,
        "resolution_m": resolution,
        "origin_xy_m": (xmin, ymin),
        "floor_height_m": floor,
        "diagnostics": {
            "physics_stage_id": stage_id,
            "static_colliders": sum(value is None for value in colliders.values()),
            "dynamic_colliders": sum(value is not None for value in colliders.values()),
            "support_samples": support_samples,
            "supported_cells": supported_cells,
            "occupied_cells": int(np.count_nonzero(result == 100)),
            "unknown_cells": int(np.count_nonzero(result == -1)),
            "meters_per_unit": frame.meters_per_unit,
            "up_axis": frame.up_axis,
            "map_origin_world_m": list(frame.origin_m),
            "map_yaw_rad": frame.yaw_rad,
            "support_samples_per_cell": len(offsets),
        },
    }


def save_occupancy_grid(data: Mapping[str, object], path_stem: Path | str) -> dict[str, Path]:
    """Save NPZ, PGM and Nav2 YAML without overwriting existing outputs.

    Args:
        data: Grid returned by :func:`build_occupancy_grid`, lengths [m].
        path_stem: Output basename without a suffix; parents may be created.

    Returns:
        Paths keyed by ``npz``, ``pgm`` and ``yaml``. PGM rows are flipped so the
        image top is maximum map Y; YAML origin is the cell (0,0) lower-left corner.
    """
    import yaml

    grid = np.asarray(data["occupancy"])
    if grid.ndim != 2 or not grid.size or not np.isin(grid, (-1, 0, 100)).all():
        raise ValueError("occupancy must be a nonempty H,W grid containing only -1, 0 and 100.")
    resolution = _number(data, "resolution_m", 0.05)
    floor = _number(data, "floor_height_m", 0.0, positive=False)
    origin = tuple(float(value) for value in data["origin_xy_m"])
    if len(origin) != 2 or not all(math.isfinite(value) for value in origin):
        raise ValueError("origin_xy_m must contain two finite coordinates [m].")
    stem = Path(path_stem)
    paths = {suffix: Path(str(stem) + "." + suffix) for suffix in ("npz", "pgm", "yaml")}
    if any(path.exists() or path.is_symlink() for path in paths.values()):
        raise FileExistsError(f"Occupancy output already exists for {stem}.")
    image = np.where(grid == 100, 0, np.where(grid == 0, 254, 205)).astype(np.uint8)
    pgm = f"P5\n{grid.shape[1]} {grid.shape[0]}\n255\n".encode() + np.flipud(image).tobytes()
    metadata = {
        "image": paths["pgm"].name,
        "mode": "trinary",
        "resolution": resolution,
        "origin": [*origin, 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
    }
    archive = io.BytesIO()
    np.savez_compressed(
        archive,
        occupancy=grid.astype(np.int8),
        resolution_m=resolution,
        origin_xy_m=np.asarray(origin),
        floor_height_m=floor,
        diagnostics=json.dumps(data.get("diagnostics", {}), sort_keys=True),
    )
    contents = {"npz": archive.getvalue(), "pgm": pgm, "yaml": yaml.safe_dump(metadata, sort_keys=False).encode()}
    stem.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        for suffix, path in paths.items():
            with path.open("xb") as output:
                created.append(path)
                output.write(contents[suffix])
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return paths


class DynamicObstacleObserver:
    """Cache scene collision topology while reading live obstacle poses on every sample.

    The listener invalidates classification after USD structural/physics edits.
    Ordinary default xform values and skeleton-pose values never rebuild the
    fixed-joint graph. No dynamic position, rotation or bounding box is cached.
    Static collider membership is fixed for the baked map; changes require a new session.
    """

    def __init__(
        self,
        stage: Usd.Stage,
        frame: StageFrame,
        excluded_roots: Sequence[str],
        dynamic_roots: Sequence[str] = (),
        *,
        rigid_pose_reader=None,
    ) -> None:
        """Own one stage's topology/listener; roots select bodies independently of scene names."""
        from pxr import Tf, Usd

        self._stage, self.frame = stage, frame
        self._rigid_pose_reader = rigid_pose_reader
        self._excluded, self._dynamic = _roots(excluded_roots), _roots(dynamic_roots)
        self._colliders: dict[str, str | None] = {}
        self._static_colliders: frozenset[str] | None = None
        self._physics_scenes: tuple[Usd.Prim, ...] = ()
        self._time_varying_xforms: set[str] = set()
        self._dirty, self.closed = True, False
        self._notice = Tf.Notice.Register(Usd.Notice.ObjectsChanged, self._on_objects_changed, stage)

    def _on_objects_changed(self, notice, sender) -> None:
        if self.closed:
            return
        if notice.GetResyncedPaths():
            self._dirty = True
            return
        properties = {
            "physics:collisionEnabled",
            "physics:rigidBodyEnabled",
            "physics:kinematicEnabled",
            "physics:jointEnabled",
            "physics:breakForce",
            "physics:breakTorque",
            "physics:body0",
            "physics:body1",
            "physics:localPos0",
            "physics:localPos1",
            "physics:localRot0",
            "physics:localRot1",
            "physics:simulationOwner",
            "physxScene:enableSceneQuerySupport",
            "prototypes",
            "protoIndices",
            "xformOpOrder",
        }
        for path in notice.GetChangedInfoOnlyPaths():
            if path.IsPrimPath():
                if {str(field) for field in notice.GetChangedFields(path)} & {
                    "apiSchemas",
                    "active",
                    "typeName",
                    "instanceable",
                }:
                    self._dirty = True
            elif path.name in properties:
                self._dirty = True
            elif path.name.startswith("xformOp:"):
                key = str(path)
                attr = self._stage.GetAttributeAtPath(path)
                varying = bool(attr and attr.GetNumTimeSamples())
                if varying != (key in self._time_varying_xforms):
                    self._dirty = True
                    if varying:
                        self._time_varying_xforms.add(key)
                    else:
                        self._time_varying_xforms.discard(key)

    def sample(self) -> list[dict[str, object]]:
        """Return fresh ROS obstacle footprints [m] and unpredicted velocity placeholders [m/s]."""
        from pxr import Usd, UsdPhysics

        if self.closed:
            raise RuntimeError("Dynamic obstacle observer is closed.")
        if self._dirty:
            colliders = _inventory(self._stage, self._excluded, self._dynamic)
            static_colliders = frozenset(path for path, owner in colliders.items() if owner is None)
            if self._static_colliders is not None and static_colliders != self._static_colliders:
                raise RuntimeError(
                    "Static collision topology changed; rebuild the occupancy map and simulation session."
                )
            if self._static_colliders is None:
                self._static_colliders = static_colliders
            scenes = tuple(
                prim for prim in self._stage.Traverse(Usd.TraverseInstanceProxies()) if prim.IsA(UsdPhysics.Scene)
            )
            self._colliders, self._physics_scenes = colliders, scenes
            self._dirty = False
        return _sample_dynamic_obstacles(
            self._stage, self.frame, self._colliders, self._physics_scenes, rigid_pose_reader=self._rigid_pose_reader
        )

    def close(self) -> None:
        """Revoke the owned USD notice and release cached topology; failed revocation can be retried."""
        self.closed = True
        if self._notice is not None:
            self._notice.Revoke()
            self._notice = None
        self._colliders.clear()
        self._static_colliders = None
        self._physics_scenes = ()
        self._stage = None


def query_dynamic_obstacles(
    stage: Usd.Stage,
    frame: StageFrame,
    excluded_roots: Sequence[str],
    *,
    dynamic_roots: Sequence[str] = (),
) -> list[dict[str, object]]:
    """Read conservative ROS obstacle discs from live rigid bodies/explicit moving roots.

    Args:
        stage: PhysX-attached stage whose enabled solid colliders are observed.
        frame: Stage/world/map conversion.
        excluded_roots: Actor/robot roots handled separately by the caller.
        dynamic_roots: Additional script-driven object roots without RigidBodyAPI.

    Returns:
        World-bounds enclosing discs with ROS ``position`` [m], ``radius_m`` [m],
        counterclockwise ``footprint_xy_m`` vertices [m], actual vertical bounds [m],
        ``velocity`` [m/s] and ``velocity_valid``. USD velocity is only authored
        state, so it is labeled invalid for prediction; zero is not a motion claim.
    """
    colliders = _inventory(stage, _roots(excluded_roots), _roots(dynamic_roots))
    return _sample_dynamic_obstacles(stage, frame, colliders)


def _convex_hull_xy(points: Sequence[Sequence[float]]) -> list[list[float]]:
    vertices = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(vertices) < 3:
        return [list(point) for point in vertices]

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    lower, upper = [], []
    for vertex in vertices:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], vertex) <= 0:
            lower.pop()
        lower.append(vertex)
    for vertex in reversed(vertices):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], vertex) <= 0:
            upper.pop()
        upper.append(vertex)
    return [list(point) for point in (*lower[:-1], *upper[:-1])]


def _sample_dynamic_obstacles(
    stage, frame, colliders, physics_scenes=None, *, rigid_pose_reader=None
) -> list[dict[str, object]]:
    from pxr import Gf, UsdGeom, UsdPhysics

    _queries(stage, frame, physics_scenes)
    roots = sorted({value for value in colliders.values() if value is not None})
    rigid_roots = {
        path
        for path in roots
        if (prim := stage.GetPrimAtPath(path)).HasAPI(UsdPhysics.RigidBodyAPI)
        and UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is not False
    }
    if rigid_pose_reader is not None:
        poses = rigid_pose_reader(tuple(sorted(rigid_roots)))
    else:
        physics = import_module("omni.physx").get_physx_interface()
        poses = {path: physics.get_rigidbody_transformation(path) for path in rigid_roots}
    time_code = _time_code(stage)
    cache = UsdGeom.BBoxCache(
        time_code, ["default", "render", "proxy", "guide"], useExtentsHint=False, ignoreVisibility=True
    )
    result = []
    for path in roots:
        prim = stage.GetPrimAtPath(path)
        # A Boundable body root can omit children; only its enabled physical shapes define this footprint.
        corners = [
            point
            for collider, owner in colliders.items()
            if owner == path
            for point in _world_corners(cache, stage.GetPrimAtPath(collider))
        ]
        if path in rigid_roots:
            pose = poses[path]
            if not pose.get("ret_val"):
                raise RuntimeError(f"Dynamic obstacle has no live PhysX pose: {path}")
            authored = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time_code)
            transform = Gf.Transform(authored)
            inverse_rotation = transform.GetRotation().GetInverse()
            position = Gf.Vec3d(*pose["position"])
            quaternion = pose["rotation"]
            if len(quaternion) != 4 or not all(math.isfinite(float(value)) for value in quaternion):
                raise RuntimeError(f"Dynamic obstacle has an invalid live rotation: {path}")
            norm = math.sqrt(sum(float(value) ** 2 for value in quaternion))
            if norm < 1e-12:
                raise RuntimeError(f"Dynamic obstacle has a degenerate live rotation: {path}")
            rotation = Gf.Rotation(Gf.Quatd(float(quaternion[3]) / norm, Gf.Vec3d(*quaternion[:3]) / norm))
            corners = [
                position + rotation.TransformDir(inverse_rotation.TransformDir(point - transform.GetTranslation()))
                for point in corners
            ]
        points = [frame.stage_to_ros(point) for point in corners]
        center = tuple((min(point[i] for point in points) + max(point[i] for point in points)) / 2 for i in range(3))
        radius = max(math.hypot(point[0] - center[0], point[1] - center[1]) for point in points)
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError(f"Dynamic obstacle has invalid physical bounds: {path}")
        result.append(
            {
                "prim_path": path,
                "position": list(center),
                "radius_m": radius,
                "footprint_xy_m": _convex_hull_xy(points),
                "z_min_m": min(point[2] for point in points),
                "z_max_m": max(point[2] for point in points),
                "velocity": [0.0, 0.0, 0.0],
                "velocity_valid": False,
            }
        )
    return result
