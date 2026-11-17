# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""An owned ROS recording viewport that never advances navigation during capture."""

from __future__ import annotations

import asyncio
import copy
import math
import struct
import sys
import zlib
from contextlib import contextmanager
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

from ..scene.namespaces import select_runtime_prim_path
from .viewport import _remove_recording_camera_specs

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..ros2_navigation.runtime import Ros2SimulationSession

_CAPTURE_TIMEOUT_S = 30.0


def _complete_png(path: Path, resolution: tuple[int, int]) -> bool:
    """Check a finished PNG's dimensions and every chunk CRC through IEND."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return False
    if len(data) < 8:
        return False
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"Recording frame is not a PNG: {path}")
    offset, header, image_data, problem = 8, False, False, None
    while offset + 12 <= len(data):
        length, kind = struct.unpack_from(">I4s", data, offset)
        end = offset + 12 + length
        if end > len(data):
            return False
        payload = data[offset + 8 : offset + 8 + length]
        checksum = struct.unpack_from(">I", data, offset + 8 + length)[0]
        if zlib.crc32(kind + payload) != checksum:
            problem = problem or "PNG checksum mismatch"
        if kind == b"IHDR":
            if header or offset != 8 or length != 13:
                problem = problem or "Invalid PNG header"
            else:
                width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", payload)
                if (width, height) != resolution:
                    problem = problem or f"PNG resolution {(width, height)} differs from configured {resolution}"
                depths = {0: (1, 2, 4, 8, 16), 2: (8, 16), 3: (1, 2, 4, 8), 4: (8, 16), 6: (8, 16)}
                if depth not in depths.get(color, ()) or compression or filtering or interlace not in (0, 1):
                    problem = problem or "Invalid PNG image format"
                header = True
        elif kind == b"IDAT":
            image_data = True
        elif kind == b"IEND":
            if length or end != len(data) or not header or not image_data:
                problem = problem or "Invalid PNG completion marker"
            if problem:
                raise RuntimeError(f"{problem}: {path}")
            return True
        offset = end
    return False


def _states(session) -> list[dict[str, object]]:
    return copy.deepcopy(session.actors.states()) if session.actors is not None else []


@contextmanager
def _paused_capture(session):
    playing = session.timeline.is_playing()
    now, before = session.timeline.get_current_time(), _states(session)
    if playing:
        session.pause()
    try:
        yield
    finally:
        original = sys.exception()
        errors = []
        try:
            if session.timeline.get_current_time() != now:
                errors.append(RuntimeError("Recording capture advanced simulation time or changed actor states."))
        except BaseException as error:
            errors.append(error)
        try:
            if _states(session) != before:
                errors.append(RuntimeError("Recording capture advanced simulation time or changed actor states."))
        except BaseException as error:
            errors.append(error)
        if playing:
            try:
                session.play()
            except BaseException as error:
                errors.append(error)
        if errors:
            if original is not None:
                original.add_note(f"Recording capture restoration failed: {errors!r}")
            else:
                for error in errors[1:]:
                    errors[0].add_note(f"Additional recording restoration failure: {error}")
                raise errors[0]


def _world_to_stage(frame, point):
    return frame.ros_to_stage(frame.world_to_ros(point))


def _world_bounds(session):
    """Bound all requested paths and full bodies in canonical world coordinates [m]."""
    from pxr import Gf, Usd, UsdGeom

    points = []
    states = _states(session)
    sizes = {}
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), ["default", "render", "proxy"], useExtentsHint=False, ignoreVisibility=False
    )
    for state in states:
        center = session.frame.ros_to_world(state["position"])
        radius, low, high = float(state["radius_m"]), -0.05, float(state["height_m"]) + 0.05
        root = session.stage.GetPrimAtPath(session.actors.root_path + "/" + state["name"])
        if root:
            box = cache.ComputeWorldBound(root)
            extent = box.GetRange()
            if not extent.IsEmpty():
                visual = [
                    session.frame.ros_to_world(session.frame.stage_to_ros(box.GetMatrix().Transform(Gf.Vec3d(*p))))
                    for p in product(*zip(extent.GetMin(), extent.GetMax()))
                ]
                radius = max(radius, *(abs(p[i] - center[i]) for p in visual for i in (0, 1)))
                low = min(low, min(p[2] for p in visual) - center[2])
                high = max(high, max(p[2] for p in visual) - center[2] + 0.05)
        sizes[state["name"]] = radius + 0.5, low, high

    def body(center, size):
        radius, low, high = size
        x, y, z = center
        points.extend(product((x - radius, x + radius), (y - radius, y + radius), (z + low, z + high)))

    for state in states:
        body(session.frame.ros_to_world(state["position"]), sizes[state["name"]])
    for plan in session.plans:
        size = sizes.get(plan.name, (0.75, -0.05, 1.85))
        if plan.position is not None:
            body(plan.position, size)
        for waypoint in plan.waypoints:
            body(waypoint.position, size)

    grid = getattr(session, "_map", None)
    if grid is not None:
        ox, oy = grid["origin_xy_m"]
        rows, columns = grid["occupancy"].shape
        resolution = float(grid["resolution_m"])
        floor = float(grid.get("floor_height_m", 0.0))
        high = max((size[2] for size in sizes.values()), default=1.85)
        points.extend(
            session.frame.ros_to_world(p)
            for p in product((ox, ox + columns * resolution), (oy, oy + rows * resolution), (floor, floor + high))
        )
    if not points:
        box = cache.ComputeWorldBound(session.stage.GetPseudoRoot())
        extent = box.GetRange()
        if not extent.IsEmpty():
            points.extend(
                session.frame.ros_to_world(session.frame.stage_to_ros(box.GetMatrix().Transform(Gf.Vec3d(*p))))
                for p in product(*zip(extent.GetMin(), extent.GetMax()))
            )
    if not points:
        raise ValueError("Cannot derive recording bounds; provide an explicit camera eye and target.")
    return tuple(min(p[i] for p in points) for i in range(3)), tuple(max(p[i] for p in points) for i in range(3))


def _camera_pose(session):
    from pxr import Gf

    recording = session.config.recording
    if recording.camera_eye_m is not None:
        eye = Gf.Vec3d(*recording.camera_eye_m)
        target = Gf.Vec3d(*recording.camera_target_m)
        if (eye - target).GetLength() < 1e-9:
            raise ValueError("Recording camera eye and target must be distinct.")
        up = Gf.Vec3d(0, 0, 1)
        if Gf.Cross(eye - target, up).GetLength() < 1e-9:
            up = Gf.Vec3d(0, 1, 0)
        return eye, target, up, None
    low, high = _world_bounds(session)
    target = Gf.Vec3d(*(0.5 * (low[i] + high[i]) for i in range(3)))
    if recording.camera_view == "overhead":
        toward_eye = Gf.Vec3d(0, 0, 1)
        camera_up = up = Gf.Vec3d(0, 1, 0)
    else:
        toward_eye = Gf.Vec3d(0.0, -0.5, math.sqrt(0.75))
        camera_up = Gf.Vec3d(0.0, math.sqrt(0.75), 0.5)
        up = Gf.Vec3d(0, 0, 1)
    tangent_x = 36.0 / (2.0 * 24.0)
    tangent_y = tangent_x * session.config.settings.height / session.config.settings.width
    relative = [Gf.Vec3d(*p) - target for p in product(*zip(low, high))]
    distance = max(
        4.0,
        max(
            Gf.Dot(p, toward_eye) + 1.15 * max(abs(p[0]) / tangent_x, abs(Gf.Dot(p, camera_up)) / tangent_y)
            for p in relative
        ),
    )
    return target + distance * toward_eye, target, up, (low, high)


class Ros2RecordingViewport:
    """Own a reusable camera/render product while observing a ROS simulation.

    Attributes:
        metadata: JSON-compatible camera coordinates in canonical world [m],
            native stage metrics and physical optics [mm].
    """

    def __init__(
        self, session: Ros2SimulationSession, viewport, camera, render_product, capture, metadata: dict[str, object]
    ) -> None:
        self._session, self._viewport = session, viewport
        self._camera, self._product, self._capture = camera, render_product, capture
        self.metadata = metadata
        self._resolution = (session.config.settings.width, session.config.settings.height)
        self._clipping_range = tuple(value / session.frame.meters_per_unit for value in metadata["clipping_range_m"])
        self._closed = False

    def _check_ownership(self) -> None:
        from pxr import Usd

        if (
            str(self._viewport.camera_path) != str(self._camera.GetPath())
            or str(self._viewport.render_product_path) != str(self._product.GetPath())
            or self._product.GetCameraRel().GetTargets() != [self._camera.GetPath()]
            or tuple(self._viewport.resolution) != self._resolution
        ):
            raise RuntimeError("Recording viewport camera, product or resolution was changed by another writer.")
        units = self._session.frame.meters_per_unit
        optics = (
            (self._camera.GetFocalLengthAttr().Get(), 0.24 / units),
            (self._camera.GetHorizontalApertureAttr().Get(), 0.36 / units),
            (self._camera.GetVerticalApertureAttr().Get(), 0.36 * self._resolution[1] / self._resolution[0] / units),
        )
        current = self._camera.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        expected = self.metadata["camera_to_world_stage"]
        clipping = self._camera.GetClippingRangeAttr().Get()
        if (
            self._camera.GetProjectionAttr().Get() != "perspective"
            or any(not math.isclose(actual, desired, rel_tol=1e-6, abs_tol=1e-10) for actual, desired in optics)
            or any(
                not math.isclose(actual, desired, rel_tol=1e-6, abs_tol=1e-10)
                for actual, desired in zip(clipping, self._clipping_range)
            )
            or any(
                abs(current[i][j] - expected[i][j]) > (1e-5 / units if i == 3 and j < 3 else 1e-5)
                for i in range(4)
                for j in range(4)
            )
        ):
            raise RuntimeError("Recording camera optics or world transform was changed by another writer.")

    def capture(self, path: Path) -> None:
        """Write one fresh frame without advancing time [s] or actor poses [m, rad]."""
        if self._closed:
            raise RuntimeError("Recording viewport is closed.")
        path = Path(path)
        if path.exists() or path.is_symlink():
            raise ValueError(f"Recording frame destination already exists: {path}")
        if not path.parent.is_dir():
            raise ValueError(f"Recording frame parent must already exist: {path.parent}")
        if not self._session.app.is_running():
            raise RuntimeError("The simulation application exited before capture.")
        with _paused_capture(self._session):
            for _ in range(2):
                self._session.app.update()
            self._check_ownership()
            request = self._capture(self._viewport, str(path))

            async def finish():
                import omni.kit.app
                import omni.kit.renderer_capture

                await request.wait_for_result(completion_frames=0)
                interface = omni.kit.renderer_capture.acquire_renderer_capture_interface()
                while True:
                    # The capture future completes before the asynchronous PNG writer.
                    interface.wait_async_capture()
                    await omni.kit.app.get_app().next_update_async()
                    if _complete_png(path, self._resolution):
                        return

            try:
                self._session.app.run_coroutine(asyncio.wait_for(finish(), timeout=_CAPTURE_TIMEOUT_S))
            except TimeoutError as error:
                raise TimeoutError(
                    f"Recording PNG writer did not finish within {_CAPTURE_TIMEOUT_S:g} wall seconds: {path}"
                ) from error
            self._check_ownership()
            self.metadata["frames_captured"] += 1
            self.metadata["last_capture_native_time_s"] = self._session.timeline.get_current_time()


@contextmanager
def ros2_recording_viewport(session: Ros2SimulationSession) -> Iterator[Ros2RecordingViewport]:
    """Create one isolated recording viewport and remove all its owned USD specs.

    Explicit camera positions and targets use canonical YAML world coordinates
    [m]; map origin/yaw and USD units/up-axis do not change that contract.
    Automatic framing includes complete bodies, original paths and known map
    bounds. The overhead view points along canonical world -Z with +Y up.
    User lighting, existing cameras and navigation remain unchanged.

    Args:
        session: An initialized simulation whose stage and application are owned
            by the calling Kit thread.

    Yields:
        One reusable frame observer with JSON-compatible camera metadata.
    """
    from omni.kit.viewport.utility import capture_viewport_to_file, create_viewport_window, get_active_viewport
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdRender

    stage = session.stage
    camera_path = select_runtime_prim_path(stage, "Ros2RecordingCamera", unique=True)
    authored = {
        layer.identifier: {path for path in camera_path.GetPrefixes() if layer.GetPrimAtPath(path) is not None}
        for layer in stage.GetLayerStack()
    }
    previous = get_active_viewport()
    restore = (
        {
            "camera_path": previous.camera_path,
            "resolution": tuple(getattr(previous, "full_resolution", previous.resolution)),
            "resolution_scale": getattr(previous, "resolution_scale", None),
        }
        if previous is not None
        else {}
    )
    layer = Sdf.Layer.CreateAnonymous("ros2_recording_viewport.usda")
    window = None
    owner = None
    try:
        with _paused_capture(session):
            eye, target, up, bounds = _camera_pose(session)
            eye_stage = Gf.Vec3d(*_world_to_stage(session.frame, eye))
            target_stage = Gf.Vec3d(*_world_to_stage(session.frame, target))
            up_stage = Gf.Vec3d(*_world_to_stage(session.frame, up)) - Gf.Vec3d(
                *_world_to_stage(session.frame, (0, 0, 0))
            )
            stage.GetSessionLayer().subLayerPaths.insert(0, layer.identifier)
            with Usd.EditContext(stage, layer):
                camera = UsdGeom.Camera.Define(stage, camera_path)
                expected_transform = Gf.Matrix4d().SetLookAt(eye_stage, target_stage, up_stage).GetInverse()
                camera.MakeMatrixXform().Set(expected_transform)
                camera.SetResetXformStack(True)
                optics = 0.01 / session.frame.meters_per_unit
                camera.CreateProjectionAttr("perspective")
                camera.CreateFocalLengthAttr(24.0 * optics)
                camera.CreateHorizontalApertureAttr(36.0 * optics)
                camera.CreateVerticalApertureAttr(
                    36.0 * session.config.settings.height / session.config.settings.width * optics
                )
                camera.CreateFocusDistanceAttr((eye_stage - target_stage).GetLength())
                clipping_range = Gf.Vec2f(
                    0.01 / session.frame.meters_per_unit,
                    max(100.0, (eye - target).GetLength() * 10) / session.frame.meters_per_unit,
                )
                camera.CreateClippingRangeAttr(clipping_range)
                render_product = UsdRender.Product.Define(stage, camera_path.AppendChild("Product"))
                render_product.CreateResolutionAttr(
                    Gf.Vec2i(session.config.settings.width, session.config.settings.height)
                )
                render_product.CreateCameraRel().SetTargets([camera_path])
                aov = UsdRender.Var.Define(stage, camera_path.AppendChild("LdrColor"))
                aov.CreateSourceNameAttr("LdrColor")
                aov.CreateSourceTypeAttr("raw")
                aov.CreateDataTypeAttr("color4f")
                render_product.CreateOrderedVarsRel().SetTargets([aov.GetPath()])
            window = create_viewport_window(
                "ROS2 recording " + camera_path.name,
                width=session.config.settings.width,
                height=session.config.settings.height,
                camera_path=camera_path,
                usd_drop_support=False,
            )
            if window is None:
                raise RuntimeError("Cannot create an independent ROS recording viewport.")
            viewport = window.viewport_api
            viewport.camera_path = camera_path
            viewport.render_product_path = str(render_product.GetPath())
            viewport.resolution = (session.config.settings.width, session.config.settings.height)
            if getattr(viewport, "resolution_scale", None) is not None:
                viewport.resolution_scale = 1.0
            for _ in range(2):
                session.app.update()
            metadata = {
                "camera": str(camera_path),
                "render_product": str(render_product.GetPath()),
                "world_frame": "canonical_z_up_m",
                "camera_view": session.config.recording.camera_view if bounds is not None else "custom",
                "camera_eye_m": list(eye),
                "camera_target_m": list(target),
                "auto_framed": bounds is not None,
                "framing_bounds_world_m": [list(point) for point in bounds] if bounds is not None else None,
                "meters_per_unit": session.frame.meters_per_unit,
                "stage_up_axis": session.frame.up_axis,
                "map_origin_m": list(session.frame.origin_m),
                "map_yaw_rad": session.frame.yaw_rad,
                "resolution": list(viewport.resolution),
                "focal_length_mm": 24.0,
                "horizontal_aperture_mm": 36.0,
                "clipping_range_m": [value * session.frame.meters_per_unit for value in clipping_range],
                "frames_captured": 0,
                "capture_wait_timeout_s": _CAPTURE_TIMEOUT_S,
                "png_completion": "renderer_flush_and_chunk_crc_iend",
                "camera_to_world_stage": [list(row) for row in expected_transform],
                "camera_readback": {
                    "projection": camera.GetProjectionAttr().Get(),
                    "focal_length_attr": camera.GetFocalLengthAttr().Get(),
                    "horizontal_aperture_attr": camera.GetHorizontalApertureAttr().Get(),
                    "vertical_aperture_attr": camera.GetVerticalApertureAttr().Get(),
                    "clipping_range_attr": list(camera.GetClippingRangeAttr().Get()),
                    "camera_to_world_stage": [
                        list(row) for row in camera.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    ],
                    "product_resolution": list(render_product.GetResolutionAttr().Get()),
                    "product_camera": [str(p) for p in render_product.GetCameraRel().GetTargets()],
                },
            }
            if bounds is not None:
                matrix = (
                    camera.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
                    * camera.GetCamera().frustum.ComputeProjectionMatrix()
                )
                ndc = [matrix.Transform(Gf.Vec3d(*_world_to_stage(session.frame, p))) for p in product(*zip(*bounds))]
                if any(abs(p[0]) >= 1 or abs(p[1]) >= 1 or abs(p[2]) >= 1 for p in ndc):
                    raise RuntimeError("Automatic recording camera does not cover the complete path and body bounds.")
                metadata["max_framing_ndc_xy"] = [max(abs(p[i]) for p in ndc) for i in (0, 1)]
            owner = Ros2RecordingViewport(session, viewport, camera, render_product, capture_viewport_to_file, metadata)
            owner._check_ownership()
        yield owner
    finally:
        original = sys.exception()
        errors = []
        if owner is not None:
            owner._closed = True
        cleanup_started = False

        def cleanup(*, flush):
            nonlocal cleanup_started
            cleanup_started = True
            if window is not None:
                try:
                    window.destroy()
                    if flush and session.app.is_running():
                        for _ in range(2):
                            session.app.update()
                except BaseException as error:
                    errors.append(error)
            try:
                _remove_recording_camera_specs(stage, camera_path, authored)
            except BaseException as error:
                errors.append(error)
            try:
                paths = stage.GetSessionLayer().subLayerPaths
                if layer.identifier in paths:
                    paths.remove(layer.identifier)
            except BaseException as error:
                errors.append(error)
            if previous is not None:
                for name, value in restore.items():
                    if value is None:
                        continue
                    try:
                        current = (
                            getattr(previous, "full_resolution", previous.resolution)
                            if name == "resolution"
                            else getattr(previous, name)
                        )
                        if current != value:
                            setattr(previous, name, value)
                    except BaseException as error:
                        errors.append(error)

        try:
            with _paused_capture(session):
                cleanup(flush=True)
        except BaseException as error:
            errors.append(error)
            if not cleanup_started:
                # A broken observer must not strand owned resources. Avoid Kit
                # updates when the strict paused guard could not be established.
                was_playing = False
                try:
                    was_playing = session.timeline.is_playing()
                except BaseException as failure:
                    errors.append(failure)
                try:
                    session.pause()
                except BaseException as failure:
                    errors.append(failure)
                cleanup(flush=False)
                if was_playing:
                    try:
                        session.play()
                    except BaseException as failure:
                        errors.append(failure)
        if errors:
            if original is not None:
                original.add_note(f"Recording viewport cleanup failed: {errors!r}")
            else:
                for error in errors[1:]:
                    errors[0].add_note(f"Additional viewport cleanup failure: {error}")
                raise errors[0]
