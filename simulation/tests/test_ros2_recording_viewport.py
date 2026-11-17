# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Owned ROS camera, paused capture and cleanup contracts without a GPU."""

import asyncio
import copy
import json
import math
import struct
import sys
import zlib
from itertools import product
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from pxr import Gf, Sdf, Usd, UsdGeom

from pedstrain_simulation.simulation.configuration.recording import RecordingSettings
from pedstrain_simulation.simulation.configuration.scenarios.models import PedestrianPlan, Waypoint
from pedstrain_simulation.simulation.recording import ros2_viewport as viewport_module
from pedstrain_simulation.simulation.recording.ros2_viewport import ros2_recording_viewport
from pedstrain_simulation.simulation.ros2_navigation.frames import StageFrame


def _png(width, height):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress((b"\x00" + b"\x00" * width * 3) * height))
        + chunk(b"IEND", b"")
    )


def _render_schema(monkeypatch):
    # UsdRender is loaded by Kit in the SDK; keep only that schema boundary fake.
    import pxr

    render = ModuleType("pxr.UsdRender")

    class Product:
        def __init__(self, prim):
            self.prim = prim

        @classmethod
        def Define(cls, stage, path):
            return cls(stage.DefinePrim(path, "RenderProduct"))

        def GetPath(self):
            return self.prim.GetPath()

        def CreateResolutionAttr(self, value):
            return self.prim.CreateAttribute("resolution", Sdf.ValueTypeNames.Int2).Set(value)

        def GetResolutionAttr(self):
            return self.prim.GetAttribute("resolution")

        def CreateCameraRel(self):
            return self.prim.CreateRelationship("camera")

        def GetCameraRel(self):
            return self.prim.GetRelationship("camera")

        def CreateOrderedVarsRel(self):
            return self.prim.CreateRelationship("orderedVars")

    class Var(Product):
        def CreateSourceNameAttr(self, value):
            return self.prim.CreateAttribute("sourceName", Sdf.ValueTypeNames.String).Set(value)

        def CreateSourceTypeAttr(self, value):
            return self.prim.CreateAttribute("sourceType", Sdf.ValueTypeNames.Token).Set(value)

        def CreateDataTypeAttr(self, value):
            return self.prim.CreateAttribute("dataType", Sdf.ValueTypeNames.Token).Set(value)

    render.Product, render.Var = Product, Var
    monkeypatch.setitem(sys.modules, "pxr.UsdRender", render)
    monkeypatch.setattr(pxr, "UsdRender", render, raising=False)


def _session(monkeypatch, *, units=1.0, up="Z", playing=True, explicit=False, population=True):
    _render_schema(monkeypatch)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, units)
    UsdGeom.SetStageUpAxis(stage, up)
    root = UsdGeom.Xform.Define(stage, "/World")
    root.AddTranslateOp().Set((30.0, 40.0, 50.0))
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Camera.Define(stage, "/World/UserCamera")
    frame = StageFrame(units, up, (5.0, -3.0, 0.2), 0.4)
    state = {
        "name": "walker",
        "position": list(frame.world_to_ros((-2.0, -1.0, 0.0))),
        "yaw": 0.0,
        "velocity": [0.0, 0.0, 0.0],
        "radius_m": 0.25,
        "height_m": 2.4,
    }
    actors = SimpleNamespace(root_path="/NoOwnedActors", states=lambda: copy.deepcopy([state])) if population else None
    timeline = SimpleNamespace(now=2.0, playing=playing)
    timeline.is_playing = lambda: timeline.playing
    timeline.get_current_time = lambda: timeline.now

    class App:
        updates = 0
        running = True
        mutate = False

        def update(self):
            self.updates += 1
            if timeline.playing:
                timeline.now += 1 / 60
            if self.mutate:
                state["position"][0] += 0.01

        def is_running(self):
            return self.running

        def run_coroutine(self, coroutine):
            return asyncio.run(coroutine)

    def pause():
        timeline.playing = False

    def play():
        timeline.playing = True

    eye = (1.0, -4.0, 8.0) if explicit else None
    target = (1.0, 2.0, 1.0) if explicit else None
    settings = RecordingSettings(camera_eye_m=eye, camera_target_m=target)
    session = SimpleNamespace(
        stage=stage,
        frame=frame,
        actors=actors,
        timeline=timeline,
        app=App(),
        pause=pause,
        play=play,
        config=SimpleNamespace(settings=SimpleNamespace(width=1280, height=720), recording=settings),
        plans=(
            PedestrianPlan(
                "walker", position=(-2.0, -1.0, 0.0), behavior="walk", waypoints=(Waypoint((12.0, 4.0, 0.0)),)
            ),
        )
        if population
        else (),
        _map=None,
    )
    modules = {}
    for name in (
        "omni",
        "omni.kit",
        "omni.kit.app",
        "omni.kit.renderer_capture",
        "omni.kit.viewport",
        "omni.kit.viewport.utility",
    ):
        modules[name] = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, modules[name])
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(modules[parent], child, modules[name])
    previous = SimpleNamespace(camera_path="/World/UserCamera", resolution=(1280, 720), resolution_scale=0.5)
    record = SimpleNamespace(
        previous=previous,
        windows=[],
        captures=[],
        fail_capture=False,
        fail_window=False,
        delayed_writer=False,
        never_complete=False,
        corrupt_png=False,
        flushes=0,
        pending=None,
        saw_partial=False,
    )

    class Window:
        def __init__(self, camera_path):
            self.destroyed = False
            self.viewport_api = SimpleNamespace(
                camera_path=camera_path, render_product_path="", resolution=(0, 0), resolution_scale=0.5
            )

        def destroy(self):
            self.destroyed = True

    def create(name, **kwargs):
        if record.fail_window:
            raise RuntimeError("deliberate viewport setup failure")
        window = Window(kwargs["camera_path"])
        record.windows.append(window)
        return window

    def capture(viewport, path):
        record.captures.append((timeline.now, copy.deepcopy(actors.states()) if actors else [], path))

        async def wait(completion_frames=0):
            session.app.update()
            if record.fail_capture:
                raise RuntimeError("deliberate capture failure")
            from pathlib import Path

            data = _png(1280, 720)
            if record.corrupt_png:
                data = data[:29] + bytes([data[29] ^ 1]) + data[30:]
            if record.delayed_writer or record.never_complete:
                record.pending = Path(path), data
            else:
                Path(path).write_bytes(data)

        return SimpleNamespace(wait_for_result=wait)

    utility = modules["omni.kit.viewport.utility"]
    utility.get_active_viewport = lambda: previous
    utility.create_viewport_window = create
    utility.capture_viewport_to_file = capture

    def flush():
        record.flushes += 1
        if record.pending is not None and not record.never_complete:
            path, data = record.pending
            if record.flushes == 1:
                path.write_bytes(data[:33])
                record.saw_partial = True
            else:
                path.write_bytes(data)
                record.pending = None

    async def next_update():
        session.app.update()
        await asyncio.sleep(0)

    modules["omni.kit.app"].get_app = lambda: SimpleNamespace(next_update_async=next_update)
    modules["omni.kit.renderer_capture"].acquire_renderer_capture_interface = lambda: SimpleNamespace(
        wait_async_capture=flush
    )
    return session, record, state


@pytest.mark.parametrize("units,up", [(1.0, "Z"), (0.01, "Z"), (0.01, "Y")])
def test_explicit_camera_uses_canonical_world_units_and_ignores_map_origin(monkeypatch, units, up):
    session, record, _ = _session(monkeypatch, units=units, up=up, explicit=True)
    original = session.stage.GetRootLayer().ExportToString()
    with ros2_recording_viewport(session) as view:
        camera = UsdGeom.Camera.Get(session.stage, view.metadata["camera"])
        expected = session.frame.ros_to_stage(session.frame.world_to_ros((1.0, -4.0, 8.0)))
        assert tuple(camera.ComputeLocalToWorldTransform(0.0).ExtractTranslation()) == pytest.approx(expected)
        assert camera.GetFocalLengthAttr().Get() * units == pytest.approx(0.24)
        assert camera.GetHorizontalApertureAttr().Get() * units == pytest.approx(0.36)
        assert view.metadata["auto_framed"] is False
        assert view.metadata["camera_view"] == "custom"
        assert view.metadata["camera_eye_m"] == [1.0, -4.0, 8.0]
        assert view.metadata["world_frame"] == "canonical_z_up_m"
        assert camera.GetPrim().GetAttribute("xformOp:transform").HasAuthoredValueOpinion()
    assert session.stage.GetRootLayer().ExportToString() == original
    assert session.stage.GetSessionLayer().subLayerPaths == []
    assert record.windows[0].destroyed


@pytest.mark.parametrize("units,up", [(1.0, "Z"), (0.01, "Y")])
@pytest.mark.parametrize("camera_view", ["oblique", "overhead"])
def test_auto_camera_contains_future_waypoints_full_heads_and_nonidentity_parent(monkeypatch, units, up, camera_view):
    session, _, _ = _session(monkeypatch, units=units, up=up)
    if camera_view == "overhead":
        session.config.recording = RecordingSettings(camera_view=camera_view)
    with ros2_recording_viewport(session) as view:
        camera = UsdGeom.Camera.Get(session.stage, view.metadata["camera"])
        transform = camera.ComputeLocalToWorldTransform(0.0)
        assert view.metadata["camera_view"] == camera_view
        if camera_view == "overhead":
            eye = Gf.Vec3d(*session.frame.stage_to_world(transform.ExtractTranslation()))
            for local, expected in (((0, 0, -1), (0, 0, -1)), ((0, 1, 0), (0, 1, 0))):
                world = Gf.Vec3d(*session.frame.stage_to_world(transform.Transform(Gf.Vec3d(*local))))
                assert tuple((world - eye).GetNormalized()) == pytest.approx(expected, abs=1e-10)
        low, high = view.metadata["framing_bounds_world_m"]
        assert high[0] >= 12.75 and high[2] >= 2.4
        matrix = transform.GetInverse() * camera.GetCamera().frustum.ComputeProjectionMatrix()
        for point in product(*zip(low, high)):
            native = session.frame.ros_to_stage(session.frame.world_to_ros(point))
            ndc = matrix.Transform(Gf.Vec3d(*native))
            assert max(abs(ndc[i]) for i in (0, 1)) < 0.9
            assert abs(ndc[2]) < 1
        assert max(view.metadata["max_framing_ndc_xy"]) < 0.9
        json.dumps(view.metadata, allow_nan=False)


@pytest.mark.parametrize("playing", [False, True])
def test_frame_capture_preserves_time_poses_play_state_and_original_viewport(monkeypatch, tmp_path, playing):
    session, record, _ = _session(monkeypatch, playing=playing)
    original_time, original_states = session.timeline.now, session.actors.states()
    before = session.stage.GetRootLayer().ExportToString()
    with ros2_recording_viewport(session) as view:
        path = tmp_path / "frame.png"
        view.capture(path)
        assert path.is_file()
        assert view.metadata["frames_captured"] == 1
        assert view.metadata["last_capture_native_time_s"] == original_time
        assert session.timeline.now == original_time and session.actors.states() == original_states
        assert session.timeline.playing is playing
        with pytest.raises(ValueError, match="already exists"):
            view.capture(path)
        camera = view.metadata["camera"]
    assert not session.stage.GetPrimAtPath(camera)
    assert record.windows[0].destroyed and len(record.windows) == 1
    assert record.previous.camera_path == "/World/UserCamera"
    assert record.previous.resolution == (1280, 720) and record.previous.resolution_scale == 0.5
    assert session.stage.GetRootLayer().ExportToString() == before
    assert session.timeline.now == original_time and session.timeline.playing is playing
    with pytest.raises(RuntimeError, match="closed"):
        view.capture(tmp_path / "after_close.png")


def test_capture_exception_is_preserved_and_owned_resources_are_removed(monkeypatch, tmp_path):
    session, record, _ = _session(monkeypatch)
    record.fail_capture = True
    before = session.stage.GetRootLayer().ExportToString()
    with pytest.raises(RuntimeError, match="deliberate capture failure"):
        with ros2_recording_viewport(session) as view:
            camera = view.metadata["camera"]
            view.capture(tmp_path / "frame.png")
    assert session.timeline.playing and session.timeline.now == 2.0
    assert record.windows[0].destroyed and not session.stage.GetPrimAtPath(camera)
    assert session.stage.GetRootLayer().ExportToString() == before
    assert session.stage.GetSessionLayer().subLayerPaths == []


@pytest.mark.parametrize("primary_error", [False, True])
def test_capture_restoration_checks_both_observers_and_resumes_when_their_reads_fail(monkeypatch, primary_error):
    session, _, _ = _session(monkeypatch)
    read_time, read_states = session.timeline.get_current_time, session.actors.states
    calls = {"time": 0, "states": 0}
    primary = ValueError("original capture error")

    def time():
        calls["time"] += 1
        if calls["time"] > 1:
            raise RuntimeError("restoration clock unavailable")
        return read_time()

    def states():
        calls["states"] += 1
        if calls["states"] > 1:
            raise RuntimeError("restoration states unavailable")
        return read_states()

    session.timeline.get_current_time, session.actors.states = time, states
    with pytest.raises(ValueError if primary_error else RuntimeError) as caught:
        with viewport_module._paused_capture(session):
            if primary_error:
                raise primary
    if primary_error:
        assert caught.value is primary
    assert session.timeline.playing and session.timeline.now == 2.0
    assert calls == {"time": 2, "states": 2}
    details = str(caught.value) + " ".join(caught.value.__notes__)
    assert "restoration clock unavailable" in details and "restoration states unavailable" in details


@pytest.mark.parametrize("primary_error", [False, True])
def test_viewport_cleanup_survives_a_capture_guard_entry_failure(monkeypatch, primary_error):
    session, record, _ = _session(monkeypatch)
    original = session.stage.GetRootLayer().ExportToString()
    primary = ValueError("original rendering error")

    def unavailable():
        raise RuntimeError("cleanup states unavailable")

    with pytest.raises(ValueError if primary_error else RuntimeError) as caught:
        with ros2_recording_viewport(session) as view:
            camera_path = view.metadata["camera"]
            session.actors.states = unavailable
            record.previous.camera_path, record.previous.resolution, record.previous.resolution_scale = (
                "/Changed",
                (2, 2),
                1.0,
            )
            if primary_error:
                raise primary
    if primary_error:
        assert caught.value is primary
    assert "cleanup states unavailable" in str(caught.value) + " ".join(getattr(caught.value, "__notes__", []))
    assert record.windows[0].destroyed and not session.stage.GetPrimAtPath(camera_path)
    assert session.stage.GetSessionLayer().subLayerPaths == []
    assert session.stage.GetRootLayer().ExportToString() == original
    assert record.previous.camera_path == "/World/UserCamera"
    assert record.previous.resolution == (1280, 720) and record.previous.resolution_scale == 0.5
    assert session.timeline.now == 2.0 and session.timeline.playing


def test_capture_detects_an_external_pose_writer_without_rewriting_the_pose(monkeypatch, tmp_path):
    session, _, state = _session(monkeypatch)
    with ros2_recording_viewport(session) as view:
        old = state["position"][0]
        session.app.mutate = True
        with pytest.raises(RuntimeError, match="changed actor states"):
            view.capture(tmp_path / "frame.png")
        session.app.mutate = False
        assert state["position"][0] > old
        assert session.timeline.playing and session.timeline.now == 2.0


def test_viewport_setup_failure_cleans_the_layer_without_touching_user_camera(monkeypatch):
    session, record, _ = _session(monkeypatch)
    before = session.stage.GetRootLayer().ExportToString()
    record.fail_window = True
    with pytest.raises(RuntimeError, match="deliberate viewport setup failure"):
        with ros2_recording_viewport(session):
            pytest.fail("The failed viewport must not be yielded")
    assert session.stage.GetRootLayer().ExportToString() == before
    assert session.stage.GetSessionLayer().subLayerPaths == []
    assert session.timeline.now == 2.0 and session.timeline.playing


def test_no_population_can_frame_the_physical_map_instead_of_fixed_scene_coordinates(monkeypatch):
    session, _, _ = _session(monkeypatch, population=False)
    session._map = {
        "occupancy": np.zeros((20, 40), dtype=np.int8),
        "origin_xy_m": (3.0, -4.0),
        "resolution_m": 0.1,
        "floor_height_m": 0.3,
    }
    with ros2_recording_viewport(session) as view:
        low, high = view.metadata["framing_bounds_world_m"]
        assert all(math.isfinite(v) for point in (low, high) for v in point)
        for point in product((3.0, 7.0), (-4.0, -2.0), (0.3, 2.15)):
            world = session.frame.ros_to_world(point)
            assert all(low[i] - 1e-12 <= world[i] <= high[i] + 1e-12 for i in range(3))


def test_exactly_vertical_explicit_camera_gets_a_nondegenerate_up_vector(monkeypatch):
    session, _, _ = _session(monkeypatch, explicit=True)
    session.config.recording = RecordingSettings(camera_eye_m=(0.0, 0.0, 8.0), camera_target_m=(0.0, 0.0, 0.0))
    with ros2_recording_viewport(session) as view:
        camera = UsdGeom.Camera.Get(session.stage, view.metadata["camera"])
        rotation = Gf.Transform(camera.ComputeLocalToWorldTransform(0.0)).GetRotation()
        assert rotation.GetQuat().GetReal() != 0 or rotation.GetQuat().GetImaginary().GetLength() > 0


@pytest.mark.parametrize("attribute", ["focal_length", "projection", "clipping_range"])
def test_capture_rejects_another_writers_lens_edit(monkeypatch, tmp_path, attribute):
    session, _, _ = _session(monkeypatch)
    with ros2_recording_viewport(session) as view:
        camera = UsdGeom.Camera.Get(session.stage, view.metadata["camera"])
        with Usd.EditContext(session.stage, session.stage.GetSessionLayer()):
            if attribute == "focal_length":
                camera.GetFocalLengthAttr().Set(camera.GetFocalLengthAttr().Get() * 2)
            elif attribute == "projection":
                camera.GetProjectionAttr().Set("orthographic")
            else:
                camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 0.02))
        with pytest.raises(RuntimeError, match="optics or world transform"):
            view.capture(tmp_path / "wrong_lens.png")
        assert session.timeline.now == 2.0 and session.timeline.playing


def test_capture_waits_for_delayed_writer_and_complete_png_after_the_future(monkeypatch, tmp_path):
    session, record, _ = _session(monkeypatch)
    record.delayed_writer = True
    before = session.actors.states()
    with ros2_recording_viewport(session) as view:
        frame = tmp_path / "late.png"
        view.capture(frame)
        assert record.flushes >= 2 and record.saw_partial
        assert frame.read_bytes() == _png(1280, 720)
        assert view.metadata["frames_captured"] == 1
        assert session.timeline.now == 2.0 and session.actors.states() == before and session.timeline.playing


def test_capture_writer_timeout_preserves_time_and_cleans_owned_resources(monkeypatch, tmp_path):
    session, record, _ = _session(monkeypatch)
    record.never_complete = True
    monkeypatch.setattr(viewport_module, "_CAPTURE_TIMEOUT_S", 0.01, raising=False)
    before = session.actors.states()
    with pytest.raises(TimeoutError):
        with ros2_recording_viewport(session) as view:
            camera = view.metadata["camera"]
            view.capture(tmp_path / "never.png")
    assert session.timeline.now == 2.0 and session.actors.states() == before and session.timeline.playing
    assert record.windows[0].destroyed and not session.stage.GetPrimAtPath(camera)
    assert session.stage.GetSessionLayer().subLayerPaths == []


def test_capture_rejects_finalized_png_with_invalid_crc(monkeypatch, tmp_path):
    session, record, _ = _session(monkeypatch)
    record.corrupt_png = True
    with ros2_recording_viewport(session) as view:
        with pytest.raises(RuntimeError, match="PNG checksum"):
            view.capture(tmp_path / "bad.png")
        assert view.metadata["frames_captured"] == 0
        assert session.timeline.now == 2.0 and session.timeline.playing
