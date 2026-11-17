# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Video sampling and failure contracts without a native Kit application."""

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from pedstrain_simulation.simulation.configuration.models import SimulationSettings
from pedstrain_simulation.simulation.configuration.navigation import NavigationConfig
from pedstrain_simulation.simulation.configuration.recording import RecordingSettings
from pedstrain_simulation.simulation.configuration.scenarios.models import PedestrianPlan
from pedstrain_simulation.simulation.recording import ros2_recorder
from pedstrain_simulation.simulation.ros2_navigation.frames import StageFrame


class _Session:
    def __init__(self, tmp_path, *, warmup=0.0, population=True):
        self.config = SimpleNamespace(
            settings=SimulationSettings(duration=0.25),
            navigation=NavigationConfig(backend="ros2"),
            pedestrians=SimpleNamespace(enabled=population),
            recording=RecordingSettings(enabled=True, output_path=tmp_path / "video.mp4", warmup_s=warmup),
        )
        self.navigation = SimpleNamespace(control_hz=60)
        self.frame = StageFrame(0.01, "Y", (2.0, -3.0, 0.0), 0.2)
        self.plans = (PedestrianPlan("walker", behavior="stand"),) if population else ()
        self.actors = SimpleNamespace(states=self.states) if population else None
        self.output = tmp_path / "runtime"
        self._last_time = self._native_origin = 0.0
        self.playing, self.pending_start, self.closed, self.end = False, False, False, 1.0
        self.timeline = SimpleNamespace(
            get_end_time=lambda: self.end,
            set_end_time=self.set_end,
            commit=lambda: None,
            is_playing=lambda: self.playing,
        )

    def set_end(self, value):
        self.end = value

    def states(self):
        return [{"name": "walker", "position": [self._last_time * 0.9, 0.0, 0.0], "velocity": [0.9, 0, 0]}]

    def play(self):
        self.playing = self.pending_start = True

    def pause(self):
        self.playing = False

    def is_running(self):
        return True

    def update(self):
        if self.playing:
            if self.pending_start:
                self.pending_start = False
            else:
                self._last_time += 1 / 60


def _environment(monkeypatch, tmp_path, *, warmup=0.0, population=True, fail_capture=False):
    session = _Session(tmp_path, warmup=warmup, population=population)
    captures = []
    encoded = []

    @contextmanager
    def viewport(current):
        def capture(path):
            current.pause()
            if fail_capture and captures:
                raise RuntimeError("capture readback failed")
            captures.append(current._last_time)
            path.write_bytes(b"PNG")
            current.play()

        try:
            yield SimpleNamespace(metadata={"kind": "fake_camera"}, capture=capture)
        finally:
            current.closed = True

    def encode(config, frames, output):
        encoded.append(len(list(frames.glob("frame_*.png"))))
        output.write_bytes(b"MP4")

    monkeypatch.setattr(ros2_recorder, "ros2_recording_viewport", viewport)
    monkeypatch.setattr(ros2_recorder, "_input_hashes", lambda session: {"input": "unchanged"})
    monkeypatch.setattr(ros2_recorder, "_source_hashes", lambda: {"source": "unchanged"})
    monkeypatch.setattr(ros2_recorder, "_encode_video", encode)
    monkeypatch.setattr(ros2_recorder, "_video_info", lambda *args: {"verified": True})
    return session, captures, encoded


@pytest.mark.parametrize("duration", [1.0e308, 10**308], ids=["float", "integer"])
def test_recording_preflight_rejects_overflowing_frame_count(tmp_path, duration):
    session = _Session(tmp_path)
    session.config.settings = SimulationSettings(duration=duration)
    with pytest.raises(ValueError, match="finite frame count"):
        ros2_recorder.validate_recording_runtime(session.config)
    assert not (tmp_path / "video.mp4").exists()


@pytest.mark.parametrize("warmup", [0.0, 0.05])
@pytest.mark.parametrize("population", [False, True])
def test_simulation_grid_samples_and_full_duration_survive_render_resume_updates(
    monkeypatch, tmp_path, warmup, population
):
    session, captures, encoded = _environment(monkeypatch, tmp_path, warmup=warmup, population=population)
    result = ros2_recorder.record_ros2_simulation(session)
    assert captures == pytest.approx([warmup + index / 12 for index in range(3)])
    assert encoded == [3] and session.closed
    assert result["passed"] and result["video_duration_s"] == 0.25
    assert result["recorded_elapsed_s"] == pytest.approx(0.25)
    assert session._last_time == pytest.approx(warmup + 0.25)
    assert result["state_frame"] == "ros_map" and result["map_origin_world_m"] == [2.0, -3.0, 0.0]
    assert result["map_yaw_rad"] == 0.2 and result["stage_up_axis"] == "Y"
    saved = json.loads((tmp_path / "video.json").read_text())
    assert saved["frames_written"] == 3 and saved["motion"]["elapsed_s"] == pytest.approx(0.25)


def test_capture_failure_preserves_partial_metadata_and_never_encodes_a_success(monkeypatch, tmp_path):
    session, captures, encoded = _environment(monkeypatch, tmp_path, fail_capture=True)
    with pytest.raises(RuntimeError, match="capture readback failed"):
        ros2_recorder.record_ros2_simulation(session)
    assert session.closed and encoded == [] and len(captures) == 1
    saved = json.loads((tmp_path / "video.json").read_text())
    assert saved["passed"] is False and saved["frames_written"] == 1
    assert "capture readback failed" in saved["error"]


def test_input_hash_read_failure_is_reported_before_any_capture(monkeypatch, tmp_path):
    session, captures, encoded = _environment(monkeypatch, tmp_path)

    def fail(current):
        raise OSError("input hash read failed")

    monkeypatch.setattr(ros2_recorder, "_input_hashes", fail)
    with pytest.raises(OSError, match="input hash read failed"):
        ros2_recorder.record_ros2_simulation(session)
    assert captures == [] and encoded == []
    saved = json.loads((tmp_path / "video.json").read_text())
    assert saved["passed"] is False and "input hash read failed" in saved["error"]
