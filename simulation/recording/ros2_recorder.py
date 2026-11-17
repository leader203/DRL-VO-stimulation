# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record the ROS-owned simulation without advancing it during image readback."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import time
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

from ..configuration.paths import ROS2_NAVIGATION_SETTINGS_PATH
from .motion import _TrajectoryMonitor
from .output import _encode_video, _recording_paths, validate_recording_runtime
from .ros2_viewport import ros2_recording_viewport

if TYPE_CHECKING:
    from ..ros2_navigation.runtime import Ros2SimulationSession


def _input_hashes(session: Ros2SimulationSession) -> dict[str, str]:
    config = session.config
    paths = [*config._source_paths, config.scene.asset_path, *config.scene.layers]
    paths.extend(robot.asset_path or robot.layer_path for robot in config.robots)
    paths.append(config.navigation.settings_path or ROS2_NAVIGATION_SETTINGS_PATH)
    paths.extend(getattr(session, "asset_sources", {}).values())
    paths.append(getattr(session, "motion_library", None))
    temporary, output = Path(session._temporary.name), session.output
    for layer in session.stage.GetUsedLayers():
        if layer.realPath:
            path = Path(layer.realPath)
            if not path.is_relative_to(temporary) and not path.is_relative_to(output):
                paths.append(path)
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted({Path(p).resolve() for p in paths if p is not None})
        if path.is_file()
    }


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).parents[1]
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.suffix in (".py", ".cpp", ".h", ".kit")
        and not {"tests", "output", "__pycache__"}.intersection(path.relative_to(root).parts)
    }


def _video_info(output: Path, width: int, height: int, fps: int, frames: int) -> dict[str, object]:
    result = subprocess.run(
        [
            shutil.which("ffprobe"),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_read_frames,duration",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError("The recording must contain one readable video stream.")
    video = streams[0]
    if (
        video.get("codec_name") != "h264"
        or (video.get("width"), video.get("height")) != (width, height)
        or int(video.get("nb_read_frames", -1)) != frames
        or abs(float(Fraction(video["avg_frame_rate"])) - fps) > 1e-6
        or abs(float(video["duration"]) - frames / fps) > 1e-3
    ):
        raise RuntimeError(f"Encoded video does not match the requested capture: {video}")
    return video


def record_ros2_simulation(session: Ros2SimulationSession) -> dict[str, object]:
    """Record configured simulation duration [s] at configured sampling rate [Hz].

    The ROS session remains the sole navigation owner. Captures freeze its native
    timeline; image readback cannot insert unobserved control or physics steps.
    Camera coordinates use canonical world [m], independently of map transforms.
    """
    config = session.config
    validate_recording_runtime(config)
    if not config.recording.enabled or config.navigation.backend != "ros2":
        raise ValueError("Enable recording on a ROS 2 configuration before recording.")
    output, frames, metadata = _recording_paths(config)
    frames.mkdir(parents=True, exist_ok=False)
    fps, duration = config.recording.fps, config.settings.duration
    frame_count = round(duration * fps)
    monitor = _TrajectoryMonitor(up_axis="Z", stationary_limit_s=config.recording.max_stationary_s)
    inputs, sources = {}, {}
    report: dict[str, object] = {
        "provenance": "simulation_ros2",
        "passed": False,
        "video": str(output),
        "frames_path": str(frames),
        "fps": fps,
        "requested_duration_s": duration,
        "warmup_s": config.recording.warmup_s,
        "width": config.settings.width,
        "height": config.settings.height,
        "control_hz": session.navigation.control_hz,
        "pedestrian_plans": [plan.to_dict() for plan in session.plans],
        "input_plans": [plan.to_dict() for plan in getattr(session, "input_plans", session.plans)],
        "coordinate_contract": getattr(session, "coordinate_contract", {}),
        "input_sha256": inputs,
        "source_sha256": sources,
        "runtime_output": str(session.output),
        "state_frame": "ros_map",
        "map_origin_world_m": list(session.frame.origin_m),
        "map_yaw_rad": session.frame.yaw_rad,
        "stage_meters_per_unit": session.frame.meters_per_unit,
        "stage_up_axis": session.frame.up_axis,
    }
    error = None
    times: list[float] = []
    deadline = time.monotonic() + max(300.0, (duration + config.recording.warmup_s) * 20)
    maximum_step_s = 0.0
    frame_samples = frames / "samples.jsonl"
    expected = {plan.name for plan in session.plans}

    def states():
        live = session.actors.states() if session.actors is not None else []
        if len(live) != len(expected) or {actor["name"] for actor in live} != expected:
            raise RuntimeError("Recorded pedestrian population changed from the prepared plan.")
        return live

    def advance(target_s: float, *, observe: bool) -> None:
        nonlocal maximum_step_s
        while session._last_time < target_s - 1e-8:
            if time.monotonic() > deadline:
                raise TimeoutError("ROS recording exceeded its wall-clock execution limit.")
            if not session.is_running() or not session.timeline.is_playing():
                raise RuntimeError("ROS recording stopped before the requested capture completed.")
            previous = session._last_time
            session.update()
            now = session._last_time
            if not math.isfinite(now) or now < previous:
                raise RuntimeError("ROS recording timeline did not advance monotonically.")
            if now == previous:
                continue
            maximum_step_s = max(maximum_step_s, now - previous)
            if observe:
                monitor.observe(now, {a["name"]: tuple(a["position"]) for a in states()})
                if stalled := monitor.stalled():
                    raise RuntimeError(f"Recorded pedestrians exceeded max_stationary_s: {stalled}")

    try:
        inputs.update(_input_hashes(session))
        sources.update(_source_hashes())
        if session._last_time != 0:
            raise ValueError("Start ROS recording from a newly prepared session at simulation time zero.")
        end = session._native_origin + config.recording.warmup_s + duration + 1.0
        session.timeline.set_end_time(max(session.timeline.get_end_time(), end))
        session.timeline.commit()
        with ros2_recording_viewport(session) as viewport, frame_samples.open("x", encoding="utf-8") as samples:
            report["camera"] = viewport.metadata
            session.play()
            advance(config.recording.warmup_s, observe=False)
            origin = session._last_time
            monitor.observe(origin, {a["name"]: tuple(a["position"]) for a in states()})
            report["recording_start_s"] = origin
            next_progress = 0.0
            for index in range(frame_count):
                target = origin + index / fps
                advance(target, observe=True)
                if not math.isclose(session._last_time, target, rel_tol=0, abs_tol=1e-7):
                    raise RuntimeError("Recording samples do not align with the configured control clock.")
                if time.monotonic() > deadline:
                    raise TimeoutError("ROS recording exceeded its wall-clock execution limit.")
                viewport.capture(frames / f"frame_{index:06d}.png")
                times.append(session._last_time)
                samples.write(
                    json.dumps({"frame": index, "time_s": times[-1], "actors": states()}, allow_nan=False) + "\n"
                )
                if index / fps >= next_progress:
                    print(
                        "ROS2_RECORDING_PROGRESS "
                        + json.dumps({"elapsed_s": index / fps, "frames": index + 1, "expected_frames": frame_count}),
                        flush=True,
                    )
                    next_progress += 5.0
            advance(origin + duration, observe=True)
            session.pause()
            report["simulation_end_time_s"] = session._last_time
            report["recorded_elapsed_s"] = session._last_time - origin
        if len(list(frames.glob("frame_*.png"))) != frame_count:
            raise RuntimeError("ROS recording did not write every requested frame.")
        _encode_video(config, frames, output)
        report["encoded_video"] = _video_info(output, config.settings.width, config.settings.height, fps, frame_count)
        report["inputs_unchanged"] = _input_hashes(session) == inputs
        report["source_unchanged"] = _source_hashes() == sources
        if not report["inputs_unchanged"] or not report["source_unchanged"]:
            raise RuntimeError("Recording inputs or functional sources changed during capture.")
        report.update(passed=True, frames=frame_count, video_duration_s=frame_count / fps)
        return report
    except BaseException as failure:
        error = failure
        report["error"] = f"{type(failure).__name__}: {failure}"
        raise
    finally:
        report.update(
            frames_written=len(times),
            simulation_times_s=times,
            maximum_simulation_step_s=maximum_step_s,
            motion=monitor.status(),
            frame_samples=str(frame_samples),
        )
        try:
            with metadata.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
                stream.write("\n")
        except Exception as metadata_error:
            if error is None:
                raise
            error.add_note(f"Recording metadata write failed: {metadata_error}")
