# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Automated workflow ownership, preservation and failure contracts without Kit."""

import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import replace
from pathlib import Path

import pytest
import tomllib
import yaml

from pedstrain_simulation.simulation.cli import launcher
from pedstrain_simulation.simulation.cli import workflow as workflow_cli
from pedstrain_simulation.simulation.configuration.loading import load_simulation_config
from pedstrain_simulation.simulation.projects import inspect_simulation_config, workflow
from pedstrain_simulation.simulation.recording import output as recording_output


@pytest.fixture
def project(tmp_path):
    scene = tmp_path / "room.usda"
    scene.write_text('#usda 1.0\n(defaultPrim="World")\ndef Xform "World" {}\n')
    characters = tmp_path / "characters"
    characters.mkdir()
    (characters / "Person.usda").write_text('#usda 1.0\ndef Xform "Person" {}\n')
    motion = tmp_path / "motion.usda"
    motion.write_text("#usda 1.0\n")
    (tmp_path / "navigation.yaml").write_text("{}\n")
    data = {
        "schema_version": 1,
        "navigation": {"backend": "ros2", "waypoint_frame": "stage_world", "settings_path": "navigation.yaml"},
        "scene": {"asset_path": str(scene)},
        "simulation": {"duration": 1.0, "headless": True, "width": 640, "height": 360},
        "pedestrians": {
            "count": 1,
            "assets": str(characters),
            "motion_library": str(motion),
            "defaults": {"behavior": "stand"},
            "actors": {"visitor": {"position": [0, 0, 0]}},
        },
    }
    path = tmp_path / "simulation.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_check_requires_disjoint_status_and_recording_paths(monkeypatch, project):
    monkeypatch.setattr(recording_output.shutil, "which", lambda name: name)
    data = yaml.safe_load(project.read_text())
    data["recording"] = {"enabled": True, "output_path": "out/nested/video.mp4"}
    for conflict in ("out", "out/nested", "out/nested/video_frames/status.json"):
        data["simulation"]["report_path"] = conflict
        project.write_text(yaml.safe_dump(data))
        with pytest.raises(ValueError, match="report_path must be separate from recording outputs"):
            inspect_simulation_config(project)

    data["simulation"]["report_path"] = "out/nested/status.json"
    project.write_text(yaml.safe_dump(data))
    assert inspect_simulation_config(project)["simulation"]["report_path"] == str(
        project.parent / "out/nested/status.json"
    )
    assert not (project.parent / "out").exists()


def _native(monkeypatch, project, *, code=0, gap=0.4, evaluation=True, mutate=False, loaded_configs=None):
    calls = []
    monkeypatch.setattr(workflow, "_ros_environment", lambda setup: {"ROS_READY": "yes"})

    def child(profile, log, environment, timeout):
        loaded = load_simulation_config(profile)
        if loaded_configs is not None:
            loaded_configs.append(loaded)
        data = tomllib.loads(profile.read_text())
        calls.append((data, environment, timeout))
        runtime = profile.parent / "native_run"
        runtime.mkdir()
        (runtime / "trace.jsonl").write_text(
            json.dumps(
                {
                    "command": {
                        "door_events": [
                            {
                                "door": "door",
                                "distance_m": gap,
                                "approach_distance_m": 0.55,
                            }
                        ]
                    }
                }
            )
            + "\n"
        )
        status = {"passed": code == 0, "output_directory": str(runtime), "error": "native failure" if code else None}
        if data["recording"]["enabled"]:
            video = Path(data["recording"]["output_path"])
            video.write_bytes(b"fake encoded video")
            status["recording"] = {"passed": True, "video": str(video)}
        Path(data["simulation"]["report_path"]).write_text(json.dumps(status))
        log.write_text("native log\n")
        if mutate:
            project.with_name("room.usda").write_text("# changed while running\n")
        return code

    def alignment(bootstrap, grid, output):
        output.mkdir()
        return {"passed": True, "output_directory": str(output)}

    monkeypatch.setattr(workflow, "_run_child", child)
    monkeypatch.setattr(workflow, "export_coordinate_alignment", alignment)
    monkeypatch.setattr(
        workflow,
        "evaluate_ros2_run",
        lambda *args, **kwargs: {
            "passed": evaluation,
            "errors": [] if evaluation else ["invalid trajectory"],
            "route_goals_complete": kwargs["require_route_goals"],
        },
    )
    return calls


def test_workflow_preserves_user_project_and_creates_a_fresh_recording_wrapper(monkeypatch, project, tmp_path):
    data = yaml.safe_load(project.read_text())
    data["recording"] = {"camera_view": "overhead"}
    project.write_text(yaml.safe_dump(data))
    original = project.read_bytes()
    calls = _native(monkeypatch, project)
    output = tmp_path / "new_run"
    report = workflow.run_simulation_workflow(project, output, record=True, require_route_goals=True)
    assert report["passed"] and project.read_bytes() == original
    wrapper, environment, timeout = calls[0]
    assert wrapper["scenario"] == str(project)
    assert wrapper["simulation"]["duration"] == 1.0 and wrapper["pedestrians"]["enabled"]
    assert wrapper["recording"]["camera_view"] == "overhead"
    assert environment == {"ROS_READY": "yes"} and timeout > 1
    assert report["video"] == str(output / "video.mp4")
    assert report["stages"]["trajectory_evaluation"]["route_goals_complete"]
    assert json.loads((output / "workflow_report.json").read_text())["passed"]


@pytest.mark.parametrize("original_exists", [False, True])
def test_workflow_navigation_override_uses_one_settings_file_throughout(
    monkeypatch, project, tmp_path, original_exists
):
    original = project.with_name("unused_navigation.yaml")
    if original_exists:
        original.write_text("unknown_setting: true\n")
    profile = project.with_name("headless.toml")
    profile.write_text(f'scenario="simulation.yaml"\n[navigation]\nsettings_path="{original.name}"\n')
    launch_dir = tmp_path / "launch"
    launch_dir.mkdir()
    override = launch_dir / "tuned.yaml"
    setup = launch_dir / "setup.bash"
    setup.write_text("# ROS setup fixture\n")
    override.write_text(f"control_hz: 30\nros_setup_path: {json.dumps(str(setup))}\n")
    original_inputs = {path: path.read_bytes() for path in (project, profile, override, setup)}
    loaded = []
    calls = _native(monkeypatch, profile, loaded_configs=loaded)
    monkeypatch.setattr(workflow, "_ros_environment", lambda path: {"ROS_SETUP": str(path)})
    monkeypatch.chdir(launch_dir)
    output = tmp_path / "new_run"

    launcher.main(["workflow", str(profile), "--output", str(output), "--navigation-settings", override.name])

    report = json.loads((output / "workflow_report.json").read_text())
    preview = json.loads((output / "preflight.json").read_text())
    assert report["passed"] and report["declared_inputs_unchanged"]
    assert str(override) in report["declared_input_sha256"]
    assert str(original) not in report["declared_input_sha256"]
    assert preview["navigation"]["settings_path"] == str(override)
    assert preview["pedestrian_parameters"]["control_hz"] == 30
    assert loaded[0].navigation.settings_path == override
    assert calls[0][1] == {"ROS_SETUP": str(setup)}
    assert all(path.read_bytes() == contents for path, contents in original_inputs.items())


@pytest.mark.parametrize("headless", [True, False])
def test_workflow_enables_viewport_when_showing_a_headless_profile(monkeypatch, project, tmp_path, headless):
    profile = tmp_path / "headless.toml"
    profile.write_text('scenario="simulation.yaml"\n[simulation]\nheadless=true\nrender_viewport=false\n')
    calls = _native(monkeypatch, profile)

    report = workflow.run_simulation_workflow(profile, tmp_path / "run", record=False, headless=headless)

    assert report["passed"]
    effective = calls[0][0]["simulation"]
    assert effective["headless"] is headless
    assert effective["render_viewport"] is not headless


@pytest.mark.parametrize("clear_robots", [False, True])
def test_toml_workflow_reloads_the_complete_effective_configuration_without_duplicate_layers(
    monkeypatch, project, tmp_path, clear_robots
):
    base_layer, extra_layer = tmp_path / "base.usda", tmp_path / "extra.usda"
    scene, robot = tmp_path / "override.usda", tmp_path / "robot.usda"
    for path in (base_layer, extra_layer, scene, robot):
        path.write_text('#usda 1.0\n(defaultPrim="Model")\ndef Xform "Model" {}\n')
    data = yaml.safe_load(project.read_text())
    # A valid wrapper can replace a missing base scene; flattening must not
    # revalidate inputs that the selected wrapper does not use.
    data["scene"]["asset_path"] = "unused_missing.usda"
    data["scene"]["layers"] = [str(base_layer)]
    data["robots"] = [{"asset_path": str(robot), "prim_path": "/Base/Robot"}]
    project.write_text(yaml.safe_dump(data))
    population = tmp_path / "population.yaml"
    data["pedestrians"]["actors"] = {"replacement": {"position": [2.0, 3.0, 0.0]}}
    population.write_text(yaml.safe_dump(data))
    navigation = tmp_path / "alternate_navigation.yaml"
    navigation.write_text("body_radius_m: 0.3\n")
    robot_value = (
        "[]"
        if clear_robots
        else '[{asset_path="robot.usda", prim_path="/Selected/Robot", source_prim_path="/Model", '
        "position=[1.0,2.0,0.0], yaw_degrees=37.0}]"
    )
    profile = tmp_path / "recording.toml"
    profile.write_text(
        'scenario="simulation.yaml"\n'
        f"robots={robot_value}\n"
        '[scene]\nasset_path="override.usda"\nlayers=["extra.usda"]\n'
        '[pedestrians]\nconfig_path="population.yaml"\n'
        '[simulation]\nduration=2.5\nwidth=800\nheight=600\nseed=19\nphysics_hz=120\nsolver="PGS"\n'
        'extensions=["omni.example"]\nheadless=true\nconsole=false\nstart_paused=false\n'
        '[navigation]\nbackend="ros2"\nwaypoint_frame="world_m"\nsettings_path="alternate_navigation.yaml"\n'
        '[recording]\nenabled=true\noutput_path="original.mp4"\nfps=20\nwarmup_s=0.15\n'
        "camera_eye_m=[3.0,-4.0,5.0]\ncamera_target_m=[1.0,0.0,0.0]\nmax_stationary_s=7.0\n"
    )
    original = {path: path.read_bytes() for path in (project, profile, population)}
    expected = load_simulation_config(profile)
    loaded = []
    calls = _native(monkeypatch, profile, loaded_configs=loaded)
    output = tmp_path / "recording_run"
    report = workflow.run_simulation_workflow(profile, output)
    assert report["passed"], report["errors"]
    actual = loaded[0]
    assert actual.settings == replace(
        expected.settings, console=False, start_paused=False, headless=True, report_path=output / "status.json"
    )
    assert actual.recording == replace(expected.recording, output_path=output / "video.mp4")
    assert actual.navigation == expected.navigation
    assert actual.pedestrians == expected.pedestrians
    assert actual.scene == expected.scene and actual.scene.layers == (base_layer, extra_layer)
    assert actual.robots == expected.robots
    assert actual.pedestrian_plans == expected.pedestrian_plans
    assert calls[0][0]["scenario"] == str(project)
    assert calls[0][0]["scene"]["layers"] == [str(extra_layer)]
    assert str(profile) in report["declared_input_sha256"]
    assert all(path.read_bytes() == contents for path, contents in original.items())


@pytest.mark.parametrize("failure", ["native", "trajectory", "door", "input_mutation"])
def test_failed_stages_cannot_be_reported_as_a_success(monkeypatch, project, tmp_path, failure):
    _native(
        monkeypatch,
        project,
        code=1 if failure == "native" else 0,
        gap=0.6 if failure == "door" else 0.4,
        evaluation=failure != "trajectory",
        mutate=failure == "input_mutation",
    )
    report = workflow.run_simulation_workflow(project, tmp_path / "failed")
    assert not report["passed"] and report["errors"]
    assert (tmp_path / "failed/native.log").is_file()
    assert json.loads((tmp_path / "failed/workflow_report.json").read_text())["passed"] is False


def test_existing_workflow_directory_is_never_replaced(project, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "user.txt"
    marker.write_text("keep")
    with pytest.raises(ValueError, match="new directory"):
        workflow.run_simulation_workflow(project, output)
    assert marker.read_text() == "keep"


def test_workflow_rejects_output_parent_symlink_loop(project, tmp_path):
    parent = tmp_path / "output"
    parent.symlink_to(parent.name)
    output = parent / "run"

    with pytest.raises(ValueError) as error:
        workflow.run_simulation_workflow(project, output)

    assert str(output) in str(error.value)


@pytest.mark.parametrize("report_writable", [True, False])
def test_cancelled_workflow_preserves_a_failed_report_and_propagates_interrupt(
    monkeypatch, project, tmp_path, report_writable
):
    _native(monkeypatch, project)

    def cancel(profile, log, environment, timeout):
        log.write_text("cancelled owned native run\n")
        raise KeyboardInterrupt("requested cancellation")

    monkeypatch.setattr(workflow, "_run_child", cancel)
    if not report_writable:
        native_open = Path.open

        def report_disk_failure(path, *args, **kwargs):
            if path.name == "workflow_report.json":
                raise OSError("report disk failure")
            return native_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", report_disk_failure)
    output = tmp_path / "cancelled"
    with pytest.raises(KeyboardInterrupt, match="requested cancellation") as caught:
        workflow.run_simulation_workflow(project, output)
    if report_writable:
        report = json.loads((output / "workflow_report.json").read_text())
        assert report["passed"] is False and report["cancelled"] is True
        assert any("KeyboardInterrupt" in error for error in report["errors"])
    else:
        assert any("report disk failure" in note for note in caught.value.__notes__)
    assert (output / "native.log").read_text() == "cancelled owned native run\n"


def test_non_bmp_project_and_output_paths_roundtrip_through_the_generated_toml(monkeypatch, project, tmp_path):
    directory = tmp_path / "场景_🚶"
    directory.mkdir()
    data = yaml.safe_load(project.read_text())
    data["navigation"]["settings_path"] = str(project.with_name("navigation.yaml"))
    profile = directory / "simulation.yaml"
    profile.write_text(yaml.safe_dump(data, allow_unicode=True))
    original = profile.read_bytes()
    _native(monkeypatch, profile)
    output = directory / "录像_📹"
    report = workflow.run_simulation_workflow(profile, output, record=True)
    assert report["passed"], report["errors"]
    wrapper = tomllib.loads((output / "workflow.toml").read_text())
    assert wrapper["scenario"] == str(profile)
    assert wrapper["simulation"]["report_path"] == str(output / "status.json")
    assert wrapper["recording"]["output_path"] == str(output / "video.mp4")
    assert profile.read_bytes() == original


def test_failed_trajectory_keeps_independent_door_video_and_provenance_results(monkeypatch, project, tmp_path):
    _native(monkeypatch, project, gap=0.6, evaluation=False)
    report = workflow.run_simulation_workflow(project, tmp_path / "complete_diagnostics", record=True)
    assert not report["passed"] and len(report["errors"]) >= 2
    assert not report["stages"]["trajectory_evaluation"]["passed"]
    assert not report["stages"]["door_admission"]["passed"]
    assert Path(report["video"]).is_file()
    assert report["source_unchanged"] and report["declared_inputs_unchanged"]


def test_native_timeout_terminates_only_its_created_process_group(monkeypatch, tmp_path):
    signals, kwargs = [], []

    class Process:
        pid = 987654
        attempts = 0

        def wait(self, timeout):
            self.attempts += 1
            if self.attempts == 1:
                raise subprocess.TimeoutExpired("native", timeout)
            return -15

        def poll(self):
            return None

    def start(*args, **options):
        kwargs.append(options)
        return Process()

    monkeypatch.setattr(workflow.subprocess, "Popen", start)
    monkeypatch.setattr(workflow.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    with pytest.raises(subprocess.TimeoutExpired):
        workflow._run_child(tmp_path / "profile.toml", tmp_path / "native.log", {}, 1.0)
    assert kwargs[0]["start_new_session"] is True
    assert signals == [(987654, signal.SIGTERM), (987654, signal.SIGKILL)]


@pytest.mark.skipif(not Path("/proc/self/stat").is_file(), reason="Requires Linux process-state observation.")
@pytest.mark.parametrize("exit_mode", ["timeout", "interrupt", "zero", "nonzero"])
def test_child_cleanup_kills_term_ignoring_descendant_after_its_leader_exits(monkeypatch, tmp_path, exit_mode):
    ready = tmp_path / "child.pid"
    child = tmp_path / "child.py"
    child.write_text(
        "import os, signal, time\nfrom pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"Path({str(ready)!r}).write_text(str(os.getpid()))\ntime.sleep(60)\n"
    )
    parent = tmp_path / "parent.py"
    parent.write_text(
        f"import subprocess, sys, time\nfrom pathlib import Path\nsubprocess.Popen([sys.executable, {str(child)!r}])\n"
        + (
            f"while not Path({str(ready)!r}).exists(): time.sleep(.01)\n"
            f"sys.exit({17 if exit_mode == 'nonzero' else 0})\n"
            if exit_mode in ("zero", "nonzero")
            else "time.sleep(60)\n"
        )
    )
    native_popen, native_killpg = subprocess.Popen, os.killpg
    owned = []

    def spawn(command, **kwargs):
        process = native_popen([sys.executable, str(parent)], **kwargs)
        owned.append(process)
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "Owned child did not initialize."
        if exit_mode == "interrupt":
            wait = process.wait
            first = True

            def interrupt_once(*args, **options):
                nonlocal first
                if first:
                    first = False
                    raise KeyboardInterrupt("cancel owned group")
                return wait(*args, **options)

            process.wait = interrupt_once
        return process

    def child_running():
        state = Path(f"/proc/{int(ready.read_text())}/stat")
        return state.exists() and state.read_text().split()[2] not in ("Z", "X")

    monkeypatch.setattr(workflow.subprocess, "Popen", spawn)
    unrelated = native_popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        if exit_mode in ("timeout", "interrupt"):
            expected = KeyboardInterrupt if exit_mode == "interrupt" else subprocess.TimeoutExpired
            with pytest.raises(expected):
                workflow._run_child(tmp_path / "unused.toml", tmp_path / "native.log", {}, 0.05)
            expected_code = -signal.SIGTERM
        else:
            expected_code = 17 if exit_mode == "nonzero" else 0
            assert workflow._run_child(tmp_path / "unused.toml", tmp_path / "native.log", {}, 5.0) == expected_code
        deadline = time.monotonic() + 2
        while child_running() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert owned[0].poll() == expected_code
        assert not child_running(), "The owned descendant survived its leader's exit."
        assert unrelated.poll() is None, "Cleanup signalled a process outside its owned group."
    finally:
        for process in owned:
            with suppress(ProcessLookupError):
                native_killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        unrelated.kill()
        unrelated.wait(timeout=5)


def test_cli_dispatches_workflow_without_using_the_live_runner(monkeypatch, project, tmp_path, capsys):
    calls = []

    def run(profile, output, **kwargs):
        calls.append((profile, output, kwargs))
        return {"passed": True, "output_directory": str(output), "errors": []}

    monkeypatch.setattr(workflow_cli, "run_simulation_workflow", run)
    launcher.main(["workflow", str(project), "--output", str(tmp_path / "out"), "--record", "--require-route-goals"])
    assert calls[0][2]["record"] and calls[0][2]["require_route_goals"]
    assert "WORKFLOW_RESULT" in capsys.readouterr().out
