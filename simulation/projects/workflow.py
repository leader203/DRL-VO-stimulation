# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run configuration, native execution and offline verification as one workflow."""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

from ..configuration.loading import _load_simulation_config, load_simulation_config
from ..configuration.models import SimulationConfig, _with_ros2_navigation_settings
from ..configuration.navigation import Ros2NavigationSettings, load_ros2_navigation_settings
from ..configuration.paths import resolve_output_path, resolve_simulation_profile
from .alignment import export_coordinate_alignment
from .evaluation import evaluate_ros2_run
from .inspection import _inspect_simulation_config


def _hashes(paths) -> dict[str, str]:
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted({Path(p).resolve() for p in paths if p is not None and "://" not in str(p)})
        if path.is_file()
    }


def _sources() -> dict[str, str]:
    root = Path(__file__).parents[1]
    return _hashes(
        path
        for path in root.rglob("*")
        if path.suffix in (".py", ".cpp", ".h", ".kit")
        and not {"tests", "output", "__pycache__"}.intersection(path.relative_to(root).parts)
    )


def _toml_value(value: object) -> str:
    if isinstance(value, (str, Path)):
        return json.dumps(str(value), ensure_ascii=False)
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key} = {_toml_value(item)}" for key, item in value.items() if item is not None) + "}"
    raise TypeError(f"Cannot serialize workflow TOML value: {type(value).__name__}")


def _workflow_wrapper(config: SimulationConfig) -> str:
    # The loader deliberately allows only a YAML scenario. Flatten wrapper
    # overrides into this private snapshot instead of recursively nesting TOML.
    scenario = next((path for path in config._source_paths if path.suffix.lower() in (".yaml", ".yml")), None)
    if scenario is None:
        raise ValueError("Workflow configuration has no consumed project YAML.")
    source = _load_simulation_config(scenario)
    prefix = source.scene.layers
    if config.scene.layers[: len(prefix)] != prefix:
        raise ValueError("Workflow scene layers do not retain their original YAML prefix.")
    lines = [f"scenario = {_toml_value(scenario)}", f"robots = {_toml_value([asdict(r) for r in config.robots])}"]
    tables = {
        "simulation": asdict(config.settings),
        "recording": asdict(config.recording),
        "navigation": asdict(config.navigation),
        "scene": {"asset_path": config.scene.asset_path, "layers": config.scene.layers[len(prefix) :]},
        "pedestrians": asdict(config.pedestrians),
    }
    for name, values in tables.items():
        lines.extend(("", f"[{name}]"))
        lines.extend(f"{key} = {_toml_value(value)}" for key, value in values.items() if value is not None)
    return "\n".join(lines) + "\n"


def _ros_environment(setup: Path) -> dict[str, str]:
    """Prepare only the child's environment before importing any native SDK."""
    if not setup.is_file():
        raise ValueError(f"ROS setup script not found: {setup}")
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", 'set -e\nsource "$1" >/dev/null\nenv -0', "ros-workflow", str(setup)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    environment = {}
    for entry in result.stdout.split(b"\0"):
        if b"=" in entry:
            key, value = entry.split(b"=", 1)
            environment[os.fsdecode(key)] = os.fsdecode(value)
    source = str(Path(__file__).resolve().parents[3])
    environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
    return environment


def _run_child(profile: Path, log: Path, environment: dict[str, str], timeout_s: float) -> int:
    command = [sys.executable, str(Path(__file__).parents[1] / "run.py"), "run", str(profile)]
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command, env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        primary: BaseException | None = None
        cleanup_errors: list[BaseException] = []
        try:
            return process.wait(timeout=timeout_s)
        except BaseException as error:
            primary = error
            raise
        finally:
            # A native crash can exit normally through wait() while its ROS
            # descendant survives. This group belongs to every exit path.
            def remember(message: str, failure: BaseException) -> None:
                if primary is None:
                    failure.add_note(message)
                    cleanup_errors.append(failure)
                else:
                    primary.add_note(message)

            try:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                except Exception as error:
                    remember(f"Cannot terminate owned process group {process.pid}: {error}", error)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                except Exception as error:
                    remember(f"Cannot wait for owned process {process.pid}: {error}", error)
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except Exception as error:
                    remember(f"Cannot kill owned process group {process.pid}: {error}", error)
                try:
                    process.wait(timeout=10)
                except Exception as error:
                    remember(f"Owned process group {process.pid} did not exit: {error}", error)
            if primary is None and cleanup_errors:
                for failure in cleanup_errors[1:]:
                    cleanup_errors[0].add_note(f"Additional owned-group cleanup failure: {failure}")
                raise cleanup_errors[0]


def run_simulation_workflow(
    project: Path,
    output: Path,
    *,
    record: bool | None = None,
    headless: bool = True,
    require_route_goals: bool = False,
    timeout_s: float | None = None,
    navigation_settings: Path | None = None,
) -> dict[str, object]:
    """Run a scene-independent ROS workflow and preserve its complete evidence.

    Args:
        project: User scenario YAML, launcher TOML or project directory.
        output: New directory; existing outputs are never replaced.
        record: Override recording selection, otherwise retain the user setting.
            Enabled videos retain FPS, warmup and camera, with a fresh local path.
        headless: Hide the native window; capture still renders when enabled.
        require_route_goals: Fail if the configured duration did not cover every
            original walking waypoint. Normally incomplete routes are reported.
        timeout_s: Optional wall-time limit [s] for this owned native process group.
        navigation_settings: Optional ROS 2 navigation YAML override. Relative
            paths resolve from the caller's working directory.

    Returns:
        JSON-compatible workflow result with preflight, alignment, trajectory and
        output provenance. Native/postprocessing failures have ``passed=False``.

    Raises:
        ValueError: Invalid inputs, unsupported backend or non-fresh destination.
    """
    project = resolve_simulation_profile(project)
    output = resolve_output_path(output)
    if output.exists() or output.is_symlink():
        raise ValueError(f"Workflow output must be a new directory: {output}")
    if navigation_settings is None:
        config = load_simulation_config(project)
    else:
        config = _with_ros2_navigation_settings(_load_simulation_config(project), navigation_settings)
        config.validate()
    if config.navigation.backend != "ros2":
        raise ValueError("The automated coordinate/map workflow requires navigation.backend=ros2.")
    if config.settings.duration <= 0 or not math.isfinite(config.settings.duration):
        raise ValueError("Automated workflows require a positive finite simulation duration [s].")
    enabled = config.recording.enabled if record is None else record
    if type(enabled) is not bool or type(headless) is not bool or type(require_route_goals) is not bool:
        raise ValueError("Workflow recording, headless and route-goal selections must be boolean.")
    budget = (
        timeout_s
        if timeout_s is not None
        else max(300.0, 180 + (config.settings.duration + config.recording.warmup_s) * 20)
    )
    if type(budget) not in (int, float) or not math.isfinite(budget) or budget <= 0:
        raise ValueError("Workflow timeout_s must be positive and finite [s].")
    effective = replace(
        config,
        settings=replace(
            config.settings,
            headless=headless,
            console=False,
            start_paused=False,
            render_viewport=enabled or not headless or config.settings.render_viewport,
            report_path=output / "status.json",
        ),
        recording=replace(
            config.recording,
            enabled=enabled,
            output_path=output / "video.mp4" if enabled else config.recording.output_path,
        ),
    )
    preview = _inspect_simulation_config(effective, project)
    navigation = (
        load_ros2_navigation_settings(config.navigation.settings_path)
        if config.pedestrians.enabled
        else Ros2NavigationSettings()
    )
    environment = _ros_environment(Path(navigation.ros_setup_path)) if config.pedestrians.enabled else dict(os.environ)
    declared = preview["dependencies"]
    inputs = _hashes([*declared["configuration_files"], *declared["assets"]])
    source = _sources()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    wrapper = output / "workflow.toml"
    wrapper.write_text(_workflow_wrapper(effective), encoding="utf-8")
    inputs.update(_hashes([wrapper]))
    (output / "preflight.json").write_text(json.dumps(preview, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = {
        "passed": False,
        "cancelled": False,
        "project": str(project),
        "output_directory": str(output),
        "requested_duration_s": config.settings.duration,
        "recording_enabled": enabled,
        "require_route_goals": require_route_goals,
        "timeout_s": budget,
        "source_sha256": source,
        "declared_input_sha256": inputs,
        "input_hash_scope": "declared local files; referenced directories are not recursively hashed",
        "stages": {"preflight": {"passed": True}},
        "errors": [],
    }

    def stage(name, action):
        try:
            result = action()
        except Exception as error:
            result = {"passed": False, "errors": [f"{type(error).__name__}: {error}"]}
        report["stages"][name] = result
        if result.get("passed") is False:
            errors = result.get("errors") or [f"{name} failed"]
            report["errors"].extend(f"{name}: {error}" for error in errors)
        return result

    interrupted = None
    try:
        print("WORKFLOW_STAGE native_execution", flush=True)
        code = _run_child(wrapper, output / "native.log", environment, budget)
        status_path = output / "status.json"
        status = json.loads(status_path.read_text()) if status_path.is_file() else {}
        report["stages"]["native_execution"] = {
            "passed": code == 0 and status.get("passed") is True,
            "exit_code": code,
            "status_path": str(status_path),
        }
        if code != 0 or status.get("passed") is not True:
            raise RuntimeError(f"Native execution failed ({code}): {status.get('error', 'inspect native.log')}")
        runtime = Path(status["output_directory"])
        if config.pedestrians.enabled:
            print("WORKFLOW_STAGE coordinate_alignment", flush=True)
            stage(
                "coordinate_alignment",
                lambda: export_coordinate_alignment(
                    runtime / "bootstrap.json", runtime / "map.npz", output / "alignment"
                ),
            )
            print("WORKFLOW_STAGE trajectory_evaluation", flush=True)

            evaluation = stage(
                "trajectory_evaluation", lambda: evaluate_ros2_run(runtime, require_route_goals=require_route_goals)
            )
            (output / "evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n")

            def door_admission():
                events = [
                    event
                    for line in (runtime / "trace.jsonl").read_text().splitlines()
                    for event in json.loads(line)["command"].get("door_events", [])
                ]
                invalid = [event for event in events if event["distance_m"] > event["approach_distance_m"] + 1e-8]
                return {
                    "passed": not invalid,
                    "events": events,
                    "invalid_events": invalid,
                    "errors": ["Door admitted outside configured approach distance."] if invalid else [],
                }

            stage("door_admission", door_admission)
        else:
            clean = status.get("cleanup", {})
            if not status.get("completed_duration") or not clean or not all(clean.values()):
                raise RuntimeError("The render-only workflow did not complete or clean up.")
            report["stages"]["coordinate_alignment"] = {
                "skipped": True,
                "reason": "pedestrians disabled; no navigation map",
            }
        if enabled:
            video = status.get("recording", {})
            if not video.get("passed") or not Path(video.get("video", "")).is_file():
                report["errors"].append("The configured recording did not produce a verified video.")
            else:
                report["video"] = video["video"]
        report["source_unchanged"] = _hashes(source) == source
        report["declared_inputs_unchanged"] = _hashes(inputs) == inputs
        if not report["source_unchanged"] or not report["declared_inputs_unchanged"]:
            report["errors"].append("Source or declared input files changed during the workflow.")
        report["passed"] = not report["errors"]
    except KeyboardInterrupt as error:
        report["passed"] = False
        report["cancelled"] = True
        report["errors"].append(f"KeyboardInterrupt: {error}")
        report["errors"].extend(getattr(error, "__notes__", []))
        interrupted = error
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
        report["errors"].extend(getattr(error, "__notes__", []))
    try:
        with (output / "workflow_report.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    except Exception as error:
        if interrupted is None:
            raise
        interrupted.add_note(f"Cannot save cancelled workflow report: {error}")
    if interrupted is not None:
        raise interrupted
    return report
