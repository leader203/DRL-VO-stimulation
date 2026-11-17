# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Create and launch per-session RViz monitoring without starting a simulator."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import yaml

from .visualization import monitor_color, monitor_world_frame


def write_monitor_files(bootstrap: Path, output: Path, namespace: str, settings: dict) -> Path:
    """Write session-bound RViz config and a read-only monitor manifest.

    The map origin/resolution [m] determine the initial view. All displays use
    ``map`` and transient-local reliable topics. A read-only static TF documents
    canonical-world alignment; no navigation tools run.

    Args:
        bootstrap: Existing ROS bootstrap JSON containing plans and map location.
        output: Owned runtime output directory.
        namespace: Actual ROS session namespace, including its unique suffix.
        settings: Effective ROS setup path and domain settings.

    Returns:
        Absolute path to the generated ``monitor.json`` manifest.
    """
    output = output.resolve()
    data = json.loads(bootstrap.read_text())
    map_path = Path(data["map_npz"])
    if not map_path.is_absolute():
        map_path = bootstrap.parent / map_path
    with np.load(map_path, allow_pickle=False) as grid:
        height, width = grid["occupancy"].shape
        resolution = float(grid["resolution_m"])
        origin = grid["origin_xy_m"].tolist()
    namespace = "/" + namespace.strip("/")
    topics = {
        "map": f"{namespace}/map",
        "markers": f"{namespace}/markers",
        "clock": "/clock",
        "tf_static": "/tf_static",
    }
    paths = {plan["name"]: f"{namespace}/{plan['name']}/path" for plan in data["plans"]}

    def topic(value):
        return {
            "Value": value,
            "Depth": 1,
            "Durability Policy": "Transient Local",
            "History Policy": "Keep Last",
            "Reliability Policy": "Reliable",
        }

    displays = [
        {
            "Class": "rviz_default_plugins/Map",
            "Name": "Static map",
            "Enabled": True,
            "Topic": topic(topics["map"]),
            "Alpha": 0.65,
            "Draw Behind": True,
            "Use Timestamp": False,
        },
        {
            "Class": "rviz_default_plugins/MarkerArray",
            "Name": "Observed scene",
            "Enabled": True,
            "Topic": topic(topics["markers"]),
        },
    ]
    for name, value in paths.items():
        color = "; ".join(str(round(part * 255)) for part in monitor_color(name))
        displays.append(
            {
                "Class": "rviz_default_plugins/Path",
                "Name": f"Route: {name}",
                "Enabled": True,
                "Topic": topic(value),
                "Color": color,
                "Line Width": 0.035,
                "Line Style": "Billboards",
            }
        )
    config = {
        "Panels": [{"Class": "rviz_common/Displays", "Name": "Displays"}],
        "Visualization Manager": {
            "Class": "",
            "Enabled": True,
            "Global Options": {"Fixed Frame": "map", "Background Color": "35; 38; 43"},
            "Displays": displays,
            "Tools": [
                {"Class": f"rviz_default_plugins/{tool}"}
                for tool in ("Interact", "MoveCamera", "Select", "FocusCamera", "Measure")
            ],
            "Views": {
                "Current": {
                    "Class": "rviz_default_plugins/TopDownOrtho",
                    "Name": "Map overview",
                    "Target Frame": "<Fixed Frame>",
                    "Angle": 0.0,
                    "X": origin[0] + width * resolution / 2,
                    "Y": origin[1] + height * resolution / 2,
                    "Scale": 500 / max(width * resolution, height * resolution, 1.0),
                }
            },
        },
        "Window Geometry": {"Width": 1600, "Height": 1000},
    }
    config_path = output / "monitor.rviz"
    manifest_path = output / "monitor.json"
    for path in (config_path, manifest_path):
        if path.exists() or path.is_symlink():
            raise ValueError(f"Monitor output already exists: {path}")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    manifest = {
        "schema_version": 1,
        "read_only": True,
        "frame_id": "map",
        "canonical_world_frame": monitor_world_frame(namespace),
        "namespace": namespace,
        "ros_domain_id": int(settings["ros_domain_id"]),
        "ros_setup_path": str(settings["ros_setup_path"]),
        "runtime_output": str(output),
        "rviz_config": config_path.name,
        "topics": {**topics, "paths": paths},
        "marker_rate_hz": 10,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def _resolve_manifest(path: Path, seen: set[Path] | None = None) -> Path:
    path = path.expanduser().resolve()
    seen = set() if seen is None else seen
    if path in seen or len(seen) > 8:
        raise ValueError("Monitor status references contain a cycle or are too deeply nested.")
    seen.add(path)
    if path.is_dir():
        for relative in ("monitor.json", "runtime/monitor.json", "status.json", "result.json"):
            candidate = path / relative
            if candidate.is_file():
                return _resolve_manifest(candidate, seen)
        raise ValueError(f"No monitor.json or runtime status in {path}; start a new ROS simulation run.")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Monitor status must be a JSON object: {path}")
    if data.get("schema_version") == 1 and "rviz_config" in data:
        return path
    for record in (data, data.get("final_status", {}), data.get("recording", {})):
        if not isinstance(record, dict):
            continue
        for field in ("output_directory", "runtime_output", "runtime_directory"):
            value = record.get(field)
            if isinstance(value, str):
                target = Path(value)
                return _resolve_manifest(target if target.is_absolute() else path.parent / target, seen)
    raise ValueError(f"No ROS runtime monitor reference in {path}.")


def launch_ros2_monitor(path: Path, *, dry_run: bool = False) -> dict[str, object]:
    """Open an existing live ROS session in RViz; never start Kit or control motion.

    Args:
        path: Runtime directory, monitor manifest or saved runtime status JSON.
        dry_run: Prepare the environment and return the exact command without a GUI.

    Returns:
        Command, session/domain binding and exit code when RViz was launched.
    """
    from ..projects.workflow import _ros_environment

    manifest_path = _resolve_manifest(path)
    manifest = json.loads(manifest_path.read_text())
    config_path = (manifest_path.parent / manifest["rviz_config"]).resolve()
    if not config_path.is_relative_to(manifest_path.parent) or not config_path.is_file():
        raise ValueError("The monitor RViz config must be an existing file inside its runtime directory.")
    domain = manifest["ros_domain_id"]
    if type(domain) is not int or not 0 <= domain <= 232 or manifest.get("read_only") is not True:
        raise ValueError("Invalid read-only ROS monitor manifest.")
    setup = Path(manifest["ros_setup_path"])
    environment = _ros_environment(setup)
    environment["ROS_DOMAIN_ID"] = str(domain)
    executable = shutil.which("rviz2", path=environment.get("PATH"))
    if executable is None:
        raise ValueError(f"rviz2 was not found after sourcing {setup}.")
    command = [executable, "-d", str(config_path), "--ros-args", "-p", "use_sim_time:=true"]
    result = {
        "read_only": True,
        "monitor_manifest": str(manifest_path),
        "rviz_config": str(config_path),
        "namespace": manifest["namespace"],
        "ros_domain_id": domain,
        "ros_setup_path": str(setup),
        "command": command,
        "dry_run": dry_run,
        "requires_live_session": True,
    }
    if not dry_run:
        result["exit_code"] = subprocess.run(command, env=environment, check=False).returncode
    return result


def main(argv: list[str] | None = None) -> None:
    """Launch the live-session monitor CLI without importing the native simulator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Runtime directory, monitor.json or saved status JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Print the exact RViz command without opening a GUI.")
    args = parser.parse_args(argv)
    try:
        result = launch_ros2_monitor(args.run, dry_run=args.dry_run)
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result.get("exit_code", 0):
        raise SystemExit(result["exit_code"])
