# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export checked coordinate tables and an SDK-free ROS occupancy diagram."""

from __future__ import annotations

import hashlib
import html
import io
import json
import math
from pathlib import Path

import numpy as np

from ..configuration._validation import _finite_number
from ..ros2_navigation.frames import StageFrame

_TOLERANCE_M = 1e-8


def _point(value: object, label: str, size: int = 3) -> np.ndarray:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != size
        or any(type(part) not in (int, float) or not math.isfinite(part) for part in value)
    ):
        raise ValueError(f"{label} requires {size} finite coordinates.")
    return np.asarray(value, dtype=float)


def _plans(value: object, label: str) -> dict[str, dict]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must contain a nonempty plan list.")
    result = {}
    for plan in value:
        name = plan.get("name") if isinstance(plan, dict) else None
        if not isinstance(name, str) or not name or name in result:
            raise ValueError(f"{label} requires unique, nonempty actor names.")
        if "position" not in plan:
            raise ValueError(f"{label}.{name}.position is required; sampled input positions are explicitly null.")
        if not isinstance(plan.get("waypoints"), list):
            raise ValueError(f"{label}.{name}.waypoints must be a list.")
        result[name] = plan
    return result


def _aligned_point(raw, world, ros, frame: StageFrame, input_frame: str, label: str, *, spawn_adjustment=None) -> dict:
    world = _point(world, f"{label}.world_m")
    ros = _point(ros, f"{label}.ros_m")
    stage = np.asarray(frame.world_to_stage(world))
    residuals = {
        "world_to_ros_m": float(np.linalg.norm(np.asarray(frame.world_to_ros(world)) - ros)),
        "ros_world_roundtrip_m": float(np.linalg.norm(np.asarray(frame.ros_to_world(ros)) - world)),
        "stage_world_roundtrip_m": float(np.linalg.norm(np.asarray(frame.stage_to_world(stage)) - world)),
        "stage_ros_roundtrip_m": float(np.linalg.norm(np.asarray(frame.stage_to_ros(frame.ros_to_stage(ros))) - ros)),
    }
    if raw is not None:
        original = _point(raw, f"{label}.input_position")
        expected = np.asarray(frame.stage_to_world(original) if input_frame == "stage_world" else original)
        restored_world = np.asarray(frame.ros_to_world(ros))
        if spawn_adjustment is not None:
            expected = expected + spawn_adjustment
            restored_world -= spawn_adjustment
        residuals["input_to_world_m"] = float(np.linalg.norm(np.asarray(expected) - world))
        restored = frame.world_to_stage(restored_world) if input_frame == "stage_world" else restored_world
        scale = frame.meters_per_unit if input_frame == "stage_world" else 1.0
        residuals["input_ros_roundtrip_m"] = float(np.linalg.norm(np.asarray(restored) - original) * scale)
    error = max(residuals.values())
    if error > _TOLERANCE_M:
        raise ValueError(f"{label}: coordinate alignment error {error:.12g} m exceeds {_TOLERANCE_M:g} m.")
    result = {
        "input_position": raw,
        "input_frame": input_frame,
        "world_m": world.tolist(),
        "ros_m": ros.tolist(),
        "stage_world": stage.tolist(),
        "sampled_spawn": raw is None,
        "residuals_m": residuals,
        "maximum_error_m": error,
    }
    if spawn_adjustment is not None:
        result["spawn_adjustment_world_m"] = spawn_adjustment.tolist()
        result["spawn_adjustment_distance_m"] = float(np.linalg.norm(spawn_adjustment))
    return result


def _aligned_seconds(points, field: str, label: str) -> float:
    input_s, world_s, ros_s = (point.get(field, 0.0) for point in points)
    if (
        any(not _finite_number(value) or value < 0 for value in (input_s, world_s, ros_s))
        or input_s != world_s
        or input_s != ros_s
    ):
        raise ValueError(f"{label}: {field} [s] differs across input, world and ROS plans or is invalid.")
    return float(input_s)


def _aligned_waypoint_heading(points, map_yaw_rad: float, label: str) -> tuple[float | None, float | None]:
    input_yaw, world_yaw, ros_yaw = (point.get("yaw") for point in points)
    if input_yaw is world_yaw is ros_yaw is None:
        return None, None
    if (
        not all(_finite_number(value) for value in (input_yaw, world_yaw, ros_yaw))
        or abs(math.remainder(input_yaw - world_yaw, 360.0)) > 1e-8
        or abs(math.remainder(world_yaw - math.degrees(map_yaw_rad) - ros_yaw, 360.0)) > 1e-8
    ):
        raise ValueError(f"{label}: waypoint yaw [deg] disagrees with the map frame.")
    return float(world_yaw), float(ros_yaw)


def _svg(occupancy: np.ndarray, resolution: float, origin: np.ndarray, actors: list[dict], contract: dict) -> str:
    height, width = occupancy.shape
    points = [p["ros_m"][:2] for actor in actors for p in [actor["spawn"], *actor["waypoints"]]]
    low = np.minimum(origin, np.min(points, axis=0)) - resolution
    high = np.maximum(origin + np.array([width, height]) * resolution, np.max(points, axis=0)) + resolution
    span = high - low
    scale = min(1120 / span[0], 760 / span[1])
    total_height = 185 + span[1] * scale

    def pixel(point):
        return 40 + (point[0] - low[0]) * scale, 130 + (high[1] - point[1]) * scale

    def text(x, y, value, **attrs):
        extra = " ".join(
            f'{key.replace("_", "-")}="{html.escape(str(value), quote=True)}"' for key, value in attrs.items()
        )
        return f'<text x="{x:.6f}" y="{y:.6f}" {extra}>{html.escape(str(value))}</text>'

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" '
        f'height="{total_height:.3f}" viewBox="0 0 1200 {total_height:.3f}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="sans-serif" font-size="14" fill="#17202a">',
        text(40, 28, "Coordinate alignment — original spawn / waypoint locations", font_size=20),
        text(
            40,
            53,
            f"Input: {contract['input_waypoint_frame']}; stage: {contract['stage_meters_per_unit']} m/unit, "
            f"{contract['stage_up_axis']}-up; displayed coordinates: ROS map [m]",
        ),
        text(
            40,
            75,
            f"Map origin in canonical world [m]: {contract['map_origin_world_m']}; "
            f"map yaw [rad]: {contract['map_yaw_rad']}",
        ),
        text(40, 97, "Circle = spawn; diamond = original goal; dashed connections are NOT planned or executed paths."),
    ]
    colors = {0: "#ffffff", -1: "#d9dde2", 100: "#4a515a"}
    # Merge equal horizontal runs vertically, preserving row zero at minimum ROS Y.
    active = {}
    rectangles = []
    for row in range(height + 1):
        runs = set()
        if row < height:
            begin = 0
            for column in range(1, width + 1):
                if column == width or occupancy[row, column] != occupancy[row, begin]:
                    runs.add((begin, column - begin, int(occupancy[row, begin])))
                    begin = column
        for key in sorted(active.keys() - runs):
            rectangles.append((*key, active[key], row - active[key]))
        active = {key: active.get(key, row) for key in sorted(runs)}
    for column, columns, value, row, rows in rectangles:
        x, y = pixel(origin + np.array([column, row + rows]) * resolution)
        parts.append(
            f'<rect data-occupancy="{value}" x="{x:.6f}" y="{y:.6f}" '
            f'width="{columns * resolution * scale:.6f}" height="{rows * resolution * scale:.6f}" '
            f'fill="{colors[value]}"/>'
        )
    palette = ("#1464a0", "#c04d17", "#267444", "#873ea3", "#ad3975", "#64701c")
    for index, actor in enumerate(actors):
        color = palette[index % len(palette)]
        route = [actor["spawn"], *actor["waypoints"]]
        coordinates = " ".join(f"{x:.6f},{y:.6f}" for x, y in (pixel(p["ros_m"]) for p in route))
        if len(route) > 1:
            parts.append(
                f'<polyline data-role="original-waypoint-connections" points="{coordinates}" '
                f'fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="6 5"/>'
            )
        for goal_index, point in enumerate(route):
            x, y = pixel(point["ros_m"])
            if goal_index == 0:
                parts.append(
                    f'<circle cx="{x:.6f}" cy="{y:.6f}" r="6" fill="white" stroke="{color}" stroke-width="3"/>'
                )
            else:
                vertices = f"{x:.6f},{y - 6:.6f} {x + 6:.6f},{y:.6f} {x:.6f},{y + 6:.6f} {x - 6:.6f},{y:.6f}"
                parts.append(f'<polygon points="{vertices}" fill="{color}" stroke="white"/>')
            label = "spawn" if goal_index == 0 else f"goal {goal_index}"
            parts.append(text(x + 9, y - 8, f"{actor['name']} {label}", fill=color, font_size=12))
    parts.extend(
        (
            text(
                40,
                total_height - 25,
                f"ROS +X →; ROS +Y ↑; raster row 0 = minimum ROS Y; cell size {resolution:g} m; "
                f"grid origin {origin.tolist()} [m]",
            ),
            "</g></svg>",
        )
    )
    return "\n".join(parts) + "\n"


def export_coordinate_alignment(bootstrap: Path, map_npz: Path, output_dir: Path) -> dict:
    """Export original and converted pedestrian positions without an SDK import.

    Args:
        bootstrap: Runtime JSON with input, canonical-world and ROS-map plans.
        map_npz: Runtime occupancy map with resolution/origin [m].
        output_dir: New output directory; existing paths are never overwritten.

    Returns:
        JSON-compatible receipt with absolute JSON/SVG paths and errors [m].

    Raises:
        ValueError: Metadata, actor pairing or coordinate residuals are invalid.
        FileExistsError: The requested output directory already exists.
    """
    output_dir = Path(output_dir).expanduser()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Coordinate output must be new: {output_dir}")
    bootstrap, map_npz, output_dir = (Path(p).expanduser().resolve() for p in (bootstrap, map_npz, output_dir))
    contents = {path: path.read_bytes() for path in (bootstrap, map_npz)}
    document = json.loads(contents[bootstrap])
    if not isinstance(document, dict):
        raise ValueError("Bootstrap must contain a JSON object.")
    contract = document.get("coordinate_contract")
    if not isinstance(contract, dict) or contract.get("input_waypoint_frame") not in ("world_m", "stage_world"):
        raise ValueError("Bootstrap requires an explicit coordinate_contract.input_waypoint_frame.")
    if contract.get("effective_plans_frame") != "world_m" or contract.get("bootstrap_plans_frame") != "ros_map":
        raise ValueError("Bootstrap canonical-world / ROS plan frame labels are inconsistent.")
    try:
        frame = StageFrame(
            contract["stage_meters_per_unit"],
            contract["stage_up_axis"],
            contract["map_origin_world_m"],
            contract["map_yaw_rad"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid bootstrap stage/map coordinate metadata: {error}") from error
    raw, world, ros = (_plans(document.get(key), key) for key in ("input_plans", "world_plans", "plans"))
    if raw.keys() != world.keys() or raw.keys() != ros.keys():
        raise ValueError("Input, canonical-world and ROS plans must have the same actor names/counts.")
    adjustments = document.get("spawn_adjustments_world_m", {})
    if not isinstance(adjustments, dict) or adjustments.keys() - raw.keys():
        raise ValueError("Spawn adjustments must reference known fixed actors.")
    if adjustments:
        settings = document.get("settings")
        tolerance = settings.get("spawn_tolerance_m") if isinstance(settings, dict) else None
        if not _finite_number(tolerance) or tolerance <= 0:
            raise ValueError("Recorded spawn adjustments require a positive spawn_tolerance_m.")
    actors = []
    for name, original in raw.items():
        targets = [p["waypoints"] for p in (original, world[name], ros[name])]
        if len({len(points) for points in targets}) != 1:
            raise ValueError(f"{name}: input/world/ROS waypoint counts differ.")
        adjustment = None
        if name in adjustments:
            adjustment = _point(adjustments[name], f"{name}.spawn_adjustment_world_m")
            if (
                original.get("position") is None
                or abs(adjustment[2]) > _TOLERANCE_M
                or np.linalg.norm(adjustment) > tolerance + _TOLERANCE_M
            ):
                raise ValueError(f"{name}: fixed spawn adjustment is outside its configured planar tolerance.")
        spawn = _aligned_point(
            original.get("position"),
            world[name].get("position"),
            ros[name].get("position"),
            frame,
            contract["input_waypoint_frame"],
            f"{name}.spawn",
            spawn_adjustment=adjustment,
        )
        waypoints = []
        for index, points in enumerate(zip(*targets)):
            label = f"{name}.waypoints[{index}]"
            if any(not isinstance(point, dict) or point.get("position") is None for point in points):
                raise ValueError(f"{label} requires three paired positions.")
            point = _aligned_point(*(p["position"] for p in points), frame, contract["input_waypoint_frame"], label)
            yaw_world, yaw_ros = _aligned_waypoint_heading(points, frame.yaw_rad, label)
            point.update(
                index=index,
                wait_s=_aligned_seconds(points, "wait", label),
                yaw_world_deg=yaw_world,
                yaw_ros_deg=yaw_ros,
            )
            waypoints.append(point)
        start = _aligned_seconds((original, world[name], ros[name]), "start_time", name)
        actors.append({"name": name, "start_time_s": start, "spawn": spawn, "waypoints": waypoints})
    try:
        with np.load(io.BytesIO(contents[map_npz]), allow_pickle=False) as archive:
            occupancy = archive["occupancy"]
            resolution = float(archive["resolution_m"])
            origin = _point(archive["origin_xy_m"].tolist(), "map origin", 2)
            diagnostics = json.loads(str(archive["diagnostics"]))
        if not isinstance(diagnostics, dict):
            raise ValueError("Map diagnostics must contain an object.")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid occupancy-map metadata: {error}") from error
    if occupancy.ndim != 2 or not occupancy.size or not np.isin(occupancy, [-1, 0, 100]).all():
        raise ValueError("Map occupancy must contain a nonempty 2D -1/0/100 raster.")
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("Map resolution must be finite and positive [m].")
    if (
        diagnostics.get("meters_per_unit") != frame.meters_per_unit
        or diagnostics.get("up_axis") != frame.up_axis
        or diagnostics.get("map_yaw_rad") != frame.yaw_rad
        or not np.allclose(
            _point(diagnostics.get("map_origin_world_m"), "map frame origin"), frame.origin_m, rtol=0, atol=_TOLERANCE_M
        )
    ):
        raise ValueError("Map stage/world/ROS frame metadata differs from the bootstrap coordinate contract.")
    json_path, svg_path = output_dir / "coordinate_alignment.json", output_dir / "coordinate_alignment.svg"
    receipt = {
        "passed": True,
        "coordinate_contract": contract,
        "actors": actors,
        "maximum_error_m": max(
            point["maximum_error_m"] for actor in actors for point in [actor["spawn"], *actor["waypoints"]]
        ),
        "tolerance_m": _TOLERANCE_M,
        "map": {
            "origin_xy_m": origin.tolist(),
            "resolution_m": resolution,
            "shape": list(occupancy.shape),
            "row_direction": "ROS +Y",
            "column_direction": "ROS +X",
        },
        "input_sha256": {str(path): hashlib.sha256(data).hexdigest() for path, data in contents.items()},
        "json_path": str(json_path),
        "svg_path": str(svg_path),
    }
    serialized = json.dumps(receipt, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    diagram = _svg(occupancy, resolution, origin, actors, contract)
    output_dir.mkdir(parents=True, exist_ok=False)
    with json_path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    with svg_path.open("x", encoding="utf-8") as stream:
        stream.write(diagram)
    return receipt
