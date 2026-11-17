# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SDK-free coordinate alignment output and diagram orientation contracts."""

import copy
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from pedstrain_simulation.simulation.projects.alignment import export_coordinate_alignment


def _inputs(tmp_path, axis="Z", units=1.0, input_frame="stage_world", name="walker"):
    world_points = [[1.25, -2.5, 0.75], [3.0, 1.0, 0.75], [-0.5, 2.0, 0.75]]
    map_origin, angle = [0.5, -1.0, 0.25], 0.43
    cosine, sine = math.cos(angle), math.sin(angle)

    def raw(point):
        x, y, z = point
        return (
            [(x if k == 0 else (z if k == 1 else -y)) / units for k in range(3)]
            if axis == "Y"
            else [v / units for v in point]
        )

    def ros(point):
        x, y, z = [p - origin for p, origin in zip(point, map_origin)]
        return [cosine * x + sine * y, -sine * x + cosine * y, z]

    def plan(points):
        return {"name": name, "position": points[0], "waypoints": [{"position": p} for p in points[1:]]}

    contract = {
        "input_waypoint_frame": input_frame,
        "effective_plans_frame": "world_m",
        "bootstrap_plans_frame": "ros_map",
        "stage_meters_per_unit": units,
        "stage_up_axis": axis,
        "map_origin_world_m": map_origin,
        "map_yaw_rad": angle,
    }
    data = {
        "input_plans": [plan([raw(p) for p in world_points] if input_frame == "stage_world" else world_points)],
        "world_plans": [plan(world_points)],
        "plans": [plan([ros(p) for p in world_points])],
        "coordinate_contract": contract,
    }
    bootstrap, map_path = tmp_path / "bootstrap.json", tmp_path / "map.npz"
    bootstrap.write_text(json.dumps(data))
    diagnostics = {"meters_per_unit": units, "up_axis": axis, "map_origin_world_m": map_origin, "map_yaw_rad": angle}
    np.savez(
        map_path,
        occupancy=np.array([[100, 0], [-1, 0]], dtype=np.int8),
        resolution_m=1.0,
        origin_xy_m=[-1.0, -1.0],
        diagnostics=json.dumps(diagnostics),
    )
    return bootstrap, map_path, data, world_points


@pytest.mark.parametrize("axis,units", [("Z", 1.0), ("Z", 0.01), ("Y", 0.01)])
@pytest.mark.parametrize("input_frame", ["stage_world", "world_m"])
def test_alignment_pairs_original_metric_ros_and_native_stage_positions(tmp_path, axis, units, input_frame):
    bootstrap, map_path, original, world = _inputs(tmp_path, axis, units, input_frame)
    before = (bootstrap.read_bytes(), map_path.read_bytes())
    report = export_coordinate_alignment(bootstrap, map_path, tmp_path / "alignment")
    assert report["passed"] and report["maximum_error_m"] < 1e-12
    actor = report["actors"][0]
    for index, actual in enumerate([actor["spawn"], *actor["waypoints"]]):
        assert actual["world_m"] == pytest.approx(world[index])
        expected_stage = [world[index][0], world[index][2], -world[index][1]] if axis == "Y" else world[index]
        assert actual["stage_world"] == pytest.approx(np.asarray(expected_stage) / units)
        raw = original["input_plans"][0]
        expected_raw = raw["position"] if index == 0 else raw["waypoints"][index - 1]["position"]
        assert actual["input_position"] == expected_raw
        assert not actual["sampled_spawn"]
    assert json.loads(Path(report["json_path"]).read_text()) == report
    assert (bootstrap.read_bytes(), map_path.read_bytes()) == before


def test_svg_places_minimum_ros_y_at_the_bottom_and_escapes_actor_text(tmp_path):
    name = '<walker & "foreign">'
    bootstrap, map_path, _, _ = _inputs(tmp_path, name=name)
    report = export_coordinate_alignment(bootstrap, map_path, tmp_path / "alignment")
    svg = Path(report["svg_path"]).read_text()
    root = ET.fromstring(svg)
    rectangles = {
        r.attrib["data-occupancy"]: r
        for r in root.iter("{http://www.w3.org/2000/svg}rect")
        if "data-occupancy" in r.attrib
    }
    assert float(rectangles["100"].attrib["y"]) > float(rectangles["-1"].attrib["y"])
    text = " ".join(root.itertext())
    assert name in text and "NOT planned or executed paths" in text
    assert "&lt;walker &amp;" in svg and "<walker" not in svg
    assert len(list(root.iter("{http://www.w3.org/2000/svg}circle"))) == 1
    assert len(list(root.iter("{http://www.w3.org/2000/svg}polygon"))) == 2
    assert report["map"]["row_direction"] == "ROS +Y"


def test_sampled_spawn_keeps_its_null_input_without_inventing_an_authored_position(tmp_path):
    bootstrap, map_path, data, _ = _inputs(tmp_path)
    data["input_plans"][0]["position"] = None
    bootstrap.write_text(json.dumps(data))
    report = export_coordinate_alignment(bootstrap, map_path, tmp_path / "alignment")
    spawn = report["actors"][0]["spawn"]
    assert spawn["input_position"] is None and spawn["sampled_spawn"]
    assert "input_to_world_m" not in spawn["residuals_m"]
    assert spawn["maximum_error_m"] < 1e-12


def test_alignment_keeps_schedule_seconds_and_rotates_the_authored_waypoint_heading(tmp_path):
    bootstrap, map_path, data, _ = _inputs(tmp_path, "Y", 0.01, "stage_world")
    angle_deg = math.degrees(data["coordinate_contract"]["map_yaw_rad"])
    for key in ("input_plans", "world_plans", "plans"):
        plan = data[key][0]
        plan["start_time"] = 5.0
        plan["waypoints"][0].update(wait=8.0, yaw=180.0 - angle_deg if key == "plans" else 180.0)
    bootstrap.write_text(json.dumps(data))
    actor = export_coordinate_alignment(bootstrap, map_path, tmp_path / "alignment")["actors"][0]
    assert actor["start_time_s"] == 5.0
    assert actor["waypoints"][0]["wait_s"] == 8.0
    assert actor["waypoints"][0]["yaw_world_deg"] == 180.0
    assert actor["waypoints"][0]["yaw_ros_deg"] == pytest.approx(180.0 - angle_deg)


@pytest.mark.parametrize("input_frame", ["stage_world", "world_m"])
def test_alignment_separates_an_explicit_planar_spawn_adjustment_from_transform_error(tmp_path, input_frame):
    bootstrap, map_path, data, _ = _inputs(tmp_path, "Y", 0.01, input_frame)
    delta = np.array([0.12, -0.05, 0.0])
    angle = data["coordinate_contract"]["map_yaw_rad"]
    cosine, sine = math.cos(angle), math.sin(angle)
    ros_delta = [cosine * delta[0] + sine * delta[1], -sine * delta[0] + cosine * delta[1], 0.0]
    for key, offset in (("world_plans", delta), ("plans", ros_delta)):
        actor = data[key][0]
        actor["position"] = (np.asarray(actor["position"]) + offset).tolist()
    name = data["plans"][0]["name"]
    data["settings"] = {"spawn_tolerance_m": 0.15}
    data["spawn_adjustments_world_m"] = {name: delta.tolist()}
    bootstrap.write_text(json.dumps(data))
    spawn = export_coordinate_alignment(bootstrap, map_path, tmp_path / "adjusted")["actors"][0]["spawn"]
    assert spawn["spawn_adjustment_distance_m"] == pytest.approx(0.13)
    assert spawn["maximum_error_m"] < 1e-12

    data["settings"]["spawn_tolerance_m"] = 0.1
    bootstrap.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="outside its configured planar tolerance"):
        export_coordinate_alignment(bootstrap, map_path, tmp_path / "rejected")


@pytest.mark.parametrize(
    "problem",
    [
        "name",
        "duplicate_name",
        "count",
        "missing_position",
        "position",
        "frame",
        "units",
        "map_frame",
        "wait",
        "start",
        "yaw",
    ],
)
def test_invalid_pairing_or_coordinate_metadata_fails_without_creating_outputs(tmp_path, problem):
    bootstrap, map_path, data, _ = _inputs(tmp_path)
    if problem == "name":
        data["plans"][0]["name"] = "another"
    elif problem == "duplicate_name":
        for key in ("plans", "world_plans", "input_plans"):
            data[key].append(copy.deepcopy(data[key][0]))
    elif problem == "count":
        data["plans"][0]["waypoints"].pop()
    elif problem == "missing_position":
        del data["input_plans"][0]["position"]
    elif problem == "position":
        data["plans"][0]["waypoints"][0]["position"][0] += 0.01
    elif problem == "frame":
        data["coordinate_contract"]["bootstrap_plans_frame"] = "world_m"
    elif problem == "units":
        data["coordinate_contract"]["stage_meters_per_unit"] = 0
    elif problem == "wait":
        data["plans"][0]["waypoints"][0]["wait"] = 1.0
    elif problem == "start":
        data["plans"][0]["start_time"] = 1.0
    elif problem == "yaw":
        for key in ("input_plans", "world_plans", "plans"):
            data[key][0]["waypoints"][0]["yaw"] = 90.0
    else:
        data["coordinate_contract"]["map_yaw_rad"] += 0.1
        # Keep all positions mutually consistent; independent map metadata must still reject this.
        data["plans"] = copy.deepcopy(data["world_plans"])
        data["coordinate_contract"]["map_origin_world_m"] = [0.0, 0.0, 0.0]
        data["coordinate_contract"]["map_yaw_rad"] = 0.0
    bootstrap.write_text(json.dumps(data))
    output = tmp_path / "alignment"
    with pytest.raises(ValueError):
        export_coordinate_alignment(bootstrap, map_path, output)
    assert not output.exists()


def test_alignment_refuses_to_overwrite_an_existing_output_directory(tmp_path):
    bootstrap, map_path, _, _ = _inputs(tmp_path)
    output = tmp_path / "alignment"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("preserved")
    with pytest.raises(FileExistsError, match="must be new"):
        export_coordinate_alignment(bootstrap, map_path, output)
    assert sentinel.read_text() == "preserved" and len(list(output.iterdir())) == 1
