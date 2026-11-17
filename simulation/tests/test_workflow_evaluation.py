# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline workflow evidence validation without Kit, ROS processes or a GPU."""

import copy
import hashlib
import json
import math

import numpy as np
import pytest

from pedstrain_simulation.simulation.projects.evaluation import evaluate_ros2_run


def _write_run(tmp_path, *, points=None, repeat=False, specs=None, hz=10, turn_rate=2.5):
    initial = {
        "name": "person",
        "position": [0.0, 0.0, 0.0],
        "velocity": [0.0, 0.0, 0.0],
        "yaw": 0.0,
        "radius_m": 0.1,
        "height_m": 1.8,
    }
    plan = {
        "name": "person",
        "position": initial["position"],
        "behavior": "walk",
        "speed": 1.0,
        "arrival_tolerance": 0.02,
        "repeat": repeat,
        "waypoints": [{"position": point, "wait": 0.0, "yaw": None} for point in (points or [[0.3, 0.0, 0.0]])],
    }
    bootstrap = {
        "plans": [plan],
        "agents": [initial],
        "settings": {
            "control_hz": hz,
            "map_clearance_m": 0.0,
            "pair_clearance_m": 0.1,
            "max_turn_rate_radps": turn_rate,
        },
    }
    if specs is None:
        specs = [(x, "walking", 0, 0) for x in (0.1, 0.2, 0.3)] + [(0.3, "complete", 0, 1)] * 2
    rows, previous = [], initial
    for index, (x, phase, waypoint, cycles) in enumerate(specs):
        after = copy.deepcopy(previous)
        after["position"] = [x, 0.0, 0.0]
        after["velocity"] = [(x - previous["position"][0]) * hz, 0.0, 0.0]
        if after["velocity"][0]:
            after["yaw"] = 0.0 if after["velocity"][0] > 0 else math.pi
        issued = {
            **copy.deepcopy(after),
            "phase": phase,
            "waypoint_index": waypoint,
            "completed_cycles": cycles,
            "error": None,
        }
        rows.append(
            {
                "time_s": (index + 1) / hz,
                "observed": [copy.deepcopy(previous)],
                "states": [after],
                "command": {"time_s": (index + 1) / hz, "agents": [issued]},
            }
        )
        previous = after
    result = {
        "passed": True,
        "completed_duration": True,
        "ira_enabled": False,
        "navmesh_enabled": False,
        "controller": "ros2_sfm_orca",
        "requested_duration_s": len(rows) / hz,
        "simulation_time_s": len(rows) / hz,
        "control_frames": len(rows),
        "cleanup": {"stage_closed": True, "temporary_removed": True, "ros_process_stopped": True},
        "pedestrian_schedules": rows[-1]["command"]["agents"],
    }
    _save(tmp_path, bootstrap, result, rows)
    np.savez(
        tmp_path / "map.npz", occupancy=np.zeros((60, 60), dtype=np.int8), resolution_m=0.1, origin_xy_m=[-3.0, -3.0]
    )
    return bootstrap, result, rows


def _save(directory, bootstrap, result, rows):
    (directory / "bootstrap.json").write_text(json.dumps(bootstrap))
    (directory / "result.json").write_text(json.dumps(result))
    (directory / "trace.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_complete_run_uses_actual_arrival_and_preserves_all_input_bytes(tmp_path):
    _write_run(tmp_path)
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in tmp_path.iterdir()}
    report = evaluate_ros2_run(tmp_path, require_route_goals=True)
    assert report["passed"], report["errors"]
    assert report["reached_original_goals"] == report["original_goal_count"] == 1
    actor = report["agents"]["person"]
    assert actor["route_completed"] is True
    assert actor["original_goals"][0]["time_s"] == pytest.approx(0.4)
    assert actor["phase_duration_s"] == pytest.approx({"walking": 0.3, "complete": 0.2})
    assert actor["distance_m"] == pytest.approx(0.3)
    assert before == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in tmp_path.iterdir()}


def test_short_bounded_run_does_not_claim_full_goal_coverage_just_because_it_moved(tmp_path):
    _write_run(tmp_path, specs=[(0.1, "walking", 0, 0), (0.2, "walking", 0, 0)])
    short = evaluate_ros2_run(tmp_path)
    assert short["passed"] and not short["route_goals_complete"]
    assert short["reached_original_goals"] == 0
    [visit] = short["agents"]["person"]["waypoint_visits"]
    assert short["waypoint_visits_frame"] == "ros_map"
    assert visit["target_position_m"] == [0.3, 0.0, 0.0]
    assert visit["arrival_time_s"] is None and visit["last_phase"] == "walking"
    assert visit["observed_dwell_s"] == 0 and not visit["dwell_complete"]
    strict = evaluate_ros2_run(tmp_path, require_route_goals=True)
    assert not strict["passed"]
    assert any("coverage is incomplete" in error for error in strict["errors"])


def test_same_tick_colocated_targets_are_all_validated_including_the_final_index_clamp(tmp_path):
    _write_run(
        tmp_path,
        points=[[0.3, 0.0, 0.0]] * 2,
        specs=[
            (0.1, "walking", 0, 0),
            (0.2, "walking", 0, 0),
            (0.3, "walking", 0, 0),
            (0.3, "complete", 1, 1),
        ],
    )
    report = evaluate_ros2_run(tmp_path, require_route_goals=True)
    assert report["passed"], report["errors"]
    assert report["reached_original_goals"] == 2
    assert [point["time_s"] for point in report["agents"]["person"]["original_goals"]] == [0.4, 0.4]


@pytest.mark.parametrize("spawn_adjusted", [False, True])
@pytest.mark.parametrize("closed", [False, True])
def test_repeat_route_expansion_does_not_confuse_return_indices_with_original_goals(tmp_path, closed, spawn_adjusted):
    if spawn_adjusted and closed:
        points = [[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]]
        specs = [
            (0.1, "walking", 0, 0),
            (0.0, "walking", 1, 0),
            (-0.1, "walking", 1, 0),
            (0.0, "walking", 0, 1),
        ]
    elif spawn_adjusted:
        points = [[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]]
        specs = [
            (0.1, "walking", 0, 0),
            (0.0, "walking", 1, 0),
            (0.1, "walking", 2, 0),
            (0.0, "walking", 3, 0),
            (0.1, "walking", 0, 1),
        ]
    elif closed:
        points = [[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]]
        specs = [(0.1, "walking", 0, 0), (0.0, "walking", 1, 0), (0.1, "walking", 0, 1)]
    else:
        points = [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]
        specs = [
            (0.1, "walking", 0, 0),
            (0.2, "walking", 1, 0),
            (0.1, "walking", 2, 0),
            (0.0, "walking", 3, 0),
            (0.1, "walking", 0, 1),
            (0.2, "walking", 1, 1),
        ]
    bootstrap, result, rows = _write_run(tmp_path, points=points, repeat=True, specs=specs, turn_rate=40.0)
    if spawn_adjusted:
        bootstrap["plans"][0]["authored_closed_route"] = closed
        _save(tmp_path, bootstrap, result, rows)
    report = evaluate_ros2_run(tmp_path, require_route_goals=True)
    assert report["passed"], report["errors"]
    assert report["reached_original_goals"] == report["original_goal_count"] == 2
    assert not report["agents"]["person"]["route_completed"]


def _write_return_heading_run(tmp_path, policy, *, closed=False, reverse_dwell=True):
    points = [[0.1, 0.0, 0.0], [0.0 if closed else 0.2, 0.0, 0.0]]
    specs = [(0.1, "walking", 0, 0), (0.1, "dwelling", 0, 0)]
    specs += [(points[1][0], "walking", 1, 0), (points[1][0], "dwelling", 1, 0)]
    if not closed:
        specs += [(0.1, "walking", 2, 0)]
        if reverse_dwell:
            specs += [(0.1, "dwelling", 2, 0)]
        specs += [(0.0, "walking", 3, 0), (0.0, "dwelling", 3, 0)]
    bootstrap, result, rows = _write_run(tmp_path, points=points, repeat=True, specs=specs, turn_rate=40.0)
    for point in bootstrap["plans"][0]["waypoints"]:
        point.update(yaw=0.0, wait=0.1)
    if policy is not None:
        bootstrap["settings"]["auto_return_heading"] = policy
    _save(tmp_path, bootstrap, result, rows)
    return bootstrap, result, rows


@pytest.mark.parametrize("policy", [None, "preserve", "follow_route"])
def test_auto_return_heading_changes_only_synthetic_reverse_heading_checks(tmp_path, policy):
    _write_return_heading_run(tmp_path, policy)
    report = evaluate_ros2_run(tmp_path, require_route_goals=True)
    assert report["passed"] is (policy == "follow_route"), report["errors"]
    assert report["reached_original_goals"] == report["original_goal_count"] == 2
    if policy != "follow_route":
        assert any("waypoint heading" in error for error in report["errors"])


@pytest.mark.parametrize("closed", [False, True])
def test_follow_route_keeps_explicit_forward_and_closed_route_headings(tmp_path, closed):
    bootstrap, result, rows = _write_return_heading_run(tmp_path, "follow_route", closed=closed)
    if not closed:
        bootstrap["plans"][0]["waypoints"][0]["yaw"] = 90.0
        _save(tmp_path, bootstrap, result, rows)
    report = evaluate_ros2_run(tmp_path, require_route_goals=True)
    assert not report["passed"]
    assert any("waypoint heading" in error for error in report["errors"])


def test_follow_route_keeps_the_original_synthetic_reverse_dwell(tmp_path):
    _write_return_heading_run(tmp_path, "follow_route", reverse_dwell=False)
    report = evaluate_ros2_run(tmp_path, require_route_goals=True)
    assert not report["passed"]
    assert any("target 2 departed before its original 0.1 [s] dwell" in error for error in report["errors"])


def test_final_arrival_coverage_does_not_claim_pending_turn_or_wait_has_completed(tmp_path):
    bootstrap, result, rows = _write_run(
        tmp_path,
        specs=[
            (0.1, "walking", 0, 0),
            (0.2, "walking", 0, 0),
            (0.3, "walking", 0, 0),
            (0.3, "turning", 0, 0),
            (0.3, "dwelling", 0, 0),
        ],
    )
    bootstrap["plans"][0]["waypoints"][0]["wait"] = 5.0
    _save(tmp_path, bootstrap, result, rows)
    report = evaluate_ros2_run(tmp_path, require_route_goals=True)
    assert report["passed"] and report["route_goals_complete"]
    assert not report["agents"]["person"]["route_completed"]
    [visit] = report["agents"]["person"]["waypoint_visits"]
    assert visit["arrival_time_s"] == pytest.approx(0.4)
    assert visit["arrival_observed_position_m"] == [0.3, 0.0, 0.0]
    assert visit["last_phase"] == "dwelling" and visit["route_advance_time_s"] is None
    assert visit["required_dwell_s"] == 5.0 and not visit["dwell_complete"]
    assert visit["observed_dwell_s"] == 0 and visit["dwell_intervals_s"] == []


@pytest.mark.parametrize("resumed_dwell_ticks", [0, 2])
def test_closing_refuge_time_does_not_count_as_the_original_dwell(tmp_path, resumed_dwell_ticks):
    bootstrap, result, rows = _write_run(
        tmp_path,
        specs=[(0.1, "walking", 0, 0), (0.2, "walking", 0, 0), (0.3, "walking", 0, 0)]
        + [(0.3, "dwelling", 0, 0)]
        + [(0.3, "door_close_wait", 0, 0)] * 4
        + [(0.3, "door_close_restore", 0, 0)]
        + [(0.3, "dwelling", 0, 0)] * resumed_dwell_ticks
        + [(0.3, "complete", 0, 1)],
    )
    bootstrap["plans"][0]["waypoints"][0]["wait"] = 0.3
    _save(tmp_path, bootstrap, result, rows)
    report = evaluate_ros2_run(tmp_path, require_route_goals=True)
    assert report["passed"] is bool(resumed_dwell_ticks), report["errors"]
    [visit] = report["agents"]["person"]["waypoint_visits"]
    assert visit["observed_dwell_s"] == pytest.approx((1 + resumed_dwell_ticks) * 0.1)
    assert visit["dwell_intervals_s"][0] == pytest.approx([0.4, 0.5])
    assert len(visit["dwell_intervals_s"]) == (2 if resumed_dwell_ticks else 1)
    assert visit["dwell_complete"] is bool(resumed_dwell_ticks)
    assert visit["max_dwell_step_m"] == 0
    if not resumed_dwell_ticks:
        assert any("original 0.3 [s] dwell" in error for error in report["errors"])


def test_terminal_door_safety_phases_keep_the_completed_route_cursor(tmp_path):
    bootstrap, result, rows = _write_run(
        tmp_path,
        points=[[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
        specs=[(0.1, "walking", 0, 0), (0.2, "walking", 1, 0), (0.2, "complete", 1, 1)]
        + [(0.2, "door_close_restore", 1, 1)] * 2
        + [(0.2, "complete", 1, 1)],
    )
    for row in rows[2:]:
        row["command"]["agents"][0]["route_phase"] = "complete"
    _save(tmp_path, bootstrap, result, rows)
    report = evaluate_ros2_run(tmp_path, require_route_goals=True)
    assert report["passed"], report["errors"]
    assert report["agents"]["person"]["route_completed"]


def test_goal_coverage_does_not_broaden_the_original_arrival_tolerance(tmp_path):
    _write_run(tmp_path, points=[[0.3, 0.0, 0.02005]])
    report = evaluate_ros2_run(tmp_path, require_route_goals=True)
    assert not report["passed"]
    assert not report["route_goals_complete"]


def test_completed_route_cannot_continue_translating_after_arrival(tmp_path):
    bootstrap, result, rows = _write_run(tmp_path)
    for snapshot in (rows[-1]["states"][0], rows[-1]["command"]["agents"][0]):
        snapshot["position"][0] = 0.4
        snapshot["velocity"][0] = 1.0
    _save(tmp_path, bootstrap, result, rows)
    report = evaluate_ros2_run(tmp_path)
    assert not report["passed"]
    assert any("stationary phase" in error for error in report["errors"])


def test_near_zero_walking_statistics_reset_when_continuous_progress_resumes(tmp_path):
    _write_run(tmp_path, specs=[(x, "walking", 0, 0) for x in (0.0, 0.0, 0.0, 0.1, 0.1)])
    report = evaluate_ros2_run(tmp_path)
    assert report["passed"]
    assert report["agents"]["person"]["walking_near_zero_total_s"] == pytest.approx(0.4)
    assert report["agents"]["person"]["walking_near_zero_longest_s"] == pytest.approx(0.3)


@pytest.mark.parametrize("height", [False, True])
def test_claimed_target_retirement_requires_actual_3d_arrival_for_every_skipped_goal(tmp_path, height):
    points = [[0.3, 0.0, 0.3]] if height else [[0.3, 0.0, 0.0], [0.9, 0.0, 0.0]]
    specs = [
        (0.1, "walking", 0, 0),
        (0.2, "walking", 0, 0),
        (0.3, "walking", 0, 0),
        (0.3, "complete", len(points) - 1, 1),
    ]
    _write_run(tmp_path, points=points, specs=specs)
    report = evaluate_ros2_run(tmp_path)
    assert not report["passed"]
    assert any("claimed arrival" in error for error in report["errors"])


@pytest.mark.parametrize("fault", ["clock", "participant", "nonfinite", "speed", "pose_jump", "angular_rate"])
def test_invalid_control_evidence_is_rejected_with_a_concrete_error(tmp_path, fault):
    bootstrap, result, rows = _write_run(tmp_path)
    expected = {
        "clock": "control clock",
        "participant": "participants",
        "nonfinite": "finite",
        "speed": "speed/displacement",
        "pose_jump": "pose jumped",
        "angular_rate": "angular rate",
    }
    if fault == "clock":
        rows[1]["time_s"] = rows[1]["command"]["time_s"] = rows[0]["time_s"]
    elif fault == "participant":
        rows[0]["observed"][0]["name"] = "unexpected"
    elif fault == "nonfinite":
        rows[0]["states"][0]["velocity"][0] = math.nan
    elif fault == "speed":
        rows[0]["states"][0]["velocity"][0] = 2.0
    elif fault == "pose_jump":
        rows[1]["observed"][0]["position"][0] = 1.0
    else:
        rows[0]["states"][0]["yaw"] = 1.0
    _save(tmp_path, bootstrap, result, rows)
    report = evaluate_ros2_run(tmp_path)
    assert not report["passed"]
    assert any(expected[fault] in error for error in report["errors"])


@pytest.mark.parametrize(
    "fault", ["cleanup", "cleanup_schema", "incomplete", "ira", "duration", "missing", "corrupt", "corrupt_map"]
)
def test_missing_or_failed_runtime_evidence_cannot_pass(tmp_path, fault):
    bootstrap, result, rows = _write_run(tmp_path)
    if fault == "cleanup":
        result["cleanup"]["ros_process_stopped"] = False
    elif fault == "cleanup_schema":
        result["cleanup"] = None
    elif fault == "incomplete":
        result["completed_duration"] = False
    elif fault == "ira":
        result["ira_enabled"] = True
    elif fault == "duration":
        result["requested_duration_s"] = 0.9
    _save(tmp_path, bootstrap, result, rows)
    if fault == "missing":
        (tmp_path / "map.npz").unlink()
    elif fault == "corrupt":
        (tmp_path / "trace.jsonl").write_text("{broken\n")
    elif fault == "corrupt_map":
        (tmp_path / "map.npz").write_bytes(b"PK\x03\x04truncated")
    report = evaluate_ros2_run(tmp_path)
    assert not report["passed"] and report["errors"]


def test_peer_swept_gap_rejects_midstep_crossing_even_when_endpoints_are_clear(tmp_path):
    bootstrap, result, rows = _write_run(tmp_path, specs=[(0.6, "walking", 0, 0)], hz=1, turn_rate=40.0)
    first = bootstrap["agents"][0]
    first["position"][0] = -0.3
    rows[0]["observed"][0]["position"][0] = -0.3
    rows[0]["states"][0]["position"][0] = rows[0]["command"]["agents"][0]["position"][0] = 0.3
    rows[0]["states"][0]["velocity"][0] = rows[0]["command"]["agents"][0]["velocity"][0] = 0.6
    second = copy.deepcopy(first)
    second.update(name="other", position=[0.3, 0.0, 0.0], yaw=math.pi)
    bootstrap["agents"].append(second)
    other_plan = copy.deepcopy(bootstrap["plans"][0])
    other_plan.update(name="other", position=second["position"])
    bootstrap["plans"].append(other_plan)
    rows[0]["observed"].append(copy.deepcopy(second))
    after = {**second, "position": [-0.3, 0.0, 0.0], "velocity": [-0.6, 0.0, 0.0]}
    rows[0]["states"].append(after)
    rows[0]["command"]["agents"].append({**after, "phase": "walking", "waypoint_index": 0, "completed_cycles": 0})
    result["pedestrian_schedules"] = rows[0]["command"]["agents"]
    _save(tmp_path, bootstrap, result, rows)
    report = evaluate_ros2_run(tmp_path)
    assert not report["passed"]
    assert report["minimum_peer_body_gap_m"] == pytest.approx(-0.2)
    assert any("swept peer gap" in error for error in report["errors"])


def test_static_swept_disk_rejects_a_thin_wall_crossing_between_clear_endpoints(tmp_path):
    bootstrap, result, rows = _write_run(tmp_path, specs=[(0.6, "walking", 0, 0)], hz=1)
    bootstrap["agents"][0]["position"][0] = rows[0]["observed"][0]["position"][0] = -0.3
    rows[0]["states"][0]["position"][0] = rows[0]["command"]["agents"][0]["position"][0] = 0.3
    rows[0]["states"][0]["velocity"][0] = rows[0]["command"]["agents"][0]["velocity"][0] = 0.6
    result["pedestrian_schedules"] = rows[0]["command"]["agents"]
    _save(tmp_path, bootstrap, result, rows)
    cells = np.zeros((60, 60), dtype=np.int8)
    cells[:, 30] = 100
    np.savez(tmp_path / "map.npz", occupancy=cells, resolution_m=0.1, origin_xy_m=[-3.0, -3.0])
    report = evaluate_ros2_run(tmp_path)
    assert not report["passed"]
    assert any("swept body disk" in error for error in report["errors"])
