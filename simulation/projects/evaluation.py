# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate saved ROS trajectories and occupancy maps without loading the SDK."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from ..ros2_navigation.planner import OccupancyGrid

_SLACK = 1e-4
_ARRIVAL_EPS_M = 1e-6
_STATIONARY_PHASES = {
    "waiting",
    "standing",
    "turning",
    "dwelling",
    "complete",
    "door_face",
    "door_wait",
    "passing_wait",
    "following_wait",
    "door_close_wait",
    "door_close_restore",
    "goal_wait",
}
_EVIDENCE_ERRORS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    IndexError,
    OverflowError,
    EOFError,
    BadZipFile,
)


def _number(value: object, label: str, *, minimum: float = -math.inf) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number.")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{label} must be finite and >= {minimum}.")
    return result


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    result = _number(value, label, minimum=minimum)
    if not result.is_integer():
        raise ValueError(f"{label} must be an integer.")
    return int(result)


def _vector(value: object, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain three finite coordinates.")
    return result


def _snapshot(records: list[dict], names: set[str], label: str) -> dict[str, dict]:
    by_name = {record["name"]: record for record in records}
    if len(by_name) != len(records) or set(by_name) != names:
        raise ValueError(f"{label} must contain exactly the configured participants {sorted(names)}.")
    return {
        name: {
            **record,
            "position": _vector(record["position"], f"{label}.{name}.position [m]"),
            "velocity": _vector(record["velocity"], f"{label}.{name}.velocity [m/s]"),
            "yaw": _number(record["yaw"], f"{label}.{name}.yaw [rad]"),
        }
        for name, record in by_name.items()
    }


def _angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _route(plan: dict, origin: np.ndarray, auto_return_heading: str = "preserve") -> dict:
    if auto_return_heading not in ("preserve", "follow_route"):
        raise ValueError("auto_return_heading must be 'preserve' or 'follow_route'.")
    targets = [dict(point) for point in plan.get("waypoints", [])] if plan.get("behavior") == "walk" else []
    for point in targets:
        point["position"] = _vector(point["position"], f"{plan['name']} waypoint [m]")
    indices = list(range(len(targets)))
    if plan.get("behavior") == "walk" and not targets:
        raise ValueError(f"{plan['name']} has no walking waypoints.")
    if targets and plan.get("repeat"):
        closed = plan.get("authored_closed_route", np.array_equal(targets[-1]["position"], origin))
        if type(closed) is not bool:
            raise ValueError("authored_closed_route must be a boolean.")
        if not closed:
            reverse = [dict(point) for point in reversed(targets[:-1])]
            if auto_return_heading == "follow_route":
                for point in reverse:
                    point["yaw"] = None
            targets += reverse
            targets.append({"position": origin, "yaw": None, "wait": 0.0})
            indices += list(reversed(indices[:-1])) + [None]
    return {
        "targets": targets,
        "original_indices": indices,
        "authored_count": len(plan.get("waypoints", [])),
        "cursor": 0,
        "reached": {},
        "visits": {},
        "route_completed": False,
        "dwell_s": Counter(),
    }


def _audit_motion(before, after, issued, previous, *, radius, speed_limit, turn_rate, dt, grid, margin, label, fail):
    motion = after["position"] - before["position"]
    distance = float(np.linalg.norm(motion))
    speed = float(np.linalg.norm(after["velocity"]))
    angular_speed = abs(_angle(after["yaw"] - before["yaw"])) / dt
    if issued["phase"] in _STATIONARY_PHASES and (speed > _SLACK or distance > _SLACK):
        fail(f"{label}: stationary phase {issued['phase']} translated the body.")
    if (
        np.linalg.norm(before["position"] - previous["position"]) > _SLACK
        or abs(_angle(before["yaw"] - previous["yaw"])) > _SLACK
    ):
        fail(f"{label}: observed pose jumped between control frames.")
    if speed > speed_limit + _SLACK or distance > speed_limit * dt + _SLACK:
        fail(f"{label}: speed/displacement exceeded the original {speed_limit} [m/s] limit.")
    if angular_speed > turn_rate + _SLACK:
        fail(f"{label}: angular rate {angular_speed} exceeded {turn_rate} [rad/s].")
    if np.linalg.norm(motion - after["velocity"] * dt) > _SLACK:
        fail(f"{label}: actual displacement does not match executed velocity.")
    if (
        np.linalg.norm(issued["position"] - after["position"]) > _SLACK
        or np.linalg.norm(issued["velocity"] - after["velocity"]) > _SLACK
        or abs(_angle(issued["yaw"] - after["yaw"])) > _SLACK
    ):
        fail(f"{label}: executed state differs from the issued command.")
    for snapshot in (before, after):
        if abs(_number(snapshot.get("radius_m", radius), "radius_m") - radius) > _SLACK:
            fail(f"{label}: body radius changed from the bootstrap geometry.")
    if not grid.segment_clear(before["position"], after["position"], radius + margin):
        fail(f"{label}: swept body disk violated static occupancy clearance.")
    return distance, speed, angular_speed


def _waypoint_visit(route, ordinal):
    visits = route["visits"]
    if ordinal not in visits:
        cycle, slot = divmod(ordinal, len(route["targets"]))
        target = route["targets"][slot]
        original = route["original_indices"][slot]
        kind = "authored" if slot < route["authored_count"] else "reverse" if original is not None else "spawn_return"
        visits[ordinal] = {
            "cycle": cycle,
            "route_index": slot,
            "original_waypoint_index": original,
            "kind": kind,
            "target_position_m": target["position"].tolist(),
            "required_dwell_s": float(target.get("wait", 0.0)),
            "arrival_time_s": None,
            "route_advance_time_s": None,
            "dwell_intervals_s": [],
            "max_dwell_step_m": 0.0,
        }
    return visits[ordinal]


def _audit_route(route, plan, observed, issued, tolerance, time_s, dt, step_distance, label, fail):
    if not route["targets"]:
        return
    count = len(route["targets"])
    index = _integer(issued["waypoint_index"], "waypoint_index")
    cycles = _integer(issued["completed_cycles"], "completed_cycles")
    phase = issued["phase"]
    if index >= count or (not plan.get("repeat") and cycles > 1):
        raise ValueError(f"{label}: route index/cycle is outside the original route.")
    complete = not plan.get("repeat") and issued.get("route_phase", phase) == "complete"
    if complete and (cycles != 1 or index != count - 1):
        raise ValueError(f"{label}: final route index/cycle is inconsistent with completion.")
    cursor = count if complete else cycles * count + index
    if cursor < route["cursor"] or cursor - route["cursor"] > 2 * count + 4:
        raise ValueError(f"{label}: route cursor skipped or reversed its original targets.")
    arrivals = list(range(route["cursor"], cursor))
    for ordinal in arrivals:
        visit = _waypoint_visit(route, ordinal)
        visit["route_advance_time_s"] = time_s
        required = visit["required_dwell_s"]
        if route["dwell_s"][ordinal] < required - 1e-7:
            fail(f"{label}: target {ordinal % count} departed before its original {required} [s] dwell.")
    if phase in ("turning", "dwelling"):
        arrivals.append(cursor)
    if phase == "dwelling":
        route["dwell_s"][cursor] += dt
        yaw = route["targets"][cursor % count].get("yaw")
        if yaw is not None and abs(_angle(issued["yaw"] - math.radians(yaw))) > 0.025 + _SLACK:
            fail(f"{label}: original waypoint heading was not restored before dwelling.")
    for ordinal in arrivals:
        slot = ordinal % count
        target = route["targets"][slot]
        error_m = float(np.linalg.norm(observed["position"] - target["position"]))
        if error_m > tolerance + _ARRIVAL_EPS_M:
            fail(f"{label}: target {slot} claimed arrival at {error_m} [m], beyond {tolerance} [m].")
            continue
        visit = _waypoint_visit(route, ordinal)
        if visit["arrival_time_s"] is None:
            visit.update(
                arrival_time_s=time_s,
                arrival_observed_position_m=observed["position"].tolist(),
                arrival_distance_m=error_m,
            )
        original_index = route["original_indices"][slot]
        if original_index is not None:
            route["reached"].setdefault(original_index, {"time_s": time_s, "distance_m": error_m})
    if not complete:
        visit = _waypoint_visit(route, cursor)
        visit["last_phase"] = phase
        visit["last_observed_position_m"] = observed["position"].tolist()
        if phase == "dwelling":
            intervals = visit["dwell_intervals_s"]
            if intervals and math.isclose(intervals[-1][1], time_s, rel_tol=0, abs_tol=1e-8):
                intervals[-1][1] = time_s + dt
            else:
                intervals.append([time_s, time_s + dt])
            visit["max_dwell_step_m"] = max(visit["max_dwell_step_m"], step_distance)
    route["cursor"] = cursor
    route["route_completed"] = complete


def _waypoint_receipt(route, trace_end_s):
    visits = list(route["visits"].values())
    for visit in visits:
        # The last command has no following observation past the saved trace end.
        intervals = [[start, min(end, trace_end_s)] for start, end in visit["dwell_intervals_s"] if start < trace_end_s]
        dwell_s = sum(end - start for start, end in intervals)
        visit.update(
            dwell_intervals_s=intervals,
            observed_dwell_s=dwell_s,
            dwell_complete=visit["arrival_time_s"] is not None and dwell_s >= visit["required_dwell_s"] - 1e-7,
        )
    return visits


def _peer_gap(observed: dict, states: dict, radii: dict, required: float, label: str, fail: Callable[[str], None]):
    minimum = math.inf
    ordered = sorted(observed)
    for i, first in enumerate(ordered):
        for second in ordered[:i]:
            relative = observed[first]["position"][:2] - observed[second]["position"][:2]
            change = (
                states[first]["position"][:2]
                - observed[first]["position"][:2]
                - states[second]["position"][:2]
                + observed[second]["position"][:2]
            )
            squared = float(np.dot(change, change))
            fraction = float(np.clip(-np.dot(relative, change) / squared, 0, 1)) if squared else 0.0
            gap = float(np.linalg.norm(relative + fraction * change)) - radii[first] - radii[second]
            minimum = min(minimum, gap)
            if gap < required - _SLACK:
                fail(f"{label}: swept peer gap {gap} [m] for {first}/{second} is below {required} [m].")
    return minimum


def evaluate_ros2_run(runtime_dir: Path, *, require_route_goals: bool = False) -> dict[str, object]:
    """Verify saved logical motion, goal arrivals and map clearance without the SDK.

    Args:
        runtime_dir: Directory containing result.json, bootstrap.json, map.npz and trace.jsonl.
        require_route_goals: Require actual arrival at every original walking waypoint.
            Short user durations may legitimately leave this coverage incomplete.

    Returns:
        JSON-compatible checks, errors, phase durations [s], near-zero walking
        intervals [s], original goal coverage and minimum swept peer body gap [m].
        Positions [m], speeds [m/s] and angular rates [rad/s] are checked against
        bootstrap plans and settings. Absolute numerical slack is 1e-4 in each
        quantity's SI units; original arrival tolerance uses only 1e-6 [m] slack.
        These are logical trajectory and occupancy-map certificates, not PhysX
        contact-depth, rendered animation or dynamic-obstacle certificates.
        Invalid or missing evidence returns passed=False and concrete errors.
        Input files are read without modification.
    """
    runtime_dir = Path(runtime_dir).expanduser().resolve()
    errors: list[str] = []
    report: dict[str, object] = {
        "passed": False,
        "runtime_directory": str(runtime_dir),
        "scope": "logical_ros2_trajectory_and_occupancy_map",
        "limitations": ["Does not measure PhysX contact depth, rendered animation or dynamic-obstacle clearance."],
        "require_route_goals": require_route_goals,
        "errors": errors,
    }

    def fail(message: str) -> None:
        if len(errors) < 100:
            errors.append(message)
        else:
            report["additional_error_count"] = int(report.get("additional_error_count", 0)) + 1

    try:
        result = json.loads((runtime_dir / "result.json").read_text(encoding="utf-8"))
        bootstrap = json.loads((runtime_dir / "bootstrap.json").read_text(encoding="utf-8"))
        plans = bootstrap["plans"]
        plan_by_name = {plan["name"]: plan for plan in plans}
        names = set(plan_by_name)
        if not names or len(names) != len(plans) or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Bootstrap plans require distinct nonempty participant names.")
        initial = _snapshot(bootstrap["agents"], names, "bootstrap.agents")
        settings = bootstrap["settings"]
        auto_return_heading = settings.get("auto_return_heading", "preserve")
        hz = _integer(settings["control_hz"], "control_hz", minimum=1)
        dt = 1.0 / hz
        margin = _number(settings.get("map_clearance_m", 0.05), "map_clearance_m", minimum=0)
        pair_gap = _number(settings.get("pair_clearance_m", 0.25), "pair_clearance_m", minimum=0)
        turn_rate = _number(settings.get("max_turn_rate_radps", 2.5), "max_turn_rate_radps", minimum=0)
        radii = {name: _number(initial[name]["radius_m"], f"{name}.radius_m", minimum=1e-12) for name in names}
        speeds = {name: _number(plan_by_name[name].get("speed", 1.0), f"{name}.speed", minimum=0) for name in names}
        tolerances = {
            name: _number(plan_by_name[name].get("arrival_tolerance", 0.15), f"{name}.arrival_tolerance", minimum=1e-12)
            for name in names
        }
        routes = {name: _route(plan_by_name[name], initial[name]["position"], auto_return_heading) for name in names}
        with np.load(runtime_dir / "map.npz", allow_pickle=False) as data:
            grid = OccupancyGrid(data["occupancy"], float(data["resolution_m"]), tuple(data["origin_xy_m"]))
        recording = result.get("recording", {})
        duration = result.get("requested_duration_s", bootstrap.get("requested_duration_s"))
        if duration is None:
            duration = recording.get("requested_duration_s")
        duration = _number(duration, "requested_duration_s", minimum=1e-12)
        warmup = _number(recording.get("warmup_s", 0.0), "warmup_s", minimum=0)
        expected_end = duration + warmup
    except _EVIDENCE_ERRORS as error:
        fail(f"Cannot load runtime evidence: {error}")
        return report

    report.update(control_hz=hz, requested_duration_s=duration, expected_end_time_s=expected_end)
    for key, expected in (
        ("passed", True),
        ("completed_duration", True),
        ("ira_enabled", False),
        ("navmesh_enabled", False),
    ):
        if result.get(key) is not expected:
            fail(f"Runtime result {key} must be {expected!r}.")
    if result.get("controller") != "ros2_sfm_orca":
        fail("Runtime result does not identify the independent ROS controller.")
    cleanup = result.get("cleanup")
    for key in ("stage_closed", "temporary_removed", "ros_process_stopped"):
        if not isinstance(cleanup, dict) or cleanup.get(key) is not True:
            fail(f"Cleanup did not confirm {key}.")
    if result.get("error") or result.get("finalization_errors"):
        fail("Runtime reported an execution or finalization error.")

    metrics = {
        name: {
            "phase_duration_s": Counter(),
            "walking_near_zero_total_s": 0.0,
            "walking_near_zero_longest_s": 0.0,
            "distance_m": 0.0,
        }
        for name in sorted(names)
    }
    near_zero = dict.fromkeys(names, 0.0)
    previous = initial
    previous_time = None
    frames = 0
    minimum_gap = math.inf
    maximum_speed = maximum_step = maximum_turn = 0.0
    last_command = None
    try:
        with (runtime_dir / "trace.jsonl").open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                    now = _number(row["time_s"], "time_s", minimum=0)
                    if previous_time is None:
                        if not any(math.isclose(now, tick * dt, abs_tol=1e-8) for tick in (1, 2)):
                            fail(f"Trace line {line_number}: first control time {now} is outside the startup interval.")
                    elif not math.isclose(now - previous_time, dt, rel_tol=1e-6, abs_tol=1e-8):
                        fail(
                            f"Trace line {line_number}: control clock advanced by {now - previous_time}, "
                            f"expected {dt} [s]."
                        )
                    command = row["command"]
                    if not math.isclose(_number(command["time_s"], "command.time_s"), now, abs_tol=1e-8):
                        fail(f"Trace line {line_number}: command time differs from the observed clock.")
                    observed = _snapshot(row["observed"], names, "observed")
                    states = _snapshot(row["states"], names, "states")
                    commands = _snapshot(command["agents"], names, "command.agents")
                    for name in sorted(names):
                        before, after, issued = observed[name], states[name], commands[name]
                        label = f"Trace line {line_number}, {name}"
                        distance, speed, angular_speed = _audit_motion(
                            before,
                            after,
                            issued,
                            previous[name],
                            radius=radii[name],
                            speed_limit=speeds[name],
                            turn_rate=turn_rate,
                            dt=dt,
                            grid=grid,
                            margin=margin,
                            label=label,
                            fail=fail,
                        )
                        maximum_speed, maximum_step, maximum_turn = (
                            max(maximum_speed, speed),
                            max(maximum_step, distance),
                            max(maximum_turn, angular_speed),
                        )
                        phase = issued["phase"]
                        if not isinstance(phase, str) or issued.get("error") or phase == "failed" or row.get("failed"):
                            fail(f"{label}: navigation reported an invalid or failed phase.")
                        metrics[name]["phase_duration_s"][phase] += dt
                        metrics[name]["distance_m"] += distance
                        near_zero[name] = near_zero[name] + dt if phase == "walking" and speed < 0.03 else 0.0
                        metrics[name]["walking_near_zero_total_s"] += dt if near_zero[name] else 0.0
                        metrics[name]["walking_near_zero_longest_s"] = max(
                            metrics[name]["walking_near_zero_longest_s"], near_zero[name]
                        )
                        _audit_route(
                            routes[name],
                            plan_by_name[name],
                            before,
                            issued,
                            tolerances[name],
                            now,
                            dt,
                            distance,
                            label,
                            fail,
                        )
                    minimum_gap = min(
                        minimum_gap, _peer_gap(observed, states, radii, pair_gap, f"Trace line {line_number}", fail)
                    )
                    previous, previous_time, last_command = states, now, commands
                    frames += 1
                except _EVIDENCE_ERRORS as error:
                    fail(f"Trace line {line_number} is invalid: {error}")
    except OSError as error:
        fail(f"Cannot read control trace: {error}")

    if not frames:
        fail("No valid control frames were recorded.")
    if previous_time is not None:
        try:
            reported_end = _number(result["simulation_time_s"], "simulation_time_s", minimum=0)
            if not math.isclose(reported_end, previous_time, abs_tol=1e-7):
                fail("Result simulation time differs from the final control trace.")
            if not expected_end - 1e-7 <= previous_time <= expected_end + dt + 1e-7:
                fail(f"Trace ended at {previous_time} [s], outside the requested bounded duration {expected_end} [s].")
            if _integer(result["control_frames"], "control_frames") != frames:
                fail("Result control frame count differs from the trace.")
            final = _snapshot(result["pedestrian_schedules"], names, "result.pedestrian_schedules")
            if any(np.linalg.norm(final[name]["position"] - last_command[name]["position"]) > _SLACK for name in names):
                fail("Result pedestrian schedules differ from the final control trace.")
        except _EVIDENCE_ERRORS as error:
            fail(f"Invalid final runtime result: {error}")
    original_count = reached_count = 0
    for name in sorted(names):
        plan, route = plan_by_name[name], routes[name]
        goals = plan.get("waypoints", []) if plan.get("behavior") == "walk" else []
        original_count += len(goals)
        reached_count += len(route["reached"])
        metrics[name].update(
            phase_duration_s=dict(metrics[name]["phase_duration_s"]),
            route_completed=route["route_completed"],
            waypoint_visits=_waypoint_receipt(route, previous_time or 0.0),
            original_goals=[
                {
                    "index": index,
                    "position_m": goal["position"],
                    "reached": index in route["reached"],
                    **route["reached"].get(index, {}),
                }
                for index, goal in enumerate(goals)
            ],
        )
    all_goals = original_count == reached_count
    if require_route_goals and not all_goals:
        fail(f"Original walking goal coverage is incomplete: {reached_count}/{original_count} actually reached.")
    report.update(
        passed=not errors,
        control_frames=frames,
        simulation_time_s=previous_time,
        agents=metrics,
        original_goal_count=original_count,
        waypoint_visits_frame="ros_map",
        reached_original_goals=reached_count,
        route_goals_complete=all_goals,
        minimum_peer_body_gap_m=None if math.isinf(minimum_gap) else minimum_gap,
        maximum_speed_mps=maximum_speed,
        maximum_step_displacement_m=maximum_step,
        maximum_turn_rate_radps=maximum_turn,
    )
    return report
