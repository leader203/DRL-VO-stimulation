# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent route scheduling, social preferences and genuine RVO2 execution."""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .goal_occupancy import GoalOccupancyCoordinator
from .interactions import DoorCoordinator, _segment_point_distance
from .passing import PassingCoordinator, _clear, _forward_path, _OverlayGrid
from .planner import OccupancyGrid, _path_lookahead, _planning_obstacles, convex_footprint, segment_polygon_clear
from .wall_force import compute_wall_force

# Allow accumulated native tick roundoff, well below one control interval [s].
_SCHEDULE_EPS_S = 1e-9


def build_solver(cache_dir: Path) -> Path:
    """Compile the vendored CPU-only RVO2 C ABI into a content-addressed cache."""
    vendor = Path(__file__).parent / "vendor"
    sources = [
        vendor / "bridge.cpp",
        *(vendor / "RVO2/src" / name for name in ("Agent.cpp", "KdTree.cpp", "Obstacle.cpp", "RVOSimulator.cpp")),
    ]
    digest = hashlib.sha256()
    for path in sorted(vendor.rglob("*")):
        if path.is_file() and path.suffix in (".cpp", ".h"):
            digest.update(path.read_bytes())
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    library = cache_dir / f"librvo2_{digest.hexdigest()[:20]}.so"
    if library.is_file():
        return library
    temporary = library.with_name(f"{library.name}.{os.getpid()}.tmp")
    try:
        completed = subprocess.run(
            [
                "g++",
                "-std=c++17",
                "-O2",
                "-shared",
                "-fPIC",
                "-I",
                str(vendor / "RVO2/src"),
                *(str(path) for path in sources),
                "-o",
                str(temporary),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(f"RVO2 compilation failed: {completed.stderr}")
        temporary.replace(library)
    finally:
        temporary.unlink(missing_ok=True)
    return library


class _RvoSolver:
    def __init__(self, walls, library_path):
        self.library = ctypes.CDLL(str(library_path))
        version = getattr(self.library, "pedestrian_rvo_abi_version", None)
        if version is None or version() != 3:
            raise RuntimeError("RVO2 library ABI mismatch; rebuild with build_solver(cache_dir).")
        double = ctypes.POINTER(ctypes.c_double)
        byte = ctypes.POINTER(ctypes.c_ubyte)
        self.library.pedestrian_rvo_create.argtypes = [double, double, ctypes.c_int]
        self.library.pedestrian_rvo_create.restype = ctypes.c_void_p
        self.library.pedestrian_rvo_destroy.argtypes = [ctypes.c_void_p]
        self.library.pedestrian_rvo_error.restype = ctypes.c_char_p
        self.library.pedestrian_rvo_step.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            double,
            double,
            double,
            double,
            double,
            byte,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            double,
        ]
        self.library.pedestrian_rvo_step.restype = ctypes.c_int
        self.library.pedestrian_rvo_heading.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            double,
        ]
        self.library.pedestrian_rvo_heading.restype = ctypes.c_int
        self._double = double
        walls = np.ascontiguousarray(walls, dtype=np.float64)
        self._static_walls = walls
        self._walls = walls
        self._wall_velocities = np.zeros((len(walls), 2), dtype=np.float64)
        self._count = None
        self.context = self.library.pedestrian_rvo_create(
            walls.ctypes.data_as(double), self._wall_velocities.ctypes.data_as(double), len(walls)
        )
        if not self.context:
            raise RuntimeError(self.library.pedestrian_rvo_error().decode())

    def solve(
        self,
        positions,
        velocities,
        preferred,
        radii,
        speeds,
        passive,
        dt,
        distance,
        horizon,
        padding,
        margin,
        obstacle_horizon=None,
    ):
        if not self.context:
            raise RuntimeError("The RVO2 solver has been closed.")
        if self._count is not None and self._count != len(positions):
            self._replace_context(self._walls, self._wall_velocities)
        self._count = len(positions)
        arrays = [
            np.ascontiguousarray(value, dtype=np.float64) for value in (positions, velocities, preferred, radii, speeds)
        ]
        flags = np.ascontiguousarray(passive, dtype=np.uint8)
        result = np.zeros_like(arrays[0])
        code = self.library.pedestrian_rvo_step(
            self.context,
            len(positions),
            *(a.ctypes.data_as(self._double) for a in arrays),
            flags.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            dt,
            distance,
            horizon,
            horizon if obstacle_horizon is None else obstacle_horizon,
            padding,
            margin,
            result.ctypes.data_as(self._double),
        )
        if code or not np.isfinite(result).all():
            raise RuntimeError(f"RVO2 failed ({code}): {self.library.pedestrian_rvo_error().decode()}")
        return result

    def _replace_context(self, walls, velocities):
        replacement = self.library.pedestrian_rvo_create(
            walls.ctypes.data_as(self._double), velocities.ctypes.data_as(self._double), len(walls)
        )
        if not replacement:
            raise RuntimeError(self.library.pedestrian_rvo_error().decode())
        self.library.pedestrian_rvo_destroy(self.context)
        self.context = replacement
        self._walls, self._wall_velocities = walls, velocities
        self._count = None

    def configure_polygons(self, polygons):
        walls = [self._static_walls]
        velocities = [np.zeros((len(self._static_walls), 2), dtype=float)]
        for obstacle in polygons:
            vertices = obstacle.footprint
            walls.append(np.hstack((vertices, np.roll(vertices, -1, axis=0))))
            velocities.append(np.tile(obstacle.velocity, (len(vertices), 1)))
        combined = np.ascontiguousarray(np.vstack(walls), dtype=np.float64)
        movement = np.ascontiguousarray(np.vstack(velocities), dtype=np.float64)
        if not np.array_equal(combined, self._walls) or not np.array_equal(movement, self._wall_velocities):
            self._replace_context(combined, movement)

    def heading_project(self, preferred, yaws, angle, speeds, agent_indices=None):
        result = np.zeros_like(preferred)
        angles = np.broadcast_to(angle, (len(result),))
        for index, agent_index in enumerate(range(len(result)) if agent_indices is None else agent_indices):
            self.library.pedestrian_rvo_heading(
                self.context,
                agent_index,
                float(yaws[index]),
                float(angles[index]),
                float(speeds[index]),
                *map(float, preferred[index]),
                result[index].ctypes.data_as(self._double),
            )
        return result

    def close(self):
        if self.context:
            self.library.pedestrian_rvo_destroy(self.context)
            self.context = None

    def __del__(self):
        if getattr(self, "context", None):
            self.close()


@dataclass
class _DynamicObstacle:
    position: np.ndarray
    velocity: np.ndarray
    radius: float
    footprint: np.ndarray | None = None
    name: str = "unnamed"


@dataclass
class _SuspendedArrival:
    remaining_dwell_s: float | None
    yaw: float | None


@dataclass
class _Route:
    plan: dict
    origin: list[float]
    targets: list[dict] = field(default_factory=list)
    index: int = 0
    cycles: int = 0
    phase: str = "waiting"
    deadline: float | None = None
    path: list = field(default_factory=list)
    path_padding_m: float = 0.0
    parked_context: tuple[tuple[str, tuple[float, float], float], ...] = ()
    parked_disks: tuple[tuple[np.ndarray, float], ...] = ()
    parked_align: bool = False
    error: str | None = None
    stalled_since: float | None = None
    progress_position: np.ndarray | None = None
    wander_target: dict | None = None
    travel_heading: float | None = None
    corner_replan_position: tuple[float, float] | None = None
    point_target: tuple[float, float] | None = None
    point_recovery: bool = False
    door_token: tuple | None = None
    suspended_arrival: _SuspendedArrival | None = None


class NavigationEngine:
    """Execute YAML plans in ROS world meters independently of rendering and IRA.

    Args:
        plans: ``PedestrianPlan.to_dict()`` records; angles in plans are [deg].
        grid: Static ROS bottom-left-origin occupancy map [m].
        settings: SI navigation settings; angular rate is [rad/s]. Agent and
            obstacle prediction horizons are [s]; an omitted obstacle horizon
            inherits ``time_horizon_s`` for direct API compatibility.
        agents: Initial bodies with name, world position [m], yaw [rad] and radius/height [m].
        library_path: Optional previously built RVO2 library.
    """

    def __init__(self, plans, grid: OccupancyGrid, settings: dict, agents, library_path: Path | None = None):
        self.grid, self.settings = grid, dict(settings)
        self.names = [str(agent["name"]) for agent in agents]
        if len(set(self.names)) != len(self.names) or not self.names:
            raise ValueError("Agents require distinct nonempty names.")
        by_name = {plan["name"]: dict(plan) for plan in plans}
        if set(by_name) != set(self.names) or len(by_name) != len(plans):
            raise ValueError("Plans and agents must have matching unique names.")
        self.bodies = {a["name"]: dict(a) for a in agents}
        self.routes = {}
        return_heading = settings.get("auto_return_heading", "preserve")
        if type(return_heading) is not str or return_heading not in ("preserve", "follow_route"):
            raise ValueError("auto_return_heading must be preserve or follow_route.")
        self.margin = float(settings.get("map_clearance_m", 0.05))
        self.path_preference = float(settings.get("preferred_path_clearance_m", 0.0))
        self.pair_clearance = float(settings.get("pair_clearance_m", 0.25))
        self.horizon = float(settings.get("time_horizon_s", 2.5))
        self.obstacle_horizon = float(settings.get("obstacle_time_horizon_s", self.horizon))
        self.neighbor_distance = float(settings.get("neighbor_distance_m", 4.0))
        self.relaxation = float(settings.get("relaxation_time_s", 0.4))
        self.turn_rate = float(settings.get("max_turn_rate_radps", 2.5))
        if (
            min(self.margin, self.pair_clearance, self.path_preference) < 0
            or min(self.horizon, self.obstacle_horizon, self.neighbor_distance, self.relaxation, self.turn_rate) <= 0
        ):
            raise ValueError("Navigation settings require positive finite physical limits.")
        if not np.isfinite(
            [
                self.margin,
                self.path_preference,
                self.pair_clearance,
                self.horizon,
                self.obstacle_horizon,
                self.neighbor_distance,
                self.relaxation,
                self.turn_rate,
            ]
        ).all():
            raise ValueError("Navigation settings must be finite.")
        for agent in agents:
            plan = by_name[agent["name"]]
            route = _Route(plan, list(agent["position"]))
            route.targets = [dict(point) for point in plan.get("waypoints", [])]
            if plan.get("behavior", "wander") == "walk":
                if not route.targets:
                    raise ValueError("Walking plans require explicit waypoints.")
                if plan.get("repeat", False):
                    closed = plan.get("authored_closed_route", route.targets[-1]["position"] == plan.get("position"))
                    if type(closed) is not bool:
                        raise ValueError("authored_closed_route must be a boolean.")
                    if not closed:
                        returning = [dict(point) for point in reversed(route.targets[:-1])]
                        if return_heading == "follow_route":
                            returning = [{**point, "yaw": None} for point in returning]
                        route.targets += returning
                        route.targets.append({"position": route.origin, "wait": 0.0, "yaw": None})
            self.routes[agent["name"]] = route
        self.rng = np.random.default_rng(settings.get("seed", 42))
        self._last_time = None
        self._encounters = {}
        self._walls = grid.wall_segments()
        self._doors = DoorCoordinator(grid, settings)
        self._goals = GoalOccupancyCoordinator(grid, settings)
        self._goal_objectives = {}
        self._planning_obstacles = ()
        self._planning_signature = None
        self._planning_grids = {}
        self._passing = PassingCoordinator(grid, settings)
        self._solver = _RvoSolver(
            self._walls, library_path or build_solver(Path(tempfile.gettempdir()) / "pedstrain_ros2_rvo_cache")
        )

    def step(self, time_s: float, dt: float, states: list[dict], obstacles=(), door_states=()) -> dict:
        """Advance a coherent crowd snapshot [m, rad, m/s] at time/dt [s].

        External obstacles require explicitly observed velocities [m/s]. Unknown
        velocities reject the tick instead of assuming an obstacle is stationary.
        """
        if not math.isfinite(time_s) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Simulation time must be finite and dt positive [s].")
        observed = {state["name"]: state for state in states}
        if set(observed) != set(self.names) or len(states) != len(self.names):
            raise ValueError("Each navigation tick requires exactly one state per configured agent.")
        if self._last_time is not None and time_s < self._last_time - 1e-9:
            for route in self.routes.values():
                route.index = route.cycles = 0
                route.phase = "waiting"
                route.deadline = None
                route.error = None
                route.path = []
                route.parked_context = route.parked_disks = ()
                route.parked_align = False
                route.stalled_since = None
                route.progress_position = None
                route.wander_target = None
                route.travel_heading = None
                route.corner_replan_position = None
                route.point_target = None
                route.point_recovery = False
                route.door_token = None
                route.suspended_arrival = None
            self._encounters.clear()
            self._doors.reset()
            self._passing.reset()
            self._goals.reset()
            self._goal_objectives.clear()
        self._last_time = time_s
        positions = np.array([observed[name]["position"][:2] for name in self.names], dtype=float)
        velocities = np.array([observed[name].get("velocity", [0, 0, 0])[:2] for name in self.names], dtype=float)
        yaws = np.array([observed[name]["yaw"] for name in self.names], dtype=float)
        radii = np.array(
            [float(observed[name].get("radius_m", self.bodies[name].get("radius_m", 0.25))) for name in self.names]
        )
        if not np.isfinite([*positions.ravel(), *velocities.ravel(), *yaws, *radii]).all() or np.any(radii <= 0):
            raise ValueError("Body states must contain finite physical quantities and positive radii.")
        intentions = {}
        for name, route in self.routes.items():
            target = (
                self._current_target(route)
                if route.plan.get("behavior", "wander") == "walk" or route.wander_target is not None
                else None
            )
            active = (
                route.error is None
                and time_s + _SCHEDULE_EPS_S >= route.plan.get("start_time", 0.0)
                and route.plan.get("behavior", "wander") in ("walk", "wander")
                and route.phase not in ("turning", "dwelling", "complete", "failed")
            )
            next_target = self._next_target(route)
            intentions[name] = {
                "target": None if target is None else target["position"],
                "active": active,
                "speed_mps": float(route.plan.get("speed", 1.0)),
                "arrival_tolerance_m": float(route.plan.get("arrival_tolerance", 0.15)),
                "route_context": (route.index, route.cycles),
                "path": route.path,
                "path_padding_m": route.path_padding_m,
                "phase": route.phase,
                "next_target": next_target,
                "scheduled_departure_s": self._scheduled_departure(route, observed[name], time_s, dt),
                "can_door_refuge": (
                    route.plan.get("behavior", "wander") == "walk"
                    and route.phase in ("turning", "dwelling", "complete")
                ),
                "prepare_door_refuge": (
                    active
                    and route.plan.get("behavior", "wander") == "walk"
                    and target is not None
                    and (target.get("wait", 0.0) > 0.0 or next_target is None)
                ),
                "settle_door_entry": (active and route.plan.get("behavior", "wander") == "walk" and target is not None),
            }
        door_bodies = [
            {
                **observed[name],
                "radius_m": float(radii[i]),
                "height_m": float(observed[name].get("height_m", self.bodies[name].get("height_m", 1.8))),
            }
            for i, name in enumerate(self.names)
        ]
        cooperation, directives, passing, goals, dynamic = self._cooperate(
            time_s, door_bodies, intentions, obstacles, door_states, observed
        )
        parked = tuple(
            (name, tuple(positions[index]), float(radii[index]))
            for index, name in enumerate(self.names)
            if time_s + _SCHEDULE_EPS_S < self.routes[name].plan.get("start_time", 0.0)
            and self.routes[name].plan.get("behavior", "wander") in ("walk", "wander")
            and not np.any(velocities[index])
            and directives.get(name, {}).get("target") is None
        )
        preferred = np.zeros_like(positions)
        caps = np.zeros(len(self.names))
        desired_speeds = np.zeros(len(self.names))
        terminal_tracking = np.zeros(len(self.names), dtype=bool)
        terminal_preferred = np.zeros_like(positions)
        point_recovery = np.zeros(len(self.names), dtype=bool)
        directions = np.zeros_like(positions)
        guidance = {}
        heading_angles = np.full(len(self.names), self.turn_rate * dt)
        for index, name in enumerate(self.names):
            route = self.routes[name]
            try:
                if not self.grid.segment_clear(positions[index], positions[index], radii[index] + self.margin):
                    raise ValueError(
                        "Actual body position is outside disk-clear occupancy space; no pose was projected."
                    )
                directive = directives.get(name)
                token = None if directive is None else tuple(directive["token"])
                if route.door_token != token:
                    # Door admission still needs the original route around jambs.
                    # Normal guidance rechecks visibility from the actual pose.
                    route.travel_heading = None
                    route.point_target = None
                    route.point_recovery = False
                    route.progress_position = None
                    route.stalled_since = None
                    route.door_token = token
                if directive is None:
                    target = self._target(route, observed[name], time_s, dt, radii[index], parked)
                else:
                    target = directive["target"]
                    if target is None:
                        route.progress_position = None
                        route.stalled_since = None
                guidance[name] = {
                    "target": None
                    if target is None
                    else [*map(float, target[:2]), float(observed[name]["position"][2])],
                    "phase": route.phase if directive is None else directive["phase"],
                }
                self._track_point(route, target)
                point_recovery[index] = route.point_recovery
                if target is not None:
                    self._observe_progress(route, positions[index], time_s)
                    delta = np.asarray(target) - positions[index]
                    length = float(np.linalg.norm(delta))
                    desired_speeds[index] = caps[index] = min(float(route.plan.get("speed", 1.0)), length / dt)
                    if length > 0.0:
                        directions[index] = delta / length
                        align_parked = (
                            directive is None
                            and route.parked_disks
                            and route.parked_align
                            and all(
                                self._dynamic_clear(positions[index], np.zeros(2), radii[index], obstacle, dt)
                                for obstacle in dynamic
                            )
                        )
                        if directives.get(name, {}).get("certified_ray") or align_parked:
                            caps[index], yaws[index], heading_angles[index] = self._refuge_departure(
                                delta, yaws[index], caps[index], dt
                            )
                            route.parked_align = bool(align_parked and caps[index] == 0.0)
                        else:
                            route.parked_align = False
                            through_corner = (
                                directive is None
                                and len(route.path) > 2
                                and tuple(target[:2]) != tuple(route.path[1][:2])
                            )
                            caps[index] = self._approach_speed(
                                caps[index], delta, yaws[index], dt, through_corner=through_corner
                            )
                            terminal_tracking[index] = length <= float(route.plan.get("speed", 1.0)) / self.turn_rate
                            terminal_preferred[index] = self._point_preference(
                                delta, yaws[index], float(route.plan.get("speed", 1.0)), dt
                            )
                            caps[index], yaws[index], heading_angles[index] = self._committed_turn(
                                route,
                                positions[index],
                                directions[index],
                                radii[index],
                                caps[index],
                                yaws[index],
                                heading_angles[index],
                                dynamic,
                                dt,
                            )
                else:
                    if directive is None:
                        yaws[index] = self._stationary_yaw(route, observed[name], dt)
                    elif directive.get("yaw") is not None:
                        difference = self._angle(directive["yaw"] - yaws[index])
                        yaws[index] = self._angle(
                            yaws[index] + float(np.clip(difference, -self.turn_rate * dt, self.turn_rate * dt))
                        )
            except (ValueError, RuntimeError) as error:
                route.error = str(error)
                route.phase = "failed"
                guidance[name] = {"target": None, "phase": "failed"}
        # Keep attraction to the point while braking execution; weakening this
        # drive as well would let soft peer repulsion prevent a safe arrival.
        force = (directions * desired_speeds[:, None] - velocities) / self.relaxation
        for i in range(len(self.names)):
            if caps[i] == 0 or self.names[i] in directives:
                continue
            for j in range(len(self.names)):
                if i == j:
                    continue
                difference = positions[i] - positions[j]
                distance = float(np.linalg.norm(difference))
                if distance > self.neighbor_distance:
                    continue
                normal = difference / max(distance, 1e-9) if distance else np.array([1.0 if i < j else -1.0, 0.0])
                force[i] += 2.0 * math.exp(min(5.0, (radii[i] + radii[j] - distance) / 0.3)) * normal
            force[i] += self._guidance_static_force(positions[i], radii[i], directions[i])
        preferred = velocities + force * dt
        biases = self._passing_bias(positions, directions, caps, radii) + self._passive_bias(
            positions, directions, caps, radii, dynamic
        )
        for i, name in enumerate(self.names):
            if name in directives:
                biases[i] = 0
        preferred += biases * (dt / self.relaxation)
        recovering = terminal_tracking & point_recovery
        preferred[recovering] = terminal_preferred[recovering]
        # Soft preferences may slow or sidestep, never reverse certified guidance.
        longitudinal = np.sum(preferred * directions, axis=1)
        preferred -= np.minimum(longitudinal, 0.0)[:, None] * directions
        lengths = np.linalg.norm(preferred, axis=1)
        preferred *= np.minimum(1.0, np.divide(caps, lengths, out=np.ones_like(caps), where=lengths > 0))[:, None]
        velocities[caps == 0] = 0
        ordinary = np.array([name not in directives for name in self.names])
        self._keep_sidestep_clearance(positions, preferred, radii, caps, directions, ordinary & ~terminal_tracking, dt)
        lateral = velocities - np.sum(velocities * directions, axis=1)[:, None] * directions
        recovery_preferred = (
            directions * caps[:, None]
            + lateral * max(0.0, 1.0 - dt / self.relaxation)
            + biases * (dt / self.relaxation)
        )
        recovery_preferred[~ordinary] = preferred[~ordinary]
        # Remove carried lateral momentum only when a nearby point needs
        # progress recovery after the reciprocal projection.
        recovery_preferred[terminal_tracking] = terminal_preferred[terminal_tracking]
        solved, guide = self._solve_crowd(
            positions,
            velocities,
            preferred,
            radii,
            caps,
            yaws,
            dt,
            dynamic,
            heading_angles,
            directions,
            recovery_preferred,
            ordinary,
            terminal_tracking,
        )
        can_follow_hold = ordinary & np.array(
            [name not in cooperation.get("committed_exits", ()) for name in self.names]
        )
        self._hold_behind_scheduled_peers(
            time_s,
            dt,
            observed,
            intentions,
            positions,
            solved,
            directions,
            desired_speeds,
            radii,
            dynamic,
            can_follow_hold,
            yaws,
            guidance,
        )
        self._check_temporary_displacements(positions, solved, directives, dt)
        self._retain_point_recovery(solved, directions, caps, terminal_tracking)
        turn_guidance = np.where(terminal_tracking[:, None], directions, guide)
        outputs = []
        for i, name in enumerate(self.names):
            route = self.routes[name]
            speed = float(np.linalg.norm(solved[i]))
            if speed > 0.0:
                yaws[i] = math.atan2(solved[i, 1], solved[i, 0])
                route.travel_heading = None
            elif (
                caps[i] > 0
                and np.linalg.norm(turn_guidance[i]) > 0.0
                and guidance[name]["phase"] != "following_wait"
                and not directives.get(name, {}).get("certified_ray")
            ):
                if (
                    route.travel_heading is None
                    or np.dot([math.cos(route.travel_heading), math.sin(route.travel_heading)], directions[i]) < 0
                ):
                    route.travel_heading = math.atan2(turn_guidance[i, 1], turn_guidance[i, 0])
                difference = self._angle(route.travel_heading - yaws[i])
                yaws[i] = self._angle(yaws[i] + float(np.clip(difference, -heading_angles[i], heading_angles[i])))
                if abs(difference) <= heading_angles[i]:
                    route.travel_heading = None
            position = [*(positions[i] + solved[i] * dt), float(observed[name]["position"][2])]
            outputs.append(
                {
                    "name": name,
                    "position": position,
                    "yaw": float(yaws[i]),
                    "velocity": [*map(float, solved[i]), 0.0],
                    "phase": guidance[name]["phase"] if route.error is None else route.phase,
                    "route_phase": route.phase,
                    "waypoint_index": route.index,
                    "completed_cycles": route.cycles,
                    "error": route.error,
                }
            )
        return {
            "time_s": time_s,
            "agents": outputs,
            "door_commands": cooperation["door_commands"],
            "door_events": cooperation.get("events", []),
            "door_closure": cooperation.get("closure_status", {}),
            "goal_occupancy": goals["status"],
            "guidance": guidance,
            "passing": passing["status"],
        }

    def _hold_behind_scheduled_peers(
        self,
        time_s,
        dt,
        observed,
        intentions,
        positions,
        solved,
        directions,
        desired_speeds,
        radii,
        dynamic,
        ordinary,
        yaws,
        guidance,
    ):
        """Suppress non-progressing creep toward a parked actor with a scheduled departure."""
        parked = []
        for index, name in enumerate(self.names):
            if np.any(solved[index]) or math.hypot(*observed[name].get("velocity", [0.0, 0.0])[:2]) > 1e-6:
                continue
            release = intentions[name]["scheduled_departure_s"]
            plan = self.routes[name].plan
            if plan.get("behavior", "wander") in ("walk", "wander") and plan.get("start_time", 0.0) > time_s:
                release = plan["start_time"]
            if release is not None and release > time_s:
                parked.append(index)
        if not parked:
            return
        for i in np.flatnonzero(ordinary & (desired_speeds > 0.0)):
            speed = float(np.linalg.norm(solved[i]))
            progress = float(np.dot(solved[i], directions[i]))
            if speed > min(0.08, 0.2 * desired_speeds[i]) or progress > max(1e-7, 0.02 * speed):
                continue
            for j in parked:
                relative = positions[j] - positions[i]
                distance = float(np.linalg.norm(relative))
                forward = float(np.dot(relative, directions[i]))
                required = radii[i] + radii[j] + self.pair_clearance
                if (
                    i == j
                    or forward <= 0.0
                    or distance > required + min(0.12, 0.25 * (radii[i] + radii[j]))
                    or distance**2 - forward**2 >= required**2
                    or (speed > 0.0 and np.dot(relative, solved[i]) <= 0.0)
                ):
                    continue
                if any(not self._dynamic_clear(positions[i], np.zeros(2), radii[i], item, dt) for item in dynamic):
                    break
                if all(
                    self._chord_distance(positions[i] - positions[k], -solved[k] * dt)
                    >= min(
                        float(np.linalg.norm(positions[i] - positions[k])), radii[i] + radii[k] + self.pair_clearance
                    )
                    for k in range(len(radii))
                    if k != i
                ):
                    solved[i] = 0.0
                    name = self.names[i]
                    heading = math.atan2(directions[i, 1], directions[i, 0])
                    measured_yaw = observed[name]["yaw"]
                    turn = self._angle(heading - measured_yaw)
                    limit = self.turn_rate * dt
                    yaws[i] = self._angle(measured_yaw + float(np.clip(turn, -limit, limit)))
                    route = self.routes[name]
                    route.travel_heading = route.progress_position = route.stalled_since = None
                    guidance[name] = {"target": None, "phase": "following_wait"}
                break

    def _cooperate(self, time_s, door_bodies, intentions, obstacles, door_states, observed):
        def intersects_actor_height(obstacle):
            bounds = (obstacle.get("z_min_m"), obstacle.get("z_max_m"))
            if not all(type(value) in (int, float) and math.isfinite(value) for value in bounds):
                return True
            low, high = bounds
            if low > high:
                return True
            for body in door_bodies:
                height = body["height_m"]
                if not math.isfinite(height) or height <= 0:
                    return True
                if high >= body["position"][2] and low <= body["position"][2] + height:
                    return True
            return False

        deferred = {
            name for name, item in self._goal_objectives.items() if item["phase"] in ("goal_wait", "goal_yield")
        }
        door_intentions = self._doors.admission_intentions(door_bodies, intentions, deferred)
        cooperation = self._doors.update(time_s, door_bodies, door_intentions, obstacles, door_states)
        directives = dict(cooperation["objectives"])
        self._suspend_arrivals(directives, observed, time_s)
        dynamic = self._dynamic_obstacles(
            [obstacle for obstacle in cooperation["obstacles"] if intersects_actor_height(obstacle)]
        )
        self._update_planning_obstacles(dynamic)
        priority = {
            name: item
            for name, item in directives.items()
            if item["phase"] != "door_wait" or item.get("safety_critical", True)
        }
        # Keep an authorized outgoing body moving until its measured sweep
        # clearance releases the lease. This marker never overrides guidance.
        portal_priority = {name: {"phase": "door_exit_owned"} for name in cooperation.get("committed_exits", ())}
        for name, marker in portal_priority.items():
            priority.setdefault(name, marker)
        for transaction in self._passing.transactions.values():
            for name in (transaction.yielder, transaction.passer):
                priority.setdefault(name, {"phase": "passing_owned"})
        for name in self._passing.waiting_actors:
            priority.setdefault(name, {"phase": "passing_owned"})
        goal_intentions = {
            name: {
                **intention,
                "door_outgoing": name in cooperation.get("outgoing_actors", ()),
                "queued_door_entry": directives.get(name, {}).get("queued_door_entry", False),
                "yield_origin_xy_m": directives.get(name, {}).get("yield_origin_xy_m"),
            }
            for name, intention in intentions.items()
        }
        goals = self._goals.update(time_s, door_bodies, goal_intentions, dynamic, priority)
        self._goal_objectives = {
            name: item
            for name, item in goals["objectives"].items()
            if directives.get(name, {}).get("phase") != "door_wait" or item["phase"] in ("goal_yield", "goal_wait")
        }
        directives.update(self._goal_objectives)
        passing = self._passing.update(time_s, door_bodies, intentions, dynamic, {**portal_priority, **directives})
        directives.update(passing["objectives"])
        directives.update(
            self._doors.prepare_closing_arrival(time_s, door_bodies, intentions, obstacles, door_states, directives)
        )
        return cooperation, directives, passing, goals, dynamic

    def _update_planning_obstacles(self, dynamic):
        observed = _planning_obstacles(dynamic, self.settings.get("door_specs", ()))
        signature = tuple(
            (
                obstacle.name,
                obstacle.footprint.tobytes()
                if obstacle.footprint is not None
                else (obstacle.position.tobytes(), obstacle.radius),
            )
            for obstacle in observed
        )
        if signature != self._planning_signature:
            self._planning_grids.clear()
            self._planning_signature = signature
        self._planning_obstacles = observed

    def _planning_grid(self, radius, parked_disks=()):
        if parked_disks:
            return _OverlayGrid(self.grid, radius, self.margin, parked_disks, self._planning_obstacles)
        if not self._planning_obstacles:
            return self.grid
        if radius not in self._planning_grids:
            self._planning_grids[radius] = _OverlayGrid(self.grid, radius, self.margin, (), self._planning_obstacles)
        return self._planning_grids[radius]

    def _route_segment_clear(self, start, end, radius, parked_disks=()):
        return _clear(
            self.grid,
            np.asarray(start[:2]),
            np.asarray(end[:2]),
            radius,
            self.margin,
            parked_disks,
            self._planning_obstacles,
        )

    def _suspend_arrivals(self, directives, observed, time_s):
        # Door clearance is temporary motion, never original waypoint dwell.
        for name, route in self.routes.items():
            directive = directives.get(name)
            closing = directive is not None and directive["phase"].startswith("door_close_")
            if closing:
                directive["certified_ray"] = True
                if route.suspended_arrival is None:
                    target = self._current_target(route)
                    yaw = None
                    if target.get("yaw") is not None and route.phase in ("turning", "dwelling", "complete"):
                        yaw = math.radians(target["yaw"]) if route.phase == "turning" else observed[name]["yaw"]
                    route.suspended_arrival = _SuspendedArrival(
                        None if route.deadline is None else max(0.0, route.deadline - time_s), yaw
                    )
            elif route.suspended_arrival is not None:
                suspended = route.suspended_arrival
                if directive is not None or (
                    suspended.yaw is not None and abs(self._angle(suspended.yaw - observed[name]["yaw"])) > 1e-7
                ):
                    if directive is None:
                        directives[name] = {
                            "target": None,
                            "yaw": suspended.yaw,
                            "phase": "door_close_restore",
                            "token": ("close_restore", route.index, route.cycles),
                        }
                else:
                    if suspended.remaining_dwell_s is not None:
                        route.deadline = time_s + suspended.remaining_dwell_s
                    route.suspended_arrival = None

    def _static_force(self, position, radius):
        return compute_wall_force(position, radius, self.margin, self._walls)

    def _check_temporary_displacements(self, positions, velocities, directives, dt):
        for i, name in enumerate(self.names):
            directive = directives.get(name)
            if directive is not None and directive.get("origin") is not None:
                displacement = np.linalg.norm(positions[i] + velocities[i] * dt - np.asarray(directive["origin"]))
                if displacement > directive["max_displacement_m"] + 1e-7:
                    raise RuntimeError(f"Temporary refuge would exceed its bounded displacement for {name}.")

    @staticmethod
    def _track_point(route, target):
        point = None if target is None else tuple(map(float, target[:2]))
        if point != route.point_target:
            route.point_target = point
            route.point_recovery = False

    def _retain_point_recovery(self, velocities, directions, caps, terminal):
        progress = np.sum(velocities * directions, axis=1)
        for index in np.flatnonzero(terminal & (caps > 0) & (progress < 0.25 * caps)):
            self.routes[self.names[index]].point_recovery = True

    def _committed_turn(self, route, position, direction, radius, speed, yaw, angle, obstacles, dt):
        """Finish an owned turn [rad] before translating, retaining dynamic escape [m/s]."""
        heading = route.travel_heading
        if heading is None:
            return speed, yaw, angle
        if np.dot([math.cos(heading), math.sin(heading)], direction) < 0 or not all(
            self._dynamic_clear(position, np.zeros(2), radius, obstacle, dt) for obstacle in obstacles
        ):
            route.travel_heading = None
            return speed, yaw, angle
        difference = self._angle(heading - yaw)
        change = float(np.clip(difference, -angle, angle))
        yaw = self._angle(yaw + change)
        remaining = max(0.0, angle - abs(change))
        if abs(difference) > angle:
            return 0.0, yaw, remaining
        route.travel_heading = None
        return speed, yaw, remaining

    def _approach_speed(self, speed, delta, yaw, dt, *, through_corner=False):
        """Bound point-tracking speed [m/s] by distance [m] and turning time [s]."""
        # A swept-clear preview on the outgoing edge advances with the actor.
        # Keep point braking for the vertex fallback, goals and directives.
        if through_corner:
            return speed
        distance = float(np.linalg.norm(delta))
        error = self._angle(math.atan2(delta[1], delta[0]) - yaw)
        # The tangent circle reaching the point has curvature 2*sin(error)/distance.
        # Brake before projecting the heading, rather than orbiting a nearby target.
        turn_time = 2.0 * abs(math.sin(error)) / self.turn_rate
        return min(speed, distance / max(dt, turn_time))

    def _point_preference(self, delta, yaw, speed, dt):
        """Return goal-directed velocity [m/s], completing a reachable final step [m]."""
        distance = float(np.linalg.norm(delta))
        error = self._angle(math.atan2(delta[1], delta[0]) - yaw)
        time_scale = max(dt, self.relaxation)
        if distance <= speed * dt * dt / self.relaxation and abs(error) <= self.turn_rate * dt:
            time_scale = dt
        return delta / time_scale

    def _refuge_departure(self, delta, yaw, speed, dt):
        # Stay on the certified segment inside the bounded refuge ball.
        error = self._angle(math.atan2(delta[1], delta[0]) - yaw)
        limit = self.turn_rate * dt
        heading = self._angle(yaw + float(np.clip(error, -limit, limit)))
        aligned = abs(error) <= limit
        return speed if aligned else 0.0, heading, 0.0 if aligned else limit

    def _guidance_static_force(self, position, radius, direction):
        force = self._static_force(position, radius)
        # The route already certifies disk clearance. Transverse wall forces
        # compete with that route and can select the longer parking turn.
        # Keep forward drive; RVO and swept guards retain lateral clearance.
        return max(0.0, float(np.dot(force, direction))) * direction

    def _keep_sidestep_clearance(self, positions, preferred, radii, caps, directions, eligible, dt):
        """Steer a wallward head-on sidestep toward a clear straight route [m/s]."""
        for index in np.flatnonzero(eligible & (caps > 0.0)):
            route = self.routes[self.names[index]]
            if route.point_target is None or len(route.path) != 2:
                continue
            position, direction = positions[index], directions[index]
            velocity = preferred[index]
            speed = float(np.linalg.norm(velocity))
            if speed < caps[index] / 3.0 or np.dot(velocity, direction) / speed >= math.cos(math.pi / 12):
                continue

            relative = positions - position
            body_extent = radii[index] + radii
            nearby = np.linalg.norm(relative, axis=1) <= 3.0 * body_extent + self.pair_clearance + self.path_preference
            opposed = (directions @ direction < -0.5) & (relative @ direction > 0.0)
            approaching = np.sum(relative * directions, axis=1) < 0.0
            if not np.any(nearby & opposed & approaching & (caps > 0.0)):
                continue

            reach = min(math.dist(position, route.point_target), 1.5 * caps[index] * self.obstacle_horizon)
            clearance = radii[index] + self.margin + self.path_preference
            if not self.grid.segment_clear(
                position, position + direction * reach, clearance
            ) or self.grid.segment_clear(position, position + velocity * (reach / speed), clearance):
                continue

            wall = self._static_force(position, radii[index])
            sideways = wall - direction * np.dot(wall, direction)
            if np.dot(sideways, velocity) < 0.0:
                # Preserve forward drive while resisting the wallward side of the avoidance.
                corrected = velocity + sideways * (2.0 * dt)
                length = float(np.linalg.norm(corrected))
                preferred[index] = corrected * (min(1.0, caps[index] / length) if length else 1.0)

    @staticmethod
    def _observe_progress(route, position, time_s):
        # A one-second checkpoint also admits valid very slow routes.
        requested_speed_mps = float(route.plan.get("speed", 1.0))
        checkpoint_m = min(0.05, requested_speed_mps * 1.0)
        if route.progress_position is None or np.linalg.norm(position - route.progress_position) > checkpoint_m:
            route.progress_position = position.copy()
            route.stalled_since = time_s
        elif route.stalled_since is not None and time_s - route.stalled_since > 12.0:
            raise RuntimeError("No safe route progress for 12 simulation seconds; original goal retained.")

    @staticmethod
    def _dynamic_obstacles(obstacles):
        dynamic = []
        for obstacle in obstacles:
            name = str(obstacle.get("path", obstacle.get("prim_path", "unnamed")))
            if obstacle.get("velocity_valid") is not True:
                raise ValueError(f"Dynamic obstacle velocity is not observed: {name}")
            position = np.asarray(obstacle["position"][:2], dtype=float)
            velocity = np.asarray(obstacle["velocity"][:2], dtype=float)
            footprint = (
                convex_footprint(obstacle["footprint_xy_m"]) if obstacle.get("footprint_xy_m") is not None else None
            )
            default_radius = (
                float(np.max(np.linalg.norm(footprint - position, axis=1))) if footprint is not None else 0.0
            )
            radius = float(obstacle.get("radius_m", default_radius))
            if (
                position.shape != (2,)
                or velocity.shape != (2,)
                or not np.isfinite([*position, *velocity, radius]).all()
                or radius <= 0
            ):
                raise ValueError("Dynamic obstacle position/velocity/radius must be finite, with positive radius [m].")
            dynamic.append(_DynamicObstacle(position, velocity, radius, footprint, name))
        return dynamic

    def _solve_crowd(
        self,
        positions,
        velocities,
        preferred,
        radii,
        caps,
        yaws,
        dt,
        dynamic,
        heading_angles=None,
        directions=None,
        recovery_preferred=None,
        ordinary=None,
        terminal_tracking=None,
    ):
        count = len(self.names)
        discs = [item for item in dynamic if item.footprint is None]
        self._solver.configure_polygons([item for item in dynamic if item.footprint is not None])
        extra_positions = np.array([item.position for item in discs], dtype=float).reshape(-1, 2)
        extra_velocities = np.array([item.velocity for item in discs], dtype=float).reshape(-1, 2)
        extra_radii = np.array([item.radius for item in discs], dtype=float)
        result = self._solver.solve(
            np.vstack((positions, extra_positions)),
            np.vstack((velocities, extra_velocities)),
            np.vstack((preferred, extra_velocities)),
            np.r_[radii, extra_radii],
            np.r_[caps, np.linalg.norm(extra_velocities, axis=1)],
            np.r_[caps == 0, np.ones(len(discs), dtype=bool)],
            dt,
            self.neighbor_distance,
            self.horizon,
            self.pair_clearance / 2,
            self.margin,
            self.obstacle_horizon,
        )[:count]
        result[caps == 0] = 0
        guide = result.copy()
        angles = np.broadcast_to(self.turn_rate * dt if heading_angles is None else heading_angles, caps.shape)
        projection_yaws, projection_angles, projection_caps = yaws, angles, caps
        if directions is not None:
            path_yaws = np.arctan2(directions[:, 1], directions[:, 0])
            progress = np.sum(guide * directions, axis=1)
            recover = (caps > 0) & (progress < -1e-8)
            if terminal_tracking is not None:
                recover |= terminal_tracking & (caps > 0) & (progress < 0.25 * caps)
            if ordinary is not None:
                near_peer = np.zeros(count, dtype=bool)
                for index in range(count):
                    separations = np.linalg.norm(positions - positions[index], axis=1)
                    contact_range = radii[index] + radii + self.pair_clearance + np.minimum(radii[index], radii)
                    close = separations <= contact_range
                    close[index] = False
                    near_peer[index] = np.any(close)
                # Recover a cancelled soft drive, not a valid velocity limited
                # by reciprocal constraints; forcing that case causes chatter.
                drive = np.sum(preferred * directions, axis=1)
                cancelled = drive < caps * min(1.0, dt / self.relaxation)
                recover |= ordinary & near_peer & (caps > 0) & (progress < 0.25 * caps) & cancelled
            if np.any(recover):
                # Reusing a wall-dominated soft preference can trap the forward query on a tangent.
                recovery = directions * caps[:, None] if recovery_preferred is None else recovery_preferred
                forward = self._solver.heading_project(recovery, path_yaws, math.pi / 2, caps)
                guide[recover] = forward[recover]
            relative = np.arctan2(np.sin(yaws - path_yaws), np.cos(yaws - path_yaws))
            lower = np.maximum(-angles, -math.pi / 2 - relative)
            upper = np.minimum(angles, math.pi / 2 - relative)
            active = caps > 0
            projection_yaws = np.where(active, yaws + (lower + upper) / 2, yaws)
            projection_angles = np.where(active, np.maximum(0.0, (upper - lower) / 2), angles)
            projection_caps = np.where(active & (lower > upper), 0.0, caps)
        # Intersect route-forward motion with the reachable turn sector before
        # selecting an ORCA candidate; any infeasible stop still needs guards.
        result = self._solver.heading_project(guide, projection_yaws, projection_angles, projection_caps)
        limits = projection_yaws, projection_angles, projection_caps
        corner_turns = self._avoid_wall_approach(
            positions, result, guide, radii, caps, directions, ordinary, terminal_tracking, dynamic, limits, dt
        )
        self._guard_motion(positions, result, radii, caps, yaws, dt, dynamic, limits, ordinary)
        terminal = corner_turns if terminal_tracking is None else terminal_tracking | corner_turns
        result, guide = self._turn_guide(
            positions, result, guide, radii, caps, yaws, dt, dynamic, directions, terminal, limits, ordinary
        )
        precision = 8 * np.finfo(np.float32).eps * caps
        speeds = np.linalg.norm(result, axis=1)
        if np.any((speeds > 0) & (speeds <= precision)):
            self._guard_motion(positions, result, radii, caps, yaws, dt, dynamic, limits, ordinary)
        guide[np.linalg.norm(guide, axis=1) <= precision] = 0.0
        return result, guide

    def _avoid_wall_approach(
        self, positions, result, guide, radii, caps, directions, ordinary, terminal, obstacles, limits, dt
    ):
        """Favor a clear route chord before a wallward velocity uses up turning room [m]."""
        turns = np.zeros(len(caps), dtype=bool)
        if ordinary is None or directions is None:
            return turns
        eligible = ordinary if terminal is None else ordinary & ~terminal
        speed = np.linalg.norm(result, axis=1)
        yaws, angles, execution_caps = limits
        for index in np.flatnonzero(eligible & (speed > 0.0)):
            route = self.routes[self.names[index]]
            nominal_speed = float(route.plan.get("speed", 1.0))
            # Let slow reciprocal maneuvers finish without restarting their turns.
            if route.point_target is None or len(route.path) < 3 or speed[index] < nominal_speed / 3.0:
                continue
            position, direction = positions[index], directions[index]
            moving = result[index] / speed[index]
            if np.dot(moving, direction) >= math.cos(math.pi / 12):
                continue
            reach = min(
                math.dist(position, route.point_target),
                2.0 * radii[index],
                nominal_speed * self.obstacle_horizon,
            )
            clearance = radii[index] + self.margin
            if not self.grid.segment_clear(position, position + result[index] * dt, clearance):
                continue  # The immediate swept-motion guard handles this step.
            if self.grid.segment_clear(position, position + moving * reach, clearance) or not self.grid.segment_clear(
                position, position + direction * reach, clearance
            ):
                continue
            sl = slice(index, index + 1)
            candidate = self._solver.heading_project(
                (direction * caps[index])[None, :], yaws[sl], angles[sl], execution_caps[sl], [int(index)]
            )[0]
            length = math.hypot(*candidate)
            if length > 0.0 and self.grid.segment_clear(position, position + candidate * (reach / length), clearance):
                result[index] = candidate
            elif all(self._dynamic_clear(position, np.zeros(2), radii[index], obstacle, dt) for obstacle in obstacles):
                # A turn in place leaves the body outside the wall's approach.
                result[index] = 0.0
                guide[index] = direction
                turns[index] = True
        return turns

    def _guard_motion(self, positions, result, radii, caps, yaws, dt, dynamic, limits, static_prefix=None):
        """Propagate stops and float-velocity cleanup through joint swept guards."""
        for _ in range(len(self.names) + 1):
            self._guard_chords(positions, result, radii, dt, limits, static_prefix)
            changed = False
            speeds = np.linalg.norm(result, axis=1)
            residuals = (speeds > 0) & (speeds <= 8 * np.finfo(np.float32).eps * caps)
            for index in np.flatnonzero(residuals):
                if all(
                    self._dynamic_clear(positions[index], np.zeros(2), radii[index], obstacle, dt)
                    for obstacle in dynamic
                ):
                    # Scale by the requested cap, preserving slow routes.
                    result[index] = 0.0
                    changed = True
            for i in range(len(self.names)):
                for obstacle in dynamic:
                    if not self._dynamic_clear(positions[i], result[i], radii[i], obstacle, dt):
                        result[i] = 0
                        changed = True
                        if not self._dynamic_clear(positions[i], np.zeros(2), radii[i], obstacle, dt):
                            details = self._dynamic_failure_details(
                                i, positions[i], yaws[i], float(radii[i]), obstacle, dt
                            )
                            raise RuntimeError(
                                "Observed dynamic motion has no certified next step within body limits. "
                                + json.dumps(details, allow_nan=False)
                            )
            if not changed:
                break

    def _turn_guide(
        self,
        positions,
        velocities,
        guide,
        radii,
        caps,
        yaws,
        dt,
        dynamic,
        directions,
        terminal,
        limits,
        static_prefix=None,
    ):
        """Prefer a nearer feasible velocity [m/s] before committing a parking turn."""
        if directions is None:
            return velocities, guide
        terminal = np.zeros(len(caps), dtype=bool) if terminal is None else np.asarray(terminal, dtype=bool)
        stopped = np.flatnonzero(np.all(velocities == 0.0, axis=1) & (caps > 0) & ~terminal)
        if not len(stopped):
            return velocities, guide
        working = velocities.copy()
        path_yaws = np.arctan2(directions[:, 1], directions[:, 0])
        execution_yaws, execution_angles, execution_caps = limits

        def representable(index, vector):
            return np.any(
                np.asarray(positions[index] + vector * dt, dtype=np.float32)
                != np.asarray(positions[index], dtype=np.float32)
            )

        def clear(index, vector, others):
            return (
                self.grid.segment_clear(positions[index], positions[index] + vector * dt, radii[index] + self.margin)
                and all(
                    self._dynamic_clear(positions[index], vector, radii[index], obstacle, dt) for obstacle in dynamic
                )
                and all(
                    self._chord_distance(positions[index] - positions[j], (vector - others[j]) * dt)
                    >= min(
                        float(np.linalg.norm(positions[index] - positions[j])),
                        radii[index] + radii[j] + self.pair_clearance,
                    )
                    for j in range(len(caps))
                    if j != index
                )
            )

        for index in stopped:
            transverse = directions[index, 0] * guide[index, 1] - directions[index, 1] * guide[index, 0]
            if (
                np.dot(guide[index], directions[index]) > 0
                and abs(transverse) <= 8 * np.finfo(np.float32).eps * np.linalg.norm(guide[index])
                and representable(index, guide[index])
            ):
                # An already feasible route-aligned guide needs its own turn,
                # rather than a nearer tangent that moves away from the route.
                continue
            facing = caps[index] * np.array([math.cos(yaws[index]), math.sin(yaws[index])])
            candidate = self._solver.heading_project(
                facing[None, :],
                np.array([path_yaws[index]]),
                math.pi / 2 - 64 * np.finfo(np.float32).eps,
                np.array([caps[index]]),
                [int(index)],
            )[0]
            magnitude = float(np.linalg.norm(candidate))
            if (
                np.dot(candidate, directions[index]) < 0
                or magnitude < 0.25 * np.linalg.norm(guide[index])
                or not representable(index, candidate)
            ):
                continue
            angle = abs(self._angle(math.atan2(candidate[1], candidate[0]) - yaws[index]))
            old_angle = abs(self._angle(math.atan2(guide[index, 1], guide[index, 0]) - yaws[index]))
            if angle >= old_angle or not clear(index, candidate, working):
                continue
            guide[index] = candidate
            if (
                magnitude <= execution_caps[index]
                and abs(self._angle(math.atan2(candidate[1], candidate[0]) - execution_yaws[index]))
                <= execution_angles[index]
            ):
                trial = working.copy()
                trial[index] = candidate
                self._guard_chords(positions, trial, radii, dt, limits, static_prefix)
                if all(clear(j, trial[j], trial) for j in range(len(caps))) and representable(index, trial[index]):
                    working = trial
        return working, guide

    def _dynamic_failure_details(self, index, position, yaw, radius, obstacle, dt):
        def clear(clearance, step):
            motion = -obstacle.velocity * step
            if obstacle.footprint is not None:
                return segment_polygon_clear(position, position + motion, obstacle.footprint, clearance)
            return self._chord_distance(position - obstacle.position, motion) >= clearance + obstacle.radius - 1e-6

        return {
            "actor": self.names[index],
            "obstacle_prim_path": obstacle.name,
            "time_s": self._last_time,
            "dt_s": float(dt),
            "actor_position_xy_m": position.tolist(),
            "actor_yaw_rad": float(yaw),
            "actor_radius_m": radius,
            "map_clearance_m": self.margin,
            "obstacle_position_xy_m": obstacle.position.tolist(),
            "obstacle_radius_m": obstacle.radius,
            "obstacle_velocity_xy_mps": obstacle.velocity.tolist(),
            "footprint_xy_m": None if obstacle.footprint is None else obstacle.footprint.tolist(),
            "current_without_margin_clear": bool(clear(radius, 0.0)),
            "current_with_margin_clear": bool(clear(radius + self.margin, 0.0)),
            "predicted_stationary_with_margin_clear": bool(clear(radius + self.margin, dt)),
            "motion_model": "current footprint plus observed translation",
        }

    def _dynamic_clear(self, position, velocity, radius, obstacle, dt):
        motion = (velocity - obstacle.velocity) * dt
        if obstacle.footprint is not None:
            return segment_polygon_clear(position, position + motion, obstacle.footprint, radius + self.margin)
        return self._chord_distance(position - obstacle.position, motion) >= radius + obstacle.radius - 1e-6

    @staticmethod
    def _chord_distance(relative, motion):
        norm = float(np.dot(motion, motion))
        fraction = float(np.clip(-np.dot(relative, motion) / norm, 0, 1)) if norm else 0.0
        return float(np.linalg.norm(relative + fraction * motion))

    def _guard_chords(self, positions, solved, radii, dt, execution_limits=None, static_prefix=None):
        pairs = []
        for i in range(len(self.names)):
            for j in range(i):
                relative = positions[i] - positions[j]
                distance = float(np.linalg.norm(relative))
                body_distance = radii[i] + radii[j]
                if distance < body_distance - 1e-9:
                    raise RuntimeError(
                        f"Actual pedestrian bodies overlap for {self.names[j]}/{self.names[i]}: "
                        f"{distance} < {body_distance} [m]; no pose was projected."
                    )
                # Existing comfort-gap deficits may separate, never deepen.
                pairs.append((i, j, relative, min(distance, body_distance + self.pair_clearance)))
        for i in range(len(self.names)):
            movement = solved[i] * dt
            clearance = radii[i] + self.margin
            if not np.any(movement) or self.grid.segment_clear(positions[i], positions[i] + movement, clearance):
                continue
            if static_prefix is None or not static_prefix[i]:
                solved[i] = 0.0
                continue
            # Keep the heading and any certified travel before the wall. The
            # following joint guard still checks the shortened ray against peers.
            lower, upper = 0.0, 1.0
            for _ in range(12):
                middle = (lower + upper) / 2.0
                if self.grid.segment_clear(positions[i], positions[i] + (middle * solved[i]) * dt, clearance):
                    lower = middle
                else:
                    upper = middle
            speed = math.hypot(*solved[i])
            if lower * speed >= min(0.05, speed / 8):
                solved[i] *= lower
            else:
                repaired = self._repair_wall_roundoff(positions[i], solved[i], clearance, dt, i, execution_limits)
                solved[i] = 0.0 if repaired is None else repaired

        def conflicts():
            return [
                (i, j, relative, required)
                for i, j, relative, required in pairs
                if self._chord_distance(relative, (solved[i] - solved[j]) * dt) < required
            ]

        # Heading/static/dynamic stops change the reciprocal solution. Scale
        # velocity rays to retain heading and swept static certificates.
        for _ in range(len(self.names)):
            violations = conflicts()
            if not violations:
                return
            for i, j, relative, required in violations:
                if self._chord_distance(relative, (solved[i] - solved[j]) * dt) >= required:
                    continue
                if execution_limits is not None:
                    repaired = self._repair_contact_roundoff(
                        positions, solved, radii, dt, i, j, required, execution_limits
                    )
                    if repaired is not None:
                        solved[:] = repaired
                        continue
                solved[[i, j]] = self._limit_peer_rays(relative, solved[[i, j]], required, dt)
        # Crowded dependency cycles can invalidate an earlier pair after a
        # later reduction. Propagate a certified stop only as the final fallback.
        for _ in range(len(self.names)):
            violations = conflicts()
            if not violations:
                return
            for i, j, _, _ in violations:
                solved[i] = solved[j] = 0
        if conflicts():
            raise RuntimeError("No certified executed pedestrian chord retained the configured pair clearance.")

    def _repair_wall_roundoff(self, position, velocity, clearance, dt, index, execution_limits):
        """Recover a near-tangent ORCA ray using the exact swept map check [m]."""
        stalled_since = self.routes[self.names[index]].stalled_since
        if execution_limits is None or stalled_since is None or self._last_time - stalled_since < 1.0:
            return None
        yaws, angles, caps = execution_limits
        epsilon = np.finfo(np.float32).eps
        limit = min(math.sqrt(epsilon) * caps[index], 8 * epsilon * max(1.0, math.hypot(*position)) / dt)
        yaw = math.atan2(velocity[1], velocity[0])
        speed = math.hypot(*velocity)
        for step in (64, 256, 1024, 4096, 8192):
            for sign in (-1, 1):
                angle = yaw + sign * step * epsilon
                if abs(self._angle(angle - yaws[index])) > angles[index]:
                    continue
                target = speed * np.array([math.cos(angle), math.sin(angle)])
                if math.dist(target, velocity) > limit or not self.grid.segment_clear(
                    position, position + target * dt, clearance
                ):
                    continue
                candidate = self._solver.heading_project(
                    target[None, :], np.array([angle]), 0.0, caps[index : index + 1], [index]
                )[0]
                if (
                    math.hypot(*candidate) >= 0.95 * speed
                    and math.dist(candidate, velocity) <= limit
                    and abs(self._angle(math.atan2(candidate[1], candidate[0]) - yaws[index])) <= angles[index] + 1e-7
                    and self.grid.segment_clear(position, position + candidate * dt, clearance)
                ):
                    return candidate
        return None

    def _repair_contact_roundoff(self, positions, velocities, radii, dt, i, j, required, execution_limits):
        """Refine float-scale contact tangents without relaxing any swept clearance [m]."""
        relative = positions[i] - positions[j]
        distance = float(np.linalg.norm(relative))
        epsilon = np.finfo(np.float32).eps
        tolerance = 8 * epsilon * max(1.0, distance, np.linalg.norm(positions[i]), np.linalg.norm(positions[j]))
        swept = self._chord_distance(relative, (velocities[i] - velocities[j]) * dt)
        if distance - required > tolerance or required - swept > tolerance:
            return None
        yaws, angles, caps = execution_limits
        inward = max(0.0, -float(np.dot(relative / distance, velocities[i] - velocities[j])))
        rays = {}
        for index, sign in ((i, 1.0), (j, -1.0)):
            velocity = velocities[index]
            if caps[index] <= 0 or np.linalg.norm(velocity) <= 1e-8:
                continue
            normal = sign * relative / distance
            derivative = float(np.dot(normal, [-velocity[1], velocity[0]]))
            if abs(derivative) <= 1e-12:
                continue
            change = math.copysign(max(64 * epsilon, 2 * inward / abs(derivative)), derivative)
            cosine, sine = math.cos(change), math.sin(change)
            target = np.array([cosine * velocity[0] - sine * velocity[1], sine * velocity[0] + cosine * velocity[1]])
            if np.linalg.norm(target - velocity) > math.sqrt(epsilon) * caps[index]:
                continue
            candidate = self._solver.heading_project(
                target[None, :],
                np.array([math.atan2(target[1], target[0])]),
                0.0,
                np.array([caps[index]]),
                [index],
            )[0]
            difference = float(np.linalg.norm(candidate - velocity))
            if (
                np.linalg.norm(candidate) <= 1e-8
                or difference > math.sqrt(epsilon) * caps[index]
                or difference * dt > tolerance
                or abs(self._angle(math.atan2(candidate[1], candidate[0]) - yaws[index])) > angles[index] + 1e-7
                or not self.grid.segment_clear(
                    positions[index], positions[index] + candidate * dt, radii[index] + self.margin
                )
            ):
                continue
            rays[index] = candidate
        options = [{index: candidate} for index, candidate in rays.items()]
        if len(rays) > 1:
            options.append(rays)
        for option in options:
            candidate = velocities.copy()
            for index, velocity in option.items():
                candidate[index] = velocity
            if all(
                self._chord_distance(positions[a] - positions[b], (candidate[a] - candidate[b]) * dt)
                >= min(float(np.linalg.norm(positions[a] - positions[b])), radii[a] + radii[b] + self.pair_clearance)
                for a in option
                for b in range(len(self.names))
                if a != b
            ):
                return candidate
        return None

    def _limit_peer_rays(self, relative, velocities, required, dt):
        def clear(candidate):
            return self._chord_distance(relative, (candidate[0] - candidate[1]) * dt) >= required

        if clear(velocities):
            return velocities

        def reduce(indices):
            candidate = velocities.copy()
            candidate[indices] = 0
            if not clear(candidate):
                return None
            lower, upper = 0.0, 1.0
            for _ in range(40):
                middle = (lower + upper) / 2
                candidate[indices] = velocities[indices] * middle
                if clear(candidate):
                    lower = middle
                else:
                    upper = middle
            candidate[indices] = velocities[indices] * lower
            return candidate

        options = [candidate for index in (0, 1) if (candidate := reduce([index])) is not None]
        if options:
            return min(options, key=lambda candidate: float(np.sum((candidate - velocities) ** 2)))
        return reduce([0, 1])

    def _target(self, route, state, time_s, dt, radius, parked=()):
        if route.error or route.phase in ("failed", "complete"):
            return None
        if time_s + _SCHEDULE_EPS_S < route.plan.get("start_time", 0.0):
            return None
        behavior = route.plan.get("behavior", "wander")
        if behavior == "stand":
            route.phase = "standing"
            return None
        for _ in range(2 * max(1, len(route.targets)) + 4):
            if route.phase == "turning":
                target = self._current_target(route)
                error = self._angle(math.radians(target.get("yaw", 0.0)) - state["yaw"])
                if abs(error) > 0.025:
                    return None
                route.phase = "dwelling"
                route.deadline = time_s + target.get("wait", 0.0)
            if route.phase == "dwelling":
                if time_s + _SCHEDULE_EPS_S < route.deadline:
                    return None
                route.deadline = None
                self._advance(route)
                if route.phase == "complete":
                    return None
            if behavior == "wander" and route.wander_target is None:
                planning_grid = self._planning_grid(radius)
                mask = planning_grid.free_mask(radius + self.margin)
                free = np.argwhere(mask)
                centers = (free[:, ::-1] + 0.5) * self.grid.resolution_m + self.grid.origin_xy_m
                distances = np.linalg.norm(centers - state["position"][:2], axis=1)
                free = free[(distances >= 1.0) & (distances <= 4.0)]
                candidates = free[self.rng.permutation(len(free))]
                attempts = 0
                while len(candidates) and attempts < 128:
                    cell, candidates = candidates[0], candidates[1:]
                    goal = self.grid.cell_center(tuple(cell))
                    attempts += 1
                    try:
                        route.path = planning_grid.plan(state["position"], goal, radius, 0.15, self.margin)
                    except ValueError:
                        if attempts == 1 and len(candidates):
                            reachable = planning_grid._reachable_mask(state["position"][:2], radius + self.margin)
                            if reachable is not None:
                                connected = reachable[candidates[:, 0], candidates[:, 1]]
                                # Missing grid connectivity is not a proof against continuous direct paths.
                                if np.any(connected):
                                    candidates = candidates[connected]
                        continue
                    route.wander_target = {
                        "position": [*goal, state["position"][2]],
                        "wait": float(self.rng.uniform(1.0, 3.0)),
                        "yaw": None,
                    }
                    break
                if route.wander_target is None:
                    raise ValueError("No reachable wander destination within 1–4 m among the sampled grid candidates.")
            target = self._current_target(route)
            tolerance = float(route.plan.get("arrival_tolerance", 0.15))
            if math.dist(state["position"], target["position"]) <= tolerance:
                route.path = []
                if target.get("yaw") is not None:
                    route.phase = "turning"
                    return None
                route.phase = "dwelling"
                route.deadline = time_s + target.get("wait", 0.0)
                if target.get("wait", 0.0) > 0:
                    return None
                continue
            cached = bool(route.path)
            if not cached:
                route.path = self._plan_from_actual(route, state, target, radius, tolerance)
                route.progress_position = None
            route.phase = "wandering" if behavior == "wander" else "walking"
            route.path = _forward_path(route.path, state["position"])
            self._refresh_parked_route(route, state, target, radius, tolerance, parked)
            visible = self._visible_route_target(route, state["position"], radius)
            if visible is not None:
                if cached and len(route.path) > 2 and tuple(visible) == tuple(route.path[1]):
                    refreshed = self._refresh_braking_corner(route, state, target, visible, radius, tolerance, dt)
                    if refreshed is not None:
                        return refreshed
                return visible
            if route.parked_disks:
                # The live body may have consumed the optional planning slack.
                # Keep the route if its remaining chords still meet the hard gap.
                route.parked_disks = tuple(
                    (np.asarray(center), radius + peer_radius + self.pair_clearance)
                    for _, center, peer_radius in route.parked_context
                )
                visible = self._visible_route_target(route, state["position"], radius)
                if visible is not None:
                    return visible
                route.parked_disks = ()
            # Reciprocal avoidance can enter a connected side bay which has no
            # line of sight to the old simplified waypoints. Retain the goal.
            route.path = self._plan_from_actual(route, state, target, radius, tolerance)
            visible = self._visible_route_target(route, state["position"], radius)
            if visible is not None:
                return visible
            raise ValueError("No reachable next path segment after replanning from the actual body position.")
        raise RuntimeError("Route contains too many immediate transitions.")

    def _refresh_parked_route(self, route, state, target, radius, tolerance, parked):
        """Try one swept-clear bypass around nearby actors that have yet to start [m]."""
        if not parked and not route.parked_context:
            return
        position = state["position"][:2]
        nearby = []
        for peer in parked:
            name, center, peer_radius = peer
            required = radius + peer_radius + self.pair_clearance
            if name == state["name"] or math.dist(position, center) > self.neighbor_distance:
                continue
            if peer in route.parked_context or any(
                _segment_point_distance(np.asarray(a), np.asarray(b), np.asarray(center)) < required
                for a, b in zip(route.path, route.path[1:])
            ):
                nearby.append(peer)
        context = tuple(nearby)
        if context == route.parked_context:
            return
        route.parked_context = context
        route.parked_disks = ()
        route.parked_align = False
        if not context:
            return
        hard = tuple((np.asarray(center), radius + size + self.pair_clearance) for _, center, size in context)
        padded = tuple(
            (center, gap + min(self.path_preference, max(0.0, math.dist(position, center) - gap - 1e-6)))
            for center, gap in hard
        )
        options = (padded, hard) if any(a[1] > b[1] for a, b in zip(padded, hard)) else (hard,)
        for disks in options:
            candidate = replace(route, parked_disks=disks)
            try:
                path = self._plan_from_actual(candidate, state, target, radius, tolerance)
            except ValueError:
                continue
            route.path, route.path_padding_m, route.parked_disks = path, candidate.path_padding_m, disks
            route.travel_heading = None
            route.parked_align = math.hypot(*state.get("velocity", (0.0, 0.0))[:2]) < 0.03
            return

    def _visible_route_target(self, route, position, radius):
        """Reuse a swept-clear cached chord [m] before paying for another search."""
        paddings = (route.path_padding_m, 0.0) if route.path_padding_m else (0.0,)
        for padding in paddings:
            for index in range(len(route.path) - 1, 0, -1):
                point = route.path[index]
                if tuple(point[:2]) != tuple(position[:2]) and self._route_segment_clear(
                    position, point, radius + padding, route.parked_disks
                ):
                    del route.path[1:index]
                    route.path_padding_m = padding
                    return self._corner_target(route, position, radius)
        return None

    def _corner_target(self, route, position, radius):
        """Look through a nearby planning corner using certified chords [m]."""
        reach = float(route.plan.get("speed", 1.0)) * max(self.relaxation, 1 / self.turn_rate)
        return _path_lookahead(
            route.path,
            position,
            reach,
            lambda start, end: self._route_segment_clear(start, end, radius + route.path_padding_m, route.parked_disks),
        )

    def _refresh_braking_corner(self, route, state, target, visible, radius, tolerance, dt):
        """Recover a smoother certified preview before cached point braking [m/s]."""
        position = np.asarray(state["position"][:2])
        speed = float(route.plan.get("speed", 1.0))
        measured = math.hypot(*state.get("velocity", (0.0, 0.0))[:2])
        yaw = state["yaw"]
        delta = np.asarray(visible) - position
        cap = self._approach_speed(speed, delta, yaw, dt)
        speed_step = speed * max(0.2, dt / max(dt, self.relaxation))
        if measured <= cap + speed_step:
            return None
        if route.corner_replan_position is not None and math.dist(position, route.corner_replan_position) < radius:
            return None
        # Try once at a location until the body progresses by its own radius.
        route.corner_replan_position = tuple(position)
        candidate = replace(route, path=[])
        try:
            candidate.path = self._plan_from_actual(candidate, state, target, radius, tolerance)
        except ValueError:
            return None
        point = self._visible_route_target(candidate, state["position"], radius)
        if (
            point is None
            or len(candidate.path) < 3
            or tuple(point) == tuple(candidate.path[1])
            or candidate.path_padding_m < route.path_padding_m
        ):
            return None
        refreshed = np.asarray(point) - position
        if math.hypot(*refreshed) <= (cap + speed_step) * dt:
            return None
        previous_turn = abs(self._angle(math.atan2(delta[1], delta[0]) - yaw))
        refreshed_turn = abs(self._angle(math.atan2(refreshed[1], refreshed[0]) - yaw))
        if refreshed_turn + self.turn_rate * dt >= previous_turn:
            return None
        route.path, route.path_padding_m = candidate.path, candidate.path_padding_m
        return point

    def _plan_from_actual(self, route, state, target, radius, tolerance):
        dz = abs(state["position"][2] - target["position"][2])
        if dz >= tolerance:
            raise ValueError("A planar occupancy map cannot reach this goal height within its arrival tolerance.")
        planar = math.sqrt(tolerance * tolerance - dz * dz)
        start, goal = state["position"], target["position"]
        disks = route.parked_disks
        path = self._planning_grid(radius, disks).plan(start, goal, radius, planar, self.margin)
        route.path_padding_m = 0.0
        padding = self.path_preference
        if padding == 0.0 or not self._route_segment_clear(start, start, radius + padding, disks):
            return path
        if all(self._route_segment_clear(a, b, radius + padding, disks) for a, b in zip(path, path[1:])):
            route.path_padding_m = padding
            return path
        try:
            preferred = self._planning_grid(radius + padding, disks).plan(
                start, goal, radius + padding, planar, self.margin
            )
        except ValueError:
            return path
        original_length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
        preferred_length = sum(math.dist(a, b) for a, b in zip(preferred, preferred[1:]))
        if preferred_length <= original_length + 2 * radius:
            route.path_padding_m = padding
            return preferred
        return path

    @staticmethod
    def _current_target(route):
        return route.wander_target if route.plan.get("behavior", "wander") == "wander" else route.targets[route.index]

    @staticmethod
    def _next_target(route):
        if route.plan.get("behavior", "wander") != "walk" or route.phase == "complete":
            return None
        index = route.index + 1
        if index < len(route.targets):
            return route.targets[index]["position"]
        return route.targets[0]["position"] if route.plan.get("repeat", False) else None

    @staticmethod
    def _advance(route):
        route.path = []
        route.parked_context = route.parked_disks = ()
        route.parked_align = False
        route.progress_position = None
        route.travel_heading = None
        route.corner_replan_position = None
        route.point_target = None
        route.point_recovery = False
        if route.plan.get("behavior", "wander") == "wander":
            route.wander_target = None
            route.phase = "waiting"
            return
        route.index += 1
        route.phase = "waiting"
        if route.index == len(route.targets):
            route.cycles += 1
            if route.plan.get("repeat", False):
                route.index = 0
            else:
                route.index = len(route.targets) - 1
                route.phase = "complete"

    def _scheduled_departure(self, route, state, time_s, dt):
        """Estimate original arrival turn/dwell completion [s], without advancing it."""
        if (
            route.plan.get("behavior", "wander") != "walk"
            or self._next_target(route) is None
            or route.error is not None
            or route.suspended_arrival is not None
        ):
            return None
        if route.phase == "dwelling" and route.deadline is not None:
            return max(time_s, route.deadline)
        target = self._current_target(route)
        if route.phase == "turning" and target.get("yaw") is not None:
            difference = self._angle(math.radians(target["yaw"]) - state["yaw"])
            return time_s + abs(difference) / self.turn_rate + dt + target.get("wait", 0.0)
        return None

    def _stationary_yaw(self, route, state, dt):
        if route.phase != "turning":
            return state["yaw"]
        target = self._current_target(route)
        error = self._angle(math.radians(target["yaw"]) - state["yaw"])
        return self._angle(state["yaw"] + float(np.clip(error, -self.turn_rate * dt, self.turn_rate * dt)))

    @staticmethod
    def _angle(value):
        return math.atan2(math.sin(value), math.cos(value))

    def _passing_bias(self, positions, directions, caps, radii):
        result = np.zeros_like(positions)
        for i in range(len(self.names)):
            for j in range(i):
                key = (j, i)
                relative = positions[i] - positions[j]
                axis = self._encounters.get(key)
                if axis is not None and (
                    np.dot(relative, axis) < -0.3
                    or np.linalg.norm(relative) > self.neighbor_distance * 1.5
                    or caps[i] == 0
                    or caps[j] == 0
                    or np.dot(directions[j], directions[i]) > 0
                ):
                    self._encounters.pop(key, None)
                    axis = None
                if axis is None and caps[i] > 0 and caps[j] > 0 and np.dot(directions[j], directions[i]) < -0.5:
                    along = float(np.dot(relative, directions[j]))
                    if 0 < along < self.neighbor_distance:
                        axis = directions[j].copy()
                        self._encounters[key] = axis
                if axis is not None:
                    motion = directions[i] * caps[i] - directions[j] * caps[j]
                    motion_sq = float(np.dot(motion, motion))
                    closing = -float(np.dot(relative, motion))
                    if closing <= 0 or motion_sq <= 1e-12:
                        continue
                    closest_time = min(self.horizon, closing / motion_sq)
                    miss_distance = float(np.linalg.norm(relative + motion * closest_time))
                    required = radii[i] + radii[j] + self.pair_clearance
                    # RVO2 owns collision avoidance. Only break symmetry when
                    # the desired trajectories conflict, fading before clear lanes.
                    fraction = float(np.clip(2.0 * (required - miss_distance) / required, 0, 1))
                    weight = fraction * fraction * (3 - 2 * fraction)
                    side = np.array([axis[1], -axis[0]])
                    result[j] += side * min(0.15, caps[j] * 0.25) * weight
                    result[i] -= side * min(0.15, caps[i] * 0.25) * weight
        return result

    def _passive_bias(self, positions, directions, caps, radii, dynamic):
        result = np.zeros_like(positions)
        for i in range(len(self.names)):
            if caps[i] <= 0:
                continue
            blockers = [(positions[j], radii[j]) for j in range(len(self.names)) if j != i and caps[j] == 0]
            blockers.extend((obstacle.position, obstacle.radius) for obstacle in dynamic if obstacle.footprint is None)
            for position, radius in blockers:
                relative = position - positions[i]
                along = float(np.dot(relative, directions[i]))
                lateral = abs(float(directions[i, 0] * relative[1] - directions[i, 1] * relative[0]))
                if 0 < along < self.neighbor_distance and lateral < radii[i] + radius + self.pair_clearance:
                    result[i] += np.array([directions[i, 1], -directions[i, 0]]) * min(0.15, caps[i] * 0.25)
        return result

    def close(self):
        """Release the CPU solver without touching ROS or simulator state."""
        self._solver.close()
