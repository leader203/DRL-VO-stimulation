# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observed goal occupancy, certified approach queues and local departure priority."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .interactions import _polygon_clearance, _segment_point_distance
from .passing import _clear, _forward_path, _OverlayGrid
from .planner import OccupancyGrid, _path_lookahead, _planning_obstacles, convex_footprint, segment_polygon_clear


def _segment_distance(a, b, c, d):
    first, second = np.asarray(b) - a, np.asarray(d) - c
    cross = float(first[0] * second[1] - first[1] * second[0])
    if abs(cross) > 1e-14:
        relative = np.asarray(c) - a
        t = float(relative[0] * second[1] - relative[1] * second[0]) / cross
        u = float(relative[0] * first[1] - relative[1] * first[0]) / cross
        if 0 <= t <= 1 and 0 <= u <= 1:
            return 0.0
    return (
        min(_segment_point_distance(np.asarray(a), np.asarray(b), np.asarray(p)) for p in (c, d))
        if np.linalg.norm(second) == 0
        else min(
            *(_segment_point_distance(np.asarray(a), np.asarray(b), np.asarray(p)) for p in (c, d)),
            *(_segment_point_distance(np.asarray(c), np.asarray(d), np.asarray(p)) for p in (a, b)),
        )
    )


class _ReservationGrid(_OverlayGrid):
    def __init__(self, base, radius, margin, peers, obstacles, reservations):
        super().__init__(base, radius, margin, peers, obstacles)
        self.reservations = reservations

    def _goal_cache_base(self, goal, tolerance, clearance):
        if type(self) is not _ReservationGrid:
            return None
        inset = self.resolution_m / math.sqrt(2) + 1e-6
        neighborhood = tolerance + self.resolution_m * math.sqrt(2) + inset + 1e-8
        point = np.asarray(goal[:2], dtype=float)
        if any(
            _segment_point_distance(np.asarray(a), np.asarray(b), point) <= required + neighborhood
            for a, b, required in self.reservations
        ):
            return None
        return self._overlay_goal_cache_base(goal, tolerance, clearance)

    def segment_clear(self, start, end, clearance_m):
        return super().segment_clear(start, end, clearance_m) and all(
            _segment_distance(start[:2], end[:2], a, b) >= required - 1e-9 for a, b, required in self.reservations
        )

    def free_mask(self, clearance_m):
        mask = super().free_mask(clearance_m)
        rows, cols = np.indices(mask.shape)
        points = np.stack(
            (
                self.origin_xy_m[0] + (cols + 0.5) * self.resolution_m,
                self.origin_xy_m[1] + (rows + 0.5) * self.resolution_m,
            ),
            axis=-1,
        )
        inset = self.resolution_m / math.sqrt(2) + 1e-6
        for a, b, required in self.reservations:
            delta = np.asarray(b) - a
            squared = float(np.dot(delta, delta))
            fraction = (
                np.clip(np.sum((points - a) * delta, axis=-1) / squared, 0, 1) if squared else np.zeros(mask.shape)
            )
            mask &= np.linalg.norm(points - a - fraction[..., None] * delta, axis=-1) > required + inset
        return mask


class _NoWaitingRefuge(RuntimeError):
    """No bounded newcomer motion certifies the holder's unchanged exit."""


@dataclass
class _DepartureYield:
    origin: np.ndarray
    refuge: np.ndarray
    waiter: str
    waiter_goal: np.ndarray
    waiter_tolerance: float
    owner_context: tuple
    owner_path: list
    started: float
    deadline: float
    ready: bool = False


@dataclass
class _Hold:
    context: tuple
    target: np.ndarray
    origin: np.ndarray
    next_target: np.ndarray | None
    started: float
    tolerance: float
    departing: bool = False
    path: list = field(default_factory=list)
    exit_path: list = field(default_factory=list)
    safety_phase: str | None = None
    safety_refuge: np.ndarray | None = None
    departure_yield: _DepartureYield | None = None
    egress_retry_s: float | None = None


@dataclass
class _Wait:
    holder: str
    context: tuple
    target: np.ndarray
    origin: np.ndarray
    started: float
    holder_context: tuple = ()
    refuge: np.ndarray | None = None
    staging: np.ndarray | None = None
    deadline: float | None = None
    admitted: bool = False
    parallel_admitted: bool = False
    parallel_prefix: np.ndarray | None = None
    parallel_prefix_retry_s: float | None = None
    admission_path: list = field(default_factory=list)
    admission_retry_s: float | None = None
    admission_blocked_since: float | None = None
    scheduled_departure_s: float | None = None


class GoalOccupancyCoordinator:
    """Coordinate nearby original goals without moving an arrived pedestrian.

    Args:
        grid: Static ROS occupancy map [m].
        settings: Existing map/pair clearance [m] and turn rate [rad/s] settings.

    This layer only returns temporary guidance. The existing SFM/RVO2 solve,
    heading cone and executed-chord certificates remain authoritative. Original
    targets, arrival tolerance, headings, waits and speed are never rewritten.
    """

    def __init__(self, grid: OccupancyGrid, settings: dict) -> None:
        self.grid = grid
        self.margin = float(settings.get("map_clearance_m", 0.05))
        self.pair_clearance = float(settings.get("pair_clearance_m", 0.25))
        self.turn_rate = float(settings.get("max_turn_rate_radps", 2.5))
        self.relaxation = float(settings.get("relaxation_time_s", 0.4))
        self.max_retreat_m = float(settings.get("max_yield_distance_m", 0.4))
        self.control_step_s = 1.0 / float(settings.get("control_hz", 60))
        self._exclusive_doors = [
            (
                convex_footprint(spec["interior_polygon_xy_m"]),
                convex_footprint(spec["swept_footprint_xy_m"]),
                float(spec.get("z_min_m", -math.inf)),
                float(spec.get("z_max_m", math.inf)),
            )
            for spec in settings.get("door_specs", ())
            if spec.get("enabled", True) and spec.get("exclusive_entry", False)
        ]
        self._holders: dict[str, _Hold] = {}
        self._waits: dict[str, _Wait] = {}
        self._paths = {}
        self._door_specs = tuple(settings.get("door_specs", ()))
        self._planning_obstacles: tuple = ()
        self._planning_signature: tuple = ()

    def reset(self) -> None:
        """Forget per-run admissions and fixed refuge origins."""
        self._holders.clear()
        self._waits.clear()
        self._paths.clear()
        self._planning_obstacles = ()
        self._planning_signature = ()

    def _observe_planning_geometry(self, obstacles):
        self._planning_obstacles = tuple(_planning_obstacles(obstacles, self._door_specs))
        signature = tuple(
            (
                obstacle.name,
                tuple(obstacle.position),
                obstacle.radius,
                None if obstacle.footprint is None else tuple(map(tuple, obstacle.footprint)),
            )
            for obstacle in self._planning_obstacles
        )
        if signature != self._planning_signature:
            self._planning_signature = signature
            self._paths.clear()

    @staticmethod
    def _context(intention):
        return tuple(intention.get("route_context", ())), tuple(intention.get("target") or ())

    def _required(self, first, second):
        return float(first["radius_m"]) + float(second["radius_m"]) + self.pair_clearance

    def _active_holder(self, wait):
        return wait.holder in self._holders and self._holders[wait.holder].context == wait.holder_context

    @staticmethod
    def _safety_phase(directive):
        phase = directive.get("phase", "")
        return phase if phase == "door_yield" or phase.startswith("door_close_") else None

    @staticmethod
    def _protect_safety_motion(hold, state, directive):
        hold.safety_phase = GoalOccupancyCoordinator._safety_phase(directive)
        points = [np.asarray(state["position"][:2])]
        if directive.get("target") is not None:
            points.append(np.asarray(directive["target"][:2]))
        if hold.safety_refuge is not None:
            points.append(hold.safety_refuge)
        hold.safety_refuge = max(points, key=lambda point: float(np.linalg.norm(point - hold.origin))).copy()

    @staticmethod
    def _hold_key(name, hold):
        return (
            name,
            hold.context,
            hold.departing,
            hold.safety_phase is not None,
            None if hold.safety_refuge is None else tuple(hold.safety_refuge),
            None if hold.departure_yield is None else (hold.departure_yield.started, hold.departure_yield.ready),
        )

    def _peers(self, name, actors, replacements=None):
        return [
            (
                np.asarray((replacements or {}).get(other, state["position"][:2]), dtype=float),
                self._required(actors[name], state),
            )
            for other, state in actors.items()
            if other != name
        ]

    def _reservations(self, name, actors):
        result = []
        for holder, hold in self._holders.items():
            if holder == name or hold.safety_phase is None:
                continue
            actual = np.asarray(actors[holder]["position"][:2])
            refuge = hold.origin if hold.safety_refuge is None else hold.safety_refuge
            result.append((hold.origin, refuge, self._required(actors[name], actors[holder])))
            if _segment_point_distance(hold.origin, refuge, actual) > 1e-6:
                result.extend(((hold.origin, actual, result[-1][2]), (actual, refuge, result[-1][2])))
        return result

    def _clear(self, name, start, end, actors, obstacles, peers=None, duration=0.0, reservation_escape=False):
        state = actors[name]
        physical = _clear(
            self.grid,
            np.asarray(start[:2]),
            np.asarray(end[:2]),
            state["radius_m"],
            self.margin,
            self._peers(name, actors) if peers is None else peers,
            obstacles,
            duration,
        )
        for a, b, required in self._reservations(name, actors):
            before = _segment_point_distance(a, b, np.asarray(start[:2]))
            threshold = min(required, before) if reservation_escape else required
            if _segment_distance(start[:2], end[:2], a, b) < threshold - 1e-9:
                return False
            if (
                reservation_escape
                and before < required
                and _segment_point_distance(a, b, np.asarray(end[:2])) <= before
            ):
                return False
        return physical

    def _plan(self, name, target, tolerance, actors, peers, obstacles=None):
        """Find an exact original-ball endpoint, including zero-area tangencies."""
        obstacles = self._planning_obstacles if obstacles is None else _planning_obstacles(obstacles, self._door_specs)
        state = actors[name]
        start = np.asarray(state["position"][:2], dtype=float)
        goal = np.asarray(target[:2], dtype=float)
        height = abs(float(state["position"][2]) - float(target[2]))
        if height > tolerance:
            return None
        planar = math.sqrt(max(0.0, tolerance * tolerance - height * height))
        candidates = [goal]
        for center, required in peers:
            difference = goal - center
            distance = float(np.linalg.norm(difference))
            if distance > 1e-12:
                unit = difference / distance
                candidates.extend((goal + unit * max(0.0, planar - min(1e-6, planar * 1e-6)), goal + unit * planar))
                along = (planar**2 - required**2 + distance**2) / (2 * distance)
                if abs(along) <= planar:
                    base = goal - unit * along
                    side = np.array([-unit[1], unit[0]]) * math.sqrt(max(0.0, planar**2 - along**2))
                    candidates.extend((base + side, base - side))
        for point in candidates:
            if (
                all(np.linalg.norm(point - center) >= required - 1e-12 for center, required in peers)
                and math.hypot(float(np.linalg.norm(point - goal)), height) <= tolerance + 1e-12
                and self._clear(name, start, point, actors, obstacles, peers)
            ):
                return [start.tolist(), point.tolist()]
        if planar <= 0:
            return None
        overlay = _ReservationGrid(
            self.grid, state["radius_m"], self.margin, peers, obstacles, self._reservations(name, actors)
        )
        try:
            return overlay.plan(start, goal, state["radius_m"], planar, self.margin)
        except ValueError:
            return None

    @staticmethod
    def _prefix(path, distance):
        result = [list(path[0])]
        for start, end in zip(path, path[1:]):
            length = math.dist(start, end)
            if length > distance:
                result.append((np.asarray(start) + (np.asarray(end) - start) * distance / length).tolist())
                return result
            result.append(list(end))
            distance -= length
        return result

    def _queue_exit_prefix(self, path, peers, reach, holder_tolerance, waiters, intentions):
        last_exit = reach
        elapsed = 0.0
        for start, end in zip(path, path[1:]):
            delta = np.asarray(end) - start
            squared = float(np.dot(delta, delta))
            length = math.sqrt(squared)
            if squared:
                for (center, required), name in zip(peers, waiters):
                    radius = required + holder_tolerance + intentions[name]["arrival_tolerance_m"]
                    relative = np.asarray(start) - center
                    projection = -float(np.dot(relative, delta)) / squared
                    perpendicular = float(np.dot(relative, relative)) - projection**2 * squared
                    window = (radius**2 - perpendicular) / squared
                    if window < 0:
                        continue
                    spread = math.sqrt(window)
                    first, last = max(0.0, projection - spread), min(1.0, projection + spread)
                    if first <= last:
                        last_exit = max(last_exit, elapsed + last * length + 0.015)
            elapsed += length
        return self._prefix(path, last_exit)

    def _exit(self, holder, newcomer, point, actors, intentions, *, replan=True):
        hold = self._holders[holder]
        if hold.safety_phase is not None:
            actual = np.asarray(actors[holder]["position"][:2])
            refuge = hold.origin if hold.safety_refuge is None else hold.safety_refuge
            required = self._required(actors[holder], actors[newcomer])
            if any(
                _segment_point_distance(a, b, np.asarray(point)) < required - 1e-9
                for a, b in ((hold.origin, refuge), (actual, hold.origin), (actual, refuge))
            ):
                return None
            actors = {**actors, holder: {**actors[holder], "position": [*hold.origin, actors[holder]["position"][2]]}}
        if hold.next_target is None:
            return []
        state = actors[holder]
        if not hold.path:
            if not replan:
                return None
            supplied = intentions[holder].get("next_path") or ()
            height = abs(state["position"][2] - hold.next_target[2])
            if height >= hold.tolerance:
                return None
            planar = math.sqrt(hold.tolerance**2 - height**2)
            if (
                len(supplied) >= 2
                and math.dist(supplied[-1][:2], hold.next_target[:2]) <= planar
                and self.grid.segment_clear(state["position"], supplied[0], state["radius_m"] + self.margin)
                and all(
                    self.grid.segment_clear(a, b, state["radius_m"] + self.margin)
                    for a, b in zip(supplied, supplied[1:])
                )
            ):
                hold.path = [list(state["position"][:2]), *(list(p[:2]) for p in supplied)]
            else:
                try:
                    hold.path = self.grid.plan(
                        state["position"],
                        hold.next_target,
                        state["radius_m"],
                        planar,
                        self.margin,
                    )
                except ValueError:
                    return None
        reach = self._required(state, actors[newcomer]) + hold.tolerance + intentions[newcomer]["arrival_tolerance_m"]
        queued = {
            name
            for name, wait in self._waits.items()
            if wait.holder == holder and wait.holder_context == hold.context and name != holder
        }
        queued.add(newcomer)
        waiters = sorted(queued)
        peers = [
            (
                np.asarray(point)
                if name == newcomer
                else np.asarray(
                    self._waits[name].refuge if self._waits[name].refuge is not None else actors[name]["position"][:2]
                ),
                self._required(state, actors[name]),
            )
            for name in waiters
        ]
        if all(
            self._clear(holder, a, b, actors, self._planning_obstacles, peers) for a, b in zip(hold.path, hold.path[1:])
        ):
            return self._execution_exit_prefix(holder, peers, reach, waiters, actors, intentions)
        if not replan:
            return None
        path = self._plan(holder, hold.next_target, hold.tolerance, actors, peers)
        if path is None:
            return None
        hold.path = path
        return self._execution_exit_prefix(holder, peers, reach, waiters, actors, intentions)

    def _execution_exit_prefix(self, holder, peers, reach, waiters, actors, intentions):
        hold = self._holders[holder]
        prefix = self._queue_exit_prefix(hold.path, peers, reach, hold.tolerance, waiters, intentions)
        if (
            hold.exit_path
            and math.dist(hold.exit_path[-1], prefix[-1]) <= 1e-9
            and all(
                self._clear(holder, a, b, actors, self._planning_obstacles, peers)
                for a, b in zip(hold.exit_path, hold.exit_path[1:])
            )
        ):
            return hold.exit_path
        return prefix

    def _exit_reached(self, name, hold, actors, intentions):
        if not hold.exit_path:
            return False
        point = np.asarray(actors[name]["position"][:2])
        end = np.asarray(hold.exit_path[-1])
        if np.linalg.norm(point - end) <= 0.015:
            return True
        if len(hold.exit_path) < 2:
            return False
        direction = end - hold.exit_path[-2]
        length = float(np.linalg.norm(direction))
        if not length:
            return False
        direction /= length
        along = float(np.dot(point - end, direction))
        if along < 0 or np.linalg.norm(point - end - along * direction) > 0.015:
            return False
        return all(
            np.linalg.norm(point - np.asarray(actors[waiter]["position"][:2]))
            >= self._required(actors[name], actors[waiter])
            + hold.tolerance
            + intentions[waiter]["arrival_tolerance_m"]
            - 0.015
            for waiter, wait in self._waits.items()
            if wait.holder == name and wait.holder_context == hold.context
        )

    def _refresh(self, time_s, actors, intentions, priority):
        before = tuple(self._hold_key(name, hold) for name, hold in self._holders.items())
        for name, state in actors.items():
            intention = intentions[name]
            phase = intention.get("phase", state.get("phase"))
            target = intention.get("target")
            context = self._context(intention)
            old = self._holders.get(name)
            safety_phase = self._safety_phase(priority.get(name, {}))
            if old is not None:
                if safety_phase is not None:
                    self._protect_safety_motion(old, state, priority[name])
                    continue
                old.safety_phase, old.safety_refuge = None, None
                if old.departure_yield is not None:
                    continue
            if phase in ("turning", "dwelling", "complete") and target is not None:
                tolerance = float(intention["arrival_tolerance_m"])
                if math.dist(state["position"], target) > tolerance + 1e-6:
                    raise ValueError(f"Goal holder {name} is outside its original arrival tolerance.")
                if old is None or old.context != context:
                    next_target = intention.get("next_target")
                    self._holders[name] = _Hold(
                        context,
                        np.asarray(target),
                        np.asarray(state["position"][:2]),
                        None if next_target is None else np.asarray(next_target),
                        time_s,
                        tolerance,
                    )
                if safety_phase is not None:
                    self._protect_safety_motion(self._holders[name], state, priority[name])
            elif name in self._holders:
                hold = self._holders[name]
                reached_next_target = False
                if intention.get("active") and context != hold.context:
                    hold.departing = True
                    if target is not None:
                        if hold.next_target is None or not np.array_equal(target, hold.next_target):
                            hold.path, hold.exit_path = [], []
                            hold.egress_retry_s = None
                        else:
                            reached_next_target = (
                                math.dist(state["position"], target) <= intention["arrival_tolerance_m"]
                            )
                        hold.next_target = np.asarray(target)
                at_exit = self._exit_reached(name, hold, actors, intentions)
                clear = (
                    np.linalg.norm(np.asarray(state["position"][:2]) - hold.origin)
                    > 2 * state["radius_m"] + self.pair_clearance + 2 * hold.tolerance
                )
                bound = any(
                    wait.holder == name and wait.holder_context == hold.context for wait in self._waits.values()
                )
                # A door crossing can reach the next original goal off the
                # reserved exit line. Release the old hold so arrival can run.
                if hold.departing and (at_exit or (clear and (not bound or reached_next_target))):
                    del self._holders[name]
                    for wait in self._waits.values():
                        if wait.holder == name and wait.holder_context == hold.context:
                            wait.admission_retry_s = None
        if tuple(self._hold_key(name, hold) for name, hold in self._holders.items()) != before:
            self._paths.clear()

    def _arrival_path(self, name, actors, intentions):
        peers = [
            (
                hold.origin if hold.safety_phase else np.asarray(actors[holder]["position"][:2]),
                self._required(actors[name], actors[holder]),
            )
            for holder, hold in self._holders.items()
            if holder != name
        ]
        key = (
            name,
            self._context(intentions[name]),
            tuple(self._hold_key(holder, hold) for holder, hold in self._holders.items()),
        )
        if key not in self._paths:
            target = intentions[name]["target"]
            tolerance = intentions[name]["arrival_tolerance_m"]
            height = abs(actors[name]["position"][2] - target[2])
            planar = math.sqrt(max(0.0, tolerance**2 - height**2))
            arrived = math.dist(actors[name]["position"], target) <= tolerance and self._clear(
                name, actors[name]["position"], actors[name]["position"], actors, (), peers
            )
            robust = all(
                np.linalg.norm(np.asarray(target[:2]) - center) + planar > required + 1e-6 for center, required in peers
            )
            self._paths[key] = self._plan(name, target, tolerance, actors, peers) if arrived or robust else None
        return self._paths[key]

    def _zero_width(self, name, actors, intention):
        target = intention["target"]
        planar = math.sqrt(
            max(0.0, intention["arrival_tolerance_m"] ** 2 - (actors[name]["position"][2] - target[2]) ** 2)
        )
        return any(
            abs(
                np.linalg.norm(
                    np.asarray(target[:2]) - (hold.origin if hold.safety_phase else actors[holder]["position"][:2])
                )
                + planar
                - self._required(actors[name], actors[holder])
            )
            <= 1e-6
            for holder, hold in self._holders.items()
            if holder != name
        )

    def _refresh_arrival_endpoint(self, name, path, actors, intentions, obstacles, holder=None):
        if len(path) < 2:
            return
        state, intention = actors[name], intentions[name]
        current = np.asarray(state["position"][:2])
        goal = np.asarray(intention["target"][:2])
        tolerance = intention["arrival_tolerance_m"]
        height = abs(state["position"][2] - intention["target"][2])
        if (
            height > tolerance
            or math.dist(state["position"], intention["target"]) <= tolerance
            or np.linalg.norm(current - path[-1]) > max(state["radius_m"], intention["speed_mps"] * self.relaxation)
        ):
            return
        planar = math.sqrt(max(0.0, tolerance * tolerance - height * height))
        peers = self._peers(name, actors)

        def margin(point):
            return min(
                [
                    planar - float(np.linalg.norm(point - goal)),
                    *(float(np.linalg.norm(point - center)) - required for center, required in peers),
                ]
            )

        candidates = [goal]
        for center, required in peers:
            difference = goal - center
            distance = float(np.linalg.norm(difference))
            if distance:
                # Balance original arrival-ball room against this body's clearance.
                offset = float(np.clip((planar + required - distance) / 2, 0.0, planar))
                candidates.append(goal + difference * (offset / distance))
        old_margin = margin(np.asarray(path[-1]))
        wait = self._waits.get(name)
        if holder is None and wait is not None and self._active_holder(wait):
            holder = wait.holder
        for point in sorted(candidates, key=lambda point: -margin(point)):
            if (
                margin(point) <= old_margin + 1e-6
                or math.hypot(float(np.linalg.norm(point - goal)), height) > tolerance
            ):
                continue
            if not self._clear(name, current, point, actors, obstacles):
                continue
            if holder is not None and self._exit(holder, name, point, actors, intentions, replan=False) is None:
                continue
            path[:] = [current.tolist(), point.tolist()]
            return

    def _approach_objective(self, name, path, actors, intentions, obstacles, token, holder=None):
        if math.dist(actors[name]["position"], intentions[name]["target"]) <= intentions[name]["arrival_tolerance_m"]:
            return None
        self._refresh_arrival_endpoint(name, path, actors, intentions, obstacles, holder)
        target = self._guide(name, path, actors, obstacles, intentions[name]["speed_mps"])
        return None if target is None else {"target": target, "phase": "goal_approach", "token": token}

    def _blocked_guidance(self, name, actors, intention):
        path = intention.get("path") or [actors[name]["position"][:2], intention["target"][:2]]
        return any(
            _segment_distance(a[:2], b[:2], c, d) < required
            for a, b in zip(path, path[1:])
            for c, d, required in self._reservations(name, actors)
        ) or any(
            _segment_point_distance(np.asarray(a[:2]), np.asarray(b[:2]), np.asarray(actors[holder]["position"][:2]))
            < self._required(actors[name], actors[holder])
            for a, b in zip(path, path[1:])
            for holder in self._holders
            if holder != name
        )

    def _release_waits(self, time_s, actors, intentions):
        for name, wait in list(self._waits.items()):
            if intentions[name].get("active") and self._context(intentions[name]) == wait.context:
                continue
            if (
                wait.admitted
                and name not in self._holders
                and math.dist(actors[name]["position"], wait.target) <= intentions[name]["arrival_tolerance_m"]
            ):
                target = intentions[name].get("target")
                self._holders[name] = _Hold(
                    wait.context,
                    wait.target,
                    np.asarray(actors[name]["position"][:2]),
                    None if target is None else np.asarray(target),
                    time_s,
                    intentions[name]["arrival_tolerance_m"],
                    departing=True,
                )
            if wait.admitted and name in self._holders:
                for other in self._waits.values():
                    if other is not wait and other.holder == wait.holder:
                        other.holder = name
                        other.holder_context = self._holders[name].context
            del self._waits[name]

    def _guide(self, name, path, actors, obstacles, speed=0.0, extra_clear=None):
        # Door interlocks own admission; execution still checks every live leaf.
        obstacles = _planning_obstacles(obstacles, self._door_specs)

        def clear(start, end):
            return self._clear(name, start, end, actors, obstacles) and (extra_clear is None or extra_clear(start, end))

        path[:] = _forward_path(path, actors[name]["position"])
        for index in range(len(path) - 1, 0, -1):
            point = path[index]
            if tuple(point[:2]) == tuple(actors[name]["position"][:2]):
                continue
            if clear(actors[name]["position"], point):
                del path[1:index]
                return _path_lookahead(
                    path,
                    actors[name]["position"],
                    speed * max(self.relaxation, 1 / self.turn_rate),
                    clear,
                )
        return None

    def _egress_guide(self, name, hold, actors, obstacles, speed, time_s):
        point = self._guide(name, hold.exit_path, actors, obstacles, speed)
        if point is not None:
            return point
        if hold.egress_retry_s is not None and time_s - hold.egress_retry_s < 0.5 - 1e-9:
            return None
        hold.egress_retry_s = time_s
        # A sidestep can lose the old exit chord. Reconnect from the measured
        # position without changing where this queue is released.
        endpoint = hold.exit_path[-1]
        path = self._plan(name, [*endpoint, actors[name]["position"][2]], 1e-6, actors, self._peers(name, actors))
        if path is None:
            return None
        if tuple(path[-1]) != tuple(endpoint):
            if not self._clear(name, path[-1], endpoint, actors, self._planning_obstacles):
                return None
            path.append(endpoint)
        point = self._guide(name, path, actors, obstacles, speed)
        if point is not None:
            hold.exit_path = path
        return point

    def _cached_departure_route(self, hold, intention, position):
        """Find the remaining route [m] to the same departure goal."""
        target = intention["target"]
        if (
            hold.safety_phase is not None
            or hold.departure_yield is not None
            or not np.array_equal(target, hold.next_target)
        ):
            return []
        path = _forward_path(intention.get("path", ()), position)
        if target is None or len(path) < 2 or math.dist([*path[-1][:2], position[2]], target) > hold.tolerance + 1e-9:
            return []
        return path

    def _admission_guide(self, name, wait, actors, obstacles, speed=0.0):
        path = wait.admission_path
        if len(path) < 2:
            return None
        return self._guide(name, path, actors, obstacles, speed)

    def _admission_objective(self, name, wait, actors, intentions, obstacles, time_s):
        wait.admitted = True
        if math.dist(actors[name]["position"], intentions[name]["target"]) <= intentions[name]["arrival_tolerance_m"]:
            wait.admission_blocked_since = None
            return None
        self._refresh_arrival_endpoint(name, wait.admission_path, actors, intentions, obstacles)
        target = self._admission_guide(name, wait, actors, obstacles, intentions[name]["speed_mps"])
        if target is None and (wait.admission_retry_s is None or time_s - wait.admission_retry_s >= 0.5 - 1e-9):
            wait.admission_retry_s = time_s
            path = self._plan(
                name,
                intentions[name]["target"],
                intentions[name]["arrival_tolerance_m"],
                actors,
                self._peers(name, actors),
            )
            if path is not None:
                wait.admission_path = path
                target = self._admission_guide(name, wait, actors, obstacles, intentions[name]["speed_mps"])
        if target is not None:
            wait.admission_blocked_since = None
            return {"target": target, "phase": "goal_approach", "token": ("goal_admit", wait.holder, name)}
        if wait.holder not in self._holders:
            if wait.admission_blocked_since is None:
                wait.admission_blocked_since = time_s
            elif time_s - wait.admission_blocked_since > 12.0:
                raise RuntimeError(
                    f"No certified goal admission progress for 12 simulation seconds after {wait.holder} "
                    f"released {name}; original goal retained."
                )
        else:
            wait.admission_blocked_since = None
        return {"target": None, "phase": "goal_wait", "token": ("goal_admit_wait", wait.holder, name)}

    def _parallel_admission(
        self, holder, hold, egress_target, actors, intentions, obstacles, time_s, objectives, *, departure_route=()
    ):
        """Resume one ordinary waiter [m] without consuming the owner's certified exit."""
        queue = [name for name, wait in self._waits.items() if wait.holder == holder and self._active_holder(wait)]
        if len(queue) != 1 or hold.safety_phase is not None or hold.departure_yield is not None:
            return
        name = queue[0]
        intention = intentions[name]
        if objectives.get(name, {}).get("phase") != "goal_wait" or any(
            item.get("queued_door_entry") or item.get("door_outgoing") for item in (intention, intentions[holder])
        ):
            return
        wait = self._waits[name]
        if wait.scheduled_departure_s is not None:
            return
        start = actors[holder]["position"][:2]
        goal = intention["target"]
        tolerance = intention["arrival_tolerance_m"]
        remaining = _forward_path(hold.exit_path, start)
        exit_chords = [(start, egress_target), *zip(remaining, remaining[1:])]
        exit_chords.extend(zip(departure_route, departure_route[1:]))
        if not self._clear(holder, start, egress_target, actors, obstacles):
            return

        current = np.asarray(actors[name]["position"][:2])
        stationary_clearance = min(
            _segment_point_distance(np.asarray(a), np.asarray(b), current) for a, b in exit_chords
        )
        hard_clearance = self._required(actors[holder], actors[name])
        # A waiter already on the safe boundary can move away from the exit tube.
        required = max(hard_clearance, min(hard_clearance + 0.001, stationary_clearance))

        def exit_clear(a, b):
            return all(_segment_distance(a[:2], b[:2], c, d) >= required for c, d in exit_chords)

        def guide(path):
            if len(path) < 2 or not all(exit_clear(a, b) for a, b in zip(path, path[1:])):
                return None
            point = self._guide(name, path, actors, obstacles, intention["speed_mps"], exit_clear)
            if point is not None and self._clear(name, actors[name]["position"], point, actors, obstacles):
                return point
            return None

        point = None
        if math.dist(start, goal[:2]) + tolerance > required:
            position = actors[name]["position"]
            path = _forward_path(wait.admission_path, position)
            point = guide(path)
            if point is None and abs(position[2] - goal[2]) < tolerance:
                path = [list(position[:2]), list(goal[:2])]
                point = guide(path)
            if point is None and (wait.admission_retry_s is None or time_s - wait.admission_retry_s >= 0.5 - 1e-9):
                wait.admission_retry_s = time_s
                path = self._plan(name, goal, tolerance, actors, self._peers(name, actors)) or []
                point = guide(path)
        if point is not None and self._parallel_step_clear(name, point, actors, obstacles, intention["speed_mps"]):
            wait.admission_path = path
            wait.admitted = True
            wait.parallel_admitted = True
            objectives[name] = {
                "target": point,
                "phase": "goal_approach",
                "token": ("goal_admit", holder, name),
                "certified_ray": True,
            }
            return

        # A waiter well clear of the exit needs no extra staging maneuver.
        if wait.parallel_prefix is None and stationary_clearance > required + 0.05:
            return
        prefix = self._parallel_prefix(name, wait, actors, intentions, obstacles, time_s, exit_clear)
        if prefix is not None:
            wait.parallel_admitted = True
            objectives[name] = {
                "target": prefix.tolist(),
                "phase": "goal_yield",
                "token": ("goal_admit_prefix", holder, name, wait.started),
                "origin": wait.origin.tolist(),
                "max_displacement_m": self.max_retreat_m,
                "certified_ray": True,
            }

    def _resume_interrupted_prefix(self, holder, hold, actors, intentions, obstacles, priority, objectives):
        """Continue an admitted prefix [m] while its owner is safely stationary."""
        if hold.safety_phase is not None or hold.departure_yield is not None:
            return
        queue = [name for name, wait in self._waits.items() if wait.holder == holder and self._active_holder(wait)]
        if len(queue) != 1:
            return
        name = queue[0]
        wait = self._waits[name]
        prefix = wait.parallel_prefix
        if (
            name in priority
            or objectives.get(name, {}).get("phase") != "goal_wait"
            or not wait.parallel_admitted
            or wait.admitted
            or wait.refuge is not None
            or wait.scheduled_departure_s is not None
            or prefix is None
            or any(
                item.get("queued_door_entry") or item.get("door_outgoing")
                for item in (intentions[name], intentions[holder])
            )
        ):
            return
        held = actors[holder]
        velocity = held.get("velocity")
        position = held["position"]
        if velocity is None or math.hypot(*velocity[:2]) > min(0.03, 0.25 * intentions[holder]["speed_mps"]):
            return
        if not self._clear(holder, position, position, actors, obstacles, duration=self.control_step_s):
            return
        start = actors[name]["position"][:2]
        if (
            math.dist(start, prefix) <= 0.015
            or math.dist(wait.origin, prefix) > self.max_retreat_m + 1e-9
            or not self._staging_clear(name, wait, start, prefix, actors, intentions, obstacles)
            or not self._parallel_step_clear(name, prefix, actors, obstacles, intentions[name]["speed_mps"])
        ):
            return
        objectives[holder] = {"target": None, "phase": "goal_wait", "token": ("goal_exit_hold", holder, wait.started)}
        objectives[name] = {
            "target": prefix.tolist(),
            "phase": "goal_yield",
            "token": ("goal_admit_prefix", holder, name, wait.started),
            "origin": wait.origin.tolist(),
            "max_displacement_m": self.max_retreat_m,
            "certified_ray": True,
        }

    def _parallel_prefix(self, name, wait, actors, intentions, obstacles, time_s, exit_clear):
        """Find one bounded forward segment [m] while the original goal is unavailable."""
        if wait.refuge is not None or wait.admitted:
            return None
        position, intention = actors[name]["position"], intentions[name]
        current, goal = np.asarray(position[:2]), np.asarray(intention["target"][:2])
        if math.dist(current, wait.origin) > self.max_retreat_m:
            return None
        speed = intention["speed_mps"]

        def clear(point):
            return (
                exit_clear(current, point)
                and self._staging_clear(name, wait, current, point, actors, intentions, obstacles)
                and self._parallel_step_clear(name, point, actors, obstacles, speed)
            )

        prefix = wait.parallel_prefix
        if prefix is not None:
            return prefix if math.dist(current, prefix) > 0.015 and clear(prefix) else None
        if wait.parallel_prefix_retry_s is not None and time_s - wait.parallel_prefix_retry_s < 0.5 - 1e-9:
            return None
        wait.parallel_prefix_retry_s = time_s
        # Keep this temporary guidance outside the original arrival ball.
        reach = min(
            self.max_retreat_m - min(0.002, 0.1 * self.max_retreat_m),
            math.dist(position, intention["target"]) - intention["arrival_tolerance_m"] - 1e-6,
        )
        if reach <= 0.05:
            return None
        distance = math.dist(current, goal)
        bearing = math.atan2(goal[1] - current[1], goal[0] - current[0])
        candidates = []
        for angle in np.linspace(-math.pi / 2, math.pi / 2, 13) + bearing:
            direction = np.array([math.cos(angle), math.sin(angle)])
            for fraction in (0.25, 0.5, 0.75, 1.0):
                point = current + direction * (reach * fraction)
                progress = distance - math.dist(point, goal)
                if progress > 0.05 and math.dist(point, wait.origin) <= self.max_retreat_m - 1e-6:
                    candidates.append((progress, point))
        for _, point in sorted(candidates, key=lambda item: -item[0]):
            if clear(point):
                wait.parallel_prefix = point
                return point
        return None

    def _parallel_step_clear(self, name, point, actors, obstacles, speed, *, lock_egress_heading=False):
        # Alignment can leave the waiter stationary. Check both possibilities
        # against the measured live footprints over one control interval.
        start = np.asarray(actors[name]["position"][:2])
        delta = np.asarray(point) - start
        distance = math.hypot(*delta)
        end = start + delta * min(1.0, speed * self.control_step_s / distance) if distance else start
        if lock_egress_heading and any(
            _segment_point_distance(start, end, center) < min(math.dist(start, center), required) - 1e-12
            for center, required in self._peers(name, actors)
        ):
            # Planning tolerates roundoff at contact. Pinning the heading on
            # such a chord can prevent the solver from steering away from it.
            return False
        return self._clear(name, start, start, actors, obstacles, duration=self.control_step_s) and self._clear(
            name, start, end, actors, obstacles, duration=self.control_step_s
        )

    def _yield_gate_clear(self, name, current, point, actors, intentions):
        state, intention = actors[name], intentions[name]
        return not any(
            z_max >= state["position"][2]
            and z_min <= state["position"][2] + state.get("height_m", 1.8)
            and _polygon_clearance(intention["target"], interior) <= 0
            and _polygon_clearance(current, interior) >= 0
            and (
                not segment_polygon_clear(current, point, interior, 0.0)
                or not segment_polygon_clear(current, point, sweep, state["radius_m"] + self.margin)
            )
            for interior, sweep, z_min, z_max in self._exclusive_doors
        )

    def _waiting_refuge(self, name, wait, actors, intentions, obstacles):
        current = np.asarray(actors[name]["position"][:2])
        hold = self._holders[wait.holder]
        reach = self.max_retreat_m - min(0.002, 0.1 * self.max_retreat_m)
        candidates = [
            wait.origin + reach * fraction * np.array([math.cos(angle), math.sin(angle)])
            for fraction in (0.25, 0.5, 0.75, 1.0)
            for angle in np.linspace(0, 2 * math.pi, 24, endpoint=False)
        ]
        modes = [True]
        if hold.safety_phase is None:
            baseline = self._plan(wait.holder, hold.next_target, hold.tolerance, actors, [])
            if baseline is not None:
                hold.path = baseline
                modes.insert(0, False)
        for replan in modes:
            for point in candidates:
                if not self._yield_gate_clear(name, current, point, actors, intentions):
                    continue
                if not self._clear(
                    name,
                    current,
                    point,
                    actors,
                    obstacles,
                    duration=math.dist(current, point) / intentions[name]["speed_mps"],
                    reservation_escape=True,
                ):
                    continue
                path = self._exit(wait.holder, name, point, actors, intentions, replan=replan)
                if path is not None:
                    wait.refuge = point
                    return path
        raise _NoWaitingRefuge(
            f"No certified goal waiting refuge within {self.max_retreat_m:g} m for {name}; "
            f"{wait.holder} and original goals retained."
        )

    def _staging_clear(self, name, wait, start, end, actors, intentions, obstacles):
        actor = actors[name]
        clearance = actor["radius_m"] + self.margin
        if not self._clear(
            name, start, end, actors, obstacles, duration=math.dist(start, end) / intentions[name]["speed_mps"]
        ):
            return False
        for interior, sweep, z_min, z_max in self._exclusive_doors:
            if z_max >= actor["position"][2] and z_min <= actor["position"][2] + actor.get("height_m", 1.8):
                if not segment_polygon_clear(start, end, interior, 0.0) or not segment_polygon_clear(
                    start, end, sweep, clearance
                ):
                    return False
        holder_path = self._holders[wait.holder].path
        required = self._required(actor, actors[wait.holder])
        # Preserve the holder's exit even while the newcomer is still moving.
        return all(_segment_distance(start, end, a, b) >= required - 1e-9 for a, b in zip(holder_path, holder_path[1:]))

    def _portal_staging(self, name, wait, actors, intentions, obstacles):
        current = wait.origin
        actor = actors[name]
        clearance = actor["radius_m"] + self.margin
        route = np.asarray(intentions[name]["target"][:2]) - current
        heading = np.array([math.cos(actor["yaw"]), math.sin(actor["yaw"])])
        reach = self.max_retreat_m - min(0.002, 0.1 * self.max_retreat_m)
        for spec in self._door_specs:
            if not spec.get("enabled", True) or not spec.get("exclusive_entry", False):
                continue
            if not (
                spec.get("z_max_m", math.inf) >= actor["position"][2]
                and spec.get("z_min_m", -math.inf) <= actor["position"][2] + actor.get("height_m", 1.8)
            ):
                continue
            portal = np.asarray(spec.get("doorway_segment_xy_m"), dtype=float)
            if portal.shape != (2, 2):
                continue
            a, b = portal
            edge = b - a
            width = math.hypot(*edge)
            if not math.isfinite(width) or width <= 1e-10:
                continue
            outward = np.array([-edge[1], edge[0]]) / width
            if np.dot(np.mean(spec["interior_polygon_xy_m"], axis=0) - a, outward) > 0:
                outward = -outward
            offset = float(np.dot(current - a, outward))
            distance = _segment_point_distance(a, b, current)
            if offset <= 0.0 or distance > clearance + reach / 2:
                continue
            for direction in (heading, outward):
                normal_progress = float(np.dot(direction, outward))
                if normal_progress < 0.5 or np.dot(direction, route) <= 0 or np.dot(direction, heading) <= 0:
                    continue
                travel = min(reach, (clearance + 0.75 * reach - offset) / normal_progress)
                for step in (travel, travel / 2):
                    point = current + step * direction
                    if step <= 0.05 or _segment_point_distance(a, b, point) <= distance + 0.05:
                        continue
                    if self._staging_clear(name, wait, current, point, actors, intentions, obstacles):
                        return point
        return None

    def _await_departure(self, name, wait, actors, intentions, obstacles, time_s):
        owner = intentions[wait.holder]
        if wait.scheduled_departure_s is None:
            hold = self._holders[wait.holder]
            departed = hold.departing and owner.get("active")
            scheduled = time_s if departed else owner.get("scheduled_departure_s")
            if (
                hold.next_target is None
                or hold.safety_phase is not None
                or not departed
                and owner.get("phase") not in ("turning", "dwelling")
                or scheduled is None
                or not math.isfinite(scheduled)
            ):
                return None
            wait.scheduled_departure_s = float(scheduled)
            wait.deadline = (
                scheduled + 4.0 + math.pi / self.turn_rate + self.max_retreat_m / intentions[name]["speed_mps"]
            )
        if time_s > wait.deadline:
            raise RuntimeError(f"Scheduled goal departure waiting timed out for {name}; original goals retained.")
        current = actors[name]["position"][:2]
        if not self._clear(name, current, current, actors, obstacles):
            raise RuntimeError(f"No certified stationary goal queue for {name}; original goals retained.")
        return {
            "target": None,
            "phase": "goal_wait",
            "token": ("goal_departure_pending", wait.holder, name, wait.started),
        }

    def _waiting_objective(self, name, wait, actors, intentions, obstacles, time_s, *, new_wait=False):
        holder = wait.holder
        current = np.asarray(actors[name]["position"][:2])
        hold = self._holders[holder]
        exit_path = self._exit(holder, name, current, actors, intentions)
        token = ("goal", holder, hold.started, name)
        if exit_path and wait.refuge is None and not hold.departing and hold.safety_phase is None:
            if new_wait:
                wait.staging = self._portal_staging(name, wait, actors, intentions, obstacles)
            if (
                wait.staging is not None
                and math.dist(current, wait.staging) > 0.015
                and time_s - wait.started < 1.0 + math.pi / self.turn_rate
                and self._staging_clear(name, wait, current, wait.staging, actors, intentions, obstacles)
            ):
                hold.exit_path = exit_path
                return {
                    "target": wait.staging.tolist(),
                    "phase": "goal_yield",
                    "token": (*token, "stage"),
                    "origin": wait.origin.tolist(),
                    "max_displacement_m": self.max_retreat_m,
                    "certified_ray": True,
                }
        wait.staging = None
        if wait.refuge is None and exit_path is None:
            if wait.scheduled_departure_s is not None:
                return self._await_departure(name, wait, actors, intentions, obstacles, time_s)
            try:
                exit_path = self._waiting_refuge(name, wait, actors, intentions, obstacles)
            except _NoWaitingRefuge:
                pending = self._await_departure(name, wait, actors, intentions, obstacles, time_s)
                if pending is None:
                    raise
                return pending
            wait.deadline = time_s + 4.0 + math.pi / self.turn_rate + self.max_retreat_m / intentions[name]["speed_mps"]
        if wait.refuge is not None and (np.linalg.norm(current - wait.refuge) > 0.015 or exit_path is None):
            if time_s > wait.deadline:
                raise RuntimeError(f"Bounded goal waiting refuge timed out for {name}; original goal retained.")
            if not self._yield_gate_clear(name, current, wait.refuge, actors, intentions) or not self._clear(
                name,
                current,
                wait.refuge,
                actors,
                obstacles,
                duration=math.dist(current, wait.refuge) / intentions[name]["speed_mps"],
                reservation_escape=True,
            ):
                raise RuntimeError(f"Goal waiting refuge no longer has a certified segment for {name}.")
            hold.exit_path = exit_path or self._exit(holder, name, wait.refuge, actors, intentions) or []
            return {
                "target": wait.refuge.tolist(),
                "phase": "goal_yield",
                "token": (*token, "yield"),
                "origin": wait.origin.tolist(),
                "max_displacement_m": self.max_retreat_m,
                "certified_ray": True,
            }
        if exit_path is None:
            raise RuntimeError(f"Queued pedestrian {name} would block {holder}'s certified exit.")
        hold.exit_path = exit_path
        return {"target": None, "phase": "goal_wait", "token": (*token, "wait")}

    def _departure_candidate(self, holder, waiter, actors, intentions, obstacles, time_s, *, first_certified=False):
        state, intention = actors[holder], intentions[holder]
        current = np.asarray(state["position"][:2])
        origin = np.asarray(
            intention.get("yield_origin_xy_m") if intention.get("yield_origin_xy_m") is not None else current,
            dtype=float,
        )
        if (
            origin.shape != (2,)
            or not np.isfinite(origin).all()
            or np.linalg.norm(current - origin) > self.max_retreat_m + 1e-7
        ):
            raise RuntimeError(
                f"Invalid fixed departure-yield origin for {holder}; the {self.max_retreat_m:g} m budget cannot renew."
            )
        options = []
        candidates = [current]
        reach = self.max_retreat_m - min(0.002, 0.1 * self.max_retreat_m)
        for angle in np.linspace(0, 2 * math.pi, 24, endpoint=False):
            direction = np.array([math.cos(angle), math.sin(angle)])
            candidates.extend(origin + reach * fraction * direction for fraction in (1.0, 0.75, 0.5, 0.25))
        if first_certified:
            candidates.sort(
                key=lambda point: (
                    float(np.linalg.norm(point - current)),
                    abs(
                        math.remainder(
                            math.atan2(point[1] - current[1], point[0] - current[0]) - state["yaw"], 2 * math.pi
                        )
                    ),
                )
            )
        goal, tolerance = intentions[waiter]["target"], intentions[waiter]["arrival_tolerance_m"]
        height = abs(actors[waiter]["position"][2] - goal[2])
        planar = math.sqrt(max(0.0, tolerance**2 - height**2))
        required = self._required(state, actors[waiter])
        for point in candidates:
            if not self._yield_gate_clear(holder, current, point, actors, intentions):
                continue
            if np.linalg.norm(np.asarray(goal[:2]) - point) + planar <= required + 1e-6:
                continue
            if not self._clear(
                holder, current, point, actors, obstacles, duration=math.dist(current, point) / intention["speed_mps"]
            ):
                continue
            hypothetical = {**actors, holder: {**state, "position": [*point, state["position"][2]]}}
            arrival = self._plan(waiter, goal, tolerance, hypothetical, self._peers(waiter, hypothetical))
            if arrival is None:
                continue
            endpoint = [*arrival[-1], actors[waiter]["position"][2]]
            hypothetical = {**hypothetical, waiter: {**actors[waiter], "position": endpoint}}
            onward = self._plan(
                holder,
                intention["target"],
                intention["arrival_tolerance_m"],
                hypothetical,
                self._peers(holder, hypothetical),
            )
            if onward is None:
                continue
            distance = sum(math.dist(a, b) for a, b in zip(arrival, arrival[1:]))
            cost = distance + sum(math.dist(a, b) for a, b in zip(onward, onward[1:])) + math.dist(current, point)
            options.append((cost, point.copy(), onward, distance))
            if first_certified:
                break
        if not options:
            raise RuntimeError(
                f"No certified departure-yield refuge within {self.max_retreat_m:g} m for {holder}/{waiter}; "
                "original goals retained."
            )
        _, refuge, path, distance = min(options, key=lambda item: item[0])
        deadline = (
            time_s
            + 4.0
            + math.pi / self.turn_rate
            + 4 * max(self.max_retreat_m / intention["speed_mps"], distance / intentions[waiter]["speed_mps"])
        )
        return _DepartureYield(
            origin.copy(),
            refuge,
            waiter,
            np.asarray(goal).copy(),
            tolerance,
            self._context(intention),
            path,
            time_s,
            deadline,
        )

    def _departure_cycles(self, time_s, actors, intentions, obstacles, priority):
        objectives = {}
        for holder, hold in self._holders.items():
            intention = intentions[holder]
            transaction = hold.departure_yield
            if transaction is None:
                if not (hold.departing and intention.get("active") and holder not in priority):
                    continue
                waiters = [
                    name
                    for name, wait in self._waits.items()
                    if wait.holder == holder
                    and self._active_holder(wait)
                    and (
                        (intention.get("queued_door_entry") and intentions[name].get("door_outgoing"))
                        or (
                            wait.scheduled_departure_s is not None
                            and name not in priority
                            and self._exit(holder, name, actors[name]["position"][:2], actors, intentions) is None
                        )
                    )
                ]
                if not waiters:
                    continue
                waiter = min(waiters, key=lambda name: (self._waits[name].started, name))
                ordinary = self._waits[waiter].scheduled_departure_s is not None
                departure_intentions = intentions
                if ordinary:
                    if time_s > self._waits[waiter].deadline:
                        raise RuntimeError(
                            f"Scheduled goal departure waiting timed out for {waiter}; original goals retained."
                        )
                    departure_intentions = {
                        **intentions,
                        holder: {
                            **intention,
                            "yield_origin_xy_m": intention.get("yield_origin_xy_m")
                            if intention.get("yield_origin_xy_m") is not None
                            else hold.origin,
                        },
                    }
                transaction = self._departure_candidate(
                    holder, waiter, actors, departure_intentions, obstacles, time_s, first_certified=ordinary
                )
                hold.departure_yield = transaction
                hold.path, hold.exit_path = transaction.owner_path, []
                self._paths.clear()
            if time_s > transaction.deadline:
                raise RuntimeError(
                    f"Door/goal departure-yield timed out for {holder}/{transaction.waiter}; original goals retained."
                )
            if self._context(intention) != transaction.owner_context:
                raise RuntimeError(f"Original route changed during departure-yield for {holder}.")
            waiter = transaction.waiter
            arrived = math.dist(actors[waiter]["position"], transaction.waiter_goal) <= transaction.waiter_tolerance
            if transaction.ready and arrived:
                onward = self._plan(
                    holder, intention["target"], intention["arrival_tolerance_m"], actors, self._peers(holder, actors)
                )
                if onward is not None:
                    hold.departure_yield = None
                    hold.path = onward
                    hold.exit_path = self._prefix(
                        onward,
                        self._required(actors[holder], actors[waiter]) + hold.tolerance + transaction.waiter_tolerance,
                    )
                    self._paths.clear()
                    continue
            current = np.asarray(actors[holder]["position"][:2])
            if not transaction.ready and np.linalg.norm(current - transaction.refuge) <= 0.015:
                transaction.ready = True
                self._paths.clear()
            token = ("goal_departure", holder, waiter, transaction.started)
            if holder not in priority:
                if transaction.ready:
                    objectives[holder] = {"target": None, "phase": "goal_wait", "token": (*token, "hold")}
                else:
                    if not self._clear(
                        holder,
                        current,
                        transaction.refuge,
                        actors,
                        obstacles,
                        duration=math.dist(current, transaction.refuge) / intention["speed_mps"],
                    ):
                        raise RuntimeError(f"Departure-yield segment is no longer certified for {holder}.")
                    objectives[holder] = {
                        "target": transaction.refuge.tolist(),
                        "phase": "goal_yield",
                        "token": (*token, "yield"),
                        "origin": transaction.origin.tolist(),
                        "max_displacement_m": self.max_retreat_m,
                        "certified_ray": True,
                    }
            if not transaction.ready and waiter not in priority:
                objectives[waiter] = {"target": None, "phase": "goal_wait", "token": (*token, "outgoing_wait")}
        return objectives

    def update(
        self,
        time_s: float,
        states: list[dict],
        intentions: dict[str, dict],
        obstacles: Sequence = (),
        priority_objectives: dict[str, dict] | None = None,
    ) -> dict[str, dict]:
        """Return temporary objectives [m] and observed occupancy/queue status.

        Intentions extend the solver's current target/context/path records with
        ``phase``, ``next_target`` [m] (None at the end), and optional static
        ``next_path`` [m]. A finite ``scheduled_departure_s`` [s] permits a safe,
        bounded queue until the original turning and dwell finish. A departure
        reservation survives a route-index change.
        Door safety and existing priority objectives always retain ownership.

        Args:
            time_s: Current simulation time [s].
            states: Measured ROS poses [m, rad], velocities [m/s] and body radii [m].
            intentions: Original goal [m], speed [m/s], tolerance [m], route context,
                phase, next original target [m] and optional cached next path [m].
            obstacles: Observed dynamic bodies and footprints [m, m/s].
            priority_objectives: Already-owned door or passing directives.

        Returns:
            Temporary planar guidance [m], waiting/yield phases and occupancy
            diagnostics. Actual original arrival remains the route owner's decision.
        """
        actors = {state["name"]: state for state in states}
        priority = priority_objectives or {}
        self._observe_planning_geometry(obstacles)
        self._refresh(time_s, actors, intentions, priority)
        self._release_waits(time_s, actors, intentions)
        objectives = self._departure_cycles(time_s, actors, intentions, obstacles, priority)
        for holder in sorted({wait.holder for wait in self._waits.values() if not self._active_holder(wait)}):
            queue = sorted(
                (name for name, wait in self._waits.items() if wait.holder == holder and not self._active_holder(wait)),
                key=lambda n: (self._waits[n].started, n),
            )
            for name in queue[1:]:
                if name not in priority:
                    objectives[name] = {"target": None, "phase": "goal_wait", "token": ("goal_fifo", holder, name)}
            if queue:
                head = queue[0]
                if head in priority:
                    self._waits[head].admission_blocked_since = None
                    continue
                directive = self._admission_objective(head, self._waits[head], actors, intentions, obstacles, time_s)
                if directive is not None:
                    objectives[head] = directive
        for name in sorted(actors):
            intention = intentions[name]
            if name in priority or name in objectives or not intention.get("active") or intention.get("target") is None:
                continue
            if math.dist(actors[name]["position"], intention["target"]) <= intention["arrival_tolerance_m"]:
                continue
            if name in self._waits:
                wait = self._waits[name]
                if not self._active_holder(wait):
                    continue
                hold = self._holders[wait.holder]
                if hold.departing and hold.exit_path and wait.parallel_admitted and hold.departure_yield is None:
                    objectives[name] = {
                        "target": None,
                        "phase": "goal_wait",
                        "token": ("goal_admit_wait", wait.holder, name),
                    }
                    continue
                path = self._arrival_path(name, actors, intentions)
                if (
                    path is not None
                    and not intention.get("queued_door_entry")
                    and (
                        not self._holders[wait.holder].departing
                        or (
                            self._holders[wait.holder].departure_yield is not None
                            and self._holders[wait.holder].departure_yield.ready
                        )
                    )
                    and self._exit(wait.holder, name, path[-1], actors, intentions) is not None
                ):
                    wait.admitted = True
                    directive = self._approach_objective(
                        name,
                        path,
                        actors,
                        intentions,
                        obstacles,
                        ("goal_available", wait.holder, name, wait.started),
                        wait.holder,
                    )
                    if directive is not None:
                        objectives[name] = directive
                    continue
                objectives[name] = self._waiting_objective(name, wait, actors, intentions, obstacles, time_s)
                continue
            for holder, hold in self._holders.items():
                if holder == name:
                    continue
                required = self._required(actors[name], actors[holder])
                near = (
                    np.linalg.norm(np.asarray(intention["target"][:2]) - hold.origin)
                    <= required + hold.tolerance + intention["arrival_tolerance_m"]
                )
                local_range = (
                    required
                    + hold.tolerance
                    + intention["arrival_tolerance_m"]
                    + self.max_retreat_m
                    + intention["speed_mps"] * self.relaxation
                )
                if not near or np.linalg.norm(np.asarray(actors[name]["position"][:2]) - hold.origin) > local_range:
                    continue
                path = self._arrival_path(name, actors, intentions)
                compatible = path is not None and self._exit(holder, name, path[-1], actors, intentions) is not None
                if compatible and not hold.departing and not intention.get("queued_door_entry"):
                    if self._blocked_guidance(name, actors, intention):
                        directive = self._approach_objective(
                            name,
                            path,
                            actors,
                            intentions,
                            obstacles,
                            ("goal_approach", holder, name, hold.started),
                            holder,
                        )
                        if directive is not None:
                            objectives[name] = directive
                    continue
                wait = _Wait(
                    holder,
                    self._context(intention),
                    np.asarray(intention["target"]),
                    np.asarray(actors[name]["position"][:2]),
                    time_s,
                    holder_context=hold.context,
                )
                self._waits[name] = wait
                objectives[name] = self._waiting_objective(
                    name, wait, actors, intentions, obstacles, time_s, new_wait=True
                )
                break
        for holder, hold in self._holders.items():
            if not hold.departing or holder in priority or holder in objectives or not hold.exit_path:
                continue
            if not intentions[holder].get("active"):
                continue
            speed = intentions[holder]["speed_mps"]
            point = self._egress_guide(holder, hold, actors, obstacles, speed, time_s)
            departure_route = []
            if point is None:
                departure_route = self._cached_departure_route(hold, intentions[holder], actors[holder]["position"])
                if departure_route:
                    point = self._guide(holder, departure_route.copy(), actors, obstacles, speed)
            if point is not None:
                objectives[holder] = {
                    "target": point,
                    "phase": "goal_egress",
                    "token": ("goal_exit", holder, hold.started),
                    "certified_ray": not departure_route
                    and self._parallel_step_clear(holder, point, actors, obstacles, speed, lock_egress_heading=True),
                }
                self._parallel_admission(
                    holder,
                    hold,
                    point,
                    actors,
                    intentions,
                    obstacles,
                    time_s,
                    objectives,
                    departure_route=departure_route,
                )
            else:
                self._resume_interrupted_prefix(holder, hold, actors, intentions, obstacles, priority, objectives)
        status = {
            "holders": [
                {
                    "name": name,
                    "phase": "departing" if hold.departing else "held",
                    "target_m": hold.target.tolist(),
                    "safety_phase": hold.safety_phase,
                    "origin_xy_m": hold.origin.tolist(),
                    "next_target_m": None if hold.next_target is None else hold.next_target.tolist(),
                    "departure_yield": None
                    if hold.departure_yield is None
                    else {
                        "waiter": hold.departure_yield.waiter,
                        "ready": hold.departure_yield.ready,
                        "origin_xy_m": hold.departure_yield.origin.tolist(),
                        "refuge_xy_m": hold.departure_yield.refuge.tolist(),
                        "started_s": hold.departure_yield.started,
                        "deadline_s": hold.departure_yield.deadline,
                    },
                }
                for name, hold in self._holders.items()
            ],
            "queue": [
                {
                    "name": name,
                    "holder": wait.holder,
                    "elapsed_s": time_s - wait.started,
                    "reason": "door_admission_queue"
                    if intentions[name].get("queued_door_entry")
                    else "zero_width_goal_occupancy"
                    if self._zero_width(name, actors, intentions[name])
                    else "permanent_goal_occupancy"
                    if self._active_holder(wait) and self._holders[wait.holder].next_target is None
                    else "awaiting_holder_departure"
                    if wait.scheduled_departure_s is not None and not wait.admitted
                    else "goal_occupancy",
                    "permanent": self._active_holder(wait)
                    and self._holders[wait.holder].next_target is None
                    and not intentions[name].get("queued_door_entry"),
                    "origin_xy_m": wait.origin.tolist(),
                    "refuge_xy_m": None if wait.refuge is None else wait.refuge.tolist(),
                    "admitted": wait.admitted,
                    "scheduled_departure_s": wait.scheduled_departure_s,
                    "waiting_deadline_s": wait.deadline if wait.scheduled_departure_s is not None else None,
                    "admission_blocked_s": None
                    if wait.admission_blocked_since is None
                    else time_s - wait.admission_blocked_since,
                }
                for name, wait in sorted(self._waits.items(), key=lambda item: (item[1].started, item[0]))
            ],
        }
        return {"objectives": objectives, "status": status}
