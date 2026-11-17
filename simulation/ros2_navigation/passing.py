# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded ROS-owned passing transactions for persistent reciprocal blockages."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .interactions import _segment_point_distance
from .planner import OccupancyGrid, _path_lookahead, _planning_obstacles, segment_polygon_clear

_RELEASE_PROGRESS_M = 0.15
_RELEASE_CLEARANCE_M = 0.05
_QUEUE_LOOKAHEAD_S = 0.9


def _forward_path(path, current):
    """Keep the remaining polyline from the actual planar position [m]."""
    if len(path) < 2:
        return list(path)
    point = np.asarray(current[:2], dtype=float)
    index = min(
        range(len(path) - 1),
        key=lambda i: (_segment_point_distance(np.asarray(path[i]), np.asarray(path[i + 1]), point), -i),
    )
    return [point.tolist(), *path[index + 1 :]]


def _approach_segments(path, distance_m):
    """Return the approach segments within the given travel distance [m]."""
    segments = []
    for start, end in zip(path, path[1:]):
        length = math.dist(start, end)
        if length == 0:
            continue
        a, b = np.asarray(start), np.asarray(end)
        if length >= distance_m:
            segments.append((a, a + (distance_m / length) * (b - a)))
            break
        segments.append((a, b))
        distance_m -= length
    return segments


def _segment_distance(a, b, c, d):
    first, second = np.asarray(b) - a, np.asarray(d) - c
    cross = float(first[0] * second[1] - first[1] * second[0])
    if abs(cross) > 1e-14:
        relative = np.asarray(c) - a
        t = float(relative[0] * second[1] - relative[1] * second[0]) / cross
        u = float(relative[0] * first[1] - relative[1] * first[0]) / cross
        if 0 <= t <= 1 and 0 <= u <= 1:
            return 0.0
    return min(
        *(_segment_point_distance(np.asarray(a), np.asarray(b), np.asarray(point)) for point in (c, d)),
        *(_segment_point_distance(np.asarray(c), np.asarray(d), np.asarray(point)) for point in (a, b)),
    )


def _polygon_clear(start, end, footprint, clearance):
    if np.any(np.max(footprint, axis=0) < np.minimum(start, end) - clearance) or np.any(
        np.min(footprint, axis=0) > np.maximum(start, end) + clearance
    ):
        return True
    return segment_polygon_clear(start, end, footprint, clearance)


def _clear(grid, start, end, radius, margin, peers, obstacles, duration=0.0):
    if not grid.segment_clear(start, end, radius + margin):
        return False
    for position, required in peers:
        if _segment_point_distance(np.asarray(start), np.asarray(end), position) < required - 1e-7:
            return False
    for obstacle in obstacles:
        relative_end = np.asarray(end) - obstacle.velocity * duration
        if obstacle.footprint is not None:
            if not _polygon_clear(start, relative_end, obstacle.footprint, radius + margin):
                return False
        elif _segment_point_distance(np.asarray(start), relative_end, obstacle.position) < radius + obstacle.radius:
            return False
    return True


class _NoPassingPath(ValueError):
    """Current geometry cannot certify a path to the unchanged original goal."""


class _NoPassingRefuge(RuntimeError):
    """Neither role has a complete bounded passing certificate."""


class _OverlayGrid(OccupancyGrid):
    """Query the original map with measured bodies as additional held obstacles."""

    def __init__(self, base, radius, margin, peers, obstacles):
        super().__init__(
            base.occupancy,
            base.resolution_m,
            base.origin_xy_m,
            goal_projection_tolerance_m=base.goal_projection_tolerance_m,
        )
        self.base, self.radius, self.margin = base, radius, margin
        self.peers, self.obstacles = peers, obstacles

    def segment_clear(self, start, end, clearance_m):
        return _clear(self.base, start, end, self.radius, self.margin, self.peers, self.obstacles)

    def _goal_cache_base(self, goal, tolerance, clearance):
        if type(self) is not _OverlayGrid:
            return None
        return self._overlay_goal_cache_base(goal, tolerance, clearance)

    def _overlay_goal_cache_base(self, goal, tolerance, clearance):
        if clearance != self.radius + self.margin:
            return None
        # Anchors and all endpoint bisections stay in this goal neighborhood.
        inset = self.resolution_m / math.sqrt(2) + 1e-6
        neighborhood = tolerance + self.resolution_m * math.sqrt(2) + inset + 1e-8
        point = np.asarray(goal[:2], dtype=float)
        for position, required in self.peers:
            if np.linalg.norm(point - position) <= neighborhood + required:
                return None
        for obstacle in self.obstacles:
            padding = neighborhood + self.radius + self.margin
            if obstacle.footprint is None:
                if np.linalg.norm(point - obstacle.position) <= padding + obstacle.radius:
                    return None
            elif not segment_polygon_clear(point, point, obstacle.footprint, padding):
                return None
        return self.base._goal_cache_base(goal, tolerance, clearance)

    def _goal_endpoints(self, goal, tolerance, mask, clearance):
        base = self._goal_cache_base(goal, tolerance, clearance)
        if base is not None:
            return base._goal_endpoints(goal, tolerance, base.free_mask(clearance), clearance)
        return super()._goal_endpoints(goal, tolerance, mask, clearance)

    def free_mask(self, clearance_m):
        mask = self.base.free_mask(clearance_m).copy()
        rows, cols = np.indices(mask.shape)
        points = np.stack(
            (
                self.origin_xy_m[0] + (cols + 0.5) * self.resolution_m,
                self.origin_xy_m[1] + (rows + 0.5) * self.resolution_m,
            ),
            axis=-1,
        )
        # Extra subcell clearance makes adjacent center edges safe for the
        # non-raster overlays too. Every returned segment is checked exactly.
        inset = self.resolution_m / math.sqrt(2) + 1e-6
        for position, required in self.peers:
            mask &= np.linalg.norm(points - position, axis=-1) > required + inset
        for obstacle in self.obstacles:
            if obstacle.footprint is None:
                mask &= np.linalg.norm(points - obstacle.position, axis=-1) > self.radius + obstacle.radius + inset
                continue
            polygon = obstacle.footprint
            reach = self.radius + self.margin + inset
            r0, c0 = self.world_to_cell(np.min(polygon, axis=0) - reach)
            r1, c1 = self.world_to_cell(np.max(polygon, axis=0) + reach)
            r0, c0 = max(0, r0), max(0, c0)
            r1, c1 = min(self.height - 1, r1), min(self.width - 1, c1)
            if r0 > r1 or c0 > c1:
                continue
            samples = points[r0 : r1 + 1, c0 : c1 + 1].reshape(-1, 2)
            edges = np.roll(polygon, -1, axis=0) - polygon
            relative = samples[:, None, :] - polygon
            fraction = np.clip(np.sum(relative * edges, axis=2) / np.sum(edges * edges, axis=1), 0, 1)
            distances = np.linalg.norm(relative - fraction[:, :, None] * edges, axis=2)
            outside = np.any(edges[:, 0] * relative[:, :, 1] - edges[:, 1] * relative[:, :, 0] < 0, axis=1)
            allowed = outside & (np.min(distances, axis=1) > reach)
            mask[r0 : r1 + 1, c0 : c1 + 1] &= allowed.reshape(r1 - r0 + 1, c1 - c0 + 1)
        return mask


@dataclass
class _Transaction:
    yielder: str
    passer: str
    origin: np.ndarray
    refuge: np.ndarray
    contexts: tuple
    started: float
    deadline: float
    path: list
    axis: np.ndarray
    phase: str = "retreat"
    suspended: bool = False
    readiness_error: str | None = None
    readiness_retry_s: float | None = None
    readiness_geometry: tuple | None = None
    approach_during_retreat: bool = False
    actual_hold_retry_s: float | None = None
    local_exchange: bool = False


@dataclass
class _PassageWait:
    transaction: tuple[str, str]
    started: float
    context: tuple
    origin: np.ndarray


@dataclass
class _PendingPassage:
    contexts: tuple
    started: float
    deadline: float
    retry_s: float
    outsiders: tuple[str, ...]


@dataclass
class _EarlyAttempt:
    context: tuple
    failures: int = 0
    failed_at_s: float = 0.0
    positions: tuple[tuple[float, float], ...] = ()


class PassingCoordinator:
    """Temporarily yield within the configured distance [m] while retaining original route goals.

    Warm, slow conflicts with observed motion may begin a fully certified
    refuge [m] before reaching the persistent close-range fallback. The other
    pedestrian waits, then follows a certified path around the held body.
    Failed optional certificates leave ordinary navigation in control.
    Warm time follows continuous active navigation across waypoint changes.
    Distant certification also requires visible centers inside the RVO range.
    Actual measured order exchange ends the transaction.
    Third-party approaches into the reserved passage wait at certified
    nonblocking poses until that exchange. Door and goal controllers retain priority.
    """

    def __init__(self, grid: OccupancyGrid, settings: dict):
        self.grid = grid
        self.margin = float(settings.get("map_clearance_m", 0.05))
        self.path_preference = float(settings.get("preferred_path_clearance_m", 0.0))
        self.pair_clearance = float(settings.get("pair_clearance_m", 0.25))
        self.horizon = float(settings.get("time_horizon_s", 2.5))
        self.neighbor_distance = float(settings.get("neighbor_distance_m", 4.0))
        self.relaxation = float(settings.get("relaxation_time_s", 0.4))
        self.turn_rate = float(settings.get("max_turn_rate_radps", 2.5))
        self._door_specs = settings.get("door_specs", [])
        self.transactions: dict[tuple[str, str], _Transaction] = {}
        self._eligible = {}
        self._blocked = {}
        self._completed = {}
        self._queued = {}
        self._pending: dict[tuple[str, str], _PendingPassage] = {}
        self._early_attempted: dict[tuple[str, str], _EarlyAttempt] = {}
        self._far_early_attempted: dict[tuple[str, str], _EarlyAttempt] = {}
        self._far_pressure_since: dict[tuple[str, str], float] = {}
        self._crossing_since: dict[tuple[str, str], float] = {}
        self._control_dt = 1.0 / float(settings.get("control_hz", 60))
        self.max_retreat_m = float(settings.get("max_yield_distance_m", 0.4))

    @property
    def waiting_actors(self) -> tuple[str, ...]:
        """Actors whose current routes wait behind an owned passing path."""
        return tuple(sorted(set(self._queued).union(name for key in self._pending for name in key)))

    def reset(self):
        """Forget transient interactions at a new simulation timeline."""
        self.transactions.clear()
        self._eligible.clear()
        self._blocked.clear()
        self._completed.clear()
        self._queued.clear()
        self._pending.clear()
        self._early_attempted.clear()
        self._far_early_attempted.clear()
        self._far_pressure_since.clear()
        self._crossing_since.clear()

    @staticmethod
    def _context(intention):
        return tuple(intention.get("route_context", ())), tuple(intention.get("target") or ())

    @staticmethod
    def _arrived(state, intention):
        target = intention.get("target")
        return target is not None and math.dist(state["position"], target) <= intention.get("arrival_tolerance_m", 0.0)

    def _peers(self, name, state, actors, replacement=None, *, holding_peer=None):
        return [
            (
                np.asarray(replacement.get(other, body["position"][:2]) if replacement else body["position"][:2]),
                float(state["radius_m"])
                + float(body["radius_m"])
                + self.pair_clearance
                + (_RELEASE_CLEARANCE_M if other == holding_peer else 0.0),
            )
            for other, body in actors.items()
            if other != name
        ]

    def _guidance(self, state, intention, obstacles=()):
        planning = _planning_obstacles(obstacles, self._door_specs)

        def clear(start, end):
            return _clear(
                self.grid,
                np.asarray(start[:2]),
                np.asarray(end[:2]),
                state["radius_m"] + intention.get("path_padding_m", 0.0),
                self.margin,
                [],
                planning,
            )

        path = _forward_path(intention.get("path", []), state["position"])
        for index in range(len(path) - 1, 0, -1):
            point = path[index]
            if tuple(point[:2]) != tuple(state["position"][:2]) and clear(state["position"], point):
                del path[1:index]
                point = _path_lookahead(
                    path, state["position"], intention["speed_mps"] * max(self.relaxation, 1 / self.turn_rate), clear
                )
                return np.asarray(point) - state["position"][:2]
        return np.asarray(intention["target"][:2]) - state["position"][:2]

    def _intended_path(self, state, intention):
        if abs(state["position"][2] - intention["target"][2]) >= intention["arrival_tolerance_m"]:
            return []
        path = intention.get("path") or []
        if len(path) >= 2:
            path = _forward_path(path, state["position"])
        elif self.grid.segment_clear(state["position"], intention["target"], state["radius_m"] + self.margin):
            path = [list(state["position"][:2]), list(intention["target"][:2])]
        else:
            return []
        if all(self.grid.segment_clear(a, b, state["radius_m"] + self.margin) for a, b in zip(path, path[1:])):
            return path
        return []

    def _reserved_segments(self, transaction, actors):
        """Protect the passing path until a conservative predicted exchange [m]."""
        held, passer = actors[transaction.yielder], actors[transaction.passer]
        path = _forward_path(transaction.path, passer["position"])
        segments = [(np.asarray(a), np.asarray(b)) for a, b in zip(path, path[1:]) if tuple(a) != tuple(b)]
        if transaction.suspended:
            return segments

        anchor = transaction.refuge if transaction.phase == "retreat" else np.asarray(held["position"][:2])
        # Keep a short reserve past the measured exchange to allow chord tracking error.
        progress = _RELEASE_PROGRESS_M + 0.10
        distance = held["radius_m"] + passer["radius_m"] + self.pair_clearance + _RELEASE_CLEARANCE_M + 0.05

        def beyond(point):
            relative = point - anchor
            return np.dot(relative, transaction.axis) >= progress and np.dot(relative, relative) >= distance**2

        for index, (a, b) in enumerate(segments):
            if not beyond(b):
                continue
            if not beyond(a):
                lower, upper = 0.0, 1.0
                for _ in range(12):
                    middle = (lower + upper) / 2
                    if beyond(a + middle * (b - a)):
                        upper = middle
                    else:
                        lower = middle
                segments[index] = (a, a + upper * (b - a))
            return segments[: index + 1]
        return segments

    def _queue_outsiders(self, actors, intentions, obstacles, priority, objectives):
        if not self.transactions:
            self._queued.clear()
            return set()
        owned = {
            name for transaction in self.transactions.values() for name in (transaction.yielder, transaction.passer)
        }
        passages = [
            (key, transaction, self._reserved_segments(transaction, actors))
            for key, transaction in sorted(self.transactions.items(), key=lambda item: item[1].started)
        ]
        queued = {}
        for name in sorted(actors):
            intention, state = intentions[name], actors[name]
            if (
                name in owned
                or name in priority
                or not intention.get("active")
                or intention.get("target") is None
                or self._arrived(state, intention)
                or not _clear(
                    self.grid,
                    np.asarray(state["position"][:2]),
                    np.asarray(state["position"][:2]),
                    state["radius_m"],
                    self.margin,
                    self._peers(name, state, actors),
                    obstacles,
                    self._control_dt,
                )
            ):
                continue
            route = self._intended_path(state, intention)
            approach = _approach_segments(route, intention["speed_mps"] * _QUEUE_LOOKAHEAD_S)
            old = self._queued.get(name)
            for key, transaction, segments in passages:
                passer = actors[transaction.passer]
                required = state["radius_m"] + passer["radius_m"] + self.pair_clearance
                if not segments or any(
                    _segment_point_distance(a, b, np.asarray(state["position"][:2])) < required for a, b in segments
                ):
                    continue
                retained = (
                    old is not None
                    and old.transaction == key
                    and old.started == transaction.started
                    and old.context == self._context(intention)
                )
                if not retained and not any(
                    _segment_distance(a, b, c, d) < required for a, b in approach for c, d in segments
                ):
                    continue
                queued[name] = (
                    old
                    if retained
                    else _PassageWait(
                        key, transaction.started, self._context(intention), np.asarray(state["position"][:2]).copy()
                    )
                )
                objectives[name] = {
                    "target": None,
                    "phase": "passing_wait",
                    "token": ("passing_queue", transaction.started, *key, name),
                }
                break
        self._queued = queued
        return set(queued)

    def update(self, time_s, states, intentions, obstacles, door_objectives):
        """Return temporary objectives and measured transaction metadata [m, s]."""
        actors = {state["name"]: state for state in states}
        self._refresh_early(actors, intentions)
        available = {
            name
            for name, intention in intentions.items()
            if intention.get("active")
            and intention.get("target") is not None
            and not self._arrived(actors[name], intention)
            and name not in door_objectives
        }
        self._refresh_eligible(time_s, actors, intentions, available)
        objectives = {}
        for key, transaction in list(self.transactions.items()):
            participants = (transaction.yielder, transaction.passer)
            if (
                any(not intentions[name].get("active") for name in participants)
                or tuple(self._context(intentions[name]) for name in participants) != transaction.contexts
            ):
                del self.transactions[key]
                continue
            if time_s > transaction.deadline:
                raise RuntimeError(
                    f"A bounded passing transaction timed out for {participants}; original goals retained."
                )
            transaction.suspended = any(name in door_objectives for name in participants)
            if transaction.suspended:
                for name in participants:
                    if name not in door_objectives:
                        objectives[name] = {
                            "target": None,
                            "phase": "passing_wait",
                            "token": ("passing", transaction.started, transaction.yielder, "door_priority"),
                        }
                continue
            held, moving = actors[transaction.yielder], actors[transaction.passer]
            hpos, mpos = np.asarray(held["position"][:2]), np.asarray(moving["position"][:2])
            required = held["radius_m"] + moving["radius_m"] + self.pair_clearance
            if (
                self._arrived(moving, intentions[transaction.passer])
                and np.linalg.norm(mpos - hpos) >= required - 1e-7
                or (
                    transaction.phase == "passing"
                    and np.dot(mpos - hpos, transaction.axis) > _RELEASE_PROGRESS_M
                    and np.linalg.norm(mpos - hpos) > required + _RELEASE_CLEARANCE_M
                )
            ):
                del self.transactions[key]
                self._blocked.pop(key, None)
                self._completed[key] = tuple(self._context(intentions[n]) for n in key)
                continue
            if transaction.phase == "retreat":
                self._update_retreat(transaction, key, hpos, actors, intentions, obstacles, time_s)
            if not self._objectives(transaction, actors, intentions, obstacles, objectives):
                del self.transactions[key]
                self._blocked.pop(key, None)
                available.difference_update(participants)
        busy = {
            name for transaction in self.transactions.values() for name in (transaction.yielder, transaction.passer)
        }
        queued = self._queue_outsiders(actors, intentions, obstacles, door_objectives, objectives)
        if self._update_pending(time_s, actors, intentions, obstacles, available - busy - queued, objectives):
            queued = self._queue_outsiders(actors, intentions, obstacles, door_objectives, objectives)
        busy.update(name for key in self._pending for name in key)
        busy.update(
            name for transaction in self.transactions.values() for name in (transaction.yielder, transaction.passer)
        )
        names = sorted(available - busy - queued)
        # Carry forward only pairs that remain eligible and visible this update.
        previous_far_pressure = self._far_pressure_since
        self._far_pressure_since = {}
        previous_crossing = self._crossing_since
        self._crossing_since = {}
        began = False
        early_available = True
        for i, first in enumerate(names):
            if began:
                break
            for second in names[i + 1 :]:
                if first in busy or second in busy:
                    continue
                key = (first, second)
                a, b = actors[first], actors[second]
                separation = np.linalg.norm(np.asarray(a["position"][:2]) - b["position"][:2])
                required = a["radius_m"] + b["radius_m"] + self.pair_clearance
                context = tuple(self._context(intentions[n]) for n in key)
                if self._completed.get(key) == context and separation < required + 0.5:
                    continue
                self._completed.pop(key, None)
                directions = [self._guidance(actors[n], intentions[n], obstacles) for n in key]
                lengths = [float(np.linalg.norm(d)) for d in directions]
                alignment = np.dot(directions[0], directions[1]) / math.prod(lengths) if min(lengths) > 1e-6 else 1.0
                opposed = alignment < -0.3
                close = separation < required + 0.08
                speeds = [np.linalg.norm(actors[n].get("velocity", [0, 0])[:2]) for n in key]
                crossing = (
                    close
                    and -0.3 <= alignment < 0.9
                    and all(speed < min(0.03, intentions[n]["speed_mps"] * 0.25) for speed, n in zip(speeds, key))
                )
                conflicting = False
                if opposed or crossing:
                    relative = np.asarray(b["position"][:2]) - a["position"][:2]
                    closing_velocity = (
                        directions[0] * intentions[first]["speed_mps"] / lengths[0]
                        - directions[1] * intentions[second]["speed_mps"] / lengths[1]
                    )
                    closest_time = float(
                        np.clip(
                            np.dot(relative, closing_velocity) / np.dot(closing_velocity, closing_velocity),
                            0,
                            self.horizon,
                        )
                    )
                    conflicting = (
                        closest_time > 0 and np.linalg.norm(relative - closing_velocity * closest_time) < required
                    )
                slow = any(speed < min(0.12, intentions[n]["speed_mps"] * 0.25) for speed, n in zip(speeds, key))
                warmed = all(time_s - self._eligible[n][1] >= 1.5 for n in key)
                if not (conflicting and slow and warmed):
                    self._blocked.pop(key, None)
                    transaction, attempted = self._try_braking_hold(
                        key,
                        context,
                        time_s,
                        actors,
                        intentions,
                        obstacles,
                        speeds,
                        separation,
                        required,
                        opposed and conflicting and warmed and early_available,
                    )
                    early_available = early_available and not attempted
                    if transaction is None:
                        continue
                elif crossing:
                    self._blocked.pop(key, None)
                    started = previous_crossing.get(key, time_s)
                    self._crossing_since[key] = started
                    if time_s < started + 0.75:
                        continue
                    try:
                        transaction = self._choose(key, time_s, actors, intentions, obstacles)
                    except _NoPassingRefuge:
                        # An optional merge recovery never stops ordinary navigation.
                        self._crossing_since[key] = time_s
                        continue
                elif close:
                    started = self._blocked.setdefault(key, time_s)
                    if time_s - started < 0.75:
                        continue
                    try:
                        transaction = self._choose(key, time_s, actors, intentions, obstacles)
                    except _NoPassingRefuge:
                        if not self._defer_certificate(key, time_s, actors, intentions, obstacles, door_objectives):
                            raise
                        self._pending_objectives(key, objectives)
                        began = True
                        break
                else:
                    self._blocked.pop(key, None)
                    transaction, attempted = self._early_transaction(
                        key,
                        context,
                        time_s,
                        actors,
                        intentions,
                        obstacles,
                        separation,
                        required,
                        previous_far_pressure,
                        early_available,
                    )
                    if attempted:
                        early_available = False
                    if transaction is None:
                        continue
                self.transactions[key] = transaction
                busy.update(key)
                self._objectives(transaction, actors, intentions, obstacles, objectives)
                self._queue_outsiders(actors, intentions, obstacles, door_objectives, objectives)
                began = True
                break
        metadata = {
            "transactions": [
                {
                    "yielder": t.yielder,
                    "passer": t.passer,
                    "phase": "door_priority" if t.suspended else t.phase,
                    "origin_xy_m": t.origin.tolist(),
                    "refuge_xy_m": t.refuge.tolist(),
                    "elapsed_s": time_s - t.started,
                    "deadline_s": t.deadline,
                    "readiness_error": t.readiness_error,
                }
                for t in self.transactions.values()
            ]
        }
        if self._queued:
            metadata["queued"] = [
                {
                    "name": name,
                    "yielder": self.transactions[wait.transaction].yielder,
                    "passer": self.transactions[wait.transaction].passer,
                    "origin_xy_m": wait.origin.tolist(),
                    "deadline_s": self.transactions[wait.transaction].deadline,
                }
                for name, wait in sorted(self._queued.items())
            ]
        if self._pending:
            metadata["pending"] = [
                {
                    "actors": list(key),
                    "elapsed_s": time_s - wait.started,
                    "deadline_s": wait.deadline,
                    "outside_actors": list(wait.outsiders),
                }
                for key, wait in sorted(self._pending.items())
            ]
        return {"objectives": objectives, "status": metadata}

    def _refresh_eligible(self, time_s, actors, intentions, available):
        changed = set()
        for name in actors:
            intention = intentions[name]
            context = self._context(intention)
            # Arrival and temporary ownership gate admission, not continuity
            # of an otherwise active navigation intent.
            if not intention.get("active") or intention.get("target") is None:
                self._eligible.pop(name, None)
                changed.add(name)
            elif name not in self._eligible:
                self._eligible[name] = (context, time_s)
                changed.add(name)
            elif self._eligible[name][0] != context:
                self._eligible[name] = (context, self._eligible[name][1])
                changed.add(name)
            if name in self._queued and name in available:
                # Queue-imposed stillness is not evidence of a new encounter.
                self._eligible[name] = (context, time_s)
        for timers in (self._blocked, self._far_pressure_since, self._crossing_since):
            for key in list(timers):
                if any(name not in available or name in changed for name in key):
                    del timers[key]

    def _refresh_early(self, actors, intentions):
        for cache in (self._early_attempted, self._far_early_attempted):
            for key, attempt in list(cache.items()):
                if (
                    any(name not in actors or name not in intentions for name in key)
                    or tuple(self._context(intentions[name]) for name in key) != attempt.context
                ):
                    del cache[key]
                    continue
                first, second = (actors[name] for name in key)
                required = first["radius_m"] + second["radius_m"] + self.pair_clearance
                if np.linalg.norm(np.asarray(first["position"][:2]) - second["position"][:2]) > max(
                    required + 2 * self.max_retreat_m, self.neighbor_distance
                ):
                    del cache[key]

    def _observed_motion(self, key, actors, intentions):
        for name in key:
            velocity = self._known_velocity(actors[name])
            if (
                velocity is None
                or np.linalg.norm(velocity[:2]) <= 8 * np.finfo(np.float32).eps * intentions[name]["speed_mps"]
            ):
                continue
            position = np.asarray(actors[name]["position"][:2])
            if np.any(
                np.asarray(position + velocity[:2] * self._control_dt, dtype=np.float32)
                != np.asarray(position, dtype=np.float32)
            ):
                return True
        return False

    def _try_braking_hold(
        self, key, context, time_s, actors, intentions, obstacles, speeds, separation, required, ready
    ):
        if not ready:
            return None, False
        ratios = [speed / intentions[name]["speed_mps"] for name, speed in zip(key, speeds)]
        if max(ratios) >= 0.68 or min(ratios) >= 0.38:
            return None, False
        # Two walkers already braking can use an open route around
        # the slower one's actual pose without ordering a retreat.
        return self._early_transaction(
            key,
            context,
            time_s,
            actors,
            intentions,
            obstacles,
            separation,
            required,
            {},
            True,
            holding_actor=key[int(np.argmin(ratios))],
        )

    def _early_transaction(
        self,
        key,
        context,
        time_s,
        actors,
        intentions,
        obstacles,
        separation,
        required,
        previous_far_pressure,
        early_available,
        *,
        holding_actor=None,
    ):
        if holding_actor is not None and (
            separation > required + 2 * self.max_retreat_m or not self._stationary_safe(key, actors, obstacles)
        ):
            return None, False
        if separation <= required + 2 * self.max_retreat_m:
            cache = self._early_attempted
        elif separation <= self.neighbor_distance and _clear(
            self.grid,
            np.asarray(actors[key[0]]["position"][:2]),
            np.asarray(actors[key[1]]["position"][:2]),
            0.0,
            0.0,
            [],
            obstacles,
        ):
            cache = self._far_early_attempted
            started = previous_far_pressure.get(key, time_s)
            self._far_pressure_since[key] = started
            # A short slowdown at a distance can resolve through ordinary ORCA.
            if time_s < started + self.relaxation:
                return None, False
        else:
            return None, False
        if not early_available or not self._observed_motion(key, actors, intentions):
            return None, False
        positions = tuple(tuple(actors[name]["position"][:2]) for name in key)
        attempt = cache.get(key)
        # An earlier failed route may open up as the bodies approach a wider section.
        if attempt is not None and (
            not 0 < attempt.failures < 3
            or time_s - attempt.failed_at_s < 0.4 - 1e-9
            or all(math.dist(a, b) < 0.05 for a, b in zip(attempt.positions, positions))
        ):
            return None, False
        try:
            transaction = (
                self._choose(key, time_s, actors, intentions, obstacles)
                if holding_actor is None
                else self._hold_for_open_route(holding_actor, key, time_s, actors, intentions, obstacles, required)
            )
        except _NoPassingRefuge:
            # A failed optional certificate leaves ordinary navigation and its
            # original close-range persistence timer in control.
            failures = 1 if attempt is None else attempt.failures + 1
            cache[key] = _EarlyAttempt(context, failures, time_s, positions)
            return None, True
        cache[key] = _EarlyAttempt(context)
        return transaction, True

    def _hold_for_open_route(self, holder, key, time_s, actors, intentions, obstacles, required):
        """Certify passage around a safely stopped body's measured position [m]."""
        passer = key[1] if key[0] == holder else key[0]
        origin = np.asarray(actors[holder]["position"][:2])
        try:
            path = self._plan(passer, actors[passer], intentions, actors, obstacles, holding_peer=holder)
        except _NoPassingPath as error:
            raise _NoPassingRefuge(str(error)) from error
        axis = self._holding_axis(path, origin, required)
        if axis is None:
            raise _NoPassingRefuge("No complete order exchange around the measured holding position.")
        transaction = _Transaction(
            holder,
            passer,
            origin.copy(),
            origin.copy(),
            tuple(self._context(intentions[name]) for name in (holder, passer)),
            time_s,
            time_s,
            path,
            axis,
        )
        self._start_passing(transaction, path, axis, intentions[passer]["speed_mps"], time_s)
        return transaction

    def _stationary_safe(self, key, actors, obstacles):
        for name in key:
            state = actors[name]
            position = np.asarray(state["position"][:2])
            if not _clear(
                self.grid,
                position,
                position,
                state["radius_m"],
                self.margin,
                self._peers(name, state, actors),
                obstacles,
                self._control_dt,
            ):
                return False
            for other, peer in actors.items():
                if other in key:
                    continue
                velocity = self._known_velocity(peer)
                if velocity is None:
                    return False
                relative_end = position - velocity[:2] * self._control_dt
                required = state["radius_m"] + peer["radius_m"] + self.pair_clearance
                if _segment_point_distance(position, relative_end, np.asarray(peer["position"][:2])) < required - 1e-7:
                    return False
        return True

    @staticmethod
    def _known_velocity(state):
        try:
            velocity = np.asarray(state.get("velocity", []), dtype=float)
        except (TypeError, ValueError):
            return None
        return velocity if velocity.shape == (3,) and np.isfinite(velocity).all() else None

    def _defer_certificate(self, key, time_s, actors, intentions, obstacles, priority):
        outsiders = []
        for name, state in actors.items():
            if name in key or name in priority or not intentions[name].get("active"):
                continue
            velocity = self._known_velocity(state)
            if velocity is not None and np.linalg.norm(velocity[:2]) > 0:
                outsiders.append(name)
        if not outsiders or not self._stationary_safe(key, actors, obstacles):
            return False
        reduced = {name: state for name, state in actors.items() if name not in outsiders}
        try:
            self._choose(key, time_s, reduced, intentions, obstacles)
        except _NoPassingRefuge:
            return False
        started = self._blocked[key]
        speed = min(intentions[name]["speed_mps"] for name in key)
        self._pending[key] = _PendingPassage(
            tuple(self._context(intentions[name]) for name in key),
            started,
            started + 4.0 + self.max_retreat_m / max(speed, 0.001),
            time_s + 0.5,
            tuple(outsiders),
        )
        return True

    def _pending_objectives(self, key, objectives):
        wait = self._pending[key]
        for name in key:
            objectives[name] = {
                "target": None,
                "phase": "passing_wait",
                "token": ("passing_pending", wait.started, *key),
            }

    def _update_pending(self, time_s, actors, intentions, obstacles, available, objectives):
        began = False
        for key, wait in list(self._pending.items()):
            if (
                any(name not in available for name in key)
                or tuple(self._context(intentions[name]) for name in key) != wait.contexts
            ):
                del self._pending[key]
                self._blocked.pop(key, None)
                continue
            if time_s > wait.deadline:
                raise RuntimeError(f"A bounded passing certificate wait timed out for {key}; original goals retained.")
            if not self._stationary_safe(key, actors, obstacles):
                del self._pending[key]
                self._blocked.pop(key, None)
                continue
            if time_s >= wait.retry_s or any(name not in actors for name in wait.outsiders):
                try:
                    transaction = self._choose(key, time_s, actors, intentions, obstacles)
                except _NoPassingRefuge:
                    if any(name not in actors for name in wait.outsiders):
                        del self._pending[key]
                        self._blocked.pop(key, None)
                        raise
                    wait.retry_s = time_s + 0.5
                else:
                    del self._pending[key]
                    self.transactions[key] = transaction
                    if self._objectives(transaction, actors, intentions, obstacles, objectives):
                        began = True
                    else:
                        del self.transactions[key]
                    continue
            self._pending_objectives(key, objectives)
        return began

    def _certify_ready(self, transaction, moving, intentions, actors, obstacles, time_s):
        at_refuge = np.linalg.norm(np.asarray(actors[transaction.yielder]["position"][:2]) - transaction.refuge) <= 1e-6
        geometry = (
            (
                tuple((name, tuple(state["position"]), state["radius_m"]) for name, state in sorted(actors.items())),
                tuple(
                    (
                        tuple(o.position),
                        tuple(o.velocity),
                        o.radius,
                        None if o.footprint is None else o.footprint.tobytes(),
                    )
                    for o in obstacles
                ),
                self._context(intentions[transaction.passer]),
                intentions[transaction.passer]["speed_mps"],
                intentions[transaction.passer]["arrival_tolerance_m"],
            )
            if at_refuge
            else None
        )
        if (
            at_refuge
            and transaction.readiness_error is not None
            and transaction.readiness_geometry == geometry
            and transaction.readiness_retry_s is not None
            and time_s - transaction.readiness_retry_s < 0.5 - 1e-9
        ):
            return
        transaction.readiness_retry_s, transaction.readiness_geometry = time_s, geometry
        try:
            path = self._plan(transaction.passer, moving, intentions, actors, obstacles)
        except _NoPassingPath as error:
            transaction.readiness_error = str(error)
            return
        axis = self._axis(path, np.asarray(actors[transaction.yielder]["position"][:2]))
        self._start_passing(transaction, path, axis, intentions[transaction.passer]["speed_mps"], time_s)

    def _update_retreat(self, transaction, key, position, actors, intentions, obstacles, time_s):
        near_refuge = np.linalg.norm(position - transaction.refuge) <= 0.015
        # A predictive speed limit can brake a bounded retreat toward
        # tangency. Before expiry, check the actual holding pose once.
        late_check = (
            not near_refuge
            and transaction.readiness_retry_s is None
            and time_s >= transaction.deadline - self.relaxation
            and self._stationary_safe(key, actors, obstacles)
        )
        if near_refuge or late_check:
            self._certify_ready(transaction, actors[transaction.passer], intentions, actors, obstacles, time_s)
            if late_check and transaction.phase == "passing":
                transaction.refuge = position.copy()
        else:
            self._try_hold_after_retreat(transaction, key, actors, intentions, obstacles, time_s)

    def _try_hold_after_retreat(self, transaction, key, actors, intentions, obstacles, time_s):
        """Recheck an actual refuge [m] once measurable retreat opens the route."""
        if transaction.approach_during_retreat or (
            transaction.actual_hold_retry_s is not None and time_s - transaction.actual_hold_retry_s < 0.5 - 1e-9
        ):
            return
        holder, passer = transaction.yielder, transaction.passer
        held, moving = actors[holder], actors[passer]
        position = np.asarray(held["position"][:2])
        velocity = self._known_velocity(held)
        if (
            math.dist(position, transaction.origin) < min(0.10, 0.25 * self.max_retreat_m)
            or velocity is None
            or math.hypot(*velocity[:2]) <= 0.03
            or np.dot(velocity[:2], transaction.refuge - position) <= 0.0
            or not self._stationary_safe(key, actors, obstacles)
        ):
            return
        transaction.actual_hold_retry_s = time_s
        try:
            path = self._plan(passer, moving, intentions, actors, obstacles, holding_peer=holder)
        except _NoPassingPath:
            return
        required = held["radius_m"] + moving["radius_m"] + self.pair_clearance
        axis = self._holding_axis(path, position, required)
        if axis is not None:
            transaction.refuge = position.copy()
            self._start_passing(transaction, path, axis, intentions[passer]["speed_mps"], time_s)

    @staticmethod
    def _start_passing(transaction, path, axis, speed, time_s):
        transaction.path = path
        transaction.axis = axis
        transaction.phase = "passing"
        transaction.readiness_error = None
        distance = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
        transaction.deadline = time_s + 6.0 + 3.0 * distance / max(speed, 0.001)

    def _plan(self, name, state, intentions, actors, obstacles, replacement=None, *, holding_peer=None):
        peers = self._peers(name, state, actors, replacement, holding_peer=holding_peer)
        overlay = _OverlayGrid(self.grid, state["radius_m"], self.margin, peers, obstacles)
        tolerance = float(intentions[name]["arrival_tolerance_m"])
        goal = intentions[name]["target"]
        height = abs(state["position"][2] - goal[2])
        if height >= tolerance:
            raise ValueError("The original passing goal is outside its planar arrival tolerance.")
        try:
            path = overlay.plan(
                state["position"], goal, state["radius_m"], math.sqrt(tolerance**2 - height**2), self.margin
            )
        except ValueError as error:
            raise _NoPassingPath(str(error)) from error
        elapsed = 0.0
        for a, b in zip(path, path[1:]):
            duration = math.dist(a, b) / max(intentions[name]["speed_mps"], 0.001)
            for obstacle in obstacles:
                relative_a = np.asarray(a) - obstacle.velocity * elapsed
                relative_b = np.asarray(b) - obstacle.velocity * (elapsed + duration)
                if obstacle.footprint is not None:
                    clear = _polygon_clear(relative_a, relative_b, obstacle.footprint, state["radius_m"] + self.margin)
                else:
                    clear = (
                        _segment_point_distance(relative_a, relative_b, obstacle.position)
                        >= state["radius_m"] + obstacle.radius
                    )
                if not clear:
                    raise _NoPassingPath("No bounded passing path has a complete observed-motion certificate.")
            elapsed += duration
        return path

    @staticmethod
    def _axis(path, refuge):
        segments = [(np.asarray(a), np.asarray(b)) for a, b in zip(path, path[1:]) if math.dist(a, b) > 1e-8]
        if not segments:
            raise ValueError("A passing path requires a nonzero segment to define its direction.")
        a, b = min(segments, key=lambda segment: _segment_point_distance(*segment, refuge))
        return (b - a) / np.linalg.norm(b - a)

    def _choose(self, key, time_s, actors, intentions, obstacles):
        options = []
        for yielder, passer in (key, key[::-1]):
            state = actors[yielder]
            origin = np.asarray(state["position"][:2])
            peers = self._peers(yielder, state, actors)
            stopped_peers = []
            for other, body in actors.items():
                if other in (yielder, passer):
                    continue
                velocity = self._known_velocity(body)
                if velocity is None or math.hypot(*velocity[:2]) > 0.03:
                    continue
                center = np.asarray(body["position"][:2])
                reserve = state["radius_m"] + body["radius_m"] + self.pair_clearance + _RELEASE_CLEARANCE_M
                stopped_peers.append((center, min(math.dist(center, origin) - 1e-6, reserve)))
            guide = self._guidance(actors[passer], intentions[passer], obstacles)
            unit = guide / max(np.linalg.norm(guide), 1e-9)
            own_guide = self._guidance(state, intentions[yielder], obstacles)
            own_unit = own_guide / max(np.linalg.norm(own_guide), 1e-9)
            heading = state.get("yaw", math.atan2(own_guide[1], own_guide[0]))
            for angle in np.linspace(0, 2 * math.pi, 24, endpoint=False):
                direction = np.asarray([math.cos(angle), math.sin(angle)])
                lower, upper = 0.0, self.max_retreat_m - min(0.002, 0.1 * self.max_retreat_m)
                if not _clear(
                    self.grid,
                    origin,
                    origin + upper * direction,
                    state["radius_m"],
                    self.margin,
                    peers,
                    obstacles,
                    upper / max(intentions[yielder]["speed_mps"], 0.001),
                ):
                    for _ in range(16):
                        middle = (lower + upper) / 2
                        if _clear(
                            self.grid,
                            origin,
                            origin + middle * direction,
                            state["radius_m"],
                            self.margin,
                            peers,
                            obstacles,
                            middle / max(intentions[yielder]["speed_mps"], 0.001),
                        ):
                            lower = middle
                        else:
                            upper = middle
                    upper = max(0.0, lower - 1e-5)
                if upper < min(0.05, 0.25 * self.max_retreat_m):
                    continue
                candidate = origin + upper * direction
                movement = candidate - actors[passer]["position"][:2]
                side = abs(float(unit[0] * movement[1] - unit[1] * movement[0]))
                backward = float(np.dot(candidate - origin, own_unit)) < -1e-9
                turn = abs(math.remainder(math.atan2(direction[1], direction[0]) - heading, 2 * math.pi))
                # A refuge at the edge of a stopped third person's clearance
                # can be geometrically valid but slow to reach under ORCA.
                tight = any(math.dist(center, candidate) < reserve for center, reserve in stopped_peers)
                options.append((backward, tight, turn, -side, yielder, passer, origin, candidate))
        selected = None
        for _, _, _, _, yielder, passer, origin, candidate in sorted(options, key=lambda item: item[:6]):
            try:
                path = self._plan(passer, actors[passer], intentions, actors, obstacles, {yielder: candidate})
            except ValueError:
                continue
            selected = yielder, passer, origin, candidate, path
            break
        if selected is None:
            raise _NoPassingRefuge(
                f"No certified bounded passing refuge within {self.max_retreat_m:g} m for {key}; "
                "original goals retained."
            )
        yielder, passer, origin, refuge, path = selected
        held, moving = actors[yielder], actors[passer]
        held_velocity, moving_velocity = self._known_velocity(held), self._known_velocity(moving)
        held_speed = math.inf if held_velocity is None else math.hypot(*held_velocity[:2])
        required = held["radius_m"] + moving["radius_m"] + self.pair_clearance
        holding_speed_limit = min(0.12, 0.25 * intentions[yielder]["speed_mps"])
        approach = (
            held_velocity is not None
            and moving_velocity is not None
            and held_speed >= holding_speed_limit
            and math.hypot(*moving_velocity[:2]) < min(0.03, 0.25 * intentions[passer]["speed_mps"])
            and math.dist(origin, moving["position"][:2]) > required + 0.08
        )
        transaction = _Transaction(
            yielder,
            passer,
            origin,
            refuge,
            tuple(self._context(intentions[n]) for n in (yielder, passer)),
            time_s,
            time_s + 4.0 + self.max_retreat_m / max(intentions[yielder]["speed_mps"], 0.001),
            path,
            self._axis(path, refuge),
            approach_during_retreat=approach,
        )
        axis = self._holding_axis(path, origin, required)
        hold_here = held_speed < 0.03
        if axis is None and held_velocity is not None and 0.03 <= held_speed < holding_speed_limit:
            # The slow actor is still moving toward a wall. Try holding here
            # before sending it farther along the same approach to a refuge.
            radius = held["radius_m"]
            ahead = origin + held_velocity[:2] * (2.0 * radius / held_speed)
            hold_here = not self.grid.segment_clear(origin, ahead, radius + self.margin)
        if (axis is not None or hold_here) and self._stationary_safe(key, actors, obstacles):
            if axis is None:
                # A slow walker can stop at its actual pose and yield
                # if the passer can route around it with the holding reserve.
                try:
                    path = self._plan(passer, moving, intentions, actors, obstacles, holding_peer=yielder)
                except _NoPassingPath:
                    pass
                else:
                    axis = self._holding_axis(path, origin, required)
            if axis is not None:
                transaction.refuge = origin.copy()
                self._start_passing(transaction, path, axis, intentions[passer]["speed_mps"], time_s)
        if transaction.phase == "retreat" and held_speed <= 0.03:
            self._try_local_exchange(transaction, key, actors, intentions, obstacles, time_s)
        return transaction

    def _try_local_exchange(self, transaction, key, actors, intentions, obstacles, time_s):
        """Avoid a wall-tight retreat when a short, certified order exchange is open [m]."""
        if self.path_preference <= 0.0 or len(transaction.path) < 2:
            return
        holder, passer = transaction.yielder, transaction.passer
        held, moving = actors[holder], actors[passer]
        origin = transaction.origin
        wall_clearance = held["radius_m"] + self.margin + self.path_preference
        if (
            not self.grid.segment_clear(origin, origin, wall_clearance)
            or self.grid.segment_clear(origin, transaction.refuge, wall_clearance)
            or not self._stationary_safe(key, actors, obstacles)
        ):
            return

        start = np.asarray(moving["position"][:2])
        first = np.asarray(transaction.path[1]) - start
        extent = math.hypot(*first)
        if extent <= 1e-8:
            return
        direction = first / extent
        progress = float(np.dot(origin - start, direction)) + moving["radius_m"]
        if progress <= 0.05 or progress > extent + 2.0 * moving["radius_m"]:
            return
        point = start + progress * direction
        path = [start.tolist(), point.tolist()]
        required = held["radius_m"] + moving["radius_m"] + self.pair_clearance
        axis = self._holding_axis(path, origin, required)
        speed = intentions[passer]["speed_mps"]
        if axis is not None and _clear(
            self.grid,
            start,
            point,
            moving["radius_m"],
            self.margin,
            self._peers(passer, moving, actors, holding_peer=holder),
            obstacles,
            progress / max(0.001, speed),
        ):
            transaction.refuge = origin.copy()
            transaction.local_exchange = True
            self._start_passing(transaction, path, axis, speed, time_s)

    def _holding_axis(self, path, origin, required):
        """Return the exchange direction if the path clears the measured holding pose [m]."""
        reserve = required + _RELEASE_CLEARANCE_M
        if any(_segment_point_distance(np.asarray(a), np.asarray(b), origin) < reserve for a, b in zip(path, path[1:])):
            return None
        axis = self._axis(path, origin)
        start, end = np.asarray(path[0]) - origin, np.asarray(path[-1]) - origin
        if np.dot(start, axis) < 0 and np.dot(end, axis) > _RELEASE_PROGRESS_M and np.linalg.norm(end) > reserve:
            return axis
        return None

    def _retreat_approach(self, transaction, actors, intentions, obstacles):
        """Select a passing prefix [m] clear of the whole remaining retreat."""
        held, moving = actors[transaction.yielder], actors[transaction.passer]
        start, retreat_start = np.asarray(moving["position"][:2]), np.asarray(held["position"][:2])
        required = held["radius_m"] + moving["radius_m"] + self.pair_clearance
        peers = self._peers(transaction.passer, moving, actors)
        speed = max(intentions[transaction.passer]["speed_mps"], 0.001)
        path = _forward_path(transaction.path, moving["position"])
        for candidate in reversed(path[1:]):
            point = np.asarray(candidate)
            distance = math.dist(start, point)
            if distance <= 0.05 or _segment_distance(start, point, retreat_start, transaction.refuge) < required + 1e-6:
                continue
            if _clear(self.grid, start, point, moving["radius_m"], self.margin, peers, obstacles, distance / speed):
                return point.tolist()
        return None

    def _objectives(self, transaction, actors, intentions, obstacles, result) -> bool:
        held, moving = transaction.yielder, transaction.passer
        if transaction.phase == "retreat":
            waiting = (
                transaction.readiness_error is not None
                and np.linalg.norm(np.asarray(actors[held]["position"][:2]) - transaction.refuge) <= 1e-6
            )
            result[held] = {
                "target": None if waiting else transaction.refuge.tolist(),
                "phase": "passing_wait" if waiting else "passing_yield",
                "origin": transaction.origin.tolist(),
                "max_displacement_m": self.max_retreat_m,
                "certified_ray": True,
                "token": ("passing", transaction.started, held, "path_wait" if waiting else "retreat"),
            }
            approach = (
                self._retreat_approach(transaction, actors, intentions, obstacles)
                if transaction.approach_during_retreat
                else None
            )
            result[moving] = {
                "target": approach,
                "phase": "passing_go" if approach is not None else "passing_wait",
                # The prefix clears the remaining retreat only along this chord.
                "certified_ray": approach is not None and self._stationary_safe((moving,), actors, obstacles),
                "token": ("passing", transaction.started, held, "approach" if approach is not None else "wait"),
            }
            return True
        if transaction.local_exchange and not self._stationary_safe((held,), actors, obstacles):
            return False
        state = actors[moving]
        peers = self._peers(moving, state, actors)
        transaction.path[:] = _forward_path(transaction.path, state["position"])
        visible = [
            (i, transaction.path[i])
            for i in range(len(transaction.path) - 1, 0, -1)
            if _clear(
                self.grid,
                np.asarray(state["position"][:2]),
                np.asarray(transaction.path[i]),
                state["radius_m"],
                self.margin,
                peers,
                obstacles,
                math.dist(state["position"][:2], transaction.path[i]) / max(intentions[moving]["speed_mps"], 0.001),
            )
        ]
        if not visible and transaction.local_exchange:
            if not self._stationary_safe((moving,), actors, obstacles):
                return False
            # This exchange only reserves a local chord. Keep that endpoint
            # available for revalidation instead of replanning to the goal.
            for name in (held, moving):
                result[name] = {
                    "target": None,
                    "phase": "passing_wait",
                    "token": ("passing", transaction.started, held, "local_wait"),
                }
            return True
        if not visible:
            try:
                path = self._plan(moving, state, intentions, actors, obstacles)
            except _NoPassingPath:
                # A peer can temporarily occupy the remote goal after the pass starts.
                if self._stationary_safe((held, moving), actors, obstacles):
                    return False
                raise
            transaction.path = path
            point = transaction.path[1]
            if not _clear(
                self.grid,
                np.asarray(state["position"][:2]),
                np.asarray(point),
                state["radius_m"],
                self.margin,
                peers,
                obstacles,
                math.dist(state["position"][:2], point) / max(intentions[moving]["speed_mps"], 0.001),
            ):
                raise RuntimeError("A bounded passing path has no currently certified next segment.")
            visible = [(1, point)]
        index, point = visible[0]
        del transaction.path[1:index]
        speed = max(intentions[moving]["speed_mps"], 0.001)
        point = _path_lookahead(
            transaction.path,
            state["position"],
            speed * max(self.relaxation, 1 / self.turn_rate),
            lambda start, end: _clear(
                self.grid,
                np.asarray(start[:2]),
                np.asarray(end[:2]),
                state["radius_m"],
                self.margin,
                peers,
                obstacles,
                math.dist(start[:2], end[:2]) / speed,
            ),
        )
        result[held] = {
            "target": None,
            "phase": "passing_wait",
            "token": ("passing", transaction.started, held, "hold"),
        }
        result[moving] = {
            "target": list(point),
            "phase": "passing_go",
            "token": ("passing", transaction.started, held, "go"),
        }
        return True
