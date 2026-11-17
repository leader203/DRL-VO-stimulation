# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ROS-owned, bounded human/door coordination using complete joint envelopes."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .planner import OccupancyGrid, convex_footprint, segment_polygon_clear


def _polygon_clearance(point, polygon):
    """Signed center clearance to a convex polygon [m], positive outside."""
    point = np.asarray(point[:2], dtype=float)
    edges = np.roll(polygon, -1, axis=0) - polygon
    relative = point - polygon
    fractions = np.clip(np.sum(relative * edges, axis=1) / np.sum(edges * edges, axis=1), 0, 1)
    distance = float(np.min(np.linalg.norm(relative - fractions[:, None] * edges, axis=1)))
    inside = bool(np.all(edges[:, 0] * relative[:, 1] - edges[:, 1] * relative[:, 0] >= -1e-12))
    return -distance if inside else distance


def _segment_point_distance(start, end, point):
    delta = end - start
    length = float(np.dot(delta, delta))
    fraction = float(np.clip(np.dot(point - start, delta) / length, 0, 1)) if length else 0.0
    return float(np.linalg.norm(point - (start + fraction * delta)))


class _NoCertifiedRefuge(RuntimeError):
    """Candidate search failed after the actual starting pose was certified."""


@dataclass
class _Access:
    context: tuple
    origin_inside: bool
    origin_side: float
    deadline: float
    crossed: bool = False


@dataclass
class _Door:
    name: str
    body: str
    interior: np.ndarray
    sweep: np.ndarray
    portal: np.ndarray
    approach_distance_m: float
    facing_half_angle_deg: float
    require_front_approach: bool
    exit_approach_distance_m: float
    specification: dict
    closed: np.ndarray | None = None
    mode: str = "idle"
    motion_started: float | None = None
    observation_started: float | None = None
    requesters: dict[str, _Access] = field(default_factory=dict)
    logged_admissions: dict[str, tuple] = field(default_factory=dict)
    released_entries: dict[str, _Access] = field(default_factory=dict)


@dataclass
class _Retreat:
    origin: np.ndarray
    target: np.ndarray
    doors: tuple[str, ...]
    started: float
    timeout: float
    cleared: bool = False


@dataclass
class _ClosingRetreat:
    context: tuple
    origin: np.ndarray
    target: np.ndarray
    doors: tuple[str, ...]
    goal: np.ndarray
    tolerance: float
    started: float
    timeout: float
    phase: str = "yield"


@dataclass
class _ClosingApproach:
    context: tuple
    origin: np.ndarray
    deadline: float


class DoorCoordinator:
    """Authorize door motion only after measured bodies clear its full swept region.

    Temporary refuge objectives stay within the configured yield distance [m] of the interaction origin.
    They do not replace the original YAML route, heading, wait or arrival radius.
    Near-portal admission remains authorized across full opening until measured
    passage and two sweep-clear observations; its original deadline never renews.
    """

    def __init__(self, grid: OccupancyGrid, settings: dict):
        self.grid = grid
        self.margin = float(settings.get("map_clearance_m", 0.05))
        self.pair_clearance = float(settings.get("pair_clearance_m", 0.25))
        self.turn_rate = float(settings.get("max_turn_rate_radps", 2.5))
        if not math.isfinite(self.turn_rate) or self.turn_rate <= 0:
            raise ValueError("Door access requires a positive finite max_turn_rate_radps [rad/s].")
        self.doors = {}
        for spec in settings.get("door_specs", []):
            if not spec.get("enabled", True):
                continue
            name = str(spec["name"])
            if name in self.doors:
                raise ValueError(f"Duplicate door specification: {name}")
            portal = np.asarray(spec.get("doorway_segment_xy_m"), dtype=float)
            approach = spec.get("approach_distance_m")
            facing = spec.get("facing_half_angle_deg", 180.0)
            front = spec.get("require_front_approach", False)
            exit_approach = spec.get("exit_approach_distance_m", approach)
            if (
                portal.shape != (2, 2)
                or not np.isfinite(portal).all()
                or np.linalg.norm(portal[1] - portal[0]) <= 1e-10
            ):
                raise ValueError(
                    f"Door {name} requires an explicit finite doorway_segment_xy_m; regenerate its descriptors."
                )
            if type(approach) not in (int, float) or not math.isfinite(approach) or approach <= 0:
                raise ValueError(f"Door {name} requires positive finite approach_distance_m from its metadata.")
            if type(facing) not in (int, float) or not 0 <= facing <= 180:
                raise ValueError(f"Door {name} requires facing_half_angle_deg in [0, 180].")
            if type(front) is not bool:
                raise ValueError(f"Door {name} require_front_approach must be boolean.")
            if type(exit_approach) not in (int, float) or not math.isfinite(exit_approach) or exit_approach <= 0:
                raise ValueError(f"Door {name} requires positive finite exit_approach_distance_m.")
            if type(spec.get("exclusive_entry", False)) is not bool:
                raise ValueError(f"Door {name} exclusive_entry must be boolean.")
            self.doors[name] = _Door(
                name,
                str(spec["body_prim_path"]),
                convex_footprint(spec["interior_polygon_xy_m"]),
                convex_footprint(spec["swept_footprint_xy_m"]),
                portal,
                float(approach),
                float(facing),
                front,
                float(exit_approach),
                dict(spec),
                closed=(convex_footprint(spec["closed_footprint_xy_m"]) if spec.get("closed_footprint_xy_m") else None),
            )
        self.previous = {}
        self.retreats: dict[str, _Retreat] = {}
        self.closures: dict[str, _ClosingRetreat] = {}
        self._closure_retries = {}
        self._closing_approaches: dict[str, _ClosingApproach] = {}
        self.max_retreat_m = float(settings.get("max_yield_distance_m", 0.4))

    def reset(self):
        """Clear per-run interactions without changing source door descriptions."""
        self.previous.clear()
        self.retreats.clear()
        self.closures.clear()
        self._closure_retries.clear()
        self._closing_approaches.clear()
        for door in self.doors.values():
            door.mode = "idle"
            door.motion_started = door.observation_started = None
            door.requesters.clear()
            door.logged_admissions.clear()
            door.released_entries.clear()

    def admission_intentions(
        self, states: list[dict], intentions: dict[str, dict], deferred_names: Sequence[str] = ()
    ) -> dict[str, dict]:
        """Copy door requests, deferring queued entries while preserving real exits.

        A valid outgoing lease survives retreat beyond the initial portal
        distance. Per-door markers also defer an entry into another room without
        cancelling an actor's current exit. This query never admits a request,
        modifies its original target, or mutates the supplied intentions.

        Args:
            states: Measured ROS-map positions [m], yaw [rad], radii/heights [m].
            intentions: Original targets [m], speed [m/s], arrival tolerance [m],
                active flags and route contexts.
            deferred_names: Actors whose goal queue has paused entry progression.

        Returns:
            Copied requests with per-door deferrals consumed by :meth:`update`.
        """
        actors = {state["name"]: state for state in states}
        result = {name: dict(intention) for name, intention in intentions.items()}
        for intention in result.values():
            intention.pop("_deferred_entry_doors", None)
        for name in set(deferred_names):
            if name not in actors or name not in result or not result[name].get("active", False):
                continue
            entry_doors = tuple(
                door.name for door in self.doors.values() if self._new_entry(door, actors[name], intentions[name])
            )
            if entry_doors:
                result[name]["_deferred_entry_doors"] = entry_doors
                if not any(self._outgoing_actor(door, actors[name], intentions[name]) for door in self.doors.values()):
                    result[name]["active"] = False
        return result

    def update(self, time_s: float, states: list[dict], intentions: dict, obstacles, door_states=()):
        """Return temporary objectives [m], conservative obstacles and door intents."""
        if not self.doors:
            return {
                "objectives": {},
                "obstacles": obstacles,
                "door_commands": [],
                "closure_status": {"transactions": {}, "blocked": {}},
                "outgoing_actors": [],
            }
        actors = {s["name"]: s for s in states}
        observed = {s["name"]: s for s in door_states}
        closing_objectives, closure_status = self._closing_objectives(time_s, actors, intentions, obstacles, observed)
        wants = {}
        fully_open = {}
        safe = {}
        commands = {}
        needed = {name: [] for name in actors}
        queued = {name: [] for name in actors}
        face_targets = {}
        position_targets = {}
        events = []

        def aim_at_door(door, state, yaw):
            name = state["name"]
            distance = _segment_point_distance(*door.portal, np.asarray(state["position"][:2]))
            candidate = (distance, door.name, yaw)
            if name not in face_targets or candidate[:2] < face_targets[name][:2]:
                face_targets[name] = candidate

        def move_to_front(door, state):
            name = state["name"]
            candidate = (_segment_point_distance(*door.portal, np.asarray(state["position"][:2])), door.name)
            if name not in position_targets or candidate < position_targets[name]:
                position_targets[name] = candidate

        for name, door in self.doors.items():
            observation = observed.get(name, {})
            valid = observation.get("valid", False) is True
            fully_open[name] = valid and observation.get("fully_open", False) is True
            interested, admitted = self._collect_requests(
                door, time_s, actors, intentions, fully_open[name], queued, needed, aim_at_door, move_to_front
            )
            events.extend(admitted)
            wants[name] = interested
            safe[name] = all(
                self._outside(state, door)
                and actor_name in self.previous
                and self._outside(self.previous[actor_name], door)
                for actor_name, state in actors.items()
            )
            if fully_open[name] and door.mode != "closing":
                door.mode = "open"
                door.motion_started = None
            if interested and not valid:
                if door.observation_started is None:
                    door.observation_started = time_s
                if time_s - door.observation_started > 5.0:
                    raise RuntimeError(f"Door {name} has no valid physical observation.")
            else:
                door.observation_started = None
            if not fully_open[name] and (interested or door.mode == "opening"):
                for actor_name, state in actors.items():
                    if not self._outside(state, door):
                        needed[actor_name].append(name)
            if fully_open[name]:
                hold = bool(interested) or not safe[name]
                commands[name] = {"name": name, "hold": hold, "request_open": bool(interested)}
                if not hold:
                    if door.mode != "closing":
                        door.motion_started = time_s
                    door.mode = "closing"
                else:
                    door.mode = "open"
                    door.motion_started = None
            elif interested or door.mode == "opening":
                authorize = safe[name] and valid and observation.get("angular_speed_degps") is not None
                evidence = None
                if (
                    authorize
                    and door.mode != "opening"
                    and (door.facing_half_angle_deg < 180 or door.require_front_approach)
                ):
                    # Refuge movement may have turned the requester away after
                    # admission. Measure again before starting the joint.
                    evidence = self._opening_facing_evidence(
                        door, interested, actors, intentions, aim_at_door, move_to_front
                    )
                    authorize = evidence is not None
                commands[name] = {"name": name, "hold": not authorize, "request_open": authorize}
                if evidence is not None:
                    commands[name]["opening_authorization"] = evidence
                if authorize:
                    if door.mode != "opening":
                        door.motion_started = time_s
                    door.mode = "opening"
            else:
                closed = self._observed_closed(door, observation)
                if closed:
                    door.mode = "idle"
                    door.motion_started = None
                    door.requesters.clear()
                    door.logged_admissions.clear()
                    door.released_entries.clear()
                closing = door.mode == "closing" and safe[name] and valid
                commands[name] = {"name": name, "hold": not closing, "request_open": False}
            if door.mode in ("opening", "closing") and door.motion_started is not None:
                nominal = float(
                    door.specification.get("target_ramp_duration_s", door.specification.get("opening_duration_s", 1.0))
                )
                if time_s - door.motion_started > max(5.0, nominal + 4.0):
                    raise RuntimeError(f"Door {name} did not reach its observed {door.mode} endpoint in time.")
        objectives = self._objectives(time_s, actors, intentions, needed, wants, fully_open, obstacles, queued=queued)
        self._positioning_objectives(position_targets, actors, intentions, obstacles, queued, commands, objectives)
        for actor_name, (_, door_name, yaw) in face_targets.items():
            existing = objectives.get(actor_name)
            if not queued[actor_name] and (existing is None or existing["phase"] == "door_wait"):
                objectives[actor_name] = {
                    "target": None,
                    "yaw": yaw,
                    "phase": "door_face",
                    "token": ("face", door_name),
                }
        objectives.update(closing_objectives)
        adjusted = self._obstacle_model(obstacles, commands, safe, fully_open)
        self.previous = {name: {**state, "position": list(state["position"])} for name, state in actors.items()}
        return {
            "objectives": objectives,
            "obstacles": adjusted,
            "door_commands": list(commands.values()),
            "events": events,
            "closure_status": closure_status,
            "committed_exits": sorted(
                {
                    actor
                    for name, door in self.doors.items()
                    if fully_open[name]
                    for actor, access in door.requesters.items()
                    if access.origin_inside
                    and (access.crossed or not self._entry_blocks_exit_goal(door, actor, actors, intentions, queued))
                }
            ),
            "outgoing_actors": sorted(
                name
                for name, state in actors.items()
                if any(self._outgoing_actor(door, state, intentions.get(name, {})) for door in self.doors.values())
            ),
        }

    def _collect_requests(
        self, door, time_s, actors, intentions, fully_open, queued, needed, aim_at_door, move_to_front
    ):
        """Admit nearby actors on certified portal routes after measuring their heading."""
        self._refresh_access(door, time_s, actors, intentions)
        events = []
        for actor_name, state in actors.items():
            intention = intentions.get(actor_name, {})
            closure = self.closures.get(actor_name)
            if closure is not None and door.name in closure.doors:
                continue
            deferred_entry = door.name in intention.get("_deferred_entry_doors", ())
            if intention.get("target") is None or not (intention.get("active", False) or deferred_entry):
                continue
            if self._queue_entry(door, state, intention, actors, queued, needed, deferred=deferred_entry):
                continue
            if deferred_entry or actor_name in door.requesters or not self._needs_portal(door, state, intention):
                continue
            if (
                door.require_front_approach
                and not fully_open
                and door.mode != "opening"
                and not (self._opening_position(door, state) and self._outside(state, door))
            ):
                move_to_front(door, state)
                continue
            yaw, error, intersects = self._portal_facing(door, state, intention)
            if not intersects or abs(error) > door.facing_half_angle_deg + 1e-8:
                aim_at_door(door, state, yaw)
                continue
            context = self._request_context(intention)
            door.requesters[actor_name] = self._new_access(door, state, intention, time_s)
            if not fully_open and door.logged_admissions.get(actor_name) != context:
                event = {
                    "door": door.name,
                    "actor": actor_name,
                    "time_s": time_s,
                    "distance_m": _segment_point_distance(*door.portal, np.asarray(state["position"][:2])),
                    "approach_distance_m": self._opening_range(door, self._inside_room(state, door)),
                    "reason": "portal_approach",
                }
                if door.facing_half_angle_deg < 180 or door.require_front_approach:
                    event.update(
                        facing_error_deg=abs(error),
                        signed_facing_error_deg=error,
                        facing_half_angle_deg=door.facing_half_angle_deg,
                    )
                events.append(event)
                door.logged_admissions[actor_name] = context
        interested = list(door.requesters)
        if not fully_open and door.facing_half_angle_deg == 180 and not door.require_front_approach:
            # Preserve support for older descriptors without a heading gate.
            interested.extend(
                name
                for name, retreat in self.retreats.items()
                if name not in door.requesters
                and door.name in retreat.doors
                and door.name not in queued[name]
                and intentions.get(name, {}).get("active", False)
            )
        return interested, events

    def _opening_facing_evidence(self, door, interested, actors, intentions, aim_at_door, move_to_front):
        awaiting_heading = []
        for name in interested:
            state = actors[name]
            position = np.asarray(state["position"][:2])
            if door.require_front_approach:
                if not self._opening_position(door, state):
                    move_to_front(door, state)
                    continue
            elif _segment_point_distance(*door.portal, position) > door.approach_distance_m + 1e-10:
                retreat = self.retreats.get(name)
                if (
                    retreat is None
                    or door.name not in retreat.doors
                    or np.linalg.norm(position - retreat.origin) > self.max_retreat_m + 1e-7
                ):
                    # A prior admission alone does not authorize an actor that
                    # has since drifted away. Only a bounded refuge may do so.
                    door.requesters.pop(name, None)
                    door.logged_admissions.pop(name, None)
                    continue
            yaw, error, intersects = self._portal_facing(door, state, intentions[name])
            if intersects and abs(error) <= door.facing_half_angle_deg + 1e-8:
                return {
                    "actor": name,
                    "signed_facing_error_deg": error,
                    "facing_half_angle_deg": door.facing_half_angle_deg,
                }
            awaiting_heading.append((state, yaw))
        for state, yaw in awaiting_heading:
            aim_at_door(door, state, yaw)
        return None

    @staticmethod
    def _opening_range(door, inside):
        return door.exit_approach_distance_m if door.require_front_approach and inside else door.approach_distance_m

    def _opening_position(self, door, state):
        """Check measured frontage and range [m] on the actor's side of the leaf."""
        if not self._at_door_height(state, door):
            return False
        a, b = door.portal
        point = np.asarray(state["position"][:2])
        edge = b - a
        width = float(np.linalg.norm(edge))
        projection = float(np.dot(point - a, edge)) / width
        inside = self._inside_room(state, door)
        # An inward-swinging leaf can leave only a jamb-side exit refuge.
        # Its resident body must still overlap the physical doorway frontage.
        allowance = float(state["radius_m"]) if inside else 0.0
        approach = self._opening_range(door, inside)
        return (
            -allowance - 1e-10 <= projection <= width + allowance + 1e-10
            and _segment_point_distance(a, b, point) <= approach + 1e-10
        )

    def _positioning_objectives(self, targets, actors, intentions, obstacles, queued, commands, objectives):
        for name, (_, door_name) in targets.items():
            if queued[name] or commands[door_name]["request_open"]:
                continue
            if name in objectives and objectives[name]["phase"] != "door_wait":
                continue
            door, state = self.doors[door_name], actors[name]
            point = self._front_waiting_pose(door, state, actors, obstacles)
            if point is None:
                continue
            speed = max(float(intentions[name].get("speed_mps", 1.0)), 0.001)
            horizon = math.dist(state["position"][:2], point) / speed + math.pi / self.turn_rate
            objectives[name] = {
                "target": point.tolist(),
                "phase": "door_position",
                "token": ("front", door_name),
                "certified_ray": self._refuge_ray_clear(state, point, actors, obstacles, horizon),
            }

    def _front_waiting_pose(self, door, state, actors, obstacles):
        """Find a nearby swept-safe opening pose [m] without crossing the leaf."""
        a, b = door.portal
        edge = b - a
        width = float(np.linalg.norm(edge))
        direction = edge / width
        normal = np.array([-direction[1], direction[0]])
        if np.dot(np.mean(door.interior, axis=0) - a, normal) > 0:
            normal = -normal
        inside = self._inside_room(state, door)
        if inside:
            normal = -normal
        allowance = float(state["radius_m"]) if inside else 0.0
        approach = self._opening_range(door, inside)
        start = np.asarray(state["position"][:2])
        reach = max(2 * approach, width)
        retreat = self.retreats.get(state["name"])
        origin = retreat.origin if retreat is not None and door.name in retreat.doors else None
        try:
            path_clear = self._refuge_path_clear(state, actors, obstacles, search_distance_m=reach)
        except RuntimeError:
            return None
        left, right = -allowance + 0.02 * width, width + allowance - 0.02 * width
        laterals = [*np.linspace(left, right, 9), float(np.clip(np.dot(start - a, direction), left, right))]
        candidates = [
            a + lateral * direction + depth * normal
            for lateral in laterals
            for depth in np.linspace(0.35 * approach, approach, 9)
        ]
        for point in sorted(candidates, key=lambda item: math.dist(start, item)):
            if math.dist(start, point) > reach or origin is not None and math.dist(origin, point) > self.max_retreat_m:
                continue
            pose = {**state, "position": [*point, state["position"][2]]}
            if self._opening_position(door, pose) and self._outside(pose, door) and path_clear(point):
                return point
        return None

    def _closing_objectives(self, time_s, actors, intentions, obstacles, observed):
        objectives, blocked, transactions = {}, {}, {}
        for name, state in actors.items():
            intention = intentions.get(name, {})
            context = self._request_context(intention)
            existing = self.closures.get(name)
            ordinary = intention.get("can_door_refuge", False) is True
            allowed = ordinary or intention.get("settle_door_entry", False)
            if existing is not None and (not allowed or existing.context != context):
                self.closures.pop(name)
                existing = None
            if existing is None:
                goal = intention.get("target")
                tolerance = float(intention.get("arrival_tolerance_m", 0.0))
                if not allowed or goal is None or math.dist(state["position"], goal) > tolerance:
                    self._closure_retries.pop(name, None)
                    continue
                approach = self._closing_approaches.get(name)
                if (
                    not ordinary
                    and approach is not None
                    and approach.context == context
                    and time_s <= approach.deadline
                    and math.dist(state["position"][:2], goal[:2]) > 1e-4
                ):
                    continue
                if not ordinary and (
                    waiting := self._wait_for_clear_entry_closure(
                        time_s, state, intention, actors, intentions, observed
                    )
                ):
                    objectives[name] = waiting
                    continue
                required = self._closing_doors(state, goal, observed)
                if not ordinary:
                    required = tuple(d for d in required if self._has_entered(self.doors[d], state, intention))
                if not required:
                    continue
                # Optional closing must never hide invalid actual poses behind
                # its geometric lower bound or a cached unsuccessful search.
                self._refuge_path_clear(state, actors, obstacles)
                radius = float(state["radius_m"]) + self.margin
                if any(
                    self.doors[d].closed is None
                    or _polygon_clearance(state["position"], self.doors[d].closed) <= radius + 1e-6
                    for d in required
                ):
                    blocked[name] = {
                        "doors": list(required),
                        "reason": "Arrival body is not certified clear of the closed leaf.",
                    }
                    continue
                origin = np.asarray(state["position"][:2], dtype=float)
                target, unavailable = self._optional_closing_refuge(
                    time_s, state, origin, context, required, actors, obstacles
                )
                if unavailable is not None:
                    blocked[name] = unavailable
                    continue
                speed = max(float(intention.get("speed_mps", 1.0)), 0.001)
                ramp = max(float(self.doors[d].specification.get("target_ramp_duration_s", 1.0)) for d in required)
                timeout = 2 * math.pi / self.turn_rate + 2 * self.max_retreat_m / speed + max(5.0, ramp + 4.0) + 2.0
                existing = _ClosingRetreat(
                    context, origin, target, required, np.asarray(goal), tolerance, time_s, timeout
                )
                self.closures[name] = existing
                self.retreats.pop(name, None)
            if time_s - existing.started > existing.timeout:
                raise RuntimeError(f"Door closing refuge timed out for {name}; original route retained.")
            for door_name in existing.doors:
                self.doors[door_name].requesters.pop(name, None)
                self.doors[door_name].logged_admissions.pop(name, None)
            position = np.asarray(state["position"][:2], dtype=float)
            exiting = any(
                self._outgoing_priority(self.doors[d], other, intentions.get(other_name, {}))
                for d in existing.doors
                for other_name, other in actors.items()
                if other_name != name
            )
            if existing.phase == "return" and exiting:
                existing.phase = "yield"
            if existing.phase == "yield" and np.linalg.norm(position - existing.target) <= 1e-4:
                existing.phase = "wait"
            closed = all(self._observed_closed(self.doors[d], observed.get(d, {})) for d in existing.doors)
            if existing.phase == "wait" and closed and not exiting:
                existing.phase = "return"
            if existing.phase == "return":
                if not closed:
                    existing.phase = "yield"
                elif (
                    np.linalg.norm(position - existing.origin) <= 1e-4
                    and math.dist(state["position"], existing.goal) <= existing.tolerance
                ):
                    self.closures.pop(name)
                    continue
            target = existing.target if existing.phase == "yield" else existing.origin
            directive = {
                "target": None if existing.phase == "wait" else target.tolist(),
                "phase": "door_close_" + existing.phase,
                "origin": existing.origin.tolist(),
                "max_displacement_m": self.max_retreat_m,
                "token": ("close", existing.phase, *existing.doors, *existing.origin.tolist()),
            }
            objectives[name] = directive
            transactions[name] = {**directive, "doors": list(existing.doors), "started_s": existing.started}
        return objectives, {"transactions": transactions, "blocked": blocked}

    def prepare_closing_arrival(self, time_s, states, intentions, obstacles, door_states, excluded):
        """Continue to a certified authored endpoint before dwell, within its original arrival ball [m]."""
        if not self.doors:
            return {}
        actors = {state["name"]: state for state in states}
        observed = {state["name"]: state for state in door_states}
        approaches, objectives = {}, {}
        for name, state in actors.items():
            intention = intentions[name]
            ordinary = intention.get("prepare_door_refuge")
            if name in excluded or not (ordinary or intention.get("settle_door_entry")):
                continue
            goal = intention["target"]
            if math.dist(state["position"], goal) > intention["arrival_tolerance_m"]:
                continue
            start = np.asarray(state["position"][:2], dtype=float)
            target = np.asarray(goal[:2], dtype=float)
            distance = float(np.linalg.norm(target - start))
            if distance <= 1e-4 or distance > self.max_retreat_m:
                continue
            context = self._request_context(intention)
            approach = self._closing_approaches.get(name)
            if approach is not None and approach.context != context:
                approach = None
            if approach is not None and time_s > approach.deadline:
                continue
            if approach is None:
                required = self._closing_doors(state, goal, observed)
                if not ordinary:
                    required = tuple(d for d in required if self._has_entered(self.doors[d], state, intention))
                radius = float(state["radius_m"]) + self.margin
                if not any(
                    radius - _polygon_clearance(start, self.doors[door].sweep) > self.max_retreat_m + 1e-9
                    for door in required
                ):
                    continue
                if any(
                    self.doors[door].closed is None
                    or _polygon_clearance(target, self.doors[door].closed) <= radius + 1e-6
                    for door in required
                ):
                    continue
            try:
                if not self._refuge_path_clear(state, actors, obstacles)(target):
                    continue
                if approach is None:
                    proposed = {**state, "position": [*target, state["position"][2]]}
                    if any(not self._outside(proposed, self.doors[door]) for door in required):
                        self._refuge(
                            proposed,
                            target,
                            [self.doors[door].sweep for door in required],
                            actors,
                            obstacles,
                            inside=[self.doors[door].interior for door in required],
                            return_blockers=[self.doors[door].closed for door in required],
                        )
            except RuntimeError:
                # An optional arrival refinement must not override ordinary arrival.
                continue
            if approach is None:
                speed = max(float(intention["speed_mps"]), 0.001)
                deadline = time_s + math.pi / self.turn_rate + 2 * distance / speed + 1
                approach = _ClosingApproach(context, start, deadline)
            approaches[name] = approach
            objectives[name] = {
                "target": target.tolist(),
                "phase": "door_arrival",
                "token": ("door_arrival", *context),
                "origin": approach.origin.tolist(),
                "max_displacement_m": self.max_retreat_m,
                "certified_ray": True,
            }
        self._closing_approaches = approaches
        return objectives

    def _closing_doors(self, state, goal, observed):
        return tuple(
            name
            for name, door in self.doors.items()
            if observed.get(name, {}).get("valid")
            and observed[name].get("fully_open")
            and not self._outside(state, door)
            and _polygon_clearance(state["position"], door.interior) <= 0
            and _polygon_clearance(goal, door.interior) <= 0
        )

    def _has_entered(self, door, state, intention):
        access = door.requesters.get(state["name"])
        return (
            access is not None
            and not access.origin_inside
            and access.context == self._request_context(intention)
            and self._inside_room(state, door)
        )

    def _wait_for_clear_entry_closure(self, time_s, state, intention, actors, intentions, observed):
        """Hold a sweep-clear arrival until the door is measured closed, within the entry lease."""
        name = state["name"]
        waiting = []
        for door in self.doors.values():
            access = door.released_entries.get(name) or door.requesters.get(name)
            observation = observed.get(door.name, {})
            if (
                access is None
                or access.origin_inside
                or access.context != self._request_context(intention)
                or time_s > access.deadline
                or not observation.get("valid")
                or self._observed_closed(door, observation)
                or not self._outside(state, door)
                or not self._inside_room(state, door)
                or _polygon_clearance(intention["target"], door.interior) > 0
                or any(other != name for other in door.requesters)
                or any(
                    other != name and self._outgoing_priority(door, body, intentions.get(other, {}))
                    for other, body in actors.items()
                )
            ):
                continue
            door.released_entries[name] = access
            door.requesters.pop(name, None)
            waiting.append(door.name)
        if not waiting:
            return None
        return {"target": None, "phase": "door_close_wait", "token": ("clear_entry_close", *waiting)}

    def _optional_closing_refuge(self, time_s, state, origin, context, required, actors, obstacles):
        radius = float(state["radius_m"]) + self.margin
        minimum = max(0.0, *(radius - _polygon_clearance(origin, self.doors[d].sweep) for d in required))
        unavailable = {
            "reason": "no_refuge",
            "doors": list(required),
            "minimum_displacement_m": minimum,
            "max_displacement_m": self.max_retreat_m,
        }
        if minimum > self.max_retreat_m + 1e-9:
            return None, {**unavailable, "message": "The sweep clearance lower bound exceeds the retreat limit."}
        key = (context, required, radius)
        cached = self._closure_retries.get(state["name"])
        if cached is not None and cached["key"] == key and time_s < cached["retry_at_s"]:
            return None, dict(cached["blocked"])
        try:
            target = self._refuge(
                state,
                origin,
                [self.doors[d].sweep for d in required],
                actors,
                obstacles,
                inside=[self.doors[d].interior for d in required],
                return_blockers=[self.doors[d].closed for d in required],
            )
        except _NoCertifiedRefuge as error:
            unavailable.update(message=str(error), search_origin_xy_m=origin.tolist(), retry_at_s=time_s + 0.5)
            self._closure_retries[state["name"]] = {
                "key": key,
                "retry_at_s": time_s + 0.5,
                "blocked": unavailable,
            }
            return None, unavailable
        self._closure_retries.pop(state["name"], None)
        return target, None

    def _outgoing_priority(self, door, state, intention):
        target = intention.get("target")
        if not self._outgoing_actor(door, state, intention):
            return False
        access = door.requesters.get(state["name"])
        if access is not None and access.origin_inside and access.context == self._request_context(intention):
            return True
        return (
            _polygon_clearance(state["position"], door.interior) <= 0
            and _polygon_clearance(target, door.interior) > 0
            and self._needs_portal(door, state, intention)
        )

    def _outgoing_actor(self, door, state, intention):
        target = intention.get("target")
        if not intention.get("active", False) or target is None or not self._at_door_height(state, door):
            return False
        if abs(state["position"][2] - target[2]) >= intention.get("arrival_tolerance_m", math.inf):
            return False
        access = door.requesters.get(state["name"])
        return bool(
            access is not None
            and access.origin_inside
            and access.context == self._request_context(intention)
            or self._inside_room(state, door)
            and _polygon_clearance(target, door.interior) > 0
        )

    def _entry_blocks_exit_goal(self, door, actor, actors, intentions, queued):
        """Keep the resource exchange when a same-door entrant occupies the exit goal."""
        intention = intentions[actor]
        target = intention["target"][:2]
        return any(
            other != actor
            and door.name in queued[other]
            and math.dist(body["position"][:2], target)
            <= actors[actor]["radius_m"]
            + body["radius_m"]
            + self.pair_clearance
            + intention.get("arrival_tolerance_m", 0.0)
            for other, body in actors.items()
        )

    def _new_entry(self, door, state, intention):
        target = intention.get("target")
        if target is None or not self._at_door_height(state, door) or self._inside_room(state, door):
            return False
        if abs(state["position"][2] - target[2]) >= intention.get("arrival_tolerance_m", math.inf):
            return False
        if self._outgoing_actor(door, state, intention):
            return False
        access = door.requesters.get(state["name"])
        return bool(
            access is not None
            and not access.origin_inside
            and access.context == self._request_context(intention)
            or _polygon_clearance(target, door.interior) <= 0
            or self._needs_portal(door, state, intention)
        )

    @staticmethod
    def _at_door_height(state, door):
        low, high = door.specification.get("z_min_m", -math.inf), door.specification.get("z_max_m", math.inf)
        return state["position"][2] <= high and state["position"][2] + state.get("height_m", 1.8) >= low

    @staticmethod
    def _inside_room(state, door):
        return (
            DoorCoordinator._at_door_height(state, door) and _polygon_clearance(state["position"], door.interior) <= 0
        )

    def _occupies_room(self, state, door):
        access = door.requesters.get(state["name"])
        # A measured outgoing lease keeps the threshold reserved until its
        # existing two-frame full-sweep clearance test releases it.
        return self._inside_room(state, door) or (access is not None and access.origin_inside)

    def _queue_entry(self, door, state, intention, actors, queued, needed, *, deferred=False):
        name = state["name"]
        closing = any(door.name in closure.doors and actor != name for actor, closure in self.closures.items())
        exclusive = door.specification.get("exclusive_entry", False) and not self._inside_room(state, door)
        if not closing and not exclusive:
            return False
        if self._outgoing_priority(door, state, intention):
            return False
        retreat = self.retreats.get(name)
        pending_retreat = (
            retreat is not None
            and door.name in retreat.doors
            and _polygon_clearance(intention["target"], door.interior) <= 0
        )
        if (
            not deferred
            and name not in door.requesters
            and not pending_retreat
            and not self._needs_portal(door, state, intention)
        ):
            return False
        occupied = exclusive and any(
            other != name and self._occupies_room(body, door) for other, body in actors.items()
        )
        if not closing and not occupied:
            return False
        door.requesters.pop(name, None)
        door.logged_admissions.pop(name, None)
        queued[name].append(door.name)
        if not self._outside(state, door):
            needed[name].append(door.name)
        return True

    @staticmethod
    def _observed_closed(door, state):
        angle, speed = state.get("actual_angle_deg"), state.get("angular_speed_degps")
        return (
            state.get("valid") is True
            and angle is not None
            and speed is not None
            and math.isfinite(angle)
            and math.isfinite(speed)
            and abs(math.remainder(angle - float(door.specification.get("closed_angle_deg", 0.0)), 360))
            <= float(door.specification.get("open_tolerance_deg", 1.0))
            and abs(speed) <= float(door.specification.get("settle_speed_degps", 5.0))
        )

    @staticmethod
    def _request_context(intention):
        return tuple(intention.get("route_context", ())), tuple(intention.get("target") or ())

    @staticmethod
    def _portal_side(door, point):
        edge = door.portal[1] - door.portal[0]
        relative = np.asarray(point[:2]) - door.portal[0]
        return float(edge[0] * relative[1] - edge[1] * relative[0]) / float(np.linalg.norm(edge))

    def _new_access(self, door, state, intention, time_s):
        span = float(np.linalg.norm(np.ptp(door.sweep, axis=0)))
        speed = max(float(intention.get("speed_mps", 1.0)), 0.001)
        turning = 2 * math.pi / self.turn_rate
        movement = 2 * (span + self.max_retreat_m + state["radius_m"] + self.margin) / speed
        ramp = float(
            door.specification.get("target_ramp_duration_s", door.specification.get("opening_duration_s", 1.0))
        )
        return _Access(
            self._request_context(intention),
            _polygon_clearance(state["position"], door.interior) <= 0,
            self._portal_side(door, state["position"]),
            time_s + max(5.0, ramp + 4.0) + turning + movement,
        )

    def _refresh_access(self, door, time_s, actors, intentions):
        for actor, access in list(door.released_entries.items()):
            state, intention = actors.get(actor), intentions.get(actor, {})
            if (
                state is None
                or not intention.get("settle_door_entry")
                or self._request_context(intention) != access.context
                or time_s > access.deadline
                or not self._inside_room(state, door)
            ):
                door.released_entries.pop(actor)
        for actor, access in list(door.requesters.items()):
            state, intention = actors.get(actor), intentions.get(actor, {})
            if state is None or not intention.get("active") or self._request_context(intention) != access.context:
                door.requesters.pop(actor)
                door.logged_admissions.pop(actor, None)
                continue
            access.crossed |= (_polygon_clearance(state["position"], door.interior) <= 0) != access.origin_inside or (
                self._portal_side(door, state["position"]) * access.origin_side < -1e-8
            )
            if (
                access.crossed
                and self._outside(state, door)
                and actor in self.previous
                and self._outside(self.previous[actor], door)
            ):
                if (
                    not access.origin_inside
                    and intention.get("settle_door_entry")
                    and self._inside_room(state, door)
                    and _polygon_clearance(intention["target"], door.interior) <= 0
                ):
                    door.released_entries[actor] = access
                door.requesters.pop(actor)
                door.logged_admissions.pop(actor, None)
            elif time_s > access.deadline:
                raise RuntimeError(f"Door access timed out for {actor} at {door.name}; original route retained.")

    @staticmethod
    def _portal_facing(door, state, intention):
        """Return the midpoint yaw [rad], signed error [deg], and finite forward-ray hit."""
        if door.facing_half_angle_deg == 180 and not door.require_front_approach:
            return 0.0, 0.0, True
        position = np.asarray(state["position"][:2], dtype=float)
        start, end = door.portal
        edge = end - start
        direction = start + 0.5 * edge - position
        if _segment_point_distance(start, end, position) <= 1e-10:
            direction = np.asarray(intention["target"][:2]) - position
        desired = math.atan2(direction[1], direction[0])
        yaw = state.get("yaw")
        if yaw is None:
            return desired, math.inf, False
        error = math.degrees(math.remainder(yaw - desired, math.tau))
        forward = math.cos(yaw), math.sin(yaw)
        denominator = float(forward[0] * edge[1] - forward[1] * edge[0])
        if abs(denominator) <= 1e-12:
            # A ray beginning on the portal already intersects it at zero range.
            return desired, error, _segment_point_distance(start, end, position) <= 1e-10
        delta = start - position
        ahead = float(delta[0] * edge[1] - delta[1] * edge[0]) / denominator
        across = float(delta[0] * forward[1] - delta[1] * forward[0]) / denominator
        return desired, error, ahead >= -1e-10 and -1e-10 <= across <= 1.0 + 1e-10

    def _needs_portal(self, door, state, intention):
        position = np.asarray(state["position"][:2], dtype=float)
        low = door.specification.get("z_min_m", -math.inf)
        high = door.specification.get("z_max_m", math.inf)
        if state["position"][2] + state.get("height_m", 1.8) < low or state["position"][2] > high:
            return False
        inside = _polygon_clearance(position, door.interior) <= 0
        if _segment_point_distance(*door.portal, position) > self._opening_range(door, inside) + 1e-10:
            return False
        target = intention["target"]
        if abs(state["position"][2] - target[2]) >= intention.get("arrival_tolerance_m", math.inf):
            return False
        if math.dist(state["position"], target) <= intention.get("arrival_tolerance_m", 0.0):
            return False
        path = intention.get("path", [])
        route = None
        for index in range(len(path) - 1, 0, -1):
            if self.grid.segment_clear(position, path[index], state["radius_m"] + self.margin):
                route = [position, *path[index:]]
                break
        if route is None:
            if not self.grid.segment_clear(position, target, state["radius_m"] + self.margin):
                return False
            route = [position, target[:2]]
        changes_room = inside != (_polygon_clearance(target, door.interior) <= 0)
        return any(
            self._crosses_portal(a, b, door.portal, state["radius_m"] + self.margin, changes_room)
            for a, b in zip(route, route[1:])
        )

    @staticmethod
    def _crosses_portal(start, end, portal, radius, allow_touch):
        start, end = np.asarray(start)[:2], np.asarray(end)[:2]
        # A body can need an open leaf even when its center skirts a jamb.
        if allow_touch and (
            min(
                _segment_point_distance(*portal, start),
                _segment_point_distance(*portal, end),
                _segment_point_distance(start, end, portal[0]),
                _segment_point_distance(start, end, portal[1]),
            )
            <= radius
        ):
            return True
        direction = end - start
        edge = portal[1] - portal[0]
        denominator = float(direction[0] * edge[1] - direction[1] * edge[0])
        if abs(denominator) < 1e-12:
            return False
        delta = portal[0] - start
        along = float(delta[0] * edge[1] - delta[1] * edge[0]) / denominator
        across = float(delta[0] * direction[1] - delta[1] * direction[0]) / denominator
        extension = radius / float(np.linalg.norm(edge))
        return -1e-10 <= along <= 1 + 1e-10 and -extension <= across <= 1 + extension

    def _obstacle_model(self, obstacles, commands, safe, fully_open):
        adjusted = []
        for obstacle in obstacles:
            path = str(obstacle.get("prim_path", obstacle.get("path", "")))
            matching = next((d for d in self.doors.values() if path == d.body or path.startswith(d.body + "/")), None)
            if matching is None:
                adjusted.append(obstacle)
                continue
            command = commands[matching.name]
            moving_envelope = safe[matching.name] and not command["hold"]
            if moving_envelope:
                continue
            if fully_open[matching.name] and command["hold"]:
                adjusted.append(
                    {
                        **obstacle,
                        "velocity": [0.0, 0.0, 0.0],
                        "velocity_valid": True,
                        "motion_model": "observed fully-open held joint",
                    }
                )
            else:
                adjusted.append(obstacle)
        for name, door in self.doors.items():
            if safe[name] and not commands[name]["hold"]:
                center = np.mean(door.sweep, axis=0)
                envelope = {
                    "prim_path": door.body,
                    "position": [*center, 0.0],
                    "radius_m": float(np.max(np.linalg.norm(door.sweep - center, axis=1))),
                    "velocity": [0.0, 0.0, 0.0],
                    "velocity_valid": True,
                    "footprint_xy_m": door.sweep.tolist(),
                    "motion_model": "complete joint swept envelope, not a stationary-leaf assumption",
                    **{key: door.specification[key] for key in ("z_min_m", "z_max_m") if key in door.specification},
                }
                bounds = (envelope.get("z_min_m"), envelope.get("z_max_m"))
                if all(type(value) in (int, float) and math.isfinite(value) for value in bounds):
                    low, high = bounds
                    if low <= high:
                        envelope["position"][2] = 0.5 * low + 0.5 * high
                adjusted.append(envelope)
        return adjusted

    def _objectives(self, time_s, actors, intentions, needed, wants, fully_open, obstacles, *, queued=None):
        objectives = {}
        queued = queued or {name: [] for name in actors}
        for actor_name, state in actors.items():
            if actor_name in self.closures:
                continue
            intention = intentions.get(actor_name, {})
            eligible = intention.get("active", False) or bool(intention.get("_deferred_entry_doors"))
            existing = self.retreats.get(actor_name)
            required = tuple(sorted(set(needed[actor_name])))
            if required and eligible:
                if existing is None or existing.doors != required:
                    origin = np.asarray(state["position"][:2], dtype=float) if existing is None else existing.origin
                    target = self._refuge(
                        state,
                        origin,
                        [self.doors[n].sweep for n in required],
                        actors,
                        obstacles,
                        outside=[self.doors[n].interior for n in queued[actor_name]],
                    )
                    speed = float(intention.get("speed_mps", 1.0))
                    started = existing.started if existing is not None else time_s
                    timeout = (
                        existing.timeout if existing is not None else 1.6 + self.max_retreat_m / max(speed, 0.001) + 4.0
                    )
                    existing = _Retreat(origin, target, required, started, timeout)
                    self.retreats[actor_name] = existing
                existing.cleared = all(self._outside(state, self.doors[n]) for n in existing.doors)
                if not existing.cleared:
                    if time_s - existing.started > existing.timeout:
                        raise RuntimeError(f"Door refuge timed out for {actor_name}; original route retained.")
                    objectives[actor_name] = {
                        "target": existing.target.tolist(),
                        "phase": "door_yield",
                        "origin": existing.origin.tolist(),
                        "max_displacement_m": self.max_retreat_m,
                        "token": ("yield", *existing.doors, *existing.target.tolist()),
                    }
                    if self._refuge_ray_clear(
                        state, existing.target, actors, obstacles, existing.started + existing.timeout - time_s
                    ):
                        objectives[actor_name]["certified_ray"] = True
            if existing is not None and not required:
                existing.cleared = True
            waiting = list(queued[actor_name])
            waiting.extend(name for name in wants if actor_name in wants[name] and not fully_open[name])
            if existing is not None and existing.cleared:
                waiting.extend(
                    name
                    for name in existing.doors
                    if not fully_open[name] and (wants[name] or self.doors[name].mode == "opening")
                )
                if not waiting:
                    self.retreats.pop(actor_name, None)
            if waiting and actor_name not in objectives and eligible:
                critical = any(
                    not self._outside(state, self.doors[door_name])
                    or door_name not in queued[actor_name]
                    and not fully_open[door_name]
                    and (
                        actor_name in self.doors[door_name].requesters
                        or existing is not None
                        and existing.cleared
                        and door_name in existing.doors
                    )
                    for door_name in waiting
                )
                objectives[actor_name] = {
                    "target": None,
                    "phase": "door_wait",
                    "token": ("wait", *sorted(set(waiting))),
                    "safety_critical": critical,
                }
                if queued[actor_name] and not critical:
                    objectives[actor_name].update(
                        queued_door_entry=True,
                        yield_origin_xy_m=(
                            existing.origin.tolist() if existing is not None else list(state["position"][:2])
                        ),
                    )
        return objectives

    def _refuge_ray_clear(self, state, target, actors, obstacles, remaining):
        """Certify the whole refuge line [m] against observed translations until its deadline [s]."""
        if not math.isfinite(remaining) or remaining <= 0:
            return False
        start, target = np.asarray(state["position"][:2]), np.asarray(target[:2])
        radius = float(state["radius_m"]) + self.margin
        if not self.grid.segment_clear(start, target, radius) or any(
            _segment_point_distance(start, target, np.asarray(other["position"][:2]))
            < float(state["radius_m"]) + float(other["radius_m"]) + self.pair_clearance
            for name, other in actors.items()
            if name != state["name"]
        ):
            return False
        for obstacle in obstacles:
            if obstacle.get("velocity_valid") is not True:
                return False
            try:
                velocity = np.asarray(obstacle.get("velocity", []), dtype=float)
            except (TypeError, ValueError):
                return False
            if velocity.shape != (3,) or not np.isfinite(velocity).all():
                return False
            shift = velocity[:2] * remaining
            if obstacle.get("footprint_xy_m") is not None:
                footprint = convex_footprint(obstacle["footprint_xy_m"])
                swept = convex_footprint(np.vstack((footprint, footprint + shift)))
            else:
                point = np.asarray(obstacle["position"][:2], dtype=float)
                other_radius = float(obstacle["radius_m"])
                if not math.isfinite(other_radius) or other_radius <= 0:
                    return False
                low = np.minimum(point, point + shift) - other_radius
                high = np.maximum(point, point + shift) + other_radius
                swept = np.array([[low[0], low[1]], [high[0], low[1]], [high[0], high[1]], [low[0], high[1]]])
            if not segment_polygon_clear(start, target, swept, radius):
                return False
        return True

    def _outside(self, state, door):
        low = door.specification.get("z_min_m", -math.inf)
        high = door.specification.get("z_max_m", math.inf)
        if state["position"][2] + state.get("height_m", 1.8) < low or state["position"][2] > high:
            return True
        return _polygon_clearance(state["position"], door.sweep) > float(state["radius_m"]) + self.margin + 1e-6

    def _refuge_path_clear(self, state, actors, obstacles, *, return_blockers=(), search_distance_m=None):
        start = np.asarray(state["position"][:2], dtype=float)
        radius = float(state["radius_m"]) + self.margin
        reach = self.max_retreat_m if search_distance_m is None else search_distance_m
        blockers = []
        for obstacle in obstacles:
            if obstacle.get("footprint_xy_m") is not None:
                polygon = convex_footprint(obstacle["footprint_xy_m"])
                if _polygon_clearance(start, polygon) <= reach + radius:
                    blockers.append(("polygon", polygon, 0.0))
            else:
                point = np.asarray(obstacle["position"][:2], dtype=float)
                other_radius = float(obstacle["radius_m"])
                if np.linalg.norm(point - start) <= reach + radius + other_radius:
                    blockers.append(("disc", point, radius + other_radius))
        for name, other in actors.items():
            if name != state["name"]:
                blockers.append(
                    (
                        "disc",
                        np.asarray(other["position"][:2], dtype=float),
                        float(state["radius_m"]) + float(other["radius_m"]) + self.pair_clearance,
                    )
                )

        def path_clear(end):
            if not self.grid.segment_clear(start, end, radius):
                return False
            if any(not segment_polygon_clear(start, end, polygon, radius) for polygon in return_blockers):
                return False
            for kind, shape, required_clearance in blockers:
                if kind == "polygon":
                    if not segment_polygon_clear(start, end, shape, radius):
                        return False
                elif _segment_point_distance(start, end, shape) <= required_clearance:
                    return False
            return True

        if not path_clear(start):
            raise RuntimeError(f"No certified door refuge from the actual body pose for {state['name']}.")
        return path_clear

    def _refuge(self, state, origin, sweeps, actors, obstacles, *, inside=(), outside=(), return_blockers=()):
        start = np.asarray(state["position"][:2], dtype=float)
        radius = float(state["radius_m"]) + self.margin
        path_clear = self._refuge_path_clear(state, actors, obstacles, return_blockers=return_blockers)
        cache = {}

        def evaluate(angle):
            key = round(angle % (2 * math.pi), 10)
            if key in cache:
                return cache[key]
            direction = np.array([math.cos(angle), math.sin(angle)])
            relative = start - origin
            b = float(np.dot(relative, direction))
            determinant = b * b + self.max_retreat_m**2 - float(np.dot(relative, relative))
            maximum = max(0.0, -b + math.sqrt(max(0.0, determinant)))
            lower, upper = 0.0, maximum
            if not path_clear(start + direction * upper):
                for _ in range(18):
                    middle = (lower + upper) / 2
                    if path_clear(start + direction * middle):
                        lower = middle
                    else:
                        upper = middle
                maximum = max(0.0, lower - 1e-6)
            point = start + direction * maximum
            clearances = [_polygon_clearance(point, poly) - radius for poly in sweeps]
            clearances.extend(-_polygon_clearance(point, polygon) - radius for polygon in inside)
            clearances.extend(_polygon_clearance(point, polygon) - radius for polygon in outside)
            score = min(clearances)
            cache[key] = (score, maximum, direction)
            return cache[key]

        angles = list(np.linspace(0, 2 * math.pi, 48, endpoint=False))
        for polygon in sweeps:
            for vertex in polygon[:: max(1, len(polygon) // 24)]:
                angles.append(math.atan2(*(vertex - start)[::-1]))
        ranked = sorted(((evaluate(angle)[0], angle) for angle in angles), reverse=True)
        candidates = []
        for _, initial in ranked[:6]:
            angle = initial
            step = 2 * math.pi / 48
            for _ in range(14):
                options = [angle - step, angle, angle + step]
                best = max(options, key=lambda value: evaluate(value)[0])
                if best == angle:
                    step *= 0.5
                else:
                    angle = best
            score, maximum, direction = evaluate(angle)
            if score <= 2e-5:
                continue
            lower, upper = 0.0, maximum
            for _ in range(24):
                middle = (lower + upper) / 2
                point = start + direction * middle
                if (
                    all(_polygon_clearance(point, poly) > radius + 1e-5 for poly in sweeps)
                    and all(_polygon_clearance(point, polygon) < -radius - 1e-5 for polygon in inside)
                    and all(_polygon_clearance(point, polygon) > radius + 1e-5 for polygon in outside)
                ):
                    upper = middle
                else:
                    lower = middle
            distance = min(maximum, upper + 0.002)
            point = start + direction * distance
            if (
                path_clear(point)
                and all(_polygon_clearance(point, poly) > radius + 1e-6 for poly in sweeps)
                and all(_polygon_clearance(point, polygon) < -radius - 1e-6 for polygon in inside)
                and all(_polygon_clearance(point, polygon) > radius + 1e-6 for polygon in outside)
            ):
                candidates.append((float(np.linalg.norm(point - origin)), point))
        if not candidates:
            raise _NoCertifiedRefuge(
                f"No certified continuous door refuge within {self.max_retreat_m:g} m for {state['name']}."
            )
        return min(candidates, key=lambda item: item[0])[1]
