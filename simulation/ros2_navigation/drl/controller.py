# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Frozen DRL-VO policy lifecycle, observation scheduling and /cmd_vel output."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ...configuration.drl import DrlPolicySettings
from ..frames import StageFrame
from ..planner import OccupancyGrid
from .clock import RateClock
from .guards import apply_safety_guards
from .local_goal import local_goal_base
from .observation import DrlObservationAdapter
from .package import DrlPackage
from .policy import FrozenDrlVoPolicy
from .robot import PhysxRobotObserver, RobotState
from .scan import overlay_dynamic_obstacles, synthesize_front_scan


def _json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.floating,)):
        return _json_safe(float(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    vector = end - start
    length_squared = float(np.dot(vector, vector))
    if length_squared <= 1e-18:
        return float(np.linalg.norm(point - start))
    fraction = max(0.0, min(1.0, float(np.dot(point - start, vector)) / length_squared))
    return float(np.linalg.norm(point - (start + fraction * vector)))


def _path_deviation(path: list[tuple[float, float]], point: tuple[float, float]) -> float:
    if len(path) < 2:
        return math.dist(path[0], point) if path else math.inf
    target = np.asarray(point, dtype=float)
    return min(
        _point_segment_distance(target, np.asarray(start, dtype=float), np.asarray(end, dtype=float))
        for start, end in zip(path, path[1:])
    )


class _FirstSubgoalAlignment:
    """Latched first-subgoal in-place turn before policy inference starts."""

    def __init__(self, settings: DrlPolicySettings) -> None:
        self.tolerance = settings.alignment_tolerance_rad
        self.max_w = settings.alignment_max_w_rad_s
        self.kp = settings.alignment_kp
        self.settle_seconds = settings.alignment_settle_s
        self.stopped_w = settings.alignment_stopped_w_rad_s
        self.reset()

    def reset(self) -> None:
        self.target_world: tuple[float, float] | None = None
        self.settle_since: float | None = None
        self.done = False

    def start(self, local_xy: np.ndarray, pose: tuple[float, float, float]) -> None:
        x, y, yaw = (float(value) for value in pose)
        lx, ly = (float(value) for value in local_xy)
        if not all(math.isfinite(value) for value in (x, y, yaw, lx, ly)):
            raise ValueError("Non-finite first subgoal or robot pose.")
        self.target_world = (x + math.cos(yaw) * lx - math.sin(yaw) * ly, y + math.sin(yaw) * lx + math.cos(yaw) * ly)

    def command(self, pose: tuple[float, float, float], actual_w: float, now_s: float) -> tuple[float, float]:
        if self.target_world is None:
            raise RuntimeError("First subgoal has not been published.")
        x, y, yaw = (float(value) for value in pose)
        bearing = math.atan2(self.target_world[1] - y, self.target_world[0] - x) - yaw
        error = math.atan2(math.sin(bearing), math.cos(bearing))
        aligned = abs(error) <= self.tolerance
        angular = 0.0 if aligned else max(-self.max_w, min(self.max_w, self.kp * error))
        if aligned and abs(actual_w) <= self.stopped_w:
            if self.settle_since is None:
                self.settle_since = now_s
            self.done = now_s - self.settle_since >= self.settle_seconds - 1e-9
        else:
            self.settle_since = None
        return 0.0, angular


class DrlVoController:
    """Own one frozen policy, its observation pipeline and optional /cmd_vel publisher."""

    def __init__(
        self,
        settings: DrlPolicySettings,
        *,
        grid: OccupancyGrid,
        frame: StageFrame,
        transport,
        output: Path,
        robots: tuple,
        stage=None,
        navigation=None,
        observer: PhysxRobotObserver | None = None,
        policy: FrozenDrlVoPolicy | None = None,
        adapter: DrlObservationAdapter | None = None,
        publisher=None,
    ) -> None:
        if not settings.enabled:
            raise ValueError("DRL policy is disabled.")
        if settings.goal_xy_m is None:
            raise ValueError("DRL policy requires goal_xy_m.")
        self.settings = settings
        self.grid = grid
        self.frame = frame
        self.transport = transport
        self.output = Path(output)
        self.robots = robots
        self.stage = stage
        self.navigation = navigation
        self._control_hz = float(getattr(navigation, "control_hz", 60.0))
        self._map_clearance = float(getattr(navigation, "map_clearance_m", 0.05))
        self._scan_clock = RateClock(settings.scan_hz, self._control_hz)
        self._policy_clock = RateClock(settings.policy_hz, self._control_hz)
        self._observer = observer
        self._policy = policy
        self._adapter = adapter
        self._publisher = publisher
        self._package: DrlPackage | None = None
        self._goal = tuple(float(value) for value in settings.goal_xy_m)
        self._path: list[tuple[float, float]] | None = None
        self._next_replan = 0.0
        self._path_ok_time = 0.0
        self._start_time: float | None = None
        self._settle_since: float | None = None
        self._alignment = _FirstSubgoalAlignment(settings)
        self._phase = "WARMUP"
        self._paused = False
        self._policy_steps = 0
        self._invalid = 0
        self._guard_stopped = False
        self._zero_reason: str | None = None
        self._last_robot: RobotState | None = None
        self._last_scan = np.zeros(720, dtype=np.float32)
        self._last_scan_stats: dict[str, object] = {}
        self._last_command: dict[str, object] | None = None
        self._policy_log = None
        self._scan_log = None
        self._observation_dir = self.output / "drl_observations"

    def start(self) -> None:
        """Load the model package and validate the initial goal/path before playback."""
        if self._policy is None:
            self._package = DrlPackage.from_settings(self.settings)
            self._adapter = self._adapter or DrlObservationAdapter(self._package.load_observation_api())
            self._policy = FrozenDrlVoPolicy(
                self._package,
                device=self.settings.device,
                expected_sha256=self.settings.expected_sha256,
            )
        if self._adapter is None:
            if self._package is None:
                raise ValueError("DRL observation adapter requires the model package.")
            self._adapter = DrlObservationAdapter(self._package.load_observation_api())
        if self._observer is None:
            self._observer = PhysxRobotObserver(self._resolve_rigid_body_path(), self.frame)
        if self._publisher is None and not self.settings.dry_run:
            from geometry_msgs.msg import Twist

            self._publisher = self.transport.create_publisher(Twist, "/cmd_vel", 10)
        try:
            initial = self._observer.read(0.0)
        except RuntimeError:
            initial = self._initial_robot_state()
        self._last_robot = initial
        self._path = self._plan(initial.pose_xyyaw)
        self._path_ok_time = 0.0
        self._next_replan = self.settings.replan_period_s
        self._policy_log = (self.output / "drl_policy.jsonl").open("w")
        self._scan_log = (self.output / "drl_scan.jsonl").open("w")
        if self.settings.save_observations:
            self._observation_dir.mkdir(exist_ok=True)
        self._write_manifest(initial)

    def _resolve_rigid_body_path(self) -> str:
        if self.settings.robot_rigid_body_path is not None:
            return self.settings.robot_rigid_body_path
        if self.stage is None or not self.robots:
            raise ValueError("DRL policy needs one robot and an open stage to resolve its rigid body.")
        from pxr import Usd, UsdPhysics

        root = self.stage.GetPrimAtPath(self.robots[0].prim_path)
        if not root:
            raise ValueError(f"Robot prim not found: {self.robots[0].prim_path}")
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            if UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is not False:
                return str(prim.GetPath())
        raise ValueError(f"No enabled rigid body under {self.robots[0].prim_path}")

    def _initial_robot_state(self) -> RobotState:
        """Use the configured spawn pose before PhysX exposes its first live pose."""
        if not self.robots or self.robots[0].position is None:
            raise RuntimeError("PhysX has no live robot pose and the robot has no configured position.")
        position = self.frame.stage_to_ros(tuple(float(value) for value in self.robots[0].position))
        yaw = math.radians(float(self.robots[0].yaw_degrees or 0.0)) - self.frame.yaw_rad
        return RobotState(
            position_ros_m=(float(position[0]), float(position[1]), float(position[2])),
            yaw_rad=float(yaw),
            pose_xyyaw=(float(position[0]), float(position[1]), float(yaw)),
            linear_velocity_ros_mps=(0.0, 0.0, 0.0),
            yaw_rate_radps=0.0,
            forward_speed_mps=0.0,
        )

    def _plan(self, pose: tuple[float, float, float]) -> list[tuple[float, float]]:
        path = self.grid.plan(
            pose[:2],
            self._goal,
            radius_m=self.settings.robot_footprint_radius_m,
            arrival_tolerance_m=self.settings.goal_tolerance_m,
            margin_m=self._map_clearance,
        )
        return [(float(point[0]), float(point[1])) for point in path]

    def _publish(self, linear: float, angular: float) -> None:
        if self.settings.dry_run or self._publisher is None:
            return
        from geometry_msgs.msg import Twist

        message = Twist()
        message.linear.x = float(linear)
        message.angular.z = float(angular)
        self._publisher.publish(message)

    def _zero(self, reason: str) -> None:
        self._zero_reason = reason
        self._publish(0.0, 0.0)

    def _write(self, stream, row: dict[str, object]) -> None:
        stream.write(json.dumps(_json_safe(row), ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()

    def _write_manifest(self, initial: RobotState) -> None:
        manifest = {
            "schema": "pedstrain.drl_vo.frozen.v1",
            "package_root": self.settings.package_root,
            "checkpoint": self._package.checkpoint if self._package is not None else self.settings.checkpoint,
            "checkpoint_sha256": getattr(self._policy, "checkpoint_sha256", None),
            "device": self.settings.device,
            "parameter_count": getattr(self._policy, "parameter_count", None),
            "robot_rigid_body_path": getattr(self._observer, "rigid_body_path", None),
            "start_pose_xyyaw_m": initial.pose_xyyaw,
            "goal_xy_m": self._goal,
            "initial_path": self._path,
            "lookahead_m": self.settings.lookahead_m,
            "scan_source": self.settings.scan_source,
            "guard_mode": self.settings.guard_mode,
            "alignment_enabled": self.settings.alignment_enabled,
            "dry_run": self.settings.dry_run,
            "limits": {
                "max_linear_m_s": self.settings.max_linear_m_s,
                "max_angular_rad_s": self.settings.max_angular_rad_s,
                "goal_tolerance_m": self.settings.goal_tolerance_m,
            },
        }
        (self.output / "drl_manifest.json").write_text(
            json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )

    def step(
        self,
        now_s: float,
        dt_s: float,
        states: list[dict[str, object]],
        obstacles: list[dict[str, object]] = (),
    ) -> None:
        """Run one control tick after the post-physics pedestrian snapshot."""
        if self._paused or self._phase in ("GOAL_REACHED", "GOAL_TIMEOUT", "FAILED"):
            return
        now = float(now_s)
        if self._start_time is None:
            self._start_time = now
        scan_due = self._scan_clock.due()
        policy_due = self._policy_clock.due()
        self._scan_clock.commit()
        self._policy_clock.commit()
        robot = self._observer.read(now)
        self._last_robot = robot
        if scan_due:
            occupancy = overlay_dynamic_obstacles(
                self.grid.occupancy,
                self.grid.resolution_m,
                self.grid.origin_xy_m,
                states,
                obstacles,
                exclude_prefix=self.robots[0].prim_path if self.robots else None,
            )
            scan, stats = synthesize_front_scan(
                occupancy,
                self.grid.resolution_m,
                self.grid.origin_xy_m,
                robot.pose_xyyaw[:2],
                robot.pose_xyyaw[2],
                min_range_m=self.settings.scan_min_range_m,
                max_range_m=self.settings.scan_max_range_m,
                sample_step_m=self.settings.scan_sample_step_m,
            )
            self._last_scan = scan
            self._last_scan_stats = stats
            self._adapter.append_scan(scan, int(round(now * 1e9)))
            if self._scan_log is not None:
                self._write(
                    self._scan_log,
                    {
                        "time_s": now,
                        "robot_pose_xyyaw_m": robot.pose_xyyaw,
                        **stats,
                    },
                )
        self._adapter.sample_history(int(round(now * 1e9)))
        if not self._adapter.history_ready:
            self._phase = "WARMUP"
            self._zero("sensor_warmup")
            return
        distance = math.dist(robot.pose_xyyaw[:2], self._goal)
        if distance <= self.settings.goal_tolerance_m:
            if self._settle_since is None:
                self._settle_since = now
            if now - self._settle_since >= self.settings.goal_settle_s:
                self._phase = "GOAL_REACHED"
                self._zero("goal_reached")
                return
            self._phase = "GOAL_SETTLE"
            self._zero("goal_settle")
            return
        self._settle_since = None
        if self.settings.goal_timeout_s and now - self._start_time >= self.settings.goal_timeout_s:
            self._phase = "GOAL_TIMEOUT"
            self._zero("goal_timeout")
            return
        if self._path is None or now >= self._next_replan or _path_deviation(self._path, robot.pose_xyyaw[:2]) > self.settings.path_deviation_m:
            try:
                self._path = self._plan(robot.pose_xyyaw)
                self._path_ok_time = now
                self._next_replan = now + self.settings.replan_period_s
            except ValueError:
                self._next_replan = now + self.settings.replan_period_s
                if now - self._path_ok_time > self.settings.replan_period_s:
                    self._phase = "PATH_UNAVAILABLE"
                    self._zero("path_unavailable")
                    return
        if self._path is None:
            self._phase = "PATH_UNAVAILABLE"
            self._zero("path_unavailable")
            return
        local_goal = local_goal_base(self._path, robot.pose_xyyaw, self.settings.lookahead_m)
        if self.settings.alignment_enabled and not self._alignment.done and not self.settings.dry_run:
            if self._alignment.target_world is None:
                self._alignment.start(local_goal, robot.pose_xyyaw)
            _, angular = self._alignment.command(robot.pose_xyyaw, robot.yaw_rate_radps, now)
            self._phase = "ALIGNING"
            self._publish(0.0, angular)
            return
        if not policy_due:
            return
        try:
            observation, observation_meta = self._adapter.build_observation(states, robot.pose_xyyaw, local_goal)
            raw, inference_meta = self._policy.infer(observation)
        except (ValueError, RuntimeError) as error:
            self._invalid += 1
            self._phase = "FAILED" if self._invalid >= self.settings.max_invalid_observations else "SAFETY_STOP"
            self._zero(f"invalid_observation:{type(error).__name__}")
            return
        clipped = np.clip(raw, -1.0, 1.0)
        desired_linear = float((clipped[0] + 1.0) * 0.25)
        desired_angular = float(clipped[1] * 2.0)
        guard = apply_safety_guards(
            commanded_linear_m_s=desired_linear,
            commanded_angular_rad_s=desired_angular,
            robot=robot,
            states=states,
            scan=self._last_scan,
            grid=self.grid,
            settings=self.settings,
            stopped=self._guard_stopped,
            map_clearance_m=self._map_clearance,
        )
        if self.settings.guard_mode == "enforce":
            self._guard_stopped = guard.override
            if self.settings.dry_run:
                published = (0.0, 0.0)
                self._phase = "DRY_RUN"
                self._zero_reason = "dry_run"
            elif guard.override and desired_linear > 0:
                prefer_clearance = "pedestrian_clearance" in (guard.reason or "")
                if prefer_clearance and self._rear_clear(robot):
                    published = (-self.settings.escape_reverse_m_s, 0.0)
                    self._phase = "RECOVERY_BACKOFF"
                else:
                    published = (
                        0.0,
                        self._escape_angular(local_goal, self._last_scan, prefer_clearance=prefer_clearance),
                    )
                    self._phase = "RECOVERY_TURN"
                self._publish(*published)
            else:
                published = (0.0, 0.0) if guard.override else (guard.linear_m_s, guard.angular_rad_s)
                self._publish(*published)
                self._phase = "SAFETY_STOP" if guard.override else "DRIVING"
        else:
            # observe/off: the policy owns the published command; the guard is logged only.
            self._guard_stopped = False
            self._zero_reason = None
            if self.settings.dry_run:
                published = (0.0, 0.0)
                self._phase = "DRY_RUN"
                self._zero_reason = "dry_run"
            else:
                published = (desired_linear, desired_angular)
                self._publish(*published)
                self._phase = "DRIVING"
        self._policy_steps += 1
        self._last_command = {
            "local_goal_base_xy_m": [float(local_goal[0]), float(local_goal[1])],
            "raw_action": raw,
            "clipped_action": clipped,
            "desired_v_m_s": desired_linear,
            "desired_w_rad_s": desired_angular,
            "published_v_m_s": published[0],
            "published_w_rad_s": published[1],
            "guard": {
                "mode": self.settings.guard_mode,
                "applied": self.settings.guard_mode == "enforce" and guard.override,
                "override": guard.override,
                "reason": guard.reason,
                **guard.details,
            },
            "inference": inference_meta,
            "observation": observation_meta,
            "scan": self._last_scan_stats,
        }
        if self._policy_log is not None:
            self._write(
                self._policy_log,
                {
                    "time_s": now,
                    "robot_pose_xyyaw_m": robot.pose_xyyaw,
                    "goal_distance_m": distance,
                    "goal_xy_m": self._goal,
                    "path_points": len(self._path),
                    "phase": self._phase,
                    **self._last_command,
                },
            )
        if self.settings.save_observations:
            np.save(self._observation_dir / f"observation_{self._policy_steps:06d}.npy", observation)

    def _rear_clear(self, robot: RobotState) -> bool:
        heading = np.array([math.cos(robot.yaw_rad), math.sin(robot.yaw_rad)], dtype=float)
        start = np.asarray(robot.pose_xyyaw[:2], dtype=float)
        end = start - heading * (self.settings.forward_guard_distance_m + self.settings.robot_length_m / 2)
        clearance = self.settings.robot_width_m / 2 + self._map_clearance
        return bool(self.grid.segment_clear(tuple(start), tuple(end), clearance))

    def _escape_angular(
        self, local_goal: np.ndarray, scan: np.ndarray, *, prefer_clearance: bool = False
    ) -> float:
        """Turn in place toward the local goal, or toward the side with more clearance."""
        limit = min(self.settings.escape_angular_rad_s, self.settings.max_angular_rad_s)
        bearing = math.atan2(float(local_goal[1]), float(local_goal[0]))
        if not prefer_clearance and abs(bearing) > math.radians(15.0):
            return math.copysign(limit, bearing)
        values = np.asarray(scan, dtype=float)
        left = values[400:700]
        right = values[20:320]
        left_open = float(left[left > 0].max()) if np.any(left > 0) else self.settings.scan_max_range_m
        right_open = float(right[right > 0].max()) if np.any(right > 0) else self.settings.scan_max_range_m
        return limit if left_open >= right_open else -limit

    def pause(self) -> None:
        self._paused = True
        self._phase = "PAUSED"
        self._zero("paused")

    def resume(self) -> None:
        self._paused = False

    def close(self) -> None:
        self._publish(0.0, 0.0)
        for stream in (self._policy_log, self._scan_log):
            if stream is not None:
                stream.close()
        self._policy_log = self._scan_log = None
        if self._policy is not None:
            self._policy.close()
        self._publisher = None

    def status(self) -> dict[str, object]:
        robot = self._last_robot
        distance = math.dist(robot.pose_xyyaw[:2], self._goal) if robot is not None else None
        path_length = None
        if self._path and len(self._path) > 1:
            path_length = sum(math.dist(a, b) for a, b in zip(self._path, self._path[1:]))
        return _json_safe({
            "enabled": True,
            "owner": "FROZEN_DRLVO",
            "phase": self._phase,
            "guard_mode": self.settings.guard_mode,
            "alignment_enabled": self.settings.alignment_enabled,
            "dry_run": self.settings.dry_run,
            "device": self.settings.device,
            "package_root": str(self.settings.package_root) if self.settings.package_root else None,
            "checkpoint_sha256": getattr(self._policy, "checkpoint_sha256", None),
            "parameter_count": getattr(self._policy, "parameter_count", None),
            "policy_steps": self._policy_steps,
            "robot_pose_xyyaw_m": robot.pose_xyyaw if robot is not None else None,
            "goal": {
                "xy_m": list(self._goal),
                "distance_m": distance,
                "tolerance_m": self.settings.goal_tolerance_m,
                "reached": self._phase == "GOAL_REACHED",
            },
            "path": {"points": len(self._path) if self._path else 0, "length_m": path_length},
            "local_goal_base_xy_m": (
                self._last_command.get("local_goal_base_xy_m") if self._last_command else None
            ),
            "last_action": self._last_command,
            "scan": {
                "ready": self._adapter.history_ready if self._adapter is not None else False,
                **self._last_scan_stats,
            },
            "zero_reason": self._zero_reason,
            "invalid_observations": self._invalid,
            "outputs": {
                "manifest": str(self.output / "drl_manifest.json"),
                "policy_log": str(self.output / "drl_policy.jsonl"),
                "scan_log": str(self.output / "drl_scan.jsonl"),
            },
        })

    def trace_state(self) -> dict[str, object]:
        return _json_safe(
            {
                "phase": self._phase,
                "policy_steps": self._policy_steps,
                "guard_stopped": self._guard_stopped,
                "zero_reason": self._zero_reason,
                "last_action": self._last_command,
            }
        )
