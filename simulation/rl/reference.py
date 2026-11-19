# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Publish the ROS-topic measurements the DRL-VO reward and observation need.

``DRLNavEnv`` measured four quantities through ROS 1 topics: the global goal
distance (``/move_base/current_goal``), the front scan (``/cnn_data``), the
body twist (``/odom``) and the tracked crowd (``/track_ped``). In the current
environment the twist and the crowd come straight from PhysX, while the goal
distance, the pure-pursuit local goal and the front scan still need a producer:

* goal distance: planar distance to the episode goal in the ROS map frame;
* local goal: :func:`..ros2_navigation.drl.local_goal.local_goal_base` along the
  A* route, 0.5 m ahead, replanned every 0.5 s like ``configs/runtime/drl_vo.yaml``;
* front scan: :func:`..ros2_navigation.drl.scan.synthesize_front_scan` on the
  same occupancy raster the deployed controller uses (``scan_source:
  occupancy_grid``), with pedestrian disks overlaid.

One :class:`DrlVoReference` is published on the environment as
``env.drl_vo_goal`` (consumed by :mod:`.rewards` and :mod:`.observations`) and
writes ``env.drl_vo_scan`` for the same step. ``update`` is idempotent within a
step, so rewards, terminations and observations may all call it.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ..ros2_navigation.drl.local_goal import local_goal_base as _pure_pursuit_local_goal
from ..ros2_navigation.drl.scan import overlay_dynamic_obstacles, synthesize_front_scan
from ..ros2_navigation.planner import OccupancyGrid


class DrlVoReference:
    """Own the episode goal, its A* route and the synthesized front scan.

    Args:
        grid: Occupancy raster in the ROS map frame, normally the one the
            training bridge already built for the pedestrians.
        frame: Stage-world to ROS map conversion.
        robot_radius_m: Planning and clearance radius [m].
        lookahead_m: Pure-pursuit lookahead that produces the local goal [m].
        replan_period_steps: Policy steps between A* replans.
        arrival_tolerance_m: Goal ball used by the source reward and by A* [m].
        margin_m: Extra clearance added to ``robot_radius_m`` when planning [m].
        goal_ros_xy: Fixed goal in the ROS map frame [m]; sampled goals are used
            when it is ``None``.
        sample_goals: Draw a new goal at every episode reset.
        min_goal_distance_m: Smallest spawn-relative goal distance when sampling [m].
        goal_clearance_m: Clearance used to pick a free goal cell [m].
        scan_min_range_m: Smallest usable front return [m].
        scan_max_range_m: Largest usable front return [m].
        scan_margin_m: Padding around the pedestrian disks in the scan [m].
        seed: Seed for goal sampling.
    """

    def __init__(
        self,
        *,
        grid: OccupancyGrid,
        frame: object,
        robot_radius_m: float = 0.37,
        lookahead_m: float = 0.5,
        replan_period_steps: int = 10,
        arrival_tolerance_m: float = 0.3,
        margin_m: float = 0.05,
        goal_ros_xy: tuple[float, float] | None = None,
        sample_goals: bool = True,
        min_goal_distance_m: float = 1.5,
        goal_clearance_m: float = 0.5,
        scan_min_range_m: float = 0.1,
        scan_max_range_m: float = 8.0,
        scan_margin_m: float = 0.03,
        seed: int = 0,
    ) -> None:
        if not all(
            math.isfinite(value) for value in (robot_radius_m, lookahead_m, arrival_tolerance_m, margin_m)
        ):
            raise ValueError("Robot geometry and planning tolerances must be finite [m].")
        if robot_radius_m <= 0 or lookahead_m <= 0 or arrival_tolerance_m <= 0 or margin_m < 0:
            raise ValueError("robot_radius_m, lookahead_m and arrival_tolerance_m must be positive.")
        if type(replan_period_steps) is not int or replan_period_steps < 1:
            raise ValueError("replan_period_steps must be a positive integer.")
        if not (0.0 < scan_min_range_m < scan_max_range_m):
            raise ValueError("Scan ranges must satisfy 0 < scan_min_range_m < scan_max_range_m.")
        self.grid = grid
        self.frame = frame
        self.robot_radius_m = float(robot_radius_m)
        self.lookahead_m = float(lookahead_m)
        self.replan_period_steps = replan_period_steps
        self.arrival_tolerance_m = float(arrival_tolerance_m)
        self.margin_m = float(margin_m)
        self.scan_min_range_m = float(scan_min_range_m)
        self.scan_max_range_m = float(scan_max_range_m)
        self.scan_margin_m = float(scan_margin_m)
        self.min_goal_distance_m = float(min_goal_distance_m)
        self.goal_clearance_m = float(goal_clearance_m)
        self.sample_goals = bool(sample_goals) and goal_ros_xy is None
        self._random = np.random.RandomState(int(seed))
        self._goal = None if goal_ros_xy is None else np.asarray(goal_ros_xy, dtype=float)
        self._path: list[tuple[float, float]] | None = None
        self._stamp: tuple[int, int] | None = None
        self._reset_stamp: tuple[int, int] | None = None
        self.distance_m: torch.Tensor | None = None
        self.local_goal_base_xy: torch.Tensor | None = None
        self.scan_ranges: torch.Tensor | None = None
        self.planned = False

    # -- public API ------------------------------------------------------

    @property
    def goal_ros_xy(self) -> tuple[float, float]:
        """Current goal in the ROS map frame [m]."""
        if self._goal is None:
            raise RuntimeError("The DRL-VO reference has no goal yet; update or reset it first.")
        return float(self._goal[0]), float(self._goal[1])

    def set_goal_ros(self, goal_ros_xy: tuple[float, float]) -> None:
        """Replace the goal with an explicit ROS map target [m]."""
        goal = np.asarray(goal_ros_xy, dtype=float)
        if goal.shape != (2,) or not np.isfinite(goal).all():
            raise ValueError("goal_ros_xy must contain two finite coordinates [m].")
        self._goal = goal.copy()
        self._path = None
        self.planned = False
        self._stamp = None

    def update(self, env: object) -> None:
        """Refresh the goal distance, local goal and front scan for this step.

        Safe to call multiple times per step: the work is keyed on the
        environment's step counters.

        Args:
            env: Environment with ``scene[robot_name]`` and, for the scan
                overlay, ``pedestrian_bridge``.
        """
        stamp = (_step_counter(env), _episode_step(env))
        if self._stamp == stamp and self.distance_m is not None:
            return
        self._stamp = stamp
        pose = self.robot_pose_ros(env)
        if self._goal is None:
            self._goal = self._sample_goal(pose)
        if self._path is None or _episode_step(env) % self.replan_period_steps == 0:
            self._path = self._plan(pose)
        local = _pure_pursuit_local_goal(np.asarray(self._path), pose, lookahead_m=self.lookahead_m)
        device = _device(env)
        self.local_goal_base_xy = torch.as_tensor(local, dtype=torch.float32, device=device).reshape(1, 2)
        distance = math.dist(pose[:2], self.goal_ros_xy)
        self.distance_m = torch.full((int(env.num_envs),), distance, dtype=torch.float32, device=device)
        self.scan_ranges = self._scan(env, pose)
        env.drl_vo_scan = self.scan_ranges

    def reset(self, env: object, env_ids: object = None) -> None:
        """Start a new episode: optionally resample the goal and drop the route.

        Args:
            env: Environment owning the robot.
            env_ids: Ignored; the single-world training task only has one robot.
        """
        self._path = None
        self.planned = False
        self._stamp = None
        if self.sample_goals:
            self._goal = None

    def robot_pose_ros(self, env: object, robot_name: str = "robot") -> tuple[float, float, float]:
        """Return the robot ``(x, y, yaw)`` in the ROS map frame [m, rad]."""
        data = env.scene[robot_name].data
        position = _tensor(getattr(data.root_pos_w, "torch", data.root_pos_w))
        if position.shape != (1, 3) or not bool(torch.isfinite(position).all()):
            raise RuntimeError("root_pos_w must be a finite [1, 3] tensor")
        quaternion = _tensor(getattr(data.root_quat_w, "torch", data.root_quat_w))
        if quaternion.shape != (1, 4):
            raise RuntimeError("root_quat_w must be a [1, 4] tensor")
        x, y, z, w = (float(value) for value in quaternion[0].cpu())
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm < 1e-9:
            raise RuntimeError("root_quat_w is degenerate")
        yaw_stage = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        forward = self.frame.stage_vector_to_ros((math.cos(yaw_stage), math.sin(yaw_stage), 0.0))
        yaw_ros = math.atan2(forward[1], forward[0])
        point = self.frame.stage_to_ros(tuple(float(value) for value in position[0].cpu()))
        return float(point[0]), float(point[1]), float(yaw_ros)

    # -- internals -------------------------------------------------------

    def _plan(self, pose: tuple[float, float, float]) -> list[tuple[float, float]]:
        goal = self.goal_ros_xy
        try:
            path = self.grid.plan(
                pose[:2], goal, radius_m=self.robot_radius_m, arrival_tolerance_m=self.arrival_tolerance_m,
                margin_m=self.margin_m,
            )
            self.planned = True
        except ValueError:
            # Training must survive an unreachable goal: keep a straight reference so
            # the heading term still points at the target for this episode.
            self.planned = False
            path = [pose[:2], goal]
        if len(path) < 2:
            path = [pose[:2], goal]
        return [(float(point[0]), float(point[1])) for point in path]

    def _sample_goal(self, pose: tuple[float, float, float]) -> np.ndarray:
        mask = self.grid.free_mask(self.goal_clearance_m)
        rows, columns = np.nonzero(mask)
        if not len(rows):
            raise RuntimeError("The occupancy raster has no free cell for a DRL-VO goal.")
        centers = np.stack(
            (
                self.grid.origin_xy_m[0] + (columns + 0.5) * self.grid.resolution_m,
                self.grid.origin_xy_m[1] + (rows + 0.5) * self.grid.resolution_m,
            ),
            axis=1,
        )
        distance = np.linalg.norm(centers - np.asarray(pose[:2], dtype=float), axis=1)
        admissible = distance >= self.min_goal_distance_m
        if not admissible.any():
            admissible = distance >= self.arrival_tolerance_m
        if not admissible.any():
            raise RuntimeError("No free cell is far enough from the robot for a DRL-VO goal.")
        candidates = centers[admissible]
        return candidates[self._random.randint(len(candidates))].astype(float)

    def _scan(self, env: object, pose: tuple[float, float, float]) -> torch.Tensor:
        pedestrians = []
        bridge = getattr(env, "pedestrian_bridge", None)
        if bridge is not None:
            state = bridge.state
            count = int(bridge.config.max_pedestrians)
            positions = _tensor(state.positions_w).reshape(-1, 3).detach().cpu().numpy()
            radii = _tensor(state.radii).reshape(-1).detach().cpu().numpy()
            valid = _tensor(state.valid).reshape(-1).detach().cpu().numpy()
            for index in range(min(count, len(positions))):
                if not bool(valid[index]) or not np.isfinite(positions[index]).all():
                    continue
                point = self.frame.world_to_ros(tuple(float(value) for value in positions[index]))
                pedestrians.append({"position": point[:2], "radius_m": float(radii[index])})
        occupied = overlay_dynamic_obstacles(
            self.grid.occupancy,
            self.grid.resolution_m,
            self.grid.origin_xy_m,
            pedestrians,
            (),
            margin_m=self.scan_margin_m,
        )
        ranges, _ = synthesize_front_scan(
            occupied,
            self.grid.resolution_m,
            self.grid.origin_xy_m,
            pose[:2],
            pose[2],
            min_range_m=self.scan_min_range_m,
            max_range_m=self.scan_max_range_m,
        )
        return torch.as_tensor(ranges, dtype=torch.float32, device=_device(env)).reshape(1, -1)


def _tensor(value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise RuntimeError("Expected a torch tensor or a Warp array with a .torch view")
    return value


def _device(env: object) -> torch.device:
    return env.device if isinstance(getattr(env, "device", None), torch.device) else torch.device("cpu")


def _step_counter(env: object) -> int:
    return int(getattr(env, "common_step_counter", 0))


def _episode_step(env: object) -> int:
    lengths = getattr(env, "episode_length_buf", None)
    if isinstance(lengths, torch.Tensor) and lengths.numel():
        return int(lengths[0])
    return 0
