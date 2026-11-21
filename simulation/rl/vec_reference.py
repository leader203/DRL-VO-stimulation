# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DRL-VO measurements for N cloned environments on one static map.

:class:`.reference.DrlVoReference` served a single robot on a single composed
stage: it read one pose, planned one route and cast one scan. The multi-environment
task spawns *identical copies* of the same room, so the map the deployed controller
would build is built once — from the first clone — and every environment is
measured against it in the room's own coordinates:

* ``env`` *i* lives at ``scene.env_origins[i]``; its pose in room coordinates is
  ``root_pos_w - env_origins[i]``, which is the same frame for all clones;
* the occupancy raster, the clearance field, the A* routes and the 720-beam scan
  are therefore all in that one room frame, and nothing has to be rebuilt per
  environment;
* the goal is sampled per environment, from free cells that are clear enough for
  the robot footprint and far enough from the current pose.

:class:`ClearanceMap` adds the one measurement the source never had: a 360°
distance to the nearest static obstacle from a Euclidean distance transform of the
raster. :mod:`.rewards` scores collisions from the 90° front scan, exactly like
``DRLNavEnv``, but a success criterion must not miss a wall the robot backed into,
so the episode verdict uses this omnidirectional clearance instead.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ..ros2_navigation.drl.local_goal import local_goal_base as _pure_pursuit_local_goal
from ..ros2_navigation.drl.scan import front_beam_angles, synthesize_front_scan

SCAN_BEAMS = 720
SCAN_BLOCK = 64


def _synthesize_scan_batch(
    occupancy: torch.Tensor,
    poses: np.ndarray,
    *,
    resolution_m: float,
    origin_xy_m: tuple[float, float],
    min_range_m: float,
    max_range_m: float,
    block: int = SCAN_BLOCK,
) -> torch.Tensor:
    """Cast all 720 beams of every robot in a few tensor passes.

    Calling the per-robot ray caster costs roughly 17 ms each and dominated a
    512-robot policy step.  The rays of different robots never interact, so they
    can be cast together; the arithmetic below is the same float64
    floor-and-gather the numpy path performs, which keeps the result
    bit-identical while moving the work onto the GPU.

    Args:
        occupancy: Boolean occupancy, ``[height, width]``, on any device.
        poses: ``(x, y, yaw)`` per robot in the ROS map frame, ``[num_envs, 3]``.
        resolution_m: Cell size of the raster [m].
        origin_xy_m: Map-frame position of cell ``(0, 0)`` [m].
        min_range_m: Smallest usable return [m].
        max_range_m: Largest usable return [m].
        block: Robots per tensor pass, bounding peak memory.

    Returns:
        ``[num_envs, 720]`` float32 ranges on the occupancy device; zero means
        no return.
    """
    device = occupancy.device
    step = resolution_m / 2.0
    # Sample distances are built with numpy.arange so they match the per-robot
    # path exactly; a torch.arange would drift in the last bits.
    distances = np.arange(min_range_m, max_range_m + step * 0.5, step, dtype=np.float64)
    samples = torch.as_tensor(distances, dtype=torch.float64, device=device)
    # A full 360-degree sweep at the same 720 beams and the same 0.5 degree
    # resolution: the episode ends on omnidirectional clearance, so a
    # front-only scan would hide the wall the robot reversed into.
    beams = torch.as_tensor(
        np.asarray(
            front_beam_angles(count=SCAN_BEAMS, angle_min=-math.pi, angle_increment=2.0 * math.pi / SCAN_BEAMS)
        ),
        dtype=torch.float64,
        device=device,
    )
    height, width = occupancy.shape
    count = int(poses.shape[0])
    result = torch.zeros((count, beams.numel()), dtype=torch.float32, device=device)

    for start in range(0, count, block):
        chunk = torch.as_tensor(poses[start : start + block], dtype=torch.float64, device=device)
        angles = chunk[:, 2:3] + beams[None, :]
        x = chunk[:, 0:1, None] + torch.cos(angles)[:, :, None] * samples[None, None, :]
        y = chunk[:, 1:2, None] + torch.sin(angles)[:, :, None] * samples[None, None, :]
        rows = torch.floor((y - float(origin_xy_m[1])) / resolution_m).long()
        columns = torch.floor((x - float(origin_xy_m[0])) / resolution_m).long()
        inside = (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width)
        rows = rows.clamp(0, height - 1)
        columns = columns.clamp(0, width - 1)
        blocked = inside & occupancy[rows, columns]
        hit = blocked.any(dim=2)
        first = blocked.to(torch.int8).argmax(dim=2)
        ranges = torch.where(hit, samples[first], torch.zeros((), dtype=torch.float64, device=device))
        ranges = ranges.to(torch.float32)
        # A floor on the near sample can land below min_range, and a beam whose
        # first hit is beyond max_range is a miss; both read as zero.
        ranges[(ranges < min_range_m) | (ranges > max_range_m)] = 0.0
        result[start : start + block] = ranges
    return result


class ClearanceMap:
    """Distance from any map cell to the nearest occupied cell [m].

    The transform is computed once per run: the static map never changes, so a
    lookup replaces a ray cast and stays exact for every heading.
    """

    def __init__(self, grid: object) -> None:
        from scipy import ndimage

        occupancy = np.asarray(grid.occupancy)
        if occupancy.ndim != 2:
            raise ValueError("The occupancy raster must be two-dimensional.")
        self.grid = grid
        self.resolution_m = float(grid.resolution_m)
        self.origin_xy_m = (float(grid.origin_xy_m[0]), float(grid.origin_xy_m[1]))
        self.meters = (ndimage.distance_transform_edt(occupancy != 100) * self.resolution_m).astype(np.float32)

    def clearance_m(self, xy: np.ndarray) -> np.ndarray:
        """Return the static clearance at each planar point [m].

        Args:
            xy: Points in the ROS map frame [m], shape ``[count, 2]``.

        Returns:
            Clearance per point [m]; a point outside the raster is reported as 0.

        Raises:
            ValueError: The input is not a finite ``[count, 2]`` array.
        """
        points = np.asarray(xy, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("clearance_m expects a finite [count, 2] array in the ROS map frame.")
        columns = np.floor((points[:, 0] - self.origin_xy_m[0]) / self.resolution_m).astype(np.int64)
        rows = np.floor((points[:, 1] - self.origin_xy_m[1]) / self.resolution_m).astype(np.int64)
        height, width = self.meters.shape
        inside = (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width)
        values = np.zeros(len(points), dtype=np.float32)
        values[inside] = self.meters[rows[inside], columns[inside]]
        return values


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Boolean mask of the largest 8-connected component, or ``mask`` if untagged."""
    try:
        from scipy import ndimage
    except ImportError:  # pragma: no cover - scipy ships with the training env
        return np.ones_like(mask, dtype=bool)
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if count <= 1:
        return np.ones_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(sizes.argmax())


def _downsample_grid(grid: object, factor: int):
    """Return a conservatively coarsened copy of ``grid`` for A*.

    Planning is the most expensive part of a step: one query on the 0.02 m raster
    costs about 124 ms, and three replans per policy step for sixteen robots adds
    up to a third of the rollout. The route only feeds a 0.5 m pure-pursuit
    lookahead on a 0.5 m/s robot, so a 0.06 m raster is indistinguishable in the
    reference while being roughly six times cheaper.

    A coarse cell is occupied if *any* of its children is, which keeps the plan on
    the safe side of every wall and every unknown cell.

    Args:
        grid: The fine :class:`OccupancyGrid` the reference was built from.
        factor: Linear downsampling factor; ``1`` returns ``grid`` unchanged.
    """
    occupancy = np.asarray(grid.occupancy)
    if factor <= 1 or occupancy.ndim != 2:
        return grid
    height = (occupancy.shape[0] // factor) * factor
    width = (occupancy.shape[1] // factor) * factor
    if height == 0 or width == 0:
        return grid
    blocks = occupancy[:height, :width].reshape(height // factor, factor, width // factor, factor)
    coarse = np.where((blocks != 0).any(axis=(1, 3)), np.int8(100), np.int8(0))
    return type(grid)(
        coarse,
        float(grid.resolution_m) * factor,
        grid.origin_xy_m,
        goal_projection_tolerance_m=getattr(grid, "goal_projection_tolerance_m", None),
    )


class VecDrlVoReference:
    """Publish goal distance, local goal, front scan and clearance for N robots.

    The object is attached as ``env.drl_vo_goal`` and exposes the same interface
    the reward, termination and observation terms already consume
    (``update``, ``reset``, ``distance_m``, ``local_goal_base_xy``,
    ``env.drl_vo_scan``), plus ``clearance_m`` for the episode verdict.

    Args:
        grid: Occupancy raster of one room, in the ROS map frame.
        frame: Stage-world to ROS map conversion of the composed stage.
        num_envs: Number of cloned environments.
        robot_radius_m: Footprint radius used by the planner [m].
        lookahead_m: Pure-pursuit lookahead for the local goal [m].
        replan_period_steps: Policy steps between A* replans.
        arrival_tolerance_m: Goal ball radius [m].
        margin_m: Extra planning clearance [m].
        min_goal_distance_m: Smallest sampled goal distance from the pose [m].
        goal_clearance_m: Static clearance a goal cell must have [m].
        start_clearance_m: Static clearance a sampled start pose must have [m].
        scan_min_range_m: Smallest usable front return [m].
        scan_max_range_m: Largest usable front return [m].
        scan_margin_m: Padding around pedestrian disks in the scan [m].
        seed: Seed for goal and start sampling.
        device: Device of the published tensors.
        map_offset: Stage-frame origin of the room the raster was built from [m].
            The occupancy raster is cached across runs, and the clone layout (and
            therefore ``scene.env_origins[0]``) depends on ``num_envs``, so the
            offset must travel with the raster instead of being re-derived from
            whatever layout the current run happens to use.
    """

    def __init__(
        self,
        *,
        grid: object,
        frame: object,
        num_envs: int,
        robot_radius_m: float = 0.37,
        lookahead_m: float = 0.5,
        replan_period_steps: int = 10,
        arrival_tolerance_m: float = 0.3,
        margin_m: float = 0.05,
        min_goal_distance_m: float = 1.5,
        goal_clearance_m: float = 0.5,
        start_clearance_m: float = 0.6,
        scan_min_range_m: float = 0.1,
        scan_max_range_m: float = 8.0,
        scan_margin_m: float = 0.03,
        seed: int = 0,
        device: torch.device | str = "cpu",
        map_offset: tuple[float, float] | None = None,
        plan_downsample: int = 3,
    ) -> None:
        if int(num_envs) < 1:
            raise ValueError("num_envs must be positive")
        if not (0.0 < scan_min_range_m < scan_max_range_m):
            raise ValueError("Scan ranges must satisfy 0 < scan_min_range_m < scan_max_range_m")
        if robot_radius_m <= 0 or lookahead_m <= 0 or arrival_tolerance_m <= 0:
            raise ValueError("Robot radius, lookahead and arrival tolerance must be positive [m]")
        if type(replan_period_steps) is not int or replan_period_steps < 1:
            raise ValueError("replan_period_steps must be a positive integer")

        self.grid = grid
        self.frame = frame
        self.plan_grid = _downsample_grid(grid, int(plan_downsample))
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.robot_radius_m = float(robot_radius_m)
        self.lookahead_m = float(lookahead_m)
        self.replan_period_steps = int(replan_period_steps)
        self.arrival_tolerance_m = float(arrival_tolerance_m)
        self.margin_m = float(margin_m)
        self.min_goal_distance_m = float(min_goal_distance_m)
        self.goal_clearance_m = float(goal_clearance_m)
        self.start_clearance_m = float(start_clearance_m)
        self.scan_min_range_m = float(scan_min_range_m)
        self.scan_max_range_m = float(scan_max_range_m)
        self.scan_margin_m = float(scan_margin_m)

        self.clearance = ClearanceMap(grid)
        self._random = np.random.RandomState(int(seed))
        self._goal = np.full((self.num_envs, 2), np.nan, dtype=np.float64)
        self._paths: list[list[tuple[float, float]] | None] = [None] * self.num_envs
        self._stamp: tuple[int, int] | None = None
        self._map_offset = (
            np.asarray(map_offset, dtype=np.float64) if map_offset is not None else np.zeros(2, dtype=np.float64)
        )
        self._offset_known = map_offset is not None

        self.distance_m = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.local_goal_base_xy = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.scan_ranges = torch.zeros((self.num_envs, SCAN_BEAMS), dtype=torch.float32, device=self.device)
        self.clearance_m = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.pose_ros = np.zeros((self.num_envs, 3), dtype=np.float64)
        self._free_centers: tuple[np.ndarray, np.ndarray] | None = None
        # Occupancy as a boolean tensor, built once: the raster is static, and
        # rebuilding it per step would cost a 313x593 host-to-device copy.
        self._scan_occupancy: torch.Tensor | None = None

    # -- geometry helpers ------------------------------------------------

    def _free_cells(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the centers of free cells and their static clearance [m].

        Only the largest connected component a 0.74 m body can traverse is kept.
        Goals and starts used to be drawn independently from every cell that
        passed a clearance test, and the washroom is not one pocket: the four
        toilet cubicles are separate components at 0.45 m clearance (their
        doors are narrower than the body), and at 0.60 m the vestibule by the
        door separates from the main room (the gap beside the washstand is
        0.67 m). A goal in one pocket with a start in another is unwinnable: A*
        returns nothing, the subgoal falls back to the straight line, and the
        robot drives into a wall for six seconds.
        """
        if self._free_centers is None:
            mask = self.grid.free_mask(self.goal_clearance_m)
            reachable = _largest_component(self.grid.free_mask(self.robot_radius_m + self.margin_m))
            mask = np.logical_and(mask, reachable)
            if not mask.any():
                raise RuntimeError("The largest traversable component has no admissible goal cell")
            rows, columns = np.nonzero(mask)
            if not len(rows):
                raise RuntimeError("The occupancy raster has no free cell for a DRL-VO goal")
            centers = np.stack(
                (
                    self.grid.origin_xy_m[0] + (columns + 0.5) * self.grid.resolution_m,
                    self.grid.origin_xy_m[1] + (rows + 0.5) * self.grid.resolution_m,
                ),
                axis=1,
            )
            self._free_centers = (centers, self.clearance.clearance_m(centers))
        return self._free_centers

    def _room_offset(self, env: object) -> np.ndarray:
        """Stage-frame origin of the room the map was rasterized from [m]."""
        if not self._offset_known:
            origins = getattr(getattr(env, "scene", None), "env_origins", None)
            if isinstance(origins, torch.Tensor) and origins.numel() >= 2:
                values = origins if origins.dtype == torch.float32 else origins.float()
                self._map_offset = values[0, :2].detach().cpu().numpy().astype(np.float64)
            self._offset_known = True
        return self._map_offset

    def poses_ros(self, env: object, robot_name: str = "robot") -> np.ndarray:
        """Return ``(x, y, yaw)`` per environment in the ROS map frame.

        Args:
            env: Environment with ``scene[robot_name]`` and ``scene.env_origins``.
            robot_name: Scene key of the robot.

        Returns:
            Array of shape ``[num_envs, 3]`` [m, m, rad].
        """
        asset = env.scene[robot_name]
        data = asset.data
        position = _torch(data.root_pos_w)
        quaternion = _torch(data.root_quat_w)
        origins = _torch(env.scene.env_origins)
        local = (position[:, :2] - origins[:, :2]).detach().cpu().numpy().astype(np.float64)
        offset = self._room_offset(env)
        yaw = _yaw_from_quaternion(quaternion).detach().cpu().numpy().astype(np.float64)
        poses = np.zeros((len(local), 3), dtype=np.float64)
        for index in range(len(local)):
            point = self.frame.stage_to_ros((local[index, 0] + offset[0], local[index, 1] + offset[1], 0.0))
            forward = self.frame.stage_vector_to_ros((math.cos(yaw[index]), math.sin(yaw[index]), 0.0))
            poses[index] = (point[0], point[1], math.atan2(forward[1], forward[0]))
        return poses

    def prepare(self, env: object) -> None:
        """Learn where the rasterized room sits in the stage, so sampling matches it.

        Args:
            env: Environment with ``scene.env_origins``.
        """
        self._room_offset(env)

    def local_from_ros(self, x: float, y: float) -> tuple[float, float]:
        """Convert a ROS map point to the clone-local stage coordinates [m]."""
        point = self.frame.ros_to_stage((float(x), float(y), 0.0))
        offset = self._map_offset
        return float(point[0] - offset[0]), float(point[1] - offset[1])

    def local_to_ros(self, x: float, y: float) -> tuple[float, float]:
        """Convert clone-local stage coordinates to the ROS map frame [m]."""
        offset = self._map_offset
        point = self.frame.stage_to_ros((float(x) + offset[0], float(y) + offset[1], 0.0))
        return float(point[0]), float(point[1])

    # -- sampling --------------------------------------------------------

    def sample_goals(self, poses: np.ndarray, env_ids: np.ndarray) -> None:
        """Draw one admissible goal per selected environment.

        Args:
            poses: Current ``(x, y, yaw)`` in the ROS map frame, ``[num_envs, 3]``.
            env_ids: Environment indices to resample, ``[count]``.
        """
        centers, clearance = self._free_cells()
        admissible_cells = clearance >= self.goal_clearance_m
        for index in env_ids:
            distance = np.linalg.norm(centers - poses[index, :2], axis=1)
            candidate = admissible_cells & (distance >= self.min_goal_distance_m)
            if not candidate.any():
                candidate = admissible_cells & (distance >= self.arrival_tolerance_m)
            if not candidate.any():
                candidate = admissible_cells
            if not candidate.any():
                raise RuntimeError("No free cell is far enough from the robot for a DRL-VO goal")
            options = np.nonzero(candidate)[0]
            self._goal[index] = centers[options[self._random.randint(len(options))]]
            self._paths[index] = None

    def sample_start_poses(self, count: int, avoid: object = None) -> np.ndarray:
        """Draw random free start poses in clone-local stage coordinates.

        Args:
            count: Number of poses to draw.
            avoid: Optional ``(x, y, radius)`` triples in clone-local coordinates
                whose disks are excluded, so a robot never spawns already touching
                a pedestrian.

        Returns:
            Array of shape ``[count, 3]`` holding ``(x, y, yaw)`` [m, rad].
        """
        centers, clearance = self._free_cells()
        candidate = centers[clearance >= self.start_clearance_m]
        if not len(candidate):
            candidate = centers[clearance >= self.goal_clearance_m]
        if not len(candidate):
            raise RuntimeError("The room has no free cell clear enough for a random start pose")
        if avoid:
            keep = np.ones(len(candidate), dtype=bool)
            for x, y, radius in avoid:
                ros_x, ros_y = self.local_to_ros(x, y)
                keep &= np.hypot(candidate[:, 0] - ros_x, candidate[:, 1] - ros_y) > float(radius)
            # Falling back keeps the draw total even if the crowd walls off the room.
            if bool(keep.any()):
                candidate = candidate[keep]
        picks = candidate[self._random.randint(len(candidate), size=int(count))]
        yaw = self._random.uniform(-math.pi, math.pi, size=int(count))
        poses = np.zeros((int(count), 3), dtype=np.float64)
        for index in range(int(count)):
            poses[index, 0], poses[index, 1] = self.local_from_ros(picks[index, 0], picks[index, 1])
            poses[index, 2] = float(yaw[index])
        return poses

    # -- publication -----------------------------------------------------

    def reset(self, env: object, env_ids: object = None) -> None:
        """Invalidate the goal and the route of the selected environments.

        Args:
            env: Environment owning the robots.
            env_ids: Index tensor, slice or ``None`` for all environments.
        """
        for index in _indices(env, env_ids):
            self._goal[index] = np.nan
            self._paths[index] = None
        self._stamp = None

    def update(self, env: object) -> None:
        """Recompute the four published measurements once per step.

        Args:
            env: Environment with the robot articulation.
        """
        stamp = (int(getattr(env, "common_step_counter", 0)), int(_episode_step(env)))
        if self._stamp == stamp and self.distance_m is not None:
            return
        self._stamp = stamp
        poses = self.poses_ros(env)
        self.pose_ros = poses
        missing = np.nonzero(~np.isfinite(self._goal[:, 0]))[0]
        if len(missing):
            self.sample_goals(poses, missing)

        step = _episode_step(env)
        distances = np.zeros(self.num_envs, dtype=np.float32)
        local_goals = np.zeros((self.num_envs, 2), dtype=np.float32)
        for index in range(self.num_envs):
            path = self._paths[index]
            if path is None or step % self.replan_period_steps == 0:
                path = self._plan(poses[index], self._goal[index])
                self._paths[index] = path
            local_goals[index] = _pure_pursuit_local_goal(np.asarray(path), poses[index], lookahead_m=self.lookahead_m)
            distances[index] = float(np.hypot(*(poses[index, :2] - self._goal[index])))
        scans = self._scans(poses)
        clearance = self.clearance.clearance_m(poses[:, :2])

        self.distance_m = torch.as_tensor(distances, dtype=torch.float32, device=self.device)
        self.local_goal_base_xy = torch.as_tensor(local_goals, dtype=torch.float32, device=self.device)
        self.scan_ranges = scans
        self.clearance_m = torch.as_tensor(clearance, dtype=torch.float32, device=self.device)
        env.drl_vo_scan = self.scan_ranges

    def _scans(self, poses: np.ndarray) -> torch.Tensor:
        """Return the 720-beam front scan of every robot.

        On a CUDA device the whole batch is cast in one sweep, which is what
        makes a 512-robot step cost milliseconds instead of seconds.  A CPU
        tensor target keeps the original per-robot numpy path, since moving the
        rays to torch there would only add overhead.

        Args:
            poses: ``(x, y, yaw)`` per robot in the ROS map frame, ``[num_envs, 3]``.

        Returns:
            ``[num_envs, 720]`` ranges on the tensor device.
        """
        # One batched kernel for every device. The per-robot numpy fallback used
        # the upstream 180-degree beams, so it would disagree with the training
        # path about what the robot can see; the batched cast is exact on CPU too.
        if self._scan_occupancy is None or self._scan_occupancy.device != self.device:
            self._scan_occupancy = torch.as_tensor(
                np.asarray(self.grid.occupancy) == 100, dtype=torch.bool, device=self.device
            )
        return _synthesize_scan_batch(
            self._scan_occupancy,
            poses,
            resolution_m=float(self.grid.resolution_m),
            origin_xy_m=(float(self.grid.origin_xy_m[0]), float(self.grid.origin_xy_m[1])),
            min_range_m=self.scan_min_range_m,
            max_range_m=self.scan_max_range_m,
        )

    def _plan(self, pose: np.ndarray, goal: np.ndarray) -> list[tuple[float, float]]:
        """Return an A* route from ``pose`` to ``goal``, or a straight fallback."""
        start = (float(pose[0]), float(pose[1]))
        target = (float(goal[0]), float(goal[1]))
        try:
            path = self.plan_grid.plan(
                start,
                target,
                radius_m=self.robot_radius_m,
                arrival_tolerance_m=self.arrival_tolerance_m,
                margin_m=self.margin_m,
            )
        except ValueError:
            # An unreachable goal must not end the episode early: keep a straight
            # reference so the heading term still points at the target.
            path = [start, target]
        if len(path) < 2:
            path = [start, target]
        return [(float(point[0]), float(point[1])) for point in path]

    def goal_ros_xy(self, index: int = 0) -> tuple[float, float]:
        """Return the goal of one environment in the ROS map frame [m]."""
        goal = self._goal[int(index)]
        return float(goal[0]), float(goal[1])

    @property
    def goals_ros_xy(self) -> np.ndarray:
        """All goals in the ROS map frame, shape ``[num_envs, 2]``."""
        return self._goal.copy()


def _torch(value: object) -> torch.Tensor:
    """Return a torch tensor behind an Isaac Lab buffer (``ProxyArray`` on 3.0)."""
    return value if isinstance(value, torch.Tensor) else value.torch


def _yaw_from_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    """Yaw of XYZW quaternions, shape ``[count]``."""
    normalised = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1e-9)
    x, y, z, w = normalised.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _indices(env: object, env_ids: object) -> list[int]:
    """Normalise a manager selection to a list of environment indices."""
    count = int(env.num_envs)
    if env_ids is None:
        return list(range(count))
    if isinstance(env_ids, slice):
        return list(range(count))[env_ids]
    if isinstance(env_ids, torch.Tensor):
        return [int(value) for value in env_ids.reshape(-1).detach().cpu()]
    return [int(value) for value in env_ids]


def _episode_step(env: object) -> int:
    """Return the episode step counter of the first environment."""
    lengths = getattr(env, "episode_length_buf", None)
    if isinstance(lengths, torch.Tensor) and lengths.numel():
        return int(lengths[0])
    return 0
