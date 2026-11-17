"""User spawn pool and original-distance random goals; no simulation or policy imports."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import maximum_filter


class TaskSamplingError(RuntimeError):
    pass


class TaskAdmissionError(TaskSamplingError):
    """Valid task request exhausted admission; configuration/safety faults differ."""
    def __init__(self, message, *, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def make_navigation(batch):
    from drlvo_heading_planner import HeadingPlanner
    batch = Path(batch)
    class BatchMap:
        resolution = .1
        _bounds = np.asarray(json.loads((batch/'MAP_OVERRIDE.json').read_text())['map_bounds_xy'], dtype=float)

        def __init__(self):
            with np.load(batch/'scene/open_doors_native_grid.npz', allow_pickle=False) as z:
                self._labels = z['labels'].copy()

        def pixel_to_world(self, cells):
            cells = np.asarray(cells)
            return np.stack((self._bounds[0]+.05+.1*cells[..., 1],
                             self._bounds[3]-.05-.1*cells[..., 0]), axis=-1)

        def world_to_pixel(self, xy):
            xy = np.asarray(xy)
            return np.stack((np.rint((self._bounds[3]-.05-xy[..., 1])/.1),
                             np.rint((xy[..., 0]-self._bounds[0]-.05)/.1)), axis=-1).astype(int)
    bounds = json.loads((batch/'SIZE_REFERENCE.json').read_text())['robot_static_asset_bounds_relative_base_link']
    return HeadingPlanner(BatchMap(), bounds['min_xyz'], bounds['max_xyz'], planning_tracking_margin=0.)


def terminal_requires_teleport(reason):
    if reason == 'goal':
        return False
    if reason in ('initial', 'collision', 'timeout'):
        return True
    raise TaskSamplingError(f'Not a normal task boundary: {reason}')


class TaskSampler:
    def __init__(self, batch, navigation, seed):
        self.batch = Path(batch)
        self.navigation = navigation
        self.rng = np.random.default_rng(int(seed))
        self.task_id = 0
        self.selection = json.loads((self.batch/'START_CANDIDATES.json').read_text())
        self.config = json.loads((self.batch/'TASK_CONFIG.json').read_text())
        if hashlib.sha256((self.batch/'SIZE_REFERENCE.json').read_bytes()).hexdigest() != self.selection['geometry_sha256']:
            raise TaskSamplingError('Selected starts and task robot geometry differ')
        if self.config['goal_distance_m'] != {'min_inclusive': 4.2, 'max_exclusive': 7.0}:
            raise TaskSamplingError('Unexpected goal-distance contract')
        self.map_sha256 = hashlib.sha256((self.batch/'scene/open_doors_native_grid.npz').read_bytes()).hexdigest()
        if self.map_sha256 != self.selection['map_sha256']:
            raise TaskSamplingError('Selected starts and task map differ')
        with np.load(self.batch/'scene/open_doors_native_grid.npz', allow_pickle=False) as z:
            if not np.array_equal(z['labels'], navigation._labels):
                raise TaskSamplingError('Sampler and navigation maps differ')
        self.starts = copy.deepcopy(self.selection['start_candidates'])
        if not self.starts or len({p['id'] for p in self.starts}) != len(self.starts):
            raise TaskSamplingError('Empty or duplicate start IDs')
        for p in self.starts:
            pose = np.r_[p['xy_m'], p['yaw_rad']]
            if not np.isfinite(pose).all() or not navigation.pose_clear(pose[:2], pose[2], .03):
                raise TaskSamplingError(f'Invalid selected start: {p["id"]}')
        # Preserve the executable original's [-cell_radius,+cell_radius) square,
        # with explicit out-of-map rejection instead of Python negative indexing.
        cell_radius = int(.8/navigation.resolution)
        self.goal_valid = maximum_filter((navigation._labels != 254).astype(np.uint8),
                                        size=2*cell_radius, mode='constant', cval=1) == 0
        cells = np.argwhere(self.goal_valid)
        self.goal_xy = navigation.pixel_to_world(cells)
        if not len(self.goal_xy):
            raise TaskSamplingError('No map-valid goal cells under original 0.8 m square check')

    def random_start(self):
        return copy.deepcopy(self.starts[int(self.rng.integers(len(self.starts)))])

    def eligible_goals(self, current_xy):
        xy = np.asarray(current_xy, dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise TaskSamplingError('Current XY must be two finite metres')
        distances = np.linalg.norm(self.goal_xy-xy, axis=1)
        return np.flatnonzero((distances >= 4.2) & (distances < 7.))

    def sample(self, current_pose, *, require_route=True, time_budget_s=None):
        pose = np.asarray(current_pose, dtype=float)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise TaskSamplingError('Current pose must be finite XY/yaw')
        indices = self.eligible_goals(pose[:2])
        if not len(indices):
            raise TaskAdmissionError(f'No valid goals at 4.2 <= d < 7 m from {pose[:2].tolist()}',
                diagnostics=dict(source_pose=pose.tolist(), eligible_goals=0, rejected_candidates=[]))
        attempts = min(len(indices), self.config['max_goal_attempts'])
        order = self.rng.choice(indices, size=attempts, replace=False)
        budget = float(self.config['sampling_budget_s'])
        if time_budget_s is not None:
            if not math.isfinite(time_budget_s) or time_budget_s < 0:
                raise TaskSamplingError('Task admission budget must be finite and nonnegative')
            budget = min(budget, time_budget_s)
        deadline = time.monotonic()+budget
        rejected = []
        p = self.navigation
        old_reference = p.reference_poses
        old_reference_only = p.reference_only
        old_yaw = p.current_yaw
        try:
            p.reference_poses = None
            p.reference_only = False
            p.current_yaw = float(pose[2])
            for index in order:
                goal = self.goal_xy[index].copy()
                poses = None
                if require_route:
                    remaining = deadline-time.monotonic()
                    if remaining <= 0:
                        break
                    path = p.astar_path(pose[:2], goal, time_budget_s=min(remaining, self.config['per_route_budget_s']))
                    if path is None or p.last_poses is None:
                        rejected.append(dict(goal_xy=goal.tolist(), reason='no_admitted_route_within_budget'))
                        continue
                    poses = p.last_poses.copy()
                self.task_id += 1
                return dict(task_id=self.task_id, start_pose=pose.tolist(), goal_xy=goal.tolist(),
                            straight_distance_m=float(np.linalg.norm(goal-pose[:2])),
                            goal_yaw_required=False, goal_neighborhood_radius_parameter_m=.8,
                            goal_map_valid=True, route_checked=bool(require_route),
                            poses_xy_yaw=None if poses is None else poses.tolist(),
                            rejected_candidates=rejected, map_sha256=self.map_sha256)
        finally:
            p.reference_poses = old_reference
            p.reference_only = old_reference_only
            p.current_yaw = old_yaw
        raise TaskAdmissionError(f'No admitted goal within bounded search: {len(rejected)} rejected, {attempts} maximum',
            diagnostics=dict(source_pose=pose.tolist(), eligible_goals=len(indices),
                max_goal_attempts=attempts, sampling_budget_s=budget,
                rejected_candidates=rejected, budget_exhausted=time.monotonic()>=deadline))

    def state(self):
        return dict(task_id=self.task_id, rng_state=copy.deepcopy(self.rng.bit_generator.state),
                    map_sha256=self.map_sha256)

    def restore(self, state):
        if state['map_sha256'] != self.map_sha256:
            raise TaskSamplingError('Task RNG state belongs to another map')
        self.rng.bit_generator.state = copy.deepcopy(state['rng_state'])
        self.task_id = int(state['task_id'])
