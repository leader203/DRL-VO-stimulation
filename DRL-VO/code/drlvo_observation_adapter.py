"""CPU observation/action preparation; no simulator, gym, training, or ROS imports.

Original ROS1 normalization is the default (no observation clipping). Runtime
sensor projection and pedestrian rasterization follow the existing a_pipeline
ROS2 contract. Pedestrian truth must be labelled ``oracle`` by the caller.
"""
from __future__ import annotations

from collections import deque
import math
import numpy as np

OBSERVATION_SIZE = 19202
PED_MAP_SHAPE = (2, 80, 80)
SCAN_HISTORY = 10
RAW_BEAMS_PER_SENSOR = 2000
SENSOR_HZ = 15


def _finite(value, shape, name):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def compress_scan_history(scan_history):
    history = _finite(scan_history, (10, 720), "scan_history")
    if np.any(history < 0):
        raise ValueError("scan_history ranges cannot be negative")
    groups = history.reshape(10, 80, 9)
    reduced = np.empty((10, 2, 80), dtype=np.float32)
    reduced[:, 0] = groups.min(axis=2)
    reduced[:, 1] = groups.mean(axis=2)
    return np.tile(reduced.reshape(-1), 4)


def build_observation(pedestrian_map_mps, scan_history, local_goal_m, *, clip_pedestrian=False):
    ped = _finite(pedestrian_map_mps, PED_MAP_SHAPE, "pedestrian_map_mps")
    goal = _finite(local_goal_m, (2,), "local_goal_m")
    # Preserve the arithmetic and flatten order in DRLNavEnv._get_observation.
    ped = 2 * (ped - (-2)) / 4 - 1
    if clip_pedestrian:
        # Explicit option matching the newer a_pipeline ROS2 adapter, which
        # differs from the original training source for |velocity| > 2 m/s.
        ped = np.clip(ped, -1, 1)
    scan = 2 * compress_scan_history(scan_history) / 30 - 1
    goal = 2 * (goal - (-2)) / 4 - 1
    return np.concatenate((ped.reshape(-1), scan, goal)).astype(np.float32)


def normalized_to_physical(action):
    action = np.clip(_finite(action, (2,), "action"), -1, 1)
    return np.asarray([(action[0] + 1) * .25, action[1] * 2], dtype=np.float32)


def rotate_world_to_base(vectors, yaw):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != 2 or not np.isfinite(vectors).all() or not math.isfinite(yaw):
        raise ValueError("vectors must be finite (N,2) and yaw finite")
    c, s = math.cos(yaw), math.sin(yaw)
    result = np.empty_like(vectors)
    result[:, 0] = c * vectors[:, 0] + s * vectors[:, 1]
    result[:, 1] = -s * vectors[:, 0] + c * vectors[:, 1]
    return result


def oracle_pedestrian_velocity_map(position_world_xy, absolute_velocity_world_xy, robot_pose_xyyaw):
    """GT roots -> absolute velocities in robot axes; do not subtract robot v.

    Legacy raster cell collisions retain the last input pedestrian. Track
    freshness/confidence selection belongs to a later perception adapter.
    """
    positions = np.asarray(position_world_xy, dtype=np.float32)
    velocities = np.asarray(absolute_velocity_world_xy, dtype=np.float32)
    pose = _finite(robot_pose_xyyaw, (3,), "robot_pose_xyyaw")
    if positions.shape != velocities.shape:
        raise ValueError("pedestrian position and velocity shapes must match")
    relative = rotate_world_to_base(positions - pose[:2], float(pose[2]))
    base_velocity = rotate_world_to_base(velocities, float(pose[2]))
    result = np.zeros(PED_MAP_SHAPE, dtype=np.float32)
    for (x, y), velocity in zip(relative, base_velocity):
        if 0 <= x <= 20 and abs(y) <= 10:
            row = min(int(math.floor(float(x) / .25)), 79)
            col = min(int(math.floor((10 - float(y)) / .25)), 79)
            result[:, row, col] = velocity
    return result


def sensor_ranges_to_base(raw_ranges, *, angle_min, angle_increment, range_min, range_max, translation_xy, yaw):
    """Raw 2000 slots -> base frame endpoints, retaining invalid slots as NaN.

    Non-finite returns have the original ROS1 zero-placeholder semantics.
    Keep them invalid during geometric projection: zero is not a hit at the
    sensor mount. Unfilled legacy bins become zero in dual_lidar_to_legacy_scan;
    a real return from the other sensor may still fill the same bin.
    Requires caller to supply measured sensor transform, never inferred mount.
    """
    raw = np.asarray(raw_ranges, dtype=np.float32)
    if raw.shape != (2000,):
        raise ValueError("raw sensor ranges must have exactly 2000 slots")
    translation = _finite(translation_xy, (2,), "translation_xy")
    if not all(math.isfinite(float(x)) for x in (angle_min, angle_increment, range_min, range_max, yaw)) or not 0 <= range_min < range_max or angle_increment <= 0:
        raise ValueError("invalid sensor metadata")
    angles = angle_min + np.arange(2000, dtype=np.float32) * angle_increment
    geometric = raw.copy()
    valid = np.isfinite(geometric) & (geometric > 0) & (geometric >= range_min) & (geometric <= range_max)
    points = np.full((2000, 2), np.nan, dtype=np.float32)
    c, s = math.cos(yaw), math.sin(yaw)
    x, y = geometric[valid] * np.cos(angles[valid]), geometric[valid] * np.sin(angles[valid])
    points[valid, 0] = translation[0] + c*x - s*y
    points[valid, 1] = translation[1] + s*x + c*y
    return np.hypot(points[:, 0], points[:, 1]), np.arctan2(points[:, 1], points[:, 0])


def dual_lidar_to_legacy_scan(base_ranges, base_angles):
    ranges = np.asarray(base_ranges)
    angles = np.asarray(base_angles)
    if ranges.shape != (4000,) or angles.shape != (4000,):
        raise ValueError("two sensors require 4000 retained slots in base frame")
    full = np.zeros(1080, dtype=np.float32)
    valid = np.isfinite(ranges) & np.isfinite(angles) & (ranges > 0)
    selected_ranges = ranges[valid].astype(np.float32)
    indices = np.rint((angles[valid].astype(np.float64) + 3*math.pi/4) / (math.pi/720)).astype(np.int64)
    inside = (indices >= 0) & (indices < 1080)
    indices, selected_ranges = indices[inside], selected_ranges[inside]
    if len(indices):
        order = np.lexsort((np.arange(len(indices)), selected_ranges, indices))
        sorted_indices = indices[order]
        first = np.concatenate(([True], sorted_indices[1:] != sorted_indices[:-1]))
        chosen = order[first]
        full[indices[chosen]] = selected_ranges[chosen]
    front = full[180:900].copy()
    nonzero = front[front > 0]
    return front, float(np.count_nonzero(front) / 720), float(nonzero.min()) if nonzero.size else math.nan


class ScanHistory:
    """Sample latest real 15 Hz scan at 20 Hz deadlines on physical ticks."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.frames = deque(maxlen=10)
        self.last_timestamp_ns = None
        self.source_timestamps_ns = deque(maxlen=10)
        self.sample_timestamps_ns = deque(maxlen=10)
        self.latest_scan = None
        self.latest_scan_timestamp_ns = None
        self.sample_origin_ns = None
        self.sample_index = 0
        self.last_sample_call_ns = None

    def append(self, front_scan, timestamp_ns):
        scan = _finite(front_scan, (720,), "front_scan")
        if np.any(scan < 0):
            raise ValueError("scan ranges cannot be negative")
        stamp = int(timestamp_ns)
        if stamp < 0:
            raise ValueError("timestamp cannot be negative")
        if self.latest_scan_timestamp_ns is not None:
            delta = stamp - self.latest_scan_timestamp_ns
            if delta <= 0:
                raise ValueError("simulation timestamps must increase; call reset on episode reset")
            if abs(delta - 1e9/15) > 2e6:
                raise ValueError("scan timestamp interval must match 15 Hz (2 ms tolerance)")
        self.latest_scan = scan.copy()
        self.latest_scan_timestamp_ns = stamp

    def sample(self, timestamp_ns):
        """Repeat latest scan when necessary; never invent/interpolate a scan."""
        now = int(timestamp_ns)
        if now < 0 or (self.last_sample_call_ns is not None and now <= self.last_sample_call_ns):
            raise ValueError('History sampling clock must strictly increase')
        self.last_sample_call_ns = now
        if self.latest_scan is None:
            return False
        if self.latest_scan_timestamp_ns > now:
            raise ValueError('Cannot sample a future scan')
        if self.sample_origin_ns is None:
            self.sample_origin_ns = now
        period = 50_000_000
        if now < self.sample_origin_ns + self.sample_index * period:
            return False
        self.frames.append(self.latest_scan.copy())
        self.last_timestamp_ns = self.latest_scan_timestamp_ns
        self.source_timestamps_ns.append(self.last_timestamp_ns)
        self.sample_timestamps_ns.append(now)
        self.sample_index = max(self.sample_index + 1, (now-self.sample_origin_ns)//period + 1)
        return True

    @property
    def ready(self):
        return len(self.frames) == 10

    def array(self):
        if not self.ready:
            raise RuntimeError("history warmup: require 10 causal 20 Hz samples before policy action")
        return np.stack(self.frames)
