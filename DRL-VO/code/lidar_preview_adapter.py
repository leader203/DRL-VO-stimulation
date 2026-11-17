"""Dual 2D LiDAR from the *current* PhysX scene, without starting Kit.

Call scan after a completed physics step using the navigation robot's actual
pose (XYZ stage units and quaternion XYZW). No static mesh cache, USD pose
write, ground-truth pedestrian injection, ROS node, or policy is used here.
Only the carrying robot is excluded. AMP limb colliders remain visible; the
rendered SMPL skin is not the geometry measured by this physics sensor.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import math
import time
import numpy as np


DEFAULT_MOUNTS = (
    ("scan_01", (0.2, 0.13, 0.208), 0.0),
    ("scan_02", (-0.2, -0.13, 0.208), math.pi),
)


def quaternion_matrix_xyzw(quaternion):
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("quaternion must be finite XYZW[4]")
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("quaternion norm must be nonzero")
    x, y, z, w = q / norm
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float64)


def path_under(path, root):
    return path == root or path.startswith(root + "/")


def hit_value(hit, name, default=None):
    if isinstance(hit, dict):
        return hit.get(name, default)
    if isinstance(hit, Mapping):
        return hit.get(name, default)
    # Native Sim 6 all-hit objects use snake_case; closest dicts use camelCase.
    object_name = "rigid_body" if name == "rigidBody" else name
    return getattr(hit, object_name, default)


class LidarPreviewAdapter:
    """15 Hz, two 2000-beam instantaneous scene queries, Z-up robot frame.

    ``query`` is omni.physx.get_physx_scene_query_interface() from the running
    application's scene. ``scan`` returns None between scheduled scans. Both
    sensors share one postphysics timestamp. Missing returns stay +Inf, with
    no preprocessing into DRL-VO network features at this stage.
    """

    def __init__(self, query, robot_root="/World/NavigationRobot", *,
                 stage_meters_per_unit=1.0, pedestrian_roots=(), mounts=DEFAULT_MOUNTS,
                 beam_count=2000, rate_hz=15.0, range_min_m=0.5, range_max_m=50.0,
                 both_sides=False, capture_self_echoes=False):
        if not isinstance(robot_root, str) or not robot_root.startswith("/") or robot_root == "/":
            raise ValueError("robot_root must be an absolute, specific navigation robot path")
        if beam_count < 2 or int(beam_count) != beam_count:
            raise ValueError("beam_count must be an integer >= 2")
        values = (stage_meters_per_unit, rate_hz, range_min_m, range_max_m)
        if not all(math.isfinite(v) for v in values) or stage_meters_per_unit <= 0 or rate_hz <= 0:
            raise ValueError("scale/rate/ranges must be finite; scale and rate positive")
        if range_min_m < 0 or range_max_m <= range_min_m:
            raise ValueError("invalid LiDAR range limits")
        if not callable(getattr(query, "raycast_all", None)):
            raise TypeError("query must provide raycast_all for self-hit filtering")
        self.query, self.robot_root = query, robot_root.rstrip("/")
        self.scale, self.rate_hz = float(stage_meters_per_unit), float(rate_hz)
        self.range_min, self.range_max = float(range_min_m), float(range_max_m)
        self.beam_count, self.both_sides = int(beam_count), bool(both_sides)
        self.capture_self_echoes = bool(capture_self_echoes)
        self.pedestrian_roots = tuple(str(p).rstrip("/") for p in pedestrian_roots)
        if any(path_under(p, self.robot_root) for p in self.pedestrian_roots):
            raise ValueError("navigation robot exclusion must not contain pedestrian roots")
        self.angles = -math.pi + np.arange(self.beam_count) * (2*math.pi/self.beam_count)
        self.mounts = []
        for name, xyz, yaw in mounts:
            xyz = np.asarray(xyz, dtype=np.float64)
            if xyz.shape != (3,) or not np.isfinite(xyz).all() or not math.isfinite(yaw):
                raise ValueError("mount offsets must be finite XYZ metres and yaw radians")
            local = self.angles + yaw
            directions = np.column_stack((np.cos(local), np.sin(local), np.zeros(self.beam_count)))
            self.mounts.append((str(name), xyz, float(yaw), directions))
        if len(self.mounts) != 2 or len({m[0] for m in self.mounts}) != 2:
            raise ValueError("exactly two distinct sensor names required")
        self.sequence = 0
        # Cache only immutable path classification, never geometry or returns.
        self._path_classification = {}
        self.reset_clock()

    def reset_clock(self):
        """Call on replay/reset; changes only the scan schedule."""
        self._epoch = self._last_time = None
        self._next_index = 0

    def _is_self(self, hit):
        return self._classify(hit)[0]

    def _classify(self, hit):
        collision = str(hit_value(hit, "collision", ""))
        body = str(hit_value(hit, "rigidBody", ""))
        key = (collision, body)
        result = self._path_classification.get(key)
        if result is None:
            carrying_robot = path_under(collision, self.robot_root) or path_under(body, self.robot_root)
            pedestrian = next((p for p in self.pedestrian_roots
                               if path_under(collision, p) or path_under(body, p)), None)
            result = (carrying_robot, pedestrian, collision or body)
            self._path_classification[key] = result
        return result

    def _nearest(self, origin, direction, stats):
        origin = tuple(float(v) for v in origin)
        direction = tuple(float(v) for v in direction)
        length = (self.range_max-self.range_min)/self.scale
        nearest = None
        closest_fn = getattr(self.query, "raycast_closest", None)
        if callable(closest_fn):
            candidate = closest_fn(origin, direction, length, self.both_sides)
            stats["closest_query_count"] += 1
            # Sim 6 misses return a dict with hit=False; tolerate object results.
            if not hit_value(candidate, "hit", bool(candidate)):
                return None
            if not self._is_self(candidate):
                return candidate
            stats["self_hits_ignored"] += 1
            path = str(hit_value(candidate, "collision", "") or hit_value(candidate, "rigidBody", ""))
            if path and self.capture_self_echoes:
                self._ray_self_paths.append(path)
        stats["all_query_count"] += 1

        def collect(candidate):
            nonlocal nearest
            if self._is_self(candidate):
                stats["self_hits_ignored"] += 1
                path = str(hit_value(candidate, "collision", "") or hit_value(candidate, "rigidBody", ""))
                if path and self.capture_self_echoes:
                    self._ray_self_paths.append(path)
            else:
                d = float(hit_value(candidate, "distance", math.inf))
                if math.isfinite(d) and d >= 0 and (nearest is None or d < float(hit_value(nearest, "distance", math.inf))):
                    # Copy callback data before native traversal advances.
                    nearest = {"distance": d, "collision": str(hit_value(candidate, "collision", "")),
                               "rigidBody": str(hit_value(candidate, "rigidBody", ""))}
            return True  # all-hit order is not guaranteed; continue traversal

        self.query.raycast_all(origin, direction, length, collect, self.both_sides)
        return nearest

    def scan(self, robot_position, robot_quaternion_xyzw, sim_time):
        sim_time = float(sim_time)
        if not math.isfinite(sim_time):
            raise ValueError("simulation time must be finite")
        if self._last_time is not None and sim_time < self._last_time - 1e-9:
            self.reset_clock()
        self._last_time = sim_time
        if self._epoch is None:
            self._epoch = sim_time
        due = self._epoch + self._next_index/self.rate_hz
        if sim_time + 1e-9 < due:
            return None
        index = max(self._next_index, int(math.floor((sim_time-self._epoch)*self.rate_hz + 1e-8)))
        missed = index-self._next_index
        position = np.asarray(robot_position, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("actual robot position must be finite XYZ[3] in stage units")
        rotation = quaternion_matrix_xyzw(robot_quaternion_xyzw)
        started = time.perf_counter()
        stats = {"closest_query_count": 0, "all_query_count": 0, "self_hits_ignored": 0,
                 "pedestrian_hit_count": 0, "scene_hit_count": 0, "unknown_path_hit_count": 0,
                 "missed_scheduled_scans": missed, "hits_by_path": {}, "pedestrian_hits_by_root": {}}
        by_path, by_ped = Counter(), Counter()
        output = {}
        for name, mount, yaw, local_directions in self.mounts:
            sensor_stage = position + rotation @ (mount/self.scale)
            directions = local_directions @ rotation.T
            origins = sensor_stage + directions*(self.range_min/self.scale)
            ranges = np.full(self.beam_count, np.inf, dtype=np.float32)
            distances = np.full(self.beam_count, np.inf, dtype=np.float64)
            points = np.full((self.beam_count, 3), np.nan, dtype=np.float64)
            paths = [""]*self.beam_count
            self_paths = [[] for _ in range(self.beam_count)] if self.capture_self_echoes else None
            sensor_ped, sensor_scene = 0, 0
            for i in range(self.beam_count):
                self._ray_self_paths = [] if self.capture_self_echoes else None
                hit = self._nearest(origins[i], directions[i], stats)
                if self.capture_self_echoes:
                    self_paths[i] = list(dict.fromkeys(self._ray_self_paths))
                if hit is None:
                    continue
                d = self.range_min + float(hit_value(hit, "distance", math.inf))*self.scale
                if not math.isfinite(d) or d < self.range_min or d > self.range_max+1e-6:
                    continue
                ranges[i] = min(d, self.range_max)
                distances[i] = d
                _, pedestrian, paths[i] = self._classify(hit)
                by_path[paths[i]] += 1
                if pedestrian:
                    by_ped[pedestrian] += 1
                    stats["pedestrian_hit_count"] += 1
                    sensor_ped += 1
                elif paths[i]:
                    stats["scene_hit_count"] += 1
                    sensor_scene += 1
                else:
                    stats["unknown_path_hit_count"] += 1
            finite = np.isfinite(ranges)
            # Use original double-precision hit distances for points; ranges
            # are FP32 storage. Avoid a tiny NumPy operation per individual ray.
            points[finite] = sensor_stage*self.scale + directions[finite]*distances[finite, None]
            output[name] = {
                "stamp_sim_time": sim_time, "frame": name, "ranges": ranges,
                "angle_min": -math.pi, "angle_max": float(self.angles[-1]),
                "angle_increment": 2*math.pi/self.beam_count, "scan_time": 1/self.rate_hz,
                "time_increment": 0.0, "range_min": self.range_min, "range_max": self.range_max,
                "sensor_position_world_m": sensor_stage*self.scale,
                "mount_position_robot_m": mount.copy(), "mount_yaw_robot_rad": yaw,
                "points_world_m": points, "hit_paths": paths,
                "finite_beams": int(np.isfinite(ranges).sum()),
                "pedestrian_hit_count": sensor_ped, "scene_hit_count": sensor_scene,
            }
            if self.capture_self_echoes:
                output[name]["self_hit_paths"] = self_paths
        stats["hits_by_path"] = dict(by_path)
        stats["pedestrian_hits_by_root"] = dict(by_ped)
        stats["query_elapsed_ms"] = (time.perf_counter()-started)*1000
        self.sequence += 1
        self._next_index = index + 1
        output["stats"] = dict(stats, sequence=self.sequence, stamp_sim_time=sim_time,
                               backend="omni.physx.scene_query", excluded_root=self.robot_root,
                               nominal_rate_hz=self.rate_hz, beams_per_sensor=self.beam_count,
                               instantaneous_scan=True, scan_merged_used=False,
                               drlvo_policy_input_ready=False,
                               pedestrian_geometry="live_PhysX_colliders_not_SMPL_visual_skin")
        return output
