# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ROS occupancy-grid planning without simulator, navigation-mesh or ROS imports."""

from __future__ import annotations

import heapq
import math
from bisect import bisect_left, bisect_right
from collections import OrderedDict, deque
from collections.abc import Collection

import numpy as np

_NEIGHBOR_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
_GOAL_CACHE_CAPACITY = 16
_GOAL_CACHE_ANCHOR_LIMIT = 4096


def _anchors_connected(mask: np.ndarray, seeds: list[tuple[int, int]], endpoints: Collection[tuple[int, int]]) -> bool:
    """Check raster reachability by traversing horizontal free runs."""
    if set(seeds).intersection(endpoints):
        return True
    # Every permitted diagonal has a cardinal alternative under A*'s corner rule.
    height = mask.shape[0]
    padded = np.pad(mask, ((0, 0), (1, 1)))
    rows, columns = np.nonzero(padded[:, 1:] != padded[:, :-1])
    splits = np.searchsorted(rows, np.arange(height + 1))
    starts = [columns[splits[row] : splits[row + 1] : 2].tolist() for row in range(height)]
    ends = [(columns[splits[row] + 1 : splits[row + 1] : 2] - 1).tolist() for row in range(height)]
    seed_runs = {(row, bisect_right(starts[row], column) - 1) for row, column in seeds}
    goal_runs = {(row, bisect_right(starts[row], column) - 1) for row, column in endpoints}
    pending = deque(seed_runs)
    visited = set(seed_runs)
    while pending:
        row, run = pending.popleft()
        if (row, run) in goal_runs:
            return True
        low, high = starts[row][run], ends[row][run]
        for other in (row - 1, row + 1):
            if not 0 <= other < height:
                continue
            first = bisect_left(ends[other], low)
            stop = bisect_right(starts[other], high)
            for index in range(first, stop):
                candidate = (other, index)
                if candidate not in visited:
                    pending.append(candidate)
                    visited.add(candidate)
    return False


def convex_footprint(points) -> np.ndarray:
    """Return a finite counterclockwise convex footprint [m], shape [N, 2]."""
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("footprint_xy_m must contain finite planar points [m].")
    unique = sorted(set(map(tuple, values)))

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    lower = []
    upper = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 1e-12:
            lower.pop()
        lower.append(point)
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 1e-12:
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=float)
    if len(hull) < 3:
        raise ValueError("footprint_xy_m must span a nondegenerate convex area [m²].")
    return hull


def segment_polygon_clear(start, end, footprint, clearance_m: float) -> bool:
    """Certify the complete swept disk against a convex polygon [m]."""
    polygon = np.asarray(footprint, dtype=float)
    a, b = np.asarray(start[:2], dtype=float), np.asarray(end[:2], dtype=float)
    if not np.isfinite([*a, *b, clearance_m]).all() or clearance_m < 0:
        return False
    # Separated bounds certify the entire chord, including the narrow-phase
    # squared-distance tolerance; nearby polygons still receive exact checks.
    padding = math.hypot(clearance_m, 1e-6)
    if np.any(np.maximum(a, b) < polygon.min(axis=0) - padding) or np.any(
        np.minimum(a, b) > polygon.max(axis=0) + padding
    ):
        return True
    edges = np.roll(polygon, -1, axis=0) - polygon
    for point in (a, b):
        relative = point - polygon
        if np.all(edges[:, 0] * relative[:, 1] - edges[:, 1] * relative[:, 0] >= -1e-12):
            return False
    delta = b - a
    length_sq = float(np.dot(delta, delta))

    def cross(first, second):
        return float(first[0] * second[1] - first[1] * second[0])

    for p, edge in zip(polygon, edges):
        q = p + edge
        denominator = cross(delta, edge)
        if abs(denominator) > 1e-14:
            t = cross(p - a, edge) / denominator
            u = cross(p - a, delta) / denominator
            if 0 <= t <= 1 and 0 <= u <= 1:
                return False
        edge_sq = float(np.dot(edge, edge))
        for point in (a, b):
            nearest = p + np.clip(np.dot(point - p, edge) / edge_sq, 0, 1) * edge
            if np.dot(point - nearest, point - nearest) <= clearance_m**2 + 1e-12:
                return False
        for point in (p, q):
            nearest = a + np.clip(np.dot(point - a, delta) / length_sq, 0, 1) * delta if length_sq else a
            if np.dot(point - nearest, point - nearest) <= clearance_m**2 + 1e-12:
                return False
    return True


def _planning_obstacles(obstacles, door_specs):
    """Retain observed geometry except bodies owned by a declared door interlock."""
    controlled = tuple(
        str(spec.get("body_prim_path", spec["name"])) for spec in door_specs if spec.get("enabled", True)
    )
    return [
        obstacle
        for obstacle in obstacles
        if not any(obstacle.name == root or obstacle.name.startswith(root + "/") for root in controlled)
    ]


def _path_lookahead(path, position, reach_m, clear_segment):
    """Look through a selected nearby planning corner using certified chords [m]."""
    vertex = path[1]
    if len(path) < 3 or math.dist(position[:2], vertex) >= reach_m:
        return list(vertex)
    edge = np.asarray(path[2]) - vertex
    length = float(np.linalg.norm(edge))
    if length == 0.0:
        return list(vertex)
    # Intermediate vertices are guides, not business arrival points.
    lower, upper = 0.0, min(1.0, reach_m / length)
    candidate = np.asarray(vertex) + upper * edge
    if clear_segment(position, candidate):
        return candidate.tolist()
    for _ in range(8):
        fraction = (lower + upper) / 2
        candidate = np.asarray(vertex) + fraction * edge
        if clear_segment(position, candidate):
            lower = fraction
        else:
            upper = fraction
    return (np.asarray(vertex) + lower * edge).tolist() if lower > 0.0 else list(vertex)


class OccupancyGrid:
    """Immutable occupancy raster; rows increase along world +Y.

    Args:
        occupancy: Cell values [-1 unknown, 0 free, 100 occupied], shape [H, W].
        resolution_m: Square cell width [m].
        origin_xy_m: World position of the raster's bottom-left corner [m].
        goal_projection_tolerance_m: Optional cap on A* goal projection [m].
    """

    def __init__(
        self,
        occupancy: np.ndarray,
        resolution_m: float,
        origin_xy_m: tuple[float, float],
        *,
        goal_projection_tolerance_m: float | None = None,
    ):
        values = np.asarray(occupancy)
        if values.ndim != 2 or not values.size or not np.isin(values, [-1, 0, 100]).all():
            raise ValueError("occupancy must be a nonempty [H, W] array containing only -1, 0 and 100.")
        if not math.isfinite(resolution_m) or resolution_m <= 0:
            raise ValueError("resolution_m must be finite and positive [m].")
        if len(origin_xy_m) != 2 or not all(math.isfinite(x) for x in origin_xy_m):
            raise ValueError("origin_xy_m requires two finite coordinates [m].")
        if goal_projection_tolerance_m is not None and (
            not math.isfinite(goal_projection_tolerance_m) or goal_projection_tolerance_m <= 0
        ):
            raise ValueError("goal_projection_tolerance_m must be finite and positive [m].")
        self.occupancy = values.astype(np.int8, copy=True)
        self.occupancy.flags.writeable = False
        self.resolution_m = float(resolution_m)
        self.origin_xy_m = tuple(float(x) for x in origin_xy_m)
        self.goal_projection_tolerance_m = goal_projection_tolerance_m
        self.height, self.width = values.shape
        self._blocked = values != 0
        self._masks: dict[float, np.ndarray] = {}
        self._occupied = np.argwhere(self._blocked)
        self._goal_endpoint_cache: OrderedDict[tuple, tuple[np.ndarray, dict]] = OrderedDict()

    def world_to_cell(self, point: tuple[float, float]) -> tuple[int, int]:
        """Return row/column indices for a world point [m]."""
        x, y = point[:2]
        return (
            math.floor((y - self.origin_xy_m[1]) / self.resolution_m),
            math.floor((x - self.origin_xy_m[0]) / self.resolution_m),
        )

    def cell_center(self, cell: tuple[int, int]) -> tuple[float, float]:
        """Return a raster cell's world center [m]."""
        row, column = cell
        return (
            self.origin_xy_m[0] + (column + 0.5) * self.resolution_m,
            self.origin_xy_m[1] + (row + 0.5) * self.resolution_m,
        )

    def free_mask(self, clearance_m: float) -> np.ndarray:
        """Return center cells clear of unknown/occupied rectangles by clearance [m]."""
        if clearance_m < 0 or not math.isfinite(clearance_m):
            raise ValueError("clearance_m must be finite and nonnegative [m].")
        key = float(clearance_m)
        if key in self._masks:
            return self._masks[key]
        blocked = self._blocked.copy()
        reach = math.ceil(clearance_m / self.resolution_m + 0.5)
        for dr in range(-reach, reach + 1):
            for dc in range(-reach, reach + 1):
                distance = math.hypot(max(abs(dr) - 0.5, 0), max(abs(dc) - 0.5, 0)) * self.resolution_m
                if distance > clearance_m + 1e-10:
                    continue
                r0, r1 = max(0, dr), min(self.height, self.height + dr)
                c0, c1 = max(0, dc), min(self.width, self.width + dc)
                if r0 < r1 and c0 < c1:
                    blocked[r0:r1, c0:c1] |= self._blocked[r0 - dr : r1 - dr, c0 - dc : c1 - dc]
        rows, cols = np.indices(blocked.shape)
        edge = np.minimum.reduce(
            (
                (rows + 0.5) * self.resolution_m,
                (self.height - rows - 0.5) * self.resolution_m,
                (cols + 0.5) * self.resolution_m,
                (self.width - cols - 0.5) * self.resolution_m,
            )
        )
        result = ~blocked & (edge >= clearance_m)
        result.flags.writeable = False
        self._masks[key] = result
        return result

    def segment_clear(self, start: tuple[float, float], end: tuple[float, float], clearance_m: float) -> bool:
        """Check the entire swept disk against occupied cells and map edges [m]."""
        a, b = np.asarray(start[:2], dtype=float), np.asarray(end[:2], dtype=float)
        if not np.isfinite([*a, *b, clearance_m]).all() or clearance_m < 0:
            return False
        lower = np.asarray(self.origin_xy_m) + clearance_m
        upper = (
            np.asarray(self.origin_xy_m)
            + [self.width * self.resolution_m, self.height * self.resolution_m]
            - clearance_m
        )
        if np.any(np.minimum(a, b) < lower - 1e-10) or np.any(np.maximum(a, b) > upper + 1e-10):
            return False
        if not len(self._occupied):
            return True
        lo = np.minimum(a, b) - clearance_m
        hi = np.maximum(a, b) + clearance_m
        r0, c0 = self.world_to_cell(lo)
        r1, c1 = self.world_to_cell(hi)
        r0, c0 = max(0, r0), max(0, c0)
        r1, c1 = min(self.height - 1, r1), min(self.width - 1, c1)
        cells = np.argwhere(self._blocked[r0 : r1 + 1, c0 : c1 + 1])
        cells += (r0, c0)
        if not len(cells):
            return True
        rect_lo = np.column_stack((cells[:, 1], cells[:, 0])) * self.resolution_m + self.origin_xy_m
        rect_hi = rect_lo + self.resolution_m
        delta = b - a
        entry = np.zeros(len(cells))
        exit_time = np.ones(len(cells))
        intersects = np.ones(len(cells), dtype=bool)
        for axis in (0, 1):
            if abs(delta[axis]) < 1e-14:
                intersects &= (a[axis] >= rect_lo[:, axis]) & (a[axis] <= rect_hi[:, axis])
            else:
                x = (rect_lo[:, axis] - a[axis]) / delta[axis]
                y = (rect_hi[:, axis] - a[axis]) / delta[axis]
                entry = np.maximum(entry, np.minimum(x, y))
                exit_time = np.minimum(exit_time, np.maximum(x, y))
        if np.any(intersects & (entry <= exit_time)):
            return False
        distance_sq = np.full(len(cells), np.inf)
        for point in (a, b):
            difference = np.maximum(np.maximum(rect_lo - point, point - rect_hi), 0)
            distance_sq = np.minimum(distance_sq, np.sum(difference * difference, axis=1))
        length_sq = float(np.dot(delta, delta))
        for x, y in ((0, 0), (0, 1), (1, 0), (1, 1)):
            corner = np.column_stack((rect_hi[:, 0] if x else rect_lo[:, 0], rect_hi[:, 1] if y else rect_lo[:, 1]))
            fraction = np.clip((corner - a) @ delta / length_sq, 0, 1) if length_sq else np.zeros(len(cells))
            difference = corner - (a + fraction[:, None] * delta)
            distance_sq = np.minimum(distance_sq, np.sum(difference * difference, axis=1))
        return bool(np.all(distance_sq > clearance_m * clearance_m + 1e-12))

    def plan(self, start, goal, radius_m: float, arrival_tolerance_m: float, margin_m: float = 0.05):
        """Plan a disk-safe A* path [m], ending inside the original arrival ball [m]."""
        clearance = radius_m + margin_m
        if radius_m <= 0 or margin_m < 0 or arrival_tolerance_m <= 0:
            raise ValueError("Planning radius/tolerance must be positive and margin nonnegative [m].")
        start, goal = tuple(start[:2]), tuple(goal[:2])
        if not self.segment_clear(start, start, clearance):
            raise ValueError("No reachable path: actual start is outside disk-clear free space.")
        if math.dist(start, goal) <= arrival_tolerance_m:
            return [start, start]
        if self.segment_clear(start, goal, clearance):
            return [start, goal]
        limit = "the original arrival tolerance"
        if self.goal_projection_tolerance_m is not None and self.goal_projection_tolerance_m < arrival_tolerance_m:
            arrival_tolerance_m = self.goal_projection_tolerance_m
            limit = f"the configured goal projection tolerance ({arrival_tolerance_m:g} m)"
        mask = self.free_mask(clearance)
        seeds = self._start_cells(start, clearance, mask)
        endpoints = self._goal_endpoints(goal, arrival_tolerance_m, mask, clearance)
        if not seeds or not endpoints:
            raise ValueError(f"No reachable path inside {limit}.")
        if not _anchors_connected(mask, seeds, endpoints):
            raise ValueError(f"No reachable path to {limit}.")
        queue = []
        costs = {}
        parent = {}
        for cell in seeds:
            cost = math.dist(start, self.cell_center(cell))
            costs[cell] = cost
            parent[cell] = None
            heapq.heappush(
                queue, (cost + max(0, math.dist(self.cell_center(cell), goal) - arrival_tolerance_m), cost, cell)
            )
        found = None
        while queue:
            _, cost, cell = heapq.heappop(queue)
            if cost > costs[cell] + 1e-10:
                continue
            if cell in endpoints:
                found = cell
                break
            r, c = cell
            for dr, dc in _NEIGHBOR_OFFSETS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.height and 0 <= nc < self.width) or not mask[nr, nc]:
                    continue
                if dr and dc and not (mask[r + dr, c] and mask[r, c + dc]):
                    continue
                # Raster rectangles attain their nearest distance to this
                # neighbor rectangle at one of its free corner centers.
                # The diagonal's two extra corners prohibit cutting corners.
                neighbor = nr, nc
                new_cost = cost + self.resolution_m * math.hypot(dr, dc)
                if new_cost >= costs.get(neighbor, math.inf) - 1e-10:
                    continue
                costs[neighbor], parent[neighbor] = new_cost, cell
                heuristic = max(0, math.dist(self.cell_center(neighbor), goal) - arrival_tolerance_m)
                heapq.heappush(queue, (new_cost + heuristic, new_cost, neighbor))
        if found is None:
            raise ValueError(f"No reachable path to {limit}.")
        path = []
        cell = found
        while cell is not None:
            path.append(self.cell_center(cell))
            cell = parent[cell]
        path = [start, *reversed(path)]
        path.append(endpoints[found])
        if self.segment_clear(path[-1], goal, clearance):
            path.append(goal)
        simplified = [path[0]]
        index = 0
        while index < len(path) - 1:
            selected = None
            for candidate in range(len(path) - 1, index, -1):
                if self.segment_clear(path[index], path[candidate], clearance):
                    selected = candidate
                    break
            if selected is None:
                raise ValueError("No reachable path with an exact swept-disk certificate.")
            simplified.append(path[selected])
            index = selected
        return simplified

    def _start_cells(self, start, clearance_m: float, mask: np.ndarray) -> list[tuple[int, int]]:
        """Connect actual start [m] to bounded raster anchors with full disk checks."""
        seed_radius = max(2 * self.resolution_m, self.resolution_m * math.sqrt(2))
        seeds = self._near_cells(start, seed_radius, mask)
        seeds = [cell for cell in seeds if self.segment_clear(start, self.cell_center(cell), clearance_m)]
        if not seeds and clearance_m > seed_radius:
            # A safe subcell pose may need a farther, still-certified center anchor.
            seeds = self._near_cells(start, clearance_m, mask)
            seeds = [cell for cell in seeds if self.segment_clear(start, self.cell_center(cell), clearance_m)]
        return seeds

    def _reachable_mask(self, start, clearance_m: float) -> np.ndarray | None:
        """Return the actual-start component at clearance [m], or None without anchors.

        Missing raster anchors leave continuous-space connectivity undecided.
        """
        mask = self.free_mask(clearance_m)
        seeds = self._start_cells(start, clearance_m, mask)
        if not seeds:
            return None
        if type(self) is OccupancyGrid and not len(self._occupied):
            reachable = mask.copy()
        else:
            reachable = np.zeros_like(mask)
            pending = deque(seeds)
            for cell in seeds:
                reachable[cell] = True
            while pending:
                row, column = pending.popleft()
                for dr, dc in _NEIGHBOR_OFFSETS:
                    nr, nc = row + dr, column + dc
                    if not (0 <= nr < self.height and 0 <= nc < self.width) or reachable[nr, nc] or not mask[nr, nc]:
                        continue
                    if dr and dc and not (mask[nr, column] and mask[row, nc]):
                        continue
                    reachable[nr, nc] = True
                    pending.append((nr, nc))
        reachable.flags.writeable = False
        return reachable

    def _goal_cache_base(self, goal, tolerance, clearance):
        """Return an exact immutable grid eligible for static goal-region reuse."""
        return self if type(self) is OccupancyGrid else None

    def _goal_endpoints(self, goal, tolerance, mask, clearance):
        cache_key = None
        if type(self) is OccupancyGrid and mask is self.free_mask(clearance):
            cache_key = (
                tuple(goal),
                tolerance,
                clearance,
                self.resolution_m,
                self.origin_xy_m,
                self.height,
                self.width,
            )
            cached = self._goal_endpoint_cache.get(cache_key)
            if cached is not None and cached[0] is mask:
                self._goal_endpoint_cache.move_to_end(cache_key)
                return cached[1].copy()
        endpoints = {}
        for cell in self._near_cells(goal, tolerance + self.resolution_m * math.sqrt(2), mask):
            center = np.asarray(self.cell_center(cell))
            if math.dist(center, goal) <= tolerance - min(1e-6, tolerance * 1e-4):
                endpoints[cell] = tuple(center)
                continue
            # A free-center anchor can certify an endpoint in a neighboring
            # cell whose center is blocked. Do not clip the arrival ball.
            nearest = np.asarray(goal, dtype=float)
            if not self.segment_clear(center, nearest, clearance):
                lower, upper = 0.0, 1.0
                for _ in range(24):
                    middle = (lower + upper) / 2
                    candidate = nearest + (center - nearest) * middle
                    if self.segment_clear(center, candidate, clearance):
                        upper = middle
                    else:
                        lower = middle
                nearest = nearest + (center - nearest) * upper
            if math.dist(nearest, goal) <= tolerance + 1e-10:
                endpoints[cell] = tuple(nearest)
        if cache_key is not None and len(endpoints) <= _GOAL_CACHE_ANCHOR_LIMIT:
            # Bound reuse across alternating actors and arbitrary wander targets.
            self._goal_endpoint_cache[cache_key] = (mask, endpoints.copy())
            self._goal_endpoint_cache.move_to_end(cache_key)
            if len(self._goal_endpoint_cache) > _GOAL_CACHE_CAPACITY:
                self._goal_endpoint_cache.popitem(last=False)
        return endpoints

    def _near_cells(self, point, distance, mask):
        row, column = self.world_to_cell(point)
        reach = math.ceil(distance / self.resolution_m) + 1
        return [
            (r, c)
            for r in range(max(0, row - reach), min(self.height, row + reach + 1))
            for c in range(max(0, column - reach), min(self.width, column + reach + 1))
            if mask[r, c] and math.dist(self.cell_center((r, c)), point) <= distance + 1e-10
        ]

    def wall_segments(self) -> np.ndarray:
        """Return merged static cell-boundary segments [m], shape [N, 4]."""
        edges: dict[tuple[str, int, int], list[int]] = {}
        for row, column in self._occupied:
            for dr, dc, axis, coordinate, index, direction in (
                (-1, 0, "h", row, column, 1),
                (1, 0, "h", row + 1, column, -1),
                (0, -1, "v", column, row, -1),
                (0, 1, "v", column + 1, row, 1),
            ):
                nr, nc = row + dr, column + dc
                if 0 <= nr < self.height and 0 <= nc < self.width and not self._blocked[nr, nc]:
                    edges.setdefault((axis, int(coordinate), direction), []).append(int(index))
        segments = []
        for (axis, coordinate, direction), indices in sorted(edges.items()):
            indices.sort()
            begin = previous = indices[0]
            for index in [*indices[1:], None]:
                if index is not None and index == previous + 1:
                    previous = index
                    continue
                a, b = (
                    ((begin, coordinate), (previous + 1, coordinate))
                    if axis == "h"
                    else ((coordinate, begin), (coordinate, previous + 1))
                )
                if direction < 0:
                    a, b = b, a
                segments.append(
                    (
                        *np.asarray(a) * self.resolution_m + self.origin_xy_m,
                        *np.asarray(b) * self.resolution_m + self.origin_xy_m,
                    )
                )
                if index is not None:
                    begin = previous = index
        x, y = self.origin_xy_m
        right, top = x + self.width * self.resolution_m, y + self.height * self.resolution_m
        segments.extend(((x, y, right, y), (right, y, right, top), (right, top, x, top), (x, top, x, y)))
        return np.asarray(segments, dtype=np.float64).reshape(-1, 4)
