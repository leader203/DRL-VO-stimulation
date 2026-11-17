"""Robot centre A* on raw closed occupancy; independent of pedestrian clearance.

Inflate once by .2235 + .05 m. Exact cell-square distances refine EDT bounds;
continuous edges are checked against raw cell squares, never inflated cells.
"""
import heapq
import math
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

STATIC_INFLATION_M = .2735
FOOTPRINT_MARGIN_M = .05


def segment_box_distances(a, b, centers, half=.05):
    """Exact Euclidean distance from a segment to each closed square."""
    a, b, centers = np.asarray(a), np.asarray(b), np.asarray(centers)
    if not len(centers):
        return np.empty(0)
    low, high = centers-half, centers+half
    delta = b-a
    endpoint = np.minimum(np.linalg.norm(np.maximum(np.maximum(low-a,a-high),0.),axis=1),
                          np.linalg.norm(np.maximum(np.maximum(low-b,b-high),0.),axis=1))
    corners = centers[:,None,:] + np.array([[-half,-half],[-half,half],[half,-half],[half,half]])
    norm = float(delta@delta)
    t = np.clip(((corners-a)@delta)/norm,0.,1.) if norm > 0 else np.zeros(corners.shape[:2])
    corner_distance = np.linalg.norm(corners-a-t[...,None]*delta,axis=-1).min(axis=1)
    enter, leave = np.zeros(len(centers)), np.ones(len(centers))
    possible = np.ones(len(centers),bool)
    for axis in range(2):
        if abs(delta[axis]) < 1e-15:
            possible &= (a[axis] >= low[:,axis]) & (a[axis] <= high[:,axis])
        else:
            t0, t1 = (low[:,axis]-a[axis])/delta[axis], (high[:,axis]-a[axis])/delta[axis]
            enter = np.maximum(enter,np.minimum(t0,t1))
            leave = np.minimum(leave,np.maximum(t0,t1))
    return np.where(possible & (enter <= leave),0.,np.minimum(endpoint,corner_distance))


class RawObstacleField:
    def __init__(self, native):
        self.native = native
        self.resolution = native.resolution
        if self.resolution != .1:
            raise ValueError('This task requires .1m source resolution')
        self.half = self.resolution/2
        self.half_diagonal = math.sqrt(2)*self.half
        self.centers = native.pixel_to_world(np.argwhere(native._labels != 254))
        self.tree = cKDTree(self.centers)
        free = native._labels == 254
        self.edt_center_distance = distance_transform_edt(np.pad(free,1,constant_values=False))[1:-1,1:-1]*.1
        self.distance = np.zeros(free.shape)
        cells = np.argwhere(free)
        self.distance[tuple(cells.T)] = self.distance_at(native.pixel_to_world(cells))

    def distance_at(self, points):
        points = np.asarray(points,dtype=float)
        shape = points.shape[:-1]; points = points.reshape(-1,2)
        if not np.isfinite(points).all():
            raise ValueError('Finite world points required')
        nearest, _ = self.tree.query(points)
        # A square is within half-diagonal of its centre. Cells outside this
        # exact search radius cannot improve the nearest square distance.
        candidates = self.tree.query_ball_point(points,nearest+self.half_diagonal+1e-12)
        result = []
        bounds = self.native._bounds
        for point, ids in zip(points,candidates):
            squares = self.centers[ids]
            gap = np.maximum(np.abs(squares-point)-self.half,0.)
            cell_distance = float(np.linalg.norm(gap,axis=1).min())
            boundary = max(0.,float(min(*(point-bounds[:2]),*(bounds[2:]-point))))
            result.append(min(cell_distance,boundary))
        return np.asarray(result).reshape(shape)

    def segment_clear(self, a, b, radius):
        a,b=np.asarray(a),np.asarray(b)
        bounds=self.native._bounds
        if min(*(a-bounds[:2]),*(bounds[2:]-a),*(b-bounds[:2]),*(bounds[2:]-b)) < radius:
            return False
        ids=self.tree.query_ball_point((a+b)/2,float(np.linalg.norm(b-a)/2)+radius+self.half_diagonal)
        return bool(np.all(segment_box_distances(a,b,self.centers[ids],self.half) >= radius-1e-12))

    def footprint(self, xy, yaw, lower, upper, margin=FOOTPRINT_MARGIN_M):
        """Oriented source rectangle + one .05m margin against RAW squares."""
        xy=np.asarray(xy);lower=np.asarray(lower);upper=np.asarray(upper)
        rect=np.array([[lower[0],lower[1]],[upper[0],lower[1]],
                       [upper[0],upper[1]],[lower[0],upper[1]]])
        rotation=np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
        polygon=rect@rotation.T+xy
        bounds=self.native._bounds
        exterior=max(0.,float(min(*(polygon.min(0)-bounds[:2]),*(bounds[2:]-polygon.max(0)))))
        radius=float(np.linalg.norm(np.maximum(abs(lower[:2]),abs(upper[:2]))))+margin
        ids=self.tree.query_ball_point(xy,radius+self.half_diagonal)
        centers=self.centers[ids]
        if len(centers):
            distances=np.full(len(centers),np.inf)
            for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
                distances=np.minimum(distances,segment_box_distances(a,b,centers,self.half))
            # A square entirely contained in the robot rectangle does not
            # intersect its boundary, but still overlaps its interior.
            local=(centers-xy)@rotation
            inside=((local>=lower[:2])&(local<=upper[:2])).all(axis=1)
            distances[inside]=0.
            closest=int(np.argmin(distances));minimum=min(exterior,float(distances[closest]))
            obstacle=centers[closest].tolist()
        else:
            minimum=exterior;obstacle=None
        return dict(clear=minimum>=margin-1e-12,clearance_lower_bound_m=min(minimum,margin),required_margin_m=margin,
                    nearest_obstacle_cell_xy=obstacle,polygon_xy=polygon.tolist())


class RobotStaticPlanner:
    def __init__(self,native,field=None):
        self.native=native;self.field=field or RawObstacleField(native)
        self.clearance_m=STATIC_INFLATION_M
        self._labels=native._labels;self._bounds=native._bounds;self.resolution=native.resolution
        self._walkable=(self._labels==254)&(self.field.distance>=self.clearance_m)
        self._walkable.setflags(write=False)
        self.world_to_pixel=native.world_to_pixel;self.pixel_to_world=native.pixel_to_world
        self._edge_cache={}

    def is_walkable(self,xy):
        return self.field.distance_at(xy)>=self.clearance_m

    def astar_path(self,start_xy,goal_xy):
        start_xy,goal_xy=np.asarray(start_xy),np.asarray(goal_xy)
        if not self.is_walkable(start_xy) or not self.is_walkable(goal_xy):return None
        start,goal=tuple(self.world_to_pixel(start_xy)),tuple(self.world_to_pixel(goal_xy))
        if not self.field.segment_clear(start_xy,self.pixel_to_world(np.array(start)),self.clearance_m):return None
        if not self.field.segment_clear(goal_xy,self.pixel_to_world(np.array(goal)),self.clearance_m):return None
        costs={start:0.};parents={};seen=set();queue=[(math.dist(start,goal),0.,start)]
        while queue:
            _,cost,node=heapq.heappop(queue)
            if node in seen:continue
            seen.add(node)
            if node==goal:
                nodes=[goal]
                while nodes[-1]!=start:nodes.append(parents[nodes[-1]])
                route=np.concatenate((start_xy[None],self.pixel_to_world(np.array(nodes[::-1])),goal_xy[None]))
                return route[np.r_[True,np.linalg.norm(np.diff(route,axis=0),axis=1)>1e-12]]
            for dy,dx in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)):
                nxt=(node[0]+dy,node[1]+dx)
                if not (0<=nxt[0]<self._walkable.shape[0] and 0<=nxt[1]<self._walkable.shape[1]) or not self._walkable[nxt]:continue
                if dy and dx and not (self._walkable[node[0]+dy,node[1]] and self._walkable[node[0],node[1]+dx]):continue
                edge=tuple(sorted((node,nxt)))
                if edge not in self._edge_cache:
                    a,b=self.pixel_to_world(np.array([node,nxt]))
                    self._edge_cache[edge]=self.field.segment_clear(a,b,self.clearance_m)
                if not self._edge_cache[edge]:continue
                new_cost=cost+math.hypot(dy,dx)
                if new_cost<costs.get(nxt,math.inf):
                    costs[nxt]=new_cost;parents[nxt]=node
                    heapq.heappush(queue,(new_cost+math.dist(nxt,goal),new_cost,nxt))
        return None
