"""SE(2) lattice A*: full source envelope + margin on raw observed squares."""
import heapq,math,time
import numpy as np
from drlvo_robot_static_planner import RobotStaticPlanner


def wrap(yaw):return (yaw+math.pi)%(2*math.pi)-math.pi


DIRECTIONS = [(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1)]
HEADINGS = [math.atan2(-dy, dx) % (2*math.pi) for dy, dx in DIRECTIONS]


class HeadingPlanner(RobotStaticPlanner):
    def __init__(self,native,lower,upper,*,planning_tracking_margin=0.):
        super().__init__(native)
        self._walkable=(self._labels==254)
        self._walkable.setflags(write=False)
        self.lower,self.upper=np.asarray(lower),np.asarray(upper)
        self.radius=float(np.linalg.norm(np.maximum(abs(self.lower[:2]),abs(self.upper[:2]))))
        self.pose_cache={};self.motion_cache={};self.current_yaw=math.pi/2;self.last_poses=None
        self.expanded=0
        self.reference_poses=None
        self.reference_only=False
        self.planning_tracking_margin=float(planning_tracking_margin)
        if not math.isfinite(self.planning_tracking_margin) or self.planning_tracking_margin<0:
            raise ValueError('Planning tracking allowance must be finite and nonnegative')

    def is_walkable(self, xy):
        if not np.isfinite(xy).all():return False
        r,c=self.world_to_pixel(xy)
        return bool(0<=r<self._labels.shape[0] and 0<=c<self._labels.shape[1] and self._labels[r,c]==254)

    def pose_clear(self,xy,yaw,margin=.05):
        return self.field.footprint(xy,yaw,self.lower,self.upper,margin=margin)['clear']

    def sweep_clear(self,a,b,*,extra_margin=0.):
        a,b=np.asarray(a,float),np.asarray(b,float)
        # Exact sufficient bound: every rotated envelope lies in its source
        # circumcircle, and centre displacement cannot exceed the segment length.
        # This only avoids redundant samples in wide-open space, with no margin change.
        if self.field.distance_at(a[:2]) >= self.radius+.030+extra_margin+np.linalg.norm(b[:2]-a[:2]):return True
        delta=wrap(b[2]-a[2]);travel=float(np.linalg.norm(b[:2]-a[:2]))+self.radius*abs(delta)
        steps=max(1,math.ceil(travel/.01))
        # Every interpolated pose is within .005m pointwise displacement of
        # a sampled envelope. The .030 m sampled gap includes .005 m between-sample allowance (continuous geometric gap >= .025 m).
        for t in np.linspace(0,1,steps+1):
            if not self.pose_clear(a[:2]+t*(b[:2]-a[:2]),a[2]+t*delta,.030+extra_margin):return False
        return True

    def connector(self,start,end,*,extra_margin=0.):
        start,end=np.asarray(start,float),np.asarray(end,float)
        d=end[:2]-start[:2]
        if np.linalg.norm(d)>1e-10:
            heading=math.atan2(d[1],d[0])
            poses=np.array([start,[*start[:2],heading],[*end[:2],heading],end])
        else:poses=np.array([start,end])
        return poses if all(self.sweep_clear(a,b,extra_margin=extra_margin) for a,b in zip(poses[:-1],poses[1:])) else None

    def _pose(self,node):return np.r_[self.pixel_to_world(np.array(node[:2])),HEADINGS[node[2]]]

    def _valid(self,node):
        if node not in self.pose_cache:
            self.pose_cache[node]=bool(self.field.distance[node[:2]]>=self.radius+.030+self.planning_tracking_margin) or self.pose_clear(self._pose(node)[:2],HEADINGS[node[2]],.030+self.planning_tracking_margin)
        return self.pose_cache[node]

    def astar_path(self,start_xy,goal_xy,*,time_budget_s=None):
        deadline=None if time_budget_s is None else time.monotonic()+time_budget_s
        self.last_poses=None
        self.expanded=0
        start_xy,goal_xy=np.asarray(start_xy,float),np.asarray(goal_xy,float)
        if not self.is_walkable(start_xy) or not self.is_walkable(goal_xy):return None
        if self.reference_poses is not None and np.linalg.norm(goal_xy-self.reference_poses[-1,:2])<1e-5:
            # Reuse the certified A* route with a freshly swept connector from
            # the measured robot pose. Never replace it with final-goal bearing.
            distances=np.linalg.norm(self.reference_poses[:,:2]-start_xy,axis=1)
            closest=int(np.argmin(distances))
            # A .4m join can omit the first heading-safe point in a .1m grid.
            # Search up to the existing 1m lookahead, then certify each join.
            candidates=np.flatnonzero((distances<=1.)&(np.arange(len(distances))>=closest))
            candidates=sorted(candidates,key=lambda i:abs(distances[i]-.15))
            for i in candidates:
                connection=self.connector(np.r_[start_xy,self.current_yaw],self.reference_poses[i])
                if connection is None:continue
                poses=np.concatenate([connection,self.reference_poses[i+1:]])
                self.last_poses=poses
                xy=poses[:,:2]
                return xy[np.r_[True,np.linalg.norm(np.diff(xy,axis=0),axis=1)>1e-10]]
            if self.reference_only:
                # An explicitly routed corner detour must not silently become
                # the old shorter route that caused the measured turn failure.
                self.last_poses=None
                return None
        start,goal=tuple(self.world_to_pixel(start_xy)),tuple(self.world_to_pixel(goal_xy))
        costs={};parents={};starts={};q=[];seen=set()
        # Attach the real pose to all safe states in a bounded 3x3 neighborhood.
        # A nearest-cell heading can be valid but have no outgoing safe edge.
        seed_nodes=[]
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                r,c=start[0]+dr,start[1]+dc
                if not (0<=r<self._labels.shape[0] and 0<=c<self._labels.shape[1]):continue
                if self._walkable[r,c]:seed_nodes.extend((r,c,h) for h in range(8))
        for node in seed_nodes:
            if deadline is not None and time.monotonic()>=deadline:return None
            h=node[2]
            if not self._valid(node):continue
            connection=self.connector(np.r_[start_xy,self.current_yaw],self._pose(node),extra_margin=self.planning_tracking_margin)
            if connection is None:continue
            cost=.15*abs(wrap(HEADINGS[h]-self.current_yaw))+float(np.linalg.norm(start_xy-self._pose(node)[:2]))
            costs[node]=cost;starts[node]=connection
            heapq.heappush(q,(cost+math.dist(node[:2],goal)*.1,cost,node))
        self.expanded=0
        # Directions match world +X/+Y (image row decreases toward +Y).
        directions=DIRECTIONS
        while q:
            if deadline is not None and time.monotonic()>=deadline:return None
            _,cost,node=heapq.heappop(q)
            if node in seen:continue
            seen.add(node);self.expanded+=1
            if node[:2]==goal:
                end=self._pose(node);finish=self.connector(end,np.r_[goal_xy,end[2]],extra_margin=self.planning_tracking_margin)
                if finish is not None:
                    nodes=[node]
                    while nodes[-1] in parents:nodes.append(parents[nodes[-1]])
                    nodes=nodes[::-1]
                    poses=np.concatenate([starts[nodes[0]],np.array([self._pose(p) for p in nodes[1:]]),finish[1:]])
                    poses=poses[np.r_[True,np.linalg.norm(np.diff(poses,axis=0),axis=1)>1e-10]]
                    self.last_poses=poses
                    xy=poses[:,:2];return xy[np.r_[True,np.linalg.norm(np.diff(xy,axis=0),axis=1)>1e-10]]
            row,col,h=node;dy,dx=directions[h]
            candidates=[((row,col,(h-1)%8),.15*abs(wrap(HEADINGS[(h-1)%8]-HEADINGS[h]))),((row,col,(h+1)%8),.15*abs(wrap(HEADINGS[(h+1)%8]-HEADINGS[h]))),((row+dy,col+dx,h),.1*math.hypot(dy,dx))]
            for nxt,step in candidates:
                r,c,nh=nxt
                if not (0<=r<self._walkable.shape[0] and 0<=c<self._walkable.shape[1]) or not self._walkable[r,c] or not self._valid(nxt):continue
                key=tuple(sorted((node,nxt)))
                if key not in self.motion_cache:self.motion_cache[key]=self.sweep_clear(self._pose(node),self._pose(nxt),extra_margin=self.planning_tracking_margin)
                if not self.motion_cache[key]:continue
                clearance=float(self.field.distance[r,c]);new=cost+step+(.002/max(.03,clearance-.2735) if (r,c)!=(row,col) else 0.)
                if new<costs.get(nxt,math.inf):
                    costs[nxt]=new;parents[nxt]=node;heapq.heappush(q,(new+math.dist(nxt[:2],goal)*.1,new,nxt))
        self.last_poses=None
        return None
