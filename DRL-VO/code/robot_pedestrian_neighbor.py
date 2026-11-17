"""Preview-only robot neighbor and transient obstacle raster; no source map edits."""
import math
import torch


def robot_motion_segment(observation, horizon_s=1.2, min_speed=.05):
    """Bounded constant measured velocity; stationary noise creates no sweep."""
    import numpy as np
    xy = np.asarray(observation['root_position_m'][:2], dtype=np.float64)
    velocity = np.asarray(observation.get('linear_velocity_world', [0., 0., 0.])[:2], dtype=np.float64)
    if not np.isfinite(xy).all() or not np.isfinite(velocity).all():
        raise ValueError('Robot prediction requires finite measured XY and velocity')
    if not math.isfinite(horizon_s) or horizon_s < 0:
        raise ValueError('Prediction horizon must be finite and nonnegative')
    end = xy+velocity*horizon_s if np.linalg.norm(velocity) >= min_speed else xy.copy()
    return xy, end


def point_segment_distance(points, start, end):
    import numpy as np
    delta = end-start
    fraction = np.clip(((points-start)*delta).sum(-1)/max(float(delta@delta), 1e-12), 0., 1.)
    return np.linalg.norm(points-(start+fraction[..., None]*delta), axis=-1)


def polyline_sweep_gap(paths, start, end):
    """Exact 2D segment distance, including crossing segments and zero speed."""
    import numpy as np
    delta = np.diff(paths, axis=1)
    denominator = np.maximum((delta*delta).sum(-1), 1e-12)
    gap = point_segment_distance(paths, start, end).min(-1)
    for endpoint in (start, end):
        fraction = np.clip(((endpoint-paths[:, :-1])*delta).sum(-1)/denominator, 0., 1.)
        nearest = paths[:, :-1]+fraction[..., None]*delta
        gap = np.minimum(gap, np.linalg.norm(nearest-endpoint, axis=-1).min(-1))
    motion = end-start
    def cross(a, b):
        return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]
    divisor = cross(delta, motion)
    safe = np.where(np.abs(divisor)>1e-12, divisor, 1.)
    offset = start-paths[:, :-1]
    t, u = cross(offset, motion)/safe, cross(offset, delta)/safe
    crossing = (np.abs(divisor)>1e-12)&(t>=0)&(t<=1)&(u>=0)&(u<=1)
    return np.where(crossing.any(-1), 0., gap)


def temporal_robot_gap(states, trajectory, observation, horizon_s=1.2):
    """Continuous relative-segment separation at matching future times."""
    steps = min(trajectory.shape[-2], round(horizon_s/.1))
    xy = trajectory[..., :steps, :2]
    c,s = states[:,3,None,None].cos(),states[:,3,None,None].sin()
    world = torch.stack((xy[...,0]*c-xy[...,1]*s,
                         xy[...,0]*s+xy[...,1]*c),-1)+states[:,None,None,:2]
    robot = states.new_tensor(observation['actual_xy'])
    velocity = states.new_tensor(observation['measured_velocity_xy'])
    if not observation['active']:
        velocity = velocity*0
    times = torch.arange(1,steps+1,device=xy.device,dtype=xy.dtype)*.1
    relative = world-robot-velocity*times[None,None,:,None]
    initial = (states[:,:2]-robot)[:,None,None].expand(-1,trajectory.shape[1],1,-1)
    relative = torch.cat((initial,relative),-2)
    delta = torch.diff(relative,dim=-2)
    alpha = (-relative[...,:-1,:]*delta).sum(-1)/delta.square().sum(-1).clamp_min(1e-12)
    closest = relative[...,:-1,:]+alpha.clamp(0.,1.)[...,None]*delta
    return closest.square().sum(-1).clamp_min(1e-12).sqrt().amin(-1)


def command_conflict(states, observation, v, *, horizon_s=1.5, clearance_m=1.46):
    """Reject wheel translation toward a predicted close pedestrian encounter."""
    robot_xy = states.new_tensor(observation['root_position_m'][:2])
    yaw = observation['yaw_rad']
    velocity = states.new_tensor([v*math.cos(yaw), v*math.sin(yaw)])
    relative = states[:,:2]-robot_xy
    change = states[:,2:4]-velocity
    time = (-(relative*change).sum(-1)/change.square().sum(-1).clamp_min(1e-8)).clamp(0.,horizon_s)
    gap = (relative+change*time[:,None]).norm(dim=-1)
    toward = (relative*velocity).sum(-1)>0.
    blocked = toward & (gap<clearance_m)
    return bool(blocked.any()), gap.tolist()


def augment_packet(packet, observation, radius=.6, *, robot_in_raster=True):
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('Positive finite robot radius required')
    result = dict(packet)
    states = packet['states']
    batch = len(states)
    xy = states.new_tensor(observation['root_position_m'][:2])
    velocity = states.new_tensor(observation['linear_velocity_world'][:2])
    speed = velocity.norm()
    heading = torch.atan2(velocity[1], velocity[0]) if speed >= .1 else states.new_tensor(observation['yaw_rad'])
    # Only current measured state is available here. Never invent past motion.
    neighbor = states.new_zeros(batch, 1, 31, 4)
    neighbor[:, 0, -1, :2] = xy
    neighbor[:, 0, -1, 2] = speed
    neighbor[:, 0, -1, 3] = heading
    mask = torch.zeros(batch, 1, 31, dtype=torch.bool, device=states.device)
    mask[:, :, -1] = True
    old = packet['neighbors']
    old_mask = packet.get('neighbor_mask', torch.ones(old.shape[:-1], dtype=torch.bool, device=states.device))
    result['neighbors'] = torch.cat((old, neighbor), 1)
    result['neighbor_mask'] = torch.cat((old_mask, mask), 1)
    ego_extent = packet.get('extents', states.new_tensor([.5, .5, 1.7]).expand(batch, 3))
    old_extent = packet.get('neighbor_extents', ego_extent[:, None].expand(batch, old.shape[1], 3))
    # Guidance uses half the XY diagonal as radius.
    extent = states.new_tensor([math.sqrt(2)*radius, math.sqrt(2)*radius, 1.0]).expand(batch, 1, 3)
    result['neighbor_extents'] = torch.cat((old_extent, extent), 1)
    if not robot_in_raster:
        # Moving occupancy is checked at matching times by the explicit
        # preview guidance. It must not become a permanent static wall.
        return result
    # Existing guidance costs only ego-to-ego pairs; put the measured robot disc
    # in a copied crop too so map cost and trajectory admission include it.
    raster = packet['raster'].clone()
    start, end = robot_motion_segment(observation)
    delta = xy[None] - states[:, :2]
    future_delta = states.new_tensor(end)[None]-states[:, :2]
    c, s = states[:, 3].cos(), states[:, 3].sin()
    lx = delta[:, 0]*c + delta[:, 1]*s
    ly = -delta[:, 0]*s + delta[:, 1]*c
    ex = future_delta[:, 0]*c+future_delta[:, 1]*s
    ey = -future_delta[:, 0]*s+future_delta[:, 1]*c
    rows, cols = torch.meshgrid(torch.arange(224, device=states.device), torch.arange(224, device=states.device), indexing='ij')
    px, py = (cols[None]-56)/12, (rows[None]-112)/12
    if (start==end).all():
        disk = (px-lx[:, None, None]).square()+(py-ly[:, None, None]).square() <= (radius+math.sqrt(2)/24)**2
    else:
        dx, dy = (ex-lx)[:, None, None], (ey-ly)[:, None, None]
        alpha = (((px-lx[:, None, None])*dx+(py-ly[:, None, None])*dy)
                 /(dx.square()+dy.square()).clamp_min(1e-12)).clamp(0., 1.)
        disk = (px-lx[:, None, None]-alpha*dx).square()+(py-ly[:, None, None]-alpha*dy).square() <= (radius+math.sqrt(2)/24)**2
    raster[:, 0].masked_fill_(disk, 0)
    raster[:, 1].masked_fill_(disk, 1)
    result['raster'] = raster
    return result


def install_robot_neighbor(robot, report, radius=.6, *, temporal_admission=False):
    import CrowdSim.humanoid.runtime.trace_pace as module
    original = module.trace_inputs_from_world
    def inputs(world, **kwargs):
        observation = robot.observe()
        packet = augment_packet(original(world, **kwargs), observation, radius,
                                robot_in_raster=not temporal_admission)
        report['robot_neighbor_updates'] = report.get('robot_neighbor_updates', 0)+1
        report['robot_neighbor_last_xy'] = observation['root_position_m'][:2]
        start, end = robot_motion_segment(observation)
        report['robot_motion_prediction'] = dict(actual_xy=start.tolist(), predicted_xy=end.tolist(),
            measured_velocity_xy=observation['linear_velocity_world'][:2], horizon_s=1.2,
            active=bool((end!=start).any()), model='bounded constant actual velocity; swept obstacle',
            perception_source='simulator actual pose/velocity; no lidar or visual detection')
        return packet
    module.trace_inputs_from_world = inputs
    report.update(robot_in_pedestrian_planning=True, robot_avoidance_radius_m=radius,
        robot_neighbor_state='actual physical pose and velocity; current frame available, past masked',
        robot_avoidance_layers=['TRACE current neighbor conditioning',
            '1.2s sweep for global detour; time-aligned robot admission; static room raster unchanged'
            if temporal_admission else '1.2s measured-velocity swept obstacle in detour and map admission'],
        robot_physical_geometry_changed=False, robot_avoidance_behavior_validated=False)


def detour_targets(states, targets, scene, observation, radius=.6, pedestrian_radius=.46, report=None,
                   robot_forecast_as_static=True):
    """A* on the current known raster plus robot disc, inflated for ego body.

    Targets remain in the authored free domain. Empty result is a real failure,
    never a fabricated successful path. This is geometric planner input.
    """
    import heapq
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    output = targets.clone()
    # Preserve the timed patrol polyline when the robot is far away. Replacing
    # an entire cyclic horizon with a shortest path to its final point can
    # erase several metres of intended walking, even on an empty scene.
    robot_xy, robot_end = robot_motion_segment(observation)
    paths = np.concatenate((states[:, None, :2].detach().cpu().numpy(),
                            targets.detach().cpu().numpy()), axis=1)
    gaps = polyline_sweep_gap(paths, robot_xy, robot_end)
    affected = gaps < radius+pedestrian_radius+.35
    if report is not None:
        report['robot_detour_candidate_obstacle'] = ('entire forecast as static' if robot_forecast_as_static
                                                   else 'current robot disc; strict temporal admission owns future collision')
        report['robot_detour_route_gap_m'] = gaps.tolist()
        report['robot_detour_affected_actors'] = np.flatnonzero(affected).tolist()
        report['robot_route_change_rms_m'] = [0.]*len(states)
    if not affected.any():
        return output
    # Raster uses velocity heading here, matching TRACE's conversion.
    nav = states.clone()
    nav[:, 4] = torch.where(nav[:, 2:4].norm(dim=-1) >= .1,
                           torch.atan2(nav[:,3],nav[:,2]),nav[:,4])
    rasters = scene.raster(nav)
    for i in range(len(states)):
        if not affected[i]:
            continue
        yaw=float(nav[i,4]);c,s=math.cos(yaw),math.sin(yaw)
        origin=states[i,:2].detach().cpu().numpy()
        def local(xy):
            d=np.asarray(xy)-origin;return np.array([d[0]*c+d[1]*s,-d[0]*s+d[1]*c])
        robot=local(observation['root_position_m'][:2])
        # Generation and hard admission are separate. Under strict temporal
        # admission, the future sweep must not turn a currently safe start
        # into a permanent wall and silently retain the old direct targets.
        # Keep the same radii/comfort inflation; every generated trajectory
        # still passes the source joint gate plus full-horizon robot checks.
        future=local(robot_end if robot_forecast_as_static else robot_xy)
        yy,xx=np.mgrid[:224,:224]
        points_xy=np.stack(((xx-56)/12,(yy-112)/12),-1)
        disk=point_segment_distance(points_xy,robot,future) <= radius+math.sqrt(2)/24
        image=rasters[i].detach().cpu().numpy()
        # Native patrol starts use .55m wall clearance. Keep that physical
        # clearance here; the robot adds a separate larger comfort buffer.
        static_free=(image[0]>=.75)&(image[1]<=.25)
        wall_clear=distance_transform_edt(np.pad(static_free,1))[1:-1,1:-1]/12
        robot_clear=distance_transform_edt(np.pad(~disk,1))[1:-1,1:-1]/12
        allowed=(wall_clear>pedestrian_radius+.1)&(robot_clear>pedestrian_radius+.35)
        start=(112,56)
        if not allowed[start]:
            continue  # Current physical state needs the per-tick stop guard.

        def astar(begin,goal):
            if begin==goal:return [begin]
            queue=[(0.,begin)];cost={begin:0.};parent={};found=False
            while queue:
                _,cell=heapq.heappop(queue)
                if cell==goal:found=True;break
                for dr,dc in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                    nxt=(cell[0]+dr,cell[1]+dc)
                    if not (0<=nxt[0]<224 and 0<=nxt[1]<224 and allowed[nxt]):continue
                    if dr and dc and not (allowed[cell[0]+dr,cell[1]] and allowed[cell[0],cell[1]+dc]):continue
                    value=cost[cell]+math.hypot(dr,dc)
                    if value>=cost.get(nxt,float('inf')):continue
                    cost[nxt]=value;parent[nxt]=cell
                    heapq.heappush(queue,(value+math.hypot(nxt[0]-goal[0],nxt[1]-goal[1]),nxt))
            if not found:return None
            path=[goal]
            while path[-1]!=begin:path.append(parent[path[-1]])
            return path[::-1]

        # Connect the intermediate timed patrol points in order. A shortest
        # path to just the last point erases loops, especially near a robot.
        path=[start]
        for waypoint in targets[i].detach().cpu().numpy():
            end=local(waypoint)
            goal=(int(round(end[1]*12+112)),int(round(end[0]*12+56)))
            if not (0<=goal[0]<224 and 0<=goal[1]<224 and allowed[goal]):
                continue  # Skip points inside the inflated robot, keep the tail.
            segment=astar(path[-1],goal)
            if segment is None:
                break  # Retain the reachable prefix; hard admission still runs.
            path.extend(segment[1:])
        path=np.asarray(path)
        points=np.column_stack(((path[:,1]-56)/12,(path[:,0]-112)/12))
        length=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
        if length[-1]<.1:
            continue
        original_length=np.linalg.norm(np.diff(paths[i],axis=0),axis=1).sum()
        distances=np.minimum(np.linspace(original_length/targets.shape[1],original_length,targets.shape[1]),length[-1])
        chosen=np.column_stack([np.interp(distances,length,points[:,axis]) for axis in range(2)])
        world=origin+np.column_stack((chosen[:,0]*c-chosen[:,1]*s,chosen[:,0]*s+chosen[:,1]*c))
        output[i]=targets.new_tensor(world)
        if report is not None:
            report['robot_detour_rewritten_routes']=report.get('robot_detour_rewritten_routes',0)+1
            report['robot_route_change_rms_m'][i]=float((output[i]-targets[i]).square().sum(-1).mean().sqrt())
    return output
