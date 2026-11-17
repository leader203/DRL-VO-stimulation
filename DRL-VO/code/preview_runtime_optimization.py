"""Opt-in manual preview: admitted route candidates, CUDA AMP, timed CPU work.

This is a different pedestrian planner mode from the released TRACE-only run.
Source maps, physical rates, body/peer/map guards, AMP weights and scans stay
unchanged. A route candidate must pass the existing joint admission; otherwise
the original 100-step TRACE planner runs. No physical pose/velocity is written.
"""
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import replace
import json
import math
import time

import numpy as np
import torch


def parallel_guidance_class(original, workers=4, robot_prediction=None):
    """Same per-raster SciPy distance fields, computed on independent CPU cores."""
    from concurrent.futures import ThreadPoolExecutor
    from scipy.ndimage import distance_transform_edt
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='preview_map_sdf')

    def field(mask):
        if not mask.any():
            raise ValueError('No known walkable cells in the planning raster')
        padded = np.pad(mask, 1, constant_values=False)
        return (distance_transform_edt(padded)-distance_transform_edt(~padded))[1:-1, 1:-1]/12.

    class ParallelGuidance(original):
        def select(self, trajectory, **kwargs):
            from CrowdSim.humanoid.diffusion.guidance import TrajectoryAdmissionError
            try:
                return super().select(trajectory, **kwargs)
            except TrajectoryAdmissionError as error:
                # Keep real generated candidates even when their all-TRACE
                # combination fails. A mixed pool must still pass joint admission.
                error.preview_candidate_states = trajectory.detach()
                raise

        def __init__(self, states, context, config, *, scene_ids,
                     local_targets=None, target_indices=None):
            self.robot_prediction = robot_prediction() if robot_prediction else None
            if states.device.type != 'cpu' or len(states) < 2:
                super().__init__(states, context, config, scene_ids=scene_ids,
                                 local_targets=local_targets, target_indices=target_indices)
                return
            self.config = config
            self.states = states.detach()
            self.extent = context['extent'].detach()
            self.scene_ids = torch.as_tensor(scene_ids, device=states.device)
            if self.scene_ids.shape != (len(states),) or self.scene_ids.dtype not in (torch.int32, torch.int64):
                raise ValueError('Explicit integer scene_ids[B] are required')
            self.targets, self.indices = local_targets, target_indices
            free = ((context['image'][:, 0] >= .75) & (context['image'][:, 1] <= .25)).cpu().numpy()
            self.sdf = torch.as_tensor(np.stack(list(pool.map(field, free))),
                                       device=states.device, dtype=states.dtype)[:, None]
            self.pair_mask = ((self.scene_ids[:, None] == self.scene_ids[None, :])
                             & ~torch.eye(len(states), device=states.device, dtype=torch.bool))
        def components(self, trajectory):
            parts = super().components(trajectory)
            allowed = getattr(self, 'preview_candidate_allowed', None)
            if allowed is not None and allowed.shape == trajectory.shape[:2]:
                parts['preview_candidate_family_disallowed'] = ~allowed
            must_move = getattr(self,'preview_must_move',None)
            if must_move is not None:
                displacement = trajectory[...,:2].norm(dim=-1).amax(-1)
                parts['preview_progress_disallowed'] = must_move[:,None] & (displacement < .15)
            if self.robot_prediction is not None:
                from robot_pedestrian_neighbor import temporal_robot_gap
                gap = temporal_robot_gap(self.states,trajectory,self.robot_prediction)
                radius = self.extent[:,:2].norm(dim=-1)/2
                physical = radius+.6
                current = (self.states[:,:2]-self.states.new_tensor(self.robot_prediction['actual_xy'])).norm(dim=-1)
                # Permit escape from an already reduced comfort gap, but never
                # admit new physical overlap or a worsening close approach.
                required = torch.maximum(physical,torch.minimum(physical+.2,current-.005))
                violation = (required[:,None]-gap).clamp_min(0.)
                parts['minimum_robot_distance_m'] = gap
                parts['maximum_robot_admission_margin_violation_m'] = violation
                parts['maximum_robot_physical_overlap_m'] = (physical[:,None]-gap).clamp_min(0.)
                if getattr(self, 'preview_manual_robot_hold', False):
                    # A stationary balancing command cannot guarantee clearance
                    # from a manually driven incoming robot. Disclose this case
                    # separately; never turn it into an admitted walking route.
                    stationary = trajectory[...,:2].norm(dim=-1).amax(-1) <= .01
                    brake = getattr(self, 'preview_robot_hold_trajectory', None)
                    exact_brake = False if brake is None else (trajectory-brake).abs().amax(dim=(-1,-2)) <= 1e-6
                    slow = (self.states[:,2] < .1)[:,None]
                    moving_robot = self.states.new_tensor(self.robot_prediction['measured_velocity_xy']).norm() > .05
                    parts['preview_manual_robot_hold'] = stationary & exact_brake & slow & moving_robot & (current >= physical)[:,None]
                parts['agent'] = parts['agent']+(physical[:,None]+self.config.agent_buffer_m-gap).clamp_min(0.).square()
            return parts
    return ParallelGuidance


def temporal_motion_admission(original):
    """Keep source admission and add the explicit moving-robot constraint."""
    def admit(parts, config):
        accepted = original(parts, config)
        violation = parts.get('maximum_robot_admission_margin_violation_m')
        if violation is not None:
            accepted = accepted & ((violation <= 1e-6) | parts.get('preview_manual_robot_hold', False))
        disallowed = parts.get('preview_progress_disallowed')
        if disallowed is not None:
            accepted = accepted & ~disallowed
        family_disallowed = parts.get('preview_candidate_family_disallowed')
        if family_disallowed is not None:
            accepted = accepted & ~family_disallowed
        return accepted
    return admit


def json_value(value):
    if isinstance(value, torch.Tensor):
        return json_value(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def skip_robot_blocked_goals(routes, indices, robot_xy, clearance=1.41, robot_end=None):
    """Change desired patrol phase around the robot; never count it as arrival."""
    updated = indices.clone()
    skipped = torch.zeros_like(indices)
    delta = torch.zeros_like(robot_xy) if robot_end is None else robot_end-robot_xy
    for actor in range(len(indices)):
        for _ in range(routes.shape[1]):
            point = routes[actor, updated[actor]]
            fraction = ((point-robot_xy)*delta).sum()/delta.square().sum().clamp_min(1e-12)
            gap = (point-robot_xy-fraction.clamp(0., 1.)*delta).norm()
            if float(gap) >= clearance:
                break
            updated[actor] = (updated[actor]+1) % routes.shape[1]
            skipped[actor] += 1
        if skipped[actor] == routes.shape[1]:
            updated[actor] = indices[actor]
            skipped[actor] = 0  # Entire loop blocked: retain safety waiting.
    return updated, skipped


class Timings:
    def __init__(self, report):
        self.report = report
        self.samples = defaultdict(lambda: deque(maxlen=300))
        self.started = time.monotonic()
        self.last = self.started
        self.previous_controls = 0

    @contextmanager
    def measure(self, name):
        begin = time.perf_counter()
        try:
            yield
        finally:
            self.samples[name].append(time.perf_counter() - begin)

    def wrap(self, name, function):
        def measured(*args, **kwargs):
            with self.measure(name):
                return function(*args, **kwargs)
        return measured

    def update(self):
        now = time.monotonic()
        controls = self.report.get('actual_controls', 0)
        if now - self.last < 5 or controls < self.previous_controls:
            if controls < self.previous_controls:
                self.previous_controls, self.last = controls, now
            return
        self.report['performance'] = {
            'window_wall_seconds': now - self.last,
            'window_controls': controls - self.previous_controls,
            'real_time_factor': (controls - self.previous_controls) / 30 / (now - self.last),
            'timings_ms': {k: {'mean': 1000 * float(np.mean(v)),
                              'p95': 1000 * float(np.percentile(v, 95)),
                              'count': len(v)} for k, v in self.samples.items() if v},
        }
        self.previous_controls, self.last = controls, now


class RouteTargetCache:
    """Refresh on a replan, route change, or expiry; all guards use live state."""
    def __init__(self, interval=15):
        self.interval = interval
        self.clear()

    def clear(self):
        self.control = -1
        self.indices = None
        self.value = None

    def get(self, control, indices, produce, *, force=False):
        changed = self.indices is None or not torch.equal(indices, self.indices)
        stale = control < self.control or control - self.control >= self.interval
        # Initial outer-loop query and inner replan share this exact control.
        if self.value is None or changed or stale or (force and control != self.control):
            self.value = produce()
            self.indices = indices.clone()
            self.control = control
        return self.value


def route_candidates(states, local_targets, config, speeds=(.35, .55, .7), *, pivot_first=False):
    """Continuous unicycle rollouts from measured speed, not position jumps."""
    targets = local_targets.detach().cpu().numpy().astype(np.float64)
    initial_speed = states[:, 2].detach().cpu().numpy().astype(np.float64)
    batch = len(states)
    result = np.zeros((batch, len(speeds), 52, 6), dtype=np.float32)
    for actor in range(batch):
        path = np.concatenate((np.zeros((1, 2)), targets[actor]))
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        for sample, cruise in enumerate(speeds):
            x = y = yaw = 0.
            velocity = float(initial_speed[actor])
            progress = 0.
            for step in range(52):
                # Monotone projection prevents reversing to an earlier waypoint.
                delta = np.diff(path, axis=0)
                alpha = np.clip(((np.array([x, y]) - path[:-1]) * delta).sum(1)
                                / np.maximum((delta * delta).sum(1), 1e-10), 0., 1.)
                projected = path[:-1] + alpha[:, None] * delta
                nearest = int(np.argmin(np.linalg.norm(projected - [x, y], axis=1)))
                progress = max(progress, float(arc[nearest] + alpha[nearest] * (arc[nearest+1] - arc[nearest])))
                lookahead = min(arc[-1], progress + .35 + .5 * max(velocity, 0.))
                target = np.array([np.interp(lookahead, arc, path[:, axis]) for axis in range(2)])
                offset = target - [x, y]
                error = math.atan2(math.sin(math.atan2(offset[1], offset[0]) - yaw),
                                   math.cos(math.atan2(offset[1], offset[0]) - yaw))
                angular = float(np.clip(2. * error, -config.max_yaw_rate_rad_s, config.max_yaw_rate_rad_s))
                desired = min(cruise, config.max_speed_m_s) * max(.15, math.cos(error))
                if pivot_first and abs(error) > .25:
                    desired = 0.
                desired = min(desired, max(0., arc[-1] - progress) / .5)
                acceleration = float(np.clip((desired - velocity) / .3,
                                            -config.max_acceleration_m_s2, config.max_acceleration_m_s2))
                next_speed = max(0., velocity + acceleration * .1)
                acceleration = (next_speed - velocity) / .1
                # Same midpoint-speed / old-heading Euler convention as TRACE.
                distance = .1 * (velocity + next_speed) / 2
                x += distance * math.cos(yaw)
                y += distance * math.sin(yaw)
                yaw += .1 * angular
                velocity = next_speed
                result[actor, sample, step] = [x, y, velocity, yaw, acceleration, angular]
    return states.new_tensor(result)


def try_route_plan(planner, packet, report, *, manual_robot_hold=False, comfort_escape=False,
                   trace_candidates=None, recovery_actors=None):
    from CrowdSim.humanoid.diffusion.guidance import PedestrianGuidance, TrajectoryAdmissionError
    from CrowdSim.humanoid.diffusion.planner import DiffusionPlan, _local

    cfg = packet.get('pedestrian_guidance')
    if cfg is None or packet.get('waypoints') is None:
        return None
    states = packet['states']
    if comfort_escape:
        # Recovery from an existing comfort encroachment may first pivot while
        # balancing. Keep physical radii, full map/robot/motion checks and the
        # existing .02m tracking tolerance; do not demand immediate comfort-gap
        # growth during that pivot. Normal planning retains its .05m/s ramp.
        cfg = replace(cfg, comfort_recovery_speed_m_s=0.)
    allowed = ['states', 'history', 'neighbors', 'raster', 'history_mask',
               'neighbor_mask', 'extents', 'neighbor_extents']
    context = planner.prepare_context(**{k: v for k, v in packet.items() if k in allowed})
    waypoints = packet['waypoints']
    local = _local(torch.cat((waypoints, torch.zeros_like(waypoints)), -1), states)[..., :2]
    times = packet.get('waypoint_times')
    if times is None:
        times = torch.arange(1, local.shape[1]+1, device=states.device).mul(50/local.shape[1]).round().mul(.1)
    indices = (times/.1).round().long()-1
    guide = PedestrianGuidance(states, context, cfg, scene_ids=packet['scene_ids'],
                              local_targets=local, target_indices=indices)
    guide.preview_manual_robot_hold = manual_robot_hold
    candidates = route_candidates(states, local, cfg)
    trace_start = trace_end = candidates.shape[1]
    if trace_candidates is not None:
        trace_candidates = trace_candidates.to(states.device)
        if (trace_candidates.shape[0] != len(states) or trace_candidates.shape[2:] != (52, 6)
                or not torch.isfinite(trace_candidates).all()):
            raise ValueError('Finite actual TRACE candidates[B,S,52,6] required')
        trace_end += trace_candidates.shape[1]
        candidates = torch.cat((candidates, trace_candidates), 1)
    recovery = (None if recovery_actors is None else
                torch.as_tensor(recovery_actors, device=states.device, dtype=torch.bool))
    if recovery is not None and (recovery.shape != (len(states),) or trace_end == trace_start):
        raise ValueError('Per-actor recovery requires matched mask and actual TRACE candidates')

    def select_options(pool, *, fixed_samples=None, has_brake=False):
        if recovery is not None:
            allowed = torch.ones(pool.shape[:2], device=states.device, dtype=torch.bool)
            allowed[recovery] = False
            allowed[recovery, trace_start:trace_end] = True
            if has_brake:
                allowed[recovery, -1] = True
            guide.preview_candidate_allowed = allowed
        return guide.select(pool, fixed_samples=fixed_samples)

    brake = packet.get('analytic_brake_candidates')
    guide.preview_robot_hold_trajectory = brake
    route_count = candidates.shape[1]
    fixed = None
    if brake is not None:
        candidates = torch.cat((candidates, brake), 1)
        forced = packet.get('force_brake_mask')
        if forced is not None:
            fixed = torch.where(forced, route_count, -1).long()
    forced = packet.get('force_brake_mask')
    if forced is None:
        forced = torch.zeros(len(states), dtype=torch.bool, device=states.device)
    # A closing pair need not both stand still. Search admitted combinations
    # of moving and braking candidates; the same joint separation checks run.
    # Map/body hazards retain their mandatory braking candidates.
    if report.get('preview_guard_map_body_mask') is not None:
        map_body = torch.tensor(report['preview_guard_map_body_mask'],device=states.device,dtype=torch.bool)
        forced = forced & map_body
        fixed = torch.where(forced,route_count,-1).long() if brake is not None else None
    # Braking is a safety option, not a cheap substitute for feasible progress.
    # First seek a jointly admitted moving combination. With a real hazard,
    # or no such combination, retain the admitted mixed moving/braking plan.
    try:
        if not bool(forced.any()):
            try:
                selected, diagnostics = select_options(candidates[:, :route_count])
            except TrajectoryAdmissionError:
                selected, diagnostics = select_options(candidates, fixed_samples=fixed, has_brake=brake is not None)
        else:
            selected, diagnostics = select_options(candidates, fixed_samples=fixed, has_brake=brake is not None)
    except TrajectoryAdmissionError as error:
        report['fast_route_last_rejection'] = error.diagnostics
        # Previously the side steps were tried only AFTER a successful mixed
        # plan. An actor with no forward/braking option never reached them.
        extra = yielding_candidates(states,cfg)
        if comfort_escape:
            extra = torch.cat((extra, yielding_candidates(states,cfg,pivot_first=True)),1)
        expanded = torch.cat((candidates[:,:route_count],extra),1)
        expanded_route_count = expanded.shape[1]
        if brake is not None:
            expanded = torch.cat((expanded,brake),1)
        expanded_fixed = torch.where(forced,expanded_route_count,-1).long() if brake is not None else None
        try:
            selected,diagnostics = select_options(expanded,fixed_samples=expanded_fixed,has_brake=brake is not None)
        except TrajectoryAdmissionError as expanded_error:
            report['fast_route_expanded_rejection'] = expanded_error.diagnostics
            return None
        candidates,route_count = expanded,expanded_route_count
        report['pedestrian_yield_recovery_plans'] = report.get('pedestrian_yield_recovery_plans',0)+1
    # Straight route speeds cannot resolve a narrow counterflow: followers
    # then brake forever behind an already stopped person. Try a small local
    # side step with the SAME full-horizon joint/map/robot admission.
    recovering = (selected >= route_count) & (states[:,2] < .1) & ~forced
    if bool(recovering.any()):
        extra = yielding_candidates(states,cfg)
        expanded = torch.cat((candidates[:,:route_count],extra),1)
        expanded_route_count = expanded.shape[1]
        if brake is not None:
            expanded = torch.cat((expanded,brake),1)
        expanded_fixed = torch.where(forced,expanded_route_count,-1).long() if brake is not None else None
        guide.preview_must_move = recovering
        try:
            recovery_selected,recovery_diag = select_options(expanded,fixed_samples=expanded_fixed,has_brake=brake is not None)
        except TrajectoryAdmissionError:
            # A rejected side step is never permission to move unsafely.
            pass
        else:
            selected,diagnostics = recovery_selected,recovery_diag
            candidates,route_count = expanded,expanded_route_count
            report['pedestrian_yield_recovery_plans'] = report.get('pedestrian_yield_recovery_plans',0)+1
            report['pedestrian_yield_recovery_selected'] = selected.tolist()
        finally:
            del guide.preview_must_move
    if bool((selected >= route_count).all()) and not bool(forced.all()):
        report['fast_route_last_rejection'] = {'reason': 'all route candidates blocked', 'selected': selected.tolist()}
        return None
    picked = candidates[torch.arange(len(states), device=states.device), selected, :50]
    c, s = states[:, 3, None].cos(), states[:, 3, None].sin()
    xy = torch.stack((picked[..., 0]*c-picked[..., 1]*s,
                      picked[..., 0]*s+picked[..., 1]*c), -1)+states[:, None, :2]
    yaw = torch.remainder(picked[..., 3]+states[:, 3, None]+math.pi, 2*math.pi)-math.pi
    report['fast_route_plans'] = report.get('fast_route_plans', 0)+1
    report['preview_selected_candidate_sources'] = [
        'brake' if int(value) >= route_count else
        'TRACE' if trace_start <= int(value) < trace_end else 'route_or_yield'
        for value in selected]
    robot_gap_violation = diagnostics.get('selected_joint_maximum_robot_admission_margin_violation_m')
    holding = (selected >= route_count) & (states[:,2] < .1)
    degraded = manual_robot_hold and robot_gap_violation is not None and bool((robot_gap_violation > 1e-6).any())
    report['manual_robot_yield_active'] = degraded
    report['manual_robot_yield_actors'] = torch.where(holding)[0].tolist() if degraded else []
    report['manual_robot_yield_full_admission'] = not degraded
    if degraded:
        report['manual_robot_yield_plans'] = report.get('manual_robot_yield_plans',0)+1
        report['manual_robot_yield_diagnostics'] = json_value(diagnostics)
    else:
        report['fast_route_last_rejection'] = None
    return DiffusionPlan(
        xy=torch.cat((states[:, None, :2], xy), 1),
        yaw=torch.cat((states[:, 3:4], yaw), 1),
        vel=torch.cat((states[:, 2:3], picked[..., 2]), 1),
        time=torch.arange(51, device=states.device)*.1,
        sample_states=candidates,
        selected_sample=torch.where(selected >= route_count, -1, selected),
        reference=('manual incoming robot: stationary balancing hold exception; robot forecast NOT admitted; map/peer/locomotion gates retained'
                   if degraded else 'per-actor route/TRACE candidate pool + original joint admission'
                   if trace_candidates is not None else 'preview bounded route candidates + original joint admission; TRACE on rejection'),
        guidance_diagnostics=diagnostics,
        sampling_dynamics_limits={'kind': 'explicit bounded CPU route-following preview candidate'},
    )


def yielding_candidates(states,config, *, pivot_first=False):
    """Bounded turn-and-walk side steps; ordinary AMP executes their targets."""
    variants = []
    points = [(0.,-.65),(0.,.65)]
    if pivot_first:
        points += [(-.8,-.35),(-.8,.35),(-.8,0.)]
    for point in points:
        target = states.new_tensor(point).expand(len(states),10,2)
        variants.append(route_candidates(states,target,config,speeds=(.25,.4),pivot_first=pivot_first))
    return torch.cat(variants,1)


def install(m):
    """Only for --optimized-preview, before the GUI responsiveness wrapper."""
    report = m.report
    import CrowdSim.humanoid.diffusion.guidance as guidance_module
    guidance_module.motion_admission = temporal_motion_admission(guidance_module.motion_admission)
    def robot_prediction():
        if getattr(m.args, 'continuous_trace', False):
            from continuous_trace_preview import PREDICTION
            snapshot = getattr(PREDICTION, 'value', None)
            if snapshot is not None:
                return snapshot
        return report.get('robot_motion_prediction')
    guidance_module.PedestrianGuidance = parallel_guidance_class(guidance_module.PedestrianGuidance,
        robot_prediction=robot_prediction)
    report['robot_local_admission'] = 'continuous time-aligned 1.2s robot/ego relative segments; static room raster; buffered footprint; no physical overlap'
    report['guidance_map_cpu_workers'] = 4
    report['robot_pedestrian_command_guard_enabled'] = False
    report['robot_pedestrian_command_guard'] = {'enabled':False,
        'mode':'user requested direct manual wheel control; linear cap remains 2m/s'}
    perf = Timings(report)
    m.preview_timings = perf
    m.route_target_cache = RouteTargetCache(m.profile['replan_controls'])
    original_advance = m.native.advance_route
    patrol_routes = torch.as_tensor(m.native._routes, dtype=torch.float32, device=m.world.device)
    def advance_route(positions, indices):
        updated, reached = original_advance(positions, indices)
        observation = m.robot.observe()
        from robot_pedestrian_neighbor import robot_motion_segment
        start, end = robot_motion_segment(observation)
        robot_xy, robot_end = positions.new_tensor(start), positions.new_tensor(end)
        report['robot_route_phase_before_awareness'] = updated.tolist()
        updated, skipped = skip_robot_blocked_goals(patrol_routes, updated, robot_xy, robot_end=robot_end)
        report['robot_route_phase_after_awareness'] = updated.tolist()
        report['robot_blocked_goals_skipped_this_control'] = skipped.tolist()
        previous = report.get('robot_blocked_route_goals_skipped_per_actor', [0]*len(indices))
        report['robot_blocked_route_goals_skipped_per_actor'] = [a+int(b) for a,b in zip(previous, skipped.tolist())]
        return updated, reached
    m.native.advance_route = advance_route
    report['robot_blocked_goal_skip_is_physical_arrival'] = False
    # SMPL skinning is independent of the PhysX body state. Move this larger
    # matrix workload too; USD still receives actual physical-pose vertices.
    visual = m.mesh.visualizer
    neutral_pose = torch.zeros(m.count, 69)
    neutral_root = torch.zeros(m.count, 3)
    neutral_quat = torch.zeros(m.count, 4)
    neutral_quat[:, 0] = 1.
    with torch.no_grad():
        cpu_vertices = visual.compute_world_vertices(neutral_pose, neutral_root, neutral_quat).cpu()
        visual.device = torch.device('cuda:0')
        visual.cfg = replace(visual.cfg, device='cuda:0')
        visual.smpl_model.to(visual.device)
        visual.betas = visual.betas.to(visual.device)
        visual.shape_floor_offsets = visual.shape_floor_offsets.to(visual.device)
        gpu_vertices = visual.compute_world_vertices(neutral_pose, neutral_root, neutral_quat).cpu()
    error = float((cpu_vertices-gpu_vertices).abs().max())
    if error > 1e-4:
        raise ValueError(f'CUDA skinning numerical comparison failed: {error}')
    report.update(skinning_device='cuda:0', skinning_neutral_max_abs_error_m=error)
    controller = m.controller
    from preview_progress import PreviewProgress
    execution_progress = PreviewProgress()
    amp_stream = (m.OUT/'amp_execution_targets.jsonl').open('a', buffering=1)
    # Observations/terrain/guards stay on CPU; transfer only 4x1401 inputs and
    # 4x69 physical-control outputs. Published weights and normalization stay.
    controller.network.to('cuda:0').eval()

    @torch.no_grad()
    def gpu_act(state, trajectory, terrain, *, trajectory_frame='world',
                return_raw=False, deterministic=True, generator=None):
        controller._require_checkpoint()
        with perf.measure('amp_observation_cpu'):
            observation = controller.raw_observation(state, trajectory, terrain, trajectory_frame=trajectory_frame)
            normalized = controller.observation_normalizer(observation.float())
        with perf.measure('amp_actor_gpu_with_transfer'):
            mean, log_std = controller.network.eval_actor(normalized.to('cuda:0'))
            if not deterministic:
                raise ValueError('Optimized manual preview uses deterministic AMP inference only')
            action = mean.clamp(-1., 1.).to(state.root_pos.device)
        if m.control % m.profile['replan_controls'] == 0:
            amp_stream.write(json.dumps(json_value(dict(
                control=m.control, root_xy=state.root_pos[:, :2],
                actual_velocity_xy=state.root_vel[:, :2],
                final_amp_world_targets=trajectory,
                canonical_action_max_abs=action.abs().amax(-1))))+'\n')
        return action if return_raw else controller.decode_action(action)

    controller.act = gpu_act
    # Stop/wait protection is installed around this GPU actor below.
    from pedestrian_robot_stop_guard import install_stop_guard
    install_stop_guard(controller, m.robot, report)
    original_plan = m.planner.plan
    manual_holds_enabled = bool(m.args.manual_robot)
    admission_events = (m.OUT/'manual_robot_admission_events.jsonl').open('a', buffering=1)
    def hybrid_plan(**kwargs):
        stalled = report.get('pedestrian_execution_stalled', [])
        use_trace = manual_holds_enabled and any(stalled)
        if use_trace:
            report['execution_stall_trace_replans'] = report.get('execution_stall_trace_replans', 0)+1
            report['trace_fallback_plans'] = report.get('trace_fallback_plans', 0)+1
            trace_error = None
            traced = None
            with perf.measure('trace_fallback_gpu'):
                try:
                    traced = original_plan(**kwargs)
                    samples = traced.sample_states
                except guidance_module.TrajectoryAdmissionError as error:
                    trace_error = error
                    samples = getattr(error, 'preview_candidate_states', None)
                    if samples is None:
                        raise
            # Only stalled actors lose the ineffective route family. Healthy
            # actors keep their route/side-step options. All choices, including
            # mixed TRACE/route pairs, are checked together against the same gates.
            samples = samples[:, :kwargs['num_samples']].detach().cpu()
            with perf.measure('route_candidate_admission_cpu'):
                candidate = try_route_plan(m.planner, kwargs, report,
                    trace_candidates=samples, recovery_actors=stalled)
                if candidate is None:
                    candidate = try_route_plan(m.planner, kwargs, report,
                        trace_candidates=samples, recovery_actors=stalled, manual_robot_hold=True)
            if candidate is not None:
                report['per_actor_recovery_plans'] = report.get('per_actor_recovery_plans', 0)+1
                return candidate
            if trace_error is not None:
                raise trace_error
            report['preview_selected_candidate_sources'] = [
                'brake' if int(value) < 0 else 'TRACE' for value in traced.selected_sample]
            return traced
        else:
            with perf.measure('route_candidate_admission_cpu'):
                candidate = try_route_plan(m.planner, kwargs, report)
        if candidate is not None:
            return candidate
        if not use_trace and manual_holds_enabled and report.get('manual_robot_yield_active'):
            candidate = try_route_plan(m.planner, kwargs, report, manual_robot_hold=True)
            if candidate is not None:
                return candidate
        report['trace_fallback_plans'] = report.get('trace_fallback_plans', 0)+1
        with perf.measure('trace_fallback_gpu'):
            try:
                result = original_plan(**kwargs)
            except guidance_module.TrajectoryAdmissionError as error:
                if not manual_holds_enabled or use_trace:
                    raise
                report['manual_robot_yield_original_rejection'] = json_value(error.diagnostics)
                admission_events.write(json.dumps(json_value({
                    'control': m.control, 'states': kwargs['states'],
                    'waypoints': kwargs.get('waypoints'),
                    'robot_prediction': report.get('robot_motion_prediction'),
                    'normal_admission_failed': True, 'diagnostics': error.diagnostics}))+'\n')
                candidate = try_route_plan(m.planner, kwargs, report, manual_robot_hold=True)
                if candidate is None:
                    raise
                return candidate
            report['manual_robot_yield_active'] = False
            report['manual_robot_yield_full_admission'] = True
            report['preview_selected_candidate_sources'] = [
                'brake' if int(value) < 0 else 'TRACE' for value in result.selected_sample]
            return result
    m.planner.plan = hybrid_plan
    original_new_runtime = m.new_runtime
    diagnostics_stream = (m.OUT/'planner_diagnostics.jsonl').open('a', buffering=1)
    def attach(runtime):
        original_guard = runtime.actual_velocity_hazards
        def guard(*args, **kwargs):
            hazard, detail = original_guard(*args, **kwargs)
            body = detail.get('actual_body_map_guard') or {}
            body_parts = body.get('body_parts_at_risk', [[False]]*len(hazard))
            report['preview_guard_map_body_mask'] = [bool(a or any(b))
                for a,b in zip(detail['map_at_risk'],body_parts)]
            report['preview_pair_hazard_mask'] = detail['closing_pair_at_risk']
            report['preview_peer_yield_mode'] = 'jointly admitted moving/braking combinations; map/body hazards keep mandatory braking'
            return hazard, detail
        runtime.actual_velocity_hazards = guard
        original_action = runtime.action
        def action(*args, **kwargs):
            if m.args.manual_robot:
                current = m.world.planar()
                intended = ([False]*len(current) if runtime.plan is None else
                    ((runtime.plan.xy-current[:, None, :2]).norm(dim=-1).amax(-1) >= .15).tolist())
                motion, stalled, ready = execution_progress.update(
                    args[1], current[:, :2].detach().cpu().numpy(), intended,
                    report.get('pedestrian_robot_waiting', [False]*len(current)))
                report.update(pedestrian_recent_motion_m=motion.tolist(),
                    pedestrian_execution_stalled=stalled.tolist(),
                    pedestrian_recent_motion_ready=ready)
            with perf.measure('pedestrian_control_total'):
                result, diag = original_action(*args, **kwargs)
            report['safety_braking_current'] = diag['safety_braking']
            if diag['replanned']:
                detail = json_value({'control': args[1], 'selected_sample': runtime.plan.selected_sample,
                    'reference': runtime.plan.reference,
                    'candidate_sources': report.get('preview_selected_candidate_sources'),
                    'actual_state': m.world.planar(),
                    'route_targets': args[2],
                    'route_targets_without_robot': report.get('route_targets_without_robot'),
                    'robot_motion_prediction': report.get('robot_motion_prediction'),
                    'robot_route_change_rms_m': report.get('robot_route_change_rms_m'),
                    'robot_route_phase_before_awareness': report.get('robot_route_phase_before_awareness'),
                    'robot_route_phase_after_awareness': report.get('robot_route_phase_after_awareness'),
                    'plan_xy_1s_intervals': runtime.plan.xy[:, ::10],
                    'guard': runtime.last_rejection_diagnostics,
                    'guidance': runtime.plan.guidance_diagnostics,
                    'fast_route_rejection': report.get('fast_route_last_rejection')})
                report['planner_latest_diagnostics'] = detail
                diagnostics_stream.write(json.dumps(detail)+'\n')
            return result, diag
        runtime.action = action
        return runtime
    m.runtime = attach(m.runtime)
    m.new_runtime = lambda: attach(original_new_runtime())
    m.world.step = perf.wrap('physics_cpu', m.world.step)
    m.mesh.update = perf.wrap('skinning_gpu_and_usd_copy', m.mesh.update)
    m.lidar.scan = perf.wrap('lidar_cpu', m.lidar.scan)
    m.world.simulator._sim.render = perf.wrap('render_gpu', m.world.simulator._sim.render)
    m.pump_gui = perf.wrap('kit_update', m.pump_gui)
    original_save = m.save
    def save():
        perf.update()
        original_save()
    m.save = save
    report.update(optimized_preview=True, inference_device='cuda:0', physics_device='cpu',
                  pedestrian_planner_mode='admitted bounded route candidates; original TRACE fallback',
                  training=False, planner_optimization_changes_sampling=True,
                  safety_admission_unchanged=False, sensor_contract_unchanged=True)
    report['manual_robot_yield_policy'] = 'Only stationary (<0.1m/s, <=0.01m target displacement) balancing holds may continue despite incoming manual robot prediction; ordinary walks retain full admission; current physical overlap/map/peer/locomotion failures still reject'
    report['robot_awareness_mode'] = 'pedestrians use actual robot state, detour, temporal admission and target-speed yield; robot manual proximity stop disabled'
