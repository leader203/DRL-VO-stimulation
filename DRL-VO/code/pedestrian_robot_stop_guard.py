"""Target-speed stop/wait guard for physical AMP pedestrians (no pose writes)."""
import torch


def install_stop_guard(controller, robot, report, *, robot_radius=.6,
                       pedestrian_radius=.46, margin=.2, horizon=1.2,
                       release_margin=.15, strict_planned_escape=False):
    original = controller.act
    waiting = None
    def act(state, trajectory, terrain, **kwargs):
        nonlocal waiting
        if kwargs.get('trajectory_frame', 'world') != 'world':
            raise ValueError('Robot stop guard requires world-frame trajectory')
        observation = robot.observe()
        xy = state.root_pos[:, :2]
        robot_xy = xy.new_tensor(observation['root_position_m'][:2])
        robot_velocity = xy.new_tensor(observation['linear_velocity_world'][:2])
        # Re-evaluate at every AMP control tick using measured physical state.
        # Sample the existing ten PACER waypoints at their documented .5s times.
        times = torch.arange(.1, horizon+.0001, .1, device=xy.device, dtype=xy.dtype)
        index = (times/.5).floor().long().clamp(max=trajectory.shape[1]-2)
        alpha = (times/.5-index).clamp(0, 1)
        planned = trajectory[:, index, :2]*(1-alpha[None,:,None]) + trajectory[:, index+1, :2]*alpha[None,:,None]
        robot_future = robot_xy[None,None] + robot_velocity[None,None]*times[None,:,None]
        route_gap = (planned-robot_future).norm(dim=-1).amin(1)
        # Constant actual-velocity closest approach catches current momentum too.
        relative_position = xy-robot_xy
        relative_velocity = state.root_vel[:, :2]-robot_velocity
        closest_time = (-(relative_position*relative_velocity).sum(-1)/relative_velocity.square().sum(-1).clamp_min(1e-8)).clamp(0,horizon)
        actual_gap = (relative_position+relative_velocity*closest_time[:,None]).norm(dim=-1)
        gap = torch.minimum(route_gap, actual_gap)
        if waiting is None or waiting.shape != gap.shape:
            waiting = torch.zeros_like(gap, dtype=torch.bool)
        threshold = robot_radius+pedestrian_radius+margin
        waiting = torch.where(waiting, (route_gap < threshold+release_margin) | (actual_gap < threshold), gap < threshold)
        # A stopped person must be able to start an admitted escape route.
        # Forecasting it as stationary forever while holding that very person
        # creates a self-sustaining wait even when its planned route is clear.
        current_gap = relative_position.norm(dim=-1)
        final_gap = (planned[:,-1]-robot_future[:,-1]).norm(dim=-1)
        # A waiting person cannot reach a fixed larger release gap without
        # first moving. Admit a stationary person's separating route when
        # its predicted gap does not worsen and its body remains clear.
        required = torch.minimum(current_gap-.02,xy.new_full(current_gap.shape,threshold))
        physical = robot_radius+pedestrian_radius
        escape = ((state.root_vel[:,:2].norm(dim=-1)<.1)
                  & (current_gap>=physical+.05)
                  & (route_gap>=required.clamp_min(physical))
                  & (final_gap>=current_gap+.1))
        if strict_planned_escape:
            # DRL-VO receives an admitted TRACE plan. Recheck its continuous
            # AMP target interpolation at the SAME execution margin; a safe
            # moving escape need not increase the current gap by another .1m.
            # A stationary hold never qualifies. Unknown/wall/peer/motion
            # admission remains owned by TRACE; no pose or velocity is set.
            relative = torch.cat((relative_position[:,None],planned-robot_future),1)
            segments = relative.diff(dim=1)
            fraction = (-relative[:,:-1]*segments).sum(-1)/segments.square().sum(-1).clamp_min(1e-8)
            closest = relative[:,:-1]+fraction.clamp(0.,1.)[...,None]*segments
            continuous_gap = closest.norm(dim=-1).amin(1)
            # Measured momentum must also retain the full margin during the
            # first .1s before the new response can take effect.
            first_time = closest_time.clamp_max(.1)
            first_gap = (relative_position+relative_velocity*first_time[:,None]).norm(dim=-1)
            moving = (planned-xy[:,None]).norm(dim=-1).amax(1) >= .05
            waiting = waiting | (continuous_gap<threshold) | (first_gap<threshold)
            # A moving person also needs to execute an admitted response.
            # Long constant-velocity extrapolation is a hazard trigger, not
            # proof that stopping is safer than a continuously clear plan.
            # The measured first-.1s momentum gate stays mandatory regardless
            # of speed; TRACE still owns the full motion/acceleration limits.
            escape = ((current_gap>=threshold) & (first_gap>=threshold)
                      & (continuous_gap>=threshold) & moving)
            report['pedestrian_robot_strict_escape_continuous_gap_m'] = continuous_gap.tolist()
            report['pedestrian_robot_strict_escape_first_actual_gap_m'] = first_gap.tolist()
            report['pedestrian_robot_strict_planned_escape'] = True
        waiting = waiting & ~escape
        held = trajectory.clone()
        # All future target points at current XY encode exactly zero desired
        # translation. AMP/PD still balances the body; actual speed may decay.
        held[waiting, :, :2] = xy[waiting, None, :]
        report.update(pedestrian_robot_stop_guard_enabled=True,
                      pedestrian_robot_waiting=waiting.tolist(),
                      pedestrian_robot_admitted_escape=escape.tolist(),
                      pedestrian_robot_predicted_center_gap_m=gap.tolist(),
                      pedestrian_robot_desired_speed_m_s=[0. if flag else None for flag in waiting.tolist()],
                      pedestrian_robot_stop_threshold_m=threshold,
                      pedestrian_robot_resume_threshold_m=threshold+release_margin,
                      pedestrian_robot_prediction_horizon_s=horizon,
                      pedestrian_robot_stop_guard_runtime_tested=False,
                      pedestrian_robot_stop_mode='zero target translation; AMP/PD balancing; no root velocity or pose injection')
        return original(state, held, terrain, **kwargs)
    controller.act = act
    report['pedestrian_robot_stop_guard_configured'] = True
