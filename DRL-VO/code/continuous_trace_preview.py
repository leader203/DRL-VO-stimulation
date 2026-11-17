"""One bounded TRACE process, live re-admission, and continuous performance evidence.

TRACE sampling never owns Kit or physical state. Its sampled action intents are
aged by simulation time, integrated again from the current measured state, and
jointly admitted against the current room, people and robot before execution.
Admitted CPU routes bridge sampling/rejection; provenance is explicit per plan.
"""
import json
import math
import threading
import time
from pathlib import Path

import torch

from preview_runtime_optimization import json_value, try_route_plan

PREDICTION = threading.local()


def within_stable_route(candidate, reference):
    return (bool(torch.isfinite(candidate.xy).all()) and
            bool(torch.isfinite(candidate.vel).all()) and
            bool(((candidate.xy-reference.xy).norm(dim=-1) <= .10).all()) and
            bool(((candidate.vel-reference.vel).abs() <= .15).all()))


def age_actions(samples, elapsed_s, config):
    if not math.isfinite(elapsed_s) or elapsed_s < 0 or elapsed_s > 1.5:
        raise ValueError('TRACE result is stale or from a reset')
    actions = samples[..., 4:6]
    query = torch.arange(52, dtype=actions.dtype, device=actions.device) + elapsed_s / .1
    left = query.floor().long().clamp(max=51)
    right = (left + 1).clamp(max=51)
    result = torch.lerp(actions[:, :, left], actions[:, :, right],
                        (query-query.floor())[None, None, :, None])
    # The sampled horizon ends in an explicit bounded brake, never indefinite
    # replay of the final walking action or a jump to old world coordinates.
    tail = query > 51
    result[..., tail, 0] = -config.max_acceleration_m_s2
    result[..., tail, 1] = 0.
    return result


def readmit(planner, packet, sampled, elapsed_s, *, progress_mask=None):
    from CrowdSim.humanoid.diffusion.guidance import PedestrianGuidance
    from CrowdSim.humanoid.diffusion.planner import DiffusionPlan, _local
    from CrowdSim.humanoid.diffusion.trace_helpers import unicyle_forward_dynamics
    cfg, states = packet['pedestrian_guidance'], packet['states']
    actions = age_actions(sampled.sample_states[:, :packet['num_samples']], elapsed_s, cfg)
    b, s, t, _ = actions.shape
    initial = states.new_zeros(b, s, 4)
    initial[..., 2] = states[:, None, 2]
    trajectory = unicyle_forward_dynamics(planner.model.dyn, initial.reshape(-1, 4),
        actions.reshape(-1, t, 2), .1, pedestrian_speed_ceiling=cfg.max_speed_m_s,
        pedestrian_yaw_rate_ceiling=cfg.max_yaw_rate_rad_s,
        bounded_guidance_gradient=0.).reshape(b, s, t, 4)
    candidates = torch.cat((trajectory, actions), -1)
    keys = ('states', 'history', 'neighbors', 'raster', 'history_mask', 'neighbor_mask', 'extents', 'neighbor_extents')
    context = planner.prepare_context(**{k: packet[k] for k in keys if k in packet})
    targets = packet['waypoints']
    local = _local(torch.cat((targets, torch.zeros_like(targets)), -1), states)[..., :2]
    guide = PedestrianGuidance(states, context, cfg, scene_ids=packet['scene_ids'],
        local_targets=local, target_indices=torch.arange(1, local.shape[1]+1).mul(50/local.shape[1]).round().long()-1)
    if progress_mask is not None:
        original_components = guide.components
        def components(value):
            parts = original_components(value)
            # Receding-horizon progress must occur inside the next committed
            # .5s, not after a start delay that every replan pushes back again.
            displacement = value[..., :5, :2].norm(dim=-1).amax(-1)
            parts['preview_progress_disallowed'] = progress_mask[:, None] & (displacement < .07)
            return parts
        guide.components = components
    brakes = packet.get('analytic_brake_candidates')
    fixed = None
    if brakes is not None:
        candidates = torch.cat((candidates, brakes), 1)
        forced = packet.get('force_brake_mask')
        if forced is not None:
            fixed = torch.where(forced, s, -1).long()
    selected, diagnostics = guide.select(candidates, fixed_samples=fixed)
    picked = candidates[torch.arange(b), selected, :50]
    c, sn = states[:, 3, None].cos(), states[:, 3, None].sin()
    xy = torch.stack((picked[..., 0]*c-picked[..., 1]*sn,
                      picked[..., 0]*sn+picked[..., 1]*c), -1)+states[:, None, :2]
    yaw = torch.remainder(picked[..., 3]+states[:, 3, None]+math.pi, 2*math.pi)-math.pi
    return DiffusionPlan(torch.cat((states[:, None, :2], xy), 1),
        torch.cat((states[:, 3:4], yaw), 1), torch.cat((states[:, 2:3], picked[..., 2]), 1),
        torch.arange(51)*.1, candidates, torch.where(selected >= s, -1, selected),
        reference='continuous actual TRACE sampled intents; simulation-age shift; current measured-state reintegration and full joint re-admission',
        guidance_diagnostics=diagnostics,
        sampling_dynamics_limits={'source': 'actual 100-step TRACE + original clean refinement',
                                  'sample_age_sim_seconds': elapsed_s, 'physical_state_written': False})


def install(m):
    from CrowdSim.humanoid.diffusion.planner import DiffusionPlanner
    from CrowdSim.humanoid.diffusion.guidance import TrajectoryAdmissionError
    report = m.report
    resources = m.source['resources']
    # Isolated planner: no shared mutable device, diffusion state or dynamics
    # with runtime guards. A single worker bounds queue depth and GPU ownership.
    import os
    import subprocess
    import uuid
    import numpy as np
    from dataclasses import asdict
    from types import SimpleNamespace
    root = Path(__file__).resolve().parents[1]
    log = (m.OUT/'TRACE_WORKER.log').open('a')
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    worker = subprocess.Popen([str(root/'python.sh'), str(root/'tools/continuous_trace_worker.py'),
        '--run', str(m.OUT), '--owner', str(os.getpid())], env=environment,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    state = {'pending': None, 'latest': None, 'epoch': 0, 'closed': False}
    events = (m.OUT/'continuous_trace_events.jsonl').open('a', buffering=1)

    def submit(packet):
        sequence = uuid.uuid4().hex
        arrays = {k:v.detach().cpu().numpy() for k,v in packet.items() if isinstance(v, torch.Tensor)}
        options = {k:v for k,v in packet.items() if not isinstance(v, torch.Tensor)}
        options['pedestrian_guidance'] = asdict(packet['pedestrian_guidance'])
        with (m.OUT/'trace_input.tmp').open('wb') as f:
            np.savez(f, **arrays)
        (m.OUT/'trace_input.tmp').replace(m.OUT/'trace_input.npz')
        request = dict(sequence=sequence, control=m.control, epoch=state['epoch'],
            options=options, prediction=report.get('robot_motion_prediction'))
        temporary = m.OUT/'TRACE_REQUEST.tmp'
        temporary.write_text(json.dumps(request)+'\n')
        temporary.replace(m.OUT/'TRACE_REQUEST.json')
        state['pending'] = sequence

    # Replace the old synchronous responsiveness wrapper: GUI and physics now
    # continue on the owner thread while real TRACE runs in the single inference process.
    def plan(**packet):
        if worker.poll() is not None and not state['closed']:
            raise RuntimeError('Continuous TRACE worker exited; inspect TRACE_WORKER.log')
        response = m.OUT/'TRACE_RESPONSE.json'
        if state['pending'] is not None and response.exists():
            result = json.loads(response.read_text())
            if result['sequence'] == state['pending']:
                state['pending'] = None
                if result.get('fatal'):
                    raise RuntimeError(result['traceback'])
                report['continuous_trace_completed'] = report.get('continuous_trace_completed', 0)+1
                report['continuous_trace_last_wall_seconds'] = result['wall_seconds']
                events.write(json.dumps(result)+'\n')
                if result['epoch'] == state['epoch'] and not result['rejected']:
                    with np.load(m.OUT/'trace_output.npz', allow_pickle=False) as data:
                        result['plan'] = SimpleNamespace(sample_states=torch.from_numpy(data['sample_states'].copy()))
                    state['latest'] = result
                if result['rejected']:
                    report['continuous_trace_sampling_rejected'] = report.get('continuous_trace_sampling_rejected', 0)+1
        if state['pending'] is None and not state['closed']:
            submit(packet)
            report['continuous_trace_submitted'] = report.get('continuous_trace_submitted', 0)+1
        # Establish the independently admitted stable route before TRACE takeover.
        with m.preview_timings.measure('route_candidate_admission_cpu'):
            candidate = try_route_plan(m.planner, packet, report)
            if candidate is None and m.args.manual_robot:
                candidate = try_route_plan(m.planner, packet, report, manual_robot_hold=True)
            if candidate is None:
                states = packet['states']
                radius = packet['extents'][:, :2].norm(dim=-1)/2
                gap = (states[:, None, :2]-states[None, :, :2]).norm(dim=-1)
                physical = radius[:, None]+radius[None, :]
                slow = states[:, 2] < .1
                encroached = ((gap >= physical+.05) &
                    (gap < physical+packet['pedestrian_guidance'].agent_buffer_m) &
                    slow[:, None] & slow[None, :])
                if bool(encroached.any()):
                    candidate = try_route_plan(m.planner, packet, report, comfort_escape=True)
                    if candidate is not None:
                        candidate.reference += '; existing comfort encroachment: pivot-first separation; physical radii retained'
                        report['continuous_trace_comfort_escape_plans'] = report.get('continuous_trace_comfort_escape_plans', 0)+1
        stable_route = candidate
        latest = state['latest']
        if latest is not None:
            age = (m.control-latest['control'])*m.world.dt
            if 0 <= age <= 1.5:
                try:
                    with m.preview_timings.measure('trace_current_readmission_cpu'):
                        map_body = torch.tensor(report.get('preview_guard_map_body_mask', [False]*len(packet['states'])), dtype=torch.bool)
                        candidate = readmit(m.planner, packet, latest['plan'], age, progress_mask=~map_body)
                except TrajectoryAdmissionError as error:
                    report['continuous_trace_readmission_rejected'] = report.get('continuous_trace_readmission_rejected', 0)+1
                    report['continuous_trace_last_rejection'] = json_value(error.diagnostics)
                else:
                    if stable_route is None or not within_stable_route(candidate, stable_route):
                        report['continuous_trace_stability_rejected'] = report.get('continuous_trace_stability_rejected', 0)+1
                        candidate = None
                    if candidate is not None:
                        report['continuous_trace_executed'] = report.get('continuous_trace_executed', 0)+1
                        report['continuous_trace_selected_sample'] = candidate.selected_sample.tolist()
                        report['continuous_trace_last_age_sim_seconds'] = age
                        report['manual_robot_yield_active'] = False
                        report['manual_robot_yield_full_admission'] = True
                        return candidate
            else:
                state['latest'] = None
                report['continuous_trace_stale_discarded'] = report.get('continuous_trace_stale_discarded', 0)+1
        candidate = stable_route
        if candidate is None:
            raise TrajectoryAdmissionError({'continuous_trace_bridge_unavailable': torch.ones(1, 1)})
        report['continuous_trace_bridge_plans'] = report.get('continuous_trace_bridge_plans', 0)+1
        return candidate

    m.planner.plan = plan
    def invalidate():
        state['epoch'] += 1
        state['latest'] = None
    m.invalidate_continuous_trace = invalidate
    def close():
        state['closed'] = True
        temporary = m.OUT/'TRACE_STOP.tmp'
        temporary.write_text(json.dumps({'stop':True})+'\n')
        temporary.replace(m.OUT/'TRACE_REQUEST.json')
        try:
            worker.wait(timeout=8)
        except subprocess.TimeoutExpired:
            worker.terminate()
            worker.wait(timeout=5)
        log.close()
        events.close()
    m.close_continuous_trace = close
    performance_stream = (m.OUT/'performance_trace.jsonl').open('a', buffering=1)
    original_save = m.save
    def save():
        original_save()
        performance_stream.write(json.dumps(json_value({
            'wall_time': time.time(), 'control': m.control, 'sim_time': m.control*m.world.dt,
            'status': report.get('status'), 'falls': report.get('falls'),
            'positions': report.get('positions'), 'speed_per_actor_m_s': report.get('speed_per_actor_m_s'),
            'performance': report.get('performance'),
            'trace': {k:v for k,v in report.items() if k.startswith('continuous_trace_') and k != 'continuous_trace_last_rejection'},
        }), separators=(',', ':'))+'\n')
    m.save = save
    report.update(continuous_trace_enabled=True, continuous_trace_queue_capacity=1,
        continuous_trace_ddpm_steps=m.planner.model.n_timesteps,
        continuous_trace_physics_waits_for_sampling=False,
        continuous_trace_max_age_sim_seconds=1.5,
        continuous_trace_stability_gate='all 51 XY targets within .10m and all speeds within .15m/s of independently admitted stable route; otherwise explicit route bridge',
        continuous_trace_progress_gate='at least .07m intended movement in next committed .5s unless actual map/body hazard; rejected intents use admitted bridge',
        pedestrian_planner_mode='continuous actual CUDA TRACE with current-state joint re-admission; explicitly labelled admitted route bridge',
        physics_held_during_planning=False, trace_compile_enabled=False,
        continuous_trace_rng='unseeded ongoing CUDA sampling; deterministic AMP; no RNG fork during physics',
        continuous_trace_worker_process=worker.pid,
        continuous_trace_worker_stream='independent pure Torch process; one request in flight; independent CUDA context',
        performance_trace_path=str(m.OUT/'performance_trace.jsonl'))
    m.label.text = 'Continuous CUDA TRACE + AMP; admitted route bridge'
