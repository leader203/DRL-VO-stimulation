"""Pure Torch TRACE inference process. Never imports or starts Isaac/Kit."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from CrowdSim.humanoid.diffusion.planner import DiffusionPlanner
import CrowdSim.humanoid.diffusion.guidance as gm
from preview_runtime_optimization import parallel_guidance_class, temporal_motion_admission, json_value

p = argparse.ArgumentParser()
p.add_argument('--run', type=Path, required=True)
p.add_argument('--owner', type=int, required=True)
args = p.parse_args()
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
source = json.loads((args.run/'SOURCE.json').read_text())
planner = DiffusionPlanner(source['resources']['diffusion_checkpoint']['path'],
    source['resources']['diffusion_config']['path'], device='cuda:0')
for parameter in planner.model.parameters():
    parameter.requires_grad_(False)
prediction = None
gm.motion_admission = temporal_motion_admission(gm.motion_admission)
gm.PedestrianGuidance = parallel_guidance_class(gm.PedestrianGuidance, robot_prediction=lambda: prediction)
(args.run/'TRACE_WORKER_READY.json').write_text(json.dumps({'pid':os.getpid(), 'owner':args.owner,
    'ddpm_steps':planner.model.n_timesteps, 'device':str(planner.device), 'kit_started':False})+'\n')
last_sequence = None
while Path(f'/proc/{args.owner}').exists():
    request_path = args.run/'TRACE_REQUEST.json'
    if not request_path.exists():
        time.sleep(.01); continue
    request = json.loads(request_path.read_text())
    if request.get('stop'):
        break
    if request['sequence'] == last_sequence:
        time.sleep(.01); continue
    last_sequence = request['sequence']
    with np.load(args.run/'trace_input.npz', allow_pickle=False) as data:
        packet = {k:torch.from_numpy(data[k].copy()) for k in data.files}
    packet.update(request['options'])
    packet['pedestrian_guidance'] = gm.PedestrianGuidanceConfig(**packet['pedestrian_guidance'])
    packet['seed'] = None
    prediction = request['prediction']
    started = time.perf_counter()
    result = {'sequence':last_sequence, 'control':request['control'], 'epoch':request['epoch']}
    try:
        plan = planner.plan(**packet)
        samples = plan.sample_states.detach().cpu().numpy()
        with (args.run/'trace_output.tmp').open('wb') as f:
            np.savez(f, sample_states=samples)
        (args.run/'trace_output.tmp').replace(args.run/'trace_output.npz')
        result.update(rejected=False, source_reference=plan.reference)
    except gm.TrajectoryAdmissionError as error:
        result.update(rejected=True, diagnostics=error.diagnostics)
    except Exception as error:
        import traceback
        result.update(fatal=True, error=str(error), traceback=traceback.format_exc())
    result['wall_seconds'] = time.perf_counter()-started
    temporary = args.run/'TRACE_RESPONSE.tmp'
    temporary.write_text(json.dumps(json_value(result))+'\n')
    temporary.replace(args.run/'TRACE_RESPONSE.json')
    if result.get('fatal'):
        break
