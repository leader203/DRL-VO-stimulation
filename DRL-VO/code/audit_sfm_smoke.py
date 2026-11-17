"""Audit actual 512-transition native PPO smoke; report saturation separately."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import zipfile
import torch
import numpy as np

B = Path(__file__).resolve().parents[1]
out = Path(sys.argv[1]).resolve()
assert out.parent==B/'runs'
report=json.loads((out/'REPORT.json').read_text())
restore=json.loads((out/'CHECKPOINT_RESTORE.json').read_text())
transitions=[json.loads(x) for x in (out/'transitions.jsonl').read_text().splitlines() if x]
updates=[json.loads(x) for x in (out/'updates.jsonl').read_text().splitlines() if x]
rows=list(csv.DictReader((out/'robot_motion.csv').open()))
poses={int(r['control']):(float(r['root_x']),float(r['root_y'])) for r in rows}
movements=[math.dist(poses[t['action_start_control']],poses[t['physical_control']]) for t in transitions
           if t['action_start_control'] in poses and t['physical_control'] in poses]
state=torch.load(B/'training/session_state.pt',map_location='cpu',weights_only=False)
cfg=json.loads((B/'TRAINING_CONFIG.json').read_text())
with zipfile.ZipFile(cfg['model']) as z:
 original=torch.load(io.BytesIO(z.read('policy.pth')),map_location='cpu',weights_only=False)
cnn_changed=[n for n in state['policy'] if n.startswith('features_extractor.') and not torch.equal(state['policy'][n].cpu(),original[n].cpu())]
resets=[json.loads(s) for s in (out/'RESET_EVENTS.jsonl').read_text().splitlines()] if (out/'RESET_EVENTS.jsonl').exists() else []
raw=np.array([t['raw_action'] for t in transitions])
clip_fraction=float(np.mean(np.abs(raw[:,0])>1)) if len(raw) else None
checks=dict(initial_zero_steps_updates=restore['steps']==restore['updates']==0,
 original_saved_Adam_restored=restore['original_saved_optimizer_restored'],
 CNN_trainable=restore['CNN_trainable'],
 real512=len(transitions)==512 and [t['total_step'] for t in transitions]==list(range(1,513)),
 native_update512_128_10=len(updates)==1 and updates[0]['batches']==40 and updates[0]['transitions']==512 and updates[0]['sb3_version']=='0.10.0',
 CNN_parameters_updated=bool(updates and updates[0]['changed_cnn_parameters']) and bool(cnn_changed),
 actor_updated=bool(updates and updates[0]['changed_actor_tensors']),
 finite_update=bool(updates) and all(math.isfinite(float(updates[0][k])) for k in ('policy_loss','value_loss','entropy','approx_kl','clip_fraction')),
 real_wheels=max((abs(float(r[k])) for r in rows for k in ('wheel_fl','wheel_fr','wheel_rl','wheel_rr')),default=0.)>.1,
 physical_action_displacement_excluding_teleports=bool(movements) and max(movements)>.002,
 causal_observations=all(t['observation_scan_time_ns']<=round(t['sim_time']*1e9) for t in transitions),
 quantized20Hz=all(min(abs(t['action_duration_s']-1/30),abs(t['action_duration_s']-2/30))<1e-7 for t in transitions),
 all_reward_terms_added=all(abs(t['reward']-sum(t['reward_terms'][k] for k in ('goal','obstacle','angular','vo')))<1e-9 for t in transitions),
 actual_yaw_threshold13=all(abs(t['reward_terms']['angular']-(-.1*abs(t['actual_yaw_rad_s']) if abs(t['actual_yaw_rad_s'])>1.3 else 0.))<1e-9 for t in transitions),
 task_ids=all(isinstance(t['task_id'],int) for t in transitions),
 post_update_robot_reset=report.get('smoke_post_update_robot_reset')=='PASS',
 checkpoint512=state['steps']==512 and state['updates']==1,
 four_SFM_AMP=report.get('actor_count')==4 and report.get('pedestrian_planner_mode')=='SFM to AMP')
result=dict(passed=all(checks.values()),checks=checks,new_real_transitions=len(transitions),valid_ppo_updates=len(updates),
 training_ready=all(checks.values()) and clip_fraction is not None and clip_fraction<.99,
 linear_action_saturation_fraction=clip_fraction,action_saturation_resolved=clip_fraction is not None and clip_fraction<.99,
 episode_done_reasons=[t['done_reason'] for t in transitions if t['done']],
 actual_success_keep_pose=('PASS' if any(e['reason']=='goal' and not e['teleported'] for e in resets) else 'NOT_OBSERVED'),
 actual_failure_random_reset=('PASS' if any(e['reason'] in ('collision','timeout') and e['teleported'] for e in resets) else 'NOT_OBSERVED'),
 full_512_128_10_update='PASS' if checks['native_update512_128_10'] else 'FAIL',
 checkpoint=str(B/'training/session_state.pt'),checkpoint_sha256=hashlib.sha256((B/'training/session_state.pt').read_bytes()).hexdigest(),
 contact_safety='UNVERIFIED',goal_arrivals=sum(t['done_reason']=='goal' and t['done'] for t in transitions),
 independent_evaluation='NOT_EXECUTED',long_training_started=False)
(B/'training/SMOKE_PASS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
raise SystemExit(0 if result['passed'] else 1)
