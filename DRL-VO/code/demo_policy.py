"""Strict frozen native SB3 policy; no optimizer steps or training transitions."""
from pathlib import Path
import hashlib,json,math
import numpy as np
import torch
from native_ppo import load_native_agent
from reward_logic import OriginalReward
from drlvo_observation_adapter import normalized_to_physical

def fingerprint(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

class DemoPolicy:
    def __init__(self,out):
        self.out=Path(out);batch=self.out.parents[1]
        self.cfg=json.loads((batch/'TRAINING_CONFIG.json').read_text())
        run=json.loads((self.out/'ROBOT_CONTROL.json').read_text())
        self.trained_lookahead=float(self.cfg['local_goal_lookahead_m'])
        lookahead=float(run.get('local_goal_lookahead_m',self.trained_lookahead))
        if not math.isfinite(lookahead) or lookahead<=0:raise ValueError('Demo lookahead must be finite and positive')
        self.cfg['local_goal_lookahead_m']=lookahead
        checkpoint=Path(run['model_checkpoint']);self.checkpoint=str(checkpoint)
        self.checksum=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if self.checksum!=run['model_sha256']:raise RuntimeError('Frozen checkpoint changed')
        agent=load_native_agent(self.cfg['model'],device=self.cfg['device'],seed=1337,
            rollout_steps=self.cfg['rollout_steps'],batch_size=self.cfg['ppo']['batch_size'],epochs=self.cfg['ppo']['update_epochs'])
        self.model=agent.policy
        self.model.load_state_dict(torch.load(checkpoint,map_location=self.cfg['device'],weights_only=True),strict=True)
        self.model.eval()
        for parameter in self.model.parameters():parameter.requires_grad_(False)
        self.loaded_fingerprint=fingerprint(self.model)
        self.steps=self.initial_steps=self.updates=0
        self.pending=None;self.reason=None
        self.reward=OriginalReward(41337,angular_threshold=self.cfg['reward']['angular_threshold_rad_s'])
        (self.out/'MODEL_BINDING.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),sha256=self.checksum,
            model_fingerprint=self.loaded_fingerprint,source_training_steps=480000,source_training_updates=937,
            policy_device=str(next(self.model.parameters()).device),all_parameters_frozen=True,
            source_trained_lookahead_m=self.trained_lookahead,demo_local_goal_lookahead_m=lookahead,
            inference='deterministic distribution mean; original clipping and physical mapping',
            optimizer_updates=0,training=False),indent=2)+'\n')

    def reset_episode(self):
        self.pending=None;self.reason=None;self.reward.reset_task()

    def command(self,obs,*,distance,minimum,control,sim_time,reward_context):
        obs=np.asarray(obs,dtype=np.float32)
        if obs.shape!=(19202,) or not np.isfinite(obs).all():raise RuntimeError('Invalid observation')
        terms=None
        if self.pending is not None:
            _,done,reason,terms=self.reward.step(distance=distance,minimum_scan=minimum,
                actual_v=reward_context['actual_v'],actual_w=reward_context['actual_w'],
                goal_base_xy=reward_context['goal_base_xy'],pedestrians_base_xy_vxy=reward_context['pedestrians'])
            if done:self.reason=reason
        if self.reason:
            return dict(raw_action=None,clipped_action=[-1.,0.],desired_v_m_s=0.,desired_w_rad_s=0.,
                owner='FROZEN_DRLVO_DEMO',training_done=True,training_reason=self.reason,
                goal_distance_m=distance,minimum_full_scan_m=reward_context['full_scan_minimum'],
                minimum_legacy_front_scan_m=minimum,done_terms=terms)
        with torch.no_grad():
            action,value,logp=self.model.forward(torch.as_tensor(obs[None],device=self.cfg['device']),deterministic=True)
        raw=action[0].cpu().numpy();clipped=np.clip(raw,-1.,1.);v,w=normalized_to_physical(clipped)
        self.steps+=1;self.pending=True
        return dict(raw_action=raw.tolist(),clipped_action=clipped.tolist(),desired_v_m_s=float(v),
            desired_w_rad_s=float(w),value=float(value.item()),owner='FROZEN_DRLVO_DEMO',training_done=False,
            goal_distance_m=distance,minimum_full_scan_m=reward_context['full_scan_minimum'],
            minimum_legacy_front_scan_m=minimum)

    def verify_frozen(self):
        current=fingerprint(self.model)
        if current!=self.loaded_fingerprint:raise RuntimeError('Frozen policy weights changed')
        return current
