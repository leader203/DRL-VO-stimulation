"""Original native PPO with stateful source-derived reward, real physical transitions."""
from pathlib import Path
import hashlib
import io
import json
import math
import random
import time
import numpy as np
import torch
from native_ppo import NativePPO, TransitionBuffer, cpu_policy_state
from reward_logic import OriginalReward
from drlvo_observation_adapter import normalized_to_physical


class TrainingPolicy:
    def __init__(self, out):
        self.out = Path(out)
        self.batch = self.out.parents[1]
        self.session = self.batch/'training'
        self.session.mkdir(exist_ok=True)
        self.cfg = json.loads((self.batch/'TRAINING_CONFIG.json').read_text())
        run = json.loads((self.out/'ROBOT_CONTROL.json').read_text())
        self.phase = run['training_phase']
        self.agent = NativePPO(self.cfg, device=self.cfg['device'], seed=run['policy_action_seed'])
        self.model = self.agent.policy
        self.reward = OriginalReward(run['policy_action_seed']+40000,
            angular_threshold=self.cfg['reward']['angular_threshold_rad_s'])
        self.buffer = TransitionBuffer()
        self.steps = self.updates = self.episodes = 0
        self.pending = None
        self.episode_steps = 0
        self.episode_return = 0.
        self.done, self.smoke_pass, self.reason = False, False, None
        self.returns = []
        self.best_mean_return = -math.inf
        statefile = self.session/'session_state.pt'
        if self.phase == 'smoke' and statefile.exists():
            raise RuntimeError('Fresh smoke cannot load an existing session')
        if self.phase != 'smoke' and not statefile.exists():
            raise RuntimeError('Later phases require this experiment accepted smoke archive')
        if statefile.exists():
            payload = statefile.read_bytes()
            if hashlib.sha256(payload).hexdigest() != run.get('task_resume_checkpoint_sha256'):
                raise RuntimeError('Checkpoint changed or task resume provenance missing; prepare a new run')
            state = torch.load(io.BytesIO(payload), map_location=self.agent.device, weights_only=False)
            if state['schema'] != 'newmetro.native_sb3_original_logic.v1':
                raise RuntimeError('Incompatible simplified/failed checkpoint')
            if state.get('interrupted_nonterminal_buffer') or (state['buffer']['dones'] and not state['buffer']['dones'][-1]):
                raise RuntimeError('Cannot attach an unfinished trajectory to a newly prepared scene')
            self.model.load_state_dict(state['policy'], strict=True)
            self.agent.optimizer.load_state_dict(state['optimizer'])
            self.buffer.__dict__.update(state['buffer'])
            self.steps, self.updates, self.episodes = state['steps'], state['updates'], state['episodes']
            self.reward.restore(state['reward'])
            self.reward.reset_task()  # A newly prepared physical task, never an exact scene resume.
            self.returns = state['returns']
            self.best_mean_return = state['best_mean_return']
            torch.set_rng_state(state['cpu_rng'].cpu())
            np.random.set_state(state['numpy_rng'])
            random.setstate(state['python_rng'])
            if self.agent.device.type == 'cuda': torch.cuda.set_rng_state(state['cuda_rng'].cpu(), device=self.agent.device)
        self.initial_steps, self.initial_updates = self.steps, self.updates
        self.loaded_state_sha256 = self.fingerprint()
        self.transitions = (self.out/'transitions.jsonl').open('w', buffering=1)
        self.update_log = (self.out/'updates.jsonl').open('w', buffering=1)
        (self.out/'CHECKPOINT_RESTORE.json').write_text(json.dumps(dict(
            source=str(statefile) if statefile.exists() else self.cfg['model'],
            sha256=hashlib.sha256(statefile.read_bytes()).hexdigest() if statefile.exists() else self.cfg['model_sha256'],
            steps=self.steps, updates=self.updates, buffer_length=len(self.buffer),
            original_saved_optimizer_restored=True, CNN_trainable=all(p.requires_grad for p in self.model.features_extractor.parameters()),
            total_parameter_tensors=len(list(self.model.parameters())),
            trainable_parameter_tensors=sum(p.requires_grad for p in self.model.parameters()),
            model_fingerprint=self.loaded_state_sha256, exact_physical_scene_resume=False), indent=2)+'\n')

    def fingerprint(self):
        digest = hashlib.sha256()
        for name, value in self.model.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def command(self, obs, *, distance, minimum, control, sim_time, reward_context):
        if self.pending is not None:
            prev = self.pending
            reward, done, reason, terms = self.reward.step(distance=distance, minimum_scan=minimum,
                actual_v=reward_context['actual_v'], actual_w=reward_context['actual_w'],
                goal_base_xy=reward_context['goal_base_xy'], pedestrians_base_xy_vxy=reward_context['pedestrians'])
            self.buffer.add(prev['observation'], prev['raw'], reward, done, prev['logp'], prev['value'])
            self.steps += 1
            self.episode_steps += 1
            self.episode_return += reward
            self.transitions.write(json.dumps(dict(task_id=getattr(self,'task_id',None), total_step=self.steps,
                episode=self.episodes+1, episode_step=self.episode_steps, physical_control=control, sim_time=sim_time,
                action_start_control=prev['control'], action_start_sim_time=prev['sim_time'],
                action_duration_s=sim_time-prev['sim_time'], raw_action=prev['raw'].tolist(), effective_action=prev['clipped'].tolist(),
                log_probability=prev['logp'], value=prev['value'], reward=reward, reward_terms=terms, done=done, done_reason=reason,
                minimum_legacy_front_scan_m=minimum, minimum_full_dual_scan_m=reward_context['full_scan_minimum'],
                actual_forward_m_s=reward_context['actual_v'], actual_yaw_rad_s=reward_context['actual_w'],
                collision_signal='accumulated_legacy_lidar_distance_proxy' if reason=='collision' else None,
                contact_verified=False, previous_goal_distance=prev['distance'], goal_distance=distance,
                observation_scan_time_ns=reward_context['scan_timestamp_ns'], oracle_pedestrian_input=True, observation_size=19202))+'\n')
            self.pending = None
            if done:
                self.episodes += 1
                self.done, self.reason = True, reason
                self.returns.append(self.episode_return)
                with (self.session/'episodes.jsonl').open('a') as file:
                    file.write(json.dumps(dict(episode=self.episodes,total_steps=self.steps,steps=self.episode_steps,
                        return_sum=self.episode_return,reason=reason,phase=self.phase,task_id=getattr(self,'task_id',None)))+'\n')
            if len(self.buffer) == self.cfg['rollout_steps']:
                metrics = self.agent.update(self.buffer, 0. if done else self.agent.value(obs))
                self.updates += 1
                metrics.update(total_steps=self.steps, update=self.updates, real_physical_rollout=True,
                               oracle_pedestrian_input=True, CNN_frozen=False)
                self.update_log.write(json.dumps(metrics)+'\n')
                self.checkpoint()
                if self.phase == 'smoke' and self.steps >= self.cfg['smoke_real_transitions']:
                    self.smoke_pass, self.done, self.reason = True, True, 'REAL512_NATIVE_PPO_SMOKE_PASS'
            if self.steps % 5000 == 0 and self.returns:
                mean = float(np.mean(self.returns[-100:]))
                if mean > self.best_mean_return:
                    self.best_mean_return = mean
                    self.agent.engine.num_timesteps = self.steps
                    self.agent.engine.save(self.session/'best_model.zip')
            if self.steps % 20000 == 0:
                self.agent.engine.num_timesteps = self.steps
                self.agent.engine.save(self.session/f'model_{self.steps}.zip')
            if self.steps >= self.cfg['total_timesteps'] and len(self.buffer)==0:
                self.done, self.reason = True, 'total_timesteps_reached'
        if self.done:
            return dict(raw_action=None, clipped_action=[-1.,0.], desired_v_m_s=0., desired_w_rad_s=0.,
                        owner='DRLVO_TRAIN_TERMINAL', training_done=True, training_reason=self.reason)
        raw, clipped, logp, value = self.agent.select_action(obs)
        v, w = normalized_to_physical(clipped)
        self.pending = dict(observation=obs.copy(), raw=raw.copy(), clipped=clipped.copy(), logp=logp,
                            value=value, distance=distance, control=control, sim_time=sim_time)
        return dict(raw_action=raw.tolist(),clipped_action=clipped.tolist(),desired_v_m_s=float(v),desired_w_rad_s=float(w),
                    value=value, log_probability=logp, owner='DRLVO_NATIVE_SB3_PPO', training_done=False)

    def checkpoint(self):
        state = dict(schema='newmetro.native_sb3_original_logic.v1', policy=cpu_policy_state(self.model),
            optimizer=self.agent.optimizer.state_dict(), buffer=self.buffer.__dict__, reward=self.reward.state(),
            steps=self.steps, updates=self.updates, episodes=self.episodes, returns=self.returns,
            best_mean_return=self.best_mean_return, cpu_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
            python_rng=random.getstate(), cuda_rng=torch.cuda.get_rng_state(self.agent.device) if self.agent.device.type=='cuda' else None,
            interrupted_nonterminal_buffer=bool(getattr(self,'interrupted',False) and len(self.buffer) and not self.buffer.dones[-1]))
        if hasattr(self,'task_state_provider'): state['task_sampling'] = self.task_state_provider()
        temporary = self.session/'session_state.tmp'
        torch.save(state, temporary)
        temporary.replace(self.session/'session_state.pt')
        from fall_recovery import preserve_boundary
        preserve_boundary(self.session, state)
        temporary = self.session/'latest.tmp'
        torch.save(state['policy'], temporary)
        temporary.replace(self.session/'latest.pt')
        progress = dict(phase=self.phase,total_steps=self.steps,total_updates=self.updates,episodes=self.episodes,
            episode_steps=self.episode_steps,smoke_pass=self.smoke_pass,reason=self.reason,
            initial_state_sha256=self.loaded_state_sha256,final_state_sha256=self.fingerprint(),updated_at=time.time(),
            real_rollouts=True,original_model_unchanged=True)
        (self.out/'TRAINING_STATUS.json').write_text(json.dumps(progress,indent=2)+'\n')
        temporary = self.session/'PROGRESS.tmp'
        temporary.write_text(json.dumps(progress,indent=2)+'\n')
        temporary.replace(self.session/'PROGRESS.json')

    def reset_episode(self):
        if not self.done or self.pending is not None or self.reason not in ('collision','goal','timeout'):
            raise RuntimeError('Invalid task reset boundary')
        self.done, self.reason, self.smoke_pass = False, None, False
        self.episode_steps, self.episode_return = 0, 0.
        self.reward.reset_task()

    def close(self, failed=False):
        self.interrupted = self.pending is not None or (not self.done and bool(self.buffer.dones) and not self.buffer.dones[-1])
        self.pending = None
        if failed:
            torch.save(cpu_policy_state(self.model),self.out/'failed_candidate.pt')
            (self.out/'TRAINING_REJECTED.json').write_text(json.dumps(dict(total_steps=self.steps,reason=self.reason,accepted_as_next_training_episode=False)))
        elif not self.interrupted:
            self.checkpoint()
        else:
            (self.out/'INTERRUPTION.json').write_text(json.dumps(dict(real_completed_steps=self.steps,pending_discarded=True,
                partial_buffer_discarded=True,no_synthetic_reward=True,no_synthetic_done=True)))
        self.transitions.close()
        self.update_log.close()
