"""Use the checkpoint's real SB3 0.10.0 PPO and protected original CustomCNN."""
from pathlib import Path
import collections
import hashlib
import importlib.util
import json
import io
import sys
import zipfile
import os

import numpy as np
import torch

BATCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BATCH/'reference/python'))
import gym
from stable_baselines3 import PPO
from stable_baselines3.common import logger
from stable_baselines3.common.policies import ActorCriticCnnPolicy


def original_cnn():
    name = '_newmetro_original_custom_cnn'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, BATCH/'reference/custom_cnn_full.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].CustomCNN


def load_native_agent(model, *, device='cpu', seed=1337, rollout_steps=512, batch_size=128, epochs=10):
    # v0.10 cannot override serialized Python-3.8 closures during load. Use its
    # native constructor and set_parameters with the SAME saved policy/Adam states.
    class ShapeOnlyEnv(gym.Env):
        observation_space = gym.spaces.Box(-1., 1., shape=(19202,), dtype=np.float32)
        action_space = gym.spaces.Box(-1., 1., shape=(2,), dtype=np.float32)
        def seed(self, value=None): return [value]
        def reset(self): return np.zeros(19202, np.float32)
        def step(self, action): raise RuntimeError('Shape-only env cannot execute transitions')
    if model is not None:
        with zipfile.ZipFile(model) as archive:
            metadata = json.loads(archive.read('data'))
            kwargs = metadata['policy_kwargs']
            if kwargs['net_arch'] != [dict(pi=[256], vf=[128])]:
                raise RuntimeError('Unsupported checkpoint architecture')
            parameters = {key: torch.load(io.BytesIO(archive.read(key+'.pth')),
                                          map_location=device, weights_only=False)
                          for key in ('policy', 'policy.optimizer')}
    agent = PPO(ActorCriticCnnPolicy, ShapeOnlyEnv(), device=device,
                policy_kwargs=dict(features_extractor_class=original_cnn(),
                    features_extractor_kwargs=dict(features_dim=256),
                    net_arch=[dict(pi=[256], vf=[128])]),
                n_steps=rollout_steps, batch_size=batch_size, n_epochs=epochs,
                learning_rate=1e-3, seed=int(seed), verbose=0, tensorboard_log=None)
    if model is not None:
        agent.set_parameters(parameters, exact_match=True, device=device)
    agent.policy.train(True)  # v0.10.0's collect_rollouts does not switch to eval.
    if not all(p.requires_grad for p in agent.policy.parameters()):
        raise RuntimeError('Original loaded policy unexpectedly contains frozen parameters')
    return agent


class TransitionBuffer:
    def __init__(self):
        self.first_episode_start = True
        self.clear()

    def clear(self):
        for key in ('observations', 'actions', 'rewards', 'dones', 'log_probabilities', 'values'):
            setattr(self, key, [])

    def __len__(self):
        return len(self.rewards)

    def add(self, observation, action, reward, done, log_probability, value):
        obs, act = np.asarray(observation, np.float32), np.asarray(action, np.float32)
        if obs.shape != (19202,) or act.shape != (2,):
            raise ValueError('Transition dimensions damaged')
        if not np.isfinite(obs).all() or not np.isfinite(act).all() or not np.isfinite([reward, log_probability, value]).all():
            raise ValueError('Non-finite transition')
        for key, item in dict(observations=obs.copy(), actions=act.copy(), rewards=float(reward),
                              dones=bool(done), log_probabilities=float(log_probability), values=float(value)).items():
            getattr(self, key).append(item)


class NativePPO:
    def __init__(self, config, *, device, seed):
        self.config = config
        self.device = torch.device(device)
        if config['initialization'] != 'random_native_PPO_constructor_no_pretrained_no_old_checkpoint' or config['model'] is not None:
            raise RuntimeError('This batch requires random initialization and forbids old weights')
        self.engine = load_native_agent(None, device=device, seed=seed,
            rollout_steps=config['rollout_steps'], batch_size=config['ppo']['batch_size'],
            epochs=config['ppo']['update_epochs'])
        if self.engine.policy.optimizer.state:
            raise RuntimeError('Fresh optimizer unexpectedly has state')
        self.policy = self.engine.policy
        self.optimizer = self.policy.optimizer
        logger.configure(str(BATCH/'audit/native_ppo_logs'), format_strings=[])

    def select_action(self, observation):
        self.policy.train(True)
        with torch.no_grad():
            obs = torch.as_tensor(np.asarray(observation, np.float32)[None], device=self.device)
            action, value, logp = self.policy.forward(obs)
        raw = action[0].cpu().numpy()
        return raw, np.clip(raw, -1., 1.), float(logp.item()), float(value.item())

    def value(self, observation):
        self.policy.train(True)
        with torch.no_grad():
            # Same native forward used for rollout bootstrap in SB3 0.10.0.
            _, value, _ = self.policy.forward(torch.as_tensor(observation[None], device=self.device))
        return float(value.item())

    def update(self, buffer, bootstrap_value):
        rb = self.engine.rollout_buffer
        if len(buffer) != rb.buffer_size:
            raise RuntimeError('Native PPO requires the full configured rollout')
        rb.reset()
        previous_done = buffer.first_episode_start
        if os.environ.get('NEWMETRO_BULK_PPO', '1') == '1':
            # Same FP32 host storage as SB3 add(), no per-transition GPU scalars.
            rb.observations[:,0] = np.asarray(buffer.observations, np.float32)
            rb.actions[:,0] = np.asarray(buffer.actions, np.float32)
            rb.rewards[:,0] = np.asarray(buffer.rewards, np.float32)
            rb.dones[:,0] = np.asarray([previous_done]+buffer.dones[:-1],np.float32)
            rb.values[:,0] = np.asarray(buffer.values,np.float32)
            rb.log_probs[:,0] = np.asarray(buffer.log_probabilities,np.float32)
            rb.pos=rb.buffer_size;rb.full=True
            previous_done=buffer.dones[-1]
            rb.compute_returns_and_advantage(torch.tensor([bootstrap_value]),np.array([previous_done]))
        else:
            for obs, action, reward, done, logp, value in zip(buffer.observations, buffer.actions,
                    buffer.rewards, buffer.dones, buffer.log_probabilities, buffer.values):
                rb.add(obs, action, reward, previous_done,
                       torch.tensor([value], device=self.device), torch.tensor([logp], device=self.device))
                previous_done = done
            rb.compute_returns_and_advantage(torch.tensor([bootstrap_value], device=self.device),
                                             np.array([previous_done]))
        before = {name: p.detach().clone() for name, p in self.policy.named_parameters()}
        if os.environ.get('NEWMETRO_CACHE_ROLLOUT_GPU','1')=='1':
            from stable_baselines3.common.type_aliases import RolloutBufferSamples
            # Preserve SB3's env-major flattening and NumPy permutations.
            cached=[torch.as_tensor(rb.swap_and_flatten(getattr(rb,k)),device=self.device)
                    for k in ('observations','actions','values','log_probs','advantages','returns')]
            original_get=rb._get_samples
            def cached_samples(indices,env=None):
                index=torch.as_tensor(indices,device=self.device)
                data=[t.index_select(0,index) for t in cached]
                return RolloutBufferSamples(data[0],data[1],*(x.flatten() for x in data[2:]))
            rb._get_samples=cached_samples
            try:self.engine.train()
            finally:rb._get_samples=original_get
        else:self.engine.train()
        after = dict(self.policy.named_parameters())
        changed = [name for name in before if not torch.equal(before[name], after[name].detach())]
        metrics = logger.get_log_dict()
        output = dict(policy_loss=float(metrics['train/policy_gradient_loss']),
                      value_loss=float(metrics['train/value_loss']),
                      entropy=-float(metrics['train/entropy_loss']),
                      approx_kl=float(metrics['train/approx_kl']),
                      clip_fraction=float(metrics['train/clip_fraction']),
                      transitions=len(buffer), batches=self.engine.n_epochs*(len(buffer)//self.engine.batch_size),
                      changed_actor_tensors=[n for n in changed if n.startswith(('action_net.', 'mlp_extractor.policy_net.')) or n=='log_std'],
                      changed_cnn_parameters=[n for n in changed if n.startswith('features_extractor.')],
                      optimizer_epsilon=float(self.optimizer.param_groups[0]['eps']),
                      optimizer_learning_rate=float(self.optimizer.param_groups[0]['lr']),
                      sb3_version='0.10.0')
        if not np.isfinite([output[k] for k in ('policy_loss', 'value_loss', 'entropy', 'approx_kl', 'clip_fraction')]).all():
            raise RuntimeError('Native PPO produced non-finite metrics')
        buffer.first_episode_start = previous_done
        buffer.clear()
        return output


def cpu_policy_state(policy):
    return {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()}
