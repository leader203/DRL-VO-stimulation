"""Isaac-side callers of the unchanged, successful Gazebo PPO implementation."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from online_ppo import OnlinePPO, PPOConfig

MODEL = Path('/home/user/navigation_project/a_pipeline/github_src/drl_vo_nav-drl_vo/drl_vo/src/model/drl_vo.zip')
MODEL_SHA256 = '78a6bc9918b092e2bd26782664248d9ce06a5498e7cd086260dbfe6548f99645'
CONTROL_PERIOD = 0.1
LIDAR_RANGE = (0.1, 8.0)
LOOKAHEAD = 1.0  # Existing fixed-dual Gazebo launch argument.


def verified_model():
    if hashlib.sha256(MODEL.read_bytes()).hexdigest() != MODEL_SHA256:
        raise RuntimeError('Gazebo initialization model changed; stop before training')
    return MODEL


class CheckedPPO(OnlinePPO):
    """Same PPO math/config; reject inconsistent action probabilities before update."""
    def __init__(self, policy, device, seed=1337):
        super().__init__(policy, device, PPOConfig(), seed=seed)

    def update(self, buffer, bootstrap_value):
        self.policy.eval()
        discrepancies = []
        rechecked_single_max_error = 0.
        rechecked_count = 0
        # Small chunks avoid a second rollout-sized CNN allocation.
        with torch.no_grad():
            for start in range(0, len(buffer), 16):
                obs = torch.as_tensor(np.stack(buffer.observations[start:start+16]), device=self.device)
                actions = torch.as_tensor(np.stack(buffer.actions[start:start+16]), device=self.device)
                dist, _ = self._distribution(obs)
                current = dist.log_prob(actions).sum(-1).cpu().numpy()
                old = np.asarray(buffer.log_probabilities[start:start+16])
                errors = np.abs(current-old)
                if not np.isfinite(errors).all():
                    raise RuntimeError('PPO nonfinite raw-action/log-prob before update')
                discrepancies.extend(errors.tolist())
                # Sampling used one observation. Batched CNN arithmetic can
                # differ slightly; verify only apparent mismatches in that mode.
                for offset in np.flatnonzero(errors > 2e-4):
                    single_dist, _ = self._distribution(obs[offset:offset+1])
                    single = float(single_dist.log_prob(actions[offset:offset+1]).sum())
                    single_error = abs(single-float(old[offset]))
                    rechecked_count += 1
                    rechecked_single_max_error = max(rechecked_single_max_error, single_error)
                    if not np.isfinite(single_error) or single_error > 2e-4:
                        raise RuntimeError(
                            f'PPO raw-action/log-prob mismatch before update: '
                            f'batch={float(errors[offset])}, single={single_error}, '
                            f'buffer_index={start+int(offset)}')
        max_error = max(discrepancies, default=0.)
        before = {k: v.detach().clone() for k, v in self.policy.state_dict().items()
                  if k == 'log_std' or k.startswith(('mlp_extractor.policy_net.', 'action_net.'))}
        metrics = super().update(buffer, bootstrap_value)
        changed = [k for k, v in before.items() if not torch.equal(v, self.policy.state_dict()[k])]
        metrics.update(preupdate_logprob_max_error=max_error,
                       preupdate_single_recheck_max_error=rechecked_single_max_error,
                       preupdate_single_recheck_count=rechecked_count,
                       changed_actor_tensors=changed)
        if not changed:
            raise RuntimeError('PPO update changed no Actor tensors; stop preserving checkpoint evidence')
        if not np.isfinite(metrics['approx_kl']) or metrics['approx_kl'] > .1:
            raise RuntimeError(f"PPO KL acceptance failed: {metrics['approx_kl']}")
        if metrics['clip_fraction'] >= .99:
            raise RuntimeError('PPO clipping saturated; stop before another rollout')
        return metrics


def path_local_goal(native, robot_xy, goal_xy, yaw, lookahead=LOOKAHEAD):
    """Follow admitted A* path; never substitute a straight line when disconnected."""
    path = native.astar_path(np.asarray(robot_xy), np.asarray(goal_xy))
    if path is None:
        raise RuntimeError('Robot has no admitted A* path to goal; stop without straight-line fallback')
    distances = np.linalg.norm(path-np.asarray(robot_xy), axis=1)
    index = int(np.argmin(distances))
    target, travelled = path[-1], 0.
    for j in range(index, len(path)-1):
        travelled += float(np.linalg.norm(path[j+1]-path[j]))
        if travelled >= lookahead:
            target = path[j+1]
            break
    from drlvo_observation_adapter import rotate_world_to_base
    local = rotate_world_to_base((target-np.asarray(robot_xy))[None], yaw)[0]
    return local, path
