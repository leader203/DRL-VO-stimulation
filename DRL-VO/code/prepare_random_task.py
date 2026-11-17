"""Prepare a fresh random task and run inputs, without creating SimulationApp."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil

from task_sampling import TaskSampler, make_navigation


def restore_task_sampler(batch, sampler, phase):
    """Resume the task RNG before drawing the new scene's start or goal."""
    if phase == 'smoke':
        return None
    import torch
    checkpoint = Path(batch)/'training/session_state.pt'
    payload = checkpoint.read_bytes()
    state = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)
    if state.get('schema') != 'newmetro.native_sb3_original_logic.v1':
        raise RuntimeError('Incompatible task-resume checkpoint')
    if state.get('interrupted_nonterminal_buffer') or (state['buffer']['dones'] and not state['buffer']['dones'][-1]):
        raise RuntimeError('Cannot attach an unfinished trajectory to a newly prepared scene')
    saved = state.get('task_sampling', {}).get('sampler')
    if saved is None:
        raise RuntimeError('Checkpoint lacks task sampler state; refusing to restart its RNG silently')
    sampler.restore(saved)
    return dict(path=str(checkpoint.resolve()), sha256=hashlib.sha256(payload).hexdigest(),
                previous_task_id=sampler.task_id, checkpoint_steps=state['steps'])


def main():
    batch = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--phase', choices=('smoke', 'long'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    if not args.name.replace('_', '').replace('-', '').isalnum():
        parser.error('simple unique run name required')
    out = batch/'runs'/args.name
    if out.exists():
        parser.error('Run directory already exists')
    task_seed = 23000+args.seed
    sampler = TaskSampler(batch, make_navigation(batch), task_seed)
    resumed = restore_task_sampler(batch, sampler, args.phase)
    start = sampler.random_start()
    task = sampler.sample([*start['xy_m'], start['yaw_rad']])
    task['spawn_candidate_id'] = start['id']
    task['sampler_state'] = sampler.state()
    task['sampler_resume'] = resumed
    out.mkdir()
    for name in ('SOURCE.json', 'scene_config.json', 'profile.json', 'preview_manifest.json'):
        shutil.copy2(batch/'scene'/name, out/name)
    source = json.loads((out/'SOURCE.json').read_text())
    source.update(robot_spawn_xy=start['xy_m'], robot_spawn_yaw_rad=start['yaw_rad'],
                  robot_goal_xy=task['goal_xy'], random_task_id=task['task_id'])
    (out/'SOURCE.json').write_text(json.dumps(source, indent=2)+'\n')
    (out/'INITIAL_TASK.json').write_text(json.dumps(task, indent=2)+'\n')
    config = dict(training_phase=args.phase, policy_action_seed=1337+args.seed,
                  task_resume_checkpoint_sha256=None if resumed is None else resumed['sha256'],
                  task_sampling_seed=task_seed, robot_goal=task['goal_xy'],
                  wall_limit_s=1200 if args.phase == 'smoke' else 43200,
                  mode='USER_START_RANDOM_GOAL_FOUR_SFM_AMP_PHYSICAL_EVT_DRLVO',
                  sensor=dict(count=2, beams_each=2000, rate_hz=15, range_m=[.1, 8.]))
    (out/'ROBOT_CONTROL.json').write_text(json.dumps(config, indent=2)+'\n')
    print(out)


if __name__ == '__main__':
    main()
