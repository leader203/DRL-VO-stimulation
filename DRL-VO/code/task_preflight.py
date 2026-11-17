"""Verify saved CPU task checks and unchanged resources without importing Isaac/PPO."""
import ast
import hashlib
import json
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(8*1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    batch = Path(__file__).resolve().parents[1]
    report = json.loads((batch/'OFFLINE_TASK_CHECK.json').read_text())
    assert report['status'] == 'OFFLINE_TASK_AND_MOCK_RESET_PASS'
    core = ('task_sampling.py', 'drlvo_heading_planner.py', 'sfm_training_bridge.py',
            'training_driver.py', 'training_policy.py', 'robot_preview_adapter.py',
            'validate_random_tasks_cpu.py')
    for name in core:
        assert digest(batch/'code'/name) == report['code_hashes'][name], f'Rerun task checks: {name}'
    source = json.loads((batch/'scene/SOURCE.json').read_text())
    resources = dict(robot=dict(path=source['robot_usd'], sha256=source['robot_sha256']),
                     scene=source['scene_source'], AMP=dict(path=source['checkpoint'], sha256=source['checkpoint_sha256']))
    config = json.loads((batch/'TRAINING_CONFIG.json').read_text())
    resources['DRLVO'] = dict(path=config['model'],sha256=config['model_sha256'])
    adaptation = json.loads((batch/'audit/ADAPTATION_CPU_CHECK.json').read_text())
    assert adaptation['status']=='SOURCE_REWARD_DONE_OBSERVATION_TIMING_PASS'
    for name in ('reward_logic.py','local_goal.py','drlvo_observation_adapter.py','native_ppo.py'):
        assert digest(batch/'code'/name)==adaptation['tested_core_hashes'][name], name
    diagnosis = json.loads((batch/'audit/ACTION_DIAGNOSIS.json').read_text())
    assert diagnosis['reconstructed_loader_mismatch_observed'] is False
    for name, entry in resources.items():
        assert digest(entry['path']) == entry['sha256'], f'Resource changed: {name}'
    for path in (batch/'code').glob('*.py'): compile(path.read_text(), str(path), 'exec')
    result = dict(status='OFFLINE_TASK_PREFLIGHT_PASS', resources=resources,
                  CPU_tasks_and_mock_resets='PASS', start_count=report['source_start_count'],
                  config_hashes={name:digest(batch/name) for name in ('TASK_CONFIG.json', 'TRAINING_CONFIG.json', 'START_CANDIDATES.json')},
                  code_hashes={p.name:digest(p) for p in (batch/'code').glob('*.py')},
                  real_simulator_reset='NOT_EXECUTED', current_PhysX_routes='NOT_EXECUTED',
                  physical_contact_safety='UNVERIFIED', action_saturation_cause='original_checkpoints_also_saturate_on_identical_NewMetro_inputs; observation_distribution_not_ruled_out',
                  native_reference_hashes={str(p.relative_to(batch)):digest(p) for p in (batch/'reference').rglob('*.py')},
                  complete_original_DRLVO_alignment=False, training_started=False)
    (batch/'OFFLINE_PREFLIGHT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('code_hashes', 'resources','native_reference_hashes')}))


if __name__ == '__main__': main()
