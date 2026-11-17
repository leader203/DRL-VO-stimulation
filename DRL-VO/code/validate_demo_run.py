"""Audit an owned, completed two-goal PhysX demo from recorded evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    batch = Path(__file__).resolve().parents[1]
    read = lambda name: json.loads((run / name).read_text())
    lines = lambda name: [json.loads(line) for line in (run / name).read_text().splitlines()]
    report, exit_report = read('REPORT.json'), read('EXIT.json')
    assert exit_report['demo_success'] and exit_report['exit_code'] == 0
    assert report['status'] == 'DEMO_USER_CLOSED'
    assert report['training'] is False and report['RL_training'] is False
    assert report['actor_count'] == 4 and report['falls'] == 0
    assert not report.get('error')
    frozen = read('FROZEN_POLICY_CHECK.json')
    assert frozen['unchanged'] and frozen['optimizer_updates'] == 0
    assert frozen['model_fingerprint'] == 'd5abb3e46251825fa1ef1e0f66fbfe2f9d273f9068dad9588b947219fdf2b81a'
    events = lines('GOAL_EVENTS.jsonl')
    assert [e['event'] for e in events] == ['accepted', 'ready', 'accepted', 'ready']
    ready = [e for e in events if e['event'] == 'ready']
    assert all(e['robot_stopped'] and e['arrival_dwell_s'] >= .5 and e['final_goal_distance_m'] <= .3 for e in ready)
    scans = lines('lidar.jsonl')
    assert len(scans) == report['lidar_scan_count'] and len(scans) > 100
    assert np.allclose(np.diff([s['sim_time'] for s in scans]), 1 / 15, atol=1e-8)
    assert all(s['stats']['beams_per_sensor'] == 2000 and s['stats']['missed_scheduled_scans'] == 0 for s in scans)
    assert all(s['fixed_self_mask_beams'] == {'scan_01': 1006, 'scan_02': 1017} for s in scans)
    policy = lines('policy.jsonl')
    actions = [p for p in policy if p['raw_action'] is not None]
    assert len(actions) == report['demo_actions']
    for p in policy:
        assert p['history_sampling_hz'] == 20
        assert all(source <= sample <= round(p['sim_time'] * 1e9) for source, sample in zip(p['history_source_scan_times_ns'], p['history_sample_times_ns']))
        assert not p['external_safety_override']
    intervals = []
    for a, b in zip(policy, policy[1:]):
        if a['global_goal'] == b['global_goal']:
            intervals.append(b['control'] - a['control'])
    assert set(intervals) <= {1, 2}
    with (run / 'robot_motion.csv').open() as stream:
        motion = list(csv.DictReader(stream))
    displacement = float(np.hypot(float(motion[-1]['root_x']) - float(motion[0]['root_x']), float(motion[-1]['root_y']) - float(motion[0]['root_y'])))
    assert displacement > 3
    for event in ready:
        row = next(r for r in motion if int(r['control']) == event['control'])
        assert abs(float(row['actual_forward_v'])) < .05 and abs(float(row['actual_yaw_w'])) < .10
    provenance = json.loads((batch / 'PROVENANCE.json').read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in provenance['protected_sha256'].items())
    result = dict(status='TWO_GOAL_PHYSX_DEMO_PASS', run=str(run), training=False,
                  source_training_steps=270848, source_training_updates=529,
                  goal_errors_m=[e['final_goal_distance_m'] for e in ready], stopped_and_dwelled=True,
                  physical_controls=report['actual_controls'], frozen_policy_actions=len(actions),
                  real_robot_displacement_m=displacement, lidar_scans=len(scans), lidar_hz=15,
                  beams_per_sensor=2000, fixed_masks=[1006, 1017], causal_history_hz=20,
                  action_control_intervals=sorted(set(intervals)), actor_count=4, falls=0,
                  weights_unchanged=True, optimizer_updates=0, protected_sources_unchanged=True,
                  physical_contact_safety='UNVERIFIED')
    (batch / 'REAL_DEMO_CHECK.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
