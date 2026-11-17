#!/usr/bin/env python3
"""Keep Isaac, a recurrent 2D goal picker, and RViz open for frozen DRL-VO inference."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

DEMO = Path(__file__).resolve().parent
HERE = DEMO/'ui'
sys.path.insert(0,str(HERE))
PROJECT = Path('/home/user/workspace/social_navigation_runtime_20261010')
EXPECTED_MODEL_SHA = 'bef18b93c76f5eef245942b0aa1bede03faaf14efcc7a0fbc38e972a989996ca'


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def prepare_run(run, args):
    for name in ('SOURCE.json', 'scene_config.json', 'profile.json', 'preview_manifest.json'):
        shutil.copy2(DEMO / 'scene' / name, run / name)
    checkpoint = DEMO / 'frozen_policy.pt'
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != EXPECTED_MODEL_SHA:
        raise RuntimeError('Frozen model checkpoint hash mismatch')
    atomic_json(run / 'ROBOT_CONTROL.json', {
        'mode': 'FROZEN_DRLVO_RECURRENT_CLICK_DEMO',
        'model_checkpoint': str(checkpoint), 'model_sha256': digest,
        'local_goal_lookahead_m': args.lookahead_m, 'source_trained_lookahead_m': 0.5,
        'first_subgoal_alignment': {
            'enabled': True, 'heading_tolerance_rad': math.radians(5),
            'max_w_rad_s': .8, 'kp': 2., 'settle_seconds': .15, 'stopped_w_rad_s': .1},
        'max_controls': args.max_controls, 'max_wall_seconds': args.max_wall_seconds,
        'goal_tolerance_m': .3, 'done_contract': 'original accumulated legacy LiDAR proxy / 512-step timeout; unchanged'})


def ros_process(command, logfile):
    # ROS Humble is deliberately outside the Isaac Python process.
    script = 'source /opt/ros/humble/setup.bash && exec "$@"'
    stream = logfile.open('w')
    try:
        return subprocess.Popen(['bash', '-lc', script, 'ros-demo', *command],stdout=stream, stderr=subprocess.STDOUT,env={**os.environ,'ROS_DOMAIN_ID':'209','ROS_LOCALHOST_ONLY':'1'})
    finally:
        stream.close()


def run_demo(run, args):
    import newmetro_drl_demo_prep as picker_map
    labels = picker_map.load_map()
    expected_hash = picker_map.map_hash(labels)
    if args.goal:
        for goal in args.goal:
            picker_map.route_between(picker_map.clearance_grid(labels),
                                     picker_map.DEFAULT_START, tuple(goal))
    lock = (DEMO / 'ISAAC_OWNER.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    subprocess.run(['python3', str(PROJECT / 'tools/check_single_isaac.py')], check=True)
    cmd = ['taskset','--cpu-list','0-15',str(PROJECT / 'python.sh'), str(DEMO / 'code/show_newmetro_robot_preview.py'),
           '--output', str(run), '--sfm-preview', '--manual-robot', '--open-doors', '--isolated-demo']
    if args.headless:
        cmd.append('--demo-headless')
    env = os.environ.copy()
    env.update(OMNI_KIT_ACCEPT_EULA='YES', OMNI_KIT_DISABLE_TELEMETRY='1',PYTHONDONTWRITEBYTECODE='1',NEWMETRO_TORCH_THREADS='1',NEWMETRO_GPU_EXECUTION='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    stream = (run / 'launcher.log').open('w')
    child = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT, env=env)
    stat = Path(f'/proc/{child.pid}/stat').read_text().rsplit(') ', 1)[1].split()
    atomic_json(run / 'OWNER.json', {'pid': child.pid, 'uid': os.getuid(),
        'argv': cmd, 'start_ticks': stat[19], 'launcher_pid': os.getpid(),
        'mode': 'frozen_drlvo_recurrent_click_demo'})
    atomic_json(DEMO / 'LATEST_DEMO.json', {
        'run_dir': str(run), 'pid': child.pid, 'start_ticks': stat[19],
        'uid': os.getuid(), 'launcher_pid': os.getpid(),
        'mode': 'headless_scripted_check' if args.headless else 'interactive_map_picker',
        'training': False, 'source_training_steps': 480000, 'source_training_updates': 937,
        'checkpoint': str(DEMO / 'frozen_policy.pt'), 'checkpoint_sha256': EXPECTED_MODEL_SHA,
        'local_goal_lookahead_m': args.lookahead_m,
        'limits': {'physical_controls': args.max_controls, 'wall_seconds': args.max_wall_seconds},
        'report': str(run / 'REPORT.json'), 'created_at': time.time()})
    print(f'Isaac 已启动；运行目录 {run}', flush=True)
    picker = bridge = rviz = None
    dispatched = 0
    ready = 0
    last_event = None
    last_print = None
    startup = time.monotonic()
    last_live = None
    code = None
    try:
        while True:
            code = child.poll()
            report = picker_map.owned_live_report(run, expected_hash)
            if code is not None:
                break
            if not (run / 'VIEW_REQUEST.json').exists():
                for name, proc in (('2D 选点窗口', picker), ('ROS/RViz 数据桥', bridge), ('RViz', rviz)):
                    if proc is not None and proc.poll() is not None:
                        raise RuntimeError(f'{name} 意外退出，见运行目录内对应日志')
            if report is None:
                if last_live is not None and (run / 'REPORT.json').exists():
                    final = json.loads((run / 'REPORT.json').read_text())
                    if final.get('pid') == child.pid and final.get('window_open') is False:
                        code = child.wait(timeout=25)
                        break
                if last_live is None and time.monotonic() - startup > 240:
                    raise RuntimeError('Isaac did not publish an owned live report within 240 seconds')
                if last_live is not None and time.monotonic() - last_live > 120:
                    raise RuntimeError('Owned Isaac demo report stopped updating')
                time.sleep(1)
                continue
            last_live = time.monotonic()
            if report.get('status', '').startswith('DEMO_WAITING'):
                if args.goal:
                    if dispatched < len(args.goal) and dispatched == ready:
                        dispatched += 1
                        goal = args.goal[dispatched - 1]
                        atomic_json(run / 'GOAL_REQUEST.json', {
                            'schema': 'newmetro_drl_demo_goal_request/v1',
                            'sequence': dispatched, 'map_labels_sha256': expected_hash,
                            'goal_xy_m': goal, 'source': 'scripted_check'})
                        print(f'发送第 {dispatched} 个目标：{goal}', flush=True)
                elif picker is None:
                    picker = subprocess.Popen([sys.executable, str(HERE / 'newmetro_drl_live_picker.py'),
                                               '--run', str(run)],
                                              stdout=(run / 'picker.log').open('w'),
                                              stderr=subprocess.STDOUT)
                    print('2D 选点窗口已打开；到达并停稳后会再次弹出。', flush=True)
                if not args.headless and not args.no_rviz and bridge is None:
                    bridge = ros_process(['/usr/bin/python3', str(HERE / 'newmetro_drl_rviz_bridge.py'),
                                          '--run', str(run)], run / 'rviz_bridge.log')
                    rviz = ros_process(['/opt/ros/humble/bin/rviz2', '-d',
                                        str(HERE / 'newmetro_drl_demo.rviz')], run / 'rviz.log')
                    print('RViz 已启动：静态地图、真实 LiDAR、机器人、行人、目标与路线。', flush=True)
            event_path = run / 'GOAL_EVENT.json'
            if event_path.exists():
                try:
                    event = json.loads(event_path.read_text())
                    marker = (event.get('sequence'), event.get('event'))
                    if marker != last_event:
                        last_event = marker
                        print('目标事件：' + json.dumps(event, ensure_ascii=False), flush=True)
                        if event.get('event') == 'ready':
                            ready = int(event['sequence'])
                            if args.goal and ready == len(args.goal):
                                atomic_json(run / 'VIEW_REQUEST.json', {'close': True})
                        if event.get('event') in ('failed', 'rejected') and args.goal:
                            atomic_json(run / 'VIEW_REQUEST.json', {'close': True})
                except (OSError, ValueError, KeyError):
                    pass
            snapshot = (report.get('status'), report.get('actual_controls', 0) // 30,
                        report.get('demo_completed_goals', 0))
            if snapshot != last_print:
                print(f"{snapshot[0]}  控制步={report.get('actual_controls', 0)}  已到达={snapshot[2]}  距离={report.get('demo_last_goal_distance_m')}", flush=True)
                last_print = snapshot
            time.sleep(.5)
    except KeyboardInterrupt:
        atomic_json(run / 'VIEW_REQUEST.json', {'close': True})
        print('已请求结束演示，等待 Isaac 保存报告…', flush=True)
    except Exception as exc:
        atomic_json(run / 'LAUNCH_FAILURE.json', {
            'status': 'DEMO_LAUNCH_FAILED', 'error': str(exc),
            'demo_success': False, 'training': False, 'recorded_at': time.time()})
        atomic_json(run / 'VIEW_REQUEST.json', {'close': True})
        raise
    finally:
        if child.poll() is None:
            try:
                code = child.wait(timeout=25)
            except subprocess.TimeoutExpired:
                child.terminate()
                code = child.wait(timeout=20)
        for proc in (picker, bridge, rviz):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        stream.close()
    report_path = run / 'REPORT.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    status = report.get('status', 'NO_REPORT')
    demo_success = code == 0 and status == 'DEMO_USER_CLOSED' and (not args.goal or ready == len(args.goal))
    result = {'exit_code': code, 'status': status, 'demo_success': demo_success, 'source_training_steps':480000, 'training':False,
              'local_goal_lookahead_m': args.lookahead_m,
              'terminal_reason': report.get('demo_done_reason'),
              'completed_goals': report.get('demo_completed_goals', 'NOT_OBSERVED'),
              'actual_controls': report.get('actual_controls', 'NOT_OBSERVED'),
              'demo_actions': report.get('demo_actions', 'NOT_OBSERVED'),
              'last_goal_distance_m': report.get('demo_last_goal_distance_m', 'NOT_OBSERVED'),
              'report': str(report_path), 'finished_at': time.time()}
    atomic_json(run / 'EXIT.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if demo_success else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--goal', nargs=2, type=float, action='append', metavar=('X', 'Y'),
                        help='scripted goal; repeat for a multi-goal check instead of opening picker')
    parser.add_argument('--headless', action='store_true', help='hide Isaac for scripted check')
    parser.add_argument('--no-rviz', action='store_true', help='skip RViz viewer')
    parser.add_argument('--lookahead-m', type=float, default=.5, help='demo-only pure-pursuit subgoal lookahead in metres (default 0.5)')
    parser.add_argument('--max-controls', type=int, default=18000)
    parser.add_argument('--max-wall-seconds', type=int, default=3600)
    args = parser.parse_args()
    if not math.isfinite(args.lookahead_m) or args.lookahead_m <= 0:
        parser.error('lookahead must be finite and positive')
    if args.max_controls <= 0 or args.max_wall_seconds <= 0:
        parser.error('limits must be positive')
    if args.headless and not args.goal:
        parser.error('--headless requires at least one scripted --goal')
    run = DEMO / 'runs' / ('click_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run.mkdir(parents=True, exist_ok=False)
    prepare_run(run, args)
    return run_demo(run, args)


if __name__ == '__main__':
    raise SystemExit(main())
