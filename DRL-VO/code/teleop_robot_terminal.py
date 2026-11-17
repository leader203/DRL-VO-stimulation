"""Gazebo teleop_twist_keyboard style, driving the existing physical preview."""
import argparse
import json
import math
import os
from pathlib import Path
import select
import sys
import termios
import time
import tty
import uuid

ROOT = Path(__file__).resolve().parents[1]
MOVES = {'i':(1,0), 'u':(1,1), 'o':(1,-1), 'j':(0,1), 'l':(0,-1),
         ',':(-1,0), '.':(-1,1), 'm':(-1,-1), 'k':(0,0), ' ':(0,0)}
SPEEDS = {'q':(1.1,1.1), 'z':(.9,.9), 'w':(1.1,1), 'x':(.9,1),
          'e':(1,1.1), 'c':(1,.9)}
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('key', nargs='?', choices=list('wsadxrpk'))
p.add_argument('--speed', type=float, default=.18)
p.add_argument('--turn', type=float, default=.4)
p.add_argument('--run', type=Path, help='Bind to this exact live preview directory')
args = p.parse_args()
if not all(math.isfinite(x) and x >= 0 for x in (args.speed, args.turn)):
    p.error('--speed 和 --turn 必须是有限非负数')
def live_candidate(candidate):
    try:
        candidate = candidate.resolve(strict=True)
        if candidate.parent != (ROOT / 'evidence').resolve() or not candidate.name.startswith('newmetro_robot_preview_'):
            return False
        report = json.loads((candidate / 'REPORT.json').read_text())
        pid = int(report['pid'])
        os.kill(pid, 0)
        argv = (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
        decoded = [item.decode(errors='replace') for item in argv if item]
        return (str(ROOT / 'tools/show_newmetro_robot_preview.py') in decoded
                and '--output' in decoded and str(candidate) in decoded
                and report.get('window_open') and report.get('terminal_teleop_enabled')
                and report.get('gui_update_count', 0) > 0 and report.get('status') != 'FAILED')
    except (OSError, ValueError, KeyError):
        return False

run = None
candidates = ([args.run] if args.run else
              sorted((ROOT / 'evidence').glob('newmetro_robot_preview_*'), reverse=True))
for candidate in candidates:
    if live_candidate(candidate):
        run = candidate.resolve()
        break
if run is None:
    sys.exit('指定的机器人预览尚未就绪或 PID/加载路径不匹配。')

def send(key, v=0., w=0.):
    sequence = uuid.uuid4().hex
    message = dict(sequence=sequence, key=key, v=v, w=w, sent_at=time.time())
    target = run / 'ROBOT_REQUEST.json'
    temporary = run / ('ROBOT_REQUEST_' + sequence + '.tmp')
    temporary.write_text(json.dumps(message)+'\n')
    temporary.replace(target)
    return sequence

if args.key:
    sequence = send('x' if args.key == 'k' else args.key)
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline:
        try:
            ack = json.loads((run / 'ROBOT_ACK.json').read_text())
            if ack.get('sequence') == sequence:
                print(ack['result'])
                sys.exit(0)
        except (OSError, ValueError):
            pass
        time.sleep(.05)
    sys.exit('命令未获确认；请核对预览窗口是否响应。')
if not sys.stdin.isatty():
    sys.exit('交互遥控需要终端。')
print('Gazebo键位：\n   u   i   o\n   j   k   l\n   m   ,   .\n')
print('i前进，,后退，j/l原地左右转；u/o前进转弯，m/.倒车转弯。')
print('k或空格停止；q/z总体加减速，w/x线速度加减，e/c角速度加减（10%）。')
print('方向持续生效，直到换方向或停止；Ctrl+C停止并退出。')
print('连接：'+run.name, flush=True)
speed = min(args.speed, 2.0)
turn = args.turn
direction = (0,0)
old = termios.tcgetattr(sys.stdin)
last_send = 0.
try:
    tty.setcbreak(sys.stdin.fileno())
    while True:
        if select.select([sys.stdin], [], [], .05)[0]:
            key = sys.stdin.read(1).lower()
            if key == '\x03':
                break
            if key in MOVES:
                direction = MOVES[key]
            elif key in SPEEDS:
                a,b = SPEEDS[key]
                speed = min(2.0, speed*a)
                turn *= b
                if not math.isfinite(speed) or not math.isfinite(turn):
                    raise ValueError('速度数值溢出')
            else:
                direction = (0,0)
            v,w = direction[0]*speed, direction[1]*turn
            send('velocity', v, w)
            last_send = time.monotonic()
            print(f'\rv={v:+.3f} m/s  w={w:+.3f} rad/s  speed={speed:.3f} turn={turn:.3f}     ', end='', flush=True)
        if time.monotonic()-last_send >= .1:
            send('velocity', direction[0]*speed, direction[1]*turn)
            last_send = time.monotonic()
except KeyboardInterrupt:
    pass
finally:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
    send('velocity', 0., 0.)
    print('\n已发送停止，退出遥控。')
