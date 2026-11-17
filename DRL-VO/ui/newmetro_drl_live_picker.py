#!/usr/bin/env python3
"""Persistent 2D goal picker for one owned NewMetro Isaac demo run."""
import argparse
import json
import math
from pathlib import Path

import newmetro_drl_demo_prep as base


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


class LivePicker(base.Picker):
    def __init__(self, run):
        labels = base.load_map()
        self.sequence = 0
        self.last_event = None
        self.shown_initial = False
        self.pending = False
        super().__init__(labels, base.clearance_grid(labels), run,
                         run / 'GOAL_REQUEST.json')
        self.root.title('NewMetro · DRL 交互选点')
        self.root.protocol('WM_DELETE_WINDOW', self.root.withdraw)
        self.root.withdraw()
        self.tk.Button(self.root, text='结束演示', command=self.close_demo).pack(pady=(0, 8))
        for item in self.root.winfo_children():
            if isinstance(item, self.tk.Frame):
                for button in item.winfo_children():
                    if isinstance(button, self.tk.Button) and button.cget('text') == '导出选点请求':
                        button.configure(text='开始导航')
        self.status.set('等待 Isaac 场景启动…')

    def show_picker(self, message):
        self.pending = False
        self.status.set(message)
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def poll(self):
        report = base.owned_live_report(self.run, self.expected_hash)
        if report is None:
            self.status.set('等待当前 Isaac 演示的实时报告和地图…')
        else:
            obs = report.get('robot_observation') or {}
            pose = obs.get('root_position_m')
            if pose and all(math.isfinite(float(v)) for v in pose[:2]):
                self.robot = tuple(map(float, pose[:2]))
            self.people = [tuple(map(float, p[:2])) for p in report.get('positions', [])
                           if len(p) >= 2 and all(math.isfinite(float(v)) for v in p[:2])]
            base.ROBOT_YAW=float(obs.get('yaw_rad',math.pi/2))
            self.draw()
            if not self.shown_initial and report.get('status', '').startswith('DEMO_WAITING'):
                self.shown_initial = True
                self.show_picker('Isaac 已就绪；请选择目标。蓝色机器人和橙色行人持续更新。')
            event_path = self.run / 'GOAL_EVENT.json'
            if event_path.exists():
                try:
                    event = json.loads(event_path.read_text())
                    marker = (event.get('sequence'), event.get('event'))
                    if marker != self.last_event and event.get('schema') == 'newmetro_drl_demo_goal_event/v1':
                        self.last_event = marker
                        if event.get('event') == 'ready':
                            self.selected = None
                            self.route = []
                            self.show_picker('已到达并停稳；可再次选点继续导航。')
                        elif event.get('event') == 'rejected':
                            self.show_picker('选点未被 Isaac 接受：' + str(event.get('reason')))
                        elif event.get('event') == 'failed':
                            self.show_picker('本次导航停止：' + str(event.get('reason')))
                except (OSError, ValueError, KeyError):
                    pass
        self.root.after(500, self.poll)

    def export(self):
        if self.pending:
            return
        if self.selected is None or not self.route:
            self.status.set('请先选一个当前可达的目标')
            return
        if base.owned_live_report(self.run, self.expected_hash) is None:
            self.status.set('Isaac 运行身份或地图未确认，不能发目标')
            return
        self.sequence += 1
        atomic_json(self.output, {
            'schema': 'newmetro_drl_demo_goal_request/v1',
            'sequence': self.sequence,
            'map_labels_sha256': self.expected_hash,
            'robot_xy_at_request_m': self.robot,
            'goal_xy_m': self.selected,
            'static_2d_preview_cells': len(self.route),
        })
        self.pending = True
        self.root.withdraw()

    def close_demo(self):
        atomic_json(self.run / 'VIEW_REQUEST.json', {'close': True})
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    LivePicker(args.run.resolve()).root.mainloop()


if __name__ == '__main__':
    main()
