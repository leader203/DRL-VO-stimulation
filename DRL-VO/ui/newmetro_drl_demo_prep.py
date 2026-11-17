#!/usr/bin/env python3
"""NewMetro 2D goal-picker preparation. It never commands the running robot."""

import argparse
import hashlib
import heapq
import json
import math
from pathlib import Path
import time
import sys

import numpy as np
from PIL import Image, ImageTk


HERE = Path(__file__).resolve().parent
MAP_FILE = HERE / "newmetro_open_doors_labels.png"
ORIGIN = (-15.0, -6.2)
RESOLUTION = 0.1
SCALE = 4
CLEARANCE_M = 0.55
DEFAULT_START = (-6.85, -1.65)
COLORS = {0: (39, 48, 58), 205: (172, 179, 187), 254: (248, 247, 242)}


def map_hash(labels):
    return hashlib.sha256(labels.tobytes()).hexdigest()


def load_map():
    labels = np.asarray(Image.open(MAP_FILE).convert("L"))
    if labels.shape != (76, 262) or not set(np.unique(labels)).issubset(COLORS):
        raise ValueError("Unexpected NewMetro label map")
    return labels


def world_to_cell(x, y, shape):
    col = math.floor((x - ORIGIN[0]) / RESOLUTION)
    row = shape[0] - 1 - math.floor((y - ORIGIN[1]) / RESOLUTION)
    if 0 <= row < shape[0] and 0 <= col < shape[1]:
        return row, col
    return None


def cell_to_world(row, col, shape):
    return (ORIGIN[0] + (col + 0.5) * RESOLUTION,
            ORIGIN[1] + (shape[0] - row - 0.5) * RESOLUTION)


def canvas_to_world(px, py, shape):
    return cell_to_world(int(py // SCALE), int(px // SCALE), shape)


def world_to_canvas(x, y, shape):
    return ((x - ORIGIN[0]) / RESOLUTION * SCALE,
            (shape[0] - (y - ORIGIN[1]) / RESOLUTION) * SCALE)


ROBOT_YAW=math.pi/2
_PLANNER=None

def clearance_grid(labels):
    return labels==254

def route_between(allowed,start_xy,goal_xy):
    global _PLANNER
    if _PLANNER is None:
        sys.path.insert(0,str(HERE.parent/'code'))
        from task_sampling import make_navigation
        _PLANNER=make_navigation(HERE.parent)
    _PLANNER.current_yaw=ROBOT_YAW
    _PLANNER.reference_poses=None
    route=_PLANNER.astar_path(start_xy,goal_xy,time_budget_s=6.)
    if route is None:raise ValueError('当前位姿到目标未找到通过车身检查的静态路线')
    return route.tolist()


def owned_live_report(run, expected_hash):
    """Read a report only when its recorded process still owns this exact run."""
    try:
        owner = json.loads((run / "OWNER.json").read_text())
        pid = int(owner["pid"])
        proc = Path("/proc") / str(pid)
        if proc.stat().st_uid != int(owner["uid"]):
            return None
        stat = (proc / "stat").read_text()
        start_ticks = stat.rsplit(") ", 1)[1].split()[19]
        argv = [part.decode(errors="replace") for part in (proc / "cmdline").read_bytes().split(b"\0") if part]
        if (start_ticks != str(owner["start_ticks"]) or
                "--output" not in argv or
                argv[argv.index("--output") + 1] != str(run) or
                not any(arg.endswith("/show_newmetro_robot_preview.py") for arg in argv)):
            return None
        with np.load(run / "robot_navigation_grid.npz", allow_pickle=False) as data:
            if map_hash(data["labels"]) != expected_hash:
                raise ValueError("Live run map differs from the prepared map")
        report_path = run / "REPORT.json"
        if time.time() - report_path.stat().st_mtime > 120:
            return None
        report = json.loads(report_path.read_text())
        if int(report["pid"]) != pid:
            return None
        return report
    except (FileNotFoundError, ProcessLookupError, json.JSONDecodeError, KeyError, ValueError, IndexError):
        return None


def make_background(labels):
    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label, color in COLORS.items():
        rgb[labels == label] = color
    image = Image.fromarray(rgb, "RGB")
    resampling = getattr(Image, "Resampling", Image)
    return image.resize((labels.shape[1] * SCALE, labels.shape[0] * SCALE), resampling.NEAREST)


class Picker:
    def __init__(self, labels, allowed, run, output, *, exit_on_export=False):
        import tkinter as tk

        self.tk = tk
        self.labels, self.allowed, self.run, self.output = labels, allowed, run, output
        self.exit_on_export = exit_on_export
        self.expected_hash = map_hash(labels)
        self.robot = DEFAULT_START
        self.people = []
        self.selected = None
        self.route = []
        self.root = tk.Tk()
        self.root.title("NewMetro · DRL 2D 地图选点准备")
        self.root.configure(bg="#e8edf2")
        self.photo = ImageTk.PhotoImage(make_background(labels))
        tk.Label(self.root, text="白色=已知空地  灰色=未知  深色=障碍  ·  蓝色=机器人  橙色=行人",
                 bg="#e8edf2", anchor="w").pack(fill="x", padx=10, pady=(8, 2))
        self.canvas = tk.Canvas(self.root, width=self.photo.width(), height=self.photo.height(),
                                highlightbackground="#7a8793", highlightthickness=1)
        self.canvas.pack(padx=10, pady=4)
        self.canvas.bind("<Button-1>", self.on_click)
        bar = tk.Frame(self.root, bg="#e8edf2")
        bar.pack(fill="x", padx=10, pady=5)
        tk.Label(bar, text="目标 X", bg="#e8edf2").pack(side="left")
        self.x_var, self.y_var = tk.StringVar(), tk.StringVar()
        tk.Entry(bar, textvariable=self.x_var, width=10).pack(side="left", padx=4)
        tk.Label(bar, text="Y", bg="#e8edf2").pack(side="left")
        tk.Entry(bar, textvariable=self.y_var, width=10).pack(side="left", padx=4)
        tk.Button(bar, text="预览路线", command=self.from_entry).pack(side="left", padx=5)
        tk.Button(bar, text="导出选点请求", command=self.export).pack(side="left", padx=5)
        self.status = tk.StringVar(value="请选择地图上的目标。此窗口不会控制训练中的机器人。")
        tk.Label(self.root, textvariable=self.status, bg="#e8edf2", anchor="w",
                 wraplength=1040, justify="left").pack(fill="x", padx=10, pady=(2, 8))
        self.draw()
        self.root.after(500, self.poll)

    def on_click(self, event):
        if not (0 <= event.x < self.photo.width() and 0 <= event.y < self.photo.height()):
            return
        self.select(canvas_to_world(event.x, event.y, self.labels.shape))

    def from_entry(self):
        try:
            self.select((float(self.x_var.get()), float(self.y_var.get())))
        except ValueError as exc:
            self.status.set(str(exc))

    def select(self, goal):
        if not all(math.isfinite(v) for v in goal):
            self.status.set("目标坐标必须是有限数字")
            return
        try:
            route = route_between(self.allowed, self.robot, goal)
        except ValueError as exc:
            self.selected, self.route = None, []
            self.status.set(f"选点无效：{exc}")
            self.draw()
            return
        self.selected, self.route = goal, route
        self.x_var.set(f"{goal[0]:.3f}")
        self.y_var.set(f"{goal[1]:.3f}")
        self.status.set(f"已选 ({goal[0]:.3f}, {goal[1]:.3f})，静态路线 {len(route)} 个网格点；尚未发送给导航器")
        self.draw()

    def draw(self):
        c = self.canvas
        c.delete("all")
        c.create_image(0, 0, image=self.photo, anchor="nw")
        if self.route:
            points = [v for p in self.route for v in world_to_canvas(*p, self.labels.shape)]
            if len(points) >= 4:
                c.create_line(*points, fill="#00a4dc", width=3)
        for x, y in self.people:
            px, py = world_to_canvas(x, y, self.labels.shape)
            c.create_oval(px - 5, py - 5, px + 5, py + 5, fill="#ec7738", outline="white")
        if self.selected:
            px, py = world_to_canvas(*self.selected, self.labels.shape)
            c.create_oval(px - 8, py - 8, px + 8, py + 8, fill="#f2d53c", outline="#bc1717", width=3)
        px, py = world_to_canvas(*self.robot, self.labels.shape)
        c.create_oval(px - 8, py - 8, px + 8, py + 8, fill="#267ed4", outline="white", width=2)

    def poll(self):
        if self.run is not None:
            try:
                report = owned_live_report(self.run, self.expected_hash)
                if report is None:
                    self.status.set("训练进程身份或地图不匹配；当前不显示为实时数据")
                else:
                    obs = report.get("robot_observation") or {}
                    pose = obs.get("root_position_m")
                    if pose and all(math.isfinite(float(v)) for v in pose[:2]):
                        self.robot = (float(pose[0]), float(pose[1]))
                    self.people = [tuple(map(float, p[:2])) for p in report.get("positions", [])
                                   if len(p) >= 2 and all(math.isfinite(float(v)) for v in p[:2])]
                    if self.selected:
                        try:
                            self.route = route_between(self.allowed, self.robot, self.selected)
                        except ValueError:
                            self.route = []
                    self.status.set(f"只读观察：{report.get('status')} · DRL 步数 {report.get('training_total_steps', '?')} · 更新 {report.get('training_updates', '?')} · 选点尚未发送")
                    self.draw()
            except (FileNotFoundError, json.JSONDecodeError):
                self.status.set("等待运行报告；当前未确认实时数据")
        self.root.after(1000, self.poll)

    def export(self):
        if self.selected is None or not self.route:
            self.status.set("先选择可到达目标，再导出")
            return
        record = {
            "schema": "newmetro_drl_goal_preparation/v1",
            "dispatch": "NOT_SENT",
            "map_labels_sha256": self.expected_hash,
            "map_origin_xy_m": ORIGIN,
            "map_resolution_m": RESOLUTION,
            "static_clearance_m": CLEARANCE_M,
            "robot_xy_at_export_m": self.robot,
            "goal_xy_m": self.selected,
            "route_preview_xy_m": self.route,
            "route_preview_scope": "static grid only; not policy output or physical safety evidence",
            "observed_run": str(self.run) if self.run else None,
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        temp = self.output.with_suffix(self.output.suffix + ".tmp")
        temp.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        temp.replace(self.output)
        self.status.set(f"已导出 {self.output}；目标尚未发送给导航器")
        if self.exit_on_export:
            self.root.after(0, self.root.destroy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="optional current DRL run directory, read only")
    parser.add_argument("--output", type=Path, default=HERE / "goal_request.json")
    parser.add_argument("--check", action="store_true", help="headless map, owner, and route check")
    parser.add_argument("--goal", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--select-and-exit", action="store_true",
                        help="close after exporting one goal; fail if closed without a goal")
    args = parser.parse_args()
    if args.select_and_exit and args.output.exists():
        parser.error("--select-and-exit output already exists; choose a new path")
    labels = load_map()
    allowed = clearance_grid(labels)
    run = args.run.resolve() if args.run else None
    report = owned_live_report(run, map_hash(labels)) if run else None
    pose = (report or {}).get("robot_observation", {}).get("root_position_m")
    start = tuple(map(float, pose[:2])) if pose else DEFAULT_START
    if args.check:
        result = {"map": str(MAP_FILE), "map_labels_sha256": map_hash(labels),
                  "free_cells_with_clearance": int(allowed.sum()), "live_owner": report is not None,
                  "robot_xy_m": start, "goal_dispatch": "NOT_SENT"}
        if args.goal:
            result["route_preview_cells"] = len(route_between(allowed, start, tuple(args.goal)))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    Picker(labels, allowed, run, args.output,
           exit_on_export=args.select_and_exit).root.mainloop()
    if args.select_and_exit and not args.output.is_file():
        raise SystemExit("No goal selected")


if __name__ == "__main__":
    main()
