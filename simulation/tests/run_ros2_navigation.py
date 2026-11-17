# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observe genuine ROS/Isaac navigation; positions [m], yaw [rad], time [s]."""

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from collections import Counter
from dataclasses import replace
from itertools import combinations
from pathlib import Path

_SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SIM.parents[1]))

from pedstrain_simulation.simulation import load_simulation_config
from pedstrain_simulation.simulation.configuration.paths import (
    ROS2_NAVIGATION_SETTINGS_PATH,
    resolve_simulation_profile,
)
from pedstrain_simulation.simulation.ros2_navigation.runtime import Ros2SimulationSession


def _hashes(paths):
    return {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted({p for p in paths if p is not None})
        if p.is_file()
    }


def _closest(a, b):
    delta = tuple(y - x for x, y in zip(a, b))
    norm = sum(x * x for x in delta)
    t = max(0.0, min(1.0, -sum(x * d for x, d in zip(a, delta)) / norm)) if norm else 0.0
    return math.sqrt(sum((x + t * d) ** 2 for x, d in zip(a, delta)))


class _Audit:
    def __init__(self, session):
        from pxr import Gf, UsdGeom, UsdPhysics

        self.plans = {p.name: p for p in session.plans}
        self.targets, self.capsules, self.previous, self.dwell = {}, {}, {}, {}
        self.previous = {s["name"]: {**s, "schedule": {}} for s in session.actors.states()}
        self.stats, self.arrivals, self.delays = {}, [], Counter()
        self.low_speed_runs = {}
        self.min_gap = self.min_physical_gap = math.inf
        auto_return_heading = session.navigation.to_dict().get("auto_return_heading", "preserve")
        assert auto_return_heading in ("preserve", "follow_route"), "Unknown automatic return heading policy"
        authored_plans = {plan.name: plan for plan in session.input_plans}
        for plan in session.plans:
            authored = authored_plans[plan.name]
            targets = [
                {
                    "position": session.frame.world_to_ros(w.position),
                    "wait": w.wait,
                    "yaw": None if w.yaw is None else math.radians(w.yaw) - session.frame.yaw_rad,
                }
                for w in plan.waypoints
            ]
            if plan.behavior == "walk" and plan.repeat and authored.waypoints[-1].position != authored.position:
                reverse = [dict(point) for point in reversed(targets[:-1])]
                if auto_return_heading == "follow_route":
                    for point in reverse:
                        point["yaw"] = None
                targets += [
                    *reverse,
                    {"position": session.frame.world_to_ros(plan.position), "wait": 0.0, "yaw": None},
                ]
            self.targets[plan.name] = targets
            root = session.stage.GetPrimAtPath(session.actors.root_path + "/" + plan.name)
            body = UsdGeom.Capsule.Get(session.stage, root.GetPath().AppendChild("BodyCollider"))
            assert body and UsdPhysics.CollisionAPI(body.GetPrim()).GetCollisionEnabledAttr().Get()
            matrix = body.ComputeLocalToWorldTransform(0.0)
            scales = Gf.Transform(matrix).GetScale()
            axis = "XYZ".index(str(body.GetAxisAttr().Get()))
            assert axis == 2, "Physical capsule must follow actor-local +Z"
            scale = max(abs(scales[i]) for i in range(3) if i != axis)
            radius = body.GetRadiusAttr().Get() * scale * session.frame.meters_per_unit
            height = body.GetHeightAttr().Get() * abs(scales[axis]) * session.frame.meters_per_unit + 2 * radius
            assert UsdPhysics.RigidBodyAPI(root).GetKinematicEnabledAttr().Get()
            self.capsules[plan.name] = (str(root.GetPath()), radius, height)

    def _accumulate_dwell(self, name, old, schedule, dt, changed):
        tracked = self.dwell.get(name)
        if tracked is None:
            return
        if old.get("phase") == "dwelling":
            tracked["elapsed_s"] += dt
        # Temporary door phases pause the business clock. Only actual route
        # progression or resumed walking/completion ends the original dwell.
        if changed or schedule.get("phase") in ("walking", "complete"):
            self.dwell.pop(name)
            assert tracked["elapsed_s"] >= tracked["wait_s"] - 1e-5, (
                name,
                "dwell ended early",
                tracked["elapsed_s"],
                tracked["wait_s"],
            )

    def sample(self, session, now, dt, schedules):
        from omni.physx import get_physx_interface

        states = {s["name"]: s for s in session.actors.states()}
        assert set(states) == set(self.plans), "Actor population changed"
        for name, state in states.items():
            point, yaw = state["position"], state["yaw"]
            assert len(point) == 3 and all(math.isfinite(v) for v in (*point, yaw, *state["velocity"])), name
            schedule, plan = schedules.get(name, {}), self.plans[name]
            assert not schedule.get("error") and schedule.get("phase") != "failed", (name, schedule)
            prior = self.previous.get(name)
            stats = self.stats.setdefault(name, {"distance_m": 0.0, "max_step_m": 0.0, "physics_samples": 0})
            phase = schedule.get("phase", "initializing")
            times = stats.setdefault("phase_time_s", {})
            times[phase] = times.get(phase, 0.0) + dt
            low_speed = phase == "walking" and math.hypot(*state["velocity"][:2]) < 0.03
            self.low_speed_runs[name] = self.low_speed_runs.get(name, 0.0) + dt if low_speed else 0.0
            stats["max_walking_below_003_mps_s"] = max(
                stats.get("max_walking_below_003_mps_s", 0.0), self.low_speed_runs[name]
            )
            if prior:
                step = math.dist(point, prior["position"])
                assert step <= plan.speed * dt + 2e-5, (name, "speed cap", step, dt)
                measured = [(point[i] - prior["position"][i]) / dt for i in range(3)]
                assert math.dist(state["velocity"], measured) < 2e-5, (name, "velocity differs from travel")
                angle = abs(math.remainder(yaw - prior["yaw"], 2 * math.pi))
                assert angle <= session.navigation.max_turn_rate_radps * dt + 1e-5, (name, "turn-rate cap", angle)
                stats["distance_m"] += step
                stats["max_step_m"] = max(stats["max_step_m"], step)
                old = prior["schedule"]
                changed = any(schedule.get(k) != old.get(k) for k in ("waypoint_index", "completed_cycles"))
                arrival = (
                    changed
                    or old.get("phase") not in ("turning", "dwelling", "complete")
                    and schedule.get("phase") in ("turning", "dwelling", "complete")
                )
                if arrival and self.targets[name] and old.get("waypoint_index") is not None:
                    goal = self.targets[name][old["waypoint_index"]]["position"]
                    distance = _closest(
                        tuple(x - g for x, g in zip(prior["position"], goal)), tuple(x - g for x, g in zip(point, goal))
                    )
                    assert distance <= plan.arrival_tolerance + 1e-4, f"{name}: arrival distance {distance}"
                    self.arrivals.append(
                        {
                            "name": name,
                            "time_s": now,
                            "distance_m": distance,
                            "tolerance_m": plan.arrival_tolerance,
                            "waypoint_index": old["waypoint_index"],
                            "completed_cycles": old.get("completed_cycles", 0),
                        }
                    )
                self._accumulate_dwell(name, old, schedule, dt, changed)
            if schedule.get("phase") == "dwelling" and self.targets[name]:
                target = self.targets[name][schedule["waypoint_index"]]
                self.dwell.setdefault(name, {"elapsed_s": 0.0, "wait_s": target["wait"]})
                assert math.dist(point, target["position"]) <= plan.arrival_tolerance + 1e-4
                if target["yaw"] is not None:
                    assert abs(math.remainder(yaw - target["yaw"], 2 * math.pi)) <= math.radians(5.0) + 1e-5
            state["schedule"] = schedule
            physical = get_physx_interface().get_rigidbody_transformation(self.capsules[name][0])
            if physical.get("ret_val"):
                actual = tuple(session.frame.stage_to_ros(tuple(float(v) for v in physical["position"])))
                q = tuple(float(v) for v in physical["rotation"])
                assert len(q) == 4 and all(math.isfinite(v) for v in q) and abs(sum(v * v for v in q) - 1) < 1e-4
                x, y, z, w = q
                forward = session.frame.stage_vector_to_ros(
                    (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y))
                )
                up = session.frame.stage_vector_to_ros(
                    (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))
                )
                assert up[2] > 0 and math.hypot(up[0], up[1]) <= abs(up[2]) * 1e-4, "Physical capsule tilted"
                heading = math.atan2(forward[1], forward[0])
                current = math.dist(actual, point) <= 2e-4 and abs(math.remainder(heading - yaw, 2 * math.pi)) <= 1e-4
                previous = (
                    prior
                    and math.dist(actual, prior["position"]) <= 2e-4
                    and abs(math.remainder(heading - prior["yaw"], 2 * math.pi)) <= 1e-4
                )
                assert current or previous, f"{name}: physical root pose != current/previous command"
                self.delays["current" if current else "previous_tick"] += 1
                state["physical_root_m"] = actual
                state["physical_rotation_xyzw"], state["physical_yaw_rad"] = q, heading
                stats["physics_samples"] += 1
            else:
                self.delays["not_ready"] += 1
        for (a_name, a), (b_name, b) in combinations(states.items(), 2):
            if max(a["position"][2], b["position"][2]) >= min(
                a["position"][2] + a["height_m"], b["position"][2] + b["height_m"]
            ):
                continue
            for physical in (False, True):
                key = "physical_root_m" if physical else "position"
                if key not in a or key not in b:
                    continue
                end = tuple(a[key][i] - b[key][i] for i in range(2))
                old_a, old_b = self.previous.get(a_name, a), self.previous.get(b_name, b)
                start = tuple(old_a.get(key, a[key])[i] - old_b.get(key, b[key])[i] for i in range(2))
                radii = (
                    self.capsules[a_name][1] + self.capsules[b_name][1] if physical else a["radius_m"] + b["radius_m"]
                )
                gap = _closest(start, end) - radii
                assert gap >= -2e-5, (a_name, b_name, "swept body overlap", physical, gap)
                if physical:
                    self.min_physical_gap = min(self.min_physical_gap, gap)
                else:
                    self.min_gap = min(self.min_gap, gap)
        self.previous = states
        return states


def _capture(session, path, *, overhead=False):
    from omni.kit.viewport.utility import capture_viewport_to_file, create_viewport_window
    from pxr import Gf, Usd, UsdGeom, UsdRender

    was_playing = session.timeline.is_playing()
    session.pause()
    now, before = session.timeline.get_current_time(), session.actors.states()
    window = None
    try:
        from itertools import product as corners

        width, height = session.config.settings.width, session.config.settings.height
        points = []
        bounds = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), ["default", "render", "proxy"], useExtentsHint=False, ignoreVisibility=False
        )
        for state in before:
            x, y, z = state["position"]
            padding = max(0.75, state["radius_m"])
            points.extend(corners((x - padding, x + padding), (y - padding, y + padding), (z, z + state["height_m"])))
            root = session.stage.GetPrimAtPath(getattr(session.actors, "root_path", "") + "/" + state["name"])
            if root:
                box = bounds.ComputeWorldBound(root)
                extent = box.GetRange()
                if not extent.IsEmpty():
                    points.extend(
                        session.frame.stage_to_ros(box.GetMatrix().Transform(Gf.Vec3d(*p)))
                        for p in corners(*zip(extent.GetMin(), extent.GetMax()))
                    )
        center = [(min(p[i] for p in points) + max(p[i] for p in points)) / 2 for i in range(3)]
        toward_eye = Gf.Vec3d(0.0, 0.0, 1.0) if overhead else Gf.Vec3d(0.0, -0.5, math.sqrt(0.75))
        camera_up = Gf.Vec3d(0.0, 1.0, 0.0) if overhead else Gf.Vec3d(0.0, math.sqrt(0.75), 0.5)
        tangent_x, tangent_y = 36.0 / (2 * 24.0), 36.0 * height / width / (2 * 24.0)
        relative = [Gf.Vec3d(*(p[i] - center[i] for i in range(3))) for p in points]
        distance = max(
            4.0,
            max(
                Gf.Dot(p, toward_eye) + 1.15 * max(abs(p[0]) / tangent_x, abs(Gf.Dot(p, camera_up)) / tangent_y)
                for p in relative
            ),
        )
        eye = session.frame.ros_to_stage(tuple(center[i] + distance * toward_eye[i] for i in range(3)))
        look = session.frame.ros_to_stage(center)
        up = Gf.Vec3d(*session.frame.ros_to_stage((0.0, 1.0, 0.0) if overhead else (0.0, 0.0, 1.0))) - Gf.Vec3d(
            *session.frame.ros_to_stage((0.0, 0.0, 0.0))
        )
        with Usd.EditContext(session.stage, session.stage.GetSessionLayer()):
            name = "/Ros2AcceptanceCamera"
            while session.stage.GetPrimAtPath(name):
                name += "_1"
            camera = UsdGeom.Camera.Define(session.stage, name)
            camera.SetResetXformStack(True)
            camera.MakeMatrixXform().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*look), up).GetInverse())
            optical_unit = 0.01 / session.frame.meters_per_unit
            camera.CreateProjectionAttr("perspective")
            camera.CreateFocalLengthAttr(24.0 * optical_unit)
            camera.CreateHorizontalApertureAttr(36.0 * optical_unit)
            camera.CreateVerticalApertureAttr(36.0 * height / width * optical_unit)
            camera.CreateFocusDistanceAttr((Gf.Vec3d(*eye) - Gf.Vec3d(*look)).GetLength())
            camera.CreateClippingRangeAttr(
                Gf.Vec2f(
                    0.01 / session.frame.meters_per_unit, max(100.0, distance * 10) / session.frame.meters_per_unit
                )
            )
            product = UsdRender.Product.Define(session.stage, camera.GetPath().AppendChild("Product"))
            product.CreateResolutionAttr(Gf.Vec2i(width, height))
            product.CreateCameraRel().SetTargets([camera.GetPath()])
            aov = UsdRender.Var.Define(session.stage, camera.GetPath().AppendChild("LdrColor"))
            aov.CreateSourceNameAttr("LdrColor")
            aov.CreateSourceTypeAttr("raw")
            aov.CreateDataTypeAttr("color4f")
            product.CreateOrderedVarsRel().SetTargets([aov.GetPath()])
        window = create_viewport_window(
            "ROS2 acceptance " + camera.GetPath().name,
            width=width,
            height=height,
            camera_path=camera.GetPath(),
            usd_drop_support=False,
        )
        assert window is not None, "Independent capture viewport could not be created"
        viewport = window.viewport_api
        viewport.render_product_path = product.GetPath().pathString
        viewport.camera_path = camera.GetPath()
        viewport.resolution = (width, height)
        for _ in range(2):
            session.app.update()
        assert str(viewport.camera_path) == str(camera.GetPath()), "Capture camera was replaced by scene settings"
        assert str(viewport.render_product_path) == str(product.GetPath()), "Capture render product was replaced"
        session.app.run_coroutine(capture_viewport_to_file(viewport, str(path)).wait_for_result())
        assert str(viewport.camera_path) == str(camera.GetPath()), "Capture camera changed during readback"
        assert product.GetCameraRel().GetTargets() == [camera.GetPath()], "Capture product lost its owned camera"
        assert session.timeline.get_current_time() == now and session.actors.states() == before, (
            "Capture advanced time or changed actor poses"
        )
        usd_view = camera.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
        usd_projection = camera.GetCamera(Usd.TimeCode.Default()).frustum.ComputeProjectionMatrix()
        world_to_clip = usd_view * usd_projection
        projected = [tuple(world_to_clip.Transform(Gf.Vec3d(*session.frame.ros_to_stage(p)))) for p in points]
        assert all(abs(p[0]) < 1 and abs(p[1]) < 1 and abs(p[2]) < 1 for p in projected), (
            "Actor bounds outside capture frustum"
        )
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "time_s": now,
                    "view": "overhead" if overhead else "oblique",
                    "camera": str(camera.GetPath()),
                    "render_product": str(product.GetPath()),
                    "states": before,
                    "meters_per_unit": session.frame.meters_per_unit,
                    "focal_length_mm": 24.0,
                    "horizontal_aperture_mm": 36.0,
                    "framing_center_ros_m": center,
                    "framing_distance_m": distance,
                    "max_actor_bound_ndc_xy": [max(abs(p[i]) for p in projected) for i in (0, 1)],
                    "camera_readback": {
                        "projection": camera.GetProjectionAttr().Get(),
                        "focal_length_attr": camera.GetFocalLengthAttr().Get(),
                        "horizontal_aperture_attr": camera.GetHorizontalApertureAttr().Get(),
                        "vertical_aperture_attr": camera.GetVerticalApertureAttr().Get(),
                        "camera_to_world": [
                            list(row) for row in camera.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                        ],
                        "viewport_view": [list(row) for row in viewport.view],
                        "usd_projection": [list(row) for row in usd_projection],
                        "viewport_projection_ui": [list(row) for row in viewport.projection],
                        "viewport_resolution": list(viewport.resolution),
                        "product_resolution": list(product.GetResolutionAttr().Get()),
                        "product_camera": [str(p) for p in product.GetCameraRel().GetTargets()],
                    },
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        if window is not None:
            window.destroy()
        assert session.timeline.get_current_time() == now and session.actors.states() == before, (
            "Capture cleanup advanced time or changed actor poses"
        )
        if was_playing:
            session.play()


def _overlap_diagnostics(session):
    from omni.physx import get_physx_scene_query_interface
    from pxr import PhysicsSchemaTools, Sdf

    if session.actors is None:
        return {"available": False, "reason": "Actor shapes were not created"}
    result = {
        "time_s": session.timeline.get_current_time(),
        "scope": "Current native shape overlaps only; includes floor contact, not a future sweep or penetration depth",
        "actors": {},
    }
    query = get_physx_scene_query_interface()
    for plan in session.plans:
        root = session.actors.root_path + "/" + plan.name
        body = root + "/BodyCollider"
        hits = []

        def report(hit):
            collision, rigid = str(hit.collision), str(hit.rigid_body)
            if not any(value == root or value.startswith(root + "/") for value in (collision, rigid)):
                hits.append({"collision": collision, "rigid_body": rigid, "proto_index": int(hit.protoIndex)})
            return True

        try:
            encoded = PhysicsSchemaTools.encodeSdfPath(Sdf.Path(body))
            count = query.overlap_shape(int(encoded[0]), int(encoded[1]), report, False)
            result["actors"][plan.name] = {"shape": body, "sdk_hit_count": count, "non_self_hits": hits}
        except Exception:
            result["actors"][plan.name] = {"shape": body, "query_error": traceback.format_exc()}
    return result


def main() -> int:
    """Run one unchanged project through the isolated ROS 2 backend."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, help="Simulation duration [s]; default uses the project.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument(
        "--require-first-arrivals",
        action="store_true",
        help="Require every walking actor to reach an original goal during this run.",
    )
    parser.add_argument(
        "--require-route-goals",
        action="store_true",
        help="Require every original walking waypoint to be reached during this run.",
    )
    args = parser.parse_args()
    config = load_simulation_config(args.project)
    duration = config.settings.duration if args.duration is None else args.duration
    if not math.isfinite(duration) or duration <= 0.0 or args.capture and not config.settings.render_viewport:
        parser.error("Use a positive duration; --capture requires render_viewport=true.")
    settings = replace(
        config.settings,
        duration=duration,
        headless=args.headless or config.settings.headless,
        report_path=None,
        console=False,
        start_paused=False,
    )
    config = replace(config, settings=settings)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = _hashes(
        p
        for p in _SIM.rglob("*")
        if p.suffix in (".py", ".cpp", ".h", ".kit")
        and not {"tests", "output", "__pycache__"}.intersection(p.relative_to(_SIM).parts)
    )
    sources.update(_hashes([Path(__file__).resolve()]))
    inputs = _hashes(
        [
            resolve_simulation_profile(args.project),
            *config._source_paths,
            config.scene.asset_path,
            *config.scene.layers,
            config.navigation.settings_path or ROS2_NAVIGATION_SETTINGS_PATH,
            config.pedestrians.config_path,
        ]
    )
    report = {
        "passed": False,
        "backend": config.navigation.backend,
        "source_sha256": sources,
        "input_sha256": inputs,
        "requested_duration_s": duration,
    }
    session, audit, process = Ros2SimulationSession(config, output=output / "runtime"), None, None
    temporary = Path(session._temporary.name)
    started, playback, previous, captures = time.monotonic(), None, 0.0, set()
    try:
        session.setup()
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()
        forbidden = ("isaacsim.replicator.agent.core", "omni.anim.navigation.core", "omni.anim.behavior.core")
        process = session.transport._process
        audit = _Audit(session)
        layers = (Path(layer.realPath) for layer in session.stage.GetUsedLayers() if layer.realPath)
        inputs.update(_hashes(p for p in layers if not p.is_relative_to(temporary) and not p.is_relative_to(output)))
        inputs.update(
            _hashes([getattr(session, "motion_library", None), *getattr(session, "asset_sources", {}).values()])
        )
        report["effective_plans"] = [p.to_dict() for p in session.plans]
        report["control_hz"] = session.navigation.control_hz
        playback, progress = time.monotonic(), 0.0
        periodic_capture = 5.0
        session.play()
        with (output / "audit.jsonl").open("w") as trace:
            while previous < duration - 1e-8:
                if not session.is_running() or time.monotonic() - playback > max(180.0, duration * 20):
                    raise RuntimeError("Application exited or exceeded the acceptance wall-time budget.")
                session.update()
                status = session.status()
                now, schedules = status["simulation_time_s"], {s["name"]: s for s in status["pedestrian_schedules"]}
                if now <= previous:
                    continue
                report["extension_flags"] = {name: manager.is_extension_enabled(name) for name in forbidden}
                assert not any(report["extension_flags"].values()), "A competing motion runtime was enabled"
                states = audit.sample(session, now, now - previous, schedules)
                trace.write(json.dumps({"time_s": now, "states": states}, allow_nan=False) + "\n")
                phases = {s["phase"] for s in schedules.values()}
                capture = sorted(phases - captures)
                if args.capture and (capture or now >= periodic_capture - 1e-8):
                    _capture(session, output / f"capture_{now:.3f}.png")
                    _capture(session, output / f"capture_{now:.3f}_overhead.png", overhead=True)
                    captures.update(capture)
                    if now >= periodic_capture - 1e-8:
                        periodic_capture += 5.0
                if now >= progress:
                    print("ROS2_ACCEPTANCE_PROGRESS", json.dumps({"time_s": now, "actors": audit.stats}), flush=True)
                    progress = math.floor(now) + 1.0
                previous = now
        movers = [
            p
            for p in session.plans
            if p.behavior != "stand"
            and p.start_time + 2 < duration
            and (
                p.behavior == "wander"
                or any(math.dist(p.position, w.position) > p.arrival_tolerance for w in p.waypoints)
            )
        ]
        assert all(audit.stats[p.name]["distance_m"] > 0.01 for p in movers), "A requested mover never moved"
        if args.require_first_arrivals:
            arrived = {a["name"] for a in audit.arrivals}
            assert all(p.name in arrived for p in movers if p.behavior == "walk"), (
                "A walker never reached its first goal"
            )
        if args.require_route_goals:
            coverage = {}
            for plan in session.plans:
                if plan.behavior != "walk":
                    continue
                reached = {a["waypoint_index"] for a in audit.arrivals if a["name"] == plan.name}
                coverage[plan.name] = sorted(set(range(len(plan.waypoints))) - reached)
            assert not any(coverage.values()), f"Original route goals were not reached: {coverage}"
        assert all(s["physics_samples"] for s in audit.stats.values()), "No actual physical root observations"
        report.update(run_completed=True, final_status=session.status())
    except BaseException:
        report["error"] = traceback.format_exc()
        try:
            report["status_at_failure"] = session.status()
        except Exception:
            report["status_error"] = traceback.format_exc()
        try:
            report["physx_overlap_shape_at_failure"] = _overlap_diagnostics(session)
        except Exception:
            report["physx_overlap_shape_error"] = traceback.format_exc()
    finally:
        if audit:
            report["observation"] = {
                "actors": audit.stats,
                "arrivals": audit.arrivals,
                "physical_tick_alignment": dict(audit.delays),
                "min_swept_logical_gap_m": None if math.isinf(audit.min_gap) else audit.min_gap,
                "min_swept_physical_capsule_gap_m": None
                if math.isinf(audit.min_physical_gap)
                else audit.min_physical_gap,
            }
        try:
            report["cleanup"] = session.close_scene()
            assert not temporary.exists() and (process is None or process.poll() is not None), "Resources remain"
        except BaseException:
            report["cleanup_error"] = traceback.format_exc()
        report["input_sha256"] = inputs
        report["source_unchanged"] = _hashes(map(Path, sources)) == sources
        report["inputs_unchanged"] = _hashes(map(Path, inputs)) == inputs
        report["wall_time_s"] = time.monotonic() - started
        report["playback_wall_time_s"] = None if playback is None else time.monotonic() - playback
        report["passed"] = bool(
            report.get("run_completed")
            and not any(report.get(k) for k in ("error", "cleanup_error"))
            and all(report[k] for k in ("source_unchanged", "inputs_unchanged"))
        )
        (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print("ROS2_ACCEPTANCE_RESULT", json.dumps({"passed": report["passed"], "output": str(output)}), flush=True)
        if session.app:
            session.app.close(exit_code=0 if report["passed"] else 1)
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
