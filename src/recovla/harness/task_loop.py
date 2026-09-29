"""目標書 v2 の複数手順の試行（E7 の 3 個の連続タスク）。世界は物理の周期で必ず進み、実行器（runtime/executor.py）はその中で呼ばれる。

    meta, arrays, runtime_log = run_task_trial(world, suite, make_task_runtime, layout, text, seed)

評価の道具（真値）: 手順ごとの真の成功の時刻（目標が箱の成功の体積で 1 s 静止＝旧版と同じ）、終わりに箱に入っている色、
真値のこま（20 Hz）。G1〜G3 の監査は harness/loop.py と同じ。
"""
import numpy as np

from recovla.common.seeds import COLORS
from recovla.expert.script import PhaseParams
from recovla.harness.loop import TRUTH_KEYS, g1_audit
from recovla.harness.robot_io import SimRobotIO
from recovla.harness.setup import nominal_setup
from recovla.record import episode as E
from recovla.sim import frames
from recovla.sim.rig import quiet


def run_task_trial(world, suite, make_task_runtime, layout, text: str, seed: int, cfg: dict = None,
                   time_limit_s: float = 200.0):
    cfg = cfg or world.cfg
    ev = cfg["eval"]
    rest_speed, rest_hold = float(ev["success"]["rest_speed"]), float(ev["success"]["rest_hold_s"])
    pp = PhaseParams.from_config()
    world.reset(layout)
    setup = suite.start_trial(seed, world.data, nominal_setup(cfg))
    suite.prime(world.data)
    rtv = cfg.get("runtime_v2", {})
    io = SimRobotIO(world, suite, seed, fixed_latency={"perception": float(rtv.get("perception_latency_s", 0.0))})
    ex = make_task_runtime(io, setup)
    ex.start(text, seed)
    dt = world.timestep
    st = {"hold": {}, "truth_t": {}}
    truth_log = []

    def capture():
        world.integrator.x_cmd = ex.prt.motion.x_cmd
        world.controller.gripper_closed = bool(ex.prt.closed)
        color = ex.plan["steps"][ex.j] if (ex.plan and 0 <= ex.j < len(ex.plan["steps"])) else COLORS[0]
        f, _ = E.capture_frame(world, color, 0.0, pp, render=False)
        f["x_des"] = ex.prt.motion.x_cmd
        f["gripper_closed"] = bool(ex.prt.closed)
        f["plan_step"] = ex.j
        truth_log.append(f)

    def on_step(r):
        d = r.data
        for i, c in enumerate(COLORS):                    # 手順によらず、色ごとに真の成功の時刻を記録する
            if c in st["truth_t"]:
                continue
            v = r.cube_vadr[i]
            if frames.in_box(d.xpos[r.cube_ids[i]], r.box) and float(np.linalg.norm(d.qvel[v:v + 3])) < rest_speed:
                st["hold"][c] = st["hold"].get(c, 0.0) + dt
                if st["hold"][c] >= rest_hold - 1e-9:
                    st["truth_t"][c] = float(r.step * dt)
            else:
                st["hold"][c] = 0.0
        if r.step % r.record_every == 0:
            capture()

    stops = 0
    audits = [g1_audit(ex, world, suite)]
    capture()
    with quiet():
        while not ex.finished and world.step * dt < time_limit_s - 1e-9:
            t_before = float(world.data.time)
            ex.tick()
            world.apply_joint_commands(io.take_commands())
            world.physics_step(on_step)
            suite.on_physics_step(world.data)
            if abs(float(world.data.time) - t_before - dt) > 1e-9:
                stops += 1
            if world.step % 2500 == 0:
                audits.append(g1_audit(ex, world, suite))
    audits.append(g1_audit(ex, world, suite))
    final_in_box = {c: bool(frames.in_box(world.data.xpos[world.cube_ids[i]], world.box)) for i, c in enumerate(COLORS)}
    rec = ex.record()
    plan_steps = (rec["plan"] or {}).get("steps") or []
    arrays = {k: np.array([f[k] for f in truth_log]) for k in TRUTH_KEYS if k != "target"}
    arrays["plan_step"] = np.array([f["plan_step"] for f in truth_log], dtype=np.int16)
    ioa = io.audit()
    meta = {
        "seed": int(seed), "text": text, "layout": {"kind": layout.kind, "start": layout.start, "seed": layout.seed,
                                                    "table_colors": list(layout.table_colors)},
        "detected": rec["detected"], "plan": rec["plan"], "steps": rec["steps"], "returns": rec["returns"],
        "stopped": rec["stopped"], "startup": rec["startup"], "truth_success_t": st["truth_t"],
        "final_in_box": final_in_box, "all_planned_in_box": bool(plan_steps) and all(final_in_box[c] for c in plan_steps),
        "all_three_in_box": all(final_in_box.values()), "t_end": float(world.data.time), "timed_out": not ex.finished,
        "audit": {"g1": {"checks": len(audits), "violations": sum(len(x["violations"]) for x in audits),
                         "violating_types": sorted({v for x in audits for v in x["violations"]})},
                  "g2": {"world_stops": stops, "early_use": ioa["early_use"], "n_compute": ioa["n_compute"]},
                  "g3": world.audit_summary()},
        "sensor_trial_values": suite.trial_values,
    }
    return meta, arrays, {"executor": {"judge_rows": rec["judge_rows"]}, "runtime": ex.prt.trace(),
                          "latency": ioa["latency_by_kind"]}
