"""目標書 v2 の試行の繰り返し（評価の枠）。世界は物理の周期で必ず進み、実行系はその中で呼ばれるだけ（G2）。

    meta, truth, runtime_log = run_policy_trial(world, suite, make_runtime, layout, target, seed, inducer=None)

1 手（2 ms）ごと: 実行系の tick（腕とハンドの指令を io に出す）→ 世界に指令を渡す → 物理の 1 手 → センサの模型が撮る時刻を控える。
評価の道具（真値を使ってよい）: 成否の判定（旧版と同じ: 目標が箱の成功の体積の中で 0.01 m/s 未満が 1.0 s）、失敗注入（行動の上書き）、
真値の記録（こま 20 Hz、旧版の記録と同じ欄。x_des には実行系の x_cmd、gripper_closed には実行系の開閉の指令を入れる）。
実行系には真値を渡さない。記録は実行系の記録（runtime_log）と真値の記録（truth）で分ける（0107 の 2-2）。
"""
import dataclasses

import numpy as np

from recovla.common import config
from recovla.common.seeds import COLORS
from recovla.expert.script import PhaseParams
from recovla.harness.robot_io import SimRobotIO
from recovla.harness.setup import nominal_setup
from recovla.record import episode as E
from recovla.sim import contact, frames
from recovla.sim.rig import quiet

TRUTH_KEYS = ("step", "sim_time", "ee_pos", "ee_quat", "fingertip", "fingers", "x_des", "gripper_closed",
              "cube_pos", "cube_quat", "cube_linvel", "cube_in_box", "target", "phase", "contact_robot",
              "contact_cube_cube", "min_dist", "safety_active")


class HarnessHook:
    """評価の道具（失敗注入）が実行系に差し込む口。実機では外からの乱れにあたる。到達検査はこの中に入らない（境界）。"""

    def __init__(self, fn):
        self._fn = fn

    def __call__(self, *a):
        return self._fn(*a)


def g1_audit(rt, world, suite) -> dict:
    """G1 の到達検査: 実行系から世界の物（world・MjData・MjModel・センサの模型の中身・真値）に届かないこと。"""
    from recovla.expert.script import Truth
    from recovla.harness.audit import reachable_forbidden
    from recovla.harness.sensors import SensorSuite
    from recovla.sim.rig import SimRig
    ids = {id(x) for x in (world, world.data, world.model, world.scratch, world.meter, world.meter.ddata,
                           world.meter.dmodel, suite, suite.scratch)}
    return reachable_forbidden(rt, ids, (SimRig, SensorSuite, Truth), (SimRobotIO, HarnessHook))


def instruction(color: str, cfg: dict) -> str:
    return cfg["convert"]["instruction"].format(color=color)


def run_policy_trial(world, suite, make_runtime, layout, target: str, seed: int, time_limit_s: float = None,
                     inducer=None, cfg: dict = None):
    cfg = cfg or world.cfg
    ev = cfg["eval"]
    time_limit_s = float(ev["time_limit_s"]) if time_limit_s is None else float(time_limit_s)
    rest_speed, rest_hold = float(ev["success"]["rest_speed"]), float(ev["success"]["rest_hold_s"])
    pp = PhaseParams.from_config()
    world.reset(layout)
    setup = suite.start_trial(seed, world.data, nominal_setup(cfg))
    suite.prime(world.data)                                     # カメラは始める前から流れている（場面は止まっている）
    rtv = cfg.get("runtime_v2", {})
    fixed = {"perception": float(rtv.get("perception_latency_s", 0.0))}
    fixed.update(rtv.get("diag_fixed_latency") or {})                  # 診断だけ（82 の --ablate fixedinf・freeze）
    io = SimRobotIO(world, suite, seed, fixed_latency=fixed)
    rt = make_runtime(io, setup)
    task = instruction(target, cfg)
    rt.start(task, seed)
    ti = COLORS.index(target)
    dt = world.timestep
    state = {"hold": 0.0, "success_t": None, "rest": 0.0, "induced": []}
    truth_log = []

    def truth():
        return dataclasses.replace(world.truth(target), x_cmd=rt.motion.x_cmd, gripper_closed=bool(rt.closed))

    if inducer is not None:
        def act_filter(k, a):
            out = inducer.filter(k, a, truth())
            state["induced"].append((k, bool(inducer.active)))
            rt.injecting = bool(inducer.active)                 # 上書きの間は安全フィルタを切る（旧版と同じ）
            return out
        rt.action_filter = HarnessHook(act_filter)

    def capture():
        world.integrator.x_cmd = rt.motion.x_cmd                # 段階の判定（記録だけ）が読む値を実行系の指令に合わせる
        world.controller.gripper_closed = bool(rt.closed)
        f, _ = E.capture_frame(world, target, state["rest"], pp, render=False)
        f["x_des"] = rt.motion.x_cmd
        f["gripper_closed"] = bool(rt.closed)
        la = rt.log_act[-1] if getattr(rt, "log_act", None) else None      # 旧版の記録と同じ欄（集計の互換。実行系の記録から写す）
        f["action"] = np.asarray(la[4], float) if la is not None and not la[3] else np.full(7, np.nan)
        f["chunk_id"] = -1 if la is None or la[2] is None else int(la[2])
        truth_log.append(f)

    def on_step(r):
        d = r.data
        pos = d.xpos[r.cube_ids[ti]]
        v = r.cube_vadr[ti]
        speed = float(np.linalg.norm(d.qvel[v:v + 3]))
        state["rest"] = state["rest"] + dt if speed < pp.rest_speed else 0.0
        if state["success_t"] is None:
            if frames.in_box(pos, r.box) and speed < rest_speed:
                state["hold"] += dt
                t = r.step * dt
                if state["hold"] >= rest_hold - 1e-9 and t <= time_limit_s + 1e-9:
                    state["success_t"] = float(t)
            else:
                state["hold"] = 0.0
        if r.step % r.record_every == 0:
            capture()

    stops = 0
    audits = [g1_audit(rt, world, suite)]                        # 試行の始め・5 s ごと・終わり
    capture()
    with quiet():
        while state["success_t"] is None and world.step * dt < time_limit_s - 1e-9:
            if inducer is not None and world.step > 0 and world.step % 50 == 0:
                inducer.after(rt.k, truth())
            t_before = float(world.data.time)
            rt.tick()
            world.apply_joint_commands(io.take_commands())
            world.physics_step(on_step)
            suite.on_physics_step(world.data)
            if abs(float(world.data.time) - t_before - dt) > 1e-9:
                stops += 1                                      # 世界が 1 手ぶん進まなかった（G2 の監査。0 のはず）
            if world.step % 2500 == 0:
                audits.append(g1_audit(rt, world, suite))
    audits.append(g1_audit(rt, world, suite))
    arrays = {k: np.array([f[k] for f in truth_log]) for k in TRUTH_KEYS}
    arrays["target"] = np.full(len(truth_log), ti, dtype=np.int8)
    arrays["phase"] = arrays["phase"].astype(np.int8)
    cid = np.array([f["chunk_id"] for f in truth_log], dtype=np.int32)
    sw = np.zeros(len(cid), dtype=bool)
    prev = -1
    for i, c in enumerate(cid):                                 # 塊が切り替わった最初のこま（旧版 scene_trial と同じ）
        if c >= 0 and c != prev and prev >= 0:
            sw[i] = True
        if c >= 0:
            prev = c
    arrays.update({"action": np.array([f["action"] for f in truth_log]), "chunk_id": cid, "chunk_switch": sw,
                   "induce_active": np.zeros(len(cid), dtype=bool)})
    ioa = io.audit()
    g3 = world.audit_summary()
    meta = {
        "record_version": 1, "steps": [{"target": target, "instruction": task, "t_start": 0.0, "t_end": float(world.data.time),
                                         "success": state["success_t"] is not None, "t_success": state["success_t"]}],
        "inference": [{"wall_s": e["latency_s"]} for e in rt.log_inf],   # 集計の互換: シミュレーションの時刻の上の推論の時間
        "seed": int(seed), "target": target, "instruction": task, "success": state["success_t"] is not None,
        "t_success": state["success_t"], "time_limit_s": time_limit_s, "t_end": float(world.data.time),
        "layout": {"kind": layout.kind, "start": layout.start, "prefilled": sorted(layout.prefilled)},
        "obstacles": list(contact.COLUMNS),                              # contact_robot の列（旧版の記録と同じ欄）
        "induce": inducer.record() if inducer is not None else {"kind": None, "established": False},
        "audit": {"g1": {"checks": len(audits), "violations": sum(len(x["violations"]) for x in audits),
                         "violating_types": sorted({v for x in audits for v in x["violations"]}),
                         "max_visited": max(x["visited"] for x in audits), "truncated": any(x["truncated"] for x in audits)},
                  "g2": {"world_stops": stops, "physics_steps": int(world.step),
                         "time_consistent": abs(world.step * dt - float(world.data.time)) < 1e-6,
                         "early_use": ioa["early_use"], "n_compute": ioa["n_compute"]},
                  "g3": g3},
        "sensor_trial_values": suite.trial_values,
    }
    return meta, arrays, {"runtime": rt.trace(), "latency": ioa["latency_by_kind"], "compute": io.compute_log}
