"""G3（作動）の測定: 今の実行系とエキスパートの腕・グリッパへの指令を物理の周期（500 Hz）で記録し、
Franka Panda の公称の上限（目標書 v2 の G3）と比べる（0106 の設計 7。学習用のシードだけを使う）。

    .venv\\Scripts\\python.exe scripts\\80_g3_measure.py expert            # 台本（通常 10 本、注入 A・B・C 各 3 本）
    .venv\\Scripts\\python.exe scripts\\80_g3_measure.py policy --checkpoint CKPT --trials natural:58100:5 --tag R2_nat
    .venv\\Scripts\\python.exe scripts\\80_g3_measure.py policy --checkpoint CKPT --trials induced:58200:10 --induce P1 --tag R2_P1
    .venv\\Scripts\\python.exe scripts\\80_g3_measure.py summary           # outputs/g3/*.npz → outputs/g3/summary.json

指令とみなすもの（0106 の設計 7）:
  - 腕: 位置サーボへの関節の目標 q_des（data.ctrl、500 Hz）。速度・加速度・躍度は 2 ms の差分で求める
  - 直交座標: q_des の順運動学での手先（hand）の位置。速度・加速度・躍度を同じ差分で求める
  - トルク: 位置サーボが出したトルク（actuator_force）と、その 2 ms の差分から変化の速さ
  - グリッパ: 指 1 本あたりの速さ（関節の速度）と、指 1 本あたりの力（腱の力の半分）
コードの本体（凍結した実行系）は変えない。SimRig を継いで pad_read の物理ステップごとに記録するだけ。
"""
import argparse
import json
import os
import pathlib
import time

import mujoco
import numpy as np

from recovla.common import config

CFG = config.load()
os.environ["HF_HOME"] = str(config.path(CFG["paths"]["models_home"]))
os.environ["HF_HUB_OFFLINE"] = "1"
OUT = config.path(CFG["paths"]["outputs"]) / "g3"

# 目標書 v2 の G3（Franka Panda の公称の上限。libfranka の rate_limiting.h と同じ値）
LIM = {
    "joint_vel": np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]),
    "joint_acc": np.array([15, 7.5, 10, 12.5, 15, 20, 20.0]),
    "joint_jerk": np.array([7500, 3750, 5000, 6250, 7500, 10000, 10000.0]),
    "torque": np.array([87, 87, 87, 87, 12, 12, 12.0]),
    "torque_rate": np.full(7, 1000.0),
    "cart_vel": 1.7, "cart_acc": 13.0, "cart_jerk": 6500.0,
    "finger_speed": 0.05, "finger_force": 70.0,
}


def make_rig_class():
    from recovla.sim.rig import SimRig

    class LogRig(SimRig):
        """pad_read の物理ステップごとに、指令と作動の値を記録する。"""

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            c = self.controller
            self.arm_act = np.array([x[0] for x in c.arm])
            self.log = None

        def start_log(self):
            self.log = {k: [] for k in ("t", "q_des", "q", "dq", "tau", "finger_q", "finger_dq", "grip_ctrl", "grip_force")}

        def pad_read(self, vel=None, press: bool = False, on_step=None) -> None:
            def rec(rig):
                d = rig.data
                if rig.log is not None:
                    L = rig.log
                    L["t"].append(d.time)
                    L["q_des"].append(d.ctrl[rig.arm_act].copy())
                    L["q"].append(d.qpos[rig.arm_qadr].copy())
                    L["dq"].append(d.qvel[rig.arm_vadr].copy())
                    L["tau"].append(d.actuator_force[rig.arm_act].copy())
                    L["finger_q"].append(d.qpos[rig.finger_qadr].copy())
                    L["finger_dq"].append(d.qvel[rig.finger_vadr].copy())
                    L["grip_ctrl"].append(float(d.ctrl[rig.grip_act]))
                    L["grip_force"].append(float(d.actuator_force[rig.grip_act]))
                if on_step is not None:
                    on_step(rig)
            super().pad_read(vel, press, rec)

        def take_log(self, name: str, meta: dict) -> None:
            OUT.mkdir(parents=True, exist_ok=True)
            arr = {k: np.asarray(v) for k, v in self.log.items()}
            np.savez_compressed(OUT / f"{name}.npz", **arr, meta=json.dumps(meta, ensure_ascii=False))
            self.log = None

    return LogRig


# ------------------------------------------------------------------------------ runs
def cmd_expert(a) -> None:
    from recovla.expert import generate as G
    LogRig = make_rig_class()
    rig = LogRig(render=False)
    specs = [G.EpisodeSpec(s, None, "empty", "n") for s in range(58000, 58010)]
    for kind, base in (("A", 58010), ("B", 58013), ("C", 58016)):
        specs += [G.EpisodeSpec(s, None, None, kind) for s in range(base, base + 3)]
    from recovla.sim import scene
    for sp in specs:
        lay = scene.sample_layout(sp.layout_seed, sp.layout_kind)
        sp = G.EpisodeSpec(sp.layout_seed, lay.table_colors[0], sp.layout_kind, sp.kind)
        rig.start_log()
        r = G.run_attempt(rig, sp, 0, None, False)
        rig.take_log(f"expert_{sp.kind}_{sp.layout_seed}_{sp.color}", {"source": "expert", "kind": sp.kind,
                     "seed": sp.layout_seed, "color": sp.color, "success": bool(r["success"])})
        print(f"[g3] expert {sp.kind} {sp.layout_seed} {sp.color} success {r['success']}", flush=True)
    rig.close()


def cmd_policy(a) -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location("r41", config.ROOT / "scripts" / "41_results.py")
    r41 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r41)
    from recovla.eval import induce as I
    from recovla.eval import scene_trial as T
    from recovla.policy.runner import RuntimeConfig, SceneRunner
    from recovla.policy.scene_policy import ScenePolicy
    pol = ScenePolicy(pathlib.Path(a.checkpoint))
    rt = RuntimeConfig(a.mode, int(CFG["runtime"]["exec_interval"]), a.d if a.mode != "sync" else None,
                       execution_horizon=int(CFG["runtime"]["rtc_guidance_horizon"]))
    runner = SceneRunner(pol, rt, {"schedule": CFG["runtime"]["rtc_schedule"],
                                   "max_guidance_weight": CFG["runtime"]["rtc_max_guidance_weight"]})
    LogRig = make_rig_class()
    rig = LogRig(render=True)
    for i, (seed, lay, tgt) in enumerate(r41.trial_list(a.trials)):
        runner.start_trial(seed)
        ind = I.Inducer(a.induce, seed, lay, tgt, rig) if a.induce else None
        rig.start_log()
        meta, _, _ = T.run_trial(rig, lay, tgt, runner, {"trial": i, "seed": seed, "experiment": "G3", "condition": a.tag,
                                                          "model": {"name": a.tag, "checkpoint": str(a.checkpoint)},
                                                          "runtime": runner.runtime_record()}, inducer=ind)
        rig.take_log(f"policy_{a.tag}_{i:03d}_{seed}_{tgt}", {"source": "policy", "tag": a.tag, "seed": seed, "target": tgt,
                                                              "success": bool(meta["success"]), "induce": a.induce})
        print(f"[g3] {a.tag} {i} seed {seed} {tgt} success {meta['success']}", flush=True)
    rig.close()


# ---------------------------------------------------------------------------- summary
def _diffs(x, dt):
    v = np.diff(x, axis=0) / dt
    acc = np.diff(v, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    return v, acc, jerk


def analyse(path: pathlib.Path, fk) -> dict:
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    t, qd = z["t"], z["q_des"]
    # 試行の切れ目（reset で時刻が 0 に戻る。台本の作り直しなど）をまたぐ差分を除く
    seg = np.flatnonzero(np.diff(t) <= 0) + 1
    dt = float(np.median(np.diff(t)[np.diff(t) > 0]))
    out = {"meta": meta, "seconds": float(len(t) * dt)}
    worst = {}
    counts = {}
    for part in np.split(np.arange(len(t)), seg):
        if len(part) < 5:
            continue
        v, acc, jerk = _diffs(qd[part], dt)
        tau = z["tau"][part]
        tr = np.diff(tau, axis=0) / dt
        x = fk(qd[part])
        cv, ca, cj = _diffs(x, dt)
        fv = np.abs(z["finger_dq"][part])
        ff = np.abs(z["grip_force"][part]) * 0.5
        items = {
            "joint_vel": np.abs(v) / LIM["joint_vel"], "joint_acc": np.abs(acc) / LIM["joint_acc"],
            "joint_jerk": np.abs(jerk) / LIM["joint_jerk"], "torque": np.abs(tau) / LIM["torque"],
            "torque_rate": np.abs(tr) / LIM["torque_rate"],
            "cart_vel": np.linalg.norm(cv, axis=1) / LIM["cart_vel"], "cart_acc": np.linalg.norm(ca, axis=1) / LIM["cart_acc"],
            "cart_jerk": np.linalg.norm(cj, axis=1) / LIM["cart_jerk"],
            "finger_speed": fv / LIM["finger_speed"], "finger_force": ff / LIM["finger_force"],
        }
        for k, r in items.items():
            worst[k] = max(worst.get(k, 0.0), float(np.max(r)))
            counts[k] = counts.get(k, 0) + int(np.sum(np.any(r.reshape(len(r), -1) > 1.0, axis=1)))
    out["max_ratio"] = worst                 # 上限に対する比の最大（1 を超えたら違反）
    out["violating_steps"] = counts          # 違反した 2 ms の刻みの数
    return out


def cmd_summary(a) -> None:
    from recovla.sim import scene
    m = scene.build_model("3cube")
    d = mujoco.MjData(m)
    arm_q = np.array([m.joint(f"joint{i}").qposadr[0] for i in range(1, 8)])
    hand = m.body("hand").id

    def fk(Q):
        out = np.empty((len(Q), 3))
        for i, q in enumerate(Q):
            d.qpos[arm_q] = q
            mujoco.mj_kinematics(m, d)
            out[i] = d.xpos[hand]
        return out

    rows = [analyse(p, fk) for p in sorted(OUT.glob("*.npz"))]
    groups = {}
    for r in rows:
        g = r["meta"]["source"] if r["meta"]["source"] == "expert" else r["meta"]["tag"]
        G = groups.setdefault(g, {"n": 0, "seconds": 0.0, "max_ratio": {}, "violating_steps": {}, "episodes_violating": {}})
        G["n"] += 1
        G["seconds"] += r["seconds"]
        for k, v in r["max_ratio"].items():
            G["max_ratio"][k] = round(max(G["max_ratio"].get(k, 0.0), v), 3)
            G["violating_steps"][k] = G["violating_steps"].get(k, 0) + r["violating_steps"][k]
            G["episodes_violating"][k] = G["episodes_violating"].get(k, 0) + int(v > 1.0)
    res = {"limits": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in LIM.items()},
           "note": "max_ratio は上限に対する比の最大。1 を超えると違反。差分は 2 ms（物理の周期）",
           "groups": groups, "episodes": rows, "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    (OUT / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(groups, ensure_ascii=False, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("expert")
    p = sub.add_parser("policy")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--trials", required=True)
    p.add_argument("--induce", default=None)
    p.add_argument("--tag", required=True)
    p.add_argument("--mode", default="naive")
    p.add_argument("--d", type=int, default=4)
    sub.add_parser("summary")
    a = ap.parse_args(argv)
    {"expert": cmd_expert, "policy": cmd_policy, "summary": cmd_summary}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
