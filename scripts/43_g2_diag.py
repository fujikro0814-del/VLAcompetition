"""G2 の不合格の原因の切り分け（決裁 0068 の 1）と、R2 の計画のための引き継ぎの試し（0068 の 2）。

    .venv\\Scripts\\python.exe scripts\\43_g2_diag.py hold      # 1 の (3): 手を止める P2（B の生成と同じ形、診断用）R1 で 20 回
    .venv\\Scripts\\python.exe scripts\\43_g2_diag.py pilot     # 2: 誘発の成立（＋0〜3 s）で台本に引き継ぎ、台本が立て直せるか
    .venv\\Scripts\\python.exe scripts\\43_g2_diag.py states    # 1 の (4): 制御が戻った時点の状態を A・B・C の開始状態と並べる

- 1 の (1)・(2) は `41_results.py run --experiment G2diag`（R1_P1・R1_P3・N1_P2）で回す
- 誘発の定義（recovla.eval.induce、6f6d2bc で固定）は変えない。hold はこのスクリプトの中の派生で、条件の名前に diag を付ける。
  評価の条件ではなく、R1 が B の開始状態からなら立ち直れるかの確認に限る（報告の本文の P2 の数字に入れない）
- pilot はデータセットを作らない（成功率と引き継ぎの時点の状態だけを記録）。種は学習データ用の帯（49000〜）
- 「制御が戻った時点」: 誘発が上書きした最後のこまの次のこま（induce_active が最後に真だったこまの次）。
  そのこまが、方策が自分の行動を出す直前の観測
"""
import argparse
import json
import os
import pathlib
import time

import numpy as np

from recovla.common import config

CFG = config.load()
os.environ["HF_HOME"] = str(config.path(CFG["paths"]["models_home"]))
os.environ["HF_HUB_OFFLINE"] = "1"

OUTPUTS = config.path(CFG["paths"]["outputs"])
EVAL_OUT = OUTPUTS / "eval" / "G2diag"
RES_OUT = OUTPUTS / "results" / "G2diag"
R1_30K = config.ROOT / "outputs/train/train_R1_20260926-153959_20260926-153959/checkpoints/030000/pretrained_model"
N1_30K = config.ROOT / "outputs/train/train_N1_20260926-191928_20260926-191928/checkpoints/030000/pretrained_model"
HOLD_SEEDS = (195300, 20)
PILOT_SEEDS = (49000, 10)            # 学習データ用の帯（R2 の引き継ぎの配置 40000〜49999 の末尾を試しに使う）
PILOT_DELAY_S = (0.0, 3.0)           # 成立の後に方策をそのまま走らせる時間（一様、0068 の 2）
PILOT_NATURAL_S = (1.0, 9.0)         # 自然な走行のランダムな引き継ぎの時刻（一様）
PILOT_TIME_LIMIT_S = 60.0
OUTSIDE_TOL_M = 0.001                # 範囲の外の判定の許容
PILOT_RNG_KEY = 4300                # 引き継ぎの時刻の乱数: SeedSequence([種, 4300, 種類の番号])


def _runner(ckpt, mode="rtc", horizon=None):
    """既定は評価の設定（rtc・s=10・d=4・範囲 40・指数）。mode="sync"・"naive"（d=4、RTC なし）は診断用。"""
    from recovla.policy.runner import SceneRunner
    from recovla.policy.schedule import RuntimeConfig
    from recovla.policy.scene_policy import ScenePolicy
    rt = CFG["runtime"]
    pol = ScenePolicy(pathlib.Path(ckpt))
    r = SceneRunner(pol, RuntimeConfig(mode, int(rt["exec_interval"]), None if mode == "sync" else int(rt["delay_steps"]),
                                       execution_horizon=int(horizon or rt["rtc_guidance_horizon"])),
                    {"schedule": rt["rtc_schedule"], "max_guidance_weight": rt["rtc_max_guidance_weight"]})
    return pol, r


def _induced(base, n):
    from recovla.eval import scene_trial as T
    from recovla.sim import scene
    seeds_ = list(range(base, base + n))
    lays = [scene.sample_layout(s) for s in seeds_]
    return list(zip(seeds_, lays, T.choose_targets(seeds_, lays)))


# ------------------------------------------------------------------ 1 の (3)

def _hold_inducer():
    from recovla.eval.induce import Inducer

    class HoldP2(Inducer):
        """診断用: P2 と同じ発動・同じ成立の条件で、落ちてから静止するまで手を止める（B の生成と同じ形）。"""

        def _p2(self, k, a, tr):
            a = super()._p2(k, a, tr)
            if self.stage == "dropped":
                a[:3] = 0.0
            return a

        def record(self):
            r = super().record()
            r["diag"] = "hold_hand_during_fall"
            return r
    return HoldP2


def cmd_hold(a) -> None:
    from recovla.eval import scene_trial as T
    from recovla.eval.closed_loop import write_mp4
    from recovla.sim.rig import SimRig
    HoldP2 = _hold_inducer()
    ckpt = {"R1": R1_30K, "N1": N1_30K}[a.model]
    pol, runner = _runner(ckpt, a.mode)
    cond = f"{a.model}_P2hold_diag" + ("" if a.mode == "rtc" else f"_{a.mode}")
    out = EVAL_OUT / cond
    if out.exists() and any(out.glob("trial_*.json")):
        raise SystemExit(f"{out} already has trials")
    rig = SimRig(render=True)
    try:
        for i, (seed, lay, tgt) in enumerate(_induced(*HOLD_SEEDS)):
            runner.start_trial(seed)
            ind = HoldP2("P2", seed, lay, tgt, rig)
            meta, arr, video = T.run_trial(rig, lay, tgt, runner, {
                "trial": i, "seed": seed, "experiment": "G2diag", "condition": cond,
                "model": {"name": f"{a.model}_30000", "checkpoint": str(ckpt)}, "runtime": runner.runtime_record()},
                inducer=ind)
            meta["input_check"] = pol.input_check
            T.write_trial(out, i, meta, arr)
            if i < a.videos:
                write_mp4(out / f"trial_{i:04d}_raw.mp4", video, 20)
            r = meta["induce"]
            print(f"[hold] {i:2d} seed {seed} {tgt:5s} success {meta['success']} fired {r['fired']} "
                  f"est {r['established']} reason {r['reason']}", flush=True)
    finally:
        rig.close()


# ------------------------------------------------------------------ 2 の試し

class _Handover:
    """方策で走らせ、引き継ぎの時刻が来たら台本（入力の読み取りごと）に切り替える。"""

    def __init__(self, runner, rig, seed, target, inducer, delay_s=None, at_s=None):
        from recovla.common import seeds
        from recovla.expert import generate as G
        from recovla.expert import script as S
        self.runner, self.rig, self.target, self.ind = runner, rig, target, inducer
        self.delay_s, self.at_s = delay_s, at_s
        self.t_handover = at_s
        self.script = None
        self.state = None
        self._params = S.sample_params(seeds.script_rng(seed, target, G.HANDOVER_RETRY_KEY), CFG)
        self._S = S
        self.active = False

    # run_trial から act として呼ばれる
    def __call__(self, k, frame, raw, task):
        if self.script is not None:
            return np.zeros(7)
        return self.runner(k, frame, raw, task)

    # run_trial から inducer として呼ばれる（誘発の上書き・成立の判定は元の Inducer のまま）
    def filter(self, k, a, truth):
        if self.script is not None or self.ind is None:
            self.active = False
            return a
        out = self.ind.filter(k, a, truth)
        self.active = self.ind.active
        return out

    def after(self, k, truth):
        if self.ind is not None and self.script is None:
            self.ind.after(k, truth)
            if self.ind.established and self.t_handover is None:
                self.t_handover = float(self.ind.t_established) + float(self.delay_s)
        if self.script is None and self.t_handover is not None and truth.t >= self.t_handover - 1e-9:
            dt = self.rig.steps_per_read * self.rig.timestep
            self.script = self._S.Expert(self._params, dt, CFG)
            fp = truth.fingertip
            tp = truth.target_pos
            self.state = {"t": float(truth.t), "tip_to_cube_xy": float(np.hypot(*(fp[:2] - tp[:2]))),
                          "tip_z": float(fp[2]), "finger_gap": float(np.sum(truth.fingers)),
                          "gripper_closed": bool(truth.gripper_closed),
                          "holding": bool(self._S.holding(truth, self.script.pp)),
                          "target_z": float(tp[2])}

    def record(self):
        r = self.ind.record() if self.ind is not None else {
            "kind": None, "params": {}, "fired": False, "t_fire": None, "established": False,
            "t_established": None, "t_failure": None, "reason": None}
        r["handover"] = {"delay_s": self.delay_s, "at_s": self.at_s, "t": self.t_handover, "state": self.state}
        return r

    def execute(self, orig, rig, a, on_step):
        if self.script is None:
            return orig(rig, a, on_step)
        from recovla.eval import scene_trial as T
        for _ in range(T.STEPS_PER_ACTION // rig.steps_per_read):
            cmd = self.script.act(rig.truth(self.target))
            rig.pad_read(cmd.vel, cmd.press, on_step)


def cmd_pilot(a) -> None:
    from recovla.eval import induce as I
    from recovla.eval import scene_trial as T
    from recovla.sim.rig import SimRig
    pol, runner = _runner(R1_30K)
    orig_exec = T.execute_action
    rig = SimRig(render=True)
    rows = []
    try:
        for ki, kind in enumerate(("P1", "P2", "P3", "natural")):
            for seed, lay, tgt in _induced(*PILOT_SEEDS):
                rng = np.random.default_rng(np.random.SeedSequence([seed, PILOT_RNG_KEY, ki]))
                runner.start_trial(seed)
                if kind == "natural":
                    h = _Handover(runner, rig, seed, tgt, None, at_s=float(rng.uniform(*PILOT_NATURAL_S)))
                else:
                    h = _Handover(runner, rig, seed, tgt, I.Inducer(kind, seed, lay, tgt, rig),
                                  delay_s=float(rng.uniform(*PILOT_DELAY_S)))
                T.execute_action = lambda rig_, a_, on_step=None, _h=h: _h.execute(orig_exec, rig_, a_, on_step)
                try:
                    meta, arr, _ = T.run_trial(rig, lay, tgt, h, {
                        "trial": len(rows), "seed": seed, "experiment": "G2diag_pilot", "condition": kind},
                        time_limit_s=PILOT_TIME_LIMIT_S, render=True, inducer=h)
                finally:
                    T.execute_action = orig_exec
                r = meta["induce"]
                hs = r["handover"]
                ok = bool(meta["success"])
                t_s = meta["steps"][0]["t_success"]
                row = {"kind": kind, "seed": seed, "target": tgt, "layout_kind": lay.kind, "fired": r["fired"],
                       "established": r["established"], "reason": r["reason"], "handed_over": hs["state"] is not None,
                       "handover_t": hs["t"], "delay_s": hs["delay_s"], "state": hs["state"],
                       "success": ok, "success_after_handover": bool(ok and hs["state"] is not None and t_s is not None
                                                                     and t_s >= hs["state"]["t"]),
                       "recovery_s": (t_s - hs["state"]["t"]) if (ok and hs["state"] is not None and t_s) else None}
                rows.append(row)
                print(f"[pilot] {kind:7s} seed {seed} est {r['established']} handed {row['handed_over']} "
                      f"success {ok} state {hs['state']}", flush=True)
    finally:
        rig.close()
    summ = {}
    for kind in ("P1", "P2", "P3", "natural"):
        rs = [r for r in rows if r["kind"] == kind]
        hd = [r for r in rs if r["handed_over"]]
        summ[kind] = {"n": len(rs), "handed_over": len(hd), "script_success": sum(r["success_after_handover"] for r in hd),
                      "success_before_handover": sum(r["success"] and not r["success_after_handover"] for r in rs),
                      "gripper_closed_at_handover": sum(r["state"]["gripper_closed"] for r in hd),
                      "tip_to_cube_xy_median": float(np.median([r["state"]["tip_to_cube_xy"] for r in hd])) if hd else None,
                      "recovery_s_median": float(np.median([r["recovery_s"] for r in hd if r["recovery_s"] is not None]))
                      if any(r["recovery_s"] is not None for r in hd) else None}
    RES_OUT.mkdir(parents=True, exist_ok=True)
    (RES_OUT / "pilot.json").write_text(json.dumps({
        "note": "R2 の計画のための引き継ぎの試し（データセットは作らない）。R1 の 3 万手、rtc・s=10・d=4・範囲 40・指数",
        "seeds": [PILOT_SEEDS[0], PILOT_SEEDS[0] + PILOT_SEEDS[1] - 1], "delay_s": PILOT_DELAY_S,
        "natural_at_s": PILOT_NATURAL_S, "time_limit_s": PILOT_TIME_LIMIT_S, "summary": summ, "rows": rows,
        "written": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1))


# ------------------------------------------------------------------ 1 の (4)

def _state(fp, cube, fingers, closed):
    return {"tip_to_cube_xy": float(np.hypot(*(fp[:2] - cube[:2]))), "tip_to_cube_3d": float(np.linalg.norm(fp - cube)),
            "finger_gap": float(np.sum(fingers)), "tip_z": float(fp[2]), "gripper_closed": bool(closed)}


def _eval_states(path):
    """試行の記録から、誘発が成立した試行の「制御が戻った時点」の状態。"""
    out = []
    for p in sorted(pathlib.Path(path).glob("trial_*.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        if not m["induce"]["established"]:
            continue
        a = np.load(p.with_suffix(".npz"))
        act = np.flatnonzero(a["induce_active"])
        if not act.size:
            continue
        f = int(act[-1]) + 1
        ti = int(a["target"][0])
        s = _state(a["fingertip"][f], a["cube_pos"][f, ti], a["fingers"][f], a["gripper_closed"][f])
        f1 = min(f + 20, len(a["sim_time"]) - 1)             # 1 s 後（20 fps）: 方策が戻ってすぐ閉じたか
        s.update(seed=m["seed"], success=bool(m["success"]), t=float(a["sim_time"][f]),
                 gripper_closed_after_1s=bool(a["gripper_closed"][f1]),
                 tip_to_cube_xy_after_1s=float(np.hypot(*(a["fingertip"][f1, :2] - a["cube_pos"][f1, ti, :2]))))
        out.append(s)
    return out


def _data_states():
    data = json.loads((OUTPUTS / "f" / "data.json").read_text(encoding="utf-8"))
    run = config.ROOT / data["run"]
    out = {"A": [], "B": [], "C": []}
    for c in data["chosen"]:
        d = np.load(run / c["recovery"] / "data.npz")
        ti = int(d["target"][0])
        out[c["kind"]].append(_state(d["fingertip"][0], d["cube_pos"][0, ti], d["fingers"][0], d["gripper_closed"][0]))
    return out


def cmd_states(a) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = _data_states()
    ev = {"eval P1 (R1)": _eval_states(EVAL_OUT / "R1_P1"),
          "eval P2 (R1, G2)": _eval_states(OUTPUTS / "eval" / "G2" / "R1_P2"),
          "eval P3 (R1)": _eval_states(EVAL_OUT / "R1_P3"),
          "eval P2 (N1)": _eval_states(EVAL_OUT / "N1_P2"),
          "diag P2 hold (R1)": _eval_states(EVAL_OUT / "R1_P2hold_diag"),
          "diag P2 hold (R1, sync)": _eval_states(EVAL_OUT / "R1_P2hold_diag_sync")}
    groups = {"data A": data["A"], "data B": data["B"], "data C": data["C"], **ev}
    pair = {"eval P1 (R1)": "A", "eval P2 (R1, G2)": "B", "eval P3 (R1)": "C", "eval P2 (N1)": "B", "diag P2 hold (R1)": "B",
            "diag P2 hold (R1, sync)": "B"}
    keys = [("tip_to_cube_xy", "hand-cube horizontal distance [m]"), ("finger_gap", "finger opening [m]"),
            ("tip_z", "fingertip height [m]")]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
    names = list(groups)
    for ax, (k, lab) in zip(axs, keys):
        for i, g in enumerate(names):
            v = np.array([s[k] for s in groups[g]])
            if not v.size:
                continue
            x = i + np.random.default_rng(i).uniform(-0.18, 0.18, v.size)
            col = "C0" if g.startswith("data") else ("C7" if g.startswith("diag") else "C3")
            ax.scatter(x, v, s=10, color=col, alpha=0.6)
            ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, "k-", lw=2)
        ax.set_xticks(range(len(names)), [n.replace(" (", "\n(") for n in names], rotation=45, ha="right", fontsize=8)
        ax.set_title(lab)
    fig.suptitle("state when control returns to the policy (eval) vs first recorded frame (recovery data)")
    fig.tight_layout()
    RES_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(RES_OUT / "handover_states.png", dpi=110)
    plt.close(fig)
    summ = {}
    for g, rows in groups.items():
        if not rows:
            summ[g] = {"n": 0}
            continue
        s = {"n": len(rows), "gripper_closed": sum(r["gripper_closed"] for r in rows)}
        if "gripper_closed_after_1s" in rows[0]:
            s["gripper_closed_after_1s"] = sum(r["gripper_closed_after_1s"] for r in rows)
            s["tip_to_cube_xy_after_1s_median"] = float(np.median([r["tip_to_cube_xy_after_1s"] for r in rows]))
        for k, _ in keys + [("tip_to_cube_3d", "")]:
            v = np.array([r[k] for r in rows])
            s[k] = {"median": float(np.median(v)), "p5": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)),
                    "min": float(v.min()), "max": float(v.max())}
        if g in pair:                                    # 対応するデータの種類の最小〜最大の外の割合
            ref = data[pair[g]]
            s["outside_data_minmax"] = {}
            for k, _ in keys:                            # 1 mm の許容（指の開き 0.080 どうしの数値の揺れを外に数えない）
                lo, hi = min(r[k] for r in ref), max(r[k] for r in ref)
                s["outside_data_minmax"][k] = sum(not (lo - OUTSIDE_TOL_M <= r[k] <= hi + OUTSIDE_TOL_M) for r in rows)
            s["success"] = sum(r.get("success", False) for r in rows)
        summ[g] = s
    (RES_OUT / "handover_states.json").write_text(json.dumps({
        "definition": "評価: 誘発が上書きした最後のこまの次のこま（成立した試行）。データ: 復帰 90 本の保存を始めた最初のこま",
        "pair": pair, "summary": summ, "rows": groups, "written": time.strftime("%Y-%m-%d %H:%M:%S")},
        ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1))


# ------------------------------------------------------------------ 0070 の 5 (b)

def cmd_fdcheck(a) -> None:
    """G2 で成功した試行（R1・N1 の自然）に失敗の検出 (i)〜(iv) を当て、成功の前に誤って発火した件数（0070 の 5 (b)）。"""
    import collections
    from recovla.eval import failure_detect as FD
    res = {}
    for cond in ("R1_nat", "N1_nat"):
        rows = []
        for p in sorted((OUTPUTS / "eval" / "G2" / cond).glob("trial_*.json")):
            m = json.loads(p.read_text(encoding="utf-8"))
            if not m["success"]:
                continue
            arr = np.load(p.with_suffix(".npz"))
            evs = FD.detect_trial(arr, t_end=float(m["steps"][0]["t_success"]))
            rows.append({"seed": m["seed"], "t_success": m["steps"][0]["t_success"],
                         "events": [{"kind": e.kind, "t": e.t, **{k: v for k, v in e.info.items()
                                                                   if not isinstance(v, list)}} for e in evs]})
        fired = [r for r in rows if r["events"]]
        res[cond] = {"successes": len(rows), "trials_with_false_fire": len(fired),
                     "by_kind": dict(collections.Counter(e["kind"] for r in rows for e in r["events"])),
                     "fired_rows": fired}
        print(cond, {k: v for k, v in res[cond].items() if k != "fired_rows"}, flush=True)
    RES_OUT.mkdir(parents=True, exist_ok=True)
    (RES_OUT / "failure_detect_check.json").write_text(json.dumps({
        "note": "0070 の 5 (b): 成功した試行で、成功の時刻より前に発火した検出。閾値は configs の eval.failure_detect",
        "thresholds": CFG["eval"]["failure_detect"], "results": res, "written": time.strftime("%Y-%m-%d %H:%M:%S")},
        ensure_ascii=False, indent=2, default=float), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("hold")
    s.add_argument("--videos", type=int, default=3)
    s.add_argument("--mode", choices=["rtc", "sync", "naive"], default="rtc", help="sync・naive は実行の方式の切り分け用（診断）")
    s.add_argument("--model", choices=["R1", "N1"], default="R1")
    sub.add_parser("pilot")
    sub.add_parser("states")
    sub.add_parser("fdcheck")
    a = ap.parse_args(argv)
    {"hold": cmd_hold, "pilot": cmd_pilot, "states": cmd_states, "fdcheck": cmd_fdcheck}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
