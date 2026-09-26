"""評価の試行を条件ごとに回し、結果の表と図を出す（手順書 Step G の 8・9、docs/interfaces/results.md）。

    .venv\\Scripts\\python.exe scripts\\41_results.py run --experiment G2 --condition R1_nat --checkpoint CKPT
        --mode rtc --s 10 --d 2 --trials natural:194000:33 [--induce P2] [--videos 3]
    .venv\\Scripts\\python.exe scripts\\41_results.py report --experiment G2 [--pair R1_nat:N1_nat]
    .venv\\Scripts\\python.exe scripts\\41_results.py dcal --checkpoint CKPT [--n 60]      # 遅れ d の決定（runtime.delay_rule）

試行の並び（--trials）:
  natural:<種の先頭>:<配置の数>   空の箱・既定の開始姿勢の配置それぞれで、机上の 3 色を目標に（E1・G2 の自然 99 回など）
  induced:<種の先頭>:<数>          種ごとに配置（配置の種類・開始姿勢は乱数列のまま）と目標を 1 つ（choose_targets）。誘発と一緒に使う
  selection:<種の先頭>:<数>        空の箱・既定の開始姿勢、種ごとに目標を 1 つ（保存点の選択の 30 回など）
出力: outputs/eval/<実験>/<条件>/trial_NNNN.{json,npz}（と先頭の mp4）、集計は outputs/results/<実験>/。
学習と同時に回さない（推論の時間と d の測定が学習で遅く出るため。決裁の先回りの回答、0055）。
"""
import argparse
import json
import math
import os
import pathlib
import time

import numpy as np

from recovla.common import config

CFG = config.load()
# lerobot・huggingface_hub を import する前に決める（ネットワークなし。02_g0_check.py と同じ）
os.environ["HF_HOME"] = str(config.path(CFG["paths"]["models_home"]))
os.environ["HF_HUB_OFFLINE"] = "1"
EVAL_OUT = config.path(CFG["paths"]["outputs"]) / "eval"
RES_OUT = config.path(CFG["paths"]["outputs"]) / "results"


def trial_list(spec: str) -> list:
    """[(種, 配置, 目標)]。"""
    from recovla.eval import scene_trial as T
    from recovla.sim import scene
    kind, base, n = spec.split(":")
    base, n = int(base), int(n)
    if kind == "natural":
        out = []
        for s in range(base, base + n):
            lay = scene.sample_layout(s, "empty", start="home")
            out += [(s, lay, c) for c in lay.table_colors]
        return out
    seeds_ = list(range(base, base + n))
    if kind == "induced":
        lays = [scene.sample_layout(s) for s in seeds_]
    elif kind == "selection":
        lays = [scene.sample_layout(s, "empty", start="home") for s in seeds_]
    else:
        raise SystemExit(f"--trials {spec!r}: kind must be natural, induced or selection")
    return list(zip(seeds_, lays, T.choose_targets(seeds_, lays)))


def _set_num_steps(pol, n) -> None:
    """流れの積分の刻み数（SmolVLA の num_steps、既定 10）を推論で使う値にする（0072 の 1。学習は 10 のまま）。"""
    if n is None:
        return
    from recovla.policy.runner import _base_policy
    _base_policy(pol.policy).config.num_steps = int(n)


def cmd_run(a) -> None:
    from recovla.eval import induce as I
    from recovla.eval import scene_trial as T
    from recovla.eval.closed_loop import write_mp4
    from recovla.policy.runner import RuntimeConfig, SceneRunner
    from recovla.policy.scene_policy import ScenePolicy
    from recovla.sim.rig import SimRig
    out = EVAL_OUT / a.experiment / a.condition
    if out.exists() and any(out.glob("trial_*.json")):
        raise SystemExit(f"{out} already has trials (use a new --condition)")
    pol = ScenePolicy(pathlib.Path(a.checkpoint))
    _set_num_steps(pol, a.num_steps)
    rt = RuntimeConfig(a.mode, a.s, a.d if a.mode != "sync" else None,
                       execution_horizon=a.horizon or int(CFG["runtime"]["rtc_guidance_horizon"]))
    runner = SceneRunner(pol, rt, {"schedule": a.schedule or CFG["runtime"]["rtc_schedule"],
                                   "max_guidance_weight": CFG["runtime"]["rtc_max_guidance_weight"]})
    trials = trial_list(a.trials)
    rig = SimRig(render=True)
    rows = []
    t0 = time.perf_counter()
    try:
        for i, (seed, lay, tgt) in enumerate(trials):
            runner.start_trial(seed)
            ind = I.Inducer(a.induce, seed, lay, tgt, rig) if a.induce else None
            meta, arr, video = T.run_trial(rig, lay, tgt, runner, {
                "trial": i, "seed": seed, "experiment": a.experiment, "condition": a.condition,
                "model": {"name": a.model or a.condition, "checkpoint": str(a.checkpoint)},
                "runtime": runner.runtime_record()}, inducer=ind)
            meta["input_check"] = pol.input_check
            T.write_trial(out, i, meta, arr)
            if i < a.videos:
                write_mp4(out / f"trial_{i:04d}_raw.mp4", video, 20)
            rows.append({"trial": i, "seed": seed, "target": tgt, "success": meta["success"],
                         "induce": meta["induce"]["kind"], "established": meta["induce"]["established"]})
            print(f"[g] {a.condition} {i:3d} seed {seed} {tgt:5s} success {meta['success']} "
                  f"induce {meta['induce']['kind']} fired {meta['induce']['fired']} est {meta['induce']['established']}",
                  flush=True)
    finally:
        rig.close()
    (out / "run.json").write_text(json.dumps({
        "experiment": a.experiment, "condition": a.condition, "checkpoint": str(a.checkpoint), "trials_spec": a.trials,
        "induce": a.induce, "runtime": runner.runtime_record(), "n": len(rows),
        "successes": sum(r["success"] for r in rows), "wall_s": round(time.perf_counter() - t0, 1),
        "written": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_report(a) -> None:
    from recovla.eval import report
    base = EVAL_OUT / a.experiment
    dirs = [d for d in sorted(base.iterdir()) if d.is_dir() and any(d.glob("trial_*.json"))]
    rows = report.collect(dirs)
    pairs = [tuple(p.split(":")) for p in (a.pair or [])]
    info = report.write_all(RES_OUT / a.experiment, rows, a.experiment, pairs)
    print(json.dumps(info, ensure_ascii=False, indent=1))


def cmd_dcal(a) -> None:
    """遅れ d（runner.md §6、runtime.delay_rule）: RTC ありの推論を n 回（最初の 1 回を除く）測り、d = ceil(p95 / 0.1 s)。
    観測は選択用の種の配置で、台本の代わりに方策そのものを rtc・s=10 で回して得る（閉ループの試行の推論の時間を使う）。"""
    from recovla.eval import scene_trial as T
    from recovla.policy.runner import RuntimeConfig, SceneRunner
    from recovla.policy.scene_policy import ScenePolicy
    from recovla.sim.rig import SimRig
    rule = CFG["runtime"]["delay_rule"]
    pol = ScenePolicy(pathlib.Path(a.checkpoint))
    _set_num_steps(pol, a.num_steps)
    rt = RuntimeConfig("rtc", int(CFG["runtime"]["exec_interval"]), 1,
                       execution_horizon=int(CFG["runtime"]["rtc_guidance_horizon"]))
    runner = SceneRunner(pol, rt, {"schedule": CFG["runtime"]["rtc_schedule"],
                                   "max_guidance_weight": CFG["runtime"]["rtc_max_guidance_weight"]})
    walls, parts, trials_run = [], [], 0
    rig = SimRig(render=True)
    try:
        for seed, lay, tgt in trial_list(f"selection:{a.seed}:{a.max_trials}"):
            runner.start_trial(seed)
            T.run_trial(rig, lay, tgt, runner, {"trial": trials_run, "seed": seed, "experiment": "dcal",
                                                "condition": "dcal"}, render=True)
            trials_run += 1
            inf = [e for e in runner.trace()["inference"] if e["wall_s"] is not None]
            walls += [e["wall_s"] for e in inf]
            parts += [e["wall_breakdown_s"] for e in inf]
            if len(walls) - 1 >= a.n:
                break
    finally:
        rig.close()
    used_seeds = [a.seed, a.seed + trials_run - 1]
    w = np.array(walls[1:])                         # 最初の 1 回（ウォームアップ）を除く
    p95 = float(np.percentile(w, float(rule["percentile"])))
    d = int(math.ceil(p95 / float(rule["action_dt_s"])))
    res = {"n": int(w.size), "trials": trials_run, "seeds_used": used_seeds, "wall_mean_s": float(w.mean()), "wall_p95_s": p95,
           "wall_max_s": float(w.max()),
           "breakdown_mean_s": {k: float(np.mean([p[k] for p in parts[1:]])) for k in ("preprocess", "policy")},
           "d": d, "stop": d > int(rule["max_d"]), "rule": rule,
           "checkpoint": str(a.checkpoint), "tf32": pol.config.get("tf32"),
           "num_steps": runner.runtime_record()["num_steps"],
           "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    RES_OUT.mkdir(parents=True, exist_ok=True)
    out = RES_OUT / f"dcal{a.tag}.json"
    if a.tag and out.exists():
        raise SystemExit(f"{out} already exists")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=1))


def cmd_induce_script(a) -> None:
    """誘発の下見（Step G 完了条件 4 の、方策の代わりに台本を使った版。GPU なし）。種類ごとに n 回、発動・成立の割合。
    本番の完了条件 4 は、学習の後に方策で同じ数を回す（run --induce）。"""
    import collections
    import numpy as np
    from recovla.common import seeds
    from recovla.eval import induce as I
    from recovla.eval import scene_trial as T
    from recovla.expert import script as S
    from recovla.sim.rig import SimRig

    class ScriptPolicy:
        def __init__(self, rig, target, seed):
            self.rig, self.target = rig, target
            self.ex = S.Expert(S.sample_params(seeds.script_rng(seed, target, 0)), T.ACTION_DT)

        def __call__(self, k, frame, raw, task):
            cmd = self.ex.act(self.rig.truth(self.target))
            out = np.zeros(7)
            out[:3] = cmd.vel * T.ACTION_DT
            out[6] = 1.0 if bool(self.rig.controller.gripper_closed) != bool(cmd.press) else -1.0
            return out

    rig = SimRig(render=False)
    res = {}
    try:
        for kind in I.KINDS:
            rows = []
            for seed, lay, tgt in trial_list(f"induced:{a.seed}:{a.n}"):
                ind = I.Inducer(kind, seed, lay, tgt, rig)
                meta, arr, _ = T.run_trial(rig, lay, tgt, ScriptPolicy(rig, tgt, seed),
                                           {"trial": 0, "seed": seed, "experiment": "induce_script", "condition": kind},
                                           render=False, inducer=ind)
                r = meta["induce"]
                land = (r.get("info") or {}).get("landing") or {}
                clear = land.get("clearance_m") or {}
                near = sorted(k for k, v in clear.items() if v < float(CFG["inject"]["landing"]["min_clearance_m"]))
                ti = ("red", "green", "blue").index(tgt)
                rows.append({"seed": seed, "fired": r["fired"], "established": r["established"], "reason": r["reason"],
                             "success": meta["success"], "t_fire": r["t_fire"], "layout_kind": lay.kind,
                             "landing_fail": land.get("landing_fail"), "too_close_to": near,
                             "tilt_deg": land.get("tilt_deg"),
                             "final_target_speed": float(np.linalg.norm(arr["cube_linvel"][-1, ti])),
                             "final_target_z": float(arr["cube_pos"][-1, ti, 2]), "info": r.get("info")})
            n = len(rows)
            res[kind] = {"n": n, "fired": sum(r["fired"] for r in rows), "established": sum(r["established"] for r in rows),
                         "success": sum(r["success"] for r in rows),
                         "reasons": dict(collections.Counter(r["reason"] for r in rows if not r["established"])),
                         "establish_rate": sum(r["established"] for r in rows) / n, "rows": rows}
            print(kind, {k: v for k, v in res[kind].items() if k != "rows"}, flush=True)
    finally:
        rig.close()
    RES_OUT.mkdir(parents=True, exist_ok=True)
    (RES_OUT / "induce_script.json").write_text(json.dumps({
        "note": "台本（真値を読む Expert）を方策の代わりにした誘発の下見。種は Step G の検査の帯", "seed": a.seed,
        "induction_min_rate": CFG["eval"]["induction_min_rate"], "results": res,
        "written": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------------------------------ 判断の決まり（0062・0063。結果を見る前に固めた）

RTC_COMBOS = [(10, "LINEAR"), (10, "EXP"), (40, "LINEAR"), (40, "EXP")]
RTC_SUCCESS_WITHIN = 3          # 最良の組の成功数から 3 回以内（30 回中）
RTC_SEAM_SAME_REL = 0.10        # 継ぎ目の跳びの中央値が最小の 10% 以内なら「同じ」
RTC_DEFAULT = (10, "EXP")       # 同点のときに近い方を採る既定


def rtc_condition(h: int, sched: str) -> str:
    return f"E{h}_{sched}"


def cmd_decide_rtc(a) -> None:
    """RTC の設定の 2×2（0062 の 2・0063 の 2）。outputs/eval/RTC2x2/<条件>/ の試行から決める。"""
    from recovla.eval import report
    base = EVAL_OUT / "RTC2x2"
    rows = {}
    for h, sc in RTC_COMBOS:
        c = rtc_condition(h, sc)
        rows[c] = report.collect([base / c])
    stat = {}
    for (h, sc) in RTC_COMBOS:
        c = rtc_condition(h, sc)
        rs = rows[c]
        est = [r for r in rs if r["induce_established"]]
        seam = [r["seam_jump_mean"] for r in rs if r["seam_jump_mean"] is not None and not math.isnan(r["seam_jump_mean"])]
        react = [r["reaction_time_s"] for r in est if r["reaction_time_s"] is not None and not math.isnan(r["reaction_time_s"])]
        stat[c] = {"horizon": h, "schedule": sc, "n": len(rs), "successes": sum(bool(r["success"]) for r in rs),
                   "established": len(est), "establish_rate": len(est) / len(rs) if rs else None,
                   "seam_jump_median": float(np.median(seam)) if seam else float("nan"),
                   "reaction_time_median_s": float(np.median(react)) if react else float("nan"),
                   "seeds": sorted({r["seed"] for r in rs})}
    seeds_sets = {tuple(v["seeds"]) for v in stat.values()}
    best = max(v["successes"] for v in stat.values())
    step1 = [c for c, v in stat.items() if v["successes"] >= best - RTC_SUCCESS_WITHIN]
    smin = min(stat[c]["seam_jump_median"] for c in step1)
    step2 = [c for c in step1 if stat[c]["seam_jump_median"] <= smin * (1 + RTC_SEAM_SAME_REL)]
    if len(step2) > 1:
        rmin = min(stat[c]["reaction_time_median_s"] for c in step2)
        step3 = [c for c in step2 if stat[c]["reaction_time_median_s"] == rmin]
    else:
        step3 = step2

    def closeness(c):
        v = stat[c]
        return (v["horizon"] != RTC_DEFAULT[0], v["schedule"] != RTC_DEFAULT[1])
    chosen = sorted(step3, key=closeness)[0]
    res = {"rule": {"success_within": RTC_SUCCESS_WITHIN, "seam_same_rel": RTC_SEAM_SAME_REL, "default": RTC_DEFAULT,
                    "success_denominator": "P2 が不成立の試行も含めた全試行（0063 の 2）", "source": "0062・0063"},
           "same_seeds_all_combos": len(seeds_sets) == 1, "stats": stat, "step1_success": step1,
           "step2_seam": step2, "step3_reaction": step3, "chosen": chosen,
           "chosen_horizon": stat[chosen]["horizon"], "chosen_schedule": stat[chosen]["schedule"],
           "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    RES_OUT.mkdir(parents=True, exist_ok=True)
    (RES_OUT / "rtc_decision.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "stats"}, ensure_ascii=False, indent=1))


def cmd_decide_ckpt(a) -> None:
    """保存点の選択（手順書 Step H の 2・0062 の 3）。成功数の多い方、同数なら 30000 手。"""
    from recovla.eval import report
    base = EVAL_OUT / f"select_{a.model}"
    out = {}
    for step in (20000, 30000):
        rs = report.collect([base / f"{a.model}_{step}"])
        out[step] = {"n": len(rs), "successes": sum(bool(r["success"]) for r in rs),
                     "seeds": sorted({r["seed"] for r in rs})}
    s20, s30 = out[20000]["successes"], out[30000]["successes"]
    chosen = 20000 if s20 > s30 else 30000
    res = {"model": a.model, "rule": "成功数の多い方、同数なら 30000 手（手順書 Step H の 2・0062・0063）",
           "candidates": out, "same_seeds": out[20000]["seeds"] == out[30000]["seeds"], "chosen_step": chosen,
           "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    RES_OUT.mkdir(parents=True, exist_ok=True)
    (RES_OUT / f"ckpt_decision_{a.model}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("run")
    s.add_argument("--experiment", required=True)
    s.add_argument("--condition", required=True)
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--model", default=None)
    s.add_argument("--mode", choices=["sync", "naive", "rtc"], default="rtc")
    s.add_argument("--s", type=int, default=int(CFG["runtime"]["exec_interval"]))
    s.add_argument("--d", type=int, default=CFG["runtime"]["delay_steps"])
    s.add_argument("--horizon", type=int, default=None)
    s.add_argument("--schedule", default=None)
    s.add_argument("--trials", required=True)
    s.add_argument("--induce", choices=["P1", "P2", "P3"], default=None)
    s.add_argument("--videos", type=int, default=3)
    s.add_argument("--num-steps", type=int, default=None, help="流れの積分の刻み数（既定は保存点の設定＝10）")
    s = sub.add_parser("report")
    s.add_argument("--experiment", required=True)
    s.add_argument("--pair", action="append", help="条件 a:b（同じ種で対にする）")
    s = sub.add_parser("dcal")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--n", type=int, default=100, help="最初の 1 回を除いた回数（0062・0063: 100 回以上）")
    s.add_argument("--seed", type=int, default=198600, help="0062・0063 の割り当て（198600〜、使った範囲を記録）")
    s.add_argument("--max-trials", type=int, default=100)
    s.add_argument("--num-steps", type=int, default=None)
    s.add_argument("--tag", default="", help="結果を dcal<tag>.json に書く（元の dcal.json を上書きしない）")
    sub.add_parser("decide-rtc")
    s = sub.add_parser("decide-ckpt")
    s.add_argument("--model", choices=["R1", "N1"], required=True)
    s = sub.add_parser("induce-script")
    s.add_argument("--seed", type=int, default=198100, help="Step G の検査の帯 198000〜198999")
    s.add_argument("--n", type=int, default=20)
    a = ap.parse_args(argv)
    {"run": cmd_run, "report": cmd_report, "dcal": cmd_dcal, "induce-script": cmd_induce_script,
     "decide-rtc": cmd_decide_rtc, "decide-ckpt": cmd_decide_ckpt}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
