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
import pathlib
import time

import numpy as np

from recovla.common import config

CFG = config.load()
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
    rt = RuntimeConfig("rtc", int(CFG["runtime"]["exec_interval"]), 1,
                       execution_horizon=int(CFG["runtime"]["rtc_guidance_horizon"]))
    runner = SceneRunner(pol, rt, {"schedule": CFG["runtime"]["rtc_schedule"],
                                   "max_guidance_weight": CFG["runtime"]["rtc_max_guidance_weight"]})
    walls, trials_run = [], 0
    rig = SimRig(render=True)
    try:
        for seed, lay, tgt in trial_list(f"selection:{a.seed}:{a.max_trials}"):
            runner.start_trial(seed)
            T.run_trial(rig, lay, tgt, runner, {"trial": trials_run, "seed": seed, "experiment": "dcal",
                                                "condition": "dcal"}, render=True)
            trials_run += 1
            walls += [e["wall_s"] for e in runner.trace()["inference"] if e["wall_s"] is not None]
            if len(walls) - 1 >= a.n:
                break
    finally:
        rig.close()
    w = np.array(walls[1:])                         # 最初の 1 回（ウォームアップ）を除く
    p95 = float(np.percentile(w, float(rule["percentile"])))
    d = int(math.ceil(p95 / float(rule["action_dt_s"])))
    res = {"n": int(w.size), "trials": trials_run, "wall_mean_s": float(w.mean()), "wall_p95_s": p95,
           "wall_max_s": float(w.max()), "d": d, "stop": d > int(rule["max_d"]), "rule": rule,
           "checkpoint": str(a.checkpoint), "tf32": pol.config.get("tf32"),
           "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    RES_OUT.mkdir(parents=True, exist_ok=True)
    (RES_OUT / "dcal.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=1))


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
    s = sub.add_parser("report")
    s.add_argument("--experiment", required=True)
    s.add_argument("--pair", action="append", help="条件 a:b（同じ種で対にする）")
    s = sub.add_parser("dcal")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--n", type=int, default=int(CFG["runtime"]["delay_rule"]["min_samples"]))
    s.add_argument("--seed", type=int, default=198000, help="Step G の検査の帯 198000〜198999（B_提案書 §9）")
    s.add_argument("--max-trials", type=int, default=30)
    a = ap.parse_args(argv)
    {"run": cmd_run, "report": cmd_report, "dcal": cmd_dcal}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
