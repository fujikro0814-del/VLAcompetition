"""目標書 v2 の実行系で、方策の試行を回す（1 つのサブタスク。失敗注入 P1〜P3 も）。記録は真値と実行系で分ける。

    .venv\\Scripts\\python.exe scripts\\82_v2_eval.py run --experiment V2DEV --condition R2_nat --model R2 --trials natural:59050:1
        [--mode naive|sync|rtc] [--induce P1] [--no-limiter]

試行の並びは 41_results.py と同じ（natural・induced・selection）。出力: outputs/v2eval/<実験>/<条件>/trial_NNNN.{json,npz}
（json に成否・監査・センサの試行ごとの値、npz に真値のこま）と trial_NNNN_runtime.json（実行系の記録: 推論・行動・計算の時間）。
"""
import argparse
import importlib.util
import json
import os
import pathlib
import time

import numpy as np

from recovla.common import config

CFG = config.load_v2()
os.environ["HF_HOME"] = str(config.path(CFG["paths"]["models_home"]))
os.environ["HF_HUB_OFFLINE"] = "1"
OUT = config.path(CFG["paths"]["outputs"]) / "v2eval"
CKPT = {
    "R1": "outputs/train/train_R1_20260926-153959_20260926-153959/checkpoints/030000/pretrained_model",
    "N1": "outputs/train/train_N1_20260926-191928_20260926-191928/checkpoints/030000/pretrained_model",
    "R2": "outputs/train/train_R2_20260927-145256_20260927-145256/checkpoints/010000/pretrained_model",
    "R1plus": "outputs/train/train_R1plus_20260927-160208_20260927-160208/checkpoints/010000/pretrained_model",
}


def trial_list(spec: str) -> list:
    s = importlib.util.spec_from_file_location("r41", config.ROOT / "scripts" / "41_results.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m.trial_list(spec)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


ABLATIONS = ("nocolor", "wrist60", "nolatency", "nocalib", "noholes")


def ablate(cfg: dict, names) -> dict:
    """診断だけに使う（評価では使わない）: センサの模型の要素を 1 つずつ外す。"""
    import copy
    c = copy.deepcopy(cfg)
    s = c["sensor"]
    for n in names:
        if n == "nocolor":
            s["color_noise"].update(read_sigma=0.0, shot_gain=0.0, exposure_gain=[1.0, 1.0], white_balance=[1.0, 1.0])
        elif n == "wrist60":
            s["cameras"]["wrist"]["color"] = {"width": 256, "height": 256, "fovy": 60.0, "out": 256}
        elif n == "nolatency":
            s["latency_ms"] = [0.0, 0.0]
        elif n == "nocalib":
            for k in ("overhead", "wrist"):
                s["calibration"][k] = {"trans_sigma": 0.0, "rot_sigma_deg": 0.0}
            s["calibration"]["table"] = {"z_sigma": 0.0, "tilt_sigma_deg": 0.0}
        elif n == "noholes":
            s["depth_noise"]["hole_area"] = 0.0
        else:
            raise SystemExit(f"--ablate {n}: one of {ABLATIONS}")
    return c


def cmd_run(a) -> None:
    global CFG
    if a.ablate:
        CFG = ablate(CFG, a.ablate.split(","))
    from recovla.eval import induce as I
    from recovla.harness.loop import run_policy_trial
    from recovla.harness.sensors import SensorSuite
    from recovla.harness.world import WorldRig
    from recovla.runtime import cue as C
    from recovla.runtime.motion import Motion
    from recovla.runtime.policy import SensorPolicy
    from recovla.runtime.perception import Params, Perception
    from recovla.runtime.runner import PolicyRuntime, disable_rtc_for, enable_rtc_for
    from recovla.runtime.safety import PerceptionSafetyFilter
    out = OUT / a.experiment / a.condition
    if out.exists() and any(out.glob("trial_*.json")):
        raise SystemExit(f"{out} already has trials")
    out.mkdir(parents=True, exist_ok=True)
    world = WorldRig(render=False, cfg=CFG, gravcomp=False if a.diag_no_gravcomp else None)
    suite = SensorSuite(world.model, CFG)
    rt_cfg, act = CFG["runtime"], CFG["actuation"]
    cache = {}

    def make_runtime(io, setup):
        if "pol" not in cache:
            mo = Motion(setup)
            cache["fk_motion"] = mo
            cache["pol"] = SensorPolicy(config.path(CKPT[a.model]), setup, mo.hand_pose)
            if a.mode == "rtc":
                enable_rtc_for(cache["pol"], int(rt_cfg["rtc_guidance_horizon"]), rt_cfg["rtc_schedule"],
                               float(rt_cfg["rtc_max_guidance_weight"]))
            else:
                disable_rtc_for(cache["pol"])
        pol = cache["pol"]
        pol.setup = setup
        if pol.cue is not None:                         # 試行ごとの較正誤差（信じている値）で作り直す
            pol.cue = C.TargetCue(C.calibration_from_setup(setup.cameras["overhead"]), pol.cue.thr,
                                  setup.table_z + 0.5 * setup.cube_size, setup.cue_fallback_xy)
        rtv = CFG["runtime_v2"]
        per = Perception(setup, Params.from_config(rtv["perception"]), C.Thresholds.from_dict(CFG["planner"]["color_detect"]))
        sf = None
        if not a.no_safety:
            sf = PerceptionSafetyFilter(setup, CFG["safety_filter"], float(rtv["safety_extra_margin_m"] or 0.0))
        return PolicyRuntime(io, setup, pol, perception=per, safety=sf, checks=rtv["checks"],
                             tip_offset=float(CFG["sim"]["fingertip_offset"]), mode=a.mode, s=int(rt_cfg["exec_interval"]), d_init=int(rt_cfg["delay_steps"]),
                             rtc_horizon=int(rt_cfg["rtc_guidance_horizon"]),
                             motion=Motion(setup, limiter_enabled=not a.no_limiter, margin=float(act["limiter_margin"]),
                                           ik_on=a.diag_ik))

    rows = []
    t0 = time.perf_counter()
    for i, (seed, lay, tgt) in enumerate(trial_list(a.trials)):
        ind = I.Inducer(a.induce, seed, lay, tgt, world) if a.induce else None
        w0 = time.perf_counter()
        meta, arrays, rlog = run_policy_trial(world, suite, make_runtime, lay, tgt, seed, inducer=ind, cfg=CFG)
        meta.update({"trial": i, "experiment": a.experiment, "condition": a.condition, "model": {"name": a.model}, "ablate": a.ablate,
                     "runtime": {"mode": a.mode, "exec_interval": int(rt_cfg["exec_interval"]), "delay_steps": "sampled",
                                 "safety_filter": not a.no_safety},
                     "mode": a.mode, "limiter": not a.no_limiter, "safety": not a.no_safety, "wall_s": round(time.perf_counter() - w0, 2)})
        np.savez(out / f"trial_{i:04d}.npz", **arrays)
        (out / f"trial_{i:04d}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=_json_default), encoding="utf-8")
        (out / f"trial_{i:04d}_runtime.json").write_text(json.dumps(rlog, ensure_ascii=False, default=_json_default), encoding="utf-8")
        rows.append({"trial": i, "seed": seed, "target": tgt, "success": meta["success"],
                     "established": meta["induce"].get("established")})
        g = meta["audit"]
        print(f"[v2] {a.condition} {i:3d} seed {seed} {tgt:5s} success {meta['success']} est {meta['induce'].get('established')} "
              f"stops {g['g2']['world_stops']} early {g['g2']['early_use']} g3 {g['g3']['total_violations']} "
              f"wall {meta['wall_s']}", flush=True)
    suite.close()
    (out / "run.json").write_text(json.dumps({"experiment": a.experiment, "condition": a.condition, "model": a.model,
                                               "mode": a.mode, "trials": a.trials, "induce": a.induce, "n": len(rows),
                                               "successes": sum(r["success"] for r in rows),
                                               "wall_s": round(time.perf_counter() - t0, 1),
                                               "written": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False, indent=1),
                                   encoding="utf-8")


def cmd_task(a) -> None:
    """複数手順（E7 の 3 個の連続タスク）。配置は旧版と同じく空の箱・既定の開始姿勢（scene.sample_layout(種, "empty", "home")）。"""
    from recovla.harness.sensors import SensorSuite
    from recovla.harness.task_loop import run_task_trial
    from recovla.harness.world import WorldRig
    from recovla.planner import decompose as D
    from recovla.runtime import cue as C
    from recovla.runtime.executor import TaskRuntime
    from recovla.runtime.judge import JudgeV2
    from recovla.runtime.motion import Motion
    from recovla.runtime.perception import Params, Perception
    from recovla.runtime.policy import SensorPolicy
    from recovla.runtime.runner import PolicyRuntime, disable_rtc_for
    from recovla.runtime.safety import PerceptionSafetyFilter
    from recovla.sim import scene
    out = OUT / a.experiment / a.condition
    if out.exists() and any(out.glob("run_*.json")):
        raise SystemExit(f"{out} already has runs")
    out.mkdir(parents=True, exist_ok=True)
    world = WorldRig(render=False, cfg=CFG)
    suite = SensorSuite(world.model, CFG)
    rt_cfg, act, rtv = CFG["runtime"], CFG["actuation"], CFG["runtime_v2"]
    thr = C.Thresholds.from_dict(CFG["planner"]["color_detect"])
    cache = {}

    def make(io, setup):
        if "pol" not in cache:
            cache["pol"] = SensorPolicy(config.path(CKPT[a.model]), setup, Motion(setup).hand_pose)
            disable_rtc_for(cache["pol"])
        pol = cache["pol"]
        pol.setup = setup
        if pol.cue is not None:
            pol.cue = C.TargetCue(C.calibration_from_setup(setup.cameras["overhead"]), pol.cue.thr,
                                  setup.table_z + 0.5 * setup.cube_size, setup.cue_fallback_xy)
        per = Perception(setup, Params.from_config(rtv["perception"]), thr)
        sf = None if a.no_safety else PerceptionSafetyFilter(setup, CFG["safety_filter"], float(rtv["safety_extra_margin_m"]))
        prt = PolicyRuntime(io, setup, pol, perception=per, safety=sf, checks=rtv["checks"],
                            tip_offset=float(CFG["sim"]["fingertip_offset"]), mode="naive", s=int(rt_cfg["exec_interval"]),
                            d_init=int(rt_cfg["delay_steps"]), motion=Motion(setup, margin=float(act["limiter_margin"])))
        judge = JudgeV2(setup, per, thr, rtv["judge"])
        e = CFG["expert"]
        mp = {"gain": float(e["gain_per_s"]), "xy_max": float(e["speed_ref"]["xy"]), "z_max": float(e["speed_ref"]["z"]),
              "z_tol": float(e["move_tol_m"]), "tol": float(e["phase"]["retreat_tol_m"])}
        return TaskRuntime(io, setup, prt, judge, CFG["planner"], mp, D.decompose, CFG["convert"]["instruction"])

    base, n = (int(x) for x in a.trials.split(":"))
    rows = []
    for i, seed in enumerate(range(base, base + n)):
        lay = scene.sample_layout(seed, "empty", start="home")
        w0 = time.perf_counter()
        meta, arrays, rlog = run_task_trial(world, suite, make, lay, a.text, seed, cfg=CFG)
        meta.update({"run": i, "model": a.model, "wall_s": round(time.perf_counter() - w0, 1)})
        np.savez(out / f"run_{i:04d}.npz", **arrays)
        (out / f"run_{i:04d}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=_json_default), encoding="utf-8")
        (out / f"run_{i:04d}_runtime.json").write_text(json.dumps(rlog, ensure_ascii=False, default=_json_default), encoding="utf-8")
        rows.append({"seed": seed, "all_three": meta["all_three_in_box"]})
        g = meta["audit"]
        print(f"[v2task] {i} seed {seed} plan {(meta['plan'] or {}).get('steps')} final {meta['final_in_box']} "
              f"stopped {bool(meta['stopped'])} g1 {g['g1']['violations']} g2 {g['g2']['world_stops']}/{g['g2']['early_use']} "
              f"g3 {g['g3']['total_violations']} wall {meta['wall_s']}", flush=True)
    suite.close()
    (out / "run.json").write_text(json.dumps({"n": len(rows), "all_three": sum(r["all_three"] for r in rows),
                                               "text": a.text, "model": a.model, "trials": a.trials}, ensure_ascii=False,
                                              indent=1), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--experiment", required=True)
    p.add_argument("--condition", required=True)
    p.add_argument("--model", required=True, choices=sorted(CKPT))
    p.add_argument("--trials", required=True)
    p.add_argument("--mode", default="naive", choices=("naive", "sync", "rtc"))
    p.add_argument("--induce", default=None)
    p.add_argument("--no-limiter", action="store_true")
    p.add_argument("--no-safety", action="store_true", help="安全フィルタを切る（E5 の比べる側）")
    p.add_argument("--diag-ik", default="commanded", choices=("commanded", "measured"), help="診断だけ")
    p.add_argument("--diag-no-gravcomp", action="store_true", help="診断だけ")
    p.add_argument("--ablate", default=None, help="診断だけ: " + ",".join(ABLATIONS))
    p = sub.add_parser("task")
    p.add_argument("--experiment", required=True)
    p.add_argument("--condition", required=True)
    p.add_argument("--model", default="R2", choices=sorted(CKPT))
    p.add_argument("--trials", required=True, help="<種の先頭>:<数>")
    p.add_argument("--text", default="全部片付けて")
    p.add_argument("--no-safety", action="store_true")
    a = ap.parse_args(argv)
    {"run": cmd_run, "task": cmd_task}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
