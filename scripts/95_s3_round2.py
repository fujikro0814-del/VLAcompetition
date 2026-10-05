"""段階 3 の 2 周目（案 3）: 方策から台本へ閉じる前に引き継ぐ区間の集めと検査（0126・0138、決まりは 0139）。

    .venv\\Scripts\\python.exe scripts\\95_s3_round2.py smoke --model R1v3 [--n 6]          # 移植の検査（49030〜、学習に使わない）
    .venv\\Scripts\\python.exe scripts\\95_s3_round2.py gen --model R1v3 [--shard 0/2]      # 本番（44000〜44399、R・N に同じ種の並び）
    .venv\\Scripts\\python.exe scripts\\95_s3_round2.py check --model R1v3 --runs <フォルダ…> [--rerun 10]

方策は 1 周目で採った R1v3・N1v3（2 万手）。実行のしかたは 0 周目の判断（naive・6 行、安全フィルタなし。
docs/results/s3_round0_*.json）で、評価（scripts/82_v2_eval.py run）と同じ組み立て。世界は v2 の設定に configs/expert_v3.yaml を
重ねたもの（引き継ぎの後の台本は 1 周目と同じ案 1 の台本）。引き継ぎの決まりは src/recovla/expert/handover_v2.py。
種の並び: 種 s を R と N の両方で 1 回ずつ試す（同じ配置・目標・ランダムな時刻）。分けて回すとき（--shard i/n）は
(s − 44000) mod n = i の種だけを、その分け前（ceil(本数 / n)）に達するまで順に回す。集めた区間から、種の小さい順に本数だけ使う。
"""
import argparse
import importlib.util
import json
import math
import os
import pathlib
import time

import numpy as np

from recovla.common import config

CFG = config.load()
os.environ["HF_HOME"] = str(config.path(CFG["paths"]["models_home"]))
os.environ["HF_HUB_OFFLINE"] = "1"
GEN = config.path(CFG["paths"]["outputs"]) / "gen"
RES = config.ROOT / "docs" / "results"
MODELS = ("R1v3", "N1v3")
SEED_BASE, SEED_SPAN = 44000, 400          # 2 周目の帯（0126 の 3）
QUOTA = 120                                # R・N それぞれ
SMOKE_BASE = 49030                         # 移植の検査（R2 の帯の未使用、学習に使わない）


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, config.ROOT / "scripts" / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def runtime_choice() -> dict:
    """0 周目の判断（実行のしかた・安全フィルタ）。"""
    r0, r1 = _load("r0", "92_s3_round0.py"), _load("r1", "94_s3_round1.py")
    ch = r1.round0_choice()
    mode, s = r0.ARMS[ch["arm"]]
    return {"arm": ch["arm"], "mode": mode, "exec_interval": s, "safety": ch["safety"]}


def runtime_factory(model: str, choice: dict):
    """82_v2_eval.py run の make_runtime と同じ組み立て（制限層あり、診断の切り替えなし）。"""
    from recovla.runtime import cue as C
    from recovla.runtime.motion import Motion
    from recovla.runtime.perception import Params, Perception
    from recovla.runtime.policy import SensorPolicy
    from recovla.runtime.runner import PolicyRuntime, disable_rtc_for
    from recovla.runtime.safety import PerceptionSafetyFilter
    ev = _load("ev82", "82_v2_eval.py")
    cfg = config.load_v2()                     # 評価と同じ設定（実行系の値。expert_v3 は台本の値だけ）
    rt_cfg, act, rtv = cfg["runtime"], cfg["actuation"], cfg["runtime_v2"]
    if choice["mode"] == "rtc":
        raise SystemExit("rtc は 0 周目で選ばれていない")
    cache = {"ckpt": ev.CKPT[model]}

    def make_runtime(io, setup):
        if "pol" not in cache:
            mo = Motion(setup)
            cache["pol"] = SensorPolicy(config.path(cache["ckpt"]), setup, mo.hand_pose)
            disable_rtc_for(cache["pol"])
        pol = cache["pol"]
        pol.setup = setup
        if pol.cue is not None:
            pol.cue = C.TargetCue(C.calibration_from_setup(setup.cameras["overhead"]), pol.cue.thr,
                                  setup.table_z + 0.5 * setup.cube_size, setup.cue_fallback_xy)
        per = Perception(setup, Params.from_config(rtv["perception"]), C.Thresholds.from_dict(config.color_detect(cfg)))
        sf = None
        if choice["safety"]:
            sf = PerceptionSafetyFilter(setup, cfg["safety_filter"], float(rtv["safety_extra_margin_m"] or 0.0))
        return PolicyRuntime(io, setup, pol, perception=per, safety=sf, checks=rtv["checks"],
                             gripper_gate=dict(rtv.get("gripper_gate") or {}),
                             tip_offset=float(cfg["sim"]["fingertip_offset"]), mode=choice["mode"],
                             s=int(choice["exec_interval"]), d_init=int(rt_cfg["delay_steps"]),
                             rtc_horizon=int(rt_cfg["rtc_guidance_horizon"]),
                             motion=Motion(setup, margin=float(act["limiter_margin"]), xcmd_leash_m=rtv.get("xcmd_leash_m"),
                                           cart_margin=rtv.get("cart_margin")))
    return make_runtime, cache["ckpt"]


def _rig():
    from recovla.expert import generate as G
    from recovla.harness.gen_v2 import SensedDrivenRig
    return SensedDrivenRig(G.rig_config("v3"))


def _collect(model: str, seeds_iter, quota: int, run_dir: pathlib.Path, log_name: str = "handover.jsonl") -> dict:
    from recovla.common import code_version
    from recovla.expert.handover_v2 import run_handover_attempt
    choice = runtime_choice()
    make_runtime, ckpt = runtime_factory(model, choice)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "start.json").write_text(json.dumps({
        "model": model, "checkpoint": ckpt, "runtime": choice, "quota": quota, "code_version": code_version.code_version(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    rig = _rig()
    got, n = 0, 0
    t0 = time.perf_counter()
    try:
        with open(run_dir / log_name, "w", encoding="utf-8") as log:
            for seed in seeds_iter:
                if got >= quota:
                    break
                out = run_handover_attempt(rig, make_runtime, seed, run_dir, model)
                n += 1
                got += int(out["saved"])
                log.write(json.dumps(out, ensure_ascii=False, default=float) + "\n")
                log.flush()
                h = out["handover"]
                print(f"[s3h] {model} {seed} {out['color']:5s} trigger {h['trigger']} t {h['t_handover']:5.1f} "
                      f"saved {out['saved']} discard {out['discard']} got {got}/{quota} wall {out['wall_s']:.0f}s", flush=True)
    finally:
        rig.close()
    (run_dir / "run.json").write_text(json.dumps({"model": model, "attempts": n, "saved": got, "quota": quota,
                                                  "wall_s": round(time.perf_counter() - t0, 1),
                                                  "written": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2), encoding="utf-8")
    return {"attempts": n, "saved": got}


def cmd_smoke(a) -> None:
    run = GEN / f"S3H_smoke_{a.model}_{time.strftime('%Y%m%d-%H%M%S')}"
    _collect(a.model, range(SMOKE_BASE, SMOKE_BASE + a.n), a.n, run)
    print(run)


def cmd_gen(a) -> None:
    i, n = (int(v) for v in a.shard.split("/"))
    seeds_ = [s for s in range(SEED_BASE, SEED_BASE + SEED_SPAN) if (s - SEED_BASE) % n == i]
    run = GEN / f"S3H_{a.model}_{time.strftime('%Y%m%d-%H%M%S')}_s{i}of{n}"
    _collect(a.model, seeds_, math.ceil(QUOTA / n), run)


def _rows(runs) -> list:
    out = []
    for r in runs:
        p = config.path(r) / "handover.jsonl"
        out += [dict(json.loads(l), _run=str(r)) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return sorted(out, key=lambda x: x["seed"])


def cmd_check(a) -> None:
    """数え上げ（引き継ぎの理由・捨てた理由・状態）、記録の最初のこまが 10 Hz の境目か、回し直しの一致（--rerun 本）。"""
    import collections
    a.runs = [str(config.path(r).resolve().relative_to(config.ROOT)) for r in a.runs]
    rows = _rows(a.runs)
    saved = [r for r in rows if r["saved"]]
    split = int(CFG["sim"]["record_every"]) * int(CFG["sim"]["stride"])
    res = {"model": a.model, "runs": a.runs, "attempts": len(rows), "saved": len(saved),
           "discards": dict(collections.Counter(r["discard"] for r in rows if not r["saved"])),
           "triggers_all": dict(collections.Counter(str(r["handover"]["trigger"]) for r in rows)),
           "triggers_saved": dict(collections.Counter(r["handover"]["trigger"] for r in saved)),
           "boundary_ok": all(r["record_start_step"] % split == 0 for r in saved),
           "g3_violations_saved": sum(int(r["g3"]["total_violations"]) for r in saved),
           "g3_discards": sum(r["discard"] == "g3" for r in rows)}
    keys = ("tip_to_cube_xy", "hand_z_minus_grasp_z", "lead_xy", "t")
    res["state_at_handover_saved"] = {k: {"median": float(np.median([r["handover"]["state_at_handover"][k] for r in saved])),
                                          "p10": float(np.percentile([r["handover"]["state_at_handover"][k] for r in saved], 10)),
                                          "p90": float(np.percentile([r["handover"]["state_at_handover"][k] for r in saved], 90))}
                                      for k in keys} if saved else None
    res["phase_at_handover_saved"] = dict(collections.Counter(r["handover"]["state_at_handover"]["phase"] for r in saved))
    if a.rerun:
        res["rerun"] = _rerun(a.model, saved[:a.rerun])
    out = RES / f"s3_round2_check_{a.model}{a.tag}.json"
    res["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "rerun"}, ensure_ascii=False, indent=1, default=float))
    if a.rerun:
        print("rerun:", [(x["name"], x["same_handover"], x["arrays_identical"]) for x in res["rerun"]])


TRUTH_KEYS = ("cube_pos", "cube_quat", "fingertip", "x_des", "step_x_des", "step_gripper_cmd", "gripper_closed", "phase", "step")
SENSED_KEYS = ("ee_pos", "joints", "fingers")


def _rerun(model: str, rows: list) -> list:
    """0069 の (b) にあたる検査。v2 の移植は世界を止めずに続けて進める（保存した開始状態から復元しない）ので、保存した区間の
    種を回し直して確かめる。方策の推論（GPU）は回すたびに少し変わるので、前半は記録した行動（行動の口に入った後の値）を同じ
    区切りで再生し、同じ物理の手で引き継ぐ。記録の真値の列（TRUTH_KEYS）が全こまでビット一致すれば通る。センサの値
    （SENSED_KEYS。雑音の乱数の消費が前半のカメラの読み出しの数で変わる）は差を記録するだけ。"""
    from recovla.data import convert as CV
    from recovla.expert.handover_v2 import run_handover_attempt
    make_runtime, _ = runtime_factory(model, runtime_choice())
    rig = _rig()
    out = []
    tmp = GEN / f"S3H_rerun_{model}_{time.strftime('%Y%m%d-%H%M%S')}"
    try:
        for r in rows:
            h = r["handover"]
            o = run_handover_attempt(rig, make_runtime, r["seed"], tmp, model,
                                     replay={"actions": h["actions"], "step": h["step_handover"]})
            row = {"name": r["name"], "same_handover": o["handover"]["step_handover"] == h["step_handover"],
                   "saved_again": o["saved"]}
            if o["saved"]:
                _, d0 = CV.load_raw(pathlib.Path(r["path"]))
                _, d1 = CV.load_raw(pathlib.Path(o["path"]))
                same = {k: bool(np.shape(d0[k]) == np.shape(d1[k]) and np.array_equal(d0[k], d1[k])) for k in TRUTH_KEYS}
                diff = {k: (float(np.max(np.abs(np.asarray(d0[k], float) - np.asarray(d1[k], float))))
                            if np.shape(d0[k]) == np.shape(d1[k]) else None) for k in SENSED_KEYS}
                row.update(n_frames=[int(len(d0["step"])), int(len(d1["step"]))], truth_identical=same,
                           sensed_max_abs_diff=diff, arrays_identical=all(same.values()))
            else:
                row["arrays_identical"] = False
            out.append(row)
            print(f"[rerun] {row}", flush=True)
    finally:
        rig.close()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke")
    s.add_argument("--model", choices=MODELS, required=True)
    s.add_argument("--n", type=int, default=6)
    s = sub.add_parser("gen")
    s.add_argument("--model", choices=MODELS, required=True)
    s.add_argument("--shard", default="0/1")
    s = sub.add_parser("check")
    s.add_argument("--model", choices=MODELS, required=True)
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--rerun", type=int, default=0)
    s.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    {"smoke": cmd_smoke, "gen": cmd_gen, "check": cmd_check}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
