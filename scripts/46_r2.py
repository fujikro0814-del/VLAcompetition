"""R2 の引き継ぎ区間の生成と検査（決裁 0069 の 3・0070 の 1。生成は 0070 の 2 の報告への回答の後）。

    .venv\\Scripts\\python.exe scripts\\46_r2.py smoke                      # 実装の検査（種 49010〜、各種類 2 本、別のフォルダ）
    .venv\\Scripts\\python.exe scripts\\46_r2.py gen --kind P2 [--tag TAG]  # 本番: 種類ごとに割り当ての本数に達するまで
    .venv\\Scripts\\python.exe scripts\\46_r2.py check --run outputs\\gen\\R2_...   # 生成の後の検査 (a)(b)(c)

割り当て（0069 の 3）: 乙 P1・P2・P3 各 40（引き継ぎの時点は成立の時点 20・成立の後 0〜3 s 20）、甲 natural 60。
種: P1 40000〜40099、P2 41000〜41099、P3 42000〜42099、natural 43000〜43099（先頭から、割り当てに達するまで）。
引き継ぎの時点の半々は種の偶奇で決める（偶数＝成立の時点、奇数＝成立の後 0〜3 s）。結果を見る前に決まる。
方策は R1 の 3 万手、評価と同じ設定（rtc・s=10・d=4・範囲 40・指数）。評価用の帯の種は使わない。
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
R1_30K = config.ROOT / "outputs/train/train_R1_20260926-153959_20260926-153959/checkpoints/030000/pretrained_model"
SEED_BASE = {"P1": 40000, "P2": 41000, "P3": 42000, "natural": 43000}
SEED_SPAN = 100
QUOTA = {"P1": {"at_est": 20, "delayed": 20}, "P2": {"at_est": 20, "delayed": 20},
         "P3": {"at_est": 20, "delayed": 20}, "natural": {"detect_or_random": 60}}
SMOKE_BASE = 49010
SMOKE_PER_KIND = 2
SCRIPT_SUCCESS_MIN = 0.9          # 検査 (a)
REPLAY_N = 10                     # 検査 (b)
VIDEOS_PER_KIND = 5               # 検査 (c)


def mode_of(kind: str, seed: int) -> str:
    if kind == "natural":
        return "detect_or_random"
    return "at_est" if seed % 2 == 0 else "delayed"


def _runner():
    from recovla.policy.runner import SceneRunner
    from recovla.policy.schedule import RuntimeConfig
    from recovla.policy.scene_policy import ScenePolicy
    rt = CFG["runtime"]
    pol = ScenePolicy(R1_30K)
    return pol, SceneRunner(pol, RuntimeConfig("rtc", int(rt["exec_interval"]), int(rt["delay_steps"]),
                                               execution_horizon=int(rt["rtc_guidance_horizon"])),
                            {"schedule": rt["rtc_schedule"], "max_guidance_weight": rt["rtc_max_guidance_weight"]})


def _generate(kind, seeds_iter, quota, run_dir, log):
    from recovla.expert.r2_collect import R2Spec, run_r2_attempt
    from recovla.sim.rig import SimRig
    pol, runner = _runner()
    rig = SimRig(render=True)
    got = {m: 0 for m in quota}
    try:
        for seed in seeds_iter:
            mode = mode_of(kind, seed)
            if got.get(mode, 0) >= quota.get(mode, 0):
                if all(got[m] >= quota[m] for m in quota):
                    break
                continue
            out = run_r2_attempt(rig, runner, R2Spec(seed, kind, mode), run_dir)
            if out["saved"]:
                got[mode] += 1
            log.write(json.dumps(out, ensure_ascii=False, default=float) + "\n")
            log.flush()
            print(f"[r2] {kind} {seed} {mode} saved {out['saved']} discard {out['discard']} "
                  f"trigger {out['r2']['trigger']} got {got}", flush=True)
    finally:
        rig.close()
    return got


def cmd_gen(a) -> None:
    kind = a.kind
    run_dir = OUTPUTS / "gen" / f"R2_{kind}_{time.strftime('%Y%m%d-%H%M%S')}{a.tag}"
    run_dir.mkdir(parents=True, exist_ok=False)
    base = SEED_BASE[kind]
    with open(run_dir / "r2_generation.jsonl", "w", encoding="utf-8") as log:
        got = _generate(kind, range(base, base + SEED_SPAN), QUOTA[kind], run_dir, log)
    (run_dir / "run.json").write_text(json.dumps({"kind": kind, "quota": QUOTA[kind], "got": got,
                                                  "seed_range": [base, base + SEED_SPAN - 1], "checkpoint": str(R1_30K),
                                                  "written": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2),
                                      encoding="utf-8")


def cmd_smoke(a) -> None:
    run_dir = OUTPUTS / "gen" / f"R2_smoke_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    with open(run_dir / "r2_generation.jsonl", "w", encoding="utf-8") as log:
        for kind in SEED_BASE:
            seeds_ = range(SMOKE_BASE, SMOKE_BASE + 20)
            quota = {m: SMOKE_PER_KIND // len(QUOTA[kind]) or 1 for m in QUOTA[kind]}
            _generate(kind, seeds_, quota, run_dir, log)
    print(run_dir)


def cmd_check(a) -> None:
    """(a) 種類ごとの台本の成功率、(b) 最初のこまが 10 Hz の境目・保存した開始状態からの再生、(c) 映像と状態の図の材料。"""
    import collections
    from recovla.common import seeds
    from recovla.record import replay as legacy
    from recovla.record import replay_scene as RS
    from recovla.sim.rig import SimRig
    run = pathlib.Path(a.run)
    rows = [json.loads(l) for l in (run / "r2_generation.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    split = int(CFG["sim"]["record_every"]) * int(CFG["sim"]["stride"])
    res = {"run": str(run), "by_kind": {}}
    for kind in SEED_BASE:
        rs = [r for r in rows if r["kind"] == kind]
        handed = [r for r in rs if r["r2"]["t_handover"] is not None and not str(r["discard"] or "").endswith("before_handover")
                  and not str(r["discard"] or "").startswith("induce_")]
        res["by_kind"][kind] = {
            "attempts": len(rs), "handed_over": len(handed), "saved": sum(r["saved"] for r in rs),
            "script_success_rate": (sum(r["saved"] for r in handed) / len(handed)) if handed else None,
            "discards": dict(collections.Counter(r["discard"] for r in rs if not r["saved"])),
            "by_mode": dict(collections.Counter(r["mode"] for r in rs if r["saved"])),
            "triggers": dict(collections.Counter(r["r2"]["trigger"] for r in rs if r["saved"]))}
    res["a_pass"] = all(v["script_success_rate"] is None or v["script_success_rate"] >= SCRIPT_SUCCESS_MIN
                        for v in res["by_kind"].values())
    saved = [r for r in rows if r["saved"]]
    res["b_boundary"] = all(r["record_start_step"] % split == 0 for r in saved)
    rng = seeds.stream(49900, "order")
    pick = [saved[i] for i in rng.choice(len(saved), min(REPLAY_N, len(saved)), replace=False)] if saved else []
    rig = SimRig(render=True)
    rep_rows = []
    try:
        for r in pick:
            ep = legacy.load_episode(pathlib.Path(r["path"]))
            row = {"episode": r["name"]}
            for mode in ("step", "window"):
                rep = RS.replay(ep, mode, rig, render=True)
                c = RS.compare(ep, rep, images=True)
                c["pass"] = RS.verdict(mode, c)
                row[mode] = c
            rep_rows.append(row)
    finally:
        rig.close()
    res["b_replay"] = rep_rows
    res["b_pass"] = res["b_boundary"] and bool(rep_rows) and all(x["step"]["pass"] and x["window"]["pass"] for x in rep_rows)
    res["states_at_handover"] = {kind: [r["r2"]["state_at_handover"] for r in saved if r["kind"] == kind] for kind in SEED_BASE}
    res["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (run / "check.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("b_replay", "states_at_handover")},
                     ensure_ascii=False, indent=1, default=float))
    print("replay:", [(x["episode"], x["step"]["pass"], x["window"]["pass"]) for x in rep_rows])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke")
    s = sub.add_parser("gen")
    s.add_argument("--kind", choices=list(SEED_BASE), required=True)
    s.add_argument("--tag", default="")
    s = sub.add_parser("check")
    s.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    {"smoke": cmd_smoke, "gen": cmd_gen, "check": cmd_check}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
