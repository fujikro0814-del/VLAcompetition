"""R2 の引き継ぎ区間の生成と検査（決裁 0069 の 3・0070 の 1。生成は 0070 の 2 の報告への回答の後）。

    .venv\\Scripts\\python.exe scripts\\46_r2.py smoke                      # 実装の検査（種 49010〜、各種類 2 本、別のフォルダ）
    .venv\\Scripts\\python.exe scripts\\46_r2.py gen --kind P2 [--tag TAG]  # 本番: 種類ごとに割り当ての本数に達するまで
    .venv\\Scripts\\python.exe scripts\\46_r2.py check --run outputs\\gen\\R2_...   # 生成の後の検査 (a)(b)
    .venv\\Scripts\\python.exe scripts\\46_r2.py media --runs <4 つのフォルダ> --out 2026-09-27_R2   # (c) 映像と状態の図

割り当て（0069 の 3）: 乙 P1・P2・P3 各 40（引き継ぎの時点は成立の時点 20・成立の後 0〜3 s 20）、甲 natural 60。
種: P1 40000〜40099、P2 41000〜41099、P3 42000〜42099、natural 43000〜43099（先頭から、割り当てに達するまで）。
引き継ぎの時点の半々は種の偶奇で決める（偶数＝成立の時点、奇数＝成立の後 0〜3 s）。結果を見る前に決まる。
方策は R1 の 3 万手、評価と同じ設定（configs の runtime: s=10・d=4、RTC の有無と範囲は 0075 の 1 で決め直した値）。
評価用の帯の種は使わない。
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
    # 本線の設定（configs の runtime。RTC の有無と範囲は 0075 の 1 の決め直しで決めた値）
    return pol, SceneRunner(pol, RuntimeConfig(rt["mode"], int(rt["exec_interval"]), int(rt["delay_steps"]),
                                               execution_horizon=int(rt["rtc_guidance_horizon"])),
                            {"schedule": rt["rtc_schedule"], "max_guidance_weight": rt["rtc_max_guidance_weight"]})


def _write_video(path, preview, episode_dir, label):
    """目視用の映像（検査 (c)）: 前半（方策、記録しない、10 Hz を 2 回ずつ）と、保存した区間（引き継ぎの後、台本）をつなぐ。
    俯瞰と手首を横に並べた 512×256、20 fps。"""
    import cv2
    from recovla.record.recorder import read_png
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (512, 256))

    def put(imgs, text):
        im = cv2.cvtColor(np.hstack(imgs), cv2.COLOR_RGB2BGR)
        cv2.putText(im, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(im)
    for t, imgs in preview:
        for _ in range(2):
            put(imgs, f"{label} t={t:5.2f}s policy (not recorded)")
    meta = json.loads((episode_dir / "meta.json").read_text(encoding="utf-8"))
    cams = meta["cameras"]["names"]
    n = len(list((episode_dir / cams[0]).glob("*.png")))
    t0 = float(meta["t_record_start"])
    for i in range(n):
        put([read_png(episode_dir / c / f"{i:06d}.png") for c in cams],
            f"{label} t={t0 + i * float(meta['record_dt']):5.2f}s script (recorded)")
    vw.release()


def _generate(kind, seeds_iter, quota, run_dir, log, videos=0):
    from recovla.expert.r2_collect import R2Spec, run_r2_attempt
    from recovla.sim.rig import SimRig
    pol, runner = _runner()
    rig = SimRig(render=True)
    got = {m: 0 for m in quota}
    n_video = 0
    try:
        for seed in seeds_iter:
            mode = mode_of(kind, seed)
            if got.get(mode, 0) >= quota.get(mode, 0):
                if all(got[m] >= quota[m] for m in quota):
                    break
                continue
            preview = [] if n_video < videos else None
            out = run_r2_attempt(rig, runner, R2Spec(seed, kind, mode), run_dir, preview=preview)
            if out["saved"]:
                got[mode] += 1
                if preview is not None:
                    (run_dir / "videos").mkdir(exist_ok=True)
                    _write_video(run_dir / "videos" / f"{out['name']}.mp4", preview, pathlib.Path(out["path"]),
                                 f"{out['name']} {out['color']} {out['r2']['trigger']}")
                    n_video += 1
            log.write(json.dumps(out, ensure_ascii=False, default=float) + "\n")
            log.flush()
            print(f"[r2] {kind} {seed} {mode} saved {out['saved']} discard {out['discard']} "
                  f"trigger {out['r2']['trigger']} got {got}", flush=True)
    finally:
        rig.close()
    return got


def cmd_gen(a) -> None:
    from recovla.common import code_version
    kind = a.kind
    cv = code_version.code_version()           # 始めた時点のコードの版（未 commit の変更がないことも記録に残す）
    run_dir = OUTPUTS / "gen" / f"R2_{kind}_{time.strftime('%Y%m%d-%H%M%S')}{a.tag}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "start.json").write_text(json.dumps({"kind": kind, "code_version": cv, "runtime": CFG["runtime"],
                                                    "started": time.strftime("%Y-%m-%d %H:%M:%S")},
                                                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    base = SEED_BASE[kind]
    with open(run_dir / "r2_generation.jsonl", "w", encoding="utf-8") as log:
        got = _generate(kind, range(base, base + SEED_SPAN), QUOTA[kind], run_dir, log, videos=VIDEOS_PER_KIND)
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
            _generate(kind, seeds_, quota, run_dir, log, videos=1)
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


def cmd_media(a) -> None:
    """検査 (c) の資料: 種類ごとの映像（生成中に書いた先頭 5 本）を docs/media/<題>/ に写し、引き継ぎの時点の状態の図を
    Step F の復帰 90 本の最初のこま（A・B・C）と並べて描く。README は本線が結果を見て書く。"""
    import shutil
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = config.ROOT / "docs" / "media" / a.out
    out.mkdir(parents=True, exist_ok=True)
    groups, videos = {}, []
    for run in map(pathlib.Path, a.runs):
        rows = [json.loads(l) for l in (run / "r2_generation.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        for r in rows:
            if r["saved"]:
                groups.setdefault(f"R2 {r['kind']}", []).append(r["r2"]["state_at_handover"])
        for v in sorted((run / "videos").glob("*.mp4")):
            shutil.copy2(v, out / v.name)
            videos.append(v.name)
    data = json.loads((OUTPUTS / "f" / "data.json").read_text(encoding="utf-8"))
    drun = config.ROOT / data["run"]
    for c in data["chosen"]:
        d = np.load(drun / c["recovery"] / "data.npz")
        ti = int(d["target"][0])
        tip, cube = d["fingertip"][0], d["cube_pos"][0, ti]
        groups.setdefault(f"data {c['kind']}", []).append(
            {"tip_to_cube_xy": float(np.hypot(*(tip[:2] - cube[:2]))), "finger_gap": float(np.sum(d["fingers"][0])),
             "tip_z": float(tip[2]), "gripper_closed": bool(d["gripper_closed"][0])})
    names = [n for n in ("data A", "data B", "data C", "R2 P1", "R2 P2", "R2 P3", "R2 natural") if n in groups]
    keys = [("tip_to_cube_xy", "hand-cube horizontal distance [m]"), ("finger_gap", "finger opening [m]"),
            ("tip_z", "fingertip height [m]")]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (k, lab) in zip(axs, keys):
        for i, g in enumerate(names):
            v = np.array([s[k] for s in groups[g]])
            cl = np.array([s["gripper_closed"] for s in groups[g]])
            x = i + np.random.default_rng(i).uniform(-0.18, 0.18, v.size)
            base = "C0" if g.startswith("data") else "C3"
            ax.scatter(x[~cl], v[~cl], s=12, color=base, alpha=0.6, marker="o")
            ax.scatter(x[cl], v[cl], s=16, color=base, alpha=0.8, marker="x")
            ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, "k-", lw=2)
        ax.set_xticks(range(len(names)), [f"{n}\n(n={len(groups[n])})" for n in names], rotation=45, ha="right", fontsize=8)
        ax.set_title(lab)
    fig.suptitle("state at handover (R2, red) vs first recorded frame of Step F recovery data (blue); o = open, x = closed")
    fig.tight_layout()
    fig.savefig(out / "handover_states.png", dpi=110)
    plt.close(fig)
    summ = {g: {"n": len(v), "gripper_closed": sum(s["gripper_closed"] for s in v),
                **{k: {"median": float(np.median([s[k] for s in v])), "min": float(min(s[k] for s in v)),
                       "max": float(max(s[k] for s in v))} for k, _ in keys}} for g, v in groups.items()}
    (out / "handover_states.json").write_text(json.dumps({"runs": a.runs, "videos": videos, "summary": summ,
                                                          "written": time.strftime("%Y-%m-%d %H:%M:%S")},
                                                         ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke")
    s = sub.add_parser("gen")
    s.add_argument("--kind", choices=list(SEED_BASE), required=True)
    s.add_argument("--tag", default="")
    s = sub.add_parser("check")
    s.add_argument("--run", required=True)
    s = sub.add_parser("media")
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--out", required=True, help="docs/media の下のフォルダ名（例 2026-09-27_R2）")
    a = ap.parse_args(argv)
    {"smoke": cmd_smoke, "gen": cmd_gen, "check": cmd_check, "media": cmd_media}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
