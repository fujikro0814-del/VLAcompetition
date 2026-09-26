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


def _write_video(path, preview, episode_dir, label, mark_t=None):
    """目視用の映像（検査 (c)）: 前半（方策、記録しない、10 Hz を 2 回ずつ）と、保存した区間（引き継ぎの後、台本）をつなぐ。
    俯瞰と手首を横に並べた 512×256、20 fps。"""
    import cv2
    from recovla.record.recorder import read_png
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (512, 256))

    def put(imgs, text):
        im = cv2.cvtColor(np.hstack(imgs), cv2.COLOR_RGB2BGR)
        for i, line in enumerate(text.split("\n")):             # 1 行目に名前、2 行目に時刻と区間
            cv2.putText(im, line, (4, 14 + 14 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(im)
    for t, imgs in preview:
        for _ in range(2):
            mark = " <<SEAM JUMP" if mark_t is not None and abs(t - mark_t) <= 0.5 else ""
            put(imgs, f"{label}\nt={t:5.2f}s policy (not recorded){mark}")
    meta = json.loads((episode_dir / "meta.json").read_text(encoding="utf-8"))
    cams = meta["cameras"]["names"]
    n = len(list((episode_dir / cams[0]).glob("*.png")))
    t0 = float(meta["t_record_start"])
    for i in range(n):
        put([read_png(episode_dir / c / f"{i:06d}.png") for c in cams],
            f"{label}\nt={t0 + i * float(meta['record_dt']):5.2f}s script (recorded)")
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


def cmd_seam_video(a) -> None:
    """目視用（0077 への回答の 2）: 保存した区間の種を、同じ設定で回し直し（保存しない）、前半の方策の行動と塊の切り替わりを
    記録して、継ぎ目の跳び（方策が出した行動の、塊の切り替わりでの速度の差。誘発が行動を上書きしたこまは除く）が
    いちばん大きい種を映像にする。回し直しは GPU の描画の揺れでビットでは再現しないので、引き継ぎの時刻と状態が
    保存した記録と一致したものだけを候補にする。"""
    from recovla.eval import induce as I
    from recovla.expert.r2_collect import R2Spec, run_r2_attempt
    from recovla.sim.rig import SimRig
    run = pathlib.Path(a.run)
    rows = [json.loads(l) for l in (run / "r2_generation.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    saved = [r for r in rows if r["saved"]][a.skip:a.skip + a.n]
    active_log = []
    orig_filter = I.Inducer.filter

    def filt(self, k, act, truth):
        out = orig_filter(self, k, act, truth)
        active_log.append((int(k), bool(self.active)))
        return out
    I.Inducer.filter = filt
    pol, runner = _runner()
    acts = []

    class Rec:
        def __init__(self, r):
            self.r = r

        def __getattr__(self, name):
            return getattr(self.r, name)

        def __call__(self, k, frame, raw, task):
            out = self.r(k, frame, raw, task)
            acts.append(np.asarray(out, float).copy())
            return out
    rig = SimRig(render=True)
    res = []
    try:
        for r in saved:
            acts.clear()
            active_log.clear()
            preview = []
            out = run_r2_attempt(rig, Rec(runner), R2Spec(r["seed"], r["kind"], r["mode"]), None, preview=preview)
            ex = runner.trace()["exec"]
            A = np.array(acts)
            chunk = np.array([e[0] for e in ex[:len(A)]])
            over = dict(active_log)
            v = A[:, :3] / 0.1
            jumps = []
            for k in range(1, len(A)):
                if chunk[k] != chunk[k - 1] and not over.get(k, False) and not over.get(k - 1, False):
                    jumps.append((float(np.linalg.norm(v[k] - v[k - 1])), k))
            h0, h1 = r["r2"]["state_at_handover"], out["r2"]["state_at_handover"]
            same = (r["r2"]["t_handover"] == out["r2"]["t_handover"]
                    and abs(h0["tip_to_cube_xy"] - h1["tip_to_cube_xy"]) < 0.005)
            best = max(jumps) if jumps else (0.0, -1)
            res.append({"name": r["name"], "seed": r["seed"], "reproduced": bool(same), "seam_max": best[0], "k": best[1],
                        "seam_median": float(np.median([j for j, _ in jumps])) if jumps else None,
                        "t_handover_saved": r["r2"]["t_handover"], "t_handover_rerun": out["r2"]["t_handover"],
                        "preview": preview, "path": r["path"], "color": r["color"], "trigger": r["r2"]["trigger"]})
            print(r["name"], "reproduced", same, "seam max %.3f at k=%d" % best, flush=True)
    finally:
        rig.close()
        I.Inducer.filter = orig_filter
    cand = [x for x in res if x["reproduced"]] or res
    pick = max(cand, key=lambda x: x["seam_max"])
    vdir = run / "videos_seam"
    vdir.mkdir(exist_ok=True)
    k = pick["k"]
    t_jump = pick["preview"][k][0] if 0 <= k < len(pick["preview"]) else None
    pv = [(t, im) for t, im in pick["preview"]]
    label = f"{pick['name']} RERUN, seam {pick['seam_max']:.2f} m/s at {t_jump:.1f}s"
    _write_video(vdir / f"{pick['name']}_seam.mp4", pv, pathlib.Path(pick["path"]), label, mark_t=t_jump)
    (vdir / "seam_pick.json").write_text(json.dumps({
        "note": "回し直し（保存しない）。継ぎ目の跳びは方策の出した行動、誘発の上書きを除く。映像の後半は保存した区間",
        "picked": {k2: v2 for k2, v2 in pick.items() if k2 != "preview"}, "t_jump": t_jump,
        "rows": [{k2: v2 for k2, v2 in x.items() if k2 != "preview"} for x in res],
        "written": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print("picked", pick["name"], pick["seam_max"], t_jump)


MEDIA_ORIGINAL_PER_KIND = 4       # 生成中に書いた映像のうち写す本数（残りの 1 本は継ぎ目の跳びの回し直し＝0077 への回答の 2）
EVAL_NAIVE = {"P1": "RTCREDO/NV_P1", "P2": "RTCREDO/NV_P2", "P3": "RTCREDO/NV_P3"}   # 決め直しの naive の評価


def _state_row(tip, cube, fingers, closed):
    from recovla.sim import frames
    return {"tip_to_cube_xy": float(np.hypot(*(tip[:2] - cube[:2]))), "finger_gap": float(np.sum(fingers)),
            "tip_z": float(tip[2]), "target_rise": float(cube[2] - frames.CUBE_REST_Z), "gripper_closed": bool(closed)}


def cmd_media(a) -> None:
    """検査 (c) の資料（0077 への回答の 2）: 引き継ぎの時点の状態の分布を、決め直しの naive の評価で制御が戻った時点
    （誘発が上書きした最後のこまの次、成立した試行）の状態と重ねた図、Step F の復帰 90 本の最初のこまとの図、
    種類ごとの映像（生成中に書いた先頭 4 本＋継ぎ目の跳びが目立つ回し直し 1 本）を docs/media/<題>/ に置く。
    README は本線が結果を見て書く。"""
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
        seam_names = {v.stem[:-len("_seam")] for v in (run / "videos_seam").glob("*_seam.mp4")}
        orig = [v for v in sorted((run / "videos").glob("*.mp4")) if v.stem not in seam_names]   # 回し直しと同じ種は除く
        for v in orig[:MEDIA_ORIGINAL_PER_KIND]:
            shutil.copy2(v, out / v.name)
            videos.append(v.name)
        for v in sorted((run / "videos_seam").glob("*.mp4")):
            shutil.copy2(v, out / v.name)
            videos.append(v.name)
        if (run / "videos_seam" / "seam_pick.json").is_file():
            shutil.copy2(run / "videos_seam" / "seam_pick.json", out / f"seam_pick_{run.name.split('_')[1]}.json")
    for kind, d in EVAL_NAIVE.items():
        for p in sorted((OUTPUTS / "eval" / d).glob("trial_*.json")):
            m = json.loads(p.read_text(encoding="utf-8"))
            if not m["induce"]["established"]:
                continue
            arr = np.load(p.with_suffix(".npz"))
            act = np.flatnonzero(arr["induce_active"])
            if not act.size:
                continue
            f = min(int(act[-1]) + 1, len(arr["sim_time"]) - 1)
            ti = int(arr["target"][0])
            groups.setdefault(f"eval naive {kind}", []).append(
                _state_row(arr["fingertip"][f], arr["cube_pos"][f, ti], arr["fingers"][f], arr["gripper_closed"][f]))
    data = json.loads((OUTPUTS / "f" / "data.json").read_text(encoding="utf-8"))
    drun = config.ROOT / data["run"]
    for c in data["chosen"]:
        d = np.load(drun / c["recovery"] / "data.npz")
        ti = int(d["target"][0])
        groups.setdefault(f"data {c['kind']}", []).append(
            _state_row(d["fingertip"][0], d["cube_pos"][0, ti], d["fingers"][0], d["gripper_closed"][0]))
    keys = [("tip_to_cube_xy", "hand-cube horizontal distance [m]"), ("finger_gap", "finger opening [m]"),
            ("tip_z", "fingertip height [m]"), ("target_rise", "target cube rise above table rest [m]")]
    color = {"R2": "C3", "eval": "C2", "data": "C0"}

    def figure(names, path, title):
        fig, axs = plt.subplots(1, len(keys), figsize=(20, 5))
        for ax, (k, lab) in zip(axs, keys):
            for i, g in enumerate(names):
                v = np.array([s[k] for s in groups[g]])
                cl = np.array([s["gripper_closed"] for s in groups[g]])
                x = i + np.random.default_rng(i).uniform(-0.18, 0.18, v.size)
                base = color[g.split()[0]]
                ax.scatter(x[~cl], v[~cl], s=12, color=base, alpha=0.6, marker="o")
                ax.scatter(x[cl], v[cl], s=16, color=base, alpha=0.8, marker="x")
                ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, "k-", lw=2)
            ax.set_xticks(range(len(names)), [f"{n}\n(n={len(groups[n])})" for n in names], rotation=45, ha="right",
                          fontsize=8)
            ax.set_title(lab)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out / path, dpi=110)
        plt.close(fig)
    figure([n for n in ("R2 P1", "eval naive P1", "R2 P2", "eval naive P2", "R2 P3", "eval naive P3", "R2 natural")
            if n in groups], "handover_vs_eval_naive.png",
           "R2 state at handover (red) vs naive-eval state when control returns (green); o = open, x = closed")
    figure([n for n in ("data A", "data B", "data C", "R2 P1", "R2 P2", "R2 P3", "R2 natural") if n in groups],
           "handover_states.png",
           "R2 state at handover (red) vs first recorded frame of Step F recovery data (blue); o = open, x = closed")
    summ = {g: {"n": len(v), "gripper_closed": sum(s["gripper_closed"] for s in v),
                **{k: {"median": float(np.median([s[k] for s in v])), "min": float(min(s[k] for s in v)),
                       "max": float(max(s[k] for s in v))} for k, _ in keys}} for g, v in groups.items()}
    cover = {}                                      # naive の評価の状態のうち、同じ種類の R2 の最小〜最大の外にあるもの
    for kind in EVAL_NAIVE:
        ev, r2 = groups.get(f"eval naive {kind}", []), groups.get(f"R2 {kind}", [])
        if not ev or not r2:
            continue
        cover[kind] = {"n_eval": len(ev), "outside_R2_minmax": {
            k: sum(not (min(s[k] for s in r2) - 0.001 <= e[k] <= max(s[k] for s in r2) + 0.001) for e in ev)
            for k, _ in keys},
            "outside_any": sum(any(not (min(s[k] for s in r2) - 0.001 <= e[k] <= max(s[k] for s in r2) + 0.001)
                                   for k, _ in keys) for e in ev),
            "gripper_closed_eval": sum(e["gripper_closed"] for e in ev),
            "gripper_closed_R2": sum(s["gripper_closed"] for s in r2)}
    (out / "handover_states.json").write_text(json.dumps({
        "runs": a.runs, "eval_naive": EVAL_NAIVE, "videos": videos, "summary": summ, "eval_outside_R2": cover,
        "note": "評価の状態は制御が戻った時点（誘発が上書きした最後のこまの次）、R2 は引き継ぎの時点、データは最初のこま。"
                "範囲の外の許容は 1 mm", "written": time.strftime("%Y-%m-%d %H:%M:%S")},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(cover, ensure_ascii=False, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke")
    s = sub.add_parser("gen")
    s.add_argument("--kind", choices=list(SEED_BASE), required=True)
    s.add_argument("--tag", default="")
    s = sub.add_parser("check")
    s.add_argument("--run", required=True)
    s = sub.add_parser("seam-video")
    s.add_argument("--run", required=True)
    s.add_argument("--n", type=int, default=10)
    s.add_argument("--skip", type=int, default=0)
    s = sub.add_parser("media")
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--out", required=True, help="docs/media の下のフォルダ名（例 2026-09-27_R2）")
    a = ap.parse_args(argv)
    {"smoke": cmd_smoke, "gen": cmd_gen, "check": cmd_check, "media": cmd_media, "seam-video": cmd_seam_video}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
