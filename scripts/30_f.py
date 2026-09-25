"""Step F の段取り（手順書 §7）。結果は outputs/f/<項目>.json。

    .venv\\Scripts\\python.exe scripts\\30_f.py sweep [--workers 8] [--n 120] [--smoke]   # 注入の値の振り（掲示板 0033・0034）
    ... check-gen [--workers 8] [--n 60] [--smoke]   # 完了条件 1〜4 の試験の生成（描画あり、作り直しあり）
    ... check-eval [--smoke]                         # 完了条件 1〜4 の測定 → outputs/f/check.json（tests/test_f_recovery.py が判定）

振り（描画なし、作り直しなし＝retry 0）:
  A: 閉じる高さの上げ幅の中心 2.0〜4.5 cm（幅 ±0.5 cm、上げる型だけ＝raise_ratio 1）と、横ずらしだけ（raise_ratio 0）
  B: 箱の中心からの距離の下限 0.15〜0.23 m
  C: 今の値
種（Step D・F の試験の帯。Step D は 50000〜50624・50900、Step F の試しは 50000〜50039・50100〜50119 を使った）:
  A 51000〜、B 51200〜、C 51400〜（各 n 個。色は種ごとに机上の色から 1 つ）。同じ種類の中では、どの点も同じ種を
  使う（対応のある比較。注入の乱数列は種類によらず同じ順に引くので、点の違いは値の違いだけになる）
"""
import argparse
import collections
import copy
import json
import multiprocessing
import os
import pathlib
import time

from recovla.common import config
from recovla.eval.stats import wilson_interval

CFG = config.load()
OUT = config.path(CFG["paths"]["outputs"]) / "f"
SEED_BASE = {"A": 51000, "B": 51200, "C": 51400}
A_RAISE = [0.020, 0.025, 0.030, 0.035, 0.040, 0.045]
A_HALF = 0.005
B_DMIN = [0.15, 0.17, 0.19, 0.21, 0.23]
CATS = ("not_effective", "landing_invalid", "recovery_failed", "success")


def points() -> list:
    pts = [{"id": f"A_raise_{r * 100:.1f}cm", "kind": "A",
            "overrides": {"A": {"raise_close_m": [round(r - A_HALF, 4), round(r + A_HALF, 4)], "raise_ratio": 1.0}}}
           for r in A_RAISE]
    pts.append({"id": "A_lateral_only", "kind": "A", "overrides": {"A": {"raise_ratio": 0.0}}})
    pts += [{"id": f"B_dmin_{d:.2f}m", "kind": "B", "overrides": {"B": {"min_dist_from_box_m": d}}} for d in B_DMIN]
    pts.append({"id": "C_current", "kind": "C", "overrides": {}})
    return pts


def apply_overrides(cfg: dict, overrides: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    for k, sub in overrides.items():
        cfg["inject"][k].update(sub)
    return cfg


def write(name: str, obj: dict) -> pathlib.Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.json"
    p.write_text(json.dumps({"check": name, "written": time.strftime("%Y-%m-%d %H:%M:%S"), **obj},
                            ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[f] wrote {p}", flush=True)
    return p


# ------------------------------------------------------------------------------------ the workers

_RIG = None


def _init() -> None:
    global _RIG
    from recovla.sim.rig import SimRig
    _RIG = SimRig(render=False)


def _job(job):
    from recovla.expert import generate as G
    from recovla.sim import scene
    pid, kind, overrides, seeds_ = job
    _RIG.cfg = apply_overrides(CFG, overrides)            # run_attempt・注入は rig.cfg を使う（物理の設定は変えない）
    rows = []
    for seed in seeds_:
        lay = scene.sample_layout(seed)
        color = lay.table_colors[seed % len(lay.table_colors)]
        t0 = time.perf_counter()
        a = G.run_attempt(_RIG, G.EpisodeSpec(seed, color, kind=kind), 0, None, False)
        a.pop("_frames", None)
        a["point"] = pid
        a["wall_attempt_s"] = round(time.perf_counter() - t0, 3)
        a["worker_pid"] = os.getpid()
        rows.append(a)
    return rows


# ------------------------------------------------------------------------------------ the summary

def category(a: dict) -> str:
    if a["success"]:
        return "success"
    if a["failure"] == "inject_not_effective":
        return "not_effective"
    if a["failure"] == "inject_landing_invalid":
        return "landing_invalid"
    if a["inject"]["status"] == "confirmed":
        return "recovery_failed"
    return "excluded"                                      # 注入の前の自然な失敗など（件数だけ別に数える）


def landing_reasons(a: dict, min_clear: float) -> set:
    """着地が不自然の内訳（決裁 0034 の 4）: 壁・箱の縁／先客（箱の中）の立方体／机上の他の立方体／その他。
    立方体は「隙間が min_clear 未満」か「記録の前に目標が触れた」のどちらか。"""
    inj = a["inject"]
    info = inj["info"]
    clear = info.get("clearance_m", {})
    out = set()
    if info.get("over_box") or any(v < min_clear for k, v in clear.items() if k.startswith("box_wall")):
        out.add("wall_or_box_edge")
    for k, v in clear.items():
        if not k.startswith("cube_"):
            continue
        c = k[len("cube_"):]
        if v < min_clear or c in inj.get("cube_contact_before", []):
            out.add("prefilled_cube" if c in inj.get("prefilled", []) else "table_cube")
    rest = set(info.get("landing_fail", [])) - {"over_box", "clearance"}
    if inj.get("reason") == "no_rest":
        rest.add("no_rest")
    if rest or not out:
        out.add("other:" + ",".join(sorted(rest)) if rest else "other")
    return out


def wilson(k: int, n: int) -> list:
    lo, hi = wilson_interval(k, n)
    return [None if lo is None else round(lo, 4), None if hi is None else round(hi, 4)]


def summarize(rows: list) -> list:
    split = float(CFG["inject"]["landing_orientation_split_deg"])
    min_clear = float(CFG["inject"]["landing"]["min_clearance_m"])
    by = collections.defaultdict(list)
    for a in rows:
        by[a["point"]].append(a)
    out = []
    for pt in points():
        rs = by.get(pt["id"], [])
        if not rs:
            continue
        n = len(rs)
        cnt = collections.Counter(category(a) for a in rs)
        excluded = collections.Counter(a["failure"] for a in rs if category(a) == "excluded")
        eff = cnt["success"] + cnt["recovery_failed"] + cnt["landing_invalid"]
        valid = cnt["success"] + cnt["recovery_failed"]
        succ = [a for a in rs if a["success"]]
        within = sum(abs(a["inject"]["info"].get("yaw_rel_deg", 0.0)) <= split for a in succ)
        row = {
            "point": pt["id"], "kind": pt["kind"], "overrides": pt["overrides"], "n": n,
            "seeds": [min(a["layout_seed"] for a in rs), max(a["layout_seed"] for a in rs)],
            "counts": {c: cnt[c] for c in CATS}, "excluded": dict(excluded),
            "wilson_of_n": {c: wilson(cnt[c], n) for c in CATS},
            "not_effective_rate": [cnt["not_effective"], n, wilson(cnt["not_effective"], n)],
            "landing_invalid_ratio": [cnt["landing_invalid"], eff, wilson(cnt["landing_invalid"], eff)],
            "recovery_success": [cnt["success"], valid, wilson(cnt["success"], valid)],
            "success_yaw_within": [within, len(succ)], "success_yaw_beyond": [len(succ) - within, len(succ)],
            "wall_attempt_s_mean": round(sum(a["wall_attempt_s"] for a in rs) / n, 3),
        }
        if pt["kind"] == "A":
            row["a_mode_counts"] = dict(collections.Counter(a["inject"]["params"].get("a_mode") for a in rs))
            row["not_effective_reason"] = dict(collections.Counter(
                a["inject"]["reason"] for a in rs if category(a) == "not_effective"))
        inv = [a for a in rs if category(a) == "landing_invalid"]
        reasons = collections.Counter(r for a in inv for r in landing_reasons(a, min_clear))
        row["landing_invalid_breakdown"] = dict(reasons)
        row["landing_invalid_examples"] = [
            {"name": a["name"], "reasons": sorted(landing_reasons(a, min_clear)),
             "clearance_m": a["inject"]["info"].get("clearance_m"), "target_pos": a["inject"]["info"].get("target_pos"),
             "cube_contact_before": a["inject"].get("cube_contact_before"), "prefilled": a["inject"].get("prefilled")}
            for a in inv[:8]]
        row["touched_other_cube_but_landing_ok"] = sum(
            1 for a in rs if category(a) in ("success", "recovery_failed") and a["inject"].get("cube_contact_before"))
        out.append(row)
    return out


def cmd_sweep(a) -> None:
    n = 2 if a.smoke else a.n
    jobs = []
    for pt in points():
        base = SEED_BASE[pt["kind"]]
        seeds_ = list(range(base, base + n))
        for i in range(0, n, a.chunk):
            jobs.append((pt["id"], pt["kind"], pt["overrides"], seeds_[i:i + a.chunk]))
    OUT.mkdir(parents=True, exist_ok=True)
    name = "sweep_smoke" if a.smoke else "sweep"
    rows_path = OUT / f"{name}_rows.jsonl"
    t0 = time.perf_counter()
    rows = []
    with open(rows_path, "w", encoding="utf-8") as log:
        if a.workers <= 1:
            _init()
            it = map(_job, jobs)
            pool = None
        else:
            pool = multiprocessing.get_context("spawn").Pool(a.workers, initializer=_init)
            it = pool.imap_unordered(_job, jobs)
        for rs in it:
            for r in rs:
                rows.append(r)
                log.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            log.flush()
            print(f"[f] {len(rows)}/{len(points()) * n} {time.perf_counter() - t0:.0f}s", flush=True)
        if pool is not None:
            pool.close()
            pool.join()
    write(name, {"n_per_point": n, "workers": a.workers, "wall_s": round(time.perf_counter() - t0, 1),
                 "seed_ranges": {k: [v, v + n - 1] for k, v in SEED_BASE.items()},
                 "rows": str(rows_path.relative_to(config.ROOT)), "points": summarize(rows)})


# ------------------------------------------------------------------------ completion conditions 1〜4

GEN = config.path(CFG["paths"]["outputs"]) / "gen"
CHECK_SEED_BASE = {"A": 52000, "B": 52200, "C": 52400}   # 完了条件の試験（振りの種と重ねない）
REPLAY_PER_KIND = 3
VIDEOS_PER_KIND = 5


def check_specs(n: int) -> list:
    from recovla.expert import generate as G
    from recovla.sim import scene
    specs = []
    for kind, base in CHECK_SEED_BASE.items():
        for seed in range(base, base + n):
            lay = scene.sample_layout(seed)
            specs.append(G.EpisodeSpec(seed, lay.table_colors[seed % len(lay.table_colors)], lay.kind, kind))
    return specs


def cmd_check_gen(a) -> None:
    """完了条件の試験の生成（描画あり、作り直しあり）。学習と同時に回さない（GPU の描画）。"""
    from recovla.expert import generate as G
    n = 2 if a.smoke else a.n
    run = GEN / f"f_check{'_smoke' if a.smoke else ''}_{time.strftime('%Y%m%d-%H%M%S')}"
    G.generate(check_specs(n), run, workers=a.workers, render=True)
    print(f"[f] generated {run}", flush=True)


def latest_run(prefix: str) -> pathlib.Path:
    runs = sorted(p for p in GEN.glob(f"{prefix}_2*") if p.is_dir())
    if not runs:
        raise SystemExit(f"no run outputs/gen/{prefix}_* (run check-gen first)")
    return runs[-1]


def cmd_check_eval(a) -> None:
    import numpy as np
    from recovla.common import seeds
    from recovla.expert import inject
    from recovla.record import replay as legacy, replay_scene as RS
    from recovla.sim.rig import SimRig
    run = latest_run("f_check_smoke" if a.smoke else "f_check")
    results = [json.loads(l) for l in (run / "generation.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    attempts = [dict(t, point=t["kind"]) for r in results for t in r["attempts"]]
    # 1: 試みごとの 4 区分（作り直しの各回を 1 回と数える）
    split = float(CFG["inject"]["landing_orientation_split_deg"])
    table = {}
    for kind in CHECK_SEED_BASE:
        rs = [t for t in attempts if t["kind"] == kind]
        cnt = collections.Counter(category(t) for t in rs)
        valid = cnt["success"] + cnt["recovery_failed"]
        eff = valid + cnt["landing_invalid"]
        succ = [t for t in rs if t["success"]]
        within = sum(abs(t["inject"]["info"].get("yaw_rel_deg", 0.0)) <= split for t in succ)
        table[kind] = {"attempts": len(rs), "counts": {c: cnt[c] for c in CATS},
                       "excluded": dict(collections.Counter(t["failure"] for t in rs if category(t) == "excluded")),
                       "recovery_success": [cnt["success"], valid, wilson(cnt["success"], valid)],
                       "landing_invalid_ratio": [cnt["landing_invalid"], eff, wilson(cnt["landing_invalid"], eff)],
                       "success_yaw_within": [within, len(succ)], "success_yaw_beyond": [len(succ) - within, len(succ)],
                       "specs": sum(1 for r in results if r["kind"] == kind),
                       "specs_saved": sum(1 for r in results if r["kind"] == kind and r["success"])}
    # 2: 保存した全エピソードの最初のこま
    saved = sorted(p for p in run.iterdir() if p.is_dir() and not p.name.endswith(".partial"))
    cond2 = []
    for p in saved:
        ep = legacy.load_episode(p)
        c = inject.start_condition(ep.data, ep.meta)
        cond2.append({"episode": p.name, "ok": c["ok"], "checks": c["checks"]})
    # 3: 種類ごとに無作為の 3 本を再生（step はビット一致と画像、window は手先 5 mm 以内）
    rng = seeds.stream(52900, "order")
    rig = SimRig(render=True)
    replay_rows = []
    try:
        for kind in CHECK_SEED_BASE:
            eps = [p for p in saved if p.name.startswith(kind + "_")]
            for i in rng.choice(len(eps), min(REPLAY_PER_KIND, len(eps)), replace=False):
                ep = legacy.load_episode(eps[i])
                row = {"episode": eps[i].name}
                for mode in ("step", "window"):
                    rep = RS.replay(ep, mode, rig, render=True)
                    r = RS.compare(ep, rep, images=True)
                    r["pass"] = RS.verdict(mode, r)
                    row[mode] = r
                replay_rows.append(row)
    finally:
        rig.close()
    # 4: 種類ごとに 5 本の映像（俯瞰と手首を横に並べる。Step D の videos と同じ作り）
    import cv2
    import shutil
    import tempfile
    from recovla.record.recorder import read_png
    vdir = OUT / "videos"
    vdir.mkdir(parents=True, exist_ok=True)
    made = []
    for kind in CHECK_SEED_BASE:
        for p in [p for p in saved if p.name.startswith(kind + "_")][:VIDEOS_PER_KIND]:
            n = json.loads((p / "meta.json").read_text(encoding="utf-8"))["n_frames"]
            tmp = pathlib.Path(tempfile.mkdtemp()) / "v.mp4"
            vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 20, (512, 256))
            for i in range(n):
                im = np.hstack([read_png(p / v / f"{i:06d}.png") for v in CFG["sim"]["cameras"]])
                vw.write(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            vw.release()
            dst = vdir / f"{p.name}.mp4"
            shutil.move(str(tmp), str(dst))
            made.append(str(dst.relative_to(config.ROOT)))
    write("check_smoke" if a.smoke else "check", {
        "run": str(run.relative_to(config.ROOT)), "table": table,
        "cond1_pass": all(t["recovery_success"][1] > 0 and t["recovery_success"][0] / t["recovery_success"][1] >= 0.9
                          and (t["landing_invalid_ratio"][1] == 0 or t["landing_invalid_ratio"][0] / t["landing_invalid_ratio"][1]
                               <= float(CFG["inject"]["landing"]["max_invalid_ratio"])) for t in table.values()),
        "cond2_episodes": len(cond2), "cond2_pass": bool(cond2) and all(c["ok"] for c in cond2),
        "cond2_failures": [c for c in cond2 if not c["ok"]],
        "cond3_rows": replay_rows,
        "cond3_pass": bool(replay_rows) and all(r["step"]["pass"] and r["window"]["pass"] for r in replay_rows),
        "cond4_videos": made,
        "cond4_pass": all(sum(1 for m in made if pathlib.Path(m).name.startswith(k + "_")) >= VIDEOS_PER_KIND
                          for k in CHECK_SEED_BASE),
        "seed_ranges": {k: [v, v + a.n - 1] for k, v in CHECK_SEED_BASE.items()}, "replay_order_seed": 52900})


# ---------------------------------------------------------------------------- R1・N1 の計画（7）

# 計画書 §7: 通常 240 本（空の箱 60 配置 x 3 色、先客 1 個 15 配置 x 2 色、先客 2 個 30 配置 x 1 色）、R1・N1 共通。
# 復帰 90 本（A40・B30・C20）は通常とは別の種（30000〜）で、配置の種類の割合（75・12.5・12.5%）を通常と揃える。
# N1 の追加 90 本は、復帰と同じ配置・同じ目標の通常デモ（一度で成功するもの）。
NORMAL_SEEDS = {"empty": (20000, 60), "prefilled_1": (20100, 15), "prefilled_2": (20200, 30)}
RECOVERY_NEED = {"A": 40, "B": 30, "C": 20}
RECOVERY_SEED_BASE = {"A": 30000, "B": 31000, "C": 32000}
RECOVERY_KIND_SHARE = {"empty": 0.75, "prefilled_1": 0.125, "prefilled_2": 0.125}
CANDIDATE_FACTOR = 2.0               # 捨てる分を見込んで、各枠の候補をこの倍数だけ先に決めておく（使うのは先頭から）


def split_counts(total: int) -> dict:
    """種類ごとの必要数を配置の種類へ割り振る（最大剰余法。通常の割合 180:30:30 と同じ）。"""
    raw = {k: total * s for k, s in RECOVERY_KIND_SHARE.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[:total - sum(out.values())]:
        out[k] += 1
    return out


def cmd_plan(a) -> None:
    from recovla.sim import scene
    normal = []
    for lk, (base, n) in NORMAL_SEEDS.items():
        for seed in range(base, base + n):
            lay = scene.sample_layout(seed, lk)
            normal += [{"seed": seed, "color": c, "layout_kind": lk, "start": lay.start} for c in lay.table_colors]
    recovery = {}
    for kind, need in RECOVERY_NEED.items():
        cells = {}
        seed = RECOVERY_SEED_BASE[kind]
        for lk, k in split_counts(need).items():
            cand = []
            for i in range(int(round(k * CANDIDATE_FACTOR))):
                lay = scene.sample_layout(seed, lk)
                cols = lay.table_colors
                cand.append({"seed": seed, "color": cols[i % len(cols)], "layout_kind": lk, "start": lay.start})
                seed += 1
            cells[lk] = {"need": k, "candidates": cand}
        recovery[kind] = cells

    def comp(rows):
        return {"n": len(rows), "color": dict(collections.Counter(r["color"] for r in rows)),
                "layout_kind": dict(collections.Counter(r["layout_kind"] for r in rows)),
                "start": dict(collections.Counter(r["start"] for r in rows))}
    first = [c for cells in recovery.values() for cell in cells.values() for c in cell["candidates"][:cell["need"]]]
    # 所要時間の見積もり: 振りの 1 本の時間（描画なし）と、Step D の描画ありの並列の本/時（outputs/d/throughput.json）
    est = {}
    sweep = OUT / "sweep.json"
    if sweep.is_file():
        pts = json.loads(sweep.read_text(encoding="utf-8"))["points"]
        est["recovery_attempt_s_no_render"] = {p["point"]: p["wall_attempt_s_mean"] for p in pts}
    thr = config.path(CFG["paths"]["outputs"]) / "d" / "throughput.json"
    if thr.is_file():
        est["normal_throughput_d"] = {r["workers"]: r["episodes_per_hour"]
                                      for r in json.loads(thr.read_text(encoding="utf-8"))["rows"]}
    write("plan", {"normal": {"specs": normal, "composition": comp(normal), "seed_ranges": NORMAL_SEEDS},
                   "recovery": recovery, "recovery_first_choice_composition": comp(first),
                   "recovery_seed_base": RECOVERY_SEED_BASE, "candidate_factor": CANDIDATE_FACTOR,
                   "n1_extra": "復帰に採った 90 の（種、色）で通常デモ（kind n）を生成する。どちらかが作り直しを含めて失敗したら"
                               "その候補を両方から外し、同じ枠の次の候補を使う",
                   "estimate_inputs": est})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--n", type=int, default=120)
    s.add_argument("--chunk", type=int, default=10)
    s.add_argument("--smoke", action="store_true")
    for name in ("check-gen", "check-eval"):
        s = sub.add_parser(name)
        s.add_argument("--workers", type=int, default=1)
        s.add_argument("--n", type=int, default=60)
        s.add_argument("--smoke", action="store_true")
    sub.add_parser("plan")
    a = ap.parse_args(argv)
    {"sweep": cmd_sweep, "check-gen": cmd_check_gen, "check-eval": cmd_check_eval, "plan": cmd_plan}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
