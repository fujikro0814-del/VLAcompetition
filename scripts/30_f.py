"""Step F の段取り（手順書 §7）。結果は outputs/f/<項目>.json。

    .venv\\Scripts\\python.exe scripts\\30_f.py sweep [--workers 8] [--n 120] [--smoke]   # 注入の値の振り（掲示板 0033・0034）

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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--n", type=int, default=120)
    s.add_argument("--chunk", type=int, default=10)
    s.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    {"sweep": cmd_sweep}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
