"""段階 4 の関門 1 の二重集計の B（独立の集計）。試行の生の記録から指標を数え直し、関門の規則も自分で当てはめる。

    .venv\\Scripts\\python.exe scripts\\98_s4_gate1_b.py tally [--out outputs\\s4\\gate1_result_b.json]
    .venv\\Scripts\\python.exe scripts\\98_s4_gate1_b.py compare [--a-input outputs\\s4\\gate1_input.json] ^
        [--a-result outputs\\s4\\gate1_result.json] [--b-result outputs\\s4\\gate1_result_b.json] [--out outputs\\s4\\gate1_compare.json]

独立の守り（掲示板 0155 の 2-7、目標書 3-6）:
  - recovla.diag・recovla.eval（gate1・time_scoring）・98_s4_d_* を読み込まない。ファイルの列挙・記録の読み方・中央値・検定は自前。
    読み込むのは numpy・scipy と、定数の recovla.sim.frames（CUBE_REST_Z）だけ。
  - 定義は目標書 8-1・8-2・8-3・8-4、s4_gates.json の metrics、掲示板 0154・0155、docs/stage4/bundle1_defs/ の掲示の文から書いた。
  - 規則の数値は目標書 8-2 の表から写した定数（下の RULES）。s4_gates.json の SHA-256（改行 LF）が掲示板 0162 の値と同じときだけ使う。
  - 判定できない条件の扱い: 値がない・分母が 0 は「判定できない」（None）。all_of は 1 つでも偽なら偽、そうでなく 1 つでも None なら None。
書くもの: tally は --out の JSON（指標・判定・結論・E7 の本数の提案・列挙の点検）。compare は A と B の照らし合わせ（--out）。
終了コード: 0 済み / 1 列挙の点検または照らし合わせで食い違い / 2 読めない。
"""
import argparse
import hashlib
import json
import math
import pathlib
import re
import sys
import time
from fractions import Fraction

import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recovla.sim.frames import CUBE_REST_Z  # noqa: E402  定数だけ

V2 = ROOT / "outputs" / "v2eval"
S4 = ROOT / "outputs" / "s4"
GATES = ROOT / "configs" / "s4_gates.json"
POSTED_LF_SHA = "dcf0dd4c1b2a4f906936bf478f012d3aa6e9a51a81aeb12c6cdcbcf35f33362d"   # 掲示板 0162
COLORS = ("red", "green", "blue")
HS = (10, 20, 30, 40)

# 目標書 8-2・8-3・8-4 の表の数値（写し）
RULES = {
    "R": {"c1_mm": Fraction(10), "c2": Fraction("0.95"), "c3": 4, "c4": Fraction("0.019"), "top": 2, "b1_mm": Fraction(15),
          "candidates": ["paper_formula_range44_cap5", "range40_cap5", "range10_cap5", "ZEROS"], "base": "naive"},
    "T": {"c1": Fraction("0.60"), "margin_min": Fraction("0.25"), "margin_add": Fraction("0.10"), "b1": Fraction("0.60"),
          "b2": Fraction("0.20"), "tie": "EH"},
    "S": {"high": Fraction("0.40"), "diff": Fraction("0.25"), "p": 0.05},
    "XPL": {"rep": Fraction("0.40"), "both": Fraction("0.25"), "each": Fraction("0.15"), "repeats": 2},
    "C": {"c1": Fraction("0.20"), "c2": Fraction("0.10"), "c3": 20, "L": 30.0, "variant": "fall_with_hold"},
    "K": {"n": 99, "warn": Fraction("0.364")},
}
LIFT_M = 0.02          # 持ち上がり: 目標が CUBE_REST_Z + 2 cm を超えた
PLUS_Y_M = 0.01        # +y のずれ: 指先 − 目標の y > 1 cm
NEED_MIN_M = 0.02      # 移動の比: 必要な移動の |u 成分| がこれ以下の試行は除く
DIR_MIN_M = 0.03       # 影・X2: 向きを決める距離の下限
DT_ACT = 0.1           # 行動は 10 Hz


# ------------------------------------------------------------------ 読み込みと列挙（自前）
def rd(p):
    return json.loads(pathlib.Path(p).read_text(encoding="utf-8-sig"))


def listing(d: pathlib.Path, prefix: str) -> list:
    """d の直下の <prefix>_NNNN.json を番号順に（_incomplete_* などの下の階層は見ない）。"""
    rx = re.compile(rf"^{prefix}_(\d{{4}})\.json$")
    out = []
    for p in d.iterdir():
        m = rx.match(p.name)
        if p.is_file() and m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def color_of(target) -> int:
    return COLORS.index(str(target).split(">")[0])


def med(xs):
    xs = [float(x) for x in xs if x is not None and math.isfinite(x)]
    return (float(np.median(xs)) if xs else None), len(xs)


def first_true(b):
    i = np.flatnonzero(b)
    return int(i[0]) if i.size else None


def rises(gc) -> np.ndarray:
    """gripper_closed の 0→1 のこま（こま 0 で閉じていればこま 0 も数える）。"""
    g = np.asarray(gc).astype(np.int8)
    return np.flatnonzero(np.diff(np.r_[np.int8(0), g]) == 1)


def lifted(z, f: int, ci: int) -> bool:
    """閉じたこま f から次に開くまでに、目標の高さが CUBE_REST_Z + 2 cm を超えたか。"""
    gc = np.asarray(z["gripper_closed"]).astype(bool)
    after = np.flatnonzero(~gc[f + 1:])
    f2 = f + 1 + int(after[0]) if after.size else len(gc)
    return bool(np.any(z["cube_pos"][f:f2, ci, 2] - CUBE_REST_Z > LIFT_M))


def plus_y(z, f: int, ci: int) -> bool:
    return bool(z["fingertip"][f, 1] - z["cube_pos"][f, ci, 1] > PLUS_Y_M)


def succ_at(m, L: float) -> bool:
    ts = m.get("t_success")
    return bool(m.get("success")) and ts is not None and float(ts) <= L


def est_before(m, L: float) -> bool:
    """誘発が L 秒より前に成立（掲示板 0155 の 1: t_established < L）。"""
    ind = m.get("induce") or {}
    te = ind.get("t_established")
    return bool(ind.get("established")) and te is not None and float(te) < L


def kn(k, n):
    return {"k": int(k), "n": int(n)}


# ------------------------------------------------------------------ 列挙の点検（帯と本数）
def band(gates, aid):
    a = next(x for x in gates["bands"]["allocations"] if x["id"] == aid)
    return list(range(a["range"][0], a["range"][1] + 1))


def check_seeds(name, seeds, expect_seeds, per_seed, problems, n_expect=None):
    from collections import Counter
    c = Counter(seeds)
    want = {s: per_seed for s in expect_seeds}
    ok = dict(c) == want
    if not ok:
        problems.append(f"{name}: 種の集合・本数が帯と違う（{len(seeds)} 試行、種 {len(c)} 個）")
    if n_expect is not None and len(seeds) != n_expect:
        problems.append(f"{name}: 本数 {len(seeds)} が {n_expect} でない")
    return {"n": len(seeds), "seeds": f"{min(seeds) if seeds else None}〜{max(seeds) if seeds else None}", "per_seed": per_seed, "ok": ok}


# ------------------------------------------------------------------ 関門 R
def rtc_trial(m, z, rt):
    ci = color_of(m["target"])
    t = z["sim_time"]
    gc = np.asarray(z["gripper_closed"]).astype(bool)
    tip, cube = z["fingertip"], z["cube_pos"][:, ci]
    acts = rt["actions"]
    A = np.array([a["a"] for a in acts], float)
    ta = np.array([a["t"] for a in acts], float)
    held = np.array([bool(a["held"]) for a in acts])
    ch = np.array([-1 if a["chunk"] is None else int(a["chunk"]) for a in acts])
    r = {"seed": m["seed"], "target": m["target"], "s30": succ_at(m, 30.0), "s60": succ_at(m, 60.0)}
    # 継ぎ目の跳び（行動の時刻 < 30 s の継ぎ目だけ）
    v = A[:, :3] / DT_ACT
    jump = np.linalg.norm(np.diff(v, axis=0), axis=1)
    seam = (~held[1:] & ~held[:-1]) & (ch[1:] != ch[:-1]) & (ta[1:] < 30.0) & (ta[:-1] < 30.0)
    r["seam"] = float(jump[seam].mean()) if seam.any() else None
    k1 = first_true(gc)
    r["t_close"] = float(t[k1]) if k1 is not None else None
    r["radial"] = r["ratio"] = None
    if k1 is not None:
        u = cube[k1, :2] / np.linalg.norm(cube[k1, :2])
        r["radial"] = float((tip[k1, :2] - cube[k1, :2]) @ u * 1e3)
        if t[k1] <= 30.0:
            use = (ta < t[k1]) & ~held
            cmd = A[use, :2].sum(axis=0)
            need = float((cube[k1, :2] - tip[0, :2]) @ u)
            r["ratio"] = float(cmd @ u / need) if abs(need) > NEED_MIN_M else None
    return r


def shadow_rows(m, z, rt, dz):
    """影の推論: (影 − 誘導) の行 0〜h−1 の水平の和を u に射影 [mm]。"""
    ci = color_of(m["target"])
    t = z["sim_time"]
    gc = np.asarray(z["gripper_closed"]).astype(bool)
    k1 = first_true(gc)
    tclose = float(t[k1]) if k1 is not None else math.inf
    inf = {int(e["i"]): e for e in rt["inference"]}
    out = []
    for j, ii in enumerate(dz["inf_index"]):
        if not bool(dz["has_shadow"][j]):
            continue
        tobs = float(inf[int(ii)]["t_obs"])
        if not (tobs < tclose and tobs <= 30.0):
            continue
        f = int(np.searchsorted(t, tobs + 1e-9, side="right") - 1)
        d = z["cube_pos"][f, ci, :2] - z["x_des"][f, :2]
        if np.linalg.norm(d) < DIR_MIN_M:
            continue
        u = d / np.linalg.norm(d)
        sh, gd = dz["shadow_post"][j].astype(float), dz["guided_post"][j].astype(float)
        out.append({h: float((sh[:h, :2].sum(axis=0) - gd[:h, :2].sum(axis=0)) @ u * 1e3) for h in HS})
    return out


def x2_rows(gen_dir: pathlib.Path, ev_path: pathlib.Path):
    name = ev_path.stem
    seed = int(name.split("_")[1])
    gdir = gen_dir / f"part_{seed}" / name
    meta = rd(gdir / "meta.json")
    g = np.load(gdir / "data.npz")
    ev = np.load(ev_path)
    ci = color_of(meta.get("target") or meta["color"])
    f1 = first_true(np.asarray(g["gripper_closed"]).astype(bool))
    xd, cube = g["x_des"], g["cube_pos"][:, ci]
    out = []
    for i, k in enumerate(ev["k_index"]):
        k = int(k)
        f = 2 * k
        d = cube[f, :2] - xd[f, :2]
        if np.linalg.norm(d) < DIR_MIN_M:
            continue
        u = d / np.linalg.norm(d)
        row = {}
        for h in HS:
            if f1 is None or k + h > f1 // 2 or 2 * (k + h) >= len(xd):
                continue
            e = xd[2 * (k + h), :2] - xd[f, :2]
            p = ev["pred_post"][i, :h, :2].astype(float).sum(axis=0)
            row[h] = float((e - p) @ u * 1e3)
        if row:
            out.append(row)
    return out, {"episode": name, "f1_own": f1, "f_close_eval": int(ev["f_close"])}


def tally_R(gates, problems):
    base = V2 / "S4DRTC"
    settings = {}
    for name in ["current_repro", "naive"] + RULES["R"]["candidates"]:
        d = base / name
        rows = []
        for n, p in listing(d, "trial"):
            m = rd(p)
            if (m.get("diag") or {}).get("setting", {}).get("name") not in (name, None):
                problems.append(f"R {name} trial_{n:04d}: diag の設定名がフォルダ名と違う")
            rows.append(rtc_trial(m, np.load(d / f"trial_{n:04d}.npz"), rd(d / f"runtime_{n:04d}.json")["runtime"]))
        chk = check_seeds(f"R {name}", [r["seed"] for r in rows], band(gates, "D_RTC"), 3, problems, 30)
        rg, nr = med(r["radial"] for r in rows)
        mr, nm = med(r["ratio"] for r in rows)
        sj, ns = med(r["seam"] for r in rows)
        settings[name] = {"n_trials": len(rows), "radial_gap_mm": rg, "n_radial": nr, "move_ratio": mr, "n_move_ratio": nm,
                          "seam_jump_mps": sj, "n_seam": ns, "natural_success_30": sum(r["s30"] for r in rows),
                          "natural_success_60": sum(r["s60"] for r in rows),
                          "first_close_after_30s": sum(1 for r in rows if r["t_close"] is not None and r["t_close"] > 30.0),
                          "n_no_close": sum(1 for r in rows if r["t_close"] is None), "listing": chk}
    # 影の推論
    d = base / "shadow_current_repro"
    sh, seeds = [], []
    for n, p in listing(d, "trial"):
        m = rd(p)
        seeds.append(m["seed"])
        sh += shadow_rows(m, np.load(d / f"trial_{n:04d}.npz"), rd(d / f"runtime_{n:04d}.json")["runtime"], np.load(d / f"diag_{n:04d}.npz"))
    chk_sh = check_seeds("R shadow", seeds, band(gates, "D_RTC_shadow"), 3, problems, 12)
    shadow = {str(h): med(r[h] for r in sh)[0] for h in HS}
    shadow_n = {str(h): med(r[h] for r in sh)[1] for h in HS}
    # X2
    ev_dir, gen_dir = S4 / "x2" / "eval_X2_gen_R1v3", S4 / "x2" / "gen" / "X2_gen"
    xr, eps = [], []
    for p in sorted(ev_dir.glob("n_*_r[0-9].npz")):
        rows, info = x2_rows(gen_dir, p)
        xr += rows
        eps.append(info)
        if info["f1_own"] != info["f_close_eval"]:
            problems.append(f"X2 {info['episode']}: 最初の閉じのこまが eval の記録と違う（{info['f1_own']} と {info['f_close_eval']}）")
    if len(eps) != 20:
        problems.append(f"X2: エピソードが {len(eps)} 本（20 本のはず）")
    x2 = {str(h): med(r.get(h) for r in xr)[0] for h in HS}
    x2_n = {str(h): med(r.get(h) for r in xr)[1] for h in HS}
    return {"settings": settings, "shadow_plan_shorter_mm": shadow, "shadow_n": shadow_n, "shadow_listing": chk_sh,
            "x2_shortfall_mm": x2, "x2_n": x2_n, "x2_episodes": len(eps)}


# ------------------------------------------------------------------ 関門 T（E7）
def e7_run(m, z):
    t = z["sim_time"]
    rs = rises(z["gripper_closed"])
    steps = m.get("steps") or []

    def first_attempt(step):
        att = step.get("attempts") or []
        if not att:
            return None
        t0 = float(att[0]["t_begin"])
        t1 = float(att[1]["t_begin"]) if len(att) > 1 else float(step.get("t_end") if step.get("t_end") is not None else m["t_end"])
        fs = [int(f) for f in rs if t0 <= t[f] < t1]
        if not fs:
            return {"closed": False}
        f = fs[0]
        ci = COLORS.index(step["color"])
        return {"closed": True, "lift": lifted(z, f, ci), "plus_y": plus_y(z, f, ci)}

    return {"seed": m["seed"], "all3": bool(m.get("all_three_in_box")), "timed_out": bool(m.get("timed_out")),
            "s0": first_attempt(steps[0]) if len(steps) > 0 else None, "s0_color": steps[0]["color"] if steps else None,
            "s1": first_attempt(steps[1]) if len(steps) > 1 else None, "s1_started": len(steps) > 1}


def tally_T(gates, problems):
    base = V2 / "S4DE7"
    arms, raw = {}, {}
    for arm in ("E0_run1", "E0_run2", "EH", "ES", "EO"):
        d = base / arm
        rows = []
        for n, p in listing(d, "run"):
            m = rd(p)
            want = arm.split("_")[0]
            if (m.get("diag") or {}).get("arm") != want:
                problems.append(f"T {arm} run_{n:04d}: diag の腕が {want} でない")
            rows.append(e7_run(m, np.load(d / f"run_{n:04d}.npz")))
        chk = check_seeds(f"T {arm}", [r["seed"] for r in rows], band(gates, "D_E7"), 1, problems, 40)
        st = [r for r in rows if r["s1_started"]]
        cl = [r for r in st if r["s1"] and r["s1"].get("closed")]
        arms[arm] = {"first_close_lift": kn(sum(1 for r in cl if r["s1"]["lift"]), len(st)),
                     "plus_y_shift": kn(sum(1 for r in cl if r["s1"]["plus_y"]), len(cl)),
                     "all_three_true": kn(sum(r["all3"] for r in rows), len(rows)),
                     "timed_out": sum(r["timed_out"] for r in rows), "listing": chk}
        raw[arm] = rows
    eo = raw["EO"]
    s0 = [r for r in eo if r["s0"] is not None]
    eo_green = {"first_close_lift": kn(sum(1 for r in s0 if r["s0"].get("closed") and r["s0"]["lift"]), len(s0)),
                "colors": sorted({r["s0_color"] for r in eo if r["s0_color"]})}
    # K の E7 版（並べるだけ）: 同じ種の対で 3 個とも（真値）の食い違いと phi
    a = {r["seed"]: r["all3"] for r in raw["E0_run1"]}
    b = {r["seed"]: r["all3"] for r in raw["E0_run2"]}
    keys = sorted(set(a) & set(b))
    x = np.array([a[k] for k in keys], float)
    y = np.array([b[k] for k in keys], float)
    phi = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None
    k_e7 = {"pairs": len(keys), "all_three_discordant": int(np.sum(x != y)), "rho_hat_phi": phi}
    return {"arms": arms, "EO_green_first": eo_green, "K_e7": k_e7}


# ------------------------------------------------------------------ 関門 S と移植の腕
def single_trial(m, z):
    ci = color_of(m["target"])
    rs = rises(z["gripper_closed"])
    t = z["sim_time"]
    r = {"seed": m["seed"], "closed0": bool(np.asarray(z["gripper_closed"])[0]), "close": False}
    if rs.size:
        f = int(rs[0])
        r.update(close=True, t_close=float(t[f]), plus_y=plus_y(z, f, ci), lift=lifted(z, f, ci))
    return r


def single_summary(rows):
    cl = [r for r in rows if r["close"]]
    return {"n": len(rows), "plus_y_shift": kn(sum(r["plus_y"] for r in cl), len(cl)),
            "first_close_lift": kn(sum(r["lift"] for r in cl), len(cl)),
            "first_close_after_30s": sum(1 for r in cl if r["t_close"] > 30.0),
            "closed_at_frame0": sum(r["closed0"] for r in rows)}


def tally_S(gates, problems):
    conds, xpl = {}, {}
    base = V2 / "S4DSTART"
    for d in sorted(x for x in base.iterdir() if x.is_dir()):
        rows, keys = [], set()
        for n, p in listing(d, "trial"):
            m = rd(p)
            dg = m.get("diag") or {}
            keys.add((dg.get("diag"), dg.get("start"), dg.get("prior")))
            rows.append(single_trial(m, np.load(d / f"trial_{n:04d}.npz")))
        if not rows:
            continue
        if len(keys) != 1 or next(iter(keys))[0] != "D-single-start":
            problems.append(f"S {d.name}: diag の鍵が 1 つでない {sorted(map(str, keys))}")
            continue
        _, st, pr = next(iter(keys))
        if d.name != f"{st}_{pr}":
            problems.append(f"S {d.name}: フォルダ名と diag（{st}・{pr}）が違う")
        key = f"{st}|{pr}"
        if key in conds:
            problems.append(f"S: 同じ鍵 {key} のフォルダが 2 つ")
        conds[key] = dict(single_summary(rows), listing=check_seeds(f"S {key}", [r["seed"] for r in rows],
                                                                    band(gates, "D_single_start"), 3, problems, 33))
    base = V2 / "S4XPL"
    for d in sorted(x for x in base.iterdir() if x.is_dir()):
        rows, keys = [], set()
        for n, p in listing(d, "trial"):
            m = rd(p)
            dg = m.get("diag") or {}
            keys.add((dg.get("diag"), dg.get("arm"), dg.get("rep")))
            if m.get("target") != "green":
                problems.append(f"XPL {d.name} trial_{n:04d}: 目標が緑でない")
            rows.append(single_trial(m, np.load(d / f"trial_{n:04d}.npz")))
        if not rows:
            continue
        if len(keys) != 1 or next(iter(keys))[0] != "XPL":
            problems.append(f"XPL {d.name}: diag の鍵が 1 つでない")
            continue
        _, arm, rep = next(iter(keys))
        if d.name != f"{arm.replace('XPL_', '')}_r{rep}":
            problems.append(f"XPL {d.name}: フォルダ名と diag（{arm}・{rep}）が違う")
        xpl.setdefault(arm, {})[f"rep{rep}"] = dict(single_summary(rows), listing=check_seeds(
            f"XPL {arm} rep{rep}", [r["seed"] for r in rows], band(gates, "D_transplant"), 1, problems, 20))
    return {"conditions": conds}, {"arms": xpl}


# ------------------------------------------------------------------ 関門 C
def tally_C(gates, problems):
    base = V2 / "S4DREC"
    out = {}
    for variant in ("fall_with_hold", "fall_as_is", "misplace", "grasp_failure"):
        recs = {}
        for model in ("R1v3", "N1v3"):
            d = base / f"{model}_{variant}"
            rs = {}
            for n, p in listing(d, "trial"):
                m = rd(p)
                dg = m.get("diag") or {}
                if dg.get("variant") != variant or (m.get("model") or {}).get("name") != model:
                    problems.append(f"C {d.name} trial_{n:04d}: diag の版・モデルがフォルダ名と違う")
                key = (m["seed"], m["target"])
                if key in rs:
                    problems.append(f"C {d.name}: 同じ種と色が 2 回 {key}")
                rs[key] = m
            check_seeds(f"C {d.name}", [k[0] for k in rs], band(gates, "D_recovery"), 1, problems, 50)
            recs[model] = rs
        by_L = {}
        for L in (30.0, 60.0):
            ent = {}
            for model, lab in (("R1v3", "recovery_R"), ("N1v3", "recovery_N")):
                est = [m for m in recs[model].values() if est_before(m, L)]
                ent[lab] = kn(sum(succ_at(m, L) for m in est), len(est))
            keys = sorted(k for k in set(recs["R1v3"]) & set(recs["N1v3"])
                          if est_before(recs["R1v3"][k], L) and est_before(recs["N1v3"][k], L))
            ro = sum(1 for k in keys if succ_at(recs["R1v3"][k], L) and not succ_at(recs["N1v3"][k], L))
            no = sum(1 for k in keys if succ_at(recs["N1v3"][k], L) and not succ_at(recs["R1v3"][k], L))
            ent["paired"] = {"pairs": len(keys), "r_only": ro, "n_only": no}
            ent["established_exactly_L"] = sum(1 for model in recs for m in recs[model].values()
                                               if (m.get("induce") or {}).get("t_established") is not None
                                               and float(m["induce"]["t_established"]) == L)
            by_L[f"{L:g}"] = ent
        out[variant] = {"by_L": by_L}
    return out


# ------------------------------------------------------------------ 関門 K
def tally_K(gates, problems):
    base = V2 / "S4K"
    runs = {}
    for r in ("K1", "K2"):
        runs[r] = {n: rd(p) for n, p in listing(base / r, "trial")}
        check_seeds(f"K {r}", [m["seed"] for m in runs[r].values()], band(gates, "K"), 3, problems, 99)
    a, b = runs["K1"], runs["K2"]
    same = set(a) == set(b) and all((a[n]["seed"], a[n]["target"], json.dumps(a[n].get("layout"), sort_keys=True)) ==
                                    (b[n]["seed"], b[n]["target"], json.dumps(b[n].get("layout"), sort_keys=True)) for n in a)
    done = {r: sum(1 for m in runs[r].values() if m.get("t_end") is not None and m.get("success") is not None) for r in runs}
    keys = sorted(set(a) & set(b))
    d0 = {}
    for L in (30.0, 60.0):
        d0[f"{L:g}"] = kn(sum(succ_at(a[n], L) != succ_at(b[n], L) for n in keys), len(keys))
    return {"run1_completed": done["K1"], "run2_completed": done["K2"], "pairs_matched": bool(same), "d0": d0,
            "success_30": [sum(succ_at(a[n], 30.0) for n in keys), sum(succ_at(b[n], 30.0) for n in keys)]}


# ------------------------------------------------------------------ 規則の当てはめ（自前）
def fr(x) -> Fraction:
    return Fraction(x["k"], x["n"]) if isinstance(x, dict) else Fraction(repr(float(x)))


def all3(vals):
    vals = list(vals)
    if any(v is False for v in vals):
        return False
    return None if any(v is None for v in vals) else True


def fisher_greater(ka, na, kb, nb) -> float:
    """片側（a の割合 > b）のフィッシャーの正確検定: 超幾何分布の上側。"""
    return float(stats.hypergeom.sf(ka - 1, na + nb, ka + kb, na))


def rule_R(R):
    rr, S = RULES["R"], R["settings"]
    nv = S[rr["base"]]
    out, passed = {}, []
    for name in rr["candidates"]:
        s = S[name]
        c = {"R.c1": fr(s["radial_gap_mm"]) >= fr(nv["radial_gap_mm"]) - rr["c1_mm"],
             "R.c2": fr(s["move_ratio"]) >= rr["c2"],
             "R.c3": (s["natural_success_30"] >= nv["natural_success_30"] - rr["c3"]) if s["n_trials"] == nv["n_trials"] else None,
             "R.c4": fr(s["seam_jump_mps"]) <= rr["c4"]}
        ok = all3(c.values())
        out[name] = {"conditions": c, "pass": ok}
        if ok:
            passed.append(name)
    rank = sorted(passed, key=lambda n: (-S[n]["natural_success_30"], S[n]["seam_jump_mps"], abs(S[n]["radial_gap_mm"]),
                                         rr["candidates"].index(n)))
    b1 = {"shadow_any_ge": any(v is not None and fr(v) >= rr["b1_mm"] for v in R["shadow_plan_shorter_mm"].values()),
          "x2_any_ge": any(v is not None and fr(v) >= rr["b1_mm"] for v in R["x2_shortfall_mm"].values())}
    b1["applies"] = b1["shadow_any_ge"] and b1["x2_any_ge"]
    return {"settings": out, "passed": passed, "ranking": rank, "top": rank[:rr["top"]],
            "branch": "continue" if passed else "stop", "R.b2": len(passed) == 1, "R.b1": b1}


def rule_T(T):
    tr, A = RULES["T"], T["arms"]
    l1, l2 = fr(A["E0_run1"]["first_close_lift"]), fr(A["E0_run2"]["first_close_lift"])
    ref, d = max(l1, l2), abs(l1 - l2)
    margin = max(tr["margin_min"], d + tr["margin_add"])
    e0_all3 = max(A["E0_run1"]["all_three_true"]["k"], A["E0_run2"]["all_three_true"]["k"])
    arms = {}
    for arm in ("EH", "ES"):
        a = A[arm]
        lift = fr(a["first_close_lift"])
        same_n = len({a["all_three_true"]["n"], A["E0_run1"]["all_three_true"]["n"], A["E0_run2"]["all_three_true"]["n"]}) == 1
        c = {"T.c1": lift >= tr["c1"], "T.c2": lift >= ref + margin,
             "T.c3": (a["all_three_true"]["k"] >= e0_all3) if same_n else None}
        arms[arm] = {"conditions": c, "pass": all3(c.values()), "lift": lift}
    passed = [k for k in ("EH", "ES") if arms[k]["pass"]]
    if passed:
        res = "pass"
        ret = passed[0] if len(passed) == 1 else (tr["tie"] if arms["EH"]["lift"] == arms["ES"]["lift"]
                                                  else max(passed, key=lambda k: arms[k]["lift"]))
    elif all(arms[k]["pass"] is False for k in ("EH", "ES")):
        res, ret = "fail", None
    else:
        res, ret = "undetermined", None
    eo = T["EO_green_first"]
    b1 = (fr(eo["first_close_lift"]) <= tr["b1"]) if eo["colors"] == ["green"] else None
    return {"E0_ref_lift": float(ref), "d_E0": float(d), "required_margin": float(margin), "E0_all_three_max": e0_all3,
            "arms": {k: {"conditions": v["conditions"], "pass": v["pass"], "lift": float(v["lift"])} for k, v in arms.items()},
            "result": res, "return_to": ret, "T.b1": b1, "T.b2": d > tr["b2"]}


def rule_S(S):
    sr, C = RULES["S"], S["conditions"]
    out = {}
    for cid, a, b in (("S.pose", "standby_end|on_grid", "standby_start|on_grid"),
                      ("S.prior", "standby_end|wall_side", "standby_end|on_grid")):
        pa, pb = C[a]["plus_y_shift"], C[b]["plus_y_shift"]
        p = fisher_greater(pa["k"], pa["n"], pb["k"], pb["n"])
        c = {"high": max(fr(pa), fr(pb)) >= sr["high"], "diff": fr(pa) - fr(pb) >= sr["diff"], "p": p < sr["p"]}
        out[cid] = {"a": a, "b": b, "a_kn": pa, "b_kn": pb, "diff": float(fr(pa) - fr(pb)), "p_one_sided": p,
                    "conditions": c, "effect": all3(c.values())}
    return out


def rule_XPL(X):
    xr, A = RULES["XPL"], X["arms"]

    def pooled(arm):
        ks = sum(A[arm][f"rep{i}"]["plus_y_shift"]["k"] for i in range(1, xr["repeats"] + 1))
        ns = sum(A[arm][f"rep{i}"]["plus_y_shift"]["n"] for i in range(1, xr["repeats"] + 1))
        return Fraction(ks, ns), kn(ks, ns)

    as_v, as_kn = pooled("XPL_as")
    rep = as_v >= xr["rep"]
    out = {"XPL.rep": {"value": float(as_v), "k_n": as_kn, "satisfied": rep}}
    for cid, other in (("XPL.pose", "XPL_home"), ("XPL.prior", "XPL_grid")):
        ov, okn = pooled(other)
        each = [fr(A["XPL_as"][f"rep{i}"]["plus_y_shift"]) - fr(A[other][f"rep{i}"]["plus_y_shift"])
                for i in range(1, xr["repeats"] + 1)]
        c = {"rep": rep, "both": as_v - ov >= xr["both"], **{f"rep{i + 1}": e >= xr["each"] for i, e in enumerate(each)}}
        out[cid] = {"drop_both": float(as_v - ov), "drop_each": [float(e) for e in each], "other_k_n": okn,
                    "conditions": c, "satisfied": all3(c.values())}
    return out


def rule_C(Cin):
    cr = RULES["C"]
    e = Cin[cr["variant"]]["by_L"][f"{cr['L']:g}"]
    p = e["paired"]
    c = {"C.c1": fr(e["recovery_R"]) >= cr["c1"],
         "C.c2": (Fraction(p["r_only"] - p["n_only"], p["pairs"]) >= cr["c2"]) if p["pairs"] else None,
         "C.c3": p["pairs"] >= cr["c3"]}
    return {"conditions": c, "pass": all3(c.values()), "R30": float(fr(e["recovery_R"])),
            "R_minus_N_30": (p["r_only"] - p["n_only"]) / p["pairs"] if p["pairs"] else None, "pairs": p["pairs"]}


def rule_K(K):
    kr = RULES["K"]
    c1 = K["run1_completed"] == kr["n"] and K["run2_completed"] == kr["n"] and K["pairs_matched"]
    over = {L: fr(K["d0"][L]) > kr["warn"] for L in ("30", "60")}
    return {"K.c1": c1, "d0": {L: float(fr(K["d0"][L])) for L in ("30", "60")}, "over": over,
            "branch": ("K.b2" if any(over.values()) else "K.b1") if c1 else "stop", "warning": bool(c1 and any(over.values()))}


def rule_candidates(S, X, T):
    rows = [(1, "internal_state", None if X["XPL.rep"]["satisfied"] is None else not X["XPL.rep"]["satisfied"]),
            (2, "prior_cube", all3([S["S.prior"]["effect"], X["XPL.prior"]["satisfied"]])),
            (3, "pose", all3([S["S.pose"]["effect"], X["XPL.pose"]["satisfied"]])),
            (4, "none", True)]
    trace = []
    for order, verdict, hit in rows:
        trace.append({"row": order, "hit": hit})
        if hit is None:
            return {"row": None, "verdict": None, "trace": trace, "ex_arm": None, "start_candidate": None}
        if hit:
            break
    out = {"row": order, "verdict": verdict, "trace": trace, "start_candidate": verdict == "prior_cube"}
    if verdict == "internal_state":
        out["ex_arm"] = True
    elif verdict == "pose":
        out["ex_arm"] = {"pass": False, "fail": True}.get(T["result"])
        out["executor_suffices"] = {"pass": True, "fail": False}.get(T["result"])
    else:
        out["ex_arm"] = False
    return out


def conclude(R, T, C, K, cand):
    start, slip = cand["start_candidate"], C["pass"]
    if start:
        b4 = "start_state_prior_cube"
    elif start is False and slip:
        b4 = "slip"
    elif start is False and slip is False:
        b4 = "none"
    else:
        b4 = None
    research = []
    if start and slip:
        research.append("滑りのデータ（開始状態を優先）")
    if cand["verdict"] in ("internal_state", "none"):
        research.append("開始状態のデータ")
    if cand["verdict"] == "pose" and cand.get("executor_suffices"):
        research.append("姿勢のデータ（実行器で足りる）")
    if slip is False:
        research.append("落下と置き損ねの改善（C のやめる枝）")
    if R["R.b1"]["applies"]:
        research.append("学習時の RTC（R.b1）")
    if T["T.b1"]:
        research.append("弱い層のデータ（T.b1）")
    if b4 == "none":
        research.append("束 4（候補なし）")
    warnings = []
    if K["warning"]:
        warnings.append("K.b2")
    if T["T.b2"]:
        warnings.append("T.b2")
    return {"bundle2": {"rtc_settings": R["top"] if R["branch"] == "continue" else [], "b2": R["branch"] == "continue",
                        "v3a_return_to": T["return_to"] if T["result"] == "pass" else None, "v3a_include": T["result"] == "pass"},
            "bundle4": b4, "ex_arm": cand["ex_arm"], "research_path": research, "warnings": warnings}


# ------------------------------------------------------------------ E7 の本数の提案（e7_sample_size_rule）
def mcnemar_power(n: int, p0: float, p1: float, rho: float, alpha: float = 0.05) -> float:
    """対の二値（周辺 p0・p1、phi 相関 rho）で、McNemar の正確検定（両側）の検出力。"""
    p11 = p0 * p1 + rho * math.sqrt(p0 * (1 - p0) * p1 * (1 - p1))
    p10, p01 = p1 - p11, p0 - p11          # 新しい腕だけ成功・今の腕だけ成功
    if min(p10, p01) < 0:
        raise ValueError("rho が周辺の割合と両立しない")
    pd = p10 + p01
    q = p10 / pd
    power = 0.0
    for D in range(0, n + 1):
        wD = stats.binom.pmf(D, n, pd)
        if wD < 1e-15 or D == 0:
            continue
        b = np.arange(D + 1)
        pv = np.minimum(1.0, 2 * np.minimum(stats.binom.cdf(b, D, 0.5), stats.binom.sf(b - 1, D, 0.5)))
        power += wD * float(np.sum(stats.binom.pmf(b, D, q)[pv <= alpha]))
    return power


def e7_proposal(T):
    A = T["arms"]
    p0 = (A["E0_run1"]["all_three_true"]["k"] + A["E0_run2"]["all_three_true"]["k"]) / \
         (A["E0_run1"]["all_three_true"]["n"] + A["E0_run2"]["all_three_true"]["n"])
    rho = T["K_e7"]["rho_hat_phi"]
    p1 = p0 + 0.15
    pw = {n: round(mcnemar_power(n, p0, p1, rho), 4) for n in (100, 150, 200)}
    pick = next((n for n in (100, 150, 200) if pw[n] >= 0.79), 200)
    sens = {f"rho={r}": {n: round(mcnemar_power(n, p0, p1, r), 4) for n in (100, 150, 200)} for r in (0.0, 0.2)}
    return {"p0": p0, "p1": p1, "rho_hat": rho, "alpha": 0.05, "test": "McNemar の正確検定（両側）",
            "power": pw, "proposal_n": pick, "rule": "検出力 >= 0.79 になる最小の本数を {100, 150, 200} から。上限 200",
            "sensitivity_rho": sens,
            "model": "対の二値の同時分布 p11 = p0·p1 + rho·sqrt(p0(1−p0)p1(1−p1))（rho は E0 の 2 回の 3 個とも（真値）の phi）"}


# ------------------------------------------------------------------ 本体
def jsonable(x):
    if isinstance(x, Fraction):
        return float(x)
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def cmd_tally(a) -> int:
    raw = GATES.read_bytes()
    lf = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    if lf != POSTED_LF_SHA:
        print(f"s4_gates.json の SHA-256 が掲示板 0162 の値と違う（{lf}）。規則の写しを使えない", file=sys.stderr)
        return 2
    gates = json.loads(raw.decode("utf-8"))
    problems = []
    t0 = time.time()
    R = tally_R(gates, problems)
    T = tally_T(gates, problems)
    S, X = tally_S(gates, problems)
    C = tally_C(gates, problems)
    K = tally_K(gates, problems)
    rR, rT, rS, rX, rC, rK = rule_R(R), rule_T(T), rule_S(S), rule_XPL(X), rule_C(C), rule_K(K)
    cand = rule_candidates(rS, rX, rT)
    res = {"schema": "recovery_vla.s4_gate1_result_b/1", "written": time.strftime("%Y-%m-%d %H:%M:%S"),
           "by": "scripts/98_s4_gate1_b.py（二重集計の B。recovla.diag・recovla.eval・98_s4_d_* を読み込まない）",
           "gates_sha256_lf": lf, "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
           "definitions": {
               "radial_gap_mm": "最初に gripper_closed が真のこまの (指先 − 目標)·u、u = 目標の xy / |xy|。閉じた試行の中央値（時間によらない）",
               "move_ratio": "最初の閉じが 30 s 以内の試行。閉じる前（行動の時刻 < 閉じの時刻）の保持でない行動の xy の和·u ÷ (目標(閉じ) − 指先(0))·u。|分母| <= 20 mm は除く。中央値",
               "seam_jump_mps": "v = a[:3]/0.1、続く 2 行が保持でなく塊の番号が変わり、両方の時刻 < 30 s の |Δv| の試行ごとの平均 → 中央値",
               "natural_success_30": "success かつ t_success <= 30",
               "shadow": "影がある推論で t_obs < 最初の閉じかつ t_obs <= 30、t_obs のこま（sim_time <= t_obs の最後）で |目標 − x_des| >= 30 mm。(影 − 誘導) の行 0〜h−1 の xy の和·u [mm] の中央値",
               "x2": "10 fps のこま k、k + h <= f1 // 2（f1 はエキスパートの最初の閉じのこま）、|目標 − x_des|(2k) >= 30 mm。(x_des[2(k+h)] − x_des[2k] − 予測の行 0〜h−1 の和)·u [mm] の中央値",
               "e7_first_close_lift": "2 番目の手順の 1 回目の試み [t_begin, 次の試み or 手順の終わり) の最初の 0→1 から次に開くまでに目標の z − CUBE_REST_Z > 0.02。分母は 2 番目の手順が始まった試行",
               "plus_y_shift": "最初の閉じのこまで 指先 y − 目標 y > 0.01。分母は最初の閉じが起きた試行",
               "single_first_close": "試行全体で最初の 0→1",
               "recovery": "分母は induce.established かつ t_established < L、成功は success かつ t_success <= L。対は (種, 色)"},
           "metrics": {"R": R, "T": T, "S": S, "XPL": X, "C": C, "K": K},
           "verdicts": {"R": rR, "T": rT, "S": rS, "XPL": rX, "C": rC, "K": rK, "candidate_selection": cand},
           "conclusion": conclude(rR, rT, rC, rK, cand),
           "e7_sample_size": e7_proposal(T),
           "listing_problems": problems, "elapsed_s": round(time.time() - t0, 1)}
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jsonable(res), ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(jsonable(res["conclusion"]), ensure_ascii=False, indent=1))
    for p in problems:
        print(f"[gate1_b] 列挙の点検: {p}", file=sys.stderr)
    print(f"[gate1_b] 書いた: {out}（{res['elapsed_s']} s）")
    return 1 if problems else 0


# ------------------------------------------------------------------ A と B の照らし合わせ
MM_TOL = 0.5                 # 目標書 3-6: 中央値は ±0.5 mm
OTHER_TOL = {"move_ratio": 0.005, "seam_jump_mps": 0.0005}   # mm でない中央値の目安（B の決め。判定の一致を別に求める）


def research_keys(items) -> list:
    """研究の道へ回すものを、括弧の前の語で並べる（A と B で添え書きの文が違うため）。"""
    return sorted(str(x).split("（")[0].strip() for x in items)


def compare(ain, ares, b) -> dict:
    rows = []

    def add(item, va, vb, kind):
        if kind == "count":
            ok = va == vb
        elif kind == "mm":
            ok = va is not None and vb is not None and abs(va - vb) <= MM_TOL
        elif kind in OTHER_TOL:
            ok = va is not None and vb is not None and abs(va - vb) <= OTHER_TOL[kind]
        else:
            ok = va == vb
        rows.append({"item": item, "A": va, "B": vb, "kind": kind, "agree": bool(ok),
                     "abs_diff": (abs(va - vb) if isinstance(va, (int, float)) and isinstance(vb, (int, float)) else None)})

    BM = b["metrics"]
    for name, sa in ain["R"]["settings"].items():
        sb = BM["R"]["settings"][name]
        add(f"R.{name}.n_trials", sa["n_trials"], sb["n_trials"], "count")
        add(f"R.{name}.natural_success_30", sa["natural_success_30"], sb["natural_success_30"], "count")
        add(f"R.{name}.first_close_after_30s", sa["first_close_after_30s"], sb["first_close_after_30s"], "count")
        add(f"R.{name}.n_no_close", sa.get("n_no_close"), sb["n_no_close"], "count")
        add(f"R.{name}.radial_gap_mm", sa["radial_gap_mm"], sb["radial_gap_mm"], "mm")
        add(f"R.{name}.move_ratio", sa["move_ratio"], sb["move_ratio"], "move_ratio")
        add(f"R.{name}.seam_jump_mps", sa["seam_jump_mps"], sb["seam_jump_mps"], "seam_jump_mps")
    for h in ("10", "20", "30", "40"):
        add(f"R.shadow_plan_shorter_mm.{h}", ain["R"]["shadow_plan_shorter_mm"][h], BM["R"]["shadow_plan_shorter_mm"][h], "mm")
        add(f"R.x2_shortfall_mm.{h}", ain["R"]["x2_shortfall_mm"][h], BM["R"]["x2_shortfall_mm"][h], "mm")
    for arm, xa in ain["T"]["arms"].items():
        xb = BM["T"]["arms"][arm]
        for k in ("first_close_lift", "all_three_true"):
            add(f"T.{arm}.{k}", [xa[k]["k"], xa[k]["n"]], [xb[k]["k"], xb[k]["n"]], "count")
    eo_a, eo_b = ain["T"]["EO_green_first"], BM["T"]["EO_green_first"]
    add("T.EO_green_first.first_close_lift", [eo_a["first_close_lift"]["k"], eo_a["first_close_lift"]["n"]],
        [eo_b["first_close_lift"]["k"], eo_b["first_close_lift"]["n"]], "count")
    add("T.EO_green_first.colors", eo_a["colors"], eo_b["colors"], "count")
    ke = ain["T"].get("K_e7") or {}
    add("T.K_e7.all_three_discordant", (ke.get("all_three_discordance") or {}).get("k"), BM["T"]["K_e7"]["all_three_discordant"], "count")
    if ke.get("rho_hat_phi") is not None:
        add("T.K_e7.rho_hat_phi（4 桁）", round(ke["rho_hat_phi"], 4), round(BM["T"]["K_e7"]["rho_hat_phi"], 4), "count")
    for key, ca in ain["S"]["conditions"].items():
        cb = BM["S"]["conditions"][key]
        add(f"S.{key}.plus_y_shift", [ca["plus_y_shift"]["k"], ca["plus_y_shift"]["n"]], [cb["plus_y_shift"]["k"], cb["plus_y_shift"]["n"]], "count")
        add(f"S.{key}.first_close_after_30s", ca.get("first_close_after_30s"), cb["first_close_after_30s"], "count")
    for arm, ra in ain["XPL"]["arms"].items():
        for rep, va in ra.items():
            vb = BM["XPL"]["arms"][arm][rep]["plus_y_shift"]
            add(f"XPL.{arm}.{rep}", [va["k"], va["n"]], [vb["k"], vb["n"]], "count")
    for variant, va in ain["C"].items():
        for L, ea in va["by_L"].items():
            eb = BM["C"][variant]["by_L"].get(L)
            if eb is None:
                continue
            for lab in ("recovery_R", "recovery_N"):
                add(f"C.{variant}.{L}.{lab}", [ea[lab]["k"], ea[lab]["n"]], [eb[lab]["k"], eb[lab]["n"]], "count")
            add(f"C.{variant}.{L}.paired", ea["paired"], eb["paired"], "count")
    for k in ("run1_completed", "run2_completed", "pairs_matched"):
        add(f"K.{k}", ain["K"][k], BM["K"][k], "count")
    for L in ("30", "60"):
        add(f"K.d0.{L}", [ain["K"]["d0"][L]["k"], ain["K"]["d0"][L]["n"]], [BM["K"]["d0"][L]["k"], BM["K"]["d0"][L]["n"]], "count")

    # 判定
    V, G, ca, cb = b["verdicts"], ares["gates"], ares["conclusion"], b["conclusion"]
    pas = lambda r: {"pass": True, "fail": False}.get(r)        # noqa: E731
    ver = []

    def vadd(item, va, vb):
        ver.append({"item": item, "A": va, "B": vb, "agree": va == vb})

    for name in RULES["R"]["candidates"]:
        vadd(f"R.{name}", pas(G["R"]["settings"][name]["result"]), V["R"]["settings"][name]["pass"])
        for rec in G["R"]["settings"][name]["conditions"]:
            vadd(f"R.{name}.{rec['id']}", pas(rec["result"]), V["R"]["settings"][name]["conditions"][rec["id"]])
    vadd("R.passed", sorted(G["R"]["passed"]), sorted(V["R"]["passed"]))
    vadd("R.top", G["R"]["top"], V["R"]["top"])
    vadd("R.branch", G["R"]["branch"], V["R"]["branch"])
    vadd("R.b1", G["R"]["branches"]["R.b1"].get("applies"), V["R"]["R.b1"]["applies"])
    vadd("R.b2", bool(G["R"]["branches"].get("R.b2", {}).get("applies", False)), V["R"]["R.b2"])
    for arm in ("EH", "ES"):
        vadd(f"T.{arm}", pas(G["T"]["arms"][arm]["result"]), V["T"]["arms"][arm]["pass"])
        for rec in G["T"]["arms"][arm]["conditions"]:
            vadd(f"T.{arm}.{rec['id']}", pas(rec["result"]), V["T"]["arms"][arm]["conditions"][rec["id"]])
    vadd("T.result", G["T"]["result"], V["T"]["result"])
    vadd("T.return_to", G["T"]["return_to"], V["T"]["return_to"])
    vadd("T.b1", G["T"]["branches"]["T.b1"]["applies"], V["T"]["T.b1"])
    vadd("T.b2", G["T"]["branches"]["T.b2"]["applies"], V["T"]["T.b2"])
    for cid in ("S.pose", "S.prior"):
        vadd(f"{cid}.effect", G["S"]["contrasts"][cid]["effect"], V["S"][cid]["effect"])
        pa = next(c for c in G["S"]["contrasts"][cid]["conditions"] if c["id"].endswith(".p"))["value"]
        add(f"{cid}.p（片側の正確検定）", round(pa, 6), round(V["S"][cid]["p_one_sided"], 6), "count")
    for cid in ("XPL.rep", "XPL.pose", "XPL.prior"):
        vadd(f"{cid}", G["XPL"][cid]["satisfied"], V["XPL"][cid]["satisfied"])
    vadd("cand.row", ares["candidate_selection"]["row"], V["candidate_selection"]["row"])
    vadd("cand.verdict", ares["candidate_selection"]["verdict"], V["candidate_selection"]["verdict"])
    vadd("C.result", pas(G["C"]["result"]), V["C"]["pass"])
    for rec in G["C"]["conditions"]:
        vadd(f"C.{rec['id']}", pas(rec["result"]), V["C"]["conditions"][rec["id"]])
    vadd("K.c1", G["K"]["K.c1"]["satisfied"], V["K"]["K.c1"])
    vadd("K.warning", G["K"]["warning"], V["K"]["warning"])
    vadd("結論: 束 2 の RTC の設定", ca["bundle2"]["rtc"]["settings"], cb["bundle2"]["rtc_settings"])
    vadd("結論: 束 2 の v3 (a) を入れるか", ca["bundle2"]["v3a"]["include"], cb["bundle2"]["v3a_include"])
    vadd("結論: v3 (a) の戻し先", ca["bundle2"]["v3a"]["return_to"], cb["bundle2"]["v3a_return_to"])
    vadd("結論: 束 4 の候補", ca["bundle4"]["candidate"], cb["bundle4"])
    vadd("結論: EX の腕", ca["ex_arm"], cb["ex_arm"])
    vadd("結論: 研究の道へ（括弧の前の語で）", research_keys(ca["research_path"]), research_keys(cb["research_path"]))
    vadd("結論: 警告の数", len(ca["warnings"]), len(cb["warnings"]))
    bad_m = [r for r in rows if not r["agree"]]
    bad_v = [r for r in ver if not r["agree"]]
    mm = [r["abs_diff"] for r in rows if r["kind"] == "mm" and r["abs_diff"] is not None]
    return {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "tolerance": {"count": "完全一致", "mm": MM_TOL, **OTHER_TOL},
            "metrics": rows, "verdicts": ver, "n_metrics": len(rows), "n_metrics_disagree": len(bad_m),
            "n_verdicts": len(ver), "n_verdicts_disagree": len(bad_v), "max_mm_diff": max(mm) if mm else None,
            "agree": not bad_m and not bad_v}


def cmd_compare(a) -> int:
    res = compare(rd(a.a_input), rd(a.a_result), rd(a.b_result))
    out = pathlib.Path(a.out)
    out.write_text(json.dumps(jsonable(res), ensure_ascii=False, indent=1), encoding="utf-8")
    for r in res["metrics"] + res["verdicts"]:
        if not r["agree"]:
            print(f"[gate1_b] 食い違い {r['item']}: A {r['A']} / B {r['B']}", file=sys.stderr)
    print(f"[gate1_b] 指標 {res['n_metrics']} 項目（食い違い {res['n_metrics_disagree']}）、判定 {res['n_verdicts']} 項目"
          f"（食い違い {res['n_verdicts_disagree']}）、mm の最大の差 {res['max_mm_diff']}。書いた: {out}")
    return 0 if res["agree"] else 1


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:                            # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("tally", help="生の記録から数え直して判定する")
    p.add_argument("--out", default=str(S4 / "gate1_result_b.json"))
    p = sub.add_parser("compare", help="A（gate1.py）と B を照らす")
    p.add_argument("--a-input", default=str(S4 / "gate1_input.json"))
    p.add_argument("--a-result", default=str(S4 / "gate1_result.json"))
    p.add_argument("--b-result", default=str(S4 / "gate1_result_b.json"))
    p.add_argument("--out", default=str(S4 / "gate1_compare.json"))
    a = ap.parse_args(argv)
    try:
        return {"tally": cmd_tally, "compare": cmd_compare}[a.cmd](a)
    except (OSError, KeyError, ValueError) as e:
        print(f"読めない: {type(e).__name__}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
