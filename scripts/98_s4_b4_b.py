"""段階 4 束 4 の解析（実装 B）。実装 A（src/recovla/eval/b4.py）と同じ結果の形を、別のコードで出す（二重集計の片方。
事前登録 v2 の草案 第 7 節 1、掲示板 0155 の 2-7）。

    .venv\\Scripts\\python.exe scripts\\98_s4_b4_b.py --layout <layout.json> --params <params.json> [--out b.json]
    （ふだんは scripts\\98_s4_b4_check.py check が A と B を両方呼んで照らす）

A と共有しないもの（独立の度合い）:
  - recovla も scipy も numpy も import しない（標準ライブラリだけ）。
  - ファイルの列挙は os.listdir と正規表現（A は pathlib の glob）。
  - 正確な二項検定は整数の組み合わせの和で、両側は「観測より偏った側の確率の 2 倍（上限 1）」（A は scipy の binomtest）。
  - Wilson の区間の z は statistics.NormalDist、Newcombe の区間（対あり方法 10・φ の補正あり、対なし方法 10）も自前の式
    （A は recovla.eval.stats）。
  - 入口の点検・帯の表・数え方も別に書いた（関門 2・3 の帯は事前登録 v2 の草案 12-2 の表から、テスト 2 は s4_gates.json を読む）。
同じなのは、読む記録のファイルと、s4_gates.json の帯、決まり（事前登録 v2 の草案・s4_gates.json の bundle4_gates・掲示板 0155・0165）だけ。
作者の判断（10/09）: 守りの P1 の差と関門 3 の R4s1001 − N4s1001 は同じ種で両方とも誘発が成立した対で数える。関門 2 の置き損ね（P3）は
欠けても未完にしない。ドライバが 610.88 と違っても止めず、警告として残す。
"""
import argparse
import json
import math
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
GATES_PATH = os.path.join(REPO, "configs", "s4_gates.json")
SIG = 0.05
T30, T45, T60 = 30.0, 45.0, 60.0
TINY = 1e-9
TRIAL_RE = re.compile(r"^trial_(\d{4})\.json$")
COLOR_LIST = ["red", "green", "blue"]
ENV_FIELDS = ["driver", "torch", "torch_cuda", "os_build"]
DEFAULTS = {"guard_mode": "point", "h2_in_family": True, "g3_with_r1v3s1001": True, "ckpt_sha256": {}, "p_fill": {}}
# 関門 2・3（事前登録 v2 の草案 12-2）。テスト 2 は s4_gates.json の test2_*
GATE_TABLE = {
    "G2": [("nat", 192100, 192165, ["R4", "R1v3"]), ("P1", 192200, 192299, ["R4", "N4", "R1v3"]),
           ("P2", 192300, 192399, ["R4", "R1v3"]), ("P3", 192600, 192699, ["R4", "N4", "R1v3"])],
    "G3": [("P1", 192400, 192499, ["R4s1001", "N4s1001"]), ("P2", 192500, 192599, ["R4s1001", "R1v3s1001"])],
}
TEST2_IDS = [("nat", "test2_natural", ["R4", "R1v3"]), ("P1", "test2_P1", ["R4", "N4", "R1v3"]),
             ("P2", "test2_drop", ["R4", "N4", "R1v3"]), ("P3", "test2_misplace", ["R4", "N4", "R1v3"])]
INDUCE_OF = {"nat": None, "P1": "P1", "P2": "P2", "P3": "P3"}
LIMIT_SINGLE = 60.0
DESCRIPTIVE_PART = {"G2": "P3"}          # 関門 2 の置き損ねは記述だけ（欠けても未完にしない。作者の判断 10/09）
DRIVER_WANT = "610.88"                  # 違っても止めない。1 枚に警告
GATES_REL = "configs/s4_gates.json"
GATES_POSTED = "7f2f651cafa9cf97b5548324d3fb8ea0bec5e891cca9c8859c7dd9c0347646e9"   # 掲示板 0165（改訂 4）


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_records(folder):
    names = sorted(n for n in os.listdir(folder) if TRIAL_RE.match(n))
    return [read_json(os.path.join(folder, n)) for n in names]


def lim_key(x):
    return str(int(x)) if float(x).is_integer() else repr(x)


# ================================================================ 統計（自前）
def exact_two_sided(b, c):
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    tail = sum(math.comb(n, i) for i in range(lo + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


Z95 = statistics.NormalDist().inv_cdf(0.975)


def wilson95(k, n):
    if n == 0:
        return None
    ph = k / n
    den = 1.0 + Z95 ** 2 / n
    mid = (ph + Z95 ** 2 / (2.0 * n)) / den
    half = Z95 * math.sqrt(ph * (1.0 - ph) / n + Z95 ** 2 / (4.0 * n * n)) / den
    return [max(0.0, mid - half), min(1.0, mid + half)]


def newcombe_pair(a11, a10, a01, a00):
    n = a11 + a10 + a01 + a00
    if n == 0:
        return None
    p1, p2 = (a11 + a10) / n, (a11 + a01) / n
    l1, u1 = wilson95(a11 + a10, n)
    l2, u2 = wilson95(a11 + a01, n)
    den = (a11 + a10) * (a01 + a00) * (a11 + a01) * (a10 + a00)
    if den == 0:
        phi = 0.0
    else:
        num = a11 * a00 - a10 * a01
        if num > n / 2.0:
            num = num - n / 2.0
        elif num >= 0:
            num = 0.0
        phi = num / math.sqrt(den)
    d = p1 - p2
    lower = d - math.sqrt(max(0.0, (p1 - l1) ** 2 - 2.0 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    upper = d + math.sqrt(max(0.0, (u1 - p1) ** 2 - 2.0 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return [lower, upper]


def newcombe_indep(k1, n1, k2, n2):
    if n1 == 0 or n2 == 0:
        return None
    w1, w2 = wilson95(k1, n1), wilson95(k2, n2)
    p1, p2 = k1 / n1, k2 / n2
    d = p1 - p2
    return {"diff": d, "ci95": [d - math.hypot(p1 - w1[0], w2[1] - p2), d + math.hypot(w1[1] - p1, p2 - w2[0])]}


def holm_b(pvals):
    names = list(pvals)
    names.sort(key=lambda k: (pvals[k], k))
    m = len(names)
    out, best, still = {}, 0.0, True
    for idx, k in enumerate(names):
        mult = m - idx
        best = max(best, min(1.0, mult * pvals[k]))
        if still and not (pvals[k] <= SIG / mult + 1e-15):
            still = False
        out[k] = {"rank": idx + 1, "p": pvals[k], "p_holm": best, "rejected_stepdown": still}
    return {"order": names, "m": m, "by": out}


# ================================================================ 計画と入口の点検
def fill(params):
    q = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    for k, v in (params or {}).items():
        if k not in DEFAULTS:
            raise ValueError(f"知らない params: {k}")
        q[k] = v
    if q["guard_mode"] not in ("point", "interval"):
        raise ValueError("guard_mode は point か interval")
    if type(q["h2_in_family"]) is not bool or type(q["g3_with_r1v3s1001"]) is not bool:
        raise ValueError("h2_in_family・g3_with_r1v3s1001 は真偽")
    if not isinstance(q["ckpt_sha256"], dict):
        raise ValueError("ckpt_sha256 は辞書")
    return q


def plan_of(phase):
    """{"<部分>.<モデル>": (部分, 下, 上)}。"""
    out = {}
    if phase in GATE_TABLE:
        for part, lo, hi, models in GATE_TABLE[phase]:
            for m in models:
                out[f"{part}.{m}"] = (part, lo, hi)
        return out
    if phase != "T2":
        raise ValueError(f"知らない段階 {phase}")
    al = {}
    for a in read_json(GATES_PATH)["bands"]["allocations"]:
        al[a["id"]] = a["range"]
    for part, aid, models in TEST2_IDS:
        lo, hi = al[aid]
        for m in models:
            out[f"{part}.{m}"] = (part, int(lo), int(hi))
    return out


def is_descriptive(phase, name):
    return DESCRIPTIVE_PART.get(phase) == name.split(".")[0]


def needed(phase, q):
    names = [n for n in plan_of(phase) if not is_descriptive(phase, n)]
    if phase == "G3" and not q["g3_with_r1v3s1001"]:
        names = [n for n in names if n != "P2.R1v3s1001"]
    return names


def inspect(phase, name, folder, model, experiment, q):
    part, lo, hi = plan_of(phase)[name]
    issues = []
    res = {"ok": False, "problems": issues, "git_heads": [], "ckpt_sha256": None, "env": None, "files_sha256": None, "n": 0}
    if not os.path.isdir(folder):
        issues.append(f"{name}: フォルダがない")
        return res
    rj = os.path.join(folder, "run.json")
    run = read_json(rj) if os.path.isfile(rj) else None
    if run is None:
        issues.append(f"{name}: run.json がない")
    ga = os.path.join(folder, "G_AUDIT.json")
    if not os.path.isfile(ga):
        issues.append(f"{name}: G_AUDIT.json がない")
    elif read_json(ga).get("met") is not True:
        issues.append(f"{name}: G_AUDIT の met が真でない")
    recs = list_records(folder)
    res["n"] = len(recs)
    keys = []
    for r in recs:
        keys.append((int(r["seed"]), r.get("target") if part == "nat" else "*"))
    if part == "nat":
        plan = set((s, c) for s in range(lo, hi + 1) for c in COLOR_LIST)
    else:
        plan = set((s, "*") for s in range(lo, hi + 1))
    if len(set(keys)) != len(keys):
        issues.append(f"{name}: 重なる記録")
    if set(keys) != plan:
        issues.append(f"{name}: 記録の集合が計画と違う")
    if any(not (lo <= s <= hi) for s, _ in keys):
        issues.append(f"{name}: 帯の外の種")
    folder_name = os.path.basename(os.path.normpath(folder))
    bad_mark = False
    for r in recs:
        mm = r.get("model")
        mname = mm.get("name") if isinstance(mm, dict) else mm
        b4 = r.get("b4") or {}
        rt = r.get("runtime") or {}
        if mname != model:
            issues.append(f"{name}: モデル名の印")
            bad_mark = True
        if r.get("experiment") != experiment or r.get("condition") != folder_name:
            issues.append(f"{name}: 実験名・条件名の印")
            bad_mark = True
        if b4.get("phase") != phase or b4.get("part") != part or b4.get("model") != model:
            issues.append(f"{name}: b4 の印")
            bad_mark = True
        if rt.get("mode") != "naive" or rt.get("exec_interval") != 6 or rt.get("safety_filter") is not False:
            issues.append(f"{name}: 実行のしかた")
            bad_mark = True
        if bad_mark:
            break
    kinds = set()
    for r in recs:
        ind = r.get("induce")
        kinds.add(ind.get("kind") if isinstance(ind, dict) else None)
    if recs and kinds != {INDUCE_OF[part]}:
        issues.append(f"{name}: 誘発の印")
    tl = set()
    for r in recs:
        tl.add(float(r.get("time_limit_s", -1)))
    if recs and tl != {LIMIT_SINGLE}:
        issues.append(f"{name}: 制限時間")
    if run is not None:
        t = (run.get("time_limits") or {}).get("time_limit_s")
        if t is None or float(t) != LIMIT_SINGLE:
            issues.append(f"{name}: run.json の制限時間")
        sg = run.get("env_segments")
        if not isinstance(sg, list) or len(sg) != 1:
            issues.append(f"{name}: env_segments")
    env_set = set()
    for r in recs:
        e = r.get("env")
        env_set.add(json.dumps({k: e.get(k) for k in ENV_FIELDS}, sort_keys=True) if isinstance(e, dict) else None)
    if None in env_set or len(env_set) > 1:
        issues.append(f"{name}: env")
    elif env_set:
        res["env"] = list(env_set)[0]
    heads = set()
    no_head = False
    for r in recs:
        h = (r.get("env") or {}).get("git_head")
        if not h:
            no_head = True
        heads.add(str(h))
    if no_head:
        issues.append(f"{name}: git_head")
    res["git_heads"] = sorted(heads)
    per_file = {}
    key_sets = set()
    lacking = False
    for r in recs:
        fs = (r.get("b4") or {}).get("files_sha256") or {}
        if not fs:
            lacking = True
        key_sets.add(tuple(sorted(fs)))
        for f, v in fs.items():
            per_file.setdefault(f, set()).add(v)
    if lacking:
        issues.append(f"{name}: ファイルの SHA-256 なし")
    elif [f for f, v in per_file.items() if len(v) != 1] or len(key_sets) > 1:
        issues.append(f"{name}: ファイルの SHA-256")
    elif recs:
        res["files_sha256"] = {f: list(per_file[f])[0] for f in sorted(per_file)}
        if res["files_sha256"].get(GATES_REL) != GATES_POSTED:
            issues.append(f"{name}: s4_gates.json の SHA-256 が掲示と違う")
    cks = set()
    for r in recs:
        cks.add(((r.get("b4") or {}).get("ckpt") or {}).get("sha256"))
    if len(recs) == 0 or len(cks) != 1 or None in cks:
        issues.append(f"{name}: 保存点の SHA-256")
    else:
        res["ckpt_sha256"] = list(cks)[0]
        posted = q["ckpt_sha256"].get(model)
        if posted and posted != res["ckpt_sha256"]:
            issues.append(f"{name}: 保存点の SHA-256 が掲示と違う")
    res["ok"] = len(issues) == 0
    return res


def across(checks, model_of):
    issues = []
    shas = {}
    for name in checks:
        shas.setdefault(model_of[name], []).append(checks[name]["ckpt_sha256"])
    models = {}
    for m in sorted(shas):
        u = set(shas[m])
        if len(u) == 1 and None not in u:
            models[m] = list(u)[0]
        else:
            models[m] = None
            issues.append(f"{m}: 保存点")
    envs = set(c["env"] for c in checks.values())
    drv = None
    if None in envs or len(envs) != 1:
        issues.append("環境")
    else:
        drv = json.loads(list(envs)[0])["driver"]
    # 使ったファイルの SHA-256 は段階の全条件で 1 種類
    files_all = [c["files_sha256"] for c in checks.values()]
    same_files = all(f is not None for f in files_all) and all(f == files_all[0] for f in files_all)
    if not same_files:
        issues.append("ファイル")
    return {"ok": len(issues) == 0, "problems": issues, "models": models, "env_kinds": len(envs),
            "files_sha256": dict(files_all[0]) if same_files and files_all else None,
            "driver": drv, "driver_expected": DRIVER_WANT, "driver_warning": drv != DRIVER_WANT}


# ================================================================ 数える（自前）
def established_before(r, lim):
    ind = r.get("induce")
    if not isinstance(ind, dict) or not ind.get("established"):
        return False
    t = ind.get("t_established")
    return t is not None and float(t) < lim - TINY


def success_within(r, lim):
    if not r.get("success"):
        return False
    t = r.get("t_success")
    return t is not None and float(t) <= lim + TINY


def arm_rate(folder, part, lim):
    k = n = 0
    for r in list_records(folder):
        if part != "nat" and not established_before(r, lim):
            continue
        n += 1
        if success_within(r, lim):
            k += 1
    return {"k": k, "n": n, "rate": (k / n) if n else None, "wilson95": wilson95(k, n)}


def paired_induced(fx, fy, lim):
    X, Y = {}, {}
    for r in list_records(fx):
        X[int(r["seed"])] = r
    for r in list_records(fy):
        Y[int(r["seed"])] = r
    b = c = both = neither = pairs = 0
    for s in sorted(X):
        if s not in Y:
            continue
        if not (established_before(X[s], lim) and established_before(Y[s], lim)):
            continue
        pairs += 1
        x, y = success_within(X[s], lim), success_within(Y[s], lim)
        if x and y:
            both += 1
        elif x:
            b += 1
        elif y:
            c += 1
        else:
            neither += 1
    return {"pairs": pairs, "b": b, "c": c, "both": both, "neither": neither, "p": exact_two_sided(b, c)}


def paired_natural(fx, fy, lim):
    X, Y = {}, {}
    for r in list_records(fx):
        X[(int(r["seed"]), r["target"])] = r
    for r in list_records(fy):
        Y[(int(r["seed"]), r["target"])] = r
    a11 = a10 = a01 = a00 = 0
    for k in X:
        if k not in Y:
            continue
        x, y = success_within(X[k], lim), success_within(Y[k], lim)
        a11 += x and y
        a10 += x and not y
        a01 += y and not x
        a00 += (not x) and (not y)
    n = a11 + a10 + a01 + a00
    return {"pairs": n, "x_k": a11 + a10, "y_k": a11 + a01, "x_only": a10, "y_only": a01,
            "diff": ((a10 - a01) / n) if n else None, "newcombe95": newcombe_pair(a11, a10, a01, a00)}


def guard_rate(fx, fy, part, lim, thr, mode="point"):
    x, y = arm_rate(fx, part, lim), arm_rate(fy, part, lim)
    nc = newcombe_indep(x["k"], x["n"], y["k"], y["n"])
    if nc is None:
        ok = False
    elif mode == "point":
        ok = nc["diff"] >= thr - TINY
    else:
        ok = nc["ci95"][0] >= thr - TINY
    return {"x": x, "y": y, "diff": None if nc is None else nc["diff"], "newcombe95": None if nc is None else nc["ci95"],
            "threshold": thr, "mode": mode, "pass": bool(ok)}


def guard_pair(fx, fy, lim, thr, mode="point"):
    """守りの P1（作者の判断 10/09）: 同じ種で両方の腕とも誘発が lim より前に成立した対の差 (b − c) / 対の数。"""
    t = paired_induced(fx, fy, lim)
    n = t["pairs"]
    ci = newcombe_pair(t["both"], t["b"], t["c"], t["neither"])
    d = None if n == 0 else (t["b"] - t["c"]) / n
    if d is None:
        ok = False
    elif mode == "point":
        ok = d >= thr - TINY
    else:
        ok = ci[0] >= thr - TINY
    return {"pairs": n, "b": t["b"], "c": t["c"], "both": t["both"], "neither": t["neither"],
            "x_k": t["both"] + t["b"], "y_k": t["both"] + t["c"], "diff": d, "newcombe95": ci,
            "threshold": thr, "mode": mode, "pass": bool(ok)}


def guard_nat(fx, fy, lim, thr, mode="point"):
    g = paired_natural(fx, fy, lim)
    if mode == "point":
        ok = g["diff"] is not None and g["diff"] >= thr - TINY
    else:
        ok = g["newcombe95"] is not None and g["newcombe95"][0] >= thr - TINY
    g["threshold"] = thr
    g["mode"] = mode
    g["pass"] = bool(ok)
    return g


def rate_table(D):
    out = {}
    for name in sorted(D):
        part = name.split(".")[0]
        out[name] = {lim_key(L): arm_rate(D[name], part, L) for L in (T30, T45, T60)}
    return out


# ================================================================ 段階ごと
def b_gate2(D):
    g = {"natural": guard_nat(D["nat.R4"], D["nat.R1v3"], T30, -0.05),
         "p1_r4_vs_r1v3": guard_pair(D["P1.R4"], D["P1.R1v3"], T30, -0.10),
         "p1_r4_minus_n4": guard_pair(D["P1.R4"], D["P1.N4"], T30, 0.20)}
    gp = g["natural"]["pass"] and g["p1_r4_vs_r1v3"]["pass"] and g["p1_r4_minus_n4"]["pass"]
    aim = guard_rate(D["P2.R4"], D["P2.R1v3"], "P2", T30, 0.15)
    aim["pairs"] = paired_induced(D["P2.R4"], D["P2.R1v3"], T30)
    rates3, used3, skip3 = {}, [], []
    for m in ["R4", "N4", "R1v3"]:
        if "P3." + m not in D:
            skip3.append(m)
            continue
        used3.append(m)
        rates3[m] = {lim_key(L): arm_rate(D["P3." + m], "P3", L) for L in (T30, T60)}
    if gp and aim["pass"]:
        nxt = "test2"
    elif not gp:
        nxt = "research_guard"
    else:
        nxt = "no_test2"
    return {"guards": g, "guards_pass": bool(gp), "aim": aim,
            "p3": {"label": "記述だけ（関門 2 の判定に使わない。掲示板 0165）", "used": used3, "not_used": skip3, "rates": rates3},
            "rates": rate_table(D),
            "decision": {"pass": bool(gp and aim["pass"]), "next": nxt}}


def b_gate3(D, q):
    rn = guard_pair(D["P1.R4s1001"], D["P1.N4s1001"], T30, 0.20)
    dr = None
    if q["g3_with_r1v3s1001"]:
        dr = guard_rate(D["P2.R4s1001"], D["P2.R1v3s1001"], "P2", T30, 0.0)
        dr["pass"] = bool(dr["diff"] is not None and dr["diff"] > TINY)
        dr["threshold"] = "> 0"
    ok = rn["pass"] and (dr is None or dr["pass"])
    return {"direction": dr, "direction_used": dr is not None, "p1_r4_minus_n4": rn, "rates": rate_table(D),
            "decision": {"pass": bool(ok), "wording": "two_seeds" if ok else "one_seed_diagnostic"}}


def b_test2(D, q):
    h1 = paired_induced(D["P2.R4"], D["P2.R1v3"], T30)
    h2 = paired_induced(D["P2.R4"], D["P2.N4"], T30)
    prim = {"H1": h1, "H2": h2}
    family = {"H1": h1["p"]}
    if q["h2_in_family"]:
        family["H2"] = h2["p"]
    hb = holm_b(family)
    for k in ("H1", "H2"):
        v = prim[k]
        v["in_family"] = k in family
        if k in family:
            v["p_holm"] = hb["by"][k]["p_holm"]
            v["established"] = bool(v["p_holm"] < SIG and v["b"] > v["c"])
        else:
            v["p_holm"] = None
            v["established"] = None
    mode = q["guard_mode"]
    guards = {"natural": guard_nat(D["nat.R4"], D["nat.R1v3"], T30, -0.05, mode),
              "p1_r4_vs_r1v3": guard_pair(D["P1.R4"], D["P1.R1v3"], T30, -0.10, mode),
              "p1_r4_minus_n4": guard_pair(D["P1.R4"], D["P1.N4"], T30, 0.20, mode)}
    p3 = {"label": "副次・族の外（Holm の補正をしない。基準を置かない）"}
    for L in (T30, T60):
        p3[lim_key(L)] = {"R4_vs_R1v3": paired_induced(D["P3.R4"], D["P3.R1v3"], L),
                          "R4_vs_N4": paired_induced(D["P3.R4"], D["P3.N4"], L),
                          "rates": {m: arm_rate(D["P3." + m], "P3", L) for m in ["R4", "N4", "R1v3"]}}
    sixty = {"label": "副次・族の外", "H1_form": paired_induced(D["P2.R4"], D["P2.R1v3"], T60),
             "H2_form": paired_induced(D["P2.R4"], D["P2.N4"], T60)}
    gp = all(v["pass"] for v in guards.values())
    return {"primary": prim, "holm": hb, "secondary": {"guards": guards, "guards_pass": bool(gp), "p3": p3, "sixty": sixty,
                                                       "rates": rate_table(D)}}


def run_b(layout, params):
    q = fill(params)
    phase = layout["phase"]
    plan = plan_of(phase)
    base = os.path.join(layout["root"], layout["experiment"])
    D, model_of = {}, {}
    for name, v in (layout.get("conds") or {}).items():
        if name not in plan:
            raise ValueError(f"計画に無い条件 {name}")
        if name.split(".", 1)[1] != v["model"]:
            raise ValueError(f"{name}: モデル名が条件名と違う")
        D[name] = os.path.join(base, v["cond"])
        model_of[name] = v["model"]
    want = needed(phase, q)
    desc = [n for n in plan if is_descriptive(phase, n)]
    checks, core = {}, {}
    for name in want:
        if name in D:
            core[name] = checks[name] = inspect(phase, name, D[name], model_of[name], layout["experiment"], q)
    for name in desc:
        if name in D and os.path.isdir(D[name]):
            checks[name] = inspect(phase, name, D[name], model_of[name], layout["experiment"], q)
    missing = [n for n in want if n not in D]
    if core:
        cr = across(core, model_of)
    else:
        cr = {"ok": False, "problems": ["条件なし"], "models": {}, "env_kinds": 0, "files_sha256": None,
              "driver": None, "driver_expected": DRIVER_WANT, "driver_warning": True}
    done = bool(core) and all(c["ok"] for c in core.values()) and not missing and cr["ok"]
    # 記述だけの条件は、点検と条件をまたぐ点検（判定に使う条件と合わせて）を満たすものだけ使う
    opt = {"used": [], "missing": [], "excluded": []}
    for name in desc:
        if name not in checks:                  # 計画に無いか、フォルダが無い
            opt["missing"].append(name)
            continue
        trial = dict(core)
        trial[name] = checks[name]
        good = checks[name]["ok"] and bool(core) and across(trial, model_of)["ok"]
        opt["used" if good else "excluded"].append(name)
    out = {"schema": "recovery_vla.s4_b4_result/1", "phase": phase, "status": "complete" if done else "incomplete", "params": q,
           "checks": checks, "missing_conditions": missing, "optional": opt, "cross": cr, "result": None}
    if not done:
        return out
    Du = {n: D[n] for n in want + opt["used"]}
    if phase == "G2":
        out["result"] = b_gate2(Du)
    elif phase == "G3":
        out["result"] = b_gate3(Du, q)
    else:
        out["result"] = b_test2(Du, q)
    return out


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                    # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--layout", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = run_b(read_json(a.layout), read_json(a.params))
    text = json.dumps(res, ensure_ascii=False, indent=1)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)
    return 0 if res["status"] == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
