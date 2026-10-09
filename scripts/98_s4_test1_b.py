"""段階 4 テスト 1 の解析（実装 B）。実装 A（src/recovla/eval/test1.py）と同じ結果の形を、別のコードで出す（二重集計の片方。
事前登録の案 第 7 節 1、掲示板 0155 の 2-7）。

    .venv\\Scripts\\python.exe scripts\\98_s4_test1_b.py --layout <layout.json> --params <params.json> [--out b.json]
    （ふだんは scripts\\98_s4_test1.py check が A と B を両方呼んで照らす）

A と共有しないもの（独立の度合い）:
  - recovla も scipy も numpy も import しない（標準ライブラリだけ）。
  - ファイルの列挙は os.listdir と正規表現（A は pathlib の glob）。
  - 正確な二項検定は整数の組み合わせの和で、両側は「観測より偏った側の確率の 2 倍（上限 1）」（事前登録の案 第 2 節の定義そのもの。
    A は scipy の binomtest）。
  - Wilson の区間の z は statistics.NormalDist、Newcombe の区間（対あり方法 10・φ の補正あり、対なし方法 10）も自前の式。
  - 介入・success@k・時間（E7）は scripts/56_intervention_s3.py の定義を、ここで書き下した（A は 56 の count_run を呼ぶ）。
  - 中央値は並べ替えて真ん中を取る（A は numpy.median）。
  - 入口の点検も別に書いた。版の照合 (b)（登録版 第 7 節 1、判断の紙の問 5）と案 B の G-P1 の同じ種の対（問 7）も、A と別に書いた。
同じなのは、読む記録のファイルと、s4_gates.json の帯、決まり（事前登録の案）だけ。
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
LIMIT_30, LIMIT_45, LIMIT_60 = 30.0, 45.0, 60.0
TINY = 1e-9
TRIAL_RE = re.compile(r"^trial_(\d{4})\.json$")
RUN_RE = re.compile(r"^run_(\d{4})\.json$")
COLOR_LIST = ["red", "green", "blue"]
ENV_FIELDS = ["driver", "torch", "torch_cuda", "os_build"]
MUST = ["h1", "rtc_arm"]                     # P-1・P-3（既定を置かない）
GATES_POSTED = "7f2f651cafa9cf97b5548324d3fb8ea0bec5e891cca9c8859c7dd9c0347646e9"   # 掲示板 0165（LF）
REL_GATES = "configs/s4_gates.json"
REL_ENTRY = "scripts/98_s4_test1_eval.py"
REL_EXEC_V3 = "src/recovla/runtime/executor_v3.py"
V3_NAME = "v3.1"
CKPT_RUNS = {"R1v3": "outputs/train/train_R1v3_20261005-180404_20261005-180404",
             "N1v3": "outputs/train/train_N1v3_20261005-202158_20261005-202158",
             "R1v3s1001": "outputs/s4/train/train_R1v3s1001_20261009-002457_20261009-002457",
             "N1v3s1001": "outputs/s4/train/train_N1v3s1001_20261009-024259_20261009-024259",
             "R1v3s1002": "outputs/s4/train/train_R1v3s1002_20261009-050112_20261009-050112",
             "N1v3s1002": "outputs/s4/train/train_N1v3s1002_20261009-071920_20261009-071920"}
CKPT_TAIL = "/checkpoints/020000/pretrained_model"
VERSION_FIELDS = ["entry_sha256", "executor_v3_sha256", "rtc_setting", "rtc_module_sha256", "ckpt_sha256"]
DEFAULTS = {"e7_n": None, "plan": "A", "guard_mode": "point", "ni_margin": 0.10, "h2_layers": ["1001", "1002"],
            "rtc_p1_n": 50, "e7_band_extended": False, "c4_on_time": None, "p_fill": {}, "versions": None}
KNOWN_RETURNS = {"placed": "scripted_return", "retry": "retry", "replan": "replan"}


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_records(folder, single):
    rx = TRIAL_RE if single else RUN_RE
    names = sorted(n for n in os.listdir(folder) if rx.match(n))
    return [read_json(os.path.join(folder, n)) for n in names]


# ================================================================ 統計（自前）
def exact_two_sided(b, c):
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    tail = sum(math.comb(n, i) for i in range(lo + 1))
    p = 2.0 * tail / (2 ** n)
    return min(1.0, p)


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
    return {"diff": d, "ci95": [lower, upper]}


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


def middle(values):
    v = sorted(values)
    if not v:
        return None
    h = len(v) // 2
    return float(v[h]) if len(v) % 2 == 1 else (v[h - 1] + v[h]) / 2.0


# ================================================================ 条件と入口の点検
def fill(params):
    q = {k: (list(v) if isinstance(v, list) else (dict(v) if isinstance(v, dict) else v)) for k, v in DEFAULTS.items()}
    params = params or {}
    for k in MUST:
        if k not in params:
            raise ValueError(f"params に {k} が要る")
    for k, v in params.items():
        if k not in q and k not in MUST:
            raise ValueError(f"知らない params: {k}")
        q[k] = v
    if type(q["h1"]) is not bool or type(q["rtc_arm"]) is not bool:
        raise ValueError("h1・rtc_arm の値")
    if q["plan"] not in ("A", "B") or q["guard_mode"] not in ("point", "interval"):
        raise ValueError("plan・guard_mode の値")
    if q["plan"] == "B" and q["rtc_arm"] is not True:
        raise ValueError("案 B なら rtc_arm が要る")
    want_rtc_n = {"A": 50, "B": 100}[q["plan"]]
    if int(q["rtc_p1_n"]) != want_rtc_n:
        raise ValueError("rtc_p1_n の値")
    if q["h1"] and q["e7_n"] not in (100, 150, 200):
        raise ValueError("h1 なら e7_n は 100・150・200")
    if q["versions"] is not None:
        if type(q["versions"]) is not dict or [k for k in q["versions"] if k not in VERSION_FIELDS]:
            raise ValueError("versions の鍵")
        if type(q["versions"].get("ckpt_sha256") or {}) is not dict:
            raise ValueError("versions.ckpt_sha256 は辞書")
    return q


def needed(q):
    out = []
    layers = ["1000"]
    layers.extend(q["h2_layers"])
    for s in layers:
        for kind in ("p1", "natural"):
            out.append(f"{kind}.{s}.R")
            out.append(f"{kind}.{s}.N")
    if q["h1"]:
        out.extend(["e7.v3", "e7.cur", "e7.n1v3_v3"])
    if q["rtc_arm"]:
        out.extend(["rtc.natural", "rtc.p1"])
    if q["plan"] == "B":
        out.append("rtc.p1_n")
    return out


def name_model(x):
    if isinstance(x, dict):
        return x["cond"], x.get("model")
    return x, None


def plan_conditions(layout, q):
    base = os.path.join(layout["root"], layout["experiment"])
    found = {}
    e7 = layout.get("e7") or {}
    if q["h1"]:
        for arm in ("v3", "cur", "n1v3_v3"):
            if arm in e7:
                cond, model = name_model(e7[arm])
                found["e7." + arm] = {"kind": "task", "role": "e7", "dir": os.path.join(base, cond), "model": model,
                                      "keys": {(160000 + i, None) for i in range(int(q["e7_n"]))}, "v3": arm != "cur"}
    for layer in (layout.get("p1") or {}):
        for side in ("R", "N"):
            cond, model = name_model(layout["p1"][layer][side])
            found[f"p1.{layer}.{side}"] = {"kind": "single", "role": "p1", "dir": os.path.join(base, cond), "model": model,
                                           "keys": {(162000 + i, "*") for i in range(100)}, "induce": "P1", "rtc_arm": False}
    for layer in (layout.get("natural") or {}):
        last = 161065 if layer == "1000" else 161032
        for side in ("R", "N"):
            cond, model = name_model(layout["natural"][layer][side])
            found[f"natural.{layer}.{side}"] = {"kind": "single", "role": "natural", "dir": os.path.join(base, cond), "model": model,
                                                "keys": {(s, c) for s in range(161000, last + 1) for c in COLOR_LIST}, "induce": None,
                                                "rtc_arm": False}
    rtc = layout.get("rtc") or {}
    if "natural" in rtc:
        cond, model = name_model(rtc["natural"])
        found["rtc.natural"] = {"kind": "single", "role": "natural", "dir": os.path.join(base, cond), "model": model,
                                "keys": {(s, c) for s in range(161000, 161066) for c in COLOR_LIST}, "induce": None, "rtc_arm": True}
    if "p1" in rtc:
        cond, model = name_model(rtc["p1"])
        found["rtc.p1"] = {"kind": "single", "role": "p1", "dir": os.path.join(base, cond), "model": model,
                           "keys": {(162000 + i, "*") for i in range(int(q["rtc_p1_n"]))}, "induce": "P1", "rtc_arm": True}
    if "p1_n" in rtc:
        cond, model = name_model(rtc["p1_n"])
        found["rtc.p1_n"] = {"kind": "single", "role": "p1", "dir": os.path.join(base, cond), "model": model,
                             "keys": {(162000 + i, "*") for i in range(100)}, "induce": "P1", "rtc_arm": True}
    for spec in found.values():
        spec["experiment"] = layout["experiment"]
    return found


def allocations():
    out = {}
    for a in read_json(GATES_PATH)["bands"]["allocations"]:
        out[a["id"]] = (a["range"][0], a["range"][1])
    return out


def inspect(name, spec, q):
    why = []
    d = spec["dir"]
    if not os.path.isdir(d):
        return ["no_dir"]
    run_json = os.path.join(d, "run.json")
    run = read_json(run_json) if os.path.exists(run_json) else None
    if run is None:
        why.append("no_run_json")
        run = {}
    g = os.path.join(d, "G_AUDIT.json")
    if not os.path.exists(g):
        why.append("no_g_audit")
    elif read_json(g).get("met") is not True:
        why.append("g_audit_not_met")
    recs = list_records(d, spec["kind"] == "single")
    keys = []
    for r in recs:
        tgt = r.get("target") if spec["role"] == "natural" else ("*" if spec["role"] == "p1" else None)
        keys.append((int(r["seed"]), tgt))
    if len(set(keys)) != len(keys):
        why.append("duplicate_keys")
    if set(keys) != spec["keys"]:
        why.append("seed_set")
    if spec["kind"] == "single":
        seen = {float(r.get("time_limit_s", -1)) for r in recs}
        if recs and seen != {60.0}:
            why.append("time_limits")
    else:
        seen = set()
        for r in recs:
            tl = r.get("time_limits") or {}
            seen.add((float(tl.get("step_timeout_s", -1)), float(tl.get("retry", -1)), float(tl.get("task_time_limit_s", -1))))
        if recs and seen != {(30.0, 1.0, 200.0)}:
            why.append("time_limits")
    tl = run.get("time_limits")
    if tl:
        if spec["kind"] == "single":
            if float(tl.get("time_limit_s", -1)) != 60.0:
                why.append("run_json_time_limits")
        elif (float(tl.get("step_timeout_s", -1)), float(tl.get("retry", -1)), float(tl.get("task_time_limit_s", -1))) \
                != (30.0, 1.0, 200.0):
            why.append("run_json_time_limits")
    segs = run.get("env_segments")
    if type(segs) is not list or len(segs) != 1:
        why.append("env_segments")
    env_kinds = set()
    for r in recs:
        e = r.get("env")
        env_kinds.add(tuple(e.get(k) for k in ENV_FIELDS) if isinstance(e, dict) else "none")
    if "none" in env_kinds or len(env_kinds) > 1:
        why.append("env_mixed")
    for r in recs:
        if not (r.get("env") or {}).get("git_head"):
            why.append("no_git_head")
            break
    sha_seen = {}
    for r in recs:
        dg = r.get("diag") or {}
        for k in dg:
            if k.endswith("sha256"):
                sha_seen.setdefault(k, set()).add(str(dg[k]))
    if any(len(v) > 1 for v in sha_seen.values()):
        why.append("diag_sha_mixed")
    if spec["kind"] == "single":
        folder = os.path.basename(os.path.normpath(d))
        if any(r.get("experiment") != spec["experiment"] or r.get("condition") != folder for r in recs):
            why.append("experiment_condition_label")
    if spec["model"] is not None:
        for r in recs:
            mm = r.get("model")
            label = mm["name"] if isinstance(mm, dict) else mm
            if label != spec["model"]:
                why.append("model_label")
                break
    if "induce" in spec and recs:
        if {(r.get("induce") or {}).get("kind") for r in recs} != {spec["induce"]}:
            why.append("induce_label")
    if "v3" in spec and recs:
        if any(("v3" in r) != spec["v3"] for r in recs):
            why.append("executor_label")
    al = allocations()
    lo, hi = {"e7": al["test1_E7"], "natural": al["test1_natural"], "p1": al["test1_P1"]}[spec["role"]]
    if spec["role"] == "e7" and q["e7_band_extended"]:
        hi = 160199
    if any(not (lo <= s <= hi) for s, _ in keys):
        why.append("band")
    return why


# ================================================================ 版の照合 (b)（自前）
def single_value(vals):
    """全部が同じなら、その値。違う・空なら None。"""
    keys = set()
    for v in vals:
        keys.add(json.dumps(v, sort_keys=True))
    if len(keys) != 1:
        return None
    return vals[0]


def must_check_versions(specs, q):
    if q["versions"] is not None:
        return True
    for spec in specs.values():
        if os.path.isdir(spec["dir"]):
            for r in list_records(spec["dir"], spec["kind"] == "single"):
                if "t1" in r:
                    return True
    return False


def versions_of(name, spec, q):
    """(理由の列, 要約)。"""
    why = []
    out = {"entry_sha256": None, "ckpt_sha256": None, "executor_v3_sha256": None, "rtc_setting": None}
    want = q["versions"] or {}
    if not want:
        why.append("no_versions_param")
    recs = list_records(spec["dir"], spec["kind"] == "single") if os.path.isdir(spec["dir"]) else []
    if len(recs) == 0 or any(type(r.get("t1")) is not dict for r in recs):
        why.append("no_t1")
        return why, out
    t1 = [r["t1"] for r in recs]
    fsha = [x.get("files_sha256") or {} for x in t1]
    for f in fsha:
        if f.get(REL_GATES) != GATES_POSTED:
            why.append("gates_sha")
            break
    e = single_value([f.get(REL_ENTRY) for f in fsha])
    if e is None:
        why.append("entry_mixed")
    elif e != want.get("entry_sha256"):
        why.append("entry_not_posted")
    out["entry_sha256"] = e
    for x in t1:
        if x.get("role") != name:
            why.append("t1_role")
            break
    model = spec["model"]
    cpath = single_value([(x.get("ckpt") or {}).get("path") for x in t1])
    if cpath is None or model not in CKPT_RUNS or cpath != CKPT_RUNS[model] + CKPT_TAIL:
        why.append("ckpt_path")
    csha = single_value([(x.get("ckpt") or {}).get("sha256") for x in t1])
    if csha is None or csha != (want.get("ckpt_sha256") or {}).get(model):
        why.append("ckpt_sha")
    out["ckpt_sha256"] = csha
    if spec["kind"] == "task":
        if spec.get("v3"):
            if any("v3" not in r for r in recs) or single_value([(r.get("v3") or {}).get("settings") for r in recs]) is None:
                why.append("v3_settings")
            for r in recs:
                if (r.get("b1") or {}).get("executor_version") != V3_NAME:
                    why.append("v3_version")
                    break
            xs = single_value([((r.get("b1") or {}).get("files_sha256") or {}).get(REL_EXEC_V3) for r in recs])
            if xs is None or xs != want.get("executor_v3_sha256"):
                why.append("executor_v3_sha")
            out["executor_v3_sha256"] = xs
        else:
            for r in recs:
                if (r.get("b1") or {}).get("executor_version") != "current":
                    why.append("current_version")
                    break
    elif spec.get("rtc_arm"):
        st = single_value([(x.get("rtc") or {}).get("setting") for x in t1])
        bad = st is None or st != want.get("rtc_setting")
        for r in recs:
            if (r.get("diag") or {}).get("arm") != st:
                bad = True
        if bad:
            why.append("rtc_setting")
        mod = single_value([(x.get("rtc") or {}).get("module_sha256") for x in t1])
        if mod is None or mod != want.get("rtc_module_sha256"):
            why.append("rtc_module_sha")
        out["rtc_setting"] = st
    else:
        for x, r in zip(t1, recs):
            if x.get("rtc") is not None or (r.get("diag") or {}).get("arm") not in (None, "naive"):
                why.append("rtc_mark_on_naive")
                break
    return why, out


def versions_across(checks, specs):
    why = []
    entries = set()
    by_model = {}
    v3s = set()
    for name, c in checks.items():
        v = c.get("versions") or {}
        entries.add(v.get("entry_sha256"))
        by_model.setdefault(specs[name]["model"], set()).add(v.get("ckpt_sha256"))
        if v.get("executor_v3_sha256"):
            v3s.add(v["executor_v3_sha256"])
    if len(entries) != 1:
        why.append("entry_across")
    for m in by_model:
        if len(by_model[m]) != 1:
            why.append(f"ckpt_across:{m}")
    if len(v3s) > 1:
        why.append("executor_v3_across")
    return why


# ================================================================ 数える（自前）
def established_before(r, lim):
    ind = r.get("induce") or {}
    if not ind.get("established"):
        return False
    t = ind.get("t_established")
    return t is not None and float(t) + TINY < lim


def success_within(r, lim):
    if not r.get("success"):
        return False
    t = r.get("t_success")
    return t is not None and float(t) <= lim + TINY


def by_seed(recs):
    out = {}
    for r in recs:
        out[int(r["seed"])] = r
    return out


def layer_counts(dir_r, dir_n, lim):
    rr, nn = by_seed(list_records(dir_r, True)), by_seed(list_records(dir_n, True))
    pairs = b = c = both = neither = 0
    for s in sorted(rr):
        if s not in nn:
            continue
        x, y = rr[s], nn[s]
        if not (established_before(x, lim) and established_before(y, lim)):
            continue
        pairs += 1
        sx, sy = success_within(x, lim), success_within(y, lim)
        if sx and sy:
            both += 1
        elif sx:
            b += 1
        elif sy:
            c += 1
        else:
            neither += 1
    return {"pairs": pairs, "b": b, "c": c, "both": both, "neither": neither}


def rate_after_induce(d, lim):
    use = [r for r in list_records(d, True) if established_before(r, lim)]
    ok = [r for r in use if success_within(r, lim)]
    return {"k": len(ok), "n": len(use), "rate": (len(ok) / len(use)) if use else None, "wilson95": wilson95(len(ok), len(use)),
            "t_success_median": middle([float(r["t_success"]) for r in ok]) if ok else None}


def nat_compare(da, db, lim):
    a = {(int(r["seed"]), r["target"]): success_within(r, lim) for r in list_records(da, True)}
    b = {(int(r["seed"]), r["target"]): success_within(r, lim) for r in list_records(db, True)}
    keys = [k for k in sorted(a) if k in b]
    t11 = t10 = t01 = t00 = 0
    for k in keys:
        if a[k] and b[k]:
            t11 += 1
        elif a[k]:
            t10 += 1
        elif b[k]:
            t01 += 1
        else:
            t00 += 1
    n = len(keys)
    return {"pairs": n, "a_k": t11 + t10, "b_k": t11 + t01, "a_wilson95": wilson95(t11 + t10, n), "b_wilson95": wilson95(t11 + t01, n),
            "a_only": t10, "b_only": t01, "mcnemar_p": exact_two_sided(t10, t01), "newcombe_a_minus_b": newcombe_pair(t11, t10, t01, t00)}


def task_compare(da, db):
    a = {int(r["seed"]): bool(r.get("all_three_in_box")) for r in list_records(da, False)}
    b = {int(r["seed"]): bool(r.get("all_three_in_box")) for r in list_records(db, False)}
    t11 = t10 = t01 = t00 = 0
    for s in sorted(a):
        if s not in b:
            continue
        x, y = a[s], b[s]
        t11 += x and y
        t10 += x and not y
        t01 += y and not x
        t00 += (not x) and (not y)
    return {"pairs": t11 + t10 + t01 + t00, "b": t10, "c": t01, "both": t11, "neither": t00,
            "newcombe_a_minus_b": newcombe_pair(t11, t10, t01, t00)}


def interventions_of(meta):
    """56_intervention_s3.py の定義を書き下したもの（種類別の数、総数、LLM の呼び出しは別に読む）。"""
    rets = meta.get("returns") or []
    kinds = [r.get("kind") for r in rets]
    for k in kinds:
        if k not in KNOWN_RETURNS:
            raise ValueError(f"知らない戻す動きの種類 {k}")
    steps = meta.get("steps") or []
    retry = 0
    for s in steps:
        retry += max(0, len(s.get("attempts") or []) - 1)
    if kinds.count("retry") and kinds.count("retry") != retry:
        raise ValueError("出し直しの数が合わない")
    plan = meta.get("plan")
    replan = (len(plan) - 1) if isinstance(plan, list) else 0
    reps = meta.get("replans") or []
    n_cont = len([r for r in reps if (r.get("applied") or {}).get("kind") == "continue"])
    if kinds.count("replan") > n_cont:
        raise ValueError("立て直しの戻す動きが多い")
    replan += len([r for r in reps if r.get("intervention_kind") == "plan_change"])
    counts = {"scripted_return": kinds.count("placed"), "retry": retry, "replan": max(0, replan),
              "judge_override": len([s for s in steps if s.get("judged_complete") and s.get("t_judge") is None])}
    return counts


def task_arm(d):
    recs = list_records(d, False)
    n = len(recs)
    tot = {"scripted_return": 0, "retry": 0, "replan": 0, "judge_override": 0}
    k = done = stored = llm = timed = 0
    t_end_sum = 0.0
    at_k = {"0": 0, "1": 0, "2": 0}
    t3 = []
    for m in recs:
        cnt = interventions_of(m)
        for key in tot:
            tot[key] += cnt[key]
        ok = bool(m.get("all_three_in_box"))
        k += ok
        total = sum(cnt.values())
        for kk in at_k:
            at_k[kk] += ok and total <= int(kk)
        plan = m.get("plan")
        plan0 = plan[0] if isinstance(plan, list) and plan else plan
        steps = m.get("steps") or []
        n_plan = len((plan0 or {}).get("steps") or [])
        if (m.get("stopped") is None and not m.get("timed_out") and len(steps) == n_plan and n_plan > 0
                and all(s.get("judged_complete") for s in steps)):
            done += 1
        stored += len([c for c in COLOR_LIST if (m.get("final_in_box") or {}).get(c)])
        rt = os.path.join(d, "run_%04d_runtime.json" % int(m["run"]))
        lat = read_json(rt).get("latency") if os.path.exists(rt) else None
        llm += len(lat["llm"]) if (lat is not None and "llm" in lat) else (1 if plan0 else 0)
        timed += bool(m.get("timed_out"))
        t_end_sum += float(m["t_end"])
        tr = m.get("truth_success_t") or {}
        if ok and all(tr.get(c) is not None for c in COLOR_LIST):
            t3.append(max(float(tr[c]) for c in COLOR_LIST))
    return {"n": n, "all_three": k, "wilson95": wilson95(k, n), "judged_all_done": done, "mean_stored": (stored / n) if n else None,
            "success_at_k": at_k, "interventions": tot, "llm_calls": llm, "timed_out": timed,
            "success_per_sim_hour": (k / (t_end_sum / 3600.0)) if t_end_sum > 0 else None, "t_all_three_median": middle(t3)}


# ================================================================ 本体
def run_b(layout, params):
    q = fill(params)
    specs = plan_conditions(layout, q)
    checks = {}
    do_versions = must_check_versions(specs, q)
    for name in specs:
        w = inspect(name, specs[name], q)
        heads = []
        if os.path.isdir(specs[name]["dir"]):
            heads = sorted({str((r.get("env") or {}).get("git_head"))
                            for r in list_records(specs[name]["dir"], specs[name]["kind"] == "single")})
        checks[name] = {"ok": len(w) == 0, "problems": w, "git_heads": heads}
        if do_versions:
            vw, vs = versions_of(name, specs[name], q)
            checks[name]["problems"] = w + vw
            checks[name]["ok"] = len(w) == 0 and len(vw) == 0
            checks[name]["versions"] = vs
    vcheck = {"done": do_versions, "ok": None, "problems": []}
    if do_versions and len(checks) > 0:
        vcheck["problems"] = versions_across(checks, specs)
        vcheck["ok"] = len(vcheck["problems"]) == 0 and all(v["ok"] for v in checks.values())
    missing = [x for x in needed(q) if x not in checks]
    complete = not missing and all(v["ok"] for v in checks.values()) and len(vcheck["problems"]) == 0
    res = {"schema": "recovery_vla.s4_test1_result/1", "status": "complete" if complete else "incomplete", "params": q,
           "checks": checks, "missing_conditions": missing, "version_check": vcheck, "primary": None, "holm": None,
           "secondary": None, "face_switch": None}
    if not complete:
        return res
    D = {k: v["dir"] for k, v in specs.items()}
    prim, pv = {}, {}
    if q["h1"]:
        t = task_compare(D["e7.v3"], D["e7.cur"])
        del t["newcombe_a_minus_b"]
        t["p"] = exact_two_sided(t["b"], t["c"])
        prim["H1"] = t
        pv["H1"] = t["p"]
    lay = {}
    for s in q["h2_layers"]:
        lay[s] = layer_counts(D[f"p1.{s}.R"], D[f"p1.{s}.N"], LIMIT_30)
    bb = sum(v["b"] for v in lay.values())
    cc = sum(v["c"] for v in lay.values())
    prim["H2"] = {"layers": lay, "pairs": sum(v["pairs"] for v in lay.values()), "b": bb, "c": cc, "p": exact_two_sided(bb, cc)}
    pv["H2"] = prim["H2"]["p"]
    if q["plan"] == "B":                       # 改良版の R と N（RTC の設定。事前登録の案 12-2）
        r_arm, n_arm = "rtc.p1", "rtc.p1_n"
    else:
        r_arm, n_arm = "p1.1000.R", "p1.1000.N"
    h3 = layer_counts(D[r_arm], D[n_arm], LIMIT_30)
    h3["arms"] = [r_arm, n_arm]
    h3["p"] = exact_two_sided(h3["b"], h3["c"])
    prim["H3"] = h3
    pv["H3"] = h3["p"]
    hb = holm_b(pv)
    for k in prim:
        prim[k]["p_holm"] = hb["by"][k]["p_holm"]
        prim[k]["established"] = bool(prim[k]["p_holm"] < SIG and prim[k]["b"] > prim[k]["c"])
    res["primary"] = prim
    res["holm"] = hb
    res["secondary"] = second(D, q)
    res["face_switch"] = face(prim, D, q)
    return res


def second(D, q):
    out = {"e7_arms": {}, "e7_r_vs_n": None, "p1_rates": {}, "p1_layers_by_limit": {}, "natural": {}, "rtc": None}
    for arm in ("v3", "cur", "n1v3_v3"):
        if "e7." + arm in D:
            out["e7_arms"][arm] = task_arm(D["e7." + arm])
    if "e7.v3" in D and "e7.n1v3_v3" in D:
        t = task_compare(D["e7.v3"], D["e7.n1v3_v3"])
        t["mcnemar_p"] = exact_two_sided(t["b"], t["c"])
        out["e7_r_vs_n"] = t
    for name in sorted(k for k in D if k.startswith("p1.") or k in ("rtc.p1", "rtc.p1_n")):
        out["p1_rates"][name] = {"30": rate_after_induce(D[name], LIMIT_30), "45": rate_after_induce(D[name], LIMIT_45),
                                 "60": rate_after_induce(D[name], LIMIT_60)}
    layers = sorted({k.split(".")[1] for k in D if k.startswith("p1.")})
    for lim, key in ((LIMIT_30, "30"), (LIMIT_45, "45"), (LIMIT_60, "60")):
        tab = {s: layer_counts(D[f"p1.{s}.R"], D[f"p1.{s}.N"], lim) for s in layers}
        b = sum(v["b"] for v in tab.values())
        c = sum(v["c"] for v in tab.values())
        out["p1_layers_by_limit"][key] = {"layers": tab, "b": b, "c": c, "p_all_layers": exact_two_sided(b, c)}
    for s in sorted({k.split(".")[1] for k in D if k.startswith("natural.")}):
        out["natural"][s] = {"30": nat_compare(D[f"natural.{s}.R"], D[f"natural.{s}.N"], LIMIT_30),
                             "60": nat_compare(D[f"natural.{s}.R"], D[f"natural.{s}.N"], LIMIT_60)}
    if "rtc.natural" in D and "natural.1000.R" in D:
        ni = nat_compare(D["rtc.natural"], D["natural.1000.R"], LIMIT_30)
        nc = ni["newcombe_a_minus_b"]
        out["rtc"] = {"natural_vs_naive": ni, "ni_margin": q["ni_margin"],
                      "non_inferior": None if nc is None else bool(nc["ci95"][0] > -q["ni_margin"])}
        if "rtc.p1" in D and "p1.1000.R" in D:
            x, y = rate_after_induce(D["rtc.p1"], LIMIT_30), rate_after_induce(D["p1.1000.R"], LIMIT_30)
            out["rtc"]["p1_vs_naive"] = {"rtc": x, "naive": y, "diff": newcombe_indep(x["k"], x["n"], y["k"], y["n"])}
    return out


def face(prim, D, q):
    c1 = {"by": "H3", "value": prim["H3"]["established"], "pass": prim["H3"]["established"]}
    if q["plan"] == "A":
        nat_new, nat_old, p1_new, p1_old = D["natural.1000.R"], D["natural.1000.R"], D["p1.1000.R"], D["p1.1000.R"]
    else:
        nat_new, nat_old, p1_new, p1_old = D["rtc.natural"], D["natural.1000.R"], D["rtc.p1"], D["p1.1000.R"]
    g = nat_compare(nat_new, nat_old, LIMIT_30)
    x, y = rate_after_induce(p1_new, LIMIT_30), rate_after_induce(p1_old, LIMIT_30)
    gp = newcombe_indep(x["k"], x["n"], y["k"], y["n"])                 # 各腕の分母（記述だけ）
    pr = layer_counts(p1_new, p1_old, LIMIT_30)                        # 同じ種で両方とも成立した対
    pci = newcombe_pair(pr["both"], pr["b"], pr["c"], pr["neither"])
    pd = None if pr["pairs"] == 0 else (pr["b"] - pr["c"]) / pr["pairs"]
    nd = (g["a_k"] - g["b_k"]) / g["pairs"] if g["pairs"] else None
    gci = g["newcombe_a_minus_b"]["ci95"] if g["newcombe_a_minus_b"] else None
    if q["guard_mode"] == "point":
        ok_nat = nd is not None and nd + 0.05 >= -TINY
        ok_p1 = pd is not None and pd + 0.10 >= -TINY
    else:
        ok_nat = gci is not None and gci[0] + 0.05 >= -TINY
        ok_p1 = pci is not None and pci["ci95"][0] + 0.10 >= -TINY
    if q["plan"] == "A":
        ok_p1 = True                           # 同じ記録どうし（構造上 0）
    c2 = {"plan": q["plan"], "mode": q["guard_mode"], "g_nat_diff": nd, "g_nat_ci95": gci,
          "g_p1_count": "paired" if q["plan"] == "B" else "structural",
          "g_p1_pairs": pr["pairs"], "g_p1_b": pr["b"], "g_p1_c": pr["c"], "g_p1_diff": pd,
          "g_p1_ci95": pci["ci95"] if pci else None,
          "g_p1_unpaired": {"imp_k": x["k"], "imp_n": x["n"], "base_k": y["k"], "base_n": y["n"],
                            "diff": gp["diff"] if gp else None, "ci95": gp["ci95"] if gp else None},
          "g_nat": bool(ok_nat), "g_p1": bool(ok_p1), "pass": bool(ok_nat and ok_p1)}
    has1 = "H1" in prim
    c3 = {"by": "H1", "present": has1, "pass": bool(has1 and prim["H1"]["established"])}
    c4 = {"by": "params.c4_on_time（人が確かめる）", "pass": q["c4_on_time"]}
    first3 = bool(c1["pass"] and c2["pass"] and c3["pass"])
    return {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c1_to_c3": first3,
            "switch": None if q["c4_on_time"] is None else bool(first3 and q["c4_on_time"])}


def main(argv=None):
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
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
