"""段階 4 束 4 の解析（実装 A）。関門 2・関門 3（configs/s4_gates.json の bundle4_gates.G2・G3）とテスト 2（事前登録 v2 の草案の
H1・H2・Holm・副次）を、記録から出す。実装 B（scripts/98_s4_b4_b.py）とは何も共有しない（ファイルの列挙・JSON の読み方・数え方・
検定）。照合は scripts/98_s4_b4_check.py check。

    from recovla.eval import b4 as B4
    res = B4.analyze(layout, params)     # layout は scripts/98_s4_b4_eval.py layout が書くもの、params は docs/stage4/bundle4_eval.md

A の作り: ファイルは pathlib の glob で列挙、検定は scipy.stats（binomtest・norm）、対の差の区間は recovla.eval.stats.paired_diff_ci
（Newcombe の方法 10、φ の補正あり）、単一の割合は recovla.eval.stats.wilson_interval。
時刻の境界（掲示板 0155 の 1）: 誘発の成立は t_established < L（ちょうど L は入れない）、成功は t_success <= L。
割合の差（守り・狙い）は、各腕で L より前に誘発が成立した試行を分母にした割合の差（事前登録 v2 の草案 12-2）。自然は 198 対の対の差。
入口の点検（0155 の 2 節）を満たさない条件が 1 つでもあれば status = "incomplete" とし、検定・判定は出さない（result = None）。
条件をまたぐ点検のうち、同じモデルの保存点の SHA-256 と環境は A・B がそれぞれ行い、git の HEAD の照らし合わせと台帳は入口
（scripts/98_s4_b4_check.py）が 1 回だけ行う。
置き損ね（P3）は族に入れない（掲示板 0165）。Holm の族は H1・H2（params.h2_in_family が偽なら H1 だけ）。
"""
import json
import math
import pathlib

from scipy import stats as st

from recovla.eval import stats as S

SCHEMA = "recovery_vla.s4_b4_result/1"
ALPHA = 0.05
LIMITS = (30.0, 45.0, 60.0)
L_MAIN, L_SUB = 30.0, 60.0
EPS = 1e-9
COLORS = ("red", "green", "blue")
ROOT = pathlib.Path(__file__).resolve().parents[3]
GATES = ROOT / "configs" / "s4_gates.json"
SINGLE_LIMIT_S = 60.0
ENV_KEYS = ("driver", "torch", "torch_cuda", "os_build")
RUNTIME_WANT = {"mode": "naive", "exec_interval": 6, "safety_filter": False}
PARAM_DEFAULTS = {"guard_mode": "point", "h2_in_family": True, "g3_with_r1v3s1001": True, "ckpt_sha256": {}, "p_fill": {}}
THRESH = {"g_nat": -0.05, "g_p1": -0.10, "g_rn": 0.20, "aim": 0.15}
# 段階ごとの条件（"<部分>.<モデル>"）と帯。テスト 2 は s4_gates.json の test2_*、関門 2・3 は事前登録 v2 の草案 12-2 の表
G_BANDS = {"G2": {"nat": (192100, 192165), "P1": (192200, 192299), "P2": (192300, 192399), "P3": (192600, 192699)},
           "G3": {"P1": (192400, 192499), "P2": (192500, 192599)}}
T2_ALLOC = {"nat": "test2_natural", "P1": "test2_P1", "P2": "test2_drop", "P3": "test2_misplace"}
REQUIRED = {"G2": ["nat.R4", "nat.R1v3", "P1.R4", "P1.N4", "P1.R1v3", "P2.R4", "P2.R1v3", "P3.R4", "P3.N4", "P3.R1v3"],
            "G3": ["P1.R4s1001", "P1.N4s1001", "P2.R4s1001", "P2.R1v3s1001"],
            "T2": ["nat.R4", "nat.R1v3", "P1.R4", "P1.N4", "P1.R1v3", "P2.R4", "P2.N4", "P2.R1v3", "P3.R4", "P3.N4", "P3.R1v3"]}
INDUCE = {"nat": None, "P1": "P1", "P2": "P2", "P3": "P3"}


# ================================================================ 読み込み
def params_with_defaults(params: dict) -> dict:
    p = dict(PARAM_DEFAULTS)
    p.update(params or {})
    unknown = sorted(set(p) - set(PARAM_DEFAULTS))
    if unknown:
        raise ValueError(f"知らない params: {unknown}")
    if p["guard_mode"] not in ("point", "interval"):
        raise ValueError("guard_mode は point か interval")
    for k in ("h2_in_family", "g3_with_r1v3s1001"):
        if not isinstance(p[k], bool):
            raise ValueError(f"{k} は真偽")
    if not isinstance(p["ckpt_sha256"], dict):
        raise ValueError("ckpt_sha256 は {モデル: SHA-256}")
    return p


def bands(phase: str) -> dict:
    if phase in G_BANDS:
        return dict(G_BANDS[phase])
    g = json.loads(GATES.read_text(encoding="utf-8"))
    al = {a["id"]: tuple(a["range"]) for a in g["bands"]["allocations"]}
    return {part: al[aid] for part, aid in T2_ALLOC.items()}


def required(phase: str, p: dict) -> list:
    need = list(REQUIRED[phase])
    if phase == "G3" and not p["g3_with_r1v3s1001"]:
        need.remove("P2.R1v3s1001")
    return need


def records(d: pathlib.Path) -> list:
    return [json.loads(x.read_text(encoding="utf-8")) for x in sorted(d.glob("trial_[0-9][0-9][0-9][0-9].json"))]


def plan_keys(part: str, lo: int, hi: int) -> set:
    if part == "nat":
        return {(s, c) for s in range(lo, hi + 1) for c in COLORS}
    return {(s, "*") for s in range(lo, hi + 1)}


def rec_key(part: str, r: dict) -> tuple:
    return (int(r["seed"]), r.get("target") if part == "nat" else "*")


# ================================================================ 入口の点検（0155 の 2 節）
def check_condition(phase: str, name: str, d: pathlib.Path, model: str, experiment: str, p: dict) -> dict:
    part = name.split(".", 1)[0]
    lo, hi = bands(phase)[part]
    prob = []
    out = {"ok": False, "problems": prob, "git_heads": [], "ckpt_sha256": None, "env": None, "n": 0}
    if not d.is_dir():
        prob.append(f"{name}: フォルダがない {d}")
        return out
    run = {}
    if (d / "run.json").is_file():
        run = json.loads((d / "run.json").read_text(encoding="utf-8"))
    else:
        prob.append(f"{name}: run.json がない")
    if not (d / "G_AUDIT.json").is_file():
        prob.append(f"{name}: G_AUDIT.json がない")
    elif json.loads((d / "G_AUDIT.json").read_text(encoding="utf-8")).get("met") is not True:
        prob.append(f"{name}: G_AUDIT.json の met が真でない")
    recs = records(d)
    out["n"] = len(recs)
    got = [rec_key(part, r) for r in recs]
    want = plan_keys(part, lo, hi)
    if len(got) != len(set(got)):
        prob.append(f"{name}: 同じ対の鍵が 2 回ある")
    if set(got) != want:
        prob.append(f"{name}: 本数・種の集合が計画と違う（記録 {len(got)}、計画 {len(want)}）")
    outside = [k[0] for k in got if not (lo <= k[0] <= hi)]
    if outside:
        prob.append(f"{name}: 帯 {lo}〜{hi} の外の種 {outside[0]}")
    # 印（0155 の 2-3）
    for r in recs:
        m = r.get("model")
        if (m.get("name") if isinstance(m, dict) else m) != model:
            prob.append(f"{name}: モデルの印が {model} でない（seed {r.get('seed')}）")
            break
    for r in recs:
        if r.get("experiment") != experiment or r.get("condition") != d.name:
            prob.append(f"{name}: 実験名・条件名の印がフォルダと違う（seed {r.get('seed')}）")
            break
    if recs and {(r.get("induce") or {}).get("kind") for r in recs} != {INDUCE[part]}:
        prob.append(f"{name}: 誘発の印が {INDUCE[part]} でない")
    for r in recs:
        b4 = r.get("b4") or {}
        if (b4.get("phase"), b4.get("part"), b4.get("model")) != (phase, part, model):
            prob.append(f"{name}: b4 の印（段階・部分・モデル）が計画と違う（seed {r.get('seed')}）")
            break
    for r in recs:
        rt = r.get("runtime") or {}
        if any(rt.get(k) != v for k, v in RUNTIME_WANT.items()):
            prob.append(f"{name}: 実行のしかた（naive・6 行・安全フィルタなし）が違う（seed {r.get('seed')}）")
            break
    # 制限時間・環境（0155 の 2-4）
    lims = {float(r.get("time_limit_s", -1)) for r in recs}
    if recs and lims != {SINGLE_LIMIT_S}:
        prob.append(f"{name}: 制限時間が 60 s でないか 2 種類以上")
    rtl = (run.get("time_limits") or {}).get("time_limit_s")
    if run and (rtl is None or float(rtl) != SINGLE_LIMIT_S):
        prob.append(f"{name}: run.json の制限時間が 60 s でない")
    segs = run.get("env_segments")
    if run and (not isinstance(segs, list) or len(segs) != 1):
        prob.append(f"{name}: run.json の env_segments が 1 つでない")
    envs = {json.dumps({k: r["env"].get(k) for k in ENV_KEYS}, sort_keys=True) if isinstance(r.get("env"), dict) else None
            for r in recs}
    if None in envs or len(envs) > 1:
        prob.append(f"{name}: 試行の env が無いか 2 種類以上")
    elif envs:
        out["env"] = next(iter(envs))
    # 版（0155 の 2-5）: HEAD、使ったファイルと保存点の SHA-256
    if any(not (r.get("env") or {}).get("git_head") for r in recs):
        prob.append(f"{name}: env.git_head の無い試行がある")
    out["git_heads"] = sorted({str((r.get("env") or {}).get("git_head")) for r in recs})
    fsha = {}
    for r in recs:
        for k, v in ((r.get("b4") or {}).get("files_sha256") or {}).items():
            fsha.setdefault(k, set()).add(v)
    if any(len(v) > 1 for v in fsha.values()):
        prob.append(f"{name}: 使ったファイルの SHA-256 が 2 種類以上")
    ck = {((r.get("b4") or {}).get("ckpt") or {}).get("sha256") for r in recs}
    if not recs or None in ck or len(ck) != 1:
        prob.append(f"{name}: 保存点の SHA-256 が無いか 2 種類以上")
    else:
        out["ckpt_sha256"] = next(iter(ck))
        want_sha = p["ckpt_sha256"].get(model)
        if want_sha and out["ckpt_sha256"] != want_sha:
            prob.append(f"{name}: 保存点の SHA-256 が掲示した値と違う")
    out["ok"] = not prob
    return out


def cross_check(checks: dict, models: dict) -> dict:
    """同じモデルの保存点の SHA-256 が条件をまたいで同じか、環境が段階の全条件で 1 種類か。"""
    prob = []
    by_model = {}
    for name, c in checks.items():
        by_model.setdefault(models[name], set()).add(c["ckpt_sha256"])
    out_models = {}
    for m in sorted(by_model):
        v = by_model[m]
        if len(v) != 1 or None in v:
            prob.append(f"モデル {m} の保存点の SHA-256 が条件によって違う")
            out_models[m] = None
        else:
            out_models[m] = next(iter(v))
    envs = {c["env"] for c in checks.values()}
    if len(envs) != 1 or None in envs:
        prob.append("環境（ドライバ・torch・CUDA・OS）が条件によって違うか、分からない")
    return {"ok": not prob, "problems": prob, "models": out_models, "env_kinds": len(envs)}


# ================================================================ 検定・区間
def binom_two_sided(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else float(st.binomtest(b, n, 0.5, alternative="two-sided").pvalue)


def wilson(k: int, n: int):
    if n == 0:
        return None
    lo, hi = S.wilson_interval(k, n, float(st.norm.ppf(0.975)))
    return [lo, hi]


def newcombe_paired(n11, n10, n01, n00):
    d, lo, hi = S.paired_diff_ci(n11, n10, n01, n00)
    return None if math.isnan(d) else {"diff": d, "ci95": [lo, hi]}


def newcombe_unpaired(k1, n1, k2, n2):
    if n1 == 0 or n2 == 0:
        return None
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return {"diff": d, "ci95": [d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)]}


def holm(ps: dict) -> dict:
    order = sorted(ps, key=lambda k: (ps[k], k))
    m = len(order)
    out, run_max, going = {}, 0.0, True
    for i, k in enumerate(order):
        run_max = max(run_max, min(1.0, (m - i) * ps[k]))
        going = going and ps[k] <= ALPHA / (m - i) + 1e-15
        out[k] = {"rank": i + 1, "p": ps[k], "p_holm": run_max, "rejected_stepdown": going}
    return {"order": order, "m": m, "by": out}


# ================================================================ 数える
def est_before(r, L) -> bool:
    ind = r.get("induce") or {}
    t = ind.get("t_established")
    return bool(ind.get("established")) and t is not None and float(t) < L - EPS


def succ_by(r, L) -> bool:
    t = r.get("t_success")
    return bool(r.get("success")) and t is not None and float(t) <= L + EPS


def rate(d, part, L) -> dict:
    recs = records(d)
    if part != "nat":
        recs = [r for r in recs if est_before(r, L)]
    k = sum(succ_by(r, L) for r in recs)
    return {"k": k, "n": len(recs), "rate": k / len(recs) if recs else None, "wilson95": wilson(k, len(recs))}


def induced_pairs(dX, dY, L) -> dict:
    X = {int(r["seed"]): r for r in records(dX)}
    Y = {int(r["seed"]): r for r in records(dY)}
    pairs = [(X[s], Y[s]) for s in sorted(set(X) & set(Y)) if est_before(X[s], L) and est_before(Y[s], L)]
    rr = [(succ_by(a, L), succ_by(b, L)) for a, b in pairs]
    b = sum(x and not y for x, y in rr)
    c = sum(y and not x for x, y in rr)
    return {"pairs": len(pairs), "b": b, "c": c, "both": sum(x and y for x, y in rr), "neither": sum(not x and not y for x, y in rr),
            "p": binom_two_sided(b, c)}


def natural_pairs(dX, dY, L) -> dict:
    X = {(int(r["seed"]), r["target"]): r for r in records(dX)}
    Y = {(int(r["seed"]), r["target"]): r for r in records(dY)}
    xy = [(succ_by(X[k], L), succ_by(Y[k], L)) for k in sorted(set(X) & set(Y))]
    n11 = sum(a and b for a, b in xy)
    n10 = sum(a and not b for a, b in xy)
    n01 = sum(b and not a for a, b in xy)
    n00 = len(xy) - n11 - n10 - n01
    nc = newcombe_paired(n11, n10, n01, n00)
    return {"pairs": len(xy), "x_k": n11 + n10, "y_k": n11 + n01, "x_only": n10, "y_only": n01,
            "diff": ((n10 - n01) / len(xy)) if xy else None, "newcombe95": None if nc is None else nc["ci95"]}


def rate_diff(dX, dY, part, L, thr: float, mode: str = "point") -> dict:
    """各腕の割合（誘発は L より前に成立した試行が分母）の差と、点推定（mode=interval なら区間の下限）での判定。"""
    x, y = rate(dX, part, L), rate(dY, part, L)
    nc = newcombe_unpaired(x["k"], x["n"], y["k"], y["n"])
    diff = None if nc is None else nc["diff"]
    if mode == "point":
        ok = diff is not None and diff >= thr - EPS
    else:
        ok = nc is not None and nc["ci95"][0] >= thr - EPS
    return {"x": x, "y": y, "diff": diff, "newcombe95": None if nc is None else nc["ci95"], "threshold": thr, "mode": mode,
            "pass": bool(ok)}


def nat_guard(dX, dY, L, thr: float, mode: str = "point") -> dict:
    g = natural_pairs(dX, dY, L)
    if mode == "point":
        ok = g["diff"] is not None and g["diff"] >= thr - EPS
    else:
        ok = g["newcombe95"] is not None and g["newcombe95"][0] >= thr - EPS
    return dict(g, threshold=thr, mode=mode, **{"pass": bool(ok)})


def all_rates(dirs: dict) -> dict:
    return {name: {f"{L:g}": rate(d, name.split(".", 1)[0], L) for L in LIMITS} for name, d in sorted(dirs.items())}


# ================================================================ 段階ごと
def gate2(dirs: dict) -> dict:
    g = {"natural": nat_guard(dirs["nat.R4"], dirs["nat.R1v3"], L_MAIN, THRESH["g_nat"]),
         "p1_r4_vs_r1v3": rate_diff(dirs["P1.R4"], dirs["P1.R1v3"], "P1", L_MAIN, THRESH["g_p1"]),
         "p1_r4_minus_n4": rate_diff(dirs["P1.R4"], dirs["P1.N4"], "P1", L_MAIN, THRESH["g_rn"])}
    guards_ok = all(v["pass"] for v in g.values())
    aim = rate_diff(dirs["P2.R4"], dirs["P2.R1v3"], "P2", L_MAIN, THRESH["aim"])
    aim["pairs"] = induced_pairs(dirs["P2.R4"], dirs["P2.R1v3"], L_MAIN)
    p3 = {"label": "記述だけ（関門 2 の判定に使わない。掲示板 0165）",
          "rates": {m: {f"{L:g}": rate(dirs[f"P3.{m}"], "P3", L) for L in (L_MAIN, L_SUB)} for m in ("R4", "N4", "R1v3")}}
    nxt = "test2" if guards_ok and aim["pass"] else ("research_guard" if not guards_ok else "no_test2")
    return {"guards": g, "guards_pass": guards_ok, "aim": aim, "p3": p3, "rates": all_rates(dirs),
            "decision": {"pass": bool(guards_ok and aim["pass"]), "next": nxt}}


def gate3(dirs: dict, p: dict) -> dict:
    rn = rate_diff(dirs["P1.R4s1001"], dirs["P1.N4s1001"], "P1", L_MAIN, THRESH["g_rn"])
    direction = None
    if p["g3_with_r1v3s1001"]:
        x = rate_diff(dirs["P2.R4s1001"], dirs["P2.R1v3s1001"], "P2", L_MAIN, 0.0)
        x["pass"] = bool(x["diff"] is not None and x["diff"] > EPS)            # 向きが正（> 0）
        x["threshold"] = "> 0"
        direction = x
    ok = rn["pass"] and (direction is None or direction["pass"])
    return {"direction": direction, "direction_used": direction is not None, "p1_r4_minus_n4": rn, "rates": all_rates(dirs),
            "decision": {"pass": bool(ok), "wording": "two_seeds" if ok else "one_seed_diagnostic"}}


def test2(dirs: dict, p: dict) -> dict:
    prim = {"H1": induced_pairs(dirs["P2.R4"], dirs["P2.R1v3"], L_MAIN),
            "H2": induced_pairs(dirs["P2.R4"], dirs["P2.N4"], L_MAIN)}
    fam = ["H1", "H2"] if p["h2_in_family"] else ["H1"]
    hm = holm({k: prim[k]["p"] for k in fam})
    for k, v in prim.items():
        v["in_family"] = k in fam
        if k in fam:
            v["p_holm"] = hm["by"][k]["p_holm"]
            v["established"] = bool(v["p_holm"] < ALPHA and v["b"] > v["c"])
        else:
            v["p_holm"], v["established"] = None, None
    mode = p["guard_mode"]
    guards = {"natural": nat_guard(dirs["nat.R4"], dirs["nat.R1v3"], L_MAIN, THRESH["g_nat"], mode),
              "p1_r4_vs_r1v3": rate_diff(dirs["P1.R4"], dirs["P1.R1v3"], "P1", L_MAIN, THRESH["g_p1"], mode),
              "p1_r4_minus_n4": rate_diff(dirs["P1.R4"], dirs["P1.N4"], "P1", L_MAIN, THRESH["g_rn"], mode)}
    p3 = {"label": "副次・族の外（Holm の補正をしない。基準を置かない）"}
    for L in (L_MAIN, L_SUB):
        p3[f"{L:g}"] = {"R4_vs_R1v3": induced_pairs(dirs["P3.R4"], dirs["P3.R1v3"], L),
                        "R4_vs_N4": induced_pairs(dirs["P3.R4"], dirs["P3.N4"], L),
                        "rates": {m: rate(dirs[f"P3.{m}"], "P3", L) for m in ("R4", "N4", "R1v3")}}
    sixty = {"label": "副次・族の外", "H1_form": induced_pairs(dirs["P2.R4"], dirs["P2.R1v3"], L_SUB),
             "H2_form": induced_pairs(dirs["P2.R4"], dirs["P2.N4"], L_SUB)}
    sec = {"guards": guards, "guards_pass": all(v["pass"] for v in guards.values()), "p3": p3, "sixty": sixty,
           "rates": all_rates(dirs)}
    return {"primary": prim, "holm": hm, "secondary": sec}


# ================================================================ 本体
def analyze(layout: dict, params: dict) -> dict:
    p = params_with_defaults(params)
    phase = layout["phase"]
    if phase not in REQUIRED:
        raise ValueError(f"phase は {sorted(REQUIRED)} のどれか")
    root = pathlib.Path(layout["root"]) / layout["experiment"]
    dirs, models, checks = {}, {}, {}
    for name, v in sorted((layout.get("conds") or {}).items()):
        if name not in REQUIRED[phase]:
            raise ValueError(f"{phase} に無い条件 {name}")
        dirs[name] = root / v["cond"]
        models[name] = v["model"]
        if name.split(".", 1)[1] != v["model"]:
            raise ValueError(f"{name} のモデル {v['model']} が条件名と違う")
    use = [n for n in dirs if n in required(phase, p)]
    for name in use:
        checks[name] = check_condition(phase, name, dirs[name], models[name], layout["experiment"], p)
    missing = [n for n in required(phase, p) if n not in dirs]
    cross = cross_check(checks, models) if checks else {"ok": False, "problems": ["条件がない"], "models": {}, "env_kinds": 0}
    status = "complete" if checks and all(c["ok"] for c in checks.values()) and not missing and cross["ok"] else "incomplete"
    res = {"schema": SCHEMA, "phase": phase, "status": status, "params": p, "checks": checks, "missing_conditions": missing,
           "cross": cross, "result": None}
    if status != "complete":
        return res
    d = {n: dirs[n] for n in use}
    res["result"] = gate2(d) if phase == "G2" else (gate3(d, p) if phase == "G3" else test2(d, p))
    return res


# ================================================================ 要約（Markdown）
def _pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def _diff(v):
    return "—" if v.get("diff") is None else f"{100 * v['diff']:+.1f} ポイント"


def summary_md(res: dict) -> str:
    title = {"G2": "関門 2", "G3": "関門 3", "T2": "テスト 2"}[res["phase"]]
    o = [f"# 束 4 {title}の判定（1 枚）", "", f"- 状態: {'そろった（判定した）' if res['status'] == 'complete' else '未完（判定しない）'}"]
    if res["status"] != "complete":
        o += ["", "## 入口の点検で満たさなかったもの", ""]
        o += [f"- {pr}" for c in res["checks"].values() for pr in c["problems"]]
        o += [f"- 条件がない: {x}" for x in res["missing_conditions"]]
        o += [f"- 条件をまたぐ点検: {x}" for x in res["cross"]["problems"]]
        o += [f"- 条件をまたぐ点検（入口）: {x}" for x in (res.get("entry_audit") or {}).get("problems", [])]
        return "\n".join(o) + "\n"
    r = res["result"]
    if res["phase"] == "G2":
        g = r["guards"]
        o += ["", "## 守り（全部満たす。30 s、点推定）", "",
              f"- 自然の成功 R4 − R1v3: {_diff(g['natural'])}（{g['natural']['pairs']} 対、基準 −5 ポイント）→ {'満たす' if g['natural']['pass'] else '割った'}",
              f"- 把持失敗の復帰 R4 − R1v3: {_diff(g['p1_r4_vs_r1v3'])}（基準 −10 ポイント）→ {'満たす' if g['p1_r4_vs_r1v3']['pass'] else '割った'}",
              f"- 把持失敗の復帰 R4 − N4: {_diff(g['p1_r4_minus_n4'])}（基準 +20 ポイント）→ {'満たす' if g['p1_r4_minus_n4']['pass'] else '割った'}",
              "", "## 狙い（落下の 30 s の復帰）", "",
              f"- R4 {_pct(r['aim']['x']['rate'])}（{r['aim']['x']['k']}/{r['aim']['x']['n']}）、R1v3 {_pct(r['aim']['y']['rate'])}"
              f"（{r['aim']['y']['k']}/{r['aim']['y']['n']}）、差 {_diff(r['aim'])}（基準 +15 ポイント）→ {'届いた' if r['aim']['pass'] else '届かない'}",
              "", "- 分母: 各腕で誘発が 30 s より前に成立した試行（自然は 198 対）。置き損ね（P3）は記述だけ。",
              "", f"## 結論: {({'test2': 'テスト 2 へ進む（事前登録 v2 の登録の後）', 'research_guard': '守りを割った。研究の道で原因を調べる（テスト 2 は回さない）', 'no_test2': '狙いに届かない。テスト 2 へ進まない'})[r['decision']['next']]}"]
    elif res["phase"] == "G3":
        dr = r["direction"]
        o += ["", "- 向き（落下の 30 s の復帰 R4s1001 − R1v3s1001 > 0）: "
              + ("使わない（R1v3s1001 がない。P-3）" if dr is None else f"{_diff(dr)} → {'正' if dr['pass'] else '正でない'}"),
              f"- 把持失敗の復帰 R4s1001 − N4s1001: {_diff(r['p1_r4_minus_n4'])}（基準 +20 ポイント）→ {'満たす' if r['p1_r4_minus_n4']['pass'] else '満たさない'}",
              "", f"## 結論: {'種 2 つで同じ向き' if r['decision']['pass'] else '「種 1 つの結果」と明記し、診断として載せる'}"]
    else:
        o += ["", "## 主要評価項目（Holm、α = 0.05）", "", "| 項目 | 組 | b | c | p | Holm 補正後の p | 成立 |", "|---|---|---|---|---|---|---|"]
        for k in ("H1", "H2"):
            v = r["primary"][k]
            ph = "族の外" if v["p_holm"] is None else f"{v['p_holm']:.6g}"
            est = "—" if v["established"] is None else ("○" if v["established"] else "×")
            o.append(f"| {k} | {v['pairs']} | {v['b']} | {v['c']} | {v['p']:.6g} | {ph} | {est} |")
        o += ["", "- 分母: 同じ種で両方とも落下の誘発が 30 s より前に成立した組（成功は 30 s 以内）。b は R4 だけ、c は相手だけ。",
              "- H1 は R4 対 R1v3、H2 は R4 対 N4（どちらも落下 P2、帯 167000〜167099）。"]
        p3 = r["secondary"]["p3"]["30"]
        o += ["", "## 置き損ね（P3）の 30 s の復帰（副次・族の外）", "",
              f"- R4 対 R1v3: 組 {p3['R4_vs_R1v3']['pairs']}、b {p3['R4_vs_R1v3']['b']}、c {p3['R4_vs_R1v3']['c']}、p {p3['R4_vs_R1v3']['p']:.6g}（副次・族の外）",
              f"- R4 対 N4: 組 {p3['R4_vs_N4']['pairs']}、b {p3['R4_vs_N4']['b']}、c {p3['R4_vs_N4']['c']}、p {p3['R4_vs_N4']['p']:.6g}（副次・族の外）"]
        g = r["secondary"]["guards"]
        o += ["", f"## 守り（副次、{g['natural']['mode']}）: {'満たす' if r['secondary']['guards_pass'] else '割った'}"]
    return "\n".join(o) + "\n"
