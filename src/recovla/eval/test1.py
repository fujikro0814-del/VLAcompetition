"""段階 4 テスト 1 の解析（実装 A）。事前登録 v1 の案（docs/stage4/prereg_test1_v1_draft.md）の H1〜H3・Holm・副次・掲示板 0157 の
顔の切り替えの条件 1〜3 を、記録から出す。実装 B（scripts/98_s4_test1_b.py）とは何も共有しない（ファイルの列挙・JSON の読み方・
数え方・検定・中央値）。照合は scripts/98_s4_test1.py check。

    from recovla.eval import test1 as T1
    res = T1.analyze(layout, params)            # layout・params の形は docs/stage4/test1_analysis.md

A の作り: ファイルは pathlib の glob で列挙、検定は scipy.stats（binomtest・norm）、対の差の区間は recovla.eval.stats.paired_diff_ci
（Newcombe の方法 10、φ の補正あり）、E7 の介入・success@k・時間は scripts/56_intervention_s3.py の count_run（importlib）、
中央値は numpy.median。
時刻の境界（掲示板 0155 の 1）: 誘発の成立は t_established < L（ちょうど L は入れない）、成功は t_success <= L。
3 個の連続タスク（E7）の主な指標は all_three_in_box（0155 の 2）。3 個そろった時刻は all_three_in_box が真の試行だけ、3 色の
truth_success_t の最大。
入口の点検（0155 の 2 節）を満たさない条件が 1 つでもあれば status = "incomplete" とし、検定・判定は出さない（None）。
条件をまたぐ点検（2-5 の HEAD の間の照らし合わせ、2-6 の台帳）は入口（scripts/98_s4_test1.py）が 1 回だけ行う。
H3 の腕: 案 A は種 1000 の naive の P1（p1.1000.R 対 p1.1000.N）、案 B は RTC の設定の P1（rtc.p1 対 rtc.p1_n。事前登録の案 12-2）。
版の照合 (b)（登録版 第 7 節 1 の (b-1)〜(b-5)、判断の紙の問 5）: params.versions（「枝の確定」の掲示の値）があるか、どれかの試行に
起動の入口（scripts/98_s4_test1_eval.py）の印 "t1" があれば、条件ごとに s4_gates.json の SHA-256（LF）が掲示の値（0165）、入口の版・
保存点のパスと SHA-256・v3 の設定と版（v3.1）と executor_v3.py の SHA-256・RTC の設定名と rtc.py の SHA-256 がそれぞれ 1 種類で
掲示の値と同じこと、naive の条件に RTC の印がないことを確かめ、条件をまたいで入口の版・同じモデルの保存点・v3 の版が 1 種類であることを
確かめる（version_check）。満たさなければ未完。どちらも無い（入口を通っていない合成の記録）ときは照らさず、1 枚に「照らしていない」と書く。
案 B の G-P1（登録版 12-3、判断の紙の問 7）: 両方の腕で誘発が 30 s より前に成立した同じ種の対で、差 (b − c)/対の数、区間は Newcombe の
対あり（束 4 の paired_guard と同じ）。対が 0 なら満たさない。各腕の分母の差（対なし）は記述として並べる。案 A は構造上 0 で満たす。
"""
import importlib.util
import json
import math
import pathlib

import numpy as np
from scipy import stats as st

from recovla.eval import stats as S

SCHEMA = "recovery_vla.s4_test1_result/1"
ALPHA = 0.05
L_MAIN, L_DESC, L_SUB = 30.0, 45.0, 60.0
EPS = 1e-9
COLORS = ("red", "green", "blue")
ROOT = pathlib.Path(__file__).resolve().parents[3]
GATES = ROOT / "configs" / "s4_gates.json"
E7_LIMITS = {"step_timeout_s": 30.0, "retry": 1, "task_time_limit_s": 200.0}
SINGLE_LIMIT_S = 60.0
ENV_KEYS = ("driver", "torch", "torch_cuda", "os_build")        # 96_s4_resume.ENV_STOP_KEYS と同じ
POSTED_GATES_SHA256 = "7f2f651cafa9cf97b5548324d3fb8ea0bec5e891cca9c8859c7dd9c0347646e9"   # 掲示板 0165（LF）
GATES_KEY = "configs/s4_gates.json"
ENTRY_KEY = "scripts/98_s4_test1_eval.py"
EXEC_V3_KEY = "src/recovla/runtime/executor_v3.py"
V3_VERSION = "v3.1"
# 登録版 第 5 節のチェックポイント（2 万手）
REGISTERED_CKPT = {
    "R1v3": "outputs/train/train_R1v3_20261005-180404_20261005-180404/checkpoints/020000/pretrained_model",
    "N1v3": "outputs/train/train_N1v3_20261005-202158_20261005-202158/checkpoints/020000/pretrained_model",
    "R1v3s1001": "outputs/s4/train/train_R1v3s1001_20261009-002457_20261009-002457/checkpoints/020000/pretrained_model",
    "N1v3s1001": "outputs/s4/train/train_N1v3s1001_20261009-024259_20261009-024259/checkpoints/020000/pretrained_model",
    "R1v3s1002": "outputs/s4/train/train_R1v3s1002_20261009-050112_20261009-050112/checkpoints/020000/pretrained_model",
    "N1v3s1002": "outputs/s4/train/train_N1v3s1002_20261009-071920_20261009-071920/checkpoints/020000/pretrained_model",
}
VERSION_KEYS = ("entry_sha256", "executor_v3_sha256", "rtc_setting", "rtc_module_sha256", "ckpt_sha256")
# h1（P-1）と rtc_arm（P-3）は結果で埋まる所なので既定を置かない（params に必ず書く）
PARAM_REQUIRED = ("h1", "rtc_arm")
PARAM_DEFAULTS = {"e7_n": None, "plan": "A", "guard_mode": "point", "ni_margin": 0.10, "h2_layers": ["1001", "1002"],
                  "rtc_p1_n": 50, "e7_band_extended": False, "c4_on_time": None, "p_fill": {}, "versions": None}


# ================================================================ 読み込み
def params_with_defaults(params: dict) -> dict:
    params = dict(params or {})
    lack = [k for k in PARAM_REQUIRED if k not in params]
    if lack:
        raise ValueError(f"params に {lack} が要る（P-1・P-3。既定を置かない）")
    p = dict(PARAM_DEFAULTS)
    p.update(params)
    unknown = sorted(set(p) - set(PARAM_DEFAULTS) - set(PARAM_REQUIRED))
    if unknown:
        raise ValueError(f"知らない params: {unknown}")
    if not isinstance(p["h1"], bool) or not isinstance(p["rtc_arm"], bool):
        raise ValueError("h1・rtc_arm は真偽")
    if p["plan"] not in ("A", "B"):
        raise ValueError("plan は A か B")
    if p["plan"] == "B" and not p["rtc_arm"]:
        raise ValueError("案 B は RTC の腕（P-3）があるときだけ")
    if int(p["rtc_p1_n"]) != (100 if p["plan"] == "B" else 50):
        raise ValueError("rtc_p1_n は案 A なら 50、案 B なら 100（事前登録の案 第 4 節）")
    if p["guard_mode"] not in ("point", "interval"):
        raise ValueError("guard_mode は point か interval")
    if p["h1"] and p["e7_n"] not in (100, 150, 200):
        raise ValueError("h1 が真なら e7_n（P-4）は 100・150・200 のどれか")
    v = p["versions"]
    if v is not None:
        if not isinstance(v, dict) or set(v) - set(VERSION_KEYS):
            raise ValueError(f"versions は {list(VERSION_KEYS)} の辞書")
        if not isinstance(v.get("ckpt_sha256") or {}, dict):
            raise ValueError("versions.ckpt_sha256 は {モデル: SHA-256}")
    return p


def _entry(x):
    return (x, None) if isinstance(x, str) else (x["cond"], x.get("model"))


def _records(d: pathlib.Path, kind: str) -> list:
    pat = "trial_[0-9][0-9][0-9][0-9].json" if kind == "single" else "run_[0-9][0-9][0-9][0-9].json"
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob(pat))]


def _latency(d: pathlib.Path, meta: dict):
    p = d / f"run_{int(meta['run']):04d}_runtime.json"
    return json.loads(p.read_text(encoding="utf-8")).get("latency") if p.is_file() else None


def _bands() -> dict:
    g = json.loads(GATES.read_text(encoding="utf-8"))
    return {a["id"]: tuple(a["range"]) for a in g["bands"]["allocations"]}


def conditions(layout: dict, p: dict) -> list:
    """[(名前, 種類 single|task, 役, フォルダ, 期待するモデル名, 期待する対の鍵の集合, 期待する印)]。"""
    root = pathlib.Path(layout["root"]) / layout["experiment"]
    out = []
    if p["h1"]:
        seeds = list(range(160000, 160000 + int(p["e7_n"])))
        for arm in ("v3", "cur", "n1v3_v3"):
            if arm in (layout.get("e7") or {}):
                c, m = _entry(layout["e7"][arm])
                out.append((f"e7.{arm}", "task", "e7", root / c, m, {(s, None) for s in seeds},
                            {"v3": arm != "cur"}))
    for layer, arms in (layout.get("p1") or {}).items():
        for side in ("R", "N"):
            c, m = _entry(arms[side])
            out.append((f"p1.{layer}.{side}", "single", "p1", root / c, m, {(s, "*") for s in range(162000, 162100)},
                        {"induce": "P1", "rtc_arm": False}))
    for layer, arms in (layout.get("natural") or {}).items():
        hi = 161065 if layer == "1000" else 161032
        for side in ("R", "N"):
            c, m = _entry(arms[side])
            out.append((f"natural.{layer}.{side}", "single", "natural", root / c, m,
                        {(s, col) for s in range(161000, hi + 1) for col in COLORS}, {"induce": None, "rtc_arm": False}))
    rtc = layout.get("rtc") or {}
    if "natural" in rtc:
        c, m = _entry(rtc["natural"])
        out.append(("rtc.natural", "single", "natural", root / c, m, {(s, col) for s in range(161000, 161066) for col in COLORS},
                    {"induce": None, "rtc_arm": True}))
    if "p1" in rtc:
        c, m = _entry(rtc["p1"])
        out.append(("rtc.p1", "single", "p1", root / c, m, {(s, "*") for s in range(162000, 162000 + int(p["rtc_p1_n"]))},
                    {"induce": "P1", "rtc_arm": True}))
    if "p1_n" in rtc:                                   # 案 B だけ: N1v3＋RTC の設定の P1（162000〜162099）
        c, m = _entry(rtc["p1_n"])
        out.append(("rtc.p1_n", "single", "p1", root / c, m, {(s, "*") for s in range(162000, 162100)},
                    {"induce": "P1", "rtc_arm": True}))
    return out


def required_conditions(p: dict) -> list:
    """事前登録の案で回す条件（params で決まる）。1 つでも layout に無ければ未完。"""
    need = []
    for s in ["1000"] + list(p["h2_layers"]):
        need += [f"p1.{s}.R", f"p1.{s}.N", f"natural.{s}.R", f"natural.{s}.N"]
    if p["h1"]:
        need += ["e7.v3", "e7.cur", "e7.n1v3_v3"]
    if p["rtc_arm"]:
        need += ["rtc.natural", "rtc.p1"]
    if p["plan"] == "B":
        need += ["rtc.p1_n"]
    return need


# ================================================================ 入口の点検（0155 の 2 節）
def check_condition(name, kind, role, d: pathlib.Path, model, keys: set, mark: dict, p: dict, experiment: str = None) -> tuple:
    """(満たさない点の文の列, 試行の env.git_head の集合の並び)。"""
    prob = check_condition_items(name, kind, role, d, model, keys, mark, p, experiment)
    heads = sorted({str((r.get("env") or {}).get("git_head")) for r in _records(d, kind)}) if d.is_dir() else []
    return prob, heads


def check_condition_items(name, kind, role, d: pathlib.Path, model, keys: set, mark: dict, p: dict, experiment) -> list:
    prob = []
    if not d.is_dir():
        return [f"{name}: フォルダがない {d}"]
    if not (d / "run.json").is_file():
        prob.append(f"{name}: run.json がない")
        run = {}
    else:
        run = json.loads((d / "run.json").read_text(encoding="utf-8"))
    ga = d / "G_AUDIT.json"
    if not ga.is_file():
        prob.append(f"{name}: G_AUDIT.json がない")
    elif json.loads(ga.read_text(encoding="utf-8")).get("met") is not True:
        prob.append(f"{name}: G_AUDIT.json の met が真でない")
    recs = _records(d, kind)
    got = [(int(r["seed"]), r.get("target") if role == "natural" else ("*" if role == "p1" else None)) for r in recs]
    if len(got) != len(set(got)):
        prob.append(f"{name}: 同じ対の鍵が 2 回ある")
    if set(got) != keys:
        prob.append(f"{name}: 本数・種の集合が計画と違う（記録 {len(got)}、計画 {len(keys)}、"
                    f"足りない {len(keys - set(got))}、余分 {len(set(got) - keys)}）")
    lims = set()
    for r in recs:
        if kind == "single":
            lims.add(float(r.get("time_limit_s", -1)))
        else:
            tl = r.get("time_limits") or {}
            lims.add(tuple(float(tl.get(k, -1)) for k in ("step_timeout_s", "retry", "task_time_limit_s")))
    want = {SINGLE_LIMIT_S} if kind == "single" else {tuple(float(v) for v in E7_LIMITS.values())}
    if recs and lims != want:
        prob.append(f"{name}: 制限時間が計画と違うか 2 種類以上（{sorted(lims, key=str)}）")
    rtl = run.get("time_limits")
    if rtl:
        got_rtl = (float(rtl.get("time_limit_s", -1)) if kind == "single"
                   else tuple(float(rtl.get(k, -1)) for k in ("step_timeout_s", "retry", "task_time_limit_s")))
        if got_rtl not in want:
            prob.append(f"{name}: run.json の制限時間が計画と違う（{got_rtl}）")
    # 環境（0155 の 2-4）: run.json の env_segments がちょうど 1 つ、試行の env（ドライバ・torch・CUDA・OS）も 1 種類
    segs = run.get("env_segments")
    if not isinstance(segs, list) or len(segs) != 1:
        prob.append(f"{name}: run.json の env_segments が 1 つでない（{None if segs is None else len(segs)}。"
                    f"2 つ以上なら区切りごとに分けて出す。自動では判定しない）")
    envs = {json.dumps({k: r["env"].get(k) for k in ENV_KEYS}, sort_keys=True) if isinstance(r.get("env"), dict) else None
            for r in recs}
    if None in envs or len(envs) > 1:
        prob.append(f"{name}: 試行の env（{'・'.join(ENV_KEYS)}）が無いか 2 種類以上")
    # 版（0155 の 2-5）: 全試行に git の HEAD があり、diag の *_sha256 はそれぞれ 1 種類。HEAD が 2 つ以上のときの
    # 子が読み込むファイルの照らし合わせは、条件をまたいで入口が行う
    if any(not (r.get("env") or {}).get("git_head") for r in recs):
        prob.append(f"{name}: env.git_head の無い試行がある")
    shas = {}
    for r in recs:
        for k, v in (r.get("diag") or {}).items():
            if k.endswith("sha256"):
                shas.setdefault(k, set()).add(str(v))
    multi = sorted(k for k, v in shas.items() if len(v) > 1)
    if multi:
        prob.append(f"{name}: diag の SHA-256 が 2 種類以上（{multi}）")
    # 印（0155 の 2-3）: 単発の試行の実験名・条件名がフォルダと同じ
    if kind == "single" and experiment is not None:
        for r in recs:
            if r.get("experiment") != experiment or r.get("condition") != d.name:
                prob.append(f"{name}: 試行の実験名・条件名 {r.get('experiment')!r}/{r.get('condition')!r} がフォルダと違う"
                            f"（seed {r.get('seed')}）")
                break
    for r in recs:
        m = r.get("model")
        mname = m.get("name") if isinstance(m, dict) else m
        if model is not None and mname != model:
            prob.append(f"{name}: モデルの印 {mname!r} が計画 {model!r} と違う（seed {r.get('seed')}）")
            break
    if "induce" in mark:
        kinds = {(r.get("induce") or {}).get("kind") for r in recs}
        if recs and kinds != {mark["induce"]}:
            prob.append(f"{name}: 誘発の印 {sorted(map(str, kinds))} が計画 {mark['induce']!r} と違う")
    if "v3" in mark:
        has = {("v3" in r) for r in recs}
        if recs and has != {mark["v3"]}:
            prob.append(f"{name}: 実行器の印（v3 の欄）が計画と違う")
    b = _bands()
    lo, hi = {"e7": b["test1_E7"], "natural": b["test1_natural"], "p1": b["test1_P1"]}[role]
    if role == "e7" and p["e7_band_extended"]:
        hi = 160199
    bad = [s for s, _ in got if not (lo <= s <= hi)]
    if bad:
        prob.append(f"{name}: テストの帯 {lo}〜{hi} の外の種 {bad[0]}（{len(bad)} 本）")
    return prob


# ================================================================ 版の照合 (b)（登録版 第 7 節 1、判断の紙の問 5）
def version_needed(conds: list, p: dict) -> bool:
    """params.versions があるか、どれかの試行に入口の印 "t1" があれば照らす。"""
    if p["versions"] is not None:
        return True
    for _, kind, _, d, _, _, _ in conds:
        if d.is_dir() and any("t1" in r for r in _records(d, kind)):
            return True
    return False


def _one(values: list):
    """全部が同じ 1 つの値ならその値、そうでなければ None（空も None）。"""
    u = {json.dumps(v, sort_keys=True) for v in values}
    return values[0] if len(u) == 1 else None


def version_items(name, kind, d: pathlib.Path, model, mark: dict, p: dict) -> tuple:
    """(満たさない点の文の列, 条件の版の要約)。"""
    recs = _records(d, kind) if d.is_dir() else []
    ver = p["versions"] or {}
    prob, summ = [], {"entry_sha256": None, "ckpt_sha256": None, "executor_v3_sha256": None, "rtc_setting": None}
    if not ver:
        prob.append(f"{name}: params.versions（「枝の確定」の掲示の値）が無い")
    t1s = [r.get("t1") for r in recs]
    if not recs or any(not isinstance(t, dict) for t in t1s):
        prob.append(f"{name}: 起動の入口の印 t1 の無い試行がある")
        return prob, summ
    fs = [t.get("files_sha256") or {} for t in t1s]
    if any(f.get(GATES_KEY) != POSTED_GATES_SHA256 for f in fs):
        prob.append(f"{name}: (b-1) s4_gates.json の SHA-256 が掲示の値（0165）でない試行がある")
    ent = _one([f.get(ENTRY_KEY) for f in fs])
    if ent is None:
        prob.append(f"{name}: (b-5) 入口の SHA-256 が無いか 2 種類以上")
    elif ver.get("entry_sha256") != ent:
        prob.append(f"{name}: (b-5) 入口の SHA-256 が掲示の値と違う")
    summ["entry_sha256"] = ent
    if any(t.get("role") != name for t in t1s):
        prob.append(f"{name}: 入口の役の印（t1.role）が条件と違う")
    path = _one([(t.get("ckpt") or {}).get("path") for t in t1s])
    sha = _one([(t.get("ckpt") or {}).get("sha256") for t in t1s])
    if path is None or path != REGISTERED_CKPT.get(model):
        prob.append(f"{name}: (b-4) 保存点のパスが登録版 第 5 節のもの（{REGISTERED_CKPT.get(model)}）でないか 2 種類以上")
    if sha is None or sha != (ver.get("ckpt_sha256") or {}).get(model):
        prob.append(f"{name}: (b-4) 保存点の SHA-256 が無いか 2 種類以上か、掲示の値と違う")
    summ["ckpt_sha256"] = sha
    if kind == "task":
        b1s = [r.get("b1") or {} for r in recs]
        if mark.get("v3"):
            st_ = _one([(r.get("v3") or {}).get("settings") for r in recs])
            if st_ is None or any("v3" not in r for r in recs):
                prob.append(f"{name}: (b-2) v3.settings が無いか 2 種類以上")
            if any(b.get("executor_version") != V3_VERSION for b in b1s):
                prob.append(f"{name}: (b-2) b1.executor_version が {V3_VERSION} でない試行がある")
            ex = _one([(b.get("files_sha256") or {}).get(EXEC_V3_KEY) for b in b1s])
            if ex is None or ex != ver.get("executor_v3_sha256"):
                prob.append(f"{name}: (b-2) executor_v3.py の SHA-256 が無いか 2 種類以上か、掲示の値と違う")
            summ["executor_v3_sha256"] = ex
        elif any(b.get("executor_version") != "current" for b in b1s):
            prob.append(f"{name}: (b-2) 今の実行器の腕の b1.executor_version が current でない")
    elif mark.get("rtc_arm"):
        rs = _one([(t.get("rtc") or {}).get("setting") for t in t1s])
        if rs is None or rs != ver.get("rtc_setting") or any((r.get("diag") or {}).get("arm") != rs for r in recs):
            prob.append(f"{name}: (b-3) RTC の設定名が無いか 2 種類以上か、掲示の値と違う")
        ms = _one([(t.get("rtc") or {}).get("module_sha256") for t in t1s])
        if ms is None or ms != ver.get("rtc_module_sha256"):
            prob.append(f"{name}: (b-3) RTC の設定（rtc.py）の SHA-256 が無いか 2 種類以上か、掲示の値と違う")
        summ["rtc_setting"] = rs
    elif any(t.get("rtc") is not None or (r.get("diag") or {}).get("arm") not in (None, "naive") for t, r in zip(t1s, recs)):
        prob.append(f"{name}: (b-3) naive の条件に RTC の設定の印がある")
    return prob, summ


def version_cross(checks: dict, models: dict) -> list:
    """条件をまたぐ版の照合: 入口の版が全条件で 1 種類、同じモデルの保存点が同じ、v3 の腕の executor_v3.py が 1 種類。"""
    prob = []
    vs = {n: c.get("versions") or {} for n, c in checks.items()}
    if len({v.get("entry_sha256") for v in vs.values()}) != 1:
        prob.append("入口の版（files_sha256）が条件によって違う")
    by = {}
    for n, v in vs.items():
        by.setdefault(models[n], set()).add(v.get("ckpt_sha256"))
    for m, s in sorted(by.items(), key=lambda x: str(x[0])):
        if len(s) != 1:
            prob.append(f"モデル {m} の保存点の SHA-256 が条件によって違う")
    ex = {v["executor_v3_sha256"] for v in vs.values() if v.get("executor_v3_sha256")}
    if len(ex) > 1:
        prob.append("v3 の腕の executor_v3.py の SHA-256 が条件によって違う")
    return prob


# ================================================================ 検定・区間
def binom_two_sided(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else float(st.binomtest(b, n, 0.5, alternative="two-sided").pvalue)


def wilson(k: int, n: int):
    if n == 0:
        return None
    z = float(st.norm.ppf(0.975))
    lo, hi = S.wilson_interval(k, n, z)
    return [lo, hi]


def newcombe_paired(n11, n10, n01, n00):
    d, lo, hi = S.paired_diff_ci(n11, n10, n01, n00)
    return None if math.isnan(d) else {"diff": d, "ci95": [lo, hi]}


def newcombe_unpaired(k1, n1, k2, n2):
    """独立な 2 つの割合の差の区間（Newcombe 1998a の方法 10、Wilson の区間をつなぐ）。"""
    if n1 == 0 or n2 == 0:
        return None
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return {"diff": d, "ci95": [d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)]}


def holm(ps: dict) -> dict:
    """{名前: p} → 名前ごとの順位・補正後の p（(m−i+1)p(i) の累積の最大、上限 1）・逐次の棄却（p(i) <= α/(m−i+1)、止まった後は全部偽）。"""
    order = sorted(ps, key=lambda k: (ps[k], k))
    m = len(order)
    out, run_max, going = {}, 0.0, True
    for i, k in enumerate(order):
        adj = min(1.0, (m - i) * ps[k])
        run_max = max(run_max, adj)
        going = going and ps[k] <= ALPHA / (m - i) + 1e-15
        out[k] = {"rank": i + 1, "p": ps[k], "p_holm": run_max, "rejected_stepdown": going}
    return {"order": order, "m": m, "by": out}


# ================================================================ 数える
def _est_before(r, L) -> bool:
    ind = r.get("induce") or {}
    t = ind.get("t_established")
    return bool(ind.get("established")) and t is not None and float(t) < L - EPS


def _succ_by(r, L) -> bool:
    t = r.get("t_success")
    return bool(r.get("success")) and t is not None and float(t) <= L + EPS


def p1_layer(dR, dN, L) -> dict:
    """同じ種で両方とも誘発が L より前に成立した組。b = R だけ復帰、c = N だけ復帰。"""
    R = {int(r["seed"]): r for r in _records(dR, "single")}
    N = {int(r["seed"]): r for r in _records(dN, "single")}
    pairs = [(R[s], N[s]) for s in sorted(set(R) & set(N)) if _est_before(R[s], L) and _est_before(N[s], L)]
    rr = [(_succ_by(a, L), _succ_by(b, L)) for a, b in pairs]
    return {"pairs": len(pairs), "b": sum(a and not b for a, b in rr), "c": sum(b and not a for a, b in rr),
            "both": sum(a and b for a, b in rr), "neither": sum(not a and not b for a, b in rr)}


def recovery_rate(d, L) -> dict:
    recs = [r for r in _records(d, "single") if _est_before(r, L)]
    k = sum(_succ_by(r, L) for r in recs)
    return {"k": k, "n": len(recs), "rate": k / len(recs) if recs else None, "wilson95": wilson(k, len(recs)),
            "t_success_median": float(np.median([float(r["t_success"]) for r in recs if _succ_by(r, L)]))
            if k else None}


def natural_pairs(dA, dB, L) -> dict:
    A = {(int(r["seed"]), r["target"]): r for r in _records(dA, "single")}
    B = {(int(r["seed"]), r["target"]): r for r in _records(dB, "single")}
    common = sorted(set(A) & set(B))
    xy = [(_succ_by(A[k], L), _succ_by(B[k], L)) for k in common]
    n11 = sum(a and b for a, b in xy)
    n10 = sum(a and not b for a, b in xy)
    n01 = sum(b and not a for a, b in xy)
    n00 = len(xy) - n11 - n10 - n01
    ka, kb = n11 + n10, n11 + n01
    return {"pairs": len(xy), "a_k": ka, "b_k": kb, "a_wilson95": wilson(ka, len(xy)), "b_wilson95": wilson(kb, len(xy)),
            "a_only": n10, "b_only": n01, "mcnemar_p": binom_two_sided(n10, n01), "newcombe_a_minus_b": newcombe_paired(n11, n10, n01, n00)}


def e7_pairs(dA, dB) -> dict:
    A = {int(r["seed"]): bool(r.get("all_three_in_box")) for r in _records(dA, "task")}
    B = {int(r["seed"]): bool(r.get("all_three_in_box")) for r in _records(dB, "task")}
    common = sorted(set(A) & set(B))
    n11 = sum(A[s] and B[s] for s in common)
    n10 = sum(A[s] and not B[s] for s in common)
    n01 = sum(B[s] and not A[s] for s in common)
    n00 = len(common) - n11 - n10 - n01
    return {"pairs": len(common), "b": n10, "c": n01, "both": n11, "neither": n00,
            "newcombe_a_minus_b": newcombe_paired(n11, n10, n01, n00)}


def _m56():
    spec = importlib.util.spec_from_file_location("intervention_s3_test1a", ROOT / "scripts" / "56_intervention_s3.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def e7_arm(d: pathlib.Path) -> dict:
    m56 = _m56()
    recs = _records(d, "task")
    rows = [m56.count_run(r, _latency(d, r)) for r in recs]
    n = len(rows)
    k = sum(r["success"] for r in rows)
    t_sum = sum(r["t_end"] for r in rows)
    t3 = [r["t_all_three"] for r in rows if r["success"] and r["t_all_three"] is not None]
    return {"n": n, "all_three": k, "wilson95": wilson(k, n), "judged_all_done": sum(r["judged_all_done"] for r in rows),
            "mean_stored": (sum(r["n_in_box"] for r in rows) / n) if n else None,
            "success_at_k": {str(kk): sum(1 for r in rows if r["success"] and r["total"] <= kk) for kk in (0, 1, 2)},
            "interventions": {t: sum(r["interventions"][t] for r in rows) for t in m56.KINDS},
            "llm_calls": sum(r["llm_calls"] for r in rows), "timed_out": sum(r["timed_out"] for r in rows),
            "success_per_sim_hour": (k / (t_sum / 3600.0)) if t_sum > 0 else None,
            "t_all_three_median": float(np.median(t3)) if t3 else None}


# ================================================================ 本体
def analyze(layout: dict, params: dict) -> dict:
    p = params_with_defaults(params)
    conds = conditions(layout, p)
    checks = {}
    vneed = version_needed(conds, p)
    for name, kind, role, d, model, keys, mark in conds:
        pr, heads = check_condition(name, kind, role, d, model, keys, mark, p, layout["experiment"])
        checks[name] = {"ok": not pr, "problems": pr, "git_heads": heads}
        if vneed:
            vp, vs = version_items(name, kind, d, model, mark, p)
            checks[name].update(ok=not pr and not vp, problems=pr + vp, versions=vs)
    vc = {"done": vneed, "ok": None, "problems": []}
    if vneed and checks:
        vc["problems"] = version_cross(checks, {n: m for n, _, _, _, m, _, _ in conds})
        vc["ok"] = not vc["problems"] and all(c["ok"] for c in checks.values())
    missing = [x for x in required_conditions(p) if x not in checks]
    status = "complete" if all(c["ok"] for c in checks.values()) and not missing and not vc["problems"] else "incomplete"
    res = {"schema": SCHEMA, "status": status, "params": p, "checks": checks, "missing_conditions": missing, "version_check": vc,
           "primary": None, "holm": None, "secondary": None, "face_switch": None}
    if status != "complete":
        return res
    dirs = {name: d for name, _, _, d, _, _, _ in conds}
    prim, ps = {}, {}
    if p["h1"]:
        h1 = e7_pairs(dirs["e7.v3"], dirs["e7.cur"])
        h1.pop("newcombe_a_minus_b")
        h1["p"] = binom_two_sided(h1["b"], h1["c"])
        prim["H1"], ps["H1"] = h1, h1["p"]
    layers = {s: p1_layer(dirs[f"p1.{s}.R"], dirs[f"p1.{s}.N"], L_MAIN) for s in p["h2_layers"]}
    b, c = sum(x["b"] for x in layers.values()), sum(x["c"] for x in layers.values())
    prim["H2"] = {"layers": layers, "pairs": sum(x["pairs"] for x in layers.values()), "b": b, "c": c, "p": binom_two_sided(b, c)}
    ps["H2"] = prim["H2"]["p"]
    h3_arms = ["p1.1000.R", "p1.1000.N"] if p["plan"] == "A" else ["rtc.p1", "rtc.p1_n"]
    h3 = p1_layer(dirs[h3_arms[0]], dirs[h3_arms[1]], L_MAIN)
    h3["arms"] = h3_arms
    h3["p"] = binom_two_sided(h3["b"], h3["c"])
    prim["H3"], ps["H3"] = h3, h3["p"]
    hm = holm(ps)
    for k, v in prim.items():
        v["p_holm"] = hm["by"][k]["p_holm"]
        v["established"] = bool(v["p_holm"] < ALPHA and v["b"] > v["c"])
    res["primary"], res["holm"] = prim, hm
    res["secondary"] = secondary(dirs, p)
    res["face_switch"] = face_switch(prim, dirs, p)
    return res


def secondary(dirs: dict, p: dict) -> dict:
    sec = {"e7_arms": {}, "e7_r_vs_n": None, "p1_rates": {}, "p1_layers_by_limit": {}, "natural": {}, "rtc": None}
    for arm in ("v3", "cur", "n1v3_v3"):
        if f"e7.{arm}" in dirs:
            sec["e7_arms"][arm] = e7_arm(dirs[f"e7.{arm}"])
    if "e7.v3" in dirs and "e7.n1v3_v3" in dirs:
        x = e7_pairs(dirs["e7.v3"], dirs["e7.n1v3_v3"])
        x["mcnemar_p"] = binom_two_sided(x["b"], x["c"])
        sec["e7_r_vs_n"] = x
    for name in sorted(d for d in dirs if d.startswith("p1.") or d.startswith("rtc.p1")):
        sec["p1_rates"][name] = {f"{L:g}": recovery_rate(dirs[name], L) for L in (L_MAIN, L_DESC, L_SUB)}
    layers = sorted({n.split(".")[1] for n in dirs if n.startswith("p1.")})
    for L in (L_MAIN, L_DESC, L_SUB):
        tab = {s: p1_layer(dirs[f"p1.{s}.R"], dirs[f"p1.{s}.N"], L) for s in layers}
        b, c = sum(x["b"] for x in tab.values()), sum(x["c"] for x in tab.values())
        sec["p1_layers_by_limit"][f"{L:g}"] = {"layers": tab, "b": b, "c": c, "p_all_layers": binom_two_sided(b, c)}
    for s in sorted({n.split(".")[1] for n in dirs if n.startswith("natural.")}):
        sec["natural"][s] = {f"{L:g}": natural_pairs(dirs[f"natural.{s}.R"], dirs[f"natural.{s}.N"], L) for L in (L_MAIN, L_SUB)}
    if "rtc.natural" in dirs and "natural.1000.R" in dirs:
        ni = natural_pairs(dirs["rtc.natural"], dirs["natural.1000.R"], L_MAIN)
        lo = (ni["newcombe_a_minus_b"] or {}).get("ci95", [None])[0]
        sec["rtc"] = {"natural_vs_naive": ni, "ni_margin": p["ni_margin"],
                      "non_inferior": None if lo is None else bool(lo > -p["ni_margin"])}
        if "rtc.p1" in dirs and "p1.1000.R" in dirs:
            a, bb = recovery_rate(dirs["rtc.p1"], L_MAIN), recovery_rate(dirs["p1.1000.R"], L_MAIN)
            sec["rtc"]["p1_vs_naive"] = {"rtc": a, "naive": bb, "diff": newcombe_unpaired(a["k"], a["n"], bb["k"], bb["n"])}
    return sec


def face_switch(prim: dict, dirs: dict, p: dict) -> dict:
    """掲示板 0157 の条件 1〜3（事前登録の案 第 12 節）。条件 4（日程）は params.c4_on_time（人が確かめる）。"""
    c1 = {"by": "H3", "value": prim["H3"]["established"], "pass": prim["H3"]["established"]}
    if p["plan"] == "A":
        base_n, base_p = dirs["natural.1000.R"], dirs["p1.1000.R"]
        imp_n, imp_p = base_n, base_p                       # 案 A: 単発の試行の改良版の腕は段階 3 の構成と同じ記録
    else:
        base_n, base_p, imp_n, imp_p = dirs["natural.1000.R"], dirs["p1.1000.R"], dirs["rtc.natural"], dirs["rtc.p1"]
    gn = natural_pairs(imp_n, base_n, L_MAIN)
    ra, rb = recovery_rate(imp_p, L_MAIN), recovery_rate(base_p, L_MAIN)
    gu = newcombe_unpaired(ra["k"], ra["n"], rb["k"], rb["n"])          # 各腕の分母の差（記述）
    gq = p1_layer(imp_p, base_p, L_MAIN)                                 # 同じ種の対（b＝改良版だけ復帰、c＝段階 3 の構成だけ）
    gpn = newcombe_paired(gq["both"], gq["b"], gq["c"], gq["neither"])
    p1_diff = (gq["b"] - gq["c"]) / gq["pairs"] if gq["pairs"] else None
    nat_diff = (gn["a_k"] - gn["b_k"]) / gn["pairs"] if gn["pairs"] else None
    if p["guard_mode"] == "point":
        g_nat = nat_diff is not None and nat_diff >= -0.05 - EPS
        g_p1 = p1_diff is not None and p1_diff >= -0.10 - EPS
    else:
        g_nat = gn["newcombe_a_minus_b"] is not None and gn["newcombe_a_minus_b"]["ci95"][0] >= -0.05 - EPS
        g_p1 = gpn is not None and gpn["ci95"][0] >= -0.10 - EPS
    if p["plan"] == "A":                                # 改良版と段階 3 の構成が同じ記録: 構造上 0 で満たす（登録版 12-3）
        g_p1 = True
    c2 = {"plan": p["plan"], "mode": p["guard_mode"], "g_nat_diff": nat_diff, "g_nat_ci95": (gn["newcombe_a_minus_b"] or {}).get("ci95"),
          "g_p1_count": "structural" if p["plan"] == "A" else "paired",
          "g_p1_pairs": gq["pairs"], "g_p1_b": gq["b"], "g_p1_c": gq["c"], "g_p1_diff": p1_diff,
          "g_p1_ci95": None if gpn is None else gpn["ci95"],
          "g_p1_unpaired": {"imp_k": ra["k"], "imp_n": ra["n"], "base_k": rb["k"], "base_n": rb["n"],
                            "diff": None if gu is None else gu["diff"], "ci95": None if gu is None else gu["ci95"]},
          "g_nat": bool(g_nat), "g_p1": bool(g_p1), "pass": bool(g_nat and g_p1)}
    h1 = prim.get("H1")
    c3 = {"by": "H1", "present": h1 is not None, "pass": bool(h1 is not None and h1["established"])}
    c4 = {"by": "params.c4_on_time（人が確かめる）", "pass": p["c4_on_time"]}
    switch = None if c4["pass"] is None else bool(c1["pass"] and c2["pass"] and c3["pass"] and c4["pass"])
    return {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c1_to_c3": bool(c1["pass"] and c2["pass"] and c3["pass"]), "switch": switch}


# ================================================================ 要約（Markdown）
def summary_md(res: dict) -> str:
    o = ["# テスト 1 の判定（1 枚）", "", f"- 状態: {'そろった（判定した）' if res['status'] == 'complete' else '未完（判定しない）'}"]
    vc = res.get("version_check") or {}
    o.append("- 版の照合 (b): " + ("照らしていない（params.versions も入口の印 t1 も無い）" if not vc.get("done")
                                  else ("満たす" if vc.get("ok") else "満たさない")))
    if res["status"] != "complete":
        o += ["", "## 入口の点検で満たさなかったもの", ""]
        o += [f"- {pr}" for c in res["checks"].values() for pr in c["problems"]]
        o += [f"- 条件がない: {x}" for x in res["missing_conditions"]]
        o += [f"- 版の照合（条件をまたぐ）: {x}" for x in vc.get("problems") or []]
        o += [f"- 条件をまたぐ点検: {x}" for x in (res.get("entry_audit") or {}).get("problems", [])]
        return "\n".join(o) + "\n"
    o += ["", "## 主要評価項目（Holm、α = 0.05）", "", "| 項目 | 組 | b | c | p | Holm 補正後の p | 成立 |", "|---|---|---|---|---|---|---|"]
    for k in ("H1", "H2", "H3"):
        v = res["primary"].get(k)
        if v is None:
            o.append(f"| {k} | — | — | — | — | — | 置かない（P-1） |")
            continue
        o.append(f"| {k} | {v['pairs']} | {v['b']} | {v['c']} | {v['p']:.6g} | {v['p_holm']:.6g} | {'○' if v['established'] else '×'} |")
    o += ["", "- 分母: H1 は同じ種で両方の腕の記録がそろった種。H2・H3 は同じ種で R・N の両方とも誘発が 30 s より前に成立した組"
          "（成功は 30 s 以内）。b は前の腕（v3・R）だけ、c は後の腕だけ。",
          f"- H3 の腕: {' 対 '.join(res['primary']['H3']['arms'])}（案 {res['params']['plan']}）。"]
    fs = res["face_switch"]
    o += ["", "## 掲示板 0157 の条件", "",
          f"- 条件 1（H3）: {'満たす' if fs['c1']['pass'] else '満たさない'}",
          f"- 条件 2（守り、案 {fs['c2']['plan']}・{fs['c2']['mode']}）: {'満たす' if fs['c2']['pass'] else '満たさない'}"
          f"（自然の差 {fs['c2']['g_nat_diff']}、把持失敗の復帰の差 {fs['c2']['g_p1_diff']}"
          f"{'（構造上 0）' if fs['c2']['g_p1_count'] == 'structural' else '（同じ種の対 ' + str(fs['c2']['g_p1_pairs']) + ' 組で数えた）'}）",
          f"- 条件 3（H1）: {'満たす' if fs['c3']['pass'] else '満たさない'}",
          f"- 条件 4（日程）: {'人が確かめる' if fs['c4']['pass'] is None else ('満たす' if fs['c4']['pass'] else '満たさない')}",
          f"- 切り替え: {'条件 4 の確かめ待ち' if fs['switch'] is None else ('切り替える' if fs['switch'] else '切り替えない')}"]
    return "\n".join(o) + "\n"
