"""段階 4 束 2 の採否（B1 実行器 v3・B2 RTC の設定）を、採る条件に機械的に当てはめる。数え方は 2 通り（別のコードの経路）で、
件数が完全に一致しなければ止まる（目標書_段階4.md 第 3 節 6。中央値は ±1e-6 の相対差まで）。

使い方（読むだけ。GPU・シミュレーションは使わない）:
    .venv\\Scripts\\python.exe scripts\\98_s4_b_decide.py b1 [--experiment S4B1] [--out outputs\\s4\\b_decide\\b1.json --md ...b1.md]
    .venv\\Scripts\\python.exe scripts\\98_s4_b_decide.py b2 --settings range10_cap5,ZEROS [--experiment S4B2 --drtc-experiment S4DRTC]
        （--settings の代わりに --gate1 outputs\\s4\\gate1_result.json で関門 1 の判定から読む）
    smoke（実験名が S4SMOKE で始まるときだけ --smoke）: 本数・種を計画と照らさない（比べる腕どうしの一致は照らす）。評価の数字にしない。
終了コード: 0 判定した（採る・採らないのどちらも）/ 1 2 通りの数え方が一致しない（判定しない）/ 2 記録を読めない /
  3 未完（解析の入口の点検を満たさない、または 96_s4_resume.py の採点の直しがまだ入っていない。判定しない。何も書かない）。

解析の入口の点検（掲示板 0155 の 2。判定の前に全条件で。1 つでも満たさなければ「未完」で止まる）:
  1. 条件のフォルダに run.json と G_AUDIT.json（met が真）がある。ただし B1 は G_AUDIT.json があれば met が偽でも止めず、
     違反は c4 で数えて「採らない」にする（手順書 第 4 節「c3・c4 を満たさない場合は、採らないうえに、原因を調べて掲示する」）。
     B2 は met が真でなければ未完（手順書 第 7 節「B2 の条件にも、違反のある記録は使わない」）。
  2. 本数と種（B2 は種と目標の色）の並びが計画と完全に一致する: B1 は R1v3_cur・R1v3_v3 が 191100〜191159 の 60 本、N1v3_v3 が
     191100〜191129 の 30 本（s4_gates.json の bundle2_confirm）。B2 は設定ごとに S4DRTC\\<設定> 30（D_RTC の natural:190200:10）、
     S4B2\\<設定>_ext 69（bundle2_rtc_ext）、S4B2\\<設定>_P1 50（bundle2_p1、induced）。
  3. 比べる腕どうしで種の並びが同じ（B1: R1v3_cur と R1v3_v3 が同じ、N1v3_v3 はその先頭。B2: 部分ごとに naive と同じ）。
  4. 制限時間が 1 種類（腕・設定をまたいでも 1 種類）、run.json の env_segments が 1 つ。
  5. 試行のラベルがフォルダと合う: B1 は meta.b1.arm と meta.b1.executor（今の実行器の腕に v3 の記録がない、v3 の腕に v3 の記録が
     ある）。B2 は meta.diag.arm（影でない）と、S4B2 では meta.b2.setting・meta.b2.part。
  6. B1 の v3 の腕（R1v3_v3・N1v3_v3）の v3 の設定（meta.v3.settings）と実行器の版（meta.b1.executor_version）が全試行で 1 つ。
  7. B2 の A の数え方が使う 96_s4_resume.py の score_condition が、誘発の分母を t_established < L（0155 の 1-1）で数えること。
     ちょうど L 秒に成立した合成の記録 1 本を一時フォルダに書いて確かめ、分母に入るなら（まだ <= L）判定しない
     （96 の直し＝枝 s4-scoring-fix は束 1 の後に入れる。直す前の値は使わない）。

採る条件（作者が承認した束 2 の計画。結果を見る前に固定。docs/stage4/bundle2_protocol.md 第 4 節）:
  B1（R1v3_cur と R1v3_v3 は同じ 60 種の対。N1v3_v3 は 30 本）。すべて満たせば v3 を採る:
    c1 3 個とも（真値）: R1v3_v3 の本数 − R1v3_cur の本数 >= 9（/60）
    c2 2 番目の手順の 1 回目の閉じの持ち上がり（分母は 2 番目の手順が始まった試行。s4_gates.json の first_close_lift）:
       R1v3_v3 の割合 − R1v3_cur の割合 >= 0.25
    c3 誤った「完了」（完了の判定が出たのに、その色が試行の中で一度も箱の中で 1 s 静止しなかった＝truth_success_t にない）:
       v3 の腕（R1v3_v3・N1v3_v3）で 0
    c4 監査: G1〜G3 の違反がどれか 1 つでもある試行（監査のない試行を含む）の本数。3 つの腕すべてで 0
    並べて出すもの（判定には使わない。目標書 第 7 節 6・手順書 第 7 節）: N1v3_v3 の腕、success@0（介入 0 回で成功）、介入の種類別の数・
    時間（scripts/56_intervention_s3.py の数え方）、全体 200 s の打ち切りが効いた試行（timed_out）、計画役の API の回し直し
    （meta.b1.api_retry のある試行）。
  B2（naive と設定 S。自然は S4DRTC\\<S>（30）＋ S4B2\\<S>_ext（69）= 99 試行、把持失敗は S4B2\\<S>_P1 の 50 試行）。
    すべて満たす設定のうち、躍度が最も小さいものを採る（なければ採らない＝naive のまま）:
    c1 自然の成功（30 s の採点）: S の割合 − naive の割合 >= −0.05
    c2 把持失敗からの復帰（30 s の採点。分母は誘発が 30 s より前に成立した試行＝t_established < 30）: S の割合 − naive の割合 >= −0.10
    c3 躍度（自然の試行の、指令の手先 x_des を 10 Hz にした 3 階差分の RMS [m/s³]。最初の 30 s に限った値の中央値を主、
       記録全体の値を副に出す）: S の中央値 < naive の中央値
2 通りの数え方（ファイルの列挙・中央値もそれぞれ別に実装し、共有しない）:
  A: ファイルの列挙は os.scandir と正規表現、中央値は statistics.median。diag/e7.py の summarize_condition（3 個とも・持ち上がり・
     timed_out）、96_s4_resume.py の score_condition（30 s の採点）、eval/metrics.py の jerk_rms（躍度）、meta の steps を手順ごとに
     回す誤った完了、試行ごとの違反の数の和 > 0 で数える監査。
  B: ファイルの列挙は pathlib の glob、中央値は並べて真ん中を取る自前の実装。final_in_box の値、npz の plan_step で 2 番目の手順の
     こまを選ぶ持ち上がり、meta の audit の項目ごとの真偽の「または」で数える監査、試行の json の success・t_success・induce を直接
     読む採点（t_established < L）、こまの間隔から刻みを決める躍度、集合で数える誤った完了。
"""
import argparse
import importlib.util
import json
import math
import os
import pathlib
import re
import statistics
import sys
import tempfile
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
GATES = ROOT / "configs" / "s4_gates.json"
OUTD = ROOT / "outputs" / "s4" / "b_decide"
B1_ARMS = ("R1v3_cur", "R1v3_v3", "N1v3_v3")
B1_V3_ARMS = ("R1v3_v3", "N1v3_v3")
B1_RULES = {"c1_all_three_gain_min": 9, "c2_lift_gain_min": 0.25, "c3_false_complete_max": 0, "c4_audit_trials_max": 0,
            "c4_definition": "G1〜G3 の違反がどれか 1 つでもある試行（監査のない試行を含む）の本数"}
B2_RULES = {"c1_natural_drop_max": 0.05, "c2_recovery_drop_max": 0.10, "c3_jerk_smaller": True, "horizon_s": 30.0,
            "induced_denominator": "t_established < L（0155 の 1-1）"}
B2_PARTS = ("natural_drtc", "ext", "P1")
EPS = 1e-9
LIFT_M = 0.02


def _load(path, name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _r96():
    """A の数え方が使う 96_s4_resume.py（書き換えない）。"""
    return _load(ROOT / "scripts" / "96_s4_resume.py", "s4_resume_decide")


class Mismatch(RuntimeError):
    pass


class Incomplete(RuntimeError):
    """未完（解析の入口の点検を満たさない・96 の採点の直しが入っていない）。判定しない。"""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def _same(name, a, b, rel=0.0):
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None or (isinstance(a, float) and math.isnan(a)):
            ok = (a is None and b is None) or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))
        else:
            ok = abs(a - b) <= rel * max(1.0, abs(a), abs(b))
    else:
        ok = a == b
    if not ok:
        raise Mismatch(f"{name}: A={a!r} B={b!r}")
    return a


# ================================================================ 計画（s4_gates.json の allocations）
def _alloc(aid: str) -> tuple:
    g = json.loads(GATES.read_text(encoding="utf-8"))
    return tuple(next(a["range"] for a in g["bands"]["allocations"] if a["id"] == aid))


def b1_plan() -> dict:
    """腕 → 種の並び。R1v3 の 2 つの腕は bundle2_confirm の 60 種、N1v3 はその先頭 30 種（191100〜191129）。"""
    lo, hi = _alloc("bundle2_confirm")
    r = list(range(lo, hi + 1))
    return {"R1v3_cur": r, "R1v3_v3": r, "N1v3_v3": list(range(lo, 191130))}


def b2_plan() -> dict:
    """部分 → [(種, 目標の色)]（41_results.trial_list。96 が回す並びと同じ）。"""
    r41 = _load(ROOT / "scripts" / "41_results.py", "r41_decide")
    out = {}
    for part, aid, kind in (("natural_drtc", "D_RTC", "natural"), ("ext", "bundle2_rtc_ext", "natural"), ("P1", "bundle2_p1", "induced")):
        lo, hi = _alloc(aid)
        out[part] = [(s, t) for s, _, t in r41.trial_list(f"{kind}:{lo}:{hi - lo + 1}")]
    return out


# ================================================================ 解析の入口の点検（0155 の 2）
def _limits_key(lim) -> str:
    return json.dumps({k: v for k, v in (lim or {}).items() if k != "source"}, sort_keys=True)


def entry_audit_dir(d: pathlib.Path, kind: str, expect, label, require_met: bool = True) -> tuple:
    """(問題の列, 情報)。kind は "run"（単発）か "task"（3 個の連続タスク）。expect は計画の並び（None なら照らさない）。
    label(meta) は合わなければ理由の文字列。require_met が偽なら G_AUDIT.json は「あって met の欄がある」だけを求める
    （B1: 監査の違反は c4 で数えて「採らない」にする。手順書 第 4 節・第 7 節）。"""
    name = f"{d.parent.name}\\{d.name}"
    if not d.is_dir():
        return [f"{name}: フォルダがない"], None
    pre = "trial" if kind == "run" else "run"
    files = [p for p in sorted(d.glob(f"{pre}_[0-9][0-9][0-9][0-9].json"))]
    metas = [json.loads(p.read_text(encoding="utf-8")) for p in files]
    probs = []
    if not (d / "run.json").is_file():
        probs.append("run.json がない（全部そろっていない）")
    else:
        segs = json.loads((d / "run.json").read_text(encoding="utf-8")).get("env_segments")
        if not isinstance(segs, list) or len(segs) != 1:
            probs.append(f"env_segments が 1 つでない（{len(segs) if isinstance(segs, list) else segs}）")
    ga = d / "G_AUDIT.json"
    met = json.loads(ga.read_text(encoding="utf-8")).get("met") if ga.is_file() else None
    if not isinstance(met, bool) or (require_met and met is not True):
        probs.append("G_AUDIT.json がない、または met が真でない" if require_met else "G_AUDIT.json がない（met の欄がない）")
    got = [(m.get("seed"), m.get("target")) for m in metas] if kind == "run" else [m.get("seed") for m in metas]
    if expect is not None and got != list(expect):
        probs.append(f"本数・種が計画と違う（記録 {len(got)} 本、計画 {len(expect)} 本）")
    lims = {_limits_key(m.get("time_limits")) for m in metas}
    if len(lims) != 1:
        probs.append(f"制限時間が 1 種類でない: {sorted(lims)}")
    bad = [why for why in (label(m) for m in metas) if why]
    if bad:
        probs.append(f"記録のラベルがフォルダと合わない試行 {len(bad)} 本（例 {bad[0]}）")
    return [f"{name}: {x}" for x in probs], {"pairs": got, "limits": lims, "metas": metas}


def b1_entry_audit(base: pathlib.Path, plan) -> dict:
    """B1 の入口の点検。満たさなければ Incomplete。plan が None なら本数・種を計画と照らさない（smoke）。"""
    probs, info = [], {}
    for arm in B1_ARMS:
        want = "v3" if arm in B1_V3_ARMS else "current"

        def label(m, arm=arm, want=want):
            b = m.get("b1") or {}
            if b.get("arm") != arm:
                return f"b1.arm={b.get('arm')!r}"
            if b.get("executor") != want:
                return f"b1.executor={b.get('executor')!r}"
            if want == "v3" and not isinstance(m.get("v3"), dict):
                return "v3 の腕に v3 の記録がない"
            if want == "current" and "v3" in m:
                return "今の実行器の腕に v3 の記録がある"
            return ""
        p, inf = entry_audit_dir(base / arm, "task", None if plan is None else plan[arm], label, require_met=False)
        probs += p
        info[arm] = inf
    if all(info.values()):
        if info["R1v3_cur"]["pairs"] != info["R1v3_v3"]["pairs"]:
            probs.append("R1v3_cur と R1v3_v3 の種の並びが違う")
        n1 = info["N1v3_v3"]["pairs"]
        if n1 != info["R1v3_v3"]["pairs"][:len(n1)]:
            probs.append("N1v3_v3 の種の並びが R1v3 の腕の先頭と違う")
        lims = set().union(*(info[a]["limits"] for a in B1_ARMS))
        if len(lims) != 1:
            probs.append(f"制限時間が腕をまたいで 1 種類でない: {sorted(lims)}")
        v3m = [m for a in B1_V3_ARMS for m in info[a]["metas"]]
        sets = {json.dumps((m.get("v3") or {}).get("settings"), sort_keys=True) for m in v3m}
        vers = {(m.get("b1") or {}).get("executor_version") for m in v3m}
        if len(sets) != 1:
            probs.append(f"v3 の腕の v3 の設定が 1 つでない（{len(sets)} 種類）")
        if len(vers) != 1:
            probs.append(f"v3 の腕の実行器の版が 1 つでない: {sorted(map(str, vers))}")
    if probs:
        raise Incomplete(probs)
    return {arm: {"n": len(info[arm]["pairs"])} for arm in B1_ARMS}


def b2_entry_audit(b2_base: pathlib.Path, drtc_base: pathlib.Path, arms: list, plan) -> dict:
    probs, info = [], {}
    for s in arms:
        for part, d in (("natural_drtc", drtc_base / s), ("ext", b2_base / f"{s}_ext"), ("P1", b2_base / f"{s}_P1")):
            def label(m, s=s, part=part):
                dg = m.get("diag") or {}
                if dg.get("arm") != s or dg.get("shadow"):
                    return f"diag.arm={dg.get('arm')!r} shadow={dg.get('shadow')!r}"
                if part != "natural_drtc":
                    b = m.get("b2") or {}
                    if b.get("setting") != s or b.get("part") != part:
                        return f"b2.setting={b.get('setting')!r} b2.part={b.get('part')!r}"
                return ""
            p, inf = entry_audit_dir(d, "run", None if plan is None else plan[part], label)
            probs += p
            info[(s, part)] = inf
    if all(info.values()):
        for s in arms[1:]:
            for part in B2_PARTS:
                if info[(s, part)]["pairs"] != info[("naive", part)]["pairs"]:
                    probs.append(f"{s} と naive の {part} の種・目標の並びが違う")
        lims = set().union(*(v["limits"] for v in info.values()))
        if len(lims) != 1:
            probs.append(f"制限時間が条件をまたいで 1 種類でない: {sorted(lims)}")
    if probs:
        raise Incomplete(probs)
    return {f"{s}.{part}": len(info[(s, part)]["pairs"]) for s in arms for part in B2_PARTS}


def score_fix_landed(r96) -> bool:
    """96 の score_condition が、ちょうど L 秒に成立した誘発の試行を分母から外すか（0155 の 1-1。t_established < L）。
    合成の記録 1 本を一時フォルダに書いて確かめる（outputs には書かない）。"""
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "trial_0000.json"
        p.write_text(json.dumps({"trial": 0, "seed": 0, "target": "red", "success": True, "t_success": 20.0,
                                 "time_limits": {"time_limit_s": 60.0},
                                 "induce": {"kind": "P1", "established": True, "t_established": 30.0}}), encoding="utf-8")
        s = r96.score_condition(pathlib.Path(td), (30.0,))["at"]["30"]
    return s["n"] == 0


# ================================================================ A の道具（B と共有しない）
_RX_A = {"run": re.compile(r"^run_\d{4}\.json$"), "trial": re.compile(r"^trial_\d{4}\.json$")}


def _files_a(d: pathlib.Path, prefix: str) -> list:
    with os.scandir(d) as it:
        names = sorted(e.name for e in it if e.is_file() and _RX_A[prefix].match(e.name))
    return [d / n for n in names]


def _median_a(xs):
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return statistics.median(vals) if vals else None


# ================================================================ B の道具（A と共有しない）
def _files_b(d: pathlib.Path, prefix: str) -> list:
    k = len(prefix) + 1
    return [p for p in sorted(d.glob(f"{prefix}_*.json")) if len(p.stem) == k + 4 and p.stem[k:].isdigit()]


def _median_b(xs):
    xs = sorted(x for x in xs if x is not None and not math.isnan(x))
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


# ================================================================ B1
def b1_count_a(d: pathlib.Path) -> dict:
    from recovla.diag import e7 as E7
    s = E7.summarize_condition(d)
    false = bad = api = 0
    for p in _files_a(d, "run"):
        m = json.loads(p.read_text(encoding="utf-8"))
        truth = m.get("truth_success_t") or {}
        for st in m.get("steps") or []:
            if st.get("judged_complete") and st["color"] not in truth:
                false += 1
        au = m.get("audit") or {}
        try:
            v = (int(au["g1"]["violations"]) + int(au["g2"]["world_stops"]) + int(au["g2"]["early_use"])
                 + int(au["g3"]["total_violations"]))
        except (KeyError, TypeError):
            v = 1                                                   # 監査のない試行は違反のある試行に数える
        bad += int(v > 0)
        api += int((m.get("b1") or {}).get("api_retry") is not None)
    lift = s["step2"]["first_close_lift"]
    return {"n": s["n"], "seeds": sorted(int(x) for x in s["per_seed"]), "all_three": s["all_three_true"]["k"],
            "lift_k": lift["k"], "lift_n": lift["n"], "false_complete": false, "audit_bad_trials": bad,
            "timed_out": int(s["timed_out"]), "api_retries": api}


def _first_close_lift_b(z, ci: int, sel: np.ndarray, rest_z: float):
    """sel（こまの真偽）の中で最初に指が閉じたこま（0→1）から、次に開くまでに目標が rest_z + 2 cm を超えたか。閉じなければ None。"""
    gc = np.asarray(z["gripper_closed"]).astype(bool)
    idx = np.flatnonzero(sel)
    for f in idx:
        if f > 0 and gc[f] and not gc[f - 1]:
            g = f + 1
            while g < len(gc) and gc[g]:
                g += 1
            return bool(np.max(np.asarray(z["cube_pos"])[f:g, ci, 2]) - rest_z > LIFT_M)
    return None


def b1_count_b(d: pathlib.Path) -> dict:
    from recovla.common.seeds import COLORS
    from recovla.sim import frames
    out = {"n": 0, "seeds": [], "all_three": 0, "lift_k": 0, "lift_n": 0, "false_complete": 0, "audit_bad_trials": 0,
           "timed_out": 0, "api_retries": 0}
    for p in _files_b(d, "run"):
        m = json.loads(p.read_text(encoding="utf-8"))
        out["n"] += 1
        out["seeds"].append(int(m["seed"]))
        fin = m.get("final_in_box") or {}
        out["all_three"] += int(len(fin) == 3 and all(bool(v) for v in fin.values()))
        steps = m.get("steps") or []
        if len(steps) >= 2:                                         # 2 番目の手順が始まった試行
            s2 = steps[1]
            out["lift_n"] += 1
            z = np.load(p.with_suffix(".npz"))
            t = np.asarray(z["sim_time"])
            att = s2["attempts"]
            t_hi = att[1]["t_begin"] if len(att) > 1 else (s2.get("t_end") if s2.get("t_end") is not None else float(t[-1]) + 1.0)
            sel = (np.asarray(z["plan_step"]) == 1) & (t >= float(att[0]["t_begin"]) - EPS) & (t < float(t_hi) - EPS)
            lf = _first_close_lift_b(z, COLORS.index(s2["color"]), sel, frames.CUBE_REST_Z)
            out["lift_k"] += int(bool(lf))
        judged = {s["color"] for s in steps if s.get("judged_complete") is True}
        out["false_complete"] += len(judged - set((m.get("truth_success_t") or {}).keys()))
        au = m.get("audit") or {}
        bad = (not all(k in au for k in ("g1", "g2", "g3")) or int(au["g1"]["violations"]) > 0
               or int(au["g2"]["world_stops"]) > 0 or int(au["g2"]["early_use"]) > 0 or int(au["g3"]["total_violations"]) > 0)
        out["audit_bad_trials"] += int(bad)
        out["timed_out"] += int(m.get("timed_out") is True)
        out["api_retries"] += int(bool((m.get("b1") or {}).get("api_retry")))
    out["seeds"].sort()
    return out


def b1_count(d: pathlib.Path) -> dict:
    a, b = b1_count_a(d), b1_count_b(d)
    for k in a:
        _same(f"{d.name}.{k}", a[k], b[k])
    return a


def b1_display(d: pathlib.Path) -> dict:
    """並べて出すもの（判定に使わない）: success@k・介入の種類別の数・時間（56 の数え方）。"""
    m56 = _load(ROOT / "scripts" / "56_intervention_s3.py", "intervention_s3_decide")
    rows = m56.load_runs(d)
    res = m56.aggregate(rows)
    return {k: res[k] for k in ("per_type", "interventions_per_run", "success_at_k", "llm", "stopped", "time")}


def b1_decide(base: pathlib.Path, plan=None, smoke: bool = False) -> dict:
    """plan: 腕 → 種の並び（None なら s4_gates.json の計画）。smoke なら本数・種を計画と照らさない。"""
    audit = b1_entry_audit(base, None if smoke else (plan or b1_plan()))
    c = {arm: b1_count(base / arm) for arm in B1_ARMS}
    cur, v3 = c["R1v3_cur"], c["R1v3_v3"]
    if cur["seeds"] != v3["seeds"] or cur["n"] == 0:
        raise ValueError(f"R1v3_cur と R1v3_v3 の種がそろわない（{cur['n']} 本・{v3['n']} 本）")
    rate = lambda x: x["lift_k"] / x["lift_n"] if x["lift_n"] else None          # noqa: E731
    d_all = v3["all_three"] - cur["all_three"]
    d_lift = None if rate(v3) is None or rate(cur) is None else rate(v3) - rate(cur)
    false = c["R1v3_v3"]["false_complete"] + c["N1v3_v3"]["false_complete"]
    bad = sum(x["audit_bad_trials"] for x in c.values())
    conds = {"c1": {"value": d_all, "rule": f">= {B1_RULES['c1_all_three_gain_min']}", "pass": d_all >= B1_RULES["c1_all_three_gain_min"]},
             "c2": {"value": None if d_lift is None else round(d_lift, 4), "rule": f">= {B1_RULES['c2_lift_gain_min']}",
                    "pass": d_lift is not None and d_lift >= B1_RULES["c2_lift_gain_min"] - EPS},
             "c3": {"value": false, "rule": "== 0（v3 の腕）", "pass": false == 0},
             "c4": {"value": bad, "rule": "== 0（3 つの腕。違反のある試行の本数）", "pass": bad == 0}}
    return {"status": "判定した" if not smoke else "smoke（評価の数字にしない）", "entry_audit": audit, "counts": c,
            "conditions": conds, "adopt_v3": all(x["pass"] for x in conds.values()),
            "display": {arm: b1_display(base / arm) for arm in B1_ARMS}}


# ================================================================ B2
def _jerk_a(z, horizon_s):
    from recovla.eval import metrics as M
    t = np.asarray(z["sim_time"], float)
    k = np.abs(t * 10.0 - np.round(t * 10.0)) < 1e-6                 # 10 Hz のこま（時刻が 0.1 s の倍数）
    if horizon_s is not None:
        k &= t <= horizon_s + EPS
    return M.jerk_rms(np.asarray(z["x_des"], float)[k], 0.1)


def _jerk_b(z, horizon_s):
    t = np.asarray(z["sim_time"], float)
    x = np.asarray(z["x_des"], float)
    dt = float(np.median(np.diff(t)))
    stride = int(round(0.1 / dt))
    first = int(np.argmin(np.abs(t[:stride] - np.round(t[:stride] * 10.0) / 10.0)))
    idx = list(range(first, len(t), stride))
    if horizon_s is not None:
        idx = [i for i in idx if t[i] <= horizon_s + EPS]
    if len(idx) < 4:
        return math.nan
    xs = x[idx]
    vals = []
    for i in range(3, len(xs)):
        j = (xs[i] - 3 * xs[i - 1] + 3 * xs[i - 2] - xs[i - 3]) / 0.1 ** 3
        n = float(np.sqrt(np.sum(j ** 2)))
        if math.isfinite(n):
            vals.append(n * n)
    return float(math.sqrt(sum(vals) / len(vals))) if vals else math.nan


def b2_count_a(dirs: list, horizon: float) -> dict:
    r96 = _r96()
    k = n = n_tr = 0
    jerks, jerks_all = [], []
    for d in dirs:
        s = r96.score_condition(d, (horizon,))["at"][f"{horizon:g}"]
        k, n, n_tr = k + s["successes"], n + s["n"], n_tr + s["n_trials"]
        for p in _files_a(d, "trial"):
            z = np.load(p.with_suffix(".npz"))
            jerks.append(_jerk_a(z, horizon))
            jerks_all.append(_jerk_a(z, None))
    return {"k": k, "n": n, "n_trials": n_tr, "jerk_median": _median_a(jerks), "jerk_median_all": _median_a(jerks_all)}


def b2_count_b(dirs: list, horizon: float) -> dict:
    k = n = n_tr = 0
    jerks, jerks_all = [], []
    for d in dirs:
        for p in _files_b(d, "trial"):
            m = json.loads(p.read_text(encoding="utf-8"))
            n_tr += 1
            ind = m.get("induce") or {}
            if ind.get("kind"):
                te = ind.get("t_established")
                if not (ind.get("established") and te is not None and float(te) < horizon - EPS):   # 0155 の 1-1: L より前に成立
                    continue
            n += 1
            ts = m.get("t_success")
            k += int(bool(m.get("success")) and ts is not None and float(ts) <= horizon + EPS)
            z = np.load(p.with_suffix(".npz"))
            jerks.append(_jerk_b(z, horizon))
            jerks_all.append(_jerk_b(z, None))
    return {"k": k, "n": n, "n_trials": n_tr, "jerk_median": _median_b(jerks), "jerk_median_all": _median_b(jerks_all)}


def b2_count(name: str, dirs: list, horizon: float, with_jerk: bool) -> dict:
    a, b = b2_count_a(dirs, horizon), b2_count_b(dirs, horizon)
    for key in ("k", "n", "n_trials"):
        _same(f"{name}.{key}", a[key], b[key])
    if with_jerk:
        for key in ("jerk_median", "jerk_median_all"):
            _same(f"{name}.{key}", a[key], b[key], rel=1e-6)
    else:
        a.pop("jerk_median"), a.pop("jerk_median_all")
    return a


def b2_decide(b2_base: pathlib.Path, drtc_base: pathlib.Path, settings: list, plan=None, smoke: bool = False) -> dict:
    """plan: 部分 → [(種, 目標)]（None なら s4_gates.json の計画）。smoke なら本数・種を計画と照らさない。"""
    h = B2_RULES["horizon_s"]
    arms = ["naive"] + [s for s in settings if s != "naive"]
    probs = []
    try:
        audit = b2_entry_audit(b2_base, drtc_base, arms, None if smoke else (plan or b2_plan()))
    except Incomplete as e:
        probs += e.problems
    if not score_fix_landed(_r96()):
        probs.append("96_s4_resume.py の score_condition が、まだ誘発の分母を t_established <= L で数える（0155 の 1-1 は < L）。"
                     "枝 s4-scoring-fix を入れてから判定する（直す前の値は使わない）")
    if probs:
        raise Incomplete(probs)
    c = {}
    for s in arms:
        nat_dirs = [drtc_base / s, b2_base / f"{s}_ext"]
        c[s] = {"natural": b2_count(f"{s}.natural", nat_dirs, h, True),
                "recovery": b2_count(f"{s}.P1", [b2_base / f"{s}_P1"], h, False)}
    nv = c["naive"]
    rate = lambda x: x["k"] / x["n"] if x["n"] else None                          # noqa: E731
    res = {}
    for s in arms[1:]:
        x = c[s]
        dn = None if rate(x["natural"]) is None or rate(nv["natural"]) is None else rate(x["natural"]) - rate(nv["natural"])
        dr = None if rate(x["recovery"]) is None or rate(nv["recovery"]) is None else rate(x["recovery"]) - rate(nv["recovery"])
        js, jn = x["natural"]["jerk_median"], nv["natural"]["jerk_median"]
        conds = {"c1": {"value": None if dn is None else round(dn, 4), "rule": f">= -{B2_RULES['c1_natural_drop_max']}",
                        "pass": dn is not None and dn >= -B2_RULES["c1_natural_drop_max"] - EPS},
                 "c2": {"value": None if dr is None else round(dr, 4), "rule": f">= -{B2_RULES['c2_recovery_drop_max']}",
                        "pass": dr is not None and dr >= -B2_RULES["c2_recovery_drop_max"] - EPS},
                 "c3": {"value": [js, jn], "rule": "設定の躍度の中央値 < naive の中央値（最初の 30 s）",
                        "pass": js is not None and jn is not None and js < jn}}
        res[s] = {"conditions": conds, "pass": all(v["pass"] for v in conds.values())}
    passing = [s for s in res if res[s]["pass"]]
    adopt = min(passing, key=lambda s: c[s]["natural"]["jerk_median"]) if passing else None
    return {"status": "判定した" if not smoke else "smoke（評価の数字にしない）", "entry_audit": audit, "counts": c, "settings": res,
            "adopt": adopt, "adopt_note": "条件を満たす中で躍度が最も小さい設定。なければ naive のまま"}


# ================================================================ 出力
def md_b1(r: dict) -> str:
    o = ["# 束 2 B1（実行器 v3）の採否", "", f"- 状態: {r['status']}", f"- 結論: {'v3 を採る' if r['adopt_v3'] else 'v3 を採らない'}", "",
         "| 条件 | 値 | 基準 | 満たす |", "|---|---|---|---|"]
    for k, v in r["conditions"].items():
        o.append(f"| {k} | {v['value']} | {v['rule']} | {'○' if v['pass'] else '×'} |")
    o += ["", "| 腕 | 本数 | 3 個とも | 2 番目の 1 回目の持ち上がり | 誤った完了 | 監査の違反のある試行 | success@0 | 介入/本 | "
          "200 s の打ち切り（timed_out） | API の回し直し |", "|---|---|---|---|---|---|---|---|---|---|"]
    for arm, x in r["counts"].items():
        dsp = r["display"][arm]
        o.append(f"| {arm} | {x['n']} | {x['all_three']} | {x['lift_k']}/{x['lift_n']} | {x['false_complete']} | {x['audit_bad_trials']} | "
                 f"{dsp['success_at_k']['0']['k']}/{dsp['success_at_k']['0']['n']} | {dsp['interventions_per_run']} | "
                 f"{x['timed_out']} | {x['api_retries']} |")
    o += ["", "介入の種類別の数（56_intervention_s3.py の数え方。判定には使わない）:", ""]
    for arm in r["counts"]:
        o.append(f"- {arm}: " + "、".join(f"{v['label']} {v['total']}" for v in r["display"][arm]["per_type"].values()))
    n_to = sum(x["timed_out"] for x in r["counts"].values())
    o += ["", f"全体 200 s の打ち切りが効いた試行: {n_to} 本（1 本でもあれば手順書 第 12 節に逸脱として書く）。",
          "", "解析の入口の点検（0155 の 2）を満たし、2 通りの数え方（A・B）は件数が完全に一致した。"]
    return "\n".join(o) + "\n"


def md_b2(r: dict) -> str:
    o = ["# 束 2 B2（RTC の設定）の採否", "", f"- 状態: {r['status']}", f"- 結論: {r['adopt'] or '採る設定なし（naive のまま）'}", "",
         "| 設定 | 自然（30 s） | 把持失敗の復帰（30 s） | 躍度の中央値（30 s / 全体） | c1 | c2 | c3 | 採れる |", "|---|---|---|---|---|---|---|---|"]
    for s, x in r["counts"].items():
        st = r["settings"].get(s)
        flag = (lambda k: "○" if st["conditions"][k]["pass"] else "×") if st else (lambda k: "—")
        jm, ja = x["natural"]["jerk_median"], x["natural"]["jerk_median_all"]
        o.append(f"| {s} | {x['natural']['k']}/{x['natural']['n']} | {x['recovery']['k']}/{x['recovery']['n']} | "
                 f"{'—' if jm is None else f'{jm:.3f}'} / {'—' if ja is None else f'{ja:.3f}'} | {flag('c1')} | {flag('c2')} | {flag('c3')} | "
                 f"{'—' if not st else ('○' if st['pass'] else '×')} |")
    o += ["", "解析の入口の点検（0155 の 2）を満たし、2 通りの数え方（A・B）は件数が完全に一致し、躍度の中央値は相対 1e-6 以内で一致した。"]
    return "\n".join(o) + "\n"


def _write(p, text):
    p = pathlib.Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def settings_from(a) -> list:
    if a.settings:
        return [x for x in a.settings.split(",") if x]
    if a.gate1:
        rtc = json.loads(pathlib.Path(a.gate1).read_text(encoding="utf-8"))["conclusion"]["bundle2"]["rtc"]
        if rtc.get("result") == "undetermined":
            raise ValueError("関門 1 の判定で B2 の設定が判定できない")
        return list(rtc.get("settings") or []) if rtc.get("b2") else []
    raise ValueError("--settings か --gate1 が要る")


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:                                    # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("b1")
    p.add_argument("--experiment", default="S4B1")
    p.add_argument("--root", default=str(ROOT / "outputs" / "v2eval"))
    p.add_argument("--out", default=None)
    p.add_argument("--md", default=None)
    p.add_argument("--smoke", action="store_true", help="S4SMOKE の実験だけ: 本数・種を計画と照らさない（評価の数字にしない）")
    p = sub.add_parser("b2")
    p.add_argument("--experiment", default="S4B2")
    p.add_argument("--drtc-experiment", default="S4DRTC")
    p.add_argument("--settings", default=None)
    p.add_argument("--gate1", default=None)
    p.add_argument("--root", default=str(ROOT / "outputs" / "v2eval"))
    p.add_argument("--out", default=None)
    p.add_argument("--md", default=None)
    p.add_argument("--smoke", action="store_true", help="S4SMOKE の実験だけ: 本数・種を計画と照らさない（評価の数字にしない）")
    a = ap.parse_args(argv)
    root = pathlib.Path(a.root)
    if a.smoke and not a.experiment.startswith("S4SMOKE"):
        print(f"--smoke は実験名が S4SMOKE で始まるときだけ（{a.experiment}）", file=sys.stderr)
        return 2
    try:
        if a.cmd == "b1":
            r = b1_decide(root / a.experiment, smoke=a.smoke)
            md = md_b1(r)
        else:
            st = settings_from(a)
            if not st:
                r = {"settings": {}, "adopt": None, "note": "関門 R で B2 に送る設定がない（B2 は回していない）"}
                md = "# 束 2 B2（RTC の設定）の採否\n\n- 関門 R で B2 に送る設定がない（B2 は回していない）\n"
            else:
                r = b2_decide(root / a.experiment, root / a.drtc_experiment, st, smoke=a.smoke)
                md = md_b2(r)
    except Incomplete as e:
        print("未完（判定しない。何も書かない）:", file=sys.stderr)
        for x in e.problems:
            print(f"  - {x}", file=sys.stderr)
        print(json.dumps({"status": "未完", "problems": e.problems}, ensure_ascii=False, indent=1))
        return 3
    except Mismatch as e:
        print(f"2 通りの数え方が一致しない（判定しない）: {e}", file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError) as e:
        print(f"記録を読めない・そろわない: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    r = dict(r, rules=B1_RULES if a.cmd == "b1" else B2_RULES, written=time.strftime("%Y-%m-%d %H:%M:%S"),
             double_count="A と B の件数が完全に一致（躍度の中央値は相対 1e-6 以内）")
    text = json.dumps(r, ensure_ascii=False, indent=1, default=float)
    _write(a.out or OUTD / f"{a.cmd}_{a.experiment}.json", text)
    _write(a.md or OUTD / f"{a.cmd}_{a.experiment}.md", md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
