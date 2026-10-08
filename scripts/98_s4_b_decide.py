"""段階 4 束 2 の採否（B1 実行器 v3・B2 RTC の設定）を、採る条件に機械的に当てはめる。数え方は 2 通り（別のコードの経路）で、
件数が完全に一致しなければ止まる（目標書_段階4.md 第 3 節 6。中央値は ±1e-6 の相対差まで）。

使い方（読むだけ。GPU・シミュレーションは使わない）:
    .venv\\Scripts\\python.exe scripts\\98_s4_b_decide.py b1 [--experiment S4B1] [--out outputs\\s4\\b_decide\\b1.json --md ...b1.md]
    .venv\\Scripts\\python.exe scripts\\98_s4_b_decide.py b2 --settings range10_cap5,ZEROS [--experiment S4B2 --drtc-experiment S4DRTC]
        （--settings の代わりに --gate1 outputs\\s4\\gate1_result.json で関門 1 の判定から読む）
終了コード: 0 判定した（採る・採らないのどちらも）/ 1 2 通りの数え方が一致しない（判定しない）/ 2 記録を読めない・対がそろわない。

採る条件（作者が承認した束 2 の計画。結果を見る前に固定。docs/stage4/bundle2_protocol.md 第 4 節）:
  B1（R1v3_cur と R1v3_v3 は同じ 60 種の対。N1v3_v3 は 30 本）。すべて満たせば v3 を採る:
    c1 3 個とも（真値）: R1v3_v3 の本数 − R1v3_cur の本数 >= 9（/60）
    c2 2 番目の手順の 1 回目の閉じの持ち上がり（分母は 2 番目の手順が始まった試行。s4_gates.json の first_close_lift）:
       R1v3_v3 の割合 − R1v3_cur の割合 >= 0.25
    c3 誤った「完了」（完了の判定が出たのに、その色が試行の中で一度も箱の中で 1 s 静止しなかった＝truth_success_t にない）:
       v3 の腕（R1v3_v3・N1v3_v3）で 0
    c4 監査（G1〜G3）の違反のある試行・監査のない試行: 3 つの腕すべてで 0
    並べて出すもの（判定には使わない。目標書 第 7 節 6）: N1v3_v3 の腕、success@0（介入 0 回で成功）、介入の種類別の数・時間
    （scripts/56_intervention_s3.py の数え方）。
  B2（naive と設定 S。自然は S4DRTC\\<S>（30）＋ S4B2\\<S>_ext（69）= 99 試行、把持失敗は S4B2\\<S>_P1 の 50 試行）。
    すべて満たす設定のうち、躍度が最も小さいものを採る（なければ採らない＝naive のまま）:
    c1 自然の成功（30 s の採点）: S の割合 − naive の割合 >= −0.05
    c2 把持失敗からの復帰（30 s の採点。分母は誘発が 30 s 以内に成立した試行）: S の割合 − naive の割合 >= −0.10
    c3 躍度（自然の試行の、指令の手先 x_des を 10 Hz にした 3 階差分の RMS [m/s³]。最初の 30 s に限った値の中央値を主、
       記録全体の値を副に出す）: S の中央値 < naive の中央値
2 通りの数え方:
  A: diag/e7.py の summarize_condition（3 個とも・持ち上がり）、eval/gate.py の audit_dir（監査）、96_s4_resume.py の score_condition
     （30 s の採点）、eval/metrics.py の jerk_rms（躍度）、meta の steps を手順ごとに回す誤った完了。
  B: このファイルの中の別の実装（final_in_box の値、npz の plan_step で 2 番目の手順のこまを選ぶ持ち上がり、meta の audit を直接
     数える監査、試行の json の success・t_success・induce を直接読む採点、こまの間隔から刻みを決める躍度、集合で数える誤った完了）。
"""
import argparse
import importlib.util
import json
import math
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTD = ROOT / "outputs" / "s4" / "b_decide"
B1_ARMS = ("R1v3_cur", "R1v3_v3", "N1v3_v3")
B1_RULES = {"c1_all_three_gain_min": 9, "c2_lift_gain_min": 0.25, "c3_false_complete_max": 0, "c4_audit_trials_max": 0}
B2_RULES = {"c1_natural_drop_max": 0.05, "c2_recovery_drop_max": 0.10, "c3_jerk_smaller": True, "horizon_s": 30.0}
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


def _runs(d: pathlib.Path, pat: str) -> list:
    return [p for p in sorted(d.glob(pat)) if not p.name.endswith("_runtime.json")]


class Mismatch(RuntimeError):
    pass


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


# ================================================================ B1
def b1_count_a(d: pathlib.Path) -> dict:
    from recovla.diag import e7 as E7
    from recovla.eval import gate
    s = E7.summarize_condition(d)
    au = gate.audit_dir(d)
    false = 0
    for p in _runs(d, "run_[0-9][0-9][0-9][0-9].json"):
        m = json.loads(p.read_text(encoding="utf-8"))
        truth = m.get("truth_success_t") or {}
        for st in m.get("steps") or []:
            if st.get("judged_complete") and st["color"] not in truth:
                false += 1
    lift = s["step2"]["first_close_lift"]
    return {"n": s["n"], "seeds": sorted(int(x) for x in s["per_seed"]), "all_three": s["all_three_true"]["k"],
            "lift_k": lift["k"], "lift_n": lift["n"], "false_complete": false,
            "audit_bad_trials": au["g1_trials"] + au["g2_trials"] + au["g3_trials"] + au["trials_without_audit"]}


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
    out = {"n": 0, "seeds": [], "all_three": 0, "lift_k": 0, "lift_n": 0, "false_complete": 0, "audit_bad_trials": 0}
    for p in _runs(d, "run_[0-9][0-9][0-9][0-9].json"):
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
               or int(au["g2"]["world_stops"]) + int(au["g2"]["early_use"]) > 0 or int(au["g3"]["total_violations"]) > 0)
        out["audit_bad_trials"] += int(bad)
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


def b1_decide(base: pathlib.Path) -> dict:
    c = {arm: b1_count(base / arm) for arm in B1_ARMS}
    cur, v3 = c["R1v3_cur"], c["R1v3_v3"]
    if cur["seeds"] != v3["seeds"] or cur["n"] == 0:
        raise ValueError(f"R1v3_cur と R1v3_v3 の種がそろわない（{cur['n']} 本・{v3['n']} 本）")
    rate = lambda x: x["lift_k"] / x["lift_n"] if x["lift_n"] else None          # noqa: E731
    d_all = v3["all_three"] - cur["all_three"]
    d_lift = None if rate(v3) is None or rate(cur) is None else rate(v3) - rate(cur)
    false = c["R1v3_v3"]["false_complete"] + c["N1v3_v3"]["false_complete"]
    audit = sum(x["audit_bad_trials"] for x in c.values())
    conds = {"c1": {"value": d_all, "rule": f">= {B1_RULES['c1_all_three_gain_min']}", "pass": d_all >= B1_RULES["c1_all_three_gain_min"]},
             "c2": {"value": None if d_lift is None else round(d_lift, 4), "rule": f">= {B1_RULES['c2_lift_gain_min']}",
                    "pass": d_lift is not None and d_lift >= B1_RULES["c2_lift_gain_min"] - EPS},
             "c3": {"value": false, "rule": "== 0（v3 の腕）", "pass": false == 0},
             "c4": {"value": audit, "rule": "== 0（3 つの腕）", "pass": audit == 0}}
    return {"counts": c, "conditions": conds, "adopt_v3": all(x["pass"] for x in conds.values()),
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


def _median(xs):
    xs = sorted(x for x in xs if x is not None and not math.isnan(x))
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def b2_count_a(dirs: list, horizon: float) -> dict:
    r96 = _load(ROOT / "scripts" / "96_s4_resume.py", "s4_resume_decide")
    k = n = n_tr = 0
    jerks, jerks_all = [], []
    for d in dirs:
        s = r96.score_condition(d, (horizon,))["at"][f"{horizon:g}"]
        k, n, n_tr = k + s["successes"], n + s["n"], n_tr + s["n_trials"]
        for p in sorted(d.glob("trial_[0-9][0-9][0-9][0-9].json")):
            z = np.load(p.with_suffix(".npz"))
            jerks.append(_jerk_a(z, horizon))
            jerks_all.append(_jerk_a(z, None))
    return {"k": k, "n": n, "n_trials": n_tr, "jerk_median": _median(jerks), "jerk_median_all": _median(jerks_all)}


def b2_count_b(dirs: list, horizon: float) -> dict:
    k = n = n_tr = 0
    jerks, jerks_all = [], []
    for d in dirs:
        for p in sorted(d.glob("trial_[0-9][0-9][0-9][0-9].json")):
            m = json.loads(p.read_text(encoding="utf-8"))
            n_tr += 1
            ind = m.get("induce") or {}
            if ind.get("kind"):
                te = ind.get("t_established")
                if not (ind.get("established") and te is not None and float(te) <= horizon + EPS):
                    continue
            n += 1
            ts = m.get("t_success")
            k += int(bool(m.get("success")) and ts is not None and float(ts) <= horizon + EPS)
            z = np.load(p.with_suffix(".npz"))
            jerks.append(_jerk_b(z, horizon))
            jerks_all.append(_jerk_b(z, None))
    return {"k": k, "n": n, "n_trials": n_tr, "jerk_median": _median(jerks), "jerk_median_all": _median(jerks_all)}


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


def b2_decide(b2_base: pathlib.Path, drtc_base: pathlib.Path, settings: list) -> dict:
    h = B2_RULES["horizon_s"]
    arms = ["naive"] + [s for s in settings if s != "naive"]
    c = {}
    for s in arms:
        nat_dirs = [drtc_base / s, b2_base / f"{s}_ext"]
        for d in nat_dirs + [b2_base / f"{s}_P1"]:
            if not d.is_dir():
                raise ValueError(f"記録がない: {d}")
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
    return {"counts": c, "settings": res, "adopt": adopt, "adopt_note": "条件を満たす中で躍度が最も小さい設定。なければ naive のまま"}


# ================================================================ 出力
def md_b1(r: dict) -> str:
    o = ["# 束 2 B1（実行器 v3）の採否", "", f"- 結論: {'v3 を採る' if r['adopt_v3'] else 'v3 を採らない'}", "",
         "| 条件 | 値 | 基準 | 満たす |", "|---|---|---|---|"]
    for k, v in r["conditions"].items():
        o.append(f"| {k} | {v['value']} | {v['rule']} | {'○' if v['pass'] else '×'} |")
    o += ["", "| 腕 | 本数 | 3 個とも | 2 番目の 1 回目の持ち上がり | 誤った完了 | 監査の違反 | success@0 | 介入/本 |", "|---|---|---|---|---|---|---|---|"]
    for arm, x in r["counts"].items():
        dsp = r["display"][arm]
        o.append(f"| {arm} | {x['n']} | {x['all_three']} | {x['lift_k']}/{x['lift_n']} | {x['false_complete']} | {x['audit_bad_trials']} | "
                 f"{dsp['success_at_k']['0']['k']}/{dsp['success_at_k']['0']['n']} | {dsp['interventions_per_run']} |")
    o += ["", "介入の種類別の数（56_intervention_s3.py の数え方。判定には使わない）:", ""]
    for arm in r["counts"]:
        o.append(f"- {arm}: " + "、".join(f"{v['label']} {v['total']}" for v in r["display"][arm]["per_type"].values()))
    o += ["", "2 通りの数え方（A・B）は件数が完全に一致した。"]
    return "\n".join(o) + "\n"


def md_b2(r: dict) -> str:
    o = ["# 束 2 B2（RTC の設定）の採否", "", f"- 結論: {r['adopt'] or '採る設定なし（naive のまま）'}", "",
         "| 設定 | 自然（30 s） | 把持失敗の復帰（30 s） | 躍度の中央値（30 s / 全体） | c1 | c2 | c3 | 採れる |", "|---|---|---|---|---|---|---|---|"]
    for s, x in r["counts"].items():
        st = r["settings"].get(s)
        flag = (lambda k: "○" if st["conditions"][k]["pass"] else "×") if st else (lambda k: "—")
        jm, ja = x["natural"]["jerk_median"], x["natural"]["jerk_median_all"]
        o.append(f"| {s} | {x['natural']['k']}/{x['natural']['n']} | {x['recovery']['k']}/{x['recovery']['n']} | "
                 f"{'—' if jm is None else f'{jm:.3f}'} / {'—' if ja is None else f'{ja:.3f}'} | {flag('c1')} | {flag('c2')} | {flag('c3')} | "
                 f"{'—' if not st else ('○' if st['pass'] else '×')} |")
    o += ["", "2 通りの数え方（A・B）は件数が完全に一致し、躍度の中央値は相対 1e-6 以内で一致した。"]
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
    except Exception:                                    # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("b1")
    p.add_argument("--experiment", default="S4B1")
    p.add_argument("--root", default=str(ROOT / "outputs" / "v2eval"))
    p.add_argument("--out", default=None)
    p.add_argument("--md", default=None)
    p = sub.add_parser("b2")
    p.add_argument("--experiment", default="S4B2")
    p.add_argument("--drtc-experiment", default="S4DRTC")
    p.add_argument("--settings", default=None)
    p.add_argument("--gate1", default=None)
    p.add_argument("--root", default=str(ROOT / "outputs" / "v2eval"))
    p.add_argument("--out", default=None)
    p.add_argument("--md", default=None)
    a = ap.parse_args(argv)
    root = pathlib.Path(a.root)
    try:
        if a.cmd == "b1":
            r = b1_decide(root / a.experiment)
            md = md_b1(r)
        else:
            st = settings_from(a)
            if not st:
                r = {"settings": {}, "adopt": None, "note": "関門 R で B2 に送る設定がない（B2 は回していない）"}
                md = "# 束 2 B2（RTC の設定）の採否\n\n- 関門 R で B2 に送る設定がない（B2 は回していない）\n"
            else:
                r = b2_decide(root / a.experiment, root / a.drtc_experiment, st)
                md = md_b2(r)
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
