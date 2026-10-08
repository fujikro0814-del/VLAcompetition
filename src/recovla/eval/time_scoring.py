"""段階 4 の時間ごとの採点。60 s で回した試行の記録（trial_*.json）から、30・45・60 s の成否を後から出す。

    from recovla.eval import time_scoring as TS
    recs = TS.load_trials(d)                                   # 記録のフォルダの trial_*.json
    TS.success_at(rec, 30.0)                                   # 30 s で採点した成否
    TS.time_report(recs_a, recs_b, ("復帰デモあり", "復帰デモなし"), induced=False)

後から採点し直せる根拠: harness/loop.py の run_policy_trial で制限時間が効くのは、打ち切り（while の条件）と成功時刻の
判定（t <= time_limit_s）だけ。60 s で回した試行の最初の 30 s は、30 s で回した試行と同じ経過になる。
t_success は「目標が箱の成功の体積の中で静止の保持（rest_hold）を満たした時刻」なので、30 s での成否は
t_success <= 30 と同じ。回した時間（time_limit_s）より長い制限では採点できない（例外にする）。

主な指標は 30 s の 1 つだけ（2 条件を種の対で比べた正確な McNemar）。45・60 s と時間ごとの曲線は補正なしの記述
（出力の JSON では primary と secondary に分ける）。
意図的な失敗の条件では、分母を「その制限時間より前に失敗が成立した試行」（induce.t_established < T。ちょうど T に
成立した試行は入れない。掲示板 0155 の 1-1）に絞れる（50_e_eval.py の summary・paired_binary と同じ考え方。対は両方で
成立したものだけ）。段階 3 は 30 s で打ち切ったので、成立は 29.9 s 以前だけだった（loop の while が t = 30.0 の前に終わり、
誘発の判定が t = 30.0 で呼ばれない）。60 s の記録で 30 s ちょうど・30 s より後に成立した試行を 30 s の分母に入れると、
段階 3 と同じ定義にならない。成功の側（t_success <= T）は変えない。96_s4_resume.py の score・diag/recovery.py と同じ規則。
時間ごとの曲線（誘発の条件）は、各時刻 t の分母を「t より前に成立した試行」にする（0155 の 1-3。曲線の 30 s の高さが
主な指標の 30 s の値と一致する）。曲線の n_at に時刻ごとの分母を残す。
差の区間は 50_e_eval.py の _newcombe と同じ φ の補正なし（段階 3 の報告の値とそろえる。stats.paired_diff_ci の既定は補正あり）。
time_report は、制限時間が 2 種類以上・環境（ドライバ・torch・CUDA・OS）が 2 つ以上・誘発の有無と induced の指定の
食い違い（自然と誘発の混在を含む）があれば、数えずに ValueError にする（査読の重要 1・軽微 3）。

3 個の連続タスク（run_*.json）は採点し直さない。1 手順の持ち時間が切れると、やり直しや次の手順へ進むので、制限時間
によって経過そのものが変わる（60 s で回した記録の最初の 30 s は、30 s で回した記録と同じにならない）。
連続タスクは task_step_times で各手順の成功時刻の分布を、task_all_three で「3 個そろった時刻」の分布を出す。
「3 個そろった時刻」は、終わりに 3 個とも箱の中（all_three_in_box）が真の試行だけについて、3 色の truth_success_t の最大
（0155 の 1-2。途中で箱から出た試行が曲線で成功に数えられないように）。
"""
import json
import math
import pathlib
import re

import numpy as np

from recovla.eval import stats

PRIMARY_LIMIT_S = 30.0
DEFAULT_LIMITS = (30.0, 45.0, 60.0)
GRID_STEP_S = 0.5               # 曲線の格子。物理の 1 手（2 ms）より粗いが、図と表には十分
EPS = 1e-9                      # harness/loop.py の比較と同じ幅
TASK_T_MAX_S = 200.0            # 連続タスクの曲線の終わり（試行全体の打ち切り。task_loop の既定・段階 3 と同じ）
ENV_KEYS = ("driver", "torch", "torch_cuda", "os_build")    # 96_s4_resume.ENV_STOP_KEYS と同じ（環境の区切りの鍵）
_TRIAL = re.compile(r"trial_\d+\.json")
_RUN = re.compile(r"run_\d+\.json")      # run_0000_runtime.json は除く


def _load(d, pat) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(pathlib.Path(d).iterdir()) if pat.fullmatch(p.name)]


def load_trials(d) -> list[dict]:
    """1 つの条件の試行の記録（trial_0000.json …）。"""
    return _load(d, _TRIAL)


def load_runs(d) -> list[dict]:
    """3 個の連続タスクの記録（run_0000.json …）。"""
    return _load(d, _RUN)


def is_induced(rec: dict) -> bool:
    """誘発の試行か（induce.kind がある）。集計の行（metrics.trial_metrics の "induce" は種類の名前か "none"）も読む。"""
    ind = rec.get("induce")
    if isinstance(ind, dict):
        return bool(ind.get("kind"))
    return ind not in (None, "", "none")


def established(rec: dict, by_s: float = None) -> bool:
    """意図的な失敗が起きた試行か。集計の行（induce_established）と生の記録（induce.established）のどちらも読む。
    by_s を渡すと、by_s 秒より前に成立したものだけを真にする（induce.t_established < by_s。ちょうど by_s は入れない。
    掲示板 0155 の 1-1）。成立の時刻が分からず、by_s が回した時間より短いときは判断できないので ValueError
    （回した時間以上なら、成立は打ち切りより前なので真）。"""
    if "induce_established" in rec:
        ok = bool(rec["induce_established"])
        t_est = rec.get("induce_t_established")
    else:
        ind = rec.get("induce") or {}
        ok = bool(ind.get("kind")) and bool(ind.get("established"))    # metrics.trial_metrics と同じ
        t_est = ind.get("t_established")
    if not ok or by_s is None:
        return ok
    if t_est is None:
        if float(by_s) >= float(rec["time_limit_s"]) - EPS:
            return True
        raise ValueError(f"seed {rec.get('seed')!r}: 成立の時刻がないので {float(by_s):g} s より前に成立したか分からない")
    return float(t_est) < float(by_s) - EPS


def success_at(rec: dict, limit_s: float) -> bool:
    """limit_s 秒で採点した成否。回した時間より長い制限では採点できないので ValueError。"""
    limit_s = float(limit_s)
    ran = float(rec["time_limit_s"])
    if ran < limit_s - EPS:
        raise ValueError(f"seed {rec.get('seed')!r}: {ran:g} s で回した試行を {limit_s:g} s で採点できない")
    t = rec.get("t_success")
    return t is not None and float(t) <= limit_s + EPS


def _key(lim) -> str:
    return f"{float(lim):g}"


def _use(recs, induced, by_s=None):
    return [r for r in recs if established(r, by_s)] if induced else list(recs)


def condition_scores(recs, limits=DEFAULT_LIMITS, induced: bool = False) -> dict:
    """1 つの条件の、制限時間ごとの成功数・分母・Wilson の 95% 信頼区間。
    induced なら分母をその制限時間までに失敗が成立した試行に絞る（制限時間ごとに分母が違いうる）。"""
    out = {"n_records": len(recs), "n": len(_use(recs, induced)), "by_limit": {}}
    for lim in limits:
        use = _use(recs, induced, lim)
        k = sum(success_at(r, lim) for r in use)
        lo, hi = stats.wilson_interval(k, len(use))
        out["by_limit"][_key(lim)] = {"limit_s": float(lim), "successes": k, "n": len(use),
                                      "rate": k / len(use) if use else None, "wilson95": [lo, hi]}
    return out


def success_curve(recs, t_max: float = 60.0, step: float = GRID_STEP_S, induced: bool = False) -> dict:
    """累積の成功率を 0〜t_max の格子（step 刻み）で。t における値は success_at(r, t) の割合（階段）。
    induced なら、各時刻 t の分母は t より前に失敗が成立した試行（established(r, t)。0155 の 1-3）で、n_at に時刻ごとの
    分母を残す（分母が 0 の時刻の rate は None）。そうでなければ分母は全試行で一定。
    n は t_max での分母。t_success の生の値（t_max の分母の中で成功した試行だけ、小さい順）も付ける。"""
    n_grid = int(round(t_max / step))
    grid = [round(i * step, 9) for i in range(n_grid + 1)]
    rates, n_at = [], []
    for t in grid:
        use = _use(recs, induced, t)
        n_at.append(len(use))
        rates.append(sum(success_at(r, t) for r in use) / len(use) if use else None)
    use = _use(recs, induced, t_max)
    times = sorted(float(r["t_success"]) for r in use if success_at(r, t_max))
    return {"n": len(use), "step_s": step, "t": grid, "rate": rates, "n_at": n_at,
            "denominator": ("各時刻 t より前に失敗が成立した試行（induce.t_established < t）" if induced else "すべての試行"),
            "t_success": times}


def check_records(recs, induced: bool) -> list:
    """数える前の確かめ（査読の重要 1・軽微 3）。問題の文の並び（空なら問題なし）。
    - 制限時間（time_limit_s）が 2 種類以上
    - 環境（ENV_KEYS）が 2 つ以上（記録に env がある場合。区切りごとに分けて出す）
    - 誘発の試行があるのに induced でない／induced なのに誘発の試行がない／自然と誘発が混ざっている"""
    probs = []
    lims = sorted({float(r["time_limit_s"]) for r in recs if r.get("time_limit_s") is not None})
    if len(lims) > 1:
        probs.append(f"制限時間が {len(lims)} 種類ある: {lims}（同じ制限時間の記録だけで数える）")
    envs = {json.dumps({k: r["env"].get(k) for k in ENV_KEYS}, sort_keys=True) for r in recs if isinstance(r.get("env"), dict)}
    if len(envs) > 1:
        probs.append(f"環境（{'・'.join(ENV_KEYS)}）が {len(envs)} つある（区切りごとに分けて出す）: {sorted(envs)}")
    n_ind = sum(is_induced(r) for r in recs)
    if n_ind and n_ind < len(recs):
        probs.append(f"誘発の試行 {n_ind} 本と自然の試行 {len(recs) - n_ind} 本が混ざっている（条件を分ける）")
    elif n_ind and not induced:
        probs.append(f"誘発の試行（{n_ind} 本）なのに induced でない（分母を成立した試行に絞らないと、96 の score と食い違う）")
    elif induced and recs and not n_ind:
        probs.append("induced なのに誘発の試行が 1 本もない")
    return probs


def _pair_rows(recs):
    """対の鍵は (種, 目標)。自然の試行は 1 つの種で 3 色を回すので、種だけでは対が決まらない（50_e_eval.py の _rows と同じ）。"""
    return [{"pair": (r["seed"], r.get("target")), "rec": r} for r in recs]


def compare_conditions(recs_a, recs_b, limits=DEFAULT_LIMITS, induced: bool = False) -> dict:
    """2 条件を種で対にして、制限時間ごとに正確な McNemar と Newcombe の差の 95% 区間（stats.py のもの）。
    片方にしかない種は除く。induced なら、その制限時間までに両方で失敗が成立した対だけを使う（制限時間ごとに対の数が違いうる）。"""
    all_pairs = stats.pair_by_seed(_pair_rows(recs_a), _pair_rows(recs_b), key="pair")
    n_common = len(all_pairs)

    def _pairs(by_s=None):
        if not induced:
            return all_pairs
        return [(x, y) for x, y in all_pairs if established(x["rec"], by_s) and established(y["rec"], by_s)]

    out = {"pairs": len(_pairs()), "common_seeds": n_common, "only_a": len(recs_a) - n_common,
           "only_b": len(recs_b) - n_common, "by_limit": {}}
    for lim in limits:
        pairs = _pairs(lim)
        sa = [success_at(x["rec"], lim) for x, _ in pairs]
        sb = [success_at(y["rec"], lim) for _, y in pairs]
        n11 = sum(a and b for a, b in zip(sa, sb))
        n10 = sum(a and not b for a, b in zip(sa, sb))
        n01 = sum(b and not a for a, b in zip(sa, sb))
        n00 = len(pairs) - n11 - n10 - n01
        diff, lo, hi = stats.paired_diff_ci(n11, n10, n01, n00, phi_correction=False)    # 50_e_eval._newcombe と同じ
        out["by_limit"][_key(lim)] = {
            "limit_s": float(lim), "pairs": len(pairs), "both": n11, "a_only": n10, "b_only": n01, "neither": n00,
            "a_rate": (n11 + n10) / len(pairs) if pairs else None, "b_rate": (n11 + n01) / len(pairs) if pairs else None,
            "mcnemar_exact_p": stats.mcnemar_exact(n10, n01),
            "diff_a_minus_b": None if math.isnan(diff) else diff,
            "diff_95ci_newcombe": None if math.isnan(lo) else [lo, hi]}
    return out


def time_report(recs_a, recs_b=None, labels=("条件 A", "条件 B"), limits=DEFAULT_LIMITS, induced: bool = False,
                step: float = GRID_STEP_S) -> dict:
    """主（30 s）と副（それ以外の制限時間と曲線）を分けた結果。recs_b が None なら 1 条件だけ（比較なし）。
    条件ごとに check_records を当て、問題があれば数えずに ValueError。"""
    limits = sorted({float(x) for x in limits})
    if not any(abs(x - PRIMARY_LIMIT_S) < EPS for x in limits):
        raise ValueError(f"制限時間に主な指標の {PRIMARY_LIMIT_S:g} s が入っていない: {limits}")
    sides = [("a", labels[0], recs_a)] + ([("b", labels[1], recs_b)] if recs_b is not None else [])
    probs = [f"{lab}: {p}" for _, lab, rs in sides for p in check_records(rs, induced)]
    if probs:
        raise ValueError("数える前の確かめで止めた: " + " / ".join(probs))
    conds = {s: {"label": lab, **condition_scores(rs, limits, induced)} for s, lab, rs in sides}
    cmp = compare_conditions(recs_a, recs_b, limits, induced) if recs_b is not None else None
    pk = _key(PRIMARY_LIMIT_S)
    sec_limits = [_key(x) for x in limits if _key(x) != pk]

    def at(k):
        return {"conditions": {s: {"label": c["label"], **c["by_limit"][k]} for s, c in conds.items()},
                "compare": None if cmp is None else cmp["by_limit"][k]}

    return {
        "settings": {"limits_s": limits, "primary_limit_s": PRIMARY_LIMIT_S, "induced_only": induced, "grid_step_s": step,
                     "labels": {s: lab for s, lab, _ in sides}, "pair_key": "seed と目標の色",
                     "denominator": ("その制限時間より前に失敗が成立した試行（induce.t_established < T。対は両方で成立。"
                                     "曲線は各時刻 t より前に成立した試行で、n_at が時刻ごとの分母）"
                                     if induced else "すべての試行"),
                     "diff_ci": "Newcombe の方法 10、φ の補正なし（50_e_eval.py の _newcombe と同じ）"},
        "records": {s: {"n_records": c["n_records"], "n": c["n"]} for s, c in conds.items()},
        "pairing": None if cmp is None else {k: cmp[k] for k in ("pairs", "common_seeds", "only_a", "only_b")},
        "primary": {"metric": f"{PRIMARY_LIMIT_S:g} s 以内の成功（種の対の正確な McNemar、両側）", "limit_s": PRIMARY_LIMIT_S,
                    **at(pk)},
        "secondary": {"note": "補正なしの記述（主な指標とは別に読む）",
                      "by_limit": {k: at(k) for k in sec_limits},
                      "curves": {s: {"label": lab, **success_curve(rs, max(limits), step, induced)} for s, lab, rs in sides}},
    }


# ------------------------------------------------------------------ 3 個の連続タスク（分布だけ）

def task_step_times(runs) -> dict:
    """手順ごと（1・2・3 番目）の成功時刻の分布。採点し直しはしない（冒頭の注記）。
    t_success は真値の成功時刻（truth_success_t、試行の始めから）、since_step_start はその手順の始め（t_start）から。
    その手順の色の真値の時刻が手順の始めより前のもの（ほかの手順の間に入った。since_step_start が負）は、その手順の成功に
    数えず、before_step_start に別に出す（査読の重要 2・0155 の 1-2）。"""
    by = {}
    for r in runs:
        truth = r.get("truth_success_t") or {}
        for j, s in enumerate(r.get("steps") or []):
            b = by.setdefault(str(j + 1), {"n": 0, "t_success": [], "since_step_start": [], "before_step_start": []})
            b["n"] += 1
            t = truth.get(s["color"])
            if t is None:
                continue
            since = float(t) - float(s.get("t_start", 0.0))
            if since < -EPS:
                b["before_step_start"].append({"seed": r.get("seed"), "run": r.get("run"), "color": s["color"],
                                               "t_success": float(t), "since_step_start": since})
                continue
            b["t_success"].append(float(t))
            b["since_step_start"].append(since)
    for b in by.values():
        b["t_success"].sort()
        b["since_step_start"].sort()
        b["successes"] = len(b["t_success"])
        b["n_before_step_start"] = len(b["before_step_start"])
        b["quartiles_s"] = _quartiles(b["t_success"])
    return {"runs": len(runs), "note": "連続タスクは制限時間で経過が変わるので、30 s での採点し直しはしない。"
                                       "手順の始めより前に真値の時刻がある色は before_step_start に分けた", "by_step": by}


def all_three_time(run: dict):
    """3 個そろった時刻（0155 の 1-2）。all_three_in_box が真で、3 色とも truth_success_t があるときだけ、その最大。ほかは None。"""
    tt = run.get("truth_success_t") or {}
    if not run.get("all_three_in_box") or len(tt) != 3 or any(v is None for v in tt.values()):
        return None
    return float(max(tt.values()))


def task_all_three(runs, t_max: float = TASK_T_MAX_S, step: float = GRID_STEP_S) -> dict:
    """3 個の連続タスクの「3 個そろった時刻」の曲線（副次。0155 の 1-2）。分母は全試行（打ち切り・止まった試行を含む）。
    all_three_in_box（主な指標。段階 3 と同じ）と timed_out の本数も出す。all_three_in_box が真なのに 3 色の真値の時刻が
    そろわない試行は曲線に入らないので、all_three_without_time に数える（0 でなければ曲線の終わりが主な数より低い）。"""
    n = len(runs)
    ts = [all_three_time(r) for r in runs]
    n_grid = int(round(t_max / step))
    grid = [round(i * step, 9) for i in range(n_grid + 1)]
    rates = [sum(1 for t in ts if t is not None and t <= g + EPS) / n if n else None for g in grid]
    n_all3 = sum(bool(r.get("all_three_in_box")) for r in runs)
    return {"runs": n, "all_three_in_box": n_all3, "timed_out": sum(bool(r.get("timed_out")) for r in runs),
            "all_three_without_time": n_all3 - sum(t is not None for t in ts),
            "definition": "all_three_in_box が真の試行の、3 色の truth_success_t の最大（ほかの試行は曲線で失敗）",
            "step_s": step, "t": grid, "rate": rates, "t_all_three": sorted(t for t in ts if t is not None),
            "quartiles_s": _quartiles([t for t in ts if t is not None])}


def _quartiles(v):
    if not v:
        return None
    return [float(x) for x in np.percentile(v, [25, 50, 75])]
