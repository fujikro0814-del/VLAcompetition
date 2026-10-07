"""段階 4 の時間ごとの採点（recovla.eval.time_scoring・scripts/98_s4_time_report.py）の検査。合成の記録だけで回す。"""
import importlib.util
import json
import pathlib

import numpy as np
import pytest

from recovla.eval import stats
from recovla.eval import time_scoring as TS

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location("s4_time_report", ROOT / "scripts" / "98_s4_time_report.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def rec(seed, t_success, target="red", time_limit_s=60.0, induce=None):
    """harness/loop.py の記録と同じ欄（使うものだけ）。induce は None（通常の試行）か established の真偽。"""
    return {"seed": seed, "target": target, "success": t_success is not None, "t_success": t_success,
            "time_limit_s": time_limit_s, "t_end": t_success if t_success is not None else time_limit_s,
            "steps": [{"target": target, "t_start": 0.0, "success": t_success is not None, "t_success": t_success}],
            "induce": {"kind": None, "established": False} if induce is None else {"kind": "P1", "established": bool(induce)}}


# ------------------------------------------------------------------ success_at

def test_boundary_exactly_30():
    r = rec(1, 30.0)
    assert TS.success_at(r, 30.0)
    assert TS.success_at(r, 30)
    late = rec(2, 30.002)                       # 物理の 1 手（2 ms）遅れ
    assert not TS.success_at(late, 30.0)
    assert TS.success_at(late, 45.0)


def test_none_is_failure():
    r = rec(1, None)
    assert not any(TS.success_at(r, lim) for lim in (30, 45, 60))


def test_longer_limit_than_ran_raises():
    r = rec(1, 12.0, time_limit_s=30.0)
    assert TS.success_at(r, 30.0)               # 回した時間以下なら採点できる
    assert TS.success_at(r, 30.0 + 1e-12)       # 比較の幅の中
    with pytest.raises(ValueError):
        TS.success_at(r, 60.0)
    with pytest.raises(ValueError):
        TS.condition_scores([rec(2, None), r], limits=(30, 60))


# ------------------------------------------------------------------ 1 条件

def test_condition_scores_and_wilson():
    rs = [rec(i, t) for i, t in enumerate([5.0, 29.9, 30.0, 31.0, 44.0, 59.0, None, None])]
    sc = TS.condition_scores(rs)
    assert [sc["by_limit"][k]["successes"] for k in ("30", "45", "60")] == [3, 5, 6]
    assert all(sc["by_limit"][k]["n"] == 8 for k in ("30", "45", "60"))
    assert sc["by_limit"]["30"]["wilson95"] == list(stats.wilson_interval(3, 8))


def test_induced_narrows_denominator():
    rs = [rec(1, 10.0, induce=True), rec(2, None, induce=True), rec(3, 10.0, induce=False), rec(4, None, induce=False)]
    assert TS.condition_scores(rs)["by_limit"]["30"]["n"] == 4
    sc = TS.condition_scores(rs, induced=True)
    assert sc["n_records"] == 4 and sc["n"] == 2
    assert sc["by_limit"]["30"]["successes"] == 1
    # 集計の行（metrics.trial_metrics の induce_established）も読める
    assert TS.established({"induce_established": True}) and not TS.established({"induce_established": False})
    # 通常の試行の記録は established が真でも数えない（metrics と同じ）
    assert not TS.established({"induce": {"kind": None, "established": True}})


def test_curve_monotone_and_consistent():
    rng = np.random.default_rng(0)
    ts = [float(x) if x < 60 else None for x in rng.uniform(0, 90, 40).round(3)]
    rs = [rec(i, t) for i, t in enumerate(ts)]
    c = TS.success_curve(rs)
    assert len(c["t"]) == 121 and c["t"][0] == 0.0 and c["t"][-1] == 60.0
    assert all(b >= a for a, b in zip(c["rate"], c["rate"][1:]))
    sc = TS.condition_scores(rs)
    assert c["rate"][c["t"].index(30.0)] == pytest.approx(sc["by_limit"]["30"]["rate"])
    assert c["rate"][-1] == pytest.approx(sc["by_limit"]["60"]["rate"])
    assert c["t_success"] == sorted(t for t in ts if t is not None)


# ------------------------------------------------------------------ 2 条件の比較

def test_pairing_drops_one_sided_seeds():
    a = [rec(s, 10.0) for s in (1, 2, 3, 4, 5)]
    b = [rec(s, None) for s in (2, 3, 4, 5, 6)]
    c = TS.compare_conditions(a, b)
    assert (c["pairs"], c["only_a"], c["only_b"]) == (4, 1, 1)
    assert c["by_limit"]["30"]["a_only"] == 4


def test_pairing_uses_seed_and_target():
    # 自然の試行は 1 つの種で 3 色を回す。種だけでは対が決まらないので、目標の色も鍵に入れる
    a = [rec(7, 10.0, target=c) for c in ("red", "green", "blue")]
    b = [rec(7, None, target=c) for c in ("red", "green", "blue")]
    assert TS.compare_conditions(a, b)["pairs"] == 3
    with pytest.raises(ValueError):             # 同じ側に同じ鍵が 2 回あれば対が決まらない
        TS.compare_conditions(a + [rec(7, 1.0, target="red")], b)


def test_induced_pairs_need_both_established():
    a = [rec(1, 10.0, induce=True), rec(2, 10.0, induce=True), rec(3, 10.0, induce=False), rec(4, 10.0, induce=True)]
    b = [rec(1, None, induce=True), rec(2, None, induce=False), rec(3, None, induce=True), rec(4, 10.0, induce=True)]
    c = TS.compare_conditions(a, b, induced=True)
    assert c["common_seeds"] == 4 and c["pairs"] == 2
    d = c["by_limit"]["30"]
    assert (d["both"], d["a_only"], d["b_only"], d["neither"]) == (1, 1, 0, 0)


def test_mcnemar_per_limit_uses_stats():
    # 30 s: A だけ成功 6、B だけ成功 1、両方 2、両方失敗 3。45 s では B の遅い成功が 3 つ増える
    a = [rec(i, 10.0) for i in range(8)] + [rec(8 + i, None) for i in range(4)]
    tb = [10.0, 10.0, None, None, None, 40.0, 40.0, 40.0, 10.0, None, None, None]
    b = [rec(i, t) for i, t in enumerate(tb)]
    c = TS.compare_conditions(a, b)
    d30, d45 = c["by_limit"]["30"], c["by_limit"]["45"]
    assert (d30["both"], d30["a_only"], d30["b_only"], d30["neither"]) == (2, 6, 1, 3)
    assert d30["mcnemar_exact_p"] == stats.mcnemar_exact(6, 1)
    diff, lo, hi = stats.paired_diff_ci(2, 6, 1, 3)
    assert d30["diff_a_minus_b"] == diff and d30["diff_95ci_newcombe"] == [lo, hi]
    assert (d45["both"], d45["a_only"], d45["b_only"]) == (5, 3, 1)


# ------------------------------------------------------------------ 主と副

def _pair_set():
    a = [rec(i, t) for i, t in enumerate([5.0, 20.0, 30.0, 40.0, 50.0, None])]
    b = [rec(i, t) for i, t in enumerate([25.0, 35.0, None, 55.0, None, None])]
    return a, b


def test_primary_is_only_30():
    a, b = _pair_set()
    rep = TS.time_report(a, b, ("復帰デモあり", "復帰デモなし"))
    assert rep["primary"]["limit_s"] == 30.0
    assert rep["primary"]["conditions"]["a"]["successes"] == 3
    assert rep["primary"]["compare"]["pairs"] == 6
    assert set(rep["secondary"]["by_limit"]) == {"45", "60"}
    assert "mcnemar_exact_p" in rep["secondary"]["by_limit"]["60"]["compare"]
    assert "補正なし" in rep["secondary"]["note"]
    assert set(rep["secondary"]["curves"]) == {"a", "b"}
    with pytest.raises(ValueError):
        TS.time_report(a, b, limits=(45, 60))   # 主な指標の 30 s がない


def test_single_condition_without_compare():
    a, _ = _pair_set()
    rep = TS.time_report(a, None, ("復帰デモあり",))
    assert rep["primary"]["compare"] is None and rep["pairing"] is None
    assert set(rep["primary"]["conditions"]) == {"a"}


# ------------------------------------------------------------------ 図

@pytest.mark.parametrize("induced", [False, True])
def test_svg_has_no_forbidden_words(induced):
    m = _script()
    a, b = _pair_set()
    if induced:
        a = [dict(r, induce={"kind": "P1", "established": True}) for r in a]
        b = [dict(r, induce={"kind": "P1", "established": True}) for r in b]
    svg = m.curve_svg(TS.time_report(a, b, ("復帰デモあり", "復帰デモなし"), induced=induced))
    assert svg.startswith("<svg") and "</svg>" in svg
    assert ("意図的な失敗" if induced else "通常の試行") in svg
    assert "30 秒" in svg and "60 秒" in svg
    assert m.forbidden_hits(svg) == []


def test_forbidden_check_is_not_vacuous():
    m = _script()
    pats = m.forbidden_patterns()
    assert len(pats) > 20                        # 60_paper.py の一覧を読めている
    a, b = _pair_set()
    svg = m.curve_svg(TS.time_report(a, b, ("R1v3", "N1v3")))
    assert {w for _, w in m.forbidden_hits(svg, pats)} >= {"R1v3", "N1v3"}


# ------------------------------------------------------------------ 3 個の連続タスク

def _run(seed, truth, starts=(0.0, 20.0, 45.0)):
    steps = [{"step": j, "color": c, "t_start": t0, "attempts": [{"attempt": 0, "t_begin": t0, "t_judge": None}]}
             for j, (c, t0) in enumerate(zip(("red", "green", "blue"), starts))]
    return {"seed": seed, "steps": steps, "truth_success_t": truth, "all_three_in_box": len(truth) == 3}


def test_task_step_times():
    runs = [_run(1, {"red": 15.0, "green": 40.0, "blue": 70.0}), _run(2, {"red": 18.0})]
    out = TS.task_step_times(runs)
    assert out["runs"] == 2 and "採点し直し" in out["note"]
    s1, s2, s3 = (out["by_step"][k] for k in ("1", "2", "3"))
    assert (s1["n"], s1["successes"], s1["t_success"]) == (2, 2, [15.0, 18.0])
    assert s2["since_step_start"] == [20.0] and s3["since_step_start"] == [25.0]
    assert s1["quartiles_s"][1] == pytest.approx(16.5)


# ------------------------------------------------------------------ CLI

def _write(d, recs, name="trial_{:04d}.json"):
    d.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(recs):
        (d / name.format(i)).write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")


def test_cli_end_to_end(tmp_path):
    m = _script()
    a, b = _pair_set()
    _write(tmp_path / "A", a)
    _write(tmp_path / "B", b)
    (tmp_path / "A" / "trial_0000_runtime.json").write_text("{}", encoding="utf-8")    # 記録でないものは読まない
    out, svg = tmp_path / "res" / "t.json", tmp_path / "res" / "t.svg"
    rc = m.main(["--dirs", str(tmp_path / "A"), str(tmp_path / "B"), "--labels", "復帰デモあり", "復帰デモなし",
                 "--out", str(out), "--svg", str(svg)])
    assert rc == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["records"]["a"]["n"] == 6 and res["primary"]["limit_s"] == 30.0
    assert m.forbidden_hits(svg.read_text(encoding="utf-8")) == []


def test_cli_tasks_only_distribution(tmp_path):
    m = _script()
    _write(tmp_path / "T", [_run(1, {"red": 15.0})], name="run_{:04d}.json")
    (tmp_path / "T" / "run_0000_runtime.json").write_text("{}", encoding="utf-8")
    out = tmp_path / "t.json"
    assert m.main(["--dirs", str(tmp_path / "T"), "--labels", "連続タスク", "--out", str(out)]) == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert "primary" not in res and res["tasks"]["連続タスク"]["runs"] == 1
