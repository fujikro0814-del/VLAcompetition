"""段階 4 束 1 の採点の定義の直し（掲示板 0155 の 1 節）と解析の入口の点検（0155 の 2 節、scripts/98_s4_d_audit.py）の検査。
合成の記録だけで回す（CPU。GPU・MuJoCo・outputs は要らない）。

- 1-1 誘発の成立の境界: L 秒の分母は t_established < L（ちょうど 30.0 s の成立は 30 s の分母に入らない）。成功の側は t_success <= L。
  96 の score・time_scoring・diag/recovery・点検の数え方の 4 つで同じ。
- 1-2 3 個の連続タスクの曲線: all_three_in_box が真の試行だけ、3 色の truth_success_t の最大。score に all_three_in_box と打ち切りの本数。
- 1-3 誘発の条件の曲線は時刻ごとの分母。図に分母を書く。
- 査読の軽微 3: --induced の付け忘れで止める。査読の重要 1: 98_s4_d_start の summary の同じ鍵のフォルダで止める、二重集計の列挙の独立。
diag/recovery.py・diag/start.py は import のときに mujoco を読むので、この PC のように mujoco が無いときだけ、集計の関数を試すために
mujoco（と cv2）の代わりの空の模型を sys.modules に入れて読み込み、終わったら取り除く（試行を回す部分は呼ばない）。
"""
import contextlib
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
from unittest import mock

import pytest

from recovla.eval import time_scoring as TS

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENV = {"driver": "610.88", "torch": "2.11.0+cu126", "torch_cuda": "12.6", "os_build": "26300.9457", "git_head": "h1"}


def _script(name, modname):
    spec = importlib.util.spec_from_file_location(modname, ROOT / "scripts" / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def r96():
    return _script("96_s4_resume.py", "s4_resume_scoring_fix")


@pytest.fixture(scope="module")
def audit():
    return _script("98_s4_d_audit.py", "s4_audit_scoring_fix")


@pytest.fixture(scope="module")
def report():
    return _script("98_s4_time_report.py", "s4_time_report_scoring_fix")


def _missing(name: str) -> bool:
    if name in sys.modules:
        return False
    try:
        return importlib.util.find_spec(name) is None
    except (ImportError, ValueError):
        return False


@contextlib.contextmanager
def _diag_modules():
    """recovla.diag.recovery・start を読み込む。mujoco・cv2 が無ければ空の模型で代える（集計の関数だけを試すため）。
    代えたときは、その上で読み込んだ recovla のモジュールと代わりの模型を、終わったら sys.modules から取り除く。"""
    stubs = {n: mock.MagicMock(name=f"stub_{n}") for n in ("mujoco", "cv2") if _missing(n)}
    if not stubs:
        from recovla.diag import recovery as DR
        from recovla.diag import start as S
        yield DR, S
        return
    before = set(sys.modules)
    sys.modules.update(stubs)
    try:
        from recovla.diag import recovery as DR
        from recovla.diag import start as S
        yield DR, S
    finally:
        for k in set(sys.modules) - before:
            if k in stubs or k == "recovla" or k.startswith("recovla."):
                sys.modules.pop(k, None)


@pytest.fixture(scope="module")
def diag():
    with _diag_modules() as (DR, S):
        yield DR, S


def ind_rec(seed, t_success, t_est, target="red", established=True, kind="P2", time_limit_s=60.0, trial=None):
    return {"trial": trial if trial is not None else seed, "seed": seed, "target": target, "success": t_success is not None,
            "t_success": t_success, "time_limit_s": time_limit_s, "time_limits": {"time_limit_s": time_limit_s, "source": "test"},
            "induce": {"kind": kind, "established": established, "t_established": t_est if established else None}}


def _write(d: pathlib.Path, recs, name="trial_{:04d}.json"):
    d.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(recs):
        (d / name.format(i)).write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")


# ================================================================== 1-1 誘発の成立の境界
BOUNDARY = [ind_rec(1, 20.0, 29.9), ind_rec(2, 25.0, 30.0), ind_rec(3, None, 30.0), ind_rec(4, 30.0, 10.0),
            ind_rec(5, 50.0, 30.0), ind_rec(6, None, None, established=False)]
# 30 s: 分母は種 1（29.9）と 4（10.0）の 2 本。種 4 は t_success = 30.0 ちょうどで成功（成功の側は <= のまま）
# 60 s: 分母は 1〜5 の 5 本、成功は 1・2・4・5 の 4 本


def test_time_scoring_boundary_exactly_30():
    assert not TS.established(ind_rec(9, None, 30.0), 30.0)
    assert TS.established(ind_rec(9, None, 29.9), 30.0)
    assert TS.established(ind_rec(9, None, 30.0), 30.0 + 0.1)
    assert TS.established(ind_rec(9, None, 30.0))                     # 時刻を問わなければ成立
    sc = TS.condition_scores(BOUNDARY, (30.0, 60.0), induced=True)["by_limit"]
    assert (sc["30"]["successes"], sc["30"]["n"]) == (2, 2)
    assert (sc["60"]["successes"], sc["60"]["n"]) == (4, 5)
    assert TS.success_at(ind_rec(4, 30.0, 10.0), 30.0)               # 成功の側は変えない


def test_time_scoring_pairs_boundary():
    a = [ind_rec(1, 20.0, 29.9), ind_rec(2, 20.0, 30.0)]
    b = [ind_rec(1, None, 5.0), ind_rec(2, None, 5.0)]
    c = TS.compare_conditions(a, b, (30.0, 60.0), induced=True)
    assert c["by_limit"]["30"]["pairs"] == 1 and c["by_limit"]["60"]["pairs"] == 2


def test_96_score_boundary(r96, tmp_path):
    _write(tmp_path / "P2", BOUNDARY)
    r = r96.score_condition(tmp_path / "P2", [30.0, 60.0])
    assert (r["at"]["30"]["successes"], r["at"]["30"]["n"]) == (2, 2)
    assert (r["at"]["60"]["successes"], r["at"]["60"]["n"]) == (4, 5)
    assert "< T" in r["at"]["30"]["denominator"]


def test_recovery_boundary_and_cross_check(diag):
    DR, _ = diag
    ra = DR.recovery_at(BOUNDARY, 30.0)
    assert (ra["recovered"], ra["established"]) == (2, 2)
    assert (DR.recovery_at(BOUNDARY, 60.0)["recovered"], DR.recovery_at(BOUNDARY, 60.0)["established"]) == (4, 5)
    n = [ind_rec(s, None, 29.0) for s in range(1, 7)]
    p = DR.paired_at(BOUNDARY, n, 30.0)
    assert p["pairs"] == 2 and p["r_only"] == 2
    assert DR.cross_check(BOUNDARY, n)["agree"]


def test_four_implementations_agree(r96, audit, diag, tmp_path):
    """96 の score・time_scoring・diag/recovery・点検の数え方が、境界の合成の記録で同じ件数を出す。"""
    DR, _ = diag
    _write(tmp_path / "c", BOUNDARY)
    s96 = r96.score_condition(tmp_path / "c", [30.0, 60.0])["at"]
    ts = TS.condition_scores(TS.load_trials(tmp_path / "c"), (30.0, 60.0), induced=True)["by_limit"]
    recs, _ = audit.enumerate_records(tmp_path / "c", "run")
    mine = audit.tally({"kind": "run"}, recs)["at"]
    for L in ("30", "60"):
        dr = DR.recovery_at(DR.load_condition(tmp_path / "c"), float(L))
        got = {(s96[L]["successes"], s96[L]["n"]), (ts[L]["successes"], ts[L]["n"]), (dr["recovered"], dr["established"]),
               (mine[L]["successes"], mine[L]["n"])}
        assert len(got) == 1, (L, got)


# ================================================================== 1-2 3 個の連続タスク
def _task(run, seed, truth, all3, timed_out=False, starts=(0.0, 20.0, 45.0)):
    steps = [{"step": j, "color": c, "t_start": t0} for j, (c, t0) in enumerate(zip(("red", "green", "blue"), starts))]
    return {"run": run, "seed": seed, "steps": steps, "truth_success_t": truth, "all_three_in_box": all3, "timed_out": timed_out,
            "time_limits": {"step_timeout_s": 30.0, "retry": 1, "task_time_limit_s": 200.0, "source": "test"}}


TASKS = [_task(0, 1, {"red": 15.0, "green": 40.0, "blue": 70.0}, True),
         _task(1, 2, {"red": 15.0, "green": 40.0, "blue": 90.0}, False),          # 途中で 1 個目を箱から出した: 曲線で成功にしない
         _task(2, 3, {"red": 18.0}, False, timed_out=True),
         _task(3, 4, {"red": 12.0, "green": 30.0, "blue": 150.0}, True)]


def test_96_score_task_curve_needs_all_three(r96, tmp_path):
    _write(tmp_path / "E7", TASKS, name="run_{:04d}.json")
    r = r96.score_condition(tmp_path / "E7", [100.0, 200.0])
    assert r["kind"] == "task" and r["all_three_in_box"] == 2 and r["timed_out"] == 1
    assert (r["at"]["200"]["successes"], r["at"]["200"]["n"]) == (2, 4)          # 前の版は 3/4（試行 1 を成功に数えた）
    assert r["at"]["100"]["successes"] == 1
    assert r["t_success"] == [70.0, None, None, 150.0]
    assert "problems" not in r


def test_96_score_task_all_three_without_time(r96, tmp_path):
    _write(tmp_path / "E7", TASKS + [_task(4, 5, {"red": 10.0, "green": 20.0}, True)], name="run_{:04d}.json")
    r = r96.score_condition(tmp_path / "E7", [200.0])
    assert r["all_three_in_box"] == 3 and r["all_three_without_truth_time"] == 1
    assert any("そろわない" in p for p in r["problems"])


def test_time_scoring_task_all_three():
    out = TS.task_all_three(TASKS)
    assert (out["runs"], out["all_three_in_box"], out["timed_out"], out["all_three_without_time"]) == (4, 2, 1, 0)
    assert out["t_all_three"] == [70.0, 150.0]
    assert out["t"][-1] == 200.0 and out["rate"][-1] == pytest.approx(2 / 4)
    assert out["rate"][out["t"].index(100.0)] == pytest.approx(1 / 4)


def test_task_step_times_drops_color_before_step_start():
    # 2 番目の手順（緑、t_start 20）の始めより前に、緑の真値の時刻（12 s）がある試行は、2 番目の成功に数えず別に出す
    runs = [_task(0, 1, {"red": 15.0, "green": 12.0, "blue": 70.0}, True), _task(1, 2, {"red": 15.0, "green": 40.0}, False)]
    out = TS.task_step_times(runs)["by_step"]
    assert out["2"]["successes"] == 1 and out["2"]["since_step_start"] == [20.0]
    assert out["2"]["n_before_step_start"] == 1 and out["2"]["before_step_start"][0]["since_step_start"] == pytest.approx(-8.0)
    assert out["1"]["n_before_step_start"] == 0 and out["3"]["successes"] == 1


def test_time_report_cli_tasks_has_all_three(report, tmp_path):
    _write(tmp_path / "T", TASKS, name="run_{:04d}.json")
    out = tmp_path / "t.json"
    assert report.main(["--dirs", str(tmp_path / "T"), "--labels", "連続タスク", "--out", str(out)]) == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["all_three"]["連続タスク"]["all_three_in_box"] == 2


# ================================================================== 1-3 曲線の分母
def test_curve_uses_denominator_per_time():
    recs = [ind_rec(1, 20.0, 5.0), ind_rec(2, 50.0, 35.0), ind_rec(3, None, 10.0)]
    c = TS.success_curve(recs, 60.0, induced=True)
    i5, i30, i40 = c["t"].index(5.0), c["t"].index(30.0), c["t"].index(40.0)
    assert c["n_at"][0] == 0 and c["rate"][0] is None and c["n_at"][i5] == 0     # 5.0 ちょうどはまだ入らない
    assert c["n_at"][i30] == 2 and c["rate"][i30] == pytest.approx(1 / 2)       # 前の版は分母 3（60 s までに成立）で 1/3
    assert c["n_at"][i40] == 3 and c["n"] == 3
    sc = TS.condition_scores(recs, (30.0, 60.0), induced=True)["by_limit"]
    assert c["rate"][i30] == pytest.approx(sc["30"]["rate"]) and c["rate"][-1] == pytest.approx(sc["60"]["rate"])
    assert all(n == 3 for n in TS.success_curve([ind_rec(s, 1.0, 1.0, kind=None) for s in range(3)], 60.0)["n_at"])


def test_svg_writes_denominator(report):
    a = [ind_rec(1, 20.0, 5.0), ind_rec(2, 50.0, 35.0), ind_rec(3, None, 10.0)]
    b = [ind_rec(1, None, 6.0), ind_rec(2, 40.0, 12.0), ind_rec(3, None, 31.0)]
    rep = TS.time_report(a, b, ("復帰デモあり", "復帰デモなし"), induced=True)
    svg = report.curve_svg(rep)
    assert "各時刻より前" in svg and "分母 30 秒 2・60 秒 3" in svg and "分母 30 秒 2・60 秒 3" in svg
    assert report.forbidden_hits(svg) == []
    assert "M" in svg and "None" not in svg                                    # 分母 0 の時刻は描かない


# ================================================================== 査読の軽微 3・重要 1（98_s4_time_report・time_report）
def test_cli_stops_without_induced(report, tmp_path, capsys):
    _write(tmp_path / "A", [ind_rec(s, 20.0, 5.0, trial=s) for s in range(3)])
    assert report.main(["--dirs", str(tmp_path / "A"), "--out", str(tmp_path / "x.json")]) == 2
    assert "induced" in capsys.readouterr().err
    assert report.main(["--dirs", str(tmp_path / "A"), "--induced", "--out", str(tmp_path / "y.json")]) == 0
    _write(tmp_path / "N", [ind_rec(s, 20.0, None, kind=None, established=False) for s in range(3)])
    assert report.main(["--dirs", str(tmp_path / "N"), "--induced", "--out", str(tmp_path / "z.json")]) == 2


def test_time_report_refuses_mixed_inputs():
    nat = [ind_rec(s, 20.0, None, kind=None, established=False) for s in range(3)]
    ind = [ind_rec(s, 20.0, 5.0) for s in range(3)]
    with pytest.raises(ValueError, match="混ざって"):
        TS.time_report(nat + ind, None, ("x",), induced=True)
    with pytest.raises(ValueError, match="induced でない"):
        TS.time_report(ind, None, ("x",))
    with pytest.raises(ValueError, match="制限時間"):
        TS.time_report(nat + [ind_rec(9, 20.0, None, kind=None, established=False, time_limit_s=30.0)], None, ("x",),
                       limits=(30.0,))
    e2 = [dict(r, env=dict(ENV, driver="560.94" if r["seed"] == 0 else "610.88")) for r in nat]
    with pytest.raises(ValueError, match="環境"):
        TS.time_report(e2, None, ("x",))
    assert TS.check_records(nat, False) == [] and TS.check_records(ind, True) == []


def test_96_score_flags_mixed_limits_and_env(r96, tmp_path):
    recs = [ind_rec(s, 20.0, None, kind=None, established=False, trial=s) for s in range(3)]
    recs[0]["env"], recs[1]["env"], recs[2]["env"] = ENV, ENV, dict(ENV, driver="560.94")
    recs[2]["time_limits"] = {"time_limit_s": 30.0}
    _write(tmp_path / "c", recs)
    r = r96.score_condition(tmp_path / "c", [30.0])
    assert len(r["problems"]) == 2 and len(r["env_segments"]) == 2


# ================================================================== 査読の重要 1: 98_s4_d_start の summary
def _start_trial(d, i, diag):
    d.mkdir(parents=True, exist_ok=True)
    (d / f"trial_{i:04d}.json").write_text(json.dumps({"trial": i, "diag": diag}), encoding="utf-8")


def test_d_start_summary_stops_on_duplicate_key(tmp_path):
    m = _script("98_s4_d_start.py", "s4_d_start_scoring_fix")
    base = tmp_path / "S4DSTART"
    for name in ("send_wall", "send_wall_again"):
        _start_trial(base / name, 0, {"diag": "D-single-start", "start": "standby_end", "prior": "wall_side"})
    _start_trial(base / "home_grid", 0, {"diag": "D-single-start", "start": "home", "prior": "on_grid"})
    _start_trial(base / "mixed", 0, {"diag": "D-single-start", "start": "home", "prior": "wall_side"})
    _start_trial(base / "mixed", 1, {"diag": "D-single-start", "start": "standby_end", "prior": "wall_side"})
    _start_trial(base / "as_r1", 0, {"diag": "XPL", "arm": "XPL_as", "rep": 1})
    starts, xpl, unknown, probs = m.summary_dirs(base)
    assert set(starts) == {"standby_end|wall_side", "home|on_grid"} and starts["standby_end|wall_side"].name == "send_wall"
    assert xpl == {"as": {1: base / "as_r1"}} and unknown == []
    assert len(probs) == 2 and any("send_wall_again" in p for p in probs) and any("mixed" in p for p in probs)


def test_d_start_cmd_summary_returns_3(tmp_path, monkeypatch):
    m = _script("98_s4_d_start.py", "s4_d_start_scoring_fix2")
    monkeypatch.setattr(m, "ROOT", tmp_path)
    base = tmp_path / "outputs" / "v2eval" / "S4XPL"
    for name in ("as_r1", "as_r1_copy"):
        _start_trial(base / name, 0, {"diag": "XPL", "arm": "XPL_as", "rep": 1})
    with _diag_modules():
        a = type("A", (), {"experiment": "S4XPL", "out": str(tmp_path / "s.json"), "with_trials": False})()
        assert m.cmd_summary(a) == 3
    assert not (tmp_path / "s.json").exists()


# ================================================================== 二重集計の列挙の独立（recovery.cross_check_dirs）
def test_cross_check_dirs_enumerates_independently(diag, tmp_path):
    DR, _ = diag
    R = [ind_rec(s, 20.0 if s % 2 else None, 5.0, trial=i) for i, s in enumerate(range(1, 7))]
    N = [ind_rec(s, None, 6.0, trial=i) for i, s in enumerate(range(1, 7))]
    _write(tmp_path / "R", R)
    _write(tmp_path / "N", N)
    ok = DR.cross_check_dirs(tmp_path / "R", tmp_path / "N")
    assert ok["agree"] and ok["enumeration"]["R"] == 6
    # 4 桁でない名前の記録は glob（こちら）と正規表現（time_scoring）で拾い方が違う → 一致にしない
    (tmp_path / "R" / "trial_00006.json").write_text(json.dumps(ind_rec(7, 20.0, 5.0, trial=6)), encoding="utf-8")
    bad = DR.cross_check_dirs(tmp_path / "R", tmp_path / "N")
    assert not bad["agree"] and bad["enumeration"]["diffs"]


# ================================================================== 0155 の 2 節: 解析の入口の点検
def _plan():
    tl_run = {"time_limit_s": 60, "scored_at_s": {"primary": 30}}
    tl_task = {"step_timeout_s": 30, "retry": 1, "task_time_limit_s": 200}
    conds = [
        {"id": "RTC_naive", "kind": "run", "enabled": True, "experiment": "S4DRTC", "record_condition": "naive", "model": "R1v3",
         "trials_spec": "natural:190200:2", "seeds": [190200, 190201], "trials": 6, "time_limits": tl_run,
         "out_dir": "outputs\\v2eval\\S4DRTC\\naive", "job": "J_RTC_rotate"},
        {"id": "RC_R1v3_fall_with_hold", "kind": "run", "enabled": True, "experiment": "S4DREC",
         "record_condition": "R1v3_fall_with_hold", "model": "R1v3", "trials_spec": "induced:190500:3", "seeds": [190500, 190502],
         "trials": 3, "time_limits": tl_run, "out_dir": "outputs\\v2eval\\S4DREC\\R1v3_fall_with_hold",
         "job": "J_RC_fall_with_hold", "variant": "fall_with_hold"},
        {"id": "E7_EH", "kind": "task", "enabled": True, "experiment": "S4DE7", "record_condition": "EH", "model": "R1v3",
         "trials_spec": "190300:2", "seeds": [190300, 190301], "trials": 2, "time_limits": tl_task,
         "out_dir": "outputs\\v2eval\\S4DE7\\EH", "job": "E7_EH", "arm": "EH"},
        {"id": "E7_EX", "kind": "task", "enabled": False, "experiment": "S4DE7", "record_condition": "EX", "model": "R1v3",
         "trials_spec": "190300:2", "seeds": [190300, 190301], "trials": 2, "time_limits": tl_task,
         "out_dir": "outputs\\v2eval\\S4DE7\\EX", "job": "E7_EX", "arm": "EX"},
        {"id": "ST_home_on_grid", "kind": "run", "enabled": True, "experiment": "S4DSTART", "record_condition": "home_on_grid",
         "model": "R1v3", "trials_spec": "natural:190400:1", "seeds": [190400, 190400], "trials": 3, "time_limits": tl_run,
         "out_dir": "outputs\\v2eval\\S4DSTART\\home_on_grid", "job": "ST_home_on_grid", "start": "home", "prior": "on_grid"},
        {"id": "XPL_as_r1", "kind": "run", "enabled": True, "experiment": "S4XPL", "record_condition": "as_r1", "model": "R1v3",
         "trials_spec": "xpl:190420:2", "seeds": [190420, 190421], "trials": 2, "time_limits": tl_run,
         "out_dir": "outputs\\v2eval\\S4XPL\\as_r1", "job": "XPL_as_r1", "arm": "XPL_as", "rep": 1},
        {"id": "X2_gen", "kind": "x2", "enabled": True, "experiment": "outputs\\s4\\x2", "record_condition": "gen\\X2_gen",
         "model": None, "trials_spec": "44404:2", "seeds": [44404, 44405], "trials": 2, "time_limits": {},
         "out_dir": "outputs\\s4\\x2\\gen\\X2_gen", "job": "X2_gen"},
    ]
    jobs = [{"id": "J_RTC_rotate", "covers": ["RTC_naive", "RTC_ZEROS"], "cmd": "rotate"},
            {"id": "J_RC_fall_with_hold", "covers": ["RC_R1v3_fall_with_hold", "RC_N1v3_fall_with_hold"], "cmd": "rc"},
            {"id": "E7_EH", "covers": ["E7_EH"], "cmd": ".venv\\Scripts\\python.exe scripts\\98_s4_d_e7.py run --arm EH"},
            {"id": "E7_EX", "covers": ["E7_EX"], "cmd": "ex"},
            {"id": "ST_home_on_grid", "covers": ["ST_home_on_grid"], "cmd": "st"},
            {"id": "XPL_as_r1", "covers": ["XPL_as_r1"], "cmd": "xpl"}, {"id": "X2_gen", "covers": ["X2_gen"], "cmd": "x2"}]
    return {"conditions": conds, "jobs": jobs}


def _sha_diag(extra):
    return dict({"r96_sha256": "a" * 64, "module_sha256": "b" * 64}, **extra)


def _build(root: pathlib.Path, plan: dict, head="h1"):
    """計画どおりにそろった合成の記録を書く。"""
    env = dict(ENV, git_head=head)
    lim = {"time_limit_s": 60.0, "source": "96_s4_resume --time-limit-s"}
    tlim = {"step_timeout_s": 30.0, "retry": 1, "task_time_limit_s": 200.0, "source": "x"}
    for c in plan["conditions"]:
        d = root / pathlib.Path(c["out_dir"].replace("\\", "/"))
        if not c["enabled"]:
            continue
        if c["kind"] == "x2":
            d.mkdir(parents=True, exist_ok=True)
            (d / "gen_summary.json").write_text(json.dumps({"used_seeds": [44404, 44405], "script_sha256": "c" * 64,
                                                            "env_segments": [{"env": {"driver": "610.88"}, "seeds": [44404, 44405]}]}),
                                                encoding="utf-8")
            continue
        recs = []
        fam = c["id"].split("_", 1)[0]
        if c["kind"] == "task":
            for i in range(c["trials"]):
                recs.append({"run": i, "seed": 190300 + i, "model": "R1v3", "all_three_in_box": i == 0, "timed_out": False,
                             "truth_success_t": {"red": 10.0, "green": 50.0, "blue": 90.0} if i == 0 else {"red": 10.0},
                             "time_limits": tlim, "env": env,
                             "diag": _sha_diag({"diag": "D-E7", "arm": c["arm"], "condition": c["record_condition"]})})
            name = "run_{:04d}.json"
        else:
            kind, base, n = c["trials_spec"].split(":")
            seeds = [int(base) + i // 3 for i in range(3 * int(n))] if kind == "natural" else [int(base) + i for i in range(int(n))]
            for i, s in enumerate(seeds):
                tgt = ("red", "green", "blue")[i % 3] if kind == "natural" else "green"
                m = {"trial": i, "seed": s, "target": tgt, "success": i % 2 == 0, "t_success": 20.0 if i % 2 == 0 else None,
                     "time_limit_s": 60.0, "time_limits": lim, "env": env, "experiment": c["experiment"],
                     "condition": c["record_condition"], "model": {"name": c["model"]},
                     "induce": {"kind": None, "established": False}}
                if fam == "RTC":
                    m["diag"] = _sha_diag({"diag": "D-RTC", "arm": c["record_condition"], "shadow": False})
                elif fam == "RC":
                    m["induce"] = {"kind": "P2", "established": True, "t_established": [29.9, 30.0, 12.0][i], "variant": c["variant"]}
                    m["diag"] = _sha_diag({"name": "D_recovery", "variant": c["variant"], "model": c["model"], "induce_kind": "P2"})
                elif fam == "ST":
                    m["diag"] = _sha_diag({"diag": "D-single-start", "start": c["start"], "prior": c["prior"]})
                elif fam == "XPL":
                    m["diag"] = _sha_diag({"diag": "XPL", "arm": c["arm"], "rep": c["rep"]})
                recs.append(m)
            name = "trial_{:04d}.json"
        _write(d, recs, name)
        segs = [{"env": {k: env[k] for k in ("driver", "torch", "torch_cuda", "os_build")}, "trials": list(range(len(recs)))}]
        tl = tlim if c["kind"] == "task" else lim
        (d / "run.json").write_text(json.dumps({"n": len(recs), "time_limits": tl, "env_segments": segs}), encoding="utf-8")
        (d / "G_AUDIT.json").write_text(json.dumps({"met": True, "trials": len(recs)}), encoding="utf-8")
    pp = root / "plan.json"
    pp.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    return pp


def _run_audit(audit, root, *extra, ledger=False):
    out = root / "audit.json"
    args = ["--plan", str(root / "plan.json"), "--root", str(root), "--out", str(out), *extra]
    if not ledger:
        args.append("--no-ledger")                                       # 台帳の照合は別のテストで試す
    code = audit.main(args)
    return code, json.loads(out.read_text(encoding="utf-8"))


def _cond(res, cid):
    return next(r for r in res["conditions"] if r["id"] == cid)


def test_audit_all_ok(audit, tmp_path, capsys):
    _build(tmp_path, _plan())
    code, res = _run_audit(audit, tmp_path)
    assert code == 0, json.dumps([r for r in res["conditions"] if not r["ok"]], ensure_ascii=False)[:3000]
    assert res["n_conditions"] == 6 and res["ok"]                       # EX（enabled 偽）はフォルダが無いので点検しない
    rc = _cond(res, "RC_R1v3_fall_with_hold")["checks"]["double_count"]["detail"]
    assert rc["mine"]["at"]["30"] == {"successes": 2, "n": 2}           # t_established 30.0 の試行は 30 s の分母に入らない
    assert rc["mine"]["at"]["60"] == {"successes": 2, "n": 3}
    assert rc["mine"] == rc["score_96"]
    assert _cond(res, "X2_gen")["checks"]["g_audit"].get("na")
    assert "| RTC_naive |" in capsys.readouterr().out


def test_audit_missing_run_json_and_recall(audit, tmp_path, capsys):
    plan = _plan()
    _build(tmp_path, plan)
    (tmp_path / "outputs/v2eval/S4DRTC/naive/run.json").unlink()
    g = tmp_path / "outputs/v2eval/S4DE7/EH/G_AUDIT.json"
    g.write_text(json.dumps({"met": False}), encoding="utf-8")
    (tmp_path / "outputs/v2eval/S4DREC/R1v3_fall_with_hold/run.json").unlink()
    code, res = _run_audit(audit, tmp_path)
    assert code == 1 and res["n_bad"] == 3
    rtc = _cond(res, "RTC_naive")
    assert not rtc["checks"]["run_json"]["ok"] and "98_s4_d_rtc.py run --setting naive" in rtc["recall"]["cmd"]
    e7 = _cond(res, "E7_EH")
    assert not e7["checks"]["g_audit"]["ok"] and e7["recall"]["cmd"].endswith("--arm EH")
    rc = _cond(res, "RC_R1v3_fall_with_hold")["recall"]
    assert "--variants fall_with_hold --models R1v3" in rc["cmd"] and rc["note"]
    assert "呼び直す" in capsys.readouterr().out


def test_audit_trials_and_diag(audit, tmp_path):
    plan = _plan()
    _build(tmp_path, plan)
    d = tmp_path / "outputs/v2eval/S4DSTART/home_on_grid"
    (d / "trial_0002.json").unlink()                                     # 本数が足りない
    x = tmp_path / "outputs/v2eval/S4XPL/as_r1/trial_0001.json"
    m = json.loads(x.read_text(encoding="utf-8"))
    m["diag"]["arm"] = "XPL_home"                                        # フォルダ名（腕）と合わない
    x.write_text(json.dumps(m), encoding="utf-8")
    r = tmp_path / "outputs/v2eval/S4DRTC/naive/trial_0003.json"
    m = json.loads(r.read_text(encoding="utf-8"))
    m["seed"] = 190299                                                   # 種が計画と違う
    r.write_text(json.dumps(m), encoding="utf-8")
    (tmp_path / "outputs/v2eval/S4DE7/EH/run_0001.json").write_text("{", encoding="utf-8")   # 書きかけ
    code, res = _run_audit(audit, tmp_path)
    assert code == 1
    st = _cond(res, "ST_home_on_grid")["checks"]["trials"]
    assert not st["ok"] and st["detail"]["missing"] == [2]
    xd = _cond(res, "XPL_as_r1")["checks"]["diag"]
    assert not xd["ok"] and xd["detail"]["mismatch"][0]["item"].startswith("diag.arm")
    rt = _cond(res, "RTC_naive")["checks"]["trials"]
    assert not rt["ok"] and rt["detail"]["wrong_seed_or_index"]
    e7 = _cond(res, "E7_EH")["checks"]["trials"]
    assert not e7["ok"] and e7["detail"]["unreadable"]


def test_audit_limits_env_and_sha(audit, tmp_path):
    plan = _plan()
    _build(tmp_path, plan)
    p = tmp_path / "outputs/v2eval/S4DRTC/naive/trial_0000.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    m["time_limits"] = {"time_limit_s": 30.0}
    p.write_text(json.dumps(m), encoding="utf-8")
    p = tmp_path / "outputs/v2eval/S4XPL/as_r1/trial_0000.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    m["env"]["driver"] = "560.94"
    p.write_text(json.dumps(m), encoding="utf-8")
    p = tmp_path / "outputs/v2eval/S4DSTART/home_on_grid/trial_0001.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    m["diag"]["r96_sha256"] = "f" * 64
    p.write_text(json.dumps(m), encoding="utf-8")
    code, res = _run_audit(audit, tmp_path)
    assert code == 1
    assert not _cond(res, "RTC_naive")["checks"]["time_limits"]["ok"]
    env = _cond(res, "XPL_as_r1")["checks"]["env_segments"]
    assert not env["ok"] and [s["trials"] for s in env["detail"]["segments"]] == [[0], [1]]
    ver = _cond(res, "ST_home_on_grid")["checks"]["versions"]
    assert not ver["ok"] and len(ver["detail"]["sha256"]["r96_sha256"]) == 2


# git: 作業場所の .tools の git があればそれを、無ければ PATH の git（98_s4_d_audit._git と同じ順）
_GIT_EXE = ROOT / ".tools" / "git" / "cmd" / "git.exe"
GIT = str(_GIT_EXE) if _GIT_EXE.is_file() else shutil.which("git")
needs_git = pytest.mark.skipif(GIT is None, reason="git が無い")


@pytest.fixture
def git_on_path(monkeypatch):
    """点検の _git は tmp の作業場所に .tools の git が無いので PATH の git を使う。見つけた git を PATH の先頭に置く。"""
    monkeypatch.setenv("PATH", str(pathlib.Path(GIT).parent) + os.pathsep + os.environ.get("PATH", ""))


def _git(root, *a):
    subprocess.run([GIT, "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.invalid", *a], check=True,
                   capture_output=True)
    return subprocess.run([GIT, "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


@needs_git
def test_audit_heads_compare_child_files(audit, tmp_path, git_on_path):
    root = tmp_path
    (root / "src/recovla").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "docs").mkdir()
    (root / "src/recovla/a.py").write_text("x = 1\n", encoding="utf-8")
    (root / "scripts/96_s4_resume.py").write_text("y = 1\n", encoding="utf-8")
    subprocess.run([GIT, "init", "-q", str(root)], check=True)
    h1 = (_git(root, "add", "-A"), _git(root, "commit", "-q", "-m", "1"))[1]
    (root / "docs/note.md").write_text("掲示板\n", encoding="utf-8")                 # 子が読まないファイルだけ
    (root / "src/recovla/new_tool.py").write_text("z = 1\n", encoding="utf-8")      # 後から足しただけ
    h2 = (_git(root, "add", "-A"), _git(root, "commit", "-q", "-m", "2"))[1]
    (root / "src/recovla/a.py").write_text("x = 2\n", encoding="utf-8")            # 子が読むファイルを変えた
    h3 = (_git(root, "add", "-A"), _git(root, "commit", "-q", "-m", "3"))[1]
    same = audit.compare_heads(root, [h1, h2], "E7")
    assert same["changed"] == [] and same["added_only"] == ["src/recovla/new_tool.py"]
    diff = audit.compare_heads(root, [h1, h3], "E7")
    assert diff["changed"] == ["src/recovla/a.py"]
    plan = _plan()
    plan["conditions"] = [c for c in plan["conditions"] if c["id"] == "E7_EH"]
    _build(root, plan, head=h1)
    p = root / "outputs/v2eval/S4DE7/EH/run_0001.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    m["env"]["git_head"] = h2
    p.write_text(json.dumps(m), encoding="utf-8")
    code, res = _run_audit(audit, root)
    v = _cond(res, "E7_EH")["checks"]["versions"]
    assert code == 0 and v["ok"] and "同じ版" in v["detail"]["note"]
    m["env"]["git_head"] = h3
    p.write_text(json.dumps(m), encoding="utf-8")
    code, res = _run_audit(audit, root)
    assert code == 1 and not _cond(res, "E7_EH")["checks"]["versions"]["ok"]
    code, res = _run_audit(audit, root, "--no-git")
    assert code == 1


@needs_git
def test_audit_child_configs_exclude_demo(audit, tmp_path, git_on_path):
    """versions の照らし合わせは、子が読み込む設定のファイル（CHILD_CONFIGS）だけを見る。configs/demo の変更は数えない。"""
    root = tmp_path
    for rel, text in (("src/recovla/a.py", "x = 1\n"), ("configs/default.yaml", "a: 1\n"), ("configs/demo/video.yaml", "v: 1\n")):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    subprocess.run([GIT, "init", "-q", str(root)], check=True)
    h1 = (_git(root, "add", "-A"), _git(root, "commit", "-q", "-m", "1"))[1]
    (root / "configs/demo/video.yaml").write_text("v: 2\n", encoding="utf-8")       # 動画の場面だけ
    h2 = (_git(root, "add", "-A"), _git(root, "commit", "-q", "-m", "2"))[1]
    (root / "configs/default.yaml").write_text("a: 2\n", encoding="utf-8")           # 子が読む設定
    h3 = (_git(root, "add", "-A"), _git(root, "commit", "-q", "-m", "3"))[1]
    assert "configs/demo/video.yaml" not in audit.CHILD_CONFIGS
    assert audit.compare_heads(root, [h1, h2], "E7")["changed"] == []
    assert audit.compare_heads(root, [h1, h3], "E7")["changed"] == ["configs/default.yaml"]


def test_audit_ledger_parse_errors(audit, tmp_path, capsys):
    """0155 の 2-6: 台帳の照合の parse_errors が 0 でなければ欠け。点検する記録より古い結果は使わない。"""
    _build(tmp_path, _plan())
    lj = tmp_path / "ledger_check.json"
    lj.write_text(json.dumps({"problems": {"unreadable_records": [{"path": "outputs/x/trial_0000.json", "error": "e"}],
                                           "not_in_ledger": []}, "summary": {"parse_errors": 1}}), encoding="utf-8")
    code, res = _run_audit(audit, tmp_path, "--ledger-json", str(lj), ledger=True)
    assert code == 1 and res["conditions_ok"] and res["ledger"]["ok"] is False and res["ledger"]["parse_errors"] == 1
    lj.write_text(json.dumps({"problems": {"unreadable_records": [], "not_in_ledger": [1]}, "summary": {"parse_errors": 0}}),
                  encoding="utf-8")
    code, res = _run_audit(audit, tmp_path, "--ledger-json", str(lj), ledger=True)
    assert code == 0 and res["ledger"]["ok"] and res["ledger"]["other_problems"] == {"not_in_ledger": 1}
    os.utime(lj, (1_000_000, 1_000_000))                                 # 記録より古い結果
    code, res = _run_audit(audit, tmp_path, "--ledger-json", str(lj), ledger=True)
    assert code == 1 and "古い" in res["ledger"]["detail"]
    code, res = _run_audit(audit, tmp_path, ledger=True)                 # 台帳が無い作業場所でこの場で照合すると欠け
    assert code == 1 and "台帳が無い" in res["ledger"]["detail"]


def _plan_with_n():
    plan = _plan()
    rc = next(c for c in plan["conditions"] if c["id"] == "RC_R1v3_fall_with_hold")
    n = dict(rc, id="RC_N1v3_fall_with_hold", model="N1v3", record_condition="N1v3_fall_with_hold",
             out_dir="outputs\\v2eval\\S4DREC\\N1v3_fall_with_hold")
    plan["conditions"].append(n)
    return plan


def test_audit_gate1_cross_check(audit, tmp_path, capsys):
    """0155 の 2-7: gate1.py の入力の件数を、点検の列挙と数え方で照らす。無ければ not_available（--require-gate1 で欠け）。"""
    _build(tmp_path, _plan_with_n())
    code, res = _run_audit(audit, tmp_path)
    assert code == 0 and res["gate1_cross_check"]["status"] == "not_available"
    code, res = _run_audit(audit, tmp_path, "--require-gate1")
    assert code == 1
    gin = {"schema": "recovery_vla.s4_gate1_input/1",
           "R": {"settings": {"naive": {"natural_success_30": 3, "n_trials": 6, "radial_gap_mm": 1.0}}},
           "T": {"arms": {"EH": {"first_close_lift": {"k": 1, "n": 2}, "all_three_true": {"k": 1, "n": 2}}}},
           "C": {"fall_with_hold": {"by_L": {"30": {"recovery_R": {"k": 2, "n": 2}, "recovery_N": {"k": 2, "n": 2},
                                                    "paired": {"pairs": 2, "r_only": 0, "n_only": 0}}}}},
           "S": {"conditions": {}}}
    gp = tmp_path / "gate1_input.json"
    gp.write_text(json.dumps(gin), encoding="utf-8")
    code, res = _run_audit(audit, tmp_path, "--gate1-input", str(gp))
    g = res["gate1_cross_check"]
    assert code == 0 and g["status"] == "done" and g["ok"], g
    assert {r["item"] for r in g["rows"]} == {"R.naive", "T.EH.all_three_true", "C.fall_with_hold.30"}
    assert any(x.startswith("S:") for x in g["not_compared"])
    gin["C"]["fall_with_hold"]["by_L"]["30"]["paired"]["pairs"] = 3          # 食い違い
    gin["R"]["settings"]["naive"]["natural_success_30"] = 4
    gp.write_text(json.dumps(gin), encoding="utf-8")
    capsys.readouterr()
    code, res = _run_audit(audit, tmp_path, "--gate1-input", str(gp))
    g = res["gate1_cross_check"]
    assert code == 1 and not g["ok"]
    assert {m["item"]: m["keys"] for m in g["mismatch"]} == {"R.naive": ["natural_success_30"],
                                                            "C.fall_with_hold.30": ["paired.pairs"]}
    out = capsys.readouterr().out
    assert "natural_success_30" in out and '"gate1_input"' not in out          # 標準出力は項目の名前だけ


def test_audit_bundle1_plan_is_readable(audit, tmp_path):
    """本物の計画（docs/stage4/bundle1_defs/bundle1_plan_queue.json）で、記録の無い作業場所に当てると全条件が欠けになり、
    呼び直すコマンドが全部に付く（計画の読み方の確かめ。記録は要らない）。"""
    plan = json.loads((ROOT / "docs/stage4/bundle1_defs/bundle1_plan_queue.json").read_text(encoding="utf-8"))
    (tmp_path / "plan.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    code, res = _run_audit(audit, tmp_path)
    enabled = [c for c in plan["conditions"] if c.get("enabled", True)]
    assert code == 1 and res["n_conditions"] == len(enabled) and res["n_bad"] == len(enabled)
    assert all(r["recall"] and r["recall"]["cmd"] for r in res["conditions"])
    for c in enabled:
        if c["kind"] != "x2":
            assert len(audit.expected_trials(c)) == c["trials"], c["id"]
