"""テスト 1 の解析（実装 A: src/recovla/eval/test1.py、実装 B: scripts/98_s4_test1_b.py、照合: scripts/98_s4_test1.py）の検査。
合成の記録だけを使う（テスト用の種の帯の本物の記録は作らない・読まない。pytest の一時フォルダに書く）。CPU だけ。

使い方: PYTHONPATH=src python -m pytest -q tests/test_s4_test1.py -p no:cacheprovider
"""
import importlib.util
import json
import math
import pathlib
import sys

import pytest

from recovla.eval import test1 as T1

ROOT = pathlib.Path(__file__).resolve().parents[1]
COLORS = ("red", "green", "blue")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


B = _load(ROOT / "scripts" / "98_s4_test1_b.py", "s4_test1_b_t")
ENTRY = _load(ROOT / "scripts" / "98_s4_test1.py", "s4_test1_entry_t")
ENV = {"driver": "610.88", "torch": "2.11.0", "os_build": "x", "git_head": "abc"}


# ---------------------------------------------------------------- 合成の記録
def _cond(d: pathlib.Path):
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({"env_segments": [ENV]}), encoding="utf-8")
    (d / "G_AUDIT.json").write_text(json.dumps({"met": True}), encoding="utf-8")


def _trial(d, i, seed, target, success, t_success, model, induce=None, t_lim=60.0):
    m = {"trial": i, "seed": seed, "target": target, "success": bool(success), "t_success": t_success if success else None,
         "time_limit_s": t_lim, "time_limits": {"time_limit_s": t_lim}, "model": {"name": model}, "env": ENV,
         "induce": induce or {}, "experiment": d.parent.name, "condition": d.name}
    (d / f"trial_{i:04d}.json").write_text(json.dumps(m), encoding="utf-8")


def make_p1(d, model, outcomes, n=100, base=162000):
    """outcomes: [(t_established or None, t_success or None)]（種の順）。足りない分は誘発が成立しない試行。"""
    _cond(d)
    for i in range(n):
        te, ts = outcomes[i] if i < len(outcomes) else (None, None)
        ind = {"kind": "P1", "established": te is not None, "t_established": te}
        _trial(d, i, base + i, "red", ts is not None, ts, model, ind)


def p1_pair_outcomes(b, c, both, neither, extra=()):
    """R・N の outcome の列を、b（R だけ復帰）・c（N だけ）・both・neither の組で作る（誘発は 5 s に成立、成功は 20 s）。"""
    R, N = [], []
    for _ in range(b):
        R.append((5.0, 20.0)), N.append((5.0, None))
    for _ in range(c):
        R.append((5.0, None)), N.append((5.0, 20.0))
    for _ in range(both):
        R.append((5.0, 20.0)), N.append((5.0, 20.0))
    for _ in range(neither):
        R.append((5.0, None)), N.append((5.0, None))
    for r, nn in extra:
        R.append(r), N.append(nn)
    return R, N


def make_natural(d, model, last_seed, succ_fn):
    _cond(d)
    i = 0
    for s in range(161000, last_seed + 1):
        for col in COLORS:
            ok = succ_fn(i)
            _trial(d, i, s, col, ok, 20.0 if ok else None, model)
            i += 1


def make_e7(d, model, n, succ, v3, base=160000, interventions=None):
    _cond(d)
    for i in range(n):
        ok = succ(i)
        steps = [{"step": j, "color": c, "attempts": [{"attempt": 0, "t_begin": 10.0 * j, "t_judge": 10.0 * j + 8}],
                  "t_start": 10.0 * j, "judged_complete": ok or j == 0, "t_judge": 10.0 * j + 8 if (ok or j == 0) else None,
                  "t_end": 10.0 * j + 9} for j, c in enumerate(COLORS)]
        rets = []
        if interventions and i in interventions:                       # 1 回の出し直し（手順 1 に試み 2 つ）
            steps[1]["attempts"].insert(0, {"attempt": 0, "t_begin": 5.0, "t_judge": None})
            steps[1]["attempts"][1]["attempt"] = 1
            rets.append({"step": 1, "attempt": 0, "kind": "retry", "t_begin": 9.0})
        fin = {c: (ok or c == "red") for c in COLORS}
        truth = {"red": 8.0, **({"green": 18.0, "blue": 28.0} if ok else {})}
        m = {"run": i, "seed": base + i, "plan": {"steps": list(COLORS), "from_cache": True}, "steps": steps, "returns": rets,
             "stopped": None if ok else {"step": 1, "t": 70.0, "reply": "x"}, "truth_success_t": truth, "final_in_box": fin,
             "all_three_in_box": ok, "t_end": 30.0 if ok else 70.0, "timed_out": False, "model": model, "env": ENV,
             "time_limits": {"step_timeout_s": 30.0, "retry": 1, "task_time_limit_s": 200.0, "source": "x"}}
        if v3:
            m["v3"] = {"settings": {}}
        (d / f"run_{i:04d}.json").write_text(json.dumps(m), encoding="utf-8")


LAYOUT = {"root": None, "experiment": "S4T1SYN",
          "e7": {"v3": {"cond": "E7_v3", "model": "R1v3"}, "cur": {"cond": "E7_cur", "model": "R1v3"},
                 "n1v3_v3": {"cond": "E7_N", "model": "N1v3"}},
          "p1": {"1000": {"R": {"cond": "P1_R", "model": "R1v3"}, "N": {"cond": "P1_N", "model": "N1v3"}},
                 "1001": {"R": "P1_R1001", "N": "P1_N1001"}, "1002": {"R": "P1_R1002", "N": "P1_N1002"}},
          "natural": {"1000": {"R": "nat_R", "N": "nat_N"}, "1001": {"R": "nat_R1001", "N": "nat_N1001"},
                      "1002": {"R": "nat_R1002", "N": "nat_N1002"}}}


def build(tmp, h1=(10, 3), h2=((5, 1), (2, 0)), h3=(13, 0), e7_n=100, nat_r=lambda i: i % 3 != 0, nat_n=lambda i: i % 2 == 0,
          rtc=None, h3_extra=()):
    """h1 = (v3 だけ成功 b, 今だけ成功 c)、h2 = 層 1001・1002 の (b, c)、h3 = 種 1000 の (b, c)。"""
    base = tmp / "v2eval" / "S4T1SYN"
    b1, c1 = h1
    make_e7(base / "E7_v3", "R1v3", e7_n, lambda i: i < b1 + 20, True, interventions={0, 1})
    make_e7(base / "E7_cur", "R1v3", e7_n, lambda i: (i < 20) or (b1 + 20 <= i < b1 + 20 + c1), False)
    make_e7(base / "E7_N", "N1v3", e7_n, lambda i: i < 5, True)
    for layer, (b, c) in (("1001", h2[0]), ("1002", h2[1]), ("1000", h3)):
        R, N = p1_pair_outcomes(b, c, 10, 20, extra=h3_extra if layer == "1000" else ())
        sfx = "" if layer == "1000" else layer
        make_p1(base / f"P1_R{sfx}", "R1v3" if layer == "1000" else "R1v3s" + layer, R)
        make_p1(base / f"P1_N{sfx}", "N1v3" if layer == "1000" else "N1v3s" + layer, N)
    for layer, last in (("1000", 161065), ("1001", 161032), ("1002", 161032)):
        sfx = "" if layer == "1000" else layer
        make_natural(base / f"nat_R{sfx}", "x", last, nat_r)
        make_natural(base / f"nat_N{sfx}", "x", last, nat_n)
    lay = json.loads(json.dumps(LAYOUT))
    lay["root"] = str(tmp / "v2eval")
    if rtc:
        # rtc = (自然の成否の関数, R1v3＋RTC の P1 の outcome, 本数, N1v3＋RTC の P1 の outcome（案 B だけ。None なら置かない）)
        nat_fn, p1_out, n_p1, p1_n_out = (tuple(rtc) + (50, None))[:4] if len(rtc) == 2 else rtc
        make_natural(base / "nat_rtc", "x", 161065, nat_fn)
        make_p1(base / "P1_rtc", "R1v3", p1_out, n=n_p1)
        lay["rtc"] = {"natural": "nat_rtc", "p1": "P1_rtc"}
        if p1_n_out is not None:
            make_p1(base / "P1_rtc_N", "N1v3", p1_n_out, n=100)
            lay["rtc"]["p1_n"] = {"cond": "P1_rtc_N", "model": "N1v3"}
    return lay


PARAMS = {"h1": True, "e7_n": 100, "rtc_arm": False, "plan": "A", "c4_on_time": None}


def both(lay, params):
    a = T1.analyze(lay, params)
    b = B.run_b(lay, params)
    return a, b, ENTRY.compare(a, b)


# ---------------------------------------------------------------- 手計算の p 値
def test_exact_binomial_matches_hand_examples():
    for b, c, want in ((13, 0, 2 / 2 ** 13), (7, 1, 18 / 256), (10, 3, 756 / 8192), (0, 0, 1.0), (4, 4, 1.0), (1, 0, 1.0)):
        assert math.isclose(T1.binom_two_sided(b, c), want, rel_tol=1e-12), (b, c)
        assert math.isclose(B.exact_two_sided(b, c), want, rel_tol=1e-12), (b, c)
        assert math.isclose(B.exact_two_sided(c, b), want, rel_tol=1e-12)


def test_holm_order_and_adjusted_p():
    ps = {"H1": 18 / 256, "H2": 2 / 2 ** 13, "H3": 756 / 8192}
    for h in (T1.holm(ps), B.holm_b(ps)):
        assert h["order"] == ["H2", "H1", "H3"]
        assert math.isclose(h["by"]["H2"]["p_holm"], 3 * ps["H2"])
        assert math.isclose(h["by"]["H1"]["p_holm"], 2 * ps["H1"])                  # 0.140625
        assert math.isclose(h["by"]["H3"]["p_holm"], 2 * ps["H1"])                  # 累積の最大（1 × 0.0923 より大きい前の値）
        assert [h["by"][k]["rejected_stepdown"] for k in ("H2", "H1", "H3")] == [True, False, False]
    # p(1) が 0.05/3 ちょうどなら棄却（<=）。次で止まったら、その後ろは p が小さくても全部「通らない」
    for h in (T1.holm({"a": 0.05 / 3, "b": 0.04, "c": 0.9}), B.holm_b({"a": 0.05 / 3, "b": 0.04, "c": 0.9})):
        assert h["order"] == ["a", "b", "c"]
        assert [h["by"][k]["rejected_stepdown"] for k in h["order"]] == [True, False, False]
        assert math.isclose(h["by"]["b"]["p_holm"], 0.08) and math.isclose(h["by"]["c"]["p_holm"], 0.9)


def test_wilson_and_newcombe_agree_between_a_and_b():
    for k, n in ((0, 10), (3, 10), (10, 10), (85, 99), (1, 1)):
        a, b = T1.wilson(k, n), B.wilson95(k, n)
        assert all(math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-15) for x, y in zip(a, b))
    for cell in ((30, 10, 4, 56), (0, 5, 0, 5), (10, 0, 0, 10), (40, 20, 20, 20)):
        a, b = T1.newcombe_paired(*cell), B.newcombe_pair(*cell)
        assert math.isclose(a["diff"], b["diff"]) and all(math.isclose(x, y, abs_tol=1e-12) for x, y in zip(a["ci95"], b["ci95"]))
    a, b = T1.newcombe_unpaired(30, 50, 35, 50), B.newcombe_indep(30, 50, 35, 50)
    assert all(math.isclose(x, y, abs_tol=1e-12) for x, y in zip(a["ci95"], b["ci95"]))
    assert T1.wilson(0, 0) is None and B.wilson95(0, 0) is None


# ---------------------------------------------------------------- 全体（A と B の一致）
def test_full_analysis_a_equals_b_and_hand_values(tmp_path):
    lay = build(tmp_path)
    a, b, diffs = both(lay, PARAMS)
    assert diffs == [], diffs[:10]
    assert a["status"] == "complete"
    pr = a["primary"]
    assert (pr["H1"]["b"], pr["H1"]["c"], pr["H1"]["pairs"]) == (10, 3, 100)
    assert math.isclose(pr["H1"]["p"], 756 / 8192)
    assert (pr["H2"]["b"], pr["H2"]["c"]) == (7, 1) and math.isclose(pr["H2"]["p"], 18 / 256)
    assert pr["H2"]["layers"]["1001"] == {"pairs": 36, "b": 5, "c": 1, "both": 10, "neither": 20}
    assert (pr["H3"]["b"], pr["H3"]["c"]) == (13, 0) and math.isclose(pr["H3"]["p"], 2 / 2 ** 13)
    assert a["holm"]["order"] == ["H3", "H2", "H1"]
    assert pr["H3"]["established"] and not pr["H2"]["established"] and not pr["H1"]["established"]
    e7 = a["secondary"]["e7_arms"]["v3"]
    assert e7["all_three"] == 30 and e7["interventions"]["retry"] == 2 and e7["success_at_k"] == {"0": 28, "1": 30, "2": 30}
    assert a["secondary"]["e7_r_vs_n"]["b"] == 25 and a["secondary"]["natural"]["1000"]["30"]["pairs"] == 198
    fs = a["face_switch"]
    assert fs["c1"]["pass"] and fs["c2"]["pass"] and fs["c2"]["g_nat_diff"] == 0.0 and not fs["c3"]["pass"]
    assert fs["switch"] is None and fs["c1_to_c3"] is False                          # 条件 4 は人が確かめる


def _ledger(tmp_path, parse_errors=0):
    p = tmp_path / "ledger_check.json"                                              # 記録より後に書く（新しい）
    p.write_text(json.dumps({"problems": {"unreadable_records": [{"path": "x"}] * parse_errors},
                             "summary": {"parse_errors": parse_errors}}), encoding="utf-8")
    return p


def _entry_args(tmp_path, lay, params, out, ledger):
    lp, pp = tmp_path / "layout.json", tmp_path / "params.json"
    lp.write_text(json.dumps(lay), encoding="utf-8")
    pp.write_text(json.dumps(params), encoding="utf-8")
    return ["check", "--layout", str(lp), "--params", str(pp), "--out", str(out), "--ledger-json", str(ledger)]


def test_entry_check_writes_only_when_agree(tmp_path, monkeypatch):
    lay = build(tmp_path)
    out = tmp_path / "out" / "r.json"
    args = _entry_args(tmp_path, lay, PARAMS, out, _ledger(tmp_path))
    assert ENTRY.main(args) == 0
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["double_count"]["agree"] and r["params"]["e7_n"] == 100 and out.with_suffix(".md").is_file()
    assert r["entry_audit"]["versions"]["ok"] and r["entry_audit"]["ledger"]["parse_errors"] == 0
    # B だけがずれたら止まり、判定を書かない
    out2 = tmp_path / "out2" / "r.json"
    orig = B.exact_two_sided
    monkeypatch.setattr(B, "exact_two_sided", lambda b, c: min(1.0, orig(b, c) * 1.001))
    monkeypatch.setattr(ENTRY, "_load_b", lambda: B)
    args[args.index("--out") + 1] = str(out2)
    assert ENTRY.main(args) == 1
    assert not out2.exists()


def test_compare_is_relative_for_small_p():
    # 絶対 1e-9 では見逃す小さな p の食い違い（相対 1e-6）を、相対で捉える
    assert ENTRY.compare({"p": 2.44e-12}, {"p": 2.44e-12 * (1 + 1e-6)}) != []
    assert ENTRY.compare({"p": 2.44e-12}, {"p": 2.44e-12 * (1 + 1e-12)}) == []
    assert ENTRY.compare({"holm": {"by": {"H3": {"p_holm": 1e-20}}}}, {"holm": {"by": {"H3": {"p_holm": 2e-20}}}}) != []
    assert ENTRY.compare({"x": 0.0}, {"x": 1e-15}) == []                           # 区間の端などは 0 の近くで絶対 1e-12


# ---------------------------------------------------------------- 条件をまたぐ入口の点検（0155 の 2-5・2-6）
class _FakeAudit:
    def __init__(self, changed):
        self.changed = changed

    def order_heads(self, root, heads):
        return sorted(heads)

    def compare_heads(self, root, order, fam):
        return {"heads": order, "changed": self.changed, "added_only": []}


def _two_heads(lay):
    p = pathlib.Path(lay["root"]) / "S4T1SYN" / "P1_N1001" / "trial_0050.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    m["env"] = dict(m["env"], git_head="def")
    p.write_text(json.dumps(m), encoding="utf-8")


@pytest.mark.parametrize("changed, code", [([], 0), (["src/recovla/harness/loop.py"], 3)])
def test_entry_versions_across_heads(tmp_path, monkeypatch, changed, code):
    lay = build(tmp_path)
    _two_heads(lay)
    a, b, diffs = both(lay, PARAMS)
    assert diffs == [] and a["status"] == "complete" and a["checks"]["p1.1001.N"]["git_heads"] == ["abc", "def"]
    monkeypatch.setattr(ENTRY, "_load_script", lambda name, mod: _FakeAudit(changed))
    out = tmp_path / "out" / "r.json"
    assert ENTRY.main(_entry_args(tmp_path, lay, PARAMS, out, _ledger(tmp_path))) == code
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["entry_audit"]["versions"]["ok"] is (not changed)
    assert (r["primary"] is None) is bool(changed)
    if changed:
        assert "条件をまたぐ点検" in out.with_suffix(".md").read_text(encoding="utf-8")


def test_entry_ledger_parse_errors_and_stale(tmp_path):
    lay = build(tmp_path)
    out = tmp_path / "out" / "r.json"
    assert ENTRY.main(_entry_args(tmp_path, lay, PARAMS, out, _ledger(tmp_path, parse_errors=1))) == 3
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["status"] == "incomplete" and r["face_switch"] is None and r["entry_audit"]["ledger"]["parse_errors"] == 1
    # 記録より古い台帳の照合は使わない
    led = _ledger(tmp_path)
    import os
    old = led.stat().st_mtime - 3600
    os.utime(led, (old, old))
    assert ENTRY.main(_entry_args(tmp_path, lay, PARAMS, out, led)) == 3
    assert not json.loads(out.read_text(encoding="utf-8"))["entry_audit"]["ledger"]["ok"]


# ---------------------------------------------------------------- 分母の境界（ちょうど L 秒）
def test_boundary_established_exactly_L_is_excluded_success_exactly_L_is_included(tmp_path):
    extra = [((30.0, 40.0), (30.0, None)),        # 成立がちょうど 30 s: 30 s の分母に入れない（60 s では入る）
             ((29.9, 30.0), (29.9, None)),        # 成功がちょうど 30 s: 30 s で復帰に数える
             ((29.9, 30.1), (29.9, None))]        # 30 s を過ぎた成功: 30 s では復帰しない（60 s では復帰）
    lay = build(tmp_path, h3=(0, 0), h3_extra=extra)
    a, b, diffs = both(lay, PARAMS)
    assert diffs == []
    h3 = a["primary"]["H3"]
    assert h3["pairs"] == 32 and h3["b"] == 1 and h3["neither"] == 21               # 30 = 10 + 20 + 2 組（ちょうど 30 の成立は除く）
    l60 = a["secondary"]["p1_layers_by_limit"]["60"]["layers"]["1000"]
    assert l60["pairs"] == 33 and l60["b"] == 3
    assert a["secondary"]["p1_rates"]["p1.1000.R"]["30"]["n"] == 32


# ---------------------------------------------------------------- 未完（入口の点検）
def _edit(p, fn):
    m = json.loads(p.read_text(encoding="utf-8"))
    fn(m)
    p.write_text(json.dumps(m), encoding="utf-8")


@pytest.mark.parametrize("breaker", ["missing_trial", "gaudit", "band", "limits", "env", "model", "induce", "executor",
                                     "no_segments", "torch", "git_head", "diag_sha", "label", "run_limits"])
def test_incomplete_is_not_decided(tmp_path, breaker):
    lay = build(tmp_path)
    base = pathlib.Path(lay["root"]) / "S4T1SYN"
    if breaker == "no_segments":
        (base / "nat_N" / "run.json").write_text(json.dumps({}), encoding="utf-8")
    elif breaker == "torch":                                                      # 試行の env の torch が途中で変わった
        _edit(base / "nat_R1001" / "trial_0010.json", lambda m: m["env"].update(torch="2.12.0"))
    elif breaker == "git_head":
        _edit(base / "E7_N" / "run_0003.json", lambda m: m["env"].pop("git_head"))
    elif breaker == "diag_sha":
        for i, v in ((0, "aa"), (1, "bb")):
            _edit(base / "P1_R1002" / f"trial_{i:04d}.json", lambda m, v=v: m.update(diag={"script_sha256": v}))
    elif breaker == "label":                                                      # 別の条件の記録が混ざった
        _edit(base / "P1_N" / "trial_0004.json", lambda m: m.update(condition="P1_N1001"))
    elif breaker == "run_limits":
        (base / "nat_R" / "run.json").write_text(json.dumps({"env_segments": [ENV], "time_limits": {"time_limit_s": 30.0}}),
                                                 encoding="utf-8")
    elif breaker == "missing_trial":
        (base / "P1_N1001" / "trial_0099.json").unlink()
    elif breaker == "gaudit":
        (base / "nat_R" / "G_AUDIT.json").write_text(json.dumps({"met": False}), encoding="utf-8")
    elif breaker == "band":
        p = base / "E7_cur" / "run_0000.json"
        m = json.loads(p.read_text(encoding="utf-8"))
        m["seed"] = 159999
        p.write_text(json.dumps(m), encoding="utf-8")
    elif breaker == "limits":
        p = base / "P1_R" / "trial_0003.json"
        m = json.loads(p.read_text(encoding="utf-8"))
        m["time_limit_s"] = 30.0
        p.write_text(json.dumps(m), encoding="utf-8")
    elif breaker == "env":
        (base / "P1_R" / "run.json").write_text(json.dumps({"env_segments": [ENV, dict(ENV, driver="560.94")]}), encoding="utf-8")
    elif breaker == "model":
        p = base / "E7_v3" / "run_0005.json"
        m = json.loads(p.read_text(encoding="utf-8"))
        m["model"] = "N1v3"
        p.write_text(json.dumps(m), encoding="utf-8")
    elif breaker == "induce":
        p = base / "P1_N" / "trial_0002.json"
        m = json.loads(p.read_text(encoding="utf-8"))
        m["induce"]["kind"] = "P2"
        p.write_text(json.dumps(m), encoding="utf-8")
    elif breaker == "executor":
        p = base / "E7_cur" / "run_0007.json"
        m = json.loads(p.read_text(encoding="utf-8"))
        m["v3"] = {}
        p.write_text(json.dumps(m), encoding="utf-8")
    a, b, diffs = both(lay, PARAMS)
    assert diffs == [], diffs
    assert a["status"] == "incomplete" and a["primary"] is None and a["face_switch"] is None
    bad = [k for k, v in a["checks"].items() if not v["ok"]]
    assert len(bad) == 1, bad
    assert "未完" in T1.summary_md(a)


def test_missing_condition_and_band_extension(tmp_path):
    lay = build(tmp_path)
    del lay["p1"]["1002"]
    a, b, diffs = both(lay, PARAMS)
    assert diffs == [] and a["status"] == "incomplete" and a["missing_conditions"] == ["p1.1002.R", "p1.1002.N"]
    # H2 の層を 1001 だけにする（P-7: 種 1002 が間に合わなかった）なら、そろう
    a, b, diffs = both(lay, dict(PARAMS, h2_layers=["1001"]))
    assert diffs == [] and a["status"] == "complete" and list(a["primary"]["H2"]["layers"]) == ["1001"]
    # E7 の 200 種は、帯の追加（D1）を採ったときだけ
    lay2 = build(tmp_path / "x", e7_n=200)
    a, b, diffs = both(lay2, dict(PARAMS, e7_n=200))
    assert diffs == [] and a["status"] == "incomplete"
    a, b, diffs = both(lay2, dict(PARAMS, e7_n=200, e7_band_extended=True))
    assert diffs == [] and a["status"] == "complete"


def test_without_h1_two_hypotheses(tmp_path):
    lay = build(tmp_path)
    a, b, diffs = both(lay, {"h1": False, "rtc_arm": False, "plan": "A"})
    assert diffs == [] and a["status"] == "complete"
    assert set(a["primary"]) == {"H2", "H3"} and a["holm"]["m"] == 2
    assert a["face_switch"]["c3"] == {"by": "H1", "present": False, "pass": False}


def test_params_placeholders_are_required_and_checked():
    for impl in (T1.params_with_defaults, B.fill):
        with pytest.raises(ValueError):
            impl({"e7_n": 100, "plan": "A"})                                         # h1（P-1）・rtc_arm（P-3）に既定はない
        with pytest.raises(ValueError):
            impl(dict(PARAMS, e7_n=120))                                             # P-4 は 100・150・200
        with pytest.raises(ValueError):
            impl(dict(PARAMS, plan="B"))                                             # 案 B は RTC の腕が要る
        with pytest.raises(ValueError):
            impl(dict(PARAMS, plan="B", rtc_arm=True, rtc_p1_n=50))                  # 案 B の RTC の P1 は 100
        assert impl(dict(PARAMS, p_fill={"P-2": "ES"}))["p_fill"] == {"P-2": "ES"}


def test_required_conditions_include_natural_and_rtc(tmp_path):
    lay = build(tmp_path)
    del lay["natural"]["1000"]
    a, b, diffs = both(lay, PARAMS)
    assert diffs == [] and a["status"] == "incomplete"
    assert a["missing_conditions"] == ["natural.1000.R", "natural.1000.N"]
    lay = build(tmp_path / "y")
    a, b, diffs = both(lay, dict(PARAMS, rtc_arm=True))                             # P-3 で置くと決めたのに RTC の腕が無い
    assert diffs == [] and a["missing_conditions"] == ["rtc.natural", "rtc.p1"]


# ---------------------------------------------------------------- 掲示板 0157 の条件
def test_face_switch_all_conditions(tmp_path):
    lay = build(tmp_path, h1=(20, 3), h2=((8, 0), (6, 0)), h3=(13, 0))
    a, b, diffs = both(lay, dict(PARAMS, c4_on_time=True))
    assert diffs == []
    pr = a["primary"]
    assert pr["H1"]["established"] and pr["H2"]["established"] and pr["H3"]["established"]
    fs = a["face_switch"]
    assert fs["c1"]["pass"] and fs["c2"]["pass"] and fs["c3"]["pass"] and fs["switch"] is True
    a, b, diffs = both(lay, dict(PARAMS, c4_on_time=False))
    assert diffs == [] and a["face_switch"]["switch"] is False


@pytest.mark.parametrize("nat_drop, mode, g_nat", [(0, "point", True), (12, "point", False), (8, "interval", False)])
def test_face_switch_plan_b_guard(tmp_path, nat_drop, mode, g_nat):
    base_fn = lambda i: i % 3 != 0                                                   # noqa: E731  naive: 132/198
    rtc_nat = lambda i: base_fn(i) and not (i % 3 == 1 and i < 3 * nat_drop)         # noqa: E731  nat_drop 本だけ成功を落とす
    R, N = p1_pair_outcomes(13, 0, 10, 20)
    lay = build(tmp_path, nat_r=base_fn, rtc=(rtc_nat, R, 100, N))
    a, b, diffs = both(lay, dict(PARAMS, plan="B", rtc_arm=True, rtc_p1_n=100, guard_mode=mode))
    assert diffs == [], diffs[:5]
    c2 = a["face_switch"]["c2"]
    assert math.isclose(c2["g_nat_diff"], -nat_drop / 198)
    assert c2["g_nat"] == g_nat
    assert a["secondary"]["rtc"]["natural_vs_naive"]["pairs"] == 198


def test_plan_b_h3_uses_rtc_arms(tmp_path):
    """案 B の H3（0157 の条件 1）は改良版の R と N＝RTC の設定の R1v3 対 N1v3（事前登録の案 12-2）。naive の種 1000 ではない。"""
    R, N = p1_pair_outcomes(2, 1, 10, 20)                                            # RTC の腕: 2 対 1
    lay = build(tmp_path, h3=(13, 0), rtc=(lambda i: i % 3 != 0, R, 100, N))      # naive の種 1000: 13 対 0
    a, b, diffs = both(lay, dict(PARAMS, plan="B", rtc_arm=True, rtc_p1_n=100, c4_on_time=True))
    assert diffs == [], diffs[:5]
    h3 = a["primary"]["H3"]
    assert h3["arms"] == ["rtc.p1", "rtc.p1_n"] and (h3["b"], h3["c"]) == (2, 1)
    assert not h3["established"] and not a["face_switch"]["c1"]["pass"] and a["face_switch"]["switch"] is False
    assert a["secondary"]["p1_layers_by_limit"]["30"]["layers"]["1000"]["b"] == 13   # naive の比較は副次に残る
    assert "rtc.p1_n" in a["secondary"]["p1_rates"]
    # 案 A なら naive の種 1000 の腕
    lay2 = build(tmp_path / "z", h3=(13, 0))
    a, b, diffs = both(lay2, PARAMS)
    assert diffs == [] and a["primary"]["H3"]["arms"] == ["p1.1000.R", "p1.1000.N"]
