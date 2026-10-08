"""束 6 (ii) U4S: 入口 scripts/98_s4_u4s.py（98_s4_u4.py を包み、立て直しの計画役だけを claude-sonnet-5-5 にする）。
CPU だけ。ネットワーク・GPU・シミュレーションは使わない（LLM は偽のクライアント、実行器は偽の io・知覚・判定、96 の cmd_main は偽物）。

使い方: .venv\\Scripts\\python.exe -m pytest -q tests\\test_s4_u4s.py -p no:cacheprovider
読むもの: scripts/98_s4_u4s.py・scripts/98_s4_u4.py・scripts/96_s4_resume.py・tests/test_s4_u4.py（偽物の部品。importlib）。
書くもの: pytest の一時フォルダだけ。
"""
import importlib.util
import json
import pathlib
import sys
import types

import numpy as np
import pytest

from recovla.planner import decompose as D
from recovla.planner import replan_multi as RM
from recovla.planner import replan_s4 as R4
from recovla.runtime.executor import TaskRuntime

ROOT = pathlib.Path(__file__).resolve().parents[1]
SONNET, HAIKU = "claude-sonnet-5-5", "claude-haiku-5-5"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


T = _load(ROOT / "tests" / "test_s4_u4.py", "test_s4_u4_parts_for_u4s")     # 偽物の部品だけを借りる


@pytest.fixture(scope="module")
def s():
    return _load(ROOT / "scripts" / "98_s4_u4s.py", "s4_u4s_entry")


class Boom:
    """呼ばれたら失敗する（本物の API・鍵・GPU に届かないことの確かめ）。"""

    def __init__(self, what):
        self.what = what

    def __call__(self, *a, **kw):
        raise AssertionError(f"呼ばないはずの {self.what} を呼んだ")


@pytest.fixture(autouse=True)
def _no_real_api(tmp_path, monkeypatch):
    monkeypatch.setattr(R4, "CACHE", tmp_path / "llm_cache")
    monkeypatch.setattr(R4, "client", Boom("本物の API の口（R4.client）"))
    monkeypatch.setattr(D, "client", Boom("本物の API の口（decompose.client）"))


# ---------------------------------------------------------------- 帯と拒む場合
def test_band_check_is_the_same_as_98(s):
    s98 = s.s98
    for r in (range(191400, 191440), range(191400, 191500), range(191430, 191441), range(191490, 191501), range(190300, 190340),
              range(44400, 44402), range(44404, 44406), range(160000, 160002)):
        for sm in (False, True):
            assert s.band_check(r, sm) == s98.band_check(r, sm)
    assert s.band_check(range(191400, 191440)) == ""
    assert "X2" in s.band_check(range(44404, 44406), allow_smoke=True)
    assert s.band_check(range(44400, 44402), allow_smoke=True) == ""
    # 改訂 3 で s4_gates.json の bundle6_u4 が 191400〜191499 に広がった（98 と同じ）。191500 からは拒む
    g = json.loads((ROOT / "configs" / "s4_gates.json").read_text(encoding="utf-8"))
    assert next(x for x in g["bands"]["allocations"] if x["id"] == s98.ALLOC)["range"] == [191400, 191499]
    assert s.band_check(range(191400, 191500)) == ""
    assert s.band_check(range(191490, 191501)) and s.band_check(range(191500, 191502))


def fake_r96(s, monkeypatch, out_root, calls):
    """本物の 96 を読み、cmd_main・load_82・load_ops だけを偽物にする（GPU・シミュレーション・書き込みに届かない）。"""
    r96 = s.load_r96()

    def cmd_main(a, v82, ops):
        calls.append({"a": a, "r96": r96})
        return 0
    monkeypatch.setattr(r96, "cmd_main", cmd_main)
    monkeypatch.setattr(r96, "load_82", lambda allow: types.SimpleNamespace(OUT=out_root))
    monkeypatch.setattr(r96, "load_ops", lambda: types.SimpleNamespace())
    monkeypatch.setattr(r96, "run_json_text", lambda *x, **kw: json.dumps({"experiment": "S4U4ST"}))
    monkeypatch.setattr(s, "load_r96", lambda: r96)
    return r96


BASE = ["--experiment", "S4U4ST", "--condition", "U4S", "--model", "R1v3"]


@pytest.mark.parametrize("args", [
    ["--trials", "160000:2", "--dry-run"],                                    # 帯の外
    ["--trials", "191490:20", "--dry-run"],                                   # 帯からはみ出す（改訂 3 の後の帯の端）
    ["--trials", "190300:2", "--dry-run"],                                    # D-E7 の帯
    ["--trials", "44400:2", "--dry-run"],                                     # smoke は --allow-smoke のときだけ
    ["--trials", "44404:2", "--dry-run", "--allow-smoke"],                    # X2 の生成の帯
    ["--trials", "44420:6", "--dry-run", "--allow-smoke"],
    ["--trials", "191400:2", "--dry-run", "--exec-interval", "10"],
    ["--trials", "191400:2", "--dry-run", "--planner", "legacy"],
    ["--trials", "191400:2", "--dry-run", "--arm", "U4"],                     # この入口の腕は U4S だけ
])
def test_run_refuses_before_running(s, args, monkeypatch, tmp_path):
    calls = []
    fake_r96(s, monkeypatch, tmp_path, calls)
    if "--arm" in args:
        with pytest.raises(SystemExit):
            s.cmd_run(BASE + args)
    else:
        assert s.cmd_run(BASE + args) == 3
    assert calls == []


@pytest.mark.parametrize("cond", ["U4", "U0_run1", "U0_run2"])
def test_run_refuses_condition_names_of_98(s, cond, monkeypatch, tmp_path):
    calls = []
    fake_r96(s, monkeypatch, tmp_path, calls)
    assert s.cmd_run(["--experiment", "X", "--condition", cond, "--model", "R1v3", "--trials", "191400:2", "--dry-run"]) == 3
    assert calls == []


def test_run_accepts_191400_100_after_charter_rev3(s, monkeypatch, tmp_path):
    calls = []
    fake_r96(s, monkeypatch, tmp_path, calls)
    assert s.cmd_run(BASE + ["--trials", "191400:100", "--dry-run"]) == 0                # 改訂 3 の帯（100 種）
    assert s.cmd_run(BASE + ["--trials", "191500:2", "--dry-run"]) == 3                  # 帯の外


def test_run_needs_api_key_unless_dry_run(s, monkeypatch, tmp_path):
    calls = []
    fake_r96(s, monkeypatch, tmp_path, calls)
    monkeypatch.setattr(D, "_api_key", lambda: None)
    assert s.cmd_run(BASE + ["--trials", "191400:1"]) == 3 and calls == []
    assert s.cmd_run(BASE + ["--trials", "191400:1", "--dry-run"]) == 0 and len(calls) == 1     # --dry-run は鍵を見ない


def test_dry_run_sets_up_u4s(s, monkeypatch, tmp_path):
    calls = []
    fake_r96(s, monkeypatch, tmp_path, calls)
    monkeypatch.setattr(D, "_api_key", Boom("鍵の読み取り"))                       # --dry-run は鍵を読まない
    assert s.cmd_run(BASE + ["--trials", "44400:2", "--allow-smoke", "--dry-run"]) == 0
    a, r96 = calls[0]["a"], calls[0]["r96"]
    assert a.dry_run and a.exec_interval == 6 and a.no_safety and a.planner == "s4"
    assert a.u4_arm == "U4S" and a.u4_replanner_model == SONNET
    for k in ("u4_arm", "u4_replan_sha256", "u4_executor_sha256", "u4_replanner_model", "u4_replan_multi_sha256"):
        assert k in r96.SPEC_KEYS
    assert a.u4_replan_multi_sha256 == s.s98.sha256_file(ROOT / "src" / "recovla" / "planner" / "replan_multi.py")
    assert r96.Engine.__name__ == "U4SEngine" and r96.Engine.u4s_step.model == SONNET
    # run.json の "u4" に腕・モデル・SHA-256 が入る（元の run_json_text は fake_r96 の偽物）
    d = json.loads(r96.run_json_text(a, "task", [], 0.0, 6, {}))
    assert d["experiment"] == "S4U4ST"
    assert d["u4"]["arm"] == "U4S" and d["u4"]["replanner_model"] == SONNET and d["u4"]["planner_model"] == HAIKU
    assert d["u4"]["replanner_variant"] == "s4_replan_multi_sonnet55" and d["u4"]["script"] == "98_s4_u4s.py"
    assert set(d["u4"]["files_sha256"]) == set(s.FILES)
    assert "scripts/98_s4_u4s.py" in s.FILES and "src/recovla/planner/replan_multi.py" in s.FILES
    assert set(s.s98.FILES) <= set(s.FILES)


# ---------------------------------------------------------------- U4 の記録を続きから回さない
def _put(d, name, obj):
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def test_foreign_records(s, tmp_path):
    assert s.foreign_records(tmp_path / "none") == []
    ok = tmp_path / "ok"
    _put(ok, "resume_spec.json", {"u4_arm": "U4S", "u4_replanner_model": SONNET})
    _put(ok, "run.json", {"u4": {"arm": "U4S", "replanner_model": SONNET}})
    _put(ok, "run_0000.json", {"u4": {"arm": "U4S", "replanner_model": SONNET}})
    assert s.foreign_records(ok) == []
    for name, obj in (("resume_spec.json", {"u4_arm": "U4"}), ("resume_spec_20261009-010000.json", {"u4_arm": "U0"}),
                      ("run.json", {"u4": {"arm": "U4"}}), ("run_0001.json", {"u4": {"arm": "U4"}}),
                      ("run_0002.json", {"seed": 191400}), ("run_0003.json", {"u4": {"arm": "U4S", "replanner_model": HAIKU}})):
        d = tmp_path / name.replace(".", "_")
        _put(d, name, obj)
        assert s.foreign_records(d), name
    bad = tmp_path / "broken"
    bad.mkdir()
    (bad / "run_0000.json").write_text("{書きかけ", encoding="utf-8")
    assert "読めない" in s.foreign_records(bad)[0]


@pytest.mark.parametrize("name, obj", [("resume_spec.json", {"u4_arm": "U4", "condition": "U4S"}),
                                       ("run_0000.json", {"u4": {"arm": "U4", "condition": "U4S"}})])
def test_run_refuses_folder_made_by_u4(s, name, obj, monkeypatch, tmp_path):
    calls = []
    fake_r96(s, monkeypatch, tmp_path, calls)
    _put(tmp_path / "S4U4ST" / "U4S", name, obj)
    assert s.cmd_run(BASE + ["--trials", "191400:2", "--dry-run"]) == 3
    assert s.cmd_run(BASE + ["--trials", "191400:2", "--dry-run", "--accept-spec-change"]) == 3      # 控えの変更の許可でも通さない
    assert calls == []


def test_run_resumes_own_folder(s, monkeypatch, tmp_path):
    calls = []
    fake_r96(s, monkeypatch, tmp_path, calls)
    _put(tmp_path / "S4U4ST" / "U4S", "run_0000.json", {"u4": {"arm": "U4S", "replanner_model": SONNET}})
    assert s.cmd_run(BASE + ["--trials", "191400:2", "--dry-run"]) == 0 and len(calls) == 1


# ---------------------------------------------------------------- Engine: 立て直しだけが Sonnet 5.5 になる
class FakeBaseEngine:
    """96 の Engine の代わり。init_task が make_task を作り、run_one_task が偽の実行器を最後まで回す。"""

    doable = {"green", "blue"}

    def __init__(self, a=None):
        self.a = a

    def init_task(self):
        def make(io, setup):
            return TaskRuntime(io, setup, setup.prt, setup.judge, json.loads(json.dumps(T.PLANNER_CFG)), T.MP,
                               lambda text, table, box: {"steps": [c for c in ("red", "green", "blue") if c in table]},
                               "pick up the {color} cube")
        self.make_task = make

    def run_one_task(self, i, seed):
        clock = T.Clock()
        io, prt = T.FakeIO(clock), T.FakePRT(clock)
        setup = types.SimpleNamespace(retreat_pose=np.array([0.4, 0.0, 0.3]), gripper_speed=0.1, truth=T.Forbidden(),
                                      prt=prt, judge=T.FakeJudge(prt.wm, self.doable))
        ex = self.make_task(io, setup)
        ex.start("全部片付けて", seed)
        for _ in range(200000):
            if ex.finished:
                break
            ex.tick()
        assert ex.finished
        return {"run": i, "seed": seed}, {}, {}


def _engine(s, info=None):
    r96 = types.SimpleNamespace(Engine=FakeBaseEngine)
    info = info or {"script": "98_s4_u4s.py", "replanner_variant": RM.variant(SONNET), "files_sha256": {}}
    return s.make_engine(r96, info)()


def test_engine_uses_sonnet_only_for_replanning(s, monkeypatch):
    cli = T.FakeClient(T.raw())
    monkeypatch.setattr(R4, "client", lambda: cli)
    e = _engine(s)
    e.init_task()
    meta, _, _ = e.run_one_task(0, 191400)
    assert len(cli.calls) == 2                                                  # 立て直し 2 回（計画役は偽物で API を通らない）
    kw = cli.calls[0]
    assert kw == RM.build_params(SONNET, kw["messages"][0]["content"])           # replan_multi の Sonnet 5.5 の要求そのもの
    assert kw["model"] == SONNET and kw["thinking"] == {"type": "between_tools"} and "temperature" not in kw
    assert kw["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": R4.SCHEMA}}
    assert kw["system"] == R4.SYSTEM
    assert e.u4_last.replan.model == SONNET and e.u4_last.max_replans == R4.MAX_REPLANS
    u = meta["u4"]
    assert u["arm"] == "U4S" and u["replanner_model"] == SONNET and u["planner_model"] == HAIKU
    assert u["replanner_variant"] == "s4_replan_multi_sonnet55"
    assert u["max_replans"] == R4.MAX_REPLANS and u["max_runs_per_color"] == R4.MAX_RUNS_PER_COLOR
    assert u["interventions"] == {"plan_change": 1, "replan_requests": 2, "llm_calls": 2}
    r1, r2 = meta["replans"]
    assert r1["out"]["replanner_variant"] == "s4_replan_multi_sonnet55" and r1["out"]["kind"] == "replan_multi"
    assert r1["out"]["llm"]["requested_model"] == SONNET and r1["action"] == "reorder"
    assert r2["action"] == "stop" and r2["out"]["fallback"]                     # 弾く規則は replan_s4 と同じ
    json.dumps(meta, ensure_ascii=False)


def test_engine_report_check_is_kept(s, monkeypatch):
    cli = T.FakeClient(T.raw(report="Sonnet が赤を後に回しました。"), T.raw("stop", "none", []))
    monkeypatch.setattr(R4, "client", lambda: cli)
    e = _engine(s)
    e.init_task()
    meta, _, _ = e.run_one_task(0, 191400)
    r1 = meta["replans"][0]
    assert r1["out"]["report_source"] == "template" and "report:forbidden:Sonnet" in r1["out"]["report_rejected"]


def test_engine_replan_api_failure_falls_back_to_stop(s, monkeypatch):
    class _ApiError(Exception):
        pass
    cli = T.FakeClient(_ApiError("overloaded"), _ApiError("overloaded"))
    monkeypatch.setattr(R4, "client", lambda: cli)
    e = _engine(s)
    e.init_task()
    meta, _, _ = e.run_one_task(0, 191400)
    r1 = meta["replans"][0]
    assert len(cli.calls) == 2 and r1["action"] == "stop" and r1["out"]["rejected"][0] == "llm_failed"


# ---------------------------------------------------------------- summary（費用はモデルごと、二重集計）
def _metas(step_of, arm, rm):
    out = []
    for i, resp in enumerate([T.raw(), T.raw("skip", "red", []), T.raw("finish", "none", []), T.raw("stop", "none", [])]):
        cli = T.FakeClient(resp)
        rec = T.run(T.make_rt({"green", "blue"}, step_of(cli))[0])
        m = T.meta_of(rec, i, timed_out=(i == 3), t_end=200.0 if i == 3 else 150.0)
        m["plan"] = dict(m["plan"], from_cache=False, usage={"input_tokens": 800, "output_tokens": 30})
        m["u4"].update(arm=arm, **({"replanner_model": rm} if rm else {}))
        out.append(m)
    return out


def _sonnet_step(cli):
    return lambda **kw: RM.replan_step(**kw, cli=cli, use_cache=False, model=SONNET)


def _haiku_step(cli):
    return lambda **kw: R4.replan_step(**kw, cli=cli, use_cache=False)


def _summary(s, monkeypatch, tmp_path, conds, extra=()):
    for c, metas in conds.items():
        T._write_runs(tmp_path / "outputs" / "v2eval" / "S4U4ST" / c, metas)
    monkeypatch.setattr(s, "ROOT", tmp_path)
    monkeypatch.setattr(s, "OUTD", tmp_path / "outputs" / "s4" / "u4")
    rc = s.main(["summary", "--experiment", "S4U4ST", "--conditions", *conds, *extra])
    return rc, json.loads((tmp_path / "outputs" / "s4" / "u4" / "summary_S4U4ST_u4s.json").read_text(encoding="utf-8"))


def test_summary_prices_planner_at_haiku_and_replanner_at_sonnet(s, monkeypatch, tmp_path):
    rc, summ = _summary(s, monkeypatch, tmp_path, {"U4": _metas(_haiku_step, "U4", None),
                                                   "U4S": _metas(_sonnet_step, "U4S", SONNET)})
    assert rc == 0
    for c in ("U4", "U4S"):
        assert summ["conditions"][c]["cross_check"]["all_match"]
        assert "llm_compute" in summ["conditions"][c]["cross_check"]["items"]
    u4s = summ["conditions"]["U4S"]
    assert u4s["arms"] == {f"U4S/{SONNET}": 4}
    by = u4s["llm"]["by_model"]
    assert set(by) == {HAIKU, SONNET}
    assert by[HAIKU]["decompose_calls"] == 4 and by[HAIKU]["replan_calls"] == 0
    assert by[HAIKU]["price_per_mtok"] == [0.10, 0.50] and by[SONNET]["price_per_mtok"] == [2.0, 10.0]
    n = by[SONNET]["replan_calls"]
    assert n == u4s["llm"]["replan_calls"] and n > 0
    want = 4 * (800 * 0.10 + 30 * 0.50) / 1e6 + n * (900 * 2.0 + 60 * 10.0) / 1e6
    assert abs(u4s["llm"]["cost_usd"] - round(want, 6)) < 1e-9
    assert by[SONNET]["input_tokens"] == 900 * n and by[SONNET]["output_tokens"] == 60 * n
    # 98 の U4 は全部 Haiku の料金（記録の requested_model）
    u4 = summ["conditions"]["U4"]
    assert set(u4["llm"]["by_model"]) == {HAIKU} and u4["arms"] == {f"U4/{HAIKU}": 4}
    m = u4["llm"]["by_model"][HAIKU]
    assert abs(u4["llm"]["cost_usd"] - round((m["input_tokens"] * 0.10 + m["output_tokens"] * 0.50) / 1e6, 6)) < 1e-9
    # 料金の上書き
    rc, summ = _summary(s, monkeypatch, tmp_path / "p", {"U4S": _metas(_sonnet_step, "U4S", SONNET)},
                        ["--price", SONNET, "3", "15"])
    assert rc == 0 and summ["conditions"]["U4S"]["llm"]["by_model"][SONNET]["price_per_mtok"] == [3.0, 15.0]


def test_cost_table_matches_docs(s):
    from recovla.vlm import cost
    assert cost.PRICES[HAIKU] == (0.10, 0.50) and cost.PRICES[SONNET] == (2.00, 10.00)
    assert s.s98.PRICE_PER_MTOK[HAIKU] == cost.PRICES[HAIKU]


def test_summary_stops_on_mixed_arms(s, monkeypatch, tmp_path):
    metas = _metas(_sonnet_step, "U4S", SONNET)
    metas[1]["u4"].update(arm="U4", replanner_model=None)
    rc, summ = _summary(s, monkeypatch, tmp_path, {"U4S": metas})
    assert rc == 1 and len(summ["conditions"]["U4S"]["arms"]) == 2


def test_summary_stops_on_cross_check_mismatch(s, monkeypatch, tmp_path):
    metas = _metas(_sonnet_step, "U4S", SONNET)
    metas[2]["replans"][0]["intervention_kind"] = None                          # 56 の数え方だけが変わる記録
    rc, summ = _summary(s, monkeypatch, tmp_path, {"U4S": metas})
    assert rc == 1 and not summ["conditions"]["U4S"]["cross_check"]["items"]["plan_change"]["match"]
