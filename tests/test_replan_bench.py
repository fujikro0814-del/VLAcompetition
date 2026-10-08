"""立て直しの計画役の包み（src/recovla/planner/replan_multi.py）と、オフラインの試験台（scripts/58_replan_bench.py）。
CPU だけ。ネットワーク・GPU・シミュレーションは使わない（LLM は偽のクライアント、記録は一時フォルダに作った偽物）。

使い方: .venv\\Scripts\\python.exe -m pytest -q tests\\test_replan_bench.py -p no:cacheprovider
読むもの: replan_s4.py・replan_multi.py・executor_u4.py・scripts/58_replan_bench.py（importlib）・src/recovla/vlm/ask.py・
  scripts/check_g1_boundary.py（importlib）。書くもの: pytest の一時フォルダだけ。
"""
import ast
import importlib.util
import itertools
import json
import pathlib
import sys
import types

import pytest

from recovla.planner import replan_multi as M
from recovla.planner import replan_s4 as R4

ROOT = pathlib.Path(__file__).resolve().parents[1]
HAIKU, SONNET, OPUS = "claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def s58():
    return _load(ROOT / "scripts" / "58_replan_bench.py", "replan_bench_t")


@pytest.fixture(autouse=True)
def _tmp_out(tmp_path, monkeypatch, s58):
    monkeypatch.setattr(R4, "CACHE", tmp_path / "llm_cache")
    monkeypatch.setattr(s58, "OUT", tmp_path / "bench")

    def no_client():
        raise AssertionError("本物のクライアントを作ろうとした")
    monkeypatch.setattr(s58, "make_client", no_client)
    monkeypatch.setattr(R4, "client", no_client)


def raw(action="reorder", color="none", order=("green", "blue", "red"), reason="赤を後に回す", report=None):
    report = "赤の立方体を箱に入れられませんでした。先に緑と青を入れて、赤は最後にもう一度試します。" if report is None else report
    return json.dumps({"action": action, "color": color, "order": list(order), "reason": reason, "report": report},
                      ensure_ascii=False)


class FakeClient:
    """messages.create と messages.batches を模擬する。応答（文字列、例外、("refusal", 文字列)）を順に返し、受けた引数を残す。"""

    def __init__(self, *responses, usage=(1000, 80), batch_results=None):
        self.responses = list(responses) or [raw()]
        self.calls, self.usage = [], usage
        self.batch_requests, self.batch_results = [], batch_results
        self.messages = types.SimpleNamespace(create=self._create, batches=types.SimpleNamespace(
            create=self._b_create, retrieve=self._b_retrieve, results=self._b_results))

    def _msg(self, model, r):
        stop, text = ("end_turn", r) if isinstance(r, str) else r
        return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=text)], model=model, stop_reason=stop,
                                     usage=types.SimpleNamespace(input_tokens=self.usage[0], output_tokens=self.usage[1]))

    def _create(self, **kw):
        self.calls.append(kw)
        r = self.responses[min(len(self.calls), len(self.responses)) - 1]
        if isinstance(r, BaseException):
            raise r
        return self._msg(kw["model"], r)

    def _b_create(self, requests):
        self.batch_requests.append(requests)
        return types.SimpleNamespace(id=f"msgbatch_{len(self.batch_requests)}", processing_status="in_progress")

    def _b_retrieve(self, batch_id):
        return types.SimpleNamespace(processing_status="ended")

    def _b_results(self, batch_id):
        reqs = self.batch_requests[int(batch_id.split("_")[1]) - 1]
        for i, q in enumerate(reqs):
            how = self.batch_results(i, q) if self.batch_results else raw()
            if how == "errored":
                res = types.SimpleNamespace(type="errored", error={"type": "overloaded"})
            else:
                res = types.SimpleNamespace(type="succeeded", message=self._msg(q["params"]["model"], how))
            yield types.SimpleNamespace(custom_id=q["custom_id"], result=res)


class NoCallClient:
    def __init__(self):
        self.messages = types.SimpleNamespace(create=self._create)

    def _create(self, **kw):
        raise AssertionError("呼ばないはずの API を呼んだ")


def ctx(failed="red", table=("red", "green", "blue"), box=(), steps=None, plan=("red", "green", "blue"), replans=0,
        in_hand=(), lost=()):
    steps = [{"color": failed, "result": "timeout", "attempts": 2}] if steps is None else steps
    return R4.make_context("全部片付けて", list(plan), steps, {"table": list(table), "in_box": list(box), "in_hand": list(in_hand),
                                                         "lost": list(lost)}, len(steps) - 1, failed, replans)


# ================================================================ replan_multi: 要求の形
RESPONSES = [raw(), raw("next", "green", []), raw("skip", "red", []), raw("finish", "none", []), raw("stop", "none", []),
             raw("next", "blue", [], report="E7 の red は誘発で失敗。"), raw("dance", "none", []), "{壊れた",
             ("refusal", raw())]


@pytest.mark.parametrize("resp", RESPONSES)
def test_haiku_request_and_result_identical_to_replan_s4(resp):
    c = ctx()
    a, b = FakeClient(resp, resp), FakeClient(resp, resp)
    out_s4 = R4.replan(c, cli=a, use_cache=False)
    out_m = M.replan(c, model=HAIKU, cli=b, use_cache=False)
    assert a.calls == b.calls and len(a.calls) >= 1                                  # 要求の中身が全部同じ（回し直しも同じ回数）
    for k in ("decision", "reason", "report", "accepted", "rejected", "fallback", "report_source", "report_rejected",
              "llm_output", "candidates"):
        assert out_s4.get(k) == out_m.get(k), k
    assert out_m["replanner_variant"] == "s4_replan_multi_haiku55" and out_m["kind"] == "replan_multi"


def test_haiku_replan_step_matches_replan_s4_step():
    c = ctx()
    kw = dict(text="全部片付けて", plan_order=c["plan"], steps=c["steps"], perception=c["perception"], failed_step=0,
              failed_color="red", replans_done=0)
    o1 = R4.replan_step(**kw, cli=FakeClient(raw("next", "green", [])), use_cache=False)
    o2 = M.make_replan_step(HAIKU)(**kw, cli=FakeClient(raw("next", "green", [])), use_cache=False)
    for k in ("decision", "next_order", "plan_change", "context", "report", "accepted"):
        assert o1[k] == o2[k], k


def test_per_model_parameters_copy_vlm_ask_and_send_no_temperature():
    from recovla.vlm import ask as A
    msg = R4.user_message(ctx())
    for model in (HAIKU, SONNET, OPUS):
        cli = FakeClient()
        M.replan(ctx(), model=model, cli=cli, use_cache=False)
        kw = cli.calls[0]
        assert kw["model"] == model and "temperature" not in kw and "extra_body" not in kw and "top_p" not in kw
        assert kw["system"] == R4.SYSTEM and kw["messages"] == [{"role": "user", "content": msg}]
        assert kw["output_config"]["format"] == {"type": "json_schema", "schema": R4.SCHEMA}
        assert kw == M.build_params(model, msg)
        if model != HAIKU:                                                            # Sonnet・Opus は vlm/ask.py の設定と同じ
            s = A.MODEL_SETTINGS[model]
            assert kw["max_tokens"] == s["max_tokens"] and kw.get("thinking") == s.get("thinking")
            assert {k: v for k, v in kw["output_config"].items() if k != "format"} == s["output_config"]
    p = M.build_params(SONNET, msg)
    assert p["thinking"] == {"type": "between_tools"} and p["output_config"]["effort"] == "low" and p["max_tokens"] == 2048
    p = M.build_params(OPUS, msg)
    assert "thinking" not in p and p["output_config"]["effort"] == "low" and p["max_tokens"] == 4096
    p = M.build_params(HAIKU, msg)
    assert p["thinking"] == {"type": "disabled"} and set(p["output_config"]) == {"format"}
    with pytest.raises(ValueError):
        M.build_params("claude-haiku-4-5", msg)


def test_cache_keys_differ_by_model_repeat_and_from_replan_s4(tmp_path):
    msg = R4.user_message(ctx())
    keys = {M.cache_key(m, msg, x) for m in (HAIKU, SONNET, OPUS) for x in (None, {"repeat": 0}, {"repeat": 1})}
    assert len(keys) == 9
    assert R4.cache_key(HAIKU, msg) not in keys
    d = tmp_path / "c"
    out1 = M.replan(ctx(), model=SONNET, cli=FakeClient(), cache_dir=d)
    out2 = M.replan(ctx(), model=SONNET, cli=NoCallClient(), cache_dir=d)
    assert out2["llm"]["from_cache"] and out2["decision"] == out1["decision"]
    out3 = M.replan(ctx(), model=OPUS, cli=FakeClient(raw("stop", "none", [])), cache_dir=d)   # 別のモデルはキャッシュを共有しない
    assert not out3["llm"]["from_cache"] and out3["decision"]["action"] == "stop"
    rec = json.loads((d / f"{out1['llm']['cache']}.json").read_text(encoding="utf-8"))
    assert rec["request"]["requested_model"] == SONNET and rec["request"]["temperature"] is None
    assert rec["request"]["settings"]["thinking"] == {"type": "between_tools"}
    assert not (R4.CACHE).exists()                                                    # replan_s4 の置き場には書かない


@pytest.mark.parametrize("bad", [RuntimeError("down"), "{壊れた", ("refusal", raw())])
def test_one_retry_then_stop(bad, tmp_path):
    cli = FakeClient(bad, raw())
    out = M.replan(ctx(), model=OPUS, cli=cli, cache_dir=tmp_path)
    assert len(cli.calls) == 2 and out["accepted"] and out["llm"]["retried"]
    cli = FakeClient(bad, bad, raw())
    out = M.replan(ctx(box=("green",), table=("red", "blue")), model=OPUS, cli=cli, cache_dir=tmp_path)
    assert len(cli.calls) == 2 and out["decision"]["action"] == "stop" and out["fallback"]
    assert out["rejected"][0] == "llm_failed" and not list(tmp_path.glob(f"{out['llm']['cache']}.json"))


def test_replan_limit_and_model_name_in_report():
    out = M.replan(ctx(replans=R4.MAX_REPLANS), model=SONNET, cli=NoCallClient())
    assert out["decision"]["action"] == "stop" and out["llm"] == {"called": False, "calls": 0}
    out = M.replan(ctx(), model=SONNET, cli=FakeClient(raw(report="赤を入れられませんでした。Sonnet が緑を先にします。")),
                   use_cache=False)
    assert out["accepted"] and out["report_source"] == "template" and "report:forbidden:Sonnet" in out["report_rejected"]


def test_thinking_blocks_are_skipped_when_reading_text():
    r = types.SimpleNamespace(content=[types.SimpleNamespace(type="thinking", thinking="..."),
                                       types.SimpleNamespace(type="text", text=raw())],
                              model=OPUS, stop_reason="end_turn", usage={"input_tokens": 5, "output_tokens": 7})
    f = M.message_fields(r)
    assert f["raw"] == raw() and f["thinking_blocks"] == 1 and f["usage"] == {"input_tokens": 5, "output_tokens": 7}


def test_g1_replan_multi_reads_no_truth():
    g = _load(ROOT / "scripts" / "check_g1_boundary.py", "check_g1_boundary_rb")
    src = (ROOT / "src" / "recovla" / "planner" / "replan_multi.py").read_text(encoding="utf-8")
    assert g.scan(src) == {}
    mods = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import):
            mods |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            mods.add(n.module)
    assert not [m for m in mods if m.startswith(("recovla.sim", "recovla.eval", "recovla.expert", "recovla.record",
                                                  "recovla.harness"))]


def test_executor_accepts_multi_step():
    from recovla.runtime import executor_u4 as U4
    from recovla.runtime.executor import TaskRuntime
    tr = TaskRuntime.__new__(TaskRuntime)
    step = M.make_replan_step(SONNET)
    ex = U4.install(tr, step, max_replans=2)
    assert ex.replan is step and step.model == SONNET


# ================================================================ 試験台: 項目（真値を入力に入れない）
def _meta(run=0, stopped=True, truth=None, final=None):
    steps = [{"step": 0, "color": "red", "attempts": [{"attempt": 0, "t_begin": 1.0, "t_judge": 20.0}], "judged_complete": True,
              "t_judge": 20.0, "t_end": 20.0},
             {"step": 1, "color": "green", "attempts": [{"attempt": 0, "t_begin": 20.0, "t_judge": None},
                                                        {"attempt": 1, "t_begin": 52.0, "t_judge": None}],
              "judged_complete": False if stopped else True, "t_judge": None, "t_end": 82.0}]
    m = {"run": run, "seed": 145000 + run, "text": "全部片付けて", "model": "R1v3",
         "detected": {"table": ["blue", "green", "red"], "box": []},
         "plan": {"steps": ["red", "green", "blue"], "reply": "x"}, "steps": steps, "returns": [],
         "stopped": {"step": 1, "t": 82.0, "reply": "2 番目の手順（green）を 2 回試して終えられなかったので、止めました"} if stopped else None,
         "truth_success_t": truth if truth is not None else {"red": 15.0},
         "final_in_box": final or {"red": True, "green": False, "blue": False}, "all_three_in_box": False, "timed_out": False}
    return m


def _runtime(entries):
    return {"runtime": {"perception": [{"t": t, "t_obs": t - 0.1, "cubes": {c: {"pos": [0, 0, 0], "status": st, "in_box": ib,
                                                                                 "source": "overhead"}
                                                                             for c, (st, ib) in cubes.items()}} for t, cubes in entries]}}


def _write_runs(d, metas, runtimes):
    d.mkdir(parents=True, exist_ok=True)
    for i, (m, rt) in enumerate(zip(metas, runtimes)):
        (d / f"run_{i:04d}.json").write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
        if rt is not None:
            (d / f"run_{i:04d}_runtime.json").write_text(json.dumps(rt), encoding="utf-8")


RT = _runtime([(10.0, {"red": ("in_hand", False), "green": ("seen", False), "blue": ("seen", False)}),
               (81.9, {"red": ("seen", True), "green": ("held", False), "blue": ("seen", False)}),
               (82.1, {"red": ("seen", True), "green": ("seen", True), "blue": ("seen", False)})])


def test_input_is_what_replan_s4_would_receive(s58):
    inp, note, why = s58.build_input(s58.executor_view(_meta()), RT)
    assert why == "" and note["source"] == "runtime_log" and note["t_entry"] == 81.9      # 止まった時刻より後の記録は使わない
    assert inp == {"text": "全部片付けて", "plan_order": ["red", "green", "blue"],
                   "steps": [{"color": "red", "result": "success", "attempts": 1},
                             {"color": "green", "result": "timeout", "attempts": 2}],
                   "perception": {"table": ["blue", "green"], "in_box": ["red"], "in_hand": [], "lost": []},
                   "failed_step": 1, "failed_color": "green", "replans_done": 0}
    c = R4.make_context(**inp)
    assert R4.candidates(c) == ["green", "blue"]


def test_step_result_matches_executor_u4(s58):
    from recovla.runtime.executor_u4 import ReplanTaskRuntime
    for rec in ({"judged_complete": True}, {"judged_complete": False}, {"skipped": True}, {}, {"judged_complete": None}):
        assert s58.step_result(rec) == ReplanTaskRuntime.step_result(rec)


def test_no_ground_truth_leaks_into_inputs(s58):
    a = _meta(truth={"red": 15.0}, final={"red": True, "green": False, "blue": False})
    b = _meta(truth={"red": 15.0, "green": 40.0, "blue": 70.0}, final={"red": True, "green": True, "blue": True})
    b["all_three_in_box"] = True
    va, vb = s58.executor_view(a), s58.executor_view(b)
    assert va == vb and not set(va) & set(s58.TRUTH_KEYS)
    assert s58.build_input(va, RT) == s58.build_input(vb, RT)
    assert s58.build_input(va, None)[0] == s58.build_input(vb, None)[0]


def test_items_from_folder_and_truth_kept_apart(s58, tmp_path):
    d = tmp_path / "v2eval" / "V3S3" / "E7_R1v3"
    _write_runs(d, [_meta(0), _meta(1, stopped=False), _meta(2, truth={"red": 15.0, "green": 60.0})], [RT, RT, None])
    doc = s58.build_items([d], "s3")
    assert [x["item_id"].split("/")[-1] for x in doc["items"]] == ["run_0000", "run_0002"]
    assert doc["excluded"][0]["why"] == "no_stop"
    it0, it2 = doc["items"]
    assert it0["perception_note"]["source"] == "runtime_log" and it2["perception_note"]["source"] == "reconstructed_from_record"
    assert it2["input"]["perception"] == {"table": ["blue", "green"], "in_box": ["red"], "in_hand": [], "lost": []}
    assert it2["truth"]["box_at_request"] == ["red", "green"]                       # 真値は truth にだけある
    for it in doc["items"]:
        s = json.dumps(it["input"], ensure_ascii=False)
        assert not any(k in s for k in s58.TRUTH_KEYS + ("box_at_request",))


def test_bundle1_folders_need_flag(s58, tmp_path):
    with pytest.raises(SystemExit):
        s58.resolve_dirs([ROOT / "outputs" / "v2eval" / "S4DE7" / "E0_run1"], False)
    assert s58.resolve_dirs([ROOT / "outputs" / "v2eval" / "S4DE7" / "E0_run1"], True)
    assert s58.main(["items", "--allow-bundle1", "--dirs", str(tmp_path)]) == 2              # s3 は段階 3 だけ


# ================================================================ 試験台: ask（--execute・予算）と score
def _items_file(s58, tmp_path, metas=None, rts=None):
    d = tmp_path / "v2eval" / "V3S3" / "E7_R1v3"
    metas = metas or [_meta(0), _meta(1, truth={"red": 15.0, "green": 60.0})]
    _write_runs(d, metas, rts or [RT] * len(metas))
    doc = s58.build_items([d], "s3")
    s58.OUT.mkdir(parents=True, exist_ok=True)
    s58.items_path("s3").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return doc


def test_ask_without_execute_never_makes_a_client(s58, tmp_path):
    _items_file(s58, tmp_path)
    assert s58.main(["ask", "--models", "haiku,sonnet", "--repeats", "2"]) == 0      # 見込みだけ（make_client は失敗する作り）
    assert s58.main(["ask", "--models", "haiku", "--batch"]) == 0
    assert not (s58.OUT / "cache").exists() and not (s58.OUT / "spend_log.jsonl").exists()


def test_budget_cap_refuses_before_calling(s58, tmp_path):
    _items_file(s58, tmp_path)
    assert s58.main(["ask", "--models", "opus", "--execute", "--budget", "0.0001"]) == 3
    assert not (s58.OUT / "cache").exists()
    assert s58.DEFAULT_BUDGET_USD == 10.0


def test_budget_cap_stops_sync_midway(s58, tmp_path, monkeypatch):
    _items_file(s58, tmp_path, [_meta(0), _meta(1, truth={"red": 15.0, "green": 60.0})],
                [RT, _runtime([(81.0, {"red": ("seen", True), "green": ("seen", False), "blue": ("lost", False)})])])
    cli = FakeClient(usage=(1_000_000, 0))                                           # 1 回で $4（Opus）
    monkeypatch.setattr(s58, "make_client", lambda: cli)
    monkeypatch.setattr(s58, "estimate", lambda qs, calib=None: {"total_usd_sync": 0.0, "total_usd_batch": 0.0})
    assert s58.main(["ask", "--models", "opus", "--repeats", "3", "--execute", "--budget", "5"]) == 3
    assert len(cli.calls) == 2 and s58.spent_usd() == 8.0


def test_ask_sync_then_score_end_to_end(s58, tmp_path, monkeypatch):
    _items_file(s58, tmp_path)
    cli = FakeClient(raw("next", "green", [], report="緑の立方体を箱に入れられませんでした。箱には赤が入っており、もう一度緑を入れます。"))
    monkeypatch.setattr(s58, "make_client", lambda: cli)
    assert s58.main(["score", "--models", "haiku,sonnet", "--repeats", "2"]) == 3         # まだ答えが無い
    assert s58.main(["ask", "--models", "haiku,sonnet", "--repeats", "2", "--execute"]) == 0
    assert len(cli.calls) == 4                                                       # 同じ入力の 2 項目はキャッシュを共有（U4 と同じ）
    assert {c["model"] for c in cli.calls} == {HAIKU, SONNET}
    assert s58.main(["ask", "--models", "haiku,sonnet", "--repeats", "2", "--execute"]) == 0
    assert len(cli.calls) == 4                                                       # 2 回目は呼ばない
    assert s58.main(["score", "--models", "haiku,sonnet", "--repeats", "2", "--write"]) == 0
    res = json.loads((s58.OUT / "score_s3.json").read_text(encoding="utf-8"))
    h = res["per_model"][HAIKU]
    assert h["n"] == 4 and h["valid_action"]["k"] == 4 and h["actions_final"] == {"next": 4}
    assert h["report_correct"]["k"] == 2                                             # run 1 は真値で緑が箱の中（取りこぼし）なので 1 が誤
    assert h["truth_failed_in_box"] == 2
    assert res["agreement"][f"{HAIKU}|{SONNET}"]["decision"]["rate"] == 1.0
    assert res["sonnet_rule"]["decidable"] and res["sonnet_rule"]["add_U4S"] is False


def test_batch_submit_collect_and_retry_once(s58, tmp_path, monkeypatch):
    _items_file(s58, tmp_path, [_meta(0)])
    cli = FakeClient(batch_results=lambda i, q: "errored" if q["params"]["model"] == OPUS else raw("next", "green", []))
    monkeypatch.setattr(s58, "make_client", lambda: cli)
    assert s58.main(["ask", "--models", "haiku,opus", "--repeats", "1", "--batch", "--execute"]) == 0
    req = cli.batch_requests[0]
    assert len(req) == 2 and all("temperature" not in r["params"] for r in req)
    msg = R4.user_message(R4.make_context(**json.loads(s58.items_path("s3").read_text(encoding="utf-8"))["items"][0]["input"]))
    assert next(r for r in req if r["params"]["model"] == HAIKU)["params"] == M.build_params(HAIKU, msg)
    assert s58.main(["ask", "--models", "haiku,opus", "--repeats", "1", "--batch", "--execute"]) == 0
    assert len(cli.batch_requests) == 2 and len(cli.batch_requests[1]) == 1           # 失敗した 1 問だけを 1 回だけ出し直す
    assert s58.main(["ask", "--models", "haiku,opus", "--repeats", "1", "--batch", "--execute"]) == 0
    assert len(cli.batch_requests) == 2                                              # 2 回失敗したら出さない
    assert s58.main(["score", "--models", "haiku,opus", "--repeats", "1", "--write"]) == 0
    res = json.loads((s58.OUT / "score_s3.json").read_text(encoding="utf-8"))
    o = res["per_model"][OPUS]
    assert o["fallback_kind"] == {"llm_failed": 1} and o["actions_final"] == {"stop": 1}
    assert res["per_model"][HAIKU]["valid_action"]["k"] == 1
    assert s58.spent_usd() > 0


# ================================================================ 採点の規則（u4_protocol.md 第 5 節）
def _grade(s58, text, c, action, truth_box, color="none", order=()):
    dec = {"action": action, "color": color, "order": list(order)}
    return s58.grade_report(text, c["failed"]["color"], truth_box, dec, R4.apply(dec, c))


def test_every_template_report_scores_correct_against_its_own_action(s58):
    n = 0
    for failed in ("red", "green", "blue"):
        others = [x for x in ("red", "green", "blue") if x != failed]
        for k in range(len(others) + 1):
            for box in itertools.combinations(others, k):
                table = [x for x in ("red", "green", "blue") if x not in box]
                c = ctx(failed=failed, table=table, box=box)
                for a in R4.ACTIONS:
                    col = failed if a in ("next", "skip") else "none"
                    order = list(R4.candidates(c)) if a == "reorder" else []
                    t = R4.template_report(a, c, col, order)
                    g = _grade(s58, t, c, a, list(box), col, order)
                    assert g["c1"] == "correct" and g["c3"] == "correct" and g["correct"], (a, failed, box, t, g)
                    if a in ("finish", "stop"):
                        assert g["c2"] == "correct", (t, g)
                    n += 1
    assert n == 3 * 4 * 5


@pytest.mark.parametrize("text, action, color, order, truth, expect", [
    # 1 終わらなかった色
    ("緑の立方体を入れられませんでした。ここで止まって知らせます。", "stop", "none", (), [], ("wrong", "none", "correct", False)),
    ("うまくいきませんでした。ここで止まって知らせます。", "stop", "none", (), [], ("none", "none", "correct", False)),
    ("赤の立方体を入れられませんでした。ここで止まって知らせます。", "stop", "none", (), ["red"], ("wrong", "none", "correct", False)),
    # 2 箱の中の色
    ("赤を入れられませんでした。箱にはまだ何も入っておらず、ここで止まります。", "stop", "none", (), ["green"],
     ("correct", "wrong", "correct", False)),
    ("赤を入れられませんでした。緑と青は箱に入っています。止まって知らせます。", "stop", "none", (), ["green", "blue"],
     ("correct", "correct", "correct", True)),
    ("赤を入れられませんでした。箱には緑が入っており、ここで終えます。", "finish", "none", (), ["green", "blue"],
     ("correct", "wrong", "correct", False)),
    ("赤を入れられませんでした。緑は箱に入っていません。止まって知らせます。", "stop", "none", (), [],
     ("correct", "none", "correct", True)),
    # 3 次にすること
    ("赤を入れられませんでした。先に緑を入れます。", "stop", "none", (), [], ("correct", "none", "wrong", False)),
    ("赤を入れられませんでした。ここで止まって知らせます。", "next", "green", (), [], ("correct", "none", "wrong", False)),
    ("赤を入れられませんでした。先に緑を入れます。", "next", "green", (), [], ("correct", "none", "correct", True)),
    ("赤を入れられませんでした。先に青を入れます。", "next", "green", (), [], ("correct", "none", "wrong", False)),
    ("赤を入れられませんでした。赤は後に回して、緑と青を先に入れます。", "reorder", "none", ("green", "blue", "red"), [],
     ("correct", "none", "correct", True)),
    ("赤を入れられませんでした。赤は飛ばして、緑と青を続けます。", "skip", "red", (), [], ("correct", "none", "correct", True)),
    ("赤を入れられませんでした。もう一度赤を入れるか、止まって知らせます。", "stop", "none", (), [],
     ("correct", "none", "ambiguous", False)),
])
def test_scoring_rules_on_synthetic_reports(s58, text, action, color, order, truth, expect):
    g = _grade(s58, text, ctx(), action, truth, color, order)
    assert (g["c1"], g["c2"], g["c3"], g["correct"]) == expect, g


def test_ambiguous_rows_go_to_manual_sheet_and_manual_grades_apply(s58, tmp_path, monkeypatch):
    _items_file(s58, tmp_path, [_meta(0)])
    cli = FakeClient(raw("stop", "none", [], report="緑を入れられませんでした。もう一度緑を入れるか、止まって知らせます。"))
    monkeypatch.setattr(s58, "make_client", lambda: cli)
    assert s58.main(["ask", "--models", "haiku", "--repeats", "1", "--execute"]) == 0
    assert s58.main(["score", "--models", "haiku", "--repeats", "1", "--write"]) == 0
    sheet = json.loads((s58.OUT / "sheet_s3.json").read_text(encoding="utf-8"))
    assert len(sheet) == 1 and "model" not in json.dumps(sheet)                       # モデル名は伏せる
    man = tmp_path / "manual.json"
    man.write_text(json.dumps({sheet[0]["row_id"]: {"c3": "correct"}}), encoding="utf-8")
    assert s58.main(["score", "--models", "haiku", "--repeats", "1", "--manual", str(man), "--write"]) == 0
    res = json.loads((s58.OUT / "score_s3.json").read_text(encoding="utf-8"))
    assert res["per_model"][HAIKU]["report_correct"]["k"] == 1 and res["per_model"][HAIKU]["needs_manual"] == 0


def _pm(k_h, k_s, v_h, v_s, n=80):
    return {HAIKU: {"n": n, "report_correct": {"k": k_h}, "valid_action": {"k": v_h}, "needs_manual": 0},
            SONNET: {"n": n, "report_correct": {"k": k_s}, "valid_action": {"k": v_s}, "needs_manual": 0}}


def test_sonnet_rule_is_exact(s58):
    assert s58.sonnet_rule(_pm(40, 48, 70, 70))["add_U4S"] is True                  # ちょうど +10 ポイント
    assert s58.sonnet_rule(_pm(40, 47, 70, 70))["add_U4S"] is False                 # +8.75 ポイント
    assert s58.sonnet_rule(_pm(40, 60, 70, 69))["add_U4S"] is False                 # 受理の率が下がる
    assert s58.sonnet_rule({HAIKU: _pm(1, 1, 1, 1)[HAIKU]})["decidable"] is False


def test_wilson(s58):
    assert s58.wilson(0, 0) == [None, None]
    lo, hi = s58.wilson(50, 100)
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3
