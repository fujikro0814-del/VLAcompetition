"""Step I の上位層の単体検査（掲示板 0088〜0090）。ネットワークと GPU は使わない（LLM は模擬の応答）。

実 API での正解数（完了条件 2）、照合データでの一致率（完了条件 1）、複数手順の通し（完了条件 3）は
scripts/51_planner.py の llm-eval・judge-data・run で測る。
"""
import json
import types

import numpy as np
import pytest

from recovla.planner import decompose as D
from recovla.planner.judge import CompletionJudge


class _FakeClient:
    """messages.create を模擬する。与えた JSON の文字列を返す。"""

    def __init__(self, raw):
        self.raw = raw
        self.calls = []
        self.messages = types.SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=self.raw)], model=kw["model"],
                                     stop_reason="end_turn", usage=types.SimpleNamespace(input_tokens=10, output_tokens=5))


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "CACHE", tmp_path / "llm_cache")


def test_decompose_accepts_valid_and_sends_json_schema_and_temperature():
    cli = _FakeClient(json.dumps({"steps": ["blue", "red"], "reply": "青と赤を入れます"}))
    out = D.decompose("青と赤", ["red", "green", "blue"], [], cli=cli)
    assert out["valid"] and out["steps"] == ["blue", "red"]
    kw = cli.calls[0]
    assert kw["output_config"]["format"]["type"] == "json_schema"
    assert kw["extra_body"] == {"temperature": 0.0} and kw["model"] == "claude-haiku-4-5"
    assert json.loads(kw["messages"][0]["content"])["table_colors"] == ["red", "green", "blue"]


@pytest.mark.parametrize("steps, table, box, bad", [
    (["red"], ["green", "blue"], ["red"], "no_box_colors"),          # 箱の中にある色
    (["green"], ["red", "blue"], [], "subset_of_table"),             # 机の上にない色
    (["red", "red"], ["red", "green", "blue"], [], "no_duplicates"),  # 重複
])
def test_decompose_falls_back_to_asking(steps, table, box, bad):
    cli = _FakeClient(json.dumps({"steps": steps, "reply": "入れます"}))
    out = D.decompose("x", table, box, cli=cli)
    assert not out["valid"] and out["steps"] == [] and not out["checks"][bad]


def test_decompose_cache_reuses_the_response():
    cli = _FakeClient(json.dumps({"steps": ["red"], "reply": "赤を入れます"}))
    D.decompose("赤", ["red"], [], cli=cli)
    out = D.decompose("赤", ["red"], [], cli=_FakeClient("{}"))
    assert out["from_cache"] and out["steps"] == ["red"] and len(cli.calls) == 1


def test_decompose_broken_json_asks():
    out = D.decompose("赤", ["red"], [], cli=_FakeClient("not json"))
    assert out["steps"] == [] and not out["valid"]


class _Regions:
    """box_pixels と手首の画素を直接返す模擬（画像は使わない）。"""
    box_min, wrist_min, wrist_frac = 56, 33, 0.73
    thr = None

    def __init__(self, box_px):
        self.box_px = box_px

    def counts(self, img):
        return {"box": {"red": self.box_px, "green": 0, "blue": 0}}


def _run(judge, n, closed=False, x=(0.33, 0.15, 0.30), wrist=(500, 500), dt=0.05):
    import recovla.planner.judge as J
    orig = J.wrist_counts
    J.wrist_counts = lambda img, mask, color, thr: wrist
    try:
        out = [judge.update(i * dt, None, closed, x, np.zeros(1), np.ones(1, bool)) for i in range(n)]
    finally:
        J.wrist_counts = orig
    return out


def test_judge_needs_one_second_of_all_conditions():
    j = CompletionJudge(_Regions(100))
    j.reset("red")
    out = _run(j, 21)
    assert not any(out[:20]) and out[20]                 # 20 こま × 0.05 s = 1.0 s で完了


@pytest.mark.parametrize("kw, box_px", [
    ({"closed": True}, 100),                             # 持っている
    ({"x": (0.45, 0.25, 0.20)}, 100),                    # 待機位置にいない
    ({}, 10),                                            # 俯瞰で箱の中に見えない
    ({"wrist": (500, 1000)}, 100),                       # 手首で半分が箱の外（縁に乗っている形）
    ({"wrist": (20, 20)}, 100),                          # 手首でほとんど見えない
])
def test_judge_rejects(kw, box_px):
    j = CompletionJudge(_Regions(box_px))
    j.reset("red")
    assert not any(_run(j, 40, **kw))


def test_step_seed_is_deterministic_and_distinct():
    from recovla.planner.executor import step_seed
    assert step_seed(197000, 0, 0) == step_seed(197000, 0, 0)
    assert len({step_seed(197000, s, a) for s in range(3) for a in range(2)}) == 6


# --- 凍結（0092 の 5）。範囲を変えるときは、変える前に諮る ------------------------------------------------------

FROZEN = {"SYSTEM": "458cca994ccf384e6cce1ae05e263699b99c6006852df72c5875e4f2c633bae9",
          "SCHEMA": "a3026bd9dd85bc43ee93c2d33621c11640fff61e3cb7c7ca0cf24f282dc11484",
          "model": "f72e9feca7d458ff796aa31c70b79c9f93554d1a450f35c6f6c99bd3e24217d3",
          "temperature": "8aed642bf5118b9d3c859bd4be35ecac75b6e873cce34e7b6f554b06f75550d7",
          "check": "acbc93fab76b4976d9006f3197e98a7f3758daf42acc331484481e335b3f7551"}


def test_prompt_is_frozen():
    import hashlib
    import inspect
    from recovla.common import config
    c = config.load()["planner"]
    h = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()      # noqa: E731
    got = {"SYSTEM": h(D.SYSTEM), "SCHEMA": h(json.dumps(D.SCHEMA, sort_keys=True, ensure_ascii=False)),
           "model": h(c["model"]), "temperature": h(repr(float(c["temperature"]))), "check": h(inspect.getsource(D.check))}
    assert got == FROZEN


def test_fallback_reply_comes_from_the_checks_not_the_llm():
    cli = _FakeClient(json.dumps({"steps": ["red", "green"], "reply": "赤と青を入れることはできますが"}))
    out = D.decompose("赤と緑を入れて", ["red", "blue"], ["green"], cli=cli)
    assert out["steps"] == [] and out["reply"] == "緑はすでに箱の中にあります。赤だけ入れますか。"
    assert out["llm_reply"] == "赤と青を入れることはできますが"