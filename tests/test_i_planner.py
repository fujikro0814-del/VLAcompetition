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


# --- 待機位置へ戻す動き（0094 の 2） ----------------------------------------------------------------------------

def _integrate(mot, x, dt=0.02, t_max=10.0):
    path = [np.array(x, float)]
    for _ in range(int(t_max / dt)):
        if mot.arrived(path[-1]):
            break
        path.append(path[-1] + mot.velocity(path[-1]) * dt)
    return np.array(path)


def test_return_motion_rises_straight_up_then_moves_level_at_expert_speed():
    from recovla.common import config
    from recovla.planner.executor import ReturnMotion
    c = config.load()
    start = (0.45, -0.10, 0.12)
    mot = ReturnMotion(start, c)
    v0 = mot.velocity(start)
    assert np.allclose(v0[:2], 0.0) and np.isclose(v0[2], c["expert"]["speed_ref"]["z"])    # 真上に、台本の上限の速さで
    p = _integrate(ReturnMotion(start, c), start)
    goal = np.array(c["expert"]["retreat_pose"], float)
    assert np.linalg.norm(p[-1] - goal) <= c["expert"]["phase"]["retreat_tol_m"]
    low = p[p[:, 2] < 0.30 - c["expert"]["move_tol_m"] - 1e-9]
    assert np.allclose(low[:, :2], start[:2], atol=1e-9)                # 上がり切るまで水平には動かない
    step_xy = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1) / 0.02
    assert step_xy.max() <= c["expert"]["speed_ref"]["xy"] + 1e-9
    assert np.all(p[np.argmax(p[:, 2] >= 0.30 - c["expert"]["move_tol_m"]):, 2] >= 0.30 - 0.002)   # 以後は水平


def test_return_motion_from_above_comes_down_to_rise_z_first():
    from recovla.common import config
    from recovla.planner.executor import ReturnMotion
    c = config.load()
    p = _integrate(ReturnMotion((0.50, 0.0, 0.34), c), (0.50, 0.0, 0.34))
    assert np.linalg.norm(p[-1] - np.array(c["expert"]["retreat_pose"])) <= c["expert"]["phase"]["retreat_tol_m"]


def test_return_motion_is_only_in_the_task_executor():
    """0096 の 1 (1): 戻す動きは上位層の実行器（planner/executor.py、呼ぶのは 51_planner.py だけ）にしかなく、
    E1〜E6・E8 の経路（41_results.py → eval/closed_loop・scene_trial、20_k1.py e6）には入っていない。"""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    allowed = {root / "src" / "recovla" / "planner" / "executor.py", root / "scripts" / "51_planner.py",
               root / "scripts" / "50_e_eval.py",       # 50_e_eval.py は E7 の一覧の文に名前が出るだけ（下で import しないことを見る）
               root / "scripts" / "61_demo.py",         # 動画の場面の回し直し（3 個の連続タスクの場面だけが実行器を使う）
               root / "src" / "recovla" / "runtime" / "executor.py"}   # 目標書 v2 の上位層（E7 だけ。E1〜E6 の harness/loop.py は使わない）
    words = ("planner.executor", "TaskExecutor", "ReturnMotion", "return_to_retreat", "_return_to_retreat")
    hits = [str(p.relative_to(root)) for d in ("src", "scripts") for p in (root / d).rglob("*.py")
            if p not in allowed and any(w in p.read_text(encoding="utf-8") for w in words)]
    assert hits == []
    for p in list((root / "src" / "recovla" / "eval").rglob("*.py")) + [root / "scripts" / "41_results.py",
                                                                         root / "scripts" / "20_k1.py",
                                                                         root / "scripts" / "50_e_eval.py"]:
        assert "recovla.planner" not in p.read_text(encoding="utf-8"), p


@pytest.fixture(scope="module")
def _rig():
    from recovla.sim.rig import SimRig
    r = SimRig(render=False)
    yield r
    r.close()


def test_return_to_retreat_opens_rises_and_arrives_without_touching(_rig):
    """物理の中で: 低い所で閉じた手を、開いて真上に上げてから待機位置へ戻す。立方体と箱に触れない。"""
    from recovla.common.seeds import COLORS
    from recovla.expert.script import PhaseParams
    from recovla.planner.executor import TaskExecutor
    from recovla.record import episode as E
    from recovla.sim import scene
    from recovla.sim.rig import quiet
    rig = _rig
    rig.reset(scene.sample_layout(197000, "empty", start="home"))
    pp = PhaseParams.from_config()
    frames_log = []
    state = {"frames": frames_log, "n_frames": lambda: len(frames_log), "done_t": None, "mode": 0}

    def on_step(r):
        if r.step % r.record_every == 0:
            frames_log.append(E.capture_frame(r, COLORS[0], 0.0, pp, False)[0])
    with quiet():
        for _ in range(40):                                   # 下へ 0.8 s（約 7 cm）
            rig.pad_read(np.array([0.0, 0.0, -0.09]), False, on_step)
        rig.pad_read(np.zeros(3), True, on_step)              # 閉じる
        for _ in range(25):
            rig.pad_read(np.zeros(3), False, on_step)
        assert rig.controller.gripper_closed
        x_start = rig.integrator.x_cmd.copy()
        ex = TaskExecutor(rig, runner=None)
        rec = ex._return_to_retreat("retry", on_step, state, judge_wait=False)
    assert rec["arrived"] and rec["opened"] and not rig.controller.gripper_closed
    assert not rec["contact_cube"] and not rec["contact_box"] and max(rec["cube_moved_m"]) < 1e-3
    xd = np.array([f["x_des"] for f in frames_log[-int((rec["t_end"] - rec["t_begin"]) / 0.05):]])
    low = xd[xd[:, 2] < 0.29]
    assert len(low) and np.allclose(low[:, :2], x_start[:2], atol=2e-3)   # 上がり切るまでは真上
    assert np.linalg.norm(rig.integrator.x_cmd - np.array([0.33, 0.15, 0.30])) <= 0.003
    assert state["mode"] == 0 and not rig.safety.gate


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