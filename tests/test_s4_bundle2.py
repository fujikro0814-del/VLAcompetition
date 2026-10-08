"""段階 4 束 2: 実行器 v3（src/recovla/runtime/executor_v3.py）、入口（scripts/98_s4_b1.py・98_s4_b2.py）、採否の判定
（scripts/98_s4_b_decide.py）の検査。CPU だけ（GPU・方策・カメラの描画・API を使わない。腕の動きは本物の Motion か偽物）。

使い方: PYTHONPATH=src python -m pytest -q tests/test_s4_bundle2.py -p no:cacheprovider
読むもの: executor_v3.py、diag/e7.py（値と動きの照合）、scripts/56_intervention_s3.py・98_s4_b1.py・98_s4_b2.py・98_s4_b_decide.py・
  check_g1_boundary.py（importlib）、configs/s4_gates.json（読むだけ）。書くもの: pytest の一時フォルダだけ。
"""
import importlib.util
import inspect
import json
import os
import pathlib
import sys
import types

import numpy as np
import pytest

from recovla.runtime import executor_v3 as V3
from recovla.runtime.executor import TaskRuntime

ROOT = pathlib.Path(__file__).resolve().parents[1]
DT = 0.002


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- 偽の io・知覚・方策・判定
class Clock:
    t = 0.0


class FakeMotion:
    """指令の手先だけを持つ偽物（(b)〜(d) の検査用）。"""

    def __init__(self):
        self.x_cmd, self.v = np.array([0.45, 0.1, 0.12]), np.zeros(3)

    def set_velocity(self, v):
        self.v = np.asarray(v, float)

    def hand_pose(self, q):
        return self.x_cmd.copy(), None

    def step_fake(self):
        self.x_cmd = self.x_cmd + self.v * DT


class FakeWM:
    def __init__(self, colors=("red", "green", "blue")):
        self.cubes = {c: types.SimpleNamespace(status="seen", in_box=False) for c in colors}


class FakePRT:
    """方策の代わり: 外から止められていない間（external が偽）、手先を半径 5 cm の円で動かし（moving）、試みの始めから
    close_after_s たったら指を閉じる（close_after_s が None なら閉じない）。"""

    def __init__(self, clock, motion=None, moving=True, close_after_s=3.0, box_after=None):
        self.clock, self.motion = clock, motion or FakeMotion()
        self.box_after = dict(box_after or {})                  # 色 → この時刻から知覚が「箱の中」と見る
        self.wm, self.ready, self.stop_reason, self.closed = FakeWM(), True, None, False
        self.external, self.startup = True, {"ok": True}
        self.moving, self.close_after_s, self.t_task = moving, close_after_s, 0.0
        self.q = None

    def start(self, task, seed):
        pass

    def set_task(self, task, seed):
        self.t_task = self.clock.t

    def reset_chunks(self):
        pass

    def tick(self):
        for c, t_in in self.box_after.items():
            if self.clock.t >= t_in:
                self.wm.cubes[c].in_box = True
        if not self.external:
            el = self.clock.t - self.t_task
            if self.moving:
                w = 1.0
                self.motion.set_velocity(0.05 * w * np.array([-np.sin(w * el), np.cos(w * el), 0.0]))
            else:
                self.motion.set_velocity(np.zeros(3))
            if self.close_after_s is not None and el >= self.close_after_s:
                self.closed = True
        if isinstance(self.motion, FakeMotion):
            self.motion.step_fake()
        else:
            from recovla.runtime.types import JointState
            qs = self.motion.step(JointState(self.clock.t, self.q, np.zeros(7), self.q.copy()))
            self.q = np.asarray(qs[-1], float)
        self.clock.t += DT


class Pending:
    def __init__(self, v):
        self.v = v

    def ready(self, t):
        return True

    def result(self, t):
        return self.v


class FakeIO:
    def __init__(self, clock, prt=None):
        self.clock, self.prt, self.n_llm = clock, prt, 0

    def now(self):
        return self.clock.t

    def sense(self, cameras=True):
        q = self.prt.q if (self.prt is not None and self.prt.q is not None) else np.zeros(7)
        return types.SimpleNamespace(cameras={"overhead": None, "wrist": None} if cameras else {},
                                     joints=types.SimpleNamespace(q=np.asarray(q, float)), gripper=types.SimpleNamespace(width=0.08))

    def compute(self, kind, fn):
        self.n_llm += kind == "llm"
        return Pending(fn())

    def gripper_move(self, w, speed):
        if self.prt is not None:
            self.prt.closed = False


class FakeJudge:
    """色ごとの振る舞い: "ok"（u 回目の判定で完了）、"never"（完了しない）、"wait"（ex が (c) の待ちの間だけ完了）。
    placed=True なら、目標の色が箱に見え・待機位置から離れている（(b) の条件）。reclose_every は n 回に 1 回だけ指が閉じて見える。"""

    def __init__(self, wm, how, u=3, placed=False, reclose_every=0):
        self.p = {"gripper_open_m": 0.07, "retreat_tol_m": 0.05, "box_min_pixels": 56}
        self.wm, self.how, self.u, self.placed, self.reclose_every = wm, dict(how), u, placed, reclose_every
        self.color, self.k, self.n, self.ex, self.last = None, 0, 0, None, {}

    def reset(self, color):
        self.color, self.k = color, 0

    def update(self, sensor, wm, hand):
        self.k += 1
        self.n += 1
        how = self.how.get(self.color, "never")
        done = (how == "ok" and self.k >= self.u) or (how == "wait" and self.ex is not None and self.ex._final_wait)
        if done:
            self.wm.cubes[self.color].in_box = True
        opened = not (self.reclose_every and self.n % self.reclose_every == 0)
        self.last = {"ok": done, "held_s": 0.0, "box_pixels": 100 if self.placed else 0, "wrist_in_pixels": 0, "depth_ok": True,
                     "retreat_dist_m": 0.2 if self.placed else 0.0, "gripper_open": opened}
        return done


PLANNER_CFG = {"step_timeout_s": 30.0, "retry": 1,
               "return_to_retreat": {"enabled": True, "rise_z": 0.30, "trigger_s": 4.0, "wait_s": 2.0, "timeout_s": 10.0}}
MP = {"gain": 4.0, "xy_max": 0.3, "z_max": 0.3, "z_tol": 0.005, "tol": 0.01}


def decompose(text, table, box):
    return {"steps": [c for c in ("red", "green", "blue") if c in table], "reply": "赤・緑・青の順に入れます"}


def make(how, v3=None, judge_kw=None, prt_kw=None, cfg=None, motion=None, q0=None):
    clock = Clock()
    prt = FakePRT(clock, motion=motion, **(prt_kw or {}))
    if q0 is not None:
        prt.q = np.asarray(q0, float)
    io = FakeIO(clock, prt)
    judge = FakeJudge(prt.wm, how, **(judge_kw or {}))
    setup = types.SimpleNamespace(retreat_pose=np.array([0.4, 0.0, 0.3]), gripper_speed=0.1)
    ex = V3.make_v3(io, setup, prt, judge, json.loads(json.dumps(cfg or PLANNER_CFG)), MP, decompose, "pick up the {color} cube",
                    v3)
    judge.ex = ex
    return ex, io, prt, judge


def run(ex, limit_s=260.0):
    ex.start("全部片付けて", 191100)
    n = int(limit_s / DT)
    for _ in range(n):
        if ex.finished:
            break
        ex.tick()
    assert ex.finished
    return ex.record()


OFF = {"b_placed_fix": False, "c_final_wait": False, "d_early_abort": False}


# ---------------------------------------------------------------- 全部を切ると今の実行器と同じ
def _base(how, judge_kw=None, prt_kw=None):
    clock = Clock()
    prt = FakePRT(clock, **(prt_kw or {}))
    io = FakeIO(clock, prt)
    judge = FakeJudge(prt.wm, how, **(judge_kw or {}))
    setup = types.SimpleNamespace(retreat_pose=np.array([0.4, 0.0, 0.3]), gripper_speed=0.1)
    tr = TaskRuntime(io, setup, prt, judge, json.loads(json.dumps(PLANNER_CFG)), MP, decompose, "pick up the {color} cube")
    judge.ex = tr
    return tr


def _strip(rec):
    keep = ("steps", "returns", "stopped")
    out = json.loads(json.dumps({k: rec[k] for k in keep}, default=float))
    for r in out["returns"]:
        r.pop("v3", None)
    for s in out["steps"]:
        for a in s["attempts"]:
            a.pop("end", None)
    return out


@pytest.mark.parametrize("how", [{"red": "ok", "green": "ok", "blue": "ok"}, {"red": "ok", "green": "never", "blue": "ok"}])
def test_all_off_matches_current_executor(how):
    tr = _base(how)
    tr.start("全部片付けて", 191100)
    for _ in range(int(260 / DT)):
        if tr.finished:
            break
        tr.tick()
    ex, *_ = make(how, OFF)
    rec = run(ex)
    assert _strip(rec) == _strip(tr.record())
    assert rec["v3"]["retreat_settings"]["trigger_s"] == 4.0                      # (b) を切れば configs の値のまま


# ---------------------------------------------------------------- (b) 置いた後の強制の戻し
def test_b_reclose_does_not_reset_and_trigger_is_shorter():
    how = {"red": "never", "green": "never", "blue": "never"}
    jk = {"placed": True, "reclose_every": 4}                                        # 判定 4 回に 1 回、指が閉じて見える（閉じ直し）
    ex, *_ = make(how, {"c_final_wait": False, "d_early_abort": False}, judge_kw=jk)
    rec = run(ex)
    placed = [r for r in rec["returns"] if r["kind"] == "placed" and r.get("v3") == "placed"]
    assert placed and rec["v3"]["interventions"]["placed_kept_on_reclose"] > 0
    att0 = rec["steps"][0]["attempts"][0]["t_begin"]
    assert placed[0]["t_begin"] - att0 <= 2.0 + 0.5                                 # 2.0 s で発動（判定は 20 Hz）
    assert rec["v3"]["retreat_settings"]["trigger_s"] == 2.0 and rec["v3"]["retreat_settings"]["wait_s"] == 2.0
    # 直す前（今の実行器の数え方）: 閉じ直しで数え直しになり、4 s が続かないので発動しない
    ex0, *_ = make(how, OFF, judge_kw=jk)
    rec0 = run(ex0)
    assert not [r for r in rec0["returns"] if r["kind"] == "placed"]


# ---------------------------------------------------------------- (c) 最後の試みで知覚が箱の中と見ていれば判定を待つ
OPEN = {"close_after_s": None}                                                     # 方策が指を閉じない（(c) は指が開のときだけ）


def test_c_final_wait_when_perception_sees_target_in_box():
    how = {"red": "wait", "green": "ok", "blue": "ok"}
    ex, *_ = make(how, {"d_early_abort": False}, prt_kw=dict(OPEN, box_after={"red": 40.0}))   # 40 s から知覚は「赤は箱の中」
    rec = run(ex)
    ev = [e for e in rec["v3"]["events"] if e["kind"] == "final_wait"]
    assert len(ev) == 1 and ev[0]["judged"] is True and ev[0]["attempt"] == 1
    assert rec["steps"][0]["judged_complete"] and rec["stopped"] is None and len(rec["steps"]) == 3
    fw = [r for r in rec["returns"] if r.get("v3") == "final_wait"]
    assert len(fw) == 1 and fw[0]["kind"] == "placed" and fw[0]["judged"]          # 56 は台本の動きとして数える
    assert rec["v3"]["version"] == V3.VERSION == "v3.1"


def test_c_no_wait_when_perception_does_not_see_target_in_box():
    how = {"red": "wait", "green": "ok", "blue": "ok"}
    ex, *_ = make(how, {"d_early_abort": False}, prt_kw=OPEN)
    rec = run(ex)
    assert not [e for e in rec["v3"]["events"] if e["kind"] in ("final_wait", "final_wait_skipped")]
    assert rec["stopped"] == {"step": 0, "t": rec["stopped"]["t"],
                              "reply": "1 番目の手順（red）を 2 回試して終えられなかったので、止めました"}


def test_c_wait_without_judgement_stops_like_current_executor():
    how = {"red": "never", "green": "ok", "blue": "ok"}
    ex, *_ = make(how, {"d_early_abort": False}, prt_kw=dict(OPEN, box_after={"red": 40.0}))
    rec = run(ex)
    ev = [e for e in rec["v3"]["events"] if e["kind"] == "final_wait"]
    assert len(ev) == 1 and ev[0]["judged"] is False and rec["stopped"]["step"] == 0
    assert not rec["steps"][0]["judged_complete"]


@pytest.mark.parametrize("prt_kw, judge_kw", [({}, {}),                                    # 方策が指を閉じている（持ったまま）
                                              (OPEN, {"reclose_every": 1})])                # 指令は開でも、測った開き幅が開でない
def test_c_not_applied_when_gripper_not_open(prt_kw, judge_kw):
    """知覚の in_box は水平の位置だけなので、持ったまま箱の上にあっても「箱の中」と見る。指が開でなければ (c) で戻さない。"""
    how = {"red": "wait", "green": "ok", "blue": "ok"}
    ex, *_ = make(how, {"d_early_abort": False}, prt_kw=dict(prt_kw, box_after={"red": 40.0}), judge_kw=judge_kw)
    rec = run(ex)
    assert not [e for e in rec["v3"]["events"] if e["kind"] == "final_wait"]
    sk = [e for e in rec["v3"]["events"] if e["kind"] == "final_wait_skipped"]
    assert len(sk) == 1 and sk[0]["why"] == "gripper_closed" and rec["v3"]["interventions"]["final_wait_skipped"]["gripper_closed"] == 1
    assert not [r for r in rec["returns"] if r.get("v3") == "final_wait"]
    assert rec["stopped"]["step"] == 0 and not rec["steps"][0]["judged_complete"]


def test_c_not_repeated_after_placed_return():
    """(b) の戻し（待機位置で wait_s 待つ）が済んで指を閉じていなければ、(c) は同じ待ちを繰り返さない（台本の戻しを 2 回数えない）。"""
    how = {"red": "never", "green": "never", "blue": "never"}
    ex, *_ = make(how, {"d_early_abort": False}, prt_kw=dict(OPEN, box_after={"red": 0.0}), judge_kw={"placed": True})
    rec = run(ex)
    last = [r for r in rec["returns"] if r["step"] == 0 and r["attempt"] == 1]
    assert last and last[-1].get("v3") == "placed"                                  # 最後の試みで (b) の戻しが済んでいる
    assert not [r for r in rec["returns"] if r.get("v3") == "final_wait"]
    sk = [e for e in rec["v3"]["events"] if e["kind"] == "final_wait_skipped"]
    assert len(sk) == 1 and sk[0]["why"] == "after_placed_return" and sk[0]["attempt"] == 1
    n_placed = sum(1 for r in rec["returns"] if r["kind"] == "placed")
    assert rec["v3"]["interventions"]["scripted_return"] == n_placed == rec["v3"]["interventions"]["by_v3_kind"]["placed"]


# ---------------------------------------------------------------- (d) 早めの打ち切りは止まった・閉じないだけ
@pytest.mark.parametrize("prt_kw, why, t_end", [({"moving": False}, "stalled", 8.0), ({"close_after_s": None}, "no_close", 20.0)])
def test_d_early_abort_only_for_stall_and_no_close(prt_kw, why, t_end):
    how = {"red": "never", "green": "ok", "blue": "ok"}
    ex, *_ = make(how, {"c_final_wait": False}, prt_kw=prt_kw)
    rec = run(ex)
    a0 = rec["steps"][0]["attempts"]
    assert [a["end"] for a in a0] == [why, why]
    assert abs((a0[1]["t_begin"] - a0[0]["t_begin"]) - (t_end + rec["returns"][0]["t_end"] - rec["returns"][0]["t_begin"])) < 0.3
    assert rec["v3"]["interventions"]["early_abort"][why] == 2
    # 動いていて閉じている（どちらでもない）なら、打ち切らずに 30 s の時間切れ
    ex2, *_ = make(how, {"c_final_wait": False})
    rec2 = run(ex2)
    assert [a["end"] for a in rec2["steps"][0]["attempts"]] == ["timeout", "timeout"]
    # (d) を切れば止まっていても 30 s
    ex3, *_ = make(how, {"c_final_wait": False, "d_early_abort": False}, prt_kw=prt_kw)
    assert [a["end"] for a in run(ex3)["steps"][0]["attempts"]] == ["timeout", "timeout"]


# ---------------------------------------------------------------- (a) 関節空間の戻す動き（本物の Motion）
def _motion():
    from recovla.common import config
    from recovla.harness.setup import nominal_setup
    from recovla.runtime.motion import Motion
    return Motion(nominal_setup(config.load("sensor_v1")))


def _joints(q):
    from recovla.runtime.types import JointState
    q = np.asarray(q, float)
    return JointState(0.0, q, np.zeros(7), q.copy())


def test_goal_values_and_motion_match_diag_e7():
    from recovla.diag import e7 as E7
    assert V3.HOME_Q == E7.HOME_Q and V3.STANDBY_START_Q == E7.STANDBY_START_Q and V3.GOAL_PARAMS == E7.GOAL_PARAMS
    assert V3.GOALS["EH"] == E7.ARMS["EH"]["goal_q"] and V3.GOALS["ES"] == E7.ARMS["ES"]["goal_q"]
    outs = []
    for cls in (E7.GoalMotion, V3.JointGoalMotion):
        mo = _motion()
        mo.reset(_joints(E7.STANDBY_END_Q))
        mo.__class__ = cls
        mo.set_joint_goal(np.asarray(E7.HOME_Q), E7.GOAL_PARAMS["vmax_rad_s"], E7.GOAL_PARAMS["gain_per_s"])
        q, qs = np.asarray(E7.STANDBY_END_Q, float), []
        for _ in range(1500):
            out = mo.step(_joints(q))
            q = out[-1]
            qs += out
        e1 = mo.joint_goal_error()
        mo.clear_joint_goal()
        for _ in range(100):
            out = mo.step(_joints(q))
            q = out[-1]
            qs += out
        outs.append((np.array(qs), e1, mo.x_cmd.copy()))
    assert np.array_equal(outs[0][0], outs[1][0]) and outs[0][1] == outs[1][1] and np.array_equal(outs[0][2], outs[1][2])


@pytest.mark.parametrize("goal", ["EH", "ES"])
def test_a_goal_before_next_step_and_before_retry(goal):
    from recovla.diag import e7 as E7
    how = {"red": "ok", "green": "never", "blue": "ok"}
    mo = _motion()
    mo.reset(_joints(E7.HOME_Q))
    short = dict(PLANNER_CFG, step_timeout_s=6.0)                                     # 軽くするため持ち時間だけ縮める
    ex, _, prt, _ = make(how, {"a_goal": goal, "c_final_wait": False, "d_early_abort": False}, motion=mo, q0=E7.HOME_Q,
                         judge_kw={"u": 30}, cfg=short)
    assert isinstance(prt.motion, V3.JointGoalMotion)
    rec = run(ex)
    tags = [(r["kind"], r.get("v3"), r["step"], r["attempt"]) for r in rec["returns"]]
    assert ("placed", "goal", 1, 0) in tags                                          # 2 番目の手順の始めの前に戻す
    assert ("retry", "retry_goal", 1, 0) in tags                                     # やり直しの前（待機位置へ戻す代わり）
    assert not [r for r in rec["returns"] if r.get("v3") == "retry"]
    g = [r for r in rec["returns"] if r.get("v3") in ("goal", "retry_goal")]
    assert all(r["arrived"] and r["err_meas_rad"] <= 0.01 for r in g), [(r["arrived"], r["err_meas_rad"]) for r in g]
    assert not [r for r in rec["returns"] if r["step"] == 0]                         # 1 番目の手順の前には戻さない（既定）
    # 戻し先「なし」なら (a) は働かず、やり直しの前は今の実行器と同じ待機位置へ戻す動き
    mo2 = _motion()
    mo2.reset(_joints(E7.HOME_Q))
    ex2, _, prt2, _ = make(how, {"c_final_wait": False, "d_early_abort": False}, motion=mo2, q0=E7.HOME_Q, judge_kw={"u": 30},
                           cfg=short)
    rec2 = run(ex2)
    assert not isinstance(prt2.motion, V3.JointGoalMotion)
    assert [r.get("v3") for r in rec2["returns"]] == ["retry"]


def test_settings_validation_and_e_thresholds_untouched():
    with pytest.raises(ValueError):
        V3.settings({"a_goal": "EX"})
    with pytest.raises(ValueError):
        V3.settings({"judge_threshold": 1})
    ex, _, _, judge = make({"red": "ok", "green": "ok", "blue": "ok"})
    p0 = dict(judge.p)
    rec = run(ex)
    assert judge.p == p0 and rec["plan"]["steps"] == ["red", "green", "blue"]       # (e) 判定の閾値と色の順は変えない


# ---------------------------------------------------------------- (f) 56 で数えられる
def meta_of(rec, run_i=0, truth=None, in_box=None, t_end=150.0):
    in_box = in_box or {"red": True, "green": True, "blue": True}
    return {"run": run_i, "seed": 191100 + run_i, "plan": rec["plan"], "steps": rec["steps"], "returns": rec["returns"],
            "stopped": rec["stopped"], "truth_success_t": truth or {"red": 10.0, "green": 50.0, "blue": 90.0},
            "final_in_box": in_box, "all_three_in_box": all(in_box.values()), "t_end": t_end, "timed_out": False,
            "v3": rec["v3"]}


def _recs():
    out = []
    ex, *_ = make({"red": "never", "green": "never", "blue": "never"}, {"c_final_wait": False, "d_early_abort": False},
                  judge_kw={"placed": True, "reclose_every": 4})
    out.append(run(ex))
    ex, *_ = make({"red": "wait", "green": "ok", "blue": "ok"}, {"d_early_abort": False}, prt_kw=dict(OPEN, box_after={"red": 40.0}))
    out.append(run(ex))
    ex, *_ = make({"red": "never", "green": "ok", "blue": "ok"}, {"c_final_wait": False}, prt_kw={"close_after_s": None})
    out.append(run(ex))
    ex, *_ = make({"red": "ok", "green": "ok", "blue": "ok"})
    out.append(run(ex))
    return out


@pytest.fixture(scope="module")
def m56():
    return _load(ROOT / "scripts" / "56_intervention_s3.py", "intervention_s3_b2")


def test_f_interventions_countable_by_56(m56):
    for i, rec in enumerate(_recs()):
        assert {r["kind"] for r in rec["returns"]} <= set(V3.RETURN_KINDS_56)
        r = m56.count_run(meta_of(rec, i))                                         # 知らない種類なら ValueError
        mine = rec["v3"]["interventions"]
        for k in ("scripted_return", "retry", "replan", "judge_override"):
            assert r["interventions"][k] == mine[k], (i, k, r["interventions"], mine)
    rows = [m56.count_run(meta_of(rec, i)) for i, rec in enumerate(_recs())]
    sk = m56.success_at_k(rows)
    assert sk["0"]["n"] == 4 and m56.aggregate(rows)["llm"]["calls_per_run"] == 1.0


# ---------------------------------------------------------------- G1
def test_g1_executor_v3_strict_scan_and_reachability():
    g = _load(ROOT / "scripts" / "check_g1_boundary.py", "check_g1_boundary_b2")
    assert "src/recovla/runtime/executor_v3.py" in g.runtime_v2_files(None)
    src = (ROOT / "src" / "recovla" / "runtime" / "executor_v3.py").read_text(encoding="utf-8")
    assert g.scan_runtime(src, "src/recovla/runtime/executor_v3.py") == {}
    from recovla.harness.audit import reachable_forbidden
    world = types.SimpleNamespace(data=object())
    ex, *_ = make({"red": "ok", "green": "ok", "blue": "ok"}, {"a_goal": None})
    ex.start("全部片付けて", 1)
    assert reachable_forbidden(ex, {id(world), id(world.data)}, (), ())["violations"] == []
    ex.planted = world
    assert reachable_forbidden(ex, {id(world), id(world.data)}, (), ())["violations"]


# ================================================================ 入口（98_s4_b1.py・98_s4_b2.py）
@pytest.fixture(scope="module")
def b1():
    return _load(ROOT / "scripts" / "98_s4_b1.py", "s4_b1_t")


@pytest.fixture(scope="module")
def b2():
    return _load(ROOT / "scripts" / "98_s4_b2.py", "s4_b2_t")


def _gate1(tmp_path, v3a=None, rtc=None, name="gate1.json"):
    p = tmp_path / name
    p.write_text(json.dumps({"conclusion": {"status": "determined", "bundle2": {
        "v3a": v3a or {"result": "pass", "include": True, "return_to": "ES"},
        "rtc": rtc or {"result": "pass", "b2": True, "settings": ["range10_cap5"]}}}}), encoding="utf-8")
    return p


def test_b1_band_check(b1):
    assert b1.check_band("R1v3_v3", "191100:60", "S4B1", False) == ""
    assert b1.check_band("R1v3_cur", "191100:60", "S4B1", False) == ""
    assert b1.check_band("N1v3_v3", "191100:30", "S4B1", False) == ""
    assert b1.check_band("N1v3_v3", "191100:60", "S4B1", False)                    # N1v3 は 30 本の帯だけ
    assert b1.check_band("R1v3_v3", "191100:59", "S4B1", False)                    # 割り当てと完全に一致しない
    assert b1.check_band("R1v3_v3", "191200:60", "S4B1", False)
    assert b1.check_band("R1v3_v3", "44400:3", "S4B1", False)                      # smoke は --allow-smoke のときだけ
    assert b1.check_band("R1v3_v3", "44400:3", "S4SMOKE_B1", True) == ""
    assert b1.check_band("R1v3_v3", "44400:3", "S4B1", True)                       # smoke は S4SMOKE の実験名だけ
    assert "X2" in b1.check_band("R1v3_v3", "44402:4", "S4SMOKE_B1", True)         # X2 の帯 44404〜44423 は smoke でも拒む
    assert b1.check_band("R1v3_v3", "44424:2", "S4SMOKE_B1", True) == ""


def test_b2_band_check(b2):
    assert b2.default_trials("ext") == "natural:191200:23" and b2.default_trials("P1") == "induced:191300:50"
    assert b2.check_band("ext", "natural:191200:23", "S4B2", False) == ""
    assert b2.check_band("P1", "induced:191300:50", "S4B2", False) == ""
    assert b2.check_band("ext", "natural:191200:22", "S4B2", False)
    assert b2.check_band("P1", "natural:191300:50", "S4B2", False)                 # 種類も一致させる
    assert b2.check_band("ext", "natural:44430:1", "S4SMOKE_B2", True) == ""
    assert "X2" in b2.check_band("ext", "natural:44410:1", "S4SMOKE_B2", True)
    assert b2.check_band("ext", "natural:44430:1", "S4B2", True)


def test_b1_v3_from_gate1_and_override(b1, tmp_path):
    v3, src = b1.v3_from_gate1(_gate1(tmp_path))
    assert v3 == {"a_goal": "ES"} and src["override"] is None and src["gate1"]["v3a"]["return_to"] == "ES"
    v3, _ = b1.v3_from_gate1(_gate1(tmp_path, v3a={"result": "fail", "include": False, "return_to": None}))
    assert v3 == {"a_goal": None}                                                    # 戻し先「なし」: (a) は働かない
    with pytest.raises(SystemExit):
        b1.v3_from_gate1(_gate1(tmp_path, v3a={"result": "undetermined", "include": None}))
    v3, src = b1.v3_from_gate1(_gate1(tmp_path, v3a={"result": "undetermined"}), override="EH", off="b,d")
    assert v3 == {"a_goal": "EH", "b_placed_fix": False, "d_early_abort": False}
    assert src["override"]["v3_goal"] == "EH" and src["v3_off"] == ["b", "d"]
    V3.settings(v3)                                                                  # 実行器が受け付ける形
    with pytest.raises(SystemExit):
        b1.v3_from_gate1(None)


def test_b2_settings_from_gate1(b2, tmp_path):
    st, src, run = b2.settings_from_gate1(_gate1(tmp_path, rtc={"result": "pass", "b2": True, "settings": ["range10_cap5", "ZEROS"]}))
    assert st == ["naive", "range10_cap5", "ZEROS"] and run
    st, src, run = b2.settings_from_gate1(_gate1(tmp_path, rtc={"result": "fail", "b2": False, "settings": []}))
    assert st == [] and not run                                                      # R のやめる枝: B2 は回さない
    with pytest.raises(SystemExit):
        b2.settings_from_gate1(_gate1(tmp_path, rtc={"result": "undetermined"}))
    with pytest.raises(SystemExit):
        b2.settings_from_gate1(None, "current_repro")                               # 候補でない設定
    st, src, run = b2.settings_from_gate1(None, "range40_cap5")
    assert st == ["naive", "range40_cap5"] and src["override"]["settings"] == ["range40_cap5"]


def test_b1_b2_dry_run_writes_nothing(b1, b2, tmp_path, capsys):
    exp1, exp2 = "S4B1TESTDRY", "S4B2TESTDRY"
    r96 = _load(ROOT / "scripts" / "96_s4_resume.py", "s4_resume_b1_t")
    ops, v82 = r96.load_ops(), r96.load_82(False)
    v82.CKPT.update({"R1v3": "outputs/train/x", "N1v3": "outputs/train/y"})
    a = b1.build_parser().parse_args(["run", "--arm", "R1v3_v3", "--gate1", str(_gate1(tmp_path)), "--experiment", exp1])
    rc = b1.run_arm(r96, ops, v82, a, "R1v3_v3", ["--dry-run"])
    out = capsys.readouterr().out
    assert rc in (0, 3) and "全 60 本" in out and "[b1] R1v3_v3" in out              # 3 は GPU のドライバが読めない環境
    m96 = b2.drtc().load96()
    v82b = m96.load_82(False)
    v82b.CKPT.update({"R1v3": "outputs/train/x"})
    a = b2.build_parser().parse_args(["run", "--setting", "naive", "--part", "P1", "--gate1", str(_gate1(tmp_path)),
                                      "--experiment", exp2])
    st, src, _ = b2.settings_from_gate1(a.gate1)
    rc = b2.run_condition(m96, m96.load_ops(), v82b, a, "naive", "P1", src, ["--dry-run"])
    out = capsys.readouterr().out
    assert rc in (0, 3) and "全 50 本" in out
    assert not (ROOT / "outputs" / "v2eval" / exp1).exists() and not (ROOT / "outputs" / "v2eval" / exp2).exists()


def test_b1_spec_pins_executor_only_for_v3_and_records_version(b1, tmp_path, monkeypatch):
    r96 = _load(ROOT / "scripts" / "96_s4_resume.py", "s4_resume_b1_spec_t")
    ops, v82 = r96.load_ops(), r96.load_82(False)
    seen = {}

    def fake_main(a96, v82_, ops_):                                                  # 回さずに、96 に渡る引数の控えだけを見る
        seen[a96.b1_arm] = r96.make_spec(a96)
        return 0
    monkeypatch.setattr(r96, "cmd_main", fake_main)
    for arm in ("R1v3_cur", "R1v3_v3", "N1v3_v3"):
        a = b1.build_parser().parse_args(["run", "--arm", arm, "--gate1", str(_gate1(tmp_path)), "--experiment", "S4B1TESTSPEC"])
        assert b1.run_arm(r96, ops, v82, a, arm, []) == 0
    assert seen["R1v3_cur"]["b1_executor_sha256"] is None and seen["R1v3_cur"]["b1_executor_version"] == "current"
    for arm in ("R1v3_v3", "N1v3_v3"):
        assert len(seen[arm]["b1_executor_sha256"]) == 64 and seen[arm]["b1_executor_version"] == V3.VERSION
    assert not (ROOT / "outputs" / "v2eval" / "S4B1TESTSPEC").exists()


def test_b1_engine_resets_executor_per_trial(b1):
    class FakeEngine:
        def __init__(self, a, v82, cfg, lim, env=None):
            self.a = a

        def init_task(self):
            self.make_task = lambda io, setup: _base({"red": "ok", "green": "ok", "blue": "ok"})

        def run_one_task(self, i, seed):
            if seed % 2 == 0:                                                        # 偶数の種だけ実行器を作る
                self.make_task(None, None).start("全部片付けて", seed)
            return {"run": i}, {}, {}
    r96 = types.SimpleNamespace(Engine=FakeEngine, SPEC_KEYS=("cmd",), run_json_text=lambda *x, **k: "{}")
    b1.patch96(r96, "R1v3_v3", {"a_goal": None}, {"arm": "R1v3_v3", "executor_version": V3.VERSION})
    eng = r96.Engine(types.SimpleNamespace(model="R1v3"), None, None, None)
    eng.init_task()
    meta, _, _ = eng.run_one_task(0, 191100)
    assert meta["v3"]["version"] == V3.VERSION and meta["b1"]["api_retry"] is None
    with pytest.raises(RuntimeError):
        eng.run_one_task(1, 191101)                                                  # この試行の実行器がない: 前の試行の記録を写さない


# ---------------------------------------------------------------- rotate（続けるのは --max-new で止まったときだけ）
def _writer(p, text):
    pathlib.Path(p).write_text(text, encoding="utf-8")
    return True


def _rot(b1, tmp_path, script, totals, stop=lambda n: ""):
    """script: 名前 → 呼び出しごとの (終了コード, 子の stop_reason, 増える完全な本数) の列。"""
    done = {n: 0 for n in totals}
    calls = []

    def count(n):
        return done[n]

    def call(n):
        c, reason, inc = script[n].pop(0)
        done[n] = min(totals[n], done[n] + inc)
        calls.append(n)
        return c, {"status": {0: "done", 1: "stopped"}.get(c, "error"), "stop_reason": reason}
    prog = b1.RotateProgress(tmp_path / "rot.progress.json", {"experiment": "X", "condition": "rotate:X", "total": sum(totals.values())},
                             _writer)
    try:
        rc = b1.rotate_loop(list(totals), totals, count, call, prog, tmp_path / "rot.log.json", stop)
    finally:
        pr = json.loads((tmp_path / "rot.progress.json").read_text(encoding="utf-8"))
    return rc, calls, pr


def test_rotate_continues_only_on_max_new(b1, tmp_path):
    script = {"A": [(1, "max_new:2", 2), (0, None, 2)], "B": [(1, "max_new:2", 2), (0, None, 2)]}
    rc, calls, pr = _rot(b1, tmp_path, script, {"A": 4, "B": 4})
    assert rc == 0 and calls == ["A", "B", "A", "B"]
    assert pr["status"] == "done" and pr["done"] == 8 and pr["pid"] == os.getpid() and len(pr["calls"]) == 4
    assert json.loads((tmp_path / "rot.log.json").read_text(encoding="utf-8"))[0]["stop_reason"] == "max_new:2"


@pytest.mark.parametrize("reason", ["KeyboardInterrupt", "memory_timeout", "stop_file:C:\\x\\STOP"])
def test_rotate_stops_on_other_rc1(b1, tmp_path, reason):
    rc, calls, pr = _rot(b1, tmp_path, {"A": [(1, reason, 1)], "B": [(1, "max_new:2", 2)]}, {"A": 4, "B": 4})
    assert rc == 1 and calls == ["A"] and pr["status"] == "stopped" and reason in pr["stop_reason"]


def test_rotate_stops_on_error_no_progress_and_stop_file(b1, tmp_path):
    (tmp_path / "e").mkdir()
    rc, calls, pr = _rot(b1, tmp_path / "e", {"A": [(3, None, 0)]}, {"A": 2})
    assert rc == 3 and pr["status"] == "error"
    (tmp_path / "n").mkdir()
    with pytest.raises(RuntimeError, match="増えない"):                               # 1 巡して完全な試行が増えない
        _rot(b1, tmp_path / "n", {"A": [(1, "max_new:2", 0), (1, "max_new:2", 0)]}, {"A": 2})
    assert json.loads((tmp_path / "n" / "rot.progress.json").read_text(encoding="utf-8"))["status"] == "error"
    (tmp_path / "s").mkdir()
    rc, calls, pr = _rot(b1, tmp_path / "s", {"A": [(0, None, 2)]}, {"A": 2}, stop=lambda n: "stop_file:rotate_X.STOP")
    assert rc == 1 and calls == [] and pr["status"] == "stopped"


def test_b1_rotate_child_argv_release_and_refusals(b1, tmp_path):
    a = b1.build_parser().parse_args(["rotate", "--gate1", "g.json", "--experiment", "S4B1", "--v3-off", "d"])
    argv = b1.child_argv(a, "N1v3_v3", ["--accept-env-change"], 10)
    assert argv[0] == sys.executable and argv[2:5] == ["run", "--arm", "N1v3_v3"]
    assert argv[-3:] == ["--accept-env-change", "--max-new", "10"]
    ca, extra = b1.build_parser().parse_known_args(argv[2:])                          # 子の引数として読める
    assert ca.arm == "N1v3_v3" and ca.v3_off == "d" and ca.gate1 == "g.json" and extra == ["--accept-env-change", "--max-new", "10"]
    # --in-process: 模型が替わるときに前の模型を手放す（Engine が持つ辞書も空になる）
    b1.SHARED_CACHE.clear()
    held = b1.SHARED_CACHE.setdefault("R1v3", {"pol": object()})
    assert b1.release_models("R1v3") == [] and "pol" in held
    assert b1.release_models("N1v3") == ["R1v3"] and held == {} and "R1v3" not in b1.SHARED_CACHE
    # rotate が決める引数は渡せない（何も回さず、何も書かない）
    g = str(_gate1(tmp_path))
    assert b1.main(["rotate", "--gate1", g, "--experiment", "S4B1TESTROT", "--max-new", "3"]) == 3
    assert not (b1.OUTD / "rotate_S4B1TESTROT.progress.json").exists()


# ================================================================ 採否の判定（98_s4_b_decide.py）: 入口の点検と 2 通りの数え方の一致
@pytest.fixture(scope="module")
def dec():
    return _load(ROOT / "scripts" / "98_s4_b_decide.py", "s4_b_decide_t")


TL_TASK = {"step_timeout_s": 30.0, "retry": 1, "task_time_limit_s": 200.0, "source": "96_s4_resume"}
TL_RUN = {"time_limit_s": 60.0, "source": "96_s4_resume --time-limit-s"}
V3SET = V3.settings({"a_goal": "ES"})


def _cond_files(d, met=True, segs=1):
    """96 が全部そろったときに書く run.json（env_segments）と G_AUDIT.json。"""
    (d / "run.json").write_text(json.dumps({"env_segments": [{"env": {}, "trials": []}] * segs}), encoding="utf-8")
    (d / "G_AUDIT.json").write_text(json.dumps({"met": met}), encoding="utf-8")


def _e7_run(d, arm, i, seed, all3, lift, false=False, audit_ok=True, timed_out=False, api_retry=False):
    """3 個の連続タスクの記録 1 本（run_NNNN.json・npz）。2 番目の手順（緑）は t=10 に始まり、12 s に閉じ、15 s に開く。"""
    from recovla.sim import frames
    t = np.round(np.arange(0, 40.0, 0.05), 6)
    nf = len(t)
    cube = np.zeros((nf, 3, 3))
    cube[:, :, 2] = frames.CUBE_REST_Z
    if lift:
        cube[(t >= 13.0) & (t < 15.0), 1, 2] = frames.CUBE_REST_Z + 0.05
    gc = ((t >= 12.0) & (t < 15.0)).astype(np.int8)
    np.savez(d / f"run_{i:04d}.npz", sim_time=t, gripper_closed=gc, cube_pos=cube, fingertip=np.zeros((nf, 3)),
             plan_step=np.where(t >= 10.0, 1, 0).astype(np.int16))
    inb = {"red": True, "green": all3, "blue": all3}
    truth = {"red": 8.0} if not false else {}
    if all3:
        truth.update({"green": 20.0, "blue": 30.0})
    steps = [{"step": 0, "color": "red", "attempts": [{"attempt": 0, "t_begin": 1.0, "t_judge": 9.0}], "t_start": 1.0,
              "judged_complete": True, "t_judge": 9.0, "t_end": 10.0},
             {"step": 1, "color": "green", "attempts": [{"attempt": 0, "t_begin": 10.0, "t_judge": 22.0 if all3 else None}],
              "t_start": 10.0, "judged_complete": all3, "t_judge": 22.0 if all3 else None, "t_end": 25.0}]
    bad = 0 if audit_ok else 1                                                       # 1 本の試行に G1 と G3 の 2 種類の違反
    au = {"g1": {"violations": bad}, "g2": {"world_stops": 0, "early_use": 0}, "g3": {"total_violations": bad}}
    v3 = arm.endswith("_v3")
    m = {"run": i, "seed": seed, "plan": {"steps": ["red", "green", "blue"], "from_cache": True}, "steps": steps, "returns": [],
         "stopped": None if all3 else {"step": 1, "t": 70.0, "reply": "x"}, "truth_success_t": truth, "final_in_box": inb,
         "all_three_in_box": all(inb.values()), "t_end": 40.0, "timed_out": timed_out, "audit": au, "time_limits": TL_TASK,
         "b1": {"arm": arm, "executor": "v3" if v3 else "current", "executor_version": V3.VERSION if v3 else "current",
                "api_retry": {"n": 1, "error": "x"} if api_retry else None}}
    if v3:
        m["v3"] = {"version": V3.VERSION, "settings": V3SET}
    (d / f"run_{i:04d}.json").write_text(json.dumps(m), encoding="utf-8")


def _b1_dirs(root, cur_all3, v3_all3, cur_lift, v3_lift, n=10, false_v3=0, bad_audit=0, exp="S4B1"):
    base = root / exp
    for arm, all3, lift, nn in (("R1v3_cur", cur_all3, cur_lift, n), ("R1v3_v3", v3_all3, v3_lift, n), ("N1v3_v3", 2, 2, 5)):
        d = base / arm
        d.mkdir(parents=True)
        bad = bad_audit if arm == "R1v3_cur" else 0
        for i in range(nn):
            _e7_run(d, arm, i, 191100 + i, i < all3, i < lift, false=(arm == "R1v3_v3" and i < false_v3), audit_ok=not i < bad,
                    timed_out=(arm == "R1v3_v3" and i == 0), api_retry=(arm == "N1v3_v3" and i == 1))
        _cond_files(d, met=bad == 0)
    return base


def _b1_plan(n=10):
    return {"R1v3_cur": list(range(191100, 191100 + n)), "R1v3_v3": list(range(191100, 191100 + n)),
            "N1v3_v3": list(range(191100, 191105))}


def test_b1_decide_adopts_and_rejects(dec, tmp_path):
    r = dec.b1_decide(_b1_dirs(tmp_path / "a", 0, 9, 2, 6), plan=_b1_plan())
    assert r["adopt_v3"] and r["conditions"]["c1"]["value"] == 9 and r["conditions"]["c2"]["value"] == 0.4
    assert r["counts"]["N1v3_v3"]["n"] == 5 and "success_at_k" in r["display"]["N1v3_v3"]
    assert r["counts"]["R1v3_v3"]["timed_out"] == 1 and r["counts"]["N1v3_v3"]["api_retries"] == 1   # 第 7 節: 並べて出す
    md = dec.md_b1(r)
    assert "timed_out" in md and "API の回し直し" in md
    r = dec.b1_decide(_b1_dirs(tmp_path / "b", 0, 8, 2, 6), plan=_b1_plan())
    assert not r["adopt_v3"] and not r["conditions"]["c1"]["pass"]                   # +8 は足りない
    r = dec.b1_decide(_b1_dirs(tmp_path / "c", 0, 9, 2, 4), plan=_b1_plan())
    assert not r["conditions"]["c2"]["pass"]                                         # 持ち上がり +0.20
    r = dec.b1_decide(_b1_dirs(tmp_path / "d", 0, 9, 2, 6, false_v3=1), plan=_b1_plan())
    assert r["conditions"]["c3"]["value"] == 1 and not r["adopt_v3"]                 # 誤った完了
    r = dec.b1_decide(_b1_dirs(tmp_path / "e", 0, 9, 2, 6, bad_audit=1), plan=_b1_plan())
    assert r["conditions"]["c4"]["value"] == 1 and not r["adopt_v3"]                 # 2 種類の違反のある 1 本は 1 本と数える
    assert r["counts"]["R1v3_cur"]["audit_bad_trials"] == 1


def _mut_meta(base, arm, i, fn):
    p = base / arm / f"run_{i:04d}.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    fn(m)
    p.write_text(json.dumps(m), encoding="utf-8")


B1_BREAKS = {
    "no_run_json": lambda b: (b / "R1v3_v3" / "run.json").unlink(),
    "no_g_audit": lambda b: (b / "N1v3_v3" / "G_AUDIT.json").unlink(),
    "two_env_segments": lambda b: _cond_files(b / "R1v3_cur", segs=2),
    "arm_label": lambda b: _mut_meta(b, "R1v3_v3", 3, lambda m: m["b1"].update(arm="R1v3_cur")),
    "cur_has_v3": lambda b: _mut_meta(b, "R1v3_cur", 0, lambda m: m.update(v3={"settings": V3SET})),
    "v3_settings_differ": lambda b: _mut_meta(b, "N1v3_v3", 2, lambda m: m["v3"]["settings"].update(d_noclose_s=10.0)),
    "version_differ": lambda b: _mut_meta(b, "N1v3_v3", 2, lambda m: m["b1"].update(executor_version="v3.0")),
    "time_limit_differ": lambda b: _mut_meta(b, "R1v3_cur", 4, lambda m: m["time_limits"].update(step_timeout_s=20.0)),
    "seed_differ": lambda b: _mut_meta(b, "R1v3_v3", 9, lambda m: m.update(seed=191199)),
}


@pytest.mark.parametrize("brk", sorted(B1_BREAKS))
def test_b1_entry_audit_refuses_incomplete(dec, tmp_path, brk):
    base = _b1_dirs(tmp_path, 0, 9, 2, 6)
    B1_BREAKS[brk](base)
    with pytest.raises(dec.Incomplete):
        dec.b1_decide(base, plan=_b1_plan())


def test_b1_main_incomplete_against_plan_and_smoke_rules(dec, tmp_path):
    _b1_dirs(tmp_path, 0, 9, 2, 6)                                                    # 10 本しかない: 計画（60/60/30）と違う
    out = tmp_path / "o.json"
    assert dec.main(["b1", "--root", str(tmp_path), "--out", str(out), "--md", str(tmp_path / "o.md")]) == 3
    assert not out.exists()
    assert dec.b1_plan()["R1v3_cur"] == list(range(191100, 191160)) and dec.b1_plan()["N1v3_v3"] == list(range(191100, 191130))
    assert dec.main(["b1", "--root", str(tmp_path), "--smoke"]) == 2                  # --smoke は S4SMOKE の実験だけ


def test_b1_decide_stops_on_mismatch(dec, tmp_path):
    base = _b1_dirs(tmp_path, 0, 9, 2, 6, exp="S4SMOKE_B1T")
    p = base / "R1v3_v3" / "run_0000.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    m["all_three_in_box"] = not m["all_three_in_box"]                                # A（all_three_in_box）と B（final_in_box）が食い違う
    p.write_text(json.dumps(m), encoding="utf-8")
    assert dec.main(["b1", "--root", str(tmp_path), "--experiment", "S4SMOKE_B1T", "--smoke", "--out", str(tmp_path / "o.json"),
                     "--md", str(tmp_path / "o.md")]) == 1
    assert not (tmp_path / "o.json").exists()


def test_dual_tally_paths_do_not_share_listing_or_median(dec, tmp_path):
    for f in (dec.b1_count_a, dec.b2_count_a):                                      # 呼び出し（"名前("）で見る
        src = inspect.getsource(f)
        assert "_files_a(" in src and "_files_b(" not in src and "_median_b(" not in src
    for f in (dec.b1_count_b, dec.b2_count_b):
        src = inspect.getsource(f)
        assert "_files_b(" in src and "_files_a(" not in src and "_median_a(" not in src
    assert "_median_a(" in inspect.getsource(dec.b2_count_a) and "_median_b(" in inspect.getsource(dec.b2_count_b)
    for n in ("run_0000.json", "run_0001.json", "run_0000_runtime.json", "run_x.json", "trial_0003.json"):
        (tmp_path / n).write_text("{}", encoding="utf-8")
    assert dec._files_a(tmp_path, "run") == dec._files_b(tmp_path, "run") == [tmp_path / "run_0000.json", tmp_path / "run_0001.json"]
    assert dec._median_a([3.0, None, float("nan"), 1.0, 2.0, 10.0]) == dec._median_b([3.0, None, float("nan"), 1.0, 2.0, 10.0]) == 2.5


# ---------------------------------------------------------------- B2
def _rtc_trial(d, i, seed, success, t_success, jerky, setting, part, induce=None):
    t = np.round(np.arange(0, 60.0, 0.05), 6)
    x = np.stack([0.4 + 0.05 * np.sin(0.3 * t), 0.02 * np.cos(0.2 * t), 0.2 + 0.0 * t], axis=1)
    if jerky:
        x = x + 0.002 * np.sign(np.sin(7.0 * t))[:, None]
    np.savez(d / f"trial_{i:04d}.npz", sim_time=t, x_des=x)
    m = {"trial": i, "seed": seed, "target": "red", "success": success, "t_success": t_success if success else None,
         "time_limits": TL_RUN, "induce": induce or {}, "diag": {"arm": setting, "shadow": False}}
    if part != "natural_drtc":
        m["b2"] = {"setting": setting, "part": part}
    (d / f"trial_{i:04d}.json").write_text(json.dumps(m), encoding="utf-8")


def _b2_dirs(root, nat, rec, jerky, exp="S4B2", drtc="S4DRTC"):
    """nat・rec: 設定 → (成功の本数, 本数)。自然は D-RTC 3 本＋延長、P1 は誘発が 30 s より前に成立した試行だけが分母
    （40 s の成立と、ちょうど 30.0 s の成立の 2 本は分母に入らない。0155 の 1-1）。"""
    for s in nat:
        dd = root / drtc / s
        de, dp = root / exp / f"{s}_ext", root / exp / f"{s}_P1"
        for d in (dd, de, dp):
            d.mkdir(parents=True)
        k, n = nat[s]
        for i in range(n):
            if i < 3:
                _rtc_trial(dd, i, 191200 + i, i < k, 20.0, jerky[s], s, "natural_drtc")
            else:
                _rtc_trial(de, i - 3, 191300 + i, i < k, 20.0, jerky[s], s, "ext")
        k, n = rec[s]
        for i in range(n):
            _rtc_trial(dp, i, 191400 + i, i < k, 25.0, jerky[s], s, "P1",
                       induce={"kind": "P1", "established": True, "t_established": 5.0})
        _rtc_trial(dp, n, 191400 + n, True, 25.0, jerky[s], s, "P1", induce={"kind": "P1", "established": True, "t_established": 40.0})
        _rtc_trial(dp, n + 1, 191400 + n + 1, False, None, jerky[s], s, "P1",
                   induce={"kind": "P1", "established": True, "t_established": 30.0})          # ちょうど L 秒: 分母に入らない
        for d in (dd, de, dp):
            _cond_files(d)
    return root


def _strict96(dec):
    """直した後の 96（t_established < L）の代わり: 本物の 96 の score_condition を L − 1e-6 で呼び、鍵を L に戻す
    （この合成の記録の成功の時刻は L から離れているので、成功の側は変わらない）。96 の直しが入れば要らない。"""
    r96 = dec._r96()

    def sc(d, ats):
        out = r96.score_condition(d, tuple(T - 1e-6 for T in ats))
        out["at"] = {f"{T:g}": v for T, v in zip(ats, out["at"].values())}
        return out
    return types.SimpleNamespace(score_condition=sc)


def test_b2_refuses_until_96_scoring_fix(dec, tmp_path):
    """96 の score_condition が t_established <= L のままなら、B2 は判定しない（未完、終了コード 3、何も書かない）。"""
    assert dec.score_fix_landed(_strict96(dec))                                     # 直した 96 なら通る（探りの検査）
    if dec.score_fix_landed(dec._r96()):
        pytest.skip("96 の採点の直し（枝 s4-scoring-fix）が入っている: この拒否は働かない（判定の検査が本物の 96 で通る）")
    root = _b2_dirs(tmp_path, nat={"naive": (8, 10), "range10_cap5": (8, 10)}, rec={"naive": (5, 10), "range10_cap5": (5, 10)},
                    jerky={"naive": True, "range10_cap5": False}, exp="S4SMOKE_B2T")
    with pytest.raises(dec.Incomplete) as e:
        dec.b2_decide(root / "S4SMOKE_B2T", root / "S4DRTC", ["range10_cap5"], smoke=True)
    assert any("t_established" in x for x in e.value.problems)
    out = tmp_path / "o.json"
    assert dec.main(["b2", "--root", str(root), "--experiment", "S4SMOKE_B2T", "--settings", "range10_cap5", "--smoke",
                     "--out", str(out), "--md", str(tmp_path / "o.md")]) == 3
    assert not out.exists()


def test_b2_decide_rules_and_double_count(dec, tmp_path, monkeypatch):
    monkeypatch.setattr(dec, "_r96", lambda s=_strict96(dec): s)
    root = _b2_dirs(tmp_path, nat={"naive": (8, 10), "range10_cap5": (8, 10), "ZEROS": (7, 10)},
                    rec={"naive": (5, 10), "range10_cap5": (5, 10), "ZEROS": (5, 10)},
                    jerky={"naive": True, "range10_cap5": False, "ZEROS": False})
    r = dec.b2_decide(root / "S4B2", root / "S4DRTC", ["range10_cap5", "ZEROS"], smoke=True)
    assert r["counts"]["naive"]["natural"]["n"] == 10 and r["counts"]["naive"]["recovery"]["n"] == 10   # 40 s・30.0 s の成立は分母に入らない
    assert r["counts"]["naive"]["recovery"]["n_trials"] == 12
    assert r["settings"]["range10_cap5"]["pass"] and not r["settings"]["ZEROS"]["conditions"]["c1"]["pass"]   # −0.10 は −0.05 を超える
    assert r["adopt"] == "range10_cap5"
    js = r["counts"]["range10_cap5"]["natural"]["jerk_median"]
    assert js < r["counts"]["naive"]["natural"]["jerk_median"]
    # 躍度が naive より小さくなければ採らない
    root2 = _b2_dirs(tmp_path / "x", nat={"naive": (8, 10), "range10_cap5": (8, 10)}, rec={"naive": (5, 10), "range10_cap5": (5, 10)},
                     jerky={"naive": False, "range10_cap5": True})
    assert dec.b2_decide(root2 / "S4B2", root2 / "S4DRTC", ["range10_cap5"], smoke=True)["adopt"] is None
    # 計画と照らす（smoke でない）: 本数・種が計画と違えば未完
    with pytest.raises(dec.Incomplete):
        dec.b2_decide(root2 / "S4B2", root2 / "S4DRTC", ["range10_cap5"])
    plan = dec.b2_plan()
    assert [len(plan[p]) for p in ("natural_drtc", "ext", "P1")] == [30, 69, 50]


B2_BREAKS = {
    "g_audit_not_met": lambda r: _cond_files(r / "S4B2" / "range10_cap5_P1", met=False),
    "part_label": lambda r: _mut_b2(r / "S4B2" / "range10_cap5_ext", 0, lambda m: m["b2"].update(part="P1")),
    "drtc_shadow": lambda r: _mut_b2(r / "S4DRTC" / "naive", 1, lambda m: m["diag"].update(shadow=True)),
    "seed_differ": lambda r: _mut_b2(r / "S4B2" / "range10_cap5_P1", 2, lambda m: m.update(seed=1)),
    "time_limit_differ": lambda r: _mut_b2(r / "S4B2" / "naive_ext", 0, lambda m: m.update(time_limits={"time_limit_s": 30.0})),
}


def _mut_b2(d, i, fn):
    p = d / f"trial_{i:04d}.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    fn(m)
    p.write_text(json.dumps(m), encoding="utf-8")


@pytest.mark.parametrize("brk", sorted(B2_BREAKS))
def test_b2_entry_audit_refuses_incomplete(dec, tmp_path, monkeypatch, brk):
    monkeypatch.setattr(dec, "_r96", lambda s=_strict96(dec): s)
    root = _b2_dirs(tmp_path, nat={"naive": (8, 10), "range10_cap5": (8, 10)}, rec={"naive": (5, 10), "range10_cap5": (5, 10)},
                    jerky={"naive": True, "range10_cap5": False})
    B2_BREAKS[brk](root)
    with pytest.raises(dec.Incomplete):
        dec.b2_decide(root / "S4B2", root / "S4DRTC", ["range10_cap5"], smoke=True)


def test_b2_decide_stops_on_mismatch(dec, tmp_path, monkeypatch):
    monkeypatch.setattr(dec, "_r96", lambda s=_strict96(dec): s)
    root = _b2_dirs(tmp_path, nat={"naive": (8, 10), "range10_cap5": (8, 10)}, rec={"naive": (5, 10), "range10_cap5": (5, 10)},
                    jerky={"naive": True, "range10_cap5": False}, exp="S4SMOKE_B2T")
    orig = dec.b2_count_b

    def off_by_one(dirs, horizon):                                                   # B の数え方だけが 1 本ずれた
        r = orig(dirs, horizon)
        return dict(r, k=r["k"] + 1)
    monkeypatch.setattr(dec, "b2_count_b", off_by_one)
    rc = dec.main(["b2", "--root", str(root), "--experiment", "S4SMOKE_B2T", "--settings", "range10_cap5", "--smoke",
                   "--out", str(tmp_path / "o.json"), "--md", str(tmp_path / "o.md")])
    assert rc == 1 and not (tmp_path / "o.json").exists()
    monkeypatch.setattr(dec, "b2_count_b", orig)
    monkeypatch.setattr(dec, "_jerk_b", lambda z, h: dec._jerk_a(z, h) * 1.001)      # 躍度の中央値の相対差 1e-3 も止める
    with pytest.raises(dec.Mismatch):
        dec.b2_decide(root / "S4SMOKE_B2T", root / "S4DRTC", ["range10_cap5"], smoke=True)
