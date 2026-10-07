"""段階 4 束 1 の担当 B（D-E7・D-単発の開始・移植の腕）の検査。CPU だけ（GPU・方策・カメラの描画を使わない）。
使い方: .venv\\Scripts\\python.exe -m pytest -q tests\\test_s4_diag_B.py -p no:cacheprovider
読むもの: src\\recovla\\diag\\e7.py・start.py、scripts\\98_s4_d_e7.py・98_s4_d_start.py（importlib）、needs_outputs の検査だけ
  outputs\\v2eval\\V3S3\\E7_R1v3 と outputs\\manifests の学習データ（読むだけ）。書くもの: pytest の一時フォルダだけ。
確かめること:
  - 関節空間の戻す動き（GoalMotion）が制限層の上限の中で目標に着き、外した後に直交座標の動きへ跳びなく戻る（戻した後の差 <= 0.01 rad）
  - 置き直した世界（DiagWorldRig）の腕の関節角の差が 0.01 rad 以内、先客が箱の中・壁際の位置にある（CPU の物理だけ）
  - 格子の置き場所が sample_layout(種, "prefilled_1") と同じ、壁際の当て方・XPL_grid の空きの選び方
  - 最初の閉じの物差しが、段階 3 の E7 の 2 番目の 1 回目 5/20・+y 16/19（e7_first_grasp.py）を再現する（needs_outputs）
  - 関門 T の材料の式（E0 の基準・要求の幅・食い違い）、帯の照合
"""
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
E7_DIR = ROOT / "outputs" / "v2eval" / "V3S3" / "E7_R1v3"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- 関節空間の戻す動き
def _motion():
    from recovla.common import config
    from recovla.harness.setup import nominal_setup
    from recovla.runtime.motion import Motion
    return Motion(nominal_setup(config.load("sensor_v1")))


def _joints(q):
    from recovla.runtime.types import JointState
    q = np.asarray(q, float)
    return JointState(0.0, q, np.zeros(7), q.copy())


def test_goal_motion_reaches_goal_within_limits_and_hands_back():
    from recovla.diag import e7 as E7
    from recovla.runtime import limiter as L
    mo = _motion()
    q0 = np.asarray(E7.STANDBY_END_Q)
    mo.reset(_joints(q0))
    mo.__class__ = E7.GoalMotion
    assert not mo.joint_goal_active()
    goal = np.asarray(E7.HOME_Q)
    mo.set_joint_goal(goal, E7.GOAL_PARAMS["vmax_rad_s"], E7.GOAL_PARAMS["gain_per_s"])
    qs = [q0]
    for n in range(5000):                                   # 10 s（500 Hz）
        qs += mo.step(_joints(qs[-1]))
        e, v = mo.joint_goal_error()
        if e <= E7.GOAL_PARAMS["tol_cmd_rad"] and v <= E7.GOAL_PARAMS["tol_vel_rad_s"]:
            break
    assert e <= 1e-3, e
    Q = np.array(qs)
    vel = np.diff(Q, axis=0) / L.DT
    assert np.all(np.abs(vel) <= L.MAX_VEL * 0.99 + 1e-6)
    assert np.max(np.abs(vel)) <= E7.GOAL_PARAMS["vmax_rad_s"] * 1.02   # 関節空間の直線の速さの上限（制限層の追従の行き過ぎ 0.5% ほど）
    acc = np.diff(vel, axis=0) / L.DT
    assert np.all(np.abs(acc) <= L.MAX_ACC * 0.99 + 1e-3)
    assert np.allclose(mo.x_cmd, mo.hand_pose(Q[-1])[0], atol=1e-9)   # x_cmd は指令の手先
    mo.clear_joint_goal()
    q_before = mo.limiter.q.copy()
    for _ in range(250):                                    # 0.5 s 保つ（直交座標の動き、速さ 0）
        qs += mo.step(_joints(qs[-1]))
    assert np.max(np.abs(mo.limiter.q - goal)) <= 0.01      # 戻した後の関節角の差 <= 0.01 rad（束 1 の確かめ方）
    v2 = np.diff(np.array(qs[-501:]), axis=0) / L.DT        # 外した後: 直交座標の IK の姿勢の項でゆっくり動くだけ（跳ばない）
    assert np.max(np.abs(v2)) < 0.2 and np.all(np.abs(np.diff(v2, axis=0)) / L.DT <= L.MAX_ACC * 0.99 + 1e-3)


def test_install_swaps_only_goal_and_rebuild_arms():
    from recovla.diag import e7 as E7

    class Ex:                                               # TaskRuntime の代わり（install が触る値だけ）
        def __init__(self):
            self.prt = type("P", (), {})()
            self.prt.motion = _motion()
    for arm, swapped in (("E0", False), ("EO", False), ("EH", True), ("ES", True), ("EX", True)):
        ex = E7.install(Ex(), arm)
        assert isinstance(ex, E7.DiagTaskRuntime) == swapped, arm
        assert isinstance(ex.prt.motion, E7.GoalMotion) == (arm in ("EH", "ES")), arm
        if swapped:
            assert (ex.goal_q is not None) == (arm in ("EH", "ES"))
            assert ex.rebuild == (arm == "EX")
    assert E7.ARMS["EO"]["text"] != E7.ARMS["E0"]["text"] and E7.ARMS["EO"]["expected_order"] == ["green", "red", "blue"]
    assert E7.ARMS["EH"]["goal_q"] == (0.0, -0.684, 0.0, -2.907, 0.0, 2.216, 0.785)   # 目標書 8-5 の値そのもの


def test_goal_poses_fk_heights():
    """戻し先の手先の高さ: 待機位置の始め z 0.314、終わり z 0.3036（目標書 8-2 の S・final.md）。"""
    from recovla.diag import e7 as E7
    mo = _motion()
    z_s = mo.hand_pose(np.asarray(E7.STANDBY_START_Q))[0][2]
    z_e = mo.hand_pose(np.asarray(E7.STANDBY_END_Q))[0][2]
    assert abs(z_s - 0.3144) < 2e-3 and abs(z_e - 0.3036) < 2e-3, (z_s, z_e)


# ---------------------------------------------------------------- 最初の閉じの物差し（合成の記録）
def _fake_npz(n=40, ci=1, close_at=10, open_at=25, lift=True, dy=0.015):
    t = np.arange(n) * 0.05
    gc = np.zeros(n, bool)
    gc[close_at:open_at] = True
    cube = np.zeros((n, 3, 3))
    cube[:, :, 2] = 0.02
    cube[:, ci, :2] = [0.4, -0.1]
    if lift:
        cube[close_at + 5:open_at, ci, 2] = 0.05
    tip = np.zeros((n, 3))
    tip[:, :2] = [0.4, -0.1 + dy]
    return {"sim_time": t, "gripper_closed": gc, "cube_pos": cube, "fingertip": tip}


def test_first_close_metric_synthetic():
    from recovla.diag import e7 as E7
    z = _fake_npz()
    fc = E7.first_close(z, 1, 0, 40)
    assert fc["frame"] == 10 and fc["lift"] and fc["plus_y"] and abs(fc["dy_cm"] - 1.5) < 1e-9
    assert E7.first_close(z, 1, 11, 40) is None                        # 窓の中に 0→1 が無い
    z2 = _fake_npz(lift=False, dy=0.009)
    fc2 = E7.first_close(z2, 1, 0, 40)
    assert not fc2["lift"] and not fc2["plus_y"]
    meta = {"steps": [{"step": 0, "color": "red", "attempts": [{"attempt": 0, "t_begin": 0.0, "t_judge": 0.2}], "t_end": 0.3},
                      {"step": 1, "color": "green", "attempts": [{"attempt": 0, "t_begin": 0.3, "t_judge": None}], "t_end": None}]}
    rows = E7.attempt_first_closes(meta, z)
    assert [r["step"] for r in rows] == [0, 1] and rows[0]["first_close"] is None and rows[1]["first_close"]["lift"]


def test_gate_T_inputs_arithmetic():
    from recovla.diag import e7 as E7

    def S(k, n, seeds_ok):
        return {"dir": "x", "step2": {"first_close_lift": {"k": k, "n": n, "rate": k / n}, "plus_y_shift": {"k": 1, "n": 2, "rate": .5}},
                "step1": {"first_close_lift": {"k": 1, "n": 1, "rate": 1.0}, "colors": ["red"]},
                "all_three_true": {"k": sum(seeds_ok), "n": len(seeds_ok)}, "timed_out": 0, "plans": [],
                "per_seed": {str(i): {"all_three": bool(v)} for i, v in enumerate(seeds_ok)}}
    g = E7.gate_T_inputs({"E0_run1": S(10, 40, [1, 0, 1, 0]), "E0_run2": S(14, 40, [1, 1, 0, 0]), "EH": S(30, 40, [1, 1, 1, 1])})
    assert g["E0_ref_lift"] == 0.35 and g["d_E0"] == 0.1 and g["required_margin"] == 0.25
    g2 = E7.gate_T_inputs({"E0_run1": S(10, 40, [1, 0]), "E0_run2": S(30, 40, [1, 0])})
    assert g2["required_margin"] == 0.6                                 # max(0.25, 0.5 + 0.10)
    assert g["K_e7"]["all_three_discordance"] == {"k": 2, "n": 4, "rate": 0.5}
    assert g["E0_all_three_max"] == 2 and "判定はしない" in g["note"]


# ---------------------------------------------------------------- 配置・置き場所
def test_grid_slot_matches_sample_layout():
    from recovla.diag import start as S
    from recovla.sim import scene
    for seed in list(range(190400, 190411)) + [44450, 44451]:
        lay = scene.sample_layout(seed, "prefilled_1", start="home")
        assert list(lay.prefilled.values()) == [S.grid_slot(seed)]


def test_start_trial_spec_conditions():
    from recovla.diag import e7 as E7
    from recovla.diag import start as S
    from recovla.sim import scene
    pool = [{"from_index": 1, "rel_xy": [0.0127, -0.054], "quat": [1, 0, 0, 0]},
            {"from_index": 4, "rel_xy": [-0.059, 0.0079], "quat": [0.5, -0.5, -0.5, -0.5]}]
    lay = scene.sample_layout(190401, "empty", start="home")
    l2, ov, rec = S.start_trial_spec(190401, lay, "green", "standby_end", "wall_side", pool, 190400)
    assert l2.kind == "prefilled_1" and l2.start == "standby_end" and set(l2.prefilled) == {"red"}
    assert set(l2.cubes) == {"green", "blue"} and l2.cubes["green"] == tuple(lay.cubes["green"])
    assert ov["arm_q"] == list(E7.STANDBY_END_Q) and rec["wall"]["from_index"] == 4      # k = 1 -> pool[1]
    assert np.allclose(ov["cubes"]["red"][0][:2], np.array([0.45, 0.25]) + [-0.059, 0.0079])
    # 6 条件とも同じ置き直しの道（腕は START_Q へ、先客は置き直しで格子か壁際へ）。10/08 の直し
    l3, ov3, rec3 = S.start_trial_spec(190400, scene.sample_layout(190400, "empty", start="home"), "red", "home", "on_grid", pool, 190400)
    assert l3.start == "placed_home" and set(l3.prefilled) == {"blue"} and ov3["arm_q"] == list(E7.HOME_Q)
    assert rec3["base_start"] == "retreat"
    p3, q3 = S.slot_pose(S.grid_slot(190400))
    assert ov3["cubes"] == {"blue": (p3, q3)}
    l4, ov4, _ = S.start_trial_spec(190400, lay, "blue", "standby_start", "on_grid", pool, 190400)
    assert l4.start == "placed_standby_start" and set(l4.prefilled) == {"green"} and ov4["arm_q"] == list(E7.STANDBY_START_Q)
    for start in S.START_KEYS:
        for prior in S.PRIORS:
            _, ov_, _ = S.start_trial_spec(190402, lay, "green", start, prior, pool, 190400)
            assert ov_["arm_q"] == list(S.START_Q[start]) and set(ov_["cubes"]) == {"red"}, (start, prior)


def test_xpl_spec_grid_moves_red_to_nearest_empty_slot():
    from recovla.diag import e7 as E7
    from recovla.diag import start as S
    st = {"index": 3, "e7_seed": 145003, "target": "green", "joints": [0.1] * 7, "cube_in_box": [True, False, False],
          "cube_pos": [[0.45 + 0.0315, 0.25 - 0.0302, 0.03], [0.40, -0.10, 0.02], [0.50, 0.0, 0.02]],
          "cube_quat": [[1, 0, 0, 0]] * 3}
    l, ov, rec = S.xpl_trial_spec(190423, st, "grid", 1)
    assert rec["grid_slot"] == 1 and np.allclose(ov["cubes"]["red"][0][:2], [0.49, 0.21])   # (+0.04, -0.04)
    assert ov["arm_q"] == [0.1] * 7 and l.kind == "prefilled_1" and set(l.cubes) == {"green", "blue"}
    _, ovh, _ = S.xpl_trial_spec(190423, st, "home", 2)
    assert ovh["arm_q"] == list(E7.HOME_Q) and np.allclose(ovh["cubes"]["red"][0], st["cube_pos"][0])
    slot, _ = S.nearest_empty_slot([0.49, 0.21], [[0.49, 0.21, 0.03]])           # 一番近い置き場所が埋まっていれば次
    assert slot != 1


# ---------------------------------------------------------------- 置き直した世界（CPU の物理だけ）
def test_diag_world_override_holds_arm_and_places_prior():
    from recovla.common import config
    from recovla.diag import e7 as E7
    from recovla.diag import start as S
    from recovla.sim import frames, scene
    w = S.DiagWorldRig(render=False, cfg=config.load("sensor_v1"))
    lay = scene.sample_layout(44450, "empty", start="home")
    pool = [{"from_index": 4, "rel_xy": [-0.059, 0.0079], "quat": [0.5, -0.5, -0.5, -0.5]}]
    l2, ov, _ = S.start_trial_spec(44450, lay, "green", "standby_end", "wall_side", pool, 44450)
    w.diag_override = ov
    w.reset(l2)
    ap = w.diag_applied
    assert ap["arm_q_err_max_rad"] <= 0.01, ap
    assert abs(ap["hand"][2] - 0.3036) < 3e-3
    ri = 0
    assert ap["cube_in_box"][ri] and np.allclose(ap["cube_pos"][ri][:2], np.array([0.45, 0.25]) + [-0.059, 0.0079], atol=3e-3)
    assert w.step == 0 and float(w.data.time) == 0.0 and min(ap["fingers"]) > 0.039
    # home・standby_start も同じ置き直しの道で、その姿勢へ置かれる（腕の差 0.01 rad 以内、先客は格子の置き場所）
    for start, z in (("home", 0.3553), ("standby_start", 0.3144)):
        l5, ov5, _ = S.start_trial_spec(44450, lay, "green", start, "on_grid", pool, 44450)
        w.diag_override = ov5
        w.reset(l5)
        ap5 = w.diag_applied
        assert ap5["arm_q_err_max_rad"] <= 0.01, (start, ap5)
        assert abs(ap5["hand"][2] - z) < 3e-3, (start, ap5["hand"])
        assert ap5["cube_in_box"][ri] and np.allclose(ap5["cube_pos"][ri][:2], ov5["cubes"]["red"][0][:2], atol=3e-3)
        assert w.step == 0 and float(w.data.time) == 0.0
    w.diag_override = None                                    # 置き直しの開始の名札で置き直しが無ければ止める
    import pytest
    with pytest.raises(ValueError):
        w.reset(l5)
    w.diag_override = None                                    # 置き直しなしなら WorldRig と同じ開始
    w.reset(scene.sample_layout(44450, "empty", start="retreat"))
    assert w.diag_applied is None and abs(float(w.data.xpos[w.hand_id][2]) - 0.3144) < 1e-3
    del frames, E7


# ---------------------------------------------------------------- 帯の照合（スクリプト）
def test_band_check():
    m = _load("d_e7_t", "scripts/98_s4_d_e7.py")
    assert m.band_check(range(190300, 190340), "D_E7") == ""
    assert m.band_check(range(44450, 44452), "D_E7") == ""
    assert m.band_check(range(190300, 190341), "D_E7") != ""
    assert m.band_check([190150], "D_E7") != ""
    s = _load("d_start_t", "scripts/98_s4_d_start.py")
    assert s.band_check(range(190400, 190411), "D_single_start") == ""
    assert s.band_check(range(190420, 190440), "D_transplant") == ""
    assert s.band_check(range(190420, 190441), "D_transplant") != ""


# ---------------------------------------------------------------- 段階 3 の記録で物差しを確かめる（outputs が要る）
@pytest.mark.needs_outputs
def test_metric_reproduces_stage3_e7_first_grasp():
    """W\\upper_skeptic\\e7_first_grasp.json: 2 番目の 1 回目 5/19（閉じた 19）・+y 16/19、1 番目 19/20、2・3 番目の 1 回目の +y 22/27。
    s4_gates の first_close_lift の E7 の分母は 2 番目が始まった試行なので 5/20。"""
    from recovla.diag import e7 as E7
    if not E7_DIR.is_dir():
        pytest.skip("V3S3 の記録がない")
    s = E7.summarize_condition(E7_DIR)
    assert s["n"] == 20 and s["all_three_true"]["k"] == 6
    assert s["step2"]["first_close_lift"] == {"k": 5, "n": 20, "rate": 0.25}
    assert s["step2"]["n_close"] == 19 and s["step2"]["plus_y_shift"]["k"] == 16
    assert s["step1"]["first_close_lift"]["k"] == 19
    assert s["later_att0"]["plus_y_shift"] == {"k": 22, "n": 27, "rate": round(22 / 27, 4)}
    assert s["timed_out"] == 0 and abs(s["t_end_max"] - 136.3) < 0.1


@pytest.mark.needs_outputs
def test_xpl_states_and_wall_pool_from_stage3_records():
    from recovla.diag import start as S
    if not E7_DIR.is_dir():
        pytest.skip("V3S3 の記録がない")
    st = S.extract_xpl_states(E7_DIR, 20)
    assert len(st) == 20 and {s["target"] for s in st} == {"green"}
    assert all(s["cube_in_box"] == [True, False, False] and not s["gripper_closed"] for s in st)
    assert all(abs(s["frame_t"] - s["t_obs"]) < 0.05 + 1e-9 and s["t_obs"] >= s["t_step_start"] for s in st)
    pool = S.wall_pool(st)
    assert [w["from_index"] for w in pool] == [1, 2, 4, 8]


@pytest.mark.needs_outputs
def test_train_pose_constants_match_training_data():
    from recovla.diag import e7 as E7
    man = ROOT / "outputs" / "manifests" / "R1v3_20261005-133737.json"
    if not man.is_file():
        pytest.skip("学習データの一覧がない")
    m = E7.train_pose_means(man, ROOT)
    assert np.max(np.abs(np.asarray(E7.STANDBY_START_Q) - m["start_retreat"]["joints"])) < 1e-5
    assert np.max(np.abs(np.asarray(E7.STANDBY_END_Q) - m["end"]["joints"])) < 1e-5
    assert np.max(np.abs(np.asarray(E7.HOME_Q) - m["start_home"]["joints"])) < 1e-3


# ---------------------------------------------------------------- 計画役の API が失敗したときの回し直し（98_s4_d_e7.py の DiagE7Engine）
# 偽の例外（anthropic の本物の例外の型）を、計画役（decompose_s4.decompose）に渡す偽の client から起こす。GPU・API・シミュレーションは
# 使わない: 96 の本物の Engine.run_one_task を通し、run_task_trial（task_loop）と new_world だけを差し替える。キャッシュの書き先は
# 試験の間だけ一時フォルダに向ける（98_s4_d_e7.py・decompose_s4.py の書き先 outputs\llm_cache は変えない）。
def _api_errors():
    import anthropic
    import httpx
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return {"connection": lambda: anthropic.APIConnectionError(request=req),
            "timeout": lambda: anthropic.APITimeoutError(request=req)}


class _FakeMessages:
    """messages.create の偽物。fail の数だけ例外を起こし、その後は計画の JSON を返す。"""

    def __init__(self, fails):
        self.fails, self.calls = list(fails), 0

    def create(self, **kw):
        from types import SimpleNamespace as NS
        self.calls += 1
        if self.fails:
            raise self.fails.pop(0)()
        raw = json.dumps({"steps": ["green", "red", "blue"], "reply": "はい"}, ensure_ascii=False)
        return NS(content=[NS(type="text", text=raw)], model=kw["model"], stop_reason="end_turn", _request_id="req_test",
                  usage=NS(input_tokens=1, output_tokens=1))


def _e7_engine(monkeypatch, tmp_path, fails):
    """96 の本物の Engine を土台にした DiagE7Engine（腕 EO）と、呼ばれた記録。"""
    from types import SimpleNamespace as NS
    from recovla.diag import e7 as E7
    from recovla.harness import task_loop
    from recovla.planner import decompose_s4 as D4
    r96 = _load("s4_resume_test_api_retry", "scripts/96_s4_resume.py")
    m = _load("d_e7_api_retry", "scripts/98_s4_d_e7.py")
    monkeypatch.setattr(D4, "CACHE", tmp_path / "llm_cache")
    msgs = _FakeMessages(fails)
    cli = NS(messages=msgs)
    seen = {"trials": [], "new_world": 0}

    def fake_run_task_trial(world, suite, make_task, lay, text, seed, cfg=None, time_limit_s=None):
        seen["trials"].append({"seed": seed, "world": world, "lay": repr(lay), "text": text})
        plan = D4.decompose(text, ["red", "green", "blue"], [], cli=cli)   # 偽の client。失敗はここから上へ抜ける
        return {"all_three_in_box": False, "plan": plan, "audit": {}}, {"x": np.zeros(1)}, {"runtime": {}}

    def fake_new_world(self):
        seen["new_world"] += 1
        self.world = f"world{seen['new_world']}"

    monkeypatch.setattr(task_loop, "run_task_trial", fake_run_task_trial)
    monkeypatch.setattr(r96.Engine, "new_world", fake_new_world)
    Eng = m.make_engine(r96, "EO", {"script": "98_s4_d_e7.py", "condition": "EO"})
    a = NS(world_per_trial=True, text=E7.ARMS["EO"]["text"], model="R1v3")
    eng = Eng(a, None, {}, {"task_time_limit_s": 200.0}, {"driver": "test"})
    eng.planner_info = {"name": "s4"}
    eng.world = "world0"                                    # init_task の new_world の代わり
    eng.make_task = None                                    # init_task の make の代わり（偽の run_task_trial は使わない）
    return eng, msgs, seen


@pytest.mark.parametrize("kind", ["connection", "timeout"])
def test_e7_api_error_retries_once_with_same_seed(monkeypatch, tmp_path, kind):
    """1 回目が anthropic の例外 → 同じ番号・同じ種で 1 回だけ回し直し（世界は作り直す）、diag.api_retry に残る。
    失敗した呼び出しはキャッシュを書かず、回し直しの成功で 1 回だけ書く。"""
    eng, msgs, seen = _e7_engine(monkeypatch, tmp_path, [_api_errors()[kind]])
    meta, arrays, rlog = eng.run_one_task(7, 190307)
    assert msgs.calls == 2
    assert [t["seed"] for t in seen["trials"]] == [190307, 190307]
    assert seen["trials"][0]["lay"] == seen["trials"][1]["lay"]            # 同じ種の同じ配置
    assert seen["new_world"] == 1 and seen["trials"][1]["world"] != seen["trials"][0]["world"]   # 回し直しは新しい世界
    r = meta["diag"]["api_retry"]
    assert r["n"] == 1 and r["error"].startswith(("APIConnectionError", "APITimeoutError"))
    assert meta["diag"]["arm"] == "EO" and meta["run"] == 7 and meta["plan"]["from_cache"] is False
    assert len(list((tmp_path / "llm_cache").glob("*.json"))) == 1


def test_e7_api_error_twice_propagates(monkeypatch, tmp_path):
    """2 回目も失敗したら例外が上へ抜ける（96 の cmd_main が status=error・終了コード 2 にする）。3 回目は回さない。"""
    import anthropic
    errs = _api_errors()
    eng, msgs, seen = _e7_engine(monkeypatch, tmp_path, [errs["connection"], errs["timeout"]])
    with pytest.raises(anthropic.APITimeoutError):
        eng.run_one_task(0, 190300)
    assert msgs.calls == 2 and [t["seed"] for t in seen["trials"]] == [190300, 190300]
    assert not (tmp_path / "llm_cache").exists() or not any((tmp_path / "llm_cache").iterdir())


def test_e7_non_api_error_is_not_retried(monkeypatch, tmp_path):
    """anthropic の例外でない失敗（鍵が無い RuntimeError など）は回し直さず、そのまま上へ抜ける。成功なら api_retry は None。"""
    eng, msgs, seen = _e7_engine(monkeypatch, tmp_path, [lambda: RuntimeError("ANTHROPIC_API_KEY が設定されていない")])
    with pytest.raises(RuntimeError):
        eng.run_one_task(0, 190300)
    assert msgs.calls == 1 and len(seen["trials"]) == 1
    eng2, msgs2, _ = _e7_engine(monkeypatch, tmp_path / "ok", [])
    meta, _, _ = eng2.run_one_task(1, 190301)
    assert msgs2.calls == 1 and meta["diag"]["api_retry"] is None
