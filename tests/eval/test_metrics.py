"""metrics.py の検査（指示書 0012 §5 の 1）。跳び・時刻・接触・移動を前もって決めた合成の記録で検算する。

GPU・モデル・ネットワーク・MuJoCo なしで通る。期待値は関数を使わずに作り方から直接書く。
"""
import math
import re
import time

import numpy as np
import pytest

from recovla.common import config
from recovla.eval import metrics as mt

from . import synth
from .synth import SynthTrial, frame_time

EVAL = config.load()["eval"]


def _metrics(tr: SynthTrial, tmp_path, eval_cfg=None) -> dict:
    return mt.trial_metrics(mt.load_trial(tr.write(tmp_path)), eval_cfg or EVAL)


def _trials_csv_columns() -> list[str]:
    """docs/interfaces/results.md §2 の trials.csv の列（表の 1 列目）。"""
    text = (config.ROOT / "docs" / "interfaces" / "results.md").read_text(encoding="utf-8")
    sec = text.split("## 2. `trials.csv`", 1)[1].split("\n## ", 1)[0]
    rows = [ln for ln in sec.splitlines() if ln.startswith("|")]
    names = [r.split("|")[1].strip() for r in rows[2:]]            # 見出しと区切りの行を除く
    assert names and all(re.fullmatch(r"[a-z0-9_]+", n) for n in names), names
    return names


# ------------------------------------------------------------------------------ 形と欠けた値

def test_keys_are_the_trials_csv_columns_in_order(tmp_path):
    row = _metrics(SynthTrial(), tmp_path)
    assert list(row) == _trials_csv_columns()


def test_load_trial_round_trip(tmp_path):
    tr = SynthTrial(n_frames=50, trial=7)
    rec = mt.load_trial(tr.write(tmp_path))
    assert rec.meta == tr.meta
    assert set(rec.arrays) == set(tr.arrays)
    for k, v in tr.arrays.items():
        assert rec.arrays[k].dtype == v.dtype and np.array_equal(rec.arrays[k], v), k


def test_identity_columns(tmp_path):
    tr = SynthTrial()
    tr.meta["runtime"].update(mode="sync", delay_steps=None)
    tr.meta["layout"].update(kind="prefilled_1", start="retreat")
    tr.meta["steps"] = [dict(tr.meta["steps"][0]), dict(tr.meta["steps"][0], target="blue")]
    row = _metrics(tr, tmp_path)
    assert row["experiment"] == "E2" and row["condition"] == "R1_rtc_s10_d2" and row["model"] == "R1"
    assert row["mode"] == "sync" and row["s"] == 10 and row["d"] == 0          # sync の d は 0
    assert row["safety_filter"] is False
    assert (row["trial"], row["seed"]) == (0, 110000)
    assert (row["layout_kind"], row["start_pose"]) == ("prefilled_1", "retreat")
    assert row["target"] == "red>blue"
    assert row["induce"] == "none"


def test_missing_values_do_not_raise(tmp_path):
    """推論なし・誘発なし・成功しない・行動なし（NaN）・手順の外だけ。"""
    tr = SynthTrial(n_frames=60)
    tr.meta["inference"] = []
    tr.meta["induce"] = None
    tr.arrays["action"][:] = np.nan
    tr.arrays["chunk_switch"][:] = False
    tr.arrays["target"][:] = -1
    tr.arrays["x_des"][:] = np.nan
    row = _metrics(tr, tmp_path)
    assert row["success"] is False and math.isnan(row["t_success_s"])
    assert row["induce"] == "none" and row["induce_fired"] is False and row["induce_established"] is False
    assert row["recovered"] is None                                          # 誘発が成立していない
    for k in ("reaction_time_s", "recovery_time_s", "seam_jump_mean", "seam_jump_max", "nonseam_jump_mean",
              "seam_ee_speed_jump_mean", "jerk_rms", "inference_mean_s", "inference_p95_s"):
        assert math.isnan(row[k]), k
    assert row["stage_reached"] is None
    assert row["error"] is False and row["collateral"] is False and row["collateral_induced"] is False
    assert row["contacts_n"] == 0


def test_fired_but_not_established(tmp_path):
    tr = SynthTrial()
    tr.set_induce("P2", t_fire=5.0, established=False)
    tr.set_success(15.0)
    row = _metrics(tr, tmp_path)
    assert row["induce"] == "P2" and row["induce_fired"] is True and row["induce_established"] is False
    assert row["recovered"] is None
    assert math.isnan(row["reaction_time_s"]) and math.isnan(row["recovery_time_s"])
    assert row["success"] is True and row["t_success_s"] == 15.0


def test_very_short_record(tmp_path):
    row = _metrics(SynthTrial(n_frames=1), tmp_path)
    assert math.isnan(row["seam_jump_mean"]) and math.isnan(row["jerk_rms"])


# ------------------------------------------------------------------------------ 継ぎ目の跳び

def _chunked_velocity(n_actions: int):
    """塊 c（10 行動）の中で v_k = base_c + (0, 0, 0.02·j)（j は塊の中の添字）[m/s]。

    切り替わりでない k の跳びは |(0, 0, 0.02)| = 0.02。切り替わりの k（c ≥ 1）の跳びは
    |base_c − base_{c−1} − (0, 0, 0.02·9)|。
    """
    bases = np.array([[0.10, 0.00, 0.0], [0.10, 0.05, 0.0], [0.00, 0.05, 0.1], [-0.05, 0.0, 0.0],
                      [0.02, 0.02, 0.02], [0.0, -0.1, 0.0], [0.1, 0.1, 0.1], [0.0, 0.0, 0.0]])
    k = np.arange(n_actions)
    c, j = k // synth.CHUNK, k % synth.CHUNK
    nb = len(bases)                                     # 塊が 8 を越えたら base を繰り返す
    v = bases[c % nb] + np.outer(0.02 * j, [0, 0, 1])
    seam = [np.linalg.norm(bases[cc % nb] - bases[(cc - 1) % nb] - [0, 0, 0.02 * 9]) for cc in range(1, c[-1] + 1)]
    return v, np.array(seam)


def test_seam_jumps_direct():
    v, seam = _chunked_velocity(80)
    action = np.zeros((80, 7))
    action[:, :3] = v * 0.1
    switch = (np.arange(80) % synth.CHUNK) == 0
    got_seam, got_non = mt.seam_jumps(action, switch, dt=0.1)
    assert got_seam == pytest.approx(seam, abs=1e-12)
    assert len(got_non) == 80 - 8 and got_non == pytest.approx(np.full(72, 0.02), abs=1e-12)


def test_seam_jumps_skip_missing_actions():
    action = np.zeros((6, 7))
    action[:, 0] = [0.01, 0.02, np.nan, 0.02, 0.05, 0.05]
    switch = np.array([True, False, True, True, False, True])
    seam, non = mt.seam_jumps(action, switch)
    # k=2 は自分が NaN、k=3 は前が NaN なので除く。切り替わりは k=5（0.05→0.05）、そうでないのは k=1・4
    assert seam == pytest.approx([0.0])
    assert non == pytest.approx([0.1, 0.3])


@pytest.mark.parametrize("switch_on_both_frames", [False, True])
def test_seam_in_trial(tmp_path, switch_on_both_frames):
    tr = SynthTrial(n_frames=160)                        # 行動 80、塊 8
    v, seam = _chunked_velocity(80)
    tr.set_velocity(v, ee_gain=0.5)
    if switch_on_both_frames:                           # 同じ行動が並ぶ 2 こまの両方に旗があっても同じ
        tr.arrays["chunk_switch"] |= np.roll(tr.arrays["chunk_switch"], 1)
    row = _metrics(tr, tmp_path)
    assert row["seam_jump_mean"] == pytest.approx(seam.mean(), abs=1e-12)
    assert row["seam_jump_max"] == pytest.approx(seam.max(), abs=1e-12)
    assert row["nonseam_jump_mean"] == pytest.approx(0.02, abs=1e-12)
    # 手先は参照速度の 0.5 倍で動く → 切り替わりの時点の手先速度の差は跳びの 0.5 倍
    assert row["seam_ee_speed_jump_mean"] == pytest.approx(0.5 * seam.mean(), abs=1e-9)


# ------------------------------------------------------------------------------ 躍度

def test_jerk_cubic_direct():
    t = np.arange(40) * 0.1
    c3 = np.array([0.002, -0.001, 0.0005])
    x = np.outer(t ** 3, c3) + np.outer(t ** 2, [0.01, 0.02, -0.03]) + np.outer(t, [0.1, 0, 0]) + [0.3, 0, 0.2]
    assert mt.jerk_rms(x, 0.1) == pytest.approx(6 * np.linalg.norm(c3), rel=1e-6)


def test_jerk_in_trial_uses_the_10hz_samples(tmp_path):
    tr = SynthTrial(n_frames=200)
    t = tr.arrays["sim_time"]
    c3 = np.array([0.001, 0.0, -0.002])
    tr.arrays["x_des"] = synth.EE0 + np.outer(t ** 3, c3) + np.outer(t, [0.05, 0.0, 0.0])
    tr.arrays["x_des"][1::2] += 0.01                     # 20 Hz の間のこま（行動の間）は使わない
    assert _metrics(tr, tmp_path)["jerk_rms"] == pytest.approx(6 * np.linalg.norm(c3), rel=1e-6)


def test_jerk_short_is_nan():
    assert math.isnan(mt.jerk_rms(np.zeros((3, 3))))


# ------------------------------------------------------------------------------ 反応時間・復帰時間（P2 の筋書き）

FIRE_I, LAND_I, NEAR_I, CROSS_I, REGRASP_I, LIFT_I = 120, 126, 140, 144, 180, 187


def _p2_trial() -> SynthTrial:
    """赤を持ち上げて運ぶ → こま 120（6.00 s）で誘発（開く）→ こま 126（6.30 s）に机に着地（t_failure）。

    指先と赤の距離: 着地の時点で 0.15、こま 140 まで一定、以後 1 こまに 0.006 ずつ縮む
      → d0 − 0.02 = 0.13 以下になるのはこま 144（0.126。こま 143 は 0.132）→ 反応時間 7.20 − 6.30 = 0.90 s
    赤の高さ: こま 180 から 1 こまに 0.003 ずつ上がる → 0.02 以上になるのはこま 187（0.021。186 は 0.018）
      → 復帰時間 9.35 − 6.00 = 3.35 s。落ちている途中（こま 120〜125）は持ち上げの状態だが「再び」ではない
    """
    tr = SynthTrial(n_frames=260)
    red = synth.CUBE0[0].copy()
    z = np.full(tr.n, synth.CUBE_Z)
    z[60:100] = np.linspace(synth.CUBE_Z, 0.10, 40)                    # 持ち上げ
    z[100:FIRE_I] = 0.10
    z[FIRE_I:LAND_I + 1] = np.linspace(0.10, synth.CUBE_Z, LAND_I + 1 - FIRE_I)   # 落下
    z[REGRASP_I:] = synth.CUBE_Z + 0.003 * np.arange(tr.n - REGRASP_I)
    pos = np.tile(red, (tr.n, 1))
    pos[:, 2] = z
    tr.move_cube("red", np.arange(tr.n), pos)
    d = np.full(tr.n, 0.15)
    d[NEAR_I:] = 0.15 - 0.006 * np.arange(tr.n - NEAR_I)
    d[:LAND_I] = 0.0                                                   # 着地より前は持っている
    tr.arrays["fingertip"] = pos + np.outer(d, [0, 0, 1])
    tr.set_induce("P2", t_fire=float(frame_time(FIRE_I)), t_failure=float(frame_time(LAND_I)))
    tr.arrays["phase"][:] = 0
    return tr


def test_reaction_and_recovery_direct():
    tr = _p2_trial()
    a = tr.arrays
    t = a["sim_time"]
    red = a["cube_pos"][:, 0]
    assert mt.reaction_time(t, a["fingertip"], red, frame_time(LAND_I), 0.02) == pytest.approx(0.90, abs=1e-9)
    assert mt.recovery_time(t, red[:, 2], synth.CUBE_Z, frame_time(FIRE_I), 0.02) == pytest.approx(3.35, abs=1e-9)


def test_reaction_and_recovery_in_trial(tmp_path):
    tr = _p2_trial()
    tr.set_success(12.0)
    row = _metrics(tr, tmp_path)
    assert row["induce"] == "P2" and row["induce_fired"] is True and row["induce_established"] is True
    assert row["recovered"] is True
    assert row["reaction_time_s"] == pytest.approx(0.90, abs=1e-9)
    assert row["recovery_time_s"] == pytest.approx(3.35, abs=1e-9)


def test_established_but_not_recovered(tmp_path):
    row = _metrics(_p2_trial(), tmp_path)
    assert row["recovered"] is False


def test_reaction_and_recovery_not_found_are_nan():
    t = frame_time(np.arange(20))
    far = np.zeros((20, 3))
    tip = np.tile([0, 0, 0.1], (20, 1))
    assert math.isnan(mt.reaction_time(t, tip, far, 0.2, 0.02))       # 近づかない
    assert math.isnan(mt.reaction_time(t, tip, far, 5.0, 0.02))       # 記録の外
    assert math.isnan(mt.reaction_time(t, tip, far, None, 0.02))
    assert math.isnan(mt.recovery_time(t, np.full(20, 0.02), 0.02, 0.1, 0.02))   # 持ち上げない
    assert math.isnan(mt.recovery_time(t, np.full(20, 0.10), 0.02, 0.1, 0.02))   # ずっと持ったまま（再びでない）


def test_thresholds_come_from_eval_cfg(tmp_path):
    tr = _p2_trial()
    cfg = dict(EVAL, reaction_close_m=0.05, recovery_lift_m=0.05)
    row = _metrics(tr, tmp_path, cfg)
    # 0.15 − 0.006·m ≤ 0.10 → m = 9（こま 149、7.45 s）。0.003·m ≥ 0.05 → m = 17（こま 197、9.85 s）
    assert row["reaction_time_s"] == pytest.approx(7.45 - 6.30, abs=1e-9)
    assert row["recovery_time_s"] == pytest.approx(9.85 - 6.00, abs=1e-9)


# ------------------------------------------------------------------------------ 巻き添え・接触回数

def _push(tr: SynthTrial, color: str, start: int, dist: float, frames: int = 6):
    c = synth.COLORS.index(color)
    p0 = tr.arrays["cube_pos"][start - 1, c]
    path = p0 + np.outer(np.linspace(0, 1, frames + 1)[1:], [dist, 0, 0])
    tr.move_cube(color, np.arange(start, start + frames), path)


def test_collateral_by_hand(tmp_path):
    tr = SynthTrial()
    tr.touch("cube_green", [50, 51, 52])
    _push(tr, "green", 50, 0.03)
    row = _metrics(tr, tmp_path)
    assert row["collateral"] is True and row["collateral_induced"] is False


def test_collateral_by_the_dropped_cube(tmp_path):
    tr = _p2_trial()
    tr.cube_cube("red", "blue", [LAND_I])                  # 落ちた赤が青に当たる
    _push(tr, "blue", LAND_I, 0.03)
    tr.touch("cube_blue", [LAND_I + 2, LAND_I + 3])        # 後から手も青に触れる（同じ移動の中）
    row = _metrics(tr, tmp_path)
    assert row["collateral"] is False and row["collateral_induced"] is True


def test_collateral_both_kinds_are_counted_separately(tmp_path):
    tr = _p2_trial()
    tr.touch("cube_green", [30, 31])
    _push(tr, "green", 30, -0.025)
    tr.cube_cube("red", "blue", [LAND_I])
    _push(tr, "blue", LAND_I, 0.03)
    row = _metrics(tr, tmp_path)
    assert row["collateral"] is True and row["collateral_induced"] is True


def test_chain_collision_is_induced(tmp_path):
    """玉突き: 落ちた赤が緑に当たり、緑が青に当たる。青の区間の立方体どうしの接触（t_fire 以後）が
    手より先なので、青の移動も「誘発による移動」（掲示板 0015 の 2。青が赤に直接当たったかは問わない）。"""
    tr = _p2_trial()
    tr.cube_cube("red", "green", [LAND_I])
    _push(tr, "green", LAND_I, 0.03)
    tr.cube_cube("green", "blue", [LAND_I + 3])
    _push(tr, "blue", LAND_I + 3, 0.03)
    tr.touch("cube_blue", [LAND_I + 5])                    # 後から手も青に触れる
    row = _metrics(tr, tmp_path)
    assert row["collateral"] is False and row["collateral_induced"] is True


def test_cube_cube_before_fire_is_not_induced(tmp_path):
    """t_fire より前の立方体どうしの接触は誘発によるものでない。手が触れていれば巻き添え。"""
    tr = _p2_trial()
    tr.cube_cube("green", "blue", [40])
    tr.touch("cube_blue", [41])
    _push(tr, "blue", 40, 0.03)
    row = _metrics(tr, tmp_path)
    assert row["collateral"] is True and row["collateral_induced"] is False


def test_hand_first_then_dropped_cube_is_collateral(tmp_path):
    tr = _p2_trial()
    tr.touch("cube_blue", [LAND_I - 1])
    _push(tr, "blue", LAND_I - 1, 0.03, frames=8)
    tr.cube_cube("red", "blue", [LAND_I + 1])
    row = _metrics(tr, tmp_path)
    assert row["collateral"] is True and row["collateral_induced"] is False


def test_no_collateral(tmp_path):
    tr = SynthTrial()
    tr.touch("cube_green", [50, 51])
    _push(tr, "green", 50, 0.015)                          # 0.02 に届かない
    tr.touch("cube_red", [80, 81])
    _push(tr, "red", 80, 0.05)                             # 目標は巻き添えでない
    tr.cube_cube("green", "blue", [100])
    _push(tr, "blue", 100, 0.03)                           # 手の接触も誘発もない（誘発なしの試行）
    row = _metrics(tr, tmp_path)
    assert row["collateral"] is False and row["collateral_induced"] is False


def test_contact_before_rest_is_another_movement(tmp_path):
    """手が触れて少し動かし（止まる）、後で別の原因で 0.02 を越えても、手の移動の区間には入らない。"""
    tr = _p2_trial()
    tr.touch("cube_blue", [40])
    _push(tr, "blue", 40, 0.01)                            # 止まる
    tr.cube_cube("red", "blue", [LAND_I])
    _push(tr, "blue", LAND_I, 0.015)                       # 合わせて 0.025
    row = _metrics(tr, tmp_path)
    assert row["collateral"] is False and row["collateral_induced"] is True


def test_contacts_n(tmp_path):
    tr = SynthTrial()
    tr.touch("wall_xp", [10, 11, 12, 20])                 # 2 回
    tr.touch("cube_green", [0, 1, 30])                    # 2 回（こま 0 の前は偽）
    tr.touch("cube_red", [40, 41, 50])                    # 目標は数えない
    tr.touch("wall_yn", [60])                             # 1 回
    assert _metrics(tr, tmp_path)["contacts_n"] == 5


# ------------------------------------------------------------------------------ 段階別の到達・誤り

@pytest.mark.parametrize("phases,expected", [
    ([0, 1, 2, 4, 5, 3, 0, 1], "搬送"),
    ([0, 1, 3, 7, 1], "接近"),
    ([0, 1, 2, 3], "把持"),
    ([0, 1, 2, 4, 5, 6, 7, 8], "退避"),
    ([0, 1, 2, 4, 5, 6, 7, 8, 9], "完了"),
    ([0, 1, 2, 4, 5, 6, 7], "設置"),
    ([3, 7], None),
])
def test_stage_reached(tmp_path, phases, expected):
    tr = SynthTrial(n_frames=len(phases))
    tr.arrays["phase"] = np.array(phases, dtype=np.int8)
    assert _metrics(tr, tmp_path)["stage_reached"] == expected


def test_stage_ignores_frames_outside_a_step(tmp_path):
    tr = SynthTrial(n_frames=6)
    tr.arrays["phase"] = np.array([0, 1, 2, 9, 9, 9], dtype=np.int8)
    tr.arrays["target"] = np.array([0, 0, 0, -1, -1, -1], dtype=np.int8)
    assert _metrics(tr, tmp_path)["stage_reached"] == "把持"


def _lift(tr: SynthTrial, color: str, dz: float, frames=range(50, 70)):
    c = synth.COLORS.index(color)
    frames = np.asarray(list(frames))
    p = np.tile(tr.arrays["cube_pos"][frames[0] - 1, c], (len(frames), 1))
    p[:, 2] += np.linspace(0, dz, len(frames))
    tr.move_cube(color, frames, p)


def test_error_lifting_a_non_target(tmp_path):
    tr = SynthTrial()
    _lift(tr, "green", 0.03)
    assert _metrics(tr, tmp_path)["error"] is True


def test_error_lifting_the_cube_already_in_the_box(tmp_path):
    tr = SynthTrial()
    tr.meta["layout"].update(kind="prefilled_1", prefilled=["blue"])
    box = np.array([0.45, 0.25, synth.CUBE_Z])
    tr.arrays["cube_pos"][:, 2] = box                     # 先客は最初から箱の中
    tr.arrays["cube_in_box"][:, 2] = True
    _lift(tr, "blue", 0.04)
    assert _metrics(tr, tmp_path)["error"] is True


def test_no_error(tmp_path):
    tr = SynthTrial()
    _lift(tr, "red", 0.10)                                # 目標を持ち上げるのは誤りでない
    _lift(tr, "green", 0.015, frames=range(80, 90))       # 0.02 に届かない
    assert _metrics(tr, tmp_path)["error"] is False
    assert _metrics(tr, tmp_path, dict(EVAL, error_lift_m=0.01))["error"] is True


def test_error_follows_the_target_of_each_step(tmp_path):
    tr = SynthTrial()
    tr.arrays["target"][100:] = 2                         # 2 つ目の手順の目標は青
    _lift(tr, "blue", 0.05, frames=range(120, 140))
    assert _metrics(tr, tmp_path)["error"] is False
    _lift(tr, "red", 0.05, frames=range(150, 170))        # 手順が終わった赤を持ち上げるのは誤り
    assert _metrics(tr, tmp_path)["error"] is True


# ------------------------------------------------------------------------------ 推論時間・速さ

def test_inference_times(tmp_path):
    tr = SynthTrial()
    walls = [0.30, 0.10, 0.12, 0.11, 0.20]
    tr.meta["inference"] = [dict(tr.meta["inference"][0], i=j, wall_s=w) for j, w in enumerate(walls)]
    row = _metrics(tr, tmp_path)
    assert row["inference_mean_s"] == pytest.approx(np.mean(walls))
    # 線形補間の 95 パーセンタイル: 並べた [0.10, 0.11, 0.12, 0.20, 0.30] の位置 3.8 → 0.20 + 0.8·0.10
    assert row["inference_p95_s"] == pytest.approx(0.28)


def test_100_trials_in_a_few_seconds(tmp_path):
    base = _p2_trial()
    v, _ = _chunked_velocity(base.n // 2)
    tip = base.arrays["fingertip"].copy()
    base.set_velocity(v)
    base.arrays["fingertip"] = tip                         # 反応時間の筋書きの指先は残す
    paths = []
    for j in range(100):
        tr = base.copy()
        tr.meta["trial"] = j
        paths.append(tr.write(tmp_path))
    # 30 s 分（600 こま）の長さでも測る
    long = SynthTrial(n_frames=600, trial=100)
    long.set_velocity(np.tile([[0.05, 0.0, 0.0]], (300, 1)))
    paths.append(long.write(tmp_path))
    t0 = time.perf_counter()
    rows = [mt.trial_metrics(mt.load_trial(p), EVAL) for p in paths]
    elapsed = time.perf_counter() - t0
    assert len(rows) == 101 and all(r["reaction_time_s"] == pytest.approx(0.90) for r in rows[:100])
    assert elapsed < 5.0, elapsed
