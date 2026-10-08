"""段階 4 束 1 の D-復帰（担当 C）: 落下で手を止める版の誘発・関門 C の集計・落下の時刻の解析・98 の帯と順の決まり（CPU だけ）。

使い方: .venv\\Scripts\\python.exe -m pytest -q tests\\test_s4_diag_C.py -p no:cacheprovider
読むもの: src/recovla/diag/recovery.py、scripts/98_s4_d_recovery.py（importlib）、configs/s4_gates.json。
  needs_outputs の 1 件だけ outputs/v2eval/V3S3（段階 3 の記録）を読む。書くもの: なし（pytest の一時フォルダも使わない）。
GPU・シミュレーションの世界は作らない（配置は scene.sample_layout を CPU で引くだけ。着地の検査は偽物に差し替える）。
"""
import importlib.util
import pathlib
import sys
import types

import numpy as np
import pytest

from recovla.diag import recovery as DR
from recovla.eval import induce as I
from recovla.eval import time_scoring as TS
from recovla.expert import script as S
from recovla.sim import frames, scene

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEED = 44470


@pytest.fixture(scope="module")
def d98():
    spec = importlib.util.spec_from_file_location("d_rec_98_t", ROOT / "scripts" / "98_s4_d_recovery.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["d_rec_98_t"] = m
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- 誘発（手を止める版）
BOX = np.array([0.45, 0.25, 0.0])


def _truth(t, cube_xy, cube_z, closed=True, speed=0.0):
    cp = np.zeros((3, 3))
    cp[0] = [cube_xy[0], cube_xy[1], cube_z]
    cp[1] = [0.30, -0.20, frames.CUBE_REST_Z]
    cp[2] = [0.60, -0.20, frames.CUBE_REST_Z]
    lv = np.zeros((3, 3))
    lv[0, 2] = speed
    tip = cp[0].copy()
    return S.Truth(t=t, target=0, cube_pos=cp, cube_quat=np.tile([1.0, 0, 0, 0], (3, 1)), cube_linvel=lv,
                   fingers=np.array([0.02, 0.02]), finger_vel=np.zeros(2), gripper_closed=closed,
                   hand_pos=tip + [0, 0, 0.1], hand_vel=np.zeros(3), fingertip=tip, x_cmd=tip + [0, 0, 0.1], box=BOX)


def _rt(i, t_obs, img_t, k0=0, rows=50):
    return types.SimpleNamespace(active={"i": i, "o0": 0, "k0": k0, "post": [np.zeros(7)] * rows},
                                 log_inf=[{"i": i, "t_obs": t_obs, "img_t": {"overhead": img_t, "wrist": img_t + 0.002}}])


@pytest.fixture()
def hold(monkeypatch):
    monkeypatch.setattr(I.INJ, "landing_check", lambda rig, tr, cfg: {"landing_ok": True, "landing_fail": []})
    lay = scene.sample_layout(SEED)
    return DR.FallHoldInducer(SEED, lay, "red", rig=None), lay


def test_same_params_as_stage3_p2(hold):
    ind, lay = hold
    base = I.Inducer("P2", SEED, lay, "red", None)
    assert ind.params == base.params and ind.kind == "P2"         # 同じ乱数列・同じ順（u が同じ）


def _drive(ind, k, t, a, tr):
    out = ind.filter(k, a, tr)
    return out


def test_hold_until_fresh_chunk(hold):
    ind, _ = hold
    ind.params["u"] = 0.0                                         # 閾値 = 最小距離（決まった場所で開く）
    pol = np.array([0.01, -0.02, 0.005, 0, 0, 0, 1.0])
    far = BOX[:2] + [0.40, 0.0]
    lifted = frames.CUBE_REST_Z + 0.08
    rt = _rt(0, 0.0, 0.0)
    ind.bind_runtime(rt)
    # 掴んで持ち上げた（まだ遠い）: 方策のまま、発動しない
    out = _drive(ind, 10, 1.0, pol, _truth(1.0, far, lifted))
    assert np.allclose(out, pol) and not ind.fired and ind.stage == "armed"
    # 閾値（0.19 m）を切った: 開く。手先は止める
    near = BOX[:2] + [0.18, 0.0]
    out = _drive(ind, 11, 1.1, pol, _truth(1.1, near, lifted))
    assert ind.fired and np.allclose(out[:3], 0) and out[6] == -1.0 and ind.active
    # 落下中・着地: 着地の時刻を記録、手は止めたまま
    ind.after(12, _truth(1.2, near, lifted - 0.04, closed=False, speed=-0.5))
    assert ind.t_land is None
    for j, t in enumerate(np.arange(1.3, 1.75, 0.1)):
        ind.after(13 + j, _truth(float(t), near, frames.CUBE_REST_Z, closed=False))
        out = _drive(ind, 13 + j, float(t), pol, _truth(float(t), near, frames.CUBE_REST_Z, closed=False))
        assert np.allclose(out[:3], 0) and out[6] == -1.0
    assert ind.t_land == pytest.approx(1.3)
    assert ind.established and ind.stage == "done"
    # 成立した後でも、塊が着地の前の観測（画像 1.25 < 1.3）なら止めたまま
    rt.active, rt.log_inf = {"i": 3, "o0": 0, "k0": 15, "post": [np.zeros(7)] * 50}, [{"i": 3, "t_obs": 1.35, "img_t": {"overhead": 1.25, "wrist": 1.26}}]
    out = _drive(ind, 18, 1.8, pol, _truth(1.8, near, frames.CUBE_REST_Z, closed=False))
    assert np.allclose(out[:3], 0) and not ind.released
    # 着地の後の観測から作った塊（画像 1.35 >= 1.3）: 放す。方策の行動のまま
    rt.active, rt.log_inf = {"i": 4, "o0": 1, "k0": 18, "post": [np.zeros(7)] * 50}, rt.log_inf + [{"i": 4, "t_obs": 1.45, "img_t": {"overhead": 1.35, "wrist": 1.36}}]
    out = _drive(ind, 19, 1.9, pol, _truth(1.9, near, frames.CUBE_REST_Z, closed=False))
    assert np.allclose(out, pol) and ind.released and not ind.active
    rec = ind.record()
    h = rec["info"]["hold"]
    assert rec["kind"] == "P2" and rec["variant"] == "fall_with_hold" and rec["version"] == "P2H-v1"
    assert h["t_land"] == pytest.approx(1.3) and h["release_reason"] == "fresh_chunk" and h["release_chunk"] == 4
    assert h["release_chunk_img_t_min"] >= h["t_land"] and h["t_release"] == pytest.approx(1.9)
    assert h["hold_s"] == pytest.approx(0.8) and h["t_drop"] == pytest.approx(1.1)
    # 放した後は方策のまま
    out = _drive(ind, 20, 2.0, pol, _truth(2.0, near, frames.CUBE_REST_Z, closed=False))
    assert np.allclose(out, pol)


def test_held_chunk_row_beyond_length_keeps_holding(hold):
    ind, _ = hold
    ind.t_land = 1.0
    ind.bind_runtime(_rt(5, 2.0, 2.0, k0=0, rows=3))
    assert ind._chunk_now(5) is None                             # 塊の長さを超えた行（保持）は塊とみなさない
    assert ind._chunk_now(2)["i"] == 5


def test_unbound_runtime_raises(hold):
    ind, _ = hold
    ind.t_land = 1.0
    with pytest.raises(RuntimeError):
        ind._chunk_now(0)


def test_make_inducer_variants():
    lay = scene.sample_layout(SEED)
    assert isinstance(DR.make_inducer("fall_with_hold", SEED, lay, "red", None), DR.FallHoldInducer)
    for v, k in (("fall_as_is", "P2"), ("misplace", "P3"), ("grasp_failure", "P1")):
        ind = DR.make_inducer(v, SEED, lay, "red", None)
        assert type(ind) is I.Inducer and ind.kind == k


# ---------------------------------------------------------------- 関門 C の集計
def _rec(seed, est, t_est, succ, t_succ, limit=60.0, target="red"):
    return {"seed": seed, "target": target, "time_limit_s": limit, "success": succ, "t_success": t_succ if succ else None,
            "induce": {"kind": "P2", "established": est, "t_established": t_est if est else None}}


def _synthetic():
    R = [_rec(1, True, 5, True, 20), _rec(2, True, 5, True, 40), _rec(3, True, 35, True, 50), _rec(4, False, None, False, None),
         _rec(5, True, 8, False, None), _rec(6, True, 9, True, 29.99)]
    N = [_rec(1, True, 6, False, None), _rec(2, True, 6, True, 25), _rec(3, True, 7, True, 33), _rec(4, True, 7, True, 10),
         _rec(5, True, 31, False, None), _rec(6, True, 9, False, None), _rec(7, True, 3, True, 3)]
    return R, N


def test_gate_c_material_definitions():
    R, N = _synthetic()
    m = DR.gate_c_material(R, N)
    p30 = m["by_L"]["30"]
    assert (p30["recovery_R"]["recovered"], p30["recovery_R"]["established"]) == (2, 4)      # 種 1・6 が 30 s までに成功、種 3 は 35 s 成立で除外
    assert p30["paired"]["pairs"] == 3                                                       # 種 1・2・6（5 は N が 31 s 成立）
    assert (p30["paired"]["r_only"], p30["paired"]["n_only"], p30["paired"]["both"]) == (2, 1, 0)
    assert m["primary_30s"]["R_minus_N"] == pytest.approx(1 / 3)
    p60 = m["by_L"]["60"]
    assert (p60["recovery_R"]["recovered"], p60["recovery_R"]["established"]) == (4, 5)
    assert p60["paired"]["pairs"] == 5


def test_gate_c_matches_time_scoring():
    R, N = _synthetic()
    assert DR.cross_check(R, N)["agree"]


def test_cannot_score_beyond_time_limit():
    R = [_rec(1, True, 5, True, 20, limit=30.0)]
    with pytest.raises(ValueError):
        DR.recovery_at(R, 60.0)


# ---------------------------------------------------------------- 落下の時刻（記録から）
def _fall_record(hold_variant: bool):
    t = np.round(np.arange(0, 6.0, 0.05), 3)
    n = len(t)
    cube = np.zeros((n, 3, 3))
    z = np.full(n, frames.CUBE_REST_Z + 0.08)
    z[t >= 2.0] = np.linspace(frames.CUBE_REST_Z + 0.08, frames.CUBE_REST_Z, 7)[np.minimum(np.arange((t >= 2.0).sum()), 6)]
    cube[:, 0, 2] = z
    ft = np.zeros((n, 3))
    chunk = np.where(t < 1.0, -1, (t // 0.6).astype(int))
    meta = {"seed": 1, "trial": 0, "induce": {"kind": "P2", "fired": True, "t_fire": 2.0, "established": True, "t_established": 2.8}}
    t_land = float(t[np.argmax(z - frames.CUBE_REST_Z < DR.LAND_DZ_M)])
    inf = [{"i": i, "t_obs": 0.6 * i - 0.3, "img_t": {"overhead": 0.6 * i - 0.4}, "t_act": 0.6 * i} for i in range(10)]
    acts = []
    for k in range(60):
        tk = round(0.1 * k, 3)
        c = int(tk // 0.6) if tk >= 1.0 else None
        a = [0.01, 0.0, 0.0, 0, 0, 0, 1.0]
        if hold_variant and 2.0 <= tk < 4.2:
            a = [0.0, 0.0, 0.0, 0, 0, 0, -1.0]
        acts.append({"k": k, "t": tk, "chunk": c, "held": c is None, "a": a})
    return meta, {"sim_time": t, "cube_pos": cube, "target": np.zeros(n, np.int8), "chunk_id": chunk, "fingertip": ft}, \
        {"inference": inf, "actions": acts}, t_land


def test_fall_timing_classes():
    meta, arr, rt, t_land = _fall_record(False)
    r = DR.fall_timing(meta, arr, rt)
    assert r["t_land"] == pytest.approx(t_land) and r["t_drop"] == 2.0
    assert r["v1"]["chunk"] == 4 and r["v1"]["class_t_obs"] == "falling"                # 塊 4: t_obs 2.1（開いた後、着地の前）
    assert r["moved"]["chunk"] == 4
    meta, arr, rt, _ = _fall_record(True)
    r = DR.fall_timing(meta, arr, rt)
    assert r["moved"]["t_first_move"] == pytest.approx(4.2)                             # 止めていた間の行動は除く
    assert r["moved"]["chunk"] == 7 and r["moved"]["class_img"] == "post_land"
    s = DR.summarize_fall_timing([r])
    assert s["moved_class_img"] == {"post_land": 1}


# ---------------------------------------------------------------- 98 の帯と順
def test_band_rules(d98):
    g = d98.gates()
    assert d98.default_trials(g) == "induced:190500:50"
    assert d98.check_band("induced:190500:50", "S4DREC", g) == "production"
    assert d98.check_band("induced:44470:3", "S4SMOKE_C", g) == "smoke"
    for trials, exp in (("induced:190500:49", "S4DREC"), ("induced:190500:50", "S4SMOKE_C"), ("induced:44470:3", "S4DREC"),
                        ("natural:44470:3", "S4SMOKE_C"), ("induced:160000:5", "S4DREC")):
        with pytest.raises(SystemExit):
            d98.check_band(trials, exp, g)


def test_gates_match_variants():
    g = __import__("json").loads((ROOT / "configs" / "s4_gates.json").read_text(encoding="utf-8"))
    assert sorted(g["gates"]["C"]["runs"]["variants"]) == sorted(DR.VARIANT_ORDER)
    assert g["gates"]["C"]["gate_variant"] == "fall_with_hold" and DR.VARIANTS["fall_with_hold"]["hold"]


def test_schedule_interleaves_by_block(d98):
    todo = {"R_a": list(range(5)), "N_a": list(range(5))}
    assert d98.schedule(todo, ["R_a", "N_a"], 2) == [("R_a", 2), ("N_a", 2), ("R_a", 2), ("N_a", 2), ("R_a", 1), ("N_a", 1)]
    todo = {"R_a": [1, 4], "N_a": [0, 1, 2, 3, 4]}                                      # 再開: 欠けた試行だけ、塊の順に
    assert d98.schedule(todo, ["R_a", "N_a"], 2) == [("R_a", 1), ("N_a", 2), ("N_a", 2), ("R_a", 1), ("N_a", 1)]
    assert d98.schedule({"R_a": list(range(5)), "N_a": list(range(5))}, ["R_a", "N_a"], 2, cap=3) == [("R_a", 2), ("N_a", 1)]
    assert d98.schedule({"R_a": []}, ["R_a"], 10) == []


def _fake_96(tmp: pathlib.Path, calls: list, fail_on: str = None):
    """96 の偽物（GPU・シミュレーションなし）。cmd_main は試行のファイルを置くだけで、--max-new で止まる。"""
    import argparse

    def trial_paths(kind, out, i):
        return {"json": out / f"trial_{i:04d}.json", "npz": out / f"trial_{i:04d}.npz", "runtime": out / f"runtime_{i:04d}.json"}

    def check_complete(kind, out, i, expect):
        ok = all(p.is_file() for p in trial_paths(kind, out, i).values())
        return ok, "ok" if ok else "missing", {}

    def build_parser():
        ap = argparse.ArgumentParser()
        sub = ap.add_subparsers(dest="cmd")
        p = sub.add_parser("run")
        for k in ("--experiment", "--condition", "--model", "--trials", "--induce", "--mode", "--quiet-window"):
            p.add_argument(k, default="")
        for k in ("--time-limit-s", "--min-free-gb", "--min-commit-free-gb", "--mem-timeout-min"):
            p.add_argument(k, type=float)
        p.add_argument("--exec-interval", type=int)
        p.add_argument("--max-new", type=int, default=0)
        for k in ("--no-safety", "--dry-run", "--accept-env-change"):
            p.add_argument(k, action="store_true")
        return ap

    def cmd_main(ns, v82, ops):
        calls.append((ns.condition, ns.max_new, ns.diag_variant))
        if fail_on and ns.condition == fail_on:
            raise SystemExit("食い違い")
        out = v82.OUT / ns.experiment / ns.condition
        out.mkdir(parents=True, exist_ok=True)
        ran = 0
        for i in range(len(v82.trial_list(ns.trials))):
            if check_complete("run", out, i, None)[0]:
                continue
            if ns.max_new and ran >= ns.max_new:
                (out / "progress.json").write_text(__import__("json").dumps({"stop_reason": f"max_new:{ns.max_new}"}), encoding="utf-8")
                return 1
            for p in trial_paths("run", out, i).values():
                p.write_text("x", encoding="utf-8")
            ran += 1
        for name in ("run.json", "G_AUDIT.json"):                  # 本物の 96 と同じく、全部そろったら両方を書く
            (out / name).write_text("{}", encoding="utf-8")
        (out / "progress.json").write_text("{}", encoding="utf-8")
        return 0

    class Progress:
        def __init__(self, path, ops, base):
            self.path, self.d = path, dict(base)

        def update(self, **kw):
            self.d.update(kw)
            self.path.write_text(__import__("json").dumps(self.d, default=str), encoding="utf-8")

        def start_heartbeat(self):
            pass

        def stop_heartbeat(self):
            pass

    v82 = types.SimpleNamespace(OUT=tmp, trial_list=lambda s: [(44470 + i, None, "red") for i in range(int(s.split(":")[2]))])
    return types.SimpleNamespace(Engine=type("Engine", (), {"run_one": lambda self, *a: None}), make_spec=lambda a: {},
                                 run_json_text=lambda *a: "{}", load_ops=lambda: None, load_82=lambda allow: v82,
                                 check_complete=check_complete, trial_paths=trial_paths, build_parser=build_parser,
                                 cmd_main=cmd_main, Progress=Progress, _now_s=lambda: "now")


def test_driver_interleaves_and_stops(d98, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(d98, "load_96", lambda: _fake_96(tmp_path, calls))
    monkeypatch.setattr(d98, "OUT_S4", tmp_path / "s4")
    argv = ["run", "--experiment", "S4SMOKE_T", "--trials", "induced:44470:3", "--variants", "fall_with_hold",
            "--models", "R1v3,N1v3", "--block", "2"]
    assert d98.main(argv) == 0
    assert calls == [("R1v3_fall_with_hold", 2, "fall_with_hold"), ("N1v3_fall_with_hold", 2, "fall_with_hold"),
                     ("R1v3_fall_with_hold", 1, "fall_with_hold"), ("N1v3_fall_with_hold", 1, "fall_with_hold")]
    prog = __import__("json").loads(next((tmp_path / "s4").glob("progress_*.json")).read_text(encoding="utf-8"))
    assert prog["status"] == "done" and prog["done"] == 6 and len(prog["rss_gb_after_call"]) == 4
    calls.clear()
    assert d98.main(argv) == 0 and calls == []                                          # 全部完全なら何も回さない
    calls2 = []
    monkeypatch.setattr(d98, "load_96", lambda: _fake_96(tmp_path, calls2, fail_on="N1v3_misplace"))
    assert d98.main(["run", "--experiment", "S4SMOKE_T", "--trials", "induced:44470:3", "--variants", "misplace",
                     "--models", "R1v3,N1v3", "--block", "2"]) == 3                      # 96 の食い違い（終了コード 3）で止まる
    assert [c[0] for c in calls2] == ["R1v3_misplace", "N1v3_misplace"]


@pytest.mark.needs_outputs
def test_stage3_values():
    v = ROOT / "outputs" / "v2eval" / "V3S3"
    m = DR.gate_c_material(DR.load_condition(v / "A_P1"), DR.load_condition(v / "B_P1"), (30.0,))["primary_30s"]
    assert m["recovery_R_k_n"] == [16, 32] and (m["pairs"], m["r_only"], m["n_only"]) == (27, 13, 0)
    s = DR.summarize_fall_timing(DR.fall_timing_dir(v / "A_P2"))
    assert s["n_established"] == 38 and s["v1_class_t_obs"].get("post_land", 0) == 4                 # 34/38 が着地の前
