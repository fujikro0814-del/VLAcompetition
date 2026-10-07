"""段階 4 束 1 の担当 A（D-RTC・影の推論・X2）の単体検査。CPU だけで回る（GPU・シミュレーション・記録の読み込みなし）。

- 6 設定の定義が s4_gates.json の gates.R と一致する
- 「論文の式どおり」（範囲 44・EXP）の重みが arXiv 2506.07339 の式 5 と、lerobot 0.6.1 の get_prefix_weights と一致する
- 設定を configs の写しに重ねても元の辞書は変わらない
- 影の推論が同じ雑音で引かれ、実行（戻り値・次の推論の前の塊）は誘導した側のまま
- 関門 R の指標・影・X2 の計算を、答えの分かる人工の記録で確かめる
- 96_s4_resume.py を包んだときの記録のファイル（diag_NNNN.npz）と控え
"""
import importlib.util
import json
import math
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recovla.diag import rtc as D            # noqa: E402


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def drtc():
    return _load(ROOT / "scripts" / "98_s4_d_rtc.py", "s4_d_rtc_testA")


# ------------------------------------------------------------------ 設定
def test_settings_match_gates():
    g = json.loads((ROOT / "configs" / "s4_gates.json").read_text(encoding="utf-8"))["gates"]["R"]
    assert list(D.SETTINGS) == g["runs"]["settings"]
    assert all(c in g["candidates"] for c in D.CANDIDATES)
    assert D.SETTINGS["current_repro"] == {**D.SETTINGS["current_repro"], "mode": "rtc", "horizon": 40, "schedule": "EXP",
                                           "max_guidance_weight": 10.0}
    assert D.SETTINGS["paper_formula_range44_cap5"]["horizon"] == D.CHUNK_H - D.EXEC_S == 44
    for n in ("paper_formula_range44_cap5", "range40_cap5", "range10_cap5", "ZEROS"):
        assert D.SETTINGS[n]["max_guidance_weight"] == 5.0
    assert D.SETTINGS["naive"]["mode"] == "naive"


def test_current_repro_equals_configs():
    import yaml
    rt = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))["runtime"]
    s = D.SETTINGS["current_repro"]
    assert (rt["rtc_guidance_horizon"], rt["rtc_schedule"], float(rt["rtc_max_guidance_weight"])) == \
        (s["horizon"], s["schedule"], s["max_guidance_weight"])


def test_overlay_does_not_touch_input():
    cfg = {"runtime": {"rtc_guidance_horizon": 40, "rtc_schedule": "EXP", "rtc_max_guidance_weight": 10.0, "delay_steps": 4}}
    c2 = D.overlay_rtc(cfg, "range10_cap5")
    assert cfg["runtime"]["rtc_guidance_horizon"] == 40
    assert (c2["runtime"]["rtc_guidance_horizon"], c2["runtime"]["rtc_schedule"], c2["runtime"]["rtc_max_guidance_weight"]) == (10, "EXP", 5.0)
    assert D.overlay_rtc(cfg, "naive") == cfg
    rv = D.runtime_values(D.overlay_rtc(cfg, "ZEROS"), "ZEROS")
    assert rv["rtc_schedule"] == "ZEROS" and rv["rtc_guidance_horizon"] == 44 and rv["delay_steps_init"] == 4


# ------------------------------------------------------------------ 重み（論文の式 5 と lerobot）
@pytest.mark.parametrize("d", [1, 2, 3, 4, 5])
def test_paper_formula_equals_eq5(d):
    assert np.allclose(D.prefix_weights("EXP", d, 44), D.paper_weights(d, 50, 6), atol=1e-12)


@pytest.mark.parametrize("name", ["current_repro", "paper_formula_range44_cap5", "range40_cap5", "range10_cap5", "ZEROS"])
@pytest.mark.parametrize("d", [3, 4, 5])
def test_weights_equal_lerobot(name, d):
    from lerobot.configs import RTCAttentionSchedule
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.policies.rtc.modeling_rtc import RTCProcessor
    s = D.SETTINGS[name]
    proc = RTCProcessor(RTCConfig(prefix_attention_schedule=RTCAttentionSchedule[s["schedule"]],
                                  max_guidance_weight=s["max_guidance_weight"], execution_horizon=s["horizon"]))
    ref = proc.get_prefix_weights(d, s["horizon"], 50).numpy()
    assert np.allclose(D.prefix_weights(s["schedule"], d, s["horizon"]), ref, atol=1e-6)


def test_zeros_weights_only_first_d_rows():
    w = D.prefix_weights("ZEROS", 4, 44)
    assert w[:4].tolist() == [1, 1, 1, 1] and not w[4:].any()


# ------------------------------------------------------------------ 影の推論
class _FakePolicy:
    """infer は雑音そのもの（RTC の引数があれば +1）を返す。last_raw も同じ。"""

    def __init__(self):
        self.last_raw = None
        self.calls = 0

    def infer(self, obs, gen, rtc):
        import torch
        self.calls += 1
        x = torch.randn((50, 7), generator=gen).numpy().astype(np.float64)
        if rtc is not None:
            x = x + 1.0
        self.last_raw = np.concatenate([x, np.zeros((50, 25))], axis=1).astype(np.float32)
        return x

    def start_trial(self, seed):
        self.started = seed


def test_shadow_uses_same_noise_and_keeps_guided():
    import torch
    inner = _FakePolicy()
    p = D.DiagPolicy(inner, shadow=True)
    g = torch.Generator().manual_seed(7)
    out0 = p.infer({}, g, None)                          # 最初の推論（RTC の引数なし）は影を足さない
    g = torch.Generator().manual_seed(8)
    rtc = {"inference_delay": 4, "prev_chunk_left_over": np.zeros((44, 32))}
    out1 = p.infer({}, g, rtc)
    a = p.arrays()
    assert inner.calls == 3
    assert a["has_rtc"].tolist() == [False, True] and a["has_shadow"].tolist() == [False, True]
    assert np.allclose(a["shadow_post"][1] + 1.0, a["guided_post"][1], atol=1e-6)   # 同じ雑音・引数だけ違う
    assert np.allclose(a["guided_post"][1], out1[:, :7], atol=1e-6)
    assert np.allclose(inner.last_raw[:, :7], out1[:, :7], atol=1e-6)               # 次の推論の前の塊は誘導した側
    assert np.isnan(a["shadow_post"][0]).all() and a["inference_delay"].tolist() == [-1, 4]
    assert a["left_over_rows"].tolist() == [0, 44]
    p.start_trial(3)                                     # ほかの口は中へ渡る
    assert inner.started == 3


def test_no_shadow_records_guided_only():
    import torch
    inner = _FakePolicy()
    p = D.DiagPolicy(inner, shadow=False)
    p.infer({}, torch.Generator().manual_seed(1), {"inference_delay": 3, "prev_chunk_left_over": np.zeros((10, 32))})
    assert inner.calls == 1 and not p.arrays()["has_shadow"].any()


# ------------------------------------------------------------------ 指標（人工の記録）
def _synthetic(close_t=6.0, gap=-0.02, success_t=12.0, n_f=401):
    """目標（赤）は (0.5, 0, 0.02)。指先は (0.3, 0, 0.2) から x だけ進み、close_t に 0.5 + gap で閉じる。"""
    t = np.arange(n_f) * 0.05
    cube = np.zeros((n_f, 3, 3))
    cube[:, 0] = [0.5, 0.0, 0.02]
    cube[:, 1] = [0.3, 0.2, 0.02]
    cube[:, 2] = [0.3, -0.2, 0.02]
    tip = np.zeros((n_f, 3))
    x_end = 0.5 + gap
    tip[:, 0] = np.clip(0.3 + (x_end - 0.3) * t / close_t, 0.3, x_end)
    tip[:, 2] = np.clip(0.2 - 0.17 * t / close_t, 0.03, 0.2)
    closed = t >= close_t
    z = {"sim_time": t, "fingertip": tip, "cube_pos": cube, "gripper_closed": closed, "x_des": tip.copy()}
    # 行動（10 Hz）: 閉じるまで一様に x へ進む（指令の合計 = 実際の移動）。6 行ごとに塊が替わり、継ぎ目で 0.01 m/s 跳ぶ
    n_a = int(20 * 10)
    acts = []
    step = (x_end - 0.3) / (close_t * 10)
    for k in range(n_a):
        tk = k * 0.1
        if tk < close_t:
            a = [step + (0.001 if (k // 6) % 2 else 0.0), 0, 0, 0, 0, 0, -1]
        else:
            a = [0, 0, 0, 0, 0, 0, 1]
        acts.append({"k": k, "t": tk, "chunk": k // 6, "held": k == 0 or tk >= close_t, "a": a})   # 閉じた後は保持（継ぎ目にしない）
    rt = {"actions": acts, "inference": []}
    meta = {"trial": 0, "seed": 1, "target": "red", "success": success_t is not None, "t_success": success_t}
    return meta, z, rt


def test_trial_metrics_synthetic():
    meta, z, rt = _synthetic()
    r = D.trial_metrics(meta, z, rt, horizon_s=30.0)
    assert r["radial_gap_mm"] == pytest.approx(-20.0, abs=1e-6)
    assert r["move_ratio"] == pytest.approx(1.0, abs=0.06)
    assert r["seam_jump_mps"] == pytest.approx(0.01, abs=1e-9)           # 継ぎ目の跳びは 0.001 m / 0.1 s
    assert r["success_at"] == {"30": True, "45": True, "60": True}
    assert r["first_close_after_horizon"] is False


def test_trial_metrics_horizon_and_no_close():
    meta, z, rt = _synthetic(close_t=35.0, success_t=50.0, n_f=1201)
    r30 = D.trial_metrics(meta, z, rt, horizon_s=30.0)
    rall = D.trial_metrics(meta, z, rt, horizon_s=None)
    assert r30["first_close_after_horizon"] is True and r30["move_ratio"] is None
    assert rall["move_ratio"] is not None and r30["radial_gap_mm"] == rall["radial_gap_mm"]
    assert r30["success_at"] == {"30": False, "45": False, "60": True}
    z["gripper_closed"][:] = False
    rn = D.trial_metrics(meta, z, rt, horizon_s=30.0)
    assert rn["t_first_close"] is None and rn["radial_gap_mm"] is None
    s = D.summarize([r30, rall, rn])
    assert s["n_no_close"] == 1 and s["n_first_close_after_horizon"] == 1 and s["successes_at"]["60"] == 3


def test_shadow_trial_synthetic():
    meta, z, rt = _synthetic()
    n = 3
    g = np.zeros((n, 50, 7), np.float32)
    s = np.zeros((n, 50, 7), np.float32)
    g[:, :, 0] = 0.001                                    # 誘導: 1 行 1 mm
    s[:, :, 0] = 0.002                                    # 影: 1 行 2 mm（目標の向き = +x）
    diag = {"has_rtc": np.array([False, True, True]), "has_shadow": np.array([False, True, True]),
            "guided_post": g, "shadow_post": s}
    rt["inference"] = [{"t_obs": 0.0, "offset": 0}, {"t_obs": 1.0, "offset": 4}, {"t_obs": 8.0, "offset": 4}]   # 3 本目は閉じた後
    tr = D.shadow_trial(meta, z, rt, diag)
    assert tr["n_used"] == 1
    assert tr["h"]["10"] == [pytest.approx(10.0, abs=1e-3)] and tr["h"]["40"] == [pytest.approx(40.0, abs=1e-3)]
    assert tr["exec_rows"] == [pytest.approx(6.0, abs=1e-3)]
    sm = D.shadow_summary([tr])
    assert sm["plan_shorter_ge_15mm"] is True
    with pytest.raises(ValueError):
        D.shadow_trial(meta, z, {**rt, "inference": rt["inference"][:2]}, diag)


def test_x2_shortfall_synthetic():
    # エキスパートは 20 Hz の生のこまで x に 1 こま 1 mm（10 fps の 1 行 2 mm）進み、生のこま 200 で閉じる。立方体は (0.5, 0)
    f = 401
    xd = np.zeros((f, 3))
    xd[:, 0] = 0.2 + 0.001 * np.minimum(np.arange(f), 200)
    cube = np.tile([0.5, 0.0], (f, 1))
    ks = np.array([0, 10, 50])
    pred = np.zeros((3, 50, 7))
    pred[:, :, 0] = 0.0015                                # 予測は 1 行 1.5 mm（短い）
    sf = D.x2_episode_shortfall(pred, ks, xd, cube, f_close=200)
    assert sf["10"] == [pytest.approx(5.0)] * 3          # 10 行: エキスパート 20 mm − 予測 15 mm
    assert len(sf["40"]) == 3 and sf["40"][0] == pytest.approx(20.0)
    assert len(D.x2_episode_shortfall(pred, np.array([70]), xd, cube, f_close=200)["40"]) == 0   # 70 + 40 > 100 は使わない
    assert D.x2_summary([sf])["shortfall_ge_15mm"] is True


# ------------------------------------------------------------------ 96 を包む
def test_patch96_files_and_spec(drtc, tmp_path):
    m = _load(ROOT / "scripts" / "96_s4_resume.py", "s4_resume_testA")
    drtc.patch96(m, "ZEROS", False, "a" * 64, "b" * 64)
    ps = m.trial_paths("run", tmp_path, 3)
    assert ps["diag"] == tmp_path / "diag_0003.npz" and set(ps) == {"json", "npz", "runtime", "diag"}
    assert "diag" not in m.trial_paths("task", tmp_path, 3)
    a = m.build_parser().parse_args(drtc.argv96(type("A", (), {"experiment": "S4SMOKE_A"})(), "ZEROS", "ZEROS",
                                                "natural:44430:1", ["--dry-run"]))
    assert (a.mode, a.exec_interval, a.no_safety, a.time_limit_s, a.model) == ("rtc", 6, True, 60.0, "R1v3")
    spec = m.make_spec(a)
    assert spec["diag_rtc"]["setting"]["schedule"] == "ZEROS" and spec["diag_rtc"]["shadow"] is False
    cfg = m.overlay_limits({"eval": {"time_limit_s": 30}, "planner": {}, "runtime": {"rtc_guidance_horizon": 40, "rtc_schedule": "EXP",
                                                                                     "rtc_max_guidance_weight": 10.0}},
                           {"time_limit_s": 60.0})
    assert cfg["eval"]["time_limit_s"] == 60.0 and cfg["runtime"]["rtc_schedule"] == "ZEROS"
    # 包み直しても元の関数から包む（二重に包まない）
    drtc.patch96(m, "naive", False, "a" * 64, "b" * 64)
    assert m.overlay_limits({"eval": {}, "planner": {}, "runtime": {"rtc_schedule": "EXP"}}, {"time_limit_s": 60.0})["runtime"]["rtc_schedule"] == "EXP"


def test_check_complete_requires_diag(drtc, tmp_path):
    m = _load(ROOT / "scripts" / "96_s4_resume.py", "s4_resume_testA2")
    drtc.patch96(m, "range10_cap5", False, "a" * 64, "b" * 64)
    meta = {"success": True, "seed": 44430, "target": "red", "trial": 0, "audit": {},
            "diag": {"arm": "range10_cap5", "shadow": False}}
    (tmp_path / "trial_0000.json").write_text(json.dumps(meta), encoding="utf-8")
    (tmp_path / "runtime_0000.json").write_text(json.dumps({"runtime": {}}), encoding="utf-8")
    np.savez(tmp_path / "trial_0000.npz", a=np.zeros(3))
    ok, why, _ = m.check_complete("run", tmp_path, 0, {"seed": 44430, "target": "red"})
    assert not ok and why.startswith("missing:diag")
    np.savez(tmp_path / "diag_0000.npz", has_rtc=np.zeros(2, bool))
    ok, why, _ = m.check_complete("run", tmp_path, 0, {"seed": 44430, "target": "red"})
    assert ok, why
    drtc.patch96(m, "ZEROS", False, "a" * 64, "b" * 64)          # 別の設定の記録は完全とみなさない（混ぜない）
    ok, why, _ = m.check_complete("run", tmp_path, 0, {"seed": 44430, "target": "red"})
    assert not ok and why.startswith("diag_meta")


def test_band_check(drtc):
    drtc.check_band("natural:190200:10", False, False)
    drtc.check_band("natural:190220:4", True, False)
    drtc.check_band("natural:44430:1", False, False)
    with pytest.raises(SystemExit):
        drtc.check_band("natural:190220:4", False, False)        # 影の帯を 6 設定に使わない
    with pytest.raises(SystemExit):
        drtc.check_band("natural:140000:1", False, False)
    with pytest.raises(SystemExit):
        drtc.check_band("induced:190200:1", False, False)


def test_x2_specs_one_per_seed():
    x2 = _load(ROOT / "scripts" / "98_s4_x2.py", "s4_x2_testA")
    specs = x2.specs_for([44404, 44405, 44406])
    assert [s.layout_seed for s in specs] == [44404, 44405, 44406]
    assert all(s.kind == "n" and s.layout_kind == "empty" for s in specs)
    assert x2.x2_band() == (44404, 44423)


# ------------------------------------------------------------------ settings の掲示（回す前に掲示する定義の文）
def test_settings_definitions_before_run(drtc):
    d = drtc.definitions_before_run()
    mr = d["move_ratio_30s"]
    assert "30 s 以内" in mr["30s_version"] and "主は中央値" in mr["aggregate"] and "20 mm" in mr["definition"]
    assert "同じ観測・同じ雑音" in d["shadow"]["compare_with"] and "15 mm" in d["shadow"]["h_rule"]
    assert "10・20・30・40" in d["shadow"]["metric"]
    z = d["ZEROS"]
    assert "E = 44" in z["contents"] and "β = 5" in z["contents"]
    assert z["weights_rows0_7"]["d=4"] == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert D.SETTINGS["ZEROS"]["horizon"] == 44 and D.SETTINGS["ZEROS"]["max_guidance_weight"] == 5.0


# ------------------------------------------------------------------ X2: 環境の照合・続きから・progress.json（GPU・生成なし）
ENV0 = {"driver": "610.88", "torch": "2.11.0+cu126", "torch_cuda": "12.6", "os_build": "26300.9457", "git_head": "x"}


@pytest.fixture()
def x2env(tmp_path, monkeypatch):
    """98_s4_x2.py の gen を、偽の generate・偽の環境・偽のメモリで回す道具。"""
    import types
    x2 = _load(ROOT / "scripts" / "98_s4_x2.py", "s4_x2_testA_resume")
    m96 = x2.load96()
    state = {"env": dict(ENV0), "fail_at": None, "calls": []}
    ops = types.SimpleNamespace(memory_gb=lambda: {"phys_free_gb": 20.0, "commit_free_gb": 30.0},
                                LIVE_STATUS=("starting", "running", "loading", "memory_wait", "quiet_wait"),
                                _alive=lambda pid, ct=None: False)
    monkeypatch.setattr(m96, "load_ops", lambda: ops)
    monkeypatch.setattr(m96, "read_env", lambda ops_: dict(state["env"]))
    monkeypatch.setattr(x2, "X2", tmp_path)
    monkeypatch.setattr(x2, "GLOBAL_STOP", tmp_path / "STOP_GLOBAL")
    from recovla.expert import generate as G

    def fake_generate(specs, run_dir, workers=1, render=True, rig_kind="v1", **kw):
        s = specs[0]
        run_dir = pathlib.Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=False)          # 本物と同じく新しいフォルダにだけ書く
        state["calls"].append(s.layout_seed)
        if state["fail_at"] == s.layout_seed:
            (run_dir / "run.json").write_text("{}", encoding="utf-8")
            raise RuntimeError("途中で切れた（試験）")
        name = f"n_{s.layout_seed}_{s.color}_r0"
        ep = run_dir / name
        ep.mkdir()
        (ep / "meta.json").write_text("{}", encoding="utf-8")
        np.savez(ep / "data.npz", x=np.zeros(3))
        r = {"layout_seed": s.layout_seed, "color": s.color, "success": True, "retries": 0,
             "attempts": [{"name": name, "path": str(ep), "success": True, "failure": None, "duration_s": 1.0,
                           "layout": {"start": "home"}}]}
        (run_dir / "generation.jsonl").write_text(json.dumps(r) + "\n", encoding="utf-8")
        (run_dir / "timing.json").write_text(json.dumps({"wall_s": 1.0}), encoding="utf-8")
        return [r]
    monkeypatch.setattr(G, "generate", fake_generate)
    return x2, state, tmp_path


def test_x2_gen_resumes_and_quarantines(x2env):
    x2, state, root = x2env
    run_dir = root / "gen" / "T"
    args = ["gen", "--seeds", "44430:4", "--name", "T"]
    assert x2.main(args + ["--dry-run"]) == 0 and not run_dir.exists()          # dry-run は何も書かない
    state["fail_at"] = 44432
    assert x2.main(args) == 2                                                     # 3 本目で切れた（エラー）
    prog = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
    for k in ("status", "pid", "proc_create_time", "updated", "done", "total", "stop_reason", "error", "env"):
        assert k in prog, k
    assert prog["status"] == "error" and prog["done"] == 2 and prog["total"] == 4
    assert not (run_dir / "gen_summary.json").exists()
    state["fail_at"], state["calls"] = None, []
    state["env"] = dict(ENV0, driver="999.99")
    assert x2.main(args) == 3                                                     # 環境が前の回と違えば止める
    assert state["calls"] == []
    assert x2.main(args + ["--dry-run"]) == 3
    state["env"] = dict(ENV0)
    assert x2.main(args) == 0                                                     # 続きから: 完全な 2 本は飛ばす
    assert state["calls"] == [44432, 44433]
    inc = [p for p in run_dir.iterdir() if p.name.startswith("_incomplete_")]
    assert len(inc) == 1 and (inc[0] / "part_44432" / "run.json").is_file()       # 書きかけは退避
    summ = json.loads((run_dir / "gen_summary.json").read_text(encoding="utf-8"))
    assert summ["n_saved"] == 4 and [e["seed"] for e in summ["episodes"]] == [44430, 44431, 44432, 44433]
    assert all(e["path"].endswith(f"part_{e['seed']}\\{e['name']}") or e["path"].endswith(f"part_{e['seed']}/{e['name']}")
               for e in summ["episodes"])
    assert json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))["status"] == "done"
    log = json.loads((run_dir / "gen_log.json").read_text(encoding="utf-8"))
    assert [s["status"] for s in log["sessions"]] == ["error", "done"] and log["sessions"][1]["ran_seeds"] == [44432, 44433]
    state["calls"] = []
    assert x2.main(args) == 0 and state["calls"] == []                            # 済み


def test_x2_part_state(tmp_path):
    x2 = _load(ROOT / "scripts" / "98_s4_x2.py", "s4_x2_testA_part")
    assert x2.part_state(tmp_path, 44430) == (False, "missing", None)
    d = tmp_path / "part_44430"
    d.mkdir()
    (d / "generation.jsonl").write_text(json.dumps({"layout_seed": 44430, "success": False, "attempts": [{"name": "a"}]}) + "\n",
                                        encoding="utf-8")
    assert x2.part_state(tmp_path, 44430)[1] == "no_timing"
    (d / "timing.json").write_text("{}", encoding="utf-8")
    assert x2.part_state(tmp_path, 44430)[0] is True                               # 失敗の種も完全（作り直しの上限まで試した）
    (d / "generation.jsonl").write_text(json.dumps({"layout_seed": 44430, "success": True, "attempts": [{"name": "ep"}]}) + "\n",
                                        encoding="utf-8")
    assert x2.part_state(tmp_path, 44430)[1] == "episode_files"                    # エピソードのファイルが無い
