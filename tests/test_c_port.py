"""Step C の完了条件（手順書 §4）と、移植で動作を変えていないことの検査。

前半（速い）: 実行環境が固定一覧どおりか、場面・制御器の値・学習配置が流用元と同じか、C:\\VLA・台帳への
結合が残っていないか。
後半: 完了条件 1〜4。重い処理は scripts/02_g0_check.py（1〜3）と scripts/03_vla_listing.ps1（4）が行い、
outputs/g0/*.json に結果を書く。ここはその結果を読んで判定する（無ければ、何を実行すればよいかを示して落ちる）。
"""
import hashlib
import json
import pathlib
import re
import subprocess
import sys

import mujoco
import numpy as np
import pytest

from recovla.common import config
from recovla.record import recorder
from recovla.sim import control

ROOT = config.ROOT
CFG = config.load("g0")
G0 = CFG["g0"]
OUT = config.path(CFG["paths"]["outputs"]) / "g0"
LEGACY_RAW = config.path(G0["raw_episode"])
LEGACY_EVAL = config.path(G0["legacy_eval"])

# 流用元の teleop_scene.xml（panda.xml と同じフォルダ）をコンパイルした mj_saveModel のバイト列の SHA-256
# （2026-09-25、mujoco 3.2.3 で計測。取り込みの前に .cache\import の流用元の XML から直接求めた値）
SOURCE_SCENE_MJB_SHA256 = "9a328745ee468fe73bfa8be2a4b10c2dd8af57e3b1c778e669fdbdce60f7563f"


def _need(path: pathlib.Path, how: str) -> dict:
    if not path.is_file():
        pytest.fail(f"{path} がない。先に {how} を実行する")
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------ 移植の検査

def test_environment_matches_the_lock():
    uv = ROOT / ".tools" / "uv" / "uv.exe"
    freeze = subprocess.run([str(uv), "pip", "freeze", "--python", sys.executable],
                            capture_output=True, text=True, check=True).stdout.splitlines()
    got = sorted(line for line in freeze if line and not line.startswith("-e ") and not line.startswith("recovla"))
    lock = sorted(line for line in (ROOT / "env" / "requirements-lock.txt").read_text(encoding="utf-8").splitlines()
                  if line and not line.startswith("#"))
    assert got == lock


def test_scene_compiles_to_the_same_model_as_the_source():
    m = mujoco.MjModel.from_xml_path(control.SCENE_PATH)
    buf = np.zeros(mujoco.mj_sizeModel(m), dtype=np.uint8)
    mujoco.mj_saveModel(m, None, buf)
    assert hashlib.sha256(buf.tobytes()).hexdigest() == SOURCE_SCENE_MJB_SHA256
    control.check_timestep(m)


def test_controller_and_recording_values_equal_the_legacy_raw_meta():
    """設定ファイルから作った制御器の調整値が、予備実験の raw（流用元が meta.json に書いた値）と同じ。"""
    meta = json.loads((LEGACY_RAW / "meta.json").read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(control.SCENE_PATH)
    data = mujoco.MjData(model)
    c = control.make_collect_controller(model, data)
    for key, want in meta["controller"].items():
        if key == "class":
            assert type(c).__name__ == want
            continue
        got = getattr(c, key)
        assert (list(got) if isinstance(got, tuple) else got) == want, key
    assert c.ctrl_dt == meta["timestep"] == float(model.opt.timestep)
    assert list(control.START_POS) == meta["start_pos"]
    assert control.RECORD_EVERY == meta["record_every"]
    assert list(control.CAMERAS) == meta["cameras"]["names"] and control.IMAGE_SIZE == meta["cameras"]["width"]
    assert recorder.INSTRUCTION == meta["instruction"]


def test_training_placements_equal_the_legacy_evaluation():
    """学習配置（種 0〜2999 から 10 個）が、予備実験の評価の記録と完全に同じ（G0 の比較の前提）。"""
    placements = {p.placement_id: p for p in recorder.training_placements()}
    trials = sorted(LEGACY_EVAL.glob("trial_*.json"))
    assert len(trials) == 20
    for path in trials:
        t = json.loads(path.read_text(encoding="utf-8"))
        want = t["placement"]
        p = placements[want["placement_id"]]
        assert (p.seed, p.x, p.y, float(np.degrees(p.yaw))) == (want["placement_seed"], want["x"], want["y"],
                                                               want["yaw_deg"]), path.name


def test_no_ledger_and_no_vla_paths_in_the_code():
    """台帳との結合と、C:\\VLA のパスの直書きが残っていない（説明文の中の言及は除く）。"""
    code_re = re.compile(r"(?i)^\s*(from|import)\s+\S*ledger|C:\\\\VLA|parents\[\d\]\s*/\s*\"0\d_")
    for p in (ROOT / "src").rglob("*.py"):
        hits = [ln for ln in p.read_text(encoding="utf-8").splitlines() if code_re.search(ln)]
        assert not hits, (p, hits)
    for p in (ROOT / "configs").glob("*.yaml"):
        assert "VLA\\" not in p.read_text(encoding="utf-8") and "C:/VLA" not in p.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- 完了条件 1〜4

def test_condition_1_offline_start_model_checkpoint_and_training():
    r = _need(OUT / "offline_check.json", "scripts/02_g0_check.py offline")
    assert r["HF_HUB_OFFLINE"] == "1"
    assert r["start_model"]["ok"] and r["checkpoint"]["ok"]
    assert r["hf_home_vla_paths"] == []
    t = _need(OUT / "train10_check.json", "scripts/02_g0_check.py train10")
    assert t["verify_ok"] and t["dry_run_exit"] == 0 and t["train_exit"] == 0 and t["checkpoint_exists"]
    assert t["log_summary"]["last_step_logged"] == 10
    # B_提案書 §10 の仕組み（meta/stats.json から読まれる、--policy.path で上書きされる、学習率の上書き）
    assert t["stats_ok"], t["stats_bit_equal_to_replaced"]
    assert t["lr_peak_ok"], t["logged_lr"]


def test_condition_2_replay_of_the_copied_raw_episode():
    r = _need(OUT / "replay_check.json", "scripts/02_g0_check.py replay")
    assert r["modes"]["window"]["ee_err_max_m"] < float(G0["replay_tol_m"])
    assert r["negative_controls_detected"]


def test_condition_3_g0_closed_loop():
    r = _need(OUT / "eval_check.json", "scripts/02_g0_check.py eval")
    assert r["trials"] == int(G0["repeats"]) * len(G0["placement_ids"]) == 20
    assert r["all_same_seed_and_placement"]
    assert [row["seed"] for row in r["rows"]] == list(range(G0["first_seed"], G0["first_seed"] + 20))
    assert r["all_input_check_ok"]
    assert r["successes_new"] >= int(G0["min_successes"])
    assert r["inference"]["n"] > 0 and r["inference"]["mean_s_excluding_first"] is not None


def test_condition_4_vla_listing_unchanged():
    r = _need(OUT / "vla_listing_check.json", "scripts/03_vla_listing.ps1")
    assert r["count_before"] == r["count_now"] and r["only_before"] == 0 and r["only_now"] == 0 and r["ok"]
