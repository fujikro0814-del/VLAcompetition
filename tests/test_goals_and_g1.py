"""目標書の版の照合と G1 の境界の検査（0106）。"""
import importlib.util
import pathlib

import pytest

from recovla.common import goals

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _g1():
    spec = importlib.util.spec_from_file_location("check_g1_boundary", ROOT / "scripts" / "check_g1_boundary.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_goals_fingerprint_matches_tag():
    fp = goals.fingerprint()
    assert fp["version"] >= 2 and fp["tag"] == f"goals-v{fp['version']}" and len(fp["sha256"]) == 64


def test_goals_rejects_edited_copy(tmp_path, monkeypatch):
    real = (ROOT / goals.GOALS_PATH).read_bytes()
    monkeypatch.setattr(goals, "_blob_at", lambda root, tag: real)
    monkeypatch.setattr(goals, "_tags", lambda root: ["goals-v2"])
    (tmp_path / "docs").mkdir()
    (tmp_path / goals.GOALS_PATH).write_bytes(real + b"\n")
    with pytest.raises(goals.GoalsMismatch):
        goals.fingerprint(tmp_path)
    (tmp_path / goals.GOALS_PATH).write_bytes(real)
    assert goals.fingerprint(tmp_path)["version"] == 2
    monkeypatch.setattr(goals, "_tags", lambda root: ["goals-v3"])     # 変更履歴に v3 の行がない
    with pytest.raises(goals.GoalsMismatch):
        goals.fingerprint(tmp_path)


def test_g1_scan_finds_truth_access():
    src = ("from recovla.sim import rig as R\n"
           "import recovla.expert.script\n"
           "def f(rig, frame, cfg):\n"
           "    a = rig.data.qpos[3]\n"
           "    b = frame['cube_pos']\n"
           "    c = cfg['scene']['box']\n"
           "    return rig.controller.gripper_closed\n")
    hits = _g1().scan(src)
    for k in ("import:recovla.sim.rig", "import:recovla.expert.script", "attr:qpos", "key:cube_pos", "key:box",
              "attr:controller", "attr:gripper_closed", "name:rig"):
        assert hits[k] >= 1, k


def test_g1_scan_allows_sensor_only_code():
    src = ("def f(sensor, setup):\n"
           "    return sensor.q, sensor.gripper_width, setup.box_size\n")
    assert not _g1().scan(src)


def test_g1_no_new_violations_in_work_tree():
    assert _g1().main([]) == 0
