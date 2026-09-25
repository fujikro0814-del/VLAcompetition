"""Tests for convert_to_lerobot.py (C5). Needs the LeRobot environment:

    C:\\VLA\\02_環境\\lerobot\\.venv\\Scripts\\python.exe -m pytest tests/test_convert_to_lerobot.py

Skipped under the teleop python311 (no lerobot there). Builds a synthetic raw
episode in tmp_path, so it never touches 03_収録.
"""
import json
import pathlib

import numpy as np
import pytest

pytest.importorskip("lerobot")
from PIL import Image  # noqa: E402

import convert_to_lerobot as conv  # noqa: E402


def _axis_quat(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])


def make_raw_episode(root: pathlib.Path, episode_id=0, n=21, size=16):
    ep = root / "2026-09-16" / f"ep_{episode_id:06d}"
    rng = np.random.default_rng(episode_id)
    for cam in ("overhead", "wrist"):
        (ep / cam).mkdir(parents=True)
        for i in range(n):
            img = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
            Image.fromarray(img).save(ep / cam / f"{i:06d}.png")
    x_des = np.cumsum(rng.normal(0, 0.003, (n, 3)), axis=0) + [0.3, 0.0, 0.3]
    quats = np.array([conv.quat_mul(_axis_quat([1, 0, 0], a)[None], conv.Q_DOWN)[0]
                      for a in rng.normal(0, 0.002, n)])
    quats[::3] *= -1.0          # sign flips must not matter
    closed = np.zeros(n, bool)
    closed[8:15] = True
    data = {
        "ee_pos": x_des + 0.004, "ee_quat": quats,
        "fingers": np.stack([np.linspace(0.04, 0.02, n)] * 2, axis=1),
        "joints": rng.normal(0, 1, (n, 7)), "x_des": x_des, "gripper_closed": closed,
    }
    np.savez(ep / "data.npz", **data)
    meta = {"episode_id": episode_id, "n_frames": n, "record_hz": 20, "instruction": "put the cube in the box",
            "cameras": {"width": size}, "placement_id": 1, "operator": "01", "session": 1,
            "success": True, "retakes": 0}
    (ep / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return ep, meta, data


def test_episode_arrays_definitions(tmp_path):
    ep, meta, data = make_raw_episode(tmp_path, n=21)
    arr = conv.episode_arrays(meta, data)
    assert arr["raw_index"].tolist() == list(range(0, 20, 2))            # (21-1)//2 = 10
    a, s = arr["action"], arr["state"]
    assert a.dtype == np.float32 and s.dtype == np.float32
    assert np.allclose(a[:, :3], data["x_des"][2:21:2] - data["x_des"][0:20:2], atol=1e-7)
    assert np.all(a[:, 3:6] == 0)
    assert a[:, 6].tolist() == [1.0 if c else -1.0 for c in data["gripper_closed"][2:21:2]]
    assert np.allclose(s[:, :3], data["ee_pos"][0:20:2], atol=1e-6)
    assert np.allclose(s[:, 6], data["fingers"][0:20:2, 0]) and np.all(s[:, 7] <= 0)
    assert np.abs(s[:, 3:6]).max() < 0.01                                  # near-down, no pi


def test_orientation_deviation():
    assert np.allclose(conv.orientation_deviation(conv.Q_DOWN[None]), 0.0)
    tilt = conv.quat_mul(_axis_quat([0, 1, 0], 0.1)[None], conv.Q_DOWN)
    assert np.allclose(conv.orientation_deviation(tilt), [[0, 0.1, 0]])
    assert np.allclose(conv.orientation_deviation(-tilt), [[0, 0.1, 0]])


def test_convert_and_verify(tmp_path):
    raw = tmp_path / "03_収録" / "raw"
    for eid in (0, 1):
        make_raw_episode(raw, episode_id=eid, n=21 + 4 * eid)
    out = tmp_path / "03_収録" / "lerobot" / "unit"
    episodes = conv.raw_episodes(raw)
    conv.convert(episodes, out, "unit")
    assert conv.verify(out, tmp_path / "exports") is True
    info = json.loads((out / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["fps"] == 10 and info["codebase_version"] == "v3.0"
    assert info["features"]["observation.state"]["shape"] == [15]
    assert info["features"]["observation.images.image"]["dtype"] == "image"
    assert sorted(p.name for p in (tmp_path / "exports").iterdir()) == [
        "ep_000000_lerobot.npz", "ep_000001_lerobot.npz"]
    with pytest.raises(FileExistsError):
        conv.convert(episodes, out, "unit")


# ------------------------------------------------------------------- manifests

def make_labelled_raw(tmp_path, ids=(0, 1, 2), results=None, label_ids=None):
    """A raw folder plus the labels next to a manifests folder, as 03_収録 is laid out."""
    root = tmp_path / "03_収録"
    raw = root / "raw"
    for i in ids:
        make_raw_episode(raw, episode_id=i, n=21 + 4 * i)
    labels = root / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    results = results or {}
    with open(labels / "2026-09-16.jsonl", "w", encoding="utf-8") as f:
        for i in (label_ids if label_ids is not None else ids):
            f.write(json.dumps({"episode_id": i, "at": "2026-09-16T10:00:00", "by": "01",
                                "vocab_version": 1, "result": results.get(i, "成功"),
                                "stage": None, "recovery": "なし", "manner": [], "defects": [],
                                "reason": ""}, ensure_ascii=False) + "\n")
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    return root, raw


def write_manifest(root, name, episodes, **extra):
    record = {"name": name, "created": "2026-09-16", "base_commit": "a3f91c2",
              "rule": "動作確認", "episodes": list(episodes), "n": len(episodes),
              "teacher_frames": 999, "composition": {"total": len(episodes)}}
    record.update(extra)
    path = root / "manifests" / f"{name}.json"
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    return path


def test_manifest_picks_the_listed_episodes_in_order(tmp_path):
    root, raw = make_labelled_raw(tmp_path)
    path = write_manifest(root, "A", [2, 0])
    episodes, manifest = conv.episodes_from_manifest(path, raw_root=raw)
    assert [p.name for p in episodes] == ["ep_000002", "ep_000000"]
    assert manifest["name"] == "A"


def test_manifest_stops_when_a_recording_is_missing(tmp_path):
    root, raw = make_labelled_raw(tmp_path, ids=(0, 1))
    path = write_manifest(root, "A", [0, 7, 9])
    with pytest.raises(conv.ManifestError) as e:
        conv.episodes_from_manifest(path, raw_root=raw)
    assert "[7, 9]" in str(e.value)                    # 該当する番号を挙げる


def test_manifest_stops_on_an_episode_that_did_not_achieve_the_task(tmp_path):
    root, raw = make_labelled_raw(tmp_path, results={1: "失敗"})
    path = write_manifest(root, "A", [0, 1, 2])
    with pytest.raises(conv.ManifestError) as e:
        conv.episodes_from_manifest(path, raw_root=raw)
    assert "[1]" in str(e.value) and "失敗" in str(e.value)


def test_manifest_stops_on_an_unlabelled_episode(tmp_path):
    root, raw = make_labelled_raw(tmp_path, label_ids=(0, 2))
    path = write_manifest(root, "A", [0, 1, 2])
    with pytest.raises(conv.ManifestError) as e:
        conv.episodes_from_manifest(path, raw_root=raw)
    assert "[1]" in str(e.value) and "no label" in str(e.value)


def test_manifest_is_recorded_in_conversion_json(tmp_path):
    root, raw = make_labelled_raw(tmp_path, ids=(0, 1))
    path = write_manifest(root, "A_all_valid", [0, 1])
    episodes, manifest = conv.episodes_from_manifest(path, raw_root=raw)
    out = tmp_path / "ds"
    conv.convert(episodes, out, "unit_manifest", conv.manifest_record(path, manifest))
    record = json.loads((out / "meta" / "conversion.json").read_text(encoding="utf-8"))["manifest"]
    assert record["name"] == "A_all_valid" and record["n"] == 2
    assert record["base_commit"] == "a3f91c2" and record["rule"] == "動作確認"
    assert record["composition"] == {"total": 2}          # 中身を見にファイルを探しにいかなくてよい
    assert record["sha256"] == conv.sha256(path)


def test_normalisation_statistics_come_from_the_chosen_set(tmp_path):
    """同じ生データでも、選んだ集合が違えば統計量は違う（全件変換して学習で絞る方式を採らない理由）。"""
    root, raw = make_labelled_raw(tmp_path, ids=(0, 1, 2))
    stats = {}
    for name, ids in (("A", [0, 1]), ("B", [2])):
        path = write_manifest(root, name, ids)
        episodes, manifest = conv.episodes_from_manifest(path, raw_root=raw)
        out = tmp_path / f"ds_{name}"
        conv.convert(episodes, out, f"unit_{name}", conv.manifest_record(path, manifest))
        stats[name] = json.loads((out / "meta" / "stats.json").read_text(encoding="utf-8"))
    a = np.asarray(stats["A"]["observation.state"]["mean"], float)
    b = np.asarray(stats["B"]["observation.state"]["mean"], float)
    assert not np.allclose(a, b)
    assert stats["A"]["observation.state"]["count"] != stats["B"]["observation.state"]["count"]


def test_the_old_ways_of_calling_the_converter_still_work(tmp_path):
    """--manifest を足しても、フォルダを並べる／--raw-dir の呼び出しは変わらない。"""
    root, raw = make_labelled_raw(tmp_path, ids=(0, 1))
    listed = conv.main(["--out", str(tmp_path / "ds_list"), "--name", "list",
                        str(raw / "2026-09-16" / "ep_000000")])
    assert listed == 0
    record = json.loads((tmp_path / "ds_list" / "meta" / "conversion.json").read_text(encoding="utf-8"))
    assert "manifest" not in record and len(record["sources"]) == 1
    assert conv.main(["--out", str(tmp_path / "ds_dir"), "--name", "dir", "--raw-dir", str(raw)]) == 0
    record = json.loads((tmp_path / "ds_dir" / "meta" / "conversion.json").read_text(encoding="utf-8"))
    assert "manifest" not in record and len(record["sources"]) == 2


def test_manifest_and_raw_dir_cannot_be_given_together(tmp_path):
    root, raw = make_labelled_raw(tmp_path, ids=(0,))
    path = write_manifest(root, "A", [0])
    with pytest.raises(SystemExit):
        conv.main(["--out", str(tmp_path / "ds"), "--name", "x", "--manifest", str(path),
                   "--raw-dir", str(raw)])


def test_dataset_images_are_transformed_per_view_at_value_level(tmp_path):
    """overhead = raw mirrored left-right, wrist = raw upside down (Step E-C). Expected images use
    plain numpy slicing here, independent of vla_image_spec, and must differ from the old
    both-views-left-right behaviour."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    raw = tmp_path / "raw"
    ep, meta, data = make_raw_episode(raw, n=21)
    out = tmp_path / "ds"
    conv.convert([ep], out, "unit_views")
    ds = LeRobotDataset("local/unit_views", root=out)
    arr = conv.episode_arrays(meta, data)
    for k, i in enumerate(arr["raw_index"]):
        item = ds[k]
        for cam, key, want_fn in (("overhead", "observation.images.image", lambda a: a[:, ::-1]),
                                  ("wrist", "observation.images.image2", lambda a: a[::-1, :])):
            raw_img = np.asarray(Image.open(ep / cam / f"{i:06d}.png").convert("RGB"))
            got = np.round(item[key].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            assert np.array_equal(got, want_fn(raw_img)), (cam, k)
            if cam == "wrist":
                assert not np.array_equal(got, raw_img[:, ::-1]), "wrist still flipped left-right"
    conversion = json.loads((out / "meta" / "conversion.json").read_text(encoding="utf-8"))
    assert conversion["converter_version"] == 2
    assert conversion["image_spec_version"] == 2
    assert conversion["image_transforms"] == {"overhead": "fliplr", "wrist": "flipud"}
    assert conversion["image_keys"] == {"overhead": "observation.images.image",
                                        "wrist": "observation.images.image2"}


def test_verify_writes_raw_vs_policy_image_and_rejects_version1(tmp_path, capsys):
    raw = tmp_path / "raw"
    ep, _, _ = make_raw_episode(raw, n=21, size=16)
    out = tmp_path / "ds"
    conv.convert([ep], out, "unit_check")
    check_png = tmp_path / "check.png"
    assert conv.verify(out, None, check_png) is True
    img = np.asarray(Image.open(check_png))
    assert img.shape[1] == 2 * 16 and img.shape[0] >= 2 * 16          # 2 views x [raw | policy]
    default_png = out.parent / f"{out.name}_verify_raw_vs_policy.png"
    assert conv.verify(out, None) is True and default_png.is_file()
    # a version-1 record (no per-view spec) must fail verification
    cpath = out / "meta" / "conversion.json"
    c = json.loads(cpath.read_text(encoding="utf-8"))
    for field in ("image_spec_version", "image_transforms", "image_keys"):
        c.pop(field)
    c["converter_version"] = 1
    cpath.write_text(json.dumps(c), encoding="utf-8")
    assert conv.verify(out, None, tmp_path / "check_v1.png") is False
    assert "FAIL image spec" in capsys.readouterr().out

