"""Tests for vla_observation.py, the evaluation-time image entry (Step E-C, 2026-09-17).

The first group is numpy only (python311 and LeRobot venv). The last test needs the LeRobot venv:
it converts a synthetic episode with convert_to_lerobot.py and checks that the evaluation entry
produces exactly the images stored in the training dataset for the same raw frames.
"""
import json
import pathlib

import numpy as np
import pytest

from recovla.data import vla_image_spec as spec
from recovla.data import vla_observation as obs

H, W = 6, 8


def raw_pair(seed=0):
    rng = np.random.default_rng(seed)
    return {v: rng.integers(0, 256, (H, W, 3), dtype=np.uint8) for v in spec.CAMERAS}


def training_run(tmp_path, conversion=None):
    """Fake lerobot-train output dir; returns the pretrained_model path."""
    out = tmp_path / "train_out"
    ckpt = out / "checkpoints" / "000010" / "pretrained_model"
    ckpt.mkdir(parents=True)
    if conversion is not None:
        (out / obs.COPIED_CONVERSION).write_text(json.dumps(conversion), encoding="utf-8")
    return ckpt


def test_observation_images_value_level_and_format():
    raw = raw_pair()
    got = obs.observation_images(raw)
    over = got["observation.images.image"]
    wrist = got["observation.images.image2"]
    assert over.dtype == np.float32 and over.shape == (3, H, W)
    # expected written without the module: overhead mirrored left-right, wrist upside down
    for r in range(H):
        for c in range(W):
            assert np.array_equal(over[:, r, c] * 255.0, raw["overhead"][r, W - 1 - c].astype(np.float32))
            assert np.array_equal(wrist[:, r, c] * 255.0, raw["wrist"][H - 1 - r, c].astype(np.float32))


def test_observation_images_rejects_a_missing_view():
    raw = raw_pair()
    with pytest.raises(spec.ImageSpecError):
        obs.observation_images({"overhead": raw["overhead"]})
    with pytest.raises(spec.ImageSpecError):
        obs.observation_images({"wrist": raw["wrist"]})


def test_builder_requires_the_copied_training_conversion(tmp_path):
    with pytest.raises(spec.ImageSpecError, match="not found"):
        obs.PolicyObservationBuilder(training_run(tmp_path))


@pytest.mark.parametrize("conversion", [
    {"converter_version": 1, "images": "PNG from raw, flipped left-right (LIBERO mirror)"},
    {**spec.spec_record(), "image_transforms": {"overhead": "fliplr", "wrist": "fliplr"}},
    {**spec.spec_record(), "image_transforms": {"wrist": "flipud"}},
])
def test_builder_rejects_version1_or_different_spec(tmp_path, conversion):
    with pytest.raises(spec.ImageSpecError):
        obs.PolicyObservationBuilder(training_run(tmp_path, conversion))


def proprio(seed=0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=4)
    return {"ee_pos": rng.normal(size=3), "ee_quat": q / np.linalg.norm(q),
            "fingers": rng.uniform(0, 0.04, 2), "joints": rng.normal(size=7)}


def test_builder_accepts_matching_spec_and_builds(tmp_path):
    ckpt = training_run(tmp_path, {"converter_version": 2, **spec.spec_record()})
    for where in (ckpt, ckpt.parent.parent.parent):       # pretrained_model or the output dir itself
        b = obs.PolicyObservationBuilder(where)
    raw = raw_pair(3)
    p = proprio(1)
    o = b.build(raw, p, "put the cube in the box")
    assert set(o) == {"observation.images.image", "observation.images.image2", "observation.state", "task"}
    assert o["observation.state"].dtype == np.float32 and o["observation.state"].shape == (15,)
    assert o["task"] == "put the cube in the box"
    # the state comes from vla_state (fingers: +f1, -f2), not passed through
    assert o["observation.state"][6] == np.float32(p["fingers"][0])
    assert o["observation.state"][7] == np.float32(-p["fingers"][1])
    with pytest.raises(ValueError):
        b.build(raw, {k: v for k, v in p.items() if k != "joints"}, "x")
    with pytest.raises(ValueError):
        b.build(raw, {**p, "ee_quat": np.zeros(3)}, "x")
    with pytest.raises(spec.ImageSpecError):
        b.build({"wrist": raw["wrist"]}, p, "x")


@pytest.mark.torch                               # LeRobotDataset を作る（掲示板 0013 の 3）
def test_evaluation_entry_equals_training_dataset_images(tmp_path):
    pytest.importorskip("lerobot")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from recovla.data import convert as conv
    from tests.legacy.test_convert_to_lerobot import make_raw_episode

    raw_root = tmp_path / "raw"
    ep, meta, data = make_raw_episode(raw_root, n=21)
    out = tmp_path / "ds"
    conv.convert([ep], out, "unit_obs")
    conversion = json.loads((out / "meta" / "conversion.json").read_text(encoding="utf-8"))
    ckpt = training_run(tmp_path, conversion)                 # as the training launcher would copy it
    builder = obs.PolicyObservationBuilder(ckpt)
    ds = LeRobotDataset("local/unit_obs", root=out)
    arr = conv.episode_arrays(meta, np.load(ep / "data.npz"))
    for k, i in enumerate(arr["raw_index"]):
        raw = {v: conv.read_raw_image(ep / v / f"{i:06d}.png") for v in spec.CAMERAS}
        frame = {f: data[f][i] for f in ("ee_pos", "ee_quat", "fingers", "joints")}   # one raw frame, as in evaluation
        o = builder.build(raw, frame, meta["instruction"])
        item = ds[k]
        for key in spec.IMAGE_KEYS.values():
            stored = item[key].numpy()
            assert stored.shape == o[key].shape
            assert np.array_equal(np.round(stored * 255), np.round(o[key] * 255)), (k, key)
        # state: bit-identical to the dataset (evaluation computes it per frame, training per episode)
        assert np.array_equal(item["observation.state"].numpy(), o["observation.state"]), k
