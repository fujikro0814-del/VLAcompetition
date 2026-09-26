"""K1 の LoRA の枝（掲示板 0040）の部品: 重みつきの並べ方（決裁の条件 1）と、起動器の LoRA・重みの設定。"""
import json
import re

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("lerobot")

from recovla.policy import train_launcher as tl                     # noqa: E402
from recovla.policy.train_wrapped import WeightedEpisodeAwareSampler   # noqa: E402

pytestmark = pytest.mark.torch

# K1 の 100 配置のデータに近い形: 300 本、1 本 120〜170 こま
RNG = np.random.default_rng(0)
LENGTHS = RNG.integers(120, 171, size=300)
TO = np.cumsum(LENGTHS)
FROM = TO - LENGTHS


def sampler(weight=5.0, frames=20, seed=1000):
    return WeightedEpisodeAwareSampler(FROM.tolist(), TO.tolist(), shuffle=True, seed=seed,
                                       first_frames=frames, weight=weight)


def test_expected_share_formula():
    s = sampler()
    n_first = 20 * len(LENGTHS)
    n_rest = int(LENGTHS.sum()) - n_first
    assert s.expected_first_share == pytest.approx(5 * n_first / (5 * n_first + n_rest))
    assert s.stats()["uniform_first_share"] == pytest.approx(n_first / LENGTHS.sum())


def test_drawn_share_matches_expected():
    """条件 1: 実際に引かれた「最初の 2 秒のこま」の割合が期待値に合う（3 周、二項の 4 標準偏差以内）。"""
    s = sampler()
    idx = [i for _ in range(3) for i in s]
    n = len(idx)
    assert n == 3 * int(LENGTHS.sum())                                  # 1 周の長さは全こま数のまま
    p = s.expected_first_share
    got = s.stats()["drawn_first_share"]
    assert abs(got - p) < 4 * np.sqrt(p * (1 - p) / n), (got, p)
    # 引いたこまを、エピソードの中の位置に戻して数え直しても同じ割合になる（数え方の検算）
    ep = np.searchsorted(TO, np.asarray(idx), side="right")
    pos = np.asarray(idx) - FROM[ep]
    assert (pos < 20).mean() == pytest.approx(got)
    assert (pos >= 0).all() and (pos < LENGTHS[ep]).all()


def test_weight_one_is_uniform():
    s = sampler(weight=1.0)
    idx = [i for i in s]
    p = s.stats()["uniform_first_share"]
    assert abs(s.stats()["drawn_first_share"] - p) < 4 * np.sqrt(p * (1 - p) / len(idx))


def test_deterministic_per_seed_and_epoch():
    a, b = sampler(), sampler()
    ea1, eb1 = list(a), list(b)
    assert ea1 == eb1                                                   # 同じ種・同じ周回なら同じ列
    assert list(a) != ea1                                               # 次の周回は別の列
    assert list(sampler(seed=1001)) != ea1


# --- launcher ---------------------------------------------------------------------------------------------

LORA = {"layers": list(range(8, 16)), "modules": ["q_proj", "v_proj"], "r": 16, "alpha": 32}
MODULE_NAMES = [f"model.vlm_with_expert.vlm.model.text_model.layers.{i}.self_attn.{m}"
                for i in range(16) for m in ("q_proj", "k_proj", "v_proj", "o_proj")]


def config(tmp_path, **over):
    cfg = {"dataset": str(tmp_path / "ds"), "train_scope": "expert", "batch_size": 32, "steps": 50, **over}
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return tl.load_config(p)


def test_lora_flags_target_exactly_layers_8_to_15_q_v(tmp_path):
    cfg = config(tmp_path, lora=LORA, first_frames_weight={"frames": 20, "weight": 5})
    cmd = tl.build_command(tl.wrapper_prefix(cfg, python="py"), "SNAP", cfg, tmp_path / "out", "job")
    cut = cmd.index("--")
    assert cmd[:cut] == ["py", "-m", "recovla.policy.train_wrapped", "--reference", "--first-frames=20",
                         "--first-weight=5"]
    args = dict(a.split("=", 1) for a in cmd[cut + 1:])
    assert len(args) == len(cmd) - cut - 1                              # no key given twice
    target = args["--peft.target_modules"]
    hits = [n for n in MODULE_NAMES if re.fullmatch(target, n)]
    assert hits == [f"model.vlm_with_expert.vlm.model.text_model.layers.{i}.self_attn.{m}"
                    for i in range(8, 16) for m in ("q_proj", "v_proj")]
    assert json.loads(args["--peft.full_training_modules"]) == tl.FULL_TRAINING_MODULES
    assert args["--peft.r"] == "16" and args["--peft.lora_alpha"] == "32" and args["--peft.method_type"] == "LORA"
    assert args["--policy.train_expert_only"] == "true"                # 決裁 0040: VLM 本体は凍結のまま
    assert not any(a.startswith("--policy.optimizer_lr") for a in cmd)  # 学習率は変えない


@pytest.mark.parametrize("over, words", [
    ({"lora": {"layers": [16], "modules": ["q_proj"], "r": 16, "alpha": 32}}, "lora.layers"),
    ({"lora": {"layers": [8], "modules": ["gate"], "r": 16, "alpha": 32}}, "lora.modules"),
    ({"lora": {"layers": [8], "modules": ["q_proj"], "r": 0, "alpha": 32}}, "lora.r"),
    ({"lora": {"layers": [8], "modules": ["q_proj"], "r": 16}}, "lora must be"),
    ({"first_frames_weight": {"frames": 0, "weight": 5}}, "first_frames_weight"),
    ({"extra_args": ["--peft.r=8"]}, "--peft.r"),
])
def test_lora_config_rejects(tmp_path, over, words):
    with pytest.raises(tl.LaunchError, match=re.escape(words)):
        config(tmp_path, **over)


def test_plain_config_is_not_wrapped(tmp_path):
    assert not tl.wrapped(config(tmp_path))
