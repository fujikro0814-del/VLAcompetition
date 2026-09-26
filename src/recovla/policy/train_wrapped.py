"""lerobot-train を包む（掲示板 0038・0040）。LeRobot のファイルは書き換えず、モジュールの名前を差し替えるだけ。

    python -m recovla.policy.train_wrapped [--first-frames N --first-weight W] [--reference] -- <lerobot-train の引数>

1. 重みつきの並べ方（--first-frames・--first-weight）: lerobot-train が作る EpisodeAwareSampler を、各エピソードの
   最初の N こまを W 倍の重みで引く WeightedEpisodeAwareSampler に差し替える。1 周の長さは同じ（全こま数）で、
   (種, 周回) から決まる乱数で復元抽出する。引いたこまの数を数え、保存のたびに sampler_stats.json に書く
2. 学習時の出力の記録（--reference）: 保存のたびに、学習中の（メモリ上の）方策に、データセットの決まったこまと
   決まった雑音を与えた塊（predict_action_chunk、後処理の前）を保存点の pretrained_model/recovla_reference.pt に
   書く。PEFT のときは、アダプタを切った出力も書く（対照）。推論の経路で読み込んだ方策の出力がこれと一致すれば、
   保存と読み込みで LoRA の差分が失われていない（決裁 0040 の条件 2）
"""
import argparse
import json
import pathlib
import sys

import numpy as np

from lerobot.datasets.sampler import EpisodeAwareSampler

REFERENCE_FILE = "recovla_reference.pt"
SAMPLER_STATS = "sampler_stats.json"
REFERENCE_NOISE_SEED = 20260926
REFERENCE_ITEMS = 3                  # データセットの最初の 3 エピソードの最初のこま


class WeightedEpisodeAwareSampler(EpisodeAwareSampler):
    """最初の first_frames こま（エピソードの中の位置 < first_frames）を weight 倍の重みで引く。"""

    last = None                      # 保存のときに統計を読むため、最後に作ったものを覚えておく

    def __init__(self, *args, first_frames: int = 20, weight: float = 5.0, drop_n_first_frames: int = 0, **kw):
        super().__init__(*args, drop_n_first_frames=drop_n_first_frames, **kw)
        lengths = np.diff(np.concatenate([[0], self._cum_lengths]))
        pos = np.concatenate([np.arange(n) + drop_n_first_frames for n in lengths])   # エピソードの中の位置
        self.is_first = pos < int(first_frames)
        self.weights = np.where(self.is_first, float(weight), 1.0)
        self.first_frames, self.weight = int(first_frames), float(weight)
        self.drawn = 0
        self.drawn_first = 0
        WeightedEpisodeAwareSampler.last = self

    @property
    def expected_first_share(self) -> float:
        w = self.weights
        return float(w[self.is_first].sum() / w.sum())

    def _iter_epoch(self, epoch: int, start: int):
        import torch
        draws = torch.multinomial(torch.as_tensor(self.weights, dtype=torch.float64), self._num_frames,
                                  replacement=True, generator=self._epoch_generator(epoch))
        for k in range(start, self._num_frames):
            p = int(draws[k])
            self.drawn += 1
            self.drawn_first += int(self.is_first[p])
            yield self._frame_index(p)

    def stats(self) -> dict:
        return {"first_frames": self.first_frames, "weight": self.weight, "num_frames": int(self._num_frames),
                "first_frames_total": int(self.is_first.sum()), "expected_first_share": self.expected_first_share,
                "uniform_first_share": float(self.is_first.mean()), "drawn": self.drawn,
                "drawn_first": self.drawn_first,
                "drawn_first_share": (self.drawn_first / self.drawn) if self.drawn else None}


def write_reference(pretrained_dir: pathlib.Path, cfg, policy, preprocessor) -> dict:
    """学習中の方策の出力を保存点に書く（モジュールの説明の 2）。"""
    import torch
    from lerobot.datasets.factory import make_dataset
    ds = make_dataset(cfg)
    ep_from = [int(v) for v in ds.meta.episodes["dataset_from_index"]][:REFERENCE_ITEMS]
    items = [ds[i] for i in ep_from]
    batch = {}
    for k in items[0]:
        v = [it[k] for it in items]
        batch[k] = torch.stack(v) if isinstance(v[0], torch.Tensor) else v
    batch = preprocessor(batch)
    pcfg = policy.config
    gen = torch.Generator().manual_seed(REFERENCE_NOISE_SEED)
    noise = torch.randn((len(items), pcfg.chunk_size, pcfg.max_action_dim), generator=gen)
    was_training = policy.training
    policy.eval()
    out = {"dataset_indices": ep_from, "tasks": [it["task"] for it in items], "noise": noise}
    with torch.no_grad():
        out["chunk"] = policy.predict_action_chunk(batch, noise=noise.to(batch["observation.state"].device)).cpu()
        if hasattr(policy, "disable_adapter"):
            with policy.disable_adapter():
                out["chunk_adapter_disabled"] = policy.predict_action_chunk(
                    batch, noise=noise.to(batch["observation.state"].device)).cpu()
    if was_training:
        policy.train()
    torch.save(out, pretrained_dir / REFERENCE_FILE)
    return {"items": len(items), "has_disabled": "chunk_adapter_disabled" in out}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("usage: python -m recovla.policy.train_wrapped [options] -- <lerobot-train args>")
    cut = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-frames", type=int, default=None)
    ap.add_argument("--first-weight", type=float, default=None)
    ap.add_argument("--reference", action="store_true")
    a = ap.parse_args(argv[:cut])
    import lerobot.scripts.lerobot_train as LT

    if a.first_frames is not None:
        ff, fw = a.first_frames, float(a.first_weight)

        def make_sampler(*args, **kw):
            return WeightedEpisodeAwareSampler(*args, first_frames=ff, weight=fw, **kw)
        LT.EpisodeAwareSampler = make_sampler

    original_save = LT.save_checkpoint

    def save_checkpoint(checkpoint_dir, step, cfg, policy, *args, **kw):
        original_save(checkpoint_dir, step, cfg, policy, *args, **kw)
        pretrained = pathlib.Path(checkpoint_dir) / "pretrained_model"
        s = WeightedEpisodeAwareSampler.last
        if s is not None:
            (pathlib.Path(checkpoint_dir) / SAMPLER_STATS).write_text(
                json.dumps({"step": int(step), **s.stats()}, indent=2), encoding="utf-8")
        if a.reference:
            pre = kw.get("preprocessor", args[2] if len(args) > 2 else None)
            info = write_reference(pretrained, cfg, policy, pre)
            print(f"recovla: wrote {pretrained / REFERENCE_FILE} {info}", flush=True)
    LT.save_checkpoint = save_checkpoint

    sys.argv = ["lerobot-train"] + argv[cut + 1:]
    LT.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
