"""行動の塊の読み出しの高速化（train_wrapped --fast-query、0126 の 4）が、元の経路と同じ値を返すかの確かめ。

    .venv\\Scripts\\python.exe scripts\\91_fast_query_check.py [--n 1000] [--datasets R1v2_20260929-205251 N1v2_20260929-205251]

データセットごとに、決まった種（20261005）で引いた n 個のこまについて、元の DatasetReader._query_hf_dataset と
置き換えの両方で行動の塊を読み、全要素がビットで一致するか、型と形が同じかを比べる。1 試料あたりの時間も測る。
GPU とシミュレーションは使わない。結果は docs/results/fast_query_check.json。
"""
import argparse
import json
import os
import pathlib
import random
import statistics
import time

from recovla.common import config

CFG = config.load()
os.environ["HF_HOME"] = str(config.path(CFG["paths"]["models_home"]))
os.environ["HF_HUB_OFFLINE"] = "1"

import torch  # noqa: E402
from lerobot.datasets.dataset_reader import DatasetReader  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from recovla.policy import train_wrapped  # noqa: E402

SEED = 20261005
OUT = config.ROOT / "docs/results/fast_query_check.json"
DATASETS = config.path(CFG["paths"]["outputs"]) / "datasets"


def chunk_size() -> int:
    snap = config.path(CFG["paths"]["policy_snapshot"]) / "config.json"
    return int(json.loads(snap.read_text(encoding="utf-8"))["chunk_size"])


def delta_timestamps(root: pathlib.Path, k: int) -> dict:
    """学習と同じ（lerobot の resolve_delta_timestamps と SmolVLA の delta_indices）: 観測の列は [0]、行動は k 行。"""
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = info["fps"]
    out = {key: [0.0] for key in info["features"] if key.startswith("observation.")}
    out["action"] = [i / fps for i in range(k)]
    return out


def check(folder: str, n: int, k: int) -> dict:
    root = DATASETS / folder
    ds = LeRobotDataset(f"local/{folder}", root=root, delta_timestamps=delta_timestamps(root, k))
    rd = ds._ensure_reader()
    if rd.hf_dataset is None:
        rd.load_and_activate()
    rng = random.Random(SEED)
    idx = sorted(rng.sample(range(len(ds)), n))
    train_wrapped.install_fast_query()
    original = train_wrapped._ORIGINAL_QUERY            # 差し替える前の LeRobot の関数
    assert original is not train_wrapped.fast_query_hf_dataset and DatasetReader._query_hf_dataset is not original
    t_orig, t_fast, mismatch, worst = [], [], [], 0.0
    t0 = time.perf_counter()
    train_wrapped.fast_query_hf_dataset(rd, rd._get_query_indices(0, 0)[0])      # 表を作る（一度だけ）
    t_table = time.perf_counter() - t0
    for i in idx:
        row = rd.hf_dataset.with_format(None).select_columns(["episode_index", "index"])[i]
        q, _ = rd._get_query_indices(int(row["index"]), int(row["episode_index"]))
        t0 = time.perf_counter(); a = original(rd, q); t_orig.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); b = train_wrapped.fast_query_hf_dataset(rd, q); t_fast.append(time.perf_counter() - t0)
        same = set(a) == set(b) and all(a[key].dtype == b[key].dtype and a[key].shape == b[key].shape
                                          and torch.equal(a[key], b[key]) for key in a)
        if not same:
            mismatch.append(i)
            worst = max(worst, max(float((a[key].double() - b[key].double()).abs().max()) for key in a if key in b))
    return {"dataset": str(root), "frames": len(ds), "n": n, "seed": SEED, "keys": sorted(a),
            "dtype": {key: str(a[key].dtype) for key in a}, "shape": {key: list(a[key].shape) for key in a},
            "mismatch": len(mismatch), "mismatch_first": mismatch[:10], "max_abs_diff": worst,
            "original_ms_median": 1000 * statistics.median(t_orig), "fast_ms_median": 1000 * statistics.median(t_fast),
            "table_build_s": t_table, "pass": not mismatch}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--datasets", nargs="+", default=["R1v2_20260929-205251", "N1v2_20260929-205251"])
    a = ap.parse_args()
    k = chunk_size()
    rows = []
    for folder in a.datasets:
        r = check(folder, a.n, k)
        print(json.dumps({x: r[x] for x in ("dataset", "mismatch", "original_ms_median", "fast_ms_median", "pass")},
                         ensure_ascii=False), flush=True)
        rows.append(r)
    out = {"what": "fast_query の値の一致（0126 の 4）", "chunk_size": k, "results": rows,
           "pass": all(r["pass"] for r in rows)}
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"pass={out['pass']} -> {OUT}")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
