"""K1 の不合格の診断（掲示板 0038、決裁が承認）。結果は outputs/k1/diag.json。学習し直しはしない。

    .venv\\Scripts\\python.exe scripts\\21_k1_diag.py [--checkpoint DIR]

D1 向きの一致（GPU 不要。outputs/k1/e6.json を読む）:
   E6 の 99 件（33 配置 x 3 指示）で、(指示 c の予測の到達点 − 3 指示の予測の平均) と (指示 c の正解の到達点 −
   3 つの正解の重心) の cos。予測の到達点は 5 回の平均、正解の到達点は色 c の立方体の中心（水平）。
   あわせて 3x3 の取り違え表（指示した色 x 最寄りの色、各 5 回）
D2 教師あり損失の比較: K1 の学習データの最初の 1 秒のこま（frame_index 0〜9、全 90 本）で、正しい指示と他の 2 指示の
   損失（SmolVLAPolicy.forward の reduction="none"）を、雑音と時刻を揃えて比べる
D3 指示の経路: 同じ観測（K1 のデモの最初のこま）について、学習時の経路（LeRobotDataset → 保存点の前処理）と
   推論時の経路（vla_observation の builder → 同じ前処理）とで、指示のトークン列・注意の覆い・画像・状態が一致するか
"""
import argparse
import collections
import itertools
import json
import time

import numpy as np

from recovla.common import config
from recovla.common.seeds import COLORS

CFG = config.load()
OUT = config.path(CFG["paths"]["outputs"]) / "k1"
D2_FRAMES = range(10)          # 最初の 1 秒（10 fps）
D2_BATCH = 30


def read(name: str) -> dict:
    return json.loads((OUT / f"{name}.json").read_text(encoding="utf-8"))


def pct(x) -> list:
    return [round(float(v), 4) for v in np.percentile(np.asarray(x, float), [10, 50, 90])]


# ------------------------------------------------------------------------------------------------ D1

def d1() -> dict:
    from recovla.sim import scene
    e6 = read("e6")
    by = collections.defaultdict(dict)
    conf = {c: collections.Counter() for c in COLORS}
    for r in e6["rows"]:
        by[r["seed"]][r["instruction"]] = np.array([s["xy"] for s in r["samples"]])
        for s in r["samples"]:
            conf[r["instruction"]][s["nearest"]] += 1
    cos, rows = [], []
    for seed, d in sorted(by.items()):
        lay = scene.sample_layout(seed, "empty", start="home")
        gt = {c: np.array(lay.cubes[c][:2], float) for c in COLORS}
        pred = {c: d[c].mean(0) for c in COLORS}
        pm, gm = np.mean(list(pred.values()), 0), np.mean(list(gt.values()), 0)
        for c in COLORS:
            u, v = pred[c] - pm, gt[c] - gm
            k = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
            cos.append(k)
            rows.append({"seed": seed, "instruction": c, "cos": round(k, 4),
                         "pred_dev_m": round(float(np.linalg.norm(u)), 4), "gt_dev_m": round(float(np.linalg.norm(v)), 4)})
    cos = np.array(cos)
    return {"n": len(cos), "cos_p10_p50_p90": pct(cos), "cos_mean": round(float(cos.mean()), 4),
            "cos_positive_share": round(float((cos > 0).mean()), 4),
            "cos_above_0p5_share": round(float((cos > 0.5).mean()), 4),
            "pred_dev_over_gt_dev_p10_p50_p90": pct([r["pred_dev_m"] / r["gt_dev_m"] for r in rows]),
            "confusion_instructed_x_nearest": {c: {n: conf[c][n] for n in COLORS} for c in COLORS},
            "gt_endpoint": "色 c の立方体の中心（配置の値、水平）", "rows": rows}


# --------------------------------------------------------------------------------------- the two paths

def load_all(ckpt):
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from recovla.policy.scene_policy import ScenePolicy
    pol = ScenePolicy(ckpt)
    g = read("gen")
    root = config.path(g["dataset"])
    fps = int(CFG["convert"]["fps"])
    delta = {"action": [i / fps for i in pol.policy.config.action_delta_indices]}
    ds = LeRobotDataset(f"local/{root.name}", root=root, delta_timestamps=delta)
    return torch, pol, ds, config.path(g["run"])


def collate(items):
    import torch
    out = {}
    for k in items[0]:
        v = [it[k] for it in items]
        out[k] = torch.stack(v) if isinstance(v[0], torch.Tensor) else v
    return out


def episode_frame_indices(ds, frames) -> list:
    """(episode_index, frame_index) → データセットの通し番号。"""
    ep = np.asarray(ds.hf_dataset["episode_index"])
    fr = np.asarray(ds.hf_dataset["frame_index"])
    return [int(i) for i in np.flatnonzero(np.isin(fr, list(frames)))], ep, fr


# ------------------------------------------------------------------------------------------------ D2

def d2(torch, pol, ds) -> dict:
    idx, ep, fr = episode_frame_indices(ds, D2_FRAMES)
    instr = {c: CFG["convert"]["instruction"].format(color=c) for c in COLORS}
    gen = torch.Generator().manual_seed(20260926)
    cfg = pol.policy.config
    rows = []
    for b in range(0, len(idx), D2_BATCH):
        items = [ds[i] for i in idx[b:b + D2_BATCH]]
        n = len(items)
        noise = torch.randn((n, cfg.chunk_size, cfg.max_action_dim), generator=gen)
        t = torch.distributions.Beta(1.5, 1.0).sample((n,)) * 0.999 + 0.001      # SmolVLA の学習と同じ時刻の分布
        true_c = [next(c for c in COLORS if it["task"].strip() == instr[c]) for it in items]
        losses = {}
        for c in COLORS:
            batch = collate([dict(it, task=instr[c]) for it in items])
            batch = pol.pre(batch)
            with torch.no_grad():
                per, _ = pol.policy.forward(batch, noise=noise.to(pol.device), time=t.to(pol.device), reduction="none")
            losses[c] = per.detach().cpu().numpy()
        for j in range(n):
            tc = true_c[j]
            others = [c for c in COLORS if c != tc]
            rows.append({"episode": int(ep[idx[b + j]]), "frame": int(fr[idx[b + j]]), "true": tc,
                         "loss_true": float(losses[tc][j]), "loss_others": [float(losses[c][j]) for c in others]})
    diff = np.array([np.mean(r["loss_others"]) - r["loss_true"] for r in rows])
    rel = np.array([(np.mean(r["loss_others"]) - r["loss_true"]) / max(r["loss_true"], 1e-12) for r in rows])
    best = np.array([r["loss_true"] < min(r["loss_others"]) for r in rows])
    by_frame = {int(f): round(float(best[[r["frame"] == f for r in rows]].mean()), 4) for f in D2_FRAMES}
    return {"samples": len(rows), "episodes": len({r["episode"] for r in rows}),
            "loss_true_mean": round(float(np.mean([r["loss_true"] for r in rows])), 5),
            "loss_others_mean": round(float(np.mean([np.mean(r["loss_others"]) for r in rows])), 5),
            "diff_others_minus_true_mean": round(float(diff.mean()), 5),
            "diff_p10_p50_p90": pct(diff), "rel_diff_p10_p50_p90": pct(rel),
            "true_is_min_share": round(float(best.mean()), 4), "true_is_min_share_by_frame": by_frame,
            "noise_time": "各こまで 3 指示に同じ雑音と時刻（Beta(1.5, 1) を 0.001〜1 に写したもの）", "rows": rows}


# ------------------------------------------------------------------------------------------------ D3

def d3(torch, pol, ds, run) -> dict:
    from recovla.record.recorder import read_png
    idx, ep, fr = episode_frame_indices(ds, [0])
    names = [p for p in sorted(run.iterdir()) if p.is_dir() and not p.name.endswith(".partial")]
    gen_log = [json.loads(l) for l in (run / "generation.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    order = [a["name"] for r in gen_log for a in r["attempts"] if a["success"]]   # 変換はこの順に episode_index を振る
    rows = []
    tok = getattr(pol.pre, "steps", [])
    for k in range(3):
        i = idx[k]
        item = ds[i]
        train_b = pol.pre(collate([item]))
        name = order[int(ep[i])]
        p = run / name
        z = np.load(p / "data.npz")
        frame = {kk: z[kk][0] for kk in z.files if not kk.startswith(("start_", "step_")) and kk != "timestamp"}
        raw = {v: read_png(p / v / "000000.png") for v in CFG["sim"]["cameras"]}
        meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
        infer_b = pol.prepare(pol.builder.build(raw, frame, meta["instruction"]))
        row = {"dataset_index": i, "episode": name, "task_dataset": item["task"], "task_meta": meta["instruction"]}
        for key in ("observation.language.tokens", "observation.language.attention_mask"):
            a, b = train_b[key].cpu(), infer_b[key].cpu()
            row[key] = {"shape_train": list(a.shape), "shape_infer": list(b.shape),
                        "equal": bool(a.shape == b.shape and torch.equal(a, b)),
                        "n_attended_train": int(train_b["observation.language.attention_mask"].sum()),
                        "n_attended_infer": int(infer_b["observation.language.attention_mask"].sum())}
        m = train_b["observation.language.attention_mask"][0].bool().cpu()
        toks = train_b["observation.language.tokens"][0].cpu()[m].tolist()
        tokenizer = next((s.tokenizer for s in tok if hasattr(s, "tokenizer")), None)
        row["decoded_train"] = tokenizer.decode(toks) if tokenizer is not None else None
        m2 = infer_b["observation.language.attention_mask"][0].bool().cpu()
        row["decoded_infer"] = tokenizer.decode(infer_b["observation.language.tokens"][0].cpu()[m2].tolist()) if tokenizer else None
        for key in [k2 for k2 in train_b if k2.startswith("observation.images") or k2 == "observation.state"]:
            if key in infer_b and hasattr(train_b[key], "shape"):
                a, b = train_b[key].float().cpu(), infer_b[key].float().cpu()
                row[key] = {"shape": list(a.shape), "max_abs_diff": float((a - b).abs().max()) if a.shape == b.shape else None}
        rows.append(row)
    all_equal = all(r[k]["equal"] for r in rows for k in ("observation.language.tokens", "observation.language.attention_mask"))
    return {"episodes_checked": len(rows), "tokens_and_mask_equal_all": all_equal, "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", default=None)
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    res = {"D1": d1()}
    ckpt = a.checkpoint or str(config.path(read("train_K1")["checkpoint"]))
    torch, pol, ds, run = load_all(ckpt)
    res["D3"] = d3(torch, pol, ds, run)
    res["D2"] = d2(torch, pol, ds)
    res["checkpoint"] = ckpt
    res["wall_s"] = round(time.perf_counter() - t0, 1)
    p = OUT / "diag.json"
    p.write_text(json.dumps({"check": "diag", "written": time.strftime("%Y-%m-%d %H:%M:%S"), **res},
                            ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[k1] wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
