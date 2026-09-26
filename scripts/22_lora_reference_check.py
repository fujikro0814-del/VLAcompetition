"""保存した LoRA のモデルを推論の経路で読み、学習時の出力と一致するかを確かめる（決裁 0040 の条件 2）。

    .venv\\Scripts\\python.exe scripts\\22_lora_reference_check.py --train train_smoke_lora

学習の包み（recovla.policy.train_wrapped --reference）が保存のたびに書いた pretrained_model/recovla_reference.pt
（学習中のメモリ上の方策に、データセットの決まったこまと決まった雑音を与えた塊、後処理の前）を読み、
  1. 推論の経路（recovla.policy.scene_policy.ScenePolicy＝PolicyActions の読み込み）で保存点を読む
  2. 同じこまを同じデータセットから取り、保存点の前処理に通し、同じ雑音で predict_action_chunk
  3. 学習時の塊と一致するか（差の最大）、アダプタを切った塊（対照）とは違うか
結果は outputs/k1/lora_check<tag>.json。
"""
import argparse
import json
import time

from recovla.common import config

CFG = config.load()
OUT = config.path(CFG["paths"]["outputs"]) / "k1"
MATCH_TOL = 1e-6             # 正規化された行動の空間（後処理の前）での差の最大の許容（同じ設定ならビット一致する）


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", default="train_smoke_lora")
    ap.add_argument("--gen", default="gen_100")
    ap.add_argument("--tag", default="_smoke")
    a = ap.parse_args(argv)
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from recovla.policy.scene_policy import ScenePolicy
    from recovla.policy.train_wrapped import REFERENCE_FILE
    tr = json.loads((OUT / f"{a.train}.json").read_text(encoding="utf-8"))
    ckpt = config.path(tr["checkpoint"]).resolve()
    ref = torch.load(ckpt / REFERENCE_FILE, weights_only=False)
    pol = ScenePolicy(ckpt)
    g = json.loads((OUT / f"{a.gen}.json").read_text(encoding="utf-8"))
    root = config.path(g["dataset"])
    fps = int(CFG["convert"]["fps"])
    ds = LeRobotDataset(f"local/{root.name}", root=root,
                        delta_timestamps={"action": [i / fps for i in pol.policy.config.action_delta_indices]})
    items = [ds[i] for i in ref["dataset_indices"]]
    batch = {}
    for k in items[0]:
        v = [it[k] for it in items]
        batch[k] = torch.stack(v) if isinstance(v[0], torch.Tensor) else v
    batch = pol.pre(batch)
    # lerobot-train は起動時に torch.backends.cuda.matmul.allow_tf32 = True にする（cudnn_deterministic が false の
    # とき。lerobot_train.py）。学習時の出力は TF32 ありで出ているので、一致は同じ設定で見る。推論の経路（評価器）は
    # TF32 なしで動いているので、その差も並べて記録する
    noise = ref["noise"].to(pol.device)
    outs = {}
    for tf32 in (False, True):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        with torch.no_grad():
            outs[tf32] = pol.policy.predict_action_chunk(batch, noise=noise).cpu()
    got = outs[True]
    diff = float((got - ref["chunk"]).abs().max())
    res = {"checkpoint": str(ckpt), "peft": pol.peft, "dataset_indices": ref["dataset_indices"],
           "tasks": ref["tasks"], "tasks_now": [it["task"] for it in items],
           "max_abs_diff_vs_training": diff, "match_tol": MATCH_TOL, "match": diff <= MATCH_TOL,
           "tf32_for_match": True,
           "max_abs_diff_vs_training_tf32_off": float((outs[False] - ref["chunk"]).abs().max()),
           "note_tf32": "学習は TF32 あり、評価器は TF32 なし。TF32 の有無だけで塊が変わる大きさを参考に記録する"}
    if "chunk_adapter_disabled" in ref:
        with torch.no_grad(), pol.policy.disable_adapter():
            dis = pol.policy.predict_action_chunk(batch, noise=noise).cpu()
        res["disabled_matches_training_disabled"] = float((dis - ref["chunk_adapter_disabled"]).abs().max())
        d0 = float((got - ref["chunk_adapter_disabled"]).abs().max())
        dref = float((ref["chunk"] - ref["chunk_adapter_disabled"]).abs().max())
        res.update(max_abs_diff_vs_adapter_disabled=d0, training_vs_disabled=dref,
                   adapter_effective=d0 > 100 * max(diff, 1e-7))
    res["pass"] = res["match"] and res.get("adapter_effective", False) and pol.peft is not None
    p = OUT / f"lora_check{a.tag}.json"
    p.write_text(json.dumps({"check": "lora_check", "written": time.strftime("%Y-%m-%d %H:%M:%S"), **res},
                            ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
