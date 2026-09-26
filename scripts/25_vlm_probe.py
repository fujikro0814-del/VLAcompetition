"""情報がどこで途切れていたかの診断（決裁 0048 の任意）。結果は outputs/k1/vlm_probe.json。

    .venv\\Scripts\\python.exe scripts\\25_vlm_probe.py [--gen gen_100] [--train train_K1]

K1 の 300 本（配置 100 × 3 色）の最初のこまを、最初の K1 の保存点（VLM は固定＝smolvla_libero のまま）の前処理と
前半（embed_prefix → VLM）に通し、VLM の最後の層の後の特徴から、リッジ回帰で位置を読み取る。
  (a) 画像のトークンだけ → 3 色それぞれの位置（x, y）
  (b) 指示のトークンだけ（注意の覆いが 1 のもの）→ 指示した立方体の位置
  (c) 全トークン（画像・指示・状態）→ 指示した立方体の位置
画像と指示のトークンは前半の中で互いに注意する（同じブロック）ので、(b) は「指示のトークンが指示した立方体の位置を
画像から取り込んでいるか」を見る。交差検証は配置で分けた 5 分割（同じ配置の 3 本は同じ分割）、正則化の強さは
学習側の中の 4 分割で選ぶ（双対形のリッジ、特徴は学習側の平均と標準偏差で標準化）。誤差は水平の距離の中央値 [cm]。
基準は「常に 3 個の重心を答えた場合」。
"""
import argparse
import json
import time

import numpy as np

from recovla.common import config
from recovla.common.seeds import COLORS

CFG = config.load()
OUT = config.path(CFG["paths"]["outputs"]) / "k1"
ALPHAS = [10.0 ** k for k in range(-1, 6)]
FOLDS = 5


def ridge_dual(Xtr, Ytr, Xte, alpha):
    K = Xtr @ Xtr.T
    A = np.linalg.solve(K + alpha * np.eye(len(K)), Ytr - Ytr.mean(0))
    return (Xte @ Xtr.T) @ A + Ytr.mean(0)


def standardize(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def cv_predict(X, Y, groups, rng) -> tuple:
    """配置で分けた 5 分割。各分割の学習側で、さらに 4 分割して alpha を選ぶ。"""
    ug = np.unique(groups)
    rng.shuffle(ug)
    fold_of = {g: i % FOLDS for i, g in enumerate(ug)}
    f = np.array([fold_of[g] for g in groups])
    pred = np.zeros_like(Y)
    chosen = []
    for k in range(FOLDS):
        tr, te = f != k, f == k
        inner = np.array([fold_of[g] for g in groups[tr]])
        best = None
        for alpha in ALPHAS:
            errs = []
            for j in set(inner.tolist()):
                itr, ite = inner != j, inner == j
                a, b = standardize(X[tr][itr], X[tr][ite])
                p = ridge_dual(a, Y[tr][itr], b, alpha)
                errs.append(np.mean(np.linalg.norm((p - Y[tr][ite]).reshape(len(p), -1, 2), axis=2)))
            e = float(np.mean(errs))
            if best is None or e < best[0]:
                best = (e, alpha)
        chosen.append(best[1])
        a, b = standardize(X[tr], X[te])
        pred[te] = ridge_dual(a, Y[tr], b, best[1])
    return pred, chosen


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gen", default="gen_100")
    ap.add_argument("--train", default="train_K1")
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
    from recovla.policy.scene_policy import ScenePolicy
    tr = json.loads((OUT / f"{a.train}.json").read_text(encoding="utf-8"))
    g = json.loads((OUT / f"{a.gen}.json").read_text(encoding="utf-8"))
    pol = ScenePolicy(config.path(tr["checkpoint"]))
    root = config.path(g["dataset"])
    ds = LeRobotDataset(f"local/{root.name}", root=root,
                        delta_timestamps={"action": [i / 10 for i in pol.policy.config.action_delta_indices]})
    firsts = [int(v) for v in ds.meta.episodes["dataset_from_index"]]
    conv = json.loads((root / "meta" / "conversion.json").read_text(encoding="utf-8"))
    run = config.ROOT / g["run"]
    model = pol.policy.model
    feats = {"img": [], "lang": [], "all": []}
    meta_rows = []
    for b0 in range(0, len(firsts), 10):
        idx = firsts[b0:b0 + 10]
        items = [ds[i] for i in idx]
        batch = {}
        for k in items[0]:
            v = [it[k] for it in items]
            batch[k] = torch.stack(v) if isinstance(v[0], torch.Tensor) else v
        batch = pol.pre(batch)
        with torch.no_grad():
            images, img_masks = pol.policy.prepare_images(batch)
            state = pol.policy.prepare_state(batch)
            lt, lm = batch["observation.language.tokens"], batch["observation.language.attention_mask"]
            keep = model.prefix_length
            model.prefix_length = 0                         # 埋め草なしで並びの長さを知る
            embs, pad, att = model.embed_prefix(images, img_masks, lt, lm, state=state)
            model.prefix_length = keep
            att2d = make_att_2d_masks(pad, att)
            pos = torch.cumsum(pad, dim=1) - 1
            # 推論（sample_actions）と同じく、前半だけを KV キャッシュを満たす経路で通す（行動エキスパートは通らない）
            out, _ = model.vlm_with_expert.forward(attention_mask=att2d, position_ids=pos, past_key_values=None,
                                                   inputs_embeds=[embs, None], use_cache=True)
            h = out[0].float().cpu().numpy()
        L = h.shape[1]
        n_lang = lt.shape[1]
        l0 = L - 1 - n_lang
        n_img_cams = len(images)
        per = l0 // n_img_cams
        for j in range(len(idx)):
            lmask = lm[j].bool().cpu().numpy()
            feats["img"].append(h[j, :2 * per].reshape(-1))           # camera1（俯瞰）と camera2（手首）
            feats["lang"].append(h[j, l0:l0 + n_lang][lmask].reshape(-1))
            valid = pad[j].bool().cpu().numpy()
            feats["all"].append(h[j][valid].reshape(-1))
        meta_rows += [conv["sources"][ep] for ep in range(b0, b0 + len(idx))]
    # 真値: 最初のこまの立方体の位置
    Y3, Yt, groups, cent = [], [], [], []
    for s in meta_rows:
        z = np.load(config.ROOT / g["run"] / s["raw_episode"] / "data.npz") if not s.get("raw_path") else \
            np.load(__import__("pathlib").Path(s["raw_path"]) / "data.npz")
        cp = z["cube_pos"][0, :, :2]
        Y3.append(cp.reshape(-1))
        Yt.append(cp[COLORS.index(s["target"])])
        cent.append(cp.mean(0))
        groups.append(int(s["layout_seed"]))
    Y3, Yt, cent, groups = np.array(Y3), np.array(Yt), np.array(cent), np.array(groups)
    rng = np.random.default_rng(0)
    res = {}
    for name, X, Y in (("a_image_tokens_to_3_cubes", np.array(feats["img"]), Y3),
                       ("b_instruction_tokens_to_target", np.array(feats["lang"]), Yt),
                       ("c_all_tokens_to_target", np.array(feats["all"]), Yt)):
        pred, alphas = cv_predict(X.astype(np.float64), Y, groups, rng)
        err = np.linalg.norm((pred - Y).reshape(len(Y), -1, 2), axis=2).reshape(-1)
        base_pred = np.tile(cent, (1, Y.shape[1] // 2))
        base = np.linalg.norm((base_pred - Y).reshape(len(Y), -1, 2), axis=2).reshape(-1)
        res[name] = {"features": int(X.shape[1]), "median_err_cm": round(float(np.median(err)) * 100, 2),
                     "p90_err_cm": round(float(np.percentile(err, 90)) * 100, 2),
                     "baseline_centroid_median_err_cm": round(float(np.median(base)) * 100, 2),
                     "alphas_chosen": alphas}
    out = {"gen": a.gen, "checkpoint": tr["checkpoint"], "samples": len(Y3), "layouts": int(len(np.unique(groups))),
           "folds": FOLDS, "alphas": ALPHAS, "token_layout": {"prefix_len": int(L), "image_tokens_per_camera": int(per),
                                                               "language_tokens": int(n_lang)},
           "results": res, "wall_s": round(time.perf_counter() - t0, 1)}
    (OUT / "vlm_probe.json").write_text(json.dumps({"check": "vlm_probe", "written": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                    **out}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
