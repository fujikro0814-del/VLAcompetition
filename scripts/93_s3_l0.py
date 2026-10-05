"""段階 3 の 0 周目 L0: 開ループで、止まる位置の符号つきの偏りを測る（0126 の 4、仮説 A 対 D の判別）。

    .venv\\Scripts\\python.exe scripts\\93_s3_l0.py [--models N1v2 R1v2_20000]

観測の経路と雑音の種は scripts/88_cause.py の c1 と同じ（変換の経路の観測、雑音の種 1000 + こまの番号）。
こまごとに、予測の塊の実行する最初の 10 行の終点（行動 [:, :3] の和）と記録の行動の 10 行の終点の差を、近づく向き
（そのこまの手先 → 立方体の水平の単位ベクトル）に射影する（負＝手前で止まる）。先行は x_des − 手先 を同じ向きに射影。
対象のこま（どれもグリッパが開いている・立方体を持ち上げる前・指先が立方体より 60 mm 以上高い・水平の残り 25〜200 mm）:
  (a) 学習に使っていないエキスパート outputs/gen/C1_heldout_v2（学習用の種 59950〜59969）
  (b) R1v2 の学習データのうち N1v2 にないエピソード（復帰の区間）の、先行 < 10 mm のこま（N1v2 は学習していない）
判定は N1v2 だけで読む（回す前に固めた決まり、0126 の 4）:
  - (a) の先行 ≥ 30 mm のこまで中央値が −5〜+5 mm、かつ (b) で中央値が −20 mm 以下 → 仮説 A（先行の取り違え）が主因
  - (a) でも（先行によらず）中央値が −20 mm 以下 → D（行動の縮み・平均化）も効いている
  - どちらでもない → 判別できない
区間はエピソードを単位にしたブートストラップ（種 20261005、2000 回）の 95%。結果: docs/results/s3_round0_l0.json
"""
import argparse
import importlib.util
import json
import pathlib
import time

import numpy as np

from recovla.common import config

CFG = config.load()
OUT = config.path(CFG["paths"]["outputs"])
RES = config.ROOT / "docs" / "results"
COLORS = ["red", "green", "blue"]
R = 10
HELDOUT = OUT / "gen" / "C1_heldout_v2"
DS = {"R1v2": OUT / "datasets/R1v2_20260929-205251", "N1v2": OUT / "datasets/N1v2_20260929-205251"}
REM_MM = (25.0, 200.0)
DZ_MM = 60.0
LIFT_Z = 0.045
LEAD_EXPERT_MM = 30.0
LEAD_SMALL_MM = 10.0
BOOT_SEED, BOOT_N = 20261005, 2000


def _mod(name, file):
    spec = importlib.util.spec_from_file_location(name, config.ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def frame_geometry(p: pathlib.Path, meta: dict) -> dict:
    """10 Hz のこま（変換と同じ idx）ごとの真値: 近づく向き・先行・残り・高さ・閉じたか・持ち上げたか。"""
    from recovla.data import convert as CV
    z = np.load(p / "data.npz")
    n = int(meta["n_frames"])
    idx = np.arange((n - 1) // CV.STRIDE) * CV.STRIDE
    ti = COLORS.index(meta["target"])
    cube, ee, tip, xdes = z["cube_pos"][idx, ti], z["ee_pos"][idx], z["fingertip"][idx], z["x_des"][idx]
    closed = z["gripper_closed"][idx].astype(bool)
    rel = cube[:, :2] - ee[:, :2]
    rem = np.linalg.norm(rel, axis=1)
    u = rel / np.maximum(rem[:, None], 1e-9)
    lead = np.sum((xdes[:, :2] - ee[:, :2]) * u, axis=1) * 1e3
    kl = np.where(cube[:, 2] > LIFT_Z)[0]
    k_lift = int(kl[0]) if kl.size else len(idx)
    # 開いている・まだ持ち上げていないこま（復帰 A は空を掴んで開き直した後も含む＝診断 1 の d1_g と同じ）
    before = (~closed) & (np.arange(len(idx)) < k_lift)
    dz = (tip[:, 2] - cube[:, 2]) * 1e3
    cand = before & (dz > DZ_MM) & (rem * 1e3 > REM_MM[0]) & (rem * 1e3 < REM_MM[1])
    return {"u": u, "lead": lead, "rem_mm": rem * 1e3, "dz_mm": dz, "cand": cand}


def episodes_b() -> list:
    def names(ds):
        conv = json.loads((ds / "meta" / "conversion.json").read_text(encoding="utf-8"))
        return [pathlib.Path(s["raw_path"]) for s in conv["sources"]]
    n_set = {p.name for p in names(DS["N1v2"])}
    return [p for p in names(DS["R1v2"]) if p.name not in n_set]


def run_model(model: str, eps_a: list, eps_b: list) -> list:
    import torch
    from recovla.data import convert as CV
    from recovla.data import vla_observation, vla_state
    from recovla.harness import setup as HS
    from recovla.harness.sensors import SensorSuite
    from recovla.harness.world import WorldRig
    from recovla.runtime.motion import Motion
    from recovla.runtime.policy import SensorPolicy
    from recovla.runtime.runner import disable_rtc_for
    from recovla.sim import scene
    ck = config.path(_mod("v2e", "82_v2_eval.py").CKPT[model])
    cfg2 = config.load_v2()
    world = WorldRig(render=False, cfg=cfg2)
    suite = SensorSuite(world.model, cfg2)
    world.reset(scene.sample_layout(59950, "empty", start="home"))
    setup = suite.start_trial(59950, world.data, HS.nominal_setup(cfg2))     # 方策の読み込みに要るだけ（88_cause c1 と同じ）
    pol = SensorPolicy(ck, setup, Motion(setup).hand_pose)
    disable_rtc_for(pol)
    rec = pol.conversion["target_cue"]
    tracker = CV.cue_tracker(rec.get("thresholds_source", "training"))
    keep = "cue_visible" in rec["names"]
    rows = []
    for group, eps in (("a", eps_a), ("b", eps_b)):
        for p in eps:
            if not (p / "meta.json").is_file():
                continue
            meta, data = CV.load_raw(p)
            if group == "a" and not meta.get("success"):
                continue
            g = frame_geometry(p, meta)
            sel = g["cand"] if group == "a" else g["cand"] & (g["lead"] < LEAD_SMALL_MM)
            if not sel.any():
                continue
            arr = CV.episode_arrays(meta, data)
            etr = CV.episode_tracker(meta, tracker)
            etr.reset()
            act = arr["action"]
            last = int(np.where(sel)[0].max())
            for k, i in enumerate(arr["raw_index"][:last + 1]):
                over = CV.read_raw_image(p / "overhead" / f"{i:06d}.png")
                c = etr.update(over, meta["target"])                           # 手がかりは毎こま追う（c1 と同じ）
                if not sel[k] or len(act) - k < R:
                    continue
                raw = {"overhead": over, "wrist": CV.read_raw_image(p / "wrist" / f"{i:06d}.png")}
                state = vla_state.with_cue(arr["state"][k], c, keep_flag=keep)
                obs = {**vla_observation.observation_images(raw), "observation.state": state, "task": meta["instruction"]}
                pred = pol.infer(obs, torch.Generator().manual_seed(1000 + k))
                d = pred[:R, :3].sum(0) - act[k:k + R, :3].sum(0)
                rows.append({"model": model, "group": group, "ep": p.name, "k": int(k),
                             "bias_along_mm": float(d[:2] @ g["u"][k] * 1e3),
                             "err_norm_mm": float(np.linalg.norm(d) * 1e3), "lead_mm": float(g["lead"][k]),
                             "rem_mm": float(g["rem_mm"][k]), "dz_mm": float(g["dz_mm"][k])})
            print(f"[l0] {model} ({group}) {p.name} frames {int(sel.sum())}", flush=True)
    suite.close()
    return rows


def summ(rows: list) -> dict:
    if not rows:
        return {"frames": 0}
    x = np.array([r["bias_along_mm"] for r in rows])
    eps = sorted({r["ep"] for r in rows})
    by = {e: np.array([r["bias_along_mm"] for r in rows if r["ep"] == e]) for e in eps}
    rng = np.random.default_rng(BOOT_SEED)
    meds = [float(np.median(np.concatenate([by[eps[j]] for j in rng.integers(0, len(eps), len(eps))])))
            for _ in range(BOOT_N)]
    return {"frames": len(x), "episodes": len(eps), "median_mm": float(np.median(x)),
            "p25_mm": float(np.percentile(x, 25)), "p75_mm": float(np.percentile(x, 75)),
            "median_boot95_mm": [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))],
            "lead_median_mm": float(np.median([r["lead_mm"] for r in rows])),
            "err_norm_median_mm": float(np.median([r["err_norm_mm"] for r in rows]))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["N1v2", "R1v2_20000"])
    a = ap.parse_args(argv)
    eps_a = sorted(p for p in HELDOUT.iterdir() if p.is_dir())
    eps_b = episodes_b()
    out = {"what": "段階 3 の 0 周目 L0（0126 の 4）: 開ループの止まる位置の符号つきの偏り（近づく向き、負＝手前）",
           "rule": ["(a) 先行 ≥ 30 mm のこまで中央値が −5〜+5 mm、かつ (b) で中央値が −20 mm 以下 → 仮説 A（先行の取り違え）が主因",
                    "(a) でも中央値が −20 mm 以下（先行によらず）→ D（行動の縮み・平均化）も効いている",
                    "どちらでもない → 判別できない", "判定は N1v2 だけで読む"],
           "frames": {"common": f"グリッパが開いている・持ち上げる前・指先が立方体より {DZ_MM:.0f} mm 以上高い・水平の残り {REM_MM[0]:.0f}〜{REM_MM[1]:.0f} mm",
                      "a": f"{HELDOUT.relative_to(config.ROOT)}（学習用の種 59950〜59969、成功したエピソード）",
                      "b": f"R1v2 の学習データのうち N1v2 にないエピソード {len(eps_b)} 本（{DS['R1v2'].name} の conversion.json の sources）の、先行 < {LEAD_SMALL_MM:.0f} mm のこま"},
           "observation": "scripts/88_cause.py c1 と同じ（変換の経路、雑音の種 1000 + こまの番号）", "rows_per_chunk": R,
           "bootstrap": {"unit": "episode", "seed": BOOT_SEED, "n": BOOT_N}, "models": {}}
    all_rows = []
    for m in a.models:
        t0 = time.time()
        rows = run_model(m, eps_a, eps_b)
        all_rows += rows
        ra = [r for r in rows if r["group"] == "a"]
        out["models"][m] = {"a_all": summ(ra), "a_lead_ge_30": summ([r for r in ra if r["lead_mm"] >= LEAD_EXPERT_MM]),
                            "a_lead_lt_30": summ([r for r in ra if r["lead_mm"] < LEAD_EXPERT_MM]),
                            "b_lead_lt_10": summ([r for r in rows if r["group"] == "b"]), "wall_s": round(time.time() - t0, 1)}
    n = out["models"].get("N1v2")
    if n:
        a30, b, aall = n["a_lead_ge_30"].get("median_mm"), n["b_lead_lt_10"].get("median_mm"), n["a_all"].get("median_mm")
        if a30 is not None and b is not None and -5 <= a30 <= 5 and b <= -20:
            verdict = "仮説 A（先行の取り違え）が主因"
        elif (aall is not None and aall <= -20) or (a30 is not None and a30 <= -20):
            verdict = "D（行動の縮み・平均化）も効いている"
        else:
            verdict = "判別できない"
        out["verdict_N1v2"] = verdict
    out["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "s3_round0_l0.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "results").mkdir(parents=True, exist_ok=True)
    (OUT / "results" / "s3_round0_l0_rows.json").write_text(json.dumps(all_rows), encoding="utf-8")
    print(json.dumps({m: {k: v.get("median_mm") for k, v in s.items() if isinstance(v, dict)} for m, s in out["models"].items()},
                     ensure_ascii=False), out.get("verdict_N1v2"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
