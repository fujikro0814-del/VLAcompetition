"""Step G の完了条件のうち、方策を GPU で動かすもの（手順書 Step G の完了条件 1・2・4・6、決裁 0055・0062・0063）。

    .venv\\Scripts\\python.exe scripts\\42_g_checks.py cond1        # sync・s=50 が旧評価器（K1 の閉ループ）と同じ成否か（20 回）
    .venv\\Scripts\\python.exe scripts\\42_g_checks.py cond2        # naive・d=0 が同じ s の sync と同じ行動の列か（20 回）
    .venv\\Scripts\\python.exe scripts\\42_g_checks.py cond4        # 方策で誘発 P1〜P3 各 20 回（発動・成立の記録）
    .venv\\Scripts\\python.exe scripts\\42_g_checks.py cond6 --d D  # RTC が効いていること（同じ入力で rtc と naive の塊）

出力: outputs/g/condN.json（tests/test_g_eval.py が判定）、試行の記録は outputs/eval/Gcheck/<条件>/。
種は 0062・0063 の割り当て（docs/種の台帳.md）。学習・ほかのシミュレーションと同時に回さない。

- 完了条件 1 の照合相手は K1 の手がかりつき（closed_cue.json、191000〜191029 の先頭 20）。旧評価器は select_action の
  待ち行列（50 手）で、雑音を毎手 1 つ引く（推論 i は 50·i 手目に引いた雑音）。新しい実行器は推論 i の雑音を
  noise_generator(種, i) から引く。雑音が違うので軌跡は一致しない前提で、成否の一致を見る（手順書のとおり）
- 完了条件 4 は遅れ d の測定の前に回すので、d に依らない sync・s=10 で回す（誘発は行動の上書きなので方式に依らない）
- 完了条件 6 の「十分小さい」は結果を見る前に次で固める: 前の塊の残りがある推論ごとに、最初の d 手の前の塊との差
  （正規化された空間、行動の 7 次元の平均絶対差）を rtc と naive（同じ入力・同じ雑音で、前の塊を渡さない）で比べ、
  比 rtc/naive の中央値が 0.5 以下、かつ rtc の方が小さい推論が 9 割以上
"""
import argparse
import json
import os
import pathlib
import time

import numpy as np

from recovla.common import config

CFG = config.load()
# lerobot・huggingface_hub を import する前に決める（ネットワークなし。02_g0_check.py と同じ）
os.environ["HF_HOME"] = str(config.path(CFG["paths"]["models_home"]))
os.environ["HF_HUB_OFFLINE"] = "1"
OUT = config.path(CFG["paths"]["outputs"]) / "g"
EVAL_OUT = config.path(CFG["paths"]["outputs"]) / "eval" / "Gcheck"
R1_30K = config.ROOT / "outputs/train/train_R1_20260926-153959_20260926-153959/checkpoints/030000/pretrained_model"

COND1_SEEDS = (191000, 20)
COND2_SEEDS = (198300, 20)
COND4_SEEDS = (198400, 20)
COND6_SEEDS = (198500, 10)
COND6_RATIO_MAX = 0.5
COND6_WIN_MIN = 0.9


def _runner(ckpt, mode, s, d, horizon=None, schedule=None):
    from recovla.policy.runner import SceneRunner
    from recovla.policy.schedule import RuntimeConfig
    from recovla.policy.scene_policy import ScenePolicy
    pol = ScenePolicy(pathlib.Path(ckpt))
    rt = RuntimeConfig(mode, s, None if mode == "sync" else d,
                       execution_horizon=int(horizon or CFG["runtime"]["rtc_guidance_horizon"]))
    return pol, SceneRunner(pol, rt, {"schedule": schedule or CFG["runtime"]["rtc_schedule"],
                                      "max_guidance_weight": CFG["runtime"]["rtc_max_guidance_weight"]})


def _selection(base, n):
    from recovla.eval import scene_trial as T
    from recovla.sim import scene
    seeds_ = list(range(base, base + n))
    lays = [scene.sample_layout(s, "empty", start="home") for s in seeds_]
    return list(zip(seeds_, lays, T.choose_targets(seeds_, lays)))


def _run(rig, runner, trials, condition, ckpt, inducer_kind=None, videos=0):
    """試行を回して記録を outputs/eval/Gcheck/<条件>/ に書く。[(meta, arrays)] を返す。"""
    from recovla.eval import induce as I
    from recovla.eval import scene_trial as T
    from recovla.eval.closed_loop import write_mp4
    out = EVAL_OUT / condition
    if out.exists() and any(out.glob("trial_*.json")):
        raise SystemExit(f"{out} already has trials")
    res = []
    for i, (seed, lay, tgt) in enumerate(trials):
        runner.start_trial(seed)
        ind = I.Inducer(inducer_kind, seed, lay, tgt, rig) if inducer_kind else None
        meta, arr, video = T.run_trial(rig, lay, tgt, runner, {
            "trial": i, "seed": seed, "experiment": "Gcheck", "condition": condition,
            "model": {"name": condition, "checkpoint": str(ckpt)}, "runtime": runner.runtime_record()}, inducer=ind)
        meta["input_check"] = runner.pol.input_check
        T.write_trial(out, i, meta, arr)
        if i < videos:
            write_mp4(out / f"trial_{i:04d}_raw.mp4", video, 20)
        ir = meta["induce"]
        print(f"[g] {condition} {i:2d} seed {seed} {tgt:5s} success {meta['success']} "
              f"induce {ir['kind']} fired {ir['fired']} est {ir['established']} reason {ir['reason']}", flush=True)
        res.append((meta, arr))
    return res


def _first_diff(a, b):
    """行動の列（こまごと）が最初に違うこまと、その行動の番号・差の最大（一致なら None）。"""
    m = min(len(a), len(b))
    bad = [i for i in range(m) if not np.array_equal(a[i], b[i], equal_nan=True)]
    if not bad:
        return None if len(a) == len(b) else {"frame": m, "k": m // 2, "max_abs": None}
    f = bad[0]
    return {"frame": f, "k": f // 2, "max_abs": float(np.nanmax(np.abs(a[f] - b[f])))}


def _write(name, res):
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"check": name, **res, "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    (OUT / f"{name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, ensure_ascii=False, indent=1, default=float))


# ------------------------------------------------------------------ 完了条件 1

def cmd_cond1(a) -> None:
    from recovla.sim import scene
    from recovla.sim.rig import SimRig
    old = json.loads((config.path(CFG["paths"]["outputs"]) / "k1" / "closed_cue.json").read_text(encoding="utf-8"))
    base, n = COND1_SEEDS
    old_rows = {r["seed"]: r for r in old["rows"]}
    trials = [(s, scene.sample_layout(s, "empty", start="home"), old_rows[s]["target"]) for s in range(base, base + n)]
    ckpt = config.ROOT / old["checkpoint"]
    pol, runner = _runner(ckpt, "sync", 50, None)
    rig = SimRig(render=True)
    try:
        res = _run(rig, runner, trials, "cond1_sync_s50", ckpt, videos=a.videos)
    finally:
        rig.close()
    old_dir = config.ROOT / old["out"]
    rows = []
    for i, ((seed, _, tgt), (meta, arr)) in enumerate(zip(trials, res)):
        o = old_rows[seed]
        oarr = np.load(old_dir / f"trial_{o['trial']:04d}.npz")
        m = min(len(arr["fingertip"]), len(oarr["fingertip"]))
        rows.append({"seed": seed, "target": tgt, "old_success": bool(o["success"]), "new_success": bool(meta["success"]),
                     "match": bool(o["success"]) == bool(meta["success"]),
                     "old_t_success_s": o["t_success_s"], "new_t_success_s": meta["steps"][0]["t_success"],
                     "fingertip_max_diff_m": float(np.abs(arr["fingertip"][:m] - oarr["fingertip"][:m]).max()),
                     "first_chunk_action_max_diff": float(np.nanmax(np.abs(arr["action"][:100] - oarr["action"][:100])))})
    _write("cond1", {"old": {"file": "outputs/k1/closed_cue.json", "checkpoint": old["checkpoint"], "out": old["out"],
                             "tf32": old["policy_config"].get("tf32")},
                     "new_runtime": runner.runtime_record(), "n": n, "matches": sum(r["match"] for r in rows),
                     "pass": all(r["match"] for r in rows),
                     "note": "旧評価器は雑音を毎手引く（推論 i は 50·i 手目の雑音）。新しい実行器は noise_generator(種, i)。"
                             "雑音が違うので軌跡の差は原因の説明用で、判定は成否の一致",
                     "rows": rows})


def cmd_cond1_cause(a) -> None:
    """完了条件 1 の不一致の原因の確認: 新しい実行器に旧評価器の雑音の列（試行の種の "noise" の列から毎手 1 つ引き、
    推論 i は 50·i 番目）を与えて同じ 20 回を回し、旧の記録と軌跡がビット一致するかを見る。"""
    import torch
    from recovla.common import seeds
    from recovla.policy import runner as R
    from recovla.sim import scene
    from recovla.sim.rig import SimRig
    old = json.loads((config.path(CFG["paths"]["outputs"]) / "k1" / "closed_cue.json").read_text(encoding="utf-8"))
    base, n = COND1_SEEDS
    old_rows = {r["seed"]: r for r in old["rows"]}
    trials = [(s, scene.sample_layout(s, "empty", start="home"), old_rows[s]["target"]) for s in range(base, base + n)]
    ckpt = config.ROOT / old["checkpoint"]
    pol, runner = _runner(ckpt, "sync", 50, None)
    cfgp = R._base_policy(pol.policy).config
    shape = (1, cfgp.chunk_size, cfgp.max_action_dim)

    def old_stream(seed, i):
        g = torch.Generator().manual_seed(seeds.torch_seed(seeds.seed_sequence(seed, "noise")))
        for _ in range(50 * i):
            torch.randn(shape, generator=g)
        return g
    R.noise_generator = old_stream
    rig = SimRig(render=True)
    try:
        res = _run(rig, runner, trials, "cond1_sync_s50_oldnoise", ckpt)
    finally:
        rig.close()
    old_dir = config.ROOT / old["out"]
    new1 = json.loads((OUT / "cond1.json").read_text(encoding="utf-8"))
    new_rows = {r["seed"]: r for r in new1["rows"]}
    rows = []
    for (seed, _, tgt), (meta, arr) in zip(trials, res):
        o = old_rows[seed]
        oarr = np.load(old_dir / f"trial_{o['trial']:04d}.npz")
        same_len = arr["fingertip"].shape == oarr["fingertip"].shape
        rows.append({"seed": seed, "target": tgt, "old_success": bool(o["success"]),
                     "oldnoise_success": bool(meta["success"]), "newnoise_success": new_rows[seed]["new_success"],
                     "same_length": same_len,
                     "fingertip_equal": bool(same_len and np.array_equal(arr["fingertip"], oarr["fingertip"])),
                     "action_equal": bool(same_len and np.array_equal(arr["action"], oarr["action"], equal_nan=True)),
                     "fingertip_max_diff_m": float(np.abs(arr["fingertip"][:min(len(arr["fingertip"]), len(oarr["fingertip"]))]
                                                          - oarr["fingertip"][:min(len(arr["fingertip"]), len(oarr["fingertip"]))]).max())})
    _write("cond1_cause", {"n": n, "success_matches_old": sum(r["oldnoise_success"] == r["old_success"] for r in rows),
                           "bit_equal_trajectories": sum(r["fingertip_equal"] and r["action_equal"] for r in rows),
                           "rows": rows})


# ------------------------------------------------------------------ 完了条件 2

def _memo_render() -> dict:
    """描画の揺れ（同じ状態を描いても、まれに 1 画素が 1 段違う。GPU の OpenGL）を除くため、同じ物理の状態
    （qpos・qvel のバイト列）には最初に描いた画像を使い回す。完了条件 2 の論理の一致を、描画の揺れと切り分けて見る用。"""
    from recovla.sim.rig import SimRig
    memo = {"hits": 0, "misses": 0, "store": {}}
    orig = SimRig.render

    def render(self, data=None):
        d = self.scratch if data is None else data
        key = d.qpos.tobytes() + d.qvel.tobytes()
        got = memo["store"].get(key)
        if got is None:
            got = orig(self, data)
            memo["store"][key] = got
            memo["misses"] += 1
        else:
            memo["hits"] += 1
        return [g.copy() for g in got]
    SimRig.render = render
    return memo


def cmd_cond2(a) -> None:
    from recovla.sim.rig import SimRig
    memo = _memo_render() if a.memo_render else None
    trials = _selection(*COND2_SEEDS)
    out = {}
    for mode, d in (("sync", None), ("naive", 0)):
        pol, runner = _runner(R1_30K, mode, 10, d)
        rig = SimRig(render=True)
        try:
            out[mode] = _run(rig, runner, trials, f"cond2_{mode}_s10{a.tag}", R1_30K)
        finally:
            rig.close()
        del pol, runner
    rows = []
    for i, ((seed, _, tgt), (ms, xs), (mn, xn)) in enumerate(zip(trials, out["sync"], out["naive"])):
        same_len = xs["action"].shape == xn["action"].shape
        rows.append({"seed": seed, "target": tgt, "sync_success": bool(ms["success"]), "naive_success": bool(mn["success"]),
                     "success_match": bool(ms["success"]) == bool(mn["success"]), "same_length": same_len,
                     "action_equal": bool(same_len and np.array_equal(xs["action"], xn["action"], equal_nan=True)),
                     "fingertip_equal": bool(same_len and np.array_equal(xs["fingertip"], xn["fingertip"])),
                     "action_max_diff": float(np.nanmax(np.abs(xs["action"] - xn["action"]))) if same_len else None,
                     "first_diff": _first_diff(xs["action"], xn["action"])})
    _write("cond2" + a.tag, {"checkpoint": str(R1_30K), "s": 10, "n": len(rows), "note": a.note,
                             "memo_render": None if memo is None else {"hits": memo["hits"], "misses": memo["misses"]},
                     "success_matches": sum(r["success_match"] for r in rows),
                     "action_equal": sum(r["action_equal"] for r in rows),
                     "pass": all(r["success_match"] and r["action_equal"] for r in rows), "rows": rows})


# ------------------------------------------------------------------ 完了条件 4

def cmd_cond4(a) -> None:
    import collections
    from recovla.eval import induce as I
    from recovla.sim import scene
    from recovla.eval import scene_trial as T
    from recovla.sim.rig import SimRig
    base, n = COND4_SEEDS
    seeds_ = list(range(base, base + n))
    lays = [scene.sample_layout(s) for s in seeds_]
    trials = list(zip(seeds_, lays, T.choose_targets(seeds_, lays)))
    pol, runner = _runner(R1_30K, "sync", 10, None)
    rig = SimRig(render=True)
    res = {}
    try:
        for kind in I.KINDS:
            got = _run(rig, runner, trials, f"cond4_{kind}_sync_s10", R1_30K, inducer_kind=kind, videos=a.videos)
            rows = []
            for (seed, lay, tgt), (meta, arr) in zip(trials, got):
                r = meta["induce"]
                rows.append({"seed": seed, "target": tgt, "layout_kind": lay.kind, "fired": r["fired"],
                             "established": r["established"], "reason": r["reason"], "t_fire": r["t_fire"],
                             "success": meta["success"],
                             "induce_active_after_established": bool(
                                 r["established"] and r["t_established"] is not None and
                                 arr["induce_active"][arr["sim_time"] > float(r["t_established"]) + 1e-9].any())})
            est = sum(r["established"] for r in rows)
            res[kind] = {"n": len(rows), "fired": sum(r["fired"] for r in rows), "established": est,
                         "establish_rate": est / len(rows), "success": sum(r["success"] for r in rows),
                         "success_among_established": sum(r["success"] for r in rows if r["established"]),
                         "reasons_not_established": dict(collections.Counter(r["reason"] for r in rows
                                                                              if not r["established"])),
                         "all_recorded": len(rows) == n,
                         "below_min_rate": est / len(rows) < float(CFG["eval"]["induction_min_rate"]),
                         "rows": rows}
    finally:
        rig.close()
    _write("cond4", {"checkpoint": str(R1_30K), "runtime": runner.runtime_record(), "seeds": [base, base + n - 1],
                     "induction_min_rate": CFG["eval"]["induction_min_rate"],
                     "pass": all(v["all_recorded"] for v in res.values()), "results": res})


# ------------------------------------------------------------------ 完了条件 6

def cmd_cond6(a) -> None:
    import torch
    from recovla.policy.runner import _base_policy
    from recovla.sim.rig import SimRig
    pol, runner = _runner(R1_30K, "rtc", 10, a.d)
    base_pol = _base_policy(pol.policy)
    orig = pol.policy.predict_action_chunk
    log = []

    def wrapped(batch, noise=None, **kw):
        chunk = orig(batch, noise=noise.clone(), **kw)
        if "prev_chunk_left_over" in kw:
            with torch.no_grad():
                naive = orig(batch, noise=noise.clone())
            d = int(kw["inference_delay"])
            L = kw["prev_chunk_left_over"][0, :d, :7].float().cpu().numpy()
            r = chunk[0, :d, :7].float().cpu().numpy()
            nv = naive[0, :d, :7].float().cpu().numpy()
            log.append({"d": d, "rtc_err": float(np.abs(r - L).mean()), "naive_err": float(np.abs(nv - L).mean()),
                        "example": {"left_over": kw["prev_chunk_left_over"][0, :, :7].float().cpu().numpy(),
                                    "rtc": chunk[0, :, :7].float().cpu().numpy(),
                                    "naive": naive[0, :, :7].float().cpu().numpy()} if len(log) == a.example else None})
        return chunk
    pol.policy.predict_action_chunk = wrapped
    trials = _selection(*COND6_SEEDS)
    rig = SimRig(render=True)
    try:
        _run(rig, runner, trials, f"cond6_rtc_s10_d{a.d}", R1_30K)
    finally:
        rig.close()
    ratio = np.array([e["rtc_err"] / e["naive_err"] for e in log if e["naive_err"] > 0])
    win = float(np.mean([e["rtc_err"] < e["naive_err"] for e in log]))
    ex = next(e["example"] for e in log if e["example"] is not None)
    fig = _figure(base_pol, a.d, ex)
    _write("cond6", {"checkpoint": str(R1_30K), "runtime": runner.runtime_record(), "d": a.d,
                     "seeds": [COND6_SEEDS[0], COND6_SEEDS[0] + COND6_SEEDS[1] - 1], "inferences": len(log),
                     "rtc_err_median": float(np.median([e["rtc_err"] for e in log])),
                     "naive_err_median": float(np.median([e["naive_err"] for e in log])),
                     "ratio_median": float(np.median(ratio)), "rtc_smaller_share": win,
                     "rule": {"ratio_median_max": COND6_RATIO_MAX, "rtc_smaller_share_min": COND6_WIN_MIN},
                     "pass": bool(np.median(ratio) <= COND6_RATIO_MAX and win >= COND6_WIN_MIN), "figure": fig,
                     "rows": [{k: v for k, v in e.items() if k != "example"} for e in log]})


def _figure(base_pol, d, ex) -> str:
    """RTC の途中経過の 1 例: 前の塊への重み（範囲 E と減衰の 4 組）と、前の塊の残り・rtc・naive の塊（手先 x の増分）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lerobot.configs import RTCAttentionSchedule
    proc = base_pol.model.rtc_processor
    H = ex["rtc"].shape[0]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    keep = proc.rtc_config.prefix_attention_schedule
    for E in (10, 40):
        for sc in ("LINEAR", "EXP"):
            proc.rtc_config.prefix_attention_schedule = RTCAttentionSchedule[sc]
            ax[0].plot(proc.get_prefix_weights(d, E, H).numpy(), label=f"E={E} {sc}")
    proc.rtc_config.prefix_attention_schedule = keep
    ax[0].axvline(d - 0.5, color="gray", ls=":")
    ax[0].set(title=f"prefix weights (d={d})", xlabel="chunk step", ylabel="weight")
    ax[0].legend()
    L = ex["left_over"]
    ax[1].plot(np.arange(len(L)), L[:, 0], "k-", lw=2, label="previous chunk (left over)")
    ax[1].plot(ex["rtc"][:, 0], "C0-", label="rtc")
    ax[1].plot(ex["naive"][:, 0], "C3--", label="naive (same input, same noise)")
    ax[1].axvspan(-0.5, d - 0.5, color="gray", alpha=0.15)
    ax[1].set(title="one inference: action dim 0 (normalized)", xlabel="chunk step", xlim=(-1, 30))
    ax[1].legend()
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "cond6_rtc_example.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return str(p.relative_to(config.ROOT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("cond1")
    s.add_argument("--videos", type=int, default=2)
    sub.add_parser("cond1-cause")
    s = sub.add_parser("cond2")
    s.add_argument("--tag", default="", help="出力の名前に足す（回し直し用）")
    s.add_argument("--note", default=None)
    s.add_argument("--memo-render", action="store_true", help="同じ物理の状態には同じ画像を使う（描画の揺れを除く）")
    s = sub.add_parser("cond4")
    s.add_argument("--videos", type=int, default=2)
    s = sub.add_parser("cond6")
    s.add_argument("--d", type=int, required=True, help="遅れ d の測定で決めた値（0063 の 1）")
    s.add_argument("--example", type=int, default=5, help="図にする推論の番号（前の塊の残りがあるものの中で）")
    a = ap.parse_args(argv)
    {"cond1": cmd_cond1, "cond1-cause": cmd_cond1_cause, "cond2": cmd_cond2, "cond4": cmd_cond4, "cond6": cmd_cond6}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
