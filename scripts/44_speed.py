"""推論を速めて遅れ d を下げる手の見積もり（決裁 0070 の 2 の (2)。採用はしない）。

    .venv\\Scripts\\python.exe scripts\\44_speed.py collect   # R1 を評価の設定で走らせ、推論の入力 100 組（前処理の後の batch・雑音・前の塊の残り）と出力を保存
    .venv\\Scripts\\python.exe scripts\\44_speed.py bench     # 各案を同じ 100 組に当て、推論時間と行動の塊の差を測る

- 固定した入力: Step G の検査の帯 198700〜（空の箱・既定の開始姿勢、種ごとに目標 1 つ）で R1 の 3 万手を評価の設定
  （rtc・s=10・d=4・範囲 40・指数）で走らせ、最初の 1 回（ウォームアップ）を除いた推論を先頭から 100 組
- 時間: 方策の推論（predict_action_chunk、torch.cuda.synchronize 込み）の実時間。前処理（約 2.6 ms）は collect で測った平均を足す。
  d = ceil(p95 / 0.1 s)（runtime.delay_rule）。基準の案（今の推論をそのまま同じ入力で回し直す）の p95 も並べ、
  閉ループで測った d の p95（0.396 s）との違いを示す
- 差: 同じ入力・同じ雑音で、今の推論の出力（collect のときの出力）との差。正規化された空間の行動 7 次元と、後処理の後の値
  （手先の差分 [m]・グリッパ）の両方で、塊全体（50 手）と最初の 10 手（s）について、100 組の平均と最大
- 出力を変える手: 刻み数（num_steps）8・5・3、注意を SDPA に、行列積の TF32、cuDNN の benchmark。
  出力を変えない手: 推論の経路のコンパイル（triton がないので inductor は使えない。cudagraphs の後ろ側だけ試す）
"""
import argparse
import json
import math
import os
import pathlib
import time

import numpy as np

from recovla.common import config

CFG = config.load()
os.environ["HF_HOME"] = str(config.path(CFG["paths"]["models_home"]))
os.environ["HF_HUB_OFFLINE"] = "1"

OUT = config.path(CFG["paths"]["outputs"]) / "speed"
R1_30K = config.ROOT / "outputs/train/train_R1_20260926-153959_20260926-153959/checkpoints/030000/pretrained_model"
SEED0 = 198700
N_INPUTS = 100
REPEAT = 3                      # 1 組あたりの計測の回数（p95 は 100 × 3 回から）


def _to(x, dev):
    import torch
    if isinstance(x, torch.Tensor):
        return x.to(dev)
    if isinstance(x, dict):
        return {k: _to(v, dev) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to(v, dev) for v in x)
    return x


def _runner():
    from recovla.policy.runner import SceneRunner
    from recovla.policy.schedule import RuntimeConfig
    from recovla.policy.scene_policy import ScenePolicy
    rt = CFG["runtime"]
    pol = ScenePolicy(R1_30K)
    return pol, SceneRunner(pol, RuntimeConfig("rtc", int(rt["exec_interval"]), int(rt["delay_steps"]),
                                               execution_horizon=int(rt["rtc_guidance_horizon"])),
                            {"schedule": rt["rtc_schedule"], "max_guidance_weight": rt["rtc_max_guidance_weight"]})


# ------------------------------------------------------------------ collect

def cmd_collect(a) -> None:
    import torch
    from recovla.eval import scene_trial as T
    from recovla.sim import scene
    from recovla.sim.rig import SimRig
    pol, runner = _runner()
    orig = pol.policy.predict_action_chunk
    got, seen = [], {"n": 0}

    def wrapped(batch, noise=None, **kw):
        out = orig(batch, noise=noise, **kw)
        seen["n"] += 1
        if seen["n"] > 1 and len(got) < N_INPUTS:            # 最初の 1 回（ウォームアップ）は除く
            got.append({"batch": _to(batch, "cpu"), "noise": noise.cpu(), "kw": _to(kw, "cpu"),
                        "out": out.detach().float().cpu()})
        return out
    pol.policy.predict_action_chunk = wrapped
    rig = SimRig(render=True)
    seeds_used, pre = [], []
    try:
        s = SEED0
        while len(got) < N_INPUTS:
            lay = scene.sample_layout(s, "empty", start="home")
            tgt = T.choose_targets([s], [lay])[0]
            runner.start_trial(s)
            T.run_trial(rig, lay, tgt, runner, {"trial": len(seeds_used), "seed": s, "experiment": "speed",
                                                "condition": "collect"})
            pre += [e["wall_breakdown_s"]["preprocess"] for e in runner.trace()["inference"]]
            seeds_used.append(s)
            s += 1
    finally:
        rig.close()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(got, OUT / "inputs.pt")
    info = {"seeds_used": [seeds_used[0], seeds_used[-1]], "n_inputs": len(got),
            "with_left_over": sum("prev_chunk_left_over" in g["kw"] for g in got),
            "preprocess_mean_s": float(np.mean(pre)), "checkpoint": str(R1_30K), "runtime": runner.runtime_record(),
            "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    (OUT / "inputs.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=1))


# ------------------------------------------------------------------ variants

def _sdpa_attention(self, attention_mask, batch_size, head_dim, query_states, key_states, value_states):
    """eager_attention_forward と同じ意味の注意を scaled_dot_product_attention で（数値の丸めは変わる）。"""
    import torch
    h, kvh = self.num_attention_heads, self.num_key_value_heads
    g = h // kvh
    L = key_states.shape[1]
    k = key_states[:, :, :, None, :].expand(batch_size, L, kvh, g, head_dim).reshape(batch_size, L, h, head_dim)
    v = value_states[:, :, :, None, :].expand(batch_size, L, kvh, g, head_dim).reshape(batch_size, L, h, head_dim)
    q = query_states.to(torch.float32).transpose(1, 2)
    k = k.to(torch.float32).transpose(1, 2)
    v = v.to(torch.float32).transpose(1, 2)
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask[:, None, :, :])
    o = o.to(value_states.dtype).permute(0, 2, 1, 3)
    return o.reshape(batch_size, -1, h * head_dim)


def _variants():
    """(名前, 出力を変えるか, 準備, 片付け)。準備・片付けは (pol, base) を受け取る。"""
    import torch
    from recovla.policy.runner import _base_policy  # noqa: F401

    def steps(n):
        def on(pol, base):
            base.config.num_steps = n

        def off(pol, base):
            base.config.num_steps = 10
        return on, off

    def sdpa_on(pol, base):
        vw = base.model.vlm_with_expert
        vw._orig_attn = vw.get_attention_interface
        vw.get_attention_interface = lambda: (lambda *args: _sdpa_attention(vw, *args))

    def sdpa_off(pol, base):
        vw = base.model.vlm_with_expert
        vw.get_attention_interface = vw._orig_attn

    def tf32_on(pol, base):
        torch.backends.cuda.matmul.allow_tf32 = True

    def tf32_off(pol, base):
        torch.backends.cuda.matmul.allow_tf32 = False

    def bench_on(pol, base):
        torch.backends.cudnn.benchmark = True

    def bench_off(pol, base):
        torch.backends.cudnn.benchmark = False

    def compile_on(pol, base):
        m = base.model
        m._orig_denoise = m.denoise_step
        m.denoise_step = torch.compile(m.denoise_step, backend="cudagraphs", dynamic=False)

    def compile_off(pol, base):
        m = base.model
        m.denoise_step = m._orig_denoise
        torch._dynamo.reset()

    nop = (lambda p, b: None, lambda p, b: None)
    s8, s5, s3 = steps(8), steps(5), steps(3)
    return [("ref", False, *nop), ("steps8", True, *s8), ("steps5", True, *s5), ("steps3", True, *s3),
            ("sdpa", True, sdpa_on, sdpa_off), ("tf32", True, tf32_on, tf32_off),
            ("cudnn_benchmark", True, bench_on, bench_off), ("compile_cudagraphs", False, compile_on, compile_off),
            ("ref_again", False, *nop)]


def cmd_bench(a) -> None:
    import torch
    from recovla.policy.runner import _base_policy
    from recovla.policy.runner import enable_rtc
    from recovla.policy.scene_policy import ScenePolicy
    rt = CFG["runtime"]
    inputs = torch.load(OUT / "inputs.pt", weights_only=False)[:a.n]
    info = json.loads((OUT / "inputs.json").read_text(encoding="utf-8"))
    info["n_used"] = len(inputs)
    pol = ScenePolicy(R1_30K)
    enable_rtc(pol.policy, int(rt["rtc_guidance_horizon"]), rt["rtc_schedule"], rt["rtc_max_guidance_weight"])
    base = _base_policy(pol.policy)
    dev = pol.device
    ref = torch.stack([g["out"][0] for g in inputs])                       # (100, 50, A)
    ref_post = np.stack([pol.post(g["out"].to(dev)).detach().cpu().numpy().reshape(-1, g["out"].shape[-1])[:, :7]
                         for g in inputs])
    s = int(rt["exec_interval"])
    res = {}
    names = a.only.split(",") if a.only else None
    for name, changes, on, off in _variants():
        if names and name not in names:
            continue
        row = {"changes_output": changes}
        try:
            on(pol, base)
            outs, walls = [], []
            for w in range(3):                                         # ウォームアップ
                g = inputs[w]
                with torch.no_grad():
                    pol.policy.predict_action_chunk(_to(g["batch"], dev), noise=g["noise"].to(dev), **_to(g["kw"], dev))
            for r in range(a.repeat):
                for g in inputs:
                    b, nz, kw = _to(g["batch"], dev), g["noise"].to(dev), _to(g["kw"], dev)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        o = pol.policy.predict_action_chunk(b, noise=nz, **kw)
                    torch.cuda.synchronize()
                    walls.append(time.perf_counter() - t0)
                    if r == 0:
                        outs.append(o.detach().float().cpu()[0])
            out = torch.stack(outs)
            post = np.stack([pol.post(o[None].to(dev)).detach().cpu().numpy().reshape(-1, o.shape[-1])[:, :7] for o in outs])
            dn = (out[:, :, :7] - ref[:, :, :7]).abs().numpy()
            dp = np.abs(post - ref_post)
            w = np.array(walls)
            total_p95 = float(np.percentile(w, 95)) + info["preprocess_mean_s"]
            row.update({
                "policy_mean_s": float(w.mean()), "policy_p95_s": float(np.percentile(w, 95)), "policy_max_s": float(w.max()),
                "total_p95_s": total_p95, "d": int(math.ceil(total_p95 / float(rt["delay_rule"]["action_dt_s"]))),
                "diff_norm": {"chunk_mean": float(dn.mean()), "chunk_max": float(dn.max()),
                              "first_s_mean": float(dn[:, :s].mean()), "first_s_max": float(dn[:, :s].max())},
                "diff_post_xyz_m": {"chunk_mean": float(dp[:, :, :3].mean()), "chunk_max": float(dp[:, :, :3].max()),
                                    "first_s_mean": float(dp[:, :s, :3].mean()), "first_s_max": float(dp[:, :s, :3].max())},
                "diff_post_grip": {"chunk_mean": float(dp[:, :, 6].mean()), "chunk_max": float(dp[:, :, 6].max()),
                                   "sign_flips": int(np.sum(np.sign(post[:, :, 6]) != np.sign(ref_post[:, :, 6])))},
                "bit_equal_inputs": int(sum(bool(torch.equal(o, rr)) for o, rr in zip(outs, ref)))})
        except Exception as e:                                         # 使えない手はその旨を記録する
            row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        finally:
            try:
                off(pol, base)
            except Exception as e:
                row["cleanup_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        res[name] = row
        print(name, json.dumps({k: v for k, v in row.items() if not isinstance(v, dict)}, ensure_ascii=False), flush=True)
    res["_info"] = {**info, "repeat": a.repeat, "delay_rule": rt["delay_rule"],
                    "not_measured": {"compile_inductor": "triton がない（Windows）。triton-windows の追加は環境の変更で承認が要る"},
                    "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    (OUT / f"bench{a.tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("collect")
    s = sub.add_parser("bench")
    s.add_argument("--only", default=None, help="案の名前（, 区切り）")
    s.add_argument("--n", type=int, default=N_INPUTS, help="使う入力の組数（先頭から）")
    s.add_argument("--repeat", type=int, default=REPEAT)
    s.add_argument("--tag", default="", help="結果のファイル名に足す")
    a = ap.parse_args(argv)
    {"collect": cmd_collect, "bench": cmd_bench}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
