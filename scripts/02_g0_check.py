"""Step C の完了条件 1〜3（判断点 G0）を実行し、結果を outputs/g0/ に JSON で残す。

    .venv\\Scripts\\python.exe scripts\\02_g0_check.py offline     完了条件 1 (a)(b): 出発点モデル・前処理・後処理の読み込み、
                                                                予備実験のチェックポイントの読み込みと 1 回の推論
    .venv\\Scripts\\python.exe scripts\\02_g0_check.py train10     完了条件 1 (c): 起動器の空打ちと 10 手の学習
                                                                （あわせて B_提案書 §10 の統計量の固定と学習率の上書きを確かめる）
    .venv\\Scripts\\python.exe scripts\\02_g0_check.py replay      完了条件 2: 複写した raw 1 本の再生確認（手先 5 mm）
    .venv\\Scripts\\python.exe scripts\\02_g0_check.py eval        完了条件 3: 学習配置 20 回（種 100000〜100019）

どれもネットワークなし（HF_HUB_OFFLINE=1、HF_HOME=<ROOT>\\models\\hf_home）で動かす。tests/test_c_port.py が
ここで書いた JSON を読んで判定する。出力先が既にあれば止まる（上書きしない）。
"""
import os
import sys

from recovla.common import config

CFG = config.load("g0")
G0 = CFG["g0"]
# lerobot・huggingface_hub を import する前に決める（import 時に読まれる）
os.environ["HF_HOME"] = str(config.path(CFG["paths"]["models_home"]))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import datetime as dt  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

OUT = config.path(CFG["paths"]["outputs"]) / "g0"
CHECKPOINT = config.path(G0["checkpoint"])
LEGACY_EVAL = config.path(G0["legacy_eval"])
RAW_EPISODE = config.path(G0["raw_episode"])
SNAPSHOT = config.path(CFG["paths"]["policy_snapshot"])


def write_json(path: pathlib.Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"[g0] wrote {path}")


def stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def hf_home_vla_paths() -> list:
    """HF_HOME の下の文字のファイルに、C:\\VLA を指すパスが残っていないか（完了条件 1）。"""
    hits = []
    pat = re.compile(r"(?i)C:[\\/]+VLA")
    for p in pathlib.Path(os.environ["HF_HOME"]).rglob("*"):
        if p.is_file() and p.suffix.lower() in (".json", ".txt", ".md", ".yaml", ".yml", ".jinja", ".py"):
            if pat.search(p.read_text(encoding="utf-8", errors="replace")):
                hits.append(str(p))
    return hits


# ----------------------------------------------------------------------------------- offline

def cmd_offline(a) -> int:
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor import PolicyProcessorPipeline

    from recovla.eval import closed_loop as cle
    from recovla.record import recorder

    report = {"at": dt.datetime.now().isoformat(timespec="seconds"), "HF_HOME": os.environ["HF_HOME"],
              "HF_HUB_OFFLINE": os.environ["HF_HUB_OFFLINE"], "hf_home_vla_paths": hf_home_vla_paths()}

    # (a) 出発点モデルと前処理・後処理（SmolVLM2 は方策の中から名前で呼ばれ、同じ HF_HOME から解決される）
    t0 = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(str(SNAPSHOT))
    pre = PolicyProcessorPipeline.from_pretrained(str(SNAPSHOT), config_filename="policy_preprocessor.json")
    post = PolicyProcessorPipeline.from_pretrained(str(SNAPSHOT), config_filename="policy_postprocessor.json")
    report["start_model"] = {"path": str(SNAPSHOT), "load_s": time.perf_counter() - t0,
                             "vlm_model_name": policy.config.vlm_model_name,
                             "n_params": int(sum(p.numel() for p in policy.parameters())),
                             "preprocessor_steps": [type(s).__name__ for s in pre.steps],
                             "postprocessor_steps": [type(s).__name__ for s in post.steps], "ok": True}
    del policy, pre, post
    torch.cuda.empty_cache()

    # (b) 予備実験のチェックポイントを新しい評価器で読み、1 回推論する
    pa = cle.PolicyActions(CHECKPOINT, a.device)
    rig = cle.EvalRig()
    try:
        placement = {p.placement_id: p for p in recorder.training_placements()}[0]
        rig.reset(placement)
        pa.start_trial(int(G0["first_seed"]))
        with contextlib.redirect_stdout(io.StringIO()):
            frame, raw = rig.sampler.capture(rig.data, 0)
        action = pa(0, frame, dict(zip(cle.control.CAMERAS, raw)))
        report["checkpoint"] = {"path": str(CHECKPOINT), "input_check": pa.input_check,
                                "first_action": [float(v) for v in action], "timing": pa.timing_summary(),
                                "ok": bool(np.all(np.isfinite(action)) and action.shape == (7,))}
    finally:
        rig.close()
    report["ok"] = bool(report["start_model"]["ok"] and report["checkpoint"]["ok"] and not report["hf_home_vla_paths"])
    write_json(OUT / "offline_check.json", report)
    return 0 if report["ok"] else 1


# ----------------------------------------------------------------------------------- replay

def cmd_replay(a) -> int:
    import mujoco

    from recovla.record import replay
    from recovla.sim import control, render

    ep = replay.load_episode(RAW_EPISODE)
    model = mujoco.MjModel.from_xml_path(control.SCENE_PATH)
    renderer = mujoco.Renderer(model, control.IMAGE_SIZE, control.IMAGE_SIZE)
    report = {"at": dt.datetime.now().isoformat(timespec="seconds"), "episode": str(RAW_EPISODE),
              "n_frames": ep.n_frames, "tolerance_m": replay.WINDOW_TOL_M, "modes": {}}
    try:
        for mode in replay.MODES:
            with contextlib.redirect_stdout(io.StringIO()):
                rep = replay.replay(ep, mode, renderer, model)
            r = replay.compare(ep, rep, images=mode in ("step", "window"))
            report["modes"][mode] = r
            print(f"[g0] replay {mode:9s} ee max {r['ee_err_max_m'] * 1000:.3f} mm, bit-exact {r['bit_exact_state']}")
    finally:
        render.close_renderer(renderer)
    w = report["modes"]["window"]
    report["window_pass"] = bool(w["ee_err_max_m"] < replay.WINDOW_TOL_M)
    report["negative_controls_detected"] = bool(all(report["modes"][m]["ee_err_max_m"] >= replay.WINDOW_TOL_M
                                                    for m in ("last_step", "shifted")))
    report["ok"] = report["window_pass"] and report["negative_controls_detected"]
    write_json(OUT / "replay_check.json", report)
    return 0 if report["ok"] else 1


# ----------------------------------------------------------------------------------- eval

def cmd_eval(a) -> int:
    from recovla.eval import closed_loop as cle

    out = OUT / f"eval_{stamp()}"
    args = ["run", "--checkpoint", str(CHECKPOINT), "--out", str(out), "--device", a.device,
            "--placement-ids", *map(str, G0["placement_ids"]), "--repeats", str(G0["repeats"]),
            "--first-seed", str(G0["first_seed"])]
    t0 = time.perf_counter()
    code = cle.main(args)
    wall = time.perf_counter() - t0
    new = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    old = json.loads((LEGACY_EVAL / "summary.json").read_text(encoding="utf-8"))
    rows = []
    for n in new["trials"]:
        o = next(t for t in old["trials"] if t["trial"] == n["trial"])
        tj = json.loads((out / f"trial_{n['trial']:04d}.json").read_text(encoding="utf-8"))
        oj = json.loads((LEGACY_EVAL / f"trial_{n['trial']:04d}.json").read_text(encoding="utf-8"))
        rows.append({"trial": n["trial"], "seed": n["seed"], "placement_id": n["placement_id"],
                     "yaw_deg": tj["placement"]["yaw_deg"],
                     "same_seed_and_placement": (n["seed"], n["placement_id"]) == (o["seed"], o["placement_id"])
                     and tj["placement"] == oj["placement"],
                     "success_new": n["success"], "success_legacy": o["success"],
                     "success_time_new": n["success_time_s"], "success_time_legacy": o["success_time_s"],
                     "input_check_ok": bool(tj["input_check"]
                                            and all(tj["input_check"][v]["present"] and tj["input_check"][v]["equals_view"]
                                                    for v in ("overhead", "wrist"))),
                     "inference_mean_s": tj["inference_timing"]["inference_mean_s"],
                     "inference_max_s": tj["inference_timing"]["inference_max_s"]})
    inf = [s for n in new["trials"]
           for s in json.loads((out / f"trial_{n['trial']:04d}.json").read_text(encoding="utf-8"))["inference_timing"]["inference_s"]]
    first_proc = inf[0] if inf else None
    steady = inf[1:]
    report = {"at": dt.datetime.now().isoformat(timespec="seconds"), "out": str(out), "exit_code": code,
              "wall_s": wall, "successes_new": sum(r["success_new"] for r in rows),
              "successes_legacy": sum(r["success_legacy"] for r in rows), "trials": len(rows),
              "min_successes": int(G0["min_successes"]),
              "agree": sum(r["success_new"] == r["success_legacy"] for r in rows),
              "all_same_seed_and_placement": all(r["same_seed_and_placement"] for r in rows),
              "all_input_check_ok": all(r["input_check_ok"] for r in rows),
              "inference": {"n": len(inf), "first_in_process_s": first_proc,
                            "mean_s_excluding_first": float(np.mean(steady)) if steady else None,
                            "median_s_excluding_first": float(np.median(steady)) if steady else None,
                            "max_s_excluding_first": float(np.max(steady)) if steady else None},
              "rows": rows}
    report["ok"] = bool(code == 0 and report["successes_new"] >= report["min_successes"]
                        and report["all_input_check_ok"] and report["all_same_seed_and_placement"])
    write_json(OUT / "eval_check.json", report)
    return 0 if report["ok"] else 1


# ----------------------------------------------------------------------------------- train10

def cmd_train10(a) -> int:
    from safetensors.numpy import load_file

    from recovla.data import convert
    from recovla.policy import train_launcher as tl

    work = OUT / f"train10_{stamp()}"
    ds = work / "datasets" / "g0_one_episode"
    with contextlib.redirect_stdout(io.StringIO()):
        convert.convert([RAW_EPISODE], ds, ds.name)
    verified = convert.verify(ds, None, work / "verify_raw_vs_policy.png")

    # B_提案書 §10: 学習に渡すデータセットの写しの meta/stats.json を別の統計量に差し替え、学習後の保存点の
    # 正規化・逆正規化の統計量がその値とビット単位で一致すること（= meta/stats.json から読まれ、固定できる）
    fixed = work / "datasets" / "g0_one_episode_statsX"
    shutil.copytree(ds, fixed)
    stats = json.loads((fixed / "meta" / "stats.json").read_text(encoding="utf-8"))
    for key in ("observation.state", "action"):
        mean = np.asarray(stats[key]["mean"], dtype=np.float64)
        std = np.asarray(stats[key]["std"], dtype=np.float64)
        stats[key]["mean"] = (mean + 0.5 * std + 0.125).tolist()
        stats[key]["std"] = (std * 1.5 + 0.25).tolist()
    (fixed / "meta" / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    lr = CFG["train"]["runs"]["second_round"]
    extra = [f"--policy.optimizer_lr={lr['optimizer_lr']}", "--policy.scheduler_warmup_steps=2",
             "--policy.scheduler_decay_steps=10", f"--policy.scheduler_decay_lr={lr['decay_lr']}"]
    cfg = {"dataset": str(fixed), "train_scope": CFG["train"]["scope"], "batch_size": int(CFG["train"]["batch_size"]),
           "steps": 10, "save_freq": 10, "seed": int(CFG["train"]["seed"]), "log_freq": 1, "extra_args": extra,
           "note": "Step C 完了条件 1 (c): 10 手の学習。統計量の固定と学習率の上書きの確認（B_提案書 §10）"}
    cfg_path = work / "g0_train10.json"
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    dry = tl.run(cfg_path, work / "train", work / "logs", dry_run=True)
    code = tl.run(cfg_path, work / "train", work / "logs")
    runs = sorted((work / "train").glob("g0_train10_*"))
    report = {"at": dt.datetime.now().isoformat(timespec="seconds"), "work": str(work), "verify_ok": verified,
              "dry_run_exit": dry, "train_exit": code, "runs": [str(r) for r in runs]}
    if runs:
        run = runs[-1]
        rec = json.loads((run / tl.RUN_RECORD).read_text(encoding="utf-8"))
        ckpt = run / "checkpoints" / "000010" / "pretrained_model"
        norm = load_file(str(ckpt / "policy_preprocessor_step_5_normalizer_processor.safetensors"))
        unnorm = load_file(str(ckpt / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
        stat_check = {}
        for key, tensors in (("observation.state", (norm,)), ("action", (norm, unnorm))):
            for field in ("mean", "std"):
                want = np.asarray(stats[key][field], dtype=np.float32)
                got = [t[f"{key}.{field}"] for t in tensors if f"{key}.{field}" in t]
                stat_check[f"{key}.{field}"] = bool(got) and all(np.array_equal(g.astype(np.float32).reshape(-1), want)
                                                                 for g in got)
        log = pathlib.Path(rec["log"]).read_text(encoding="utf-8", errors="replace")
        lrs = [float(v) for v in re.findall(r" lr:([0-9.eE+-]+)", log)]
        # 学習率: ログは各手の scheduler.step() の後の値を 2 桁で出す（ピークは記録点の間に来る）。同じ設定で
        # LeRobot の予定を作り直し、各手の後の値を同じ桁に丸めたものと比べる
        import torch
        from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
        opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=float(lr["optimizer_lr"]))
        sched = CosineDecayWithWarmupSchedulerConfig(num_warmup_steps=2, num_decay_steps=10,
                                                     peak_lr=float(lr["optimizer_lr"]),
                                                     decay_lr=float(lr["decay_lr"])).build(opt, 10)
        expected = []
        for _ in range(10):
            opt.step()
            sched.step()
            expected.append(float(f"{opt.param_groups[0]['lr']:0.1e}"))
        report.update({
            "log_summary": rec.get("log_summary"), "checkpoint": str(ckpt), "checkpoint_exists": ckpt.is_dir(),
            "stats_bit_equal_to_replaced": stat_check, "stats_ok": all(stat_check.values()),
            "logged_lr": lrs, "expected_lr_after_each_step": expected, "lr_ok": lrs == expected,
            "lr_settings": {"peak": float(lr["optimizer_lr"]), "decay_lr": float(lr["decay_lr"]),
                            "warmup_steps": 2, "decay_steps": 10}, "gpu_mem_allocated_max_gib":
                (rec.get("log_summary") or {}).get("gpu_mem_allocated_max_gib")})
    report["ok"] = bool(verified and dry == 0 and code == 0 and report.get("checkpoint_exists")
                        and report.get("stats_ok") and report.get("lr_ok"))
    write_json(OUT / "train10_check.json", report)
    return 0 if report["ok"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=("offline", "train10", "replay", "eval"))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    return {"offline": cmd_offline, "train10": cmd_train10, "replay": cmd_replay, "eval": cmd_eval}[a.what](a)


if __name__ == "__main__":
    sys.exit(main())
