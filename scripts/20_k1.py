"""Step E（K1：色の見極め）の段取り（手順書 §6）。結果は outputs/k1/<項目>.json。

    .venv\\Scripts\\python.exe scripts\\20_k1.py gen            # 1. 空の箱・既定の開始姿勢の 30 配置 x 3 色 = 90 本を作り、変換・検証
    ... train smoke                                            # 2. 1000 手で最後まで通す
    ... train K1                                               # 2. 8000 手
    ... e6 [--checkpoint DIR]                                   # 3. 指示の差し替え試験（33 配置 x 3 指示 x 5 回）
    ... closed [--checkpoint DIR]                               # 4. 閉ループ 30 回（50 手・同期）

種: K1 の配置 10000〜10029（B_提案書 §9 の K1 の帯）、E6 は選択用 190000〜190032、閉ループは 191000〜191029。
"""
import argparse
import json
import pathlib
import subprocess
import sys
import time

from recovla.common import config

CFG = config.load()
OUT = config.path(CFG["paths"]["outputs"]) / "k1"
K1_SEEDS = range(10000, 10030)
E6_SEEDS = range(190000, 190033)
CLOSED_SEEDS = range(191000, 191030)


def stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def write(name: str, obj: dict) -> pathlib.Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.json"
    p.write_text(json.dumps({"check": name, "written": time.strftime("%Y-%m-%d %H:%M:%S"), **obj},
                            ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[k1] wrote {p}")
    return p


def read(name: str) -> dict:
    p = OUT / f"{name}.json"
    if not p.is_file():
        raise SystemExit(f"{p} がない。先に scripts/20_k1.py {name.split('_')[0]} を実行する")
    return json.loads(p.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ gen

def cmd_gen(a) -> None:
    from recovla.data import convert as C
    from recovla.expert import generate as G
    specs = G.plan_specs({"empty": K1_SEEDS}, start="home")
    run = config.path(CFG["paths"]["outputs"]) / "gen" / f"K1_{stamp()}"
    results = G.generate(specs, run, workers=a.workers, render=True)
    saved = [r["attempts"][-1]["name"] for r in results if r["success"]]
    name = run.name
    mpath = config.path(CFG["paths"]["outputs"]) / "manifests" / f"{name}.json"
    C.write_manifest(mpath, name, [{"run": str(run.relative_to(config.ROOT)).replace("\\", "/"), "key": k} for k in saved],
                     "Step E K1: empty box, home start, layout seeds 10000-10029 x 3 colors, normal demos only")
    ds = config.path(CFG["paths"]["outputs"]) / "datasets" / name
    log = OUT / f"convert_{name}.log"
    OUT.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8") as f:
        c1 = subprocess.run([sys.executable, "-m", "recovla.data.convert", "--manifest", str(mpath), "--out", str(ds),
                             "--name", name], stdout=f, stderr=subprocess.STDOUT)
        c2 = subprocess.run([sys.executable, "-m", "recovla.data.convert", "--verify", str(ds)],
                            stdout=f, stderr=subprocess.STDOUT)
    text = log.read_text(encoding="utf-8", errors="replace")
    fails = [l.strip() for l in text.splitlines() if l.strip().startswith("FAIL")]
    write("gen", {"run": str(run.relative_to(config.ROOT)), "manifest": str(mpath.relative_to(config.ROOT)),
                  "dataset": str(ds.relative_to(config.ROOT)), "specs": len(specs), "saved": len(saved),
                  "first_try_success": sum(r["first_try_success"] for r in results),
                  "dropped": [(r["layout_seed"], r["color"]) for r in results if not r["success"]],
                  "convert_exit": c1.returncode, "verify_exit": c2.returncode,
                  "verify_pass": "[verify] PASS" in text, "verify_fails": fails,
                  "colors": {c: sum(1 for s in specs if s.color == c) for c in ("red", "green", "blue")},
                  "log": str(log.relative_to(config.ROOT))})


# ---------------------------------------------------------------------------------------------- train

def cmd_train(a) -> None:
    from recovla.policy import train_launcher as tl
    g = read("gen")
    run_cfg = CFG["train"]["runs"][a.run]
    cfg = {"dataset": str(config.path(g["dataset"])), "train_scope": CFG["train"]["scope"],
           "batch_size": int(CFG["train"]["batch_size"]), "steps": int(run_cfg["steps"]),
           "save_freq": int(run_cfg.get("save_freq", run_cfg["steps"])), "seed": int(CFG["train"]["seed"]),
           "log_freq": int(CFG["train"]["log_freq"]), "num_workers": int(CFG["train"]["num_workers"]),
           "note": f"Step E K1 ({a.run}): {run_cfg['steps']} steps on {g['dataset']}"}
    cfg_path = OUT / f"train_{a.run}_{stamp()}.json"
    OUT.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    t0 = time.time()
    code = tl.run(cfg_path)
    wall = time.time() - t0
    runs = sorted(config.path(CFG["paths"]["train_output"]).glob(f"{cfg_path.stem}_*"))
    rec = {}
    if runs:
        rec = json.loads((runs[-1] / tl.RUN_RECORD).read_text(encoding="utf-8"))
    write(f"train_{a.run}", {"config": str(cfg_path.relative_to(config.ROOT)), "exit": code, "wall_s": round(wall, 1),
                             "run_dir": str(runs[-1].relative_to(config.ROOT)) if runs else None,
                             "log_summary": rec.get("log_summary"),
                             "checkpoint": str((runs[-1] / "checkpoints" / "last" / "pretrained_model")
                                               .relative_to(config.ROOT)) if runs else None})


def checkpoint_of(a) -> pathlib.Path:
    if a.checkpoint:
        return pathlib.Path(a.checkpoint)
    return config.path(read("train_K1")["checkpoint"])


# ------------------------------------------------------------------------------------------------- e6

E6_SAMPLES = 5


def cmd_e6(a) -> None:
    import collections
    import numpy as np
    from recovla.common import seeds
    from recovla.common.seeds import COLORS
    from recovla.eval.scene_trial import instruction
    from recovla.eval.swap_test import chunk_target, majority
    from recovla.expert.script import PhaseParams
    from recovla.policy.scene_policy import ScenePolicy
    from recovla.record import episode as E
    from recovla.sim import scene
    from recovla.sim.rig import SimRig
    ckpt = checkpoint_of(a)
    pol = ScenePolicy(ckpt)
    rig = SimRig(render=True)
    pp = PhaseParams.from_config()
    rows = []
    try:
        for seed in list(E6_SEEDS)[:a.limit]:
            lay = scene.sample_layout(seed, "empty", start="home")
            rig.reset(lay)
            frame, imgs = E.capture_frame(rig, COLORS[0], 0.0, pp, render=True)
            raw = dict(zip(rig.cameras, imgs))
            cubes_xy = {c: frame["cube_pos"][i, :2].tolist() for i, c in enumerate(COLORS)}
            pol.input_check = None
            for ci, color in enumerate(COLORS):
                samples = []
                for j in range(E6_SAMPLES):
                    gen = pol.torch.Generator().manual_seed(seeds.torch_seed(seeds.seed_sequence(seed, "noise", ci, j)))
                    ch = pol.chunk_for(raw, frame, instruction(color), gen)
                    samples.append(chunk_target(ch, frame["x_des"], cubes_xy))
                votes = collections.Counter(s["nearest"] for s in samples)
                maj = majority([s["nearest"] for s in samples])
                rows.append({"seed": seed, "instruction": color, "votes": dict(votes), "majority": maj,
                             "correct_majority": maj == color,
                             "correct_samples": sum(s["nearest"] == color for s in samples),
                             "mid_drop_samples": sum(s["mid_drop"] for s in samples), "samples": samples})
    finally:
        rig.close()
    n = len(rows)
    by_seed = collections.defaultdict(list)
    for r in rows:
        by_seed[r["seed"]].append(r["majority"])
    dependent = sum(1 for v in by_seed.values() if len(set(v)) > 1 or None in v)
    total_samples = n * E6_SAMPLES
    res = {"checkpoint": str(ckpt), "layouts": len(by_seed), "pairs": n, "samples_per_pair": E6_SAMPLES,
           "accuracy_majority": sum(r["correct_majority"] for r in rows) / n,
           "accuracy_samples": sum(r["correct_samples"] for r in rows) / total_samples,
           "ties": sum(r["majority"] is None for r in rows),
           "instruction_dependence": dependent / len(by_seed),
           "mid_drop_rate": sum(r["mid_drop_samples"] for r in rows) / total_samples,
           "closed_within_chunk_rate": sum(s["closed"] for r in rows for s in r["samples"]) / total_samples,
           "by_instruction": {c: sum(r["correct_majority"] for r in rows if r["instruction"] == c) / (n / 3) for c in COLORS},
           "input_check_last": pol.input_check, "rows": rows}
    write("e6" + a.tag, res)


# --------------------------------------------------------------------------------------------- closed

def cmd_closed(a) -> None:
    import numpy as np
    from recovla.eval import metrics, scene_trial as T
    from recovla.eval.closed_loop import write_mp4
    from recovla.policy.scene_policy import ScenePolicy
    from recovla.sim import scene
    from recovla.sim.rig import SimRig
    ckpt = checkpoint_of(a)
    pol = ScenePolicy(ckpt)
    rig = SimRig(render=True)
    seeds_ = list(CLOSED_SEEDS)[:a.limit]
    lays = [scene.sample_layout(s, "empty", start="home") for s in seeds_]
    targets = T.choose_targets(seeds_, lays)
    out = config.path(CFG["paths"]["outputs"]) / "eval" / f"K1_closed_{stamp()}"
    rows = []
    lift = float(CFG["eval"]["error_lift_m"])
    try:
        for i, (seed, lay, tgt) in enumerate(zip(seeds_, lays, targets)):
            pol.start_trial(seed)
            meta, arr, video = T.run_trial(rig, lay, tgt, pol, {
                "trial": i, "seed": seed, "experiment": "K1_closed_loop", "condition": "K1_sync_n50",
                "model": {"name": "K1", "checkpoint": str(ckpt)},
                "runtime": {"mode": "sync", "exec_interval": 50, "delay_steps": 0, "safety_filter": False}})
            meta["input_check"] = pol.input_check
            meta["inference"] = [{"i": j, "wall_s": s} for j, (s, inferred) in enumerate(pol.timing) if inferred]
            p = T.write_trial(out, i, meta, arr)
            if i < a.videos:
                write_mp4(out / f"trial_{i:04d}_raw.mp4", video, 20)
            m = metrics.trial_metrics(metrics.load_trial(p), CFG["eval"])
            z = arr["cube_pos"][:, :, 2]
            lifted = [c for c, k in zip(("red", "green", "blue"), range(3)) if (z[:, k] - z[0, k]).max() >= lift]
            rows.append({"trial": i, "seed": seed, "target": tgt, "success": m["success"], "lifted": lifted,
                         "lifted_any": bool(lifted), "lifted_wrong": bool(set(lifted) - {tgt}), "error": m["error"],
                         "contacts_n": m["contacts_n"], "stage_reached": m["stage_reached"],
                         "t_success_s": m["t_success_s"]})
            print(f"[k1] closed {i:2d} seed {seed} target {tgt:5s} success {m['success']} lifted {lifted} contacts {m['contacts_n']}")
    finally:
        rig.close()
    n = len(rows)
    lifted_n = sum(r["lifted_any"] for r in rows)
    wrong_n = sum(r["lifted_wrong"] for r in rows)
    write("closed" + a.tag, {"checkpoint": str(ckpt), "out": str(out.relative_to(config.ROOT)), "trials": n,
                     "successes": sum(r["success"] for r in rows), "lifted_any": lifted_n,
                     "lifted_any_rate": lifted_n / n, "lifted_wrong": wrong_n,
                     "wrong_among_lifted": (wrong_n / lifted_n) if lifted_n else None,
                     "contact_trials": sum(r["contacts_n"] > 0 for r in rows),
                     "contact_rate": sum(r["contacts_n"] > 0 for r in rows) / n,
                     "targets": {c: targets.count(c) for c in ("red", "green", "blue")},
                     "inference": pol.timing_summary(), "rows": rows})


COMMANDS = {"gen": cmd_gen, "train": cmd_train, "e6": cmd_e6, "closed": cmd_closed}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("gen")
    s.add_argument("--workers", type=int, default=8)
    s = sub.add_parser("train")
    s.add_argument("run", choices=["smoke", "K1"])
    for name in ("e6", "closed"):
        s = sub.add_parser(name)
        s.add_argument("--checkpoint", default=None, help="既定は train K1 の保存点（checkpoints/last）")
        s.add_argument("--limit", type=int, default=None, help="先頭から何配置・何試行だけ回す（通しの確認用）")
        s.add_argument("--tag", default="", help="結果の名前に付ける（通しの確認用。例 _smoke）")
        if name == "closed":
            s.add_argument("--videos", type=int, default=6, help="動画を書く試行の数（先頭から）")
    a = ap.parse_args(argv)
    COMMANDS[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
