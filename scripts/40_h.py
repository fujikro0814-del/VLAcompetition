"""Step H の本番学習（手順書 §9 の 1）。結果は outputs/h/<項目>.json。

    .venv\\Scripts\\python.exe scripts\\40_h.py decide               # K1 での手がかりのずらしの採否（決裁 0054 の 3）
    .venv\\Scripts\\python.exe scripts\\40_h.py train R1 --smoke     # 1000 手で最後まで通す
    .venv\\Scripts\\python.exe scripts\\40_h.py train R1             # 3 万手（5000 手ごとに保存）
    ... train N1 [--smoke]

データは目標の手がかりつき（17 次元、旗は記録のみ）の R1cue・N1cue（outputs/f/data_cue.json、掲示板 0053）。
学習の種は R1・N1 で同じ（configs の train.seed）。行動エキスパートのみ、バッチ 32。学習時の手がかりのずらし
（決裁 0054）は、K1 での採否（outputs/h/cue_aug_decision.json）が「採る」のときだけ入れる。
"""
import argparse
import json
import pathlib
import time

from recovla.common import config

CFG = config.load()
OUT = config.path(CFG["paths"]["outputs"]) / "h"


def write(name: str, obj: dict) -> pathlib.Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.json"
    p.write_text(json.dumps({"check": name, "written": time.strftime("%Y-%m-%d %H:%M:%S"), **obj},
                            ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[h] wrote {p}", flush=True)
    return p


def cmd_train(a) -> None:
    from recovla.policy import train_launcher as tl
    d = json.loads((config.path(CFG["paths"]["outputs"]) / "f" / "data_cue.json").read_text(encoding="utf-8"))
    dec_path = OUT / "cue_aug_decision.json"
    if not dec_path.is_file():
        raise SystemExit(f"{dec_path} がない（K1 での手がかりのずらしの採否を先に決める＝掲示板 0054）")
    dec = json.loads(dec_path.read_text(encoding="utf-8"))
    run = "smoke" if a.smoke else a.name
    rc = CFG["train"]["runs"][a.name]
    steps = 1000 if a.smoke else int(rc["steps"])
    cfg = {"dataset": str(config.path(d["datasets"][a.name]["dataset"])), "train_scope": CFG["train"]["scope"],
           "batch_size": int(CFG["train"]["batch_size"]), "steps": steps,
           "save_freq": steps if a.smoke else int(rc["save_freq"]), "seed": int(CFG["train"]["seed"]),
           "log_freq": int(CFG["train"]["log_freq"]), "num_workers": int(CFG["train"]["num_workers"]),
           "note": f"Step H {a.name}{' smoke' if a.smoke else ''}: {steps} steps on {d['datasets'][a.name]['dataset']} "
                   f"(target cue 17-d, cue_augment {'on' if dec['adopt'] else 'off'} per board 0054)"}
    if dec["adopt"]:
        cfg["cue_augment"] = dict(CFG["train"]["cue_augment"])
    tag = f"{a.name}{'_smoke' if a.smoke else ''}"
    cfg_path = OUT / f"train_{tag}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    OUT.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    t0 = time.time()
    code = tl.run(cfg_path)
    runs = sorted(config.path(CFG["paths"]["train_output"]).glob(f"{cfg_path.stem}_*"))
    rec = json.loads((runs[-1] / tl.RUN_RECORD).read_text(encoding="utf-8")) if runs else {}
    write(f"train_{tag}", {"config": str(cfg_path.relative_to(config.ROOT)), "exit": code,
                           "wall_s": round(time.time() - t0, 1), "cue_augment": dec["adopt"],
                           "run_dir": str(runs[-1].relative_to(config.ROOT)) if runs else None,
                           "log_summary": rec.get("log_summary"), "command": rec.get("command"),
                           "checkpoints": sorted(p.name for p in (runs[-1] / "checkpoints").iterdir()
                                                 if p.is_dir()) if runs else None})


RULE = {"none_min_success": 29, "noise2_success_more_than": 21, "noise2_along_median_less_than_m": 0.0144}


def cmd_decide(a) -> None:
    """手がかりのずらしの採否（決裁 0054 の 3。結果を見る前に固定した決まり）。"""
    import numpy as np
    k1 = config.path(CFG["paths"]["outputs"]) / "k1"
    L = lambda n: json.loads((k1 / f"{n}.json").read_text(encoding="utf-8"))
    res = {}
    for label, before, after in (("none", "closed_cue", "closed_cueaug"), ("1cm", "closed_cue_noise1cm", "closed_cueaug_noise1cm"),
                                 ("2cm", "closed_cue_noise2cm", "closed_cueaug_noise2cm")):
        row = {}
        for side, name in (("without_aug", before), ("with_aug", after)):
            c = L(name)
            al = [r["grasp_err_along_offset_m"] for r in c["rows"] if r.get("grasp_err_along_offset_m") is not None]
            row[side] = {"successes": sum(r["success"] for r in c["rows"]), "trials": len(c["rows"]),
                         "lifted": c["lifted_any"], "wrong": c["lifted_wrong"], "contact": c["contact_trials"],
                         "along_offset_median_m": float(np.median(al)) if al else None, "checkpoint": c["checkpoint"]}
        res[label] = row
    w = {k: v["with_aug"] for k, v in res.items()}
    checks = {"none_success_ge_29": w["none"]["successes"] >= RULE["none_min_success"],
              "noise2_success_gt_21": w["2cm"]["successes"] > RULE["noise2_success_more_than"],
              "noise2_along_median_lt_1.44cm": (w["2cm"]["along_offset_median_m"] is not None
                                                and w["2cm"]["along_offset_median_m"] < RULE["noise2_along_median_less_than_m"])}
    write("cue_aug_decision", {"rule": RULE, "rule_source": "決裁 0054 の 3（結果を見る前に固定）", "results": res,
                               "checks": checks, "adopt": all(checks.values())})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("decide")
    s = sub.add_parser("train")
    s.add_argument("name", choices=["R1", "N1"])
    s.add_argument("--smoke", action="store_true", help="1000 手で最後まで通す")
    a = ap.parse_args(argv)
    {"train": cmd_train, "decide": cmd_decide}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
