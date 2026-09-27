"""本線（naive d=4）の接触率の測り直しと、安全フィルタを作るかの判定（決裁 0079 の 2）。決まりは結果を見る前にここに固めた。

    .venv\\Scripts\\python.exe scripts\\49_contact.py run       # 41_results.py run を 1 本（G2 と同じ種・同じ並び）
    .venv\\Scripts\\python.exe scripts\\49_contact.py decide    # outputs/results/contact_decision.json

試行: R1 の 3 万手、本線の設定（configs の runtime: naive・s=10・d=4・刻み 10）、`selection:194000:99`
（G2 の自然の R1_nat と同じ種・同じ並び＝194000〜194098 の 99 回）。出力 outputs/eval/CONTACT/R1_nat_naive/
決め方（0079 の 2、原文のとおり）:
  - 接触率が 5% 以上なら安全フィルタを作る（作る前に設計の案を出して止まる）。5% 未満なら作らない
  - 接触の数え方は G2 と同じ: 試行ごとの metrics.contacts_n（目標以外の立方体と箱の壁への、偽→真の回数）が 1 以上の
    試行の割合。分母は 99 回すべて（成功しなかった試行も含める）
  - 「5% 以上」は 99 回中 5 回（5.05%）以上。4 回（4.04%）なら作らない
あわせて出すもの（判定には使わない）: 自然の成功数、G2（rtc 範囲 40）との種ごとの対（接触・成功、McNemar の正確な p）、
接触のあった試行の相手（壁・立方体）と時刻、Wilson の 95% 区間
"""
import argparse
import json
import math
import subprocess
import sys
import time

import numpy as np

from recovla.common import config

CFG = config.load()
OUT = config.path(CFG["paths"]["outputs"])
CKPT = r"outputs\train\train_R1_20260926-153959_20260926-153959\checkpoints\030000\pretrained_model"
SPEC = "selection:194000:99"
COND = "R1_nat_naive"
G2_DIR = OUT / "eval" / "G2" / "R1_nat"
NEW_DIR = OUT / "eval" / "CONTACT" / COND
THRESH = 0.05


def cmd_run(a) -> None:
    rt = CFG["runtime"]
    if rt["mode"] != "naive":
        raise SystemExit(f"configs runtime.mode is {rt['mode']!r}, expected the main setting 'naive'")
    cmd = [sys.executable, "scripts/41_results.py", "run", "--experiment", "CONTACT", "--condition", COND,
           "--checkpoint", CKPT, "--model", "R1", "--mode", rt["mode"], "--s", str(rt["exec_interval"]),
           "--d", str(rt["delay_steps"]), "--trials", SPEC, "--videos", "3"]
    print(" ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=config.ROOT))


def _rows(d):
    from recovla.eval import metrics as M
    out = {}
    for p in sorted(d.glob("trial_*.json")):
        rec = M.load_trial(p)
        m = rec.meta
        n = M._contacts_n(rec, np.asarray(rec.arrays["target"]))
        out[(m["seed"], m["steps"][0]["target"])] = {"trial": m["trial"], "success": bool(m["success"]), "contacts_n": n,
                                                     "events": _events(rec) if n else []}
    return out


def _events(rec):
    from recovla.common.seeds import COLORS
    cr = np.asarray(rec.arrays["contact_robot"], bool)
    tgt = np.asarray(rec.arrays["target"])
    t = np.asarray(rec.arrays["sim_time"], float)
    ev = []
    for k, name in enumerate(rec.meta.get("obstacles") or []):
        if name.startswith("wall_"):
            col = cr[:, k]
        elif name.startswith("cube_") and name[5:] in COLORS:
            col = cr[:, k] & (tgt != COLORS.index(name[5:]))
        else:
            continue
        idx = np.flatnonzero(col)
        if idx.size:
            ev.append({"with": name, "t_first": float(t[idx[0]]), "frames": int(idx.size)})
    return ev


def _wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [c - h, c + h]


def _mcnemar(b, c):
    from scipy.stats import binomtest
    return float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0


def cmd_decide(a) -> None:
    new, old = _rows(NEW_DIR), _rows(G2_DIR)
    if len(new) != 99:
        raise SystemExit(f"{NEW_DIR}: {len(new)} trials, expected 99")
    k = sum(r["contacts_n"] > 0 for r in new.values())
    rate = k / len(new)
    keys = sorted(set(new) & set(old))
    pair = {}
    for name, f in (("contact", lambda r: r["contacts_n"] > 0), ("success", lambda r: r["success"])):
        b = sum(f(new[x]) and not f(old[x]) for x in keys)
        c = sum(f(old[x]) and not f(new[x]) for x in keys)
        pair[name] = {"naive_only": b, "rtc40_only": c, "both": sum(f(new[x]) and f(old[x]) for x in keys),
                      "mcnemar_exact_p": _mcnemar(b, c)}
    res = {"rule": {"threshold": THRESH, "denominator": "99 回すべて", "count": "metrics.contacts_n > 0（G2 と同じ）",
                    "source": "0079 の 2"},
           "spec": SPEC, "runtime": CFG["runtime"]["mode"],
           "naive": {"n": len(new), "contact_trials": k, "contact_rate": rate, "wilson95": _wilson(k, len(new)),
                     "successes": sum(r["success"] for r in new.values())},
           "g2_rtc40": {"n": len(old), "contact_trials": sum(r["contacts_n"] > 0 for r in old.values()),
                        "successes": sum(r["success"] for r in old.values())},
           "paired_n": len(keys), "paired": pair,
           "contact_rows_naive": [{"seed": s, "target": t, **r} for (s, t), r in sorted(new.items()) if r["contacts_n"]],
           "contact_rows_g2": [{"seed": s, "target": t, **r} for (s, t), r in sorted(old.items()) if r["contacts_n"]],
           "build_safety_filter": rate >= THRESH, "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    p = OUT / "results" / "contact_decision.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k2: v for k2, v in res.items() if not k2.startswith("contact_rows")}, ensure_ascii=False,
                     indent=1, default=float))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    sub.add_parser("decide")
    a = ap.parse_args(argv)
    {"run": cmd_run, "decide": cmd_decide}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
