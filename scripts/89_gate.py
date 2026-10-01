"""G1〜G3 の監査の集計（0124 の ■ 3）。目標との照合の表の G1〜G3 の数字は、このスクリプトの出力だけを載せる（手で書かない）。

    .venv\\Scripts\\python.exe scripts\\89_gate.py table V2S1 V2SEL2 ...      # outputs/v2eval/<実験> の下の走行ごとの集計（Markdown の表）
    .venv\\Scripts\\python.exe scripts\\89_gate.py table --all                # outputs/v2eval の下の全部
各走行の集計は <走行>/G_AUDIT.json にも書く（recovla.eval.gate.mark）。
"""
import argparse
import json
import sys

from recovla.common import config
from recovla.eval import gate

CFG = config.load()
V2 = config.path(CFG["paths"]["outputs"]) / "v2eval"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("table")
    p.add_argument("experiments", nargs="*")
    p.add_argument("--all", action="store_true")
    p.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    exps = sorted(d.name for d in V2.iterdir() if d.is_dir()) if a.all else a.experiments
    rows = []
    for e in exps:
        for d in sorted(p for p in (V2 / e).iterdir() if p.is_dir()):
            if not any(d.glob("trial_*.json")) and not any(d.glob("run_*.json")):
                continue
            rows.append(gate.mark(d))
    print("| 走行 | 試行 | G1 違反の試行 | G2 違反の試行（停止・早すぎる使用） | G3 違反の試行（件数） | 監査なし | G を満たす |")
    print("|---|---|---|---|---|---|---|")
    for s in rows:
        name = s["dir"].replace("\\", "/").split("/v2eval/")[-1]
        print(f"| {name} | {s['trials']} | {s['g1_trials']} | {s['g2_trials']}（{s['g2_world_stops']}・{s['g2_early_use']}） | "
              f"{s['g3_trials']}（{s['g3_violations']}） | {s['trials_without_audit']} | {'はい' if s['met'] else '**いいえ**'} |")
    tot = {k: sum(s[k] for s in rows) for k in ("trials", "g1_trials", "g2_trials", "g3_trials", "trials_without_audit")}
    print(f"| 計 | {tot['trials']} | {tot['g1_trials']} | {tot['g2_trials']} | {tot['g3_trials']} | {tot['trials_without_audit']} | "
          f"{'はい' if all(s['met'] for s in rows) else '**いいえ**'} |")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"rows": rows, "total": tot}, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
