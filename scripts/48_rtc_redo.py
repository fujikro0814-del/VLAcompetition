"""本線の非同期の設定（RTC の有無と範囲）の決め直し（決裁 0074 への回答＝0075 の 1）。決まりは結果を見る前にここに固めた（2026-09-27）。

    .venv\\Scripts\\python.exe scripts\\48_rtc_redo.py plan      # 回す 12 本の 41_results.py run の引数を表示
    .venv\\Scripts\\python.exe scripts\\48_rtc_redo.py decide    # outputs/results/rtc_redo_decision.json

候補（どれも R1 の 3 万手、刻み 10、s=10、d=4。変えるのは RTC の有無と範囲だけ）:
  E40  rtc・範囲 40・指数（今の設定）
  E10  rtc・範囲 10・指数
  NV   naive・d=4（RTC なし）
試行（候補ごと。種は全候補で同じ。これまでの診断で使っていない選択用の帯の新しい種）:
  自然 30 回   selection:193200:30（193200〜193229）
  P1・P2・P3  induced:195400:30（195400〜195429、3 種類とも同じ種）。P2 は評価の P2（0.19 m、G2 と同じ定義）
出力: outputs/eval/RTCREDO/<候補>_<nat|P1|P2|P3>/

決め方（0075 の 1、原文のとおり）:
  (a) 自然の成功数が、最良の候補から 3 回以内の候補だけを残す
  (b) 残った候補のうち、P1・P2・P3 の立ち直りの合計（成立した試行が分母）が最も多い候補を選ぶ。最多から 3 以内は同じとみなす
  (c) 同じとみなした候補の中で、継ぎ目の跳び（中央値）が最小の候補を選ぶ
  (d) それでも決まらなければ、今の設定（範囲 40）に近い順（範囲 40 → 範囲 10 → naive）で選ぶ
ここで固めた数え方:
  - 立ち直り = 誘発が成立した試行のうち成功した試行。成立しなかった試行は数えない（差し替えない）
  - 継ぎ目の跳び（中央値）= その候補の全試行（自然 30＋誘発 90＝120 回）の、試行ごとの seam_jump_mean
    （eval/metrics.py、10 Hz の行動の切り替わりでの速度の差 [m/s]）の中央値。NaN の試行は除く。
    RTC の 2×2（41_results.py decide-rtc）と同じ量・同じ集め方
  - (c) の「最小」は値の比較そのまま（同じ値のときだけ (d) に進む）
  - 以後、R2 の結果がどうであってもこの決定は変えない（0075 の 1）
"""
import argparse
import json
import math
import time

import numpy as np

from recovla.common import config

CFG = config.load()
EVAL = config.path(CFG["paths"]["outputs"]) / "eval" / "RTCREDO"
RES = config.path(CFG["paths"]["outputs"]) / "results"
CKPT = r"outputs\train\train_R1_20260926-153959_20260926-153959\checkpoints\030000\pretrained_model"
CANDS = {"E40": {"mode": "rtc", "horizon": 40, "schedule": "EXP"},
         "E10": {"mode": "rtc", "horizon": 10, "schedule": "EXP"},
         "NV": {"mode": "naive", "horizon": None, "schedule": None}}
ORDER = ("E40", "E10", "NV")          # (d) 今の設定に近い順
KINDS = ("P1", "P2", "P3")
NAT_SPEC = "selection:193200:30"
IND_SPEC = "induced:195400:30"
NAT_WITHIN = 3
REC_SAME = 3


def runs():
    out = []
    for c, v in CANDS.items():
        for part in ("nat",) + KINDS:
            args = ["--experiment", "RTCREDO", "--condition", f"{c}_{part}", "--checkpoint", CKPT, "--model", "R1",
                    "--mode", v["mode"], "--s", "10", "--d", "4",
                    "--trials", NAT_SPEC if part == "nat" else IND_SPEC, "--videos", "2"]
            if v["horizon"]:
                args += ["--horizon", str(v["horizon"]), "--schedule", v["schedule"]]
            if part != "nat":
                args += ["--induce", part]
            out.append((f"{c}_{part}", args))
    return out


def cmd_plan(a) -> None:
    for name, args in runs():
        print(name, " ".join(args))


def cmd_decide(a) -> None:
    from recovla.eval import report
    stat = {}
    for c in CANDS:
        rows_all = []
        nat = report.collect([EVAL / f"{c}_nat"])
        rows_all += nat
        s = {"nat_n": len(nat), "nat_success": sum(bool(r["success"]) for r in nat), "kinds": {}}
        for k in KINDS:
            rs = report.collect([EVAL / f"{c}_{k}"])
            rows_all += rs
            est = [r for r in rs if r["induce_established"]]
            s["kinds"][k] = {"n": len(rs), "established": len(est), "recovered": sum(bool(r["success"]) for r in est)}
        s["recovered_total"] = sum(v["recovered"] for v in s["kinds"].values())
        s["established_total"] = sum(v["established"] for v in s["kinds"].values())
        seam = [r["seam_jump_mean"] for r in rows_all
                if r["seam_jump_mean"] is not None and not math.isnan(r["seam_jump_mean"])]
        s["seam_jump_median"] = float(np.median(seam)) if seam else float("nan")
        s["seam_n"] = len(seam)
        s["n_total"] = len(rows_all)
        s["seeds"] = sorted({(r["condition"].split("_", 1)[1], r["seed"]) for r in rows_all})
        stat[c] = s
        print(c, json.dumps({k: v for k, v in s.items() if k != "seeds"}, ensure_ascii=False), flush=True)
    same_seeds = len({tuple(v["seeds"]) for v in stat.values()}) == 1
    best_nat = max(v["nat_success"] for v in stat.values())
    step_a = [c for c in ORDER if stat[c]["nat_success"] >= best_nat - NAT_WITHIN]
    best_rec = max(stat[c]["recovered_total"] for c in step_a)
    step_b = [c for c in step_a if stat[c]["recovered_total"] >= best_rec - REC_SAME]
    smin = min(stat[c]["seam_jump_median"] for c in step_b)
    step_c = [c for c in step_b if stat[c]["seam_jump_median"] == smin]
    chosen = sorted(step_c, key=ORDER.index)[0]
    res = {"rule": {"nat_within": NAT_WITHIN, "recovered_same_within": REC_SAME, "order": ORDER,
                    "seam": "全 120 試行の試行ごとの seam_jump_mean の中央値", "source": "0075 の 1"},
           "same_seeds_all_candidates": same_seeds,
           "stats": {c: {k: v for k, v in s.items() if k != "seeds"} for c, s in stat.items()},
           "step_a_kept": step_a, "step_b_same": step_b, "step_c_seam": step_c, "chosen": chosen,
           "chosen_runtime": CANDS[chosen], "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "rtc_redo_decision.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "stats"}, ensure_ascii=False, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    sub.add_parser("decide")
    a = ap.parse_args(argv)
    {"plan": cmd_plan, "decide": cmd_decide}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
