"""流れの積分の刻み数（10・5・3）の採否（決裁 0072 の 1）。決まりは結果を見る前にここに固めた（2026-09-27 02:40）。

    .venv\\Scripts\\python.exe scripts\\47_steps.py decide     # outputs/results/steps_decision.json

試行（R1 の 3 万手、rtc・s=10・範囲 40・指数、d は候補ごとに閉ループで測った値: 10→4、5→3、3→2）:
  自然 30 回（193100〜193129）、P1・P3（195100〜195129）、P2（195000〜195029）各 30 回
  刻み 10 の誘発は既存の記録（G2diag/R1_P1・R1_P3、G2/R1_P2 の 195000〜195029）、自然は STEPS/S10_nat

決め方（0072 の 1、原文のとおり）:
  (a) 自然の成功数が、刻み 10 から 3 回以内の候補だけを残す
  (b) 残った候補のうち、P1・P2・P3 の復帰の合計（成立した試行が分母）が最も多い候補を選ぶ
  (c) (b) で選んだ候補の復帰の合計が、刻み 10 より 4 以上多くなければ、刻み 10 のままとする
  (d) 同じ数なら刻みの多い方とする

グリッパの開閉が原因の失敗（0072 の 1。数え方をここで固めた）: 失敗した試行（自然で成功しなかった試行と、誘発が成立して
復帰しなかった試行）の、方策が動かしていた区間（自然は全体、誘発は制御が戻った後＝誘発が上書きした最後のこまの次から）で
  g1 運んでいる途中で開いた: グリッパが閉→開に変わったこまで、目標が 2 cm 以上持ち上がっていて指先の中心から 2.5 cm 未満
     （掴んでいる）、かつ目標が箱の内側の上にない
  g2 掴む物がないのに閉じた: グリッパが開→閉に変わったこまで、指先の中心と目標の中心の距離（3 次元）が 3 cm 以上
  のどちらかがあった試行を数える（両方あれば両方に数える）。参考に成功した試行での件数も出す
P1 の「本当は掴めていたのに成立と数えた」本数（0072 の 2）: 誘発の閉じた後、開かないまま目標が 2 cm 以上持ち上がった成立の試行
"""
import argparse
import json
import pathlib
import time

import numpy as np

from recovla.common import config

CFG = config.load()
EVAL = config.path(CFG["paths"]["outputs"]) / "eval"
RES = config.path(CFG["paths"]["outputs"]) / "results"
STEPS = (10, 5, 3)
KINDS = ("P1", "P2", "P3")
P_SEEDS = {"P1": (195100, 195129), "P2": (195000, 195029), "P3": (195100, 195129)}
NAT_WITHIN = 3
REC_MARGIN = 4
G1_HELD_M = 0.025
G2_FAR_M = 0.03
LIFT_M = 0.02


def _dirs(step):
    if step == 10:
        return {"nat": EVAL / "STEPS" / "S10_nat", "P1": EVAL / "G2diag" / "R1_P1", "P2": EVAL / "G2" / "R1_P2",
                "P3": EVAL / "G2diag" / "R1_P3"}
    return {"nat": EVAL / "STEPS" / f"S{step}_nat", **{k: EVAL / "STEPS" / f"S{step}_{k}" for k in KINDS}}


def _box():
    from recovla.sim.rig import SimRig
    r = SimRig(render=False)
    b = r.box.copy()
    r.close()
    return b


def _gripper_events(a, f0, box):
    from recovla.sim import frames
    ti = int(a["target"][0])
    g = a["gripper_closed"].astype(bool)
    tip, cube = a["fingertip"], a["cube_pos"][:, ti]
    g1 = g2 = False
    for f in range(max(1, f0), len(g)):
        if g[f - 1] and not g[f]:
            held = cube[f - 1, 2] - frames.CUBE_REST_Z >= LIFT_M and np.linalg.norm(tip[f - 1] - cube[f - 1]) < G1_HELD_M
            if held and not frames.over_box_interior(cube[f - 1], box):
                g1 = True
        if g[f] and not g[f - 1] and np.linalg.norm(tip[f] - cube[f]) >= G2_FAR_M:
            g2 = True
    return g1, g2


def _load(d, seed_range=None):
    out = []
    for p in sorted(pathlib.Path(d).glob("trial_*.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        if seed_range and not (seed_range[0] <= m["seed"] <= seed_range[1]):
            continue
        out.append((m, p.with_suffix(".npz")))
    return out


def _p1_late(m, a):
    from recovla.sim import frames
    tc = float(m["induce"]["info"]["t_closed"])
    ti = int(a["target"][0])
    t, g = a["sim_time"], a["gripper_closed"].astype(bool)
    z = a["cube_pos"][:, ti, 2] - frames.CUBE_REST_Z
    after = t >= tc
    opened = np.flatnonzero(after & ~g)
    t_open = t[opened[0]] if opened.size else np.inf
    return bool(np.any(after & (z >= LIFT_M) & (t < t_open)))


def cmd_decide(a) -> None:
    box = _box()
    stats = {}
    for step in STEPS:
        d = _dirs(step)
        nat = _load(d["nat"])
        s = {"nat_n": len(nat), "nat_success": sum(bool(m["success"]) for m, _ in nat), "kinds": {},
             "gripper": {"g1_fail": 0, "g2_fail": 0, "any_fail": 0, "fail_trials": 0, "g1_success": 0, "g2_success": 0}}
        dcal = RES / ("dcal.json" if step == 10 else f"dcal_steps{step}.json")
        s["d"] = json.loads(dcal.read_text(encoding="utf-8"))["d"]
        s["dcal_p95_s"] = json.loads(dcal.read_text(encoding="utf-8"))["wall_p95_s"]

        def gripper(m, npz, f0, failed):
            arr = np.load(npz)
            g1, g2 = _gripper_events(arr, f0, box)
            G = s["gripper"]
            if failed:
                G["fail_trials"] += 1
                G["g1_fail"] += g1
                G["g2_fail"] += g2
                G["any_fail"] += (g1 or g2)
            else:
                G["g1_success"] += g1
                G["g2_success"] += g2
            return arr
        for m, npz in nat:
            gripper(m, npz, 0, not m["success"])
        for k in KINDS:
            rows = _load(d[k], P_SEEDS[k])
            est = [(m, npz) for m, npz in rows if m["induce"]["established"]]
            rec = sum(bool(m["success"]) for m, _ in est)
            late = 0
            for m, npz in est:
                arr = np.load(npz)
                act = np.flatnonzero(arr["induce_active"])
                f0 = int(act[-1]) + 1 if act.size else 0
                gripper(m, npz, f0, not m["success"])
                if k == "P1":
                    late += _p1_late(m, arr)
            s["kinds"][k] = {"n": len(rows), "established": len(est), "recovered": rec,
                             **({"p1_late_lift_established": late} if k == "P1" else {})}
        s["recovered_total"] = sum(v["recovered"] for v in s["kinds"].values())
        s["established_total"] = sum(v["established"] for v in s["kinds"].values())
        stats[step] = s
        print(step, json.dumps(s, ensure_ascii=False), flush=True)
    base = stats[10]
    step_a = [st for st in STEPS if base["nat_success"] - stats[st]["nat_success"] <= NAT_WITHIN]   # (a) 3 回以内（多いのは残す）
    best = max(stats[st]["recovered_total"] for st in step_a)
    step_b = sorted([st for st in step_a if stats[st]["recovered_total"] == best], reverse=True)   # (d) 刻みの多い方
    pick = step_b[0]
    chosen = pick if stats[pick]["recovered_total"] - base["recovered_total"] >= REC_MARGIN else 10
    res = {"rule": {"nat_within": NAT_WITHIN, "rec_margin": REC_MARGIN, "source": "0072 の 1"}, "stats": stats,
           "step_a_kept": step_a, "step_b_best": step_b, "picked_b": pick, "chosen_steps": chosen,
           "chosen_d": stats[chosen]["d"], "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "steps_decision.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "stats"}, ensure_ascii=False, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("decide")
    a = ap.parse_args(argv)
    {"decide": cmd_decide}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
