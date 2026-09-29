"""目標書 v2 の完了判定（runtime/judge.py）の照合データと、画素の閾値の決め直し（0107 の 6、0113 の 5 の 2）。

    .venv\\Scripts\\python.exe scripts\\85_v2_judge.py data --seeds 59600:40 --tag fit        # 学習用のシード: 閾値を決める例
    .venv\\Scripts\\python.exe scripts\\85_v2_judge.py fit --tag fit                          # 画素の閾値を決めて configs/runtime_v2_judge.yaml へ
    .venv\\Scripts\\python.exe scripts\\85_v2_judge.py data --seeds 196040:40 --tag pre --n-each 100   # 検証用: 照合（一致率）

例の作り方（回す前に固めた）: 種ごとに、エキスパート（harness/gen_v2.py、センサの模型つき）で目標を箱に入れ、待機位置に戻って
止まった最後の状態から、目標の立方体だけを置き直した状態を作る（物理は進めない）。どの例も、同じ状態の 21 こま（20 Hz で 1 s。
こまごとにセンサの雑音は変わる）を、知覚（起動時の確かめと箱の推定は、その種の最後の状態の 5 こま）と判定に順に与え、最後に
完了と出たかを見る。真値は目標が成功の体積の中（frames.in_box）。
  run_positive       走行の最後の状態そのまま（正例）
  placed_inside ×2   箱の中の別の場所（正例）
  placed_rim         箱の壁の上（負例）
  placed_beside      箱の外の机の上、壁から 1〜3 cm（負例）
  placed_on_cube     箱の外の、別の立方体の上に乗せる（負例。0107 の 6 で足した負例）
  run_held_over_box  走行の途中で、持って箱の上にいる状態（グリッパ閉、目標が 2 cm 以上持ち上がっている。負例）
fit: 学習用のシードの例だけで、俯瞰の箱の画素・手首の割合と画素の閾値を、旧版と同じ決まり（正しく分けた数が最も多い値、同点なら
小さい方）で決める。深度の高さの許容・開き幅・待機位置の条件は決めない（configs の値）。
data の出力: outputs/v2judge/<tag>.json（照合は、正例 n_each・負例 n_each を、正解の側と種類だけで決まった順で選ぶ）
"""
import argparse
import collections
import json
import time

import numpy as np

from recovla.common import config

CFG = config.load_v2()
OUT = config.path(CFG["paths"]["outputs"]) / "v2judge"
JUDGE_KEY = 5200


def _cfg_with_judge():
    return config.load_v2()


def cmd_data(a) -> None:
    import mujoco
    from recovla.common.seeds import COLORS
    from recovla.eval import scene_trial as T
    from recovla.expert import generate as G
    from recovla.harness.gen_v2 import SensedDrivenRig
    from recovla.runtime import cue as C
    from recovla.runtime.judge import JudgeV2
    from recovla.runtime.perception import Params, Perception
    from recovla.runtime.types import JointState, SensorFrame
    from recovla.sim import frames, scene
    from recovla.sim.rig import quiet
    cfg = _cfg_with_judge()
    rtv = cfg["runtime_v2"]
    rig = SensedDrivenRig(cfg)
    suite = rig.suite
    thr = C.Thresholds.from_dict(config.color_detect(cfg))
    base, n = map(int, a.seeds.split(":"))
    seeds_ = list(range(base, base + n))
    lays = [scene.sample_layout(s, "empty", start="home") for s in seeds_]
    targets = T.choose_targets(seeds_, lays)
    cases = []
    box = rig.box
    h = frames.CUBE_HALF
    outer = frames.box_outer_half(rig.model)
    wall_mid = (frames.BOX_INNER_HALF + outer) / 2.0
    seq = {"n": 10 ** 6}

    def frames_of(qpos, count, t0):
        """同じ状態の count こま（20 Hz）。こまの番号を変えて雑音を変える。"""
        out = []
        for i in range(count):
            seq["n"] += 1
            t = t0 + 0.05 * i
            cams = {nm: suite._frame(nm, (seq["n"], t, t, qpos)) for nm in ("overhead", "wrist")}
            out.append(cams)
        return out

    def run_case(kind, label, qpos, closed, color, seed, per_base, extra):
        d = suite.scratch
        d.qpos[:] = qpos
        mujoco.mj_forward(rig.model, d)
        q = d.qpos[rig.arm_qadr].copy()
        width = float(d.qpos[rig.finger_qadr].sum())
        per = per_base()
        judge = JudgeV2(rig.setup_believed, per, thr, rtv["judge"])
        judge.reset(color)
        t0 = 100.0
        for i in range(25):
            per.record_joints(JointState(t0 - 0.2 + 0.01 * i, q, np.zeros(7)))
        from recovla.runtime.types import GripperState
        gs = GripperState(t0, width, bool(closed and 0.035 <= width <= 0.045))
        hand = rig.motion.hand_pose(q)[0]
        done = False
        for i, cams in enumerate(frames_of(qpos, 21, t0)):
            t = t0 + 0.05 * i
            per.record_joints(JointState(t, q, np.zeros(7)))
            sf = SensorFrame(t, JointState(t, q, np.zeros(7)), gs, cams)
            wm = per.update(cams, gs, t, hand)
            done = judge.update(sf, wm, hand)
        last = judge.last
        cases.append({"id": len(cases), "seed": seed, "color": color, "kind": kind, "truth": bool(label), "judge": bool(done),
                      "conditions": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in last.items()}, **extra})

    try:
        for seed, lay, color in zip(seeds_, lays, targets):
            rng = np.random.default_rng(np.random.SeedSequence([seed, JUDGE_KEY]))
            ti = COLORS.index(color)
            held = []
            orig_step = rig.physics_step

            def hook(on_step=None):
                orig_step(on_step)
                if rig.step % 25 == 0:
                    d = rig.data
                    p = d.xpos[rig.cube_ids[ti]]
                    if rig.closed and frames.over_box_interior(p, box) and p[2] - frames.CUBE_REST_Z >= 0.02:
                        held.append(d.qpos.copy())
            rig.physics_step = hook
            with quiet():
                r = G.run_attempt(rig, G.EpisodeSpec(seed, color, "empty", "n", start="home"), 0, None, False)
            rig.physics_step = orig_step
            if not r["success"]:
                print(seed, color, "expert failed", flush=True)
                continue
            q_end = rig.data.qpos.copy()
            # 知覚の起動（テーブル面・箱）はその種の最後の状態の 5 こまで。例ごとに立方体の推定は空から始める
            startup = frames_of(q_end, 5, 50.0)
            q_arm = rig.data.qpos[rig.arm_qadr].copy()
            width_end = float(rig.data.qpos[rig.finger_qadr].sum())

            def per_base():
                per = Perception(rig.setup_believed, Params.from_config(rtv["perception"]), thr)
                for i in range(25):
                    per.record_joints(JointState(49.8 + 0.01 * i, q_arm, np.zeros(7)))
                per.check_table(startup[0]["overhead"], width_end)
                per.init_box([(c["overhead"], width_end) for c in startup])
                return per
            qadr = scene.cube_qpos_adr(rig.model, color)[0]

            def placed(xyz, yaw):
                qp = q_end.copy()
                qp[qadr:qadr + 3] = xyz
                qp[qadr + 3:qadr + 7] = frames.yaw_quat(yaw)
                d = suite.scratch
                d.qpos[:] = qp
                mujoco.mj_forward(rig.model, d)
                return qp, d.xpos[rig.cube_ids[ti]].copy()
            run_case("run_positive", frames.in_box(rig.data.xpos[rig.cube_ids[ti]], box), q_end, False, color, seed, per_base, {})
            others = [rig.data.xpos[rig.cube_ids[c]].copy() for c in range(3) if c != ti]
            inside_others = [o[:2] for o in others if frames.over_box_interior(o, box)]
            for _ in range(2):
                lim = frames.BOX_SUCCESS_HALF - h - 0.005
                for _try in range(50):
                    xy = box[:2] + rng.uniform(-lim, lim, 2)
                    if all(np.hypot(*(xy - o)) > 2 * h + 0.01 for o in inside_others):
                        break
                qp, p = placed(np.r_[xy, frames.BOX_FLOOR_Z + h + 0.001], rng.uniform(-np.pi, np.pi))
                run_case("placed_inside", frames.in_box(p, box), qp, False, color, seed, per_base, {"pos": p.tolist()})
            axis = [(0, 1), (0, -1), (1, 1), (1, -1)][int(rng.integers(4))]
            u = rng.uniform(-(frames.BOX_INNER_HALF - h), frames.BOX_INNER_HALF - h)
            rim = np.array([box[0], box[1], frames.BOX_WALL_TOP_Z + h])
            rim[axis[0]] += axis[1] * wall_mid
            rim[1 - axis[0]] += u
            qp, p = placed(rim, rng.uniform(-np.pi, np.pi))
            run_case("placed_rim", frames.in_box(p, box), qp, False, color, seed, per_base, {"pos": p.tolist()})
            beside = np.array([box[0], box[1], frames.CUBE_REST_Z])
            beside[axis[0]] += axis[1] * (outer + h + rng.uniform(0.01, 0.03))
            beside[1 - axis[0]] += u
            qp, p = placed(beside, rng.uniform(-np.pi, np.pi))
            run_case("placed_beside", frames.in_box(p, box), qp, False, color, seed, per_base, {"pos": p.tolist()})
            table_others = [o for o in others if not frames.over_box_interior(o, box) and abs(o[2] - frames.CUBE_REST_Z) < 0.01]
            if table_others:
                o = table_others[int(rng.integers(len(table_others)))]
                qp, p = placed(np.r_[o[:2], o[2] + 2 * h + 0.001], rng.uniform(-np.pi, np.pi))
                run_case("placed_on_cube", frames.in_box(p, box), qp, False, color, seed, per_base, {"pos": p.tolist()})
            if held:
                qh = held[int(rng.integers(len(held)))]
                run_case("run_held_over_box", False, qh, True, color, seed, per_base, {})
            print(seed, color, "cases", len(cases), flush=True)
    finally:
        rig.close()
    rng = np.random.default_rng(np.random.SeedSequence([base, JUDGE_KEY, 1]))
    pos = [c for c in cases if c["truth"]]
    neg = [c for c in cases if not c["truth"]]
    pick = lambda xs, m: [xs[i] for i in sorted(rng.permutation(len(xs))[:m])] if len(xs) >= m else xs   # noqa: E731
    sel = pick(pos, a.n_each) + pick(neg, a.n_each) if a.n_each else cases
    agree = sum(c["truth"] == c["judge"] for c in sel)
    by_kind = collections.defaultdict(lambda: [0, 0])
    for c in sel:
        by_kind[c["kind"]][0] += c["truth"] == c["judge"]
        by_kind[c["kind"]][1] += 1
    res = {"seeds": a.seeds, "all_cases": len(cases), "pos_all": len(pos), "neg_all": len(neg), "selected": len(sel),
           "agree": agree, "agreement": agree / len(sel) if sel else None, "by_kind": dict(by_kind),
           "judge_params": cfg["runtime_v2"]["judge"], "selected_ids": [c["id"] for c in sel], "cases": cases,
           "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("cases", "selected_ids")}, ensure_ascii=False, indent=1))


def _best_threshold(pos, neg):
    vals = sorted(set(pos) | set(neg) | {0})
    best = None
    for t in [v + 1 for v in vals]:
        correct = sum(p >= t for p in pos) + sum(n < t for n in neg)
        if best is None or correct > best[1]:
            best = (t, correct)
    return best


def cmd_fit(a) -> None:
    """学習用のシードの例だけで、画素の閾値を旧版と同じ決まりで決める（51_planner.py fit と同じ順: 俯瞰の箱、手首の割合、手首の画素）。"""
    import yaml
    d = json.loads((OUT / f"{a.tag}.json").read_text(encoding="utf-8"))
    cs = [c for c in d["cases"] if c["kind"] != "run_held_over_box"]      # 持ったままは開閉の条件で落ちるので閾値に使わない
    box_pos = [c["conditions"]["box_pixels"] for c in cs if c["truth"]]
    box_neg = [c["conditions"]["box_pixels"] for c in cs if not c["truth"]]
    bt, bc = _best_threshold(box_pos, box_neg)
    rows = [(c["truth"], c["conditions"]["wrist_in_pixels"], c["conditions"]["wrist_fraction"]) for c in cs]
    wf, wfc = None, None
    for F in [x / 100 for x in range(0, 101)]:
        cc = sum((wi >= 1 and fr >= F) == lab for lab, wi, fr in rows)
        if wfc is None or cc > wfc:
            wf, wfc = F, cc
    passing = [(lab, wi) for lab, wi, fr in rows if fr >= wf]
    wt, wtc = _best_threshold([wi for lab, wi in passing if lab], [wi for lab, wi in passing if not lab])
    out = {"runtime_v2": {"judge": {"box_min_pixels": int(bt), "wrist_min_pixels": int(wt), "wrist_min_fraction": float(wf)}},
           "fit": {"tag": a.tag, "seeds": d["seeds"], "n": len(cs), "box_correct": bc, "wrist_fraction_correct": wfc,
                   "wrist_pixels_correct": wtc, "written": time.strftime("%Y-%m-%d %H:%M:%S")}}
    path = config.CONFIG_DIR / "runtime_v2_judge.yaml"
    head = "# 自動で書いた値（scripts/85_v2_judge.py fit）。学習用のシードの例だけで決めた完了判定の画素の閾値。手で直さない\n"
    path.write_text(head + yaml.safe_dump(out, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(path.read_text(encoding="utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("data")
    p.add_argument("--seeds", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--n-each", type=int, default=0)
    p = sub.add_parser("fit")
    p.add_argument("--tag", required=True)
    a = ap.parse_args(argv)
    {"data": cmd_data, "fit": cmd_fit}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
