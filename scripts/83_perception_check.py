"""知覚の閾値を学習用のシードで決める（0108 の 2 の 4・5）と、知覚の精度の下見。真値は評価の道具としてだけ使う。

    .venv\\Scripts\\python.exe scripts\\83_perception_check.py thresholds --seeds 59100:10 [--tag v1]

エキスパート（harness/driven.py、真値で動く）を回し、0.5 s ごとのこまで、真のカメラから描いた分割（どの形状の画素か）と
真の深度（雑音なし）と照らす。俯瞰と手首で別々に数える。自己除去の太らせ方 M と、テーブル面からの閾値 H の組ごとに:
  画素の割合（0108 の 2 の 4・5 が挙げた量）
    robot_left(M, H)   ロボットの画素のうち、自己除去を通り抜けてテーブル面から H より上に残った割合
    cube_removed(M)    立方体の画素のうち、自己除去のマスクで消えた割合
    table_left(H)      テーブルの画素のうち、テーブル面から H より上に出た割合
  知覚の出力（runtime/perception.py の measure_cubes）
    detect(M, H)       上面が見えている立方体（上面の高さの画素が 15 以上）のうち、正しい色で 3 cm 以内に出した割合
    phantom(M, H)      出した立方体のうち、その色の立方体の上面が見えていないか、3 cm より離れていた割合
    err(M, H)          正しく出したときの位置の誤差（中央値・p95）
  高さは、そのカメラ自身の深度から当てはめた平面で測る（俯瞰は最初のこまで当てはめて保つ）
  table_check          俯瞰の平面の当てはめと、信じているテーブル面とのずれ（起動時の確かめの閾値を決めるため）
決め方（回す前に固めた。0108 の 2 の 4・5）:
  M: 候補のうち、robot_left(M, 0.015) ≤ 0.5%（俯瞰・手首とも）を満たす最小。そのときの cube_removed を記録する
  H: その M で、detect ≥ 99% かつ phantom ≤ 0.5%（俯瞰・手首とも）を満たす最小。満たすものがなければ detect − phantom の最大
  経緯: 最初の決まり（table_left ≤ 0.1%、立方体の画素が H 以下で落ちる割合 ≤ 10%）は、試しの 1 本（59100）で、
  (a) 灰色の机の点は色で立方体から外れるので知覚の出力に効かず、縁の空飛ぶ画素のために H を上げても 0.15% で止まる、
  (b) 立方体の側面の下の方は H によらず落ちる、ので意味がなかった。本番の前に、知覚の出力で測る上の決まりへ直した
出力: outputs/perception/thresholds_<tag>.json（写しを docs/results/perception_thresholds_<tag>.json）
"""
import argparse
import json
import math
import time

import mujoco
import numpy as np

from recovla.common import config

CFG = config.load("sensor_v1")
OUT = config.path(CFG["paths"]["outputs"]) / "perception"
H_GRID = [0.005, 0.01, 0.015, 0.02, 0.025]
M_GRID = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03]
COLORS = ("red", "green", "blue")
ROBOT_BODIES = {"link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand", "left_finger", "right_finger"}


def truth_view(suite, name, qpos):
    """真のカメラでの分割（形状）と雑音なしの深度と、真のカメラの姿勢（世界 ← カメラ）。"""
    c = suite.sc["cameras"][name]["depth"]
    m, d = suite.model, suite.scratch
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)
    cid = suite.cam_ids[name]
    key = ("seg", c["width"], c["height"])
    if key not in suite.renderers:
        r = mujoco.Renderer(m, c["height"], c["width"])
        r.enable_segmentation_rendering()
        suite.renderers[key] = r
    keyd = ("depth", c["width"], c["height"])
    if keyd not in suite.renderers:
        r = mujoco.Renderer(m, c["height"], c["width"])
        r.enable_depth_rendering()
        suite.renderers[keyd] = r
    f0 = float(m.cam_fovy[cid])
    m.cam_fovy[cid] = c["fovy"]
    try:
        suite.renderers[key].update_scene(d, camera=cid)
        seg = suite.renderers[key].render()
        suite.renderers[keyd].update_scene(d, camera=cid)
        z = suite.renderers[keyd].render().astype(np.float64)
    finally:
        m.cam_fovy[cid] = f0
    geom = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
    R, t = d.cam_xmat[cid].reshape(3, 3).copy(), d.cam_xpos[cid].copy()
    return geom, z, R, t


def categories(model):
    cat = np.full(model.ngeom, "other", dtype=object)
    for g in range(model.ngeom):
        b = model.body(model.geom_bodyid[g]).name
        cat[g] = ("robot" if b in ROBOT_BODIES else "cube" if b.startswith("cube_") else
                  "table" if b == "table" else "box" if b == "goal_box" else "other")
    return cat


def cmd_thresholds(a) -> None:
    from recovla.expert import generate as G
    from recovla.harness import setup as HS
    from recovla.harness.driven import DrivenRig
    from recovla.harness.sensors import SensorSuite, intrinsics
    from recovla.runtime import cue as Cq
    from recovla.runtime.perception import Params, Perception, pixel_rays
    from recovla.runtime.types import JointState
    from recovla.sim import scene
    rig = DrivenRig(render=False)
    suite = SensorSuite(rig.model, CFG)
    cat = categories(rig.model)
    cube_bodies = [rig.model.body(f"cube_{c}").id for c in COLORS]
    thr = Cq.Thresholds.from_dict(CFG["planner"]["color_detect"])
    rays = {n: pixel_rays(intrinsics(c["depth"]["width"], c["depth"]["height"], c["depth"]["fovy"]))
            for n, c in CFG["sensor"]["cameras"].items()}
    nM, nH = len(M_GRID), len(H_GRID)

    def zeros():
        return {"robot_left": np.zeros((nM, nH)), "robot_n": 0, "cube_removed": np.zeros(nM), "cube_n": 0,
                "table_left": np.zeros(nH), "table_n": 0, "vis": 0, "det": np.zeros((nM, nH)), "out": np.zeros((nM, nH)),
                "phantom": np.zeros((nM, nH)), "err": [[[] for _ in range(nH)] for _ in range(nM)]}
    acc = {cam: zeros() for cam in ("overhead", "wrist")}
    table_checks = []
    base, n = (int(x) for x in a.seeds.split(":"))
    n_frames = 0
    for s in range(base, base + n):
        lay = scene.sample_layout(s)
        sp = G.EpisodeSpec(s, lay.table_colors[0], lay.kind, "n")
        state = {"per": None, "n": 0}

        def sample(r):
            per = state["per"]
            sf = suite.sense(r.data, r.hand, r._last_cmd)
            if "overhead" in sf.cameras and "overhead" not in per.planes:
                table_checks.append(per.check_table(sf.cameras["overhead"], sf.gripper.width))
            for cam, fr in sf.cameras.items():
                qpos = [c for c in suite.streams[cam].captures if c[0] == fr.seq][0][3]
                geom, zt, Rt, tt = truth_view(suite, cam, qpos)
                gb = np.where(geom >= 0, rig.model.geom_bodyid[np.clip(geom, 0, None)], -1)
                gc = np.where(geom >= 0, cat[np.clip(geom, 0, None)], "none")
                pt = (rays[cam] * zt[..., None]) @ Rt.T + tt                   # 真の点（世界）
                valid = np.asarray(fr.depth) > 0
                A = acc[cam]
                cube_pos = suite.scratch.xpos[cube_bodies].copy()
                top_vis = []
                for b, cp in zip(cube_bodies, cube_pos):
                    top = (gb == b) & (pt[..., 2] > cp[2] + 0.01)
                    top_vis.append(int(top.sum()) >= 15)
                rob, cub, tab = (gc == "robot") & valid, (gc == "cube") & valid, (gc == "table") & valid
                A["robot_n"] += int(rob.sum())
                A["cube_n"] += int(cub.sum())
                A["table_n"] += int(tab.sum())
                A["vis"] += int(sum(top_vis))
                for mi, M in enumerate(M_GRID):
                    P = per.points(fr, sf.gripper.width, margin=M)
                    hmap = np.full(valid.size, np.nan)
                    hmap[P["idx"]] = P["h"]
                    hmap = hmap.reshape(valid.shape)
                    mask = P["robot_mask"]
                    A["cube_removed"][mi] += int(np.sum(cub & mask))
                    for hi, H in enumerate(H_GRID):
                        A["robot_left"][mi, hi] += int(np.sum(rob & (hmap > H)))
                        if mi == 0:
                            A["table_left"][hi] += int(np.sum(tab & (hmap > H)))
                        per.p.table_h = H
                        meas = per.measure_cubes(P)
                        for ci, c in enumerate(COLORS):
                            if c not in meas:
                                continue
                            A["out"][mi, hi] += 1
                            e = float(np.linalg.norm(meas[c]["pos"][:2] - cube_pos[ci][:2]))
                            if top_vis[ci] and e <= 0.03:
                                A["det"][mi, hi] += 1
                                A["err"][mi][hi].append(e)
                            else:
                                A["phantom"][mi, hi] += 1
            state["n"] += 1

        orig_reset, orig_step = rig.reset, rig.physics_step

        def reset_hook(layout):
            orig_reset(layout)
            setup = suite.start_trial(s, rig.data, HS.nominal_setup(CFG))
            state["per"] = Perception(setup, Params(), thr)

        def step_hook(on_step_inner=None):
            orig_step(on_step_inner)
            suite.on_physics_step(rig.data)
            d = rig.data
            state["per"].record_joints(JointState(float(d.time), d.qpos[rig.arm_qadr].copy(), d.qvel[rig.arm_vadr].copy()))
            if rig.step % 250 == 0:
                sample(rig)

        rig.reset, rig.physics_step = reset_hook, step_hook
        r = G.run_attempt(rig, sp, 0, None, False)
        rig.reset, rig.physics_step = orig_reset, orig_step
        n_frames += state["n"]
        print(f"[per] seed {s} ok {r['success']} frames {state['n']}", flush=True)
    res = {"seeds": a.seeds, "frames": n_frames, "H_grid": H_GRID, "M_grid": M_GRID, "cameras": {},
           "table_check": {"dz_m": [c["dz_m"] for c in table_checks], "angle_deg": [c["angle_deg"] for c in table_checks]}}
    for cam, A in acc.items():
        err = [[(float(np.median(x)) if x else None, float(np.percentile(x, 95)) if x else None) for x in row] for row in A["err"]]
        res["cameras"][cam] = {
            "robot_px": A["robot_n"], "cube_px": A["cube_n"], "table_px": A["table_n"], "top_visible_cubes": A["vis"],
            "robot_left": (A["robot_left"] / max(A["robot_n"], 1)).round(6).tolist(),
            "cube_removed": (A["cube_removed"] / max(A["cube_n"], 1)).round(6).tolist(),
            "table_left": (A["table_left"] / max(A["table_n"], 1)).round(6).tolist(),
            "detect": (A["det"] / max(A["vis"], 1)).round(4).tolist(),
            "phantom": (A["phantom"] / np.maximum(A["out"], 1)).round(4).tolist(),
            "err_med_p95_m": err}
    C_ = res["cameras"]
    hi15 = H_GRID.index(0.015)
    m_ok = [mi for mi in range(nM) if all(C_[c]["robot_left"][mi][hi15] <= 0.005 for c in C_)]
    mi = m_ok[0] if m_ok else nM - 1
    res["chosen_self_margin_m"] = M_GRID[mi]
    res["cube_removed_at_chosen"] = {c: C_[c]["cube_removed"][mi] for c in C_}
    h_ok = [hi for hi in range(nH) if all(C_[c]["detect"][mi][hi] >= 0.99 and C_[c]["phantom"][mi][hi] <= 0.005 for c in C_)]
    if h_ok:
        res["chosen_table_h"], res["h_rule_met"] = H_GRID[h_ok[0]], True
    else:
        score = [min(C_[c]["detect"][mi][hi] - C_[c]["phantom"][mi][hi] for c in C_) for hi in range(nH)]
        res["chosen_table_h"], res["h_rule_met"] = H_GRID[int(np.argmax(score))], False
    res["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    OUT.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(res, ensure_ascii=False, indent=1)
    (OUT / f"thresholds_{a.tag}.json").write_text(txt, encoding="utf-8")
    if a.tag != "smoke":
        (config.ROOT / "docs" / "results" / f"perception_thresholds_{a.tag}.json").write_text(txt + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("frames", "chosen_self_margin_m", "cube_removed_at_chosen", "chosen_table_h",
                                          "h_rule_met")}, ensure_ascii=False))


def cmd_accuracy(a) -> None:
    """知覚の実際の流れ（10 Hz の update、融合と隠れの保持、箱は試行の始めの 5 こま）の精度と計算時間。
    0107 の 5 の余裕（障害物の位置の誤差の p95）と、4-2 の知覚の遅れ（計算時間の p99 を 10 ms 単位で切り上げ）をここから決める。"""
    from recovla.expert import generate as G
    from recovla.harness import setup as HS
    from recovla.harness.driven import DrivenRig
    from recovla.harness.sensors import SensorSuite
    from recovla.runtime import cue as Cq
    from recovla.runtime.perception import Params, Perception
    from recovla.runtime.types import JointState
    from recovla.sim import frames as FR
    from recovla.sim import scene
    rig = DrivenRig(render=False)
    suite = SensorSuite(rig.model, CFG)
    thr = Cq.Thresholds.from_dict(CFG["planner"]["color_detect"])
    pp = Params(self_margin_m=float(a.margin), table_h=float(a.table_h))
    cube_bodies = [rig.model.body(f"cube_{c}").id for c in COLORS]
    rows, boxes, walls, checks = [], [], [], []
    base, n = (int(x) for x in a.seeds.split(":"))
    for s in range(base, base + n):
        lay = scene.sample_layout(s)
        sp = G.EpisodeSpec(s, lay.table_colors[0], lay.kind, "n")
        st = {"per": None, "boxf": []}

        def reset_hook(layout):
            orig_reset(layout)
            setup = suite.start_trial(s, rig.data, HS.nominal_setup(CFG))
            st["per"] = Perception(setup, pp, thr)
            st["boxf"] = []

        def step_hook(on_step_inner=None):
            orig_step(on_step_inner)
            suite.on_physics_step(rig.data)
            d = rig.data
            per = st["per"]
            per.record_joints(JointState(float(d.time), d.qpos[rig.arm_qadr].copy(), d.qvel[rig.arm_vadr].copy()))
            if rig.step % 50 != 0:
                return
            sf = suite.sense(d, rig.hand, rig._last_cmd)
            if "overhead" not in sf.cameras or "wrist" not in sf.cameras:
                return
            if "overhead" not in per.planes:
                checks.append(per.check_table(sf.cameras["overhead"], sf.gripper.width))
            if not per.box["ok"] and len(st["boxf"]) < 5:
                st["boxf"].append((sf.cameras["overhead"], sf.gripper.width))
                if len(st["boxf"]) == 5:
                    b = per.init_box(st["boxf"])
                    tb = rig.box
                    boxes.append({"seed": s, "ok": b["ok"], "err_m": float(np.linalg.norm(b["xy"] - tb[:2])) if b["ok"] else None,
                                  "yaw_deg": float(np.degrees(min(b["yaw"], np.pi / 2 - b["yaw"]))) if b["ok"] else None})
                return
            per.selfr.set_q(d.qpos[rig.arm_qadr], sf.gripper.width)
            tip = per.selfr.hand_pose()
            tipp = tip[0] + tip[1] @ np.array([0.0, 0.0, 0.1034])
            t0 = time.perf_counter()
            wm = per.update(sf.cameras, sf.gripper, sf.t, tipp)
            wall = time.perf_counter() - t0
            for ci, c in enumerate(COLORS):
                e = wm.cubes.get(c)
                tp = d.xpos[cube_bodies[ci]]
                tin = bool(FR.in_box(tp, rig.box))
                if e is None:
                    rows.append({"seed": s, "t": sf.t, "color": c, "status": "none", "true_in_box": tin, "wall": wall})
                    continue
                rows.append({"seed": s, "t": sf.t, "color": c, "status": e.status, "source": e.source,
                             "err3": float(np.linalg.norm(e.pos - tp)), "errxy": float(np.linalg.norm(e.pos[:2] - tp[:2])),
                             "in_box": e.in_box, "true_in_box": tin, "wall": wall,
                             "on_table": bool(tp[2] < 0.03 and not tin)})

        orig_reset, orig_step = rig.reset, rig.physics_step
        rig.reset, rig.physics_step = reset_hook, step_hook
        r = G.run_attempt(rig, sp, 0, None, False)
        rig.reset, rig.physics_step = orig_reset, orig_step
        print(f"[acc] seed {s} ok {r['success']} updates {sum(1 for x in rows if x['seed'] == s) // 3}", flush=True)

    def q(x, p):
        return float(np.percentile(x, p)) if len(x) else None
    obst = [x["err3"] for x in rows if x.get("on_table") and x["status"] in ("seen", "held")]
    walls_t = [x["wall"] for x in rows]
    res = {"seeds": a.seeds, "params": {"self_margin_m": pp.self_margin_m, "table_h": pp.table_h}, "n_updates": len(rows) // 3,
           "obstacle_err3_m": {"n": len(obst), "median": q(obst, 50), "p95": q(obst, 95), "max": max(obst) if obst else None},
           "by_status": {st_: {"n": len(v), "median_err3": q(v, 50), "p95_err3": q(v, 95)} for st_ in ("seen", "held", "in_hand", "lost")
                         for v in [[x["err3"] for x in rows if x["status"] == st_ and "err3" in x]]},
           "not_found_frac": float(np.mean([x["status"] == "none" for x in rows])) if rows else None,
           "in_box_agreement": float(np.mean([x["in_box"] == x["true_in_box"] for x in rows if "in_box" in x and x["status"] != "in_hand"])),
           "box": {"n": len(boxes), "ok": sum(b["ok"] for b in boxes), "err_m": [b["err_m"] for b in boxes],
                   "yaw_deg": [b["yaw_deg"] for b in boxes]},
           "table_check": {"dz_m": [c["dz_m"] for c in checks], "angle_deg": [c["angle_deg"] for c in checks]},
           "update_wall_s": {"median": q(walls_t, 50), "p99": q(walls_t, 99), "max": max(walls_t) if walls_t else None},
           "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    res["margin_rule"] = "安全フィルタの余裕 = 机上の立方体（見えている・隠れて保持中）の中心の位置の誤差（3 次元）の p95"
    res["latency_rule"] = "知覚の遅れ = update の計算時間の p99 を 10 ms 単位で切り上げ"
    res["chosen_extra_margin_m"] = None if res["obstacle_err3_m"]["p95"] is None else round(res["obstacle_err3_m"]["p95"], 4)
    res["chosen_perception_latency_s"] = None if res["update_wall_s"]["p99"] is None else math.ceil(res["update_wall_s"]["p99"] * 100) / 100
    OUT.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(res, ensure_ascii=False, indent=1)
    (OUT / f"accuracy_{a.tag}.json").write_text(txt, encoding="utf-8")
    if a.tag != "smoke":
        (config.ROOT / "docs" / "results" / f"perception_accuracy_{a.tag}.json").write_text(txt + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("n_updates", "obstacle_err3_m", "by_status", "not_found_frac", "in_box_agreement",
                                          "box", "update_wall_s", "chosen_extra_margin_m", "chosen_perception_latency_s")},
                     ensure_ascii=False))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("thresholds")
    p.add_argument("--seeds", default="59100:10")
    p.add_argument("--tag", default="v1")
    p = sub.add_parser("accuracy")
    p.add_argument("--seeds", default="59200:20")
    p.add_argument("--margin", required=True, type=float)
    p.add_argument("--table-h", required=True, type=float)
    p.add_argument("--tag", default="v1")
    a = ap.parse_args(argv)
    {"thresholds": cmd_thresholds, "accuracy": cmd_accuracy}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
