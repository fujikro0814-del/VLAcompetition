"""Step D の完了条件の測定（手順書 §5）。結果は outputs/d/<項目>.json に書き、tests/test_d_scene.py が読んで判定する。

    .venv\\Scripts\\python.exe scripts\\11_d_checks.py scene        # 色・発表用カメラ・視野・到達（完了条件の前提）
    ... physics      # 1・6（配置の種類ごと 120 本、描画なし）と、着地のばらつき（R3）・向きごとの成功（R4）
    ... pairs        # 2（組の最初のこまの画素一致。並列 1 と 8 の両方）と、描画の有無で物理が変わらないこと
    ... replay       # 4（無作為の 5 本の再生確認）
    ... convert      # 5（変換の検証、task と目標の色、2 視点）
    ... handover     # 3（引き継ぎ試験 50 回）
    ... throughput   # 7（1 本あたりの時間と、並列数ごとの本/時）
    ... videos       # 8（代表の組の映像）
    ... all

種はすべて Step D・F の試験の帯（50000〜59999、B_提案書 §9）から取る。
"""
import argparse
import collections
import glob
import json
import pathlib
import sys
import time

import numpy as np

from recovla.common import config, seeds
from recovla.common.seeds import COLORS
from recovla.expert import generate as G
from recovla.expert.script import Phase
from recovla.sim import contact, frames, scene

CFG = config.load()
OUT = config.path(CFG["paths"]["outputs"]) / "d"
GEN = config.path(CFG["paths"]["outputs"]) / "gen"
SEEDS = {  # 配置の種（Step D・F の試験の帯）
    "physics": {"empty": range(50000, 50040), "prefilled_1": range(50100, 50160), "prefilled_2": range(50200, 50320)},
    "pairs": {"empty": range(50000, 50010), "prefilled_1": range(50100, 50105)},
    "handover": range(50500, 50550),
    "throughput": {"empty": range(50600, 50624)},      # 24 配置 x 3 色 = 72 本（仕事の単位は配置なので、並列 12 でも余らない数）
}
WORKERS_PAIRS = (1, 8)
WORKERS_THROUGHPUT = (1, 4, 8, 12)
REPLAY_N = 5
HANDOVER_U = (0.05, 0.95)


def stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def write(name: str, obj: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    obj = {"check": name, "written": time.strftime("%Y-%m-%d %H:%M:%S"), **obj}
    (OUT / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_json), encoding="utf-8")
    print(f"[d] wrote {OUT / (name + '.json')}")


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def latest_run(prefix: str) -> pathlib.Path:
    runs = sorted(GEN.glob(f"{prefix}_*"))
    if not runs:
        raise SystemExit(f"no run outputs/gen/{prefix}_* (run the check that makes it first)")
    return runs[-1]


def read_results(run: pathlib.Path) -> list:
    return [json.loads(l) for l in (run / "generation.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def pct(x, qs=(0, 5, 50, 95, 100)) -> dict:
    x = np.asarray(x, float)
    return {f"p{q}": float(np.percentile(x, q)) for q in qs} if x.size else {}


# --------------------------------------------------------------------------------------------- scene

def check_scene() -> None:
    import mujoco
    from recovla.sim import render
    from recovla.sim.rig import SimRig
    m = scene.build_model()                                    # 色の照合を含む
    pc = CFG["scene"]["presentation_camera"]
    cam = m.camera(pc["name"])
    xy = np.array(pc["xyaxes"], float)
    xmat = np.stack([xy[:3] / np.linalg.norm(xy[:3]), xy[3:] / np.linalg.norm(xy[3:])])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.vstack([xmat, np.cross(xmat[0], xmat[1])]).T.ravel())
    cam_ok = bool(np.allclose(m.cam_pos[cam.id], pc["pos"]) and (np.allclose(m.cam_quat[cam.id], q, atol=1e-5)
                                                                  or np.allclose(m.cam_quat[cam.id], -q, atol=1e-5))
                  and abs(m.cam_fovy[cam.id] - pc["fovy"]) < 1e-9)
    # 発表用カメラ・画面外の描画領域を足しても、記録用カメラの描画が変わらない: 足さない版（MjSpec で消す）と比べる
    m0 = model_without_presentation()
    rig = SimRig(render=True)
    r0 = mujoco.Renderer(m0, 256, 256)
    d0 = mujoco.MjData(m0)
    diffs, states = [], 0
    try:
        for seed in (50000, 50100, 50200):
            lay = scene.sample_layout(seed, None)
            rig.reset(lay)
            for _ in range(3):
                s = rig.forward_scratch()
                d0.qpos[:] = s.qpos
                mujoco.mj_forward(m0, d0)
                a = rig.render(s)
                b = []
                for c in rig.cameras:
                    r0.update_scene(d0, camera=c)
                    b.append(r0.render().copy())
                diffs.append(max(int(np.abs(x.astype(int) - y.astype(int)).max()) for x, y in zip(a, b)))
                states += 1
                from recovla.sim.rig import quiet
                with quiet():
                    for _ in range(50):
                        rig.pad_read(np.array([0.05, -0.05, -0.03]), False)
        # 視野: 机上の範囲の四隅と箱の置き場所の立方体が、開始姿勢（home・retreat）で俯瞰カメラから全部見える
        vis = visibility(rig)
    finally:
        render.close_renderer(r0)
        rig.close()
    cams_without = m0.ncam
    write("scene", {"colors_match_config": True, "presentation_camera_matches_config": cam_ok,
                    "presentation_removed_ncam": {"with": int(m.ncam), "without": int(cams_without)},
                    "recording_render_max_diff_with_vs_without_presentation": max(diffs), "states_compared": states,
                    "visibility": vis, "region": CFG["scene"]["region"], "retreat_pose": CFG["expert"]["retreat_pose"]})


def model_without_presentation():
    """scene_3cube.xml から、発表用カメラと画面外の描画領域の指定（visual）を除いた模型。include とメッシュの場所は
    絶対パスにして、一時フォルダの XML から読む（リポジトリには書かない）。"""
    import re
    import tempfile
    import mujoco
    src = config.path(CFG["paths"]["scene"])
    xml = src.read_text(encoding="utf-8")
    xml, n1 = re.subn(r'<camera name="presentation".*?/>', "", xml, flags=re.S)
    xml, n2 = re.subn(r"<visual>.*?</visual>", "", xml, flags=re.S)
    base = src.parent.as_posix()
    xml = xml.replace('file="panda/panda.xml"', f'file="{base}/panda/panda.xml"')
    xml = xml.replace('meshdir="panda/assets"', f'meshdir="{base}/panda/assets"')
    if (n1, n2) != (1, 1):
        raise RuntimeError(f"presentation camera / visual not found once each: {(n1, n2)}")
    tmp = pathlib.Path(tempfile.mkdtemp()) / "scene_without_presentation.xml"
    tmp.write_text(xml, encoding="utf-8")
    return mujoco.MjModel.from_xml_path(str(tmp))


def visibility(rig) -> dict:
    import mujoco
    m = rig.model
    r = mujoco.Renderer(m, 256, 256)
    r.enable_segmentation_rendering()
    hide = mujoco.MjvOption()
    hide.geomgroup[2] = 0
    gid = {c: m.geom(frames.cube_geom(c)).id for c in COLORS}
    gid["box_floor"] = m.geom("box_bottom").id
    rx, ry = CFG["scene"]["region"]["x"], CFG["scene"]["region"]["y"]
    corner_sets = [((rx[0], ry[0]), (rx[1], ry[1]), (rx[1], ry[0])), ((rx[0], ry[1]), (sum(rx) / 2, sum(ry) / 2), (rx[1], ry[0]))]
    worst = 1.0
    rows = []
    from recovla.sim import render
    try:
        for start in scene.STARTS:
            for kind, cs, pre in [("corners", cs, {}) for cs in corner_sets] + [("box", ((0.45, -0.05),), {"red": 0, "green": 3})]:
                d = rig.data
                cubes = {c: (x, y, 0.5) for c, (x, y) in zip([c for c in COLORS if c not in pre], cs)}
                lay = scene.Layout(0, "prefilled_2" if pre else "empty", start, cubes, pre)
                rig.reset(lay)
                counts = []
                for opt in (None, hide):
                    if opt is None:
                        r.update_scene(d, camera="overhead")
                    else:
                        r.update_scene(d, camera="overhead", scene_option=opt)
                    s = r.render()
                    ids = s[..., 0][s[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)]
                    counts.append({k: int((ids == g).sum()) for k, g in gid.items()})
                frac = {k: (counts[0][k] / counts[1][k] if counts[1][k] else None) for k in gid}
                vals = [v for v in frac.values() if v is not None]
                worst = min(worst, min(vals))
                rows.append({"start": start, "kind": kind, "visible_px": counts[0], "unoccluded_px": counts[1],
                             "visible_fraction": frac})
    finally:
        render.close_renderer(r)
    return {"worst_visible_fraction": worst, "min_visible_px": min(min(v for v in row["visible_px"].values())
                                                                     for row in rows), "rows": rows}


# ------------------------------------------------------------------------------------------- physics

def check_physics() -> None:
    specs = G.plan_specs(SEEDS["physics"])
    run = GEN / f"d_physics_{stamp()}"
    t0 = time.time()
    results = G.generate(specs, run, workers=8, render=False)
    wall = time.time() - t0
    by = {}
    for kind in SEEDS["physics"]:
        rs = [r for r in results if r["layout_kind"] == kind]
        fails = collections.Counter(a["failure"] for r in rs for a in r["attempts"] if a["failure"])
        by[kind] = {"episodes": len(rs), "first_try_success": sum(r["first_try_success"] for r in rs),
                    "saved": sum(r["success"] for r in rs), "dropped": sum(not r["success"] for r in rs),
                    "retries_histogram": dict(collections.Counter(r["retries"] for r in rs)),
                    "failures_by_kind": dict(fails),
                    "first_try_rate": sum(r["first_try_success"] for r in rs) / len(rs)}
    stats = episode_stats(run, results)
    write("physics", {"run": str(run.relative_to(config.ROOT)), "wall_s": round(wall, 1), "workers": 8,
                      "by_kind": by, **stats})


def episode_stats(run: pathlib.Path, results: list) -> dict:
    """接触・最短距離（全体・配置の種類別・段階別）、着地（R3）、向き（R4）。成功したエピソードの npz から。"""
    contact_by_kind = collections.Counter()
    eps_by_kind = collections.Counter()
    md_all, md_phase = [], collections.defaultdict(list)
    md_ep = []
    landing = collections.defaultdict(list)
    yaw_ok = []
    for r in results:
        for a in r["attempts"]:
            yaw_ok.append((a["target_yaw_deg"], a["success"]))
        if not r["success"]:
            continue
        name = r["attempts"][-1]["name"]
        z = np.load(run / name / "data.npz")
        meta = json.loads((run / name / "meta.json").read_text(encoding="utf-8"))
        mask = np.array(meta["obstacle_mask"])
        kind = meta["layout_kind"]
        eps_by_kind[kind] += 1
        contact_by_kind[kind] += int(z["contact_robot"][:, mask].any())
        md = z["min_dist"][:, mask].min(axis=1)
        md_all.append(md)
        md_ep.append(md.min())
        for ph in np.unique(z["phase"]):
            md_phase[Phase(int(ph)).name].append(md[z["phase"] == ph])
        landing[kind].append(landing_of(z, meta))
    md_all = np.concatenate(md_all) if md_all else np.zeros(0)
    land = {k: summarize_landing(v) for k, v in landing.items()}
    yaws = np.array([y for y, _ in yaw_ok])
    bins = np.linspace(-30, 30, 7)
    yaw_rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = [(ok) for y, ok in yaw_ok if lo <= y < hi or (hi == 30 and y == 30)]
        yaw_rows.append({"yaw_deg": [float(lo), float(hi)], "attempts": len(sel), "success": int(sum(sel))})
    return {
        "episodes_with_obstacle_contact": {k: {"contact_episodes": contact_by_kind[k], "episodes": eps_by_kind[k]}
                                           for k in eps_by_kind},
        "episodes_total": int(sum(eps_by_kind.values())),
        "contact_episodes_total": int(sum(contact_by_kind.values())),
        "min_dist_per_frame_m": pct(md_all),
        "min_dist_per_episode_m": pct(md_ep),
        "min_dist_by_phase_m": {ph: pct(np.concatenate(v)) for ph, v in md_phase.items()},
        "landing_by_kind": land, "yaw_success": yaw_rows,
        "yaw_range_deg": [float(yaws.min()), float(yaws.max())] if yaws.size else None,
    }


def landing_of(z, meta) -> dict:
    """最後のこまの目標の着地: 置き場所からのずれ、傾き、箱の中の他の立方体・壁との隙間（厳密な距離）。"""
    t = COLORS.index(meta["target"])
    pos, quat = z["cube_pos"][-1], z["cube_quat"][-1]
    box = np.array(CFG["scene"]["box"]["pos"], float)
    slot_err = min(np.hypot(*(pos[t, :2] - frames.slot_xy(box, s))) for s in range(len(frames.BOX_SLOTS)))
    others = [i for i in range(3) if i != t and z["cube_in_box"][-1, i]]
    cube_gap = min((box_gap(pos[t], quat[t], pos[i], quat[i]) for i in others), default=None)
    wall_gap = min(wall_gaps(pos[t], quat[t], box))
    return {"slot_err_m": float(slot_err), "tilt_deg": frames.tilt_deg(quat[t]), "cube_gap_m": cube_gap,
            "wall_gap_m": float(wall_gap), "n_in_box_before": len(others)}


def _corners(pos, quat, half) -> np.ndarray:
    import mujoco
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat, float))
    c = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * np.asarray(half, float)
    return c @ R.reshape(3, 3).T + np.asarray(pos, float)


def exact_gap(A, B) -> float:
    """2 つの点集合の凸包の距離（厳密。小さい二次計画）。"""
    from scipy.optimize import minimize
    na, nb = len(A), len(B)
    f = lambda w: (lambda p: p @ p)(w[:na] @ A - w[na:] @ B)
    cons = [{"type": "eq", "fun": lambda w: w[:na].sum() - 1}, {"type": "eq", "fun": lambda w: w[na:].sum() - 1}]
    r = minimize(f, np.r_[np.ones(na) / na, np.ones(nb) / nb], bounds=[(0, 1)] * (na + nb), constraints=cons,
                 method="SLSQP", options={"ftol": 1e-15, "maxiter": 800})
    return float(np.sqrt(max(r.fun, 0.0)))


def box_gap(p1, q1, p2, q2) -> float:
    h = [frames.CUBE_HALF] * 3
    return exact_gap(_corners(p1, q1, h), _corners(p2, q2, h))


def wall_gaps(p, q, box) -> list:
    """壁の内面までの隙間（壁は軸にそろった板なので、立方体の角の座標から厳密に出る）。"""
    c = _corners(p, q, [frames.CUBE_HALF] * 3) - box
    half = frames.BOX_INNER_HALF
    return [half - c[:, 0].max(), c[:, 0].min() + half, half - c[:, 1].max(), c[:, 1].min() + half]


def summarize_landing(rows: list) -> dict:
    cfg = CFG["scene"]["box_slot_clearance_m"]
    cube = [r["cube_gap_m"] for r in rows if r["cube_gap_m"] is not None]
    wall = [r["wall_gap_m"] for r in rows]
    ok = [(r["wall_gap_m"] >= cfg["wall"]) and (r["cube_gap_m"] is None or r["cube_gap_m"] >= cfg["cube"]) for r in rows]
    return {"n": len(rows), "slot_err_m": pct([r["slot_err_m"] for r in rows]), "tilt_deg": pct([r["tilt_deg"] for r in rows]),
            "cube_gap_m": pct(cube), "wall_gap_m": pct(wall),
            "clearance_condition_met": int(sum(ok)), "clearance_condition": cfg}


# --------------------------------------------------------------------------------------------- pairs

def check_pairs() -> None:
    specs = G.plan_specs(SEEDS["pairs"])
    runs = {}
    for w in WORKERS_PAIRS:
        runs[w] = GEN / f"d_pairs_w{w}_{stamp()}"
        G.generate(specs, runs[w], workers=w, render=True)
    from recovla.record.recorder import read_png
    pair_rows = []
    groups = collections.defaultdict(list)
    for s in specs:
        groups[(s.layout_kind, s.layout_seed)].append(s)
    all_equal = True
    cross_max = 0
    for (kind, seed), ss in groups.items():
        row = {"kind": kind, "layout_seed": seed, "colors": [s.color for s in ss], "views": {}}
        for view in CFG["sim"]["cameras"]:
            row["views"][view] = {}
            by_w = {w: [read_png(ep_dir(runs[w], s) / view / "000000.png") for s in ss] for w in WORKERS_PAIRS}
            for w, imgs in by_w.items():                 # 回の中（その回の生成データ）で、組の各本が一致する
                eq = all(np.array_equal(imgs[0], im) for im in imgs[1:])
                row["views"][view][f"workers_{w}_pixel_identical"] = bool(eq)
                all_equal &= eq
            a, b = by_w[WORKERS_PAIRS[0]][0], by_w[WORKERS_PAIRS[-1]][0]   # 回をまたいだ差（参考）
            cross_max = max(cross_max, int(np.abs(a.astype(int) - b.astype(int)).max()))
        pair_rows.append(row)
    # 組の最初のこまの状態が、組の中でビット一致する（画像を共有してよい前提）。共有した本数
    state_equal = True
    shared = 0
    for (kind, seed), ss in groups.items():
        for w in WORKERS_PAIRS:
            zs = [np.load(ep_dir(runs[w], s) / "data.npz") for s in ss]
            for key in ("joints", "joint_vel", "ee_pos", "ee_quat", "fingers", "x_des", "cube_pos", "cube_quat",
                        "cube_linvel", "gripper_closed"):
                state_equal &= all(np.array_equal(zs[0][key][0], z[key][0]) for z in zs[1:])
            shared += sum(json.loads((ep_dir(runs[w], s) / "meta.json").read_text(encoding="utf-8"))
                          .get("frame0_image_shared_with_pair", False) for s in ss)
    # 並列 1 と 8 で、全こまの配列が一致する（決定性）
    arrays_equal = True
    for s in specs:
        a, b = (np.load(ep_dir(runs[w], s) / "data.npz") for w in WORKERS_PAIRS)
        arrays_equal &= all(np.array_equal(a[k], b[k]) for k in a.files)
    # 描画の有無で物理が変わらない: physics の回（描画なし）の同じ指定と、全こまの配列が一致する
    phys = latest_run("d_physics")
    render_invariant = []
    for s in specs:
        a, b = np.load(ep_dir(runs[1], s) / "data.npz"), np.load(ep_dir(phys, s) / "data.npz")
        render_invariant.append(all(np.array_equal(a[k], b[k]) for k in a.files))
    write("pairs", {"runs": {str(w): str(p.relative_to(config.ROOT)) for w, p in runs.items()},
                    "pairs": {k: sum(1 for r in pair_rows if r["kind"] == k) for k in SEEDS["pairs"]},
                    "first_frame_pixel_identical_all": bool(all_equal), "rows": pair_rows,
                    "first_frame_max_diff_across_runs": cross_max,
                    "first_frame_state_bit_identical_in_pairs": bool(state_equal),
                    "first_frame_images_shared": shared,
                    "arrays_identical_workers_1_vs_8": bool(arrays_equal), "episodes": len(specs),
                    "render_invariant_vs_physics_run": {"episodes": len(render_invariant),
                                                        "identical": int(sum(render_invariant)),
                                                        "physics_run": str(phys.relative_to(config.ROOT))}})


def ep_dir(run: pathlib.Path, spec, retry=None) -> pathlib.Path:
    found = sorted(run.glob(f"{spec.kind}_{spec.layout_seed}_{spec.color}_r*"))
    found = [p for p in found if not p.name.endswith(".partial")]
    if len(found) != 1:
        raise FileNotFoundError(f"{run}: {spec} -> {found}")
    return found[0]


# -------------------------------------------------------------------------------------------- replay

def check_replay() -> None:
    from recovla.record import replay as legacy, replay_scene as RS
    from recovla.sim.rig import SimRig
    run = latest_run("d_pairs_w1")
    eps = sorted(p for p in run.iterdir() if p.is_dir() and not p.name.endswith(".partial"))
    pick = [eps[i] for i in seeds.stream(50900, "order").choice(len(eps), REPLAY_N, replace=False)]
    rig = SimRig(render=True)
    rows = []
    try:
        for p in pick:
            ep = legacy.load_episode(p)
            row = {"episode": p.name}
            for mode in RS.MODES:
                rep = RS.replay(ep, mode, rig, render=mode in ("step", "window"))
                r = RS.compare(ep, rep, images=mode in ("step", "window"))
                r["pass"] = RS.verdict(mode, r)
                row[mode] = r
            rows.append(row)
    finally:
        rig.close()
    write("replay", {"run": str(run.relative_to(config.ROOT)), "tol_m": RS.WINDOW_TOL_M, "episodes": rows,
                     "window_pass_all": all(r["window"]["pass"] for r in rows),
                     "step_bit_exact_all": all(r["step"]["pass"] for r in rows),
                     "negative_controls_detected_all": all(r["last_step"]["pass"] and r["shifted"]["pass"] for r in rows)})


# ------------------------------------------------------------------------------------------- convert

def check_convert() -> None:
    import subprocess
    from recovla.data import convert as C
    run = latest_run("d_pairs_w1")
    eps = sorted(p.name for p in run.iterdir() if p.is_dir() and not p.name.endswith(".partial"))
    name = f"d_pairs_{stamp()}"
    mpath = config.path(CFG["paths"]["outputs"]) / "manifests" / f"{name}.json"
    C.write_manifest(mpath, name, [{"run": str(run.relative_to(config.ROOT)).replace("\\", "/"), "key": k} for k in eps],
                     "Step D completion condition 5: the rendered pairs (empty 10 pairs, prefilled_1 5 pairs)")
    out = config.path(CFG["paths"]["outputs"]) / "datasets" / name
    exe = sys.executable
    log = OUT / f"convert_{name}.log"
    OUT.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8") as f:
        c1 = subprocess.run([exe, "-m", "recovla.data.convert", "--manifest", str(mpath), "--out", str(out),
                             "--name", name], stdout=f, stderr=subprocess.STDOUT)
        c2 = subprocess.run([exe, "-m", "recovla.data.convert", "--verify", str(out), "--export-actions",
                             str(OUT / f"{name}_actions")], stdout=f, stderr=subprocess.STDOUT)
    text = log.read_text(encoding="utf-8", errors="replace")
    checks = [l.strip() for l in text.splitlines() if l.strip().startswith(("ok ", "FAIL "))]
    conv = json.loads((out / "meta" / "conversion.json").read_text(encoding="utf-8"))
    flags_ok = all(s.get("target") and s.get("pair_id") is not None and s.get("layout_kind") for s in conv["sources"])
    # 10 fps の行動（LeRobot から読み戻したもの）で手先を再生（5 mm）
    lr = lerobot_replay(OUT / f"{name}_actions")
    write("convert", {"manifest": str(mpath.relative_to(config.ROOT)), "dataset": str(out.relative_to(config.ROOT)),
                      "convert_exit": c1.returncode, "verify_exit": c2.returncode, "verify_pass": "[verify] PASS" in text,
                      "checks_ok": sum(c.startswith("ok") for c in checks), "checks_fail": [c for c in checks if c.startswith("FAIL")],
                      "task_color_checks": sum("task color == meta target" in c and c.startswith("ok") for c in checks),
                      "episodes": len(conv["sources"]), "flags_carried": bool(flags_ok),
                      "two_views_value_level": sum(("images == vla_image_spec" in c or "differs from every other" in c)
                                                   and c.startswith("ok") for c in checks),
                      "lerobot_actions_replay": lr, "log": str(log.relative_to(config.ROOT))})


def lerobot_replay(export_dir: pathlib.Path) -> dict:
    from recovla.record import replay as legacy, replay_scene as RS
    from recovla.sim.rig import SimRig
    rig = SimRig(render=False)
    rows = []
    try:
        for f in sorted(export_dir.glob("*_lerobot.npz")):
            z = np.load(f, allow_pickle=False)
            ep = legacy.load_episode(str(z["raw_path"]))
            stride = int(20 // int(z["fps"]))
            rep = RS.replay(ep, "actions", rig, actions=z["action"], frame_stride=stride, render=False)
            idx = np.arange(len(z["action"]) + 1) * stride
            err = np.linalg.norm(rep["ee_pos"][idx] - ep.data["ee_pos"][idx], axis=1)
            rows.append({"episode": f.name, "ee_err_max_m": float(err.max()), "success": bool(rep["success"])})
    finally:
        rig.close()
    return {"episodes": len(rows), "ee_err_max_m": max(r["ee_err_max_m"] for r in rows) if rows else None,
            "all_within_tol": all(r["ee_err_max_m"] < RS.WINDOW_TOL_M for r in rows), "all_success": all(r["success"] for r in rows)}


# ------------------------------------------------------------------------------------------ handover

def check_handover() -> None:
    specs = []
    for seed in SEEDS["handover"]:
        lay = scene.sample_layout(seed, None)
        rng = seeds.stream(seed, "order")
        color = lay.table_colors[int(rng.integers(len(lay.table_colors)))]
        specs.append(G.EpisodeSpec(seed, color, None, "h", float(rng.uniform(*HANDOVER_U))))
    run = GEN / f"d_handover_{stamp()}"
    results = G.generate(specs, run, workers=8, render=False)
    ph = collections.Counter(r["attempts"][0]["handover"]["phase_before"] for r in results if r["attempts"][0]["handover"])
    first = sum(r["first_try_success"] for r in results)
    write("handover", {"run": str(run.relative_to(config.ROOT)), "trials": len(results), "first_try_success": first,
                       "first_try_rate": first / len(results), "phase_at_handover": dict(ph),
                       "kinds": dict(collections.Counter(r["attempts"][0]["layout"]["kind"] for r in results)),
                       "failures": [(r["layout_seed"], r["color"], a["failure"]) for r in results for a in r["attempts"] if a["failure"]],
                       "u_range": HANDOVER_U})


# ---------------------------------------------------------------------------------------- throughput

def check_throughput() -> None:
    specs = G.plan_specs(SEEDS["throughput"])
    rows = []
    for w in WORKERS_THROUGHPUT:
        run = GEN / f"d_throughput_w{w}_{stamp()}"
        t0 = time.time()
        res = G.generate(specs, run, workers=w, render=True)
        wall = time.time() - t0
        per = [r["worker_wall_s"] for r in res]
        sim = [r["attempts"][-1]["duration_s"] for r in res]
        rows.append({"workers": w, "episodes": len(res), "wall_s": round(wall, 2),
                     "episodes_per_hour": round(len(res) / wall * 3600, 1),
                     "per_episode_wall_s": pct(per), "episode_sim_s": pct(sim),
                     "run": str(run.relative_to(config.ROOT))})
    import os
    write("throughput", {"rows": rows, "cpu_count": os.cpu_count(), "render": True,
                         "note": "起動（spawn と各プロセスの SimRig の作成）を含む実時間。学習と同時には回していない"})


# -------------------------------------------------------------------------------------------- videos

def check_videos() -> None:
    import cv2
    import shutil
    import tempfile
    from recovla.record.recorder import read_png
    run = latest_run("d_pairs_w1")
    vdir = OUT / "videos"
    vdir.mkdir(parents=True, exist_ok=True)
    reps = [("empty", s) for s in list(SEEDS["pairs"]["empty"])[:3]] + [("prefilled_1", list(SEEDS["pairs"]["prefilled_1"])[0])]
    made = []
    for kind, seed in reps:
        eps = sorted(p for p in run.glob(f"n_{seed}_*_r*") if p.is_dir())
        for p in eps:
            n = json.loads((p / "meta.json").read_text(encoding="utf-8"))["n_frames"]
            tmp = pathlib.Path(tempfile.mkdtemp()) / "v.mp4"
            vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 20, (512, 256))
            for i in range(n):
                im = np.hstack([read_png(p / v / f"{i:06d}.png") for v in CFG["sim"]["cameras"]])
                vw.write(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            vw.release()
            dst = vdir / f"{kind}_{p.name}.mp4"
            shutil.move(str(tmp), str(dst))
            made.append(str(dst.relative_to(config.ROOT)))
    write("videos", {"run": str(run.relative_to(config.ROOT)), "videos": made,
                     "pairs": [{"kind": k, "layout_seed": s} for k, s in reps]})


CHECKS = {"scene": check_scene, "physics": check_physics, "pairs": check_pairs, "replay": check_replay,
          "convert": check_convert, "handover": check_handover, "throughput": check_throughput, "videos": check_videos}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checks", nargs="+", choices=list(CHECKS) + ["all"])
    a = ap.parse_args(argv)
    names = list(CHECKS) if "all" in a.checks else a.checks
    for n in names:
        print(f"[d] === {n}")
        CHECKS[n]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
