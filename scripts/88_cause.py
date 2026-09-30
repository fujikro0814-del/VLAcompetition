"""0121 の原因の調べ（■ 2）。学習用・検証用の種の記録だけを使う（テスト用の種 130000〜 の出力は outputs/sealed/ にあり、読まない）。

    .venv\\Scripts\\python.exe scripts\\88_cause.py a1          # 描画の破壊の混入（知覚の記録と真値の照合で、こまごとに）
"""
import argparse
import json
import pathlib
import sys

import numpy as np

from recovla.common import config
from recovla.common.seeds import COLORS
from recovla.sim import frames

CFG = config.load()
OUT = config.path(CFG["paths"]["outputs"])
RES = OUT / "results"
BOX = np.asarray(CFG["scene"]["box"]["pos"], float)

# 描画が壊れたこまは深度が全画素で約 112 m になり（0120）、知覚は立方体も箱も見つけられないか、とんでもない位置を返す。
# 大外れの境: 机の上の立方体の推定が真値から 10 cm 超（E10 の p95 は約 2.9 cm、知覚の遅れ 0.16 s の間の動きも数 cm）。
FAR_M = 0.10
SEEN = ("seen", "held")

A1_DIRS = [("V2S1", "段階 1 の最終評価（120000〜、有効の 26 組）"),
           ("V2S1_invalid_0930", "0120 で無効とした組（検出できることの確かめ）"),
           ("V2SEL", "保存点の選択（199700〜）"),
           ("V2SF_s1", "安全フィルタの判定（199800〜）"),
           ("V2DEVE", "E の段取りの試し（59800〜）"),
           ("V2DEV", "0110 ごろの診断（学習用の種）"),
           ("V2DBG", "0120 の診断（59900〜）")]


def trial_frames(d: pathlib.Path, i: str) -> dict:
    """1 試行の知覚のこまごとの判定。"""
    z = np.load(d / f"trial_{i}.npz")
    rt = json.loads((d / f"runtime_{i}.json").read_text(encoding="utf-8"))["runtime"]
    t = z["sim_time"]
    n_frames, bad, blind, far, box_bad = 0, [], 0, 0, 0
    flags = []
    worst = 0.0
    for e in rt.get("perception", []):
        n_frames += 1
        k = int(np.argmin(np.abs(t - e["t_obs"])))
        seen_any, far_here = False, False
        for c, v in e["cubes"].items():
            if v["status"] in SEEN or v["status"] == "in_hand":
                seen_any = True
            tp = z["cube_pos"][k, COLORS.index(c)]
            on_table = tp[2] < 0.03 and not frames.in_box(tp, BOX)
            if on_table and v["status"] in SEEN and v.get("pos") is not None:
                err = float(np.linalg.norm(np.asarray(v["pos"]) - tp))
                worst = max(worst, err)
                if err > FAR_M:
                    far_here = True
        b = e.get("box") or {}
        if not b.get("ok", False):
            box_bad += 1
        if not seen_any:
            blind += 1
        if far_here:
            far += 1
        flags.append((not seen_any) or far_here)
        if flags[-1]:
            bad.append(round(float(e["t_obs"]), 2))
    # 壊れは戻らない（0120）: 最初の異常から終わりまでほぼ全部が異常で、最後の TAIL こまが全部異常なら「壊れの形」
    TAIL = 5
    persistent = False
    if bad and len(flags) >= TAIL and all(flags[-TAIL:]):
        k0 = flags.index(True)
        persistent = float(np.mean(flags[k0:])) >= 0.9
    st = rt.get("startup") or {}
    tb = st.get("table") or {}
    fit_fail = bool(tb) and tb.get("dz_m") == 0.0 and tb.get("angle_deg") == 0.0
    box_nf = not (st.get("box") or {}).get("ok", True)
    n_inf = len(rt.get("inference", []))
    return {"trial": i, "frames": n_frames, "bad_frames": len(bad), "blind": blind, "far": far, "box_not_ok": box_bad,
            "first_bad_t": bad[0] if bad else None, "worst_err_m": round(worst, 4), "startup_fit_fail": fit_fail,
            "startup_box_nf": box_nf, "n_inference": n_inf, "stop_reason": rt.get("stop_reason"),
            "persistent": persistent, "corrupt_like": persistent or fit_fail}


def cmd_a1(a) -> None:
    out = {"far_m": FAR_M, "groups": {}}
    for top, label in A1_DIRS:
        root = OUT / "v2eval" / top
        if not root.is_dir():
            continue
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            ids = sorted(p.stem.split("_")[1] for p in d.glob("runtime_*.json") if (d / f"trial_{p.stem.split('_')[1]}.npz").is_file())
            if not ids:
                continue
            rows = [trial_frames(d, i) for i in ids]
            anom = [r for r in rows if r["bad_frames"] or r["startup_fit_fail"] or r["startup_box_nf"]]
            key = f"{top}/{d.name}"
            out["groups"][key] = {
                "label": label, "trials": len(rows), "frames": sum(r["frames"] for r in rows),
                "bad_frames": sum(r["bad_frames"] for r in rows), "anomalous_trials": len(anom),
                "corrupt_like_trials": sum(r["corrupt_like"] for r in rows),
                "persistent_trials": sum(r["persistent"] for r in rows),
                "fit_fail_trials": sum(r["startup_fit_fail"] for r in rows),
                "anomalous": [{k: r[k] for k in ("trial", "frames", "bad_frames", "blind", "far", "first_bad_t", "worst_err_m",
                                                 "startup_fit_fail", "startup_box_nf", "n_inference", "stop_reason",
                                                 "persistent", "corrupt_like")} for r in anom],
                "worst_err_m": max(r["worst_err_m"] for r in rows)}
            g = out["groups"][key]
            print(f"{key:36s} trials {g['trials']:3d} frames {g['frames']:6d} bad {g['bad_frames']:5d} "
                  f"any_bad {g['anomalous_trials']:3d} corrupt_like {g['corrupt_like_trials']:3d} "
                  f"(persist {g['persistent_trials']}, fitfail {g['fit_fail_trials']}) worst {g['worst_err_m']:.3f}", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "cause_a1.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


# 壊れたこまの再現（0121 の A2）: 手首の色の画像は全画素が黒（std 0）、俯瞰の色の画像は変わらず深度だけが 112 m。
# 学習データは色の画像だけなので、黒いこま（最大の画素値 < 10 の割合が 0.5 超）と平坦なこま（std < 2）を異常とする。
A2_DATASETS = {"R1v2": "R1v2_20260929-205251", "N1v2": "N1v2_20260929-205251"}
VIEWS = ("observation.images.image", "observation.images.image2")


def cmd_a2(a) -> None:
    import io

    import pyarrow.parquet as pq
    from PIL import Image
    rng = np.random.default_rng(20261001)
    out = {"rule": "dark_frac > 0.5 or std < 2", "datasets": {}}
    for name, ds in A2_DATASETS.items():
        root = OUT / "datasets" / ds
        stats = {v: {"std": [], "dark": [], "bright": [], "mean": []} for v in VIEWS}
        bad, n = [], 0
        files = sorted((root / "data").rglob("*.parquet"))
        for f in files:
            t = pq.read_table(f, columns=[*VIEWS, "episode_index", "frame_index"]).to_pydict()
            for r in range(len(t["episode_index"])):
                n += 1
                for v in VIEWS:
                    g = np.asarray(Image.open(io.BytesIO(t[v][r]["bytes"])).convert("RGB"), dtype=np.float32)
                    sd, dk, br = float(g.std()), float((g.max(axis=2) < 10).mean()), float((g.min(axis=2) > 245).mean())
                    s = stats[v]
                    s["std"].append(sd); s["dark"].append(dk); s["bright"].append(br); s["mean"].append(float(g.mean()))
                    if dk > 0.5 or sd < 2:
                        bad.append({"episode": t["episode_index"][r], "frame": t["frame_index"][r], "view": v, "std": sd, "dark": dk})
            print(f"[a2] {name} {f.name} frames {n} bad {len(bad)}", flush=True)
        summ = {}
        for v, s in stats.items():
            summ[v] = {k: {"min": float(np.min(x)), "p01": float(np.percentile(x, 1)), "median": float(np.median(x)),
                           "p99": float(np.percentile(x, 99)), "max": float(np.max(x))} for k, x in s.items()}
        out["datasets"][name] = {"dataset": ds, "frames": n, "bad_frames": len(bad), "bad": bad[:200], "stats": summ}
        _grid(root, files, rng, OUT / "results" / f"cause_a2_grid_{name}.png")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "cause_a2.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: {"frames": v["frames"], "bad_frames": v["bad_frames"]} for k, v in out["datasets"].items()}))


def _grid(root, files, rng, path) -> None:
    """無作為に 20 こま（俯瞰と手首の組）を 4 列 × 5 行に並べる。"""
    import io

    import pyarrow.parquet as pq
    from PIL import Image, ImageDraw
    idx = []
    sizes = [pq.ParquetFile(f).metadata.num_rows for f in files]
    total = sum(sizes)
    for g in sorted(rng.choice(total, 20, replace=False)):
        k = 0
        while g >= sizes[k]:
            g -= sizes[k]; k += 1
        idx.append((k, int(g)))
    W = 256
    canvas = Image.new("RGB", (4 * 2 * W + 3 * 8, 5 * (W + 18)), "white")
    d = ImageDraw.Draw(canvas)
    for j, (k, r) in enumerate(idx):
        t = pq.read_table(files[k], columns=[*VIEWS, "episode_index", "frame_index"]).slice(r, 1).to_pydict()
        x0, y0 = (j % 4) * (2 * W + 8), (j // 4) * (W + 18)
        for vi, v in enumerate(VIEWS):
            canvas.paste(Image.open(io.BytesIO(t[v][0]["bytes"])).convert("RGB"), (x0 + vi * W, y0 + 18))
        d.text((x0 + 2, y0 + 2), f"ep {t['episode_index'][0]} frame {t['frame_index'][0]}", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def cmd_a3(a) -> None:
    """学習時と実行時の入力の一致。実行系の SensorPolicy.observe（模型は読まない）に、生の記録のこまを SensorFrame の形で渡す。
    手がかりの較正は、学習データと同じ信じている較正（テーブル面での補正の前）。補正の影響は別に数える（B・D）。"""
    import io
    import types

    import pyarrow.parquet as pq
    from PIL import Image

    from recovla.data import convert as CV
    from recovla.harness import setup as HS
    from recovla.runtime import cue as C
    from recovla.runtime.motion import Motion
    from recovla.runtime.policy import SensorPolicy
    cfg2 = config.load_v2()
    thr = C.Thresholds.from_dict(config.color_detect(cfg2))
    setup = HS.nominal_setup(cfg2)
    fk = Motion(setup).hand_pose
    root = OUT / "datasets" / A2_DATASETS["R1v2"]
    conv = json.loads((root / "meta" / "conversion.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(20261002)
    eps = sorted(int(e) for e in rng.choice(len(conv["sources"]), 10, replace=False))
    files = sorted((root / "data").rglob("*.parquet"))
    rows = {}
    for f in files:
        t = pq.read_table(f, columns=[*VIEWS, "observation.state", "episode_index", "frame_index", "task_index"]).to_pydict()
        for r, e in enumerate(t["episode_index"]):
            if e in eps:
                rows.setdefault(e, []).append({k: t[k][r] for k in t})
    tasks = {}
    tp = root / "meta" / "tasks.parquet"
    if tp.is_file():
        tt = pq.read_table(tp).to_pandas()
        tasks = {int(i): str(n) for n, i in zip(tt.index, tt["task_index"])} if "task_index" in tt else {}
    pol = SensorPolicy.__new__(SensorPolicy)
    pol.fk, pol.cue_keep_flag, pol.last_cue = fk, False, None
    res = {"episodes": [], "thresholds_training": conv["target_cue"]["thresholds"], "thresholds_runtime": thr.to_json()}
    worst = {"image": 0.0, "state": 0.0, "cue": 0.0}
    for e in eps:
        src = conv["sources"][e]
        path = pathlib.Path(src["raw_path"])
        meta, data = CV.load_raw(path)
        cc = meta["cue_calibration"]
        pol.cue = C.TargetCue(C.Calibration(np.array(cc["pos"], float), np.array(cc["rot"], float), float(cc["f"]),
                                            int(cc["width"]), int(cc["height"])), thr, float(cc["plane_z"]), cc["fallback"])
        pol.cue.reset()
        arr = CV.episode_arrays(meta, data)
        er = sorted(rows[e], key=lambda x: x["frame_index"])
        d_img, d_state, d_cue = 0.0, 0.0, 0.0
        for k, i in enumerate(arr["raw_index"]):
            raw = {v: CV.read_raw_image(path / v / f"{i:06d}.png") for v in ("overhead", "wrist")}
            sensor = types.SimpleNamespace(
                cameras={v: types.SimpleNamespace(rgb=raw[v]) for v in raw},
                joints=types.SimpleNamespace(q=np.asarray(data["joints"][i], float)),
                gripper=types.SimpleNamespace(width=float(2.0 * data["fingers"][i][0])))
            obs = pol.observe(sensor, meta["instruction"], cue_color=meta["target"])
            ds = er[k]
            for vk in VIEWS:
                im = np.asarray(Image.open(io.BytesIO(ds[vk]["bytes"])).convert("RGB"), np.float32).transpose(2, 0, 1) / 255.0
                d_img = max(d_img, float(np.abs(obs[vk] - im).max()))
            s_rt, s_ds = obs["observation.state"], np.asarray(ds["observation.state"], np.float32)
            d_state = max(d_state, float(np.abs(s_rt[:15] - s_ds[:15]).max()))
            d_cue = max(d_cue, float(np.abs(s_rt[15:] - s_ds[15:]).max()))
        res["episodes"].append({"episode": e, "raw": src["raw_episode"], "frames": len(er), "max_abs_image": d_img,
                                "max_abs_state": d_state, "max_abs_cue": d_cue, "task_dataset": tasks.get(er[0]["task_index"]),
                                "task_runtime_template": CFG["convert"]["instruction"].format(color=meta["target"])})
        worst = {"image": max(worst["image"], d_img), "state": max(worst["state"], d_state), "cue": max(worst["cue"], d_cue)}
        print(f"[a3] ep {e} {src['raw_episode']} frames {len(er)} image {d_img:.5f} state {d_state:.2e} cue {d_cue:.2e}", flush=True)
    res["worst"] = worst
    res["pass"] = {"image": worst["image"] <= 1 / 255 + 1e-7, "state": worst["state"] <= 1e-6, "cue": worst["cue"] <= 1e-6}
    (RES / "cause_a3.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"worst": worst, "pass": res["pass"]}))


def cmd_a4(a) -> None:
    """行動の再生: 学習データの通常のエピソード（n_）の行動（変換と同じ 10 Hz の x_des の差とグリッパ）を、評価の枠
    （run_policy_trial）と実行系の PolicyRuntime の動きの口（Motion＝IK・制限層、関節の指令、ハンドの move/grasp）で再生する。
    推論の代わりに、区切り k で k 行目を渡す（推論の遅れはない）。知覚と安全フィルタは使わない（動きの口だけを見る）。
    センサの種は生成と同じ（meta の sensor_seed。run_policy_trial は試行の種をセンサの種に使う）。"""
    from recovla.data import convert as CV
    from recovla.harness.loop import run_policy_trial
    from recovla.harness.sensors import SensorSuite
    from recovla.harness.world import WorldRig
    from recovla.runtime.motion import Motion
    from recovla.runtime.runner import PolicyRuntime
    from recovla.sim import scene

    class ReplayRuntime(PolicyRuntime):
        actions = None

        def _action_boundary(self) -> None:
            k, t = self.k, self.io.now()
            if k < len(self.actions):
                a, held = np.asarray(self.actions[k], float), False
            else:
                a, held = np.array([0, 0, 0, 0, 0, 0, 1.0 if self.closed else -1.0]), True
            self._apply(a)
            self.log_act.append((k, t, None if held else 0, held, a.copy()))

    class _NoPolicy:
        def start_trial(self, seed):
            pass

        def reset_cue(self):
            pass

    cfg2 = config.load_v2()
    act = cfg2["actuation"]
    root = OUT / "datasets" / A2_DATASETS["R1v2"]
    conv = json.loads((root / "meta" / "conversion.json").read_text(encoding="utf-8"))
    normal = [s for s in conv["sources"] if s["kind"] == "n"]
    rng = np.random.default_rng(20261003)
    pick = [normal[i] for i in sorted(rng.choice(len(normal), a.n, replace=False))][a.shard::a.shards]
    world = WorldRig(render=False, cfg=cfg2)
    suite = SensorSuite(world.model, cfg2)
    rows = []
    for src in pick:
        path = pathlib.Path(src["raw_path"])
        meta, data = CV.load_raw(path)
        arr = CV.episode_arrays(meta, data)
        L = meta["layout"]
        lay = scene.sample_layout(int(meta["layout_seed"]), L["kind"], start=L["start"])
        holder = {}

        def make(io, setup):
            rt = ReplayRuntime(io, setup, _NoPolicy(), mode="naive", motion=Motion(setup, margin=float(act["limiter_margin"])))
            rt.actions = arr["action"]
            holder["rt"] = rt
            return rt
        tl = max(float(cfg2["eval"]["time_limit_s"]), float(meta["duration_s"]) + 5.0)
        m, arrays, rlog = run_policy_trial(world, suite, make, lay, meta["target"], int(meta["sensor_seed"]), time_limit_s=tl, cfg=cfg2)
        n = min(len(arrays["ee_pos"]), len(data["ee_pos"]))
        n_act = min(n, 2 * len(arr["action"]) + 1)
        d = np.linalg.norm(arrays["ee_pos"][:n_act] - data["ee_pos"][:n_act], axis=1)
        dx = np.linalg.norm(arrays["x_des"][:n_act] - data["x_des"][:n_act], axis=1)
        r = {"raw": src["raw_episode"], "episode": src["episode_index"], "orig_success": bool(meta["success"]),
             "replay_success": bool(m["success"]), "t_success": m["t_success"], "orig_duration_s": meta["duration_s"],
             "ee_rms_m": float(np.sqrt(np.mean(d ** 2))), "ee_max_m": float(d.max()),
             "xcmd_rms_m": float(np.sqrt(np.mean(dx ** 2))), "g3": m["audit"]["g3"]["total_violations"] if "audit" in m else None,
             "motion": rlog["runtime"].get("motion"), "t_end": m["t_end"]}
        rows.append(r)
        print(f"[a4] {r['raw']} success {r['orig_success']}->{r['replay_success']} ee_rms {r['ee_rms_m'] * 1000:.1f} mm "
              f"max {r['ee_max_m'] * 1000:.1f} mm xcmd_rms {r['xcmd_rms_m'] * 1000:.1f} mm", flush=True)
    suite.close()
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"cause_a4_{a.shard}of{a.shards}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


def _dist(x) -> dict:
    x = np.asarray(x, float)
    return {"n": int(x.size), "median_mm": float(np.median(x) * 1e3), "p95_mm": float(np.percentile(x, 95) * 1e3),
            "max_mm": float(x.max() * 1e3)}


def cmd_b1(a) -> None:
    """B1: |x_cmd − 手先| の分布。エキスパートの生データは x_des（＝ x_cmd）と ee_pos（v1 は真値、v2 は測った関節角の順運動学）。
    方策の走行（trial_NNNN.npz）は x_des（実行系の x_cmd）と ee_pos（真値）。積み上がりは試行の中の時間の 4 区間で見る。"""
    out = {}
    for name, run in (("v1_expert", "F_data_20260926-113711"), ("v2_expert", "F_data_v2_20260929-205251")):
        ds, per_ep = [], []
        for p in sorted((OUT / "gen" / run).iterdir()):
            f = p / "data.npz"
            if not f.is_file():
                continue
            with np.load(f) as z:
                d = np.linalg.norm(z["x_des"] - z["ee_pos"], axis=1)
            ds.append(d)
            per_ep.append(float(np.percentile(d, 95)))
        out[name] = {**_dist(np.concatenate(ds)), "episodes": len(ds), "per_episode_p95_median_mm": float(np.median(per_ep) * 1e3)}
    for name, top in (("R1v2_runs_V2SEL_20000", "V2SEL/R1v2_20000"), ("N1v2_runs_V2SEL_20000", "V2SEL/N1v2_20000"),
                      ("R2_runs_V2S1_F_nat", "V2S1/F_nat")):
        ds, quarters = [], [[], [], [], []]
        for f in sorted((OUT / "v2eval" / top).glob("trial_*.npz")):
            with np.load(f) as z:
                d = np.linalg.norm(z["x_des"] - z["ee_pos"], axis=1)
            ds.append(d)
            for qi, part in enumerate(np.array_split(d, 4)):
                if part.size:
                    quarters[qi].append(float(np.median(part)))
        out[name] = {**_dist(np.concatenate(ds)), "trials": len(ds),
                     "median_by_time_quarter_mm": [float(np.median(q) * 1e3) for q in quarters]}
    for k, v in out.items():
        print(k, json.dumps(v))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "cause_b1.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


LIFT_Z = 0.045          # 立方体（一辺 4 cm、机の上で中心 z≈0.02）が持ち上がったとみなす高さ
NEAR_M = 0.05           # グリッパの切り替えが「掴む場面」とみなす指先と目標の立方体の距離


def _grip_analysis(t, closed, tip, cube, width, ee_z) -> dict:
    """B3・B4 の 1 試行（または 1 エピソード）分。t [s]、closed (N,) bool、tip (N,3)、cube (N,3) は目標の立方体。"""
    tr = np.where(np.diff(closed.astype(int)) != 0)[0] + 1
    near = [i for i in tr if np.linalg.norm(tip[i] - cube[i]) < NEAR_M]
    round_1s = [(a, b) for a, b in zip(tr[:-1], tr[1:]) if t[b] - t[a] <= 1.0]
    round_near = [(a, b) for a, b in round_1s if a in near or b in near]
    closes = []
    for i in tr:
        if not closed[i] or np.linalg.norm(tip[i] - cube[i]) >= NEAR_M:
            continue
        j = i
        while j + 2 < len(t) and closed[j + 1] and width[j + 1] < width[j] - 1e-4:
            j += 1
        k = i
        while k + 1 < len(t) and closed[k + 1] and ee_z[k + 1] < ee_z[i] + 0.01:
            k += 1
        end = min(len(t), i + int(3.0 / max(1e-6, t[1] - t[0])))
        lifted = bool(np.any(cube[i:end, 2] > LIFT_Z))
        closes.append({"t_close": float(t[i]), "close_to_closed_s": float(t[j] - t[i]),
                       "lift_minus_closed_s": float(t[k] - t[j]) if closed[k] else None,
                       "width_at_closed_mm": float(width[j] * 1e3), "lifted_within_3s": lifted})
    return {"toggles": int(len(tr)), "roundtrips_1s": len(round_1s), "roundtrips_1s_near": len(round_near),
            "closes_near": closes}


def _failure_stage(m, z, ti, rt) -> str:
    if m["success"]:
        return "success"
    if rt.get("stop_reason"):
        return "stopped:" + str(rt["stop_reason"])
    cube = z["cube_pos"][:, ti]
    closed = z["gripper_closed"].astype(bool)
    tip = z["fingertip"]
    near_close = np.any(closed & (np.linalg.norm(tip - cube, axis=1) < NEAR_M))
    lifted = cube[:, 2] > LIFT_Z
    if not near_close:
        return "F1_never_closed_near_target"
    if not lifted.any():
        return "F2_closed_but_never_lifted"
    over = np.array([frames.in_box(np.array([p[0], p[1], 0.03]), BOX) for p in cube])
    if not np.any(lifted & over):
        return "F3_lifted_not_over_box"
    if z["cube_in_box"][-1, ti]:
        return "F4_in_box_not_settled"
    return "F5_over_box_but_outside_at_end"


def cmd_b(a) -> None:
    """B3・B4・B5・D1〜D3・E1 と、失敗の段階の内訳（方策の走行）。エキスパート（生成の生データ）は B3・B4・E1 だけ。"""
    import collections
    out = {"policy": {}, "expert": {}}
    for top in a.dirs:
        d = OUT / "v2eval" / top
        if not d.is_dir():
            continue
        stages = collections.Counter()
        b3_trials_round_near, b3_toggles, closes = 0, [], []
        lat_s, lat_f, cue_err = [], [], []
        safety_frac, timeouts, stalled = [], 0, 0
        n = 0
        for p in sorted(d.glob("trial_*.json")):
            i = p.stem.split("_")[1]
            m = json.loads(p.read_text(encoding="utf-8"))
            rtp = d / f"runtime_{i}.json"
            if not rtp.is_file():
                continue
            rt = json.loads(rtp.read_text(encoding="utf-8"))["runtime"]
            z = np.load(d / f"trial_{i}.npz")
            ti = COLORS.index(m["target"])
            n += 1
            st = _failure_stage(m, z, ti, rt)
            stages[st.split(":")[0] if st.startswith("stopped") else st] += 1
            if st.startswith("stopped"):
                stages[st] += 0
            t = z["sim_time"]
            cube = z["cube_pos"][:, ti]
            width = z["fingers"].sum(axis=1) if z["fingers"].ndim == 2 else z["fingers"]
            g = _grip_analysis(t, z["gripper_closed"].astype(bool), z["fingertip"], cube, width, z["ee_pos"][:, 2])
            b3_toggles.append(g["toggles"])
            b3_trials_round_near += int(g["roundtrips_1s_near"] > 0)
            closes += g["closes_near"]
            lat = [e["latency_s"] for e in rt.get("inference", [])]
            if lat:
                (lat_s if m["success"] else lat_f).append(float(np.mean(lat)))
            inf = rt.get("inference", [])
            if inf and inf[0].get("cue") is not None:
                k = int(np.argmin(np.abs(t - inf[0]["t_obs"])))
                cue_err.append((float(np.linalg.norm(np.asarray(inf[0]["cue"][:2]) - cube[k, :2])), bool(m["success"])))
            sf = rt.get("safety") or {}
            if sf.get("enabled"):
                safety_frac.append(float(sf.get("active_physics_steps", 0)) / max(1.0, float(m["t_end"]) / 0.002))
            if not m["success"] and not rt.get("stop_reason"):
                timeouts += 1
                last = t > t[-1] - 10.0
                path = float(np.sum(np.linalg.norm(np.diff(z["ee_pos"][last], axis=0), axis=1)))
                stalled += int(path < 0.02)
        ce = np.array([c[0] for c in cue_err]) if cue_err else np.array([])
        cs = np.array([c[1] for c in cue_err]) if cue_err else np.array([])
        out["policy"][top] = {
            "trials": n, "stages": dict(stages),
            "B3": {"toggles_median": float(np.median(b3_toggles)) if b3_toggles else None,
                   "trials_with_roundtrip_1s_near_target": b3_trials_round_near,
                   "frac_trials_with_roundtrip_near": b3_trials_round_near / max(1, n)},
            "B4": {"closes_near_target": len(closes),
                   "lifted_within_3s_rate": float(np.mean([c["lifted_within_3s"] for c in closes])) if closes else None,
                   "close_to_closed_s_median": float(np.median([c["close_to_closed_s"] for c in closes])) if closes else None,
                   "lift_minus_closed_s_median": float(np.median([c["lift_minus_closed_s"] for c in closes
                                                                  if c["lift_minus_closed_s"] is not None])) if closes else None,
                   "frac_lift_before_closed": float(np.mean([c["lift_minus_closed_s"] is not None and c["lift_minus_closed_s"] < 0
                                                             for c in closes])) if closes else None,
                   "closes_per_trial": len(closes) / max(1, n)},
            "B5": {"mean_latency_success_s": float(np.median(lat_s)) if lat_s else None,
                   "mean_latency_failure_s": float(np.median(lat_f)) if lat_f else None},
            "D2": {"n": int(ce.size), "cue_err_median_mm": float(np.median(ce) * 1e3) if ce.size else None,
                   "cue_err_p95_mm": float(np.percentile(ce, 95) * 1e3) if ce.size else None,
                   "success_rate_err_lt_2cm": float(cs[ce < 0.02].mean()) if np.any(ce < 0.02) else None,
                   "success_rate_err_ge_2cm": float(cs[ce >= 0.02].mean()) if np.any(ce >= 0.02) else None,
                   "n_err_ge_2cm": int(np.sum(ce >= 0.02))},
            "D3": {"safety_active_frac_median": float(np.median(safety_frac)) if safety_frac else None,
                   "safety_active_frac_p90": float(np.percentile(safety_frac, 90)) if safety_frac else None},
            "E1": {"failures": n - stages.get("success", 0), "timeouts": timeouts, "stalled_last_10s": stalled}}
        print(top, json.dumps(out["policy"][top], ensure_ascii=False), flush=True)
    for name, run in (("v1", "F_data_20260926-113711"), ("v2", "F_data_v2_20260929-205251")):
        dur, closes, tog, rnear, ne = [], [], [], 0, 0
        for p in sorted((OUT / "gen" / run).iterdir()):
            if not (p / "meta.json").is_file() or not p.name.startswith("n_"):
                continue
            meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
            if not meta.get("success"):
                continue
            dur.append(float(meta["duration_s"]))
            with np.load(p / "data.npz") as z:
                ti = COLORS.index(meta["target"])
                g = _grip_analysis(z["sim_time"], z["gripper_closed"].astype(bool), z["fingertip"], z["cube_pos"][:, ti],
                                   z["fingers"].sum(axis=1), z["ee_pos"][:, 2])
            ne += 1
            tog.append(g["toggles"])
            rnear += int(g["roundtrips_1s_near"] > 0)
            closes += g["closes_near"]
        out["expert"][name] = {
            "episodes": ne, "E1_duration_median_s": float(np.median(dur)), "E1_duration_p95_s": float(np.percentile(dur, 95)),
            "E1_duration_max_s": float(np.max(dur)),
            "B3": {"toggles_median": float(np.median(tog)), "frac_with_roundtrip_near": rnear / max(1, ne)},
            "B4": {"closes_near_target": len(closes),
                   "lifted_within_3s_rate": float(np.mean([c["lifted_within_3s"] for c in closes])),
                   "close_to_closed_s_median": float(np.median([c["close_to_closed_s"] for c in closes])),
                   "lift_minus_closed_s_median": float(np.median([c["lift_minus_closed_s"] for c in closes
                                                                  if c["lift_minus_closed_s"] is not None])),
                   "frac_lift_before_closed": float(np.mean([c["lift_minus_closed_s"] is not None and c["lift_minus_closed_s"] < 0
                                                             for c in closes]))}}
        print("expert", name, json.dumps(out["expert"][name], ensure_ascii=False), flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"cause_b_{a.tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("a1", "a2", "a3", "b1"):
        sub.add_parser(c)
    p = sub.add_parser("a4")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p = sub.add_parser("b")
    p.add_argument("--dirs", nargs="+", default=["V2SEL/R1v2_20000", "V2SEL/N1v2_20000", "V2S1/F_nat"])
    p.add_argument("--tag", default="pre")
    a = ap.parse_args(argv)
    {"a1": cmd_a1, "a2": cmd_a2, "a3": cmd_a3, "a4": cmd_a4, "b1": cmd_b1, "b": cmd_b}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
