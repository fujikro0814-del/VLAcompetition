"""Step I の上位層の道具（掲示板 0088・0089）。

    .venv\\Scripts\\python.exe scripts\\51_planner.py fit         # 色の「ある」の閾値を学習データの描画だけで決める
    ... run --seeds 197000:20 --tag E7_pre / judge-data --seeds 196000:40 --tag pre / llm-eval --fixture dev_sentences

fit: R1 の学習データ（マニフェスト R1_20260926-113711 の 330 本の生の記録。学習用の帯の種 20000〜・30000〜）の俯瞰の
こまから、色ごとの画素数を数え、真値で「ある」「ない」を分ける閾値を決める。選択用・最終評価用の種では調整しない。
  箱: 指が開いているこま（持って運ぶ途中は完了判定のグリッパの条件で落とすので除く）を 5 こまおきに。真値は
      frames.in_box（成功の体積の中）。3 色それぞれを 1 件として数える
  机: 各エピソードの最初と最後のこま。真値は「箱の外にあり、机の上（中心の高さが静止の高さ ± 1 cm）」
  閾値は、正しく分けた件数が最も多い値（同点なら小さい方）。結果は outputs/planner/presence_fit.json と configs
"""
import argparse
import json
import pathlib
import time

import numpy as np

from recovla.common import config
from recovla.common.seeds import COLORS

CFG = config.load()
OUT = config.path(CFG["paths"]["outputs"]) / "planner"
MANIFEST = config.path(CFG["paths"]["outputs"]) / "manifests" / "R1_20260926-113711.json"
STRIDE = 5


def _regions():
    from recovla.perception import color as C
    from recovla.planner.detect import Regions
    from recovla.sim import frames, scene
    m = scene.build_model("3cube")
    return Regions(C.overhead_calibration(m), frames.box_outer_half(m)), frames.box_pos(m)


def placed_variants(rig, color, rng) -> list:
    """今の状態のまま、目標の立方体だけを置き直して描いたこま（照合データの「直接置いた例」と、閾値を決める
    学習用の帯の例で同じ作り方）。→ [(種類, 真値の in_box, 俯瞰の画像, 手首の画像, 手首の箱の領域, 置いた位置)]。
    物理は進めない（瞬間の描画）
      箱の縁（4 枚の壁のどれかの上に、中心を壁の厚さの中ほどに合わせて乗せる）、箱の脇（外の机の上、壁から 1〜3 cm）、
      箱の中（成功の体積の中の空いた所）× 2"""
    import mujoco
    from recovla.perception import color as PC
    from recovla.planner.detect import wrist_box_mask
    from recovla.sim import frames, scene
    box = rig.box
    ti = COLORS.index(color)
    qadr = scene.cube_qpos_adr(rig.model, color)[0]
    others = [rig.data.xpos[rig.cube_ids[c]][:2].copy() for c in range(3) if c != ti]
    outer = frames.box_outer_half(rig.model)
    wall_mid = (frames.BOX_INNER_HALF + outer) / 2.0
    h = frames.CUBE_HALF

    def place(xyz, yaw):
        s = rig.forward_scratch()
        s.qpos[qadr:qadr + 3] = xyz
        s.qpos[qadr + 3:qadr + 7] = frames.yaw_quat(yaw)
        mujoco.mj_forward(rig.model, s)
        ims = rig.render(s)
        wm = wrist_box_mask(rig.model, s, rig.renderer.width, rig.renderer.height)
        return (ims[rig.cameras.index(PC.OVERHEAD)], ims[rig.cameras.index("wrist")], wm), s.xpos[rig.cube_ids[ti]].copy()
    out = []
    axis = [(0, 1), (0, -1), (1, 1), (1, -1)][int(rng.integers(4))]
    u = rng.uniform(-(frames.BOX_INNER_HALF - h), frames.BOX_INNER_HALF - h)
    rim = np.array([box[0], box[1], frames.BOX_WALL_TOP_Z + h])
    rim[axis[0]] += axis[1] * wall_mid
    rim[1 - axis[0]] += u
    img, p = place(rim, rng.uniform(-np.pi, np.pi))
    out.append(("placed_rim", frames.in_box(p, box), *img, p))
    beside = np.array([box[0], box[1], frames.CUBE_REST_Z])
    beside[axis[0]] += axis[1] * (outer + h + rng.uniform(0.01, 0.03))
    beside[1 - axis[0]] += u
    img, p = place(beside, rng.uniform(-np.pi, np.pi))
    out.append(("placed_beside", frames.in_box(p, box), *img, p))
    inside_others = [o for o in others if frames.over_box_interior(np.r_[o, 0.0], box)]
    for _ in range(2):
        lim = frames.BOX_SUCCESS_HALF - h - 0.005
        for _try in range(50):
            xy = box[:2] + rng.uniform(-lim, lim, 2)
            if all(np.hypot(*(xy - o)) > 2 * h + 0.01 for o in inside_others):
                break
        img, p = place(np.r_[xy, frames.BOX_FLOOR_Z + h + 0.001], rng.uniform(-np.pi, np.pi))
        out.append(("placed_inside", frames.in_box(p, box), *img, p))
    return out


def _best_threshold(pos, neg):
    vals = sorted(set(pos) | set(neg) | {0})
    best = None
    for t in [v + 1 for v in vals]:                  # 「t 以上ならある」
        correct = sum(p >= t for p in pos) + sum(n < t for n in neg)
        if best is None or correct > best[1]:
            best = (t, correct)
    return best


def cmd_fit(a) -> None:
    from recovla.record.recorder import read_png
    from recovla.sim import frames
    reg, box = _regions()
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    box_pos, box_neg, tab_pos, tab_neg = [], [], [], []
    for e in man["entries"]:
        ep = config.path(e["run"]) / e["key"]
        d = np.load(ep / "data.npz")
        n = len(d["sim_time"])
        idx = sorted(set(range(0, n, STRIDE)) | {0, n - 1})
        for i in idx:
            img = read_png(ep / "overhead" / f"{i:06d}.png")
            c = reg.counts(img)
            cp = d["cube_pos"][i]
            if not bool(d["gripper_closed"][i]):
                for k, col in enumerate(COLORS):
                    (box_pos if frames.in_box(cp[k], box) else box_neg).append(c["box"][col])
            if i in (0, n - 1):
                for k, col in enumerate(COLORS):
                    on_table = (not frames.over_box_interior(cp[k], box)
                                and abs(cp[k][2] - frames.CUBE_REST_Z) < 0.01)
                    (tab_pos if on_table else tab_neg).append(c["table"][col])
    # 学習用の帯で、台本を最後まで走らせてから目標を置き直した例（縁・脇・中）。R1 のこまには縁に乗った例がなく、
    # それだけでは閾値が 1 画素になって縁を見分けられなかった（試しの種 198901〜198902 で分かった、0090）
    placed = {"placed_rim": [], "placed_beside": [], "placed_inside": []}
    wrist_rows = []                                   # (真値, 内側の画素, 全体の画素)
    if a.placed_seeds:
        from recovla.eval import scene_trial as T
        from recovla.expert import generate as G
        from recovla.sim import scene
        from recovla.sim.rig import SimRig, quiet
        rig = SimRig(render=True)
        try:
            b0, nn = map(int, a.placed_seeds.split(":"))
            ss = list(range(b0, b0 + nn))
            lays = [scene.sample_layout(s, "empty", start="home") for s in ss]
            for s, col in zip(ss, T.choose_targets(ss, lays)):
                rng = np.random.default_rng(np.random.SeedSequence([s, JUDGE_KEY, 2]))
                with quiet():
                    G.run_attempt(rig, G.EpisodeSpec(s, col, "empty", "n", start="home"), 0, None, False)
                for kind, label, img, wimg, wm, _p in placed_variants(rig, col, rng):
                    px = reg.counts(img)["box"][col]
                    placed[kind].append(px)
                    (box_pos if label else box_neg).append(px)
                    from recovla.planner.detect import wrist_counts
                    wi, wa = wrist_counts(wimg, wm, col, reg.thr)
                    wrist_rows.append((bool(label), wi, wa, kind))
        finally:
            rig.close()
    bt, bc = _best_threshold(box_pos, box_neg)
    tt, tc = _best_threshold(tab_pos, tab_neg)
    # 手首: まず内側の割合の閾値（画素の閾値 1 で、正しく分けた数が最も多い値、同点なら小さい方）、次にその割合で
    # 内側の画素数の閾値（同じ決まり）。学習用の帯の置き直した例だけで決める（0090）
    wf, wfc = None, None
    for F in [x / 100 for x in range(0, 101)]:
        c = sum((wi >= 1 and wa and wi / wa >= F) == lab for lab, wi, wa, _k in wrist_rows)
        if wfc is None or c > wfc:
            wf, wfc = F, c
    passing = [(lab, wi) for lab, wi, wa, _k in wrist_rows if wa and wi / wa >= wf]
    failing_neg = sum(1 for lab, wi, wa, _k in wrist_rows if not (wa and wi / wa >= wf) and not lab)
    wt, wtc = _best_threshold([wi for lab, wi in passing if lab], [wi for lab, wi in passing if not lab])
    wtc += failing_neg
    wrist_res = {"fraction_threshold": wf, "pixel_threshold": wt, "correct": wtc, "n": len(wrist_rows),
                 "by_kind": {k: [(wi, wa) for lab, wi, wa, kk in wrist_rows if kk == k][:60] for k in placed}}
    res = {"source": str(MANIFEST.relative_to(config.ROOT)), "stride": STRIDE, "placed_seeds": a.placed_seeds,
           "placed_box_pixels": {k: {"n": len(v), "min": min(v) if v else None, "max": max(v) if v else None,
                                     "median": float(np.median(v)) if v else None} for k, v in placed.items()},
           "box": {"threshold": bt, "correct": bc, "n": len(box_pos) + len(box_neg), "n_pos": len(box_pos),
                   "pos_min": min(box_pos), "pos_p1": float(np.percentile(box_pos, 1)), "neg_max": max(box_neg),
                   "neg_p99": float(np.percentile(box_neg, 99))},
           "table": {"threshold": tt, "correct": tc, "n": len(tab_pos) + len(tab_neg), "n_pos": len(tab_pos),
                     "pos_min": min(tab_pos), "neg_max": max(tab_neg) if tab_neg else None},
           "wrist": wrist_res,
           "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "presence_fit.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


R2_10K = "outputs/train/train_R2_20260927-145256_20260927-145256/checkpoints/010000/pretrained_model"
JUDGE_EXTRA_S = 6.0          # 真値の成功の後も走らせ続ける時間
# 走行の正例を取る時刻（真値の成功からの秒、一様）。手が待機位置へ戻るのに成功から 1.5〜2.5 s かかり（試しの種
# 198901〜198902 で確認）、完了は戻って 1 s 静止した後なので、それより後の待っている間から取る
JUDGE_POS_WINDOW = (3.5, 6.0)
JUDGE_KEY = 5100             # 照合データの乱数: SeedSequence([種, 5100, …])


def _runner(checkpoint):
    import os
    os.environ.setdefault("HF_HOME", str(config.path(CFG["paths"]["models_home"])))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from recovla.policy.runner import RuntimeConfig, SceneRunner
    from recovla.policy.scene_policy import ScenePolicy
    rt = CFG["runtime"]
    pol = ScenePolicy(config.path(checkpoint))
    return SceneRunner(pol, RuntimeConfig(rt["mode"], int(rt["exec_interval"]), int(rt["delay_steps"]),
                                          execution_horizon=int(rt["rtc_guidance_horizon"])),
                       {"schedule": rt["rtc_schedule"], "max_guidance_weight": rt["rtc_max_guidance_weight"]})


def cmd_judge_data(a) -> None:
    """完了判定の照合データ（手順書 Step I 完了条件 1・E7。作り方は 0088 の 3、ここに回す前に固めた）。

    選択用の種（予備は 196000〜196039 の 40 配置）ごとに、R2 を本線の設定で 1 手順（choose_targets の色）走らせ、
    真値の成功（評価器と同じ）の後も JUDGE_EXTRA_S まで走らせる。そこから次を作る:
      走行の正例   成功から U(3.5, 6.0) s の時刻（判定と独立に乱数で選ぶ。手が戻って待っている間）。真値: 目標が成功の体積の中
      走行の負例   持って箱の上にいる瞬間（目標が箱の内寸の上・グリッパ閉・目標が 2 cm 以上持ち上がっている）の 1 つ
      直接置いた例 走行の最後の状態で手が待機位置にいるとき（判定の (3) の定義）だけ、目標の立方体を置き直して描く:
                   箱の縁（壁の上に乗せる）・箱の脇（外の机の上、壁から 1〜3 cm）＝負例、箱の中の別の場所＝正例 2 つ
    各例は、判定に 1 s 分のこま（20 Hz）を順に与え、最後に完了と出たかを見る（直接置いた例は同じこまを 1 s 繰り返す）。
    正例 100・負例 100 を、正解の側と種類だけで決まった順（種の乱数）で選ぶ（判定の出力は見ない）。
    結果: outputs/planner/judge_<tag>.json と、各例の最後の俯瞰のこま（outputs/planner/judge_<tag>/）。"""
    import cv2
    import mujoco
    from recovla.common.seeds import COLORS
    from recovla.eval import scene_trial as T
    from recovla.expert.script import PhaseParams
    from recovla.perception import color as PC
    from recovla.planner.detect import Regions, wrist_box_mask
    from recovla.planner.judge import CompletionJudge
    from recovla.record import episode as E
    from recovla.sim import frames, scene
    from recovla.sim.rig import SimRig, quiet
    runner = _runner(a.checkpoint)
    rig = SimRig(render=True)
    reg = Regions(PC.overhead_calibration(rig.model), frames.box_outer_half(rig.model))
    pp = PhaseParams.from_config()
    rest_speed, rest_hold = float(CFG["eval"]["success"]["rest_speed"]), float(CFG["eval"]["success"]["rest_hold_s"])
    retreat = np.asarray(CFG["expert"]["retreat_pose"], float)
    tol = float(CFG["planner"]["judge"]["retreat_tol_m"])
    outdir = OUT / f"judge_{a.tag}"
    outdir.mkdir(parents=True, exist_ok=False)
    base, n = map(int, a.seeds.split(":"))
    seeds_ = list(range(base, base + n))
    lays = [scene.sample_layout(s) for s in seeds_]
    targets = T.choose_targets(seeds_, lays)
    cases = []

    def judge_window(window, color):
        j = CompletionJudge(reg)
        j.reset(color)
        done = False
        for t, img, closed, xd, wimg, wm in window:
            done = j.update(t, img, closed, xd, wimg, wm)
        return done, j.last

    def add(kind, label, window, color, seed, extra):
        done, last = judge_window(window, color)
        cid = len(cases)
        cv2.imwrite(str(outdir / f"case_{cid:04d}.png"), cv2.cvtColor(window[-1][1], cv2.COLOR_RGB2BGR))
        cases.append({"id": cid, "seed": seed, "color": color, "kind": kind, "truth": bool(label), "judge": bool(done),
                      "conditions": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in last.items()},
                      **extra})

    try:
        for seed, lay, color in zip(seeds_, lays, targets):
            rng = np.random.default_rng(np.random.SeedSequence([seed, JUDGE_KEY]))
            ti = COLORS.index(color)
            rig.reset(lay)
            runner.start_trial(seed)
            rig.safety.start_trial(color)
            log = []                                    # (t, 俯瞰, グリッパ閉, x_des, 目標の位置)
            st = {"hold": 0.0, "succ": None}

            def on_step(r):
                d = r.data
                pos = d.xpos[r.cube_ids[ti]]
                v = r.cube_vadr[ti]
                if st["succ"] is None:
                    if frames.in_box(pos, r.box) and float(np.linalg.norm(d.qvel[v:v + 3])) < rest_speed:
                        st["hold"] += r.timestep
                        if st["hold"] >= rest_hold - 1e-9:
                            st["succ"] = float(r.step * r.timestep)
                    else:
                        st["hold"] = 0.0
                if r.step % r.record_every == 0:
                    f, imgs = E.capture_frame(r, color, 0.0, pp, True)
                    raw = dict(zip(r.cameras, imgs))
                    st["raw"] = raw
                    wm = wrist_box_mask(r.model, r.scratch, r.renderer.width, r.renderer.height)
                    log.append((float(f["sim_time"]), raw[PC.OVERHEAD], bool(f["gripper_closed"]), f["x_des"].copy(),
                                f["cube_pos"][ti].copy(), raw["wrist"], wm))
            f, imgs = E.capture_frame(rig, color, 0.0, pp, True)
            st["raw"] = dict(zip(rig.cameras, imgs))
            k = 0
            limit = float(CFG["eval"]["time_limit_s"])
            with quiet():
                while rig.data.time < limit - 1e-9 and (st["succ"] is None or rig.data.time < st["succ"] + JUDGE_EXTRA_S):
                    a_ = np.asarray(runner(k, f, st["raw"], T.instruction(color)), dtype=np.float64)
                    rig.safety.gate = True
                    T.execute_action(rig, a_, on_step)
                    f = E.capture_frame(rig, color, 0.0, pp, False)[0]
                    k += 1
            rig.safety.gate = False
            times = np.array([x[0] for x in log])

            def window_at(t_end):
                i = int(np.searchsorted(times, t_end - 1e-9))
                i = min(i, len(log) - 1)
                j0 = max(0, i - 20)
                return [(x[0], x[1], x[2], x[3], x[5], x[6]) for x in log[j0:i + 1]], i
            if st["succ"] is not None:                  # 走行の正例
                t_pos = st["succ"] + rng.uniform(*JUDGE_POS_WINDOW)
                if t_pos <= times[-1]:
                    w, i = window_at(t_pos)
                    add("run_positive", frames.in_box(log[i][4], rig.box), w, color, seed, {"t": float(log[i][0])})
            held = [i for i, x in enumerate(log) if x[2] and frames.over_box_interior(x[4], rig.box)
                    and x[4][2] - frames.CUBE_REST_Z >= 0.02]
            if held:                                    # 走行の負例（持って箱の上）
                i = held[int(rng.integers(len(held)))]
                w, _ = window_at(log[i][0])
                add("run_held_over_box", frames.in_box(log[i][4], rig.box), w, color, seed, {"t": float(log[i][0])})
            # 直接置いた例（最後に手が待機位置にいるときだけ）
            if np.linalg.norm(log[-1][3] - retreat) <= tol and not log[-1][2]:
                for kind, label, img, wimg, wm, p in placed_variants(rig, color, rng):
                    w = [(log[-1][0] + 0.05 * q, img, False, log[-1][3], wimg, wm) for q in range(21)]
                    add(kind, label, w, color, seed, {"pos": p.tolist()})
            print(seed, color, "succ", st["succ"], "cases", len(cases), flush=True)
    finally:
        rig.close()
    rng = np.random.default_rng(np.random.SeedSequence([base, JUDGE_KEY, 1]))
    pos = [c for c in cases if c["truth"]]
    neg = [c for c in cases if not c["truth"]]
    pick = lambda xs, m: [xs[i] for i in sorted(rng.permutation(len(xs))[:m])] if len(xs) >= m else xs   # noqa: E731
    sel = pick(pos, a.n_each) + pick(neg, a.n_each)
    agree = sum(c["truth"] == c["judge"] for c in sel)
    import collections
    by_kind = collections.defaultdict(lambda: [0, 0])
    for c in sel:
        by_kind[c["kind"]][0] += c["truth"] == c["judge"]
        by_kind[c["kind"]][1] += 1
    res = {"seeds": a.seeds, "checkpoint": a.checkpoint, "all_cases": len(cases), "pos_all": len(pos), "neg_all": len(neg),
           "selected": len(sel), "selected_pos": sum(c["truth"] for c in sel), "agree": agree,
           "agreement": agree / len(sel) if sel else None, "by_kind": {k: v for k, v in by_kind.items()},
           "thresholds": {"presence": CFG["planner"]["presence"], "judge": CFG["planner"]["judge"],
                          "hold_s": CFG["planner"]["completion_hold_s"]},
           "selected_ids": [c["id"] for c in sel], "cases": cases, "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    (OUT / f"judge_{a.tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("cases", "selected_ids")}, ensure_ascii=False, indent=1))


def cmd_llm_eval(a) -> None:
    """日本語の指示の試験文（fixtures）を LLM で分け、期待する出力と比べる（手順書 Step I 完了条件 2・E7）。"""
    from recovla.planner import decompose as D
    fx = json.loads((config.ROOT / "tests" / "planner" / "fixtures" / f"{a.fixture}.json").read_text(encoding="utf-8"))
    rows = []
    for it in fx["items"]:
        out = D.decompose(it["text"], it["table"], it["box"])
        ok = out["steps"] == it["expected"]
        rows.append({"id": it["id"], "text": it["text"], "expected": it["expected"], "steps": out["steps"],
                     "llm_steps": out["llm_steps"], "valid": out["valid"], "reply": out["reply"], "correct": ok,
                     "from_cache": out["from_cache"], "usage": out["usage"]})
        print(it["id"], "OK " if ok else "NG ", it["text"], it["expected"], "->", out["steps"], out["reply"], flush=True)
    tok_in = sum((r["usage"] or {}).get("input_tokens", 0) for r in rows if not r["from_cache"])
    tok_out = sum((r["usage"] or {}).get("output_tokens", 0) for r in rows if not r["from_cache"])
    res = {"fixture": a.fixture, "prompt_version": D.PROMPT_VERSION, "model": CFG["planner"]["model"],
           "correct": sum(r["correct"] for r in rows), "n": len(rows), "rows": rows,
           "new_calls_tokens": {"input": tok_in, "output": tok_out, "usd_estimate": tok_in / 1e6 * 1.0 + tok_out / 1e6 * 5.0},
           "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"llm_{a.fixture}_v{D.PROMPT_VERSION}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                                                     encoding="utf-8")
    print("correct", res["correct"], "/", res["n"], res["new_calls_tokens"])


def cmd_run(a) -> None:
    """複数手順の通し（手順書 Step I 完了条件 3・E7）。空の箱・既定の開始姿勢に 3 色、指示は --text。
    実行は本線の設定（configs の runtime と安全フィルタ）、方策は最終モデル R2 の 1 万手。"""
    import os
    os.environ.setdefault("HF_HOME", str(config.path(CFG["paths"]["models_home"])))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from recovla.eval.closed_loop import write_mp4
    from recovla.planner.executor import TaskExecutor
    from recovla.policy.runner import RuntimeConfig, SceneRunner
    from recovla.policy.scene_policy import ScenePolicy
    from recovla.sim import scene
    from recovla.sim.rig import SimRig
    rt = CFG["runtime"]
    pol = ScenePolicy(config.path(a.checkpoint))
    runner = SceneRunner(pol, RuntimeConfig(rt["mode"], int(rt["exec_interval"]), int(rt["delay_steps"]),
                                            execution_horizon=int(rt["rtc_guidance_horizon"])),
                         {"schedule": rt["rtc_schedule"], "max_guidance_weight": rt["rtc_max_guidance_weight"]})
    out = OUT / "runs" / a.tag
    out.mkdir(parents=True, exist_ok=False)
    rig = SimRig(render=True)
    import copy
    cfg = copy.deepcopy(CFG)
    if a.return_to_retreat is not None:          # 待機位置へ戻す動き（0094 の 2）の入切を configs から上書き
        cfg["planner"]["return_to_retreat"]["enabled"] = a.return_to_retreat == "on"
    ex = TaskExecutor(rig, runner, cfg)
    base, n = map(int, a.seeds.split(":"))
    rows = []
    try:
        for i, seed in enumerate(range(base, base + n)):
            lay = scene.sample_layout(seed, "empty", start="home")
            meta, arr, video = ex.run(lay, a.text, seed, render_video=i < a.videos)
            meta["checkpoint"] = a.checkpoint
            (out / f"task_{i:04d}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=float),
                                                    encoding="utf-8")
            np.savez(out / f"task_{i:04d}.npz", **arr)
            if video:
                write_mp4(out / f"task_{i:04d}.mp4", video, 20)
            rows.append({"seed": seed, "plan": meta["plan"]["steps"], "all_three": meta["all_three_in_box"],
                         "stopped": meta["stopped"], "steps": [(s["color"], s["judged_complete"], len(s["attempts"]),
                                                                s["t_truth_success"] is not None) for s in meta["steps"]],
                         "returns": [(r["kind"], r["step"], r["arrived"], r["judged"]) for r in meta["returns"]]})
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    finally:
        rig.close()
    (out / "run.json").write_text(json.dumps({"text": a.text, "seeds": a.seeds, "checkpoint": a.checkpoint,
                                              "return_to_retreat": cfg["planner"]["return_to_retreat"],
                                              "all_three": sum(r["all_three"] for r in rows), "n": len(rows),
                                              "written": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False,
                                             indent=2), encoding="utf-8")


def _task_code(m) -> str:
    """0093 の表の符号: ok＝3 個とも（真値）、G2＝2 番目の緑で止まった（* は真値では置けていた＝判定の見逃し）、
    x:b＝判定はすべて完了だが青が真値で箱に入っていない。やり直しで完了した手順があれば末尾に +r<番号>。"""
    retry_done = [str(s["step"] + 1) for s in m["steps"] if s["judged_complete"] and len(s["attempts"]) > 1]
    tail = ("+r" + ",".join(retry_done)) if retry_done else ""
    if m["all_three_in_box"]:
        return "ok" + tail
    st = m["stopped"]
    if st is not None:
        s = m["steps"][st["step"]]
        return f"{st['color'][0].upper()}{st['step'] + 1}{'*' if s['t_truth_success'] is not None else ''}" + tail
    miss = [c for c in m["plan"]["steps"] if not m["final_in_box"][c]] or [c for c, v in m["final_in_box"].items() if not v]
    return "x:" + (miss[0][0] if miss else "?") + tail


def cmd_compare_return(a) -> None:
    """待機位置へ戻す動き（0094 の 2）の前後を同じ種の対で並べる（検定は付けない）。主な見方と採る基準は 0094 で
    回す前に固めた: 3 通りの合計の「3 個とも」（変更前 31/60）と、やり直しで完了した手順の数（変更前 0）。
    採る = 変更後の「3 個とも」の合計が変更前を下回らず、かつやり直しで完了した手順が 1 つ以上。"""
    def load(tag):
        d = OUT / "runs" / tag
        run = json.loads((d / "run.json").read_text(encoding="utf-8"))
        tasks = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("task_*.json"))]
        return run, tasks

    def summary(tasks):
        steps = [s for m in tasks for s in m["steps"]]
        rets = [r for m in tasks for r in m.get("returns", [])]
        pos = {}
        for m in tasks:
            for s in m["steps"]:
                key = (s["step"] + 1, s["color"])
                p = pos.setdefault(f"{key[0]}:{key[1]}", [0, 0, 0])
                p[1] += 1
                if not s["judged_complete"]:
                    p[0] += 1
                    if s["t_truth_success"] is None:
                        p[2] += 1
        return {"all_three": sum(m["all_three_in_box"] for m in tasks), "n": len(tasks),
                "retry_completed_steps": sum(s["judged_complete"] and len(s["attempts"]) > 1 for s in steps),
                "stopped_by_position": {k: {"stopped": v[0], "started": v[1], "stopped_truth_not_placed": v[2]}
                                        for k, v in sorted(pos.items())},
                "returns": {"n": len(rets), "retry": sum(r["kind"] == "retry" for r in rets),
                            "placed": sum(r["kind"] == "placed" for r in rets),
                            "not_arrived": sum(not r["arrived"] for r in rets),
                            "touched_cube_or_box": sum(r["contact_cube"] or r["contact_box"] for r in rets),
                            "touched_cube": sum(r["contact_cube"] for r in rets),
                            "touched_box": sum(r["contact_box"] for r in rets),
                            "placed_then_judged": sum(r["kind"] == "placed" and r["judged"] for r in rets),
                            "retry_then_step_completed": sum(
                                r["kind"] == "retry" and m["steps"][r["step"]]["judged_complete"]
                                for m in tasks for r in m.get("returns", []))},
                "judge_false_complete": sum(s["judged_complete"] and s["t_truth_success"] is None
                                            and not m["final_in_box"][s["color"]] for m in tasks for s in m["steps"])}
    if len(a.before) != len(a.after):
        raise SystemExit("--before と --after の数を揃える")
    res = {"pairs": [], "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    tot_b = tot_a = retry_a = 0
    for tb, ta in zip(a.before, a.after):
        rb, kb = load(tb)
        ra, ka = load(ta)
        if rb["text"] != ra["text"] or rb["seeds"] != ra["seeds"]:
            raise SystemExit(f"指示か種が違う: {tb} {ta}")
        sb, sa = summary(kb), summary(ka)
        tot_b += sb["all_three"]
        tot_a += sa["all_three"]
        retry_a += sa["retry_completed_steps"]
        res["pairs"].append({"text": ra["text"], "seeds": ra["seeds"], "before": tb, "after": ta,
                             "before_summary": sb, "after_summary": sa,
                             "by_seed": [{"seed": x["seed"], "before": _task_code(x), "after": _task_code(y)}
                                         for x, y in zip(kb, ka)]})
    res["primary"] = {"all_three_before": tot_b, "all_three_after": tot_a,
                      "n": sum(p["after_summary"]["n"] for p in res["pairs"]),
                      "retry_completed_after": retry_a,
                      "adopt": bool(tot_a >= tot_b and retry_a >= 1)}
    (OUT / f"{a.out}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res["primary"], ensure_ascii=False))
    for p in res["pairs"]:
        print(p["text"], p["before_summary"]["all_three"], "->", p["after_summary"]["all_three"],
              json.dumps(p["after_summary"]["returns"], ensure_ascii=False))
        for r in p["by_seed"]:
            print(" ", r["seed"], r["before"], "->", r["after"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("fit")
    s.add_argument("--placed-seeds", default="57000:40", help="学習用の帯で置き直した例を作る種（<先頭>:<数>）。空なら作らない")
    s = sub.add_parser("run")
    s.add_argument("--seeds", required=True, help="<先頭>:<数>")
    s.add_argument("--text", default="全部片付けて")
    s.add_argument("--tag", required=True)
    s.add_argument("--checkpoint", default=R2_10K)
    s.add_argument("--videos", type=int, default=2)
    s.add_argument("--return-to-retreat", choices=["on", "off"], default=None,
                   help="待機位置へ戻す動き（0094 の 2）。省くと configs の planner.return_to_retreat.enabled")
    s = sub.add_parser("compare-return")
    s.add_argument("--before", nargs="+", default=["E7_pre", "E7_order_GRB", "E7_order_RBG"])
    s.add_argument("--after", nargs="+", required=True)
    s.add_argument("--out", default="return_compare")
    s = sub.add_parser("judge-data")
    s.add_argument("--seeds", required=True, help="<先頭>:<数>（予備は 196000:40）")
    s.add_argument("--tag", required=True)
    s.add_argument("--n-each", type=int, default=100)
    s.add_argument("--checkpoint", default=R2_10K)
    s = sub.add_parser("llm-eval")
    s.add_argument("--fixture", default="dev_sentences")
    a = ap.parse_args(argv)
    {"fit": cmd_fit, "run": cmd_run, "judge-data": cmd_judge_data, "llm-eval": cmd_llm_eval,
     "compare-return": cmd_compare_return}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
