"""段階 4 束 1 の D-単発の開始と「移植」の腕（担当 B。docs/目標書_段階4.md 8-2 の S・8-3・8-4）。

評価の側（世界・真値。G1 の境界の外側）:
    DiagWorldRig    harness.world.WorldRig の子。reset(layout) の後に、決めた開始状態（腕の関節角・立方体の位置と向き）へ
                    置き直し、腕を位置のサーボで保ったまま settle_prefilled_s（1 s）落ち着かせてから時刻 0 にする。
                    実行系には何も渡さない（実行系は測った関節角とカメラだけを見る。PolicyRuntime.start が測った関節角から始める）
    start_trial_spec(...)   D-単発の開始の 1 試行の配置と置き直しの中身（開始 3 x 先客 2）
    xpl_trial_spec(...)     移植の腕の 1 試行の配置と置き直しの中身（XPL_as・XPL_home・XPL_grid）
    extract_xpl_states(...) 段階 3 の E7_R1v3 の記録から 2 番目の手順の始めの状態を取り出す（読むだけ）
    wall_pool(...)          「壁際」の先客の位置（E7 の 2 番目の始めの赤の位置のうち壁の近く）
集計（記録だけを読む。判定はしない）:
    single_first_close(meta, z)     単発の最初の閉じ（e7.first_close と同じ物差し）
    summarize_single(dir)           条件 1 つ（trial_NNNN.json・npz）の集計
    gate_S_inputs(summ)・xpl_inputs(summ)   関門 S・移植の腕の材料（判定はしない）

結果を見る前に決めてある項目（outputs/s4/d_start/definitions.json に書いて回す前に掲示する。掲示の後は変えない）:
  D-単発の開始（帯 190400〜190410、natural の並び＝種ごとに 3 色、33 試行/条件、6 条件で同じ種）:
  - 配置: 82 の natural と同じ scene.sample_layout(種, "empty", start="home") の机上の位置を使い、目標 c の前の色
    PRIOR_OF[c]（赤→青、緑→赤、青→緑。E7 の 2 番目＝緑の目標・赤の先客と同じ向き）を箱の中の先客にする。残りの色は机上のまま。
  - 先客 on_grid（格子の上）: 学習データの置き場所（scene.box_slots）のうち、sample_layout(種, "prefilled_1") が引くのと同じ
    1 番目の置き場所（配置の乱数列から同じ順に引く。grid_slot）。向きは scene.prefilled_yaw_deg。
  - 先客 wall_side（方策が置いたような壁際）: WALL_RULE。E7_R1v3 の 20 本の 2 番目の手順の始め（XPL_INDEX_RULE と同じ時刻）の
    赤の位置のうち、箱の中心からのずれの大きいほうの成分が 0.05 m を超えるもの（W\\task3\\s15_placement.py の「壁際」と同じ閾値）を
    run の番号順に並べ、帯の先頭からの種の順 k に pool[k mod 個数] を当てる（xy と向きは記録のまま、高さは箱の底の上）。
  - 開始の 3 通りは、どれも同じ置き直しの道を通す（10/08 の検査役の指摘で直した。前の版は standby_end だけが置き直しを通り、
    home・standby_start は標準の reset のままで、比べる効き目に準備のしかたの差が混ざっていた）: 待機位置の始めの標準の開始
    （BASE_START = starts["retreat"] からの reset。1 s の落ち着かせを含む）から、腕だけを START_Q[開始] へ瞬間移動して置き直す
    （下の「共通」の落ち着かせ）。START_Q: home = e7.HOME_Q（目標書 8-3・8-5 の値。標準の home の reset の後の関節角と 4.5e-4 rad 以内）、
    standby_start = e7.STANDBY_START_Q（学習データの 0 こま目の平均、手先 z 0.3144。標準の retreat の reset の後の関節角と 4e-5 rad
    以内）、standby_end = e7.STANDBY_END_Q（学習データの最後のこまの平均、z 0.3036）。3 つとも学習データ（目標書）由来の同じ出どころ。
  - 先客も 2 通りとも同じ置き直しの道を通す: on_grid は格子の置き場所（slot_pose）、wall_side は壁際の位置へ、置き直しで置く
    （on_grid は配置の先客としても同じ所に置いてあるので、位置は前の版と同じ）。
  移植の腕（帯 190420〜190439、種 190420+i が E7_R1v3 の run_{i:04d}、目標は 2 番目の手順の色＝緑、1 状態 x 3 通り x 2 回）:
  - XPL_INDEX_RULE: 2 番目の手順の t_start（run_NNNN.json の steps[1]）以降で最初の推論（run_NNNN_runtime.json の
    runtime.inference の t_obs >= t_start、state あり）。関節はその state[8:15]（推論時の測った関節角）。立方体は run_NNNN.npz の
    sim_time <= その t_obs の最後のこまの cube_pos・cube_quat（3 個）。
  - XPL_grid: 赤を、学習データの置き場所（box_slots）のうち赤の記録の位置に最も近い空き（ほかの立方体の中心が xy で
    GRID_OCCUPIED_M 以内に無い）へ移す。向きは prefilled_yaw_deg、高さは箱の底の上。緑・青は記録のまま。
  - XPL_home: 腕だけ e7.HOME_Q へ。立方体は記録のまま。
  共通: 置き直しの後、腕を置き直した関節角のまま位置のサーボで保ち（0.33 s・0.67 s に指令を測った関節角の差だけ直す）、ハンドは
  開いたまま、1.0 s（configs の scene.settle_prefilled_s）落ち着かせてから時刻 0。初期状態に記録した真値を使うのは場面のリセットに
  当たり（目標書 第 10 節 7）、実行時の判断には使わない。G1 は変えない。
"""
import json
import pathlib

import mujoco
import numpy as np

from recovla.common import seeds
from recovla.common.seeds import COLORS
from recovla.diag import e7 as E7
from recovla.harness.world import WorldRig
from recovla.sim import frames, scene

# 条件名 -> Layout.start（記録の名札。3 つとも DiagWorldRig で BASE_START の開始から置き直す）
START_KEYS = {"home": "placed_home", "standby_start": "placed_standby_start", "standby_end": "standby_end"}
START_Q = {"home": E7.HOME_Q, "standby_start": E7.STANDBY_START_Q, "standby_end": E7.STANDBY_END_Q}   # 条件名 -> 置き直す関節角
BASE_START = "retreat"                                # 置き直しの前の標準の開始（WorldRig.starts の鍵）
PLACED_STARTS = tuple(START_KEYS.values()) + ("xpl",)
START_RULE = ("開始の 3 通り（home・standby_start・standby_end）と移植は、どれも starts['retreat'] からの標準の reset（1 s の落ち着かせ）の後、"
              "腕を START_Q[開始]（移植は記録の関節角か HOME_Q）へ瞬間移動し、位置のサーボで保ったまま 1 s 落ち着かせて時刻 0 にする。"
              "先客は on_grid・wall_side とも置き直しで置く")
PRIORS = ("on_grid", "wall_side")
PRIOR_OF = {"red": "blue", "green": "red", "blue": "green"}
WALL_THRESH_M = 0.05
GRID_OCCUPIED_M = 0.045
SETTLE_FIXES = (1 / 3, 2 / 3)
XPL_ARMS = ("as", "home", "grid")
XPL_INDEX_RULE = ("steps[1].t_start 以降の最初の推論（runtime.inference の t_obs >= t_start - 1e-9、state あり）の state[8:15] を関節、"
                  "npz の sim_time <= t_obs + 1e-9 の最後のこまの cube_pos・cube_quat を立方体とする")
WALL_RULE = ("E7_R1v3 の run_0000〜0019 の XPL_INDEX_RULE の時刻の赤の位置のうち、max(|x - 箱x|, |y - 箱y|) > 0.05 m のものを run の"
             "番号順に並べ、種の順 k（帯の先頭から）に pool[k mod 個数] を当てる。xy と四元数は記録のまま、高さは BOX_FLOOR_Z + CUBE_HALF + 1e-4")
GRID_RULE = ("on_grid: 配置の乱数列 streams(種)['layout'] から sample_layout と同じ順に引いた置き場所の並びの 1 番目（= sample_layout(種, "
             "'prefilled_1').prefilled の置き場所）。XPL_grid: 赤の記録の xy に最も近い空きの置き場所（ほかの立方体の中心が xy で 0.045 m 以内に"
             "無い）。どちらも向き prefilled_yaw_deg、高さは箱の底の上")
LIFT = 1e-4


# ---------------------------------------------------------------- 置き場所・壁際・移植の状態
def grid_slot(seed: int, cfg: dict = None) -> int:
    """sample_layout(seed, "prefilled_1") が先客に当てるのと同じ置き場所の番号（配置の乱数列を同じ順に引く）。"""
    sc = (cfg or scene._CFG)["scene"]
    rng = seeds.stream(seed, "layout")
    rng.random()                                      # 配置の種類
    rng.random()                                      # 開始姿勢
    rng.permutation(len(COLORS))                      # 色の並び
    return int(rng.permutation(len(sc["box_slots"]))[0])


def slot_pose(slot: int, box=None) -> tuple:
    box = np.asarray(box if box is not None else scene._SCENE["box"]["pos"], float)
    sx, sy = frames.slot_xy(box, slot)
    yaw = np.radians(float(scene._SCENE["prefilled_yaw_deg"]))
    return [float(sx), float(sy), frames.BOX_FLOOR_Z + frames.CUBE_HALF + LIFT], [float(v) for v in frames.yaw_quat(yaw)]


def _box_xy() -> np.ndarray:
    return np.asarray(scene._SCENE["box"]["pos"], float)[:2]


def _e7_paths(e7_dir, i: int) -> dict:
    d = pathlib.Path(e7_dir)
    return {"json": d / f"run_{i:04d}.json", "npz": d / f"run_{i:04d}.npz", "runtime": d / f"run_{i:04d}_runtime.json"}


def extract_xpl_states(e7_dir, n: int = 20) -> list:
    """E7_R1v3 の run_0000〜 から 2 番目の手順の始めの状態（XPL_INDEX_RULE）。読むだけ。"""
    out = []
    for i in range(n):
        ps = _e7_paths(e7_dir, i)
        m = json.loads(ps["json"].read_text(encoding="utf-8"))
        rt = json.loads(ps["runtime"].read_text(encoding="utf-8"))
        z = np.load(ps["npz"])
        steps = m.get("steps") or []
        if len(steps) < 2:
            raise ValueError(f"{ps['json']}: 2 番目の手順が始まっていない")
        t0 = float(steps[1]["t_start"])
        inf = next(x for x in rt["runtime"]["inference"] if x["t_obs"] >= t0 - 1e-9 and x.get("state") is not None)
        t = np.asarray(z["sim_time"])
        f = int(np.searchsorted(t, float(inf["t_obs"]) + 1e-9) - 1)
        s = np.asarray(inf["state"], float)
        out.append({"index": i, "e7_seed": int(m["seed"]), "plan": (m.get("plan") or {}).get("steps"),
                    "target": steps[1]["color"], "prior_colors": [c for c, b in zip(COLORS, z["cube_in_box"][f]) if b],
                    "t_step_start": t0, "t_obs": float(inf["t_obs"]), "inference_i": int(inf["i"]), "frame": f,
                    "frame_t": float(t[f]), "joints": s[8:15].tolist(), "fingers_state": s[6:8].tolist(), "ee_state": s[:3].tolist(),
                    "cube_pos": np.asarray(z["cube_pos"][f]).tolist(), "cube_quat": np.asarray(z["cube_quat"][f]).tolist(),
                    "cube_in_box": [bool(b) for b in z["cube_in_box"][f]], "gripper_closed": bool(z["gripper_closed"][f]),
                    "fingertip": np.asarray(z["fingertip"][f]).tolist()})
    return out


def wall_pool(states: list, prior: str = "red") -> list:
    """WALL_RULE の壁際の位置の並び（states は extract_xpl_states の出力）。"""
    ci = COLORS.index(prior)
    pool = []
    for s in states:
        rel = np.asarray(s["cube_pos"][ci][:2]) - _box_xy()
        if s["cube_in_box"][ci] and float(np.max(np.abs(rel))) > WALL_THRESH_M:
            pool.append({"from_index": s["index"], "rel_xy": rel.tolist(), "quat": list(s["cube_quat"][ci])})
    return pool


def nearest_empty_slot(xy, others_xy) -> tuple:
    """(置き場所の番号, 距離 [m])。others_xy はほかの立方体の xy の並び。"""
    box = np.asarray(scene._SCENE["box"]["pos"], float)
    best = None
    for k in range(len(scene._SCENE["box_slots"])):
        sxy = np.asarray(frames.slot_xy(box, k), float)
        if any(np.linalg.norm(sxy - np.asarray(o, float)[:2]) < GRID_OCCUPIED_M for o in others_xy):
            continue
        dist = float(np.linalg.norm(sxy - np.asarray(xy, float)[:2]))
        if best is None or dist < best[1]:
            best = (k, dist)
    if best is None:
        raise ValueError("空いた置き場所がない")
    return best


# ---------------------------------------------------------------- 1 試行の配置と置き直し
def start_trial_spec(seed: int, lay, target: str, start: str, prior: str, pool: list, band_base: int) -> tuple:
    """D-単発の開始。返り値 (Layout, 置き直し, 記録)。lay は 82 の natural の配置（empty・home）。"""
    if start not in START_KEYS or prior not in PRIORS:
        raise ValueError(f"start {start} / prior {prior}")
    pc = PRIOR_OF[target]
    slot = grid_slot(seed)
    cubes = {c: tuple(lay.cubes[c]) for c in lay.table_colors if c != pc}
    lay2 = scene.Layout(int(seed), "prefilled_1", START_KEYS[start], cubes, {pc: slot}, lay.tries)
    # 6 条件とも同じ置き直しの道: 腕は START_Q[開始] へ、先客は格子か壁際へ（START_RULE）
    ov = {"arm_q": list(START_Q[start])}
    rec = {"diag": "D-single-start", "start": start, "prior": prior, "prior_color": pc, "target": target, "grid_slot": slot,
           "base_layout": lay.to_json(), "base_start": BASE_START, "start_rule": START_RULE}
    if prior == "wall_side":
        if not pool:
            raise ValueError("壁際の並びが空")
        k = int(seed) - int(band_base)
        w = pool[k % len(pool)]
        xy = _box_xy() + np.asarray(w["rel_xy"], float)
        ov["cubes"] = {pc: ([float(xy[0]), float(xy[1]), frames.BOX_FLOOR_Z + frames.CUBE_HALF + LIFT], list(w["quat"]))}
        rec["wall"] = dict(w, k=k)
    else:
        p, q = slot_pose(slot)
        ov["cubes"] = {pc: (p, q)}
    rec["override"] = ov
    return lay2, ov, rec


def xpl_trial_spec(seed: int, state: dict, arm: str, rep: int) -> tuple:
    """移植の腕。返り値 (Layout, 置き直し, 記録)。Layout は記録用（机上の色・先客の色）。立方体と腕はすべて置き直しで決める。"""
    if arm not in XPL_ARMS:
        raise ValueError(arm)
    pos = [list(p) for p in state["cube_pos"]]
    quat = [list(q) for q in state["cube_quat"]]
    rec = {"diag": "XPL", "arm": f"XPL_{arm}", "rep": int(rep), "state_index": state["index"], "e7_seed": state["e7_seed"],
           "target": state["target"], "index_rule": XPL_INDEX_RULE}
    if arm == "grid":
        ri = COLORS.index("red")
        others = [pos[k] for k in range(3) if k != ri]
        slot, dist = nearest_empty_slot(pos[ri], others)
        p, q = slot_pose(slot)
        rec.update(grid_slot=slot, grid_move_m=round(dist, 4), red_from=pos[ri])
        pos[ri], quat[ri] = p, q
    q_arm = list(E7.HOME_Q) if arm == "home" else list(state["joints"])
    ov = {"arm_q": q_arm, "cubes": {c: (pos[k], quat[k]) for k, c in enumerate(COLORS)}}
    table = {c: (float(pos[k][0]), float(pos[k][1]), 0.0) for k, c in enumerate(COLORS) if not state["cube_in_box"][k]}
    inbox = {c: 0 for k, c in enumerate(COLORS) if state["cube_in_box"][k]}
    kind = {0: "empty", 1: "prefilled_1", 2: "prefilled_2"}[len(inbox)]
    lay2 = scene.Layout(int(seed), kind, "xpl", table, inbox, 1)
    rec["override"] = ov
    return lay2, ov, rec


# ---------------------------------------------------------------- 世界の置き直し（評価の側）
class DiagWorldRig(WorldRig):
    """reset の後に diag_override（{"arm_q": [7], "cubes": {色: (pos3, quat4)}}）を当てる。None なら WorldRig と同じ。
    Layout.start の PLACED_STARTS（placed_home・placed_standby_start・standby_end・xpl）は、どれも待機位置の開始
    （starts[BASE_START]）から始めて置き直す（開始の 3 通りで準備のしかたをそろえる。START_RULE）。"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        for k in PLACED_STARTS:
            self.starts[k] = self.starts[BASE_START]
        self.diag_override = None
        self.diag_applied = None

    def reset(self, layout) -> None:
        if layout.start in PLACED_STARTS and not (self.diag_override or {}).get("arm_q"):
            raise ValueError(f"開始 {layout.start} は置き直し（diag_override の arm_q）が要る")   # 黙って待機位置の開始のまま回さない
        super().reset(layout)
        self.diag_applied = None
        if self.diag_override:
            self.diag_applied = self._apply_override(self.diag_override)

    def _apply_override(self, ov: dict) -> dict:
        m, d = self.model, self.data
        arm_act = np.array([a[0] for a in self.controller.arm])
        q_goal = None if ov.get("arm_q") is None else np.asarray(ov["arm_q"], float)
        for c, (pos, quat) in (ov.get("cubes") or {}).items():
            qa, va = scene.cube_qpos_adr(m, c)
            d.qpos[qa:qa + 3] = pos
            d.qpos[qa + 3:qa + 7] = np.asarray(quat, float) / np.linalg.norm(quat)
            d.qvel[va:va + 6] = 0.0
        if q_goal is not None:
            off = d.ctrl[arm_act] - d.qpos[self.arm_qadr]            # 元の開始のサーボのずれを最初の見込みにする
            d.qpos[self.arm_qadr] = q_goal
            d.qvel[self.arm_vadr] = 0.0
            d.ctrl[arm_act] = q_goal + off
        d.qacc_warmstart[:] = 0.0
        mujoco.mj_forward(m, d)
        n = int(round(float(self.cfg["scene"]["settle_prefilled_s"]) / self.timestep))
        fixes = {int(round(f * n)) for f in SETTLE_FIXES}
        for k in range(n):
            self.hand.step()                                         # 指は開いたまま（physics_step と同じ順）
            mujoco.mj_step(m, d)
            if q_goal is not None and (k + 1) in fixes:
                d.ctrl[arm_act] += q_goal - d.qpos[self.arm_qadr]   # 位置のサーボの定常のずれだけ直す
        d.time = 0.0
        mujoco.mj_forward(m, d)
        self.step = 0
        self.meter.reset_window()
        self.safety.reset_trial()
        c = self.controller
        c.q_des = d.ctrl[arm_act].copy()
        c.gripper_closed = False
        c.sync_target_to_hand()
        self.integrator.reset(c.target_pos)
        self.hand.reset(opened=True)
        self.audit.reset()
        self._last_cmd = d.ctrl[arm_act].copy()
        q = d.qpos[self.arm_qadr].copy()
        return {"arm_q_err_max_rad": None if q_goal is None else float(np.max(np.abs(q - q_goal))),
                "arm_q": q.tolist(), "hand": d.xpos[self.hand_id].tolist(), "fingers": d.qpos[self.finger_qadr].tolist(),
                "cube_pos": d.xpos[self.cube_ids].tolist(),
                "cube_in_box": [bool(frames.in_box(d.xpos[i], self.box)) for i in self.cube_ids]}


# ---------------------------------------------------------------- 集計（記録だけを読む。判定はしない）
def single_first_close(meta: dict, z) -> dict:
    """単発の試行の最初の閉じ（試行全体の中。e7.first_close と同じ物差し）。"""
    ci = COLORS.index(meta["target"])
    return E7.first_close(z, ci, 0, len(np.asarray(z["sim_time"])))


def summarize_single(d) -> dict:
    """条件のフォルダ 1 つ（run の記録 trial_NNNN.json・npz）。持ち上がりの分母は最初の閉じが起きた試行（s4_gates の
    first_close_lift の単発の分母）、+y のずれの分母も同じ。30 s より後に最初の閉じが起きた試行の数も出す（8-1）。"""
    d = pathlib.Path(d)
    rows = []
    for p in sorted(d.glob("trial_[0-9][0-9][0-9][0-9].json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        z = np.load(p.with_suffix(".npz"))
        fc = single_first_close(m, z)
        dg = m.get("diag") or {}
        rows.append({"trial": m["trial"], "seed": m["seed"], "target": m["target"], "success": bool(m["success"]),
                     "t_success": m.get("t_success"), "first_close": fc, "key": f"{m['seed']}:{m['target']}",
                     "rep": dg.get("rep"), "state_index": dg.get("state_index")})
    fcs = [r["first_close"] for r in rows if r["first_close"]]
    n = len(rows)
    return {"dir": str(d), "n": n, "n_close": len(fcs),
            "plus_y_shift": {"k": sum(f["plus_y"] for f in fcs), "n": len(fcs),
                             "rate": None if not fcs else round(sum(f["plus_y"] for f in fcs) / len(fcs), 4)},
            "first_close_lift": {"k": sum(f["lift"] for f in fcs), "n": len(fcs),
                                 "rate": None if not fcs else round(sum(f["lift"] for f in fcs) / len(fcs), 4)},
            "first_close_after_30s": sum(1 for f in fcs if f["t"] > 30.0 + 1e-9),
            "success_30": sum(1 for r in rows if r["success"] and r["t_success"] is not None and r["t_success"] <= 30.0 + 1e-9),
            "success_60": sum(1 for r in rows if r["success"]),
            "dy_cm_median": None if not fcs else round(float(np.median([f["dy_cm"] for f in fcs])), 3),
            "rows": rows}


def _fisher_one_sided(k_hi, n_hi, k_lo, n_lo):
    try:
        from scipy.stats import fisher_exact
    except Exception:                                # noqa: BLE001
        return None
    return float(fisher_exact([[k_hi, n_hi - k_hi], [k_lo, n_lo - k_lo]], alternative="greater")[1])


def contrast(a: dict, b: dict) -> dict:
    """a − b の +y のずれ（主）と 1 回目の持ち上がり（副）。p は a > b の片側のフィッシャーの正確検定（ふるい。s4_gates の S）。"""
    pa, pb = a["plus_y_shift"], b["plus_y_shift"]
    out = {"plus_y_a": pa, "plus_y_b": pb,
           "diff": None if pa["rate"] is None or pb["rate"] is None else round(pa["rate"] - pb["rate"], 4),
           "higher": None if pa["rate"] is None or pb["rate"] is None else max(pa["rate"], pb["rate"]),
           "p_one_sided_a_gt_b": None if not pa["n"] or not pb["n"] else _fisher_one_sided(pa["k"], pa["n"], pb["k"], pb["n"]),
           "lift_a": a["first_close_lift"], "lift_b": b["first_close_lift"]}
    return out


def gate_S_inputs(summ: dict) -> dict:
    """関門 S の材料。summ = {"<start>|<prior>": summarize_single}（start は home・standby_start・standby_end、prior は on_grid・wall_side）。
    判定はしない（effect_present_rule の当てはめは二重集計役）。"""
    g = lambda s, p: summ.get(f"{s}|{p}")                       # noqa: E731
    out = {"conditions": {k: {kk: v[kk] for kk in ("n", "n_close", "plus_y_shift", "first_close_lift", "first_close_after_30s",
                                                    "success_30", "success_60", "dy_cm_median")} for k, v in summ.items()}}
    if g("standby_end", "on_grid") and g("standby_start", "on_grid"):
        out["S.pose"] = contrast(g("standby_end", "on_grid"), g("standby_start", "on_grid"))
    if g("standby_end", "wall_side") and g("standby_start", "wall_side"):
        out["S.pose_replication_wall"] = contrast(g("standby_end", "wall_side"), g("standby_start", "wall_side"))
    if g("standby_end", "wall_side") and g("standby_end", "on_grid"):
        out["S.prior"] = contrast(g("standby_end", "wall_side"), g("standby_end", "on_grid"))
    for s in ("standby_start", "home"):
        if g(s, "wall_side") and g(s, "on_grid"):
            out[f"S.prior_replication_{s}"] = contrast(g(s, "wall_side"), g(s, "on_grid"))
    out["note"] = "判定はしない。S.pose・S.prior の effect_present_rule（高いほう >= 0.40、差 >= 0.25、p < 0.05）は二重集計役が当てはめる"
    return out


def _rate(rows):
    fc = [r["first_close"] for r in rows if r["first_close"]]
    return {"k": sum(f["plus_y"] for f in fc), "n": len(fc), "rate": None if not fc else round(sum(f["plus_y"] for f in fc) / len(fc), 4),
            "lift_k": sum(f["lift"] for f in fc)}


def xpl_inputs(summ: dict) -> dict:
    """移植の腕の材料。summ = {"as"|"home"|"grid": {1: summarize_single(rep1), 2: summarize_single(rep2)}}。判定はしない。"""
    out = {"arms": {}}
    for arm, reps in summ.items():
        rows = [r for s in reps.values() for r in s["rows"]]
        out["arms"][f"XPL_{arm}"] = {"both": _rate(rows), **{f"rep{k}": _rate(s["rows"]) for k, s in reps.items()}}
    a = out["arms"]
    for other in ("home", "grid"):
        x, o = a.get("XPL_as"), a.get(f"XPL_{other}")
        if not (x and o):
            continue
        dd = {"both": None if x["both"]["rate"] is None or o["both"]["rate"] is None else round(x["both"]["rate"] - o["both"]["rate"], 4)}
        for k in ("rep1", "rep2"):
            if k in x and k in o and x[k]["rate"] is not None and o[k]["rate"] is not None:
                dd[k] = round(x[k]["rate"] - o[k]["rate"], 4)
        out[f"as_minus_{other}"] = dd
    out["note"] = "判定はしない。XPL.rep（XPL_as >= 0.40）・XPL.pose・XPL.prior（差 >= 0.25、各回 >= 0.15）は二重集計役が当てはめる。p 値は出さない"
    return out
