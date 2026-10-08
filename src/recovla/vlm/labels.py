"""VLM の試験台の材料（問いの項目）と正解の規則。記録だけを読む（評価の側。真値を使ってよいのは採点だけ）。

材料（docs/stage4/vlm_bench_protocol.md の 2 節。結果を見る前に固定）:
  完了の判定（task "completion"）: 段階 3 の 3 個の連続タスク outputs/v2eval/V3S3/E7_R1v3（run_0000〜0019）の各手順について、
    基準の時刻 t_ref（判定した手順は実行器の t_judge、判定しなかった手順はその手順の t_end）と、その前後
    （t_ref − 4 s・− 2 s・± 0・+ 1 s）と、手順の始め + 1 s と、判定されずに終わった途中の試みの終わり。問いはその手順の色だけ。
    正解 = npz の cube_in_box かつ「持たれていない」（gripper_closed かつ 指先の中心と立方体の中心が held_dist_m 未満、でない）。
    取りこぼしの場面 = 判定されずに終わった試みの終わりの前に、正解の「箱の中」が FN_MIN_HOLD_S 以上続いていた。
  失敗の種類の分類（task "failure"）: 単発の試行の記録（V3S3 の A・B の自然・P1〜P3。束 1 の本番が終わった後は S4DREC など）から、
    下の規則で出来事（掴み損ね・落下・置き損ね・停滞・成功）を取り出し、基準の時刻 + FRAME_OFFSETS の 4 コマを見せる。

正解の規則（RULE。閾値は configs/default.yaml の eval.failure_detect と同じ値を使い、足した値は下の定数）:
  grasp_miss（掴み損ね）: グリッパが閉じた（gripper_closed 0→1）こまから grasp_window_s 以内に、目標が lift_m 以上持ち上がらなかった。
      基準の時刻 = 閉じた時刻 + GRASP_ANCHOR_DELAY_S。
  持ち上げた後に手を離れた（release）: 「持っている」（閉 かつ 指先と目標の中心が held_dist_m 未満）の区間で目標が lift_m 以上
      持ち上がり、その区間が終わったこま（基準の時刻）。その後 REST_SEARCH_S 以内（次に持つまで）に、目標の速さ（npz の cube_linvel）
      が rest_speed 未満で rest_hold_s 続いた最初の静止で、行き先を決める（release_class）:
        箱の中（cube_in_box）                                   → success（成功）
        机の上でない所（静止の高さ − 机の静止の高さ ≥ table_z_tol_m。箱の縁・他の立方体の上） → misplace（置き損ね）
        机の上で、手を離れた時に指が開いていない（閉じたまま滑り落ちた）              → drop（落とした）
        机の上で、手を離れる前 VZ_WINDOW_S の指先の上下の速さ > RISE_VZ（上がりながら） → drop
        机の上で、上下の速さ <= PLACE_VZ（下りながら）か、直前 STOP_WINDOW_S の指先の水平の速さ < STOP_SPEED（止まって）
                                                                                   → misplace（置く動きで、置いた所が違う）
        机の上で、運ぶ途中に指を開いた（上のどれでもない）                          → drop（運ぶ途中で落とした）
        静止が見つからない                                                       → 除外（unresolved）
      閾値は段階 3 の記録の誘発（P2: 運ぶ高さで上下 0・水平に動きながら開く、P3: 下りながら約 −0.1 m/s で開く）が分かれるように、
      VLM に問う前に決めた（手順書 3 節）。
  stall（止まった・動けない）: 指先の速さ < stall_hand_speed かつ 指の速さ < stall_finger_speed が stall_s 以上続き、その間ずっと
      目標が箱の中でない。始まりが STALL_MIN_START_S より後のものだけ。基準の時刻 = 始まり + STALL_ANCHOR_S。
  成功の出来事が出たら、その試行のそれより後の出来事は使わない（成功の後の静止を停滞と数えない）。
  誘発の試行（induce.established が真）: 誘発の時刻に合う出来事（P1: 閉じた時刻が induce.info.t_closed から ± INDUCE_MATCH_S、
      P2: 手を離れた時刻が [t_fire − INDUCE_MATCH_S, t_established]、P3: [t_fire, t_established]）の正解を誘発の種類
      （INDUCE_CLASS）にする（label_source "induce"）。規則の種類と違えば conflict（主の採点から除き、別に数える）。
      合う出来事が無ければ no_rule_match として除外する。ほかの出来事は規則の種類（label_source "rule"）。
  除外（EXCLUDE）: 4 コマのどれかが記録の外（< 0 か 記録の最後のこま + END_TOL_S より後）、前に残した出来事の基準の時刻から
      OVERLAP_S 以内、同じ試行の同じ種類の 2 つ目以降（MAX_PER_CLASS_PER_TRIAL。誘発の出来事は先に残す）、静止が見つからない
      （unresolved）、conflict・no_rule_match（問わない。記録には残す）。

描く時刻（resolve_time。腕と立方体を同じ時刻にそろえる）: 求めた時刻に最も近い推論（runtime の inference の t_obs、state あり）が
  SNAP_TOL_S 以内にあれば、その t_obs で描く（腕 = state[8:15]、指と立方体 = npz の sim_time <= t_obs の最後のこま）。
  無ければ（戻す動き・判定の待ち・方策の停止の後など）、求めた時刻の npz のこまで描き、腕は記録の手の位置・向き（ee_pos・ee_quat）から
  IK で求める（joints_source "ik"）。正解はどちらも描いたこまの値。
"""
import json
import pathlib

import numpy as np

from recovla.common import config
from recovla.common.seeds import COLORS

_CFG = config.load()
FD = {k: float(v) for k, v in _CFG["eval"]["failure_detect"].items()}
CUBE_REST_Z = float(_CFG["sim"]["table_top_z"]) + 0.5 * float(_CFG["scene"]["cube_size"])

CLASSES = ("grasp_miss", "drop", "misplace", "stall", "success")
CLASS_JA = {"grasp_miss": "掴み損ね", "drop": "持ち上げた後に落とした", "misplace": "箱の縁などに置き損ねた",
            "stall": "止まった・動けない", "success": "成功"}
FAILURE_CLASSES = ("grasp_miss", "drop", "misplace", "stall")
INDUCE_CLASS = {"P1": "grasp_miss", "P2": "drop", "P3": "misplace"}
S4_VARIANT_CLASS = {"grasp_failure": "grasp_miss", "fall_as_is": "drop", "fall_with_hold": "drop", "misplace": "misplace"}

FRAME_OFFSETS = (-2.0, -0.5, 0.5, 1.0)          # 失敗の分類の 4 コマ（基準の時刻からの秒。全部の種類で同じ）
GRASP_ANCHOR_DELAY_S = 1.5   # 掴み損ねの基準の時刻 = 閉じた時刻 + これ（4 コマが 閉じる 0.5 s 前・閉じた 1.0・2.0・2.5 s 後になる）
COMPLETION_ANCHORS = (("start+1", "start", 1.0), ("ref-4", "ref", -4.0), ("ref-2", "ref", -2.0), ("ref", "ref", 0.0),
                      ("ref+1", "ref", 1.0))
FN_MIN_HOLD_S = 2.0          # 取りこぼしの場面: 判定されずに終わった試みの終わりの前に、正解の「箱の中」がこれ以上続いていた
STOP_SPEED = 0.03            # m/s。指を開いた時の指先の水平の速さがこれ未満なら「止まって置いた」
STOP_WINDOW_S = 0.3
VZ_WINDOW_S = 0.5            # 手を離れる前の指先の上下の速さを測る長さ
PLACE_VZ = -0.05             # m/s。これ以下（下りながら）で指を開いたら置く動き
RISE_VZ = 0.02               # m/s。これより上がりながら手を離れたら「落とした」
MAX_PER_CLASS_PER_TRIAL = 1  # 1 本の試行から、同じ種類の出来事は最初の 1 つだけ使う
OPEN_MATCH_S = 0.15          # 手を離れた時刻と指を開いた時刻の対応の幅
REST_SEARCH_S = 3.0          # 手を離れてから静止を探す長さ
STALL_MIN_START_S = 1.0
STALL_ANCHOR_S = 2.0
OVERLAP_S = 3.0
INDUCE_MATCH_S = 0.3
SNAP_TOL_S = 0.35            # 推論の t_obs へ寄せる幅（推論は約 0.6 s ごと）
END_TOL_S = 0.1              # 記録の最後のこまより後でも、この幅までは最後のこまで描く（記録は 20 Hz。t_end はこまの間に落ちる）
EPS = 1e-9

RULE_VERSION = "vlm-labels-v1"


def rule_constants() -> dict:
    return {"version": RULE_VERSION, "failure_detect": dict(FD), "cube_rest_z": CUBE_REST_Z,
            "frame_offsets": list(FRAME_OFFSETS), "grasp_anchor_delay_s": GRASP_ANCHOR_DELAY_S,
            "completion_anchors": [list(a) for a in COMPLETION_ANCHORS], "fn_min_hold_s": FN_MIN_HOLD_S,
            "stop_speed": STOP_SPEED, "stop_window_s": STOP_WINDOW_S, "vz_window_s": VZ_WINDOW_S, "place_vz": PLACE_VZ,
            "rise_vz": RISE_VZ, "max_per_class_per_trial": MAX_PER_CLASS_PER_TRIAL, "open_match_s": OPEN_MATCH_S,
            "rest_search_s": REST_SEARCH_S, "stall_min_start_s": STALL_MIN_START_S, "stall_anchor_s": STALL_ANCHOR_S,
            "overlap_s": OVERLAP_S, "induce_match_s": INDUCE_MATCH_S, "snap_tol_s": SNAP_TOL_S, "end_tol_s": END_TOL_S,
            "induce_class": INDUCE_CLASS, "s4_variant_class": S4_VARIANT_CLASS}


# ---------------------------------------------------------------- 記録の読み込み（読むだけ）
def load_npz(path) -> dict:
    with np.load(path) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def load_single(d, i: int) -> dict:
    """単発の試行の記録（82・96 の run と同じ形: trial_NNNN.json・.npz・runtime_NNNN.json）。"""
    d = pathlib.Path(d)
    meta = json.loads((d / f"trial_{i:04d}.json").read_text(encoding="utf-8"))
    rt_path = d / f"runtime_{i:04d}.json"
    rt = json.loads(rt_path.read_text(encoding="utf-8")) if rt_path.is_file() else {}
    return {"meta": meta, "z": load_npz(d / f"trial_{i:04d}.npz"), "inference": _inference(rt), "dir": str(d), "index": i}


def load_task(d, i: int) -> dict:
    """3 個の連続タスクの記録（run_NNNN.json・.npz・run_NNNN_runtime.json）。"""
    d = pathlib.Path(d)
    meta = json.loads((d / f"run_{i:04d}.json").read_text(encoding="utf-8"))
    rt = json.loads((d / f"run_{i:04d}_runtime.json").read_text(encoding="utf-8"))
    return {"meta": meta, "z": load_npz(d / f"run_{i:04d}.npz"), "inference": _inference(rt),
            "judge_rows": (rt.get("executor") or {}).get("judge_rows") or [], "dir": str(d), "index": i}


def _inference(rt: dict) -> list:
    r = rt.get("runtime", rt) if isinstance(rt, dict) else {}
    out = []
    for x in r.get("inference") or []:
        if x.get("state") is not None and x.get("t_obs") is not None:
            out.append({"i": x.get("i"), "t_obs": float(x["t_obs"]), "state": list(x["state"])})
    return out


def list_indices(d, pattern: str) -> list:
    """d の中の pattern（"trial" か "run"）_NNNN.json の番号の並び。"""
    import re
    rx = re.compile(rf"^{pattern}_(\d{{4}})\.json$")
    return sorted(int(m.group(1)) for p in pathlib.Path(d).iterdir() if (m := rx.match(p.name)))


# ---------------------------------------------------------------- 描く時刻
def frame_at(t_arr, t: float) -> int:
    """sim_time <= t の最後のこま（無ければ 0）。"""
    return max(0, int(np.searchsorted(np.asarray(t_arr, float), float(t) + EPS, side="right")) - 1)


def resolve_time(t_req: float, inference: list, t_arr, tol: float = SNAP_TOL_S) -> dict:
    """描く時刻と腕の出どころ。返り値 {t_req, t_render, frame, joints_source, inference_i, joints}。"""
    best = None
    for x in inference:
        dt = abs(x["t_obs"] - t_req)
        if best is None or dt < best[0] - EPS:
            best = (dt, x)
    if best is not None and best[0] <= tol + EPS:
        x = best[1]
        f = frame_at(t_arr, x["t_obs"])
        return {"t_req": float(t_req), "t_render": float(x["t_obs"]), "frame": f, "joints_source": "inference",
                "inference_i": x.get("i"), "joints": [float(v) for v in x["state"][8:15]]}
    f = frame_at(t_arr, t_req)
    seed = None if best is None else [float(v) for v in best[1]["state"][8:15]]
    return {"t_req": float(t_req), "t_render": float(np.asarray(t_arr)[f]), "frame": f, "joints_source": "ik",
            "inference_i": None if best is None else best[1].get("i"), "joints": None, "ik_seed": seed}


# ---------------------------------------------------------------- 状態の派生量
def held_mask(z: dict, ci: int) -> np.ndarray:
    gc = np.asarray(z["gripper_closed"]).astype(bool)
    d = np.linalg.norm(np.asarray(z["fingertip"], float) - np.asarray(z["cube_pos"], float)[:, ci], axis=1)
    return gc & (d < FD["held_dist_m"])


def truth_in_box(z: dict, f: int, ci: int) -> dict:
    raw = bool(np.asarray(z["cube_in_box"])[f, ci])
    held = bool(held_mask(z, ci)[f])
    return {"in_box": raw and not held, "cube_in_box_raw": raw, "held": held}


def _rise(z, ci) -> np.ndarray:
    return np.asarray(z["cube_pos"], float)[:, ci, 2] - CUBE_REST_Z


def _runs(mask) -> list:
    """真の連続区間 [a, b) の並び。"""
    m = np.asarray(mask, bool).astype(int)
    d = np.diff(np.r_[0, m, 0])
    return list(zip(np.nonzero(d == 1)[0].tolist(), np.nonzero(d == -1)[0].tolist()))


# ---------------------------------------------------------------- 出来事（規則）
def grasp_miss_events(z: dict, ci: int) -> list:
    t = np.asarray(z["sim_time"], float)
    gc = np.asarray(z["gripper_closed"]).astype(int)
    rise = _rise(z, ci)
    out = []
    for f in (np.nonzero(np.diff(gc) == 1)[0] + 1).tolist():
        t0 = t[f]
        if t[-1] < t0 + FD["grasp_window_s"] - EPS:
            out.append({"kind": "grasp_miss", "t": float(t0), "frame": f, "unresolved": True})
            continue
        w = (t >= t0 - EPS) & (t <= t0 + FD["grasp_window_s"] + EPS)
        if float(rise[w].max()) < FD["lift_m"]:
            out.append({"kind": "grasp_miss", "t": float(t0), "frame": f})
    return out


def release_events(z: dict, ci: int) -> list:
    t = np.asarray(z["sim_time"], float)
    gc = np.asarray(z["gripper_closed"]).astype(bool)
    held = held_mask(z, ci)
    rise = _rise(z, ci)
    inbox = np.asarray(z["cube_in_box"]).astype(bool)[:, ci]
    speed = np.linalg.norm(np.asarray(z["cube_linvel"], float)[:, ci], axis=1)
    tip = np.asarray(z["fingertip"], float)
    runs = _runs(held)
    out = []
    for k, (a, b) in enumerate(runs):
        if float(rise[a:b].max()) < FD["lift_m"] or b >= len(t):
            continue
        tr = float(t[b])
        nxt = runs[k + 1][0] if k + 1 < len(runs) else len(t)
        stop = min(nxt, int(np.searchsorted(t, tr + REST_SEARCH_S + EPS, side="right")))
        rest_f, acc, start = None, 0.0, None
        for g in range(b, stop):
            if speed[g] < FD["rest_speed"]:
                if start is None:
                    start = g
                acc = t[g] - t[start]
                if acc >= FD["rest_hold_s"] - EPS:
                    rest_f = g
                    break
            else:
                start, acc = None, 0.0
        opens = [int(f) for f in (np.nonzero(np.diff(gc.astype(int)) == -1)[0] + 1)
                 if abs(t[f] - tr) <= OPEN_MATCH_S + EPS]
        opened = bool(opens) or not bool(gc[b])
        w = (t >= tr - STOP_WINDOW_S - EPS) & (t <= tr + EPS)
        idx = np.nonzero(w)[0]
        if len(idx) >= 2:
            dxy = np.linalg.norm(np.diff(tip[idx, :2], axis=0), axis=1)
            dts = np.diff(t[idx])
            hand_v = float(dxy.sum() / max(dts.sum(), EPS))
        else:
            hand_v = None
        f0 = frame_at(t, tr - VZ_WINDOW_S)
        vz = float((tip[b, 2] - tip[f0, 2]) / max(t[b] - t[f0], EPS)) if b > f0 else 0.0
        ev = {"kind": "release", "t": tr, "frame": int(b), "opened": opened, "hand_xy_speed": hand_v, "hand_vz": vz,
              "max_rise_m": float(rise[a:b].max()), "rest_frame": rest_f}
        if rest_f is None:
            ev.update(cls=None, unresolved=True)
        else:
            on_table = float(rise[rest_f]) < FD["table_z_tol_m"]
            ev.update(rest_in_box=bool(inbox[rest_f]), rest_on_table=on_table, rest_rise_m=float(rise[rest_f]))
            ev["cls"] = release_class(bool(inbox[rest_f]), on_table, opened, vz, hand_v)
        out.append(ev)
    return out


def release_class(rest_in_box: bool, rest_on_table: bool, opened: bool, vz: float, hand_xy_speed) -> str:
    """手を離れた出来事の正解（RULE の表）。置く動き = 下りながら（vz <= PLACE_VZ）か、水平に止まって（< STOP_SPEED）指を開いた。"""
    if rest_in_box:
        return "success"
    if not rest_on_table:
        return "misplace"
    if not opened or vz > RISE_VZ:
        return "drop"
    if vz <= PLACE_VZ or (hand_xy_speed is not None and hand_xy_speed < STOP_SPEED):
        return "misplace"
    return "drop"


def stall_events(z: dict, ci: int) -> list:
    t = np.asarray(z["sim_time"], float)
    tip = np.asarray(z["fingertip"], float)
    fing = np.asarray(z["fingers"], float)
    inbox = np.asarray(z["cube_in_box"]).astype(bool)[:, ci]
    dt = np.diff(t)
    hv = np.r_[np.inf, np.linalg.norm(np.diff(tip, axis=0), axis=1) / np.maximum(dt, EPS)]
    fv = np.r_[np.inf, np.abs(np.diff(fing, axis=0)).sum(axis=1) / np.maximum(dt, EPS)]
    still = (hv < FD["stall_hand_speed"]) & (fv < FD["stall_finger_speed"]) & ~inbox
    out = []
    for a, b in _runs(still):
        t0 = float(t[a - 1]) if a > 0 else float(t[a])      # 速さはこまの差分なので、止まり始めは 1 つ前のこま
        if t0 < STALL_MIN_START_S - EPS or float(t[b - 1]) - t0 < FD["stall_s"] - EPS:
            continue
        out.append({"kind": "stall", "t": t0, "frame": int(max(a - 1, 0)), "t_until": float(t[b - 1]),
                    "ended_by_record": b >= len(t)})
    return out


def trial_events(z: dict, ci: int) -> list:
    """規則の出来事を基準の時刻の順に並べる（成功の後は切る）。各要素に cls（正解の種類か None）と anchor を付ける。"""
    evs = []
    for e in grasp_miss_events(z, ci):
        evs.append(dict(e, cls=None if e.get("unresolved") else "grasp_miss", anchor=e["t"] + GRASP_ANCHOR_DELAY_S))
    for e in release_events(z, ci):
        evs.append(dict(e, anchor=e["t"]))
    for e in stall_events(z, ci):
        evs.append(dict(e, cls="stall", anchor=e["t"] + STALL_ANCHOR_S))
    evs.sort(key=lambda e: (e["anchor"], e["kind"]))
    out = []
    for e in evs:
        out.append(e)
        if e.get("cls") == "success":
            break
    return out


def _target_index(meta: dict, z: dict) -> int:
    if meta.get("target") in COLORS:
        return COLORS.index(meta["target"])
    return int(np.asarray(z["target"])[0])


def induce_class(meta: dict, cond: str = "") -> str:
    ind = meta.get("induce") or {}
    var = ind.get("variant") or ((meta.get("diag") or {}).get("variant"))
    if var in S4_VARIANT_CLASS:
        return S4_VARIANT_CLASS[var]
    for v, c in S4_VARIANT_CLASS.items():
        if cond.endswith("_" + v):
            return c
    return INDUCE_CLASS.get(ind.get("kind"))


def _induce_match(meta: dict, e: dict) -> bool:
    ind = meta.get("induce") or {}
    kind = ind.get("kind")
    if kind == "P1" or induce_class(meta) == "grasp_miss":
        tc = (ind.get("info") or {}).get("t_closed", ind.get("t_failure"))
        return e["kind"] == "grasp_miss" and tc is not None and abs(e["t"] - float(tc)) <= INDUCE_MATCH_S + EPS
    tf, te = ind.get("t_fire"), ind.get("t_established")
    if e["kind"] != "release" or tf is None or te is None:
        return False
    lo = float(tf) - (INDUCE_MATCH_S if induce_class(meta) == "drop" else 0.0)
    return lo - EPS <= e["t"] <= float(te) + EPS


def failure_items(rec: dict, source: str, cond: str = "") -> list:
    """単発の試行 1 本の出来事 → 問いの項目（除外したものも exclude の理由つきで返す）。"""
    meta, z = rec["meta"], rec["z"]
    ci = _target_index(meta, z)
    t = np.asarray(z["sim_time"], float)
    evs = trial_events(z, ci)
    ind = meta.get("induce") or {}
    established = bool(ind.get("established"))
    icls = induce_class(meta, cond) if established else None
    matched = None
    if established:
        for k, e in enumerate(evs):
            if _induce_match(meta, e):
                matched = k
                break
    items, kept_anchor, kept_cls = [], [], {}
    order = ([matched] if matched is not None else []) + [k for k in range(len(evs)) if k != matched]   # 誘発の出来事を先に残す
    for k in order:
        e = evs[k]
        label, src, excl = e.get("cls"), "rule", []
        if k == matched:
            label, src = icls, "induce"
            if e.get("cls") != icls:
                excl.append("conflict")
        if e.get("unresolved") or (e.get("cls") is None and k != matched):
            excl.append("unresolved")
        times = [e["anchor"] + o for o in FRAME_OFFSETS]
        if times[0] < -EPS or times[-1] > t[-1] + END_TOL_S:
            excl.append("window_out_of_record")
        if any(abs(e["anchor"] - a) < OVERLAP_S - EPS for a in kept_anchor):
            excl.append("overlap")
        if k != matched and kept_cls.get(label, 0) >= MAX_PER_CLASS_PER_TRIAL:
            excl.append("per_class_cap")
        if not excl:
            kept_anchor.append(e["anchor"])
            kept_cls[label] = kept_cls.get(label, 0) + 1
        frames = [resolve_time(x, rec["inference"], t) if x >= -EPS and x <= t[-1] + END_TOL_S else {"t_req": x, "out": True}
                  for x in times]
        items.append({"task": "failure", "item_id": f"{source.replace('/', '_')}_t{rec['index']:04d}_e{k}",
                      "source": source, "condition": cond, "index": rec["index"], "seed": meta.get("seed"),
                      "target": COLORS[ci], "model_policy": (meta.get("model") or {}).get("name") if isinstance(meta.get("model"), dict) else meta.get("model"),
                      "induce": {"kind": ind.get("kind"), "established": established, "class": icls,
                                 "t_fire": ind.get("t_fire"), "t_established": ind.get("t_established")},
                      "event": {k2: v for k2, v in e.items() if k2 not in ("cls",)}, "rule_class": e.get("cls"),
                      "label": label, "label_source": src, "exclude": excl, "frames": frames,
                      "trial_success": meta.get("success"), "t_success": meta.get("t_success")})
    if established and matched is None:
        items.append({"task": "failure", "item_id": f"{source.replace('/', '_')}_t{rec['index']:04d}_induce",
                      "source": source, "condition": cond, "index": rec["index"], "seed": meta.get("seed"),
                      "target": COLORS[ci], "induce": {"kind": ind.get("kind"), "established": True, "class": icls,
                                                        "t_fire": ind.get("t_fire"), "t_established": ind.get("t_established")},
                      "event": None, "rule_class": None, "label": icls, "label_source": "induce",
                      "exclude": ["no_rule_match"], "frames": []})
    return items


# ---------------------------------------------------------------- 完了の判定の項目（E7）
def _nearest_row(rows: list, step: int, t: float):
    best = None
    for r in rows:
        if r.get("step") != step:
            continue
        if best is None or abs(r["t"] - t) < abs(best["t"] - t):
            best = r
    if best is None:
        return None
    return {k: best.get(k) for k in ("t", "attempt", "phase", "ok", "held_s", "box_pixels", "wrist_in", "depth_ok",
                                      "retreat_dist_m")}


def attempt_ends(meta: dict, s: dict) -> list:
    """手順 s の試みごとの (試みの番号, 終わりの時刻, 判定されたか)。終わり = やり直しの戻す動きの始め（returns の retry）か、
    次の試みの始めか、手順の t_end。"""
    atts = s.get("attempts") or []
    rets = [r for r in (meta.get("returns") or []) if r.get("step") == s["step"]]
    out = []
    for k, a in enumerate(atts):
        if a.get("t_judge") is not None:
            out.append((a["attempt"], float(a["t_judge"]), True))
            continue
        r = next((r for r in rets if r.get("attempt") == a["attempt"] and r.get("kind") == "retry"), None)
        if r is not None:
            te = float(r["t_begin"])
        elif k + 1 < len(atts):
            te = float(atts[k + 1]["t_begin"])
        else:
            te = float(s["t_end"]) if s.get("t_end") is not None else float(meta.get("t_end"))
        out.append((a["attempt"], te, False))
    return out


def held_true_before(z: dict, ci: int, t_at: float) -> float:
    """t_at のこまから過去へ、正解の「箱の中（持たれていない）」が続いた長さ [s]（t_at で偽なら 0）。"""
    t = np.asarray(z["sim_time"], float)
    tb = np.asarray(z["cube_in_box"]).astype(bool)[:, ci] & ~held_mask(z, ci)
    f = frame_at(t, t_at)
    if not tb[f]:
        return 0.0
    g = f
    while g > 0 and tb[g - 1]:
        g -= 1
    return float(t[f] - t[g])


def completion_items(rec: dict, source: str) -> tuple:
    """3 個の連続タスク 1 本 → (項目の並び, 手順ごとの要約)。"""
    meta, z = rec["meta"], rec["z"]
    t = np.asarray(z["sim_time"], float)
    items, steps = [], []
    for s in meta.get("steps") or []:
        ci = COLORS.index(s["color"])
        t_start = float(s["t_start"])
        t_end = float(s["t_end"]) if s.get("t_end") is not None else float(meta.get("t_end", t[-1]))
        judged = bool(s.get("judged_complete"))
        t_ref = float(s["t_judge"]) if judged and s.get("t_judge") is not None else t_end
        f_ref = frame_at(t, t_ref)
        tr_ref = truth_in_box(z, f_ref, ci)
        aends = attempt_ends(meta, s)
        fn_att = [{"attempt": a, "t_end": te, "in_box_held_s": round(held_true_before(z, ci, te), 3)}
                  for a, te, j in aends if not j and held_true_before(z, ci, te) >= FN_MIN_HOLD_S - EPS]
        summ = {"run": rec["index"], "seed": meta.get("seed"), "step": s["step"], "color": s["color"],
                "judged_complete": judged, "t_judge": s.get("t_judge"), "t_start": t_start, "t_end": t_end,
                "t_ref": t_ref, "truth_at_ref": tr_ref, "attempt_ends": [list(x) for x in aends],
                "fn_attempts": fn_att, "fn_scene": bool(fn_att),
                "fp_scene": judged and not tr_ref["in_box"]}
        steps.append(summ)
        seen = set()
        anchors = [(name, (t_start if base == "start" else t_ref) + off) for name, base, off in COMPLETION_ANCHORS]
        anchors += [(f"a{a}_end", te) for a, te, j in aends if not j and abs(te - t_ref) > EPS]   # 判定されなかった途中の試みの終わり
        for name, tq in anchors:
            excl = []
            if tq < -EPS or tq > t[-1] + END_TOL_S:
                excl.append("window_out_of_record")
                fr = {"t_req": tq, "out": True}
            else:
                fr = resolve_time(tq, rec["inference"], t)
                if fr["frame"] in seen:
                    excl.append("same_frame")
                seen.add(fr["frame"])
            truth = truth_in_box(z, fr["frame"], ci) if "frame" in fr else None
            # 実行器が決めた時刻の答え: 判定した手順の ref は「完了」、判定されずに終わった試みの終わりは「未完了」、ほかは None
            decision = None
            if name == "ref" and judged:
                decision = True
            elif any(abs(tq - te) <= EPS and not j for _, te, j in aends):
                decision = False
            fn_point = any(abs(tq - x["t_end"]) <= EPS for x in fn_att)
            items.append({"task": "completion", "item_id": f"{source.replace('/', '_')}_r{rec['index']:04d}_s{s['step']}_{name}",
                          "source": source, "index": rec["index"], "seed": meta.get("seed"), "step": s["step"],
                          "color": s["color"], "anchor": name, "frames": [fr], "label": None if truth is None else truth["in_box"],
                          "truth": truth, "exclude": excl,
                          "executor": {"judged_complete": judged, "t_judge": s.get("t_judge"), "decision": decision,
                                       "row": None if "frame" not in fr else _nearest_row(rec["judge_rows"], s["step"], fr["t_render"])},
                          "scene": {"fn_scene": summ["fn_scene"], "fn_point": fn_point, "fp_scene": summ["fp_scene"]}})
    return items, steps


# ---------------------------------------------------------------- 材料の一覧（手順書 2 節）
E7_SOURCE = ("V3S3", "E7_R1v3")
S3_SINGLE = tuple((("V3S3", f"{arm}_{k}") for arm in ("A", "B") for k in ("nat", "P1", "P2", "P3")))
S4_SINGLE_DEFAULT = ("S4DREC",)      # 束 1 の本番が終わった後だけ（--include-s4）。中身は本番の後まで読まない


def v2eval_root() -> pathlib.Path:
    return config.path(_CFG["paths"]["outputs"]) / "v2eval"


def build_completion(root=None, n_runs: int = None) -> dict:
    root = pathlib.Path(root) if root else v2eval_root()
    d = root.joinpath(*E7_SOURCE)
    idx = list_indices(d, "run")
    if n_runs is not None:
        idx = idx[:n_runs]
    items, steps = [], []
    for i in idx:
        it, st = completion_items(load_task(d, i), "/".join(E7_SOURCE))
        items += it
        steps += st
    return {"task": "completion", "rule": rule_constants(), "sources": ["/".join(E7_SOURCE)], "items": items, "steps": steps}


def build_failure(root=None, include_s4=(), conds=S3_SINGLE, limit_per_cond: int = None) -> dict:
    """include_s4: 束 1 の本番が終わった後の実験名の並び（例 ("S4DREC",)）。その下の条件のフォルダ（trial_*.json のあるもの）を全部読む。"""
    root = pathlib.Path(root) if root else v2eval_root()
    pairs = [tuple(c) for c in conds]
    for exp in include_s4:
        base = root / exp
        if not base.is_dir():
            raise FileNotFoundError(f"{base} が無い")
        pairs += [(exp, p.name) for p in sorted(base.iterdir()) if p.is_dir() and any(p.glob("trial_[0-9][0-9][0-9][0-9].json"))]
    items = []
    for exp, cond in pairs:
        d = root / exp / cond
        idx = list_indices(d, "trial")
        if limit_per_cond is not None:
            idx = idx[:limit_per_cond]
        for i in idx:
            items += failure_items(load_single(d, i), f"{exp}/{cond}", cond)
    return {"task": "failure", "rule": rule_constants(), "sources": [f"{e}/{c}" for e, c in pairs], "items": items}


def included(items: list) -> list:
    return [x for x in items if not x["exclude"]]
