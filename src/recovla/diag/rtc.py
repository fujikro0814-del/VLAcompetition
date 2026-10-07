"""段階 4 束 1 の RTC の診断（担当 A）の部品: 6 設定の定義、影の推論の記録、関門 R の指標、X2 の予測の不足。

    from recovla.diag import rtc as D
    D.SETTINGS["paper_formula_range44_cap5"]        # 設定 → 実行系の値（mode・範囲 E・減衰の形・上限 β）
    cfg2 = D.overlay_rtc(cfg, "range10_cap5")        # configs を読んだ辞書の写しに重ねる（ファイルは書き換えない）
    prox = D.DiagPolicy(sensor_policy, shadow=True)  # 誘導した塊と、同じ観測・同じ雑音の誘導しない塊（影）を記録する
    row = D.trial_metrics(meta, z, rt, horizon_s=30) # 1 試行の関門 R の指標（記録から。GPU もシミュレーションも使わない）
    D.summarize(rows)                                # 条件の要約（中央値など）

決まり（docs/目標書_段階4.md 8-1・8-2、configs/s4_gates.json の gates.R。結果を見る前に固定）:
  半径方向の差 radial_gap_mm  最初に閉じたこま（npz の gripper_closed が最初に真）の、指先 − 目標の立方体の水平の差を、
                              根元（原点）→ 立方体の向き u = cube_xy/|cube_xy| に射影 [mm]。負は手前（W\\review_researcher\\r1_gate_noise.py の
                              rad、W\\rtc\\a3_travel.py の tip_minus_cube_radial_mm と同じ）。集計は全試行（閉じた試行）の中央値
  移動の比 move_ratio         最初に閉じる前の、方策の行動（保持でない行）の水平の和を u に射影したもの ÷ 必要な移動
                              （立方体（閉じた時点）− 指先（時刻 0））の u 成分。必要な移動が 20 mm 以下の試行は除く（a3_travel.py と同じ）。
                              集計は中央値（a3 と同じ）と平均を両方出す（どちらを主にするかは二重集計役が掲示する＝s4_gates.json の注）
  継ぎ目の速度の跳び seam_jump_mps  10 Hz の行動の速さ v = a[:3]/0.1 の、続く 2 行がどちらも保持でなく塊の番号が変わる所の |Δv| の
                              試行ごとの平均 → 全試行の中央値（W\\rtc\\a1_trial_metrics.py と同じ）
  成功 success_at[L]          success かつ t_success <= L（harness/loop.py 107 行と同じ判定。96_s4_resume.py score と同じ）
  30 s に限る（主。8-1 の注）: 移動の比は「最初の閉じが 30 s 以内の試行」だけ（区間が最初の閉じまでなので、これで最初の 30 s に
                              限った値になる）。継ぎ目の跳びは行動の時刻 t < 30 s の継ぎ目だけ。半径方向の差は最初の閉じで決まるので
                              制限時間によらない（30 s より後に最初に閉じた試行の数を n_first_close_after を出す）
  報告だけ（看板・条件にしない。gates.R.report_only_metrics）: 下りながらの水平の補正（下り始めで 25 mm 超の試行の中央値）と、
                              遠い空振り（閉じた時に 25 mm 超）の閉じの長さ（r1_gate_noise.py の red・far と同じ）
影の推論（R.b1 の材料。結果を見る前に固定）:
  誘導した塊 g と影 s（同じ観測・同じ雑音で RTC の引数を渡さない＝naive と同じ推論）の、行 0 から h 行の水平の変位の和の差
  (s − g)·u [mm]（u は観測の時刻の x_des → 目標の立方体の水平の向き。正は誘導した計画が短い）。最初の閉じの前で、
  |立方体 − x_des| >= 30 mm の推論だけ。h = 10・20・30・40。集計は推論を全部まとめた中央値。「先の計画が 15 mm 以上短い」は
  どれかの h の中央値 >= 15 mm
X2（R.b1 の材料）: エキスパートの記録の 10 fps のこま k の観測で推論した塊の、行 0 から h 行の変位の和 p_h と、エキスパートの実際の
  x_des の変位 e_h = x_des[2(k+h)] − x_des[2k] の差 (e_h − p_h)·u [mm]（u は x_des → 目標の立方体の水平の向き。正は予測が短い）。
  エキスパートの最初の閉じのこま f1 について k + h <= f1 // 2 の所だけ、|立方体 − x_des| >= 30 mm の所だけ
  （W\\rtc\\a7_v1_chunks.py の偏りと同じ向きの取り方）。集計は全こまの中央値。「不足が 15 mm 以上」はどれかの h の中央値 >= 15 mm
"""
from __future__ import annotations

import copy
import json
import math
import pathlib

import numpy as np

COLORS = ("red", "green", "blue")
ACTION_DT = 0.1
CHUNK_H = 50                 # SmolVLA の chunk_size（R1v3 の保存点。tests で lerobot と照合する）
EXEC_S = 6                   # 段階 3・4 の実行間隔 s（組 E・物差し K と同じ）
ROWS = (10, 20, 30, 40)      # 影の推論・X2 の「h 行先」
NEED_MIN_M = 0.02            # 移動の比: 必要な移動がこれ以下の試行は除く（a3_travel.py）
DIR_MIN_M = 0.03             # 影・X2: 向きを決める距離の下限（a7_v1_chunks.py）
SHORT_MM = 15.0              # R.b1 の 15 mm

# ------------------------------------------------------------------ 6 設定（結果を見る前に固定し、回す前に掲示する）
# 実行系の値: mode（PolicyRuntime の mode）、horizon（configs runtime.rtc_guidance_horizon＝lerobot の execution_horizon と、
# 前の塊の残りを揃える行数 normalize_left_over の E）、schedule（runtime.rtc_schedule＝prefix_attention_schedule）、
# max_guidance_weight（runtime.rtc_max_guidance_weight＝β）。共通: s = 6、d_init = 4（configs delay_steps）、
# 方策に渡す inference_delay = min(直前の推論で実際にかかった行の数, s − 1)（runtime/runner.py 255〜259 行）、
# 流れの積分の段数 10（SmolVLA の num_steps）、ヤコビアンなし（lerobot 0.6.1 modeling_rtc.py 212〜219 行）、安全フィルタなし
SETTINGS = {
    "current_repro": {
        "mode": "rtc", "horizon": 40, "schedule": "EXP", "max_guidance_weight": 10.0,
        "ja": "現行の再現（組 E と同じ。configs/default.yaml 155〜157 行）",
        "why": "段階 3 の組 E（V3S3/E_nat、25/99）を作った設定。崩れた基準として回す（候補にしない）"},
    "paper_formula_range44_cap5": {
        "mode": "rtc", "horizon": 44, "schedule": "EXP", "max_guidance_weight": 5.0,
        "ja": "論文の式どおり（範囲 H − s = 44・上限 β = 5）",
        "why": "arXiv 2506.07339 の式 5 の重み（範囲 H − s、指数の減衰）と既定の β = 5。lerobot の EXP は E = H − s で式 5 と一致する"
               "（tests/test_s4_diag_A.py）。ヤコビアン・段数 10・s/H は変えていない（98_s4_d_rtc.py の冒頭）"},
    "range40_cap5": {
        "mode": "rtc", "horizon": 40, "schedule": "EXP", "max_guidance_weight": 5.0,
        "ja": "範囲 40・上限 5", "why": "現行から上限だけを論文の既定 5 にする（W\\rtc_skeptic\\s2_out.json: 最終段の引き戻し 0.911 → 0.50）"},
    "range10_cap5": {
        "mode": "rtc", "horizon": 10, "schedule": "EXP", "max_guidance_weight": 5.0,
        "ja": "範囲 10（= s + d）・上限 5", "why": "W\\rtc の F1（範囲 = s + d）。線形の模型で最新の観測の割合 0.869（a6_toy.json）"},
    "ZEROS": {
        "mode": "rtc", "horizon": 44, "schedule": "ZEROS", "max_guidance_weight": 5.0,
        "ja": "ZEROS（重みは前の d 行だけ 1、ほかは 0。範囲 44・上限 5）",
        "why": "W\\rtc の F1（重みを前の d 行だけにする）。実行する行（d 行目から）には直接の引き戻しがかからない。範囲は効かない"
               "（重みは min(d, 範囲) 行だけ）が、論文の式どおりの設定と減衰の形だけが違うように 44・上限 5 にそろえた（実装役の決定、回す前に掲示）"},
    "naive": {
        "mode": "naive", "horizon": None, "schedule": None, "max_guidance_weight": None,
        "ja": "naive（RTC なし。物差し K・組 A と同じ実行系）", "why": "基準（R.c1・R.c3 の比べる相手）"},
}
CANDIDATES = ("paper_formula_range44_cap5", "range40_cap5", "range10_cap5", "ZEROS")   # gates.R.candidates の宣言順


def setting(name: str) -> dict:
    if name not in SETTINGS:
        raise KeyError(f"設定 {name!r} は {sorted(SETTINGS)} のどれか")
    return dict(SETTINGS[name], name=name)


def overlay_rtc(cfg: dict, name: str) -> dict:
    """configs を読んだ辞書の写しに、設定の RTC の値を重ねる（naive は重ねない）。ファイルは書き換えない。"""
    s = setting(name)
    c = copy.deepcopy(cfg)
    if s["mode"] == "rtc":
        c["runtime"]["rtc_guidance_horizon"] = int(s["horizon"])
        c["runtime"]["rtc_schedule"] = str(s["schedule"])
        c["runtime"]["rtc_max_guidance_weight"] = float(s["max_guidance_weight"])
    return c


def runtime_values(cfg: dict, name: str) -> dict:
    """記録に残す「実行系の何の値に当たるか」。cfg は overlay_rtc の後のもの。"""
    s = setting(name)
    rt = cfg["runtime"]
    out = {"setting": name, "mode": s["mode"], "exec_interval_s_rows": EXEC_S, "delay_steps_init": int(rt["delay_steps"]),
           "inference_delay_rule": "min(d_est, s - 1)、d_est は直前の推論で実際にかかった行の数（runtime/runner.py 255〜259 行）",
           "safety_filter": False}
    if s["mode"] == "rtc":
        out.update({"rtc_guidance_horizon": int(rt["rtc_guidance_horizon"]), "rtc_schedule": rt["rtc_schedule"],
                    "rtc_max_guidance_weight": float(rt["rtc_max_guidance_weight"]),
                    "left_over_rows": f"normalize_left_over(前の塊の残り, {int(rt['rtc_guidance_horizon'])})",
                    "jacobian": "identity（lerobot 0.6.1 modeling_rtc.py 212〜219 行。VJP なし）", "num_steps": 10})
    return out


# ------------------------------------------------------------------ 重み（掲示と単体検査用。lerobot の get_prefix_weights を numpy で）
def prefix_weights(schedule: str, start: int, end: int, total: int = CHUNK_H) -> np.ndarray:
    """lerobot 0.6.1 RTCProcessor.get_prefix_weights と同じ値（start = inference_delay、end = 範囲 E、total = H）。"""
    start = min(int(start), int(end))
    if schedule == "ZEROS":
        w = np.zeros(total)
        w[:start] = 1.0
        return w
    if schedule == "ONES":
        w = np.ones(total)
        w[end:] = 0.0
        return w
    n = total - max(total - end, 0) - start
    lin = np.linspace(1, 0, n + 2)[1:-1] if (end > start and n > 0) else np.zeros(0)
    if schedule == "EXP":
        lin = lin * np.expm1(lin) / (math.e - 1)
    elif schedule != "LINEAR":
        raise ValueError(schedule)
    w = np.concatenate([lin, np.zeros(max(total - end, 0))])
    w = np.concatenate([np.ones(min(start, total)), w])
    return w[:total]


def paper_weights(d: int, H: int = CHUNK_H, s: int = EXEC_S) -> np.ndarray:
    """arXiv 2506.07339 の式 5（soft masking）: i < d は 1、d <= i < H − s は c_i (e^{c_i} − 1)/(e − 1)、
    c_i = (H − s − i)/(H − s − d + 1)、i >= H − s は 0。"""
    w = np.zeros(H)
    for i in range(H):
        if i < d:
            w[i] = 1.0
        elif i < H - s:
            c = (H - s - i) / (H - s - d + 1)
            w[i] = c * math.expm1(c) / (math.e - 1)
    return w


# ------------------------------------------------------------------ 影の推論（実行は誘導した側、影は記録だけ）
class DiagPolicy:
    """SensorPolicy を包む。infer のたびに、誘導した塊（実際に使う）を記録し、shadow が真で RTC の引数があるときは、
    同じ観測・同じ雑音（雑音の生成器の状態を戻して引き直す）で RTC の引数を渡さない推論（影）を足して記録する。
    影は実行に使わない: 戻り値・last_raw・last_cue は誘導した側のまま（影の推論は誘導した推論の後に行う）。
    計算の時間はシミュレーションの上では分布から引く（harness/robot_io.py）ので、影を足しても世界の進みは変わらない。"""

    def __init__(self, inner, shadow: bool = False):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "shadow", bool(shadow))
        object.__setattr__(self, "records", [])

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __setattr__(self, name, value):
        setattr(self._inner, name, value)

    def infer(self, obs, noise_generator, rtc_kwargs=None):
        inner = self._inner
        state = noise_generator.get_state().clone()
        post = inner.infer(obs, noise_generator, rtc_kwargs)
        raw = np.array(inner.last_raw, copy=True)
        rec = {"has_rtc": rtc_kwargs is not None,
               "inference_delay": -1 if rtc_kwargs is None else int(rtc_kwargs["inference_delay"]),
               "left_over_rows": 0 if rtc_kwargs is None else int(np.asarray(rtc_kwargs["prev_chunk_left_over"]).shape[0]),
               "guided_post": np.asarray(post, np.float32)[:, :7].copy(), "guided_raw": raw[:, :7].astype(np.float32),
               "shadow_post": None, "shadow_raw": None}
        if self.shadow and rtc_kwargs is not None:
            noise_generator.set_state(state)
            sh = inner.infer(obs, noise_generator, None)
            rec["shadow_post"] = np.asarray(sh, np.float32)[:, :7].copy()
            rec["shadow_raw"] = np.array(inner.last_raw, copy=True)[:, :7].astype(np.float32)
            inner.last_raw = raw                       # 次の推論の「前の塊の残り」は誘導した塊から（実行と同じ）
        self.records.append(rec)
        return post

    def arrays(self) -> dict:
        """diag_NNNN.npz の中身。推論の順（= runtime の inference[] の順）。影のない行は NaN。"""
        n = len(self.records)
        H = self.records[0]["guided_post"].shape[0] if n else CHUNK_H
        nan = np.full((H, 7), np.nan, np.float32)
        return {"inf_index": np.arange(n, dtype=np.int32),
                "has_rtc": np.array([r["has_rtc"] for r in self.records], bool).reshape(n),
                "inference_delay": np.array([r["inference_delay"] for r in self.records], np.int16).reshape(n),
                "left_over_rows": np.array([r["left_over_rows"] for r in self.records], np.int16).reshape(n),
                "guided_post": np.array([r["guided_post"] for r in self.records], np.float32).reshape(n, H, 7),
                "guided_raw": np.array([r["guided_raw"] for r in self.records], np.float32).reshape(n, H, 7),
                "has_shadow": np.array([r["shadow_post"] is not None for r in self.records], bool).reshape(n),
                "shadow_post": np.array([r["shadow_post"] if r["shadow_post"] is not None else nan for r in self.records],
                                        np.float32).reshape(n, H, 7),
                "shadow_raw": np.array([r["shadow_raw"] if r["shadow_raw"] is not None else nan for r in self.records],
                                       np.float32).reshape(n, H, 7)}


# ------------------------------------------------------------------ 記録の読み取り
def load_record(json_path) -> tuple:
    """(meta, npz の辞書, runtime の辞書, diag の辞書 or None)。96_s4_resume.py run と同じ形の記録。"""
    p = pathlib.Path(json_path)
    meta = json.loads(p.read_text(encoding="utf-8"))
    n = int(p.stem.split("_")[1])
    with np.load(p.with_suffix(".npz"), allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    rt = json.loads((p.parent / f"runtime_{n:04d}.json").read_text(encoding="utf-8"))["runtime"]
    dp = p.parent / f"diag_{n:04d}.npz"
    diag = None
    if dp.is_file():
        with np.load(dp, allow_pickle=False) as d:
            diag = {k: d[k] for k in d.files}
    return meta, arrays, rt, diag


def _first(mask) -> int | None:
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else None


def _actions(rt: dict):
    acts = rt["actions"]
    A = np.array([a["a"] for a in acts], float).reshape(-1, 7)
    kt = np.array([a["t"] for a in acts], float)
    held = np.array([a["held"] for a in acts], bool)
    ch = np.array([(-1 if a["chunk"] is None else a["chunk"]) for a in acts])
    return A, kt, held, ch


def seam_jump_mean(A, kt, held, ch, horizon_s: float = None) -> float:
    """続く 2 行がどちらも保持でなく、塊の番号が変わる所の |Δv| の平均（a1_trial_metrics.py）。horizon_s があれば両方の行の時刻 < horizon_s。"""
    if len(A) < 2:
        return math.nan
    v = A[:, :3] / ACTION_DT
    jump = np.linalg.norm(np.diff(v, axis=0), axis=1)
    seam = ~held[1:] & ~held[:-1] & (ch[1:] != ch[:-1])
    if horizon_s is not None:
        seam &= kt[1:] < horizon_s - 1e-9
    return float(jump[seam].mean()) if seam.any() else math.nan


def trial_metrics(meta: dict, z: dict, rt: dict, horizon_s: float = None, ats=(30.0, 45.0, 60.0)) -> dict:
    """1 試行の関門 R の指標。horizon_s は区間を集める指標（移動の比・継ぎ目の跳び）を限る時刻（None は記録全体）。"""
    ti = COLORS.index(str(meta["target"]).split(">")[0])
    t, tip = z["sim_time"], z["fingertip"]
    cube = z["cube_pos"][:, ti]
    closed = z["gripper_closed"].astype(bool)
    ts = meta.get("t_success") if meta.get("success") else None
    row = {"trial": meta.get("trial"), "seed": meta.get("seed"), "target": meta.get("target"), "success": bool(meta.get("success")),
           "t_success": ts, "success_at": {f"{L:g}": bool(ts is not None and ts <= L + 1e-9) for L in ats},
           "horizon_s": horizon_s}
    A, kt, held, ch = _actions(rt)
    row["seam_jump_mps"] = seam_jump_mean(A, kt, held, ch, horizon_s)
    k1 = _first(closed)
    row["t_first_close"] = float(t[k1]) if k1 is not None else None
    row.update(radial_gap_mm=None, move_ratio=None, first_close_after_horizon=None, descent_red_mm=None, descent_start_mm=None,
               far_whiff_s=None)
    if k1 is None:
        return row
    after = horizon_s is not None and t[k1] > horizon_s + 1e-9
    row["first_close_after_horizon"] = bool(after)
    u = cube[k1, :2] / np.linalg.norm(cube[k1, :2])                 # 根元（原点）→ 立方体
    row["radial_gap_mm"] = float((tip[k1, :2] - cube[k1, :2]) @ u * 1e3)  # 最初の閉じで決まる（時間に依存しない）
    if not after:
        use = (kt < t[k1]) & ~held
        cmd = A[use, :2].sum(axis=0) @ u
        need = (cube[k1, :2] - tip[0, :2]) @ u
        row["move_ratio"] = float(cmd / need) if abs(need) > NEED_MIN_M else None
    # 報告だけ（r1_gate_noise.py の red・far）
    dxy = np.hypot(*(tip[:, :2] - cube[:, :2]).T)
    hz = tip[:k1 + 1, 2] - cube[:k1 + 1, 2]
    idx = np.flatnonzero(hz >= 0.13)
    if idx.size:
        row["descent_start_mm"] = float(dxy[idx[-1]] * 1e3)
        row["descent_red_mm"] = float((dxy[idx[-1]] - dxy[k1]) * 1e3)
    rise = cube[:, 2] - cube[0, 2]
    edges = np.flatnonzero(np.diff(np.r_[0, closed.astype(int), 0]))
    for s, e in zip(edges[::2], edges[1::2]):
        if not np.any(rise[s:e] >= 0.02):
            if dxy[s] * 1e3 > 25:
                row["far_whiff_s"] = float(t[min(e, len(t) - 1)] - t[s])
            break
    return row


def _med(xs):
    x = np.array([v for v in xs if v is not None and np.isfinite(v)], float)
    return (None if not x.size else float(np.median(x))), int(x.size)


def _mean(xs):
    x = np.array([v for v in xs if v is not None and np.isfinite(v)], float)
    return (None if not x.size else float(np.mean(x))), int(x.size)


def summarize(rows: list, ats=(30.0, 45.0, 60.0)) -> dict:
    """条件の要約。値は [値, 件数]。"""
    rad, n_rad = _med([r["radial_gap_mm"] for r in rows])
    mr_med, n_mr = _med([r["move_ratio"] for r in rows])
    mr_mean, _ = _mean([r["move_ratio"] for r in rows])
    sj, n_sj = _med([r["seam_jump_mps"] for r in rows])
    red, n_red = _med([r["descent_red_mm"] for r in rows if r["descent_start_mm"] is not None and r["descent_start_mm"] > 25])
    far, n_far = _med([r["far_whiff_s"] for r in rows])
    return {"n_trials": len(rows), "horizon_s": rows[0]["horizon_s"] if rows else None,
            "successes_at": {f"{L:g}": sum(r["success_at"][f"{L:g}"] for r in rows) for L in ats},
            "radial_gap_mm_median": [rad, n_rad],
            "move_ratio_median": [mr_med, n_mr], "move_ratio_mean": [mr_mean, n_mr],
            "seam_jump_mps_median": [sj, n_sj],
            "n_no_close": sum(1 for r in rows if r["t_first_close"] is None),
            "n_first_close_after_horizon": sum(1 for r in rows if r["first_close_after_horizon"]),
            "report_only": {"descent_red_mm_median(dstart>25mm)": [red, n_red], "far_whiff_s_median(>25mm)": [far, n_far]}}


def condition_metrics(d, horizons=(30.0, None)) -> dict:
    """条件のフォルダ（trial_NNNN.json/.npz・runtime_NNNN.json）から、horizon ごとの試行の指標と要約。None は記録全体（60 s）。"""
    d = pathlib.Path(d)
    paths = sorted(d.glob("trial_[0-9][0-9][0-9][0-9].json"))
    recs = [load_record(p)[:3] for p in paths]
    out = {"dir": str(d), "n_trials": len(paths), "by_horizon": {}}
    for h in horizons:
        rows = [trial_metrics(m, z, rt, h) for m, z, rt in recs]
        out["by_horizon"]["all" if h is None else f"{h:g}"] = {"summary": summarize(rows), "trials": rows}
    return out


# ------------------------------------------------------------------ 影の推論の指標
def _dir(cube_xy, xd_xy):
    u = np.asarray(cube_xy, float) - np.asarray(xd_xy, float)
    n = float(np.linalg.norm(u))
    return (u / n) if n >= DIR_MIN_M else None


def shadow_trial(meta: dict, z: dict, rt: dict, diag: dict, rows=ROWS, horizon_s: float = 30.0) -> dict:
    """影の推論の 1 試行。{h: [(s − g)·u mm ...]} と、実行した行（offset から s 行）の差。"""
    inf = rt["inference"]
    n = len(diag["has_rtc"])
    if n != len(inf):
        raise ValueError(f"diag の推論 {n} 本と runtime の inference {len(inf)} 本が合わない")
    ti = COLORS.index(str(meta["target"]).split(">")[0])
    t, xd = z["sim_time"], z["x_des"]
    cube = z["cube_pos"][:, ti]
    k1 = _first(z["gripper_closed"].astype(bool))
    t_close = float(t[k1]) if k1 is not None else math.inf
    out = {"h": {str(h): [] for h in rows}, "exec_rows": [], "n_used": 0, "n_shadow": int(diag["has_shadow"].sum())}
    for i in range(n):
        if not diag["has_shadow"][i]:
            continue
        t_obs = float(inf[i]["t_obs"])
        if t_obs >= t_close or (horizon_s is not None and t_obs > horizon_s):
            continue
        f = min(int(np.searchsorted(t, t_obs - 1e-9)), len(t) - 1)
        u = _dir(cube[f, :2], xd[f, :2])
        if u is None:
            continue
        g, s_ = diag["guided_post"][i], diag["shadow_post"][i]
        for h in rows:
            out["h"][str(h)].append(float((s_[:h, :2].sum(0) - g[:h, :2].sum(0)) @ u * 1e3))
        o = inf[i].get("offset")
        if o is not None:
            o = int(o)
            out["exec_rows"].append(float((s_[o:o + EXEC_S, :2].sum(0) - g[o:o + EXEC_S, :2].sum(0)) @ u * 1e3))
        out["n_used"] += 1
    return out


def shadow_summary(trials: list, rows=ROWS) -> dict:
    med = {str(h): _med([v for tr in trials for v in tr["h"][str(h)]]) for h in rows}
    ex = _med([v for tr in trials for v in tr["exec_rows"]])
    return {"shadow_minus_guided_mm_median": {h: list(v) for h, v in med.items()},
            "exec_rows_shadow_minus_guided_mm_median": list(ex),
            "plan_shorter_ge_15mm": any(v[0] is not None and v[0] >= SHORT_MM for v in med.values()),
            "rule": "どれかの h の中央値 >= 15 mm（R.b1 の影の側）", "n_inferences": sum(tr["n_used"] for tr in trials)}


# ------------------------------------------------------------------ X2（エキスパートの記録に開ループで当てる）
def x2_episode_shortfall(pred_post: np.ndarray, k_index: np.ndarray, x_des_raw: np.ndarray, cube_xy_raw: np.ndarray,
                         f_close: int | None, rows=ROWS, stride: int = 2) -> dict:
    """pred_post[j] は 10 fps のこま k_index[j]（生のこま stride·k）の観測で推論した塊（後処理の後）。
    返り値 {h: [(e_h − p_h)·u mm ...]}（正は予測が短い）。k + h <= f_close // stride の所だけ。"""
    k_close = (len(x_des_raw) - 1) // stride if f_close is None else int(f_close) // stride
    out = {str(h): [] for h in rows}
    for j, k in enumerate(np.asarray(k_index, int)):
        f = stride * k
        u = _dir(cube_xy_raw[f], x_des_raw[f, :2])
        if u is None:
            continue
        for h in rows:
            if k + h > k_close or stride * (k + h) >= len(x_des_raw):
                continue
            e = x_des_raw[stride * (k + h), :2] - x_des_raw[f, :2]
            p = np.asarray(pred_post[j][:h, :2], float).sum(0)
            out[str(h)].append(float((e - p) @ u * 1e3))
    return out


def x2_summary(episodes: list, rows=ROWS) -> dict:
    med = {str(h): _med([v for ep in episodes for v in ep[str(h)]]) for h in rows}
    return {"expert_minus_pred_mm_median": {h: list(v) for h, v in med.items()},
            "shortfall_ge_15mm": any(v[0] is not None and v[0] >= SHORT_MM for v in med.values()),
            "rule": "どれかの h の中央値 >= 15 mm（R.b1 の X2 の側）"}
