"""段階 4 束 1 の D-復帰（担当 C）: 落下・置き損ね・把持失敗からの復帰の診断の部品。関門 C の材料を作る（判定はしない）。

    from recovla.diag import recovery as DR
    ind = DR.make_inducer("fall_with_hold", seed, layout, target, rig)   # 落下で手を止める版（診断専用）
    ind.bind_runtime(rt)                                                 # 塊の観測の時刻を読むため（評価の側から実行系を読むだけ）
    DR.fall_timing(meta, arrays, runtime)                                # 着地の時刻と、手を動かした最初の塊の観測の時刻
    DR.gate_c_material(recs_R, recs_N, L=30)                             # 関門 C の指標（R の L 秒の復帰、R−N、成立した対の数）

使うところ: scripts/98_s4_d_recovery.py（試行を回す・集計する）、tests/test_s4_diag_C.py（CPU だけ）。

4 通り（configs/s4_gates.json の gates.C.runs.variants と同じ名前。docs/目標書_段階4.md 第 8 節の関門 C）:
| 名前 | 誘発 | 版 | 中身 |
|---|---|---|---|
| fall_as_is | P2 | stage3 | 段階 3 と同じ落下（recovla.eval.induce.Inducer のまま。落とした後も手先は方策のまま動く） |
| fall_with_hold | P2 | P2H-v1 | 落下で手を止める版（下の FallHoldInducer）。**診断にだけ使い、テストでは使わない**（目標書_段階4.md 第 10 節 6、s4_gates の C.b2） |
| misplace | P3 | stage3 | 段階 3 と同じ置き損ね（Inducer のまま） |
| grasp_failure | P1 | stage3 | 段階 3 と同じ把持失敗（Inducer のまま） |

落下で手を止める版（P2H-v1。結果を見る前に決めた。10/08）:
- 発動・パラメータは段階 3 の P2 と同じ（同じ乱数列から同じ順で引くので、同じ種なら u・r0 の決め方も同じ）。
- 指を開いた時刻（t_fire）から、手先の指令を止め（行動の xyz を 0）、指を開いたまま保つ（データの注入 B と同じ＝expert/inject.py 16〜18 行）。
- 着地の時刻 t_land = 開いた後で、目標の立方体の高さが静止の高さ +0.5 cm 未満になった最初の判定の時刻（10 Hz。成立の判定と同じ
  評価の枠の呼び出し）。定義は懐疑役の検算 v1（W\\skeptic_lang_recovery\\v1_p2_stale.py の t_land）と同じしきい値。
- 手を放す（方策に戻す）のは、成立の判定が済み（段階 3 の P2 と同じ着地の検査。stage が done）、かつ、その区切りで行動を出している
  塊が「着地の後の観測から作った塊」になった最初の区切り。塊の観測の時刻は、その推論に入った画像の撮影時刻の早いほう
  （runtime の inference[i].img_t の最小。無ければ t_obs）。画像の撮影時刻 ≥ t_land なら t_obs ≥ t_land も満たす（厳しいほう）。
- 着地しないまま判定が済んだ（他の立方体に乗った・静止しない）ときは、そこで放す（成立しないので分母に入らない）。
- 塊の観測の時刻を読むために、評価の側（誘発）が実行系（PolicyRuntime の active・log_inf）を読む。実行系から誘発へは届かない
  （HarnessHook は G1 の到達検査の境界。harness/loop.py 28〜35 行）ので、G1 は変わらない。
- 記録: meta["induce"] に "variant"・"version" を足し、info["hold"] に t_drop・t_land・t_release・放した区切りの塊の番号・t_obs・
  画像の撮影時刻の最小・放した理由を残す。kind は "P2" のまま（87・50_e_eval・report・time_scoring が P2 として読む）。

関門 C の指標（docs/目標書_段階4.md 8-1・8-2、s4_gates の gates.C。判定はしない＝数値を出すだけ）:
- R の L 秒の復帰 = R の試行のうち、誘発が L 秒より前に成立した試行（induce.t_established < L。ちょうど L 秒の成立は入れない。
  掲示板 0155 の 1-1）の中で、L 秒までに成功した（t_success <= L）割合。
- R−N（L 秒）= R と N の両方で L 秒より前に成立した種の対（鍵は種と目標の色）で、R の L 秒の成功率 − N の L 秒の成功率
  （= (R だけ成功 − N だけ成功) / 対の数）。
- 成立した対の数 = 上の対の数。
- L = 30 s が主、60 s が副、45 s は記述だけ。
- 実装は time_scoring.compare_conditions と独立に書いた（3-6「集計は独立な 2 つの実装で一致」の片方）。cross_check で突き合わせる。
  cross_check_dirs は、ファイルの列挙もそれぞれ独立に行う（こちらは load_condition の glob、time_scoring は load_trials の
  正規表現。査読の重要 1 の最後）。列挙した試行の番号の集合が食い違っても一致にしない。
"""
import json
import pathlib

import numpy as np

from recovla.eval.induce import Inducer
from recovla.sim import frames

VARIANTS = {
    "fall_as_is":     {"induce": "P2", "version": "stage3", "hold": False, "ja": "落下そのまま（段階 3 の P2）"},
    "fall_with_hold": {"induce": "P2", "version": "P2H-v1", "hold": True,
                       "ja": "落下で手を止める版（着地の後の観測から作った塊が動き始めるまで手を止めて指を開いたまま。診断専用）"},
    "misplace":       {"induce": "P3", "version": "stage3", "hold": False, "ja": "置き損ね（段階 3 の P3）"},
    "grasp_failure":  {"induce": "P1", "version": "stage3", "hold": False, "ja": "把持失敗（段階 3 の P1）"},
}
VARIANT_ORDER = ("fall_as_is", "fall_with_hold", "misplace", "grasp_failure")
LAND_DZ_M = 0.005              # 着地: 目標の高さが静止の高さ +0.5 cm 未満（懐疑役の検算 v1 と同じ）
EPS = 1e-9                     # harness/loop.py・time_scoring と同じ比較の幅
SCORE_AT = (30.0, 45.0, 60.0)  # 主 30 s、副 60 s、45 s は記述だけ（目標書_段階4.md 第 3-1 節）


# ---------------------------------------------------------------------- 誘発
class FallHoldInducer(Inducer):
    """落下で手を止める版（P2H-v1）。診断にだけ使い、テストでは使わない。中身はこのモジュールの冒頭。"""
    VARIANT = "fall_with_hold"
    VERSION = "P2H-v1"

    def __init__(self, seed: int, layout, target: str, rig, cfg: dict = None):
        super().__init__("P2", seed, layout, target, rig, cfg)      # 発動・パラメータは段階 3 の P2 と同じ
        self.rt = None
        self.t_land = None
        self.released = False
        self.hold = {"rule": "発動から、着地の後の観測（画像の撮影時刻 >= t_land）から作った塊が行動を出すまで、"
                             "手先の指令 0・指を開いたまま。着地しないまま判定が済んだらそこで放す",
                     "t_drop": None, "t_land": None, "t_release": None, "k_release": None, "release_reason": None,
                     "release_chunk": None, "release_chunk_t_obs": None, "release_chunk_img_t_min": None, "hold_s": None}

    def bind_runtime(self, rt) -> None:
        """塊の観測の時刻を読むために実行系をつなぐ（評価の側から読むだけ。98 の Engine が make_runtime の直後に呼ぶ）。"""
        self.rt = rt

    def _chunk_now(self, k: int):
        """この区切り k で行動を出している塊の推論の記録（PolicyRuntime.log_inf の 1 件）。塊がない（保持）なら None。
        runner._action_boundary の 3) と同じ行の決め方（row = o0 + (k − k0)、塊の長さを超えたら保持）。"""
        if self.rt is None:
            raise RuntimeError("FallHoldInducer: bind_runtime が呼ばれていない（塊の観測の時刻が読めない）")
        act = self.rt.active
        if act is None:
            return None
        row = int(act["o0"]) + (int(k) - int(act["k0"]))
        if row >= len(act["post"]):
            return None
        i = act["i"]
        return next((e for e in reversed(self.rt.log_inf) if e.get("i") == i), None)

    @staticmethod
    def chunk_obs_time(e: dict) -> float:
        img = e.get("img_t") or {}
        return float(min(img.values())) if img else float(e["t_obs"])

    def _ready_to_release(self, k: int, t: float) -> bool:
        if self.t_land is None:                                   # 着地しないまま判定が済んだ（成立しない）
            self.hold["release_reason"] = "no_landing"
            return True
        e = self._chunk_now(k)
        if e is None or self.chunk_obs_time(e) < self.t_land - EPS:
            return False
        self.hold.update(release_reason="fresh_chunk", release_chunk=int(e["i"]),
                         release_chunk_t_obs=round(float(e["t_obs"]), 4),
                         release_chunk_img_t_min=round(self.chunk_obs_time(e), 4))
        return True

    def _p2(self, k, a, tr):
        a = super()._p2(k, a, tr)                                 # 発動の判定と、落ちてから判定までの「開いたまま」は段階 3 と同じ
        if not self.fired or self.released:
            return a
        if self.hold["t_drop"] is None:
            self.hold["t_drop"] = self.t_fire
        if self.stage == "done" and self._ready_to_release(k, tr.t):
            self.released = True
            self.hold.update(t_release=round(float(tr.t), 3), k_release=int(k),
                             hold_s=round(float(tr.t) - float(self.t_fire), 3))
            self.active = False
            return a                                              # 方策の行動のまま
        self.active = True                                        # 手を止めて、指を開いたまま
        a[:3] = 0.0
        a[6] = -1.0
        return a

    def after(self, k: int, tr) -> None:
        if self.fired and self.t_land is None and float(tr.target_pos[2]) - frames.CUBE_REST_Z < LAND_DZ_M:
            self.t_land = round(float(tr.t), 3)
            self.hold["t_land"] = self.t_land
        super().after(k, tr)

    @property
    def releasing(self) -> bool:
        return self.fired and not self.released

    def record(self) -> dict:
        r = super().record()
        r["info"] = dict(r["info"], hold=dict(self.hold))
        r["variant"], r["version"] = self.VARIANT, self.VERSION
        return r


def make_inducer(variant: str, seed: int, layout, target: str, rig, cfg: dict = None):
    """4 通りの誘発。fall_with_hold だけが新しい版で、ほかは段階 3 の Inducer のまま。"""
    v = VARIANTS[variant]
    if v["hold"]:
        return FallHoldInducer(seed, layout, target, rig, cfg)
    return Inducer(v["induce"], seed, layout, target, rig, cfg)


# ---------------------------------------------------------------------- 落下の時刻（記録から。読むだけ）
def _chunk_class(t_obs, t_drop, t_land) -> str:
    if t_obs is None:
        return None
    if t_obs < t_drop - EPS:
        return "pre_drop"
    return "falling" if (t_land is None or t_obs < t_land - EPS) else "post_land"


def fall_timing(meta: dict, arrays, runtime: dict) -> dict:
    """落下（P2）の試行 1 本の、着地の時刻と、成立の後に手を動かした最初の塊の観測の時刻（真値の記録から。読むだけ）。
    runtime は runtime_NNNN.json の "runtime"（inference・actions）。arrays は trial_NNNN.npz（sim_time・cube_pos・target・
    chunk_id・fingertip）。定義:
      t_drop      induce.t_fire
      t_land      t_drop の後で、目標の高さが静止の高さ +0.5 cm 未満になった最初のこま（20 Hz。v1 と同じ）
      v1          成立の時刻の後で最初に塊の番号があるこまの塊（懐疑役の検算 v1 と同じ定義。段階 3 の 34/38）
      moved       成立の時刻の後で最初の「手を止めた形でない」行動（xyz が全部 0 かつ指が開、ではない）を出した塊。
                  手を止める版では止めていた間の行動を除けるので、方策が実際に手を動かした最初の塊になる
    各塊の観測の時刻は inference[i].t_obs と、画像の撮影時刻の最小（img_t）。分類は pre_drop・falling・post_land。"""
    ind = meta.get("induce") or {}
    out = {"seed": meta.get("seed"), "trial": meta.get("trial"), "variant": ind.get("variant") or "stage3",
           "established": bool(ind.get("established")), "kind": ind.get("kind")}
    if ind.get("kind") != "P2" or not ind.get("fired"):
        return dict(out, applicable=False)
    t = np.asarray(arrays["sim_time"], float)
    tgt = int(np.asarray(arrays["target"])[0])
    cz = np.asarray(arrays["cube_pos"], float)[:, tgt, 2]
    ft = np.asarray(arrays["fingertip"], float)
    t_drop = float(ind["t_fire"])
    m = (t >= t_drop - EPS) & (cz - frames.CUBE_REST_Z < LAND_DZ_M)
    t_land = float(t[np.argmax(m)]) if m.any() else None
    te = ind.get("t_established")
    out.update(applicable=True, t_drop=t_drop, t_land=t_land, t_established=te)
    hold = (ind.get("info") or {}).get("hold")
    if hold:
        out["hold_record"] = {k: hold.get(k) for k in ("t_land", "t_release", "release_reason", "release_chunk",
                                                       "release_chunk_t_obs", "release_chunk_img_t_min", "hold_s")}
    if te is None:
        return out
    te = float(te)
    inf = {int(e["i"]): e for e in runtime.get("inference", [])}

    def describe(i):
        e = inf.get(int(i)) if i is not None else None
        if e is None:
            return None
        img = e.get("img_t") or {}
        t_obs = float(e["t_obs"])
        t_img = float(min(img.values())) if img else t_obs
        return {"chunk": int(i), "t_obs": round(t_obs, 4), "img_t_min": round(t_img, 4),
                "t_act": e.get("t_act"), "class_t_obs": _chunk_class(t_obs, t_drop, t_land),
                "class_img": _chunk_class(t_img, t_drop, t_land)}

    cid = np.asarray(arrays["chunk_id"])
    j = int(np.argmax(t >= te - EPS)) if (t >= te - EPS).any() else len(t)
    while j < len(t) and cid[j] < 0:
        j += 1
    out["v1"] = describe(int(cid[j])) if j < len(t) else None
    mv = None
    for ac in runtime.get("actions", []):
        if float(ac["t"]) < te - EPS or ac.get("chunk") is None or ac.get("held"):
            continue
        a = ac["a"]
        if all(float(x) == 0.0 for x in a[:3]) and float(a[6]) < 0.0:
            continue                                              # 手を止めた形（止める版の上書き）
        mv = ac
        break
    if mv is not None:
        d = describe(mv["chunk"])
        if d is not None:
            d["t_first_move"] = round(float(mv["t"]), 4)
            k0 = int(np.argmax(t >= t_drop - EPS))
            k1 = int(np.argmax(t >= float(mv["t"]) - EPS)) if (t >= float(mv["t"]) - EPS).any() else len(t) - 1
            d["fingertip_xy_move_cm_drop_to_move"] = round(float(np.hypot(*(ft[k1, :2] - ft[k0, :2]))) * 100, 2)
        out["moved"] = d
    else:
        out["moved"] = None
    return out


def summarize_fall_timing(rows: list) -> dict:
    """成立した落下の試行について、v1・moved の塊の分類の数（t_obs と画像の撮影時刻の 2 通り）。"""
    est = [r for r in rows if r.get("applicable") and r.get("established")]
    out = {"n_established": len(est)}
    for key in ("v1", "moved"):
        for cls in ("class_t_obs", "class_img"):
            c = {}
            for r in est:
                d = r.get(key)
                lab = d.get(cls) if d else None
                c[str(lab)] = c.get(str(lab), 0) + 1
            out[f"{key}_{cls}"] = c
    mv = [r["moved"]["fingertip_xy_move_cm_drop_to_move"] for r in est if r.get("moved")]
    if mv:
        out["fingertip_xy_move_cm_drop_to_move_p10_50_90"] = [round(float(np.percentile(mv, q)), 2) for q in (10, 50, 90)]
    hs = [r["hold_record"]["hold_s"] for r in est if r.get("hold_record") and r["hold_record"].get("hold_s") is not None]
    if hs:
        out["hold_s_p10_50_90"] = [round(float(np.percentile(hs, q)), 2) for q in (10, 50, 90)]
    return out


# ---------------------------------------------------------------------- 関門 C の指標（判定はしない）
def _est_by(rec: dict, L: float) -> bool:
    """誘発が L 秒より前に成立したか（induce.t_established < L。ちょうど L 秒の成立は入れない。掲示板 0155 の 1-1。段階 3 は
    30 s で打ち切ったので成立は 29.9 s 以前だけだった）。成立の時刻が無い成立は、回した時間が L 以下なら真。"""
    ind = rec.get("induce") or {}
    if not (ind.get("kind") and ind.get("established")):
        return False
    te = ind.get("t_established")
    if te is None:
        if float(L) >= float(rec["time_limit_s"]) - EPS:
            return True
        raise ValueError(f"seed {rec.get('seed')}: 成立の時刻がなく、{L:g} s より前に成立したか分からない")
    return float(te) < float(L) - EPS


def _succ_by(rec: dict, L: float) -> bool:
    if float(rec["time_limit_s"]) < float(L) - EPS:
        raise ValueError(f"seed {rec.get('seed')}: {rec['time_limit_s']} s で回した試行を {L:g} s で採点できない")
    t = rec.get("t_success")
    return bool(rec.get("success")) and t is not None and float(t) <= float(L) + EPS


def recovery_at(recs: list, L: float) -> dict:
    """1 条件の L 秒の復帰（分母は L 秒以内に成立した試行）。"""
    den = [r for r in recs if _est_by(r, L)]
    k = sum(_succ_by(r, L) for r in den)
    return {"L_s": float(L), "recovered": k, "established": len(den), "n_trials": len(recs),
            "rate": (k / len(den)) if den else None}


def paired_at(recs_r: list, recs_n: list, L: float) -> dict:
    """R と N の、両方で L 秒以内に成立した種の対（鍵は種と目標の色）での L 秒の成否。"""
    br = {(r["seed"], r.get("target")): r for r in recs_r}
    bn = {(r["seed"], r.get("target")): r for r in recs_n}
    keys = sorted(k for k in set(br) & set(bn) if _est_by(br[k], L) and _est_by(bn[k], L))
    sr = [_succ_by(br[k], L) for k in keys]
    sn = [_succ_by(bn[k], L) for k in keys]
    both = sum(a and b for a, b in zip(sr, sn))
    r_only = sum(a and not b for a, b in zip(sr, sn))
    n_only = sum(b and not a for a, b in zip(sr, sn))
    n = len(keys)
    return {"L_s": float(L), "pairs": n, "both": both, "r_only": r_only, "n_only": n_only, "neither": n - both - r_only - n_only,
            "r_rate": (both + r_only) / n if n else None, "n_rate": (both + n_only) / n if n else None,
            "r_minus_n": (r_only - n_only) / n if n else None, "common_keys": len(set(br) & set(bn))}


def gate_c_material(recs_r: list, recs_n: list, ats=SCORE_AT) -> dict:
    """関門 C の数値（判定はしない）。主 30 s、副 60 s、45 s は記述だけ。名前は s4_gates の gates.C の指標に合わせる。"""
    out = {"definition": "recovery = L 秒より前に成立した試行（t_established < L）のうち L 秒までに成功した割合。R−N = R・N の"
                         "両方で L 秒より前に成立した種の対（種と目標の色）での (R だけ成功 − N だけ成功) / 対の数。pairs = その対の数",
           "by_L": {}}
    for L in ats:
        rr, rn, pr = recovery_at(recs_r, L), recovery_at(recs_n, L), paired_at(recs_r, recs_n, L)
        out["by_L"][f"{L:g}"] = {"recovery_R": rr, "recovery_N": rn, "paired": pr}
    p = out["by_L"].get("30")
    if p is not None:
        out["primary_30s"] = {"recovery_R": p["recovery_R"]["rate"], "recovery_R_k_n": [p["recovery_R"]["recovered"],
                                                                                     p["recovery_R"]["established"]],
                              "R_minus_N": p["paired"]["r_minus_n"], "pairs": p["paired"]["pairs"],
                              "r_only": p["paired"]["r_only"], "n_only": p["paired"]["n_only"]}
    s = out["by_L"].get("60")
    if s is not None:
        out["secondary_60s"] = {"recovery_R": s["recovery_R"]["rate"], "R_minus_N": s["paired"]["r_minus_n"],
                                "pairs": s["paired"]["pairs"]}
    return out


def cross_check(recs_r: list, recs_n: list, ats=SCORE_AT, ts_recs: tuple = None) -> dict:
    """もう 1 つの実装（recovla.eval.time_scoring）と件数が完全に一致するか（目標書_段階4.md 第 3 節 6）。
    ts_recs = (R, N) を渡すと、time_scoring の側はそれを数える（cross_check_dirs が別に列挙した記録を渡す）。"""
    from recovla.eval import time_scoring as TS
    ts_r, ts_n = ts_recs if ts_recs is not None else (recs_r, recs_n)
    mine = gate_c_material(recs_r, recs_n, ats)
    cr = TS.condition_scores(ts_r, ats, induced=True)
    cn = TS.condition_scores(ts_n, ats, induced=True)
    cmp = TS.compare_conditions(ts_r, ts_n, ats, induced=True)
    diffs = []
    for L in ats:
        k = f"{L:g}"
        m = mine["by_L"][k]
        pairs = [("recovery_R.recovered", m["recovery_R"]["recovered"], cr["by_limit"][k]["successes"]),
                 ("recovery_R.established", m["recovery_R"]["established"], cr["by_limit"][k]["n"]),
                 ("recovery_N.recovered", m["recovery_N"]["recovered"], cn["by_limit"][k]["successes"]),
                 ("recovery_N.established", m["recovery_N"]["established"], cn["by_limit"][k]["n"]),
                 ("paired.pairs", m["paired"]["pairs"], cmp["by_limit"][k]["pairs"]),
                 ("paired.both", m["paired"]["both"], cmp["by_limit"][k]["both"]),
                 ("paired.r_only", m["paired"]["r_only"], cmp["by_limit"][k]["a_only"]),
                 ("paired.n_only", m["paired"]["n_only"], cmp["by_limit"][k]["b_only"])]
        diffs += [{"L": k, "item": n, "mine": a, "time_scoring": b} for n, a, b in pairs if a != b]
    return {"agree": not diffs, "diffs": diffs}


def cross_check_dirs(dir_r, dir_n, ats=SCORE_AT) -> dict:
    """cross_check を、ファイルの列挙から独立にして当てる（査読の重要 1 の最後・掲示板 0155 の 2-7）。こちらは load_condition
    （glob）、time_scoring は load_trials（iterdir と正規表現）で、それぞれのフォルダを読む。読んだ試行の番号の並びも比べる。"""
    from recovla.eval import time_scoring as TS
    mine_r, mine_n = load_condition(dir_r), load_condition(dir_n)
    ts_r, ts_n = TS.load_trials(dir_r), TS.load_trials(dir_n)
    out = cross_check(mine_r, mine_n, ats, ts_recs=(ts_r, ts_n))
    lists = {"R": ([r.get("trial") for r in mine_r], [r.get("trial") for r in ts_r]),
             "N": ([r.get("trial") for r in mine_n], [r.get("trial") for r in ts_n])}
    enum = [{"side": s, "mine": a, "time_scoring": b} for s, (a, b) in lists.items() if a != b]
    out["enumeration"] = {"R": len(mine_r), "N": len(mine_n), "diffs": enum}
    out["agree"] = bool(out["agree"] and not enum)
    return out


# ---------------------------------------------------------------------- 読み込み
def load_condition(d) -> list:
    """trial_NNNN.json の並び（time_scoring.load_trials と同じ）。"""
    d = pathlib.Path(d)
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("trial_[0-9][0-9][0-9][0-9].json"))]


def fall_timing_dir(d) -> list:
    """条件のフォルダの全試行に fall_timing を当てる（読むだけ）。"""
    d = pathlib.Path(d)
    rows = []
    for p in sorted(d.glob("trial_[0-9][0-9][0-9][0-9].json")):
        meta = json.loads(p.read_text(encoding="utf-8"))
        rt = json.loads(p.with_name(p.name.replace("trial_", "runtime_")).read_text(encoding="utf-8"))
        with np.load(p.with_suffix(".npz")) as z:
            arrays = {k: z[k] for k in ("sim_time", "cube_pos", "target", "chunk_id", "fingertip")}
        rows.append(fall_timing(meta, arrays, rt.get("runtime", rt)))
    return rows
