"""説明資料の補い（10/07 の査読への対応）: 段階 3 の最終評価の記録から、主要評価項目の外の参考値を数える。数字は手で書かない。

    .venv\\Scripts\\python.exe scripts\\55_extra_s3.py          # → outputs/results/extra_s3.json

1. 把持失敗を意図的に起こした試行の、絞り込まない比較: 復帰デモあり 対 なし（非同期実行）の各 50 組で成功を比べる
   （主要評価項目は両モデルとも実際に失敗した組だけを数える。こちらは実際に失敗したかによらない）
2. 意図的な失敗なしの試行で自然に起きた把持失敗: グリッパを初めて閉じてから eval.P1.confirm_window_s 以内に、目標の立方体が
   閉じた時点から eval.P1.confirm_lift_m 以上持ち上がらなかった試行（評価の把持失敗の確定と同じ基準）。その数と、そのうち成功した数
読むもの: outputs/v2eval/V3S3/<組>_<種類>/trial_NNNN.json・.npz（A は復帰デモあり、B はなし、どちらも非同期実行）
"""
import importlib.util
import json
import sys
import time

import numpy as np

from recovla.common import config
from recovla.common.seeds import COLORS

ROOT = config.ROOT
CFG = config.load_v2()
S3 = config.path(CFG["paths"]["outputs"]) / "v2eval" / "V3S3"
OUT = config.path(CFG["paths"]["outputs"]) / "results" / "extra_s3.json"
SETS = {"A": "復帰デモあり（非同期実行）", "B": "復帰デモなし（非同期実行）"}


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def rows(cond: str) -> dict:
    out = {}
    for p in sorted((S3 / cond).glob("trial_*.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        out[(m["seed"], m["target"])] = {"meta": m, "npz": p.with_suffix(".npz")}
    return out


def natural_grasp_failure(r: dict, window_s: float, lift_m: float):
    """最初に閉じた試みで目標が持ち上がらなかったか。閉じなかった試行は None。"""
    a = np.load(r["npz"])
    closed = np.asarray(a["gripper_closed"], bool)
    if not closed.any():
        return None
    t = np.asarray(a["sim_time"], float)
    z = np.asarray(a["cube_pos"], float)[:, COLORS.index(r["meta"]["target"]), 2]
    i0 = int(np.argmax(closed))
    win = (t >= t[i0]) & (t <= t[i0] + window_s)
    return bool((z[win] - z[i0]).max() < lift_m)


def main() -> int:
    ee = _mod("e_eval", ROOT / "scripts" / "50_e_eval.py")
    p1 = CFG["eval"]["P1"]
    window_s, lift_m = float(p1["confirm_window_s"]), float(p1["confirm_lift_m"])
    res = {"sets": SETS, "criterion": {"confirm_window_s": window_s, "confirm_lift_m": lift_m}}
    # 1. 絞り込まない比較（把持失敗を意図的に起こした 50 組）
    X, Y = rows("A_P1"), rows("B_P1")
    sx = {k: {"success": v["meta"]["success"]} for k, v in X.items()}
    sy = {k: {"success": v["meta"]["success"]} for k, v in Y.items()}
    res["p1_all"] = ee.paired_binary(sx, sy, lambda r: bool(r["success"]))
    res["p1_all"]["x_k"] = sum(bool(v["success"]) for v in sx.values())
    res["p1_all"]["y_k"] = sum(bool(v["success"]) for v in sy.values())
    # 2. 意図的な失敗なしで自然に起きた把持失敗
    nat = {}
    for s in SETS:
        rs = rows(f"{s}_nat")
        fails = [r for r in rs.values() if natural_grasp_failure(r, window_s, lift_m)]
        nat[s] = {"trials": len(rs), "never_closed": sum(natural_grasp_failure(r, window_s, lift_m) is None for r in rs.values()),
                  "grasp_failed": len(fails), "recovered": sum(bool(r["meta"]["success"]) for r in fails),
                  "seeds": sorted([r["meta"]["seed"], r["meta"]["target"]] for r in fails)}
    res["natural_grasp_failure"] = nat
    # 3. 意図的な失敗なしで方策自身が起こした落下・置き損ね（recovla.eval.failure_detect の drop_table・drop_other を成功の前だけ見る）
    from recovla.eval import failure_detect as FD
    box_xy = np.asarray(CFG["scene"]["box"]["pos"][:2], float)
    near = float(CFG["eval"]["P2"]["min_dist_from_box_m"])     # 落下の意図的な失敗は箱の中心からこれ以上離れた所で開く
    drops = {}
    for s in SETS:
        out = {"trials": 0, "drop": [], "misplace": []}
        for r in rows(f"{s}_nat").values():
            out["trials"] += 1
            a = np.load(r["npz"])
            ev = [e for e in FD.detect_trial(a, t_end=r["meta"]["t_success"]) if e.kind in ("drop_table", "drop_other")]
            if not ev:
                continue
            e = ev[0]
            t = np.asarray(a["sim_time"], float)
            i = int(np.searchsorted(t, e.t))
            pos = np.asarray(a["cube_pos"], float)[min(i, len(t) - 1), COLORS.index(r["meta"]["target"])]
            dist = float(np.linalg.norm(pos[:2] - box_xy))
            kind = "misplace" if (e.kind == "drop_other" or dist < near) else "drop"
            out[kind].append({"seed": r["meta"]["seed"], "target": r["meta"]["target"], "event": e.kind, "t": round(e.t, 2),
                              "dist_from_box_m": round(dist, 3), "success": bool(r["meta"]["success"])})
        drops[s] = {"trials": out["trials"],
                    "drop": {"n": len(out["drop"]), "recovered": sum(x["success"] for x in out["drop"]), "cases": out["drop"]},
                    "misplace": {"n": len(out["misplace"]), "recovered": sum(x["success"] for x in out["misplace"]), "cases": out["misplace"]}}
    res["natural_drop_misplace"] = {"rule": f"drop_other、または着地点が箱の中心から {near} m 未満なら置き損ね、それ以外は落下", "sets": drops}
    res["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("natural_grasp_failure", "natural_drop_misplace")}, ensure_ascii=False, indent=1))
    print(json.dumps({s: {k: v for k, v in d.items() if k != "seeds"} for s, d in nat.items()}, ensure_ascii=False, indent=1))
    print(json.dumps(drops, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
