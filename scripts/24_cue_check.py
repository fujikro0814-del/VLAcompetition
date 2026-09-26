"""目標の手がかりの単体検査（決裁 0048 の 3、0050 の 1）。結果は outputs/k1/cue_check.json・outputs/f/cue_check.json。

    .venv\\Scripts\\python.exe scripts\\24_cue_check.py [--gen gen_100]        # K1 の 300 本
    .venv\\Scripts\\python.exe scripts\\24_cue_check.py --data                 # R1・N1（outputs/f/data.json のマニフェスト）

エピソードの頭から順に recovla.perception.color.TargetCue を更新し（見えないときは最後に見えた値を保つ）、
全こま（20 Hz）で手がかりの (x, y) と目標の立方体の中心（真値、水平）の差を出す。区間はこまの段階（phase）から:
接近（approach・descend）、把持（close・lift・reopen）、搬送（carry）、設置（release・settle）、退避（retreat・done）。
--data では、区間ごと・開始姿勢ごと・群（通常 n、復帰 A・B・C）ごとにも分け、学習データ（変換で使う 10 fps のこま）で
旗が 0 になる割合を R1・N1 それぞれで出す（0050 の 2: 1% 未満なら旗を方策の入力から外す）。
"""
import argparse
import collections
import json
import time

import numpy as np

from recovla.common import config
from recovla.common.seeds import COLORS
from recovla.expert.script import Phase
from recovla.perception import color as PC
from recovla.record.recorder import read_png
from recovla.sim import scene

CFG = config.load()
OUTPUTS = config.path(CFG["paths"]["outputs"])
SEGMENT = {Phase.approach: "接近", Phase.descend: "接近", Phase.close: "把持", Phase.lift: "把持",
           Phase.reopen: "把持", Phase.carry: "搬送", Phase.release: "設置", Phase.settle: "設置",
           Phase.retreat: "退避", Phase.done: "退避"}
ORDER = ("接近", "把持", "搬送", "設置", "退避")
STRIDE = int(CFG["sim"]["stride"])
FLAG_DROP_BELOW = 0.01


def stats(x) -> dict:
    x = np.asarray(x, float)
    if not x.size:
        return {"n": 0}
    return {"n": int(x.size), "median_cm": round(float(np.median(x)) * 100, 2),
            "p95_cm": round(float(np.percentile(x, 95)) * 100, 2), "max_cm": round(float(x.max()) * 100, 2)}


def episode_cues(cue, path, meta):
    """(こまの数, 3) の手がかり、(こまの数,) の差、区間。"""
    z = np.load(path / "data.npz")
    t = COLORS.index(meta["target"])
    cue.reset()
    out, errs, segs = [], [], []
    for i in range(len(z["step"])):
        c = cue.update(read_png(path / "overhead" / f"{i:06d}.png"), meta["target"])
        out.append(c)
        errs.append(float(np.linalg.norm(c[:2] - z["cube_pos"][i, t, :2])))
        segs.append(SEGMENT[Phase(int(z["phase"][i]))])
    return np.array(out), np.array(errs), segs


def summarize(rows: list) -> dict:
    """rows: [(区間, 群, 開始姿勢, 差, 見えるか)]。"""
    def block(sel):
        e = [r[3] for r in sel]
        inv = sum(1 for r in sel if not r[4])
        return {"frames": len(sel), "invisible": inv, "invisible_rate": round(inv / max(len(sel), 1), 4),
                "error_all_with_hold": stats(e), "error_visible": stats([r[3] for r in sel if r[4]])}
    out = {"by_segment": {s: block([r for r in rows if r[0] == s]) for s in ORDER if any(r[0] == s for r in rows)}}
    groups = sorted({r[1] for r in rows})
    starts = sorted({r[2] for r in rows})
    if len(groups) > 1 or len(starts) > 1:
        out["by_group"] = {g: block([r for r in rows if r[1] == g]) for g in groups}
        out["by_start"] = {s: block([r for r in rows if r[2] == s]) for s in starts}
        out["by_segment_group"] = {f"{s}|{g}": block([r for r in rows if r[0] == s and r[1] == g])
                                   for s in ORDER for g in groups if any(r[0] == s and r[1] == g for r in rows)}
        out["by_segment_start"] = {f"{s}|{st}": block([r for r in rows if r[0] == s and r[2] == st])
                                   for s in ORDER for st in starts if any(r[0] == s and r[2] == st for r in rows)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gen", default="gen_100")
    ap.add_argument("--data", action="store_true", help="R1・N1（outputs/f/data.json）")
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    calib = PC.overhead_calibration(scene.build_model("3cube"))
    thr = PC.Thresholds.from_config()
    cue = PC.TargetCue(calib, thr)
    if a.data:
        d = json.loads((OUTPUTS / "f" / "data.json").read_text(encoding="utf-8"))
        sets = {}
        for name in ("R1", "N1"):
            m = json.loads(config.path(d["datasets"][name]["manifest"]).read_text(encoding="utf-8"))
            sets[name] = [config.path(e["run"]) / e["key"] for e in m["entries"]]
        episodes = sorted({p for ps in sets.values() for p in ps})
        out_dir = OUTPUTS / "f"
        source = {"data": d["run"], "manifests": {k: d["datasets"][k]["manifest"] for k in sets}}
    else:
        g = json.loads((OUTPUTS / "k1" / f"{a.gen}.json").read_text(encoding="utf-8"))
        run = config.ROOT / g["run"]
        episodes = sorted(p for p in run.iterdir() if p.is_dir() and not p.name.endswith(".partial"))
        sets = {"K1": episodes}
        out_dir = OUTPUTS / "k1"
        source = {"gen": a.gen}
    rows, flags10 = [], {}
    for p in episodes:
        meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
        cues, errs, segs = episode_cues(cue, p, meta)
        group = meta.get("kind", "n")
        start = meta.get("start_pose") or meta.get("layout", {}).get("start") or "?"
        rows += [(s, group, start, e, bool(c[2] > 0.5)) for s, e, c in zip(segs, errs, cues)]
        n10 = (len(cues) - 1) // STRIDE                    # 変換が使う 10 fps のこま（convert.episode_arrays と同じ）
        flags10[p] = cues[np.arange(n10) * STRIDE, 2]
    res = {**source, "episodes": len(episodes), "frames": len(rows), "thresholds": thr.to_json(),
           "invisible_rate_all": round(sum(1 for r in rows if not r[4]) / max(len(rows), 1), 4), **summarize(rows)}
    flag = {}
    for name, ps in sets.items():
        f = np.concatenate([flags10[p] for p in ps])
        zero = int((f < 0.5).sum())
        flag[name] = {"frames_10fps": int(f.size), "flag0_frames": zero, "flag0_rate": round(zero / max(f.size, 1), 5),
                      "keep_flag_in_input": bool(zero / max(f.size, 1) >= FLAG_DROP_BELOW)}
    res["training_flag0"] = flag
    res["flag_rule"] = f"決裁 0050 の 2: 学習データで旗が 0 のこまが {FLAG_DROP_BELOW:.0%} 未満なら旗を方策の入力から外す"
    res["note"] = ("投影は机の面（立方体の中心の高さ）を仮定する。持ち上げた立方体（搬送・設置）は視差で跳ぶ。"
                   "真値は検査にだけ使い、方策の入力には使わない")
    res["wall_s"] = round(time.perf_counter() - t0, 1)
    (out_dir / "cue_check.json").write_text(json.dumps({"check": "cue_check", "written": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                        **res}, ensure_ascii=False, indent=2), encoding="utf-8")
    brief = {k: res[k] for k in ("episodes", "frames", "invisible_rate_all", "training_flag0", "wall_s")}
    print(json.dumps(brief, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
