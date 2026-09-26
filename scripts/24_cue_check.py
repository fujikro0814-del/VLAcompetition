"""目標の手がかりの単体検査（決裁 0048 の 3）。結果は outputs/k1/cue_check.json。

    .venv\\Scripts\\python.exe scripts\\24_cue_check.py [--gen gen_100]

K1 の 300 本の全こま（20 Hz）について、エピソードの頭から順に recovla.perception.color.TargetCue を更新し
（見えないときは最後に見えた値を保つ）、手がかりの (x, y) と、目標の立方体の中心（真値、水平）の差を出す。
区間はこまの段階（phase）から: 接近（approach・descend）、把持（close・lift・reopen）、搬送（carry）、
設置（release・settle）、退避（retreat・done）。区間ごとに、見えない割合と、差の中央値・95 百分位・最大
（見えているこまだけ／保った値を含む全こま）。
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
OUT = config.path(CFG["paths"]["outputs"]) / "k1"
SEGMENT = {Phase.approach: "接近", Phase.descend: "接近", Phase.close: "把持", Phase.lift: "把持",
           Phase.reopen: "把持", Phase.carry: "搬送", Phase.release: "設置", Phase.settle: "設置",
           Phase.retreat: "退避", Phase.done: "退避"}
ORDER = ("接近", "把持", "搬送", "設置", "退避")


def stats(x) -> dict:
    x = np.asarray(x, float)
    if not x.size:
        return {"n": 0}
    return {"n": int(x.size), "median_cm": round(float(np.median(x)) * 100, 2),
            "p95_cm": round(float(np.percentile(x, 95)) * 100, 2), "max_cm": round(float(x.max()) * 100, 2)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gen", default="gen_100")
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    g = json.loads((OUT / f"{a.gen}.json").read_text(encoding="utf-8"))
    run = config.ROOT / g["run"]
    eps = sorted(p for p in run.iterdir() if p.is_dir() and not p.name.endswith(".partial"))
    calib = PC.overhead_calibration(scene.build_model("3cube"))
    thr = PC.Thresholds.from_config()
    cue = PC.TargetCue(calib, thr)
    seg = collections.defaultdict(lambda: {"frames": 0, "invisible": 0, "err_visible": [], "err_all": []})
    first_invisible = 0
    for p in eps:
        meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
        z = np.load(p / "data.npz")
        t = COLORS.index(meta["target"])
        cue.reset()
        for i in range(len(z["step"])):
            c = cue.update(read_png(p / "overhead" / f"{i:06d}.png"), meta["target"])
            err = float(np.linalg.norm(c[:2] - z["cube_pos"][i, t, :2]))
            s = seg[SEGMENT[Phase(int(z["phase"][i]))]]
            s["frames"] += 1
            s["err_all"].append(err)
            if c[2] > 0.5:
                s["err_visible"].append(err)
            else:
                s["invisible"] += 1
                first_invisible += int(i == 0)
    out = {k: {"frames": seg[k]["frames"], "invisible": seg[k]["invisible"],
               "invisible_rate": round(seg[k]["invisible"] / max(seg[k]["frames"], 1), 4),
               "error_visible": stats(seg[k]["err_visible"]), "error_all_with_hold": stats(seg[k]["err_all"])}
           for k in ORDER if k in seg}
    tot = sum(v["frames"] for v in out.values())
    res = {"gen": a.gen, "episodes": len(eps), "frames": tot, "thresholds": thr.to_json(),
           "invisible_rate_all": round(sum(v["invisible"] for v in out.values()) / max(tot, 1), 4),
           "first_frame_invisible": first_invisible, "segments": out,
           "note": "投影は机の面（立方体の中心の高さ）を仮定する。持ち上げた立方体（搬送・設置）は視差でずれる。"
                   "真値は検査にだけ使い、方策の入力には使わない", "wall_s": round(time.perf_counter() - t0, 1)}
    (OUT / "cue_check.json").write_text(json.dumps({"check": "cue_check", "written": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                    **res}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
