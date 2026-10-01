"""Rerun（rerun.io）で学習データと評価の走行を見る。評価の環境とは別の .venv_viz で動かす（rerun-sdk・pyarrow・numpy・pillow だけ）。

    .venv_viz\\Scripts\\python.exe scripts\\90_rerun.py dataset --root outputs\\datasets\\R1v2_20260929-205251 --episodes 0,254
    .venv_viz\\Scripts\\python.exe scripts\\90_rerun.py run --dir outputs\\v2eval\\V2CAUSE\\R1v2_nat --trials 1,4
    .venv_viz\\Scripts\\rerun.exe outputs\\viz\\<名前>.rrd                    # ビューアで開く

学習データ: 俯瞰・手首の画像、状態（17 次元）、行動（7 次元）を、エピソードごとの時間（10 Hz）に載せる。
評価の走行（trial_NNNN.npz と runtime_NNNN.json）: 手先・x_cmd・指先・立方体の真値の軌跡、知覚の推定、箱、グリッパ、
推論の時刻を、シミュレーションの時間に載せる（走行の記録に画像は残っていない）。
"""
import argparse
import io
import json
import pathlib
import sys

import numpy as np
import rerun as rr

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "viz"
COLORS = ("red", "green", "blue")
RGB = {"red": (220, 40, 40), "green": (40, 180, 60), "blue": (50, 80, 230)}


def _ids(s: str) -> list:
    return [int(x) for x in s.split(",") if x.strip()]


def cmd_dataset(a) -> pathlib.Path:
    import pyarrow.parquet as pq
    from PIL import Image
    root = pathlib.Path(a.root)
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    sn = info["features"]["observation.state"]["names"]
    an = info["features"]["action"]["names"]
    eps = set(_ids(a.episodes))
    name = f"dataset_{root.name}_ep{'-'.join(str(e) for e in sorted(eps))}"
    rr.init(name)
    path = OUT / f"{name}.rrd"
    rr.save(str(path))
    cols = ["observation.images.image", "observation.images.image2", "observation.state", "action", "episode_index", "frame_index"]
    found = set()
    for f in sorted((root / "data").rglob("*.parquet")):
        t = pq.read_table(f, columns=cols).to_pydict()
        for r, e in enumerate(t["episode_index"]):
            if e not in eps:
                continue
            found.add(e)
            rr.set_time("frame", sequence=int(t["frame_index"][r]))
            rr.set_time("t", duration=t["frame_index"][r] / float(info["fps"]))
            base = f"ep{e}"
            for key, cam in (("observation.images.image", "overhead"), ("observation.images.image2", "wrist")):
                img = np.asarray(Image.open(io.BytesIO(t[key][r]["bytes"])).convert("RGB"))
                rr.log(f"{base}/camera/{cam}", rr.Image(img).compress(jpeg_quality=90))
            for i, v in enumerate(t["observation.state"][r]):
                rr.log(f"{base}/state/{sn[i]}", rr.Scalars(float(v)))
            for i, v in enumerate(t["action"][r]):
                rr.log(f"{base}/action/{an[i]}", rr.Scalars(float(v)))
    missing = eps - found
    if missing:
        print(f"見つからないエピソード: {sorted(missing)}", file=sys.stderr)
    return path


def cmd_run(a) -> pathlib.Path:
    d = pathlib.Path(a.dir)
    trials = _ids(a.trials)
    name = f"run_{d.parent.name}_{d.name}_t{'-'.join(str(i) for i in trials)}"
    rr.init(name)
    path = OUT / f"{name}.rrd"
    rr.save(str(path))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    for i in trials:
        m = json.loads((d / f"trial_{i:04d}.json").read_text(encoding="utf-8"))
        z = np.load(d / f"trial_{i:04d}.npz")
        rt = json.loads((d / f"runtime_{i:04d}.json").read_text(encoding="utf-8"))["runtime"]
        base = f"trial{i}"
        tgt = m["target"]
        ti = COLORS.index(tgt)
        rr.log(f"{base}/info", rr.TextDocument(
            f"seed {m['seed']}  target {tgt}  success {m['success']}  t_success {m.get('t_success')}\n"
            f"stop_reason {rt.get('stop_reason')}  G3 violations {m['audit']['g3']['total_violations']}"), static=True)
        t = z["sim_time"]
        rr.log(f"{base}/world/path/hand", rr.LineStrips3D([z["ee_pos"]], colors=[(200, 200, 200)]), static=True)
        rr.log(f"{base}/world/path/x_cmd", rr.LineStrips3D([z["x_des"]], colors=[(255, 160, 0)]), static=True)
        rr.log(f"{base}/world/path/fingertip", rr.LineStrips3D([z["fingertip"]], colors=[(120, 120, 255)]), static=True)
        box = rt.get("startup", {}) or {}
        bxy = (box.get("box") or {}).get("xy")
        if bxy:
            rr.log(f"{base}/world/box_estimate", rr.Boxes3D(centers=[[bxy[0], bxy[1], 0.03]], half_sizes=[[0.08, 0.08, 0.03]],
                                                            colors=[(200, 170, 40)]), static=True)
        for k in range(len(t)):
            rr.set_time("sim_time", duration=float(t[k]))
            rr.log(f"{base}/world/hand", rr.Points3D([z["ee_pos"][k]], radii=0.01, colors=[(230, 230, 230)]))
            rr.log(f"{base}/world/x_cmd", rr.Points3D([z["x_des"][k]], radii=0.008, colors=[(255, 160, 0)]))
            rr.log(f"{base}/world/fingertip", rr.Points3D([z["fingertip"][k]], radii=0.006, colors=[(120, 120, 255)]))
            rr.log(f"{base}/world/cubes_truth", rr.Boxes3D(centers=z["cube_pos"][k], half_sizes=[[0.02, 0.02, 0.02]] * 3,
                                                           colors=[RGB[c] for c in COLORS]))
            rr.log(f"{base}/signal/gripper_closed", rr.Scalars(float(z["gripper_closed"][k])))
            rr.log(f"{base}/signal/finger_width_mm", rr.Scalars(float(np.sum(z["fingers"][k]) * 1000)))
            rr.log(f"{base}/signal/x_cmd_minus_hand_mm", rr.Scalars(float(np.linalg.norm(z["x_des"][k] - z["ee_pos"][k]) * 1000)))
            rr.log(f"{base}/signal/tip_to_target_mm", rr.Scalars(float(np.linalg.norm(z["fingertip"][k] - z["cube_pos"][k, ti]) * 1000)))
            rr.log(f"{base}/signal/safety_active", rr.Scalars(float(z["safety_active"][k])))
        for e in rt.get("perception", []):
            rr.set_time("sim_time", duration=float(e["t"]))
            pts, cols = [], []
            for c, v in e["cubes"].items():
                if v.get("pos") is not None and v["status"] != "lost":
                    pts.append(v["pos"])
                    cols.append(RGB[c])
            rr.log(f"{base}/world/cubes_perceived", rr.Points3D(pts, radii=0.012, colors=cols) if pts else rr.Clear(recursive=False))
        for e in rt.get("inference", []):
            rr.set_time("sim_time", duration=float(e["t_obs"]))
            rr.log(f"{base}/events/inference", rr.TextLog(f"inference {e['i']} latency {e['latency_s']:.3f} s"))
            if e.get("cue"):
                rr.log(f"{base}/world/cue", rr.Points3D([[e["cue"][0], e["cue"][1], 0.02]], radii=0.01, colors=[(255, 0, 255)]))
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("dataset")
    p.add_argument("--root", required=True)
    p.add_argument("--episodes", default="0")
    p = sub.add_parser("run")
    p.add_argument("--dir", required=True)
    p.add_argument("--trials", default="0")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    path = {"dataset": cmd_dataset, "run": cmd_run}[a.cmd](a)
    print(f"書き出した: {path}（{path.stat().st_size / 1e6:.1f} MB）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
