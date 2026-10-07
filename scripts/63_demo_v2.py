"""動画の場面の回し直しと描画（目標書 v2 の実行系、段階 3 の最終評価と同じ設定）。0146・0147。

    .venv\\Scripts\\python.exe scripts\\63_demo_v2.py run --clip nat_140011_R --model R1v3 --trials natural:140011:1 [--no-safety]
    .venv\\Scripts\\python.exe scripts\\63_demo_v2.py task --clip task_145003 --model R1v3 --trials 145003:1 [--no-safety]
    .venv\\Scripts\\python.exe scripts\\63_demo_v2.py compare --clip nat_140011_R --source outputs/v2eval/V3S3/A_nat

評価のコード（82_v2_eval.py、凍結の版 v3-s3-freeze）をそのまま呼び、世界（WorldRig）の reset と physics_step に
発表用カメラ（1280×720、30 fps、シミュレーション時刻に同期）の描画だけを足す。描画は物理ステップの後に行い、
物理と方策の入力（センサの模型）には触れない。GPU の描画の揺れのため、回し直しは元の記録とビット一致しない。
compare で、元の記録（テスト用の評価）と同じ経過（成否と成功の時刻）になったかを判定して meta.json に書く。
出力: outputs/demo_v2/<場面>/trial_NN.mp4・trial_NN_frames.npz・meta.json（評価の記録は outputs/v2eval/V3DEMO/<場面>/）
"""
import argparse
import importlib.util
import json
import sys

import numpy as np

from recovla.common import config

ROOT = config.ROOT
CFG = config.load_v2()
OUT = config.path(CFG["paths"]["outputs"]) / "demo_v2"
EVAL = config.path(CFG["paths"]["outputs"]) / "v2eval"
EXP = "V3DEMO"
FPS = 30


def _ev():
    spec = importlib.util.spec_from_file_location("ev82", ROOT / "scripts" / "82_v2_eval.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Presentation:
    """発表用カメラで、シミュレーション時刻 k/FPS を最初に過ぎた物理ステップの後に 1 フレーム描く。"""

    def __init__(self, rig, path):
        import av
        import mujoco
        pc = CFG["scene"]["presentation_camera"]
        self.w, self.h = int(pc["width"]), int(pc["height"])
        self.renderer = mujoco.Renderer(rig.model, self.h, self.w)
        self.cam = rig.model.camera(pc["name"]).id
        self.box = av.open(str(path), mode="w")
        self.stream = self.box.add_stream("libx264", rate=FPS)
        self.stream.width, self.stream.height, self.stream.pix_fmt = self.w, self.h, "yuv420p"
        self.stream.options = {"crf": "16", "preset": "medium"}
        self.rows = []
        self.t0 = None
        self.next_i = 0

    def step(self, rig) -> None:
        import av
        from recovla.sim import frames
        t = float(rig.data.time)
        if self.t0 is None:
            self.t0 = t
        if t - self.t0 + 1e-9 < self.next_i / FPS:
            return
        s = rig.forward_scratch()
        self.renderer.update_scene(s, camera=self.cam)
        f = av.VideoFrame.from_ndarray(np.ascontiguousarray(self.renderer.render()), format="rgb24")
        for p in self.stream.encode(f):
            self.box.mux(p)
        self.rows.append({"t": t - self.t0, "fingertip": frames.fingertip_center(s, rig.hand_id).tolist(),
                          "cube_pos": s.xpos[rig.cube_ids].tolist()})
        self.next_i += 1

    def close(self, npz) -> None:
        for p in self.stream.encode():
            self.box.mux(p)
        self.box.close()
        self.renderer.close()
        r = self.rows
        np.savez(npz, t=np.array([x["t"] for x in r]), fingertip=np.array([x["fingertip"] for x in r]),
                 cube_pos=np.array([x["cube_pos"] for x in r]))


def install(clip_dir):
    """WorldRig の reset ごとに新しい動画を開き、physics_step の後に描く（評価のコードは変えない）。"""
    from recovla.harness.world import WorldRig
    state = {"pres": None, "i": 0}
    orig_reset, orig_step = WorldRig.reset, WorldRig.physics_step

    def finish():
        if state["pres"] is not None:
            state["pres"].close(clip_dir / f"trial_{state['i'] - 1:02d}_frames.npz")
            state["pres"] = None

    def reset(self, layout):
        finish()
        orig_reset(self, layout)
        state["pres"] = Presentation(self, clip_dir / f"trial_{state['i']:02d}.mp4")
        state["i"] += 1

    def physics_step(self, on_step=None):
        orig_step(self, on_step)
        if state["pres"] is not None:
            state["pres"].step(self)
    WorldRig.reset, WorldRig.physics_step = reset, physics_step
    return finish


def cmd_record(a, kind) -> None:
    clip_dir = OUT / a.clip
    if clip_dir.exists() and any(clip_dir.glob("*.mp4")):
        raise SystemExit(f"{clip_dir} にもう動画がある（消してから回す）")
    clip_dir.mkdir(parents=True, exist_ok=True)
    finish = install(clip_dir)
    argv = [kind, "--experiment", EXP, "--condition", a.clip, "--model", a.model, "--trials", a.trials,
            "--exec-interval", str(a.exec_interval)]
    if kind == "run":
        argv += ["--mode", a.mode] + (["--induce", a.induce] if a.induce else [])
    if a.no_safety:
        argv.append("--no-safety")
    rc = _ev().main(argv)
    finish()
    print(json.dumps({"clip": a.clip, "argv": argv, "rc": rc}, ensure_ascii=False))


def cmd_compare(a) -> None:
    """回し直し（V3DEMO/<場面>）と元の記録（--source）を、種と色で突き合わせる。"""
    def load(d):
        rows = {}
        for p in sorted(config.path(d).glob("trial_*.json")):
            m = json.loads(p.read_text(encoding="utf-8"))
            rows[(m["seed"], m["target"])] = {"trial": m["trial"], "success": m["success"], "t_success": m["t_success"],
                                              "established": (m.get("induce") or {}).get("established")}
        return rows
    new, src = load(EVAL / EXP / a.clip), load(a.source)
    out = []
    for k, r in new.items():
        s = src.get(k)
        same = s is not None and s["success"] == r["success"] and (
            not r["success"] or abs((s["t_success"] or 0) - (r["t_success"] or 0)) <= a.tol_s)
        out.append({"seed": k[0], "target": k[1], "trial": r["trial"], "rerun": r, "source": s, "same_course": bool(same)})
    meta = {"clip": a.clip, "source": a.source, "tol_s": a.tol_s, "rows": out,
            "note": "同じ設定で回し直した映像。ビット一致ではない。same_course が真のものだけを動画に使う"}
    (OUT / a.clip / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "task"):
        p = sub.add_parser(name)
        p.add_argument("--clip", required=True)
        p.add_argument("--model", required=True)
        p.add_argument("--trials", required=True)
        p.add_argument("--exec-interval", type=int, default=6)
        p.add_argument("--no-safety", action="store_true")
        if name == "run":
            p.add_argument("--mode", default="naive", choices=("naive", "sync", "rtc"))
            p.add_argument("--induce", default=None)
    p = sub.add_parser("compare")
    p.add_argument("--clip", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--tol-s", type=float, default=3.0)
    a = ap.parse_args(argv)
    if a.cmd == "compare":
        cmd_compare(a)
    else:
        cmd_record(a, a.cmd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
