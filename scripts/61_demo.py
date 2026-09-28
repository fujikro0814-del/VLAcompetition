"""動画の場面の回し直しと描画（手順書 Step J の 3・4）。場面の設定は configs/demo/scenes.yaml。

    .venv\\Scripts\\python.exe scripts\\61_demo.py record --clip nat_110001_R1      # 1 場面を回し直して描く
    .venv\\Scripts\\python.exe scripts\\61_demo.py record --all                     # 全部（GPU は同時 3 本まで。別々に起動する）
    .venv\\Scripts\\python.exe scripts\\61_demo.py status                           # 場面ごとの再現の判定の一覧

record: 評価と同じ設定（方策・実行の方式・安全フィルタ・失敗注入・乱数シード）で 1 試行（または 3 個の連続タスク）を
回し直し、発表用カメラ（1280×720）で 30 fps・シミュレーション時刻に同期して描く。描画は物理ステップの後に行い、
物理と方策の入力（記録用カメラ 256×256）には触れない。GPU の描画の揺れのため、回し直しは元の記録とビット一致しない。
元の記録と同じ経過になったか（expect）を判定して meta.json に書く。再現しなかった場面は動画に使わない。
出力: outputs/demo/<場面>/presentation.mp4（H.264）、frames.npz（フレームごとの時刻・手先・参照位置・指）、meta.json
"""
import argparse
import json
import pathlib
import time

import numpy as np

from recovla.common import config

CFG = config.load()
ROOT = config.ROOT
OUT = config.path(CFG["paths"]["outputs"]) / "demo"
SCENES = ROOT / "configs" / "demo" / "scenes.yaml"
FPS = 30


def load_scenes() -> dict:
    import yaml
    return yaml.safe_load(SCENES.read_text(encoding="utf-8"))


class Presentation:
    """発表用カメラで、シミュレーション時刻 k/FPS を最初に過ぎた物理ステップの後に 1 フレーム描く。"""

    def __init__(self, rig, path):
        import av
        import mujoco
        self.rig = rig
        pc = CFG["scene"]["presentation_camera"]
        self.w, self.h = int(pc["width"]), int(pc["height"])
        self.renderer = mujoco.Renderer(rig.model, self.h, self.w)
        self.cam = rig.model.camera(pc["name"]).id
        self.box = av.open(str(path), mode="w")
        self.stream = self.box.add_stream("libx264", rate=FPS)
        self.stream.width, self.stream.height, self.stream.pix_fmt = self.w, self.h, "yuv420p"
        self.stream.options = {"crf": "16", "preset": "medium"}
        self.rows = []
        self.next_i = 0                              # rig.reset の後の最初の物理ステップから数える（1 試行・1 タスクに 1 回）

    def save(self, path) -> None:
        r = self.rows
        np.savez(path, t=np.array([x["t"] for x in r]), fingertip=np.array([x["fingertip"] for x in r]),
                 x_des=np.array([x["x_des"] for x in r]), closed=np.array([x["closed"] for x in r]),
                 cube_pos=np.array([x["cube_pos"] for x in r]))

    def step(self, rig) -> None:
        import av
        from recovla.sim import frames
        t = float(rig.data.time)
        if t + 1e-9 < self.next_i / FPS:
            return
        s = rig.forward_scratch()
        self.renderer.update_scene(s, camera=self.cam)
        img = self.renderer.render()
        f = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")
        for p in self.stream.encode(f):
            self.box.mux(p)
        self.rows.append({"t": t, "fingertip": frames.fingertip_center(s, rig.hand_id).tolist(),
                          "x_des": rig.integrator.x_cmd.tolist(), "closed": bool(rig.controller.gripper_closed),
                          "cube_pos": s.xpos[rig.cube_ids].tolist()})
        self.next_i += 1

    def camera(self) -> dict:
        d = self.rig.forward_scratch()
        import mujoco
        mujoco.mj_forward(self.rig.model, d)
        return {"pos": d.cam_xpos[self.cam].tolist(), "xmat": d.cam_xmat[self.cam].reshape(3, 3).tolist(),
                "fovy": float(self.rig.model.cam_fovy[self.cam]), "width": self.w, "height": self.h}

    def close(self) -> None:
        for p in self.stream.encode():
            self.box.mux(p)
        self.box.close()
        self.renderer.close()


def hook(rig, pres) -> None:
    """rig.pad_read の on_step に描画を足す（物理と方策の入力は変えない）。"""
    orig = rig.pad_read

    def pad_read(vel=None, press=False, on_step=None):
        def both(r):
            if on_step is not None:
                on_step(r)
            pres.step(r)
        return orig(vel, press, both)
    rig.pad_read = pad_read


def _runner(model_ckpt, mode):
    import os
    os.environ.setdefault("HF_HOME", str(config.path(CFG["paths"]["models_home"])))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from recovla.policy.runner import RuntimeConfig, SceneRunner
    from recovla.policy.scene_policy import ScenePolicy
    rt = CFG["runtime"]
    pol = ScenePolicy(config.path(model_ckpt))
    rcfg = RuntimeConfig(mode, int(rt["exec_interval"]), None if mode == "sync" else int(rt["delay_steps"]),
                         execution_horizon=int(rt["rtc_guidance_horizon"]))
    return pol, SceneRunner(pol, rcfg, {"schedule": rt["rtc_schedule"], "max_guidance_weight": rt["rtc_max_guidance_weight"]})


def _source_trial(src, seed, target):
    for p in sorted(config.path(src).glob("trial_*.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        if m["seed"] == seed and m["steps"][0]["target"] == target:
            return p, m
    raise SystemExit(f"元の記録がない: {src} {seed} {target}")


def record_trial(c, ckpts, out) -> dict:
    from recovla.eval import failure_detect as FD
    from recovla.eval import induce as I
    from recovla.eval import scene_trial as T
    from recovla.sim import scene
    from recovla.sim.rig import SimRig
    pol, runner = _runner(ckpts[c["model"]], c["mode"])
    lay = scene.sample_layout(c["seed"], "empty", start="home") if c["layout"] == "natural" else scene.sample_layout(c["seed"])
    rig = SimRig(render=True)
    rig.safety.enabled = bool(c.get("safety", True))
    pres = Presentation(rig, out / "presentation.mp4")
    hook(rig, pres)
    try:
        runner.start_trial(c["seed"])
        ind = I.Inducer(c["induce"], c["seed"], lay, c["target"], rig) if c.get("induce") else None
        meta, arr, _ = T.run_trial(rig, lay, c["target"], runner, {"trial": 0, "seed": c["seed"], "experiment": "demo",
                                                                    "condition": c["id"], "model": {"name": c["model"]},
                                                                    "runtime": runner.runtime_record()}, inducer=ind)
        cam = pres.camera()
        pres.save(out / "frames.npz")
    finally:
        pres.close()
        rig.close()
    tr = runner.trace()
    ev = [{"kind": e.kind, "t": e.t} for e in FD.detect_trial(arr)]
    src_p, src = _source_trial(c["source"], c["seed"], c["target"])
    src_ev = [{"kind": e.kind, "t": e.t} for e in FD.detect_trial(np.load(src_p.with_suffix(".npz")))]
    got = {"success": bool(meta["success"]), "grasp_miss": any(e["kind"] == "grasp_miss" for e in ev),
           "established": bool(meta["induce"]["established"])}
    ok = all(got[k] == v for k, v in c["expect"].items())
    np.savez(out / "trace.npz", chunk_xdes_pred=tr["chunk_xdes_pred"], chunk_k_valid=tr["chunk_k_valid"],
             chunk_k_obs=tr["chunk_k_obs"], exec=np.array([e if e is not None else (-1, -1) for e in tr["exec"]]))
    return {"got": got, "reproduced": bool(ok), "events": ev, "t_success": meta["steps"][0]["t_success"],
            "induce": meta["induce"], "t_end": float(arr["sim_time"][-1]), "camera": cam,
            "source": {"path": str(src_p.relative_to(ROOT)), "success": src["success"],
                       "t_success": src["steps"][0]["t_success"], "events": src_ev, "induce": src["induce"]},
            "runtime": {**runner.runtime_record(), "safety_filter": bool(c.get("safety", True))}}


def record_task(c, ckpts, out) -> dict:
    import copy
    from recovla.planner.executor import TaskExecutor
    from recovla.sim import scene
    from recovla.sim.rig import SimRig
    pol, runner = _runner(ckpts[c["model"]], "naive")
    rig = SimRig(render=True)
    pres = Presentation(rig, out / "presentation.mp4")
    hook(rig, pres)
    try:
        ex = TaskExecutor(rig, runner, copy.deepcopy(CFG))
        meta, arr, _ = ex.run(scene.sample_layout(c["seed"], "empty", start="home"), c["text"], c["seed"])
        cam = pres.camera()
        pres.save(out / "frames.npz")
    finally:
        pres.close()
        rig.close()
    src = None
    for p in sorted(config.path(c["source"]).glob("task_*.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        if m["seed"] == c["seed"] and m["text"] == c["text"]:
            src = (p, m)
    if src is None:
        raise SystemExit(f"元の記録がない: {c['source']} {c['seed']}")
    retry_done = any(s["judged_complete"] and len(s["attempts"]) > 1 for s in meta["steps"])
    got = {"all_three": bool(meta["all_three_in_box"]), "retry_completed": retry_done}
    ok = all(got[k] == v for k, v in c["expect"].items())
    keep = ("plan", "steps", "returns", "stopped", "final_in_box", "detected", "text")
    return {"got": got, "reproduced": bool(ok), "task": {k: meta[k] for k in keep}, "t_end": meta["t_end"], "camera": cam,
            "source": {"path": str(src[0].relative_to(ROOT)), "all_three": src[1]["all_three_in_box"],
                       "steps": [(s["color"], s["judged_complete"], len(s["attempts"])) for s in src[1]["steps"]]}}


def cmd_record(a) -> None:
    sc = load_scenes()
    clips = [c for c in sc["clips"] if a.all or c["id"] in a.clip]
    for c in clips:
        out = OUT / c["id"]
        if (out / "meta.json").exists():
            print("済み", c["id"])
            continue
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        res = (record_trial if c["kind"] == "trial" else record_task)(c, sc["checkpoints"], out)
        res.update(clip=c, wall_s=round(time.perf_counter() - t0, 1), written=time.strftime("%Y-%m-%d %H:%M:%S"))
        (out / "meta.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        print(c["id"], "再現" if res["reproduced"] else "再現せず", json.dumps(res["got"], ensure_ascii=False), flush=True)


def cmd_status(a) -> None:
    for c in load_scenes()["clips"]:
        p = OUT / c["id"] / "meta.json"
        if not p.exists():
            print(f"{c['id']:22s} まだ")
            continue
        m = json.loads(p.read_text(encoding="utf-8"))
        print(f"{c['id']:22s} {'再現' if m['reproduced'] else '再現せず'} got={m['got']} expect={c['expect']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("record")
    s.add_argument("--clip", nargs="*", default=[])
    s.add_argument("--all", action="store_true")
    sub.add_parser("status")
    a = ap.parse_args(argv)
    {"record": cmd_record, "status": cmd_status}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
