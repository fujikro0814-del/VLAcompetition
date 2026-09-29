"""把持力の確かめ（0108 の 2 の 3）: Franka Hand の模型（recovla.harness.hand）で握る力を変え、シミュレーションの
接触が安定するか（めり込み・跳ね・滑り）を学習用のシード（59000〜）でエキスパートを回して測る。

    .venv\\Scripts\\python.exe scripts\\81_hand_check.py run --forces 30,50,70 --seeds 59000:10 [--tag base]
    .venv\\Scripts\\python.exe scripts\\81_hand_check.py run --forces 50 --seeds 59000:10 --pad-solref -30000,-300 --tag stiff

指標（握っている間 = ハンドが grasp の状態で、立方体が机から 1 cm 以上上がっている間）:
  めり込み   指と立方体の接触の距離（負）の最大の深さ [mm]、握っている間の開き幅
  滑り       手（hand）の座標で見た立方体の位置の、握った直後からのずれの最大 [mm]
  跳ね       立方体の、手に対する相対速度の中央値・最大 [mm/s]
  指の速さ   指 1 本あたりの速さの最大 [m/s]（上限 0.05）
出力: outputs/hand/<tag>.json
"""
import argparse
import json
import time

import mujoco
import numpy as np

from recovla.common import config

CFG = config.load()
OUT = config.path(CFG["paths"]["outputs"]) / "hand"


def make_rig(force: float, pad_solref=None):
    from recovla.harness.hand import FrankaHand
    from recovla.sim.rig import SimRig

    class HandRig(SimRig):
        def __init__(self):
            super().__init__(render=False)
            m = self.model
            self.hand = FrankaHand(m, self.data, self.grip_act, self.finger_qadr, self.finger_vadr)
            for st in self.starts.values():                 # 開始状態の ctrl は位置のサーボの値（255）なので、開く速さに替える
                st.ctrl[self.grip_act] = 0.045
            self.controller.gripper_indices = []            # 制御器は指を動かさない（ハンドの口だけ）
            self.force = force
            self._closed = False
            if pad_solref is not None:
                for g in range(m.ngeom):
                    name = m.geom(g).name or ""
                    b = m.body(m.geom_bodyid[g]).name
                    if b in ("left_finger", "right_finger") and m.geom_contype[g]:
                        m.geom_solref[g] = pad_solref
            self.pad_geoms = {g for g in range(m.ngeom)
                              if m.body(m.geom_bodyid[g]).name in ("left_finger", "right_finger")}
            self.cube_geom = {m.geom(f"cube_{c}_geom").id: c for c in ("red", "green", "blue")}
            self.trace = None

        def reset(self, layout):
            super().reset(layout)
            self.hand.reset(opened=True)
            self._closed = False
            self.trace = []

        def pad_read(self, vel=None, press=False, on_step=None):
            from recovla.sim.device import pad_state
            m, d = self.model, self.data
            if press:
                self._closed = not self._closed
                if self._closed:
                    self.hand.grasp(0.04, 0.08, self.force, 0.005, 0.005)
                else:
                    self.hand.move(0.08, 0.08)
            self.pad.state = pad_state(vel=np.zeros(3) if vel is None else np.asarray(vel, float), button_grip=bool(press))
            self.integrator.refresh()
            self.integrator.command_filter = None
            for _ in range(self.steps_per_read):
                self.controller.update(self.integrator)
                self.hand.step()
                mujoco.mj_step(m, d)
                self.step += 1
                self.meter.on_step(d)
                self._record()
                if on_step is not None:
                    on_step(self)

        def _record(self):
            d, m = self.data, self.model
            pen = 0.0
            for i in range(d.ncon):
                c = d.contact[i]
                pair = {c.geom1, c.geom2}
                if pair & self.pad_geoms and pair & set(self.cube_geom):
                    pen = min(pen, float(c.dist))
            hid = self.hand_id
            R = d.xmat[hid].reshape(3, 3)
            rel = []
            for cid in self.cube_ids:
                rel.append(R.T @ (d.xpos[cid] - d.xpos[hid]))
            self.trace.append((d.time, self.hand.mode, self.hand.width(), self.hand.width_speed(),
                               bool(self.hand.is_grasped()), pen, np.array(rel), d.xpos[self.cube_ids][:, 2].copy(),
                               float(d.qvel[self.finger_vadr].max()), float(np.abs(d.qvel[self.finger_vadr]).max())))

    return HandRig()


def metrics(trace, color_index, dt):
    t = np.array([r[0] for r in trace])
    mode = np.array([r[1] for r in trace])
    rel = np.array([r[6][color_index] for r in trace])
    z = np.array([r[7][color_index] for r in trace])
    pen = np.array([r[5] for r in trace])
    fspeed = np.array([r[9] for r in trace])
    grasped = np.array([r[4] for r in trace])
    held = (mode == "grasp") & (z > 0.03)
    width = np.array([r[2] for r in trace])
    out = {"finger_speed_max": float(fspeed.max()), "held_s": float(held.sum() * dt),
           "width_held_med": float(np.median(width[held])) if held.any() else None,
           "is_grasped_while_held": float(grasped[held].mean()) if held.any() else None}
    if held.any():
        idx = np.flatnonzero(held)
        r0 = rel[idx[0]]
        drift = np.linalg.norm(rel[idx] - r0, axis=1)
        vrel = np.linalg.norm(np.diff(rel[idx], axis=0), axis=1) / dt
        out.update({"penetration_max_mm": float(-pen[held].min() * 1e3), "penetration_med_mm": float(-np.median(pen[held]) * 1e3),
                    "slip_max_mm": float(drift.max() * 1e3), "relvel_med_mm_s": float(np.median(vrel) * 1e3),
                    "relvel_p99_mm_s": float(np.percentile(vrel, 99) * 1e3), "relvel_max_mm_s": float(vrel.max() * 1e3)})
    return out


def cmd_run(a) -> None:
    from recovla.common.seeds import COLORS
    from recovla.expert import generate as G
    from recovla.sim import scene
    base, n = (int(x) for x in a.seeds.split(":"))
    solref = None if not a.pad_solref else [float(x) for x in a.pad_solref.split(",")]
    res = {"seeds": a.seeds, "pad_solref": solref, "forces": {}}
    for force in (float(x) for x in a.forces.split(",")):
        rig = make_rig(force, solref)
        rows = []
        for s in range(base, base + n):
            lay = scene.sample_layout(s, "empty")
            sp = G.EpisodeSpec(s, lay.table_colors[0], "empty", "n")
            r = G.run_attempt(rig, sp, 0, None, False)
            m = metrics(rig.trace, COLORS.index(sp.color), rig.timestep)
            m.update({"seed": s, "color": sp.color, "success": bool(r["success"])})
            rows.append(m)
            print(f"[hand] F {force:4.0f} seed {s} {sp.color:5s} ok {r['success']} "
                  + " ".join(f"{k} {v:.2f}" for k, v in m.items() if isinstance(v, float)), flush=True)
        rig.close()
        keys = [k for k in rows[0] if isinstance(rows[0][k], float)]
        summ = {"success": sum(r["success"] for r in rows), "n": len(rows)}
        for k in keys:
            vals = [r[k] for r in rows if r.get(k) is not None]
            if vals:
                summ[k + "_worst"] = float(max(vals)) if k != "is_grasped_while_held" else float(min(vals))
        res["forces"][str(force)] = {"summary": summ, "episodes": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    res["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (OUT / f"{a.tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({f: v["summary"] for f, v in res["forces"].items()}, ensure_ascii=False, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--forces", default="30,50,70")
    p.add_argument("--seeds", default="59000:10")
    p.add_argument("--pad-solref", default=None)
    p.add_argument("--tag", default="base")
    a = ap.parse_args(argv)
    {"run": cmd_run}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
