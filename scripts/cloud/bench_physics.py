"""クラウドの支線の疎通確認（掲示板 cloud/setup-check）: MuJoCo の物理の速さ。描画はしない。

    .venv/bin/python scripts/cloud/bench_physics.py [--steps 5000] [--repeats 3]

assets/mjcf/scene_g0.xml を読み、10 秒分（timestep 0.002 s × 5000 手）を mj_step で回して、実時間に対する倍率
（模擬した秒 / 経過した秒）を測る。次の 2 通りを、それぞれ repeats 回測って中央値を出す。

  controller: 評価器と同じ入口（ScriptPad → VelocityCommandIntegrator → DLS の IK → mj_step。
              recovla.eval.closed_loop.execute_action と同じ並び）を毎周期通す。指令は 0.1 s ごとに向きを変える
              xy の円（半径方向の速さ 0.05 m/s）。
  physics:    制御器を通さず、落ち着かせた開始状態の ctrl のまま mj_step だけを回す。

開始状態を落ち着かせる処理（settle_start_state）と、模型の読み込みは時間に含めない。
"""
import argparse
import contextlib
import io
import json
import os
import platform
import statistics
import time

import mujoco
import numpy as np

from recovla.record import recorder
from recovla.sim import control
from recovla.sim.device import ScriptPad, pad_state

STEPS_PER_ACTION = 50          # 10 Hz の行動 1 つ = 物理 50 手（closed_loop と同じ）


def _setup():
    model = mujoco.MjModel.from_xml_path(control.SCENE_PATH)
    control.check_timestep(model)
    data = mujoco.MjData(model)
    with contextlib.redirect_stdout(io.StringIO()):
        start = control.settle_start_state(model)
        controller = control.make_collect_controller(model, data)
    pad = ScriptPad()
    integrator = control.make_integrator(pad, model, controller)
    placement = recorder.training_placements()[0]
    return model, data, start, controller, pad, integrator, placement


def _reset(model, data, start, controller, pad, integrator, placement):
    with contextlib.redirect_stdout(io.StringIO()):
        control.reset_episode(model, data, controller, integrator, start, placement)
    pad.state = pad_state()
    integrator.refresh()


def run_controller(rig, steps: int) -> float:
    model, data, start, controller, pad, integrator, placement = rig
    _reset(*rig)
    with contextlib.redirect_stdout(io.StringIO()):
        t0 = time.perf_counter()
        for k in range(steps):
            a = k // STEPS_PER_ACTION
            ang = 2.0 * np.pi * a / 40.0                      # 4 s で 1 周
            vel = 0.05 * np.array([np.cos(ang), np.sin(ang), 0.0])
            pad.state = pad_state(vel=vel)
            integrator.refresh()
            controller.update(integrator)
            mujoco.mj_step(model, data)
        t1 = time.perf_counter()
    return t1 - t0


def run_physics(rig, steps: int) -> float:
    model, data = rig[0], rig[1]
    _reset(*rig)
    t0 = time.perf_counter()
    for _ in range(steps):
        mujoco.mj_step(model, data)
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    t_setup = time.perf_counter()
    rig = _setup()
    t_setup = time.perf_counter() - t_setup
    model = rig[0]
    sim_s = args.steps * float(model.opt.timestep)
    out = {
        "mujoco": mujoco.__version__, "python": platform.python_version(), "cpu_count": os.cpu_count(),
        "scene": os.path.relpath(control.SCENE_PATH), "timestep": float(model.opt.timestep),
        "steps": args.steps, "sim_seconds": sim_s, "setup_seconds": round(t_setup, 3), "modes": {},
    }
    for name, fn in (("physics", run_physics), ("controller", run_controller)):
        walls = [fn(rig, args.steps) for _ in range(args.repeats)]
        med = statistics.median(walls)
        out["modes"][name] = {
            "wall_seconds": [round(w, 4) for w in walls],
            "median_wall_seconds": round(med, 4),
            "realtime_factor_median": round(sim_s / med, 2),
            "us_per_step_median": round(med / args.steps * 1e6, 1),
        }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
