"""Replay check for recorded episodes (C4, 2026-09-16).

タスク_テレオペシステムのデータ収集対応_v2.md §5: executing the recorded
actions in order through the same tracker + IK must reproduce the recorded
hand trajectory, and the replayed images must match the recorded ones.

Nothing here writes to the episode directory. Modes:

    step     per-physics-step x_des and gripper command (data.npz step_*).
             Checks that the start state and the recording are complete:
             the replay must match bit for bit (images up to GPU jitter).
    window   only the 20 Hz actions a policy would output:
             action_xyz[i] = x_des[i+1] - x_des[i], spread evenly over the
             25 steps of the window after frame i; gripper command of frame
             i+1 applied from the first step of that window.
    last_step, shifted
             deliberately wrong definitions (negative controls): only the
             last step's x_des change per window, and actions paired one
             window late. They must fail, or the check proves nothing.
"""
import dataclasses
import json
import pathlib

import mujoco
import numpy as np

from teleop import collect, recorder
from teleop.device import DeviceInput, DeviceState

MODES = ("step", "window", "last_step", "shifted")
WINDOW_TOL_M = 0.005          # plan C4: hand trajectory within 5 mm
IMAGE_TOL = 2                 # GPU jitter: +-1..2 LSB on a few pixels


@dataclasses.dataclass
class Episode:
    path: pathlib.Path
    meta: dict
    data: dict

    @property
    def n_frames(self) -> int:
        return int(self.meta["n_frames"])

    def image(self, camera: str, i: int) -> np.ndarray:
        return recorder.read_png(self.path / camera / f"{i:06d}.png")


def load_episode(path) -> Episode:
    path = pathlib.Path(path)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    with np.load(path / "data.npz") as npz:
        data = {k: npz[k] for k in npz.files}
    return Episode(path, meta, data)


class ScriptedPad(DeviceInput):
    """Feeds absolute x_des and a gripper-closed target to the controller,
    pressing the gripper button for one step whenever the target differs
    from the controller's state (the controller toggles on the rising
    edge, like the operator's square press). Exposes the x_cmd/reset
    surface reset_episode() uses on the real integrator."""

    def __init__(self, controller):
        self.controller = controller
        self.x_cmd = np.zeros(3)
        self.closed_target = False
        self._pressed = False

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def reset(self, pos) -> None:
        self.x_cmd = np.array(pos, dtype=float)

    def read(self) -> DeviceState:
        press = (self.closed_target != self.controller.gripper_closed) and not self._pressed
        self._pressed = press
        return DeviceState(pos=self.x_cmd.copy(), quat=np.array([1.0, 0.0, 0.0, 0.0]),
                           button_grip=press, button_clutch=True)


def _placement(meta) -> recorder.Placement:
    c = meta["cube_init"]
    return recorder.Placement(meta.get("placement_id"), int(meta["placement_seed"]),
                              float(c["x"]), float(c["y"]), float(c["yaw"]))


def _action_schedule(start, actions: np.ndarray, steps_per_action: int):
    """Each action (dx, dy, dz, rx, ry, rz, gripper +-1) spread evenly over
    steps_per_action physics steps; gripper applied from the first step."""
    pos = np.asarray(start, dtype=float).copy()
    for a in actions:
        begin = pos.copy()
        for j in range(1, steps_per_action + 1):
            pos = begin + a[:3] * (j / steps_per_action)
            yield pos, bool(a[6] > 0.0)


def _schedule(ep: Episode, mode: str, actions=None, frame_stride: int = 1):
    """Per-step (x_des, gripper_closed) targets for steps 1..n_steps."""
    d, every = ep.data, int(ep.meta["record_every"])
    if mode == "actions":  # external actions, e.g. read back from LeRobot
        yield from _action_schedule(d["x_des"][0], actions, frame_stride * every)
        return
    n_steps = int(ep.meta["n_steps"])
    if mode == "step":
        closed = d["step_gripper_cmd"] <= 0.5 * 255.0
        for k in range(1, n_steps + 1):
            yield d["step_x_des"][k], bool(closed[k])
        return
    x, g = d["x_des"], d["gripper_closed"]
    n = ep.n_frames
    acts = np.zeros((n - 1, 7))
    for i in range(n - 1):
        if mode == "window":
            acts[i, :3], grip = x[i + 1] - x[i], g[i + 1]
        elif mode == "last_step":
            s = d["step_x_des"]
            acts[i, :3], grip = s[(i + 1) * every] - s[(i + 1) * every - 1], g[i + 1]
        elif mode == "shifted":
            acts[i, :3] = x[i] - x[i - 1] if i > 0 else np.zeros(3)
            grip = g[i]
        else:
            raise ValueError(mode)
        acts[i, 6] = 1.0 if grip else -1.0
    yield from _action_schedule(x[0], acts, every)


def replay(ep: Episode, mode: str, renderer: mujoco.Renderer, model=None,
           actions=None, frame_stride: int = 1) -> dict:
    """Runs one replay; returns replayed frames (same fields as data.npz)
    and images per camera. mode "actions" executes `actions` (K, 7), each
    covering frame_stride raw frames (2 for 10 fps); frames are still
    captured at every raw frame."""
    model = model or mujoco.MjModel.from_xml_path(collect.app.SCENE_PATH)
    data = mujoco.MjData(model)
    controller = collect.make_collect_controller(model, data)
    pad = ScriptedPad(controller)
    start = collect.StartState(qpos=ep.data["start_qpos"], ctrl=ep.data["start_ctrl"],
                               q_des=ep.data["start_q_des"])
    collect.reset_episode(model, data, controller, pad, start, _placement(ep.meta))
    cameras = tuple(ep.meta["cameras"]["names"])
    sampler = recorder.FrameSampler(model, controller, renderer, cameras)
    every = int(ep.meta["record_every"])

    frames, images = [], {c: [] for c in cameras}

    def capture(step):
        f, imgs = sampler.capture(data, step)
        frames.append(f)
        for c, im in zip(cameras, imgs):
            images[c].append(im)

    capture(0)
    for k, (x_des, closed) in enumerate(_schedule(ep, mode, actions, frame_stride), start=1):
        pad.x_cmd = np.asarray(x_des, dtype=float)
        pad.closed_target = closed
        controller.update(pad)
        mujoco.mj_step(model, data)
        if k % every == 0:
            capture(k)
    out = {name: np.array([f[name] for f in frames]) for name in recorder.FRAME_FIELDS}
    out["images"] = images
    box = model.body(recorder.BOX_BODY).pos  # static body in the world
    out["success"] = recorder.cube_in_box(out["cube_pos"][-1], box)
    return out


def compare(ep: Episode, rep: dict, images: bool = True) -> dict:
    d = ep.data
    n = min(ep.n_frames, len(rep["ee_pos"]))  # 10 fps replays end at the last full 0.1 s
    ee_err = np.linalg.norm(rep["ee_pos"][:n] - d["ee_pos"][:n], axis=1)
    result = {
        "n_frames": n,
        "ee_err_max_m": float(ee_err.max()),
        "ee_err_mean_m": float(ee_err.mean()),
        "ee_err_final_m": float(ee_err[-1]),
        "joint_err_max_rad": float(np.abs(rep["joints"][:n] - d["joints"][:n]).max()),
        "x_des_err_max_m": float(np.abs(rep["x_des"][:n] - d["x_des"][:n]).max()),
        "gripper_mismatch_frames": int((rep["gripper_closed"][:n] != d["gripper_closed"][:n]).sum()),
        "cube_final_err_m": float(np.linalg.norm(rep["cube_pos"][n - 1] - d["cube_pos"][n - 1])),
        "success_recorded": bool(ep.meta["success"]),
        "success_replayed": bool(rep["success"]),
        "bit_exact_state": bool(np.array_equal(rep["joints"][:n], d["joints"][:n])
                                and np.array_equal(rep["ee_pos"][:n], d["ee_pos"][:n])),
    }
    if images:
        for cam, frames in rep["images"].items():
            worst, frac = 0, 0.0
            for i in range(n):
                diff = np.abs(frames[i].astype(int) - ep.image(cam, i).astype(int))
                worst = max(worst, int(diff.max()))
                frac = max(frac, float((diff.max(axis=2) > IMAGE_TOL).mean()))
            result[f"img_{cam}_max_diff"] = worst
            result[f"img_{cam}_worst_frame_fraction_over_tol"] = frac
    return result
