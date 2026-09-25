"""Raw episode recording (流用元 teleop/recorder.py、B_提案書 §2.2).

変えたこと: 数値は configs から読む（値は流用元と同じ。立方体 1 個の配置の範囲は configs/g0.yaml）。
流用元の収録フォルダの既定値（DEFAULT_RAW_DIR）、フォルダを数え上げる scan_raw、収録の記録
SessionLog は持ち込まない（台帳・03_収録 との結合を外す）。

Layout of one saved episode:

    <day or run dir>/ep_000123/
        overhead/000000.png ...    256x256 RGB, one per frame
        wrist/000000.png ...
        data.npz                   per-frame and per-step arrays (below)
        meta.json

An episode is written to ep_XXXXXX.partial while recording and renamed on
save; a discard deletes the .partial directory.

Frames: frame i is taken at physics step i * RECORD_EVERY (frame 0 = the
reset state, before any step). Images and state come from the same instant:
the live MjData is copied into a scratch MjData and mj_forward'ed there,
because right after mj_step the live data's positions (xpos, cameras) are
one step older than its qpos, and calling mj_forward on the live data would
change the controlled trajectory (it feeds the next IK solve).

Actions are not stored pre-cut. x_des (controller.desired_pos) is stored
absolute at every frame and every physics step; the 20 Hz action paired
with frame i is the sum over the window that FOLLOWS it:
    action_xyz[i] = x_des[i + 1] - x_des[i]          (i = 0 .. n_frames - 2)
    action_gripper[i] = gripper command at frame i + 1
matching LeRobot/LIBERO, where the action at t is what is executed after
observing t. The last frame has no action.
"""
import dataclasses
import datetime
import hashlib
import json
import os
import pathlib
import queue
import shutil
import threading
from typing import Optional

import cv2
import mujoco
import numpy as np

from recovla.common import config

_CFG = config.load("g0")
_G0 = _CFG["g0"]
_PLACE = _G0["placement"]

SCHEMA_VERSION = 1
INSTRUCTION = _G0["instruction"]

# Initial cube range (B-2) and fixed training placements (表3-5: 8-10
# placements x repetitions, not a new random placement per episode).
N_PLACEMENTS = int(_PLACE["n"])
CUBE_X_RANGE = tuple(float(v) for v in _PLACE["x_range"])
CUBE_Y_RANGE = tuple(float(v) for v in _PLACE["y_range"])
_YAW_HALF = np.pi / (180.0 / float(_PLACE["yaw_half_range_deg"]))    # 45 deg -> np.pi / 4.0 exactly as the source
CUBE_YAW_RANGE = (-_YAW_HALF, _YAW_HALF)
CUBE_HALF = float(_CFG["scene"]["cube_size"]) / 2.0
PLACEMENT_MIN_SEPARATION = float(_PLACE["min_separation"])   # [m] between training placements
TRAIN_SEED_LIMIT = int(_PLACE["train_seed_limit"])           # training seeds 0-2999, evaluation >= 100000
CUBE_BODY = _PLACE["cube_body"]

# Success = cube resting inside the goal box: same volume as the source's
# CubeGame._cube_in_box (inner_half_xy=0.05, wall_top_z=0.06).
BOX_BODY = "goal_box"
BOX_INNER_HALF_XY = float(_CFG["scene"]["box"]["success_inner_half"])
BOX_WALL_TOP_Z = float(_CFG["scene"]["box"]["wall_top_z"])

PNG_COMPRESSION = int(_CFG["sim"]["png_compression"])        # fast; PNG is lossless at any level


# ------------------------------------------------------------- placements

@dataclasses.dataclass(frozen=True)
class Placement:
    placement_id: Optional[int]   # None for evaluation seeds
    seed: int
    x: float
    y: float
    yaw: float

    def quat(self) -> np.ndarray:
        return np.array([np.cos(self.yaw / 2), 0.0, 0.0, np.sin(self.yaw / 2)])


def placement_from_seed(seed: int, placement_id: Optional[int] = None) -> Placement:
    """One uniform draw from the initial range. Evaluation can call this
    with seeds >= 100000 for placements never used in training."""
    rng = np.random.default_rng(seed)
    return Placement(placement_id, int(seed), float(rng.uniform(*CUBE_X_RANGE)),
                     float(rng.uniform(*CUBE_Y_RANGE)),
                     float(rng.uniform(*CUBE_YAW_RANGE)))


def training_placements(n: int = N_PLACEMENTS,
                        min_separation: float = PLACEMENT_MIN_SEPARATION) -> list:
    """The fixed training placements: seeds 0, 1, 2, ... drawn in order,
    keeping a draw only if it is at least min_separation from every kept
    one, until n are kept. Deterministic."""
    kept = []
    for seed in range(TRAIN_SEED_LIMIT):
        p = placement_from_seed(seed)
        if all(np.hypot(p.x - q.x, p.y - q.y) >= min_separation for q in kept):
            kept.append(dataclasses.replace(p, placement_id=len(kept)))
            if len(kept) == n:
                return kept
    raise RuntimeError("could not find enough separated training placements")


def apply_placement(model, data, placement: Placement) -> None:
    body = model.body(CUBE_BODY).id
    jnt = model.body_jntadr[body]
    qadr, vadr = model.jnt_qposadr[jnt], model.jnt_dofadr[jnt]
    data.qpos[qadr:qadr + 3] = [placement.x, placement.y, CUBE_HALF]
    data.qpos[qadr + 3:qadr + 7] = placement.quat()
    data.qvel[vadr:vadr + 6] = 0.0


def cube_in_box(cube_pos, box_pos) -> bool:
    rel = np.asarray(cube_pos, float) - np.asarray(box_pos, float)
    return bool(abs(rel[0]) < BOX_INNER_HALF_XY and abs(rel[1]) < BOX_INNER_HALF_XY
                and 0.0 < rel[2] < BOX_WALL_TOP_Z)


# ------------------------------------------------------------- placement order

def next_placement(placements: list, saved: dict) -> Placement:
    """Fewest saved episodes first, lowest id on ties: keeps the repetition
    counts balanced across sessions."""
    return min(placements, key=lambda p: (saved.get(p.placement_id, 0), p.placement_id))


def _now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def file_sha256(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


# -------------------------------------------------------------- sampling

FRAME_FIELDS = {
    # name: (shape per frame, dtype)
    "step": ((), np.int64),               # physics steps since reset
    "sim_time": ((), np.float64),         # actual simulation time [s]
    "ee_pos": ((3,), np.float64),         # hand body origin, world [m]
    "ee_quat": ((4,), np.float64),        # hand body, world, w>=0
    "ee_axisangle": ((3,), np.float64),   # same orientation as axis*angle
    "fingers": ((2,), np.float64),        # finger_joint1, finger_joint2 [m]
    "joints": ((7,), np.float64),         # joint1..7 [rad]
    "joint_vel": ((7,), np.float64),      # [rad/s]
    "x_des": ((3,), np.float64),          # controller.desired_pos (action source)
    "tracker_target": ((3,), np.float64), # controller.target_pos (after tracker)
    "gripper_cmd": ((), np.float64),      # actuator ctrl 0..255 (0 = closed)
    "gripper_closed": ((), np.bool_),
    "cube_pos": ((3,), np.float64),       # privileged, for labels/success
    "cube_quat": ((4,), np.float64),
}


class FrameSampler:
    """Same-instant images and state (see module docstring)."""

    def __init__(self, model, controller, renderer: mujoco.Renderer, cameras):
        self.model = model
        self.controller = controller
        self.renderer = renderer
        self.cameras = tuple(cameras)
        self.scratch = mujoco.MjData(model)
        self.arm_qadr = np.array([model.joint(f"joint{i}").qposadr[0] for i in range(1, 8)])
        self.arm_vadr = np.array([model.joint(f"joint{i}").dofadr[0] for i in range(1, 8)])
        self.finger_qadr = np.array([model.joint(n).qposadr[0]
                                     for n in ("finger_joint1", "finger_joint2")])
        self.cube = model.body(CUBE_BODY).id
        self.grip = controller.gripper_indices[0]

    def capture(self, data, step: int):
        s = self.scratch
        np.copyto(s.qpos, data.qpos)
        np.copyto(s.qvel, data.qvel)
        # mocap is deliberately NOT copied: the scratch keeps the model's
        # default mocap pose. The mocap target sphere is hidden from cameras
        # (geom group 5) and has no physics, yet with shadows + reflections
        # on, its position still shifted ~19 overhead-camera pixels at shadow
        # edges (measured 2026-09-16), which would leak the tracker target
        # into the recorded observation.
        np.copyto(s.ctrl, data.ctrl)
        s.time = data.time
        mujoco.mj_forward(self.model, s)

        h = self.controller.hand_body_id
        quat = s.xquat[h].copy()
        if quat[0] < 0.0:
            quat = -quat
        axisangle = np.zeros(3)
        mujoco.mju_quat2Vel(axisangle, quat, 1.0)
        frame = {
            "step": step,
            "sim_time": s.time,
            "ee_pos": s.xpos[h].copy(),
            "ee_quat": quat,
            "ee_axisangle": axisangle,
            "fingers": s.qpos[self.finger_qadr].copy(),
            "joints": s.qpos[self.arm_qadr].copy(),
            "joint_vel": s.qvel[self.arm_vadr].copy(),
            "x_des": self.controller.desired_pos.copy(),
            "tracker_target": self.controller.target_pos.copy(),
            "gripper_cmd": float(data.ctrl[self.grip]),
            "gripper_closed": bool(self.controller.gripper_closed),
            "cube_pos": s.xpos[self.cube].copy(),
            "cube_quat": s.xquat[self.cube].copy(),
        }
        images = []
        for cam in self.cameras:
            self.renderer.update_scene(s, camera=cam)
            images.append(self.renderer.render().copy())
        return frame, images


# ---------------------------------------------------------------- writer

def write_png(path, rgb) -> None:
    """cv2.imwrite cannot open non-ASCII paths on Windows (it failed on
    03_収録, 2026-09-16), so encode in memory and write with Python."""
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
    if not ok:
        raise RuntimeError(f"PNG encoding failed for {path}")
    pathlib.Path(path).write_bytes(buf.tobytes())


def read_png(path) -> np.ndarray:
    """RGB image; the non-ASCII-path-safe counterpart of write_png."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def episode_dir_name(episode_id: int) -> str:
    return f"ep_{int(episode_id):06d}"


class EpisodeWriter:
    """Streams PNGs to <day>/ep_XXXXXX.partial on a background thread
    (PNG encoding releases the GIL) so recording never stalls the control
    loop; buffers frame/step arrays in memory.

    episode_id=None (the collection session since the episode ledger, 2026-09-17): the id is assigned only
    when saving, so the recording goes to <day>/ep_pending_<time>_<pid>.partial (not an ep_N name: never
    counted, converted or duplicate-checked) and finalize(..., episode_id=N) renames it to ep_N; after a
    failed save mark_failed(N) renames it to ep_N.partial so the used id stays visible."""

    def __init__(self, root, episode_id, day: str, cameras, record_every: int,
                 nominal_dt: float):
        self.episode_id = episode_id
        self.cameras = tuple(cameras)
        self.record_every = record_every
        self.nominal_dt = nominal_dt
        self.day_dir = pathlib.Path(root) / day
        self.pending = episode_id is None
        if self.pending:
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            self.final_dir = None
            self.dir = self.day_dir / f"ep_pending_{stamp}_{os.getpid()}.partial"
            if self.dir.exists():
                raise FileExistsError(self.dir)
        else:
            self.final_dir = self.day_dir / episode_dir_name(episode_id)
            self.dir = self.final_dir.with_name(self.final_dir.name + ".partial")
            if self.final_dir.exists() or self.dir.exists():
                raise FileExistsError(self.final_dir)
        for cam in self.cameras:
            (self.dir / cam).mkdir(parents=True)
        self.frames = []
        self.step_x_des = []
        self.step_gripper_cmd = []
        self.frozen_frames = {cam: 0 for cam in self.cameras}   # see _count_frozen
        self._previous_images = None
        self._queue = queue.Queue()
        self._error = None
        self._thread = threading.Thread(target=self._work, daemon=True)
        self._thread.start()

    def _work(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                path, rgb = item
                write_png(path, rgb)
            except Exception as exc:  # noqa: BLE001
                if self._error is None:
                    self._error = repr(exc)
            finally:
                self._queue.task_done()

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def add_frame(self, frame: dict, images) -> None:
        i = len(self.frames)
        self.frames.append(frame)
        for cam, rgb in zip(self.cameras, images):
            self._queue.put((self.dir / cam / f"{i:06d}.png", rgb))
        self._count_frozen(images)

    def _count_frozen(self, images) -> None:
        """Material for the camera-stall label: how often a view was pixel-identical to the frame
        before it. A stalled camera keeps sending the same picture, while the hand always moves in
        this task. Counted only, never judged here: the technical defects of 本冊 表 3-10 are
        proposed by the machine and accepted or rejected by a person."""
        if self._previous_images is not None:
            for cam, rgb, last in zip(self.cameras, images, self._previous_images):
                if np.array_equal(rgb, last):
                    self.frozen_frames[cam] += 1
        self._previous_images = list(images)

    def add_step(self, x_des, gripper_cmd: float) -> None:
        """Call at reset (index 0) and after every physics step."""
        self.step_x_des.append(np.array(x_des, dtype=np.float64))
        self.step_gripper_cmd.append(float(gripper_cmd))

    def _stop_thread(self) -> None:
        self._queue.join()
        self._queue.put(None)
        self._thread.join()

    def abandon(self) -> None:
        """Stop the PNG thread and keep whatever is on disk (a save that did not complete)."""
        if self._thread.is_alive():
            self._stop_thread()

    def discard(self) -> None:
        self._stop_thread()
        shutil.rmtree(self.dir)

    def mark_failed(self, episode_id: int) -> pathlib.Path:
        """After a failed save of a pending recording: keep it as ep_N.partial (best effort)."""
        if not self.pending or not self.dir.name.startswith("ep_pending_") or not self.dir.exists():
            return self.dir
        target = self.day_dir / (episode_dir_name(episode_id) + ".partial")
        try:
            os.replace(self.dir, target)
            self.dir = target
        except OSError:
            pass
        return self.dir

    def finalize(self, n_keep: int, meta: dict,
                 extra_arrays: Optional[dict] = None, episode_id=None) -> pathlib.Path:
        """Keep frames [0, n_keep) and steps up to the last kept frame, write
        data.npz (+ extra_arrays, e.g. the start state) and meta.json, rename
        .partial -> final. A pending recording needs episode_id here."""
        if self.pending:
            if episode_id is None:
                raise ValueError("a pending recording needs the episode_id assigned at save time")
            self.episode_id = int(episode_id)
            final = self.day_dir / episode_dir_name(episode_id)
            if final.exists() or final.with_name(final.name + ".partial").exists():
                raise FileExistsError(final)
            self.final_dir = final
        self._stop_thread()
        if self._error:
            raise RuntimeError(f"episode {self.episode_id}: {self._error}")
        n_keep = max(1, min(n_keep, len(self.frames)))
        for cam in self.cameras:
            for i in range(n_keep, len(self.frames)):
                (self.dir / cam / f"{i:06d}.png").unlink(missing_ok=True)
        frames = self.frames[:n_keep]
        arrays = {}
        for name, (shape, dtype) in FRAME_FIELDS.items():
            arrays[name] = np.array([f[name] for f in frames], dtype=dtype).reshape(
                (n_keep,) + shape)
        # nominal 0.05 s grid (表3-7); sim_time holds the actual clock
        arrays["timestamp"] = np.arange(n_keep, dtype=np.float64) * self.nominal_dt
        n_steps = (n_keep - 1) * self.record_every
        arrays["step_x_des"] = np.array(self.step_x_des[:n_steps + 1], dtype=np.float64)
        arrays["step_gripper_cmd"] = np.array(self.step_gripper_cmd[:n_steps + 1],
                                              dtype=np.float64)
        arrays.update(extra_arrays or {})
        np.savez(self.dir / "data.npz", **arrays)
        meta = dict(meta, n_frames=n_keep, n_steps=n_steps)
        (self.dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(self.dir, self.final_dir)
        return self.final_dir
