"""3 色の場面のシミュレーションの一式（B_提案書 §7 の SimRig）。

    rig = SimRig(render=True)          # render=False なら描画器を作らない（物理だけ。掲示板 0014 の条件）
    rig.reset(layout)                  # 開始姿勢を戻し、立方体を置き、settle_prefilled_s 回して静止させる。時刻 0
    rig.pad_read(vel, press)           # 入力の読み取り 1 回分（物理 steps_per_pad_read 手）。指令は DualSense と同じ入口
    rig.truth(target)                  # 台本・評価の真値（recovla.expert.script.Truth）
    rig.capture()                      # こま（記録器と同じ「同じ時点の」画像と状態）

指令の入口は流用元と同じ: pad.state = pad_state(vel=…, button_grip=…) → integrator.refresh() →
物理ステップごとに controller.update(integrator) → mj_step。台本・方策・誘発はこれ以外の経路で腕を動かさない。
接触は物理ステップごとに ContactMeter に渡し、こまの窓で OR する。
"""
import contextlib
import io
import os

import mujoco
import numpy as np

from recovla.common import config
from recovla.common.seeds import COLORS
from recovla.expert.script import Truth
from recovla.sim import contact, control, frames, scene
from recovla.sim.device import ScriptPad, pad_state

_CFG = config.load()


@contextlib.contextmanager
def quiet():
    """制御器（流用元のまま）の診断用の print を捨てる。流用元の呼ぶ側と同じ扱い。"""
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        yield


class SimRig:
    def __init__(self, render: bool = True, cameras=control.CAMERAS, cfg: dict = None):
        cfg = cfg or _CFG
        self.cfg = cfg
        self.model = scene.build_model("3cube")
        control.check_timestep(self.model)
        self.data = mujoco.MjData(self.model)
        self.timestep = float(self.model.opt.timestep)
        self.steps_per_read = int(cfg["sim"]["steps_per_pad_read"])
        self.record_every = int(cfg["sim"]["record_every"])
        retreat = np.array(cfg["expert"]["retreat_pose"], dtype=float)
        with contextlib.redirect_stdout(io.StringIO()):
            self.starts = {"home": control.settle_start_state(self.model),
                           "retreat": control.settle_start_state(self.model, pos=retreat)}
            self.controller = control.make_collect_controller(self.model, self.data)
        self.pad = ScriptPad()
        self.integrator = control.make_integrator(self.pad, self.model, self.controller)
        self.cameras = tuple(cameras)
        self.renderer = mujoco.Renderer(self.model, control.IMAGE_SIZE, control.IMAGE_SIZE) if render else None
        self.meter = contact.ContactMeter(self.model)
        self.scratch = mujoco.MjData(self.model)
        m = self.model
        self.box = frames.box_pos(m)
        self.hand_id = self.controller.hand_body_id
        self.cube_ids = np.array([m.body(frames.cube_body(c)).id for c in COLORS])
        adr = [scene.cube_qpos_adr(m, c) for c in COLORS]
        self.cube_vadr = np.array([v for _, v in adr])
        self.arm_qadr = np.array([m.joint(f"joint{i}").qposadr[0] for i in range(1, 8)])
        self.arm_vadr = np.array([m.joint(f"joint{i}").dofadr[0] for i in range(1, 8)])
        self.finger_qadr = np.array([m.joint(n).qposadr[0] for n in ("finger_joint1", "finger_joint2")])
        self.finger_vadr = np.array([m.joint(n).dofadr[0] for n in ("finger_joint1", "finger_joint2")])
        self.grip_act = self.controller.gripper_indices[0]
        self.step = 0                               # 物理ステップの番号（reset で 0）
        self.layout = None
        self._vel6 = np.zeros(6)

    def close(self) -> None:
        if self.renderer is not None:
            from recovla.sim import render
            render.close_renderer(self.renderer)
            self.renderer = None

    # ------------------------------------------------------------------ reset
    def reset(self, layout: scene.Layout) -> None:
        m, d = self.model, self.data
        start = self.starts[layout.start]
        control.reset_episode(m, d, self.controller, self.integrator, start, None)
        scene.apply_layout(m, d, layout)
        mujoco.mj_forward(m, d)
        self.pad.state = pad_state()
        self.integrator.refresh()
        n = int(round(float(self.cfg["scene"]["settle_prefilled_s"]) / self.timestep))
        with quiet():
            for _ in range(n):
                self.controller.update(self.integrator)
                mujoco.mj_step(m, d)
        d.time = 0.0
        # 開始状態の派生量（xpos・ヤコビアン）を今の qpos に揃える。mj_step の直後は 1 手前の値が残っており、
        # 制御器はそれを読む。揃えておけば、保存した開始状態から mj_setState + mj_forward で同じ列を再現できる
        mujoco.mj_forward(m, d)
        self.step = 0
        self.layout = layout
        self.meter.reset_window()

    # ------------------------------------------------------------ one pad read
    def pad_read(self, vel=None, press: bool = False, on_step=None) -> None:
        """入力の読み取り 1 回分。on_step(rig) は物理ステップごと（mj_step の後）に呼ぶ。"""
        m, d = self.model, self.data
        self.pad.state = pad_state(vel=np.zeros(3) if vel is None else np.asarray(vel, float),
                                   button_grip=bool(press))
        self.integrator.refresh()
        for _ in range(self.steps_per_read):
            self.controller.update(self.integrator)
            mujoco.mj_step(m, d)
            self.step += 1
            self.meter.on_step(d)
            if on_step is not None:
                on_step(self)

    # ------------------------------------------------------------------ truth
    def truth(self, target: str, data=None) -> Truth:
        d = self.data if data is None else data
        m = self.model
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.hand_id, self._vel6, 0)
        linvel = np.stack([d.qvel[v:v + 3] for v in self.cube_vadr]).copy()
        return Truth(
            t=float(d.time), target=COLORS.index(target),
            cube_pos=d.xpos[self.cube_ids].copy(), cube_quat=d.xquat[self.cube_ids].copy(), cube_linvel=linvel,
            fingers=d.qpos[self.finger_qadr].copy(), finger_vel=d.qvel[self.finger_vadr].copy(),
            gripper_closed=bool(self.controller.gripper_closed),
            hand_pos=d.xpos[self.hand_id].copy(), hand_vel=self._vel6[3:].copy(),
            fingertip=frames.fingertip_center(d, self.hand_id), x_cmd=self.integrator.x_cmd.copy(),
            box=self.box.copy())

    def cubes_in_box(self, data=None) -> tuple:
        d = self.data if data is None else data
        return tuple(c for c, i in zip(COLORS, self.cube_ids) if frames.in_box(d.xpos[i], self.box))

    # ---------------------------------------------------------------- capture
    def forward_scratch(self):
        """今の状態を複写した scratch に mj_forward する（流用元 FrameSampler と同じ。mocap は複写しない）。"""
        s = self.scratch
        np.copyto(s.qpos, self.data.qpos)
        np.copyto(s.qvel, self.data.qvel)
        np.copyto(s.ctrl, self.data.ctrl)
        s.time = self.data.time
        mujoco.mj_forward(self.model, s)
        return s

    def render(self, data=None) -> list:
        if self.renderer is None:
            return []
        d = self.scratch if data is None else data
        out = []
        for cam in self.cameras:
            self.renderer.update_scene(d, camera=cam)
            out.append(self.renderer.render().copy())
        return out
