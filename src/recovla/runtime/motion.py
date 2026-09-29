"""実行系の腕の動き（0108 の 2 の 2: IK と制限層は実行系の側に置く）。

行動の手先の速さ → x_cmd の積分（流用元の VelocityCommandIntegrator、物理の周期 500 Hz）→ 安全フィルタ（差し込み口）→
追従器と IK（流用元の TeleopControllerIK。ロボットだけの模型に**測った**関節角を入れて解く）→ 制限層（limitRate、1 kHz）→
関節の位置の指令（1 kHz 相当。500 Hz の 1 手ごとに 2 刻み）。

    mo = Motion(setup)
    mo.reset(joints)                    # 試行の始め（測った関節角から。IK の出発点・x_cmd を今の手先に合わせる）
    mo.set_velocity(vel)                # 手先の速さ [m/s]（行動の xyz / 0.1 s）。0 なら保持
    cmds = mo.step(joints)              # 500 Hz の 1 手ぶん: 関節の位置の指令 2 つ（1 kHz 相当）
    mo.hand_pose(q)                     # 測った関節角の順運動学（手先の位置と四元数）

IK は、次の手の出発点に制限層の出力（最後の指令）を使う（ずれが積み上がらないように、0107 の 8-2）。
"""
import contextlib
import io as _io

import mujoco
import numpy as np

from recovla.runtime import limiter as L
from recovla.runtime import robot_model
from recovla.sim import control
from recovla.sim.device import ScriptPad, pad_state

SUBSTEPS = 2                               # 500 Hz の 1 手あたりの 1 kHz の刻み


class Motion:
    def __init__(self, setup, limiter_enabled: bool = True, margin: float = 0.99):
        self.setup = setup
        self.model = robot_model.load(setup.robot_xml)
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)
        m = self.model
        self.arm_qadr = np.array([m.joint(f"joint{i}").qposadr[0] for i in range(1, 8)])
        self.arm_vadr = np.array([m.joint(f"joint{i}").dofadr[0] for i in range(1, 8)])
        self.finger_qadr = np.array([m.joint(n).qposadr[0] for n in ("finger_joint1", "finger_joint2")])
        with contextlib.redirect_stdout(_io.StringIO()):
            self.controller = control.make_collect_controller(m, self.data)    # q_neutral は HOME（流用元と同じ）
        self.controller.gripper_indices = []                                   # 指はハンドの口で動かす
        self.arm_act = np.array([a[0] for a in self.controller.arm])
        self.pad = ScriptPad()
        self.integrator = control.make_integrator(self.pad, m, self.controller)
        ws = setup.workspace
        self.integrator.workspace_x, self.integrator.workspace_y, self.integrator.workspace_z = \
            tuple(ws["x"]), tuple(ws["y"]), tuple(ws["z"])
        self.limiter = L.JointLimiter(margin=margin, q_min=self.controller.q_min, q_max=self.controller.q_max,
                                      enabled=limiter_enabled)
        self.vel = np.zeros(3)
        self.hand_id = self.controller.hand_body_id

    # ------------------------------------------------------------------ kinematics
    def _load_state(self, q, dq=None, width=None) -> None:
        d = self.data
        d.qpos[self.arm_qadr] = q
        d.qvel[:] = 0.0
        if dq is not None:
            d.qvel[self.arm_vadr] = dq
        if width is not None:
            d.qpos[self.finger_qadr] = 0.5 * float(width)
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)

    def hand_pose(self, q) -> tuple:
        """測った関節角の順運動学: 手先（hand の原点）の位置 (3,) と四元数 (4,)。"""
        self._load_state(q)
        return self.data.xpos[self.hand_id].copy(), self.data.xquat[self.hand_id].copy()

    def body_pose(self, q, body: str) -> tuple:
        self._load_state(q)
        b = self.model.body(body).id
        return self.data.xpos[b].copy(), self.data.xmat[b].reshape(3, 3).copy()

    # ------------------------------------------------------------------ control
    def reset(self, joints) -> None:
        """試行の始め。測った関節角から、IK の出発点・追従器・x_cmd を今の手先に揃える（control.reset_episode と同じ手順）。"""
        c, integ = self.controller, self.integrator
        self._load_state(joints.q, joints.dq)
        q0 = np.array(joints.q if joints.q_d is None else joints.q_d, float)   # 前の指令から続ける（libfranka の q_d）
        c.q_des = q0.copy()
        c.gripper_closed = False
        c.sync_target_to_hand()
        integ.reset(c.target_pos)
        c.device_ref_pos = integ.x_cmd.copy()
        c.target_ref_pos = c.target_pos.copy()
        c.device_ref_quat = control._IDENTITY_QUAT.copy()
        c.target_ref_quat = c.target_quat.copy()
        c.prev_clutch = True
        self.limiter.reset(q0)
        self.vel = np.zeros(3)
        self.pad.state = pad_state(vel=self.vel)
        integ.refresh()

    def set_velocity(self, vel) -> None:
        """手先の速さ [m/s]。流用元と同じく、入力の読み取り（refresh）の時点の値を次の読み取りまで保つ。"""
        self.vel = np.asarray(vel, float).copy()
        self.pad.state = pad_state(vel=self.vel)
        self.integrator.refresh()

    def set_command_filter(self, fn) -> None:
        """安全フィルタの差し込み口: fn(x_cmd, step) -> step（流用元の integrator.command_filter と同じ）。"""
        self.integrator.command_filter = fn

    @property
    def x_cmd(self) -> np.ndarray:
        return self.integrator.x_cmd.copy()

    def step(self, joints) -> list:
        """500 Hz の 1 手: 指令の姿勢（制限層の最後の出力）で IK を 1 回解き、制限層で 1 kHz の指令を 2 つ作る。

        IK は指令の姿勢の上で解く（実機の関節の位置の制御は 1 kHz で指令をよく追うので、関節の位置の指令の経路では普通の
        作り）。測った関節角で解くと、制限層の加速度の上限の遅れがサーボの遅れと重なって振動した（学習用のシード 59012、
        待機位置から始める配置）。測った関節角は、指令が先へ行き過ぎないための綱（leash、流用元と同じ 0.10 rad）にだけ使う。"""
        self._load_state(self.limiter.q)
        self.controller.update(self.integrator)                    # 追従器 → IK（q_des を data.ctrl に書く）
        leash = self.controller.q_des_leash
        q_ik = np.clip(self.data.ctrl[self.arm_act], joints.q - leash, joints.q + leash)
        v_des = (q_ik - self.limiter.q) / self.dt                  # 500 Hz の 1 手で q_ik に着く速さ
        out = [self.limiter.step(v_des) for _ in range(SUBSTEPS)]
        self.controller.q_des = out[-1].copy()                     # IK の次の出発点は制限層の出力
        return out
