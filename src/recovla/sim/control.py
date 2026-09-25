"""制御ループの部品（流用元 teleop/app.py と teleop/collect.py から、B_提案書 §2.2）。

流用元から持ってきたもの（処理は変えていない）:
  app.py      make_controller（DLS の調整値の唯一の定義）、apply_home_pose、HOME_QPOS
  collect.py  VelocityCommandIntegrator、make_collect_controller、make_integrator、StartState、
              settle_start_state、reset_episode と定数

変えたこと:
  - 数値は configs/default.yaml（sim・controller）から読む。値は流用元と同じ
    （tests/test_c_port.py が予備実験の raw の meta.json と照合する）
  - 環境変数 TELEOP_CONTROLLER・TELEOP_DEVICE への依存を外した（常に DLS、姿勢は固定）
  - CollectSession（台帳つきの収録の状態機械）・OperatorView（操作者の画面）・main は持ち込まない

Signal path (flow unchanged from the source):

    pad (ScriptPad) -> vel
      -> VelocityCommandIntegrator (x_cmd += vel * timestep, per physics step)
      -> controller clutch (always engaged, scale 1.0) -> desired_pos = x_des
      -> tracker -> IK -> PD
"""
import contextlib
import dataclasses
import io

import mujoco
import numpy as np

from recovla.common import config
from recovla.sim.controller_ik import TeleopControllerIK
from recovla.sim.device import DeviceInput, DeviceState

CFG = config.load()
_SIM = CFG["sim"]

SCENE_PATH = str(config.path(CFG["paths"]["scene_g0"]))     # Step C は立方体 1 個の場面のまま
HOME_QPOS = [float(q) for q in _SIM["home_qpos"]]
COLLECT_WORKSPACE = dict(
    workspace_x=tuple(_SIM["workspace"]["x"]),
    workspace_y=tuple(_SIM["workspace"]["y"]),
    workspace_z=tuple(_SIM["workspace"]["z"]),
)
START_POS = np.array(_SIM["start_pos"], dtype=float)
SETTLE_SECONDS = float(_SIM["settle_s"])
FINGER_OPEN = float(_SIM["finger_open"])     # finger_joint1/2 range upper limit [m]
RECORD_EVERY = int(_SIM["record_every"])     # physics steps per 20 Hz window (500 / 20)
IMAGE_SIZE = int(_SIM["image_size"])
CAMERAS = tuple(_SIM["cameras"])
FINGERTIP_OFFSET = float(_SIM["fingertip_offset"])   # hand origin -> fingertip centre along hand z [m]
TABLE_TOP_Z = float(_SIM["table_top_z"])
STEPS_PER_PAD_READ = int(_SIM["steps_per_pad_read"])

_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


def check_timestep(model) -> None:
    """模型の timestep が設定（sim.timestep）と違えば止める。"""
    if float(model.opt.timestep) != float(_SIM["timestep"]):
        raise RuntimeError(f"model timestep {model.opt.timestep} != configs sim.timestep {_SIM['timestep']}")


# ------------------------------------------------------------------ controller (app.py)

def make_controller(model, data, **overrides):
    """The DLS teleop controller with the tuning of configs controller: (source: app.make_controller,
    dls branch, plus the collection overrides of collect.make_collect_controller). overrides replace
    individual keyword arguments (tests only)."""
    c = CFG["controller"]
    kwargs = dict(
        ik_gain=float(c["ik_gain"]),
        max_joint_step=float(c["max_joint_step"]),
        rot_weight=float(c["rot_weight"]),
        lock_orientation=bool(c["lock_orientation"]),
        null_gain=float(c["null_gain"]),
        ik_damping=float(c["ik_damping"]),
        tracker_omega=float(c["tracker_omega"]),
        max_target_vel=float(c["max_target_vel"]),
        q_des_leash=float(c["q_des_leash"]),
        max_dx_norm=float(c["max_dx_norm"]),
        scale=float(c["scale"]),
        **COLLECT_WORKSPACE,
    )
    kwargs.update(overrides)
    return TeleopControllerIK(model, data, **kwargs)


def apply_home_pose(model, data) -> None:
    """Write HOME_QPOS into the arm joints and refresh derived quantities.

    Only the 7 arm joints are touched by name, so this is independent of
    the qpos layout of the cubes' freejoints.
    """
    for i, q in enumerate(HOME_QPOS, start=1):
        data.qpos[model.joint(f"joint{i}").qposadr[0]] = q
    mujoco.mj_forward(model, data)


# ---------------------------------------------------------------- x_des (collect.py)

class VelocityCommandIntegrator(DeviceInput):
    """Turns the pad's velocity command into the absolute position the
    controller's clutch expects, advancing once per controller.update().

    refresh() polls the real pad (once per main-loop iteration); read() is
    what controller.update() calls (once per physics step) and integrates
    the latest velocity over one physics timestep.
    """

    def __init__(self, inner: DeviceInput, timestep: float,
                 workspace_x, workspace_y, workspace_z):
        self.inner = inner
        self.timestep = float(timestep)
        self.workspace_x = workspace_x
        self.workspace_y = workspace_y
        self.workspace_z = workspace_z
        self.x_cmd = np.zeros(3)
        self.latest = DeviceState(pos=np.zeros(3), quat=_IDENTITY_QUAT.copy(),
                                  button_grip=False, button_clutch=True,
                                  vel=np.zeros(3))
        # False: the arm ignores the pad (zero velocity, gripper button
        # ignored) while buttons are still read via refresh().
        self.enabled = True

    def start(self) -> None:
        self.inner.start()

    def stop(self) -> None:
        self.inner.stop()

    def reset(self, pos) -> None:
        self.x_cmd = self._clamp(np.array(pos, dtype=float))

    def refresh(self) -> DeviceState:
        self.latest = self.inner.read()
        return self.latest

    def read(self) -> DeviceState:
        s = self.latest
        if self.enabled and s.vel is not None:
            self.x_cmd = self._clamp(self.x_cmd + np.asarray(s.vel, dtype=float)
                                     * self.timestep)
        return DeviceState(pos=self.x_cmd.copy(), quat=_IDENTITY_QUAT.copy(),
                           button_grip=bool(self.enabled and s.button_grip),
                           button_clutch=True)

    def _clamp(self, pos: np.ndarray) -> np.ndarray:
        pos[0] = np.clip(pos[0], *self.workspace_x)
        pos[1] = np.clip(pos[1], *self.workspace_y)
        pos[2] = np.clip(pos[2], *self.workspace_z)
        return pos


def make_collect_controller(model, data):
    """make_controller's tracker/IK parameters with the collection-only
    clutch scale, orientation lock and workspace (all from the config).

    Writes HOME_QPOS into data first: the IK's null-space posture target
    q_neutral is the joint state at construction, and a fresh MjData is the
    upright all-zero pose (callers reset the episode state afterwards anyway)."""
    apply_home_pose(model, data)
    controller = make_controller(model, data)
    controller.ctrl_dt = float(model.opt.timestep)
    return controller


def make_integrator(device: DeviceInput, model, controller):
    return VelocityCommandIntegrator(
        device, model.opt.timestep, controller.workspace_x,
        controller.workspace_y, controller.workspace_z)


# ---------------------------------------------------------- start state (collect.py)

@dataclasses.dataclass
class StartState:
    qpos: np.ndarray
    ctrl: np.ndarray
    q_des: np.ndarray


class _Hold(DeviceInput):
    """Clutch released: the controller keeps whatever desired_pos is set."""

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def read(self) -> DeviceState:
        return DeviceState(pos=np.zeros(3), quat=_IDENTITY_QUAT.copy(),
                           button_grip=False, button_clutch=False)


def _set_fingers(model, data, opening: float) -> None:
    for name in ("finger_joint1", "finger_joint2"):
        data.qpos[model.joint(name).qposadr[0]] = opening


def _open_gripper(controller) -> None:
    for idx in controller.gripper_indices:
        controller.data.ctrl[idx] = controller.model.actuator_ctrlrange[idx][1]
    controller.gripper_closed = False


def settle_start_state(model, seconds: float = SETTLE_SECONDS,
                       tol: float = 0.010, qvel_tol: float = 1e-3, pos=None) -> StartState:
    """From HOME_QPOS, drive the hand to START_POS with the collection
    controller (tracker + IK at the physics rate, gripper open) and let it
    come to rest. The resulting qpos/ctrl/q_des include the servos' steady-
    state gravity offset, so an episode reset to them starts motionless.

    tol is loose on purpose: the unchanged DLS IK leaves a static offset
    between desired_pos and the hand (its null-space posture term is not
    exactly null under damping; measured 2026-09-16: 5 mm at START_POS,
    5-17 mm over the collection workspace, mostly +z, 0 with null_gain=0).
    The IK is not to be modified, so this only checks the arm came to rest
    near START_POS.

    pos (Step D): another hand position to settle at (the retreat pose of the 3-cube scene); None = START_POS,
    exactly the source's behaviour."""
    goal = START_POS if pos is None else np.asarray(pos, dtype=float)
    data = mujoco.MjData(model)
    apply_home_pose(model, data)
    _set_fingers(model, data, FINGER_OPEN)
    mujoco.mj_forward(model, data)
    with contextlib.redirect_stdout(io.StringIO()):  # IK diagnostic log
        controller = make_collect_controller(model, data)
        _open_gripper(controller)
        controller.sync_target_to_hand()
        controller.desired_pos = controller._clamp_workspace(goal.copy())
        hold = _Hold()
        for _ in range(int(round(seconds / model.opt.timestep))):
            controller.update(hold)
            mujoco.mj_step(model, data)
    err = float(np.linalg.norm(data.xpos[controller.hand_body_id] - goal))
    speed = float(np.linalg.norm(data.qvel[controller.dof_indices]))
    if err > tol or speed > qvel_tol:
        raise RuntimeError(
            f"start state did not settle after {seconds} s: hand "
            f"{err * 1000:.2f} mm from {goal}, |qvel| {speed:.2e} rad/s")
    return StartState(qpos=data.qpos.copy(), ctrl=data.ctrl.copy(),
                      q_des=controller.q_des.copy())


def reset_episode(model, data, controller, integrator, start: StartState,
                  placement=None) -> None:
    """Restore the settled start state exactly (cube at `placement`, a
    recorder.Placement, if given) and re-anchor x_des to it."""
    from recovla.record import recorder          # recorder imports this module's constants
    data.qpos[:] = start.qpos
    data.qvel[:] = 0.0
    data.qacc_warmstart[:] = 0.0
    data.ctrl[:] = start.ctrl
    data.time = 0.0
    if placement is not None:
        recorder.apply_placement(model, data, placement)
    mujoco.mj_forward(model, data)
    controller.q_des = start.q_des.copy()
    controller.gripper_closed = False
    controller.sync_target_to_hand()
    integrator.reset(controller.target_pos)
    # Engage the clutch here with identical references. Left to the first
    # update(), the reference would be grabbed after the integrator had
    # already advanced one step, leaving desired_pos one step (vel * dt)
    # off x_cmd for the whole episode.
    controller.device_ref_pos = integrator.x_cmd.copy()
    controller.target_ref_pos = controller.target_pos.copy()
    controller.device_ref_quat = _IDENTITY_QUAT.copy()
    controller.target_ref_quat = controller.target_quat.copy()
    controller.prev_clutch = True
