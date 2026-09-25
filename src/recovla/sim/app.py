"""Teleop application entry point.

Phase D changes:
- Device selection via the TELEOP_DEVICE environment variable:
  "mouse" (default) / "touch" (real Touch 3D via pyOpenHaptics) / "fake"
  (hardware-free circular motion, for smoke tests).
- Bug fix: _set_body_pose previously indexed data.qpos with body_jntadr,
  which is a JOINT ID, not a qpos address. Freejoint qpos addresses must be
  resolved through model.jnt_qposadr[model.body_jntadr[body_id]].
- Window ownership: only the app (via cv2.destroyAllWindows in the finally
  block) destroys OpenCV windows. MouseKeyboardInput shares the camera's
  "WRIST VIEW" window for its mouse callback; in touch/fake mode no mouse
  window is involved at all.

Phase E scaffold (disabled by default):
- Optional haptic force feedback: a clamped spring on the tracking error,
  F_world = -TELEOP_FORCE_GAIN * (target_pos - hand_pos), sent to the
  device if it exposes set_force(). Gain 0.0 keeps it off; tune on real
  hardware only (see PLAN.md section 7).

Phase F (2026-07-11), vibration/lag fixes:
- teleop/filters.py: FilterAugment (One Euro) wraps touch/fake devices by
  default, smoothing the raw stylus position before it reaches the clutch
  logic. Toggle with TELEOP_FILTER (see env vars below).
- LoopRateMonitor below: PLAN.md's "~1 kHz servo thread" describes only
  how fast the OpenHaptics callback refreshes its shared snapshot
  (teleop/touch_device.py _servo). TouchDeviceInput.read() just copies
  that snapshot once per call, and this main loop calls it once per
  iteration -- so the rate FilterAugment actually sees is this loop's
  wall-clock rate, which is bounded by mj_step + camera.render + the
  fixed time.sleep(0.002) below, not by the device's internal 1 kHz.
  LoopRateMonitor prints the real achieved rate periodically so filter
  and tracker parameters can be tuned against it.

Phase G (2026-07-12), game layer:
- teleop/game.py: CubeGame owns cube lifecycle (respawn, fall-catch,
  scoring, round timer, HUD overlay) once a round is active. app.py's own
  _reset_cubes/_check_cube_fall are NOT called from the main loop anymore
  -- having two independent systems write the same cube qpos was judged
  an unnecessary risk, so CubeGame is now the single owner during
  gameplay. The old functions are kept (unused by main()) only because
  existing tests exercise them directly; do not wire them back into the
  loop alongside CubeGame.
- 'r' now means "start/restart a round" (CubeGame.update), not just
  "respawn cubes".
- 'h' (home_reset) calls CubeGame.reset_for_home(), which respawns cubes
  AND drops the round to IDLE/score 0. Rationale: 'h' teleports the arm,
  so leaving a round timer running against cubes that just moved out from
  under the operator would be confusing mid-demo. If you want 'h' to
  preserve an in-progress round, change the call in home_reset() to
  game._reset_all_cubes() instead.

Phase H (2026-07-11), optional constraint-aware IK:
- teleop/controller_qp.py: TeleopControllerQP, a drop-in alternative to
  TeleopControllerIK that enforces joint velocity/position limits as hard
  constraints inside a box-constrained QP instead of clipping the DLS
  solution after the fact. Selected via TELEOP_CONTROLLER=qp (default:
  dls, i.e. the existing behavior is unchanged). See make_controller()
  below and controller_qp.py's module docstring for the math.

Phase H (2026-08-02), DualSense integration (see TASK_dualsense_v3.md and
PLAN.md phase H for the full rationale):
- teleop/dualsense_device.py: DualSenseInput, a second real device
  alongside touch, selected via TELEOP_DEVICE=dualsense. Two swappable
  HID backends (pydualsense full-feature / pygame axes+buttons-only
  fallback) behind one interface, plus a hardware-free _FakeBackend /
  FakeDualSenseInput pair used by tests. Wrapped with
  KeyboardButtonAugment same as touch/fake, but deliberately WITHOUT
  FilterAugment (see §12 in the task doc: the stick output is already an
  integrated, smooth signal, and touchpad-absolute input carries finger
  noise FilterAugment was never tuned for).
- _lock_orientation()/_device_kind(): mouse/touch/fake keep
  lock_orientation=True (byte-identical to before this phase); dualsense
  defaults to False so the right stick can drive hand orientation, unless
  TELEOP_LOCK_ORIENTATION overrides it either way.
- teleop/pad_feedback.py: pure functions turning IK health (manipulability
  + joint-limit margin) into a lightbar color and turning grip
  force/health into adaptive-trigger resistance targets. Real-hardware
  tuning (colors, force feel, BT vs USB) is not done here -- no DualSense
  unit is available yet; see PLAN.md phase H.

Phase I (2026-08-02), shelf pick-and-place task:
- teleop/shelf_task.py: ShelfTask, a duck-typed alternative to
  teleop/game.py's CubeGame (same update/reset_for_home/draw_overlay
  surface), selected via TELEOP_TASK=shelf against a new scene
  (assets/panda/shelf_scene.xml). TELEOP_TASK=cube (default) is the
  existing cube-drop game, unchanged byte-for-byte. teleop_scene.xml and
  teleop/game.py are NOT modified by this phase.

Env vars:
  TELEOP_DEVICE      mouse | touch | fake | dualsense   (default: mouse)
  TELEOP_FILTER      on | off  (default: on; touch/fake only. dualsense
                     never gets FilterAugment regardless of this flag --
                     see §12.)
  TELEOP_FORCE_GAIN  float, N per metre of tracking error (default: 0.0)
  TELEOP_MAX_STEPS   int, exit after N steps (default: 0 = run forever);
                     used by headless smoke tests.
  TELEOP_CONTROLLER  dls | qp   (default: dls)
                     dls: legacy damped-least-squares IK + post-hoc
                          clipping into joint limits (teleop/controller_ik.py).
                     qp:  constraint-aware differential IK; joint velocity
                          and position limits are enforced as hard
                          constraints inside a box-constrained QP instead
                          of being clipped after the solve
                          (teleop/controller_qp.py). Opt-in, does not
                          change dls behavior. PowerShell:
                          $env:TELEOP_CONTROLLER = "qp"
  TELEOP_TASK        cube | shelf   (default: cube -- current behavior,
                     scene/task selection unchanged byte-for-byte)
  TELEOP_LOCK_ORIENTATION  on|off|1|0|true|false (default: unset ->
                     True for mouse/touch/fake, False for dualsense --
                     see _lock_orientation() below)
  DUALSENSE_BACKEND  pydualsense | pygame   (default: pydualsense;
                     dualsense device only, see teleop/dualsense_device.py)
  DUALSENSE_PROFILE  default | no_left_stick | no_left_stick_min
                     (default: default; dualsense device only)
  DUALSENSE_DEADZONE float, stick radial deadzone override (dualsense only)
  TELEOP_DS_OUTPUT   on | off (default: on; kills DualSense HID *output*
                     writes -- lightbar/trigger effects -- only; never
                     affects input. See teleop/dualsense_device.py §2-2.)

Controls (touch/fake modes; stylus button 2 is physically dead):
  stylus button 1 (hold)  move the arm (clutch)
  SPACE                   toggle gripper open/close
  r                       start / restart a round (CubeGame/ShelfTask)
  h                       demo reset: arm -> HOME_QPOS, cubes/shelf item
                          respawned, round dropped to IDLE (use this, not
                          the viewer's Backspace)
  (keys register while the "WRIST VIEW" window has focus)

Controls (dualsense mode, default profile -- see TASK_dualsense_v3.md §6-1):
  left stick               x/y translation (up = +x forward, left = +y)
  right stick               hand pitch/yaw
  L2 (analog, hold)         clutch (踏んでいる間だけ追従); detent + health-
                             proportional resistance
  R2 (analog)                continuous gripper open/close command
  L1 / R1                    z translation (R1 = up)
  d-pad left/right           roll
  Circle                     start/restart a round (button_reset)
  Triangle                   home reset (button_home)
  touchpad (drag)             overhead camera azimuth/elevation (§7;
                             DualSenseInput.read_camera_delta(), NOT part
                             of DeviceState -- camera motion never affects
                             robot state)
"""
import os
import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from teleop.controller_ik import TeleopControllerIK
from teleop.cameras import WristCamera
from teleop.game import CubeGame

# Resolve the scene path relative to this file, not the process's current
# working directory. Visual Studio's default WorkingDirectory can differ
# from where a terminal happens to be (e.g. project root vs. solution
# root), so a bare relative path like "assets/..." is fragile. This file
# lives at <project_root>/teleop/app.py, so one parent-of-parent gets back
# to <project_root>.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE_PATH = os.path.join(_PROJECT_ROOT, "assets", "panda", "teleop_scene.xml")
SHELF_SCENE_PATH = os.path.join(_PROJECT_ROOT, "assets", "panda", "shelf_scene.xml")
HOME_QPOS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

# Conservative clamp for the optional force feedback (Touch-class devices
# have a low continuous force limit; touch_device.set_force clamps again).
MAX_FEEDBACK_FORCE_N = 3.0

# §7: overhead camera elevation clamp when a device drives viewer.cam via
# read_camera_delta() (currently only DualSenseInput's touchpad). Kept as
# an app.py constant, not a dualsense_device.py one, because it belongs to
# the viewer/presentation layer, not the device.
CAMERA_ELEVATION_RANGE = (-89.0, -5.0)


# --------------------------------------------------------------- device sel

def _filter_enabled() -> bool:
    return os.environ.get("TELEOP_FILTER", "on").strip().lower() not in (
        "off", "0", "false")


def _device_kind() -> str:
    """Single source of truth for TELEOP_DEVICE resolution -- used by both
    make_device() and make_controller() (via _lock_orientation) so the env
    var is never parsed/defaulted in two places that could drift apart."""
    return os.environ.get("TELEOP_DEVICE", "mouse").strip().lower()


def _lock_orientation(kind: str) -> bool:
    """§4-2: whether the controller holds the home (gripper-down)
    orientation and ignores rotation input entirely.

    TELEOP_LOCK_ORIENTATION, if set, wins outright (on/off/1/0/true/false).
    Otherwise: mouse/touch/fake default to True -- byte-identical to the
    behavior before this option existed. dualsense defaults to False, so
    the right stick actually does something (with lock_orientation=True
    the whole clutch rotation branch in TeleopControllerIK._update_clutch
    is skipped and stick rotation input is silently discarded).
    """
    env = os.environ.get("TELEOP_LOCK_ORIENTATION")
    if env is not None:
        v = env.strip().lower()
        if v in ("on", "1", "true"):
            return True
        if v in ("off", "0", "false"):
            return False
        # Unrecognized value: fall through to the per-device default
        # rather than silently picking one -- avoids a typo'd env var
        # looking like it did something when it didn't.
    if kind != "dualsense":
        return True
    # VLA collection (2026-09-16): the collect profile has no orientation
    # input and records with the hand fixed pointing down.
    from teleop.dualsense_device import _profile_from_env
    return _profile_from_env(None) == "collect"


def make_device():
    """Create the input device selected by TELEOP_DEVICE (default: mouse).

    touch/fake devices are wrapped, innermost first:
      1. FilterAugment  - One Euro smoothing on raw pos/quat (Phase F;
                           skipped when TELEOP_FILTER=off)
      2. KeyboardButtonAugment - SPACE/r keyboard buttons (stylus button 2
                           is physically dead), also pumps cv2.waitKey so
                           the wrist view repaints.
    Filtering sits closest to the hardware so every consumer downstream
    (clutch scaling, the FIX4 target tracker, the game layer, force
    feedback) sees the same smoothed stream. Buttons are keyboard-sourced
    and already clean, so they are added on the outside, after filtering.
    """
    kind = _device_kind()
    if kind == "touch":
        from teleop.device import KeyboardButtonAugment
        from teleop.touch_device import TouchDeviceInput
        from teleop.filters import FilterAugment
        force_gain = _force_gain()
        dev = TouchDeviceInput(enable_force=force_gain > 0.0)
        if _filter_enabled():
            dev = FilterAugment(dev)
        return KeyboardButtonAugment(dev)
    if kind == "fake":
        from teleop.device import KeyboardButtonAugment
        from teleop.touch_device import FakeTouchInput
        from teleop.filters import FilterAugment
        dev = FakeTouchInput()
        if _filter_enabled():
            dev = FilterAugment(dev)
        return KeyboardButtonAugment(dev)
    if kind == "dualsense":
        # §12: deliberately NOT wrapped in FilterAugment. One Euro was
        # tuned for a noisy absolute-position haptic stylus; DualSense's
        # stick output is already an integrated, smooth signal (adding
        # the filter would only add phase lag), and the alternative
        # touchpad-absolute profile is a distinct noise source that would
        # need its own tuning pass on real hardware before opting in.
        from teleop.device import KeyboardButtonAugment
        from teleop.dualsense_device import DualSenseInput
        dev = DualSenseInput(max_force_N=MAX_FEEDBACK_FORCE_N)  # §4-3
        return KeyboardButtonAugment(dev)
    if kind != "mouse":
        raise ValueError(
            f"Unknown TELEOP_DEVICE={kind!r}; expected mouse, touch, fake "
            "or dualsense.")
    from teleop.device import MouseKeyboardInput
    return MouseKeyboardInput()


# ----------------------------------------------------------- controller sel

def make_controller(model, data, **overrides):
    """Create the teleop controller selected by TELEOP_CONTROLLER
    (default: dls).

    overrides (2026-09-16, VLA collection): keyword arguments replacing
    entries of shared_kwargs below (e.g. scale, workspace_*,
    lock_orientation) so teleop/collect.py reuses exactly these tracker/IK
    parameters instead of copying them. No overrides = unchanged behavior.

    dls  TeleopControllerIK  - damped least squares + post-hoc clipping
                                into joint velocity/position limits
                                (legacy, unchanged behavior).
    qp   TeleopControllerQP  - same task-space solve, but joint velocity
                                and position limits are enforced as hard
                                constraints inside a box-constrained QP
                                instead of being clipped after the fact.
                                See teleop/controller_qp.py.

    Both branches receive identical tuning kwargs so switching modes is a
    fair A/B comparison and does not silently change any shared parameter.
    """
    shared_kwargs = dict(
        ik_gain=0.08,           # per-update task gain; at the measured
                                 # ~60 Hz loop this is ~4.8/s -- raise
                                 # toward 0.15 if tracking still feels slow.
        max_joint_step=0.03,   # per-update; ~1.8 rad/s at 60 Hz. The old
                                 # 0.01 assumed a 500 Hz loop and capped the
                                 # arm at a sluggish 0.6 rad/s.
        rot_weight=0.5,        # keep gripper pointing down
        lock_orientation=_lock_orientation(_device_kind()),
                                 # §4-2: True (mouse/touch/fake, unchanged
                                 # default) means stylus/stick rotation is
                                 # IGNORED -- the IK holds the home
                                 # (gripper-down) orientation, so the hand
                                 # can no longer be tilted at all. False
                                 # (dualsense default) re-enables rotation
                                 # teleop via the right stick. Override
                                 # either way with TELEOP_LOCK_ORIENTATION.
        workspace_x=(0.28, 0.62),
        workspace_y=(-0.30, 0.30),
        workspace_z=(0.08, 0.45),  # floor raised: z=-0.05 put the target
                                 # BELOW the table top (z=0), an unreachable
                                 # pose the arm leaned into (see 2026-07-12
                                 # log, err stuck at 0.14). 0.08 still lets
                                 # the fingertips reach cubes on the table.
    )
    shared_kwargs.update(overrides)

    kind = os.environ.get("TELEOP_CONTROLLER", "dls").strip().lower()
    if kind == "qp":
        from teleop.controller_qp import TeleopControllerQP
        print("[app] controller: QP (constraint-aware differential IK)")
        # posture_weight is the QP's posture regularizer, playing the same
        # role as the dls branch's null_gain below (pull toward home when
        # the task Jacobian leaves slack). The two are different gain
        # scales by construction (soft QP cost term vs null-space
        # projection), so this value is a starting point for tuning, not
        # a numeric equivalent of null_gain=0.05.
        return TeleopControllerQP(model, data, posture_weight=0.0,
                                   **shared_kwargs)
    if kind == "dls":
        print("[app] controller: DLS (legacy clip-based)")
        return TeleopControllerIK(model, data, null_gain=0.05,
                                   **shared_kwargs)
    raise ValueError(
        f"Unknown TELEOP_CONTROLLER={kind!r}; expected dls or qp.")


def _force_gain() -> float:
    try:
        return max(0.0, float(os.environ.get("TELEOP_FORCE_GAIN", "0.0")))
    except ValueError:
        return 0.0


def _task_kind() -> str:
    """TELEOP_TASK resolution, single source of truth for _scene_path() and
    _make_task() (§9-1: the two must never disagree about which task is
    selected)."""
    return os.environ.get("TELEOP_TASK", "cube").strip().lower()


def _scene_path() -> str:
    """§9-1: TELEOP_TASK unset/=cube returns SCENE_PATH unchanged (same
    string object even), so the cube task's behavior is byte-identical to
    before this option existed. =shelf returns the new scene."""
    kind = _task_kind()
    if kind == "shelf":
        return SHELF_SCENE_PATH
    if kind != "cube":
        raise ValueError(f"Unknown TELEOP_TASK={kind!r}; expected cube or shelf.")
    return SCENE_PATH


def _make_task(model, data):
    """§9-1: ShelfTask is a duck-typed alternative to CubeGame (both
    expose update(button_reset)/reset_for_home()/draw_overlay(img)) so
    main() never needs an isinstance branch."""
    kind = _task_kind()
    if kind == "shelf":
        from teleop.shelf_task import ShelfTask
        return ShelfTask(model, data)
    if kind != "cube":
        raise ValueError(f"Unknown TELEOP_TASK={kind!r}; expected cube or shelf.")
    return CubeGame(model, data, round_seconds=60.0)


def _max_steps() -> int:
    try:
        return max(0, int(os.environ.get("TELEOP_MAX_STEPS", "0")))
    except ValueError:
        return 0


# ------------------------------------------------------------------- cubes
#
# NOTE (2026-07-12): main() no longer calls _reset_cubes/_check_cube_fall --
# teleop/game.py's CubeGame is the single owner of cube state once a round
# is active (see module docstring, Phase G). These functions are kept only
# because existing tests exercise them directly (test_set_body_pose_*,
# test_reset_and_fall_respawn). Do not call them from the main loop
# alongside CubeGame; two independent writers of the same cube qpos is
# exactly the kind of bug this whole session has been chasing out.

def _find_cube_bodies(model):
    return [i for i in range(model.nbody)
            if model.body(i).name and "cube" in model.body(i).name.lower()]


def _freejoint_qpos_adr(model, body_id) -> int:
    """Return the qpos address of a body's freejoint.

    Bug fix: body_jntadr holds a JOINT ID; the qpos address must be looked
    up through jnt_qposadr. Indexing qpos directly with the joint id wrote
    the pose into the wrong slots for every cube after the first joint.
    """
    joint_id = model.body_jntadr[body_id]
    if joint_id < 0 or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(
            f"Body {model.body(body_id).name!r} has no freejoint.")
    return model.jnt_qposadr[joint_id]


def _set_body_pose(model, data, body_id, pos, quat):
    adr = _freejoint_qpos_adr(model, body_id)
    data.qpos[adr:adr + 3] = pos
    data.qpos[adr + 3:adr + 7] = quat
    # Zero the cube's velocity so respawned cubes do not keep falling.
    dof = model.jnt_dofadr[model.body_jntadr[body_id]]
    data.qvel[dof:dof + 6] = 0.0


def _random_cube_pose_on_table():
    pos = np.array([np.random.uniform(0.30, 0.60),
                    np.random.uniform(-0.20, 0.20), 0.05])
    return pos, np.array([1.0, 0.0, 0.0, 0.0])


def _reset_cubes(model, data, cube_ids):
    for body_id in cube_ids:
        pos, quat = _random_cube_pose_on_table()
        _set_body_pose(model, data, body_id, pos, quat)
    mujoco.mj_forward(model, data)


def _check_cube_fall(model, data, cube_ids):
    for body_id in cube_ids:
        if data.xpos[body_id][2] < -0.2:
            pos, quat = _random_cube_pose_on_table()
            _set_body_pose(model, data, body_id, pos, quat)


# ------------------------------------------------------------ force (Ph. E)

def compute_feedback_force(controller, data, gain: float) -> np.ndarray:
    """Clamped spring on the tracking error (world frame, Newtons).

    Pulls the operator's hand back toward where the robot actually is when
    the arm lags the target. Returns zeros when gain <= 0.
    """
    if gain <= 0.0:
        return np.zeros(3)
    err = controller.target_pos - data.xpos[controller.hand_body_id]
    force = -gain * err
    norm = float(np.linalg.norm(force))
    if norm > MAX_FEEDBACK_FORCE_N:
        force *= MAX_FEEDBACK_FORCE_N / norm
    return force


# ------------------------------------------------------- loop-rate (Ph. F)

class LoopRateMonitor:
    """Measures the main loop's real wall-clock rate.

    Why this exists: PLAN.md's "~1 kHz servo thread" is the OpenHaptics
    callback frequency inside teleop/touch_device.py, which only refreshes
    a shared snapshot -- it says nothing about how often this loop, and
    therefore device.read() / FilterAugment / the controller, actually
    samples that snapshot. This class measures that directly instead of
    assuming a number, so teleop/filters.py's min_cutoff/beta and
    controller_ik.py's tracker_omega can be tuned against reality.

    call tick() once per loop iteration (see main() below); it prints a
    summary every `report_every` calls and is silent otherwise. Safe to
    delete once the real rate is known and parameters are tuned -- it does
    not touch control flow, only reads the clock.
    """

    def __init__(self, window: int = 120, report_every: int = 120):
        self.window = window
        self.report_every = report_every
        self._dts = []
        self._t_prev = None
        self._n = 0

    def tick(self) -> None:
        t = time.perf_counter()
        if self._t_prev is not None:
            self._dts.append(t - self._t_prev)
            if len(self._dts) > self.window:
                self._dts.pop(0)
        self._t_prev = t
        self._n += 1
        if self._n % self.report_every == 0 and self._dts:
            dts = np.array(self._dts)
            hz_avg = 1.0 / dts.mean()
            hz_worst = 1.0 / dts.max()  # slowest single iteration
            jitter_ms = dts.std() * 1000.0
            print(
                f"[loop-rate] avg={hz_avg:6.1f} Hz  "
                f"worst={hz_worst:6.1f} Hz  jitter={jitter_ms:5.2f} ms  "
                f"(n={self._n})"
            )


# ------------------------------------------------------------------- main

def apply_home_pose(model, data) -> None:
    """Write HOME_QPOS into the arm joints and refresh derived quantities.

    Only the 7 arm joints are touched by name, so this is independent of
    the qpos layout of the cubes' freejoints. Cube poses are handled
    separately by _reset_cubes.
    """
    for i, q in enumerate(HOME_QPOS, start=1):
        data.qpos[model.joint(f"joint{i}").qposadr[0]] = q
    mujoco.mj_forward(model, data)


def home_reset(model, data, controller, game) -> None:
    """Full demo reset: arm -> HOME_QPOS, cubes/shelf item respawned, round
    stopped, target re-anchored.

    `game` is duck-typed (CubeGame or ShelfTask, see _make_task/§9-1): only
    reset_for_home() is called on it here, so this function does not care
    which task is active.

    Order matters:
      1. write home joint angles and mj_forward so xpos/xquat update,
      2. game.reset_for_home() respawns cubes AND drops the round to IDLE
         (also calls mj_forward internally),
      3. re-anchor the controller's mocap target to the now-current hand
         pose so the next control step starts at zero tracking error.
    Bind this to the 'h' key so demo staff never need the viewer's
    Backspace, which would drop the arm into the upright singular pose.
    """
    apply_home_pose(model, data)
    game.reset_for_home()
    controller.sync_target_to_hand()


def main() -> None:
    model = mujoco.MjModel.from_xml_path(_scene_path())
    data = mujoco.MjData(model)

    # Set the Panda home pose (avoids starting at the upright singularity).
    apply_home_pose(model, data)

    device = make_device()
    controller = make_controller(model, data)
    camera = WristCamera(model)
    game = _make_task(model, data)
    if hasattr(game, "attach_controller"):
        # §5-1/§9-4: SIGMA_REF is measured once, right here, at the home
        # pose apply_home_pose() already set above -- never hardcoded (a
        # different Panda model/scaling would silently invalidate a
        # constant). Optional: only ShelfTask currently implements this
        # hook; CubeGame does not need it.
        from teleop import pad_feedback
        sigma_ref = pad_feedback.measure_sigma_ref(
            model, data, controller.hand_body_id, controller.dof_indices)
        game.attach_controller(controller, sigma_ref=sigma_ref)

    force_gain = _force_gain()
    max_steps = _max_steps()
    has_force = hasattr(device, "set_force")
    has_camera_delta = hasattr(device, "read_camera_delta")
    rate_monitor = LoopRateMonitor()

    device.start()
    step = 0
    # Real-time pacing (2026-07-12): the render/viewer-bound main loop only
    # reaches ~60 Hz on the lab PC (see LoopRateMonitor), but one mj_step
    # advances sim time by just 2 ms -- so the robot used to run at
    # 60*0.002 = 0.12x wall speed, an ~8x slow-motion that dominated the
    # perceived lag. Fix: per iteration, step physics as many times as
    # needed for sim time to keep up with the wall clock (capped to avoid
    # spiral-of-death after a hiccup), and give the target tracker the
    # measured loop dt instead of the 2 ms it assumed.
    MAX_CATCHUP_STEPS = 25
    timestep = float(model.opt.timestep)
    t_prev = time.perf_counter()
    sim_debt = 0.0  # wall seconds of physics not yet simulated
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # teleop_scene.xml puts the mocap target sphere in geom group 5
            # so camera images never contain it; show it in the operator's
            # viewer only.
            with viewer.lock():
                viewer.opt.geomgroup[5] = 1
            while viewer.is_running():
                rate_monitor.tick()

                now = time.perf_counter()
                loop_dt = min(now - t_prev, 0.25)  # clamp pathological stalls
                t_prev = now

                # The tracker integrates once per controller.update call,
                # so its dt is this loop's real period, not the physics dt.
                controller.ctrl_dt = max(timestep, loop_dt)

                state = controller.update(device)

                # §7: DualSense touchpad -> overhead camera. A side-channel
                # deliberately kept OUT of DeviceState/controller.update --
                # camera motion must never affect robot state.
                # KeyboardButtonAugment always exposes read_camera_delta
                # (so has_camera_delta is True for touch/fake/dualsense,
                # False only for bare MouseKeyboardInput) but forwards to
                # the wrapped device's own method only if it has one;
                # touch/fake have none, so it returns (0.0, 0.0) and the
                # `if d_az or d_el` guard below makes this whole block a
                # true no-op for them -- viewer.cam is only ever touched
                # for dualsense.
                if has_camera_delta:
                    d_az, d_el = device.read_camera_delta()
                    if d_az or d_el:
                        viewer.cam.azimuth = (viewer.cam.azimuth + d_az) % 360.0
                        viewer.cam.elevation = float(np.clip(
                            viewer.cam.elevation + d_el,
                            CAMERA_ELEVATION_RANGE[0], CAMERA_ELEVATION_RANGE[1]))

                if state.button_home:
                    home_reset(model, data, controller, game)
                    sim_debt = 0.0

                sim_debt += loop_dt
                n_steps = 0
                while sim_debt >= timestep and n_steps < MAX_CATCHUP_STEPS:
                    mujoco.mj_step(model, data)
                    sim_debt -= timestep
                    n_steps += 1
                if n_steps == MAX_CATCHUP_STEPS:
                    sim_debt = 0.0  # drop the backlog, never spiral

                # Called after the physics catch-up so scoring/fall checks
                # see this iteration's freshest cube positions. 'r' (not
                # consumed by home_reset above) starts/restarts a round;
                # CubeGame also respawns fallen cubes every call regardless
                # of round state.
                if not state.button_home:
                    game.update(state.button_reset)

                if has_force and force_gain > 0.0:
                    device.set_force(
                        compute_feedback_force(controller, data, force_gain))

                camera.render(data, overlay_fn=game.draw_overlay)
                viewer.sync()
                time.sleep(0.001)

                step += 1
                if max_steps and step >= max_steps:
                    break
    finally:
        device.stop()
        cv2.destroyAllWindows()
