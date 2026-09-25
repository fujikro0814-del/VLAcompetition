"""VLA data-collection mode (C2, 2026-09-16).

Launch: collect_main.py (the demo entry point main.py / teleop/app.py is
unchanged). See タスク_テレオペシステムのデータ収集対応_v2.md.

Signal path:

    DualSense "collect" profile -> radial deadzone / trigger rest -> vel
      -> VelocityCommandIntegrator (x_cmd += vel * timestep, per physics step)
      -> controller clutch (always engaged, scale 1.0) -> desired_pos = x_des
      -> tracker -> IK -> PD                       (all unchanged)

Differences from the demo loop in app.main(), and why:
- Tracker + IK run once per 2 ms physics step (ctrl_dt = model timestep),
  not once per ~60 Hz main-loop iteration. The pad is still polled once per
  iteration; its latest velocity is integrated per physics step, so x_des
  moves on the simulation clock. The loop keeps sim time locked to wall
  time (RealTimeFactorMonitor reports when it cannot), so this equals the
  real polling-period integration the task doc asks for while RTF ~= 1.
- Controller scale 1.0 and the integrator clamping to the controller's own
  workspace make x_cmd == desired_pos, so the two clamps can never disagree
  (no hidden wind-up at a wall).
- Every episode starts from one settled state (settle_start_state) computed
  once at startup: arm at rest with the hand at START_POS, gripper open.
- No game layer, score, sounds or force feedback.

Recording (C3): CollectSession (below) runs the episode state machine and
teleop/recorder.py writes 03_収録/raw. Buttons (collect profile): circle =
start recording, triangle held 1 s = save, touchpad click = discard (a
retake), square = gripper, PS = quit. Launch with an operator id:

    python collect_main.py --operator 01 [--reps 8] [--raw-dir DIR]
"""
import argparse
import contextlib
import dataclasses
import datetime
import io
import pathlib
import socket
import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from teleop import app, ledger, recorder
from teleop.device import DeviceInput, DeviceState

# Collection workspace (B-2): y widened to 0.40 so the box at y=0.25 is
# reachable; z floor 0.115 keeps the fingertips just above the table so
# pressing into it is never recorded. The demo workspace is unchanged.
COLLECT_WORKSPACE = dict(
    workspace_x=(0.28, 0.62),
    workspace_y=(-0.30, 0.40),
    workspace_z=(0.115, 0.35),
)
# Episode start hand position: HOME_QPOS's hand x/y (0.307, 0) with z
# lowered from 0.590 into the collection workspace.
START_POS = np.array([0.307, 0.0, 0.35])
SETTLE_SECONDS = 3.0
FINGER_OPEN = 0.04           # finger_joint1/2 range upper limit [m]

RECORD_EVERY = 25            # physics steps per 20 Hz window (500 / 20)
IMAGE_SIZE = 256
CAMERAS = ("overhead", "wrist")
MAX_CATCHUP_STEPS = 50       # 100 ms of physics per loop iteration at most

# Material for the 中断 label of 本冊 表 3-10: loop iterations that took far longer than they
# should have (the application froze, the machine slept). Counted only; the threshold is written
# into meta.json with the count so that a different line can be drawn later.
LOOP_NOMINAL_DT = 0.02       # one loop iteration [s] (a healthy recording keeps max_loop_dt_s here)
STALL_FACTOR = 5
STALL_THRESHOLD_S = STALL_FACTOR * LOOP_NOMINAL_DT      # 0.1 s

# Operator display (never recorded). See OperatorView.
OPERATOR_WINDOW = "OPERATOR VIEW"
OPERATOR_PANEL = 400         # px per panel (offscreen buffer is 640x480)
OPERATOR_HZ = 30.0
OPERATOR_SCREEN_FRACTION = 0.85
ROBOT_ALPHA = 0.2            # arm see-through in the ortho panels: hand, fingers
                             # and wrist links stack over the cube in the top view
FINGERTIP_OFFSET = 0.1034    # hand origin -> fingertip centre along hand z [m]
TABLE_TOP_Z = 0.0

_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------- x_des

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
        # ignored) while buttons are still read via refresh(). CollectSession
        # enables it only while recording, so a stray input before circle
        # never moves the arm (2026-09-16 hands-on feedback).
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
    """app.make_controller's tracker/IK parameters, unchanged, with the
    collection-only clutch scale, orientation lock and workspace.

    Writes HOME_QPOS into data first, exactly as app.main() does before
    creating its controller: the IK's null-space posture target q_neutral
    is the joint state at construction, and a fresh MjData is the upright
    all-zero pose (callers reset the episode state afterwards anyway)."""
    app.apply_home_pose(model, data)
    controller = app.make_controller(model, data, scale=1.0,
                                     lock_orientation=True, **COLLECT_WORKSPACE)
    controller.ctrl_dt = float(model.opt.timestep)
    return controller


def make_integrator(device: DeviceInput, model, controller):
    return VelocityCommandIntegrator(
        device, model.opt.timestep, controller.workspace_x,
        controller.workspace_y, controller.workspace_z)


# ---------------------------------------------------------- start state

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
                       tol: float = 0.010, qvel_tol: float = 1e-3) -> StartState:
    """From HOME_QPOS, drive the hand to START_POS with the collection
    controller (tracker + IK at the physics rate, gripper open) and let it
    come to rest. The resulting qpos/ctrl/q_des include the servos' steady-
    state gravity offset, so an episode reset to them starts motionless.

    tol is loose on purpose: the unchanged DLS IK leaves a static offset
    between desired_pos and the hand (its null-space posture term is not
    exactly null under damping; measured 2026-09-16: 5 mm at START_POS,
    5-17 mm over the collection workspace, mostly +z, 0 with null_gain=0).
    The IK is not to be modified, so this only checks the arm came to rest
    near START_POS."""
    data = mujoco.MjData(model)
    app.apply_home_pose(model, data)
    _set_fingers(model, data, FINGER_OPEN)
    mujoco.mj_forward(model, data)
    with contextlib.redirect_stdout(io.StringIO()):  # IK diagnostic log
        controller = make_collect_controller(model, data)
        _open_gripper(controller)
        controller.sync_target_to_hand()
        controller.desired_pos = controller._clamp_workspace(START_POS.copy())
        hold = _Hold()
        for _ in range(int(round(seconds / model.opt.timestep))):
            controller.update(hold)
            mujoco.mj_step(model, data)
    err = float(np.linalg.norm(data.xpos[controller.hand_body_id] - START_POS))
    speed = float(np.linalg.norm(data.qvel[controller.dof_indices]))
    if err > tol or speed > qvel_tol:
        raise RuntimeError(
            f"start state did not settle after {seconds} s: hand "
            f"{err * 1000:.2f} mm from START_POS, |qvel| {speed:.2e} rad/s")
    return StartState(qpos=data.qpos.copy(), ctrl=data.ctrl.copy(),
                      q_des=controller.q_des.copy())


def reset_episode(model, data, controller, integrator, start: StartState,
                  placement=None) -> None:
    """Restore the settled start state exactly (cube at `placement`, a
    recorder.Placement, if given) and re-anchor x_des to it."""
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


# --------------------------------------------------------------- timing

class RealTimeFactorMonitor:
    """Simulated seconds per wall second, reported every report_s. RTF < 1
    means physics fell behind (the backlog is dropped, never accumulated),
    so the operator saw the robot slower than commanded."""

    def __init__(self, report_s: float = 2.0, time_fn=time.perf_counter,
                 verbose: bool = True):
        self.report_s = report_s
        self.time_fn = time_fn
        self.verbose = verbose
        self._t0 = None
        self._sim = 0.0
        self._iters = 0
        self._steps = 0
        self._drops = 0
        self.total_sim = 0.0
        self.total_wall = 0.0
        self.total_drops = 0
        self.min_rtf = float("inf")

    def update(self, sim_seconds: float, n_steps: int, dropped: bool) -> None:
        now = self.time_fn()
        if self._t0 is None:
            self._t0 = now
            return
        self._sim += sim_seconds
        self._iters += 1
        self._steps += n_steps
        self._drops += int(dropped)
        wall = now - self._t0
        if wall < self.report_s:
            return
        rtf = self._sim / wall
        self.min_rtf = min(self.min_rtf, rtf)
        self.total_sim += self._sim
        self.total_wall += wall
        self.total_drops += self._drops
        if self.verbose:
            print(f"[collect] RTF={rtf:5.3f}  loop={self._iters / wall:5.1f} Hz  "
                  f"steps/loop={self._steps / max(self._iters, 1):4.1f}  "
                  f"backlog_drops={self._drops}")
        self._t0 = now
        self._sim = 0.0
        self._iters = self._steps = self._drops = 0

    def summary(self) -> str:
        if self.total_wall <= 0.0:
            return "[collect] RTF summary: not enough data"
        return (f"[collect] RTF summary: mean={self.total_sim / self.total_wall:5.3f} "
                f"min={self.min_rtf:5.3f} backlog_drops={self.total_drops}")


# ----------------------------------------------------------------- main

def close_renderer(renderer: mujoco.Renderer) -> None:
    """Close a mujoco.Renderer without corrupting the others.

    mujoco 3.2.3's Renderer.close() (also run from __del__) frees its GL
    context first and its MjrContext second, so the MjrContext's GL objects
    are deleted in whichever context is current -- typically another live
    renderer's. Measured 2026-09-16: closing one renderer after the
    recording renderer had drawn changed every pixel of the next recorded
    frames. Here the renderer's own context is made current first. Never
    let a Renderer be garbage-collected while another one is still in use;
    close it with this instead.
    """
    gl = renderer._gl_context
    if gl:
        gl.make_current()
    if renderer._mjr_context:
        renderer._mjr_context.free()
    renderer._mjr_context = None
    if gl:
        gl.free()
    renderer._gl_context = None


def render_cameras(renderer: mujoco.Renderer, data) -> list:
    frames = []
    for name in CAMERAS:
        renderer.update_scene(data, camera=name)
        frames.append(renderer.render().copy())
    return frames


class OperatorView:
    """The operator's window. Nothing here is recorded: it has its own
    renderer, and all decoration goes into that renderer's scene or onto
    the displayed image. 3x2 panels:

        top (ortho, +x up, +y left) | back (ortho, +y left, +z up) | gripper
        wrist camera                | overhead camera              | (blank)

    History (2026-09-16 hands-on feedback): the oblique recording wrist
    camera W6m makes "where does a straight descent land" hard to judge (the
    old on-axis wrist camera made it obvious but closed fingers occluded it,
    and W6m is fixed because evaluation shares it). Hence:
    - top/back: magenta drop guide drawn in 2D over the image (projected
      from 3D, never hidden): circle at the fingertip centre, line straight
      down to the table, cross on the table; workspace outline; arm
      see-through; shadows/reflections off.
    - gripper: operator-only camera on the hand axis looking straight down,
      so the image centre is the drop point, marked with a magenta x;
      fingers drawn see-through so a closed grip does not hide it.
    - wrist/overhead: the RAW recording camera images, exactly as saved to
      the episode PNGs (no guide). They are NOT the policy input: images
      enter a policy only through vla_image_spec.py (currently overhead
      flipped left-right, wrist flipped up-down). The panel labels say "raw
      image" so nobody reads them as what the model sees. Operator displays
      are never transformed.

    2026-09-17: the wrist camera was C6 for a few hours (Step C-D of the
    wrist-camera remount) and is W6m again since Step E-C, with the wrist
    image flipped up-down for the policy instead.
    """

    DROP_BGR = (255, 0, 255)
    WORKSPACE_RGBA = np.array([0.1, 0.3, 0.9, 1.0], dtype=np.float32)
    GUIDE_CAMERAS = ("operator_top", "operator_back")
    GRIPPER_CAMERA = "operator_gripper"
    LAYOUT = (("operator_top", "operator_back", "operator_gripper"),
              ("wrist", "overhead", None))
    LABELS = {"operator_top": "TOP  (stick up = forward)",
              "operator_back": "BACK  (from behind the robot)",
              "operator_gripper": "GRIPPER  (x = straight down)",
              "wrist": "WRIST  (raw image)", "overhead": "OVERHEAD  (raw image)"}

    def __init__(self, model, controller, panel: int = OPERATOR_PANEL):
        self.model = model
        self.controller = controller
        self.panel = panel
        self.renderer = mujoco.Renderer(model, panel, panel)
        root = model.body_rootid[controller.hand_body_id]  # the arm's base body
        self.robot_geom = np.isin(model.geom_bodyid,
                                  np.flatnonzero(model.body_rootid == root))
        # gripper panel: the camera sits inside the hand, so the hand body and
        # fingers are drawn see-through there
        gripper = [controller.hand_body_id] + [
            model.body(n).id for n in ("left_finger", "right_finger")]
        self.gripper_geom = np.isin(model.geom_bodyid, gripper)
        self._last_show = None

    # -- scene decoration (operator renderer only) --
    def _add_capsule(self, scn, p0, p1, width, rgba) -> None:
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                            np.zeros(3), np.eye(3).flatten(), rgba)
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                             np.asarray(p0, float), np.asarray(p1, float))
        scn.ngeom += 1

    def fingertip_center(self, data) -> np.ndarray:
        h = self.controller.hand_body_id
        return data.xpos[h] + data.xmat[h].reshape(3, 3)[:, 2] * FINGERTIP_OFFSET

    @staticmethod
    def _see_through(scn, geom_mask) -> None:
        for i in range(scn.ngeom):
            g = scn.geoms[i]
            if (g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid >= 0
                    and geom_mask[g.objid]):
                g.rgba[3] = min(g.rgba[3], ROBOT_ALPHA)

    def _add_workspace_outline(self, scn) -> None:
        z = TABLE_TOP_Z + 0.001
        (x0, x1), (y0, y1) = self.controller.workspace_x, self.controller.workspace_y
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        for (ax, ay), (bx, by) in zip(corners, corners[1:] + corners[:1]):
            self._add_capsule(scn, [ax, ay, z], [bx, by, z], 0.0015,
                              self.WORKSPACE_RGBA)

    def project(self, data, camera: str, point, size: int):
        """Pixel (u, v) of a world point in a size x size render of a named
        camera, or None if it is behind a perspective camera."""
        cid = self.model.camera(camera).id
        rel = data.cam_xmat[cid].reshape(3, 3).T @ (np.asarray(point, float)
                                                     - data.cam_xpos[cid])
        if self.model.cam_orthographic[cid]:
            scale = size / self.model.cam_fovy[cid]  # fovy in metres
            return size / 2 + rel[0] * scale, size / 2 - rel[1] * scale
        depth = -rel[2]  # cameras look along their -z
        if depth <= 1e-6:
            return None
        f = (size / 2) / np.tan(np.deg2rad(self.model.cam_fovy[cid]) / 2)
        return size / 2 + f * rel[0] / depth, size / 2 - f * rel[1] / depth

    def draw_drop_guide(self, img, data, camera: str, render_size: int) -> None:
        tip = self.fingertip_center(data)
        floor = np.array([tip[0], tip[1], TABLE_TOP_Z])
        k = img.shape[0] / render_size
        a = self.project(data, camera, tip, render_size)
        b = self.project(data, camera, floor, render_size)
        if a is None or b is None:
            return
        pa = (int(round(a[0] * k)), int(round(a[1] * k)))
        pb = (int(round(b[0] * k)), int(round(b[1] * k)))
        cv2.line(img, pa, pb, self.DROP_BGR, 2, cv2.LINE_AA)
        cv2.circle(img, pa, 6, self.DROP_BGR, 2, cv2.LINE_AA)
        cv2.drawMarker(img, pb, self.DROP_BGR, cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)

    def render_panel(self, data, camera: str) -> np.ndarray:
        self.renderer.update_scene(data, camera=camera)
        scn = self.renderer.scene
        guide = camera in self.GUIDE_CAMERAS
        # wrist/overhead keep the recorded look (shadows on); operator-only
        # panels drop the arm's shadow, whose hard edge reads like an object
        operator_only = guide or camera == self.GRIPPER_CAMERA
        scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = not operator_only
        scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = not operator_only
        if guide:
            self._see_through(scn, self.robot_geom)
            self._add_workspace_outline(scn)
        elif camera == self.GRIPPER_CAMERA:
            self._see_through(scn, self.gripper_geom)
        return self.renderer.render().copy()

    def set_status(self, lines) -> None:
        """lines: [(text, (b, g, r)), ...] shown in the blank panel."""
        self.status = list(lines)

    def _status_image(self) -> np.ndarray:
        img = np.full((self.panel, self.panel, 3), 40, np.uint8)
        y = 34
        for text, color in getattr(self, "status", []):
            scale = 0.75 if y == 34 else 0.5
            cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                        2 if y == 34 else 1, cv2.LINE_AA)
            y += 40 if y == 34 else 26
        return img

    def _panel_image(self, data, camera) -> np.ndarray:
        if camera is None:
            return self._status_image()
        img = cv2.cvtColor(self.render_panel(data, camera), cv2.COLOR_RGB2BGR)
        if camera in self.GUIDE_CAMERAS:
            self.draw_drop_guide(img, data, camera, self.panel)
        elif camera == self.GRIPPER_CAMERA:
            c = (self.panel // 2, self.panel // 2)
            cv2.drawMarker(img, c, self.DROP_BGR, cv2.MARKER_TILTED_CROSS, 28, 2,
                           cv2.LINE_AA)
        label = self.LABELS[camera]
        cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def compose(self, data) -> np.ndarray:
        # All panels rendered from the current state, so the guides line up.
        return np.vstack([np.hstack([self._panel_image(data, c) for c in row])
                          for row in self.LAYOUT])

    def open_window(self) -> None:
        cv2.namedWindow(OPERATOR_WINDOW, cv2.WINDOW_NORMAL)
        rows, cols = len(self.LAYOUT), len(self.LAYOUT[0])
        h = rows * self.panel
        w = cols * self.panel
        try:  # fit the window to the screen (Windows), keeping the aspect
            import ctypes
            user32 = ctypes.windll.user32
            screen_w, screen_h = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
            s = min(screen_h * OPERATOR_SCREEN_FRACTION / h, screen_w * 0.95 / w)
            h, w = int(h * s), int(w * s)
        except Exception:  # noqa: BLE001
            pass
        cv2.resizeWindow(OPERATOR_WINDOW, w, h)

    def close(self) -> None:
        close_renderer(self.renderer)

    def maybe_show(self, data, now: float) -> None:
        if self._last_show is not None and now - self._last_show < 1.0 / OPERATOR_HZ:
            return
        self._last_show = now
        cv2.imshow(OPERATOR_WINDOW, self.compose(data))

# ------------------------------------------------------------- session

SAVE_HOLD_STEPS = 500        # triangle held 1.0 s (simulated) saves
# Practice runs (--practice) never touch 03_収録/raw.
PRACTICE_DIR = recorder.DEFAULT_RAW_DIR.parent / "practice"
LONG_EPISODE_S = 45.0        # status warning; 表3-8 expects 15-30 s
CONTROLLER_PARAMS = ("ik_gain", "max_joint_step", "rot_weight", "null_gain",
                     "tracker_omega", "max_target_vel", "ik_damping",
                     "q_des_leash", "max_dx_norm", "scale", "lock_orientation",
                     "workspace_x", "workspace_y", "workspace_z")
HASHED_FILES = ("teleop/collect.py", "teleop/recorder.py", "teleop/controller_ik.py",
                "teleop/controller_qp.py", "teleop/dualsense_device.py",
                "teleop/app.py", "assets/panda/teleop_scene.xml",
                "assets/panda/panda.xml", "teleop/ledger.py")

_WHITE, _GREEN, _RED, _YELLOW, _GRAY = ((255, 255, 255), (80, 220, 80),
                                        (60, 60, 255), (0, 220, 255), (170, 170, 170))


class CollectSession:
    """Episode state machine for one sitting. Headless: the caller runs the
    physics and calls handle_input() once per pad read, after_step() after
    every mj_step and note_loop() once per loop iteration.

        READY      circle      -> reset to start state + placement, RECORDING
                   touchpad    -> re-place the cube (reset, stay READY)
        RECORDING  triangle held SAVE_HOLD_STEPS -> save, truncated at the
                               press; next placement, READY
                   touchpad    -> discard (retake + 1), same placement, READY
        any        PS          -> discard if recording, quit

    Episode ids, session numbers and saved-per-placement counts come from the
    episode ledger (teleop/ledger.py, 2026-09-17), not from the folders in
    raw_root: `kind` (production / script / practice) picks the id band, the
    ledger session holds the ledger lock for the whole sitting, and the id is
    assigned only when a save is confirmed (a discarded retake uses no id; a
    failed save uses one for good). close(backup=True) copies the ledger to
    its read-only backup folder.
    """

    READY, RECORDING = "READY", "RECORDING"

    @property
    def state(self) -> str:
        return self._state

    @state.setter
    def state(self, value: str) -> None:
        # the pad drives the arm only while recording
        self._state = value
        self.integrator.enabled = value == self.RECORDING

    def __init__(self, model, data, controller, integrator, start: StartState,
                 renderer, raw_root, operator_id: str, reps_target: int,
                 clock=time.perf_counter, *, episode_ledger: "ledger.EpisodeLedger", kind: str):
        self.model, self.data = model, data
        self.controller, self.integrator, self.start = controller, integrator, start
        self.raw_root = pathlib.Path(raw_root)
        self.operator_id = str(operator_id)
        self.reps_target = int(reps_target)
        self.clock = clock
        self.timestep = float(model.opt.timestep)
        self.sampler = recorder.FrameSampler(model, controller, renderer, CAMERAS)
        mujoco.mj_forward(model, data)  # goal_box is static; read its position once
        self.box_pos = data.xpos[model.body(recorder.BOX_BODY).id].copy()

        self.kind = kind
        self.host = socket.gethostname()
        # takes the ledger lock; refuses unregistered/wrong-kind folders and
        # production on hosts not listed in the ledger config
        self.ledger = episode_ledger.open_session(kind, self.raw_root, self.operator_id, host=self.host)
        self.session = self.ledger.session
        self.saved = self.ledger.saved_per_placement()
        self.next_episode_id = self.ledger.next_episode_id     # display only; assigned at save
        self.last_saved = None                                 # (episode_id, path) of the last save
        self.day = datetime.date.today().isoformat()
        self.placements = recorder.training_placements()
        self.retakes = {}
        self.session_saved = 0
        self.message = ("", _GRAY)
        self.log = recorder.SessionLog(self.raw_root, self.session, self.day)
        self.log.write("session_start", session=self.session, operator=self.operator_id,
                       reps_target=self.reps_target,
                       placements=[dataclasses.asdict(p) for p in self.placements])

        self.state = self.READY
        self.writer = None
        self.episode_step = 0
        self._prev = {"reset": False, "discard": False, "quit": False}
        self._save_held = False
        self._hold_steps = 0
        self._press_step = None
        self.placement = recorder.next_placement(self.placements, self.saved)
        self.prepare()

    # -- transitions --
    def prepare(self) -> None:
        reset_episode(self.model, self.data, self.controller, self.integrator,
                      self.start, self.placement)
        self.episode_step = 0

    def _begin(self) -> None:
        self.prepare()
        self.writer = recorder.EpisodeWriter(      # id assigned at save
            self.raw_root, None, self.day, CAMERAS, RECORD_EVERY,
            RECORD_EVERY * self.timestep)
        self.recorded_at = recorder._now_iso()
        self.timing = {"wall_start": self.clock(), "drops": 0, "max_loop_dt": 0.0,
                       "stalls": 0, "sim": 0.0, "wall": 0.0}
        self._hold_steps, self._press_step, self._save_held = 0, None, False
        self.writer.add_step(self.controller.desired_pos, self._gripper_cmd())
        self.writer.add_frame(*self.sampler.capture(self.data, 0))
        self.state = self.RECORDING
        self.message = ("", _GRAY)

    def _gripper_cmd(self) -> float:
        return float(self.data.ctrl[self.controller.gripper_indices[0]])

    def discard(self, reason: str) -> None:
        pid = self.placement.placement_id
        duration = self.episode_step * self.timestep
        self.writer.discard()
        self.writer = None
        self.retakes[pid] = self.retakes.get(pid, 0) + 1
        self.ledger.ensure_session_started()
        self.log.write("episode_discarded", reason=reason, placement_id=pid,
                       retake=self.retakes[pid], duration_s=round(duration, 3))
        self.message = (f"discarded ({reason}), retake {self.retakes[pid]}", _YELLOW)
        self.state = self.READY
        self.prepare()

    def _save_not_done(self, pid, episode_id, exc, partial, message) -> None:
        # Keep the session alive and the recording on disk for inspection
        # (ep_N.partial, or ep_pending_*.partial when no id was assigned);
        # the attempt counts as a retake. An assigned id is never reused.
        self.log.write("episode_save_failed", episode_id=episode_id, placement_id=pid,
                       error=repr(exc), partial=str(partial))
        print(f"[collect] {message}: {exc}")
        self.writer = None
        self.retakes[pid] = self.retakes.get(pid, 0) + 1
        self.message = (f"{message} (see console)", _RED)
        self.next_episode_id = self.ledger.next_episode_id
        self.state = self.READY
        self.prepare()

    def save(self) -> pathlib.Path:
        pid = self.placement.placement_id
        n_keep = self._press_step // RECORD_EVERY + 1
        last = self.writer.frames[min(n_keep, self.writer.n_frames) - 1]
        success = recorder.cube_in_box(last["cube_pos"], self.box_pos)
        try:
            episode_id, seq = self.ledger.allocate(pid, self.placement.seed, self.day,
                                                   self.writer.day_dir)
        except ledger.LedgerError as exc:
            self.writer.abandon()
            self._save_not_done(pid, None, exc, self.writer.dir, "SAVE STOPPED, id check")
            return None
        meta = self._meta(episode_id, seq, n_keep, success, last)
        extra = {"start_qpos": self.start.qpos, "start_ctrl": self.start.ctrl,
                 "start_q_des": self.start.q_des,
                 "reset_qpos": self._reset_qpos()}
        try:
            path = self.writer.finalize(n_keep, meta, extra, episode_id=episode_id)
        except Exception as exc:  # noqa: BLE001
            self.writer.abandon()
            partial = self.writer.mark_failed(episode_id)
            self.ledger.record_save_failed(episode_id, partial, repr(exc))
            self._save_not_done(pid, episode_id, exc, partial, "SAVE FAILED")
            return None
        self.writer = None
        self.ledger.record_saved(episode_id, path, n_keep, success, self.retakes.get(pid, 0))
        self.saved[pid] = self.saved.get(pid, 0) + 1
        self.log.write("episode_saved", episode_id=episode_id, placement_id=pid,
                       retakes=self.retakes.get(pid, 0), n_frames=n_keep,
                       success=success, path=str(path))
        self.message = (f"saved ep_{episode_id:06d} "
                        f"({'in box' if success else 'NOT in box'})",
                        _GREEN if success else _YELLOW)
        self.last_saved = (episode_id, path)
        self.retakes[pid] = 0
        self.session_saved += 1
        self.next_episode_id = self.ledger.next_episode_id
        self.state = self.READY
        self.placement = recorder.next_placement(self.placements, self.saved)
        self.prepare()
        return path

    def _reset_qpos(self) -> np.ndarray:
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = self.start.qpos
        recorder.apply_placement(self.model, scratch, self.placement)
        return scratch.qpos.copy()

    # -- per-call hooks --
    def handle_input(self, s: DeviceState) -> bool:
        """Returns True when the session should end."""
        rise = {k: v and not self._prev[k] for k, v in
                (("reset", s.button_reset), ("discard", s.button_discard),
                 ("quit", s.button_quit))}
        self._prev = {"reset": s.button_reset, "discard": s.button_discard,
                      "quit": s.button_quit}
        if rise["quit"]:
            if self.state == self.RECORDING:
                self.discard("quit")
            return True
        if self.state == self.READY:
            if rise["reset"]:
                self._begin()
            elif rise["discard"]:
                self.prepare()
        else:
            if rise["discard"]:
                self.discard("touchpad")
            else:
                self._save_held = bool(s.button_save)
        return False

    def after_step(self) -> None:
        if self.state != self.RECORDING:
            return
        self.episode_step += 1
        self.writer.add_step(self.controller.desired_pos, self._gripper_cmd())
        if self.episode_step % RECORD_EVERY == 0:
            self.writer.add_frame(*self.sampler.capture(self.data, self.episode_step))
        if self._save_held:
            if self._press_step is None:
                self._press_step = self.episode_step - 1
            self._hold_steps += 1
            if self._hold_steps >= SAVE_HOLD_STEPS:
                self.save()
        else:
            self._press_step, self._hold_steps = None, 0

    def note_loop(self, loop_dt: float, sim_seconds: float, dropped: bool) -> None:
        if self.state != self.RECORDING:
            return
        t = self.timing
        t["drops"] += int(dropped)
        t["max_loop_dt"] = max(t["max_loop_dt"], loop_dt)
        t["stalls"] += int(loop_dt > STALL_THRESHOLD_S)
        t["sim"] += sim_seconds
        t["wall"] += loop_dt

    # -- metadata / display --
    def _meta(self, episode_id: int, ledger_seq: int, n_keep: int, success: bool, last: dict) -> dict:
        c, p, t = self.controller, self.placement, self.timing
        pad = getattr(self.integrator, "inner", None)
        root = pathlib.Path(app._PROJECT_ROOT)
        return {
            "schema_version": recorder.SCHEMA_VERSION,
            "episode_id": episode_id,
            # 2026-09-17 (episode ledger); absent in older episodes
            "episode_id_scheme": ledger.EPISODE_ID_SCHEME,
            "storage_kind": self.kind,
            "host": self.host,
            "ledger_seq": ledger_seq,
            "session": self.session,
            "operator": self.operator_id,
            "recorded_at": self.recorded_at,
            "saved_at": recorder._now_iso(),
            "instruction": recorder.INSTRUCTION,
            "placement_id": p.placement_id,
            "placement_seed": p.seed,
            "cube_init": {"x": p.x, "y": p.y, "z": recorder.CUBE_HALF, "yaw": p.yaw},
            "retakes": self.retakes.get(p.placement_id, 0),
            "success": success,
            "final_cube_pos": [float(v) for v in last["cube_pos"]],
            "control_hz": round(1.0 / self.timestep),
            "timestep": self.timestep,
            "record_hz": round(1.0 / (RECORD_EVERY * self.timestep)),
            "record_every": RECORD_EVERY,
            "duration_s": (n_keep - 1) * RECORD_EVERY * self.timestep,
            "truncated_at_step": self._press_step,
            "save_hold_steps": SAVE_HOLD_STEPS,
            "timing": {"wall_s": round(t["wall"], 3), "sim_s": round(t["sim"], 3),
                       "rtf": round(t["sim"] / t["wall"], 4) if t["wall"] > 0 else None,
                       "backlog_drops": t["drops"],
                       "max_loop_dt_s": round(t["max_loop_dt"], 4),
                       # 中断 material: loop iterations over the threshold, and the threshold itself
                       "stalls": t["stalls"],
                       "stall_threshold_s": STALL_THRESHOLD_S,
                       # camera-stall material; a person decides (recorder.EpisodeWriter)
                       "frozen_frames": dict(self.writer.frozen_frames)},
            "cameras": {"names": list(CAMERAS), "width": IMAGE_SIZE,
                        "height": IMAGE_SIZE, "format": "png", "rotation": "none"},
            "gripper": {"cmd_range": [0, 255], "closed_cmd": 0, "open_cmd": 255,
                        "note": "raw actuator command; conversion maps to -1 open / +1 close"},
            "frame_definition": "frame i at physics step i*record_every; frame 0 = reset "
                                "state; images and state from the same instant",
            "action_definition": "action_xyz[i] = x_des[i+1] - x_des[i] (sum over the "
                                 "window after frame i); action_gripper[i] from "
                                 "frame i+1; rotation 3 = 0; last frame has no action",
            "state_layout": {"ee_pos": 3, "ee_axisangle": 3, "fingers": 2, "joints": 7},
            "ee_reference": "hand body origin (x_des tracks the same point)",
            "start_pos": [float(v) for v in START_POS],
            "controller": {"class": type(c).__name__,
                           **{k: getattr(c, k) for k in CONTROLLER_PARAMS}},
            "pad": {k: (getattr(pad, k).tolist() if isinstance(getattr(pad, k, None), np.ndarray)
                        else getattr(pad, k, None))
                    for k in ("profile_name", "xy_speed", "z_speed", "deadzone",
                              "trigger_rest")},
            "software": {"mujoco": mujoco.__version__,
                         "sha256": {f: recorder.file_sha256(root / f) for f in HASHED_FILES}},
        }

    def status_lines(self) -> list:
        p = self.placement
        lines = []
        if self.state == self.RECORDING:
            secs = self.episode_step * self.timestep
            lines.append((f"REC  {secs:5.1f} s", _RED))
        else:
            lines.append(("READY  pad locked", _GREEN))
            lines.append(("press O to start recording", _GREEN))
        if self.kind != "production":
            lines.append((f"{self.kind.upper()}: saving to {self.raw_root.name}, not raw", _YELLOW))
        lines.append((f"episode {self.next_episode_id:06d}   session {self.session}", _WHITE))
        lines.append((f"operator {self.operator_id}", _WHITE))
        lines.append((f"placement {p.placement_id} (seed {p.seed})  "
                      f"saved {self.saved.get(p.placement_id, 0)}/{self.reps_target}", _WHITE))
        lines.append((f"retakes {self.retakes.get(p.placement_id, 0)}   "
                      f"this session {self.session_saved}/{recorder.SESSION_EPISODE_LIMIT}",
                      _YELLOW if self.session_saved >= recorder.SESSION_EPISODE_LIMIT
                      else _WHITE))
        if self.state == self.RECORDING:
            if self._hold_steps:
                n = int(20 * self._hold_steps / SAVE_HOLD_STEPS)
                lines.append(("SAVE [" + "#" * n + "." * (20 - n) + "]", _GREEN))
            if self.episode_step * self.timestep > LONG_EPISODE_S:
                lines.append(("long episode (15-30 s expected)", _YELLOW))
            if self.timing["drops"]:
                lines.append((f"stall x{self.timing['drops']} in this episode", _YELLOW))
        if all(self.saved.get(q.placement_id, 0) >= self.reps_target for q in self.placements):
            lines.append(("all placements reached the target", _GREEN))
        if self.message[0]:
            lines.append(self.message)
        lines.append(("O start  /\\ hold 1s save", _GRAY))
        lines.append(("pad click discard  PS quit", _GRAY))
        return lines

    def close(self, backup: bool = True) -> None:
        """backup=True only when the sitting ended normally: copies the ledger
        to its backup folder (a failed copy warns and is recorded, never
        raises)."""
        try:
            if self.writer is not None:
                self.discard("closed")
            self.log.write("session_end", saved=self.session_saved)
        finally:
            self.ledger.close(backup=backup)


# ----------------------------------------------------------------- main

def _make_pad() -> DeviceInput:
    if app._device_kind() == "fake":  # hardware-free smoke runs
        from teleop.dualsense_device import FakeDualSenseInput
        return FakeDualSenseInput(profile="collect")
    from teleop.dualsense_device import DualSenseInput
    return DualSenseInput(profile="collect")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="VLA data collection (DualSense)")
    parser.add_argument("--operator", required=True,
                        help="operator id recorded in every episode, e.g. 01")
    parser.add_argument("--reps", type=int, default=8,
                        help="target saved episodes per placement (default 8)")
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--raw-dir", default=str(recorder.DEFAULT_RAW_DIR),
                       help="raw episode root (default: 03_収録/raw)")
    where.add_argument("--practice", action="store_true",
                       help=f"save to {PRACTICE_DIR} instead of raw (training runs)")
    parser.add_argument("--ledger-dir", default=str(ledger.DEFAULT_LEDGER_DIR),
                        help="episode ledger (default: 99_台帳/episode_ledger)")
    args = parser.parse_args(argv)
    if args.practice:
        args.raw_dir = str(PRACTICE_DIR)
    # the id band follows the mode; the ledger refuses a folder registered for another kind
    args.kind = "practice" if args.practice else "production"
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    model = mujoco.MjModel.from_xml_path(app.SCENE_PATH)
    data = mujoco.MjData(model)
    timestep = float(model.opt.timestep)
    print("[collect] settling start state ...")
    start = settle_start_state(model)

    controller = make_collect_controller(model, data)
    integrator = make_integrator(_make_pad(), model, controller)
    renderer = mujoco.Renderer(model, IMAGE_SIZE, IMAGE_SIZE)
    # the episode ledger is checked before the pad is calibrated: a refused
    # start (another recording running, unregistered folder, host not allowed
    # for production) stops here with the reason
    try:
        session = CollectSession(model, data, controller, integrator, start, renderer,
                                 args.raw_dir, args.operator, args.reps,
                                 episode_ledger=ledger.EpisodeLedger(args.ledger_dir), kind=args.kind)
    except ledger.LedgerError as exc:
        close_renderer(renderer)
        print(f"\n[collect] NOT STARTED (episode ledger): {exc}\n")
        raise SystemExit(2)
    print(f"[collect] {session.kind} session {session.session}, next episode id "
          f"{session.next_episode_id:06d}, raw dir {session.raw_root}")
    operator = OperatorView(model, controller)
    monitor = RealTimeFactorMonitor()
    max_steps = app._max_steps()

    total_steps = 0
    sim_debt = 0.0
    normal_end = False
    try:
        integrator.start()  # pad calibration: hands off the pad now
        with mujoco.viewer.launch_passive(model, data) as viewer:
            with viewer.lock():
                viewer.opt.geomgroup[5] = 1  # mocap target: viewer only
            operator.open_window()
            t_prev = time.perf_counter()
            while viewer.is_running():
                now = time.perf_counter()
                loop_dt = min(now - t_prev, 0.25)
                t_prev = now

                state = integrator.refresh()
                prev_state = session.state
                if session.handle_input(state):
                    break
                if session.state != prev_state:
                    sim_debt = 0.0  # a reset happened: start the clock fresh

                sim_debt += loop_dt
                n = 0
                while sim_debt >= timestep and n < MAX_CATCHUP_STEPS:
                    controller.update(integrator)
                    mujoco.mj_step(model, data)
                    sim_debt -= timestep
                    n += 1
                    session.after_step()
                dropped = n == MAX_CATCHUP_STEPS and sim_debt >= timestep
                if dropped:
                    sim_debt = 0.0
                monitor.update(n * timestep, n, dropped)
                session.note_loop(loop_dt, n * timestep, dropped)

                operator.set_status(session.status_lines())
                operator.maybe_show(data, now)
                cv2.waitKey(1)
                viewer.sync()
                total_steps += n
                if max_steps and total_steps >= max_steps:
                    break
        normal_end = True
    finally:
        session.close(backup=normal_end)   # ledger backup only after a normal end
        integrator.stop()
        operator.close()
        close_renderer(renderer)
        cv2.destroyAllWindows()
        print(monitor.summary())
        print(f"[collect] saved {session.session_saved} episode(s) this session")
