import abc
import dataclasses
from typing import Optional
import numpy as np
import cv2

@dataclasses.dataclass
class DeviceState:
    pos: np.ndarray
    quat: np.ndarray
    button_grip: bool
    button_clutch: bool
    button_reset: bool = False
    button_home: bool = False   # demo home-reset: arm -> HOME_QPOS + cubes
    # Analog gripper command in [0, 1] (0 = fully open, 1 = fully closed),
    # e.g. DualSense R2 trigger pull. None (default) means "no analog
    # command available" -- every existing DeviceState construction site
    # is unaffected, and TeleopControllerIK._update_gripper falls back to
    # its legacy rising-edge toggle when this is None (see controller_ik.py
    # §6-2). Optional so old callers/tests that build DeviceState
    # positionally/by keyword without this field keep working unchanged.
    grip_analog: "Optional[float]" = None
    # VLA data collection (DualSense profile "collect", 2026-09-16). All
    # optional with inert defaults, so every existing construction site and
    # consumer is unaffected.
    # vel: commanded hand velocity [m/s], world frame, BEFORE integration.
    #   The collect loop integrates it once per physics step into x_des
    #   instead of using `pos` (which the device integrates at its own,
    #   wall-clock-measured read rate).
    vel: "Optional[np.ndarray]" = None
    button_save: bool = False     # held level (long-press is decided by the caller)
    button_discard: bool = False
    button_quit: bool = False

class DeviceInput(abc.ABC):
    @abc.abstractmethod
    def start(self) -> None: pass
    @abc.abstractmethod
    def read(self) -> DeviceState: pass
    @abc.abstractmethod
    def stop(self) -> None: pass


class KeyboardButtonAugment(DeviceInput):
    """Wraps another DeviceInput and merges keyboard buttons into its state.

    Motivation: stylus button 2 on the lab's Touch unit is physically dead
    (confirmed in the Touch Setup utility on 2026-07-10), so gripper toggle
    and reset moved to the keyboard:

      SPACE -> button_grip  (one-frame press; the controller edge-detects
               and toggles the gripper once per press)
      r     -> button_reset (one-frame press; respawns the cubes)
      h     -> button_home  (one-frame press; returns the arm to HOME_QPOS
               and respawns the cubes. Use this instead of the viewer's
               Backspace, which resets to the upright singular pose.)

    Keys are polled with cv2.waitKey, so AN OPENCV WINDOW MUST HAVE FOCUS
    for keys to register (in this app that is the "WRIST VIEW" window).
    Clutch stays on the working stylus button 1 via the inner device.

    Side benefit: cv2.imshow needs a waitKey pump to actually repaint; the
    mouse device used to provide it, but in touch/fake mode nothing did, so
    the wrist view would freeze. This wrapper restores the pump.
    """

    GRIP_KEYS = (ord(' '),)
    RESET_KEYS = (ord('r'),)
    HOME_KEYS = (ord('h'),)

    def __init__(self, inner: DeviceInput):
        self.inner = inner

    def start(self) -> None:
        self.inner.start()

    def read(self) -> DeviceState:
        state = self.inner.read()
        key = cv2.waitKey(1) & 0xFF
        if key in self.GRIP_KEYS:
            state.button_grip = True
        if key in self.RESET_KEYS:
            state.button_reset = True
        if key in self.HOME_KEYS:
            state.button_home = True
        return state

    def set_force(self, force_world_N) -> None:
        """Forward force feedback to the inner device when supported."""
        fn = getattr(self.inner, "set_force", None)
        if callable(fn):
            fn(force_world_N)

    def read_camera_delta(self):
        """Forward DualSenseInput's touchpad camera delta (§7) when the
        inner device supports it; (0.0, 0.0) otherwise (mouse/touch/fake
        have no camera side-channel, so this is a safe no-op default)."""
        fn = getattr(self.inner, "read_camera_delta", None)
        if callable(fn):
            return fn()
        return (0.0, 0.0)

    def set_lightbar(self, r, g, b) -> None:
        """Forward lightbar color (§5) to the inner device when supported."""
        fn = getattr(self.inner, "set_lightbar", None)
        if callable(fn):
            fn(r, g, b)

    def set_trigger(self, which, mode, params=None) -> None:
        """Forward adaptive-trigger effect (§6-3) when supported."""
        fn = getattr(self.inner, "set_trigger", None)
        if callable(fn):
            fn(which, mode, params)

    def stop(self) -> None:
        self.inner.stop()

class MouseKeyboardInput(DeviceInput):
    # Intentionally shares the wrist camera's window so mouse drags on the
    # camera image drive the robot. The window is created by whichever side
    # comes first (cv2.namedWindow is idempotent); destruction is owned by
    # the app's cv2.destroyAllWindows() in its finally block, NOT by stop().
    WINDOW_NAME = "WRIST VIEW"

    def __init__(self, init_pos: Optional[np.ndarray] = None,
                 xy_sensitivity: float = 0.002, z_sensitivity: float = 0.01,
                 window_name: Optional[str] = None) -> None:
        self.window_name = window_name or self.WINDOW_NAME
        self.pos = np.array(init_pos if init_pos is not None else [0.5, 0.0, 0.2], dtype=float)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.button_grip = False
        self.button_clutch = False
        self.button_reset = False
        self.xy_sensitivity = xy_sensitivity
        self.z_sensitivity = z_sensitivity
        self._last_x: Optional[float] = None
        self._last_y: Optional[float] = None
        self._mouse_down: bool = False
        self._running: bool = False

    def start(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        self._running = True

    def _mouse_callback(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._mouse_down = True
            self._last_x = x
            self._last_y = y
        elif event == cv2.EVENT_LBUTTONUP:
            self._mouse_down = False
            self._last_x = None
            self._last_y = None
        elif event == cv2.EVENT_MOUSEMOVE and self._mouse_down:
            if self._last_x is None:
                self._last_x = x
                self._last_y = y
                return
            self.pos[0] += (x - self._last_x) * self.xy_sensitivity
            self.pos[1] -= (y - self._last_y) * self.xy_sensitivity
            self._last_x = x
            self._last_y = y

    def _process_keyboard(self) -> None:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('g'):
            self.button_grip = not self.button_grip
        self.button_reset = (key == ord('r'))
        if key == ord('w'):
            self.pos[2] += self.z_sensitivity
        elif key == ord('s'):
            self.pos[2] -= self.z_sensitivity

    def read(self) -> DeviceState:
        if not self._running:
            self.start()
        self._process_keyboard()
        return DeviceState(
            pos=self.pos.copy(),
            quat=self.quat.copy(),
            button_grip=self.button_grip,
            button_clutch=self._mouse_down,
            button_reset=self.button_reset,
        )

    def stop(self) -> None:
        # Do NOT destroy the window here: it is shared with WristCamera and
        # the app tears everything down with cv2.destroyAllWindows().
        self._running = False