"""The ONE place that defines how recorded proprioception becomes the policy's observation.state.

Used by convert_to_lerobot.episode_arrays (training dataset) and vla_observation (evaluation
entry), so the two cannot compute the state differently. Moved here verbatim from
convert_to_lerobot.py on 2026-09-17; the converted state arrays of the Step D / E-D scripted
episodes are bit-identical before and after the move.

observation.state (15, float32), from the recorder's frame fields (teleop/recorder.py):

  eef_x, eef_y, eef_z               ee_pos: hand body origin [m, world]
  eef_rot_dev_x, _y, _z             orientation deviation from pointing straight down [rad]
  finger_left, finger_right_neg     fingers: finger_joint1, -finger_joint2 [m] (LIBERO sign)
  joint1 .. joint7                  joints [rad]

Orientation (decided 2026-09-16): world-frame axis-angle of q * conj(Q_DOWN), Q_DOWN = (0, 1, 0, 0)
the locked straight-down hand orientation, so all three values sit near 0. LIBERO's absolute
axis-angle was not used: with the hand down its x component sits near pi (the raw [-pi, pi] form
even jumps between +pi and -pi), and LeRobot 0.6.1 computes variance as mean(x^2) - mean(x)^2 in
float32, which cancels to exactly 0 for a value near pi varying by 3e-4 -- normalized states then
reached 1e5.

Depends on numpy only (python311 and the LeRobot venv both import it).
"""
import numpy as np

STATE_VERSION = 1
Q_DOWN = np.array([0.0, 1.0, 0.0, 0.0])   # w, x, y, z: pi about world x
STATE_NAMES = (["eef_x", "eef_y", "eef_z", "eef_rot_dev_x", "eef_rot_dev_y",
                "eef_rot_dev_z", "finger_left", "finger_right_neg"]
               + [f"joint{i}" for i in range(1, 8)])
STATE_DIM = len(STATE_NAMES)
FRAME_FIELDS = {"ee_pos": 3, "ee_quat": 4, "fingers": 2, "joints": 7}   # recorder field -> width


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of (N, 4) w,x,y,z arrays (b may be (4,))."""
    b = np.broadcast_to(b, a.shape)
    aw, ax, ay, az = a.T
    bw, bx, by, bz = b.T
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=1)


def orientation_deviation(quat: np.ndarray) -> np.ndarray:
    """(N, 4) hand quaternions -> (N, 3) world-frame axis-angle of
    q * conj(Q_DOWN), angle in [0, pi] (0 = pointing straight down)."""
    conj_down = Q_DOWN * np.array([1.0, -1.0, -1.0, -1.0])
    d = quat_mul(np.asarray(quat, dtype=np.float64), conj_down)
    d[d[:, 0] < 0] *= -1.0
    s = np.linalg.norm(d[:, 1:], axis=1)
    angle = 2.0 * np.arctan2(s, d[:, 0])
    scale = np.where(s > 1e-12, angle / np.maximum(s, 1e-12), 2.0)  # small-angle limit
    return d[:, 1:] * scale[:, None]


def policy_state(ee_pos, ee_quat, fingers, joints) -> np.ndarray:
    """(N, ...) recorder fields -> (N, 15) float32 observation.state."""
    arrays = {"ee_pos": np.asarray(ee_pos), "ee_quat": np.asarray(ee_quat),
              "fingers": np.asarray(fingers), "joints": np.asarray(joints)}
    n = None
    for name, width in FRAME_FIELDS.items():
        a = arrays[name]
        if a.ndim != 2 or a.shape[1] != width:
            raise ValueError(f"{name}: expected (N, {width}), got {a.shape}")
        if n is None:
            n = a.shape[0]
        elif a.shape[0] != n:
            raise ValueError(f"{name}: {a.shape[0]} rows, expected {n}")
    fingers = arrays["fingers"].copy()
    fingers[:, 1] *= -1.0
    state = np.concatenate([arrays["ee_pos"], orientation_deviation(arrays["ee_quat"]),
                            fingers, arrays["joints"]], axis=1)
    return state.astype(np.float32)


def policy_state_frame(frame: dict) -> np.ndarray:
    """One frame {ee_pos (3,), ee_quat (4,), fingers (2,), joints (7,)} -> (15,) float32."""
    missing = set(FRAME_FIELDS) - set(frame)
    if missing:
        raise ValueError(f"proprio frame is missing {sorted(missing)}")
    rows = {}
    for name, width in FRAME_FIELDS.items():
        a = np.asarray(frame[name])
        if a.shape != (width,):
            raise ValueError(f"{name}: expected ({width},), got {a.shape}")
        rows[name] = a[None]
    return policy_state(rows["ee_pos"], rows["ee_quat"], rows["fingers"], rows["joints"])[0]
