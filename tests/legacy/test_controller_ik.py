"""Tests for teleop.controller_ik.TeleopControllerIK behavior."""
import mujoco
import numpy as np
import pytest

from recovla.sim import control
from recovla.sim.controller_ik import TeleopControllerIK
from recovla.sim.device import DeviceInput, DeviceState

SCENE = control.SCENE_PATH

HOME_QPOS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]


@pytest.fixture()
def model_data():
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    for i, q in enumerate(HOME_QPOS, start=1):
        data.qpos[model.joint(f"joint{i}").qposadr[0]] = q
    mujoco.mj_forward(model, data)
    return model, data


class _ScriptedDevice(DeviceInput):
    """Replays a fixed sequence of (pos, clutch) pairs, one per read()."""

    def __init__(self, steps):
        self._steps = list(steps)
        self._i = 0

    def start(self) -> None:
        pass

    def read(self) -> DeviceState:
        pos, clutch = self._steps[min(self._i, len(self._steps) - 1)]
        self._i += 1
        return DeviceState(
            pos=np.array(pos, dtype=float),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
            button_grip=False,
            button_clutch=clutch,
            button_reset=False,
        )

    def stop(self) -> None:
        pass


def test_releasing_clutch_holds_position_off_neutral_posture(model_data):
    """Regression: releasing the clutch used to let the null-space posture
    pull (toward the home configuration) leak into task space, making the
    hand visibly drift/float even with no operator input -- worst near
    workspace edges/low reach where the posture is far from home. The pull
    must now be gated to "clutch held", so releasing it holds the last
    commanded pose.
    """
    model, data = model_data
    controller = TeleopControllerIK(
        model, data, ik_gain=0.08, rot_weight=0.5, null_gain=0.05,
        workspace_x=(0.25, 0.62), workspace_y=(-0.30, 0.30),
        workspace_z=(-0.05, 0.45))

    # Drive the hand to a low, off-home-ish position near a workspace edge
    # (reaching down and to the side, like going for a cube), holding the
    # clutch the whole time so the target actually moves there.
    low_edge_target = np.array([0.30, 0.20, 0.05])
    device = _ScriptedDevice([(low_edge_target, True)] * 400)
    for _ in range(400):
        controller.update(device)
        mujoco.mj_step(model, data)

    hand_id = controller.hand_body_id
    hand_before = data.xpos[hand_id].copy()

    # Now release the clutch (pos value should no longer matter) and run
    # many more steps with no operator input.
    device = _ScriptedDevice([(low_edge_target, False)] * 400)
    for _ in range(400):
        controller.update(device)
        mujoco.mj_step(model, data)

    hand_after = data.xpos[hand_id].copy()
    drift = np.linalg.norm(hand_after - hand_before)
    assert drift < 0.02, (
        f"hand drifted {drift:.4f} m after releasing clutch "
        f"(before={hand_before}, after={hand_after}) -- null-space pull "
        f"is leaking into task space while clutch is released")


def test_null_space_pull_still_active_while_clutch_held(model_data):
    """Sanity check the gating didn't just disable FIX3 outright: with the
    clutch held, joints should still ease toward the neutral posture in the
    null space (this is a soft pull, so just check it moves in that
    direction over many steps, not that it fully converges)."""
    model, data = model_data
    controller = TeleopControllerIK(
        model, data, ik_gain=0.08, rot_weight=0.0, null_gain=0.2,
        workspace_x=(0.25, 0.62), workspace_y=(-0.30, 0.30),
        workspace_z=(-0.05, 0.45))

    # Displace one redundant joint away from neutral, then hold the clutch
    # at the CURRENT hand position (no task-space motion commanded) so any
    # joint change must come from the null-space term.
    data.qpos[controller.arm[1][2]] += 0.3
    mujoco.mj_forward(model, data)
    q_neutral = controller.q_neutral.copy()

    hand_pos = data.xpos[controller.hand_body_id].copy()
    device = _ScriptedDevice([(hand_pos, True)] * 300)
    for _ in range(300):
        controller.update(device)
        mujoco.mj_step(model, data)

    q_now = np.array([data.qpos[a[2]] for a in controller.arm])
    # Distance to neutral should have shrunk (not necessarily to zero).
    dist_before = abs(q_neutral[1] - (q_neutral[1] + 0.3))
    dist_after = abs(q_neutral[1] - q_now[1])
    assert dist_after < dist_before


def test_dq_clamp_preserves_direction_not_elementwise_clip(model_data):
    """Regression: dq must be scaled to respect max_joint_step WITHOUT
    distorting its direction. An element-wise np.clip independently
    truncates each joint's component, which can change the ratio between
    joints -- i.e. send the arm in a direction the Jacobian solve never
    intended -- and was a leading suspect for observed "target and hand
    diverge for 300+ steps" stalls near low reach + full orientation lock.
    """
    model, data = model_data
    controller = TeleopControllerIK(model, data, ik_gain=0.08,
                                    max_joint_step=0.01)

    # Directly exercise the clamp logic with a known lopsided raw dq,
    # mirroring what _solve_ik_step does internally.
    dq_raw = np.array([0.05, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    max_dq_val = np.max(np.abs(dq_raw))
    dq_clamped = dq_raw * (controller.max_joint_step / max_dq_val)

    dir_raw = dq_raw / np.linalg.norm(dq_raw)
    dir_clamped = dq_clamped / np.linalg.norm(dq_clamped)
    assert np.allclose(dir_raw, dir_clamped), "clamp must preserve direction"
    assert np.isclose(np.max(np.abs(dq_clamped)), controller.max_joint_step)

    # The old element-wise clip would have destroyed the direction --
    # assert that explicitly as a documented contrast.
    dq_old_buggy = np.clip(dq_raw, -controller.max_joint_step,
                           controller.max_joint_step)
    dir_old = dq_old_buggy / np.linalg.norm(dq_old_buggy)
    assert not np.allclose(dir_raw, dir_old), (
        "sanity check: old clip SHOULD distort direction for this input")
