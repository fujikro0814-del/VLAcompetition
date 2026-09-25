import numpy as np
import mujoco

from teleop.device import DeviceInput, DeviceState


class TeleopControllerIK:
    """
    Differential IK (Damped Least Squares) teleop controller. v3.

    v3 changes (2026-07-11), targeting vibration + tracking lag:
      FIX4: the hard per-step rate limiter on the target (FIX1) is replaced
            by a critically damped second-order tracker. The raw desired
            position from the clutch is now only a *reference*; the actual
            mocap target follows it with dynamics
                a = w^2 (desired - target) - 2 w v
            integrated at the control dt (semi-implicit Euler). This gives
            a smooth, overshoot-free trajectory instead of the velocity-
            saturated staircase that excited the stiff joint servos
            (kp 2000-4500) into chatter. Set tracker_omega=0 to fall back
            to the old rate limiter for A/B comparison.
      FIX5: target speed limit is now time-based (max_target_vel [m/s]),
            so retuning is not needed if the loop rate changes.
      FIX6: ik_gain default raised 0.2 -> 0.5. The gain was kept low as a
            noise band-aid; with FilterAugment (teleop/filters.py) cleaning
            the device stream and FIX4 shaping the target, the loop can be
            faster. Explicitly passed values are honored unchanged.

    Fixes based on diagnostic log (2026-07-09):
      FIX1: target is clamped into the workspace at init, and target motion is
            rate-limited (max_target_step) so workspace clamping can never
            teleport the target (0.35 m jump caused the blow-up).
            (superseded by FIX4 unless tracker_omega=0)
      FIX2: anti-windup leash: q_des is never allowed to run further than
            q_des_leash [rad] ahead of the measured joint position, so hitting
            a joint limit or slow tracking cannot accumulate command error.
      FIX3: optional null-space posture term toward q_neutral (disabled by
            default; enable after the home pose is set in app.py).

    NOTE: the robot must NOT start in the upright singular pose (all q = 0).
    Set the Panda home pose in app.py before creating this controller:

        HOME_QPOS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        for i, q in enumerate(HOME_QPOS, start=1):
            data.qpos[model.joint(f"joint{i}").qposadr[0]] = q
        mujoco.mj_forward(model, data)

    Remove/comment the <equality><weld> block in teleop_scene.xml.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        scale: float = 2.0,
        workspace_x=(0.25, 0.70),
        workspace_y=(-0.35, 0.35),
        workspace_z=(-0.06, 0.60),
        ik_damping: float = 0.05,
        ik_gain: float = 0.5,           # per-step task gain (FIX6)
        max_joint_step: float = 0.01,   # [rad] per control step
        max_target_step: float = 0.004, # [m] per control step (FIX1 fallback)
        tracker_omega: float = 20.0,    # [rad/s] target tracker bandwidth (FIX4)
        max_target_vel: float = 0.8,    # [m/s] target speed limit (FIX5)
        ctrl_dt: float = 0.0,           # [s] control period; 0 = model timestep
        q_des_leash: float = 0.10,      # [rad] max lead of q_des over qpos (FIX2)
        null_gain: float = 0.0,         # null-space posture gain (FIX3, 0=off)
        max_dx_norm: float = 0.10,      # task error saturation
        pos_weight: float = 1.0,
        rot_weight: float = 0.0,        # orientation tracking off by default
        lock_orientation: bool = False, # True: clutch never rotates the target;
                                        # with rot_weight>0 the IK then holds the
                                        # initial (home) hand orientation forever.
        end_effector_body: str = "hand",
    ):
        self.model = model
        self.data = data
        self.scale = scale

        self.workspace_x = workspace_x
        self.workspace_y = workspace_y
        self.workspace_z = workspace_z

        self.ik_damping = ik_damping
        self.ik_gain = ik_gain
        self.max_joint_step = max_joint_step
        self.max_target_step = max_target_step
        self.tracker_omega = tracker_omega
        self.max_target_vel = max_target_vel
        self.ctrl_dt = ctrl_dt if ctrl_dt > 0.0 else float(model.opt.timestep)
        self.q_des_leash = q_des_leash
        self.null_gain = null_gain
        self.max_dx_norm = max_dx_norm
        self.pos_weight = pos_weight
        self.rot_weight = rot_weight
        self.lock_orientation = lock_orientation

        if model.nmocap < 1:
            raise RuntimeError("Scene must contain at least one mocap body (visual target).")
        self.mocap_id = 0

        self.hand_body_id = model.body(end_effector_body).id

        # ---- gripper actuators (by name, no hardcoding) ----
        self.gripper_indices = []
        for i in range(model.nu):
            name = model.actuator(i).name
            if name is not None:
                lname = name.lower()
                if "gripper" in lname or "finger" in lname:
                    self.gripper_indices.append(i)

        # ---- arm joint actuators: (act_idx, joint_id, qpos_adr, dof_adr) ----
        self.arm = []
        for i in range(model.nu):
            if i in self.gripper_indices:
                continue
            if model.actuator_trntype[i] != 0:  # mjTRN_JOINT
                continue
            joint_id = model.actuator_trnid[i][0]
            self.arm.append((i, joint_id,
                             model.jnt_qposadr[joint_id],
                             model.jnt_dofadr[joint_id]))
        if len(self.arm) == 0:
            raise RuntimeError("No arm joint actuators found.")

        self.dof_indices = np.array([a[3] for a in self.arm], dtype=int)
        self.n_arm = len(self.arm)
        self.q_min = np.array([model.jnt_range[a[1]][0] for a in self.arm])
        self.q_max = np.array([model.jnt_range[a[1]][1] for a in self.arm])

        mujoco.mj_forward(model, data)

        self.q_des = np.array([data.qpos[a[2]] for a in self.arm])
        self.q_neutral = self.q_des.copy()

        # target starts at current hand pose, clamped INTO the workspace (FIX1)
        self.target_pos = self._clamp_workspace(
            data.xpos[self.hand_body_id].copy())
        self.target_quat = data.xquat[self.hand_body_id].copy()

        # FIX4 tracker state: the clutch writes desired_pos; the tracker
        # moves target_pos toward it every step with critically damped
        # dynamics, including after the clutch is released (smooth settle).
        self.desired_pos = self.target_pos.copy()
        self.target_vel = np.zeros(3)

        # clutch references
        self.device_ref_pos = None
        self.target_ref_pos = None
        self.device_ref_quat = None
        self.target_ref_quat = None

        self.gripper_closed = True
        self.prev_grip = False
        self.prev_clutch = False

        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self._log_counter = 0

        # Set by _solve_ik_step (raw 6xN task Jacobian of the last IK
        # solve). teleop/pad_feedback.py reads this (via last_sigma_min)
        # to compute the "health" (manipulability) scalar for the
        # DualSense lightbar/trigger feedback -- see pad_feedback.py §5.
        # This class does not interpret it; the control law is untouched.
        self.last_jac = None

    # ------------------------------------------------------------------ misc

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def sync_target_to_hand(self) -> None:
        """Re-anchor the mocap target to the CURRENT hand pose and clear
        all tracker/clutch state.

        Call this right after externally rewriting data.qpos (e.g. a home
        reset), and after mj_forward has refreshed data.xpos/xquat. If the
        target is left at its old value while the arm is teleported home,
        the very next step sees a large target-vs-hand error and the IK
        loop lurches -- the same failure mode as the 'tilts and never
        returns' log. Zeroing target_vel and the clutch refs makes the
        reset a clean, motionless restart.
        """
        self.target_pos = self._clamp_workspace(
            self.data.xpos[self.hand_body_id].copy())
        self.target_quat = self.data.xquat[self.hand_body_id].copy()
        self.desired_pos = self.target_pos.copy()
        self.target_vel = np.zeros(3)
        self.device_ref_pos = None
        self.target_ref_pos = None
        self.device_ref_quat = None
        self.target_ref_quat = None
        # Force the clutch to re-grab a fresh reference on next press.
        self.prev_clutch = False
        self.data.mocap_pos[self.mocap_id] = self.target_pos
        self.data.mocap_quat[self.mocap_id] = self.target_quat

    @property
    def last_sigma_min(self) -> float:
        """Smallest singular value of the last task Jacobian (0.0 before
        the first IK solve). Read-only; pad_feedback.py's h_sigma consumes
        this. Kept as a property (computed on demand from last_jac) rather
        than cached at solve time so it never goes stale relative to
        last_jac if callers poke that attribute directly in tests."""
        if self.last_jac is None:
            return 0.0
        return float(np.min(np.linalg.svd(self.last_jac, compute_uv=False)))

    def set_target_quat(self, quat) -> None:
        """Snap the orientation target directly (§9-6, optional water-
        horizontal preset). Assignment + normalization only -- does not
        touch the control law, the clutch reference, or position target.
        Callers are responsible for re-anchoring the clutch (as
        sync_target_to_hand does for a full reset) if a sudden jump would
        otherwise fight an in-progress clutch drag.
        """
        q = np.array(quat, dtype=float)
        mujoco.mju_normalize4(q)
        self.target_quat = q
        self.data.mocap_quat[self.mocap_id] = self.target_quat

    def _clamp_workspace(self, pos: np.ndarray) -> np.ndarray:
        pos[0] = np.clip(pos[0], self.workspace_x[0], self.workspace_x[1])
        pos[1] = np.clip(pos[1], self.workspace_y[0], self.workspace_y[1])
        pos[2] = np.clip(pos[2], self.workspace_z[0], self.workspace_z[1])
        return pos

    # -------------------------------------------------------------- gripper

    def _update_gripper(self, grip_button: bool, grip_analog=None) -> None:
        """§6-2: analog gripper (DualSense R2), additive to the legacy
        rising-edge toggle used by Touch/mouse/fake.

        grip_analog is None (default): unchanged legacy path -- toggle
        open/closed on the rising edge of grip_button. Every existing
        caller (positional or via update()) that never passes grip_analog
        goes through exactly this branch, so behavior is byte-identical.

        grip_analog is a float: continuous command, ctrl = hi + (lo - hi)
        * clip(grip_analog, 0, 1) sent every frame (0 = open, 1 = closed).
        This branch bypasses the toggle state entirely -- an analog device
        commands position directly, it doesn't need edge-detected toggling
        -- but prev_grip is still updated so a caller can freely alternate
        between the two calling conventions (e.g. profile switch) without
        a stale edge firing.
        """
        if grip_analog is not None:
            g = float(np.clip(grip_analog, 0.0, 1.0))
            for idx in self.gripper_indices:
                lo = self.model.actuator_ctrlrange[idx][0]
                hi = self.model.actuator_ctrlrange[idx][1]
                self.data.ctrl[idx] = hi + (lo - hi) * g
            self.prev_grip = grip_button
            return

        if grip_button and not self.prev_grip:
            self.gripper_closed = not self.gripper_closed
            for idx in self.gripper_indices:
                lo = self.model.actuator_ctrlrange[idx][0]
                hi = self.model.actuator_ctrlrange[idx][1]
                self.data.ctrl[idx] = lo if self.gripper_closed else hi
        self.prev_grip = grip_button

    # --------------------------------------------------------------- clutch

    def _update_clutch(self, clutch_button, device_pos, device_quat) -> None:
        if clutch_button and not self.prev_clutch:
            self.device_ref_pos = device_pos.copy()
            self.target_ref_pos = self.target_pos.copy()
            self.device_ref_quat = device_quat.copy()
            self.target_ref_quat = self.target_quat.copy()

        if clutch_button and self.device_ref_pos is not None:
            delta_pos = device_pos - self.device_ref_pos
            desired = self.target_ref_pos + self.scale * delta_pos
            # FIX4: this is only the tracker reference; the actual target
            # is advanced in _track_target() with smooth dynamics.
            self.desired_pos = self._clamp_workspace(desired)

            if self.rot_weight > 0.0 and not self.lock_orientation:
                inv_ref = np.zeros(4)
                mujoco.mju_negQuat(inv_ref, self.device_ref_quat)
                dquat = np.zeros(4)
                mujoco.mju_mulQuat(dquat, device_quat, inv_ref)
                new_quat = np.zeros(4)
                mujoco.mju_mulQuat(new_quat, dquat, self.target_ref_quat)
                mujoco.mju_normalize4(new_quat)
                self.target_quat = new_quat

        self.prev_clutch = clutch_button

        self._track_target()

        self.data.mocap_pos[self.mocap_id] = self.target_pos
        self.data.mocap_quat[self.mocap_id] = self.target_quat

    # -------------------------------------------------- target tracker (FIX4)

    def _track_target(self) -> None:
        """Advance target_pos toward desired_pos by one control step.

        tracker_omega > 0: critically damped second-order system
        (semi-implicit Euler), speed-clamped at max_target_vel. Runs even
        when the clutch is open so a release settles smoothly instead of
        freezing mid-transient.

        tracker_omega <= 0: legacy FIX1 per-step rate limiter, kept for
        A/B comparison.
        """
        if self.tracker_omega <= 0.0:
            step = self.desired_pos - self.target_pos
            n = np.linalg.norm(step)
            if n > self.max_target_step:
                step *= self.max_target_step / n
            self.target_pos = self.target_pos + step
            return

        dt = self.ctrl_dt
        w = self.tracker_omega
        acc = w * w * (self.desired_pos - self.target_pos) \
            - 2.0 * w * self.target_vel
        self.target_vel = self.target_vel + acc * dt

        speed = np.linalg.norm(self.target_vel)
        if speed > self.max_target_vel:
            self.target_vel *= self.max_target_vel / speed

        new_pos = self.target_pos + self.target_vel * dt
        clamped = self._clamp_workspace(new_pos.copy())
        # Kill velocity into a workspace wall so the tracker cannot lean
        # on the boundary and then snap back when the wall is left.
        hit = clamped != new_pos
        if np.any(hit):
            self.target_vel[hit] = 0.0
        self.target_pos = clamped

    # ------------------------------------------------------------------- IK

    def _solve_ik_step(self) -> None:
        model, data = self.model, self.data

        hand_pos = data.xpos[self.hand_body_id]
        hand_quat = data.xquat[self.hand_body_id]

        pos_err = self.target_pos - hand_pos

        if self.rot_weight > 0.0:
            inv_hand = np.zeros(4)
            mujoco.mju_negQuat(inv_hand, hand_quat)
            err_quat = np.zeros(4)
            mujoco.mju_mulQuat(err_quat, self.target_quat, inv_hand)
            if err_quat[0] < 0.0:  # shortest path (double-cover fix)
                err_quat = -err_quat
            rot_err = np.zeros(3)
            mujoco.mju_quat2Vel(rot_err, err_quat, 1.0)
        else:
            rot_err = np.zeros(3)

        dx = np.concatenate([self.pos_weight * pos_err,
                             self.rot_weight * rot_err])
        dx_norm = np.linalg.norm(dx)
        if dx_norm > self.max_dx_norm:
            dx *= self.max_dx_norm / dx_norm

        mujoco.mj_jacBody(model, data, self._jacp, self._jacr, self.hand_body_id)
        J = np.vstack([self._jacp[:, self.dof_indices],
                       self._jacr[:, self.dof_indices]])
        self.last_jac = J  # §5-1: exposed read-only via last_sigma_min

        lam2 = self.ik_damping ** 2
        A = J @ J.T + lam2 * np.eye(6)
        J_pinv = J.T @ np.linalg.solve(A, np.eye(6))
        dq = J_pinv @ (self.ik_gain * dx)

        # FIX3: optional null-space posture term
        if self.null_gain > 0.0:
            q_now = np.array([data.qpos[a[2]] for a in self.arm])
            dq_null = self.null_gain * (self.q_neutral - q_now)
            N = np.eye(self.n_arm) - J_pinv @ J
            dq = dq + N @ dq_null

        dq = np.clip(dq, -self.max_joint_step, self.max_joint_step)
        self.q_des = np.clip(self.q_des + dq, self.q_min, self.q_max)

        # FIX2: anti-windup leash around the measured joint position
        q_now = np.array([data.qpos[a[2]] for a in self.arm])
        self.q_des = np.clip(self.q_des,
                             q_now - self.q_des_leash,
                             q_now + self.q_des_leash)

        for k, (act_idx, _, _, _) in enumerate(self.arm):
            data.ctrl[act_idx] = self.q_des[k]

        # ---- diagnostic log (comment out when stable) ----
        self._log_counter += 1
        if self._log_counter % 25 == 0:
            print(
                f"t={data.time:7.3f} "
                f"|pos_err|={np.linalg.norm(pos_err):.4f} "
                f"|dq|={np.linalg.norm(dq):.4f} "
                f"tgt=({self.target_pos[0]:.3f},{self.target_pos[1]:.3f},{self.target_pos[2]:.3f}) "
                f"hand=({hand_pos[0]:.3f},{hand_pos[1]:.3f},{hand_pos[2]:.3f})"
            )

    # ---------------------------------------------------------------- update

    def update(self, device: DeviceInput) -> DeviceState:
        state: DeviceState = device.read()
        # getattr, not state.grip_analog: some tests/older code build
        # DeviceState-like objects without this field (§6-2).
        self._update_gripper(state.button_grip,
                              getattr(state, "grip_analog", None))
        self._update_clutch(state.button_clutch, state.pos, state.quat)
        self._solve_ik_step()
        return state