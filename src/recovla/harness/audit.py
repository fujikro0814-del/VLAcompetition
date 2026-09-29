"""監査（目標書 v2 の G1〜G3。0107 の 2-3・7-3・8-2）。評価の枠の側で、試行ごとに記録して集計する。

CommandAudit（G3）: RobotIO の口を通った指令（関節の位置 1 kHz、ハンドの move・grasp）を、Franka Panda の公称の上限と比べる。
  関節の速度・加速度・躍度は 1 kHz の差分（libfranka と同じ）、直交座標は指令の順運動学の手先。上限は目標書の値そのもの
  （制限層の margin は掛けない）。位置のサーボの力とその変化（500 Hz）は参考（reference）として別に書く
"""
import numpy as np

from recovla.runtime import limiter as L

TORQUE = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
TORQUE_RATE = 1000.0
CART_VEL, CART_ACC, CART_JERK = 1.7, 13.0, 6500.0
TOL = 1e-6                                  # 数値の丸めの許容（比で）


class CommandAudit:
    def __init__(self, fk=None):
        self.fk = fk                                           # q -> 手先の位置 (3,)（直交座標の監査。None なら省く）
        self.reset()

    def reset(self) -> None:
        self.q = []                                            # 1 kHz の指令
        self.tau = []                                          # 500 Hz の作動器の力（重力の補償を含む）
        self.hand = []                                         # (時刻, 口, 引数)
        self.finger_speed_max = 0.0                            # 参考: 測った指の速さの最大（指令ではない）

    def joint_command(self, q) -> None:
        self.q.append(np.array(q, float))

    def torque(self, tau) -> None:
        self.tau.append(np.array(tau, float))

    def hand_commands(self, log) -> None:
        self.hand = list(log)

    def finger_speed(self, v) -> None:
        self.finger_speed_max = max(self.finger_speed_max, float(v))

    def summary(self) -> dict:
        out = {"n_joint_commands": len(self.q), "violations": {}, "max_ratio": {}}

        def put(name, ratio):
            r = np.asarray(ratio, float)
            out["max_ratio"][name] = float(r.max()) if r.size else 0.0
            out["violations"][name] = int(np.sum(r.reshape(len(r), -1).max(axis=1) > 1.0 + TOL)) if r.size else 0

        if len(self.q) >= 4:
            q = np.array(self.q)
            v = np.diff(q, axis=0) / L.DT
            a = np.diff(v, axis=0) / L.DT
            j = np.diff(a, axis=0) / L.DT
            put("joint_vel", np.abs(v) / L.MAX_VEL)
            put("joint_acc", np.abs(a) / L.MAX_ACC)
            put("joint_jerk", np.abs(j) / L.MAX_JERK)
            if self.fk is not None:
                x = np.array([self.fk(qq) for qq in q])
                cv = np.diff(x, axis=0) / L.DT
                ca = np.diff(cv, axis=0) / L.DT
                cj = np.diff(ca, axis=0) / L.DT
                put("cart_vel", np.linalg.norm(cv, axis=1) / CART_VEL)
                put("cart_acc", np.linalg.norm(ca, axis=1) / CART_ACC)
                put("cart_jerk", np.linalg.norm(cj, axis=1) / CART_JERK)
        if len(self.tau) >= 2:
            tau = np.array(self.tau)
            put("torque", np.abs(tau) / TORQUE)
            put("torque_rate", np.abs(np.diff(tau, axis=0)) / 0.002 / TORQUE_RATE)
        from recovla.harness.hand import MAX_FINGER_FORCE, MAX_WIDTH_SPEED
        speeds = [args[1] for _, kind, args in self.hand]
        forces = [args[2] for _, kind, args in self.hand if kind == "grasp"]
        put("gripper_speed", np.array(speeds) / MAX_WIDTH_SPEED if speeds else np.zeros(0))
        put("gripper_force", np.array(forces) / MAX_FINGER_FORCE if forces else np.zeros(0))
        out["n_hand_commands"] = len(self.hand)
        out["finger_speed_measured_max"] = self.finger_speed_max
        # 口を通る指令の項目だけを数える（0108 の 2 の 2）。位置のサーボの力（torque・torque_rate）は実機の内部の制御器の出力で、
        # 関節の位置の指令の口では指令されないので参考として別に書く（力の大きさは作動器の上限で切られる）
        ref = ("torque", "torque_rate")
        out["reference"] = {k: {"max_ratio": out["max_ratio"].pop(k, 0.0), "violations": out["violations"].pop(k, 0)} for k in ref}
        out["total_violations"] = int(sum(out["violations"].values()))
        return out
