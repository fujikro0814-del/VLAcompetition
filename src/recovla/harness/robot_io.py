"""実行系に渡す RobotIO の実装（評価の枠の側）。世界・センサの模型・計算の時間の模型をまとめる（0107 の 2-1・7）。

    io = SimRobotIO(world, suite, seed)
    io.sense(cameras)                 # SensorFrame（センサの模型を通したもの）
    io.command_joints(q)              # 1 kHz 相当の関節の位置の指令（物理の 1 手で 2 つ）
    io.gripper_move / gripper_grasp   # ハンドの口（上限を超える指令はハンドが拒む）
    fut = io.compute(kind, fn)        # 計算（推論・知覚）: 壁の時計で fn を今すぐ実行し、シミュレーションの時刻の上では
                                      # 分布から引いた時間の後に使えるようになる（Pending）。世界はその間も進む
    io.now()                          # シミュレーションの時刻（実行系の時計）

計算の時間の分布は configs/latency_v1.json（Step J の実測、前処理を含む）。知覚の遅れは固定の値（configs）。
"""
import json
import time

import numpy as np

from recovla.common import config, seeds

LATENCY = json.loads((config.ROOT / "configs" / "latency_v1.json").read_text(encoding="utf-8"))


class EarlyUse(RuntimeError):
    """使えるようになる前に結果を使った（G2 の違反）。"""


class Pending:
    def __init__(self, value, t_start: float, latency: float, wall_s: float, log: list):
        self._value = value
        self.t_start = t_start
        self.latency = float(latency)
        self.t_ready = t_start + self.latency
        self.wall_s = wall_s
        self._log = log

    def ready(self, t: float) -> bool:
        return t >= self.t_ready - 1e-12

    def result(self, t: float):
        if not self.ready(t):
            self._log.append({"early_use": True, "t": t, "t_ready": self.t_ready})
            raise EarlyUse(f"t={t:.4f} < t_ready={self.t_ready:.4f}")
        self._log.append({"use": True, "t": t, "t_ready": self.t_ready})
        return self._value


class SimRobotIO:
    def __init__(self, world, suite, seed: int, fixed_latency: dict = None):
        self.world, self.suite = world, suite
        self.rng = np.random.default_rng(seeds.seed_sequence(seed, "latency", 1))    # 計算の時間（カメラの遅延は 0）
        self.fixed = dict(fixed_latency or {})             # {kind: 秒}（知覚など、固定の遅れ）
        self.cmd_buffer = []
        self.compute_log = []
        self.use_log = []

    def now(self) -> float:
        return float(self.world.data.time)

    def sense(self, cameras: bool = True):
        return self.suite.sense(self.world.data, self.world.hand, self.world._last_cmd, cameras)

    def command_joints(self, q) -> None:
        self.cmd_buffer.append(np.asarray(q, float).copy())

    def take_commands(self) -> list:
        out, self.cmd_buffer = self.cmd_buffer, []
        return out

    def gripper_move(self, width: float, speed: float) -> None:
        self.world.hand.move(width, speed)

    def gripper_grasp(self, width, speed, force, eps_inner, eps_outer) -> None:
        self.world.hand.grasp(width, speed, force, eps_inner, eps_outer)

    def draw_latency(self, kind: str) -> float:
        if kind in self.fixed:
            return float(self.fixed[kind])
        q = LATENCY["kinds"][kind]["quantiles_s"]
        u = float(self.rng.random())
        return float(np.interp(u, np.linspace(0, 1, len(q)), q))

    def compute(self, kind: str, fn) -> Pending:
        t = self.now()
        w0 = time.perf_counter()
        value = fn()
        wall = time.perf_counter() - w0
        lat = self.draw_latency(kind)
        self.compute_log.append({"kind": kind, "t_start": t, "latency_s": lat, "t_ready": t + lat, "wall_s": wall})
        return Pending(value, t, lat, wall, self.use_log)

    def audit(self) -> dict:
        early = [e for e in self.use_log if e.get("early_use")]
        return {"n_compute": len(self.compute_log), "n_use": sum(1 for e in self.use_log if e.get("use")),
                "early_use": len(early),
                "latency_by_kind": {k: [c["latency_s"] for c in self.compute_log if c["kind"] == k]
                                    for k in {c["kind"] for c in self.compute_log}}}
