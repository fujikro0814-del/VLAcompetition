"""方策の実行（目標書 v2 の G2: シミュレーションの時刻の上で世界は一度も止まらない。0107 の 7）。

    rt = PolicyRuntime(io, setup, policy, mode="naive", s=10)
    rt.start(task, seed)            # 試行・サブタスクの始め（腕は保持から始まる）
    rt.tick()                       # 物理の 1 手（500 Hz）ごとに 1 回。腕の指令（1 kHz 相当で 2 つ）とハンドの指令を io に出す
    rt.reset_chunks()               # サブタスクの切り替え・やり直し（塊を持ち越さない）

- 推論: 行動の区切り（0.1 s）に観測を組み、io.compute で計算する。結果は「始めた時刻＋計算の時間」から使える
  （計算の時間は評価の枠が実測の分布から試行のシードで引く。前処理を含む）。使える前に使うと Pending が止める
- naive・rtc: 前の推論を始めてから s 行ごとに次の推論を始め、その間は前の塊を実行し続ける。新しい塊は、使えるように
  なった後の最初の区切りから、観測から数えた行の位置で使う。塊がない間（最初・切り替えの直後）は腕をその場で保持する
- sync（止まって推論する）: 推論の間は腕をその場で保持し（世界は進む）、使えるようになった区切りから s 行を実行して、また保持して推論する
- rtc: 方策に渡す inference_delay は実行系が見積もった値（直前の推論で実際にかかった行の数。最初は configs の d）。
  実際に使う行の位置は、実際に届いた時刻で決まる（未来の計算時間は実行系には分からないため）
- グリッパ: 行動の 7 番目が正なら閉（grasp）、そうでなければ開（move）。変わったときだけ口に出す
"""
import math

import numpy as np

from recovla.policy.schedule import normalize_left_over, noise_generator
from recovla.runtime.motion import Motion

ACTION_DT = 0.1
STEPS_PER_ACTION = 50


class PolicyRuntime:
    def __init__(self, io, setup, policy, mode: str = "naive", s: int = 10, d_init: int = 4, rtc_horizon: int = 40,
                 motion: Motion = None, limiter_enabled: bool = True, margin: float = 0.99):
        if mode not in ("sync", "naive", "rtc"):
            raise ValueError(mode)
        self.io, self.setup, self.policy = io, setup, policy
        self.mode, self.s, self.d_init, self.E = mode, int(s), int(d_init), int(rtc_horizon)
        self.motion = motion or Motion(setup, limiter_enabled=limiter_enabled, margin=margin)
        self.action_filter = None          # 評価の道具（失敗注入）が行動を上書きする口。None なら方策のまま
        self.safety = None                 # 安全フィルタ（知覚の障害物）。motion の差し込み口に付ける

    # -------------------------------------------------------------------- trial
    def start(self, task: str, seed: int) -> None:
        self.task, self.seed = task, int(seed)
        sensor = self.io.sense(cameras=False)
        self.motion.reset(sensor.joints)
        self.policy.start_trial(seed)
        self.n_tick = 0
        self.k = -1
        self.closed = False
        self.i = 0                          # 推論の通し番号（雑音の列）
        self.d_est = self.d_init
        self.log_inf, self.log_act = [], []
        self.reset_chunks()

    def reset_chunks(self) -> None:
        self.active = None                  # {"i", "post", "raw", "k0", "o0", "rows_done"}
        self.pending = None                 # {"i", "k_obs", "t_obs", "fut", "rtc"}
        self.next_infer_k = None
        self.policy.reset_cue()
        self.motion.set_velocity(np.zeros(3))

    # --------------------------------------------------------------------- tick
    def tick(self) -> None:
        if self.n_tick % STEPS_PER_ACTION == 0:
            self.k += 1
            self._action_boundary()
        self.n_tick += 1
        sensor = self.io.sense(cameras=False)
        for q in self.motion.step(sensor.joints):
            self.io.command_joints(q)

    def _action_boundary(self) -> None:
        k, t = self.k, self.io.now()
        # 1) 届いた推論を塊にする
        if self.pending is not None and self.pending["fut"].ready(t):
            p = self.pending
            post = p["fut"].result(t)
            lag = k - p["k_obs"]
            o0 = 0 if self.mode == "sync" else lag
            self.active = {"i": p["i"], "post": post, "raw": p["raw"](), "k0": k, "o0": o0, "rows_done": 0}
            self.d_est = max(1, lag)
            self.log_inf[-1].update({"k_act": k, "t_act": t, "offset": o0})
            self.pending = None
        # 2) 推論を始めるか
        start = False
        if self.pending is None:
            if self.mode == "sync":
                start = self.active is None or self.active["rows_done"] >= self.s
                if start:
                    self.active = None
            else:
                start = self.active is None or (self.next_infer_k is not None and k >= self.next_infer_k)
        if start:
            self._start_inference(k, t)
        # 3) この区切りの行動
        a = None
        if self.active is not None:
            row = self.active["o0"] + (k - self.active["k0"])
            if row < len(self.active["post"]) and not (self.mode == "sync" and self.active["rows_done"] >= self.s):
                a = np.array(self.active["post"][row], float)
                self.active["rows_done"] += 1
        held = a is None
        if held:
            a = np.zeros(7)
            a[6] = 1.0 if self.closed else -1.0
        if self.action_filter is not None:
            a = np.asarray(self.action_filter(k, a), float)
        self._apply(a)
        self.log_act.append((k, t, None if held else self.active["i"], held, a.copy()))

    def _start_inference(self, k: int, t: float) -> None:
        sensor = self.io.sense(cameras=True)
        i = self.i
        self.i += 1
        rtc = None
        if self.mode == "rtc" and self.active is not None:
            j = self.active["o0"] + (k - self.active["k0"])
            left = self.active["raw"][min(j, len(self.active["raw"])):]
            rtc = {"inference_delay": int(min(self.d_est, self.s - 1)),
                   "prev_chunk_left_over": normalize_left_over(left, self.E)}
        pol = self.policy
        holder = {}

        def work():
            obs = pol.observe(sensor, self.task)
            out = pol.infer(obs, noise_generator(self.seed, i), rtc)
            holder["raw"] = pol.last_raw.copy()
            holder["cue"] = None if pol.last_cue is None else np.asarray(pol.last_cue).copy()
            return out

        fut = self.io.compute("policy_rtc" if self.mode == "rtc" else "policy", work)
        self.pending = {"i": i, "k_obs": k, "t_obs": t, "fut": fut, "raw": lambda: holder["raw"]}
        self.next_infer_k = k + self.s
        self.log_inf.append({"i": i, "k_obs": k, "t_obs": t, "t_ready": fut.t_ready, "latency_s": fut.latency,
                             "wall_s": fut.wall_s, "rtc_delay": None if rtc is None else rtc["inference_delay"],
                             "img_t": {n: c.t_capture for n, c in sensor.cameras.items()},
                             "cue": holder.get("cue")})

    def _apply(self, a) -> None:
        self.motion.set_velocity(a[:3] / ACTION_DT)
        want = bool(a[6] > 0.0)
        if want != self.closed:
            self.closed = want
            su = self.setup
            if want:
                eps = 0.005
                self.io.gripper_grasp(su.cube_size, su.gripper_speed, su.grasp_force, eps, eps)
            else:
                self.io.gripper_move(0.08, su.gripper_speed)

    def trace(self) -> dict:
        return {"inference": self.log_inf, "actions": [
            {"k": k, "t": t, "chunk": c, "held": h, "a": a.tolist()} for k, t, c, h, a in self.log_act]}
