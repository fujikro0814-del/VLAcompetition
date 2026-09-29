"""方策の実行（目標書 v2 の G2: シミュレーションの時刻の上で世界は一度も止まらない。0107 の 7）。

    rt = PolicyRuntime(io, setup, policy, mode="naive", s=10)
    rt.start(task, seed)            # 試行・サブタスクの始め（腕は保持から始まる）
    rt.tick()                       # 物理の 1 手（500 Hz）ごとに 1 回。腕の指令（1 kHz 相当で 2 つ）とハンドの指令を io に出す
    rt.reset_chunks()               # サブタスクの切り替え・やり直し（塊を持ち越さない）

- 推論: 行動の区切り（0.1 s）に観測を組み、io.compute で計算する。結果は「始めた時刻＋計算の時間」から使える
  （計算の時間は評価の枠が実測の分布から試行のシードで引く。前処理を含む）。使える前に使うと Pending が止める
- naive・rtc: 前の推論を始めてから s 行ごとに次の推論を始め、その間は前の塊を実行し続ける。新しい塊は、使えるように
  なった後の最初の区切りから、観測から数えた行の位置で使う。塊がない間（最初・切り替えの直後）は腕をその場で保持し、
  そのとき届いた塊は 0 行目から使う（保持していた間、腕は観測の時点から動いていないため）
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


def enable_rtc_for(policy, horizon: int, schedule: str, max_guidance_weight: float) -> dict:
    """RTC を入れる（旧版の policy.runner.enable_rtc と同じ設定）。"""
    from recovla.policy.runner import enable_rtc
    return enable_rtc(policy.policy, horizon, schedule, max_guidance_weight)


def disable_rtc_for(policy) -> None:
    from recovla.policy.runner import disable_rtc
    disable_rtc(policy.policy)


class PolicyRuntime:
    def __init__(self, io, setup, policy, mode: str = "naive", s: int = 10, d_init: int = 4, rtc_horizon: int = 40,
                 motion: Motion = None, limiter_enabled: bool = True, margin: float = 0.99,
                 perception=None, safety=None, checks: dict = None, tip_offset: float = 0.1034):
        if mode not in ("sync", "naive", "rtc"):
            raise ValueError(mode)
        self.io, self.setup, self.policy = io, setup, policy
        self.mode, self.s, self.d_init, self.E = mode, int(s), int(d_init), int(rtc_horizon)
        self.motion = motion or Motion(setup, limiter_enabled=limiter_enabled, margin=margin)
        self.action_filter = None          # 評価の道具（失敗注入）が行動を上書きする口。None なら方策のまま
        self.injecting = False             # 評価の道具が上書きしている間は真（安全フィルタを切る。旧版と同じ）
        self.external = False              # 上位層が速さを直接出している間は真（方策を止め、安全フィルタを切る）
        self.perception = perception       # recovla.runtime.perception.Perception（None なら知覚なし）
        self.safety = safety               # recovla.runtime.safety.PerceptionSafetyFilter（None ならフィルタなし）
        self.checks = dict(checks or {})   # 知覚の失敗の止まり方の閾値（configs の runtime_v2.checks）
        self.tip_offset = float(tip_offset)

    # -------------------------------------------------------------------- trial
    def start(self, task: str, seed: int) -> None:
        self.task, self.seed = task, int(seed)
        sensor = self.io.sense(cameras=False)
        self.motion.reset(sensor.joints)
        self.policy.start_trial(seed)
        from recovla.runtime.cue import color_of_instruction
        self.color = color_of_instruction(task)
        self.ready = self.perception is None          # 知覚があるなら、起動時の確かめ（テーブル面・箱）の後に始める
        self.boxf = []
        self.pend_per = None
        self.wm = None
        self.stop_reason = None
        self.t_task0 = None
        self.log_per = []
        if self.safety is not None:
            self.safety.start_trial(self.color)
        self.n_tick = 0
        self.k = -1
        self.closed = False
        self.i = 0                          # 推論の通し番号（雑音の列）
        self.d_est = self.d_init
        self.log_inf, self.log_act = [], []
        self.reset_chunks()

    def set_task(self, task: str, seed: int) -> None:
        """サブタスクの切り替え・やり直し（知覚の起動時の確かめと推定は保つ）。塊は持ち越さない。雑音の列は seed から。"""
        from recovla.runtime.cue import color_of_instruction
        self.task, self.seed = task, int(seed)
        self.color = color_of_instruction(task)
        self.policy.start_trial(seed)
        self.i = 0
        self.t_task0 = None
        self.stop_reason = None if self.stop_reason == "target_not_found" else self.stop_reason
        if self.safety is not None:
            wm = self.wm
            self.safety.start_trial(self.color)
            if wm is not None:
                self.safety.set_world(wm)
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
        sensor = self.io.sense(cameras=False)
        if self.perception is not None:
            self.perception.record_joints(sensor.joints)
        if self.safety is not None and self.n_tick % 10 == 0:          # 入力の読み取り（50 Hz）ごと。旧版と同じ
            self.safety.begin_read(sensor.joints.q, sensor.gripper.width, self.motion.x_cmd)
            self.motion.set_command_filter(self.safety.filter if self.safety.on() else None)
        self.n_tick += 1
        for q in self.motion.step(sensor.joints):
            self.io.command_joints(q)

    # ------------------------------------------------------------------ perception
    def _tip(self, q) -> np.ndarray:
        pos, quat = self.motion.hand_pose(q)
        w, x, y, z = quat
        zaxis = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
        return pos + self.tip_offset * zaxis

    def _perception(self, t: float) -> None:
        per = self.perception
        if self.pend_per is not None and self.pend_per.ready(t):
            out = self.pend_per.result(t)
            self.pend_per = None
            if out["kind"] == "startup":
                self._after_startup(out)
            else:
                self.wm = out["wm"]
                if self.safety is not None:
                    self.safety.set_world(self.wm)
                self.log_per.append({"t": t, "t_obs": out["t_obs"],
                                     "cubes": {c: {"pos": e.pos.tolist(), "status": e.status, "in_box": e.in_box,
                                                   "source": e.source} for c, e in self.wm.cubes.items()},
                                     "box": {"xy": np.asarray(self.wm.box["xy"]).tolist(), "yaw": self.wm.box["yaw"],
                                             "ok": self.wm.box["ok"]}})
                self._check_target(t)
        if self.pend_per is not None or self.stop_reason:
            return
        sensor = self.io.sense(cameras=True)
        if not {"overhead", "wrist"} <= set(sensor.cameras):
            return
        if not self.ready:
            self.boxf.append((sensor.cameras["overhead"], sensor.gripper.width))
            if len(self.boxf) < int(self.checks.get("box_frames", 5)):
                return
            frames, self.boxf = self.boxf, []

            def startup():
                chk = per.check_table(frames[0][0], frames[0][1])
                box = per.init_box(frames)
                return {"kind": "startup", "table": chk, "box": box}
            self.pend_per = self.io.compute("perception", startup)
            return
        tip = self._tip(sensor.joints.q)
        t_obs = t

        def work():
            return {"kind": "update", "wm": per.update(sensor.cameras, sensor.gripper, sensor.t, tip), "t_obs": t_obs}
        self.pend_per = self.io.compute("perception", work)

    def _after_startup(self, out) -> None:
        c = self.checks
        tab, box = out["table"], out["box"]
        self.startup = {"table": tab, "box": {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v)
                                               for k, v in box.items()}}
        if abs(tab["dz_m"]) > float(c.get("table_dz_m", 1e9)) or tab["angle_deg"] > float(c.get("table_angle_deg", 1e9)):
            self.stop_reason = "table_mismatch"
        elif not box.get("ok"):
            self.stop_reason = "box_not_found"
        elif box.get("dev_m", 0.0) > float(c.get("box_dev_m", 1e9)) or box.get("dev_yaw_deg", 0.0) > float(c.get("box_dev_deg", 1e9)):
            self.stop_reason = "box_moved"
        self.ready = self.stop_reason is None
        if tab.get("corrected") and hasattr(self.policy, "set_cue_pose"):
            R, t = self.perception.camera_pose("overhead", self.io.now(), 0.08)
            self.policy.set_cue_pose(R, t)

    def _check_target(self, t: float) -> None:
        if self.t_task0 is None:
            self.t_task0 = t
        e = self.wm.cubes.get(self.color)
        missing = e is None or e.status == "lost"
        if missing and t - self.t_task0 > float(self.checks.get("target_missing_s", 1e9)) and self.active is None:
            self.stop_reason = "target_not_found"

    def _action_boundary(self) -> None:
        k, t = self.k, self.io.now()
        if self.perception is not None:
            self._perception(t)
            if not self.ready or self.stop_reason:           # 起動時の確かめの間・知覚の失敗: 腕をその場で保持する
                self.motion.set_velocity(np.zeros(3))
                self.log_act.append((k, t, None, True, np.array([0, 0, 0, 0, 0, 0, 1.0 if self.closed else -1.0])))
                return
        if self.external:                                     # 上位層が腕を動かしている間（待機位置へ戻す動き）: 方策は止める
            if self.safety is not None:
                self.safety.gate = False
            return
        # 1) 届いた推論を塊にする
        if self.pending is not None and self.pending["fut"].ready(t):
            p = self.pending
            post = p["fut"].result(t)
            lag = k - p["k_obs"]
            # 観測から今までに腕が前の塊を実行していたなら、その分だけ先の行から使う。保持していた（塊がなかった）なら、
            # 腕は観測の時点から動いていないので 0 行目から使う（最初の推論・切り替えの直後・同期）
            o0 = lag if (self.mode != "sync" and p["was_moving"]) else 0
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
        if self.safety is not None:                          # 方策が指令を出している間だけ（保持・評価の道具の上書き中は切る）
            self.safety.gate = (not held) and not self.injecting
        self._apply(a)
        self.log_act.append((k, t, None if held else self.active["i"], held, a.copy()))

    def _start_inference(self, k: int, t: float) -> None:
        sensor = self.io.sense(cameras=True)
        if not {"overhead", "wrist"} <= set(sensor.cameras):
            return                          # まだこまが届いていない（試行の始め）。保持して次の区切りでやり直す
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
        self.pending = {"i": i, "k_obs": k, "t_obs": t, "fut": fut, "raw": lambda: holder["raw"],
                        "was_moving": self.active is not None}
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
            {"k": k, "t": t, "chunk": c, "held": h, "a": a.tolist()} for k, t, c, h, a in self.log_act],
            "perception": self.log_per, "startup": getattr(self, "startup", None), "stop_reason": self.stop_reason,
            "safety": None if self.safety is None else self.safety.summary()}
