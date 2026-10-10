"""落下の誘発の実機寄りの版 P2S（引き抜き。P2S-v1）。運んでいる最中に、外から立方体を指の間から引き抜く。

    ind = PullOutInducer(seed, layout, target, world)      # 発動の時刻の決め方は段階 3 の P2 と同じ（同じ u）
    a = ind.filter(k, a, truth)                            # 行動は上書きしない（ハンドは方策の指令のまま閉じ続ける）
    ind.pre_physics_step(world)                            # 物理の 1 手ごと（harness/loop.py が呼ぶ）。引き抜く
    ind.after(k, truth)                                    # 成立の判定（P2 と同じ: 箱の外の机の上に正常に着地して静止）

なぜこの形か（10/11 の決定）:
- 段階 3 の P2 は、指を開いて落とす（行動の 7 番目を開に上書き）。実機で落とすと、grasp の指令は閉じ続けるので指は
  空を掴んで開き幅が 0 へ向かう。P2 はこの点が実機と逆。
- 実機の立方体は 4 cm の PLA（約 50 g）。Franka Hand の grasp は数十 N で握るので、運ぶだけでは滑らない（約 1 N で足りる）。
  摩擦を下げて滑らせるのは現実的でない。実機で試すなら、人が手で立方体を引き抜く・はたき落とす。それを真似る。
- 引き抜き（人の手でしっかりつまんで引き抜くのを真似る）: 発動から、物理の 1 手ごとに、指に対する立方体の位置と速さの
  「引く向きの成分」を、pull_speed で離れていく道に合わせる（立方体の位置と速度を直接書く。ほかの向きの成分・回転・重力・
  指との接触は物理のまま）。指の摩擦はこれに
  逆らう力として腕と指にかかる（人が引くのと同じ向き）。立方体の中心が指先の中心から引く向きに exit_m 離れたら手を離す
  （max_s で打ち切り）。向きは指の溝の向き（hand の x 軸を水平にしたもの。指の閉じる向きと直交）の ±に、下向きを
  down_deg で混ぜたもの。
  力で引く形（xfrc_applied、ばね・ダンパでも）と、速さだけを書く形は試して捨てた。速さだけでは、指の摩擦が 1 手で
  3 m/s ぶんの速さを打ち消して動かない。力で引くと: 静止摩擦（40 N/本 × 2 × μ 1 = 約 80 N）が切れた瞬間に、
  硬い当て板の接触の押し込みが戻って立方体が 6〜10 m/s で飛んだ（10/11 の試し。2 ms の陽な力では抑えられない）。- 引き抜きの向き・下向きの角・速さは、試行の種から引く（seed_sequence(seed, "induce", 2)。段階 3 の P2 の乱数列（"induce"）は
  触らないので、同じ種なら u・r0 の決め方は P2 と同じ）。
- 成立は P2 と同じ（stage3 の Inducer.after。着地の検査）。ただし P2 の「指が開いている」の代わりに「立方体が手の中にない
  （指先の中心から drop_dist より遠い）」を使う（指は閉じたままなので）。
- 記録: meta["induce"] の kind は "P2"（集計は P2 として読む）、variant="pull_out"・version="P2S-v1"、info["pull"] に
  引いた向き・抜けた時刻・理由・抜けたときの指に対する速さ・位置を書き換えた量の最大。
ほかの実機寄りの落ちる原因（今は作らない）: ヨーのずれによる角・縁の掴み（立方体のヨーは ±30°、指のヨーは固定）、浅い掴み、
箱の壁や他の立方体への衝突。
"""
import dataclasses

import numpy as np

from recovla.common import seeds
from recovla.eval.induce import Inducer
from recovla.sim import frames, scene

PULL = {"speed_m_s": (0.2, 0.5),   # 引き抜く速さ（指に対する立方体の速さ）の範囲 [m/s]
        "down_deg": (0.0, 60.0),   # 下向きを混ぜる角の範囲 [°]（0 = 水平に溝の向き、90 = 真下）
        "exit_m": 0.045,           # 指先の中心から引く向きにこれだけ離れたら手を離す [m]（立方体の半分 2 cm＋当て板）
        "max_s": 0.4}              # 引く時間の上限 [s]


class PullOutInducer(Inducer):
    VARIANT = "pull_out"
    VERSION = "P2S-v1"

    def __init__(self, seed: int, layout, target: str, rig, cfg: dict = None, pull: dict = None):
        super().__init__("P2", seed, layout, target, rig, cfg)           # u は段階 3 の P2 と同じ列から
        self.pc = dict(PULL, **(pull or {}))
        rng = seeds.generator(seeds.seed_sequence(seed, "induce", 2))
        side = 1.0 if rng.random() < 0.5 else -1.0
        down = float(rng.uniform(*self.pc["down_deg"]))
        speed = float(rng.uniform(*self.pc["speed_m_s"]))
        self.params["pull"] = {"side": side, "down_deg": down, "speed_m_s": speed}
        self.pull = {"t_start": None, "t_end": None, "end_reason": None, "dir": None, "exit_speed_m_s": None,
                     "max_dp_m": None}
        self._on = False
        self._tip_prev = None
        self._p0 = None

    # ------------------------------------------------------------------ 行動（上書きしない）
    def _p2(self, k, a, tr):
        g = float(a[6])
        a = super()._p2(k, a, tr)              # 発動の判定は段階 3 の P2 と同じ（落とした後に開に上書きする部分は戻す）
        if self.fired:
            a[6] = g
            self.active = False
            if self.pull["t_start"] is None:
                self._start(tr)
        return a

    def _start(self, tr) -> None:
        d = self.rig.data
        R = d.xmat[self.rig.hand_id].reshape(3, 3)
        x = np.array([R[0, 0], R[1, 0], 0.0])
        x /= max(np.linalg.norm(x), 1e-9)
        p = self.params["pull"]
        th = np.radians(p["down_deg"])
        u = np.cos(th) * p["side"] * x + np.sin(th) * np.array([0.0, 0.0, -1.0])
        self._dir = u / np.linalg.norm(u)
        self.pull.update(t_start=round(float(tr.t), 3), dir=[round(float(v), 4) for v in self._dir])
        self._on, self._tip_prev, self._p0, self._t0 = True, None, None, float(d.time)

    # ------------------------------------------------------------------ 物理の 1 手ごと
    def pre_physics_step(self, world) -> None:
        if not self._on:
            return
        d, m = world.data, world.model
        ti = self.rig_target
        b, va = world.cube_ids[ti], int(world.cube_vadr[ti])
        tip = frames.fingertip_center(d, world.hand_id)
        dt = float(m.opt.timestep)
        v_tip = np.zeros(3) if self._tip_prev is None else (tip - self._tip_prev) / dt
        self._tip_prev = tip.copy()
        p = float(np.dot(d.xpos[b] - tip, self._dir))
        el = float(d.time) - self._t0
        if p > self.pc["exit_m"] or el > self.pc["max_s"]:
            self._on = False
            v_rel = float(np.dot(d.qvel[va:va + 3] - v_tip, self._dir))
            self.pull.update(t_end=round(float(d.time), 4), end_reason="exit" if p > self.pc["exit_m"] else "max_s",
                             exit_speed_m_s=round(v_rel, 3))
            return
        vt = float(self.params["pull"]["speed_m_s"])
        if self._p0 is None:
            self._p0 = p
        qa = scene.cube_qpos_adr(m, self.target)[0]
        dp = (self._p0 + vt * el) - p               # 人の手: 指に対する立方体の位置を、引く向きに vt で動かす（他の向きは物理のまま）
        d.qpos[qa:qa + 3] = d.qpos[qa:qa + 3] + dp * self._dir
        v = d.qvel[va:va + 3]
        d.qvel[va:va + 3] = v + ((float(np.dot(v_tip, self._dir)) + vt) - float(np.dot(v, self._dir))) * self._dir
        self.pull["max_dp_m"] = round(max(float(self.pull.get("max_dp_m") or 0.0), abs(dp)), 5)
    @property
    def rig_target(self) -> int:
        from recovla.common.seeds import COLORS
        return COLORS.index(self.target)

    # ------------------------------------------------------------------ 成立（P2 と同じ。指の開きの代わりに「手の中にない」）
    def after(self, k: int, tr) -> None:
        if self.fired and self.stage == "dropped":
            in_hand = float(np.linalg.norm(tr.target_pos - tr.fingertip)) <= self.pp.drop_dist
            tr = dataclasses.replace(tr, gripper_closed=bool(in_hand))
        super().after(k, tr)

    @property
    def releasing(self) -> bool:
        return False                           # 指を開いたまま保つことはしない

    def record(self) -> dict:
        if self._on:                           # 試行が引いている途中で終わった
            self._on = False
            self.pull.update(t_end=round(float(self.rig.data.time), 4), end_reason="trial_end")
        r = super().record()
        r["info"] = dict(r["info"], pull=dict(self.pull))
        r["variant"], r["version"] = self.VARIANT, self.VERSION
        return r
