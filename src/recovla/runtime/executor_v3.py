"""実行器 v3（段階 4 束 2 の B1）。凍結した runtime/executor.py の TaskRuntime を書き換えずに、子として (a)〜(f) を足す。

    ex = make_v3(io, setup, prt, judge, planner_cfg, motion_params, decompose, fmt, v3={"a_goal": "EH"})
    # または 96_s4_resume.py の写しの make が作った TaskRuntime に install(ex, v3) で差し替える（diag/e7.py の install と同じやり方）

(a) 戻し先（関門 T で合格した EH・ES。「なし」なら働かない）へ関節空間で戻す: 2 番目以降の手順の始め（前の手順の完了の判定の後）と、
    やり直しの前（今の実行器の「待機位置へ戻す動き」の代わり）。1 番目の手順の前は a_before_first_step（既定 False。試行は
    home から始まり、D-E7 の EH・ES も 1 番目の前には戻さない）。戻し先の関節角と動かし方は D-E7 の腕と同じ値
    （src/recovla/diag/e7.py の HOME_Q・STANDBY_START_Q・GOAL_PARAMS。runtime からは diag を import できない＝G1 の検査
    なので値を写し、tests/test_s4_bundle2.py が一致と動きの一致を確かめる）。動きは e7 の GoalMotion と DiagTaskRuntime の
    _goal_tick と同じ（指を開く → 低ければ真上に rise_z まで → 関節空間の直線 → 測った関節角で 1 回だけ目標をずらす → 保つ）。
(b) 置いた後の強制の戻し（configs の planner の待機位置へ戻す動きの設定）の発動条件: 今の実行器は「箱の中に目標の色・指が開・待機位置から離れている」が
    trigger_s（4.0 s）続くと戻すが、方策が指を閉じ直すと続きが切れて数え直しになる。v3 は、数え始めた後は指の開閉で切らず
    （箱の中に目標の色・待機位置から離れている、だけで続ける）、trigger_s を b_trigger_s（既定 2.0 s）にする。wait_s は
    b_wait_s（既定 2.0 s = configs と同じ。判定の 1 s 静止＋余裕）。configs/default.yaml は変えず、この子の中で上書きする。
(c) 最後の試みが時間切れ（または (d) の打ち切り）になった時点で、知覚（WorldModel。真値ではない）が「目標の色は箱の中」と
    見ていれば、止める前に待機位置へ戻して wait_s の間だけ完了の判定を待つ（判定が出れば完了、出なければ今の実行器と同じに止まる）。
    判定の閾値は変えない（(e)）。ただし指が開いているときだけ（自分の指令が開＝PolicyRuntime.closed が偽、かつ直近の判定で測った
    開き幅が開＝judge.last["gripper_open"]）。知覚の in_box は水平の位置だけで見るので、持ったままの立方体を箱の上で「箱の中」と
    見ることがあり、そのまま台本で戻すと落とすため（作者の既定の決定、結果を見る前に固定）。また、同じ試みの中で (b) の戻しが
    済んだ後に指を一度も閉じていなければ、(b) が同じ待ち（待機位置で wait_s）を済ませているので (c) はしない（同じ戻しを 2 回
    数えない）。しなかったときは v3.events に final_wait_skipped（why = gripper_closed・after_placed_return）を残す。
(d) 早めの打ち切り（試みの持ち時間 step_timeout_s を待たずに次へ）は、次の 2 つだけ:
    止まった（stalled）: 試みの始めから d_stall_s 以上たち、直近 d_stall_s の間の指令の手先（x_cmd）の動きが d_stall_move_m 未満
    閉じない（no_close）: 試みの始めから d_noclose_s たっても、指を一度も閉じていない（PolicyRuntime.closed）
    置いた後の戻しを数えている間（(b)）は打ち切らない。打ち切った後は時間切れと同じ道（やり直し、最後なら (c)）。
(e) 完了の判定（JudgeV2）の閾値と色の順（計画役の出力）は変えない。
(f) 介入の記録: 戻す動きは meta["returns"] に、scripts/56_intervention_s3.py が知っている種類だけで残す
    （置いた後の戻し・(a) の手順の前の戻し・(c) の待ち = "placed"＝台本の動き、やり直しの前の戻し（(a) の関節空間の戻しを含む）
    = "retry"＝出し直し）。中身の区別は各行の "v3" の欄（"placed"・"goal"・"final_wait"・"retry_goal"）。試みの終わり方は
    steps[].attempts[].end（"timeout"・"stalled"・"no_close"）。record() の "v3" に設定・出来事・介入の種類別の数を足す。
    計画の変更と判定の上書きは v3 では起きない（0）。API の呼び出しは起動時の計画役の 1 回（介入とは別）。
各部は v3 の設定で個別に切れる（a_goal None・b_placed_fix・c_final_wait・d_early_abort）。全部を切ると今の実行器と同じ動きになる。
G1: runtime の入れ物の中にあり、scripts/check_g1_boundary.py の厳しい検査を受ける。使うのは RobotIO の口・知覚の WorldModel・
  自分の指令（x_cmd・closed）・測った関節角だけ。
"""
import numpy as np

from recovla.runtime.executor import TaskRuntime
from recovla.runtime.motion import SUBSTEPS, Motion
from recovla.sim import control
from recovla.sim.device import pad_state

VERSION = "v3.1"                    # 実行器の版（記録の meta.v3.version・meta.b1.executor_version。(c) の指の条件と (b)→(c) の重複なしを入れた版）

# ---------------------------------------------------------------- 戻し先（diag/e7.py と同じ値。テストで一致を確かめる）
HOME_Q = (0.0, -0.684, 0.0, -2.907, 0.0, 2.216, 0.785)
STANDBY_START_Q = (0.162272, -0.394371, 0.27625, -2.802853, 0.159527, 2.408117, 1.081562)
GOALS = {"EH": HOME_Q, "ES": STANDBY_START_Q}
GOAL_PARAMS = {"vmax_rad_s": 0.5, "gain_per_s": 2.0, "tol_cmd_rad": 1e-3, "tol_vel_rad_s": 5e-3, "hold_s": 0.5,
               "timeout_s": 10.0, "open_wait_max_s": 1.0, "meas_corrections": 1, "meas_settle_s": 0.25}

V3_DEFAULTS = {
    "a_goal": None,                 # "EH"・"ES"・None（関門 T の結果。None なら (a) は働かない）
    "a_before_first_step": False,
    "a_before_retry": True,
    "b_placed_fix": True,
    "b_trigger_s": 2.0,
    "b_wait_s": 2.0,
    "c_final_wait": True,
    "d_early_abort": True,
    "d_stall_s": 8.0,
    "d_stall_move_m": 0.01,
    "d_noclose_s": 20.0,
}
RETURN_KINDS_56 = ("placed", "retry")   # 56_intervention_s3.py が知っている戻す動きの種類（これ以外は書かない）


def settings(v3: dict = None) -> dict:
    s = dict(V3_DEFAULTS, **(v3 or {}))
    unknown = set(s) - set(V3_DEFAULTS)
    if unknown:
        raise ValueError(f"v3 の設定に知らない項目 {sorted(unknown)}")
    if s["a_goal"] not in (None, "EH", "ES"):
        raise ValueError(f"a_goal は EH・ES・None のどれか: {s['a_goal']!r}")
    return s


# ---------------------------------------------------------------- 関節空間の戻す動き（diag/e7.py の GoalMotion と同じ）
class JointGoalMotion(Motion):
    """Motion の子（install が __class__ を差し替える。__init__ は呼ばないので、足す値は getattr で既定を持つ）。"""

    def set_joint_goal(self, q, vmax: float, gain: float) -> None:
        self._jgoal = (np.asarray(q, float).copy(), float(vmax), float(gain))
        self.set_velocity(np.zeros(3))

    def joint_goal_active(self) -> bool:
        return getattr(self, "_jgoal", None) is not None

    def joint_goal_error(self) -> tuple:
        g = self._jgoal[0]
        return float(np.max(np.abs(g - self.limiter.q))), float(np.max(np.abs(self.limiter.v)))

    def clear_joint_goal(self) -> None:
        self._jgoal = None
        c, integ = self.controller, self.integrator
        q = self.limiter.q.copy()
        self._load_state(q)
        c.q_des = q.copy()
        c.gripper_closed = False
        c.sync_target_to_hand()
        integ.reset(c.target_pos)
        c.device_ref_pos = integ.x_cmd.copy()
        c.target_ref_pos = c.target_pos.copy()
        c.device_ref_quat = control._IDENTITY_QUAT.copy()
        c.target_ref_quat = c.target_quat.copy()
        c.prev_clutch = True
        self.vel = np.zeros(3)
        self.pad.state = pad_state(vel=self.vel)
        integ.refresh()

    def step(self, joints) -> list:
        jg = getattr(self, "_jgoal", None)
        if jg is None:
            return Motion.step(self, joints)
        goal, vmax, gain = jg
        v_des = gain * (goal - self.limiter.q)
        s = float(np.max(np.abs(v_des))) / vmax
        if s > 1.0:
            v_des = v_des / s
        out = []
        for _ in range(SUBSTEPS):
            v = self.limiter.propose(v_des)
            if self.limiter.enabled:
                v = self._cartesian_limit(v)
            q = self.limiter.commit(v)
            self._x_hist.append(self._fk_hand(q))
            del self._x_hist[:-4]
            out.append(q)
        self.integrator.x_cmd = self._fk_hand(out[-1])
        self.controller.q_des = out[-1].copy()
        return out


# ---------------------------------------------------------------- 実行器 v3
class V3TaskRuntime(TaskRuntime):
    """TaskRuntime の子。v3_setup を start の前に呼ぶ（make_v3・install が呼ぶ）。"""

    def v3_setup(self, v3: dict = None) -> None:
        self.v3 = settings(v3)
        self.goal_q = None if self.v3["a_goal"] is None else np.asarray(GOALS[self.v3["a_goal"]], float)
        self.gp = dict(GOAL_PARAMS)
        if self.v3["b_placed_fix"]:
            self.ret = dict(self.ret, trigger_s=float(self.v3["b_trigger_s"]), wait_s=float(self.v3["b_wait_s"]))
        if self.goal_q is not None and not isinstance(self.prt.motion, JointGoalMotion):
            self.prt.motion.__class__ = JointGoalMotion

    def start(self, text: str, seed: int) -> None:
        super().start(text, seed)
        self.goal_state, self.v3_events = None, []
        self._first_goal_done, self._final_wait = False, False
        self._xhist, self._closed_seen, self._kept, self._tag_return = [], False, 0, None
        self._b_after, self._closed_since_b, self._fw_event = None, False, None

    # ------------------------------------------------------------ 判定（(b) の数え方だけ変える。それ以外は TaskRuntime と同じ）
    def _judge(self, t: float) -> None:
        sensor = self.io.sense(cameras=True)
        if not {"overhead", "wrist"} <= set(sensor.cameras):
            return
        done = self.judge.update(sensor, self.prt.wm, self._hand(sensor.joints.q))
        last = self.judge.last
        in_box = self.phase == "run" and last["box_pixels"] >= self.judge.p["box_min_pixels"]
        away = last["retreat_dist_m"] > self.judge_tol
        placed = in_box and last["gripper_open"] and away
        if self.v3["b_placed_fix"] and self.placed_since is not None and in_box and away and not placed:
            placed = True                                     # 数え始めた後は、指の閉じ直しで切らない
            self._kept += 1
        if not placed:
            self.placed_since = None
        elif self.placed_since is None:
            self.placed_since = t
        self.rows.append({"t": t, "step": self.j, "attempt": self.attempt, "phase": self.phase, "ok": last["ok"],
                          "held_s": last["held_s"], "box_pixels": last["box_pixels"], "wrist_in": last["wrist_in_pixels"],
                          "depth_ok": last["depth_ok"], "retreat_dist_m": last["retreat_dist_m"]})
        if done and self.done_t is None:
            self.done_t = t

    # ------------------------------------------------------------ 制御
    def _control(self, t: float) -> None:
        if self.phase == "goal":
            self._goal_tick(t)
            return
        if self.phase != "run":
            super()._control(t)
            return
        rec = self.steps[-1]
        t_att = t - rec["attempts"][-1]["t_begin"]
        self._track(t)
        if self.done_t is not None:
            self._step_done(t, True)
        elif (bool(self.ret["enabled"]) and self.placed_since is not None
              and t - self.placed_since >= float(self.ret["trigger_s"]) - 1e-9):
            self._begin_return(t, "placed", judge_wait=True)
            self._tag_return = "placed"
        elif t_att >= self.step_timeout - 1e-9:
            self._attempt_over(t, "timeout")
        else:
            why = self._early_reason(t, t_att)
            if why:
                self._attempt_over(t, why)

    def _track(self, t: float) -> None:
        if self.prt.closed:
            self._closed_seen = True
            self._closed_since_b = True
        self._xhist.append((t, np.asarray(self.prt.motion.x_cmd, float).copy()))
        keep = t - float(self.v3["d_stall_s"]) - 0.05
        while len(self._xhist) > 2 and self._xhist[1][0] <= keep:
            self._xhist.pop(0)

    def _early_reason(self, t: float, t_att: float):
        if not self.v3["d_early_abort"] or self.placed_since is not None:
            return None
        s = float(self.v3["d_stall_s"])
        if t_att >= s - 1e-9 and self._xhist and self._xhist[0][0] <= t - s + 1e-9:
            x_now = self._xhist[-1][1]
            win = [x for tt, x in self._xhist if tt >= t - s - 1e-9]
            if max(float(np.linalg.norm(x - x_now)) for x in win) < float(self.v3["d_stall_move_m"]):
                return "stalled"
        if not self._closed_seen and t_att >= float(self.v3["d_noclose_s"]) - 1e-9:
            return "no_close"
        return None

    def _begin_step(self, t: float, j: int, attempt: int) -> None:
        if (j == 0 and attempt == 0 and self.goal_q is not None and self.v3["a_before_first_step"]
                and not self._first_goal_done):
            self._first_goal_done = True
            self._begin_goal(t, 0, 0, "placed", "goal")
            return
        super()._begin_step(t, j, attempt)
        self._xhist, self._closed_seen = [], False

    def _attempt_over(self, t: float, why: str) -> None:
        """試みの終わり（時間切れか (d) の打ち切り）。今の実行器の時間切れの道に、(a) と (c) を挟む。"""
        att = self.steps[-1]["attempts"][-1]
        att["end"] = why
        if why != "timeout":
            self.v3_events.append({"t": t, "kind": "early_abort", "why": why, "step": self.j, "attempt": self.attempt})
        if self.attempt < self.retries:
            if self.goal_q is not None and self.v3["a_before_retry"]:
                self._begin_goal(t, self.j, self.attempt + 1, "retry", "retry_goal")
            elif bool(self.ret["enabled"]):
                self._begin_return(t, "retry", judge_wait=False)
                self._tag_return = "retry"
            else:
                self._begin_step(t, self.j, self.attempt + 1)
            return
        if self.v3["c_final_wait"] and bool(self.ret["enabled"]) and self._target_in_box_by_perception():
            skip = self._final_wait_skip()
            if skip:
                self.v3_events.append({"t": t, "kind": "final_wait_skipped", "why": skip, "step": self.j, "attempt": self.attempt})
            else:
                self._fw_event = {"t": t, "kind": "final_wait", "step": self.j, "attempt": self.attempt}
                self.v3_events.append(self._fw_event)
                self._final_wait = True
                self._begin_return(t, "placed", judge_wait=True)
                self._tag_return = "final_wait"
                return
        self._step_done(t, False)

    def _target_in_box_by_perception(self) -> bool:
        wm = self.prt.wm
        if wm is None:
            return False
        e = wm.cubes.get(self.steps[-1]["color"])
        return e is not None and e.status in ("seen", "held") and bool(e.in_box)

    def _final_wait_skip(self):
        """(c) をしない理由（する場合は None）。指が開いていない・(b) の待ちが済んで指を閉じていない。"""
        last = getattr(self.judge, "last", None) or {}
        if self.prt.closed or last.get("gripper_open") is not True:
            return "gripper_closed"
        if self._b_after == (self.j, self.attempt) and not self._closed_since_b:
            return "after_placed_return"
        return None

    def _step_done(self, t: float, ok: bool) -> None:
        if not (ok and self.goal_q is not None and self.j + 1 < len(self.plan["steps"])):
            super()._step_done(t, ok)
            return
        rec = self.steps[-1]                                  # TaskRuntime._step_done と同じ記録
        rec["attempts"][-1]["t_judge"] = self.done_t
        rec.update(judged_complete=True, t_judge=self.done_t, t_end=t)
        self._begin_goal(t, self.j + 1, 0, "placed", "goal")

    def _return_tick(self, t: float) -> None:
        rs = self.ret_state
        tag = getattr(self, "_tag_return", None) or (rs or {}).get("kind")
        final = self._final_wait and rs is not None
        step_rec = self.steps[-1] if self.steps else None   # 待っている手順の記録（判定が出ると TaskRuntime が次の手順へ進む）
        n0 = len(self.returns)
        super()._return_tick(t)
        if len(self.returns) > n0:
            row = self.returns[-1]
            row["v3"] = tag
            self._tag_return = None
            if tag == "placed":                               # (b) の戻しが済んだ（(c) と重ねないための印）
                self._b_after, self._closed_since_b = (row["step"], row["attempt"]), False
        if final and self.ret_state is None:
            self._final_wait = False
            ok = bool(step_rec is not None and step_rec.get("judged_complete"))
            self._fw_event["judged"] = ok
            if not ok:
                self._step_done(t, False)                     # 判定が出なかった: 今の実行器と同じに止まる

    # ------------------------------------------------------------ (a) 関節空間の戻す動き（e7.DiagTaskRuntime と同じ段階）
    def _begin_goal(self, t: float, nxt: int, attempt: int, kind: str, tag: str) -> None:
        prt = self.prt
        prt.external = True
        prt.motion.set_velocity(np.zeros(3))
        opened = False
        if prt.closed:
            self.io.gripper_move(0.08, self.setup.gripper_speed)
            prt.closed = False
            opened = True
        self.goal_state = {"next": nxt, "next_attempt": attempt, "kind": kind, "tag": tag, "from_step": self.j,
                           "from_attempt": self.attempt, "t_begin": t, "stage": "open", "opened": opened, "rest": 0.0,
                           "last_w": None, "t_last": t, "rose": False, "t_joint": None, "t_arrive": None, "arrived": False}
        self.phase = "goal"

    def _goal_tick(self, t: float) -> None:
        prt, gs, gp = self.prt, self.goal_state, self.gp
        mo = prt.motion
        sensor = self.io.sense(cameras=False)
        if gs["stage"] == "open":
            w = sensor.gripper.width
            dt = t - gs["t_last"]
            if gs["last_w"] is not None and dt > 0:
                gs["rest"] = gs["rest"] + dt if abs(w - gs["last_w"]) / dt < 0.005 and w >= self.open_m else 0.0
            gs["last_w"], gs["t_last"] = w, t
            if not gs["opened"] or gs["rest"] >= 0.3 - 1e-9 or t - gs["t_begin"] >= gp["open_wait_max_s"] - 1e-9:
                if mo.x_cmd[2] < self.rm["rise_z"] - self.rm["z_tol"]:
                    gs["stage"], gs["rose"] = "rise", True
                else:
                    gs["stage"], gs["t_joint"] = "joint", t
                    mo.set_joint_goal(self.goal_q, gp["vmax_rad_s"], gp["gain_per_s"])
            return
        if gs["stage"] == "rise":
            z = mo.x_cmd[2]
            if abs(z - self.rm["rise_z"]) <= self.rm["z_tol"] or t - gs["t_begin"] >= gp["timeout_s"] - 1e-9:
                mo.set_velocity(np.zeros(3))
                gs["stage"], gs["t_joint"] = "joint", t
                mo.set_joint_goal(self.goal_q, gp["vmax_rad_s"], gp["gain_per_s"])
            else:
                vz = float(np.clip(self.rm["gain"] * (self.rm["rise_z"] - z), -self.rm["z_max"], self.rm["z_max"]))
                mo.set_velocity(np.array([0.0, 0.0, vz]))
            return
        if gs["stage"] == "joint":
            e, v = mo.joint_goal_error()
            timeout = t - gs["t_begin"] >= gp["timeout_s"] - 1e-9
            if e <= gp["tol_cmd_rad"] and v <= gp["tol_vel_rad_s"]:
                if gs.setdefault("n_corr", 0) < int(gp["meas_corrections"]) and not timeout:
                    gs["stage"], gs["t_settle"] = "settle_meas", t
                    return
                gs["arrived"], gs["t_arrive"] = True, t
            elif not timeout:
                return
            gs["err_cmd_rad"] = e
            gs["stage"], gs["t_hold"] = "hold", t
            return
        if gs["stage"] == "settle_meas":
            if t - gs["t_settle"] < gp["meas_settle_s"] - 1e-9:
                return
            q = np.asarray(sensor.joints.q, float)
            cmd = gs.get("goal_cmd", self.goal_q) + (self.goal_q - q)
            gs.setdefault("corr", []).append({"t": t, "err_meas_rad": float(np.max(np.abs(self.goal_q - q)))})
            gs["goal_cmd"], gs["n_corr"] = cmd, gs["n_corr"] + 1
            mo.set_joint_goal(cmd, gp["vmax_rad_s"], gp["gain_per_s"])
            gs["stage"] = "joint"
            return
        if gs["stage"] == "hold" and t - gs["t_hold"] >= gp["hold_s"] - 1e-9:
            q = np.asarray(sensor.joints.q, float)
            retry = gs["kind"] == "retry"
            self.returns.append({"step": gs["from_step"] if retry else gs["next"],
                                 "attempt": gs["from_attempt"] if retry else 0, "kind": gs["kind"], "v3": gs["tag"],
                                 "goal": self.v3["a_goal"], "t_begin": gs["t_begin"], "t_joint": gs["t_joint"],
                                 "t_arrive": gs["t_arrive"], "t_end": t, "arrived": gs["arrived"], "opened": gs["opened"],
                                 "rose": gs["rose"], "judged": False, "err_cmd_rad": gs.get("err_cmd_rad"),
                                 "corrections": gs.get("corr", []), "err_meas_rad": float(np.max(np.abs(self.goal_q - q)))})
            self.goal_state = None
            mo.clear_joint_goal()
            self._begin_step(t, gs["next"], gs["next_attempt"])

    # ------------------------------------------------------------ (f) 記録
    def interventions(self) -> dict:
        """種類別の数（56_intervention_s3.py と同じ種類。数え方は別の経路: returns の種類と attempts の数）。"""
        return {"scripted_return": sum(1 for r in self.returns if r.get("kind") == "placed"),
                "retry": sum(max(0, len(s["attempts"]) - 1) for s in self.steps),
                "replan": 0,
                "judge_override": sum(1 for s in self.steps if s.get("judged_complete") and s.get("t_judge") is None),
                "by_v3_kind": {k: sum(1 for r in self.returns if r.get("v3") == k)
                               for k in ("placed", "goal", "final_wait", "retry", "retry_goal")},
                "early_abort": {w: sum(1 for e in self.v3_events if e.get("why") == w) for w in ("stalled", "no_close")},
                "final_wait_skipped": {w: sum(1 for e in self.v3_events if e.get("kind") == "final_wait_skipped" and e.get("why") == w)
                                       for w in ("gripper_closed", "after_placed_return")},
                "placed_kept_on_reclose": self._kept}

    def record(self) -> dict:
        d = super().record()
        d["v3"] = {"version": VERSION, "settings": dict(self.v3), "goal_q": None if self.goal_q is None else self.goal_q.tolist(),
                   "goal_params": dict(self.gp), "events": self.v3_events, "interventions": self.interventions(),
                   "retreat_settings": dict(self.ret)}
        return d


def install(ex, v3: dict = None):
    """96_s4_resume.py の写しの make が作った TaskRuntime を v3 に差し替える（start の前に呼ぶ）。返り値は ex。"""
    ex.__class__ = V3TaskRuntime
    ex.v3_setup(v3)
    return ex


def make_v3(io, setup, prt, judge, planner_cfg: dict, motion_params: dict, decompose, instruction_fmt: str, v3: dict = None):
    return install(TaskRuntime(io, setup, prt, judge, planner_cfg, motion_params, decompose, instruction_fmt), v3)
