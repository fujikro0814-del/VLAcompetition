"""手順の実行器（手順書 Step I の 4、設計は掲示板 0088）。

    ex = TaskExecutor(rig, runner)                       # runner = policy.runner.SceneRunner（本線の設定）
    meta, arrays, video = ex.run(layout, "全部片付けて", seed)

1. 最初の俯瞰の画像で机上・箱の中の色を判定し（planner.detect）、LLM で手順の色の列に分ける（planner.decompose）
2. 手順ごとに、指示文と目標の手がかりの色を切り替え、実行器の塊を捨て（runner.start_trial。推論の雑音の列は
   (種, 手順の番号, やり直しの回数) から作る）、安全フィルタの障害物の組を作り直す（rig.safety.start_trial）
3. 完了判定（planner.judge）が真になったら次の手順へ。1 手順 planner.step_timeout_s（30 s）を超えたら、方策と
   実行器の状態を初期化して、その場から同じ手順を planner.retry（1）回だけやり直す。それでも駄目なら止めて返答で知らせる
記録（評価のため。方策と判定には渡さない）: こまごとの真値（scene_trial と同じ形）、手順の番号・やり直しの回数・
完了判定の条件、手順ごとの真値の成功の時刻（目標が箱の成功の体積で 1 s 静止＝評価器と同じ）
"""
import time

import numpy as np

from recovla.common import code_version, config
from recovla.common.seeds import COLORS
from recovla.eval import scene_trial as T
from recovla.expert.script import PhaseParams
from recovla.perception import color as PC
from recovla.planner import decompose as D
from recovla.planner.detect import Regions, wrist_box_mask
from recovla.planner.judge import CompletionJudge
from recovla.record import episode as E
from recovla.sim import frames

_CFG = config.load()
NOISE_KEY = 7100


def step_seed(seed: int, step: int, attempt: int) -> int:
    return int(np.random.SeedSequence([int(seed), NOISE_KEY, int(step), int(attempt)]).generate_state(1)[0])


class TaskExecutor:
    def __init__(self, rig, runner, cfg: dict = None, decompose=None):
        self.rig, self.runner = rig, runner
        self.cfg = cfg or _CFG
        self.regions = Regions(PC.overhead_calibration(rig.model), frames.box_outer_half(rig.model))
        self.judge = CompletionJudge(self.regions, self.cfg)
        self.decompose = decompose or D.decompose
        p = self.cfg["planner"]
        self.step_timeout = float(p["step_timeout_s"])
        self.retries = int(p["retry"])

    def run(self, layout, text: str, seed: int, render_video: bool = False):
        rig = self.rig
        ev = self.cfg["eval"]
        rest_speed, rest_hold = float(ev["success"]["rest_speed"]), float(ev["success"]["rest_hold_s"])
        pp = PhaseParams.from_config()
        rig.reset(layout)
        dt = rig.timestep
        f0, imgs0 = E.capture_frame(rig, COLORS[0], 0.0, pp, True)
        raw0 = dict(zip(rig.cameras, imgs0))
        table = self.regions.table_colors(raw0[PC.OVERHEAD])
        inbox = self.regions.box_colors(raw0[PC.OVERHEAD])
        plan = self.decompose(text, table, inbox)
        frames_log, rows, video, steps_rec = [], [], [], []
        state = {"step": -1, "attempt": 0, "color": None, "hold": 0.0, "truth_t": None, "raw": raw0, "done_t": None}

        def capture(color):
            f, imgs = E.capture_frame(rig, color, 0.0, pp, True)
            state["raw"] = dict(zip(rig.cameras, imgs))
            wm = wrist_box_mask(rig.model, rig.scratch, rig.renderer.width, rig.renderer.height)
            done = self.judge.update(f["sim_time"], state["raw"][PC.OVERHEAD], f["gripper_closed"], f["x_des"],
                                     state["raw"]["wrist"], wm)
            frames_log.append(f)
            rows.append({"step": state["step"], "attempt": state["attempt"], "judge_ok": self.judge.last["ok"],
                         "judge_held_s": self.judge.last["held_s"], "box_pixels": self.judge.last["box_pixels"],
                         "retreat_dist_m": self.judge.last["retreat_dist_m"]})
            if render_video:
                video.append(np.hstack(imgs))
            if done and state["done_t"] is None:
                state["done_t"] = float(f["sim_time"])
            return f

        def on_step(r):
            ti = COLORS.index(state["color"])
            d = r.data
            pos = d.xpos[r.cube_ids[ti]]
            v = r.cube_vadr[ti]
            speed = float(np.linalg.norm(d.qvel[v:v + 3]))
            if state["truth_t"] is None:
                if frames.in_box(pos, r.box) and speed < rest_speed:
                    state["hold"] += dt
                    if state["hold"] >= rest_hold - 1e-9:
                        state["truth_t"] = float(r.step * dt)
                else:
                    state["hold"] = 0.0
            if r.step % r.record_every == 0:
                capture(state["color"])

        stopped = None
        wall0 = time.perf_counter()
        for j, color in enumerate(plan["steps"]):
            state.update(step=j, color=color, hold=0.0, truth_t=None)
            rec = {"step": j, "color": color, "attempts": [], "t_start": float(rig.data.time)}
            finished = False
            for attempt in range(self.retries + 1):
                state.update(attempt=attempt, done_t=None)
                self.runner.start_trial(step_seed(seed, j, attempt))
                rig.safety.start_trial(color)
                self.judge.reset(color)
                task = T.instruction(color)
                t_begin = float(rig.data.time)
                f = frames_log[-1] if frames_log else f0
                raw = state["raw"]
                k = 0
                from recovla.sim.rig import quiet
                with quiet():
                    while state["done_t"] is None and float(rig.data.time) - t_begin < self.step_timeout - 1e-9:
                        a = np.asarray(self.runner(k, f, raw, task), dtype=np.float64)
                        rig.safety.gate = True
                        T.execute_action(rig, a, on_step)
                        f, raw = frames_log[-1], state["raw"]
                        k += 1
                rig.safety.gate = False
                rec["attempts"].append({"attempt": attempt, "t_begin": t_begin, "t_judge": state["done_t"],
                                        "policy_actions": k})
                if state["done_t"] is not None:
                    finished = True
                    break
            rec.update(judged_complete=finished, t_judge=state["done_t"], t_truth_success=state["truth_t"],
                       truth_in_box_at_end=bool(frames.in_box(rig.data.xpos[rig.cube_ids[COLORS.index(color)]], rig.box)))
            steps_rec.append(rec)
            if not finished:
                stopped = {"step": j, "color": color, "reply": f"{j + 1} 番目の手順（{color}）を {self.retries + 1} 回試して"
                                                               "終えられなかったので、止めました"}
                break
        final_in_box = {c: bool(frames.in_box(rig.data.xpos[rig.cube_ids[i]], rig.box)) for i, c in enumerate(COLORS)}
        planned_ok = bool(plan["steps"]) and all(final_in_box[c] for c in plan["steps"])
        meta = {"seed": int(seed), "text": text, "layout": {"kind": layout.kind, "start": layout.start, "seed": layout.seed,
                                                              "table_colors": list(layout.table_colors)},
                "detected": {"table": table, "box": inbox}, "plan": plan, "steps": steps_rec, "stopped": stopped,
                "final_in_box": final_in_box, "all_planned_in_box": planned_ok,
                "all_three_in_box": all(final_in_box.values()), "t_end": float(rig.data.time),
                "runtime": {**self.runner.runtime_record(), "safety_filter": rig.safety.enabled},
                "wall_s": round(time.perf_counter() - wall0, 1), "code_version": code_version.code_version()}
        keys = ("step", "sim_time", "ee_pos", "fingertip", "fingers", "x_des", "gripper_closed", "cube_pos",
                "cube_in_box", "phase", "contact_robot", "min_dist", "safety_active")
        arrays = {k: np.array([f[k] for f in frames_log]) for k in keys} if frames_log else {}
        for k in ("step", "attempt", "judge_ok", "judge_held_s", "box_pixels", "retreat_dist_m"):
            arrays["plan_" + k if k in ("step",) else k] = np.array([r[k] for r in rows]) if rows else np.zeros(0)
        return meta, arrays, video
