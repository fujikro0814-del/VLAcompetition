"""R2 の引き継ぎ区間を集める（手順書 Step H の 4、決裁 0068 の 2・0069 の 3・0070 の 1）。

    spec = R2Spec(seed=40000, kind="P2", mode="delayed")
    out = run_r2_attempt(rig, runner, spec, run_dir)      # 方策で走らせ、引き継ぎの時点から台本で立て直して保存

- 前半: R1 を評価と同じ設定（SceneRunner。rtc・s=10・d=4・範囲 40・指数）で 10 fps に走らせる。記録しない
  - 乙（kind P1・P2・P3）: 評価と同じ誘発（recovla.eval.induce.Inducer、定義は 6f6d2bc のまま）。成立の時刻を t_est として、
    mode "at_est" は t_est、mode "delayed" は t_est + U(0, 3) s（種ごとの専用の乱数列）で引き継ぐ
  - 甲（kind natural）: 誘発なし。失敗の検出（recovla.eval.failure_detect、0070 の 5）が発火した時点と、ランダムな時刻
    U(1, 9) s の早い方で引き継ぐ（どちらで引き継いだかを記録する）
  - 引き継ぎの判定は行動の境目（物理 50 手＝10 Hz のこまの境目）で行うので、引き継ぎの時点はそのまま Step F の分割点になる
- 後半: 引き継ぎの時点から記録を始め（snapshot.capture → restore、Step F の復帰と同じ）、新しい台本（引き継ぎの台本の種
  HANDOVER_RETRY_KEY）が入力の読み取りごとに立て直す。失敗の区分・成功の判定・保存の形は generate.run_attempt と同じ。
  立て直しの上限は inject.recovery_time_limit_s
- 捨てる: 誘発が成立しなかった、引き継ぐ前に成功した・時間切れ、台本が立て直せなかった。理由は結果に残す
"""
import dataclasses
import time

import numpy as np

from recovla.common import config, seeds
from recovla.common.seeds import COLORS
from recovla.eval import failure_detect as FD
from recovla.eval import induce as I
from recovla.eval import scene_trial as T
from recovla.expert import generate as G
from recovla.expert import script as S
from recovla.record import episode as E
from recovla.record import snapshot
from recovla.sim import contact, frames, scene

_CFG = config.load()
KINDS = ("P1", "P2", "P3", "natural")
DELAY_S = (0.0, 3.0)              # 乙の delayed: 成立の後に方策をそのまま走らせる時間（0068 の 2）
NATURAL_AT_S = (1.0, 9.0)         # 甲のランダムな引き継ぎの時刻
R2_RNG_KEY = 4600                 # 引き継ぎの時刻の乱数: SeedSequence([種, 4600, 種類の番号])


@dataclasses.dataclass(frozen=True)
class R2Spec:
    seed: int
    kind: str                     # P1 P2 P3 natural
    mode: str = "at_est"          # 乙: at_est / delayed。甲: detect_or_random

    def layout(self):
        return scene.sample_layout(self.seed)

    def target(self, layout) -> str:
        return T.choose_targets([self.seed], [layout])[0]

    def name(self) -> str:
        return f"R2{self.kind}_{self.seed}_{self.mode}"

    def handover_rng(self):
        return np.random.default_rng(np.random.SeedSequence([self.seed, R2_RNG_KEY, KINDS.index(self.kind)]))


def run_r2_attempt(rig, runner, spec: R2Spec, run_dir=None, render: bool = True, cfg: dict = None) -> dict:
    cfg = cfg or _CFG
    layout = spec.layout()
    color = spec.target(layout)
    ti = COLORS.index(color)
    rig.reset(layout)
    runner.start_trial(spec.seed)
    pp = S.PhaseParams.from_config(cfg)
    rng = spec.handover_rng()
    ind = I.Inducer(spec.kind, spec.seed, layout, color, rig, cfg) if spec.kind != "natural" else None
    delay = float(rng.uniform(*DELAY_S)) if spec.mode == "delayed" else 0.0
    t_random = float(rng.uniform(*NATURAL_AT_S)) if spec.kind == "natural" else None
    det = FD.FailureDetector(ti, cfg) if spec.kind == "natural" else None
    split_every = rig.record_every * int(cfg["sim"]["stride"])
    assert split_every == T.STEPS_PER_ACTION, "行動の境目が 10 Hz のこまの境目と一致する前提"
    time_limit = float(cfg["eval"]["time_limit_s"])
    wall0 = time.perf_counter()
    # ---------------------------------------------------------------- 前半: 方策（記録しない）
    k = 0
    frame, imgs = E.capture_frame(rig, color, 0.0, pp, render)
    raw = dict(zip(rig.cameras, imgs))
    task = T.instruction(color)
    t_handover, trigger, discard = None, None, None
    from recovla.sim.rig import quiet
    with quiet():
        while True:
            tr = rig.truth(color)
            if frames.in_box(tr.target_pos, tr.box) and tr.target_speed < pp.rest_speed and k > 0:
                discard = "success_before_handover"
            elif tr.t >= time_limit - 1e-9:
                discard = "timeout_before_handover"
            if discard:
                break
            if t_handover is not None and tr.t >= t_handover - 1e-9:
                break
            a = np.asarray(runner(k, frame, raw, task), dtype=np.float64)
            if ind is not None:
                a = np.asarray(ind.filter(k, a, tr), dtype=np.float64)
            T.execute_action(rig, a)
            tr = rig.truth(color)
            if ind is not None:
                ind.after(k, tr)
                if ind.reason is not None and not ind.established and ind.stage != "dropped":
                    discard = f"induce_{ind.reason}"
                elif ind.established and t_handover is None:
                    t_handover, trigger = float(ind.t_established) + delay, "established"
            else:
                ev = det.update(tr.t, tr.fingertip, tr.fingers, tr.gripper_closed, tr.cube_pos,
                                [frames.in_box(p, tr.box) for p in tr.cube_pos])
                if ev is not None and t_handover is None:
                    t_handover, trigger = tr.t, f"detect_{ev.kind}"
                elif t_handover is None and tr.t >= t_random - 1e-9:
                    t_handover, trigger = tr.t, "random"
            if discard:
                break
            frame, imgs = E.capture_frame(rig, color, 0.0, pp, render)
            raw = dict(zip(rig.cameras, imgs))
            k += 1
    tr = rig.truth(color)
    hand_state = {"t": round(float(tr.t), 3), "tip_to_cube_xy": float(np.hypot(*(tr.fingertip[:2] - tr.target_pos[:2]))),
                  "tip_z": float(tr.fingertip[2]), "finger_gap": float(np.sum(tr.fingers)),
                  "gripper_closed": bool(tr.gripper_closed), "holding": bool(S.holding(tr, pp)),
                  "target_rise": float(tr.target_pos[2] - frames.CUBE_REST_Z)}
    r2info = {"kind": spec.kind, "mode": spec.mode, "delay_s": delay, "t_random": t_random, "trigger": trigger,
              "t_handover": t_handover, "state_at_handover": hand_state, "policy_actions": k,
              "induce": ind.record() if ind is not None else None,
              "detect": [dataclasses.asdict(e) for e in det.events] if det is not None else None,
              "runtime": runner.runtime_record()}
    if discard:
        return {"name": spec.name(), "seed": spec.seed, "kind": spec.kind, "mode": spec.mode, "color": color,
                "saved": False, "discard": discard, "r2": r2info, "wall_s": round(time.perf_counter() - wall0, 3)}
    assert rig.step % split_every == 0
    # ---------------------------------------------------------------- 後半: 台本（記録する）
    dt = rig.steps_per_read * rig.timestep
    hp = S.sample_params(seeds.script_rng(spec.seed, color, G.HANDOVER_RETRY_KEY), cfg)
    expert = S.Expert(hp, dt, cfg)
    mask = contact.ContactMeter.obstacle_mask(color, rig.cubes_in_box())
    every = rig.record_every
    frames_log = []
    start = snapshot.capture(rig.model, rig.data, rig.controller, rig.integrator, rig.step)
    snapshot.restore(rig.model, rig.data, rig.controller, rig.integrator, start)
    rig.meter.reset_window()
    writer = E.SceneEpisodeWriter(run_dir, spec.name(), rig.cameras if render else ()) if run_dir else None

    def capture():
        f, im = E.capture_frame(rig, color, expert.clocks.target_rest_s, pp, render)
        frames_log.append(f)
        if writer is not None:
            writer.add_frame(f, im)

    def on_step(r):
        if writer is not None:
            writer.add_step(r.controller.desired_pos, float(r.data.ctrl[r.grip_act]))
        if r.step % every == 0:
            capture()

    if writer is not None:
        writer.add_step(rig.controller.desired_pos, float(rig.data.ctrl[rig.grip_act]))
    capture()
    t0 = float(rig.data.time)
    limit = float(cfg["inject"]["recovery_time_limit_s"])
    first, failure, t_fail = {}, None, None
    with quiet():
        while True:
            tr = rig.truth(color)
            cmd = expert.act(tr)
            ph = cmd.phase
            first.setdefault(ph.name, round(tr.t, 3))
            if "release" in first and ph in (S.Phase.approach, S.Phase.descend, S.Phase.close, S.Phase.lift,
                                             S.Phase.carry, S.Phase.reopen):
                failure = "place"
            elif "carry" in first and "release" not in first and ph in (S.Phase.settle, S.Phase.reopen):
                failure = "slip"
            elif tr.t - t0 > limit:
                failure = "timeout"
            if failure:
                t_fail = round(tr.t, 3)
                break
            if cmd.finished:
                break
            rig.pad_read(cmd.vel, cmd.press, on_step)
    tr = rig.truth(color)
    success = failure is None and frames.in_box(tr.target_pos, tr.box) and tr.target_speed < pp.rest_speed
    if failure is None and not success:
        failure = "place"
    cr = np.array([f["contact_robot"] for f in frames_log])[:, mask]
    summary = {"name": spec.name(), "seed": spec.seed, "kind": spec.kind, "mode": spec.mode, "color": color,
               "layout": layout.to_json(), "script": hp.to_json(), "saved": bool(success), "success": bool(success),
               "discard": None if success else f"script_{failure}", "failure": failure, "t_failure": t_fail,
               "t_record_start": round(t0, 3), "record_start_step": int(start.step),
               "recovery_duration_s": round(float(tr.t) - t0, 3), "n_frames": len(frames_log),
               "phase_first_t": first, "r2": r2info, "obstacle_mask": mask.tolist(),
               "contact_obstacle_frames": int(cr.any(axis=1).sum()) if cr.size else 0,
               "final_target_pos": [round(float(v), 5) for v in tr.target_pos],
               "wall_s": round(time.perf_counter() - wall0, 3)}
    if writer is not None:
        if success:
            meta = {**summary, "kind": f"R2{spec.kind}", "instruction": T.instruction(color), "target": color,
                    "layout_seed": spec.seed, "pair_id": spec.seed, "layout_kind": layout.kind, "start_pose": layout.start,
                    "duration_s": round(float(tr.t), 3), "record_every": every, "record_dt": every * rig.timestep,
                    "record_hz": round(1.0 / (every * rig.timestep)), "timestep": rig.timestep,
                    "cameras": {"names": list(rig.cameras) if render else [],
                                "width": rig.renderer.width if rig.renderer else None,
                                "height": rig.renderer.height if rig.renderer else None},
                    "rendered": bool(render), "frame0_image_shared_with_pair": False}
            summary["path"] = str(writer.finalize(meta, start.to_arrays(), every))
        else:
            writer.discard()
    return summary
