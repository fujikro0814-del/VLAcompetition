"""段階 3 の案 3: 方策を目標書 v2 の実行系で走らせ、閉じる前に台本へ引き継ぐ区間を集める（r2_collect の v2 への移植。
0126・0138、決まりは 0139）。

    rig = SensedDrivenRig(generate.rig_config("v3"))
    out = run_handover_attempt(rig, make_runtime, seed, run_dir)     # make_runtime(io, setup) -> PolicyRuntime

- 世界は 1 つ（harness/gen_v2.SensedDrivenRig: 世界・センサの模型・ハンド・G3 の監査）。前半は評価の枠（harness/loop.py の
  run_policy_trial）と同じ順（実行系の tick → 指令を世界へ → 物理の 1 手 → センサの模型）で方策を走らせる。記録しない
- 引き継ぎの判定は生成の道具の側に置く（真値を使ってよい。実行系には入れない）。判定は行動の境目（物理 50 手＝10 Hz の
  こまの境目＝実行系の行動の区切り）で行い、引き継ぎの時点はそのまま記録の最初のこまになる
    (i)   固まった: 開いていて、方策が行動を出し始めた後、手（hand の原点、真値）の速さ < STILL_SPEED が STILL_S 続いた
    (ii)  方策が閉じる指令を出したとき、指先の中心が目標の中心から水平 CLOSE_XY_M 超、または手が把持の高さ（台本の grasp_z）
          より CLOSE_Z_M 超上にある: その指令を開のままに替え（評価の道具の口 action_filter）、次の境目で引き継ぐ
    (iii) 種ごとに確率 RANDOM_P で、U(RANDOM_T_S) s のランダムな時刻（R と N で同じ値。種だけから引く）
- 後半: 台本（v3 の設定、引き継ぎの台本の種）が、実行系の Motion（x_cmd・追従器・IK・制限層の状態）とハンドをそのまま
  受け継いで立て直す（同じ世界を続けて進めるので、状態の保存と復元はしない）。記録・失敗の区分・保存の形は
  generate.run_attempt（v2 の生成）と同じ
- 捨てる: 誘発なしで引き継ぐ前に成功・時間切れ、引き継ぎの時点で閉じている、立方体が動いた、台本が立て直せなかった、
  G1〜G3 の違反（G3 は前半・引き継ぎ・後半を通した世界の監査）。理由は結果に残す
"""
import dataclasses
import time

import mujoco
import numpy as np

from recovla.common import seeds
from recovla.common.seeds import COLORS
from recovla.eval import scene_trial as T
from recovla.expert import generate as G
from recovla.expert import script as S
from recovla.harness.loop import HarnessHook, g1_audit
from recovla.harness.robot_io import SimRobotIO
from recovla.record import episode as E
from recovla.record import snapshot
from recovla.sim import contact, frames, scene
from recovla.sim.rig import quiet

STILL_SPEED = 0.01            # (i) 手の速さ [m/s]（台本が閉じる前に待つ hand_still_speed 0.005 の 2 倍）
STILL_S = 0.3                 # (i) 続いた時間 [s]
CLOSE_XY_M = 0.015            # (ii) 指先の中心と目標の中心の水平の差
CLOSE_Z_M = 0.015             # (ii) 手（hand の原点）の高さ − 台本の把持の高さ grasp_z
RANDOM_P = 0.3                # (iii)
RANDOM_T_S = (1.0, 8.0)
MOVED_M = 0.01                # 立方体が動いた: どれかの立方体の中心が開始時から 10 mm 超
HANDOVER_RNG_KEY = 4610       # (iii) の乱数: SeedSequence([種, 4610])
STEPS_PER_ACTION = 50
KIND = "H"                    # 記録の種類（変換は種類を区別しない）


@dataclasses.dataclass(frozen=True)
class HSpec:
    seed: int

    def layout(self):
        return scene.sample_layout(self.seed)

    def target(self, layout) -> str:
        return T.choose_targets([self.seed], [layout])[0]

    def name(self, color: str) -> str:
        return f"{KIND}_{self.seed}_{color}_r0"

    def random_time(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, HANDOVER_RNG_KEY]))
        u, t = float(rng.random()), float(rng.uniform(*RANDOM_T_S))
        return t if u < RANDOM_P else None


def _own_motion(rig):
    if not hasattr(rig, "_own_motion"):
        rig._own_motion = rig.motion
    return rig._own_motion


def _replay_boundary(rt, actions):
    """検査用（回し直し）: 方策の代わりに、記録した前半の行動（行動の口に入った後の値）を同じ区切りで出す。"""
    def boundary():
        k = rt.k
        a = np.asarray(actions[k] if k < len(actions) else [0, 0, 0, 0, 0, 0, 1.0 if rt.closed else -1.0], float)
        rt._apply(a)
        rt.log_act.append((k, rt.io.now(), None, False, a.copy()))
    return boundary


def run_handover_attempt(rig, make_runtime, seed: int, run_dir=None, model_name: str = None, replay: dict = None) -> dict:
    """replay = {"actions": 前半の行動の列, "step": 引き継ぎの物理の手}: 検査用。方策を走らせずに記録した行動を再生し、
    同じ手で引き継ぐ（台本の側と世界の側が決定的かを確かめる）。"""
    cfg = rig.cfg
    spec = HSpec(int(seed))
    layout = spec.layout()
    color = spec.target(layout)
    ti = COLORS.index(color)
    pp = S.PhaseParams.from_config(cfg)
    rig.motion = _own_motion(rig)                      # 前の試みで実行系の Motion に替えていたら戻す（reset が使う）
    rig.reset(layout)
    rtv = cfg.get("runtime_v2", {})
    io = SimRobotIO(rig, rig.suite, spec.seed, fixed_latency={"perception": float(rtv.get("perception_latency_s", 0.0))})
    rt = make_runtime(io, rig.setup_believed)
    rig.motion = rt.motion                             # 世界の側の記録（x_des・真値の x_cmd）を実行系の指令に合わせる
    task = T.instruction(color)
    rt.start(task, spec.seed)
    t_random = spec.random_time()
    time_limit = float(cfg["eval"]["time_limit_s"])
    cube0 = rig.data.xpos[rig.cube_ids].copy()
    vel6 = np.zeros(6)
    st = {"still_s": 0.0, "started": False, "pending": None, "close_checks": []}

    def act_filter(k, a):
        a = np.asarray(a, float)
        if a[6] > 0.0 and not rt.closed and st["pending"] is None:
            tr = rig.truth(color)
            dxy = float(np.hypot(*(tr.fingertip[:2] - tr.target_pos[:2])))
            dz = float(tr.hand_pos[2] - pp.grasp_z)
            st["close_checks"].append({"k": int(k), "t": round(tr.t, 3), "dxy_mm": round(dxy * 1e3, 2), "dz_mm": round(dz * 1e3, 2)})
            if dxy > CLOSE_XY_M or dz > CLOSE_Z_M:
                st["pending"] = "close_off"
        if st["pending"] == "close_off" and a[6] > 0.0:
            a = a.copy()
            a[6] = -1.0                                # 閉じる指令を開のままに替える（次の境目で引き継ぐ）
        st["started"] = st["started"] or rt.active is not None      # 方策の塊を実行し始めた（起動時の保持の後）
        return a
    rt.action_filter = HarnessHook(act_filter)
    if replay is not None:
        rt._action_boundary = _replay_boundary(rt, replay["actions"])
        st["started"] = True

    def on_step(r):
        mujoco.mj_objectVelocity(r.model, r.data, mujoco.mjtObj.mjOBJ_BODY, r.hand_id, vel6, 0)
        moving = float(np.linalg.norm(vel6[3:])) >= STILL_SPEED
        st["still_s"] = 0.0 if (moving or rt.closed or not st["started"]) else st["still_s"] + r.timestep

    wall0 = time.perf_counter()
    dt = rig.timestep
    stops = 0
    audits = [g1_audit(rt, rig, rig.suite)]
    trigger, discard = None, None
    with quiet():
        while True:
            if rig.step % STEPS_PER_ACTION == 0:
                tr = rig.truth(color)
                if rig.step > 0 and frames.in_box(tr.target_pos, tr.box) and tr.target_speed < pp.rest_speed:
                    discard = "success_before_handover"
                elif tr.t >= time_limit - 1e-9:
                    discard = "timeout_before_handover"
                elif replay is not None:
                    trigger = "replay" if rig.step >= int(replay["step"]) else None
                elif st["pending"] is not None:
                    trigger = st["pending"]
                elif st["still_s"] >= STILL_S - 1e-9:
                    trigger = "stalled"
                elif t_random is not None and tr.t >= t_random - 1e-9:
                    trigger = "random"
                if discard or trigger:
                    break
            t_before = float(rig.data.time)
            rt.tick()
            rig.closed = bool(rt.closed)
            rig.apply_joint_commands(io.take_commands())
            rig.physics_step(on_step)                  # SensedDrivenRig: 物理の 1 手 → センサの模型
            if abs(float(rig.data.time) - t_before - dt) > 1e-9:
                stops += 1
            if rig.step % 2500 == 0:
                audits.append(g1_audit(rt, rig, rig.suite))
    audits.append(g1_audit(rt, rig, rig.suite))
    rt.action_filter = None
    tr = rig.truth(color)
    moved = float(np.max(np.linalg.norm(tr.cube_pos - cube0, axis=1)))
    if discard is None and rt.closed:
        discard = "closed_at_handover"
    elif discard is None and moved > MOVED_M:
        discard = "cube_moved"
    ioa = io.audit()
    g12 = {"g1_violations": sum(len(x["violations"]) for x in audits), "g2_world_stops": stops,
           "g2_early_use": ioa["early_use"]}
    hand_state = {"t": round(float(tr.t), 3), "tip_to_cube_xy": float(np.hypot(*(tr.fingertip[:2] - tr.target_pos[:2]))),
                  "tip_z": float(tr.fingertip[2]), "hand_z_minus_grasp_z": float(tr.hand_pos[2] - pp.grasp_z),
                  "finger_gap": float(np.sum(tr.fingers)), "gripper_closed": bool(rt.closed),
                  "lead_xy": float(np.hypot(*(tr.x_cmd[:2] - tr.hand_pos[:2]))), "cube_moved_m": moved,
                  "phase": S.Phase(S.phase_of(tr, 0.0, pp)).name}
    hinfo = {"model": model_name, "trigger": trigger, "t_random": t_random, "t_handover": round(float(tr.t), 3),
             "step_handover": int(rig.step), "state_at_handover": hand_state, "close_checks": st["close_checks"],
             "policy_actions": int(rt.k + 1), "audit_policy": g12,
             "actions": [[float(v) for v in a] for _, _, _, _, a in rt.log_act],       # 行動の口に入った後の値（回し直しの検査用）
             "rules": {"still_speed": STILL_SPEED, "still_s": STILL_S, "close_xy_m": CLOSE_XY_M, "close_z_m": CLOSE_Z_M,
                       "random_p": RANDOM_P, "random_t_s": list(RANDOM_T_S), "moved_m": MOVED_M}}
    base = {"name": spec.name(color), "seed": spec.seed, "color": color, "handover": hinfo}
    if discard is None and (g12["g1_violations"] or stops or ioa["early_use"]):
        discard = "g1_g2"
    if discard:
        return {**base, "saved": False, "discard": discard, "g3": rig.audit_summary(),
                "wall_s": round(time.perf_counter() - wall0, 3)}
    assert rig.step % (rig.record_every * int(cfg["sim"]["stride"])) == 0
    # ---------------------------------------------------------------- 後半: 台本（記録する）
    rt_dt = rig.steps_per_read * rig.timestep
    hp = S.sample_params(seeds.script_rng(spec.seed, color, G.HANDOVER_RETRY_KEY), cfg)
    expert = S.Expert(hp, rt_dt, cfg)
    mask = contact.ContactMeter.obstacle_mask(color, rig.cubes_in_box())
    every = rig.record_every
    frames_log = []
    start = snapshot.capture(rig.model, rig.data, rig.controller, rig.integrator, rig.step)   # 記録の互換（start_*）
    rig.meter.reset_window()
    name = spec.name(color)
    writer = E.SceneEpisodeWriter(run_dir, name, rig.cameras) if run_dir else None

    def capture():
        f, im = E.capture_frame(rig, color, expert.clocks.target_rest_s, pp, writer is not None)
        f = rig.postprocess_frame(f)
        frames_log.append(f)
        if writer is not None:
            writer.add_frame(f, im)

    def on_rec(r):
        if writer is not None:
            writer.add_step(r.controller.desired_pos, float(r.data.ctrl[r.grip_act]))
        if r.step % every == 0:
            capture()

    rig.controller.desired_pos = rig.motion.x_cmd
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
            rig.pad_read(cmd.vel, cmd.press, on_rec)
    tr = rig.truth(color)
    success = failure is None and frames.in_box(tr.target_pos, tr.box) and tr.target_speed < pp.rest_speed
    if failure is None and not success:
        failure = "place"
    g3 = rig.audit_summary()
    g3_bad = int(g3["total_violations"])
    saved = bool(success) and g3_bad == 0
    cr = np.array([f["contact_robot"] for f in frames_log])[:, mask]
    summary = {**base, "layout": layout.to_json(), "script": hp.to_json(), "saved": saved, "success": bool(success),
               "discard": None if saved else (f"script_{failure}" if not success else "g3"), "failure": failure,
               "t_failure": t_fail, "t_record_start": round(t0, 3), "record_start_step": int(start.step),
               "recovery_duration_s": round(float(tr.t) - t0, 3), "n_frames": len(frames_log),
               "phase_first_t": first, "obstacle_mask": mask.tolist(), "g3": g3,
               "contact_obstacle_frames": int(cr.any(axis=1).sum()) if cr.size else 0,
               "final_target_pos": [round(float(v), 5) for v in tr.target_pos],
               "wall_s": round(time.perf_counter() - wall0, 3)}
    if writer is not None:
        if saved:
            meta = {**summary, "kind": KIND, "instruction": T.instruction(color), "target": color,
                    "layout_seed": spec.seed, "pair_id": spec.seed, "retry": 0, "layout_kind": layout.kind,
                    "start_pose": layout.start, "duration_s": round(float(tr.t), 3), "record_every": every,
                    "record_dt": every * rig.timestep, "record_hz": round(1.0 / (every * rig.timestep)),
                    "timestep": rig.timestep, "rendered": True, "frame0_image_shared_with_pair": False}
            meta.update(rig.episode_meta())
            summary["path"] = str(writer.finalize(meta, start.to_arrays(), every))
        else:
            writer.discard()
    return summary
