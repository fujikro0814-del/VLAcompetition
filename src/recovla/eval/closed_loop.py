"""Closed-loop evaluation, minimal version (scripted route check, 2026-09-17).

Run with the LeRobot venv (mujoco 3.2.3 was added to it on 2026-09-17 for this):

    C:\\VLA\\02_環境\\lerobot\\.venv\\Scripts\\python.exe closed_loop_eval.py run --checkpoint CKPT --out DIR
        (--placement-ids 4 [--repeats 10] [--first-seed 100000] | --eval-seeds 100000 100019)
    ... closed_loop_eval.py replay-dataset --dataset DIR [--episode 0] [--shift 1] --out DIR
    ... closed_loop_eval.py check-observation --dataset DIR [--episode 0] --out DIR
    ... closed_loop_eval.py open-loop --checkpoint CKPT --dataset DIR [--episode 0] --out DIR

Timing (Step 0 report §1): one 10 fps action covers 50 physics steps (RECORD_EVERY 25 x STRIDE 2, 0.1 s),
as in the conversion (action k = x_des[raw 2k+2] - x_des[raw 2k]) and the Step D 10 fps replay.

Entry: the action goes where the DualSense goes. ActionPad (a scripted_demo.ScriptPad) is the inner device of
collect.VelocityCommandIntegrator; for 50 physics steps it holds velocity = action_xyz / 0.1 s, so the
integrator adds action_xyz / 50 per step and x_des moves by exactly action_xyz. The gripper target is the sign
of action[6] (+1 close, -1 open); when it differs from the controller's state, square is pressed for the first
physics step of the window (the controller toggles on the rising edge; one press closes fully). Tracker, IK
and joints are never touched.

Observation: only vla_observation.PolicyObservationBuilder (raw renders from recorder.FrameSampler, the
recorder's same-instant capture). Its constructor checks <training output dir>/conversion.json against
vla_image_spec before the policy is loaded; the conversion table is never taken from an argument. At the first
inference the preprocessed batch is checked at value level: both camera slots present, each equal to the view
it should hold, and different from each other.

Success (本冊 3.2): within 30 s of the start, the cube is inside the box and its speed stays below 1 cm/s for
1 s. Judged every physics step; the trial ends at success or at 30 s. The simulated truth is used only for
this test and for the record, never to intervene.

Recorded per trial (materials only; no stage or failure classification):
  trial_NNNN.npz   20 Hz frames: hand pose, x_des, gripper command/state, fingers, joints, cube pose and
                   velocity, the 10 fps action executed after the frame, and for the 25 physics steps ending at
                   the frame: left/right finger touched the cube at least once, cube speed maximum
  trial_NNNN.json  seed, placement, checkpoint, code version, box opening and table height, chunk settings,
                   success and its time
  trial_NNNN_raw.mp4           raw renders (overhead | wrist), 20 fps
  trial_NNNN_policy_input.mp4  the transformed images given to the policy (overhead | wrist), 10 fps
"""
import argparse
import contextlib
import dataclasses
import io
import json
import os
import pathlib
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import code_version  # noqa: E402
import scripted_demo  # noqa: E402
import vla_image_spec as spec  # noqa: E402
import vla_observation  # noqa: E402
import vla_state  # noqa: E402
from teleop import app, collect, recorder  # noqa: E402

STRIDE = 2                                       # convert_to_lerobot.STRIDE (20 Hz raw -> 10 fps)
STEPS_PER_ACTION = STRIDE * collect.RECORD_EVERY  # 50
ACTION_DT = STEPS_PER_ACTION * 0.002             # checked against the model timestep in EvalRig
TIME_LIMIT_S = 30.0
REST_SPEED = 0.01
REST_HOLD_S = 1.0
TABLE_TOP_Z = collect.TABLE_TOP_Z
BOX_PHYSICAL_INNER_HALF_XY = 0.06                # teleop_scene.xml walls at +-0.065, 5 mm thick
EVAL_SEED_BASE = scripted_demo.EVAL_SEED_BASE
TASK = recorder.INSTRUCTION


# ------------------------------------------------------------------------------------------- rig

class EvalRig:
    """Simulation with the collection controller and the action pad in the DualSense's place."""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(app.SCENE_PATH)
        self.data = mujoco.MjData(self.model)
        if abs(STEPS_PER_ACTION * float(self.model.opt.timestep) - ACTION_DT) > 1e-12:
            raise RuntimeError(f"timestep {self.model.opt.timestep} does not give {ACTION_DT} s per action")
        with contextlib.redirect_stdout(io.StringIO()):
            self.start = collect.settle_start_state(self.model)
            self.controller = collect.make_collect_controller(self.model, self.data)
        self.pad = scripted_demo.ScriptPad()
        self.integrator = collect.make_integrator(self.pad, self.model, self.controller)
        self.renderer = mujoco.Renderer(self.model, collect.IMAGE_SIZE, collect.IMAGE_SIZE)
        self.sampler = recorder.FrameSampler(self.model, self.controller, self.renderer, collect.CAMERAS)
        mujoco.mj_forward(self.model, self.data)
        self.box_pos = self.data.xpos[self.model.body(recorder.BOX_BODY).id].copy()
        self.cube_id = self.model.body(recorder.CUBE_BODY).id
        jnt = self.model.body_jntadr[self.cube_id]
        self.cube_vadr = int(self.model.jnt_dofadr[jnt])
        self.left_id = self.model.body("left_finger").id
        self.right_id = self.model.body("right_finger").id

    def close(self) -> None:
        collect.close_renderer(self.renderer)

    def reset(self, placement: recorder.Placement) -> None:
        collect.reset_episode(self.model, self.data, self.controller, self.integrator, self.start, placement)
        self.pad.state = scripted_demo.pad_state()
        self.integrator.refresh()

    def cube_linvel(self) -> np.ndarray:
        return self.data.qvel[self.cube_vadr:self.cube_vadr + 3].copy()      # world frame (free joint)

    def finger_contacts(self):
        """(left finger touches cube, right finger touches cube) in the current contact list."""
        m, d = self.model, self.data
        left = right = False
        for i in range(d.ncon):
            c = d.contact[i]
            bodies = (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2])
            if self.cube_id in bodies:
                left |= self.left_id in bodies
                right |= self.right_id in bodies
        return left, right


def gripper_target(action) -> bool:
    """True = closed. +1 close / -1 open (LIBERO sign, convert_to_lerobot)."""
    return bool(float(action[6]) > 0.0)


# ------------------------------------------------------------------------------------ trial log

class SuccessJudge:
    def __init__(self, box_pos, dt: float):
        self.box_pos = np.asarray(box_pos, float)
        self.dt = float(dt)
        self.hold = 0.0
        self.success_time = None

    def update(self, cube_pos, speed: float, t: float) -> None:
        if self.success_time is not None:
            return
        if recorder.cube_in_box(cube_pos, self.box_pos) and speed < REST_SPEED:
            self.hold += self.dt
            if self.hold >= REST_HOLD_S - 1e-9 and t <= TIME_LIMIT_S + 1e-9:
                self.success_time = float(t)
        else:
            self.hold = 0.0


class TrialLog:
    FRAME_KEYS = ("step", "sim_time", "ee_pos", "ee_quat", "fingers", "joints", "x_des", "gripper_cmd",
                  "gripper_closed", "cube_pos", "cube_quat")

    def __init__(self, rig: EvalRig):
        self.rig = rig
        self.frames = {k: [] for k in self.FRAME_KEYS}
        self.extra = {"cube_linvel": [], "cube_angvel": [], "action": [], "contact_left_any": [],
                      "contact_right_any": [], "cube_speed_max": []}
        self.raw_video, self.policy_video = [], []
        self._left = self._right = False
        self._speed_max = 0.0
        self.judge = SuccessJudge(rig.box_pos, float(rig.model.opt.timestep))

    def on_step(self) -> None:
        l, r = self.rig.finger_contacts()
        self._left |= l
        self._right |= r
        speed = float(np.linalg.norm(self.rig.cube_linvel()))
        self._speed_max = max(self._speed_max, speed)
        self.judge.update(self.rig.data.xpos[self.rig.cube_id], speed, float(self.rig.data.time))

    def capture(self, step: int):
        frame, images = self.rig.sampler.capture(self.rig.data, step)
        for k in self.FRAME_KEYS:
            self.frames[k].append(frame[k])
        q = self.rig.sampler.scratch.qvel
        v = self.rig.cube_vadr
        self.extra["cube_linvel"].append(q[v:v + 3].copy())
        self.extra["cube_angvel"].append(q[v + 3:v + 6].copy())
        self.extra["contact_left_any"].append(self._left)
        self.extra["contact_right_any"].append(self._right)
        self.extra["cube_speed_max"].append(self._speed_max)
        self.extra["action"].append(np.full(7, np.nan))
        self._left = self._right = False
        self._speed_max = 0.0
        raw = dict(zip(collect.CAMERAS, images))
        self.raw_video.append(np.hstack([raw["overhead"], raw["wrist"]]))
        return frame, raw

    def set_action(self, frame_index: int, action) -> None:
        self.extra["action"][frame_index] = np.asarray(action, dtype=np.float64)

    def arrays(self) -> dict:
        out = {k: np.asarray(v) for k, v in self.frames.items()}
        out.update({k: np.asarray(v) for k, v in self.extra.items()})
        return out


# ------------------------------------------------------------------------------------ execution

def execute_action(rig: EvalRig, action, on_step=None) -> None:
    """One 10 fps action through the DualSense's entry point: 50 physics steps."""
    a = np.asarray(action, dtype=np.float64)
    vel = a[:3] / ACTION_DT
    press = gripper_target(a) != bool(rig.controller.gripper_closed)
    for j in range(STEPS_PER_ACTION):
        rig.pad.state = scripted_demo.pad_state(vel=vel, button_grip=bool(press and j == 0))
        rig.integrator.refresh()
        rig.controller.update(rig.integrator)
        mujoco.mj_step(rig.model, rig.data)
        if on_step is not None:
            on_step()
    rig.pad.state = scripted_demo.pad_state()
    rig.integrator.refresh()


def run_trial(rig: EvalRig, placement: recorder.Placement, act, time_limit_s: float = TIME_LIMIT_S) -> TrialLog:
    """act(k, frame, raw_by_view) -> 7-d action for 10 fps step k (frame = recorder frame dict)."""
    rig.reset(placement)
    log = TrialLog(rig)
    every = collect.RECORD_EVERY
    step = 0
    k = 0
    with contextlib.redirect_stdout(io.StringIO()):
        while True:
            frame, raw = log.capture(step)
            if log.judge.success_time is not None or step * rig.model.opt.timestep >= time_limit_s - 1e-9:
                break
            action = act(k, frame, raw)
            log.set_action(len(log.frames["step"]) - 1, action)
            half = {"n": 0}

            def on_step():
                log.on_step()
                half["n"] += 1
                if half["n"] == every:                # the 20 Hz frame inside this 10 fps action
                    log.capture(step + every)
                    log.set_action(len(log.frames["step"]) - 1, action)

            execute_action(rig, action, on_step)
            step += STEPS_PER_ACTION
            k += 1
    return log


# ------------------------------------------------------------------------------ policy actions

class PolicyActions:
    """SmolVLA from a training checkpoint. Construction order: image-spec check, then the policy."""

    def __init__(self, checkpoint, device: str = "cuda"):
        self.checkpoint_given = str(checkpoint)
        self.checkpoint = pathlib.Path(checkpoint).resolve()                        # checkpoints/last -> step
        self.builder = vla_observation.PolicyObservationBuilder(self.checkpoint)    # raises before loading
        import torch
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.converters import (batch_to_transition, policy_action_to_transition,
                                                  transition_to_batch, transition_to_policy_action)
        self.torch = torch
        self.device = device
        self.policy = SmolVLAPolicy.from_pretrained(str(self.checkpoint)).to(device).eval()
        self.pre = PolicyProcessorPipeline.from_pretrained(
            str(self.checkpoint), config_filename="policy_preprocessor.json",
            overrides={"device_processor": {"device": device}},
            to_transition=batch_to_transition, to_output=transition_to_batch)
        self.post = PolicyProcessorPipeline.from_pretrained(
            str(self.checkpoint), config_filename="policy_postprocessor.json",
            to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
        self.rename = {}
        for step in self.pre.steps:
            if hasattr(step, "rename_map"):
                self.rename.update(step.rename_map)
        self.config = {"chunk_size": int(self.policy.config.chunk_size),
                       "n_action_steps": int(self.policy.config.n_action_steps),
                       "n_obs_steps": int(self.policy.config.n_obs_steps),
                       "rename_map": dict(self.rename)}
        self.input_check = None
        self.generator = None
        self.policy_frames = []

    def start_trial(self, seed: int) -> None:
        self.policy.reset()
        self.generator = self.torch.Generator().manual_seed(int(seed))
        self.policy_frames = []

    def check_inputs(self, obs: dict, batch: dict) -> dict:
        """Value-level check that each dataset view reaches its camera slot (rename_map pitfall)."""
        t = self.torch
        result = {}
        slots = []
        for view, key in spec.IMAGE_KEYS.items():
            slot = self.rename.get(key, key)
            present = slot in batch
            same = bool(present and t.equal(batch[slot].detach().cpu().reshape(-1),
                                            t.from_numpy(obs[key]).reshape(-1)))
            result[view] = {"slot": slot, "present": present, "equals_view": same}
            slots.append(slot)
            if not (present and same):
                raise RuntimeError(f"{view} image does not reach policy slot {slot}: {result[view]}")
        if t.equal(batch[slots[0]], batch[slots[1]]):
            raise RuntimeError(f"camera slots {slots} hold identical images")
        state = batch["observation.state"]
        result["state_shape"] = list(state.shape)
        result["state_normalized_max_abs"] = float(state.abs().max())
        return result

    def prepare(self, obs: dict):
        t = self.torch
        batch = {key: t.from_numpy(val) for key, val in obs.items() if isinstance(val, np.ndarray)}
        batch["task"] = obs["task"]
        batch = self.pre(batch)
        if self.input_check is None:
            self.input_check = self.check_inputs(obs, batch)
        return batch

    def noise(self):
        return self.torch.randn((1, self.policy.config.chunk_size, self.policy.config.max_action_dim),
                                generator=self.generator).to(self.device)

    def __call__(self, k, frame, raw_by_view):
        obs = self.builder.build(raw_by_view, frame, TASK)
        self.policy_frames.append(np.hstack([
            np.round(obs[spec.IMAGE_KEYS[v]].transpose(1, 2, 0) * 255.0).astype(np.uint8)
            for v in ("overhead", "wrist")]))
        batch = self.prepare(obs)
        with self.torch.no_grad():
            action = self.policy.select_action(batch, noise=self.noise())
        action = self.post(action)
        return action.detach().cpu().numpy().reshape(-1)[:7].astype(np.float64)

    def chunk(self, raw_by_view, proprio) -> np.ndarray:
        """Open loop: the whole predicted chunk (chunk_size, 7) for one observation, unnormalized."""
        batch = self.prepare(self.builder.build(raw_by_view, proprio, TASK))
        with self.torch.no_grad():
            chunk = self.policy.predict_action_chunk(batch, noise=self.noise())
        return self.post(chunk).detach().cpu().numpy().reshape(-1, chunk.shape[-1])[:, :7].astype(np.float64)


# ------------------------------------------------------------------------------------- outputs

def write_mp4(path, frames, fps: int) -> None:
    """cv2.VideoWriter cannot open non-ASCII paths on Windows: write to a temporary ASCII path, then move."""
    path = pathlib.Path(path)
    h, w = frames[0].shape[:2]
    tmp_dir = pathlib.Path(tempfile.mkdtemp())
    tmp = tmp_dir / "video.mp4"
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    shutil.move(str(tmp), str(path))
    shutil.rmtree(tmp_dir, ignore_errors=True)


def save_trial(out_dir, index: int, log: TrialLog, attrs: dict, policy_frames=None) -> dict:
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"trial_{index:04d}"
    arrays = log.arrays()
    np.savez(out_dir / f"{name}.npz", **arrays)
    success_time = log.judge.success_time
    record = {
        **attrs,
        "success": success_time is not None,
        "success_time_s": success_time,
        "n_frames": int(len(arrays["step"])),
        "duration_s": float(arrays["sim_time"][-1]),
        "success_rule": {"time_limit_s": TIME_LIMIT_S, "rest_speed_m_s": REST_SPEED, "rest_hold_s": REST_HOLD_S,
                         "judged": "every physics step, live cube position and free-joint linear speed"},
        "constants": {
            "box_center": [float(v) for v in log.rig.box_pos],
            "box_success_inner_half_xy": recorder.BOX_INNER_HALF_XY,
            "box_physical_inner_half_xy": BOX_PHYSICAL_INNER_HALF_XY,
            "box_wall_top_z": recorder.BOX_WALL_TOP_Z,
            "table_top_z": TABLE_TOP_Z,
            "cube_half": recorder.CUBE_HALF,
        },
        "timing": {"physics_dt": float(log.rig.model.opt.timestep), "steps_per_action": STEPS_PER_ACTION,
                   "record_every": collect.RECORD_EVERY},
        "frame_fields": {
            "action": "10 fps action executed during the physics steps after this frame (NaN: none)",
            "contact_left_any/contact_right_any": "finger body touched the cube in at least one of the 25 "
                                                  "physics steps ending at this frame (frame 0: none before)",
            "cube_speed_max": "max free-joint linear speed [m/s] over the same 25 steps",
            "cube_linvel": "world frame [m/s]", "cube_angvel": "free-joint angular velocity, body frame [rad/s]",
        },
        "files": {"npz": f"{name}.npz", "raw_video": f"{name}_raw.mp4",
                  "policy_input_video": f"{name}_policy_input.mp4" if policy_frames else None},
    }
    (out_dir / f"{name}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False, default=float),
                                          encoding="utf-8")
    write_mp4(out_dir / f"{name}_raw.mp4", log.raw_video, 20)
    if policy_frames:
        write_mp4(out_dir / f"{name}_policy_input.mp4", policy_frames, 10)
    return record


def placement_attrs(p: recorder.Placement) -> dict:
    return {"placement_id": p.placement_id, "placement_seed": p.seed, "x": p.x, "y": p.y,
            "yaw_deg": float(np.degrees(p.yaw))}


# ----------------------------------------------------------------------------------- commands

def cmd_run(a) -> int:
    out = pathlib.Path(a.out)
    if out.exists():
        raise FileExistsError(f"{out} exists; evaluation runs are written to a new directory, not appended to")
    policy = PolicyActions(a.checkpoint, a.device)          # spec check happens here, before anything runs
    rig = EvalRig()
    training = {p.placement_id: p for p in recorder.training_placements()}
    trials = []
    if a.eval_seeds:
        lo, hi = a.eval_seeds
        trials = [(s, recorder.placement_from_seed(s)) for s in range(lo, hi + 1)]
    else:
        s = a.first_seed
        for _ in range(a.repeats):
            for pid in a.placement_ids:
                trials.append((s, training[pid]))
                s += 1
    if min(s for s, _ in trials) < EVAL_SEED_BASE:
        raise ValueError(f"evaluation seeds must be >= {EVAL_SEED_BASE}")
    version = code_version.code_version()
    out = pathlib.Path(a.out)
    results = []
    try:
        for i, (seed, placement) in enumerate(trials):
            policy.start_trial(seed)
            log = run_trial(rig, placement, policy)
            rec = save_trial(out, i, log, {
                "mode": "policy", "seed": seed, "placement": placement_attrs(placement),
                "checkpoint": str(policy.checkpoint), "checkpoint_given": policy.checkpoint_given,
                "code_version": version,
                "policy_config": policy.config, "input_check": policy.input_check}, policy.policy_frames)
            results.append({"trial": i, "seed": seed, "placement_id": placement.placement_id,
                            "success": rec["success"], "success_time_s": rec["success_time_s"]})
            print(f"[eval] trial {i} seed {seed} placement {placement.placement_id}: "
                  f"{'SUCCESS' if rec['success'] else 'fail'} ({rec['duration_s']:.1f} s)")
    finally:
        rig.close()
    (out / "summary.json").write_text(json.dumps({"checkpoint": str(policy.checkpoint), "trials": results,
                                                  "successes": sum(r["success"] for r in results)}, indent=2),
                                      encoding="utf-8")
    print(f"[eval] {sum(r['success'] for r in results)}/{len(results)} successes")
    return 0


def load_dataset_episode(dataset, episode: int):
    """10 fps actions of one episode read back from the LeRobot dataset, and its raw source."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root = pathlib.Path(dataset).resolve()
    ds = LeRobotDataset(f"local/{root.name}", root=root)
    ep = ds.meta.episodes[episode]
    lo, hi = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
    conv = json.loads((root / "meta" / "conversion.json").read_text(encoding="utf-8"))
    source = next(s for s in conv["sources"] if s["episode_index"] == episode)
    return ds, lo, hi, conv, source


def raw_placement(meta) -> recorder.Placement:
    c = meta["cube_init"]
    return recorder.Placement(meta.get("placement_id"), int(meta["placement_seed"]), float(c["x"]),
                              float(c["y"]), float(c["yaw"]))


def cmd_replay_dataset(a) -> int:
    ds, lo, hi, conv, source = load_dataset_episode(a.dataset, a.episode)
    actions = np.stack([ds.hf_dataset[i]["action"].numpy() for i in range(lo, hi)]).astype(np.float64)
    raw_path = pathlib.Path(source["raw_path"])
    meta = json.loads((raw_path / "meta.json").read_text(encoding="utf-8"))
    with np.load(raw_path / "data.npz") as z:
        raw = {k: z[k] for k in z.files}
    hold = np.array([0, 0, 0, 0, 0, 0, -1.0])
    if a.shift > 0:                                      # every action a.shift windows late
        hold_open = hold.copy()
        seq = np.vstack([np.repeat(hold_open[None], a.shift, axis=0), actions])
    else:
        seq = actions
    rig = EvalRig()
    try:
        start_equal = bool(np.array_equal(rig.start.qpos, raw["start_qpos"]))

        def act(k, frame, raw_by_view):
            if k < len(seq):
                return seq[k]
            last = hold.copy()
            last[6] = seq[-1][6]
            return last

        placement = raw_placement(meta)
        log = run_trial(rig, placement, act)
        arr = log.arrays()
        n = min(len(arr["ee_pos"]), len(raw["ee_pos"]), 2 * len(actions) + 1)
        ee_err = np.linalg.norm(arr["ee_pos"][:n] - raw["ee_pos"][:n], axis=1)
        rec = save_trial(a.out, a.shift, log, {
            "mode": "replay-dataset", "dataset": str(pathlib.Path(a.dataset).resolve()), "episode": a.episode,
            "raw_path": str(raw_path), "shift_actions": a.shift, "n_dataset_actions": int(len(actions)),
            "placement": placement_attrs(placement), "code_version": code_version.code_version(),
            "start_state_equals_raw": start_equal,
            "trajectory_vs_raw": {"frames_compared": int(n), "ee_err_max_m": float(ee_err.max()),
                                  "ee_err_mean_m": float(ee_err.mean())}})
    finally:
        rig.close()
    print(f"[replay-dataset] shift {a.shift}: success {rec['success']} at {rec['success_time_s']} s, "
          f"ee vs raw max {ee_err.max() * 1000:.3f} mm mean {ee_err.mean() * 1000:.3f} mm over {n} frames, "
          f"start state == raw {start_equal}")
    return 0


def cmd_check_observation(a) -> int:
    """The observation builder given the dataset's own frames must reproduce the dataset values."""
    import torch
    ds, lo, hi, conv, source = load_dataset_episode(a.dataset, a.episode)
    out = pathlib.Path(a.out)
    fake_run = out / "fake_training_output"                 # holds only the copied conversion.json
    fake_run.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pathlib.Path(a.dataset) / "meta" / "conversion.json", fake_run / "conversion.json")
    builder = vla_observation.PolicyObservationBuilder(fake_run)
    raw_path = pathlib.Path(source["raw_path"])
    meta = json.loads((raw_path / "meta.json").read_text(encoding="utf-8"))
    with np.load(raw_path / "data.npz") as z:
        raw = {k: z[k] for k in z.files}
    worst = {"image": 0.0, "state": 0.0}
    mismatched = []
    for k, i in enumerate(range(lo, hi)):
        item = ds[i]
        r = STRIDE * k
        raw_by_view = {v: recorder.read_png(raw_path / v / f"{r:06d}.png") for v in spec.CAMERAS}
        proprio = {f: raw[f][r] for f in vla_state.FRAME_FIELDS}
        obs = builder.build(raw_by_view, proprio, TASK)
        for key in spec.IMAGE_KEYS.values():
            d = float((torch.from_numpy(obs[key]) - item[key]).abs().max())
            worst["image"] = max(worst["image"], d)
            if d != 0.0:
                mismatched.append((k, key, d))
        d = float((torch.from_numpy(obs["observation.state"]) - item["observation.state"]).abs().max())
        worst["state"] = max(worst["state"], d)
        if d != 0.0:
            mismatched.append((k, "observation.state", d))

    # the evaluator's own renderer (this venv's mujoco) at the reset state vs the recorded frame 0
    rig = EvalRig()
    try:
        rig.reset(raw_placement(meta))
        frame, images = rig.sampler.capture(rig.data, 0)
        render_diff = {v: int(np.abs(img.astype(int) - recorder.read_png(raw_path / v / "000000.png").astype(int)).max())
                       for v, img in zip(collect.CAMERAS, images)}
        obs0 = builder.build(dict(zip(collect.CAMERAS, images)), frame, TASK)
        item0 = ds[lo]
        live = {key: float((torch.from_numpy(obs0[key]) - item0[key]).abs().max())
                for key in (*spec.IMAGE_KEYS.values(), "observation.state")}
        start_equal = bool(np.array_equal(rig.start.qpos, raw["start_qpos"]))
    finally:
        rig.close()
    result = {"dataset": str(pathlib.Path(a.dataset).resolve()), "episode": a.episode, "frames": hi - lo,
              "max_abs_diff_dataset_frames": worst, "mismatched_entries": mismatched[:20],
              "live_render_frame0_max_pixel_diff_vs_raw_png": render_diff,
              "live_frame0_observation_max_abs_diff_vs_dataset": live,
              "evaluator_start_state_equals_raw": start_equal}
    out.mkdir(parents=True, exist_ok=True)
    (out / "observation_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if not mismatched else 1


OPEN_LOOP_DIMS = (("dx [mm]", 0, 1000.0), ("dy [mm]", 1, 1000.0), ("dz [mm]", 2, 1000.0), ("gripper", 6, 1.0))


def plot_open_loop(recorded, predicted, path, title: str, size=(1000, 820)) -> None:
    """Rows dx, dy, dz, gripper over the episode frames: recorded (black) vs the policy's first chunk action
    (red). PIL only."""
    from PIL import Image, ImageDraw
    w, h = size
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    d.text((10, 5), title, fill="black")
    n = len(recorded)
    row_h = (h - 40) // len(OPEN_LOOP_DIMS)
    left, right = 70, 15
    for r, (label, dim, scale) in enumerate(OPEN_LOOP_DIMS):
        top = 25 + r * row_h + 5
        bottom = top + row_h - 25
        rec = recorded[:, dim] * scale
        pred = predicted[:, dim] * scale
        lo, hi = float(min(rec.min(), pred.min())), float(max(rec.max(), pred.max()))
        if hi - lo < 1e-9:
            lo, hi = lo - 1.0, hi + 1.0
        pad = 0.05 * (hi - lo)
        lo, hi = lo - pad, hi + pad

        def px(i, v):
            return left + (w - left - right) * i / max(n - 1, 1), bottom - (bottom - top) * (v - lo) / (hi - lo)

        d.rectangle([left, top, w - right, bottom], outline="black")
        if lo < 0 < hi:
            d.line([px(0, 0), px(n - 1, 0)], fill=(200, 200, 200))
        d.text((5, top), label, fill="black")
        d.text((5, top + 14), f"{hi:.2f}", fill="gray")
        d.text((5, bottom - 12), f"{lo:.2f}", fill="gray")
        d.line([px(i, v) for i, v in enumerate(rec)], fill="black", width=2)
        d.line([px(i, v) for i, v in enumerate(pred)], fill=(220, 30, 30), width=1)
    d.text((left, h - 14), f"frame (10 fps), 0 .. {n - 1}    black: recorded action    red: policy (first action "
                           f"of the chunk predicted from the recorded observation)", fill="black")
    img.save(path)


def cmd_open_loop(a) -> int:
    policy = PolicyActions(a.checkpoint, a.device)
    training_sources = json.loads((vla_observation.training_output_dir(policy.checkpoint) /
                                   vla_observation.COPIED_CONVERSION).read_text(encoding="utf-8"))["sources"]
    ds, lo, hi, conv, source = load_dataset_episode(a.dataset, a.episode)
    if source["raw_path"] not in {s["raw_path"] for s in training_sources}:
        raise ValueError(f"{source['raw_path']} was not in the training data of {policy.checkpoint}")
    recorded = np.stack([ds.hf_dataset[i]["action"].numpy() for i in range(lo, hi)]).astype(np.float64)
    raw_path = pathlib.Path(source["raw_path"])
    with np.load(raw_path / "data.npz") as z:
        raw = {k: z[k] for k in z.files}
    policy.start_trial(a.seed)
    first = []
    for k in range(hi - lo):
        r = STRIDE * k
        raw_by_view = {v: recorder.read_png(raw_path / v / f"{r:06d}.png") for v in spec.CAMERAS}
        first.append(policy.chunk(raw_by_view, {f: raw[f][r] for f in vla_state.FRAME_FIELDS})[0])
    predicted = np.stack(first)
    err = np.abs(predicted - recorded)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"open_loop_ep{a.episode:03d}"
    grip_agree = float(np.mean(np.sign(predicted[:, 6]) == np.sign(recorded[:, 6])))
    result = {"checkpoint": str(policy.checkpoint), "dataset": str(pathlib.Path(a.dataset).resolve()),
              "episode": a.episode, "raw_path": str(raw_path), "frames": int(hi - lo), "noise_seed": a.seed,
              "mae_xyz_mm": [float(v) * 1000 for v in err[:, :3].mean(axis=0)],
              "max_err_xyz_mm": [float(v) * 1000 for v in err[:, :3].max(axis=0)],
              "recorded_mean_abs_xyz_mm": [float(v) * 1000 for v in np.abs(recorded[:, :3]).mean(axis=0)],
              "gripper_sign_agreement": grip_agree, "input_check": policy.input_check,
              "code_version": code_version.code_version(),
              "note": "open-loop agreement is not evidence of closed-loop success"}
    (out / f"{name}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez(out / f"{name}.npz", recorded=recorded, predicted_first=predicted)
    plot_open_loop(recorded, predicted, out / f"{name}.png",
                   f"open loop, episode {a.episode} ({raw_path.name}), {policy.checkpoint}")
    print(json.dumps({k: result[k] for k in ("frames", "mae_xyz_mm", "max_err_xyz_mm", "recorded_mean_abs_xyz_mm",
                                             "gripper_sign_agreement")}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--checkpoint", required=True, help="checkpoints/<step>/pretrained_model of a training run")
    r.add_argument("--out", required=True, help="output directory (must not exist)")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--placement-ids", type=int, nargs="+")
    g.add_argument("--eval-seeds", type=int, nargs=2, metavar=("FIRST", "LAST"))
    r.add_argument("--repeats", type=int, default=1)
    r.add_argument("--first-seed", type=int, default=EVAL_SEED_BASE)
    r.add_argument("--device", default="cuda")
    d = sub.add_parser("replay-dataset")
    d.add_argument("--dataset", required=True)
    d.add_argument("--episode", type=int, default=0)
    d.add_argument("--shift", type=int, default=0)
    d.add_argument("--out", required=True)
    o = sub.add_parser("check-observation")
    o.add_argument("--dataset", required=True)
    o.add_argument("--episode", type=int, default=0)
    o.add_argument("--out", required=True)
    ol = sub.add_parser("open-loop")
    ol.add_argument("--checkpoint", required=True)
    ol.add_argument("--dataset", required=True)
    ol.add_argument("--episode", type=int, default=0)
    ol.add_argument("--seed", type=int, default=EVAL_SEED_BASE)
    ol.add_argument("--out", required=True)
    ol.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    return {"run": cmd_run, "replay-dataset": cmd_replay_dataset, "check-observation": cmd_check_observation,
            "open-loop": cmd_open_loop}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
