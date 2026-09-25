"""Scripted demonstrations recorded through the real collection path (scripted route check, 2026-09-17).

Task: タスク：台本データによる経路確認と、復帰入りデータの予備確認（通し） v2. Normal successful
pick-and-place episodes to check recording -> conversion -> training -> closed-loop evaluation before human
collection. Built on the wrist-mount Step D script (02_環境/lerobot/wrist_mount/stepd).

The script plays only the DualSense's role. ScriptPad emits DeviceState (stick velocity `vel`, square =
button_grip, circle = button_reset, triangle = button_save held, PS = button_quit); everything downstream is
the unmodified collection stack, in collect.main()'s order:

    integrator.refresh() -> session.handle_input(state)
    per physics step: controller.update(integrator) -> mj_step -> session.after_step()

so x_des, the gripper (one square press closes fully) and the saved episode come from the same entry point
and the same recorder as a human's. Speeds stay within the collect profile. The script reads the simulated
cube pose for its waypoints and for its own verdict (it is a script, not a policy).

Motion (placement fixed by the caller; variation drawn from one seed, see ScriptParams):
  circle -> via point above the cube (offset) -> over the cube -> descend -> square (close) -> lift 5 cm
  -> raise to the carry height -> over the box (offset) -> lower -> square (open) -> wait -> up
  -> triangle held (save)
Every episode is saved whatever happened. The script's verdict (first failed stage: grasp, carry, place)
goes to the generation log with the numbers behind it; training_list() keeps only verdict "success".

Output never goes to 03_収録 (refused). Seeds: TRAIN_SEED_LIMIT (3000) <= seed < EVAL_SEED_BASE (100000),
disjoint from the placement seeds and from evaluation.

Episode ids come from the episode ledger (teleop/ledger.py) with kind "script" (band 800000-, since
2026-09-17); raw_root must lie under a save folder registered in the ledger as "script". The ledger backup
is copied when a generation run ends normally.

    C:\\VLA\\pytools\\python311\\python.exe scripted_demo.py --raw-root DIR --log FILE.jsonl
        --placements 4 [5 ...] [--per-placement 1] [--seed-base 10000] [--ledger-dir DIR]
"""
import argparse
import contextlib
import dataclasses
import io
import json
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import code_version  # noqa: E402
from teleop import app, collect, ledger, recorder  # noqa: E402
from teleop.device import DeviceInput, DeviceState  # noqa: E402
from teleop.dualsense_device import PROFILES  # noqa: E402

EVAL_SEED_BASE = 100000
SCRIPT_OPERATOR = "script"
STEPS_PER_LOOP = 10               # 20 ms of physics per pad read, as in Step D
XY_MAX = PROFILES["collect"]["xy_speed"]
Z_MAX = PROFILES["collect"]["z_speed"]
MOVE_TOL_M = 0.001
MOVE_TIMEOUT_S = 20.0
TOP_Z = collect.START_POS[2]
LOW_Z = collect.COLLECT_WORKSPACE["workspace_z"][0]
LIFT_M = 0.05
RELEASE_Z = 0.18                  # hand height over the box when opening (Step D)
LIFT_OK_M = 0.03                  # grasp ok: cube raised at least this above resting after the lift
DROP_DIST_M = 0.035               # carry failed: cube centre farther than this from the fingertip centre
REST_SPEED = 0.01                 # m/s, 本冊 3.2
SAVE_HOLD_S = 1.2                 # triangle held longer than collect.SAVE_HOLD_STEPS (1.0 s)
STAGES = ("grasp", "carry", "place")


@dataclasses.dataclass(frozen=True)
class ScriptParams:
    seed: int
    via_offset: tuple        # (dx, dy) [m] of the approach via point above the cube
    speed_scale: float       # fraction of the collect profile's maximum speeds
    gain: float              # [1/s] velocity = gain * remaining distance (clipped)
    dwell_offset: float      # [s] added to every wait
    carry_z: float           # [m] hand height while carrying
    release_offset: tuple    # (dx, dy) [m] of the release point from the box centre


def sample_params(seed: int) -> ScriptParams:
    seed = int(seed)
    if not recorder.TRAIN_SEED_LIMIT <= seed < EVAL_SEED_BASE:
        raise ValueError(f"script seed {seed} outside [{recorder.TRAIN_SEED_LIMIT}, {EVAL_SEED_BASE})")
    rng = np.random.default_rng(seed)
    return ScriptParams(
        seed=seed,
        via_offset=tuple(float(v) for v in rng.uniform(-0.01, 0.01, 2)),
        speed_scale=float(rng.uniform(0.8, 1.0)),
        gain=float(rng.uniform(3.0, 5.0)),
        dwell_offset=float(rng.uniform(-0.1, 0.1)),
        carry_z=float(rng.uniform(0.22, 0.27)),
        release_offset=tuple(float(v) for v in rng.uniform(-0.01, 0.01, 2)),
    )


def pad_state(**kw) -> DeviceState:
    return DeviceState(pos=np.zeros(3), quat=np.array([1.0, 0.0, 0.0, 0.0]), button_clutch=True,
                       **{"button_grip": False, "vel": np.zeros(3), **kw})


class ScriptPad(DeviceInput):
    """The DualSense's place in the chain: the integrator reads this once per pad read."""

    def __init__(self):
        self.state = pad_state()

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def read(self) -> DeviceState:
        return self.state


def velocity_to(x_cmd, target, params: ScriptParams):
    d = np.asarray(target, float) - np.asarray(x_cmd, float)
    v = params.gain * d
    xy_max, z_max = XY_MAX * params.speed_scale, Z_MAX * params.speed_scale
    n = np.linalg.norm(v[:2])
    if n > xy_max:
        v[:2] *= xy_max / n
    v[2] = np.clip(v[2], -z_max, z_max)
    return v, float(np.linalg.norm(d))


def refuse_production_root(raw_root) -> pathlib.Path:
    root = pathlib.Path(raw_root).resolve()
    production = recorder.DEFAULT_RAW_DIR.parent.resolve()          # 03_収録
    if root == production or production in root.parents:
        raise ValueError(f"{root} is inside {production}; scripted data never goes to 03_収録")
    return root


class Rig:
    """One simulated collection sitting with the script in the pad's place."""

    def __init__(self, raw_root, ledger_dir=ledger.DEFAULT_LEDGER_DIR, operator: str = SCRIPT_OPERATOR,
                 kind: str = "script"):
        self.raw_root = refuse_production_root(raw_root) if kind == "script" else pathlib.Path(raw_root).resolve()
        self.model = mujoco.MjModel.from_xml_path(app.SCENE_PATH)
        self.data = mujoco.MjData(self.model)
        with contextlib.redirect_stdout(io.StringIO()):
            self.start = collect.settle_start_state(self.model)
            self.controller = collect.make_collect_controller(self.model, self.data)
        self.pad = ScriptPad()
        self.integrator = collect.make_integrator(self.pad, self.model, self.controller)
        self.renderer = mujoco.Renderer(self.model, collect.IMAGE_SIZE, collect.IMAGE_SIZE)
        try:
            self.session = collect.CollectSession(self.model, self.data, self.controller, self.integrator,
                                                  self.start, self.renderer, self.raw_root, operator, 1,
                                                  episode_ledger=ledger.EpisodeLedger(ledger_dir), kind=kind)
        except Exception:
            collect.close_renderer(self.renderer)
            raise
        self.ts = float(self.model.opt.timestep)
        self.cube_id = self.model.body(recorder.CUBE_BODY).id
        self.box_pos = self.session.box_pos.copy()
        self.placements = {p.placement_id: p for p in recorder.training_placements()}
        self.on_loop = None       # callable() after every pad read's physics, e.g. the carry monitor

    def close(self, backup: bool = True) -> None:
        try:
            self.session.close(backup=backup)
        finally:
            collect.close_renderer(self.renderer)

    # -- the pad's hands --
    def loop(self, n_loops: int = 1, **pad) -> bool:
        for _ in range(n_loops):
            self.pad.state = pad_state(**pad)
            if self.session.handle_input(self.integrator.refresh()):
                return True
            with contextlib.redirect_stdout(io.StringIO()):
                for _ in range(STEPS_PER_LOOP):
                    self.controller.update(self.integrator)
                    mujoco.mj_step(self.model, self.data)
                    self.session.after_step()
            self.session.note_loop(STEPS_PER_LOOP * self.ts, STEPS_PER_LOOP * self.ts, False)
            if self.on_loop is not None:
                self.on_loop()
        return False

    def clamp(self, target) -> np.ndarray:
        c = self.controller
        t = np.array(target, dtype=float)
        for i, (lo, hi) in enumerate((c.workspace_x, c.workspace_y, c.workspace_z)):
            t[i] = np.clip(t[i], lo, hi)
        return t

    def move(self, target, params: ScriptParams) -> None:
        target = self.clamp(target)
        for _ in range(int(MOVE_TIMEOUT_S / (STEPS_PER_LOOP * self.ts))):
            v, dist = velocity_to(self.integrator.x_cmd, target, params)
            if dist < MOVE_TOL_M:
                return
            self.loop(vel=v)
        raise RuntimeError(f"move to {target} timed out at x_cmd {self.integrator.x_cmd}")

    def wait(self, seconds: float, **pad) -> None:
        self.loop(max(1, int(round(seconds / (STEPS_PER_LOOP * self.ts)))), **pad)

    def press_grip(self) -> None:
        self.loop(button_grip=True)
        self.loop()

    # -- simulated truth (script only) --
    def cube_pos(self) -> np.ndarray:
        return self.data.xpos[self.cube_id].copy()

    def cube_speed(self) -> float:
        jnt = self.model.body_jntadr[self.cube_id]
        vadr = self.model.jnt_dofadr[jnt]
        return float(np.linalg.norm(self.data.qvel[vadr:vadr + 3]))

    def fingertip_center(self) -> np.ndarray:
        h = self.controller.hand_body_id
        return self.data.xpos[h] + self.data.xmat[h].reshape(3, 3)[:, 2] * collect.FINGERTIP_OFFSET


def run_episode(rig: Rig, placement_id: int, params: ScriptParams) -> dict:
    """Record and save one scripted episode; returns its generation record."""
    s = rig.session
    placement = rig.placements[int(placement_id)]
    s.placement = placement
    s.prepare()
    events = []

    def note(label, **extra):
        events.append({"label": label, "step": s.episode_step, **extra})

    rig.wait(0.2)
    rig.loop(button_reset=True)                    # circle: start recording
    rig.loop()
    if s.state != s.RECORDING:
        raise RuntimeError("recording did not start")
    previous_save = s.last_saved
    note("recording")
    rest_z = recorder.CUBE_HALF
    details = {"lift_m": None, "drop_step": None, "max_slip_m": 0.0}
    failed = None

    rig.wait(0.5 + params.dwell_offset)
    cube = rig.cube_pos()
    rig.move((cube[0] + params.via_offset[0], cube[1] + params.via_offset[1], TOP_Z), params)
    note("via_point")
    rig.move((cube[0], cube[1], TOP_Z), params)
    rig.move((cube[0], cube[1], LOW_Z), params)
    note("descended")
    rig.wait(0.3 + params.dwell_offset)
    rig.press_grip()
    note("close")
    rig.wait(1.0 + params.dwell_offset)
    rig.move((cube[0], cube[1], LOW_Z + LIFT_M), params)
    rig.wait(0.4 + params.dwell_offset)
    details["lift_m"] = float(rig.cube_pos()[2] - rest_z)
    note("lifted", cube_lift_m=round(details["lift_m"], 4))
    if details["lift_m"] < LIFT_OK_M:
        failed = "grasp"

    offset0 = rig.cube_pos() - rig.fingertip_center()

    def carry_monitor():
        rel = rig.cube_pos() - rig.fingertip_center()
        details["max_slip_m"] = max(details["max_slip_m"], float(np.linalg.norm(rel - offset0)))
        if details["drop_step"] is None and np.linalg.norm(rel) > DROP_DIST_M:
            details["drop_step"] = s.episode_step

    if failed is None:
        rig.on_loop = carry_monitor
    try:
        rig.move((cube[0], cube[1], params.carry_z), params)
        box = rig.box_pos
        release = (box[0] + params.release_offset[0], box[1] + params.release_offset[1])
        rig.move((release[0], release[1], params.carry_z), params)
        rig.move((release[0], release[1], RELEASE_Z), params)
    finally:
        rig.on_loop = None
    if failed is None and details["drop_step"] is not None:
        failed = "carry"
    note("over_box", drop_step=details["drop_step"], max_slip_m=round(details["max_slip_m"], 4))
    rig.press_grip()
    note("open")
    rig.wait(1.0 + params.dwell_offset)
    rig.move((release[0], release[1], params.carry_z), params)
    rig.wait(0.3 + params.dwell_offset)
    final = rig.cube_pos()
    speed = rig.cube_speed()
    in_box = recorder.cube_in_box(final, rig.box_pos)
    details.update(final_cube_pos=[round(float(v), 4) for v in final], final_cube_speed=round(speed, 5),
                   in_box=bool(in_box))
    if failed is None and not (in_box and speed < REST_SPEED):
        failed = "place"
    note("before_save")
    rig.wait(SAVE_HOLD_S, button_save=True)        # triangle held: save
    rig.wait(0.1)
    if s.state != s.READY or s.last_saved is None or s.last_saved is previous_save:
        raise RuntimeError(f"the episode was not saved ({s.message[0]})")
    episode_id, path = s.last_saved
    path = pathlib.Path(path)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    return {
        "episode_id": episode_id, "path": str(path), "placement_id": placement.placement_id,
        "placement": {"x": placement.x, "y": placement.y, "yaw_deg": round(float(np.degrees(placement.yaw)), 2)},
        "params": dataclasses.asdict(params), "verdict": "success" if failed is None else "failed",
        "failed_stage": failed, "details": details, "meta_success": bool(meta["success"]),
        "n_frames": int(meta["n_frames"]), "duration_s": float(meta["duration_s"]), "events": events,
    }


# ------------------------------------------------------------------------------------------ logs

def read_log(log_path) -> list:
    p = pathlib.Path(log_path)
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def training_list(records) -> list:
    """Episode directories usable for training: the script's verdict is success and the recorder agreed."""
    return [r["path"] for r in records if r["verdict"] == "success" and r["meta_success"]]


def summarize(records) -> dict:
    """Per placement: episodes, successes, rate, failures by stage (for the recovery-demo design)."""
    out = {}
    for r in records:
        row = out.setdefault(r["placement_id"], {"yaw_deg": r["placement"]["yaw_deg"], "n": 0, "success": 0,
                                                 **{f"failed_{st}": 0 for st in STAGES}})
        row["n"] += 1
        if r["verdict"] == "success" and r["meta_success"]:
            row["success"] += 1
        elif r["failed_stage"]:
            row[f"failed_{r['failed_stage']}"] += 1
    for row in out.values():
        row["rate"] = row["success"] / row["n"]
    return dict(sorted(out.items()))


def generate(raw_root, log_path, plan, ledger_dir=ledger.DEFAULT_LEDGER_DIR) -> list:
    """plan: [(placement_id, seed), ...]; appends one record per episode to log_path (jsonl)."""
    log_path = pathlib.Path(log_path)
    used = {r["params"]["seed"] for r in read_log(log_path)}
    clash = sorted(used & {int(seed) for _, seed in plan})
    if clash:
        raise ValueError(f"seeds already in {log_path}: {clash[:5]}")
    params = [(pid, sample_params(seed)) for pid, seed in plan]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    version = code_version.code_version()
    rig = Rig(raw_root, ledger_dir)
    records = []
    normal_end = False
    try:
        for pid, p in params:
            rec = run_episode(rig, pid, p)
            rec["code_version"] = version["code_sha256"]
            records.append(rec)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[script] ep_{rec['episode_id']:06d} placement {pid} seed {p.seed}: {rec['verdict']}"
                  f"{' (' + rec['failed_stage'] + ')' if rec['failed_stage'] else ''} "
                  f"lift {rec['details']['lift_m']:.3f} m, slip {rec['details']['max_slip_m'] * 1000:.1f} mm, "
                  f"{rec['n_frames']} frames")
        normal_end = True
    finally:
        rig.close(backup=normal_end)
    return records


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--log", required=True, help="generation log (jsonl, appended)")
    ap.add_argument("--placements", type=int, nargs="+", required=True)
    ap.add_argument("--per-placement", type=int, default=1)
    ap.add_argument("--seed-base", type=int, default=10000)
    ap.add_argument("--ledger-dir", default=str(ledger.DEFAULT_LEDGER_DIR))
    a = ap.parse_args(argv)
    plan, seed = [], a.seed_base
    for _ in range(a.per_placement):
        for pid in a.placements:
            plan.append((pid, seed))
            seed += 1
    generate(a.raw_root, a.log, plan, a.ledger_dir)
    for pid, row in summarize(read_log(a.log)).items():
        print(f"placement {pid} (yaw {row['yaw_deg']:+.1f}): {row['success']}/{row['n']}  "
              + "  ".join(f"{st} {row['failed_' + st]}" for st in STAGES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
