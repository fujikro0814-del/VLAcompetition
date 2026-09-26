"""Training launcher: one lerobot-train run on one converted dataset, described by one config JSON.

流用元 train_launcher.py。変えたのはパス（HF_HOME、出力先、ログ先、出発点のスナップショット）を
configs/default.yaml の paths から取ることと、数値（GPU の上限、log_freq の既定）の出所だけ。起動器の検査はそのまま。

What it fixes, from the data-management survey
(<共有フォルダ>\\03_収録\\データ管理と学習データ選定_調査報告.md, 2026-09-17):

  lerobot-train.exe (python -m lerobot.scripts.train does not exist in LeRobot 0.6.1)
  HF_HUB_OFFLINE=1, HF_HOME=<ROOT>\\models\\hf_home (configs paths.models_home)
  --policy.path = local snapshot dir of lerobot/smolvla_libero (a repo id fails on Windows)
  --dataset.repo_id=local/<dataset folder> + --dataset.root (repo_id alone looks in HF_HOME)
  --rename_map for both views (a one-view map silently trains with one image)
  --policy.push_to_hub=false (the snapshot config pushes to its author's repo and exits 1)
  --wandb.enable=false, --policy.device=cuda
  a new --output_dir per run (lerobot-train refuses an existing one), console output saved to a log
  <output_dir>\\conversion.json = copy of the dataset's meta\\conversion.json (checked by
  vla_observation at evaluation), copied as soon as lerobot-train has created <output_dir>
  <output_dir>\\train_run.json = config, command, dataset fingerprint, times, exit code, code version
  (git commit + dirty flag + file SHA-256, code_version.py) and a summary of the log: the frames and
  episodes lerobot-train actually read, loss curve statistics, peak GPU memory against the 14 GiB limit
  <output_dir>\\loss.csv, loss.png = the logged training loss per step (log_freq fixed by the launcher,
  because lerobot prints step numbers >= 1000 rounded, e.g. "1K")

What it does NOT fix (undecided; each run's config says it explicitly): expert-only or full training,
batch size, steps, and how a subset of episodes is chosen (a dataset converted from a manifest, or
--dataset.episodes / --dataset.eval_split passed through "extra_args").

Config JSON (unknown keys are rejected):
  dataset      LeRobot dataset dir made by convert_to_lerobot.py (has meta/conversion.json)   required
  train_scope  "expert" (action expert only, vision encoder frozen) or "full"                   required
  batch_size   int; above the Step C 14 GiB measurement (expert 62, full 12) is refused
               unless "allow_batch_over_14gib": true                                             required
  steps        int                                                                               required
  save_freq    int (default = steps)
  seed         int (default 1000)
  log_freq     int (default 50)
  num_workers  int (default: lerobot's)
  extra_args   list of further "--key=value" lerobot-train arguments (launcher-owned keys refused)
  note         free text, copied to train_run.json
  lora         {"layers": [8, ..., 15], "modules": ["q_proj", "v_proj"], "r": 16, "alpha": 32}: LoRA on those
               SmolVLM2 text layers; the action expert and the projections are fully trained
               (--peft.full_training_modules). Board 0040
  first_frames_weight  {"frames": 20, "weight": 5}: draw the first N frames of every episode W times as often
               (recovla.policy.train_wrapped). Board 0038/0040
  cue_augment  {"prob": 0.5, "max_m": 0.02}: shift the target cue (x, y) of each training sample with
               probability prob by U(0, max_m) in a uniform direction (board 0054). Actions unchanged
  With lora, first_frames_weight or cue_augment, lerobot-train runs through recovla.policy.train_wrapped, which also
  writes the in-memory policy's output on a fixed input (recovla_reference.pt) at every save.

    .venv\\Scripts\\python.exe -m recovla.policy.train_launcher CONFIG.json [--confirm] [--dry-run]
                                                 [--output-root DIR] [--log-root DIR]
"""
import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import shutil
import socket
import subprocess
import sys
import tempfile

from recovla.common import code_version, config
from recovla.data import vla_image_spec as spec
from recovla.data import vla_observation

_CFG = config.load()

LAUNCHER_VERSION = 3          # 2: log_freq, log summary, loss.csv/png, code version (2026-09-17)
                              # 3: lora, first_frames_weight, train_wrapped (2026-09-26, board 0040)
DEFAULT_LOG_FREQ = int(_CFG["train"]["log_freq"])
# lerobot logs mem_gb = torch.cuda.max_memory_allocated() / 1024**3 per logging interval, i.e. GiB of
# allocated tensors (the CUDA caching allocator reserves somewhat more).
GPU_LIMIT_GIB = float(_CFG["train"]["gpu_limit_gib"])
LOSS_CSV = "loss.csv"
LOSS_PNG = "loss.png"
HF_HOME = config.path(_CFG["paths"]["models_home"])
OUTPUT_ROOT = config.path(_CFG["paths"]["train_output"])
LOG_ROOT = config.path(_CFG["paths"]["train_logs"])
POLICY_SNAPSHOT = config.path(_CFG["paths"]["policy_snapshot"])   # lerobot/smolvla_libero
RUN_RECORD = "train_run.json"
CONFIG_COPY = "train_launch_config.json"
# policy camera slots of smolvla_libero <- dataset keys written by convert_to_lerobot.py
RENAME_MAP = {spec.IMAGE_KEYS["overhead"]: "observation.images.camera1",
              spec.IMAGE_KEYS["wrist"]: "observation.images.camera2"}
if RENAME_MAP != dict(_CFG["train"]["rename_map"]):
    raise ValueError(f"configs train.rename_map {_CFG['train']['rename_map']} != {RENAME_MAP}")
SCOPE_FLAGS = {"expert": ["--policy.train_expert_only=true", "--policy.freeze_vision_encoder=true"],
               "full": ["--policy.train_expert_only=false", "--policy.freeze_vision_encoder=false"]}
MAX_BATCH_14GIB = {"expert": 62, "full": 12}                 # Step C, per-process cap 14 GiB
REQUIRED = ("dataset", "train_scope", "batch_size", "steps")
OPTIONAL = ("save_freq", "seed", "log_freq", "num_workers", "extra_args", "note", "allow_batch_over_14gib",
            "lora", "first_frames_weight", "cue_augment")
OWNED = ("--policy.path", "--policy.push_to_hub", "--policy.repo_id", "--policy.device",
         "--policy.train_expert_only", "--policy.freeze_vision_encoder", "--dataset.root",
         "--dataset.repo_id", "--rename_map", "--output_dir", "--job_name", "--batch_size", "--steps",
         "--save_freq", "--seed", "--log_freq", "--num_workers", "--wandb.enable", "--resume", "--config_path",
         "--peft.target_modules", "--peft.full_training_modules", "--peft.method_type", "--peft.r",
         "--peft.lora_alpha", "--peft.init_type")
VLM_TEXT_LAYER = r"model\.vlm_with_expert\.vlm\.model\.text_model\.layers\.({layers})\.self_attn\.({modules})"
NUM_VLM_TEXT_LAYERS = 16             # SmolVLA reduces SmolVLM2 to 16 text layers ("Reducing the number of VLM layers")
# fully trained with LoRA: the action expert and the projections that are new in SmolVLA (B_提案書 §11)
FULL_TRAINING_MODULES = ["lm_expert", "state_proj", "action_in_proj", "action_out_proj", "action_time_mlp_in",
                         "action_time_mlp_out"]


def lora_flags(lora: dict) -> list:
    layers = [int(v) for v in lora["layers"]]
    target = VLM_TEXT_LAYER.format(layers="|".join(map(str, layers)), modules="|".join(lora["modules"]))
    return ["--peft.method_type=LORA", f"--peft.target_modules={target}",
            f"--peft.full_training_modules={json.dumps(FULL_TRAINING_MODULES)}",
            f"--peft.r={int(lora['r'])}", f"--peft.lora_alpha={int(lora['alpha'])}"]


def check_lora(path, lora) -> None:
    if not isinstance(lora, dict) or set(lora) != {"layers", "modules", "r", "alpha"}:
        raise LaunchError(f"{path}: lora must be {{layers, modules, r, alpha}}, got {lora!r}")
    if not lora["layers"] or not all(isinstance(v, int) and 0 <= v < NUM_VLM_TEXT_LAYERS for v in lora["layers"]):
        raise LaunchError(f"{path}: lora.layers must be text layer numbers 0..{NUM_VLM_TEXT_LAYERS - 1}")
    if not lora["modules"] or not all(m in ("q_proj", "k_proj", "v_proj", "o_proj") for m in lora["modules"]):
        raise LaunchError(f"{path}: lora.modules must be among q_proj, k_proj, v_proj, o_proj")
    for k in ("r", "alpha"):
        if not isinstance(lora[k], int) or isinstance(lora[k], bool) or lora[k] < 1:
            raise LaunchError(f"{path}: lora.{k} must be a positive integer")


def check_first_frames_weight(path, w) -> None:
    if not isinstance(w, dict) or set(w) != {"frames", "weight"}:
        raise LaunchError(f"{path}: first_frames_weight must be {{frames, weight}}, got {w!r}")
    if not isinstance(w["frames"], int) or w["frames"] < 1 or not isinstance(w["weight"], (int, float)) or w["weight"] <= 0:
        raise LaunchError(f"{path}: first_frames_weight needs frames >= 1 and weight > 0")


def check_cue_augment(path, c) -> None:
    if (not isinstance(c, dict) or set(c) != {"prob", "max_m"} or not 0.0 < float(c["prob"]) <= 1.0
            or not 0.0 < float(c["max_m"]) <= 0.1):
        raise LaunchError(f"{path}: cue_augment must be {{prob in (0, 1], max_m in (0, 0.1]}}, got {c!r}")


def wrapped(cfg) -> bool:
    return "lora" in cfg or "first_frames_weight" in cfg or "cue_augment" in cfg


def wrapper_prefix(cfg, python=None) -> list:
    """python -m recovla.policy.train_wrapped [...] -- (then the lerobot-train arguments)."""
    out = [str(python or sys.executable), "-m", "recovla.policy.train_wrapped", "--reference"]
    w = cfg.get("first_frames_weight")
    if w:
        out += [f"--first-frames={w['frames']}", f"--first-weight={w['weight']}"]
    c = cfg.get("cue_augment")
    if c:
        out += [f"--cue-aug-prob={c['prob']}", f"--cue-aug-max-m={c['max_m']}"]
    return out + ["--"]


class LaunchError(ValueError):
    pass


def load_config(path) -> dict:
    path = pathlib.Path(path)
    try:
        cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as e:
        raise LaunchError(f"cannot read config {path}: {e}") from e
    if not isinstance(cfg, dict):
        raise LaunchError(f"{path}: top level must be a JSON object")
    unknown = sorted(set(cfg) - set(REQUIRED) - set(OPTIONAL))
    if unknown:
        raise LaunchError(f"{path}: unknown keys {unknown} (allowed: {list(REQUIRED) + list(OPTIONAL)})")
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise LaunchError(f"{path}: missing required keys {missing}")
    if cfg["train_scope"] not in SCOPE_FLAGS:
        raise LaunchError(f"{path}: train_scope must be one of {list(SCOPE_FLAGS)}, got {cfg['train_scope']!r}")
    for k in ("batch_size", "steps", "save_freq", "seed", "log_freq", "num_workers"):
        if k in cfg and (not isinstance(cfg[k], int) or isinstance(cfg[k], bool) or cfg[k] < (0 if k in ("seed", "num_workers") else 1)):
            raise LaunchError(f"{path}: {k} must be a positive integer, got {cfg[k]!r}")
    limit = MAX_BATCH_14GIB[cfg["train_scope"]]
    if cfg["batch_size"] > limit and cfg.get("allow_batch_over_14gib") is not True:
        raise LaunchError(f"{path}: batch_size {cfg['batch_size']} > {limit}, the largest {cfg['train_scope']} batch "
                          f"measured under 14 GiB; beyond it the GPU spills into system memory and slows down "
                          f"without an error. Set \"allow_batch_over_14gib\": true to run anyway")
    extra = cfg.get("extra_args", [])
    if not isinstance(extra, list) or not all(isinstance(a, str) and a.startswith("--") for a in extra):
        raise LaunchError(f"{path}: extra_args must be a list of \"--key=value\" strings")
    for a in extra:
        if a.split("=", 1)[0] in OWNED:
            raise LaunchError(f"{path}: extra_args may not set {a.split('=', 1)[0]} (set by the launcher or a config key)")
    if "lora" in cfg:
        check_lora(path, cfg["lora"])
    if "first_frames_weight" in cfg:
        check_first_frames_weight(path, cfg["first_frames_weight"])
    if "cue_augment" in cfg:
        check_cue_augment(path, cfg["cue_augment"])
    return cfg


def check_dataset(dataset) -> dict:
    """The dataset must be a converter-v2 LeRobot dataset; returns its conversion.json."""
    root = pathlib.Path(dataset)
    for rel in ("meta/info.json", "meta/stats.json", "meta/conversion.json"):
        if not (root / rel).is_file():
            raise LaunchError(f"{root}: {rel} not found; give a dataset made by convert_to_lerobot.py")
    conversion = json.loads((root / "meta" / "conversion.json").read_text(encoding="utf-8"))
    spec.check_record(conversion, str(root / "meta" / "conversion.json"))
    return conversion


def dataset_fingerprint(dataset) -> dict:
    """Content identity of a dataset. Data parquet bytes differ between identical conversions (HF datasets
    fingerprint) and conversion.json differs in converted_at, so hash the deterministic parts instead."""
    root = pathlib.Path(dataset)
    conversion = json.loads((root / "meta" / "conversion.json").read_text(encoding="utf-8"))
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    parts = {
        "info_json_sha256": hashlib.sha256((root / "meta" / "info.json").read_bytes()).hexdigest(),
        "stats_json_sha256": hashlib.sha256((root / "meta" / "stats.json").read_bytes()).hexdigest(),
        "conversion_sha256_without_converted_at": hashlib.sha256(json.dumps(
            {k: v for k, v in conversion.items() if k != "converted_at"}, sort_keys=True).encode()).hexdigest(),
    }
    return {"total_episodes": info.get("total_episodes"), "total_frames": info.get("total_frames"),
            "converter_version": conversion.get("converter_version"),
            "raw_episode_ids": [s.get("raw_episode_id") for s in conversion.get("sources", [])], **parts,
            "fingerprint": hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()}


def build_command(trainer, policy_dir, cfg, output_dir, job_name) -> list:
    """trainer = command prefix as a list, normally [lerobot-train.exe]."""
    dataset = pathlib.Path(cfg["dataset"]).resolve()
    cmd = [*map(str, trainer), f"--policy.path={policy_dir}", f"--dataset.repo_id=local/{dataset.name}",
           f"--dataset.root={dataset}", f"--rename_map={json.dumps(RENAME_MAP)}",
           *SCOPE_FLAGS[cfg["train_scope"]], f"--batch_size={cfg['batch_size']}", f"--steps={cfg['steps']}",
           f"--save_freq={cfg.get('save_freq', cfg['steps'])}", f"--seed={cfg.get('seed', 1000)}",
           f"--log_freq={cfg.get('log_freq', DEFAULT_LOG_FREQ)}",
           f"--output_dir={output_dir}", f"--job_name={job_name}", "--policy.push_to_hub=false",
           "--policy.device=cuda", "--wandb.enable=false"]
    if "num_workers" in cfg:
        cmd.append(f"--num_workers={cfg['num_workers']}")
    if "lora" in cfg:
        cmd += lora_flags(cfg["lora"])
    return cmd + list(cfg.get("extra_args", []))


def policy_snapshot() -> pathlib.Path:
    """The local snapshot folder of lerobot/smolvla_libero under HF_HOME (a repo id fails on Windows)."""
    if not (POLICY_SNAPSHOT / "config.json").is_file():
        raise LaunchError(f"{POLICY_SNAPSHOT}: config.json not found; copy the starting model first "
                          f"(scripts/01_import_from_vla.ps1 -Part assets)")
    return POLICY_SNAPSHOT


def check_symlinks(where) -> None:
    """lerobot-train links checkpoints/last; without Developer Mode that fails only at the first save."""
    pathlib.Path(where).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=where) as d:
        try:
            os.symlink(pathlib.Path(d), pathlib.Path(d) / "link", target_is_directory=True)
        except OSError as e:
            raise LaunchError(f"cannot create symbolic links ({e}); enable Windows Developer Mode, "
                              f"lerobot-train needs one for checkpoints\\last") from e


def copy_conversion(dataset, output_dir) -> pathlib.Path:
    dst = pathlib.Path(output_dir) / vla_observation.COPIED_CONVERSION
    shutil.copyfile(pathlib.Path(dataset) / "meta" / "conversion.json", dst)
    vla_observation.load_training_spec(output_dir)          # the evaluation entry accepts it
    return dst


def sha256_file(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


_TRAIN_LINE = re.compile(r"step:(\S+) smpl:\S+ ep:\S+ epch:\S+ loss:([-\d.eE+naif]+) .*?mem_gb:([\d.]+)")
_EVAL_LINE = re.compile(r"step (\d+): eval_loss=([-\d.eE+naif]+)")


def parse_log(text: str, log_freq: int) -> dict:
    """What lerobot-train printed: frames/episodes read, loss per logged step, peak memory, eval losses.
    Steps come from the order of the log lines x log_freq (the printed step is rounded above 999)."""
    def last_int(pattern):
        found = re.findall(pattern, text)
        return int(found[-1]) if found else None

    rows = []
    for i, m in enumerate(_TRAIN_LINE.finditer(text)):
        printed, loss, mem = m.groups()
        step = (i + 1) * log_freq
        if printed.isdigit() and int(printed) != step:
            raise LaunchError(f"log line {i}: printed step {printed} != {step} (log_freq {log_freq})")
        rows.append({"step": step, "loss": float(loss), "mem_gb": float(mem)})
    losses = [r["loss"] for r in rows]
    n10 = max(1, len(losses) // 10)
    summary = {
        "dataset_num_frames": last_int(r"dataset\.num_frames=(\d+)"),
        "dataset_num_episodes": last_int(r"dataset\.num_episodes=(\d+)"),
        "train_eval_split": (re.findall(r"Train/eval split: ([^\r\n]+)", text) or [None])[-1],
        "log_points": len(rows),
        "last_step_logged": rows[-1]["step"] if rows else None,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_min": min(losses) if losses else None,
        "loss_mean_last_10pct": sum(losses[-n10:]) / n10 if losses else None,
        "loss_mean_previous_10pct": (sum(losses[-2 * n10:-n10]) / n10) if len(losses) >= 2 * n10 else None,
        "gpu_mem_allocated_max_gib": max((r["mem_gb"] for r in rows), default=None),
        "gpu_limit_gib": GPU_LIMIT_GIB,
        "eval_losses": [{"step": int(s), "eval_loss": float(v)} for s, v in _EVAL_LINE.findall(text)],
    }
    summary["gpu_over_limit"] = (summary["gpu_mem_allocated_max_gib"] is not None
                                 and summary["gpu_mem_allocated_max_gib"] > GPU_LIMIT_GIB)
    return {"rows": rows, "summary": summary}


def write_loss_files(rows, output_dir) -> None:
    out = pathlib.Path(output_dir)
    with open(out / LOSS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["step", "loss", "mem_gb"])
        w.writeheader()
        w.writerows(rows)
    if rows and importlib.util.find_spec("PIL") is not None:       # PIL: LeRobot venv only
        plot_loss([r["step"] for r in rows], [r["loss"] for r in rows], out / LOSS_PNG)


def plot_loss(steps, losses, path, size=(900, 500)) -> None:
    """Loss per logged step (thin) and its running mean over 10 % of the points (thick). PIL only
    (matplotlib is not installed in either environment)."""
    from PIL import Image, ImageDraw
    w, h = size
    left, right, top, bottom = 70, 20, 30, 50
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    y_max = max(losses) * 1.05
    y_min = 0.0
    x_max = max(steps)

    def px(s, v):
        return (left + (w - left - right) * s / x_max, top + (h - top - bottom) * (1 - (v - y_min) / (y_max - y_min)))

    d.rectangle([left, top, w - right, h - bottom], outline="black")
    for k in range(5):
        v = y_min + (y_max - y_min) * k / 4
        y = px(0, v)[1]
        d.line([left - 4, y, left, y], fill="black")
        d.text((5, y - 6), f"{v:.3f}", fill="black")
    for k in range(5):
        s = x_max * k / 4
        x = px(s, 0)[0]
        d.line([x, h - bottom, x, h - bottom + 4], fill="black")
        d.text((x - 15, h - bottom + 8), f"{int(s)}", fill="black")
    d.line([px(s, v) for s, v in zip(steps, losses)], fill=(150, 170, 220), width=1)
    n = max(1, len(losses) // 10)
    mean = [sum(losses[max(0, i - n + 1):i + 1]) / len(losses[max(0, i - n + 1):i + 1]) for i in range(len(losses))]
    d.line([px(s, v) for s, v in zip(steps, mean)], fill=(20, 40, 160), width=3)
    d.text((left, 8), f"training loss (thin: logged every {steps[0]} steps, thick: running mean of {n} points)",
           fill="black")
    d.text((w // 2 - 20, h - 20), "step", fill="black")
    img.save(path)


def run(config_path, output_root=OUTPUT_ROOT, log_root=LOG_ROOT, confirm=False, dry_run=False,
        trainer=None, policy_dir=None, now=None) -> int:
    config_path = pathlib.Path(config_path).resolve()
    cfg = load_config(config_path)
    dataset = pathlib.Path(cfg["dataset"]).resolve()
    check_dataset(dataset)
    fingerprint = dataset_fingerprint(dataset)
    stamp = (now or dt.datetime.now()).strftime("%Y%m%d-%H%M%S")
    run_name = f"{config_path.stem}_{stamp}"
    output_dir = pathlib.Path(output_root).resolve() / run_name
    log_path = pathlib.Path(log_root).resolve() / f"{run_name}.log"
    if output_dir.exists():
        raise LaunchError(f"{output_dir} already exists")
    if trainer is None:
        trainer = wrapper_prefix(cfg) if wrapped(cfg) else [pathlib.Path(sys.executable).with_name("lerobot-train.exe")]
    if not pathlib.Path(trainer[0]).is_file():
        raise LaunchError(f"{trainer[0]} not found; run with the LeRobot venv python")
    policy_dir = policy_dir or policy_snapshot()
    cmd = build_command(trainer, policy_dir, cfg, output_dir, run_name)
    print(f"\n config   {config_path}\n dataset  {dataset}\n          {fingerprint['total_episodes']} episodes, "
          f"{fingerprint['total_frames']} frames, raw ids {fingerprint['raw_episode_ids']}\n"
          f" scope    {cfg['train_scope']}   batch {cfg['batch_size']}   steps {cfg['steps']}\n"
          f" extra    {cfg.get('extra_args', [])}\n output   {output_dir}\n log      {log_path}\n")
    if dry_run:
        print(" command:\n  " + "\n  ".join(cmd))
        return 0
    check_symlinks(pathlib.Path(output_root))
    if confirm and input(" Start training? (y/n): ").strip()[:1].lower() != "y":
        print(" Cancelled.")
        return 0

    record = {"launcher_version": LAUNCHER_VERSION, "launcher_sha256": sha256_file(__file__),
              "code_version": code_version.code_version(), "run_name": run_name, "host": socket.gethostname(), "config_path": str(config_path),
              "config": cfg, "command": cmd, "dataset": {"root": str(dataset), **fingerprint},
              "policy_path": str(policy_dir), "output_dir": str(output_dir), "log": str(log_path),
              "started_at": dt.datetime.now().isoformat(timespec="seconds")}
    try:
        import lerobot
        record["lerobot_version"] = getattr(lerobot, "__version__", None)
    except ImportError:
        record["lerobot_version"] = None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HF_HOME=str(HF_HOME), HF_HUB_OFFLINE="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    copied = None
    exit_code = None
    with open(log_path, "wb") as log:
        log.write(("launcher: " + json.dumps(cmd, ensure_ascii=False) + "\n").encode("utf-8"))
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            while True:
                chunk = proc.stdout.read1(4096) if hasattr(proc.stdout, "read1") else proc.stdout.read(4096)
                if not chunk:
                    break
                log.write(chunk)
                log.flush()                                 # the log can be followed while training runs
                sys.stdout.buffer.write(chunk)
                sys.stdout.flush()
                if copied is None and output_dir.is_dir():
                    copied = copy_conversion(dataset, output_dir)
            exit_code = proc.wait()
        except KeyboardInterrupt:
            proc.wait()
            exit_code = "interrupted"
        finally:
            record["ended_at"] = dt.datetime.now().isoformat(timespec="seconds")
            record["exit_code"] = exit_code
            where = output_dir if output_dir.is_dir() else log_path.parent
            log.flush()                                     # still open: flush before reading it back
            try:
                parsed = parse_log(log_path.read_text(encoding="utf-8", errors="replace"),
                                   cfg.get("log_freq", DEFAULT_LOG_FREQ))
                record["log_summary"] = parsed["summary"]
            except (OSError, LaunchError) as e:
                parsed = None
                record["log_summary"] = {"error": repr(e)}
            if output_dir.is_dir():
                if copied is None:
                    copy_conversion(dataset, output_dir)
                shutil.copyfile(config_path, output_dir / CONFIG_COPY)
                if parsed is not None:
                    write_loss_files(parsed["rows"], output_dir)
            name = RUN_RECORD if where == output_dir else f"{run_name}.{RUN_RECORD}"
            (where / name).write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    s = record.get("log_summary", {})
    print(f"\n exit code {exit_code}\n output    {output_dir}\n log       {log_path}\n"
          f" read      {s.get('dataset_num_frames')} frames, {s.get('dataset_num_episodes')} episodes\n"
          f" loss      first {s.get('loss_first')}  last {s.get('loss_last')}  "
          f"mean last 10% {s.get('loss_mean_last_10pct')}  previous 10% {s.get('loss_mean_previous_10pct')}\n"
          f" gpu       max allocated {s.get('gpu_mem_allocated_max_gib')} GiB"
          f"{'  OVER 14 GiB' if s.get('gpu_over_limit') else ''}")
    return exit_code if isinstance(exit_code, int) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--confirm", action="store_true", help="ask y/n before starting")
    ap.add_argument("--dry-run", action="store_true", help="check the config and print the command only")
    ap.add_argument("--output-root", default=str(OUTPUT_ROOT))
    ap.add_argument("--log-root", default=str(LOG_ROOT))
    a = ap.parse_args(argv)
    try:
        return run(a.config, a.output_root, a.log_root, confirm=a.confirm, dry_run=a.dry_run)
    except (LaunchError, spec.ImageSpecError) as e:
        print(f"\n ERROR: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
