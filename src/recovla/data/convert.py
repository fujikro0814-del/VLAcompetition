"""Convert raw teleop episodes (teleop/recorder.py) to a LeRobot v3.0 dataset (C5).

Run with the LeRobot environment, NOT the teleop python311 (lerobot 0.6.1
needs Python >= 3.12 and numpy 2):

    C:\\VLA\\02_環境\\lerobot\\.venv\\Scripts\\python.exe convert_to_lerobot.py
        --out DIR --name NAME (--raw-dir RAW | EPISODE_DIR ... | --manifest M.json)
    ... convert_to_lerobot.py --verify DIR [--export-actions NPZ_DIR] [--check-image PNG]

Only reads the raw episodes (03_収録/raw is never modified). Format follows
HuggingFaceVLA/libero (LeRobot v3.0, images embedded, 10 fps), approved
2026-09-16:

  observation.images.image    overhead camera  } raw PNG -> policy image via
  observation.images.image2   wrist camera     } vla_image_spec.to_policy_image
                              (the only place that defines image transforms:
                              currently overhead = left-right flip, wrist =
                              up-down flip, both net; the spec is recorded in
                              meta/conversion.json and checked by --verify and
                              at evaluation time)
  observation.state (15)      eef pos 3 [m, world], eef orientation 3 [rad]
                              as the deviation from pointing straight down
                              (see below), fingers 2 [m] (second negated,
                              as LIBERO), joints 7 [rad]
  action (7)                  x_des change over the 0.1 s AFTER the frame
                              (3, metres, world), rotation 3 = 0,
                              gripper +1 close / -1 open (state at the end
                              of that 0.1 s, as in LIBERO)

10 fps from the 20 Hz raw frames: frame k = raw frame 2k; action k =
x_des[2k+2] - x_des[2k]. The last raw frame(s) without a following 0.1 s are
dropped.

observation.state is computed by vla_state.policy_state (the only definition;
the evaluation entry vla_observation.py uses the same function). The
orientation representation and why LIBERO's absolute axis-angle was not used
are documented there.
"""
import argparse
import datetime
import hashlib
import json
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import vla_image_spec as spec  # noqa: E402
import vla_state  # noqa: E402
from teleop import recorder  # noqa: E402  (only for DEFAULT_RAW_DIR: one definition of the path)
# re-exported for existing callers/tests; the definitions live in vla_state
from vla_state import Q_DOWN, STATE_NAMES, orientation_deviation, quat_mul  # noqa: E402,F401

# 2: per-view net image transforms from vla_image_spec, recorded in conversion.json (2026-09-17);
# 1: both views flipped left-right, no recorded spec
CONVERTER_VERSION = 2
FPS = 10
RAW_HZ = 20
STRIDE = RAW_HZ // FPS
IMAGE_KEYS = spec.IMAGE_KEYS
ACTION_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
ZERO_STD_EPS = 1e-8   # LeRobot MEAN_STD: (x - mean) / (std + eps)


def features(size: int = 256) -> dict:
    img = {"dtype": "image", "shape": [size, size, 3], "names": ["height", "width", "channel"]}
    return {
        **{IMAGE_KEYS[view]: dict(img) for view in spec.CAMERAS},
        "observation.state": {"dtype": "float32", "shape": [len(STATE_NAMES)], "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": [len(ACTION_NAMES)], "names": ACTION_NAMES},
    }


# ------------------------------------------------------------- raw side

def load_raw(path: pathlib.Path):
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    with np.load(path / "data.npz") as npz:
        data = {k: npz[k] for k in npz.files}
    return meta, data


def raw_episodes(raw_dir: pathlib.Path) -> list:
    eps = [p for p in raw_dir.glob("*/ep_*")
           if p.is_dir() and not p.name.endswith(".partial") and (p / "meta.json").is_file()]
    return sorted(eps, key=lambda p: json.loads((p / "meta.json").read_text(encoding="utf-8"))["episode_id"])


def episode_arrays(meta: dict, data: dict) -> dict:
    """10 fps state/action arrays + the raw frame indices they come from."""
    if int(meta["record_hz"]) != RAW_HZ:
        raise ValueError(f"episode {meta['episode_id']}: record_hz {meta['record_hz']} != {RAW_HZ}")
    n = int(meta["n_frames"])
    k_frames = (n - 1) // STRIDE
    if k_frames < 1:
        raise ValueError(f"episode {meta['episode_id']}: too short ({n} raw frames)")
    idx = np.arange(k_frames) * STRIDE
    nxt = idx + STRIDE
    state = vla_state.policy_state(data["ee_pos"][idx], data["ee_quat"][idx],
                                   data["fingers"][idx], data["joints"][idx])
    grip = np.where(data["gripper_closed"][nxt], 1.0, -1.0)[:, None]
    action = np.concatenate([data["x_des"][nxt] - data["x_des"][idx],
                             np.zeros((k_frames, 3)), grip], axis=1)
    return {"raw_index": idx, "state": state, "action": action.astype(np.float32)}


def read_raw_image(path: pathlib.Path) -> np.ndarray:
    """The recorded PNG as H x W x 3 uint8, untransformed."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB")).copy()


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------ conversion

class ManifestError(ValueError):
    """The manifest cannot be converted. Raised before any frame is written."""


def raw_episodes_by_id(raw_root: pathlib.Path) -> dict:
    return {json.loads((p / "meta.json").read_text(encoding="utf-8"))["episode_id"]: p
            for p in raw_episodes(raw_root)}


def read_labels(labels_root: pathlib.Path) -> dict:
    """{episode_id: the latest line}. Same rule as the review screen: append only, latest wins."""
    latest = {}
    if not labels_root.is_dir():
        return latest
    for path in sorted(labels_root.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                latest[int(row["episode_id"])] = row
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return latest


def episodes_from_manifest(manifest_path, raw_root=None, labels_root=None):
    """(episode dirs in manifest order, the manifest dict). Stops before converting when

      * an episode id has no folder under the raw root,
      * an episode was judged 失敗 (the task was not achieved), or
      * an episode has no label at all,

    each with the offending ids. A manifest is a file a person can write by hand, and the screen
    is not the only way here, so the converter checks rather than trusting the caller.
    """
    manifest_path = pathlib.Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids = [int(i) for i in manifest.get("episodes", [])]
    if not ids:
        raise ManifestError(f"{manifest_path}: no episodes listed")
    raw_root = pathlib.Path(raw_root) if raw_root else recorder.DEFAULT_RAW_DIR
    # manifests/<name>.json and labels/<day>.jsonl live side by side under the same folder
    labels_root = pathlib.Path(labels_root) if labels_root else manifest_path.parent.parent / "labels"
    found = raw_episodes_by_id(raw_root)
    missing = [i for i in ids if i not in found]
    if missing:
        raise ManifestError(f"{manifest_path}: no recording under {raw_root} for episode(s) "
                            f"{missing}")
    labels = read_labels(labels_root)
    unlabelled = [i for i in ids if i not in labels]
    if unlabelled:
        raise ManifestError(f"{manifest_path}: episode(s) {unlabelled} have no label in "
                            f"{labels_root}; every episode in a manifest must be reviewed first")
    invalid = [i for i in ids if labels[i].get("result") == "失敗"]
    if invalid:
        raise ManifestError(f"{manifest_path}: episode(s) {invalid} are labelled 失敗 "
                            f"(the task was not achieved) and cannot be training data")
    return [found[i] for i in ids], manifest


def manifest_record(manifest_path, manifest: dict) -> dict:
    """What goes into conversion.json: enough to see what was converted without opening the
    manifest file again."""
    return {"path": str(pathlib.Path(manifest_path).resolve()),
            "name": manifest.get("name"), "created": manifest.get("created"),
            "base_commit": manifest.get("base_commit"), "rule": manifest.get("rule"),
            "n": manifest.get("n"), "teacher_frames": manifest.get("teacher_frames"),
            "composition": manifest.get("composition"),
            "sha256": sha256(pathlib.Path(manifest_path))}


def convert(episodes: list, out: pathlib.Path, name: str, manifest: dict = None) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if out.exists():
        raise FileExistsError(f"{out} exists; converted datasets are rebuilt, not appended to")
    first_meta, _ = load_raw(episodes[0])
    ds = LeRobotDataset.create(repo_id=f"local/{name}", fps=FPS, features=features(
        int(first_meta["cameras"]["width"])), root=out, robot_type="panda", use_videos=False)
    sources = []
    try:
        for ep_index, path in enumerate(episodes):
            meta, data = load_raw(path)
            arr = episode_arrays(meta, data)
            for k, i in enumerate(arr["raw_index"]):
                images = spec.policy_images({view: read_raw_image(path / view / f"{i:06d}.png")
                                             for view in spec.CAMERAS})
                ds.add_frame({
                    **images,
                    "observation.state": arr["state"][k],
                    "action": arr["action"][k],
                    "task": meta["instruction"],
                })
            ds.save_episode()
            sources.append({"episode_index": ep_index, "raw_episode_id": meta["episode_id"],
                            "raw_path": str(path), "frames": int(len(arr["raw_index"])),
                            "raw_frames": int(meta["n_frames"]),
                            "placement_id": meta.get("placement_id"),
                            "operator": meta.get("operator"), "session": meta.get("session"),
                            "success": meta.get("success"), "retakes": meta.get("retakes"),
                            "meta_sha256": sha256(path / "meta.json"),
                            "data_sha256": sha256(path / "data.npz")})
            print(f"[convert] ep {ep_index}: raw ep_{meta['episode_id']:06d} "
                  f"{meta['n_frames']} raw frames -> {len(arr['raw_index'])} frames")
    finally:
        ds.finalize()
    (out / "meta" / "conversion.json").write_text(json.dumps({
        "converter_version": CONVERTER_VERSION,
        "converted_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "fps": FPS, "raw_hz": RAW_HZ, "stride": STRIDE,
        **spec.spec_record(),
        "images": "PNG from raw -> vla_image_spec net transform per view (image_transforms)",
        "state": STATE_NAMES, "action": ACTION_NAMES,
        "action_definition": "xyz = x_des[raw 2k+2] - x_des[raw 2k] (m, world); rot = 0; "
                             "gripper = +1 closed / -1 open at raw frame 2k+2",
        "orientation": "world-frame axis-angle of ee_quat * conj(q_down), q_down = (0,1,0,0) "
                       "(w,x,y,z); 0 = pointing straight down",
        "fingers": "finger_joint1, -finger_joint2 (LIBERO sign)",
        **({"manifest": manifest} if manifest else {}),
        "sources": sources,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------- verification

def write_check_image(path: pathlib.Path, raw_by_view: dict, policy_by_view: dict, caption: str) -> None:
    """One PNG: a row per view, [raw PNG | policy input as stored in the dataset], same frame."""
    size = next(iter(raw_by_view.values())).shape[0]
    label_h = 18
    canvas = Image.new("RGB", (2 * size, len(spec.CAMERAS) * (size + label_h) + label_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 3), caption, fill=(255, 255, 0))
    for r, view in enumerate(spec.CAMERAS):
        y = label_h + r * (size + label_h)
        draw.text((4, y + 3), f"{view}: raw PNG (as recorded / operator view)", fill=(255, 255, 255))
        draw.text((size + 4, y + 3), f"{view}: policy input ({spec.VIEW_TRANSFORMS[view]})", fill=(255, 255, 255))
        canvas.paste(Image.fromarray(raw_by_view[view]), (0, y + label_h))
        canvas.paste(Image.fromarray(policy_by_view[view]), (size, y + label_h))
    canvas.save(path)


def verify(out: pathlib.Path, export_dir, check_image=None) -> bool:
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    conv = json.loads((out / "meta" / "conversion.json").read_text(encoding="utf-8"))
    ds = LeRobotDataset(repo_id=f"local/{out.name}", root=out)
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  ok   " if cond else "  FAIL ") + msg)
        ok &= bool(cond)

    print(f"[verify] {out}: {ds.num_episodes} episodes, {ds.num_frames} frames, fps {ds.fps}")
    try:
        spec.check_record(conv, str(out / "meta" / "conversion.json"))
        check(True, f"image spec v{spec.SPEC_VERSION} recorded == runtime {spec.VIEW_TRANSFORMS}")
    except spec.ImageSpecError as exc:
        check(False, f"image spec: {exc}")
    check(ds.fps == FPS, f"fps == {FPS}")
    check(ds.num_episodes == len(conv["sources"]), "episode count matches conversion.json")
    check(ds.num_frames == sum(s["frames"] for s in conv["sources"]), "frame count matches")

    # stats: finite, and normalization never divides by zero
    stats = ds.meta.stats
    for key in ("observation.state", "action"):
        mean = np.asarray(stats[key]["mean"], dtype=np.float64)
        std = np.asarray(stats[key]["std"], dtype=np.float64)
        check(np.all(np.isfinite(mean)) and np.all(np.isfinite(std)), f"{key} stats finite")
        names = ds.meta.features[key]["names"]
        tiny = [f"{n}={s:.2e}" for n, s in zip(names, std) if s < 1e-3]
        print(f"       {key} std: " + ", ".join(f"{n}={s:.4g}" for n, s in zip(names, std)))
        if tiny:
            print(f"       {key} dims with std < 1e-3 (normalization amplifies noise): {tiny}")

    # every frame: shapes, dtypes, task, normalized finite; compare with raw
    exports = {}
    max_norm = {"observation.state": 0.0, "action": 0.0}
    # frames where the expected transform differs from every other net transform, per view: without
    # at least one, a wrong transform could not have been detected (e.g. symmetric images)
    discriminating = {view: 0 for view in spec.CAMERAS}
    check_frame = None
    for src in conv["sources"]:
        meta, data = load_raw(pathlib.Path(src["raw_path"]))
        arr = episode_arrays(meta, data)
        ep = src["episode_index"]
        start = int(ds.meta.episodes[ep]["dataset_from_index"])
        stop = int(ds.meta.episodes[ep]["dataset_to_index"])
        check(stop - start == src["frames"], f"ep {ep}: {stop - start} frames")
        acts, states = [], []
        img_err = 0
        for k in range(stop - start):
            item = ds[start + k]
            if k == 0:
                check(item["task"] == meta["instruction"], f"ep {ep}: task '{item['task']}'")
                check(item["observation.state"].dtype == torch.float32
                      and tuple(item["observation.state"].shape) == (15,), f"ep {ep}: state float32 (15,)")
                h, w, c = ds.meta.features[IMAGE_KEYS["overhead"]]["shape"]
                check(tuple(item[IMAGE_KEYS["overhead"]].shape) == (c, h, w),
                      f"ep {ep}: image tensor ({c}, {h}, {w})")
            st = item["observation.state"].numpy()
            ac = item["action"].numpy()
            states.append(st)
            acts.append(ac)
            for key, v in (("observation.state", st), ("action", ac)):
                z = (v - np.asarray(stats[key]["mean"])) / (np.asarray(stats[key]["std"]) + ZERO_STD_EPS)
                max_norm[key] = max(max_norm[key], float(np.abs(z).max()))
            if k % 10 == 0 or k == stop - start - 1 or (ep == 0 and k == (stop - start) // 2):  # sample frames
                i = int(arr["raw_index"][k])
                raw_imgs, got_imgs = {}, {}
                for view in spec.CAMERAS:
                    key = IMAGE_KEYS[view]
                    got = (item[key].numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
                    raw = read_raw_image(pathlib.Path(src["raw_path"]) / view / f"{i:06d}.png")
                    want = spec.to_policy_image(view, raw)
                    img_err = max(img_err, int(np.abs(got.astype(np.int16) - want.astype(np.int16)).max()))
                    others = [t for t in spec.NET_TRANSFORMS if t != spec.VIEW_TRANSFORMS[view]]
                    if all(not np.array_equal(want, spec.apply_net_transform(t, raw)) for t in others):
                        discriminating[view] += 1
                    raw_imgs[view], got_imgs[view] = raw, got
                if ep == 0 and k == (stop - start) // 2:
                    check_frame = (raw_imgs, got_imgs, f"{out.name} ep 0 frame {k} (raw frame {i})")
        states, acts = np.array(states), np.array(acts)
        check(np.allclose(states, arr["state"], atol=1e-6), f"ep {ep}: state == raw (10 fps)")
        check(np.allclose(acts, arr["action"], atol=1e-7), f"ep {ep}: action == raw x_des differences")
        check(img_err == 0, f"ep {ep}: images == vla_image_spec(raw PNG) per view (max diff {img_err})")
        check(set(np.unique(acts[:, 6]).tolist()) <= {-1.0, 1.0}, f"ep {ep}: gripper in {{-1, +1}}")
        check(np.all(acts[:, 3:6] == 0.0), f"ep {ep}: rotation actions all zero")
        jumps = int((np.abs(np.diff(states[:, 3:6], axis=0)) > 0.5).sum())
        check(jumps == 0, f"ep {ep}: orientation continuous (jumps {jumps})")
        exports[src["raw_episode_id"]] = {"action": acts, "state": states, "raw_path": src["raw_path"]}
    for view, n in discriminating.items():
        check(n > 0, f"{view}: {n} sampled frame(s) where {spec.VIEW_TRANSFORMS[view]} differs from every other "
                     f"net transform (the image check can detect a wrong transform)")
    if check_frame is not None:
        path = pathlib.Path(check_image) if check_image else out.parent / f"{out.name}_verify_raw_vs_policy.png"
        write_check_image(path, *check_frame)
        print(f"[verify] raw vs policy input image: {path}")
    for key, v in max_norm.items():
        check(np.isfinite(v), f"{key} normalized values finite (max |z| {v:.1f})")
    # LeRobot's float32 mean-of-squares variance must agree with float64:
    # catches cancellation in any dimension, not just the orientation
    for key, field in (("observation.state", "state"), ("action", "action")):
        allv = np.concatenate([e[field] for e in exports.values()]).astype(np.float64)
        true_std = allv.std(axis=0)
        got = np.asarray(stats[key]["std"], dtype=np.float64)
        bad = [n for n, t, g in zip(ds.meta.features[key]["names"], true_std, got)
               if abs(t - g) > max(1e-6, 1e-2 * t)]
        check(not bad, f"{key} stats std == float64 std (mismatch: {bad})")

    if export_dir:
        export_dir = pathlib.Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        for eid, e in exports.items():
            np.savez(export_dir / f"ep_{eid:06d}_lerobot.npz", action=e["action"],
                     state=e["state"], fps=FPS, raw_path=e["raw_path"])
        print(f"[verify] exported actions for the 10 fps replay check to {export_dir}")
    print("[verify] " + ("PASS" if ok else "FAIL"))
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="raw teleop episodes -> LeRobot v3.0 dataset")
    ap.add_argument("episodes", nargs="*", help="raw episode directories")
    ap.add_argument("--raw-dir", help="convert every saved episode under this raw root")
    ap.add_argument("--manifest", help="convert the episodes listed in a manifest "
                                       "(03_収録\\manifests\\<name>.json); not with --raw-dir")
    ap.add_argument("--raw-root", help="with --manifest: where to look the episode ids up "
                                       "(default: the production raw folder)")
    ap.add_argument("--labels-root", help="with --manifest: where the labels are "
                                          "(default: the labels folder next to the manifests)")
    ap.add_argument("--out", help="output dataset directory (must not exist)")
    ap.add_argument("--name", help="dataset name (repo_id local/NAME)")
    ap.add_argument("--verify", help="verify an existing converted dataset directory")
    ap.add_argument("--export-actions", help="with --verify: write per-episode action npz here")
    ap.add_argument("--check-image", help="with --verify: raw-vs-policy-input PNG path "
                                          "(default: next to the dataset, <name>_verify_raw_vs_policy.png)")
    args = ap.parse_args(argv)
    if args.verify:
        return 0 if verify(pathlib.Path(args.verify), args.export_actions, args.check_image) else 1
    manifest = None
    if args.manifest:
        if args.raw_dir or args.episodes:
            ap.error("--manifest names the episodes; do not also pass --raw-dir or episode paths")
        episodes, loaded = episodes_from_manifest(args.manifest, args.raw_root, args.labels_root)
        manifest = manifest_record(args.manifest, loaded)
        print(f"[convert] manifest {manifest['name']} ({manifest['n']} episodes, "
              f"base_commit {manifest['base_commit']})")
    else:
        episodes = [pathlib.Path(p) for p in args.episodes]
        if args.raw_dir:
            episodes += raw_episodes(pathlib.Path(args.raw_dir))
    if not episodes or not args.out or not args.name:
        ap.error("need --out, --name and episodes (paths, --raw-dir or --manifest)")
    convert(episodes, pathlib.Path(args.out), args.name, manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
