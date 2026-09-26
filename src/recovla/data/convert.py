"""Convert raw episodes (recovla.record.recorder) to a LeRobot v3.0 dataset (流用元 convert_to_lerobot.py).

変えたこと（B_提案書 §2.2）: recorder.DEFAULT_RAW_DIR への依存と、ラベル（03_収録\\labels）を要求する
マニフェストを外した。新しいマニフェスト（§6）は生成の回ができる Step D で足す。episode_arrays と
--verify はそのまま。

    .venv\\Scripts\\python.exe -m recovla.data.convert
        --out DIR --name NAME (--raw-dir RAW | EPISODE_DIR ...)
    ... -m recovla.data.convert --verify DIR [--export-actions NPZ_DIR] [--check-image PNG]

Only reads the raw episodes (they are never modified). Format follows
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

from recovla.common import config
from recovla.data import vla_image_spec as spec
from recovla.data import vla_state
# re-exported for existing callers/tests; the definitions live in vla_state
from recovla.data.vla_state import Q_DOWN, STATE_NAMES, orientation_deviation, quat_mul  # noqa: F401

_CFG = config.load()

# 2: per-view net image transforms from vla_image_spec, recorded in conversion.json (2026-09-17);
# 1: both views flipped left-right, no recorded spec
CONVERTER_VERSION = 2
FPS = int(_CFG["convert"]["fps"])
RAW_HZ = round(1.0 / (_CFG["sim"]["record_every"] * _CFG["sim"]["timestep"]))     # 20
STRIDE = RAW_HZ // FPS
if STRIDE != int(_CFG["sim"]["stride"]):
    raise ValueError(f"configs: record rate {RAW_HZ} Hz / fps {FPS} != sim.stride {_CFG['sim']['stride']}")
IMAGE_KEYS = spec.IMAGE_KEYS
ACTION_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
ZERO_STD_EPS = 1e-8   # LeRobot MEAN_STD: (x - mean) / (std + eps)
# Step D: 3 色の場面のエピソード（recovla.record.episode）から conversion.json の sources に持ち越す旗
FLAG_KEYS = ("name", "kind", "layout_kind", "target", "instruction", "pair_id", "start_pose", "layout_seed", "retry")


def episode_label(meta: dict) -> str:
    """エピソードの呼び名: 3 色の場面は名前（n_20000_red_r0 など）、流用元の記録は ep_000123。"""
    return meta["name"] if meta.get("name") else f"ep_{int(meta['episode_id']):06d}"


def _sort_key(meta: dict):
    return (1, meta["name"]) if meta.get("name") else (0, f"{int(meta['episode_id']):09d}")


# ------------------------------------------------------------- manifest (B_提案書 §6)

def write_manifest(path, name: str, entries: list, rule: str, base_commit=None) -> dict:
    """entries: [{"run": 回のフォルダ（<ROOT> からの相対でも可）, "key": エピソードの名前}]。"""
    comp = {}
    for e in entries:
        k = e["key"].split("_")
        comp_key = k[0] if len(k) < 3 else f"{k[0]}_{k[2]}"
        comp[comp_key] = comp.get(comp_key, 0) + 1
    manifest = {"name": name, "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                "base_commit": base_commit, "rule": rule, "entries": entries, "composition": comp}
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def manifest_episodes(manifest: dict) -> list:
    """マニフェストに載った回と名前だけを読む。1 つでも見つからなければ、変換の前に止める。"""
    paths, missing = [], []
    for e in manifest["entries"]:
        p = config.path(e["run"]) / e["key"]
        (paths if (p / "meta.json").is_file() and (p / "data.npz").is_file() else missing).append(p)
    if missing:
        raise FileNotFoundError(f"manifest {manifest['name']}: {len(missing)} episode(s) missing, e.g. {missing[:3]}")
    return paths


CUE_MODES = (None, "xyv", "xy")     # なし / 手がかり 3（旗を含む、18 次元）/ 旗を外した 2（17 次元、決裁 0050 の 2）


def _cue_mode(target_cue):
    """旧来の真偽値（True は xyv）も受ける。"""
    mode = "xyv" if target_cue is True else (None if target_cue is False else target_cue)
    if mode not in CUE_MODES:
        raise ValueError(f"target_cue {target_cue!r} not in {CUE_MODES}")
    return mode


def state_names(target_cue=None) -> list:
    mode = _cue_mode(target_cue)
    return list(STATE_NAMES) + ([] if mode is None else vla_state.cue_names(mode == "xyv"))


def features(size: int = 256, target_cue=None) -> dict:
    img = {"dtype": "image", "shape": [size, size, 3], "names": ["height", "width", "channel"]}
    names = state_names(target_cue)
    return {
        **{IMAGE_KEYS[view]: dict(img) for view in spec.CAMERAS},
        "observation.state": {"dtype": "float32", "shape": [len(names)], "names": names},
        "action": {"dtype": "float32", "shape": [len(ACTION_NAMES)], "names": ACTION_NAMES},
    }


# ------------------------------------------------------------- target cue (決裁 0048)

def cue_tracker():
    """学習と評価で同じ部品・同じ閾値（recovla.perception.color、configs の planner.color_detect）。"""
    from recovla.perception import color as PC
    from recovla.sim import scene
    return PC.TargetCue(PC.overhead_calibration(scene.build_model("3cube")), PC.Thresholds.from_config())


def cue_record(tracker, mode: str = "xyv") -> dict:
    """conversion.json の target_cue。評価の入口（vla_observation）はこれを見て同じ手がかりを足す。
    names が方策の入力に入る手がかりの次元（xy のときは旗を外し、旗は sources の cue_flag0_frames にだけ残す）。"""
    keep = mode == "xyv"
    return {"version": 1, "mode": mode, "names": vla_state.cue_names(keep), "thresholds": tracker.thr.to_json(),
            "plane_z": float(tracker.plane_z), "fallback_xy": [float(v) for v in tracker.fallback],
            "calibration": tracker.calib.to_json(), "fixed_stats": vla_state.CUE_FIXED_STATS if keep else {},
            "definition": "overhead raw render → pixels of the instructed color (recovla.perception.color) → "
                          "centroid → ray to the plane z = cube rest height → (x, y); flag 1 if >= min_pixels "
                          "pixels, else 0 and the last seen (x, y) is kept (region centre if never seen). "
                          "Recomputed at every 10 fps frame in episode order. No simulator truth."}


def fix_cue_stats(out: pathlib.Path) -> None:
    """旗など、学習データで一定になりうる次元の正規化の値を固定する（vla_state.CUE_FIXED_STATS。xyv のときだけ）。"""
    path = out / "meta" / "stats.json"
    stats = json.loads(path.read_text(encoding="utf-8"))
    names = state_names("xyv")
    for name, fixed in vla_state.CUE_FIXED_STATS.items():
        i = names.index(name)
        for k, v in fixed.items():
            stats["observation.state"][k][i] = v
    path.write_text(json.dumps(stats, indent=4), encoding="utf-8")


# ------------------------------------------------------------- raw side

def load_raw(path: pathlib.Path):
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    with np.load(path / "data.npz") as npz:
        data = {k: npz[k] for k in npz.files}
    return meta, data


def raw_episodes(raw_dir: pathlib.Path) -> list:
    eps = [p for p in raw_dir.glob("*/ep_*")
           if p.is_dir() and not p.name.endswith(".partial") and (p / "meta.json").is_file()]
    return sorted(eps, key=lambda p: _sort_key(json.loads((p / "meta.json").read_text(encoding="utf-8"))))


def episode_arrays(meta: dict, data: dict) -> dict:
    """10 fps state/action arrays + the raw frame indices they come from."""
    if int(meta["record_hz"]) != RAW_HZ:
        raise ValueError(f"episode {episode_label(meta)}: record_hz {meta['record_hz']} != {RAW_HZ}")
    n = int(meta["n_frames"])
    k_frames = (n - 1) // STRIDE
    if k_frames < 1:
        raise ValueError(f"episode {episode_label(meta)}: too short ({n} raw frames)")
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

def convert(episodes: list, out: pathlib.Path, name: str, manifest: dict = None, target_cue=None) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    mode = _cue_mode(target_cue)
    if out.exists():
        raise FileExistsError(f"{out} exists; converted datasets are rebuilt, not appended to")
    first_meta, _ = load_raw(episodes[0])
    ds = LeRobotDataset.create(repo_id=f"local/{name}", fps=FPS, features=features(
        int(first_meta["cameras"]["width"]), mode), root=out, robot_type="panda", use_videos=False)
    tracker = cue_tracker() if mode else None
    sources = []
    try:
        for ep_index, path in enumerate(episodes):
            meta, data = load_raw(path)
            arr = episode_arrays(meta, data)
            flag0 = 0
            if tracker is not None:
                tracker.reset()
            for k, i in enumerate(arr["raw_index"]):
                raw = {view: read_raw_image(path / view / f"{i:06d}.png") for view in spec.CAMERAS}
                images = spec.policy_images(raw)
                state = arr["state"][k]
                if tracker is not None:
                    c = tracker.update(raw["overhead"], meta["target"])
                    flag0 += int(c[2] < 0.5)
                    state = vla_state.with_cue(state, c, keep_flag=mode == "xyv")
                ds.add_frame({
                    **images,
                    "observation.state": state,
                    "action": arr["action"][k],
                    "task": meta["instruction"],
                })
            ds.save_episode()
            sources.append({"episode_index": ep_index, "raw_episode_id": meta.get("episode_id"),
                            "raw_episode": episode_label(meta), "run": path.parent.name,
                            **{k: meta.get(k) for k in FLAG_KEYS},
                            "raw_path": str(path), "frames": int(len(arr["raw_index"])),
                            "raw_frames": int(meta["n_frames"]),
                            "placement_id": meta.get("placement_id"),
                            "operator": meta.get("operator"), "session": meta.get("session"),
                            "success": meta.get("success"), "retakes": meta.get("retakes"),
                            "meta_sha256": sha256(path / "meta.json"),
                            "data_sha256": sha256(path / "data.npz"),
                            **({"cue_flag0_frames": flag0} if tracker is not None else {})})
            print(f"[convert] ep {ep_index}: raw {episode_label(meta)} "
                  f"{meta['n_frames']} raw frames -> {len(arr['raw_index'])} frames")
    finally:
        ds.finalize()
    if mode == "xyv":
        fix_cue_stats(out)
    (out / "meta" / "conversion.json").write_text(json.dumps({
        "converter_version": CONVERTER_VERSION,
        "converted_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "fps": FPS, "raw_hz": RAW_HZ, "stride": STRIDE,
        **spec.spec_record(),
        "images": "PNG from raw -> vla_image_spec net transform per view (image_transforms)",
        "state": state_names(mode), "action": ACTION_NAMES,
        **({"target_cue": cue_record(tracker, mode)} if tracker is not None else {}),
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

    mode = conv["target_cue"].get("mode", "xyv") if "target_cue" in conv else None
    names_state = state_names(mode)
    check(ds.meta.features["observation.state"]["names"] == names_state and conv["state"] == names_state,
          f"state names == {len(names_state)} (target_cue {mode})")
    tracker = cue_tracker() if mode else None
    if tracker is not None:
        check(conv["target_cue"]["thresholds"] == tracker.thr.to_json(),
              f"target_cue thresholds recorded == configs planner.color_detect {tracker.thr.to_json()}")
    fixed = vla_state.CUE_FIXED_STATS if mode == "xyv" else {}

    # stats: finite, and normalization never divides by zero
    stats = ds.meta.stats
    for n, want in fixed.items():
        i = names_state.index(n)
        got = {k: float(np.asarray(stats["observation.state"][k])[i]) for k in want}
        check(got == want, f"observation.state {n} stats fixed to {want} (got {got})")
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
        if tracker is not None:            # 手がかりを生の俯瞰画像から計算し直して、期待する 18 次元を作る
            tracker.reset()
            cues = np.array([tracker.update(read_raw_image(pathlib.Path(src["raw_path"]) / "overhead" / f"{int(i):06d}.png"),
                                            meta["target"]) for i in arr["raw_index"]])
            arr["state"] = vla_state.with_cue(arr["state"], cues, keep_flag=mode == "xyv")
            check(int((cues[:, 2] < 0.5).sum()) == src.get("cue_flag0_frames", int((cues[:, 2] < 0.5).sum())),
                  f"ep {src['episode_index']}: cue flag-0 frames recorded {src.get('cue_flag0_frames')}")
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
                if meta.get("target"):        # 3 色の場面: task の色がメタデータの目標の色と一致する
                    want = _CFG["convert"]["instruction"].format(color=meta["target"])
                    check(item["task"] == want and src.get("target") == meta["target"],
                          f"ep {ep}: task color == meta target '{meta['target']}'")
                check(item["observation.state"].dtype == torch.float32
                      and tuple(item["observation.state"].shape) == (len(names_state),),
                      f"ep {ep}: state float32 ({len(names_state)},)")
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
        exports[src.get("raw_episode") or episode_label(meta)] = {"action": acts, "state": states, "raw_path": src["raw_path"]}
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
               if n not in fixed and abs(t - g) > max(1e-6, 1e-2 * t)]   # 固定した次元は上で確かめた
        check(not bad, f"{key} stats std == float64 std (mismatch: {bad})")

    if export_dir:
        export_dir = pathlib.Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        for eid, e in exports.items():
            np.savez(export_dir / f"{eid}_lerobot.npz", action=e["action"],
                     state=e["state"], fps=FPS, raw_path=e["raw_path"])
        print(f"[verify] exported actions for the 10 fps replay check to {export_dir}")
    print("[verify] " + ("PASS" if ok else "FAIL"))
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="raw episodes -> LeRobot v3.0 dataset")
    ap.add_argument("episodes", nargs="*", help="raw episode directories")
    ap.add_argument("--raw-dir", help="convert every saved episode under this raw root")
    ap.add_argument("--manifest", help="convert exactly the episodes listed in this manifest (B_提案書 §6)")
    ap.add_argument("--out", help="output dataset directory (must not exist)")
    ap.add_argument("--name", help="dataset name (repo_id local/NAME)")
    ap.add_argument("--verify", help="verify an existing converted dataset directory")
    ap.add_argument("--export-actions", help="with --verify: write per-episode action npz here")
    ap.add_argument("--check-image", help="with --verify: raw-vs-policy-input PNG path "
                                          "(default: next to the dataset, <name>_verify_raw_vs_policy.png)")
    ap.add_argument("--target-cue", action="store_true",
                    help="add the target position cue to observation.state (15 -> 18; board 0048)")
    ap.add_argument("--cue-without-flag", action="store_true",
                    help="with --target-cue: leave the visibility flag out of the input (15 -> 17; board 0050-2)")
    args = ap.parse_args(argv)
    if args.verify:
        return 0 if verify(pathlib.Path(args.verify), args.export_actions, args.check_image) else 1
    episodes = [pathlib.Path(p) for p in args.episodes]
    manifest = None
    if args.manifest:
        manifest = json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))
        episodes += manifest_episodes(manifest)
    if args.raw_dir:
        episodes += raw_episodes(pathlib.Path(args.raw_dir))
    if not episodes or not args.out or not args.name:
        ap.error("need --out, --name and episodes (paths, --manifest or --raw-dir)")
    mode = ("xy" if args.cue_without_flag else "xyv") if args.target_cue else None
    convert(episodes, pathlib.Path(args.out), args.name, manifest, target_cue=mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
