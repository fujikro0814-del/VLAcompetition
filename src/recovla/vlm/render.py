"""VLM の試験台の描き直し（評価の側。記録だけを読み、凍結のモジュールは import して使うだけ）。

    r = SceneRenderer()
    st = r.state_from_record(z, frame_info)                 # 腕・指・立方体（labels.resolve_time の結果から）
    img = r.render(st, "overhead256", trial_values, seed)   # (H, W, 3) uint8

視点（VIEWS。結果を見る前に固定、手順書 4 節）:
  overhead256  評価の観測と同じ俯瞰の色の画像。場面の overhead カメラを 520x520・fovy 45° で描き、センサの模型 v1 の色の処理
               （harness.sensors.color_model: 試行の露出・白の釣り合いは記録の sensor_trial_values、読み出し・ショットの雑音は
               乱数 default_rng([種, 57, こま]) で、面積平均で 256x256 に縮める）を通す。完了の判定の主
  overhead512  同じカメラ・同じ画角を 512x512 で描いたもの（雑音・露出の処理なし）。完了の判定の副、失敗の分類の副
  presentation 発表用カメラ（場面の presentation、configs の presentation_camera と同じ。63_demo_v2.py と同じカメラ）を
               1280x720 で描き、面積平均で 768x432 に縮める。失敗の分類の主
腕: labels.resolve_time が推論の t_obs に寄せたときは state[8:15]（推論時の測った関節角）。寄せられないときは、記録のこまの手の
  位置・向き（ee_pos・ee_quat。手の体の xpos・xquat）へ減衰つき最小二乗の IK で合わせる（種は最も近い推論の関節角）。
指: 記録のこまの fingers。立方体: 同じこまの cube_pos・cube_quat。速さは 0（描くだけで物理は進めない）。
"""
import hashlib
import json
import os
import pathlib
import platform

import cv2
import mujoco
import numpy as np

from recovla.common import config
from recovla.common.seeds import COLORS
from recovla.sim import scene

VIEWS = {
    "overhead256": {"camera": "overhead", "render": (520, 520), "fovy": 45.0, "out": (256, 256), "sensor_color": True},
    "overhead512": {"camera": "overhead", "render": (512, 512), "fovy": 45.0, "out": (512, 512), "sensor_color": False},
    "presentation": {"camera": "presentation", "render": (1280, 720), "fovy": None, "out": (768, 432), "sensor_color": False},
}
NOISE_KEY = 57
HOME_Q = (0.0, -0.684, 0.0, -2.907, 0.0, 2.216, 0.785)   # IK の種が無いときだけ（目標書 8-3・8-5 の home。diag/e7.py の HOME_Q と同じ値）
IK_ITERS = 200
IK_TOL_POS = 1e-4
IK_TOL_ROT = 1e-3
IK_DAMP = 1e-3


def free_phys_gb():
    """空きの物理メモリ [GB]（Windows の GlobalMemoryStatusEx。ほかの OS は None）。"""
    if os.name != "nt":
        return None
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    s = MS()
    s.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.ullAvailPhys / 1024 ** 3


def git_sha(root=None):
    """.git の HEAD を読むだけ（git は呼ばない）。"""
    g = pathlib.Path(root or config.ROOT) / ".git"
    try:
        head = (g / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            ref = head.split(" ", 1)[1]
            p = g / ref
            if p.is_file():
                return p.read_text(encoding="utf-8").strip()
            packed = g / "packed-refs"
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line.endswith(" " + ref):
                    return line.split(" ")[0]
            return None
        return head
    except OSError:
        return None


def environment() -> dict:
    return {"mujoco": mujoco.__version__, "opencv": cv2.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "os": platform.platform(), "git_sha": git_sha()}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class SceneRenderer:
    def __init__(self, views=tuple(VIEWS)):
        self.model = scene.build_model("3cube")
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.arm_qadr = np.array([m.joint(f"joint{i}").qposadr[0] for i in range(1, 8)])
        self.arm_vadr = np.array([m.joint(f"joint{i}").dofadr[0] for i in range(1, 8)])
        self.finger_qadr = np.array([m.joint(n).qposadr[0] for n in ("finger_joint1", "finger_joint2")])
        self.hand_id = m.body("hand").id
        self.cube_adr = {c: scene.cube_qpos_adr(m, c) for c in COLORS}
        self.views = tuple(views)
        self.renderers = {}
        sc = config.load("sensor_v1")["sensor"]
        self.color_noise = sc["color_noise"]
        self.home_q = np.array(HOME_Q, float)

    def close(self) -> None:
        from recovla.sim import render as R
        for r in self.renderers.values():
            R.close_renderer(r)
        self.renderers = {}

    def _renderer(self, w: int, h: int):
        if (w, h) not in self.renderers:
            self.renderers[(w, h)] = mujoco.Renderer(self.model, h, w)
        return self.renderers[(w, h)]

    # ---------------------------------------------------------- 状態
    def set_state(self, joints, fingers, cube_pos, cube_quat) -> None:
        d = self.data
        d.qvel[:] = 0.0
        d.qpos[self.arm_qadr] = np.asarray(joints, float)
        d.qpos[self.finger_qadr] = np.asarray(fingers, float)
        for k, c in enumerate(COLORS):
            qa, _ = self.cube_adr[c]
            d.qpos[qa:qa + 3] = np.asarray(cube_pos[k], float)
            q = np.asarray(cube_quat[k], float)
            d.qpos[qa + 3:qa + 7] = q / np.linalg.norm(q)
        mujoco.mj_forward(self.model, d)

    def hand_pose(self):
        return self.data.xpos[self.hand_id].copy(), self.data.xquat[self.hand_id].copy()

    def ik(self, pos, quat, seed=None) -> dict:
        """手の体の位置・向きへ関節を合わせる（減衰つき最小二乗）。指・立方体は今の data のまま。"""
        m, d = self.model, self.data
        q = np.asarray(seed if seed is not None else d.qpos[self.arm_qadr], float).copy()
        lo = m.jnt_range[[m.joint(f"joint{i}").id for i in range(1, 8)], 0]
        hi = m.jnt_range[[m.joint(f"joint{i}").id for i in range(1, 8)], 1]
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        target_q = np.asarray(quat, float) / np.linalg.norm(quat)
        err_p = err_r = None
        for it in range(IK_ITERS):
            d.qpos[self.arm_qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            ep = np.asarray(pos, float) - d.xpos[self.hand_id]
            er_b = np.zeros(3)
            mujoco.mju_subQuat(er_b, target_q, d.xquat[self.hand_id])        # 手の体の座標での回転のずれ
            er = d.xmat[self.hand_id].reshape(3, 3) @ er_b                     # 世界の座標へ（jacr は世界の座標）
            err_p, err_r = float(np.linalg.norm(ep)), float(np.linalg.norm(er))
            if err_p < IK_TOL_POS and err_r < IK_TOL_ROT:
                break
            mujoco.mj_jacBody(m, d, jp, jr, self.hand_id)
            J = np.vstack([jp[:, self.arm_vadr], jr[:, self.arm_vadr]])
            e = np.r_[ep, er]
            dq = J.T @ np.linalg.solve(J @ J.T + IK_DAMP * np.eye(6), e)
            q = np.clip(q + dq, lo, hi)
        d.qpos[self.arm_qadr] = q
        mujoco.mj_forward(m, d)
        return {"joints": q.tolist(), "ik_err_pos_m": err_p, "ik_err_rot_rad": err_r, "ik_iters": it + 1}

    def state_from_record(self, z: dict, fr: dict) -> dict:
        """labels.resolve_time の結果 fr と記録の配列 z から状態を置く。返り値は描いた状態の記録（照合用）。"""
        f = int(fr["frame"])
        fingers = np.asarray(z["fingers"], float)[f]
        cp, cq = np.asarray(z["cube_pos"], float)[f], np.asarray(z["cube_quat"], float)[f]
        out = {"frame": f, "t_render": fr["t_render"], "joints_source": fr["joints_source"]}
        if fr["joints_source"] == "inference":
            self.set_state(fr["joints"], fingers, cp, cq)
            hp, _ = self.hand_pose()
            out["fk_vs_record_m"] = float(np.linalg.norm(hp - np.asarray(z["ee_pos"], float)[f]))
            out["joints"] = list(fr["joints"])
        else:
            seed = fr.get("ik_seed")
            self.set_state(seed if seed is not None else self.home_q, fingers, cp, cq)
            r = self.ik(np.asarray(z["ee_pos"], float)[f], np.asarray(z["ee_quat"], float)[f], seed)
            out.update(r)
            out["fk_vs_record_m"] = r["ik_err_pos_m"]
        return out

    # ---------------------------------------------------------- 描画
    def render(self, view: str, trial_values: dict = None, seed: int = 0, frame: int = 0) -> np.ndarray:
        v = VIEWS[view]
        m = self.model
        cid = m.camera(v["camera"]).id
        w, h = v["render"]
        fovy0 = float(m.cam_fovy[cid])
        try:
            if v["fovy"] is not None:
                m.cam_fovy[cid] = v["fovy"]
            r = self._renderer(w, h)
            r.disable_depth_rendering()
            r.update_scene(self.data, camera=cid)
            rgb = r.render().copy()
        finally:
            m.cam_fovy[cid] = fovy0
        if v["sensor_color"]:
            from recovla.harness.sensors import color_model
            tv = trial_values or {}
            gain = float((tv.get("gain") or {}).get("overhead", 1.0))
            wb = np.asarray((tv.get("white_balance") or {}).get("overhead", [1.0, 1.0, 1.0]), float)
            rng = np.random.default_rng([int(seed), NOISE_KEY, int(frame)])
            return color_model(rgb, rng, gain, wb, self.color_noise, None, int(v["out"][0]))
        if (w, h) != tuple(v["out"]):
            rgb = cv2.resize(rgb, tuple(v["out"]), interpolation=cv2.INTER_AREA)
        return rgb


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG に書けない")
    return buf.tobytes()


def render_items(doc: dict, view: str, out_dir, root=None, limit: int = None, item_ids=None, log=print) -> dict:
    """項目の並び（labels.build_* の出力）のうち除外でないものを描き、PNG と目録（SHA-256・描いた状態）を書く。"""
    from recovla.vlm import labels as L
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = [x for x in L.included(doc["items"]) if item_ids is None or x["item_id"] in item_ids]
    if limit is not None:
        items = items[:limit]
    root = pathlib.Path(root) if root else L.v2eval_root()
    r = SceneRenderer()
    cache = {}
    manifest = {"view": view, "view_spec": VIEWS[view], "task": doc["task"], "env": environment(), "items": {}}
    try:
        for n, it in enumerate(items):
            key = (it["source"], it["index"])
            if key not in cache:
                cache.clear()
                d = root.joinpath(*it["source"].split("/"))
                cache[key] = L.load_task(d, it["index"]) if doc["task"] == "completion" else L.load_single(d, it["index"])
            rec = cache[key]
            files = []
            for k, fr in enumerate(it["frames"]):
                st = r.state_from_record(rec["z"], fr)
                img = r.render(view, rec["meta"].get("sensor_trial_values"), rec["meta"].get("seed") or 0, fr["frame"])
                b = encode_png(img)
                name = f"{it['item_id']}_f{k}.png"
                (out_dir / name).write_bytes(b)
                files.append({"file": name, "sha256": sha256_bytes(b), "shape": list(img.shape), "state": st})
            manifest["items"][it["item_id"]] = files
            if log and (n + 1) % 50 == 0:
                log(f"[render] {view} {n + 1}/{len(items)}")
    finally:
        r.close()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest
