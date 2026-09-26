"""色の判定と、目標の位置の手がかり（決裁 0048。Step I の色の判定と同じ部品）。

    calib = overhead_calibration(model)                 # 俯瞰カメラの内部・外部（静的なカメラ。模型から読む）
    thr = Thresholds.from_config()                      # configs の planner.color_detect（学習データの種だけで決めた値）
    det = detect(raw_overhead, "red", thr)              # 画素の数・重心（生の画像の座標）
    cue = TargetCue(calib, thr); cue.reset()
    x, y, visible = cue.update(raw_overhead, "red")     # 机の面（立方体の中心の高さ）へ投影した位置と、見えているかの旗

決まり（0048 の 2）: 手がかりは俯瞰画像（描画そのまま、方策の画像の反転の前）からだけ計算する。シミュレータの真値は
使わない。見えないとき（画素が min_pixels 未満）は最後に見えた値を保ち、旗を 0 にする。一度も見えていなければ
机上の配置の範囲（scene.region）の中心を返す（旗 0）。

画素の分類: 色 c の画素 = そのチャンネルが min_value 以上、かつ ほかの 2 チャンネルの大きい方より min_margin 以上大きい。
箱（黄）は赤と緑が両方高いので、どちらの色にも入らない。
"""
import dataclasses

import mujoco
import numpy as np

from recovla.common import config
from recovla.common.seeds import COLORS
from recovla.sim import frames

_CFG = config.load()
CHANNEL = {"red": 0, "green": 1, "blue": 2}
OVERHEAD = "overhead"


@dataclasses.dataclass(frozen=True)
class Calibration:
    pos: np.ndarray          # (3,) カメラの位置（世界）
    rot: np.ndarray          # (3, 3) 列: カメラの x（右）・y（上）・z（後ろ向き）
    f: float                 # 焦点距離 [画素]（fovy は縦の画角）
    width: int
    height: int

    def to_json(self) -> dict:
        return {"pos": [float(v) for v in self.pos], "rot": [[float(v) for v in r] for r in self.rot],
                "f": float(self.f), "width": self.width, "height": self.height}


def overhead_calibration(model, size: int = None) -> Calibration:
    """俯瞰カメラの較正の値。frames.project と同じ透視投影（流用元 OperatorView.project）。"""
    size = int(size or _CFG["sim"]["image_size"])
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    cam = model.camera(OVERHEAD).id
    f = 0.5 * size / np.tan(np.radians(model.cam_fovy[cam]) / 2.0)
    return Calibration(d.cam_xpos[cam].copy(), d.cam_xmat[cam].reshape(3, 3).copy(), float(f), size, size)


def pixel_to_plane(calib: Calibration, u: float, v: float, z: float) -> np.ndarray:
    """画素 (u, v)（生の画像。frames.project の逆）を通る光線と、高さ z の水平面の交点 (x, y)。"""
    d_cam = np.array([(u - calib.width / 2.0) / calib.f, -(v - calib.height / 2.0) / calib.f, -1.0])
    d = calib.rot @ d_cam
    if abs(d[2]) < 1e-12:
        raise ValueError("ray parallel to the plane")
    t = (z - calib.pos[2]) / d[2]
    p = calib.pos + t * d
    return p[:2].copy()


@dataclasses.dataclass(frozen=True)
class Thresholds:
    min_value: int           # その色のチャンネルの下限（0〜255）
    min_margin: int          # ほかの 2 チャンネルの大きい方との差の下限
    min_pixels: int          # これ以上の画素があれば「見えている」

    @classmethod
    def from_config(cls, cfg: dict = None) -> "Thresholds":
        c = (cfg or _CFG)["planner"]["color_detect"]
        if c is None:
            raise ValueError("planner.color_detect is not set (scripts/23_color_fit.py で学習データの種から決める)")
        return cls(int(c["min_value"]), int(c["min_margin"]), int(c["min_pixels"]))

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


def color_mask(img: np.ndarray, color: str, thr: Thresholds) -> np.ndarray:
    """(H, W, 3) uint8 → (H, W) bool。"""
    a = np.asarray(img).astype(np.int16)
    ch = CHANNEL[color]
    others = np.max(np.delete(a, ch, axis=2), axis=2)
    return (a[:, :, ch] >= thr.min_value) & (a[:, :, ch] - others >= thr.min_margin)


def detect(img: np.ndarray, color: str, thr: Thresholds) -> dict:
    """画素の数と重心（画素の中心の座標、生の画像）。"""
    m = color_mask(img, color, thr)
    n = int(m.sum())
    if n == 0:
        return {"pixels": 0, "u": None, "v": None, "visible": False}
    rows, cols = np.nonzero(m)
    return {"pixels": n, "u": float(cols.mean() + 0.5), "v": float(rows.mean() + 0.5),
            "visible": n >= thr.min_pixels}


class TargetCue:
    """推論（またはデータの 1 こま）ごとに update する。見えないときは最後に見えた値を保つ。"""

    def __init__(self, calib: Calibration, thr: Thresholds, cfg: dict = None):
        cfg = cfg or _CFG
        self.calib = calib
        self.thr = thr
        self.plane_z = frames.CUBE_REST_Z
        r = cfg["scene"]["region"]
        self.fallback = np.array([np.mean(r["x"]), np.mean(r["y"])])
        self.reset()

    def reset(self) -> None:
        self.last = None
        self.color = None

    def update(self, raw_overhead: np.ndarray, color: str) -> np.ndarray:
        """→ (3,) [x, y, 見えているか（1 / 0）] float32。色が変わったら（指示の切り替え）最後の値を捨てる。"""
        if color not in CHANNEL:
            raise ValueError(f"color {color!r} not in {COLORS}")
        if color != self.color:
            self.last, self.color = None, color
        det = detect(raw_overhead, color, self.thr)
        if det["visible"]:
            self.last = pixel_to_plane(self.calib, det["u"], det["v"], self.plane_z)
            return np.array([self.last[0], self.last[1], 1.0], dtype=np.float32)
        xy = self.last if self.last is not None else self.fallback
        return np.array([xy[0], xy[1], 0.0], dtype=np.float32)


def color_of_instruction(task: str) -> str:
    """指示文（convert.instruction の書式）から色を取り出す。LLM の手順の色と同じもの。"""
    words = [w for w in str(task).replace(",", " ").split() if w in CHANNEL]
    if len(set(words)) != 1:
        raise ValueError(f"instruction {task!r}: expected exactly one color word")
    return words[0]
