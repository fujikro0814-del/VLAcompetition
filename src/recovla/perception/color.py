"""色の判定と、目標の位置の手がかり（決裁 0048。Step I の色の判定と同じ部品）。

    calib = overhead_calibration(model)                 # 俯瞰カメラの内部・外部（静的なカメラ。模型から読む）
    thr = Thresholds.from_config()                      # configs の planner.color_detect（学習データの種だけで決めた値）
    det = detect(raw_overhead, "red", thr)              # 画素の数・重心（生の画像の座標）
    cue = TargetCue(calib, thr); cue.reset()
    x, y, visible = cue.update(raw_overhead, "red")     # 机の面（立方体の中心の高さ）へ投影した位置と、見えているかの旗

決まり（0048 の 2）: 手がかりは俯瞰画像（描画そのまま、方策の画像の反転の前）からだけ計算する。シミュレータの真値は
使わない。見えないとき（画素が min_pixels 未満）は最後に見えた値を保ち、旗を 0 にする。一度も見えていなければ
机上の配置の範囲（scene.region）の中心を返す（旗 0）。

目標書 v2（0108）: 判定の関数は recovla.runtime.cue に移した（実行系と同じ関数を使うため）。ここは学習の変換と旧版の
実行系のための入口で、較正は模型の本当のカメラ、机の面は既知の高さを使う（振る舞いは移す前と同じ）。
"""
import mujoco
import numpy as np

from recovla.common import config
from recovla.runtime import cue as _cue
from recovla.runtime.cue import CHANNEL, Calibration, color_mask, color_of_instruction, detect, pixel_to_plane  # noqa: F401
from recovla.sim import frames

_CFG = config.load()
OVERHEAD = "overhead"


class Thresholds(_cue.Thresholds):
    @classmethod
    def from_config(cls, cfg: dict = None) -> "Thresholds":
        c = _cue.Thresholds.from_dict((cfg or _CFG)["planner"]["color_detect"])
        return cls(c.min_value, c.min_margin, c.min_pixels)


def overhead_calibration(model, size: int = None) -> Calibration:
    """俯瞰カメラの較正の値。frames.project と同じ透視投影（流用元 OperatorView.project）。"""
    size = int(size or _CFG["sim"]["image_size"])
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    cam = model.camera(OVERHEAD).id
    f = 0.5 * size / np.tan(np.radians(model.cam_fovy[cam]) / 2.0)
    return Calibration(d.cam_xpos[cam].copy(), d.cam_xmat[cam].reshape(3, 3).copy(), float(f), size, size)


class TargetCue(_cue.TargetCue):
    """旧版の入口: 机の面は既知の高さ（frames.CUBE_REST_Z）、一度も見えていなければ scene.region の中心。"""

    def __init__(self, calib: Calibration, thr, cfg: dict = None):
        r = (cfg or _CFG)["scene"]["region"]
        super().__init__(calib, thr, frames.CUBE_REST_Z, [np.mean(r["x"]), np.mean(r["y"])])
