"""実行系が読むロボットの模型（目標書 v2 の G1 の「ロボットのモデル」）。

読んでよいのは assets/mjcf/robot_only.xml（panda.xml と制御器の目標の印だけ）。場面の XML（机・立方体・箱・俯瞰カメラ）を
読まないことを、G1 の静的な検査が確かめる（0107 の 2-3 の (i)）。実行系の中で mujoco の模型を読むのはこのファイルだけ。
"""
import pathlib

import mujoco

ALLOWED = "robot_only.xml"


def load(robot_xml: str) -> mujoco.MjModel:
    p = pathlib.Path(robot_xml)
    if p.name != ALLOWED:
        raise ValueError(f"実行系が読めるロボットの模型は {ALLOWED} だけ: {p}")
    return mujoco.MjModel.from_xml_path(str(p))
