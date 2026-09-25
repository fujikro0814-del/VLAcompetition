"""3 色の場面の配置（手順書 Step D の 1、B_提案書 §7）。

    layout = sample_layout(seed)                  # 種から配置を決める（配置の乱数列だけを使う）
    apply_layout(model, data, layout)             # 立方体 3 個を置く（静止させるのは SimRig.reset）

配置の決め方（すべて配置の乱数列 streams(seed)["layout"] から、この順に引く）:
  1. 配置の種類（kind を渡さないとき scene.layout_kinds の割合で。渡しても 1 つ引いて捨て、以後の列を揃える）
  2. 開始姿勢（scene.start_at_retreat_prob で retreat、ほかは home）
  3. 先客の色と置き場所（先客 1 個・2 個のとき。色の並びと置き場所の並びを無作為に並べ替えて先頭から使う）
  4. 机上の立方体（残りの色、色の並びの順に）: 範囲 scene.region の一様、向き ±scene.yaw_range_deg の一様。
     すでに置いた机上の立方体との中心間距離が scene.min_center_dist 未満なら、その 1 個を引き直す
"""
import dataclasses

import mujoco
import numpy as np

from recovla.common import config, seeds
from recovla.common.seeds import COLORS
from recovla.sim import frames

_CFG = config.load()
_SCENE = _CFG["scene"]
KINDS = ("empty", "prefilled_1", "prefilled_2")
N_PREFILLED = {"empty": 0, "prefilled_1": 1, "prefilled_2": 2}
STARTS = ("home", "retreat")


@dataclasses.dataclass(frozen=True)
class Layout:
    seed: int
    kind: str                         # empty | prefilled_1 | prefilled_2
    start: str                        # home | retreat
    cubes: dict                       # 机上の色 -> (x, y, yaw)
    prefilled: dict                   # 箱の中の色 -> 置き場所の番号（scene.box_slots の添字）
    tries: int = 1                    # 机上の立方体を引いた回数（棄却を含む）

    @property
    def table_colors(self) -> tuple:
        return tuple(c for c in COLORS if c in self.cubes)

    def to_json(self) -> dict:
        return {"seed": self.seed, "kind": self.kind, "start": self.start,
                "cubes": {c: [float(v) for v in self.cubes[c]] for c in self.table_colors},
                "prefilled": {c: int(s) for c, s in self.prefilled.items()}, "tries": self.tries}

    @classmethod
    def from_json(cls, d: dict) -> "Layout":
        return cls(int(d["seed"]), d["kind"], d["start"], {c: tuple(v) for c, v in d["cubes"].items()},
                   {c: int(s) for c, s in d["prefilled"].items()}, int(d.get("tries", 1)))


def sample_layout(seed: int, kind: str = None, cfg: dict = None) -> Layout:
    sc = (cfg or _CFG)["scene"]
    rng = seeds.stream(seed, "layout")
    probs = np.array([float(sc["layout_kinds"][k]) for k in KINDS])
    u = rng.random()
    drawn = KINDS[int(np.searchsorted(np.cumsum(probs / probs.sum()), u, side="right"))]
    kind = kind or drawn
    if kind not in KINDS:
        raise ValueError(f"kind {kind!r} not in {KINDS}")
    start = "retreat" if rng.random() < float(sc["start_at_retreat_prob"]) else "home"
    color_order = [COLORS[i] for i in rng.permutation(len(COLORS))]
    slot_order = [int(i) for i in rng.permutation(len(sc["box_slots"]))]
    n = N_PREFILLED[kind]
    prefilled = {color_order[i]: slot_order[i] for i in range(n)}
    rx, ry = sc["region"]["x"], sc["region"]["y"]
    half_yaw = np.radians(float(sc["yaw_range_deg"]))
    min_d = float(sc["min_center_dist"])
    cubes, tries = {}, 0
    for color in COLORS:
        if color in prefilled:
            continue
        while True:
            tries += 1
            if tries > int(sc["layout_max_tries"]):
                raise RuntimeError(f"layout seed {seed}: no placement after {tries - 1} draws")
            x, y = float(rng.uniform(*rx)), float(rng.uniform(*ry))
            yaw = float(rng.uniform(-half_yaw, half_yaw))
            if all(np.hypot(x - p[0], y - p[1]) >= min_d for p in cubes.values()):
                cubes[color] = (x, y, yaw)
                break
    return Layout(int(seed), kind, start, cubes, prefilled, tries)


def cube_qpos_adr(model, color: str) -> tuple:
    jnt = model.body(frames.cube_body(color)).jntadr[0]
    return int(model.jnt_qposadr[jnt]), int(model.jnt_dofadr[jnt])


def apply_layout(model, data, layout: Layout) -> None:
    """立方体 3 個の位置・姿勢を書き、速さを 0 にする。先客は置き場所の中心、箱の底からわずかに浮かせる
    （着地させて静止させるのは SimRig.reset の役目）。mj_forward は呼ぶ側で行う。"""
    box = frames.box_pos(model)
    pre_yaw = np.radians(float(_SCENE["prefilled_yaw_deg"]))
    lift = 1e-4                                            # 底の上面との初期の重なりを避ける
    for color in COLORS:
        qa, va = cube_qpos_adr(model, color)
        if color in layout.cubes:
            x, y, yaw = layout.cubes[color]
            pos = (x, y, frames.CUBE_REST_Z)
        elif color in layout.prefilled:
            sx, sy = frames.slot_xy(box, layout.prefilled[color])
            pos, yaw = (sx, sy, frames.BOX_FLOOR_Z + frames.CUBE_HALF + lift), pre_yaw
        else:
            raise ValueError(f"layout {layout.seed}: color {color} neither on the table nor in the box")
        data.qpos[qa:qa + 3] = pos
        data.qpos[qa + 3:qa + 7] = frames.yaw_quat(yaw)
        data.qvel[va:va + 6] = 0.0


def check_scene_colors(model) -> None:
    """場面の XML の立方体の色が設定（scene.colors）と同じであること。"""
    for color in COLORS:
        got = [float(v) for v in model.geom_rgba[model.geom(frames.cube_geom(color)).id]]
        want = [float(v) for v in _SCENE["colors"][color]]
        if not np.allclose(got, want):
            raise RuntimeError(f"scene color {color}: xml {got} != configs {want}")


def build_model(variant: str = "3cube") -> mujoco.MjModel:
    key = {"3cube": "scene", "g0": "scene_g0"}[variant]
    model = mujoco.MjModel.from_xml_path(str(config.path(_CFG["paths"][key])))
    if variant == "3cube":
        check_scene_colors(model)
    return model
