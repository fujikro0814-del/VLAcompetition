"""誘発 P1〜P3（Step G の 3）と、run_trial への組み込み。方策の代わりに台本（真値を読む Expert を 10 fps の行動に
直したもの）を使い、描画なし・GPU なしで回す。種ごとの結果はこの PC で選んだので windows の目印。"""
import numpy as np
import pytest

from recovla.common import config, seeds
from recovla.eval import induce as I
from recovla.eval import scene_trial as T
from recovla.expert import script as S
from recovla.sim import scene

CFG = config.load()


class ScriptPolicy:
    """台本を 10 fps の行動にする（評価器の指令の入口の確認用）。"""

    def __init__(self, rig, target, seed):
        self.rig, self.target = rig, target
        self.ex = S.Expert(S.sample_params(seeds.script_rng(seed, target, 0)), T.ACTION_DT)

    def __call__(self, k, frame, raw, task):
        tr = self.rig.truth(self.target)
        cmd = self.ex.act(tr)
        closed = bool(self.rig.controller.gripper_closed) != bool(cmd.press)
        a = np.zeros(7)
        a[:3] = cmd.vel * T.ACTION_DT
        a[6] = 1.0 if closed else -1.0
        return a


@pytest.fixture(scope="module")
def rig():
    from recovla.sim.rig import SimRig
    r = SimRig(render=False)
    yield r
    r.close()


def run(rig, kind, seed):
    lay = scene.sample_layout(seed, "empty", start="home")
    target = lay.table_colors[seed % len(lay.table_colors)]
    ind = I.Inducer(kind, seed, lay, target, rig) if kind else None
    meta, arr, _ = T.run_trial(rig, lay, target, ScriptPolicy(rig, target, seed),
                               {"trial": 0, "seed": seed, "experiment": "test", "condition": kind or "none"},
                               time_limit_s=30.0, render=False, inducer=ind)
    return meta, arr


def test_params_drawn_up_front_and_deterministic():
    lay = scene.sample_layout(53000, "empty", start="home")
    t = lay.table_colors[0]
    a = I.Inducer("P3", 53000, lay, t, rig=None)
    b = I.Inducer("P3", 53000, lay, t, rig=None)
    assert a.params == b.params
    p1 = I.Inducer("P1", 53000, lay, t, rig=None).params
    lo, hi = CFG["eval"]["P1"]["lateral_offset_m"]
    assert lo <= abs(p1["offset_m"]) <= hi
    assert 0.0 <= I.Inducer("P2", 53000, lay, t, rig=None).params["u"] < 1.0


def test_no_inducer_keeps_old_behaviour(rig):
    meta, arr = run(rig, None, 53001)
    assert meta["success"] and meta["induce"]["kind"] is None and not arr["induce_active"].any()


@pytest.mark.windows
@pytest.mark.parametrize("kind", I.KINDS)
def test_inducer_fires_and_is_recorded(rig, kind):
    """発動し、成立の判定まで進み、記録（meta.induce と induce_active）に残る。台本は立て直すので成功もする。"""
    got = []
    for seed in range(53010, 53014):
        meta, arr = run(rig, kind, seed)
        ind = meta["induce"]
        assert ind["kind"] == kind and ind["fired"], ind
        assert arr["induce_active"].any()
        got.append(ind["established"])
    assert sum(got) >= 3, got                  # 4 回中 3 回以上成立（台本の相手なので、ほぼ成立する）
