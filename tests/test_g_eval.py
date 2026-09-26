"""Step G の完了条件（手順書 §8）のうち、方策を GPU で動かすもの（1・2・4・6）。重い測定は scripts/42_g_checks.py が
行い、outputs/g/*.json に結果を書く。ここはその結果を読んで判定する（無ければ、何を実行すればよいかを示して落ちる）。
3 は tests/eval/test_induce.py、5 は tests/eval/test_metrics.py（合成軌跡）。7 は安全フィルタを作った場合だけ。"""
import json
import warnings

import pytest

from recovla.common import config

pytestmark = pytest.mark.needs_outputs
CFG = config.load()
OUT = config.path(CFG["paths"]["outputs"]) / "g"


def _need(name: str, how: str) -> dict:
    p = OUT / f"{name}.json"
    if not p.is_file():
        pytest.fail(f"{p} がない。先に scripts/42_g_checks.py {how} を実行する")
    return json.loads(p.read_text(encoding="utf-8"))


def test_cond1_sync_s50_matches_old_evaluator():
    """1: sync・s=50 で、旧評価器（K1 の閉ループ）と同じ種の 20 回の成否が一致する。不一致があれば原因を示す。
    旧評価器は雑音を毎手引く（推論 i は 50·i 番目）ので、新しい実行器に同じ雑音の列を与えた回し直し（cond1_cause）で
    20 回の成否が旧と一致すれば、不一致の原因は雑音の列の違いと示せたとする。"""
    r = _need("cond1", "cond1")
    assert r["n"] == 20
    if r["pass"]:
        return
    c = _need("cond1_cause", "cond1-cause")
    assert c["n"] == 20 and c["success_matches_old"] == 20, c
    # 旧の雑音でのビット一致は描画の揺れ（1 画素・1 段）で崩れることがある。大半は一致していること
    assert c["bit_equal_trajectories"] >= 15, c


def test_cond2_naive_d0_equals_sync():
    """2: naive で d=0 にすると、同じ s の sync と一致する。成否は 20 回すべて一致。行動の列のビット一致は、描画の
    揺れを除いた回し直し（同じ物理の状態には同じ画像）で 20 回すべて。"""
    r = _need("cond2", "cond2")
    assert r["n"] == 20 and r["success_matches"] == 20, r
    m = _need("cond2_memo", "cond2 --tag _memo --memo-render")
    assert m["n"] == 20 and m["success_matches"] == 20 and m["action_equal"] == 20, m
    assert m["memo_render"]["hits"] > 0


@pytest.mark.parametrize("kind", ["P1", "P2", "P3"])
def test_cond4_induction_with_policy(kind):
    """4: 誘発が条件どおりに発生する（各 20 回）。発生しなかった試行と成立しなかった試行は記録に残る。
    成立の後は方策に戻っている（誘発の上書きが続かない）。成立率が 80% 未満なら報告（警告）。"""
    r = _need("cond4", "cond4")["results"][kind]
    assert r["n"] == 20 and r["all_recorded"]
    assert all(("fired" in x and "established" in x and "reason" in x) for x in r["rows"])
    assert all((x["established"] or x["reason"]) for x in r["rows"]), "成立しなかった試行に理由がない"
    assert not any(x["induce_active_after_established"] for x in r["rows"])
    if r["below_min_rate"]:
        warnings.warn(f"{kind} の成立率 {r['establish_rate']:.2f} が {CFG['eval']['induction_min_rate']} 未満（報告する）")


def test_cond6_rtc_is_effective():
    """6: 同じ入力で、rtc の塊の最初の d 手と前の塊の差が、naive の同じ量より十分小さい（決まりは 42_g_checks.py の
    冒頭: 比の中央値 0.5 以下、rtc の方が小さい推論が 9 割以上）。途中経過の図が 1 枚ある。"""
    r = _need("cond6", "cond6 --d D")
    assert r["d"] == CFG["runtime"]["delay_steps"], "設定の d と違う d で回している"
    assert r["inferences"] >= 50 and r["pass"], {k: r[k] for k in ("ratio_median", "rtc_smaller_share")}
    assert (config.ROOT / r["figure"]).is_file()
