"""Step F の完了条件（手順書 §7）。重い測定は scripts/30_f.py が行い、outputs/f/*.json に結果を書く。
ここはその結果を読んで判定する（無ければ、何を実行すればよいかを示して落ちる）。
完了条件 5（R1・N1 のデータ）は、生成の件数と所要時間の承認の後に足す。"""
import json

import pytest

from recovla.common import config

pytestmark = pytest.mark.needs_outputs
CFG = config.load()
OUT = config.path(CFG["paths"]["outputs"]) / "f"
KINDS = ("A", "B", "C")


def _need(name: str, how: str) -> dict:
    p = OUT / f"{name}.json"
    if not p.is_file():
        pytest.fail(f"{p} がない。先に scripts/30_f.py {how} を実行する")
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.mark.parametrize("kind", KINDS)
def test_cond1_recovery_rate_and_landing(kind):
    """1: 種類ごとに、成功 ÷（注入が効き、着地が正常だった数）≥ 90%、着地が不自然な割合 ≤ 2 割。"""
    t = _need("check", "check-gen と check-eval")["table"][kind]
    k, n, _ = t["recovery_success"]
    assert n > 0 and k / n >= 0.9, t
    k, n, _ = t["landing_invalid_ratio"]
    assert n > 0 and k / n <= float(CFG["inject"]["landing"]["max_invalid_ratio"]), t


def test_cond2_first_frame_all_episodes():
    """2: 保存した全エピソードの最初のこまが、保存を始める時点の条件を満たす（全件を機械的に）。"""
    r = _need("check", "check-gen と check-eval")
    assert r["cond2_episodes"] > 0 and r["cond2_pass"], r["cond2_failures"][:5]


def test_cond3_replay_from_saved_state():
    """3: 保存した状態からの再生が、種類ごとに無作為の 3 本で通る（step はビット一致、window は手先 5 mm 以内）。"""
    r = _need("check", "check-gen と check-eval")
    for kind in KINDS:
        assert sum(1 for row in r["cond3_rows"] if row["episode"].startswith(kind + "_")) >= 3, kind
    assert r["cond3_pass"], [(row["episode"], row["step"]["pass"], row["window"]["pass"]) for row in r["cond3_rows"]]


def test_cond4_videos():
    """4: 種類ごとに 5 本の映像。"""
    r = _need("check", "check-gen と check-eval")
    assert r["cond4_pass"], r["cond4_videos"]
    for v in r["cond4_videos"]:
        assert (config.ROOT / v).is_file(), v
