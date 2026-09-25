"""名前つきの独立した乱数列（B_提案書 §9、docs/interfaces/README.md。型は掲示板 0022 の案を 0025 で採ったもの）。

1 つの試行の種から、用途ごとに別の乱数列を作る。番号（STREAM_ID）は固定し、変えない。

    stream(seed, "layout")                    配置     SeedSequence(seed, spawn_key=(0,))
    stream(seed, "induce")                    誘発     (1,)
    torch_seed(seed_sequence(seed, "noise", i))   k 回目（i 回目）の推論の方策の雑音   (2, i)
    script_rng(layout_seed, color, retry)     生成の台本   (3, 色の添字, 作り直しの回数)
    inject_rng(layout_seed, color, retry)     生成の注入   (4, 色の添字, 作り直しの回数)
"""
import numpy as np

STREAM_ID = {"layout": 0, "induce": 1, "noise": 2, "script": 3, "inject": 4, "order": 5}   # 変えない
COLORS = ("red", "green", "blue")


def seed_sequence(seed: int, name: str, *sub: int) -> np.random.SeedSequence:
    """SeedSequence(seed, spawn_key=(STREAM_ID[name], *sub))。noise の推論 i は sub=(i,)。"""
    return np.random.SeedSequence(int(seed), spawn_key=(STREAM_ID[name],) + tuple(int(s) for s in sub))


def generator(ss: np.random.SeedSequence) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(ss))


def stream(seed: int, name: str) -> np.random.Generator:
    """np.random.Generator(np.random.PCG64(seed_sequence(seed, name)))。"""
    return generator(seed_sequence(seed, name))


def streams(seed: int) -> dict:
    """B_提案書 §9 の streams(seed)。STREAM_ID の全部の名前について stream(seed, name)。"""
    return {name: stream(seed, name) for name in STREAM_ID}


def torch_seed(ss: np.random.SeedSequence) -> int:
    """torch.Generator.manual_seed に入れる 64 ビットの種（掲示板 0025 で確定）。"""
    return int(ss.generate_state(1, dtype=np.uint64)[0])


def script_rng(layout_seed: int, color: str, retry: int) -> np.random.Generator:
    """生成の台本の乱数列。記録には (layout_seed, color, retry) を残す（B_提案書 §9、掲示板 0025）。"""
    return generator(seed_sequence(layout_seed, "script", COLORS.index(color), retry))


def inject_rng(layout_seed: int, color: str, retry: int) -> np.random.Generator:
    """生成の注入の乱数列（Step F）。"""
    return generator(seed_sequence(layout_seed, "inject", COLORS.index(color), retry))
