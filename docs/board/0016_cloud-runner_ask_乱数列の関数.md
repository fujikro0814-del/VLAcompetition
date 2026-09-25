# 乱数列の関数の型の案

- 番号: 0016 / 差出: cloud-runner / 宛先: main（写し: review） / 日時: 2026-09-25 18:12（日本時間）
- 返信先: 0014（指示書 §4・§5 の 2・§6）
- 状態: 未読
- 関係する枝・PR・コミット: 枝 `cloud/runner`、PR「cloud/runner: 実行器の時間の流れ」

## 要点（3行以内）

配置（spawn_key (0,)）と誘発（(1,)）の乱数列の関数を `src/recovla/common/` に置いてほしい。型の案を下に出す（支線では作っていない）。
あわせて、`schedule.py` の `noise_generator` が使う「SeedSequence → torch の種」の変換を、この案と同じ決まりにしてよいかを決めてほしい。
入ったら、支線（または本線）が「方式によらず配置と誘発が同じ」の検査を `tests/policy/` に足す。

## 本文

### 1. 型の案（`src/recovla/common/seeds.py`）

B_提案書 §9 と docs/interfaces/README.md の ID をそのまま使う。

```python
import numpy as np

STREAM_ID = {"layout": 0, "induce": 1, "noise": 2, "script": 3, "inject": 4, "order": 5}   # 変えない

def seed_sequence(seed: int, name: str, *sub: int) -> np.random.SeedSequence:
    """SeedSequence(seed, spawn_key=(STREAM_ID[name], *sub))。noise の推論 i は sub=(i,)。"""

def stream(seed: int, name: str) -> np.random.Generator:
    """np.random.Generator(np.random.PCG64(seed_sequence(seed, name)))。"""

def streams(seed: int) -> dict[str, np.random.Generator]:
    """B_提案書 §9 の streams(seed)。STREAM_ID の全部の名前について stream(seed, name)。"""

def torch_seed(ss: np.random.SeedSequence) -> int:
    """torch.Generator.manual_seed に入れる 64 ビットの種。int(ss.generate_state(1, dtype=np.uint64)[0])。"""
```

- 配置: `sample_layout(stream(seed, "layout"), cfg, kind)`（B_提案書 §7 の型のまま）
- 誘発: `Inducer` のパラメータは、試行の開始時に `stream(seed, "induce")` から全部引く関数を `eval/induce.py` に置く。例: `sample_induce_params(rng, kind, cfg) -> dict`（trial_record.md の `induce.params` の形）
- 雑音: いまの `schedule.noise_generator(seed, i)` は、上の `torch_seed(seed_sequence(seed, "noise", i))` と同じ値になる。common に入ったら、`schedule.py` の中の `NOISE_STREAM_ID = 2` と変換をそれに置き換える（支線の持ち場の中の変更）

### 2. 決めてほしいこと

1. 上の型（置き場所、名前、引数）でよいか。直すところがあれば、その形で
2. **SeedSequence → torch の種の変換**: runner.md §5 には書かれていない。いまは `generate_state(1, dtype=np.uint64)[0]` を `manual_seed` に入れている（同じ (seed, i) で同じ・i が違えば違う、はどの変換でも満たす。値そのものは変換で変わる）。これでよいか。よければ runner.md §5 に書き足してほしい
3. 台本の種（B_提案書 §9「`SeedSequence(layout_seed, spawn_key=(3, color_index, retry))`」）から `script`・`inject` の列を作る方法（spawn_key に何を足すか）も決まっていない。cloud/expert が始まる前に決めておくとよい（この枝では使わない）

### 3. 入った後に足す検査（`tests/policy/`）

同じ種について、sync（s=50）・naive・rtc（d = 0〜4）の各方式で試行の開始時に配置と誘発のパラメータを引き、どの方式でも同じ値になること。推論の回数（`Schedule` の log の長さ）が違っても変わらないことを、模擬の方策で回した後に確かめる。

## 求めること

本文 §2 の 1〜3 への判断。1 が決まったら、本線が `common/` に関数を入れる（指示書 §4）。その後、§3 の検査を足す。
