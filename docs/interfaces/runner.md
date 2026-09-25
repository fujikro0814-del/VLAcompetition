# 方策の実行器の時間の流れ（runner）

版: 1（2026-09-25）
- 本線: `src/recovla/policy/runner.py`（LeRobot の方策を呼ぶ部分、Step G）
- cloud/runner: `src/recovla/policy/schedule.py`（時間の流れだけ。方策は呼ばない）と `tests/policy/`

時間の流れは、方策と物理から切り離した純粋な部品（`Schedule`）にする。支線はこれを、決まった塊を返す模擬の方策で作って検査する。本線は同じ `Schedule` を実物の方策と評価器につなぐ。

## 1. 記号と時間

| 記号 | 中身 | 設定 |
|---|---|---|
| k | 10 Hz の行動の番号（0, 1, 2, …）。行動 k は物理ステップ 50k〜50k+49 に効く | |
| H | 塊の長さ = 50 | 方策の `chunk_size` |
| s | 実行間隔（何手ごとに推論を始めるか） | `runtime.exec_interval`（10。sync の卒研の既定は 50） |
| d | 遅延（推論を始めてから結果が有効になるまでの手数）。0 ≤ d < s、かつ s + d ≤ H | `runtime.delay_steps` |
| E | RTC で前の塊に合わせる範囲（LeRobot の `execution_horizon`） | `runtime.rtc_guidance_horizon`（10 と 40 を比べる） |

- 観測は行動 k の直前（こま 2k）のもの
- 模擬の中では、推論は物理を止めて計算する。そのため、結果が「d 手後に有効になる」ことは記録の上で表す。実時間は別に測る（§4）

## 2. 3 つの方式

以下、塊 i が「有効になる手」を v_i、「有効になったときに実行する塊の中の添字」を o_i とする。

### sync（推論の間は物理を止める）

- 推論 i は k_i = i·s で始め、観測は k_i のもの
- v_i = k_i、o_i = 0。塊の添字 0〜s−1 を実行する
- s = 50 は卒研の既定

### naive（非同期の模擬、つなぎなし）

- 最初の塊（i = 0）は sync と同じ（k_0 = 0 で物理を止めて推論し、v_0 = 0、o_0 = 0）
- i ≥ 1 の推論は k_i = k_{i−1} + s で始め、観測は k_i のもの
  - 結果は v_i = k_i + d から有効で、o_i = d（新しい塊の添字 d から実行する）
  - k_i〜k_i + d − 1 の d 手は、前の塊の続きを実行する
- 1 つの塊から実行する範囲は、塊 0 が添字 0〜s+d−1、i ≥ 1 は添字 d〜s+d−1（ともに次の切り替えの直前まで）
- **d = 0 のとき、sync と同じ行動の列になる**（Step G 完了条件 2）

### rtc

- 時間の流れは naive と同じ
- 推論 i（i ≥ 1）には、LeRobot の `predict_action_chunk(batch, noise=…, inference_delay=d, prev_chunk_left_over=L_i)` を使う
- 前の塊の残り L_i: 前の塊のうち、推論を始めた時点 k_i でまだ実行していない部分
  - 前の塊の k_i での添字は j = o_{i−1} + (k_i − v_{i−1})。L_i = 前の塊[j : H]
  - 空間は**後処理の前**（正規化された、`predict_action_chunk` の出力のまま、`(H, 7)`。lerobot 0.6.1 は詰め物の次元を外して返す。LeRobot 自身の実行器と同じ扱い。掲示板 0024 への 0025 の回答で A を採った）
  - 長さを E に揃える。長ければ先頭から E 行に切り詰め、短ければ 0 で詰める。同梱の `lerobot/rollout/inference/rtc.py` の `_normalize_prev_actions_length` と同じ扱い
- RTC を有効にする場所: 読み込み後に `policy.config.rtc_config = RTCConfig(enabled=True, prefix_attention_schedule=…, max_guidance_weight=…, execution_horizon=E)` を入れて `policy.init_rtc_processor()` を呼ぶ（本線の担当）
- `select_action`（方策の内部の待ち行列）は使わない

### 持ち越さない時点

次の 3 つでは、前の塊を持ち越さない。そこを i = 0 と同じに扱う（物理を止めて通常の生成、v = k、o = 0、L = なし）。
- 最初
- 指示を切り替えた直後（`reset()` を呼ぶ）
- やり直しの直後（`reset()` を呼ぶ）

## 3. 部品の型（`src/recovla/policy/schedule.py`）

```python
from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np

@dataclass(frozen=True)
class RuntimeConfig:
    mode: Literal["sync", "naive", "rtc"]
    exec_interval: int          # s
    delay_steps: int            # d（sync では無視して 0 とみなす）
    chunk_size: int = 50        # H
    execution_horizon: int = 10 # E（rtc だけ）
    # 生成時に検査: 1 <= s <= H、0 <= d < s、s + d <= H。違えば ValueError

@dataclass(frozen=True)
class Decision:
    k: int
    infer: bool                         # この k の観測で推論を始める
    block: bool                         # 推論の間、物理を止めて待つ（最初・reset 後と、sync の全推論）
    left_over: Optional[np.ndarray]     # rtc の i >= 1 のときだけ、(E, A) に揃えた前の塊の残り。それ以外は None
    inference_delay: int                # 方策に渡す d（rtc のとき。それ以外は 0）
    execute: tuple[int, int]            # この k で実行する (塊の番号 i, 塊の中の添字)

class Schedule:
    def __init__(self, cfg: RuntimeConfig): ...
    def reset(self) -> None             # 次の推論を「持ち越さない」扱いにする
    def decide(self, k: int) -> Decision
        # k は 0 から 1 ずつ増やして呼ぶ（飛ばしたら ValueError）。infer=True を返したら、呼び出し側は
        # その k のうちに deliver() を呼ぶ（推論は模擬の中では瞬時）
    def deliver(self, chunk: np.ndarray) -> int
        # 直前の infer=True に対する塊（(H, A)、後処理の前）を渡す。返り値は塊の番号 i
    def action(self, chunk_post: dict[int, np.ndarray]) -> np.ndarray
        # 補助: 直前の decide の execute に当たる行（後処理の後の塊から）を返す
    @property
    def log(self) -> list[dict]         # §4 の記録（推論ごと）
```

A は行動の次元（`predict_action_chunk` の出力は 7。`Schedule` は A を問わないので、検査は A = 32 でもよい。掲示板 0025）。

## 4. 記録（推論ごと。`trial_record.md` の json `inference` と同じ形）

| キー | 中身 |
|---|---|
| i | 塊の番号 |
| k_obs | 推論を始めた手（観測の時点） |
| k_valid | v_i |
| offset | o_i |
| left_over_len | 渡した残りの長さ（揃える前の長さ。持ち越さなければ 0） |
| reset | 持ち越さない推論だったか |
| wall_s | 推論の実時間（本線が測る。`torch.cuda.synchronize` 込み） |
| wall_breakdown_s | 内訳（本線が測って記録するだけ。掲示板 0010）: `preprocess`（前処理）、`vlm_prefix`（VLM の前半、KV キャッシュまで）、`denoise`（10 段の生成。RTC の誘導を含む） |

こまの記録の `chunk_id`・`chunk_index`・`chunk_switch` は、`decide` が返した `execute` から作る。

## 5. 乱数

- 推論 i の雑音は `SeedSequence(seed, spawn_key=(2, i))` から作る `torch.Generator` で引く（形 (1, H, 32)。雑音は詰め物の次元を含む）
  - 変換は `torch.Generator("cpu").manual_seed(seeds.torch_seed(seeds.seed_sequence(seed, "noise", i)))`（`src/recovla/common/seeds.py`。`generate_state(1, dtype=uint64)[0]`。掲示板 0022 への 0025 の回答）
- 方式が違っても、i 回目どうしは同じ雑音になる
- 配置（spawn_key (0,)）と誘発（(1,)）は試行の開始時に全部引くので、方式によらず同じになる（Step G 完了条件 3）

## 6. d の決め方（掲示板 0010）

1. Step G の最初に、RTC ありで推論を 50 回以上測る（最初の 1 回を除く）
2. d = ceil(95 パーセンタイル / 0.1 s) とする。s = 10 は変えない
3. d > 5 なら止めて報告する
4. 決めた値は `configs/default.yaml` の `runtime.delay_steps` に書く。それまでは null
5. 推論時間の内訳は測って記録するだけで、高速化はしない

## 7. cloud/runner の完了条件（B_提案書 §15.2 から）

模擬の方策（塊の中身が (i, 添字) から決まるもの）で、次を検査する。
- naive・d = 0 が sync と同じ `execute` の列になる
- 同じ種なら、方式によらず配置と誘発の条件が同じ（乱数列の作り方の検査）
- rtc の `left_over` が、上の j の定義と E への揃え方（切り詰め・0 詰め）どおり。同梱の `rtc.py` の `_normalize_prev_actions_length` と同じ結果になる
- `reset()` の直後は持ち越さない
- s + d > H などの不正な設定は ValueError
