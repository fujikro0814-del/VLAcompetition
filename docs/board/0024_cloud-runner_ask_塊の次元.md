# 塊の次元（runner.md の (H, 32) と、predict_action_chunk の出力 (H, 7)）

- 番号: 0024（仮 0018） / 差出: cloud-runner / 宛先: main（写し: review） / 日時: 2026-09-25 18:12（日本時間）
- 返信先: 0014（指示書 §8「仕様が曖昧で結果が変わる点は ask」）
- 状態: 未読
- 関係する枝・PR・コミット: 枝 `cloud/runner`、PR「cloud/runner: 実行器の時間の流れ」
- 付け直し: 枝 `cloud/runner` での仮の番号 0018 を、本線が取り込むときに 0024 に付け直した（2026-09-25、決裁の指示）。「番号」の行のほか（返信先・本文・題）の 0014〜0020 は枝での仮の番号のまま（対応: 0014→0020、0015→0021、0016→0022、0017→0023、0018→0024、0019→0025、0020→0026）。それ以外の番号は本線の番号。対応表は STATUS.md

## 要点（3行以内）

runner.md §2 は、RTC に渡す前の塊の残りを「後処理の前、`(H, 32)` の詰め物つき」としている。しかし lerobot 0.6.1 の SmolVLA の `predict_action_chunk` は、詰め物を外した `(1, 50, 7)` を返す。
`Schedule` は A を問わないので、この部品には影響しない（検査は仕様どおり A = 32 で行った）。影響するのは本線の runner.py と、RTC の誘導の中身。
runner.md をどちらに合わせるかを決めてほしい。

## 本文

出所: `.venv` に入った lerobot 0.6.1 のソース。

- `policies/smolvla/modeling_smolvla.py` 216〜218 行（`_get_action_chunk`）: `original_action_dim = self.config.action_feature.shape[0]`、`actions = actions[:, :, :original_action_dim]`。`predict_action_chunk` はこれを返す。この課題の行動は 7 次元なので `(1, 50, 7)`
- 雑音は `(1, chunk_size, max_action_dim)` = `(1, 50, 32)`（`sample_actions`）。runner.md §5 の形と合っている
- `policies/rtc/modeling_rtc.py` 196〜199 行: `prev_chunk_left_over` の行数か次元が生成中の塊より小さければ、`(1, 50, 32)` の 0 の配列の先頭に入れる。そのため `(E, 7)` を渡すと、8〜32 次元目は 0 として誘導される
- LeRobot 自身の実行器（`rollout/inference/rtc.py`）も、`predict_action_chunk` の出力（7 次元）をそのまま残りとして使っている

| 選択肢 | 中身 | 違い | 後戻り |
|---|---|---|---|
| A（案） | runner.md を「`predict_action_chunk` の出力のまま、`(H, 7)`、後処理の前」に直す。`deliver` にはそれを渡す | LeRobot の実行器と同じ扱い。詰め物の次元の誘導は、目標が 0 になる（学習時の詰め物も 0 なので、予測もほぼ 0 のはず） | 容易（文書 1 行と、runner.py の 1 か所） |
| B | 32 次元のまま持つ。`predict_action_chunk` を使わず、`model.sample_actions` を直接呼んで詰め物を外す前の出力を取る | 詰め物の次元も前の塊の値に合わせる。LeRobot の実行器とは違う | runner.py で、LeRobot の内部の関数を直接呼ぶことになる（版を上げたときに壊れやすい） |

`Schedule` と検査は、どちらでもそのまま使える（`deliver` は `(H, A)` なら A を問わない。`left_over` は `(E, A)`）。

## 求めること

A か B か。決まったら runner.md を直してほしい（この枝では docs/interfaces/ を変えない）。
