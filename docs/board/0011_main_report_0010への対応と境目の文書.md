# 0010 への対応と、境目の文書（docs/interfaces/）

- 番号: 0011 / 差出: main / 宛先: human、review / 日時: 2026-09-25 18:00（日本時間）
- 返信先: 0010
- 状態: 未読
- 関係する枝・PR・コミット: main（このファイルを含むコミット）

## 要点（3行以内）

0010 の 1〜4 を行った。境目の文書 trial_record.md・results.md・runner.md を main に入れた。
これで cloud/metrics と cloud/runner を始められる。監督に、この 2 本の指示書をお願いしたい。
本線は Step D に進む。

## 本文

1. **d の決め方**: `configs/default.yaml` の `runtime.delay_steps` を null にし、`runtime.delay_rule`（50 回以上、95 パーセンタイル、0.1 s、上限 5）を置いた。d は Step G の最初に測って書く。推論時間の内訳（前処理・VLM の前半・10 段の生成）は記録の項目として `runner.md` §4 に入れた。高速化はしない
2. **境目の文書**（`docs/interfaces/`。本線が持ち、支線は変えない）
   - `README.md`: 共通の決まり（単位 m・s・rad、MuJoCo の世界座標、20 Hz のこまと 10 Hz の行動の格子、行動の 7 次元、色の並び red・green・blue、乱数列の番号）
   - `trial_record.md`: 評価の 1 試行の記録。json と npz のキー・形・型、判定の定義（持ち上げ・誤り・巻き添え・反応時間・復帰時間・継ぎ目の跳び・躍度・接触回数）、段階の符号、`metrics.py` の関数の型
   - `results.md`: 結果の CSV（trials・summary・paired・continuous）の列、統計の関数の型（Wilson、正確な McNemar、Newcombe の方法 10、Wilcoxon）、図
   - `runner.md`: sync・naive・rtc の時間の流れ（有効になる手、塊の中の添字、前の塊の残りの切り出しと E への揃え方）、`schedule.py` の型、推論ごとの記録、雑音の乱数列、d の決め方、cloud/runner の完了条件
   - あわせて、支線用の固定一覧 `env/requirements-cloud.txt`（lock から torch・torchvision を除いたもの。CPU 版の torch を同じ版で入れる手順つき）
3. **pytest**: `pyproject.toml` に `faulthandler_timeout = 300` を入れた
4. **掲示板**: 以後、既存のファイルは直接直さない。訂正は新しい番号で出す（STATUS.md を除く）
   - 0009 の日時「17:55」は誤りで、実際は 17:48（この行で訂正する）

## 求めること

- 監督: cloud/metrics と cloud/runner の指示書（掲示板の order）
  - 境目の文書に足りない点や、決め方に異論があれば、ask で返してほしい
  - 本線は Step D と並行して対応する
