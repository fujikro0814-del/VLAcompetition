# PR #2・#3 の取り込み、番号の付け直し、0022・0024・0019 への対応

- 番号: 0028 / 差出: main / 宛先: human、review / 日時: 2026-09-25 18:45（日本時間）
- 返信先: 0027
- 状態: 未読
- 関係する枝・PR・コミット: PR #2（取り込み 59caa9d）、PR #3（取り込み 5ebc2c3）、main（このファイルを含むコミット）

## 要点（3行以内）

PR #2（metrics）・#3（runner）を取り込んだ（#1 は 0015 で取り込み済み）。取り込む前に Windows の torch ありの環境で tests/eval・tests/policy を全件回し、191 件合格・飛ばし 0。
枝の掲示板の番号を取り込む順に付け直した（metrics 0012〜0015 → 0016〜0019、runner 0014〜0020 → 0020〜0026）。対応表は STATUS。以後、枝の中は cNN（README）。
common/seeds.py を入れ（0022）、runner.md を (H, 7)・後処理の前に直し（0024）、results.md に phi_correction と trial_metrics の全列を書き足した（0019）。

## 本文

### 1. 取り込みと検査

| PR | 枝 | 取り込みのコミット | 取り込む前の検査（この PC、`.venv`、torch 2.11.0+cu126） |
|---|---|---|---|
| #1 | cloud/setup-check | 8c74e0d（0015） | — |
| #2 | cloud/metrics | 59caa9d | `pytest tests/eval` → 67 件合格 |
| #3 | cloud/runner | 5ebc2c3 | `pytest -rs tests/policy tests/eval` → **191 件合格、飛ばし 0**（tests/policy 124 件。`test_rtc_compare.py` の 60 件＝`rtc.py` の `_normalize_prev_actions_length` との照合 56 件と雑音 4 件を含む。0023 の (b)） |

- push の前に、全体の検査（`pytest -p no:cacheprovider`）と `scripts/check_before_push.ps1`（`scripts/push.ps1` の中）を通した。結果はこのコミットの push のときの出力（下の 5）
- `tests/policy/test_rtc_compare.py` に目印 `torch` を足した（0013 の 3 の決まり。枝で書かれたときはまだ決まりがなかった）。中身は変えていない

### 2. 番号の付け直し（取り込んだ順）

| 枝 | 枝での仮の番号 | 正式な番号 | 中身 |
|---|---|---|---|
| cloud/metrics | 0012 | 0016 | 監督の指示書（cloud/metrics） |
| | 0013 | 0017 | 報告: 指標と統計 |
| | 0014 | 0018 | ask: 指標の定義の曖昧な 4 点 |
| | 0015 | 0019 | 監督の回答（ファイル名の「0014 への回答」も 0018 に） |
| cloud/runner | 0014 | 0020 | 監督の指示書（cloud/runner） |
| | 0015 | 0021 | 報告: 実行器の時間の流れ |
| | 0016 | 0022 | ask: 乱数列の関数 |
| | 0017 | 0023 | ask: torch が入らず照合できない |
| | 0018 | 0024 | ask: 塊の次元 |
| | 0019 | 0025 | 監督の回答（ファイル名の「0016-0018 への回答」も 0022-0024 に） |
| | 0020 | 0026 | 報告: torch の再取得（403 のまま） |

- 変えたのは、ファイル名と「番号」の行（`0017（仮 0013）` の形）、付け直しの注記 1 行だけ。返信先・本文・題の中の番号は枝で書かれたまま残し、注記に対応を書いた
- 手違い 1 件（コミットの前に直した）: 付け直しの道具が、最初に番号だけでファイルを選んだため、本線の 0012（setup-check の報告）と 0015（本線の報告）を誤って改名・編集した。コミットの前に気づき、両方を HEAD から戻した（HEAD との差 0 を確かめた）。以後は番号と名前の一部の両方で選ぶようにした
- 支線は以後、枝の中で仮の番号 `cNN` を使う（`docs/board/README.md` に書き足した。取り込みのときに本線が正式な番号を振り、STATUS に対応表を残す）

### 3. 0022（乱数列の関数）: common/seeds.py

0025 で採った型のとおり: `STREAM_ID`、`seed_sequence(seed, name, *sub)`、`stream`、`streams`、`torch_seed(ss)`（`generate_state(1, dtype=uint64)[0]`）。台本と注入の列は `script_rng`・`inject_rng`（spawn_key (3, 色の添字, 作り直しの回数)・(4, …)）。検査 `tests/test_seeds.py`（4 件、numpy だけ）。

- `schedule.py` の `noise_generator` をこれに置き換えるのと、0022 §3 の「方式によらず配置と誘発が同じ」の検査は、0025 のとおり支線の別の PR で行う（本線は schedule.py を変えていない）
- `docs/interfaces/README.md` の乱数の行と、`runner.md` §5 に変換を書き足した

### 4. 0024（塊の次元）と 0019（metrics への回答）: 境目の文書

- `runner.md` §2: RTC に渡す前の塊の残りは「後処理の前、`predict_action_chunk` の出力のまま `(H, 7)`」に直した（0025 の A）。§3 の A の説明も直した
- `results.md` §2: `trial_metrics` は trials.csv の全列を表と同じ並びで返す。反応時間・復帰時間は誘発が成立した試行だけ、`recovered` は成立しなければ空欄
- `results.md` §5: `paired_diff_ci(..., *, phi_correction: bool = True)`（キーワード専用。方法 10 の φ の補正の定義）

### 5. Step D

並行して進めている（未コミット）。3 色の場面・配置・台本（`phase_of` は純関数）・生成の枠（描画なしでも回る）・記録・再生確認まで動いている。完了条件の測定はこの後。

## 求めること

判断を求めるものではない（報告）。runner の支線に、`noise_generator` の置き換えと 0022 §3 の検査の PR（枝の中は cNN）をお願いしたい。
