# 指示書 — 支線 cloud/metrics（指標と統計）

差出: 監督 / 宛先: 支線（クラウドの Claude Code） / 日付: 2026-09-25 / 返信先: 掲示板 0011
リポジトリ: fujikro0814-del/VLAcompetition（公開）/ 枝: `cloud/metrics`

## 0. 最初にすること

1. 枝 `cloud/metrics` を main から切る。`main` と他の枝には push しない
2. この指示書を、そのまま `docs/board/NNNN_review_order_cloud-metrics指示書.md` として保存する（NNNN は、その時点の最大番号＋1。取り込みのときに番号がぶつかったら本線が付け直す）
3. 次を読む: `docs/board/README.md`（運用規則）、`docs/interfaces/README.md`・`trial_record.md`・`results.md`（**これが仕様。食い違ったらこちらが正**）、`docs/B_提案書.md` の §7 と §15、`src/recovla/eval/stats.py`（流用した `wilson_interval`）

## 1. 目的

評価の記録（1 試行＝json と npz）から指標を計算する純関数と、条件間の比較の統計を作る。本線が Step G で評価器を拡張したとき、ここで作った関数がそのまま使える状態にする。MuJoCo・方策・GPU は使わない。

## 2. 持ち場（書いてよいパスはこれだけ）

- `src/recovla/eval/metrics.py`（新規）
- `src/recovla/eval/stats.py`（既存の `wilson_interval` は**変えずに**、関数を足す）
- `tests/eval/`（新規。合成の記録を作る補助もここに置く）
- `docs/board/` の自分の報告・ask

`docs/interfaces/`、`configs/`、`src/recovla/` のほかの場所、`env/` は変えない。変えたいときは ask を出す。

## 3. 環境

`env/requirements-cloud.txt` の冒頭の注記どおりに、リポジトリ直下の `.venv`（Git 管理外）へ入れる（Python 3.12、torch と torchvision は CPU 版を同じ版で）。版の固定一覧は書き換えない。入らないものがあれば止まって ask。

## 4. 作るもの

### 4.1 `metrics.py`（`trial_record.md` §4・§6 の定義どおり）

- `TrialRecord`、`load_trial`、`trial_metrics`、`seam_jumps`、`reaction_time`、`recovery_time`、`jerk_rms`
- `trial_metrics` の返り値のキーは `results.md` §2 の `trials.csv` の列名と同じにする
- 段階別の到達（`stage_reached`）は `trial_record.md` §5 の符号と区分から出す
- 巻き添えと「誘発による移動」の区別は §4 の定義どおり
- 欠けた値の扱いは `interfaces/README.md` の決まり（数値は NaN、ない出来事は null）。誘発が成立していない試行の `recovered` は None（CSV では空欄）
- 判定の閾値は引数の `eval_cfg`（`configs/default.yaml` の `eval` を読み込んだ dict）から取り、数値を直書きしない
- ループではなく numpy の配列演算で書く。100 試行を数秒で処理できること

### 4.2 `stats.py` に足すもの（`results.md` §5 の型どおり）

- `mcnemar_exact`、`paired_diff_ci`（Newcombe の方法 10）、`wilcoxon_paired`、`pair_by_seed`

## 5. 検査（`tests/eval/`。GPU・モデル・ネットワークなしで通ること）

1. **合成の記録**: 跳び・時刻・接触・移動を前もって決めた json と npz を作る補助を書き、各関数がその値を返すことを確かめる。少なくとも次を含める
   - 継ぎ目の跳び: 切り替わりの時点だけ速度が段差になる行動列。跳びの平均・最大と、切り替わりでない時点の量
   - 反応時間・復帰時間: 距離と高さの時系列を決め、期待する時刻と一致する（こまの格子 0.05 s の上で）
   - 躍度: 3 次式の x_des で解析値と一致する
   - 巻き添え: 手の接触による移動と、誘発で落ちた立方体が当たった移動を分けて数える
   - 段階別の到達、誤り（先客を持ち上げた場合を含む）
   - 欠けた値（推論なし、誘発なし、成功しない）で例外を出さない
2. **Wilson**: 流用元の関数が変わっていないこと（18/20、61/100 の区間が、変更前のコミットの値と一致）
3. **McNemar**: 手計算できる例で一致する（例: n10=0・n01=5 → p=0.0625、n10=n01=0 → 1.0）。scipy の binomtest を使うこと
4. **Newcombe の方法 10**: 文献（Newcombe, R. G., Statistics in Medicine 17, 2635–2650, 1998）の数値例と照合する
   - 文献の表の値を引用するときは、表の番号と値を検査のコメントに書く
   - **文献に当たれなかった場合は、値を推測で書かない。** 別の書き方の独立な実装（式を別の順で組んだもの）と一致することだけを確かめ、報告に「文献との照合は未了」と書く
5. **Wilcoxon**: scipy の `wilcoxon`（`zero_method="wilcox"`）と一致。NaN を含む対を除くこと、対が 0 のとき `(nan, nan, 0)` を返すこと
6. 既存の検査（`tests/legacy/` など）のうち、クラウドで通っていたものが引き続き通ること

## 6. 返すもの（完了条件）

- 枝を push し、PR を作る（題: 「cloud/metrics: 指標と統計」）
- 掲示板に報告 `NNNN_cloud-metrics_report_指標と統計.md` を書き、PR に含める。中身は次のとおり
  - 作った関数の一覧
  - 検査の結果（件数、合否）
  - Newcombe の照合が文献か独立実装か
  - 仕様（interfaces）で曖昧だった点と、どう解釈したか
  - 本線に確かめてほしい点
- PR の変更は §2 の持ち場の中だけ

## 7. やらないこと

- 評価器・描画・結果の表の書き出し（`report.py` は cloud/figures の担当）
- `docs/interfaces/` と設定ファイルの変更
- 依存の追加（固定一覧にないパッケージを入れない）
- 学習、方策の推論、MuJoCo の実行

## 8. 迷ったとき

仕様が曖昧で結果が変わる点は、仮定で埋めずに ask（`NNNN_cloud-metrics_ask_…`）を書いて PR に含め、その点は止まる。影響しない部分は先に進めてよい。
