# 可視化の手順（Rerun）

学習データ（LeRobot 形式）と評価の走行の記録を、Rerun（rerun.io）のビューアで見る手順。

- スクリプト: `scripts/90_rerun.py`
- 環境: `.venv_viz`（評価用の `.venv` とは別。rerun-sdk・pyarrow・numpy・pillow だけを入れてある。Git の管理外）
- 出力: `outputs/viz/<名前>.rrd`（Git の管理外）
- 評価のコード（`recovla`）は読み込まないので、評価を回している最中でも使ってよい

## 1. 見る（いちばんよく使う）

作ってある `.rrd` を開く。PowerShell で、作業場所は `C:\PAI\recovery_vla`。

```
.venv_viz\Scripts\rerun.exe outputs\viz\<名前>.rrd
```

複数のファイルを並べて渡すと、まとめて開く。

## 2. 学習データを作る

```
.venv_viz\Scripts\python.exe scripts\90_rerun.py dataset --root outputs\datasets\R1v2_20260929-205251 --episodes 0,254
```

- `--root`: データセットの場所（`outputs\datasets\` の下）。今の学習データは `R1v2_20260929-205251`・`N1v2_20260929-205251`
- `--episodes`: エピソードの番号（カンマ区切り）。番号と元の生成の対応は `meta\conversion.json` の `sources`（`raw_episode` が `n_…` なら通常、`A_…`・`B_…`・`C_…` なら復帰）
- 出力: `outputs\viz\dataset_<データセット>_ep<番号>.rrd`

ビューアで見えるもの（エピソードごとに `ep<番号>/` の下）:

| 名前 | 中身 |
|---|---|
| `camera/overhead`・`camera/wrist` | 方策に入る俯瞰・手首の画像（256×256、学習データの向きのまま） |
| `state/…` | 状態 17 次元（手先の位置 `eef_x/y/z`、向きの偏差、指、関節 7、手がかり `cue_x/y`） |
| `action/…` | 行動 7 次元（x_cmd の 10 Hz の差 `dx/dy/dz`、回転は 0、グリッパ `+1` 閉・`-1` 開） |

時間の軸は `frame`（10 Hz のこまの番号）と `t`（秒）。

## 3. 評価の走行を作る

```
.venv_viz\Scripts\python.exe scripts\90_rerun.py run --dir outputs\v2eval\V2CAUSE\R1v2_nat --trials 1,4
```

- `--dir`: 走行の場所（`outputs\v2eval\<実験>\<条件>`。`trial_NNNN.npz`・`trial_NNNN.json`・`runtime_NNNN.json` がある所）
- `--trials`: 試行の番号（ファイル名の NNNN）。成功・失敗は `trial_NNNN.json` の `success`
- 出力: `outputs\viz\run_<実験>_<条件>_t<番号>.rrd`
- 走行の記録に画像は残っていないので、軌跡と信号だけ

ビューアで見えるもの（試行ごとに `trial<番号>/` の下）:

| 名前 | 中身 |
|---|---|
| `info` | 種・目標の色・成否・止まった理由・G3 の違反の数 |
| `world/path/hand`（灰）・`x_cmd`（橙）・`fingertip`（青紫） | 試行全体の軌跡 |
| `world/hand`・`x_cmd`・`fingertip` | その時刻の位置 |
| `world/cubes_truth` | 立方体の真値（赤・緑・青の箱） |
| `world/cubes_perceived` | 知覚の推定（色つきの点。見失った色は出ない） |
| `world/cue` | 方策に渡した目標の手がかり（桃色の点、推論のたびに更新） |
| `world/box_estimate` | 起動時に知覚で当てはめた箱 |
| `signal/gripper_closed`・`finger_width_mm` | グリッパの開閉の指令と開き幅 |
| `signal/x_cmd_minus_hand_mm` | 参照位置と手先の差（0123 の B1） |
| `signal/tip_to_target_mm` | 指先と目標の立方体の距離 |
| `signal/safety_active` | 安全フィルタが働いているか |
| `events/inference` | 推論の時刻と計算時間 |

時間の軸は `sim_time`（シミュレーションの秒）。

## 4. 見方のこつ

- 3 次元の画面で `world` を開き、下の時間の棒を動かすと、手先・x_cmd・立方体の動きを追える
- 失敗した試行では、`tip_to_target_mm` が数 cm で止まったまま `gripper_closed` が何度も切り替わる様子が見える（0123 の「目標の手前の高い位置で閉じる」）
- 学習データと並べるときは、`dataset` と `run` の `.rrd` を一緒に開く

## 5. 注意

- 種の決まり（`docs/種の台帳.md`）: ここで見るのは、すでに回した走行と学習データの記録だけ。**封をした段階 2 の結果（`outputs/sealed/`）は開かない**（0121・0124）
- 新しい試行を回すわけではないので、種の台帳への記入は要らない
- 学習データの `.rrd` は、画像を JPEG（品質 90）で縮めて入れる（1 エピソードでおよそ 4 MB）
- Rerun の使用状況の送信（匿名の利用の記録）は、`rerun analytics disable` で切ってある。入れ直したときは、もう一度切る
- 環境を作り直すとき:

```
C:\PAI\recovery_vla\.python\cpython-3.12.14-windows-x86_64-none\python.exe -m venv .venv_viz
.venv_viz\Scripts\python.exe -m pip install rerun-sdk pyarrow numpy pillow
.venv_viz\Scripts\rerun.exe analytics disable
```
