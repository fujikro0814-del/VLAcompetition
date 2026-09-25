# 結果の表と統計（results）

版: 1（2026-09-25）
- 書く: `src/recovla/eval/report.py`（cloud/figures）。統計は `src/recovla/eval/stats.py`（cloud/metrics）
- 読む: 本線の `scripts/41_results.py`（Step G の 9。E1〜E8 を条件ごとに 1 コマンドで出す）

1 試行の指標は `trial_record.md` §6 の `trial_metrics` が作る。ここでは、それを並べた表と、条件間の比較の形を決める。

## 1. 置き場所

```
outputs/results/<実験>/          例: outputs/results/E3/
  trials.csv                     1 試行 1 行（全条件）
  summary.csv                    1 条件 1 行
  paired.csv                     条件の組ごとの比較（成否・復帰）
  continuous.csv                 条件の組ごとの比較（連続量）
  tables.md                      上の 4 つを人が読む形にしたもの
  fig_*.png                      図（§5）
```

CSV は UTF-8（BOM なし）、区切りはカンマ、1 行目が見出し。真偽は `true` と `false`、欠けた値は空欄。

## 2. `trials.csv`

| 列 | 型 | 中身 |
|---|---|---|
| experiment | str | E1〜E8 または選択用の名前 |
| condition | str | 条件の名前（json の `condition`） |
| model | str | R1・N1・K1 など |
| mode | str | sync・naive・rtc |
| s | int | 実行間隔 |
| d | int | 遅延（sync は 0） |
| safety_filter | bool | |
| trial | int | |
| seed | int | 条件をまたいで対にする鍵 |
| layout_kind | str | empty・prefilled_1・prefilled_2 |
| start_pose | str | home・retreat |
| target | str | red・green・blue（複数手順は `red>blue` のように > でつなぐ） |
| induce | str | P1・P2・P3・none |
| success | bool | |
| t_success_s | float | |
| error | bool | |
| collateral | bool | |
| collateral_induced | bool | |
| stage_reached | str | 接近・把持・搬送・設置・退避・完了 |
| induce_fired | bool | |
| induce_established | bool | |
| recovered | bool | 誘発が成立した試行だけ値を持つ（それ以外は空欄） |
| reaction_time_s | float | |
| recovery_time_s | float | |
| seam_jump_mean | float | 継ぎ目の跳び [m/s] |
| seam_jump_max | float | 継ぎ目の跳び [m/s] |
| nonseam_jump_mean | float | 切り替わりでない時点の同じ量 [m/s] |
| seam_ee_speed_jump_mean | float | 手先速度の跳び [m/s] |
| jerk_rms | float | [m/s³] |
| contacts_n | int | |
| inference_mean_s | float | |
| inference_p95_s | float | |

列の名前は `trial_metrics` の返り値の名前と同じにする。

## 3. `summary.csv`（1 条件 1 行）

| 列 | 中身 |
|---|---|
| experiment | |
| condition | |
| n | 試行数 |
| successes | 成功数 |
| success_rate | 成功率 |
| success_lo | Wilson 95% 区間の下限 |
| success_hi | Wilson 95% 区間の上限 |
| error_rate | 誤りの率 |
| collateral_rate | 巻き添えの率 |
| induced_n | 誘発した試行数 |
| established_n | 誘発が成立した試行数 |
| establish_rate | 成立率。80% 未満なら report.py が警告の行を tables.md に書く |
| recovered_n | 復帰した試行数 |
| recovery_rate | 復帰率 |
| recovery_lo | 復帰率の Wilson 区間の下限 |
| recovery_hi | 復帰率の Wilson 区間の上限 |
| reaction_time_median_s | 反応時間の中央値 |
| recovery_time_median_s | 復帰時間の中央値 |
| seam_jump_median | 継ぎ目の跳びの中央値 |
| nonseam_jump_median | 切り替わりでない時点の同じ量の中央値 |
| jerk_rms_median | 躍度の中央値 |
| contact_rate | 接触が 1 回以上あった試行の割合 |
| stage_接近 … stage_完了 | 段階別の到達の割合（その区分以上に到達した試行の割合） |

## 4. 比較（同じ種で対にする）

`paired.csv`（成否・復帰）の列:

| 列 | 中身 |
|---|---|
| experiment | |
| metric | `success`（全ての種）または `recovered`（誘発が両方の条件で成立した種だけ） |
| condition_a | |
| condition_b | |
| n_pairs | 対の数 |
| n11 | a・b とも真 |
| n10 | a だけ真 |
| n01 | b だけ真 |
| n00 | a・b とも偽 |
| rate_a | |
| rate_b | |
| diff | rate_a − rate_b |
| diff_lo | 対応ありの差の 95% 区間の下限（Newcombe の方法 10） |
| diff_hi | 同じく上限 |
| mcnemar_p | 正確な McNemar 検定（両側） |

`continuous.csv`（連続量）の列:

| 列 | 中身 |
|---|---|
| experiment | |
| metric | `seam_jump_mean`・`jerk_rms`・`reaction_time_s` など |
| condition_a | |
| condition_b | |
| n_pairs | 両方に値がある種の数 |
| median_a | |
| median_b | |
| median_diff | 種ごとの差（a − b）の中央値 |
| wilcoxon_stat | |
| wilcoxon_p | 符号順位検定（両側、差が 0 の対は除く＝scipy の既定 `zero_method="wilcox"`） |

## 5. 統計の関数の型（`src/recovla/eval/stats.py`）

```python
def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]
    # 流用元のまま（trials <= 0 なら (None, None)）
def mcnemar_exact(n10: int, n01: int) -> float
    # 両側の p 値。n10 + n01 == 0 なら 1.0。scipy.stats.binomtest(n10, n10 + n01, 0.5).pvalue
def paired_diff_ci(n11: int, n10: int, n01: int, n00: int, alpha: float = 0.05) -> tuple[float, float, float]
    # (差, 下限, 上限)。Newcombe (1998) の方法 10（各割合の Wilson 区間と φ 係数による補正）
def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]
    # (統計量, p, 対の数)。NaN を含む対は除く。対が 0 なら (nan, nan, 0)
def pair_by_seed(rows_a: list[dict], rows_b: list[dict], key: str = "seed") -> list[tuple[dict, dict]]
    # 同じ種の行を対にする。片方にしかない種は除き、除いた数を警告として返す仕組みは report.py 側で持つ
```

**検算**（cloud/metrics の完了条件）:
- Wilson: 流用元の報告値 61/100、18/20 と一致すること
- McNemar と Newcombe の区間: 公表されている数値例と一致すること
  - Newcombe (1998) の方法 10 は、論文の数値例の表と照合する
  - 例の値は支線が文献から引き、出典（表の番号）とともに検査に書く。本線はここに値を書かない（確かめていない値を約束にしないため）

## 6. 図（cloud/figures。合成データで作り、本番は本線が同じ関数で描く）

| ファイル | 中身 |
|---|---|
| `fig_success.png` | 条件ごとの成功率と Wilson 区間（横棒） |
| `fig_recovery.png` | 誘発ごとの成立率・復帰率 |
| `fig_seam.png` | 継ぎ目の跳びと、切り替わりでない時点の同じ量の分布（条件ごと、箱ひげ） |
| `fig_reaction.png` | 反応時間の分布 |
| `fig_stage.png` | 段階別の到達（積み上げ） |

図の色は条件ごとに固定し、同じ条件はどの図でも同じ色にする。
