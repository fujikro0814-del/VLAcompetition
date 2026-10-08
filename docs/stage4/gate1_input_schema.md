# 関門 1 の判定の入力の形（`recovery_vla.s4_gate1_input/1`）

- 使う道具: `src/recovla/eval/gate1.py`、`scripts/98_s4_gate1.py`（検査は `tests/test_gate1.py`）。
- 規則と数値は `configs/s4_gates.json` から読む（このファイルには数値を書かない。下の閾値は説明のための写し）。
- 入力は「測った値」だけ。合否・対比・p 値・基準（E0_ref_lift など）は入力に入れず、道具が計算する。
- 欄の名前は `s4_gates.json` の `metrics` の名前と、各条件の `metric` の名前に合わせた。
- 割合は必ず `{"k": 分子, "n": 分母}` で渡す（`rate` を添えてもよいが、k/n と 5e-5 より食い違えば「形が違う」）。
  割合は k/n の分数のまま閾値と比べるので、ちょうど閾値の値は合格になる。
- 欄がない・`null`・型が違う・分母が 0 の値を使う条件は「判定できない」になり、合格にしない。入力の全体が辞書でないときだけ、道具は止まる（終了コード 2）。

## 全体

```json
{
  "schema": "recovery_vla.s4_gate1_input/1",
  "source": {"自由": "出どころ（コマンド・ファイル・コミット）"},
  "R": {...}, "T": {...}, "S": {...}, "XPL": {...}, "C": {...}, "K": {...}
}
```

関門ごとの欄は、どれを省いてもよい（省いた関門は判定できないになる）。

## R（RTC の設定。`gates.R`）

```json
"R": {
  "settings": {
    "<設定名>": {
      "n_trials": 30,
      "radial_gap_mm": -4.8,
      "move_ratio": 0.99,
      "seam_jump_mps": 0.0135,
      "natural_success_30": 25,
      "first_close_after_30s": 0
    }
  },
  "shadow_plan_shorter_mm": {"10": 3.1, "20": 7.9, "30": 11.0, "40": 14.2},
  "x2_shortfall_mm": {"10": 2.0, "20": 5.5, "30": 9.0, "40": 12.5}
}
```

| 欄 | 意味 | 使う条件 |
|---|---|---|
| `settings` の鍵 | `gates.R.runs.settings` の名前。`naive` と `gates.R.candidates` の 4 設定が要る（`current_repro` はあってもよいが判定に使わない） | |
| `n_trials` | その設定の試行数 | R.c3（naive と違えば判定できない） |
| `radial_gap_mm` | 最初に閉じた位置の半径方向の差の中央値 [mm]（閉じた試行。最初の閉じで決まるので時間によらない） | R.c1、順位 3 |
| `move_ratio` | 移動の比の中央値。最初の閉じが 30 s 以内の試行だけ（掲示 0154 の決定 1、`d_rtc_settings.json` の `definitions_before_run.move_ratio_30s`） | R.c2 |
| `seam_jump_mps` | 継ぎ目の速度の跳びの中央値 [m/s]。行動の時刻 < 30 s の継ぎ目だけ（`definitions_before_run.seam_jump_30s`） | R.c4、順位 2 |
| `natural_success_30` | 自然の試行の成功数（30 s の採点。`t_success <= 30`） | R.c3、順位 1 |
| `first_close_after_30s` | 30 s より後に最初に閉じた試行の数（並べるだけ） | なし |
| `shadow_plan_shorter_mm` | 影の推論の (影 − 誘導)·u の中央値 [mm]。鍵は h（`gates.R.branching_rule` の R.b1 の文の行の並び） | R.b1 |
| `x2_shortfall_mm` | X2 の (エキスパート − 予測)·u の中央値 [mm]。鍵は h | R.b1 |

作り方: `98_s4_d_rtc.py metrics` の出力の `conditions[設定名].by_horizon["30"].summary` から
`n_trials`・`radial_gap_mm_median[0]`・`move_ratio_median[0]`・`seam_jump_mps_median[0]`・`successes_at["30"]`・`n_first_close_after_horizon`、
`shadow-metrics` の出力の `summary.shadow_minus_guided_mm_median[h][0]`、`98_s4_x2.py` の `summary.json` の
`summary.expert_minus_pred_mm_median[h][0]`（`gate1.input_from_rtc`）。

## T（2 番目の手順の始めに戻す。`gates.T`）

```json
"T": {
  "arms": {
    "E0_run1": {"first_close_lift": {"k": 5, "n": 40}, "all_three_true": {"k": 12, "n": 40}},
    "E0_run2": {"first_close_lift": {"k": 7, "n": 40}, "all_three_true": {"k": 11, "n": 40}},
    "EH":      {"first_close_lift": {"k": 26, "n": 40}, "all_three_true": {"k": 14, "n": 40}},
    "ES":      {"first_close_lift": {"k": 21, "n": 40}, "all_three_true": {"k": 12, "n": 40}}
  },
  "EO_green_first": {"first_close_lift": {"k": 30, "n": 40}, "colors": ["green"]},
  "K_e7": {"並べるだけ": "98_s4_d_e7.py の gate_T_inputs.K_e7 をそのまま"}
}
```

| 欄 | 意味 | 使う条件 |
|---|---|---|
| `first_close_lift` | 2 番目の手順の 1 回目の試みの最初の閉じで持ち上がった数 / 2 番目の手順が始まった試行（`metrics.first_close_lift` の E7 の分母） | T.c1、T.c2、基準（E0_ref_lift・d_E0・required_margin は道具が計算） |
| `all_three_true` | 3 個とも（真値）の本数 / 試行数 | T.c3（4 つの腕の n が違えば判定できない） |
| `EO_green_first.first_close_lift` | EO の 1 番目の手順の 1 回目の閉じの持ち上がり / 1 番目の手順が始まった試行 | T.b1 |
| `EO_green_first.colors` | EO の 1 番目の手順の色の一覧。`["green"]` でなければ T.b1 は判定できない | T.b1 |

作り方: `98_s4_d_e7.py summary` の出力の `gate_T_inputs.arms[腕].first_close_lift`・`all_three_true` の k と n、
`gate_T_inputs.EO_green_first`（`gate1.input_from_e7`。`E0_ref_lift`・`d_E0`・`required_margin` は写さない）。

## S（単発の開始。`gates.S`）

```json
"S": {
  "conditions": {
    "standby_end|on_grid":   {"plus_y_shift": {"k": 20, "n": 33}, "first_close_after_30s": 0},
    "standby_start|on_grid": {"plus_y_shift": {"k": 3, "n": 33}},
    "standby_end|wall_side": {"plus_y_shift": {"k": 25, "n": 33}}
  }
}
```

- 鍵は `<start_pose>|<prior_cube>`（`gates.S.runs.factorial` の名前）。判定に使うのは上の 3 条件。ほかの条件（home など）はあってもよい（使わない）。
- `plus_y_shift` は最初の閉じで +y に 1 cm を超えてずれた数 / 最初の閉じが起きた試行（`metrics.plus_y_shift`）。
- 姿勢の効き（S.pose）＝ `standby_end|on_grid` − `standby_start|on_grid`、先客の効き（S.prior）＝ `standby_end|wall_side` − `standby_end|on_grid`。
  組み合わせは `gates.S.contrasts.*.value` と `fix` の文から読む。p 値は道具が片側（a > b）のフィッシャーの正確検定で出す。
- 作り方: `98_s4_d_start.py summary` の出力の `gate_S_inputs.conditions[鍵].plus_y_shift` の k と n（`gate1.input_from_start`）。

## XPL（移植の腕。`transplant_arm`）

```json
"XPL": {
  "arms": {
    "XPL_as":   {"rep1": {"k": 16, "n": 20}, "rep2": {"k": 15, "n": 20}},
    "XPL_home": {"rep1": {"k": 4, "n": 20},  "rep2": {"k": 5, "n": 20}},
    "XPL_grid": {"rep1": {"k": 15, "n": 20}, "rep2": {"k": 14, "n": 20}}
  }
}
```

- 値は +y のずれ（`plus_y_shift`）の数 / 最初の閉じが起きた試行。回ごと（`rep1`〜`rep{transplant_arm.repeats}`）に渡す。2 回合わせた値は道具が k と n を足して出す。
- 作り方: `98_s4_d_start.py summary` の出力の `xpl_inputs.arms[腕].rep1`・`rep2` の k と n。

## C（落下・手を止める版。`gates.C`）

```json
"C": {
  "fall_with_hold": {
    "by_L": {
      "30": {"recovery_R": {"k": 10, "n": 40}, "recovery_N": {"k": 3, "n": 40},
             "paired": {"pairs": 38, "r_only": 8, "n_only": 2}},
      "60": {"recovery_R": {"k": 14, "n": 44}, "paired": {"pairs": 42, "r_only": 9, "n_only": 3}}
    }
  },
  "fall_as_is": {"by_L": {"30": {"recovery_R": {"k": 2, "n": 38}, "paired": {"pairs": 36, "r_only": 2, "n_only": 1}}}}
}
```

| 欄 | 意味 | 使う条件 |
|---|---|---|
| 版の鍵 | `gates.C.runs.variants` の名前。判定に使うのは `gate_variant`（fall_with_hold） | |
| `by_L` の鍵 | 採点の時間 L [s]。判定に使うのは C.c1〜c3 の文にある 30。60 は並べるだけ | |
| `recovery_R` | 復帰デモありのモデルで、L 秒以内に誘発が成立した試行（`induce.t_established <= L`）のうち L 秒までに成功した数 / その試行の数 | C.c1 |
| `paired` | 両方のモデルで L 秒以内に成立した種の対（種と目標の色）の数 `pairs`、ありだけ成功 `r_only`、なしだけ成功 `n_only` | C.c2（(r_only − n_only)/pairs）、C.c3（pairs） |

作り方: `98_s4_d_recovery.py score` の出力の `variants[版].material.by_L[L]` の `recovery_R.recovered`・`established`、
`paired.pairs`・`r_only`・`n_only`（`gate1.input_from_recovery`）。

## K（物差し。`gates.K`）

```json
"K": {
  "run1_completed": 99, "run2_completed": 99, "pairs_matched": true,
  "d0": {"30": {"k": 12, "n": 99}, "60": {"k": 7, "n": 99}}
}
```

| 欄 | 意味 | 使う条件 |
|---|---|---|
| `run1_completed`・`run2_completed` | 最後まで終わった試行の数（1 回目・2 回目） | K.c1（文の 99 と比べる） |
| `pairs_matched` | 2 回の種・配置・色の対応が一致しているか（真偽） | K.c1 |
| `d0["30"]`・`d0["60"]` | 2 回で成否が食い違った試行の数 / 試行数（30 s の採点・60 s の採点。鍵は K.b1・K.b2 の文の `d0_30`・`d0_60` の数） | K.b1、K.b2 |

束 0 の値（掲示 0154）は上の例のとおり（30 s 12/99、60 s 7/99）。`outputs/s4/k_d0.json` の形は決まっていないので、変換の関数はない（手で写す）。

## 結果（`recovery_vla.s4_gate1_result/1`、`--out`）の主な欄

- `gates_file`: 読んだ `s4_gates.json` の場所と SHA-256（そのまま・改行 LF）、掲示 0153 の値と一致するか、`status`。
- `gates.R|T|S|XPL|C|K`: 条件ごとに `id`・`rule`（読んだ文）・`value`・`threshold`・`op`・`result`（pass / fail / undetermined）・`reason`。
  R は `passed`・`ranking`・`top`・`branch`（continue / stop / undetermined）・`branches`（R.b1・R.b2）。
  T は `reference`（E0 の基準・d_E0・要求の幅）・`return_to`・`branches`（T.b1・T.b2）。S は `contrasts`、XPL は XPL.rep・XPL.pose・XPL.prior、
  C は `result`・`branch`・`by_variant_report_only`（60 s と C.b2 の材料。並べるだけ）、K は K.c1・`d0`・`warning`。
- `candidate_selection`: 当たった行・verdict・各行の当てはめ（`trace`）・`ex_arm`・行 3 のときの `executor_suffices`。
- `conclusion`: 束 2（RTC の設定、v3 (a) の戻し先）、束 4 の候補（`start_state_prior_cube` / `slip` / `none`）、EX の腕を足すか、
  研究の道へ回すもの、警告、`status`（determined / undetermined）と判定できない所。
- `rules_unreadable`: 文が想定の形と一致せず当てはめなかった規則（ファイルの行つき）。
- `not_machine_readable`: 機械が読める形で書かれていないので当てはめなかった規則（ファイルの行と理由）。
- `read_with_outside_definitions`: `s4_gates.json` の外の定義を合わせて読んだ所（作者が確かめる）。
- `input`・`input_sha256`: 使った入力。

## 判定できないときの扱い

- 条件の all_of は三値で当てはめる: 1 つでも不合格なら不合格、そうでなく 1 つでも判定できなければ判定できない。
- `s4_gates.json` には判定できない条件があるときの枝の規則がないので、結論がそれで変わる所は「判定できない」として止める
  （CLI の終了コード 3）。結論が変わらない所（例: 開始状態の候補が残れば、関門 C の合否によらず束 4 は開始状態）はそのまま出す。
