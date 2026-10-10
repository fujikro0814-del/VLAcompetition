# 握り損ねの反射の確かめ: 事前登録 v1（学習なしの実験）

- 書いた日: 2026-10-11（草案。登録は本線が掲示板で行う）。実験はまだ回していない。結果を見る前に固める。
- 道具: `src/recovla/runtime/reflex.py`・`src/recovla/eval/induce_slip.py`・`scripts/96_s4_resume.py`（ブランチ `s4-reflex`）。
  起動の前に、使う版の SHA（git の HEAD）をこの文書の第 9 節に書き、全条件で同じ版を使う。
- 中身の説明: `docs/stage4/reflex_protocol.md`。
- 記号: P2＝段階 3 の落下の誘発（指を開いて落とす）。P2S＝引き抜きの落下（P2S-v1。指は閉じたまま空を掴む。実機に近い形）。
  drop_only＝反射の設定の名前（落下だけで発火。空掴みは切）。McNemar＝同じ種の対の正確な検定。

## 1. 問い

1. **主**: 反射（drop_only）を入れると、R1v3 の P2S からの 30 s の復帰が上がるか（同じ種の対で、切 対 入）。
2. 副: P2 でも同じ形で効くか（R1v3・P2・入。P2 の切は段階 4 の S4DREC の fall_as_is 0/23 を参考にする。種が違うので対にはしない）。
3. 副: 落下の実演を学んでいない N1v3 でも上がるか（N1v3・P2S・入 を、R1v3・P2S・入 と同じ種で並べる）。「データに合わせる」が
   効いているなら、R1v3 の方が高い。
4. 守り: 自然な試行の成功を下げないか、誤発火が少ないか（R1v3・自然、切 対 入）。

## 2. 固定する条件

- 模型: R1v3・N1v3（段階 3 の凍結のチェックポイント。`82_v2_eval.py` の CKPT）。学習はしない。
- 実行: `--mode naive --exec-interval 6 --no-safety`、制限時間 60 s（`--time-limit-s 60`、既定）、試行ごとに世界を作り直す（既定）。
- 反射: `--reflex drop_only`（`--reflex-set` は使わない）。引数は `ReflexParams` の既定に `fire_on_miss=false` を重ねたもの
  （持つ範囲 3.0〜6.0 cm、持ち上げ 3 cm、落下のしきい値 4 mm、止め方 measured・0.25 m/s、待ち 0.5 s かつ開き幅 7.5 cm、上限 2.0 s）。
  run.json の `"reflex"` に `{"preset": "drop_only", "params": …, "overrides": {}}` が残ることを、起動の後に確かめる。
- 主な指標: 30 s の復帰（段階 4 と同じ定義。分母は誘発が 30 s より前に成立した試行、分子は 30 s までに成功）。60 s は副。
- 実験の名前: 予備 `S4RFXP`、本番 `S4RFX`。

## 3. 種の割り当て（帯 191500〜191699、200 個のうち 93 個を使う）

| 用途 | `--trials` | 種 | 試行の数（各条件） | 使う条件 |
|---|---|---|---|---|
| 予備 | `induced:191500:10` | 191500〜191509 | 10 | R1v3・P2S・切／入 |
| 本番の誘発 | `induced:191510:50` | 191510〜191559 | 50 | R1v3・P2S・切、R1v3・P2S・入、R1v3・P2・入、N1v3・P2S・入 |
| 本番の自然 | `natural:191560:33` | 191560〜191592 | 99（33 種 × 3 色） | R1v3・自然・切／入 |
| 予備の残り | なし | 191593〜191699 | — | 使わない（使うなら改訂で決める） |

- 誘発の 4 条件は、全く同じ `--trials` の指定にする。目標の色は並びの中の番号で決まる（`choose_targets`）ので、指定を変えると色が
  変わる。同じ指定なら、種・配置・目標の色・発動のしきい値（P2 の u）が 4 条件で同じになり、P2 と P2S も同じ点で発動する
  （`tests/test_s4_reflex.py` の `test_p2_and_p2s_fire_at_the_same_point_on_the_same_seed`）。
- 自然の試行は、誘発の種を使い回さず、別の 33 種にした。理由: (1) 自然の配置（`empty`・`home`）は誘発の配置と引き方が違い、同じ種でも
  同じ場面にならないので、使い回しても対の利点がない。(2) 種の台帳で 1 つの種を 1 つの用途に結べる。(3) 帯の中で足りる（93/200）。
- 帯を種の台帳（`docs/種の台帳.md`）に書くのは本線（このブランチでは触らない）。

## 4. 回す順とコマンド

作業場所 `C:\PAI\recovery_vla`。起動の前に、各コマンドに `--dry-run` を付けて回す本数を見る。

**予備（先に回す。判定には使わない）**
```
.venv\Scripts\python.exe scripts\96_s4_resume.py run --experiment S4RFXP --condition R1v3_P2S_off --model R1v3 --trials induced:191500:10 --induce P2S --mode naive --exec-interval 6 --no-safety
.venv\Scripts\python.exe scripts\96_s4_resume.py run --experiment S4RFXP --condition R1v3_P2S_on  --model R1v3 --trials induced:191500:10 --induce P2S --mode naive --exec-interval 6 --no-safety --reflex drop_only
```
予備で見ること（どれかが外れたら本番に進まず、本線に戻す）: 例外・監査（G1〜G3）の違反 0、P2S の成立が 10 本中 3 本以上、
入の腕で成立した試行のうち反射が発火しなかったものが 10% 以下、`info.pull` と `"reflex"` が記録に残る。

**本番**（この順で 1 本ずつ。同じ環境の区切りで回す）
```
.venv\Scripts\python.exe scripts\96_s4_resume.py run --experiment S4RFX --condition R1v3_P2S_off --model R1v3 --trials induced:191510:50 --induce P2S --mode naive --exec-interval 6 --no-safety
.venv\Scripts\python.exe scripts\96_s4_resume.py run --experiment S4RFX --condition R1v3_P2S_on  --model R1v3 --trials induced:191510:50 --induce P2S --mode naive --exec-interval 6 --no-safety --reflex drop_only
.venv\Scripts\python.exe scripts\96_s4_resume.py run --experiment S4RFX --condition R1v3_P2_on   --model R1v3 --trials induced:191510:50 --induce P2  --mode naive --exec-interval 6 --no-safety --reflex drop_only
.venv\Scripts\python.exe scripts\96_s4_resume.py run --experiment S4RFX --condition N1v3_P2S_on  --model N1v3 --trials induced:191510:50 --induce P2S --mode naive --exec-interval 6 --no-safety --reflex drop_only
.venv\Scripts\python.exe scripts\96_s4_resume.py run --experiment S4RFX --condition R1v3_nat_off --model R1v3 --trials natural:191560:33 --mode naive --exec-interval 6 --no-safety
.venv\Scripts\python.exe scripts\96_s4_resume.py run --experiment S4RFX --condition R1v3_nat_on  --model R1v3 --trials natural:191560:33 --mode naive --exec-interval 6 --no-safety --reflex drop_only
.venv\Scripts\python.exe scripts\96_s4_resume.py score --experiment S4RFX --condition R1v3_P2S_off R1v3_P2S_on R1v3_P2_on N1v3_P2S_on R1v3_nat_off R1v3_nat_on --at 30,45,60
```
止まったら同じコマンドで続きから回す（96 の決まり）。時間の目安: 誘発 200 本 × 約 115 s ≈ 6.4 h、自然 198 本 × 約 35 s ≈ 1.9 h、
予備 20 本 ≈ 0.6 h。合わせて約 9 h（1 本ずつ）。

## 5. 解析（結果を見る前に決める）

1. **主の検定**: R1v3・P2S の切と入で、両方とも 30 s より前に成立した種の対（鍵は種と目標の色）を作り、30 s の成否の食い違い
   （入だけ成功 b、切だけ成功 c）で McNemar の正確な検定（両側）。効果は対の上での復帰の差 (b − c) / 対の数。副に、各腕の
   分母（それぞれで成立した試行）での復帰を Fisher の正確な検定で比べる。
2. **発火しなかった落下**: 入の 3 条件（R1v3・P2S、R1v3・P2、N1v3・P2S）それぞれで、誘発が成立した試行のうち、誘発の発動から
   1.0 s 以内に反射が発火しなかった割合（試行の json の `"reflex".events` と `induce.t_fire` から）。
3. **自然な試行**: 成功率の差（入 − 切。点推定、全 99 本が分母）と、対の食い違い。誤発火＝入の腕で、反射が発火した試行のうち、
   真値で「発火の前 0.3 s に立方体が指の間にあり、発火の後 0.4 s で 2 cm 以上離れた」に当たらないもの（オフラインの評価の
   `label` と同じ定義）。誤発火の割合＝誤発火のあった試行の数 / 99。
4. 記述: 発火の理由（collapse・open）、発火までの時間（誘発の発動から）、待ちの長さ、再開の塊の観測の時刻、再開の時の手先と
   立方体の水平の距離（真値）、P2S の `info.pull`（力積・接触力・腕のトルクの変化）。
5. **同じ種で回し直しても成否が変わる**ことに注意する: 段階 4 の S4K の K1 と K2（同じ種・同じ条件の 2 回）で、99 本中 7 本（7.1%）の
   成否が入れ替わった（成功 91 本と 88 本）。対の食い違いには、この揺らぎの分が入る。自然な試行の 99 対では、反射に関係なく
   7 本前後の食い違いが出うる。

## 6. 判定の決まり

**反射を採る**のは、次の全部を満たすとき。
1. R1v3・P2S の 30 s の復帰が、入で切より **20 ポイント以上** 高く（対の上の差）、McNemar で **p < 0.05**。
2. 入の 3 条件のどれでも、成立した落下の試行のうち反射が発火しなかったものが **10% 以下**。
3. 自然な試行の成功率の低下が **5 ポイント以下**（点推定。入 − 切 ≥ −5 ポイント）、かつ誤発火の割合が **1% 以下**。

**採らないときの次の手**
- 1 の差が **10 ポイント未満**: 止める位置と待ちを調べる（`stop=freeze` との違い、settle_s、再開の時の手先と立方体の距離）。
  データの生成には進まない。
- 1 の差が 10〜20 ポイント、または p ≥ 0.05: 判断を保留し、本線が決める（予備の残りの種を使うなら改訂で決める）。
- 2 または 3 が外れた: 外れた原因（発火しない落下の型、誤発火の場面）を記録の再生で調べて、反射の決まりを直す。

**学習データが要るか**（反射を採った場合）
- **データは要らない**のは、次の全部を満たすとき: 入の腕の R1v3・P2S の 30 s の復帰が **35% 以上**、かつ **再開の観測での d2 被覆が
  50% 以上**、かつ **残った失敗の多くが掴み直しの精度**による。
  - d2 被覆: 再開の観測（反射の待ちの後の最初の塊の観測）が、学習データの落下の実演の始まりの範囲に入る割合。定義と道具は本線の
    d2 の解析に従う（**登録の前に本線が定義をここに書き込む**）。
  - 掴み直しの精度による失敗: 入の腕の失敗した P2S の試行のうち、方策が立方体の 2 cm 以内で指を閉じた（掴みに行った）が持ち上げられ
    なかったもの（真値で判定）。「多く」＝失敗の半分以上。
- それ以外は、**束 5 のデータを、反射を入れた実行で生成する**（反射の後の状態からの実演を集める）。

## 7. 検出力の見込み

段階 4 の P2 では、50 本のうち成立が約 24 本（48/100）だった。P2S も同じくらいと見込むと、対は約 24 本。
- 切 0〜5%・入 35〜40%（P2H の 10/25 と同じくらい）なら、食い違いは約 9 本でほぼ全部が入の側 → p ≈ 0.004。
- 入が 25% なら、食い違い 6:0 で p ≈ 0.031、5:0 で p ≈ 0.063。**20 ポイントちょうどの差は、見えるかどうかの境目**。
- 自然な試行の 99 対では、回し直しの揺らぎで 7 本前後の食い違いが出るので、5 ポイントの低下は点推定で見る（検定はしない）。

## 8. 変えないこと・してはいけないこと

- 結果を見た後に、条件・本数・種・引数・指標・判定の決まりを変えない（変えるなら改訂として掲示し、理由を書く）。
- 予備の結果は判定に入れない。
- テストの記録（`S4T1`）・`V3S3`・`outputs\sealed` は使わない。

## 9. 起動の前に書き込むもの

- 使う版の SHA（git の HEAD）: （起動の前に書く）
- d2 被覆の定義と道具: （本線が書く）
- 種の台帳への帯 191500〜191699 の登録: （本線が行う）
- 予備の結果の要約（第 4 節の見ること）: （予備の後に書く）
