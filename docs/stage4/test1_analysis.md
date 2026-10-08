# テスト 1 の解析（独立な 2 つの実装と照合）

テスト 1（帯 160000〜162099）を回し終えたら、すぐ判定できるように先に作った解析の道具の説明。決まりは `docs/stage4/prereg_test1_v1_draft.md`（事前登録の案）、`docs/目標書_段階4.md`、`configs/s4_gates.json`、掲示板 0155・0157・0158。どれも読むだけで、この道具は書き換えない。

## 1. 道具

| ファイル | 役目 |
|---|---|
| `src/recovla/eval/test1.py` | 実装 A。scipy（binomtest・norm）、`recovla.eval.stats`（Wilson・Newcombe の対あり）、`scripts/56_intervention_s3.py` の `count_run`（E7 の介入・success@k・時間）、numpy の中央値、pathlib の glob |
| `scripts/98_s4_test1_b.py` | 実装 B。標準ライブラリだけ（recovla・scipy・numpy を import しない）。os.listdir と正規表現、整数の組み合わせの和の正確な二項検定、statistics.NormalDist の z、Newcombe の式・Holm・中央値・入口の点検・56 の介入の定義を書き下した |
| `scripts/98_s4_test1.py` | 入口。`check` は A と B を同じ記録に当て、一致したときだけ判定の 1 枚（JSON と Markdown）を書く。`example` は layout・params の雛形を書く |
| `tests/test_s4_test1.py` | 合成の記録で、手計算の p 値、A と B の一致、Holm の順、分母の境界、未完、0157 の条件を確かめる |

```
.venv\Scripts\python.exe scripts\98_s4_test1.py example --out outputs\s4\test1\
（layout.json・params.json を事前登録の値に直す）
.venv\Scripts\python.exe scripts\98_s4_test1.py check --layout outputs\s4\test1\layout.json --params outputs\s4\test1\params.json
```

終了コード: 0 一致して判定した / 1 A と B が一致しない（判定を書かない）/ 2 入力を読めない / 3 未完（入口の点検を満たさない。A・B の点検の結果は一致）。

## 2. 入力

### 2-1 layout（記録の置き場所）

`root`（既定 `outputs/v2eval`）と `experiment`（案 S4T1）の下の条件のフォルダの名前。各項目は条件名の文字列か、`{"cond": 条件名, "model": モデル名}`（モデル名を書けば試行の json の `model` と照らす）。

| 鍵 | 中身 | 記録 |
|---|---|---|
| `e7.v3`・`e7.cur`・`e7.n1v3_v3` | E7 の 3 腕（R1v3＋v3、R1v3＋今の実行器、N1v3＋v3） | `run_NNNN.json`（96 の task。`run_NNNN_runtime.json` があれば LLM の呼び出しの数に使う） |
| `p1.<種>.R`・`p1.<種>.N` | P1 の R・N。種は `1000`（R1v3・N1v3）・`1001`・`1002` | `trial_NNNN.json`（96 の run） |
| `natural.<種>.R`・`natural.<種>.N` | 自然の R・N | `trial_NNNN.json` |
| `rtc.natural`・`rtc.p1` | RTC の腕（P-3 で B2 を採った場合。R1v3＋RTC の設定。P1 は案 A で 50 本、案 B で 100 本） | `trial_NNNN.json` |
| `rtc.p1_n` | 案 B だけ: N1v3＋RTC の設定の P1（100 本）。案 B の H3 の N の腕 | `trial_NNNN.json` |

### 2-2 params（結果で埋まる所と作者の判断）

| 鍵 | 意味 | 事前登録の案 |
|---|---|---|
| `h1` | H1 と E7 の 3 腕を置くか。**既定なし（必ず書く）** | P-1（束 2 B1 の採否） |
| `rtc_arm` | RTC の腕を置くか。**既定なし（必ず書く）**。真なら `rtc.natural`・`rtc.p1` が要る | P-3（束 2 B2 の採否） |
| `e7_n` | E7 の種の数（100・150・200 のどれか。ほかの値は読まない） | P-4 |
| `e7_band_extended` | 160150〜160199 を使えるか | D1 |
| `plan` | 改良版の案（A: 実行器だけ／B: 単発に RTC の設定も）。B は `rtc_arm` が真のときだけ | D3・P-6 |
| `guard_mode` | 条件 2 の守りを点推定（point）か区間（interval）で判定 | D6 |
| `ni_margin` | RTC の非劣性の余裕（既定 0.10） | D5 |
| `h2_layers` | H2 の層（既定 1001・1002） | P-7（間に合わなかった種は外す） |
| `rtc_p1_n` | RTC の P1 の本数。案 A は 50、案 B は 100 でなければ読まない（照らすために書く） | 第 4 節・第 12 節の案 B |
| `c4_on_time` | 0157 の条件 4（10/21 の値の確定までに判定が終わった）。人が確かめて真偽を書く。null なら切り替えの結論を出さない | 第 12-5 節 |
| `p_fill` | P-1〜P-10・D1〜D8 の値と出どころの控え（解析には使わず、結果の JSON にそのまま残す） | 第 13・14 節 |

## 3. 何をどう数えるか

### 3-1 入口の点検（掲示板 0155 の 2 節。満たさない条件が 1 つでもあれば「未完」で、検定・判定を出さない）

条件のフォルダごとに:

1. `run.json` がある。`G_AUDIT.json` があり `met` が真。
2. 本数と種（自然は種と色、P1 は種）の集合が計画と一致する: E7 は 160000〜160000+n−1、自然の種 1000 と RTC は 161000〜161065 の 3 色（198）、種 1001・1002 は 161000〜161032 の 3 色（99）、P1 は 162000〜162099（RTC の P1 は 162000 から `rtc_p1_n` 本）。同じ鍵が 2 回あれば未完。
3. 制限時間が 1 種類で計画どおり（単発 60 s、E7 は 1 手順 30 s・やり直し 1 回・全体 200 s）。run.json に `time_limits` があればそれも同じ。
4. 環境（0155 の 2-4）: run.json の `env_segments` がちょうど 1 つ（無い・2 つ以上なら未完。2 つ以上は区切りごとに分けて出す必要があるので、自動では判定しない）。試行の `env` のドライバ・torch・CUDA・OS（96 の ENV_STOP_KEYS）も 1 種類。
5. 版（0155 の 2-5、条件ごと）: 全試行に `env.git_head` があり、`diag` の `*_sha256` はそれぞれ 1 種類。条件ごとの HEAD の集合を `checks.<条件>.git_heads` に出す（A と B で照らす）。
6. 印（0155 の 2-3）: 単発の試行の `experiment`・`condition` がフォルダと同じ。モデル名（layout に書いたとき）、誘発（P1 は全部 `P1`、自然は誘発なし）、実行器（E7 の v3 の腕は試行の json に `v3` の欄があり、今の実行器の腕にはない＝`98_s4_b1.py` の記録の形）。
7. 帯: 種がテストの帯（`s4_gates.json` の `test1_E7`・`test1_natural`・`test1_P1`）の中。E7 の 160150〜160199 は `e7_band_extended` のときだけ。
8. 事前登録の案で回す条件が全部 layout にある: P1 と自然の種 1000 と H2 の層（R・N）、`h1` なら E7 の 3 腕、`rtc_arm` なら `rtc.natural`・`rtc.p1`、案 B なら `rtc.p1_n`。

条件をまたぐ点検（入口 `98_s4_test1.py` が、A と B が一致した後に 1 回だけ行う。満たさなければ未完で、検定・判定を消す。結果は `entry_audit`）:

- 2-5 版: 全条件の試行の HEAD が 2 つ以上なら、`98_s4_d_audit.py` の `order_heads`・`compare_heads`（束 1 と同じ「子が読み込むファイル」の定義。読み込むだけ）で、いちばん古い HEAD にあるファイルが後の HEAD で変わっていないことを git で確かめる。
- 2-6 台帳: `97_s4_ledger_check.py` の `build_report`（または `--ledger-json` の結果。点検する記録のどれよりも新しいこと）で parse_errors が 0。

### 3-2 主要評価項目（族は H1・H2・H3。H1 が無いときは H2・H3）

- 時刻の境界（0155 の 1）: 誘発の成立は `t_established < L`（ちょうど L は分母に入れない）、成功は `t_success <= L`。L = 30 s が主、45 s は記述、60 s は副。
- **H1**: E7 の `all_three_in_box`。同じ種で v3 と今の実行器の記録がそろった種の対。b = v3 だけ成功、c = 今だけ成功。
- **H2**: 層（種 1001・1002）ごとに、同じ種で R・N の両方とも誘発が 30 s より前に成立した組。b_s = R だけ 30 s 以内に復帰、c_s = N だけ。b = Σb_s、c = Σc_s。
- **H3**: 改良版の R と N（0157 の条件 1）の同じ数え方。案 A は種 1000 の naive の P1（`p1.1000.R` 対 `p1.1000.N`）、案 B は RTC の設定の P1（`rtc.p1` 対 `rtc.p1_n`、事前登録の案 12-2）。使った腕を `primary.H3.arms` に出す。案 B でも naive の種 1000 の比較は副次の `p1_layers_by_limit` に残る。
- H2 の層別の検定について: 帰無仮説の下では各層の b_s が Bin(b_s＋c_s, 0.5) に従うので、b＋c を条件にした Σb_s の正確な分布は Bin(b＋c, 0.5) になる（層ごとの分布の畳み込みと一致することを確かめた）。つまり、合計の食い違った組の McNemar の正確検定と同じ値で、層を分けても p は変わらない（層ごとの b_s・c_s は表に並べるだけ）。
- 検定: b＋c 回の二項（確率 0.5）の両側。p = 観測より偏った側の確率の 2 倍（上限 1）、b＋c = 0 なら 1。A は scipy の binomtest（確率 0.5 では同じ値になる）、B は整数の和。
- Holm（α = 0.05）: p の小さい順（同じ p は名前の順）。補正後の p は (m−i+1)·p(i) の累積の最大（上限 1）。逐次の棄却（p(i) <= α/(m−i+1)、止まった後は全部偽）も `holm.by.*.rejected_stepdown` に出す。
- 成立: 補正後の p < 0.05 かつ b > c。

### 3-3 副次・記述（族に入れない）

| 出力の鍵 | 中身 |
|---|---|
| `secondary.e7_arms.<腕>` | 3 個とも（Wilson の区間）、判定での成功（止まらず全手順の完了の判定が真）、平均の収納数、success@0・1・2、介入の種類別の数（台本の動き・出し直し・計画の変更・判定の上書き。56 の定義）、LLM の呼び出しの数（介入とは別）、全体の上限が効いた件数、シミュレーションの時間 1 時間あたりの成功、3 個そろった時刻の中央値（`all_three_in_box` が真の試行だけ、3 色の `truth_success_t` の最大。0155 の 2） |
| `secondary.e7_r_vs_n` | R1v3＋v3 対 N1v3＋v3（McNemar の正確検定、Newcombe の区間） |
| `secondary.p1_rates.<条件>.<L>` | 各モデルの復帰（分母はそのモデルで誘発が L より前に成立した試行）、Wilson の区間、成功時刻の中央値。L = 30・45・60 |
| `secondary.p1_layers_by_limit.<L>` | 種 1000・1001・1002 の 3 層の b_s・c_s と合計（記述。H2・H3 と別に並べるだけ） |
| `secondary.natural.<種>.<L>` | 自然の R 対 N（種と色の対、McNemar、Wilson、Newcombe）。L = 30・60 |
| `secondary.rtc` | RTC の腕があるとき: 自然の RTC 対 naive（198 対）の差と Newcombe の区間、非劣性（下限 > −`ni_margin`）、P1 の RTC 対 naive の復帰の差（対なしの Newcombe） |

区間: Wilson（単一の割合）、Newcombe の方法 10（対ありは φ の補正あり＝`recovla.eval.stats.paired_diff_ci` の既定、対なしは Wilson の区間をつなぐ形）。

### 3-4 掲示板 0157 の条件（事前登録の案 第 12 節）

| 条件 | 当てはめ |
|---|---|
| 1 | H3 が成立（補正後の p < 0.05 かつ b > c） |
| 2 | 案 A: 単発の試行の改良版の腕は段階 3 の構成と同じ記録なので、同じフォルダどうしで差を出す（構造上 0）。案 B: 自然（RTC 対 naive、198 対）の差 >= −0.05、P1（RTC 対 naive、各腕で 30 s より前に成立した試行が分母）の差 >= −0.10。`guard_mode` が interval なら区間の下限で判定 |
| 3 | H1 が成立。H1 が無ければ満たさない |
| 4 | `c4_on_time`（人が確かめる） |
| 切り替え | 1〜4 を全部満たすときだけ真。`c4_on_time` が null なら結論を出さない（`c1_to_c3` だけ出す） |

## 4. 出力（判定の 1 枚）

- JSON（既定 `outputs/s4/test1/result.json`）: `schema`（`recovery_vla.s4_test1_result/1`）、`status`（complete・incomplete）、`params`、`checks`（条件ごとの `ok` と理由）、`missing_conditions`、`primary`（H1〜H3 の組の数・b・c・p・補正後の p・成立。H2 は層ごとの表つき）、`holm`、`secondary`、`face_switch`、`double_count`（A と B の一致）、`layout`、`written`。
- Markdown（既定は JSON と同じ名前の `.md`）: 状態、主要評価項目の表（分母の書き方つき）、0157 の条件 1〜4 と切り替えの結論。未完なら、満たさなかった点検の一覧だけ。
- A と B の一致: 件数・真偽・文字列は完全に一致、実数（p・割合・区間・中央値）は相対 1e-9 以内（p 値は相対だけ。ほかは 0 の近くだけ絶対 1e-12）。入口の点検は条件ごとの `ok`・`git_heads` と足りない条件の一覧を照らす（理由の文は照らさない）。
- `entry_audit`: 条件をまたぐ点検（3-1 の 2-5・2-6）の結果。満たさなければ `status` は incomplete、終了コード 3。

## 5. 事前登録の案との対応

| 事前登録の案 | この道具 |
|---|---|
| 第 2 節 H1・H2・H3、Holm、両側の p の定義、b > c | 3-2 |
| 第 2 節 復帰の定義（成立 `<`、成功 `<=`） | 3-2（0155 の 1） |
| 第 3 節 E7 の内訳・E7 の R 対 N・自然の R 対 N・層別の値・60 s・45 s | 3-3 |
| 第 3 節 E7 の仕組みの指標（持ち上がり・+y）、方策自身の把持失敗からの復帰、時間ごとの曲線、段階 3 の数字との並べ | **この道具には入れていない**（`scripts/98_s4_d_e7.py summary`・`55_extra_s3.py`・`96_s4_resume.py score`・`time_scoring.py` で出す。二重集計が要るなら別に足す） |
| 第 4 節 帯と本数 | 3-1 の 2・6 |
| 第 5 節 制限時間・環境 | 3-1 の 3・4 |
| 第 6 節 RTC の非劣性 | 3-3 の `secondary.rtc`（躍度の Wilcoxon は入れていない） |
| 第 7 節 1 二重集計（ファイルの列挙も別） | A と B（1 節）と照合（4 節） |
| 第 7 節 2 検定・区間 | 3-2・3-3 |
| 第 7 節 9 切り替えの判定も二重集計 | 3-4（A と B の両方で出して照らす） |
| 第 12 節 条件 1〜4、案 A・B | 3-4 |
| 第 13 節 P-1・P-4・P-6・P-7、第 14 節 D1・D3・D5・D6 | params（2-2）。値は結果の JSON に残る |
| 0155 の 2 節 入口の点検 | 3-1（2-1〜2-5 は条件ごと、2-5 の HEAD の間の照らし合わせと 2-6 の台帳は入口が条件をまたいで 1 回。2-7 は A と B） |
| 第 13 節 P-3、第 4 節の RTC の本数 | params の `rtc_arm`・`rtc_p1_n`（案 A は 50、案 B は 100） |
