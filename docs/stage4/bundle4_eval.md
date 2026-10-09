# 束 4 の評価の入口と二重集計（関門 2・関門 3・テスト 2）

- 状態: **道具だけ**。まだ何も回していない（評価・GPU・API は使っていない。確かめたのは dry-run と合成の記録のテストだけ）。
- 書いた日: 2026-10-09（掲示板 0165 の後。束 4 のデータ・学習・評価の結果はどれもまだない）。同じ日に点検の指摘と作者の判断（第 10 節の F1・F3〜F5）で直した。
- 決まり: 事前登録 v2 の草案（`docs/stage4/prereg_test2_v2_draft.md` 第 2〜5・7・12 節）、`configs/s4_gates.json` の `bundle4_gates`・`bands`、掲示板 0155（採点の定義・解析の入口の点検）・0165（置き損ねを足す）。どれも読むだけ。
- 道具（新しいファイルだけ。既存のファイルは変えていない）:

| ファイル | 役目 |
|---|---|
| `scripts/98_s4_b4_eval.py` | 評価の入口。`96_s4_resume.py run` を包む。`plan`・`ckpt`・`run`・`rotate`・`layout` |
| `src/recovla/eval/b4.py` | 集計の実装 A（pathlib・scipy・`recovla.eval.stats`） |
| `scripts/98_s4_b4_b.py` | 集計の実装 B（標準ライブラリだけ。os.listdir・自前の正確な二項検定・自前の区間） |
| `scripts/98_s4_b4_check.py` | A と B を同じ記録に当てて照らし、一致したときだけ判定の 1 枚（JSON・Markdown）を書く。条件をまたぐ点検（HEAD・台帳） |
| `tests/test_s4_b4_eval.py` | 合成の記録と偽の保存点での検査（CPU だけ） |

## 1. 順番と回し方

```
0. smoke（作者の PC。第 9 節）
1. D3        事前登録 v2 を登録する（関門 2 を回す前。草案の冒頭「登録の時期」）
2. 関門 2     rotate --phase G2 → layout → check（G2）→ 判定（テスト 2 へ進むか）
3. 学習       99_s4_train_b4.py R4・N4 --seed 1001（最後の開始 10/15 24:00）
4. 関門 3     rotate --phase G3 → layout → check（G3）→ 書き方（種 1 つの結果か）
5. テスト 2   rotate --phase T2 → layout → check（T2）→ H1・H2（Holm）・副次
```

作業場所は `C:\PAI\recovery_vla`。**試すときは必ず `--dry-run` を付ける**（付けないと GPU で評価を回し、帯の種を使う）。

```
.venv\Scripts\python.exe scripts\98_s4_b4_eval.py ckpt                                  # 保存点の対応と SHA-256 → outputs\s4\b4_eval\ckpt.json
.venv\Scripts\python.exe scripts\98_s4_b4_eval.py plan --phase G2                       # 条件・帯・見込みの時間
.venv\Scripts\python.exe scripts\98_s4_b4_eval.py rotate --phase G2 --dry-run           # 全条件の何を飛ばし何を回すか
.venv\Scripts\python.exe scripts\98_s4_b4_eval.py rotate --phase G2 --expect R4=<SHA-256> --expect N4=<SHA-256>
.venv\Scripts\python.exe scripts\96_s4_ops.py wait --progress outputs\s4\b4_eval\rotate_S4B4G2.progress.json
.venv\Scripts\python.exe scripts\98_s4_b4_eval.py layout --phase G2 --out outputs\s4\b4_eval\layout_G2.json
.venv\Scripts\python.exe scripts\98_s4_b4_check.py example --out outputs\s4\b4_eval\    # params.json の雛形
.venv\Scripts\python.exe scripts\98_s4_b4_check.py check --layout outputs\s4\b4_eval\layout_G2.json --params outputs\s4\b4_eval\params.json
```

- `rotate` は段階の全条件（部分 × モデル）を、同じ種で **10 種の塊ごとに交互に**回す（事前登録 v2 の草案 第 5 節「組の並べ方」。自然は 1 種 = 3 色なので 30 試行の塊）。腕の 1 回の呼び出しごとに子プロセスを起こすので、模型は 1 つずつしか持たない。
- 止め方・進み具合は `98_s4_b1.py` の `rotate_loop` と同じ。続けるのは子が `--max-new` で止まったときだけ。Ctrl+C・メモリ待ちの時間切れ・止める合図・エラーでは止まる。1 巡して完全な試行が増えなければエラー。止める合図: `<条件>\STOP`、`outputs\s4\STOP`、`outputs\s4\b4_eval\rotate_<実験>.STOP`。
- 再開: 同じコマンドで続きから回る（96 が完全な記録を飛ばし、書きかけを `_incomplete_<時刻>` に退避して同じ番号・同じ種で回し直す）。保存点が変わっていれば、96 の引数の控え（`resume_spec.json` の `b4_ckpt_sha256`）の照合で止まる。
- 1 条件だけ: `run --phase G2 --cond P1_R4 [--max-new N] [--dry-run]`。
- 96 の続きの引数（`--accept-env-change`・`--min-free-gb` など）は後ろに付ける。モデル・帯・実行のしかた・誘発・制限時間は入口が決めるので渡せない（渡すと止まる）。`--induc`・`--time-limit 30` のような省略形も前方一致で拒む（96 の引数の解析は変えていない）。
- 3 本並行にするなら、段階を部分で分けて別のプロセスにする（例: `rotate --phase T2 --parts P2` と `--parts P1,nat` と `--parts P3`。実験名は変えない）。進み具合は `rotate_<実験>_<部分>.progress.json` に分かれ、同じ組の rotate を 2 つ起こすと pid で拒む。同じ条件を 2 つのプロセスで回すことも 96 が拒む。止める合図は `rotate_<実験>.STOP`（全部）か `rotate_<実験>_<部分>.STOP`（その組だけ）。

## 2. 条件名と帯

条件名は `<部分>_<モデル>`（記録は `outputs\v2eval\<実験>\<条件>\`）。実行のしかたは段階 3 と同じ: naive、行動の区切り 6 行、安全フィルタなし、単発の試行 60 s（30 s の採点が主）、試行ごとに世界を作り直す（96 の既定）。誘発は `recovla.eval.induce.Inducer` のまま（関門 C の手を止める版は使わない）。

| 段階（実験名） | 部分 | 帯（`--trials`） | 試行 / モデル | モデル | 使い方 |
|---|---|---|---|---|---|
| 関門 2（S4B4G2） | nat | natural:192100:66 | 198 | R4・R1v3 | 守り 1 |
| | P1 | induced:192200:100 | 100 | R4・N4・R1v3 | 守り 2・3 |
| | P2 | induced:192300:100 | 100 | R4・R1v3 | 狙い |
| | P3 | induced:192600:100 | 100 | R4・N4・R1v3 | 記述だけ（0165） |
| 関門 3（S4B4G3） | P1 | induced:192400:100 | 100 | R4s1001・N4s1001 | R4 − N4 |
| | P2 | induced:192500:100 | 100 | R4s1001・R1v3s1001 | 向き |
| テスト 2（S4T2） | nat | natural:165000:66（`test2_natural`） | 198 | R4・R1v3 | 守り（副次） |
| | P1 | induced:166000:100（`test2_P1`） | 100 | R4・N4・R1v3 | 守り（副次） |
| | P2 | induced:167000:100（`test2_drop`） | 100 | R4・N4・R1v3 | H1・H2 |
| | P3 | induced:168000:100（`test2_misplace`） | 100 | R4・N4・R1v3 | 副次・族の外 |
| smoke（S4SMOKE*） | 全部 | 種 44680〜44683 だけ | — | — | `--allow-smoke --experiment S4SMOKE_B4 --trials induced:44680:2`（rotate の自然は `--trials-nat natural:44680:1`） |

- 帯の受け付け: 本番は上の指定と**完全に一致**し（種類も）、その段階の本番の実験名（S4B4G2・S4B4G3・S4T2）のときだけ。テスト 2 の帯は毎回 `s4_gates.json` の `test2_*` と照らし、関門 2・3 の帯は `bundle4_gate2`（192100〜192999）の中で互いに重ならないことを照らす（違えば止める）。192700〜192999 は使わない（草案 12-2）。
- smoke は `--allow-smoke`・実験名 S4SMOKE*・種 44680〜44683 だけ。この種はデータ生成の smoke（`97_s4_b4_data.py --smoke`）と**兼用**する（作者の判断 10/09。どちらも smoke で、学習・評価の本番に入らない）。

## 3. 保存点の渡し方

`82_v2_eval.py` の `CKPT` の表に R4・N4 は無い。入口は 82・96 を書き換えず、**読み込んだ 82 の写しの `CKPT` に、決めた保存点を足して**から 96 を呼ぶ（96 は `config.path(v82.CKPT[--model])` で読む）。すでに同じ名前で別のパスがあれば止める。

| モデル | 保存点 | 選び方 |
|---|---|---|
| R4・N4（種 1000）、R4s1001・N4s1001 | `outputs\s4\train_b4\train_<R4\|N4>s<種>_<日時>_*\checkpoints\020000\pretrained_model` | `outputs\s4\b4_wrap\postcheck_<実行名>.json`（`99_s4_train_b4.py --post-check`）が通った実行（train_config_ok・exit_code 0・checkpoints_ok）だけ。smoke（名前に `_smoke`）は入らない。通った実行が 2 つ以上なら止める（`--run R4=<実行名>` で選ぶ。記録に残る） |
| R1v3s1001 | `outputs\s4\train\train_R1v3s1001_*\checkpoints\020000\pretrained_model`（束 3） | `outputs\s4\seed_wrap\postcheck_<実行名>.json` が通ったもの（train_config_only_seed_and_names・same_dataset_fingerprint・exit_code 0・checkpoints_ok） |
| R1v3 | 82 の `CKPT["R1v3"]`（段階 3 の 2 万手） | 草案 第 5 節の実行 `train_R1v3_20261005-180404_20261005-180404` と同じことを確かめる |

- 保存点の SHA-256（`ckpt_digest`）: `pretrained_model` の中のファイルを相対パスの順に並べ、「相対パス<TAB>ファイルの SHA-256<LF>」をつないだ UTF-8 の SHA-256。ファイルごとの値も `ckpt` の出力に残す。
- 記録: 試行の json と run.json の `"b4"`（段階・部分・モデル・保存点のパス・実行名・後の点検の JSON・SHA-256・使ったファイルの SHA-256）、`resume_spec.json` の `b4_ckpt_path`・`b4_ckpt_sha256`。
- `--expect R4=<SHA-256>`: 掲示した値（事前登録 v2 の P-2）と違えば起動しない（草案 第 8 節「SHA-256 が掲示の値と違う → 起動しない」）。集計でも params の `ckpt_sha256` に書けば記録の値と照らす。
- **マニフェストの SHA-256 は評価の入口では見ない。** 照合は学習の側（束 4 の学習の入口 `99_s4_train_b4.py`）で行う。保存点はその学習の出力で、評価の入口は保存点の SHA-256 だけを照らす。ただし今の `99_s4_train_b4.py` が照らすのは「build のときの値（`outputs\s4\b4\data_b4.json` の `manifest_sha256`）と今のファイルが同じ」「`conversion.json` の写しと同じ」の 2 つで、**掲示した値（草案 P-1）とは照らしていない**（`--expect` に当たる引数が無い）。掲示の値との照合は、学習を起こす前に人が `dryrun_<名前>_s<種>.json` の `manifest.manifest_sha256` を P-1 と見比べる（2026-10-09 に確かめた）。

## 4. 環境の照合

- 96 は条件ごとに、完全な記録・前の回と今の環境（ドライバ・torch・CUDA・OS）を照らす。入口はそれに足して、回の始めに**段階の全条件**の完全な記録の env と今の環境を照らす。違えば止める（終了コード 3）。96 の `--accept-env-change` のときだけ進め、96 が環境の区切りを書く（報告で分ける）。
- **ドライバの版が 610.88（草案 第 5 節）と違うだけなら止めない**（作者の判断 10/09）。入口は警告を出して続ける。版は試行の `env.driver` に残り（試行の `b4.driver_expected` に決めた版）、集計の 1 枚の頭に「ドライバ: <版>（決めた版 610.88）」と、違えば**警告**が出る（`cross.driver`・`driver_warning`）。
- 集計（A・B）は条件の中の env が 1 種類、`env_segments` が 1 つ、段階の全条件で env が 1 種類であることを確かめる（満たさなければ未完）。

## 5. 判定の数え方（A・B とも。結果を見る前に決めた）

- 時刻の境界（掲示板 0155 の 1）: 誘発の成立は `t_established < L`（ちょうど L は入れない）、成功は `t_success <= L`。主は L = 30 s、60 s は副次、45 s は記述だけ。
- 割合: 誘発の条件は、L より前に誘発が成立した試行を分母にした成功（復帰）の割合。Wilson の 95% 区間。自然は 198 試行が分母。
- 対: 誘発は同じ種で両方とも L より前に成立した組（b = 前の腕だけ成功、c = 後の腕だけ成功）。自然は同じ（種、色）の 198 対。
- **守りの P1 の差は対で数える**（作者の判断 10/09。テスト 1 の `test1.p1_layer`・関門 1 の R − N と同じ）: 同じ種で両方の腕とも誘発が 30 s より前に成立した対の、成功の割合の差 (b − c) / 対の数。区間は Newcombe の対あり（方法 10、φ の補正あり）。1 枚に対の数（分母）と b・c を出す。

### 5-1 関門 2（`bundle4_gates.G2`、草案 12-2。判定は点推定）

| 項目 | 数え方 | 基準 |
|---|---|---|
| 守り 1 | 自然の成功 R4 − R1v3（198 対の対の差。Newcombe の対ありの区間を並べる） | ≥ −5 ポイント |
| 守り 2 | P1 の 30 s の復帰 R4 − R1v3（同じ種で両方とも成立した対の差。Newcombe の対ありの区間を並べる） | ≥ −10 ポイント |
| 守り 3 | P1 の 30 s の復帰 R4 − N4（同じく対の差） | ≥ +20 ポイント |
| 狙い | P2（落下）の 30 s の復帰の割合 R4 − R1v3（各腕の分母。草案 12-2 のまま。対の b・c も参考に並べる） | ≥ +15 ポイント |
| 置き損ね | P3 の 30・60 s の復帰の割合（R4・N4・R1v3、Wilson。1 枚に k/n を出す） | 記述だけ（判定に使わない） |

- 結論（`decision.next`）: 守りを全部満たし狙いに届けば `test2`（テスト 2 へ）。守りを割れば `research_guard`（研究の道で原因を調べる。テスト 2 は回さない）。守りは満たすが狙いに届かなければ `no_test2`。
- **置き損ね（P3）がそろわなくても関門 2 は未完にしない**（作者の判断 10/09）。P3 の条件のうち、フォルダが無いもの（`optional.missing`）と、条件の点検か「判定に使う条件と合わせた条件をまたぐ点検」を満たさないもの（`optional.excluded`）は使わず、1 枚に注記する。使ったものだけ割合を出す（`p3.used`・`p3.not_used`）。

### 5-2 関門 3（`bundle4_gates.G3`、草案 12-3）

- 向き: P2 の 30 s の復帰の割合 R4s1001 − R1v3s1001 > 0（各腕の分母。狙いと同じ数え方）。params の `g3_with_r1v3s1001` が偽（束 3 の R1v3s1001 が無い。P-3）なら使わず、`direction_used = false` と書く。
- R4s1001 − N4s1001（P1 の 30 s の復帰）≥ +20 ポイント。関門 2 の守り 3 と同じく**対で数える**（関門 3 は関門 2 の規則を種 1001 で当てはめ直すものなので、同じ数え方にそろえた。作者の判断の外の当てはめ）。
- 満たせば `two_seeds`、満たさなければ `one_seed_diagnostic`（「種 1 つの結果」と明記し、診断として載せる）。書き方だけを決め、テスト 2 を回すかには使わない。

### 5-3 テスト 2（草案 第 2・3・7 節）

- H1: 落下（P2）の 30 s の復帰、R4 対 R1v3。H2: 同じく R4 対 N4。両側の正確な二項検定（b＋c 回、確率 0.5）。
- **Holm の族は H1・H2**（α = 0.05。params の `h2_in_family` が偽なら H1 だけ＝草案 D4）。成立 = Holm 補正後の p < 0.05 かつ b > c。
- **置き損ね（P3）は族に入れない**（掲示板 0165）: R4 対 R1v3・R4 対 N4 の正確な二項検定の p を「副次・族の外」と印を付けて出し、Holm の計算に入れない。各腕の割合に Wilson の区間。30 s と 60 s。
- 守り（副次）: 関門 2 と同じ 3 つを T2 の帯で、**同じ数え方**（P1 は対の差。草案 第 7 節 2「守りの差は Newcombe（対あり）」とも合う）。params の `guard_mode` が `point`（既定、草案 D2）なら点推定、`interval` なら Newcombe の対ありの区間の下限で判定する。
- 60 s の採点（副次・族の外）: H1・H2 と同じ形の比べ方を、分母を `t_established < 60` にして出す。
- 全条件の 30・45・60 s の割合（曲線の材料）。
- 1 枚の順（草案 第 7 節 5）: H1 → H2 → 守り → 置き損ね → 60 s → 曲線の材料（全条件の 30・45・60 s の割合の表）。

## 6. 解析の入口の点検（掲示板 0155 の 2 節。満たさない条件があれば「未完」で、判定を書かない）

条件ごと（A・B がそれぞれ別のコードで行う）:

1. run.json がある、G_AUDIT.json の met が真。
2. 本数と種の集合が計画（第 2 節の帯）と一致する（自然は（種、色）、誘発は種）。重なりがない。帯の外の種がない。
3. 試行の印がフォルダと計画に合う: 実験名・条件名、モデル名、誘発の種類、`b4` の段階・部分・モデル、実行のしかた（naive・6 行・安全フィルタなし）。
4. 制限時間が 60 s だけ（試行と run.json）、`env_segments` が 1 つ、試行の env（ドライバ・torch・CUDA・OS）が 1 種類。
5. 全試行に git の HEAD がある。全試行に使ったファイル（`b4.files_sha256`）があり、各ファイルの SHA-256 と保存点の SHA-256 がそれぞれ 1 種類。記録の `configs/s4_gates.json` の SHA-256 が掲示の値（改訂 4、掲示板 0165、`7f2f651c…`）と一致する。params に掲示した保存点の値があれば一致する。

条件をまたぐもの:

- A・B: 同じモデルの保存点の SHA-256 が条件をまたいで同じ。**使ったファイルの SHA-256（`98_s4_b4_eval.py`・96・82・`98_s4_b1.py`・`s4_gates.json`）が段階の全条件で 1 種類**（`cross.files_sha256`）。環境が段階の全条件で 1 種類。
- 入口（`98_s4_b4_check.py`、A・B が一致した後に 1 回だけ）: HEAD が 2 つ以上なら `98_s4_d_audit.py` の `compare_heads` で子が読み込むファイルが同じか（2-5。子のファイルに `98_s4_b4_eval.py` を足す。足すのは読み込んだ写しの `FAMILY_SCRIPT` だけで、`98_s4_d_audit.py` は変えていない）。**今の `configs/s4_gates.json`（LF にした SHA-256、`gate1.py` と同じ）と記録の値が掲示の値と一致**（`entry_audit.gates_sha256`）。台帳の照合の parse_errors が 0（2-6。`--ledger-json` は記録より新しいこと）。
- 関門 2 の置き損ね（P3）で使わなかった条件は、条件をまたぐ点検にも HEAD の照合にも入れない。

足りない条件（第 2 節の表の条件。関門 3 で `g3_with_r1v3s1001` が偽なら P2_R1v3s1001 は要らない。関門 2 の置き損ねは要らない）があれば未完。

## 7. 計算量（推測。1 試行の値は草案 第 5 節、置き損ねは 0165 の 0.030）

| 段階 | 試行 | プロセス時間 | 3 本並行の実時間 |
|---|---|---|---|
| 関門 2 | 1,196 | 約 24.6 | 約 8.6 時間 |
| 関門 3 | 400 | 約 9.6 | 約 3.4 時間 |
| テスト 2 | 1,296 | 約 27.3 | 約 9.6 時間 |

- `plan --phase <段階>` が同じ表を出す。`rotate` は 1 つのプロセスで条件を順に回すので、1 本だけで回すとプロセス時間がそのまま実時間になる（3 本並行は第 1 節の分け方で）。
- 学習中は評価を足さない（掲示板 0158・0164。`bundle4_protocol.md` 第 5 節）。

## 8. 出力の形

| ファイル | 中身 |
|---|---|
| `outputs\v2eval\<実験>\<条件>\` | 96 と同じ（trial_NNNN.json/.npz・runtime_NNNN.json・run.json・G_AUDIT.json・progress.json・resume_spec.json・resume_log.json）。試行の json と run.json に `"b4"` |
| `outputs\s4\b4_eval\ckpt.json` | `{"models": {名前: {path, run, postcheck, source, sha256, files_sha256, digest_rule, expected_sha256}}, "problems": {名前: 理由}}` |
| `outputs\s4\b4_eval\rotate_<実験>[_<部分>][_<モデル>].progress.json`・`.log.json` | rotate の進み具合（pid・status・done・total・counts・calls） |
| `outputs\s4\b4_eval\layout_<段階>.json` | `{"root", "phase", "experiment", "conds": {"<部分>.<モデル>": {"cond", "model"}}}` |
| `outputs\s4\b4_eval\result_<段階>.json`・`.md` | 判定の 1 枚（下） |

判定の JSON（A・B と同じ形。`98_s4_b4_check.py` が `entry_audit`・`double_count`・`layout`・`written` を足す）:

```
{"schema": "recovery_vla.s4_b4_result/1", "phase": "G2|G3|T2", "status": "complete|incomplete", "params": {...},
 "checks": {"<部分>.<モデル>": {"ok", "problems", "git_heads", "ckpt_sha256", "env", "files_sha256", "n"}},
 "missing_conditions": [...], "optional": {"used", "missing", "excluded"},
 "cross": {"ok", "problems", "models": {モデル: SHA-256}, "env_kinds", "files_sha256", "driver", "driver_expected", "driver_warning"},
 "result": null | G2: {"guards": {"natural", "p1_r4_vs_r1v3", "p1_r4_minus_n4"}, "guards_pass", "aim", "p3": {"label", "used", "not_used", "rates"}, "rates", "decision": {"pass", "next"}}
                | G3: {"direction", "direction_used", "p1_r4_minus_n4", "rates", "decision": {"pass", "wording"}}
                | T2: {"primary": {"H1", "H2"}, "holm": {"order", "m", "by"}, "secondary": {"guards", "guards_pass", "p3", "sixty", "rates"}},
 "entry_audit": {"versions", "gates_sha256", "ledger", "problems"}, "double_count": {...}}
```

- 割合: `{"k", "n", "rate", "wilson95"}`。対（誘発）: `{"pairs", "b", "c", "both", "neither", "p"}`（H1・H2 には `in_family`・`p_holm`・`established`）。対（自然）: `{"pairs", "x_k", "y_k", "x_only", "y_only", "diff", "newcombe95"}`。割合の差（狙い・関門 3 の向き）: `{"x", "y", "diff", "newcombe95", "threshold", "mode", "pass"}`。守りの P1・関門 3 の R − N（対の差）: `{"pairs", "b", "c", "both", "neither", "x_k", "y_k", "diff", "newcombe95", "threshold", "mode", "pass"}`。
- `optional` は関門 2 の置き損ね（P3）だけ。ほかの段階では空の 3 つの列。
- 一致の基準: 件数・真偽・文字列は完全一致、実数は相対 1e-9（p 値以外は 0 の近くだけ絶対 1e-12）。理由の文（problems）は実装ごとに違うので照らさない。
- 終了コード（check）: 0 一致して判定した、1 A と B が一致しない（何も書かない）、2 入力を読めない、3 未完。

params（`98_s4_b4_check.py example` が雛形を書く）:

| 鍵 | 既定 | 意味 |
|---|---|---|
| `guard_mode` | point | テスト 2 の守りを点推定で判定するか、区間の下限で判定するか（草案 D2）。関門 2 はいつも点推定 |
| `h2_in_family` | true | H2 を Holm の族に入れるか（草案 D4） |
| `g3_with_r1v3s1001` | true | 関門 3 の向きを R1v3s1001 で当てはめるか（草案 P-3） |
| `ckpt_sha256` | {} | 掲示した保存点の SHA-256（P-2）。書けば記録の値と照らす |
| `p_fill` | {} | 結果で埋まる所・作者の判断の値と出どころ（記録に残すだけ） |

## 9. 作者の PC の smoke で確かめる点

1. `ckpt`: R4・N4（学習の後）と R1v3 の保存点が決まり、SHA-256 が出る。R1v3 が `train_R1v3_20261005-180404_...` を指す。
2. `run --phase T2 --cond P2_R4 --allow-smoke --experiment S4SMOKE_B4 --trials induced:44680:2`（と P3・P1・nat）が最後まで回り、試行の json に `"b4"`（保存点のパス・SHA-256）、`model.name = R4` が入る。96 の `resume_spec.json` に `b4_ckpt_sha256` がある。
3. 同じコマンドをもう一度回すと、完全な記録を飛ばして何も回さない（再開）。途中で Ctrl+C して回し直すと、書きかけが退避されて続きから回る。
4. `rotate --phase G3 --allow-smoke --experiment S4SMOKE_B4G3 --trials induced:44680:2 --block 1` で、R4s1001・N4s1001・R1v3s1001 が交互に回り、progress.json に calls が並ぶ（関門 3 の学習の後）。
5. 環境の照合: nvidia-smi のドライバが 610.88 と出る。違えば警告が出て、止まらずに続く（試行の `env.driver` に版が残る）。
6. smoke の記録に `layout`（`--experiment S4SMOKE_B4`）→ `98_s4_b4_check.py check` を当てると、帯の本数が計画と違うので「未完」（終了コード 3）になり、A と B の点検の結果が一致する。
7. 1 試行の実時間が見込み（P2 約 1.6 分、P3 約 1.8 分のプロセス時間）から大きくずれていない。

## 10. 作者の判断（10/09 に決まった）

| 番号 | 決めること | 決定 |
|---|---|---|
| F1 | 守り 2・3 の差を「各腕の分母の割合の差」で数えるか、「同じ種で両方成立した組の差」で数えるか | **対で数える**（両方の腕で誘発が成立した同じ種の対。テスト 1 の `test1.p1_layer`・関門 1 の R − N と同じ）。テスト 2 の守りも同じ。関門 3 の R4s1001 − N4s1001 も守り 3 にそろえた。狙い（落下）と関門 3 の向きは草案 12-2 のとおり各腕の分母のまま |
| F2 | 関門 3 の「向きが正」を、落下の 30 s の復帰の割合の差 > 0（R4s1001 − R1v3s1001）とするか | そうする（草案 12-3）。R1v3s1001 が無いときは R4s1001 − N4s1001 だけ |
| F3 | ドライバ 610.88 以外で止めるか | **止めない**。版を試行の記録と 1 枚に残し、610.88 と違えば 1 枚に警告として出す。記録と今の環境の食い違い（途中で変わった）は今までどおり止める（96 の `--accept-env-change` で進め、報告で分ける） |
| F4 | 関門 2 の置き損ね（P3）がそろわないときも、関門 2 を「未完」にするか | **未完にしない**。欠け・使わなかった条件は 1 枚に注記する |
| F5 | smoke の種 44680〜44683 を、データ生成の smoke（`97_s4_b4_data.py --smoke`、同じ種）と兼ねてよいか | **兼ねてよい**（どちらも smoke で、学習・評価の本番に入らない。台帳は作者が書く） |
