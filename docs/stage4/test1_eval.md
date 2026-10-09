# テスト 1 の起動の入口と解析の版の照合（手順書）

- 状態: **道具だけ**。クラウドでは評価・GPU・API を使っていない（確かめたのは CPU のテストと、偽の保存点での `--dry-run` だけ）。
- 決まり: 登録版の事前登録 v1 `docs/stage4/prereg_test1_v1.md`（掲示板 0168。第 4・5・7・8・12・13-1・15 節）、判断の紙 `docs/stage4/d2_decision_sheet.md`（問 5・問 7・問 8）、掲示板 0155・0165・0166・0167。登録版を正とする。
- 道具:

| ファイル | 役目 | 新しい／直した |
|---|---|---|
| `scripts/98_s4_test1_eval.py` | テスト 1 の起動の入口（`plan`・`ckpt`・`run`・`rotate`・`health`・`layout`）。96 の run・task を包む | 新しい |
| `src/recovla/eval/test1.py`（A） | 版の照合 (b) と、案 B の G-P1 を同じ種の対で数える直し | 直した |
| `scripts/98_s4_test1_b.py`（B） | 同じ直しを別のコードで（recovla を import しない） | 直した |
| `scripts/98_s4_test1.py` | 条件をまたぐ点検に s4_gates.json の SHA-256（LF）と、HEAD の照合の「子が読み込むファイル」に入口を足した。例の layout・params を案 B の形に | 直した |
| `tests/test_s4_test1_eval.py` | 上の検査（CPU だけ） | 新しい |

既存のスクリプト（96・82・98_s4_b1・98_s4_b2・98_s4_d_rtc）・configs は変えていない。入口はそれらを読み込んだ写しの上で包む（82 の CKPT も写しに足す）。

## 1. 起動の手順（登録版 第 15 節）

前提: 束 2 の判定（`98_s4_b_decide.py b1`・`b2`、二重集計）→「枝の確定」の掲示（枝 1〜4、RTC の設定、`executor_v3.py`・RTC の設定・保存点 6 本・入口の SHA-256）。枝の値は引数 `--branch` で渡すが、束 2 の判定を引数で上書きしない（掲示の値をそのまま渡す）。

作業場所は `C:\PAI\recovery_vla`。下の `<枝>` は 1〜4、`<設定>` は枝 1・3 だけ（束 2 の判定の掲示の設定。関門 R の候補、ZEROS か range10_cap5 の見込み）。

```
:: 0. 版と保存点（読むだけ）
.venv\Scripts\python.exe scripts\98_s4_test1_eval.py ckpt                 # 6 本の SHA-256、executor_v3・rtc.py・入口・s4_gates.json（LF）
.venv\Scripts\python.exe scripts\98_s4_test1_eval.py plan --branch <枝> [--rtc-setting <設定>]

:: 1. smoke（44400〜44799 の未使用の種。台帳役が割り当てる。E7 は計画役の API を呼ぶ）
.venv\Scripts\python.exe scripts\98_s4_test1_eval.py rotate --branch <枝> [--rtc-setting <設定>] --allow-smoke ^
    --experiment S4SMOKE_T1 --trials <種>:<数> --dry-run
    （smoke の --trials は条件の形に合わせる: E7 は 44700:2、単発は induced:44700:2 / natural:44700:1。形の違う条件は
      --groups e7・--groups p1・--groups nat で分けて回す）

:: 2. 健全性の確認（190600〜190632、4 本 × 33。テスト 1 の直前。問 8）
.venv\Scripts\python.exe scripts\98_s4_test1_eval.py health --dry-run
.venv\Scripts\python.exe scripts\98_s4_test1_eval.py health --worker 1     （3 本なら --worker 2・3 も別の窓で）

:: 3. dry-run（全条件の何を飛ばし何を回すか。環境とドライバの表示）
.venv\Scripts\python.exe scripts\98_s4_test1_eval.py rotate --branch <枝> [--rtc-setting <設定>] --dry-run

:: 4. 3 本並行（同じコマンドを --worker 1・2・3 で 3 つ。--expect は「枝の確定」の掲示の値）
.venv\Scripts\python.exe scripts\98_s4_test1_eval.py rotate --branch <枝> [--rtc-setting <設定>] --worker 1 ^
    --expect R1v3=<SHA> --expect N1v3=<SHA> --expect R1v3s1001=<SHA> --expect N1v3s1001=<SHA> ^
    --expect R1v3s1002=<SHA> --expect N1v3s1002=<SHA> --expect executor_v3=<SHA> --expect rtc_module=<SHA> --expect entry=<SHA>

:: 5. 監視（ワーカーごと）
.venv\Scripts\python.exe scripts\96_s4_ops.py wait --progress outputs\s4\test1_eval\rotate_S4T1_b<枝>_w1.progress.json

:: 6. 二重集計（終わったら）
.venv\Scripts\python.exe scripts\98_s4_test1_eval.py layout --branch <枝> [--rtc-setting <設定>] --out outputs\s4\test1\
    （params.json の versions を「枝の確定」の掲示の値と見比べる。c4_on_time は 0157 の条件 4 の確認の後に書く）
.venv\Scripts\python.exe scripts\98_s4_test1.py check --layout outputs\s4\test1\layout.json --params outputs\s4\test1\params.json
```

- **試すときは必ず `--dry-run`**（付けないと GPU で評価を回し、E7 は計画役の API を呼び、帯の種を使う）。
- 止める合図: `outputs\s4\STOP`（全部）、`outputs\s4\test1_eval\rotate_S4T1.STOP`（テスト 1 の全ワーカー）、`rotate_S4T1_b<枝>_w<番号>.STOP`（そのワーカー）、`outputs\v2eval\S4T1\<条件>\STOP`。今の塊が終わってから止まる。
- 再起動で止まったら、サインインの後に同じコマンドで続きから（96 が完全な記録を飛ばし、書きかけを `_incomplete_<時刻>` に退避して同じ番号・同じ種で回し直す）。死んだワーカーの占有のファイルは、次に回るワーカーが外す。

## 2. 枝ごとの条件・試行数・時間

| 枝 | B1・B2（案） | 条件 | 試行 | プロセス時間 | 3 本並行（推測） |
|---|---|---|---|---|---|
| 1 | 両方採る（案 B） | E7 3 腕・P1 6 モデル・自然 6・RTC 3 | 2,240 | 72.2 | 約 25.3 h |
| 2 | B1 だけ（案 A） | E7 3 腕・P1 6・自然 6 | 1,842 | 63.9 | 約 22.4 h |
| 3 | B2 だけ（案 B） | P1 6・自然 6・RTC 3 | 1,790 | 31.3 | 約 11.0 h |
| 4 | どちらも採らない（案 A） | P1 6・自然 6 | 1,392 | 22.9 | 約 8.0 h |
| 健全性の確認 | — | 種 1001・1002 の 4 モデル × 33 | 132 | 約 1.6 | 約 35 分 |

`plan` が同じ値を出す（1 試行の値は登録版 第 5 節の表から割り戻した。丸めで ±0.1 ずれる）。

条件名（実験名 S4T1。`98_s4_test1.py example` の layout と同じ）:

| 組 | 条件 | 帯（`--trials`） | 実行 |
|---|---|---|---|
| E7（枝 1・2） | E7_R1v3_cur・E7_R1v3_v3・E7_N1v3_v3 | 160000:150（`test1_E7`） | 96 の task。v3.1・(a) なし（`98_s4_b1.py` の patch96）、計画役は 3 腕とも s4（Haiku 5.5）、1 手順 30 s・やり直し 1 回・全体 200 s、6 行・安全フィルタなし |
| P1（全部） | P1_R1v3・P1_N1v3・P1_R1v3s1001・P1_N1v3s1001・P1_R1v3s1002・P1_N1v3s1002 | induced:162000:100（`test1_P1`） | naive・6 行・安全フィルタなし・60 s（主な採点は 30 s） |
| 自然（全部） | nat_R1v3・nat_N1v3 | natural:161000:66（198 試行） | 同上 |
| | nat_R1v3s1001・nat_N1v3s1001・nat_R1v3s1002・nat_N1v3s1002 | natural:161000:33（99 試行） | 同上 |
| RTC（枝 1・3） | nat_R1v3_rtc | natural:161000:66 | `--rtc-setting` の設定（`98_s4_d_rtc.py` の patch96。diag_NNNN.npz も同じ形） |
| | P1_R1v3_rtc・P1_N1v3_rtc | induced:162000:100 | 同上（試行の diag の誘発の欄は P1 に直す。B2 と同じ） |
| 健全性（S4T1HC） | health_R1v3s1001・health_N1v3s1001・health_R1v3s1002・health_N1v3s1002 | selection:190600:33（既定。1 種 1 色）か natural:190600:33（`--health-spec`） | 自然と同じ |

- 帯は `s4_gates.json` の `test1_*`・`seed_copy_health` と毎回照らし、表の指定と完全に一致するときだけ通す（`e7_band_extended` は偽）。本番の実験名は S4T1（健全性は S4T1HC）だけ。smoke は `--allow-smoke`・実験名 S4SMOKE*・44400〜44799 の中で X2（44404〜44423）と台帳の「使用済み」の外。
- 枝 1・3 では `--rtc-setting` が要り、枝 2・4 では渡すと止まる。

### 2-1 3 本並行（rotate のワーカー）

- 3 つのワーカーは同じ条件の表を分け合う。1 回の呼び出しで塊 1 つ（E7 5 種、P1 10 種、自然 11 種＝33 試行。登録版 第 5 節「組の並べ方」）を子プロセスで回す（模型は 1 つずつ）。
- 次に回す条件は、ほかのワーカーが回していない条件のうち、塊の番号がいちばん若いもの（同じなら E7 → P1 → 自然、条件の順）。同じ組（E7・P1・自然）の中で、1 つの条件だけが 1 塊より先に進むことはしない（登録版 第 8 節「交互の塊が崩れた」を起こさない）。回せる条件が無ければ 30 秒ごとに待つ（6 時間で止まる）。
- 止め方は `98_s4_b1.py` の rotate と同じ: 続けるのは子が 0 で終わるか `--max-new` で止まったときだけ。Ctrl+C・止める合図・エラーでは止まる。呼び出しで完全な試行が増えなければエラー。進み具合は `outputs\s4\test1_eval\rotate_S4T1_b<枝>_w<番号>.progress.json`（pid 付き、1 分ごとに心拍）と `.log.json`。
- 2 重に回さない: 条件ごとの占有のファイル `outputs\s4\test1_eval\claims\S4T1\<条件>.claim` と、96 の progress.json の pid。

## 3. 保存点の渡し方

- 82 の `CKPT` の写しに、束 3 の 4 本（R1v3s1001・N1v3s1001・R1v3s1002・N1v3s1002）を足す。R1v3・N1v3 は 82 の `CKPT` のまま（登録版 第 5 節の実行と同じことを確かめ、違えば止める）。
- 4 本は登録版 第 5 節の実行名の `checkpoints\020000\pretrained_model`。`outputs\s4\seed_wrap\postcheck_<実行名>.json` が通っていること（train_config_only_seed_and_names・same_dataset_fingerprint・exit_code 0・checkpoints_ok）。smoke の実行は `--run` でも使えない。
- `checkpoints\last` があれば、020000 と同じ中身か（同じ場所か同じ SHA-256）を確かめ、違えば警告を出す（使うのは 020000）。
- SHA-256 は `98_s4_b4_eval.py` と同じ `ckpt_digest`（相対パスの順に「相対パス<TAB>ファイルの SHA-256<LF>」をつないだもの）。`--expect 名前=<SHA>` と違えば起動しない。名前は 6 本のモデルと `executor_v3`・`rtc_module`・`entry`。
- 記録: 試行の json と run.json に `"t1"`（枝・条件・役＝layout の鍵・モデル・保存点のパスと SHA-256・RTC の設定と rtc.py の SHA-256・実行器・使ったファイルの SHA-256。s4_gates.json は LF にそろえた値）。E7 は `98_s4_b1.py` と同じ `"v3"`・`"b1"`（`executor_version`・`files_sha256`）。`resume_spec.json` に `t1_ckpt_sha256`・`t1_rtc`・`t1_entry_sha256`・`b1_*`（再開のとき違えば 96 が止める）。
- ドライバが 610.88 と違っても止めない（警告だけ。版は試行の env に残る。0166）。記録と今の環境（ドライバ・torch・CUDA・OS）が違えば止める（96 の `--accept-env-change` のときだけ進め、96 が環境の区切りを書く）。
- 禁止の引数（モデル・帯・実行のしかた・誘発・制限時間・計画役。rotate は `--max-new` なども）は省略形も前方一致で拒む（0166）。

## 4. 解析の版の照合 (b)（登録版 第 7 節 1、判断の紙の問 5）

A（`test1.py`）と B（`98_s4_test1_b.py`）が、別のコードで次を確かめる。満たさない条件があれば「未完」で、判定を出さない。

| 項目 | 条件ごと | 条件をまたぐ |
|---|---|---|
| (b-1) | 全試行の `t1.files_sha256["configs/s4_gates.json"]` が 7f2f651c…（0165） | 入口（`98_s4_test1.py`）が今のファイル（LF）を 7f2f651c… と照らす |
| (b-2) | v3 の腕: `v3.settings` が 1 種類、`b1.executor_version` が v3.1、`executor_v3.py` の SHA-256 が 1 種類で掲示の値。今の実行器の腕: `b1.executor_version` が current、v3 の欄なし | v3 の腕で `executor_v3.py` が 1 種類 |
| (b-3) | RTC の腕: 設定名（`t1.rtc.setting` と `diag.arm`）が 1 種類で掲示の値、rtc.py の SHA-256 が掲示の値。naive の条件に RTC の印がない | — |
| (b-4) | 保存点のパスが登録版 第 5 節のもの、SHA-256 が 1 種類で掲示の値 | 同じモデルの保存点が条件をまたいで同じ |
| (b-5) | 入口の SHA-256 が 1 種類で掲示の値。`t1.role` が条件と同じ | 入口の版が全条件で 1 種類。HEAD が 2 つ以上のときの「子が読み込むファイル」に入口・98_s4_b1.py・98_s4_d_rtc.py を足す |

- 掲示の値は params の `versions`（`entry_sha256`・`executor_v3_sha256`・`rtc_setting`・`rtc_module_sha256`・`ckpt_sha256`）。`98_s4_test1_eval.py layout` が今のファイルから計算した値で書くので、「枝の確定」の掲示の値と見比べてから使う。
- 照らすのは、`versions` があるか、どれかの試行に入口の印 `t1` があるとき。記録に `t1` があるのに `versions` が無ければ未完。どちらも無い（入口を通っていない合成の記録）ときは照らさず、1 枚に「版の照合 (b): 照らしていない」と書く（今までのテストの記録のため）。本番の記録は必ず入口を通るので、必ず照らされる。
- A と B の照合は、合否・版の要約（`checks.*.versions`）・`version_check.ok` を照らす。理由の文は実装ごとに違うので照らさない。

## 5. 案 B の G-P1（登録版 12-3、判断の紙の問 7）

- 改良版（R1v3＋RTC の設定）の P1 と段階 3 の構成（R1v3 naive）の P1（162000〜162099）を、**両方の腕で誘発が 30 s より前に成立した同じ種の対**で数える。差は (b − c)/対の数（b＝RTC だけ復帰、c＝naive だけ復帰）、区間は Newcombe の対あり。点推定（`guard_mode` が interval なら区間の下限）で −0.10 以上なら満たす。対が 0 なら満たさない。束 4 の `paired_guard`・`guard_pair` と同じ数え方。
- 1 枚（`face_switch.c2`）に `g_p1_pairs`・`g_p1_b`・`g_p1_c`・`g_p1_diff`・`g_p1_ci95` を出し、各腕の分母の差（対なし）を `g_p1_unpaired` に記述として並べる。
- 案 A（枝 2・4）は改良版と段階 3 の構成が同じ記録なので、構造上 0 で満たす（`g_p1_count` = structural）。

## 6. 手元で回す前に確かめる点

1. `ckpt` で 6 本の SHA-256 が出て、警告（`checkpoints\last` が 020000 と違う）が無い。値を「枝の確定」の掲示に写す。
2. `plan --branch <枝>` の試行数が登録版 第 13-1 節と同じ（2,240・1,842・1,790・1,392）。
3. smoke: E7 の 3 腕・P1・自然・RTC の 3 条件・種 1001・1002 のモデルの記録に `"t1"` が入り、E7 の v3 の腕に `"v3"`・`"b1"`（executor_version v3.1）、RTC の条件に `diag.arm` と diag_NNNN.npz がある。`resume_spec.json` に `t1_ckpt_sha256` がある。
4. smoke の記録に `layout`（`--experiment` を S4SMOKE_T1 にした layout に直す）→ `98_s4_test1.py check` を当て、止まる理由が帯・本数の違いだけ（版の照合の項目では止まらない）であることを確かめる（登録版 第 15 節 4）。
5. 同じコマンドをもう一度回すと、完全な記録を飛ばして何も回さない。1 つのワーカーを Ctrl+C して起こし直すと、続きから回る。
6. 3 つのワーカーを起こしたとき、同じ条件を 2 つが回さない（占有のファイル）。progress.json の calls の並びで、E7 の 3 腕が 1 塊以上離れない。
7. nvidia-smi のドライバ（610.88 と違えば警告が出るだけ）と、空きメモリ 12 GB 以上・学習が動いていないこと（登録版 第 5 節）。
8. 健全性の確認の色の扱い（下の G2）を掲示してから回す。

## 7. 食い違いと判断が要る点

| 番号 | 中身 | この道具の扱い |
|---|---|---|
| G1 | 依頼の文は種のモデルの保存点を `checkpoints/last/pretrained_model` と書いているが、登録版 第 5 節は `checkpoints/020000/pretrained_model`（2 万手） | 登録版を正として 020000 を使う。`last` があれば中身を照らし、違えば警告 |
| G2 | 健全性の確認（190600〜190632、4 本 × 33 試行）の色の扱いは「回す前に掲示」（登録版 第 4 節）で、まだ決まっていない | 既定は `selection:190600:33`（1 種 1 色、`choose_targets` で決まる色。33 試行）。`--health-spec natural:190600:33` なら 1 種 3 色（99 試行 × 4）。どちらかを掲示する |
| G3 | 3 本並行の分け方は登録版に無い（「3 本並行」と「塊ごとに交互」だけ） | ワーカーが条件を分け合い、同じ組で 1 塊より先に進まない作り（第 2-1 節）。E7 は 3 腕が別のワーカーで同時に回ることが多い（同じ種・同じ時期） |
| G4 | 版の照合 (b) を、入口の印も `versions` も無い記録では照らさない | 今までのテストの合成の記録のため。本番は必ず照らされる（第 4 節） |
| G5 | RTC の設定の「SHA-256」の定義が登録版に無い | 設定名と `src/recovla/diag/rtc.py` の SHA-256（束 2 の `b2.files_sha256` と同じ値）で照らす。設定の中身の SHA-256（`setting_sha256`）も記録に残す |
| G6 | `s4_gates.json` の `induced_denominator_rule` の文字列が `<= L` のまま（判断の紙 3 の 4） | 道具は 0155・登録版どおり `<`。記録だけ |
