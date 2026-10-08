# 段階 4 の運用（束 0 運用役、2026-10-08）

段階 4 の評価を、夜間の停止（Windows Update の再起動・電源断）で 1 晩を失わずに回すための道具と手順。
道具は 3 つ。どれも作業場所 `C:\PAI\recovery_vla`、Python は `.venv\Scripts\python.exe`。

| 道具 | 役目 | GPU |
|---|---|---|
| `scripts\96_s4_ops.py` | wu-check（再起動の恐れ）、env-record（環境の記録）、backup-manifest／backup-verify（バックアップの一覧と照合）、wait（監視役） | 使わない |
| `scripts\96_s4_resume.py` | 82 の run・task を試行ごとに続きから回す包み。score（記録から 30・45・60 秒の採点） | run・task だけ使う |
| `scripts\96_s4_replay_check.py` | 記録した行動を開ループで再生し、止めて再開した試行と続けて回した試行のビット一致を確かめる（層 (i)、5-1 節） | 方策の推論はしない（シミュレーションとカメラの模型は動かすので、運用役だけが回す） |

82_v2_eval.py・configs は書き換えていない。96_s4_resume.py は 82 の cmd_run・cmd_task の本体を写しており、82 の SHA-256（`e36e12ed…2807`、タグ v3-s3-freeze の版）を固定して照合する。82 が変わったら写しを見直す。

## 1. 使い方

### 1-1 評価を回す（96_s4_resume.py）

```
.venv\Scripts\python.exe scripts\96_s4_resume.py run  --experiment S4K --condition K1 --model R1v3 --trials natural:190100:33 --mode naive --exec-interval 6 --no-safety
.venv\Scripts\python.exe scripts\96_s4_resume.py task --experiment S4E7 --condition speed --model R1v3 --trials 190150:6 --exec-interval 6
.venv\Scripts\python.exe scripts\96_s4_resume.py run  ... --dry-run      # 何を飛ばし何を回すかだけ見る（何も書かない）
```

- 引数は 82 と同じ。同じコマンドをもう一度打てば続きから回る（完全な試行は飛ばす）。
- 試行 i が「完全」とは、`trial_NNNN.json`・`.npz`・`runtime_NNNN.json`（task は `run_NNNN.json`・`.npz`・`run_NNNN_runtime.json`）が全部あり、json が読めて種・目標・番号が並びと合い、npz の CRC が通ること。どれかが欠けるか壊れていれば、その試行のファイルを `_incomplete_<時刻>\` に退避して同じ番号・同じ種で回し直す。完全な記録には触れない。
- 全部そろったときだけ `run.json` と `G_AUDIT.json`（82 と同じ gate）を書く。**途中の条件に 87_v2_e.py run を使わない**（87 run は run.json の無い条件を消して最初からやり直す）。段階 4 の実行は 96_s4_resume.py だけで回す。
- 新しい試行を始める前に、次を確かめる。
  - 止める合図: `<条件>\STOP` か `outputs\s4\STOP` があれば、今の試行を終えてから止まる（status=stopped、終了コード 1）。
  - 時間の窓: 既定では無い（01:45〜02:45 の窓は作者の決定 10/08 で必須から外した）。`--quiet-window HH:MM-HH:MM` を指定したときだけ、その窓では新しい試行を始めず窓の終わりまで待つ（`--ignore-quiet` は互換のため残し、指定した窓を無効にする）。
  - 空きメモリ: 物理 12 GB 未満かコミット 6 GB 未満なら待つ（運用の決まり「空きメモリ 12 GB 未満なら動かさない」。`--min-free-gb`・`--min-commit-free-gb`、120 分で諦める `--mem-timeout-min`）。
  - 束 1 のように 3 本を並行で回すときも、各プロセスは「今の空き ≥ 12 GB」だけを見る（空きをプロセス数で割らない）。1 プロセスの作業セットは最大 3.83 GB（5 節）なので、ほかのプロセスが試行の途中で増える分は 1 本あたり 4 GB 弱。3 本目が始まる時点で空きが 12 GB あれば、全部が最大になっても 4 GB 以上残る見込みで、この仕組みで足りる。ほかの重い処理（学習・別の評価）は重ねない（前例: 評価 3 本と別の処理を重ねて落ちた）。
- 同じ条件を 2 つのプロセスで回そうとすると止める（前の progress.json の pid が生きていれば終了コード 3）。
- **環境の記録（目標書_段階4.md 5-1・11 節 5、prereg_template 8 節）:** 回の始めに nvidia-smi のドライバ、torch・CUDA、OS（CurrentBuildNumber.UBR）、git の HEAD を読み、`resume_log.json` の session の `env`、`progress.json` の `env`、`run.json` の `env`・`env_segments`、各試行の json の `env`（要約）に書く。
  - ドライバ・torch・CUDA・OS の版が、完全な記録（`env` の無い記録を含む）か前の回と違えば、何も回さずに終了コード 3 で止まる。ドライバの版を読めないときも止まる（記録のない実行は報告に使わない）。
  - 意図して続けるときだけ `--accept-env-change` を付ける。その回の session に `env_segment` が書かれ、試行の `env` で前後を分けて報告する（別の条件名で回し直す方が単純）。
  - git の HEAD の違いでは止めない（記録だけ。メインのコミットで変わるため）。
- **世界の作り直し（既定）:** 試行ごとに WorldRig・SensorSuite を作り直す（方策の模型は使い回す）。82 と同じ使い回しは `--reuse-world`。理由は 5 節の行動の再生の確かめ（使い回すと、止めて再開した試行が続けて回した場合とビット一致しない）。試行の json・`run.json`・`resume_spec.json` の `world_per_trial` に残る。
- **計画役（task。目標書_段階4.md 第 10 節 17）:** 既定 `--planner s4` は `src\recovla\planner\decompose_s4.py`（claude-haiku-5-5、温度は送らない、thinking disabled、鍵にモデル名を入れたキャッシュ）。`--planner legacy` は凍結の `decompose.py`（Haiku 4.5、温度 0）。82 は書き換えず、写した make の中で TaskRuntime に渡す関数だけを選ぶ。使った計画役は `run.json`・`resume_spec.json` の `planner`、試行の json の `planner` と `plan.planner_variant`（s4 のとき）に残る。
- 終了コード: 0 全部そろった、1 途中で止まった（合図・`--max-new`・メモリ待ちの時間切れ・Ctrl+C）、2 エラー、3 引数・前提の食い違い（制限時間・環境の食い違い、ドライバの版が読めない、を含む）。
- 書くもの（`outputs\v2eval\<実験>\<条件>\`）: 82 と同じ試行の記録、`run.json`、`G_AUDIT.json`、`progress.json`（進捗）、`resume_spec.json`（引数の控え。再開で食い違えば止める）、`resume_log.json`（回ごとの環境・飛ばした本数・回した試行・退避したファイル・メモリの最大）。

### 1-2 制限時間（単発は 60 秒統一 10/08 01:59、E7 は段階 3 と同じ 10/08。どちらもユーザーの決定）

| 引数 | 既定 | 効くところ |
|---|---|---|
| run `--time-limit-s` | 60 | 試行の打ち切りと成功の時刻の判定（`src\recovla\harness\loop.py` 107・118 行） |
| task `--step-timeout-s` | 30 | 1 手順の持ち時間（configs の `planner.step_timeout_s` と同じ値を写しの辞書に重ねる）。やり直しは configs の `retry: 1` のまま。段階 3 と同じ |
| task `--task-time-limit-s` | 200 | 試行全体の打ち切り（`task_loop.py` の既定と同じ。段階 3 と同じ） |

- 段階 3 の E7 は 20 本とも全体の打ち切りなし、t_end の最大 136.3 s（`outputs\v2eval\V3S3\E7_R1v3\run_*.json` の timed_out・t_end）。理由: 段階 3 の 6/20 とそのまま比べるため。止まる主な原因は 2 番目の手順で動けなくなることで、時間切れではない。
- 経緯: 改訂 1 では手順 60 s・全体 480 s を既定にしていた（S4SMOKE60\task_e7 はこの値で回したので、今の既定のまま続きを回すと制限時間の食い違いで止まる）。
- 使った値は試行の json の `"time_limits"`（run は 82 と同じ `"time_limit_s"` も）、`run.json` の `"time_limits"`、`progress.json`、`resume_spec.json` に残る。
- 既にある完全な記録の制限時間が今回と違えば、混ぜずに止める（終了コード 3）。段階 3 の V3S3 の条件に既定のまま当てると、30 と 60 の食い違いで止まる（確かめ済み）。

### 1-3 30・45・60 秒の採点（score）

```
.venv\Scripts\python.exe scripts\96_s4_resume.py score --experiment S4K --condition K1 K2 --at 30,45,60 [--out outputs\s4\score_K.json]
```

- 主な指標は 30 s の採点（作者の決定 10/08）。60 s の採点と時間ごとの曲線は副次、45 s は記述だけ。
- run の試行: 「success かつ t_success ≤ T」を T 秒の成功とする。制限時間が効くのは打ち切りと成功の時刻の判定だけで、打ち切りの前の経過は制限時間によらないので、60 秒の記録から 30・45 秒の採点が出せる。t_success（シミュレーションの時刻）は試行の json に残る。
- 誘発の試行の分母: 誘発が T 秒より前に成立した試行（`induce.t_established < T`。ちょうど T 秒の成立は入れない）だけ。成立しなかった・T 秒ちょうど以後に成立した試行は、その T の分母にも分子にも入れない（段階 3 は 30 s で打ち切ったので成立は 29.9 s 以前だけ。30 s の採点が段階 3 と同じ定義になる。批判役の指摘と掲示板 0155 の 1-1）。成功の側（`t_success ≤ T`）は変えない。出力の `at[T]` は `successes`（分子）・`n`（分母）・`n_trials`（記録の本数）・`denominator`（分母の定義）。
- 成功の時刻の分布（時間ごとの成功率の曲線）は `t_success` の並びから描く。誘発の条件の曲線は、各時刻 t の分母を「t より前に成立した試行」にして描き、図に分母を書く（0155 の 1-3。`scripts\98_s4_time_report.py`。記録に誘発があるのに `--induced` が無い・自然と誘発が混ざっている・制限時間や環境が混ざっている場合は数えずに終了コード 2）。
- 成否以外の指標（接触・巻き添え・段階など）は試行全体（最長 60 s）で数えた値なので、30 秒の版は npz を 30 s で切って数え直す必要がある（score は成否だけを出す）。
- task は、段階 3 と同じ 1 手順 30 s・全体 200 s で回すので、記録の 3 個とも（真値。`all_three_in_box`）がそのまま段階 3 と同じ定義の採点になる。score は `all_three_in_box`（主）と `timed_out`（全体の打ち切りが効いた本数）の本数を出す。時間ごとの曲線は、`all_three_in_box` が真の試行についてだけ、真値の 3 色の成功の時刻の最大 ≤ T を「T 秒までに 3 個そろった」とする（0155 の 1-2。途中で箱から出した試行を曲線で成功に数えない）。
- 制限時間より長い T を頼むと警告を出す。制限時間が 2 種類以上・環境の区切り（`env_segments`）が 2 つ以上なら `problems` に書き、終了コード 1（区切りごとに分けて出す）。

### 1-3-1 解析の入口の点検（98_s4_d_audit.py。掲示板 0155 の 2 節）

```
.venv\Scripts\python.exe scripts\98_s4_d_audit.py [--only <条件の id> ...] [--out outputs\s4\audit\bundle1_audit.json]
```

- 束 1 が終わったら、解析の前に全条件で回す（記録を読むだけ）。計画（`docs\stage4\bundle1_defs\bundle1_plan_queue.json`）の各条件のフォルダで、run.json と G_AUDIT（met=true）、本数と種の集合、試行の diag とフォルダ名、制限時間が 1 種類、env_segments が 1 つ、git の HEAD と各 SHA-256（HEAD が違えば子が読み込むファイルの中身を git で照らす）、96 の score との件数の一致を確かめる。
- 1 つでも欠ければ終了コード 1 で、欠けた条件ごとに呼び直すコマンドを出す。満たさない条件の結果は報告に使わない。

### 1-4 監視役（wait）

```
.venv\Scripts\python.exe scripts\96_s4_ops.py wait --progress outputs\v2eval\S4K\K1\progress.json [ほかの progress.json ...] [--log <ログ>]
```

- メインが背景で動かし、終わったら通知を受ける。終了コード: 0 完了、1 途中で止まった（合図）、2 エラー、3 異常終了（状態が最終にならないままプロセスが消えた＝再起動・強制終了）、4 progress.json の更新が 30 分途絶えた、5 時間切れ。
- PID だけを見る場合は `--pid N`。

### 1-5 再起動の恐れ（wu-check）と環境の記録（env-record）

```
.venv\Scripts\python.exe scripts\96_s4_ops.py wu-check      # 確認用（任意）。終了コード 0 なし、1 再起動待ち、2 読めない、3 印はないが入っていないドライバ等あり
.venv\Scripts\python.exe scripts\96_s4_ops.py env-record    # outputs\s4\env_record.json（前の記録は env_record_history.jsonl に残す）
```

- wu-check は読むだけ（レジストリの RebootRequired・RebootPending・PendingFileRenameOperations〔Edge の一時ファイルは無視〕、COM の更新の履歴 7 日〔ドライバは driver_updates_7d に分ける〕、入っていない更新〔端末の控えだけ、ネットに出ない〕、System イベントの電源の記録、最後の起動、参考に 01:45〜02:45 までの分数）。`outputs\s4\wu_check.jsonl` に 1 行足す。
- env-record は nvidia-smi のドライバ、OS、Python・torch・lerobot・mujoco の版、git の HEAD とタグ v3-s3-freeze の SHA、メモリ、主なスクリプトの SHA-256 を残す。Windows Update の履歴の NVIDIA の行と V3S3 の記録の時刻から、段階 3 の時点のドライバを推す。

| 項目 | 段階 3（V3S3、10/06 21:34〜10/07 10:05） | 段階 4（10/08 02:01 以降） |
|---|---|---|
| NVIDIA ドライバ | 560.94（32.0.15.6094、2025-10-09 導入。推定） | 610.88（32.0.16.1088、10/08 02:01 導入） |
| torch | 2.11.0+cu126 | 2.11.0+cu126（CUDA は動作確認済み） |

出どころ: `outputs\s4\env_record.json` の gpu_driver_history（COM の QueryHistory）。09/16 に 591.86 の導入が 6 回失敗している（同じ表）。段階 4 の GPU の測定（物差し K を含む）は段階 3 と別の環境として扱う。

### 1-6 バックアップの一覧（backup-manifest）

```
.venv\Scripts\python.exe scripts\96_s4_ops.py backup-manifest [--force]
.venv\Scripts\python.exe scripts\96_s4_ops.py backup-verify --dest <コピー先> [--quick]
```

- 組: V3S3 の記録（ログ・キューを含む）、outputs\results、R1v3・N1v3 の 2 万手のチェックポイント、outputs\demo_v2、paper、docs\freeze。10/08 03:00 の一覧は 4,446 ファイル・3.17 GB、manifest_sha256 `528ea076…f946`（`outputs\s4\backup_manifest.json`）。
- **コピーは実行していない。置き場所は作者が決める。** 手順は一覧の `copy_instructions`（robocopy の例と照合）。paper は編集中なので、コピーの直前に `--force` で一覧を作り直す。
- backup-verify は読むだけ。作業場所そのものに当てると 4,446 ファイル全部一致（道具の確かめ）。

## 2. 毎晩の手順

時間帯を避ける規則（01:45〜02:45 に新しい試行を始めない、17:00 の確認）は、作者の決定（10/08）で必須から外した。再起動は再起動の要る更新が入った日だけで曜日の決まりがなく（過去 120 日で 10/02 と 10/08 の 2 回）、試行ごとに続きから回せるので回復できる。避けるほうが作業を縛る。

1. 実行の前に `env-record`（ドライバが変わっていないか）。`wu-check` は確認用（任意。再起動待ちかどうかを見たいときに回す）。
2. メインが評価を切り離して起動（`96_s4_resume.py run|task ...`）し、必ず監視役 `96_s4_ops.py wait --progress ...` を背景で付ける。
3. 朝: wait の要約を見る。0 なら集計へ。3（異常終了＝再起動・強制終了）なら 3 節の手順で、サインインの後に `96_s4_resume.py` の同じコマンドで続きから回す（切れた試行は `_incomplete_*` に退避して同じ種で回し直す）。
4. 1 日の終わりに引き継ぎのメモを更新する。

## 3. 再起動の前後の手順（10/08 の例）

**起きたこと（10/08）:**
- 01:38 の確認で RebootRequired・RebootPending は両方 False。
- 01:58〜02:00 に smoke（S4SMOKE\stop、natural:44400:1 の 3 本）を回していた。
- 02:00:02 に MoUsoCoreWorker が再起動を始め（System イベント 1074）、02:01:08 に NVIDIA のドライバ 32.0.16.1088 が入った（COM の履歴）。最後の起動は 02:01:31。サインインは 02:38（人の手。自動サインインなし）。
- 試行 0・1 は完全、試行 2 は 02:00:01 に始まったところで切れた（ファイルは何も残らなかった）。progress.json は status=running のまま残った。

**再開の手順:**
1. サインインしたら `wu-check` と `env-record`。ドライバが変わっていれば、以後の GPU の測定は別の環境として記録する（10/08 は 560.94 → 610.88）。96_s4_resume.py は、前の回とドライバ・torch・OS の版が違えば終了コード 3 で止まる（1-1）。そのときは、別の条件名で最初から回すか、`--accept-env-change` で env_segment として分けて続けるかを決める（物差し K の 2 回の途中なら、決まりの文書どおり止めて分けて報告する）。
2. `wait --progress <その条件>\progress.json` を 1 回動かす。プロセスが消えているので outcome=crashed・終了コード 3 が出る（10/08 の S4SMOKE\stop で確かめた）。
3. 同じコマンドに `--dry-run` を付けて、飛ばす試行と回す試行を見る（S4SMOKE\stop: 完全 2 本、回す 1 本〔missing:json,npz,runtime〕）。
4. 同じコマンドで回す。切れた試行だけが同じ種で回り直る。完全な試行のファイルは SHA-256 が変わらない（S4SMOKE\stop で試行 0・1 の 6 ファイルとも一致を確かめた）。
5. 回し直した本数は resume_log.json の sessions に残る。報告の「回し直し」の欄に使う（回し直しは、落ちた試行を同じ種で回す場合だけ＝共通の評価の作法 6）。

**再起動の前（予告がある場合）:** `<条件>\STOP` か `outputs\s4\STOP` を置くと、今の試行を終えてから止まる。全部止めたら `wait` が 1 を返す。再起動の後は STOP を消してから同じコマンドで再開する。

## 4. 確かめ方の 3 層（批判 #4）の運用での意味

GPU の推論は毎回少し揺らぐので、閉ループの試行の回し直しは成否・行動が一致しない（引き継ぎ 22 行）。そこで確かめ方を 3 層に分ける。

| 層 | 求めること | 運用での中身 |
|---|---|---|
| (i) 方策を通さない部分 | ビット一致 | 再開で完全な記録のファイルが SHA-256 で変わらない。**止めて再開した試行と、同じプロセスで続けて回した試行が、記録した行動の再生で真値の npz の全部の配列がビット一致する**（`scripts\96_s4_replay_check.py check`。5 節。これは試行ごとに世界を作り直す既定のときだけ成り立つ）。試行の並び（種・配置・目標）は 82 と同じ 41_results.trial_list から作る。バックアップは一覧の SHA-256 と一致。 |
| (ii) 新しいコードと凍結版 | 同じ入力で同じ出力（1e-5 以内） | 96_s4_resume.py は 82 の SHA-256 を固定して本体を写す。記録の欄は V3S3 と同じ（足したのは `time_limits` だけ）で、87・50_e_eval・recovla.eval.report・gate がそのまま読む。 |
| (iii) 閉ループの試行 | 分布として一致 | 回し直した試行は元の実行と一致を求めない。物差し K（同じ種で 2 回回した食い違い d0）を回し直しのぶれとして全報告に載せる。 |

## 5. 10/08 の確かめ（学習用の帯 44400〜、R1v3・naive・6 行・フィルタなし）

| 確かめ | 結果 | 出どころ |
|---|---|---|
| (a) 最後まで回す（natural:44400:1、3 本、60 s） | 3/3 成功、run.json・G_AUDIT（G1〜G3 違反 0）、wait が done・0 を返す。試行の wall 49.6・27.0・33.1 s（1 本目はモデルの読み込みを含む）、全体 115 s | `outputs\v2eval\S4SMOKE60\full` |
| (b) 止めて再開（induced:44400:3 --induce P1） | 1 回目: STOP で 2 本で止まる（終了コード 1）。2 回目: 試行 2 の途中 14 s で強制終了（再起動のまね）。82 が残しうる書きかけの json を植えた。3 回目: 書きかけを `_incomplete_*` に退避して試行 2 だけ回し、完全 3/3。試行 0・1 の 6 ファイルの SHA-256 は 3 回目の後も一致 | `outputs\v2eval\S4SMOKE60\resume_p1` |
| (c) 10/08 02:00 の再起動で切れた記録の再開（30 s の記録なので `--time-limit-s 30`） | 試行 2 だけ回し、完全 3/3。試行 0・1 の 6 ファイルの SHA-256 は一致 | `outputs\v2eval\S4SMOKE\stop` |
| 制限時間が記録に残る | 試行の json の time_limit_s = 60（S4SMOKE60）・30（S4SMOKE）。resume_p1 の試行 1 は t_success 56.7 s で成功＝30 s の採点 2/3、60 s の採点 3/3（score） | 同上 |
| task（44403:1、60 s の手順） | 3 番目の手順（青）の 1 回目 39.8 s → 2 回目 103.8 s → 163.8 s で止まる（持ち時間 60 s が効いている）。計画は LLM のキャッシュから。wall 514.7 s（シミュレーションの 3.1 倍。段階 3 の E7 は中央値 2.96 倍） | `outputs\v2eval\S4SMOKE60\task_e7` |
| 読み込み | 87 の _audits・_perception_accuracy、50_e_eval の summary、recovla.eval.report.collect、gate.require が 3 条件とも通る | scratchpad の check_readers.py |
| メモリ | 1 プロセスの作業セットの最大 3.83 GB、コミットの最大 6.94 GB、GPU の使用の最大 2,396 MiB（待機時 約 750 MiB を含む）、空きの物理メモリの最小 19.3 GB | `S4SMOKE60\full\resume_log.json`、nvidia-smi を 3 s ごと |

### 5-1 行動の再生でのビット一致（層 (i)。10/08 03:00〜03:45 の修理で追加）

やり方（`scripts\96_s4_replay_check.py check --dir <条件> --target 2`）: 完全な記録の `runtime_NNNN.json` の行動（区切りごとの a。誘発の上書きの後の値）を、方策の代わりに開ループで流す。推論・知覚を始める時刻、センサの読み、計算の時間の引き方は PolicyRuntime と同じに保ち、方策だけを呼ばない（GPU の推論なし。試行の枠は `harness.loop.run_policy_trial` のまま）。
- 経路 1: 新しいプロセス・新しい WorldRig で試行 2 だけを流す（＝止めて再開した試行の世界）。
- 経路 2: 新しいプロセスで試行 0・1 を流した後に、試行 2 を流す（＝続けて回した試行の世界）。
- 2 つの経路の試行 2 の npz の 22 の配列（cube_pos・cube_quat・cube_linvel・ee_pos・fingers・min_dist・contact・action など全部）をバイト列で比べる。参考に、元の閉ループの記録とも比べる。

| 確かめ | 結果 | 出どころ |
|---|---|---|
| 再生の忠実さ | 元の記録と同じ世界の状態から流すと、閉ループの記録とビット一致（resume_p1 の試行 0・1・2、world_env の試行 0・1・2。行動の食い違い 0、区切りの数も一致） | 下の 2 つの report.json の `reference_vs_closed_loop_record` |
| (A) 82 と同じ使い回し（S4SMOKE60\resume_p1、修理の前の包みの記録） | **一致しない**（22 の配列のうち 13 が一致、9 が不一致）。開始のこま（時刻 0）から違う: fingers 最大 9.4e-4 m（指 1 本で約 0.94 mm）、cube_pos 最大 2.0e-10 m、ee_pos 1.5e-10 m。原因: `sim\rig.py` 81〜95 行の reset は、前の試行が残したハンドの力の上限・指令のまま 1.0 秒（configs の scene.settle_prefilled_s）落ち着かせ、ハンドを戻すのはその後（`harness\world.py` 78〜79 行）。そのため、1 本目と 2 本目以降で開始の指の開きが違う（別に確かめた: 新しい世界の reset の後の指 0.041749 m、掴む口〔力の上限 ±20 → ±40〕の後の reset で 0.043178 m、先に hand.reset してから reset すると 0.041749 m に戻り差 0。scratchpad の reset_cause.py、読み取りと CPU の物理だけ）。経路 1 は元の記録（再開した試行 2）とビット一致、経路 2 の試行 0・1 も元の記録とビット一致 | `outputs\s4\replay_check\S4SMOKE60_resume_p1\report.json` |
| (B) 試行ごとに世界を作り直す（同じ記録、`--world-per-trial`） | 経路 1 と経路 2 が **22/22 ビット一致**（プロセスに残る状態は世界のほかにない）。試行 1 は使い回しで記録したので、作り直しの再生とは一致しない（(A) と同じ原因の裏づけ） | `outputs\s4\replay_check\S4SMOKE60_resume_p1_world_per_trial\report.json` |
| (C) 直した包みで止めて再開（S4SMOKE60\world_env、natural:44400:1、R1v3・naive・6 行・フィルタなし・60 s、既定の作り直し） | 1 回目 `--max-new 2` で 2 本で止まる（終了コード 1）、2 回目に試行 2 だけ回して 3/3 成功・G1〜G3 違反 0（終了コード 0）。試行 0・1 の 6 ファイルの SHA-256 は再開の後も一致。再生の確かめ: 経路 1 と経路 2 が 22/22 ビット一致、3 本とも閉ループの記録とビット一致。記録の `env`（ドライバ 610.88、torch 2.11.0+cu126、CUDA 12.6、OS 26300.9457、HEAD ce6b20d2）・`world_per_trial: true`・`time_limit_s: 60` が全試行に入り、87・50_e_eval・report.collect・gate が読める。score: 30 s 2/3、45 s 3/3、60 s 3/3（t_success 15.81・15.80・31.02 s） | `outputs\v2eval\S4SMOKE60\world_env`、`outputs\s4\replay_check\S4SMOKE60_world_env_world_per_trial\report.json` |
| 時間・メモリ | 包みの 2 回の実時間 1.44 分・1.20 分（試行の wall 51.1・31.8・69.0 s、1 本目と再開の 1 本目はモデルの読み込みを含む）。作業セットの最大 3.83 GB、空きの物理メモリの最小 20.4 GB。再生は経路 1 0.4〜0.7 分、経路 2 1.4〜2.3 分（GPU を使わない） | `S4SMOKE60\world_env\resume_log.json`、report.json の runs |

決めたこと: 段階 4 の包みは**試行ごとに世界を作り直す**（既定）。こうすれば、止めて再開しても、方策を通さない部分は続けて回した場合とビット一致する（(B)・(C)）。目標書_段階4.md 11 節 4 の「止めて再開して、行動の再生でビット一致（層 (i)）」は、この既定で満たす。段階 3 の 82 の記録（V3S3）は使い回しで回したので、各条件・各プロセスの 1 本目と 2 本目以降で開始の指の開きが違う。段階 4 の中の比較は全部作り直しで回すので、この差は入らない。段階 3 の数字との比較には、ドライバ・制限時間に加えて、この差も入る（大きさは開始の指の開き 1 mm 以下。成否への効き目は測っていない）。

修理の前の包みで作った S4SMOKE60\full・resume_p1・S4SMOKE\stop には `env` が無いので、続きを回そうとすると終了コード 3 で止まる（`unknown_records`。dry-run で確かめた）。smoke の記録なので、報告には使わない。

**計算時間の見積もりの注意:** 制限時間が 60 s になると、失敗した試行は 60 s 回り切る。実時間 ≈（シミュレーションの時刻）×（実時間の倍率）で、倍率は 1 本で 1.6 前後（S4SMOKE60\full の 2・3 本目）、3 本並行の段階 3 の A_nat で中央値 1.86。task は約 3 倍。55 s の実測で、復帰デモありは 1.36 倍・なしは 1.66 倍（10/08 のユーザーの指示）なので、D-復帰などは失敗の多い組ほど 60 s 回り切る本数で見積もる。

## 6. 結果を見る前に決めたこと・見た後に決めること

- 見る前に決めた（この文書と 2 つのスクリプトの既定値）: 完全の判定の規則、退避の規則、時間の窓は既定で無し（10/08 の作者の決定。改訂 1 の 01:45〜02:45 の窓は外した）、空きメモリ 12 GB（運用の決まり。並行の本数で割らない）・コミット 6 GB、制限時間 run 60 s・task の手順 30 s・全体 200 s（段階 3 と同じ。改訂 1 の 60 s・480 s は戻した）、30・45・60 秒の採点の規則（主は 30 s、誘発の分母は T 秒以内に成立した試行）、計画役の既定（decompose_s4、Haiku 5.5）、wait と wu-check の終了コードの割り当て、バックアップの組、環境の記録と食い違いで止める規則（ドライバ・torch・CUDA・OS。git の HEAD は記録だけ）。
- 見た後に決めた（成績ではなく、確かめの結果を見て）: 試行ごとに世界を作り直す既定（5-1 の (A) で使い回しが層 (i) を割ることを見てから決めた。成否・成績は見ていない。段階 4 の束 1・K の結果はまだ一つもない）。task の手順 30 s・全体 200 s は、作者が段階 3 の 6/20 とそのまま比べるために決めた（10/08。段階 4 の E7 の結果はまだ一つもない）。
