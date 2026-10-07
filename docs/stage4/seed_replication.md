# 束 3　学習の種の複製（包みのスクリプト scripts\99_s4_train_seed.py）

R1v3・N1v3 を、マニフェスト・データ・手順をそのままに train.seed だけ 1001・1002 に変えて学習し直す（計 4 本）ための包み。
40_h.py と configs\default.yaml は書き換えていない（タグ v3-s3-freeze の中身のまま）。作ったのは次の 4 つ。

| ファイル | 役割 |
|---|---|
| scripts\99_s4_train_seed.py | 包み本体。設定を作り、train_launcher を通して学習を起動する |
| configs\s4_seed_1001.yaml・s4_seed_1002.yaml | default.yaml の上に重ねる設定。中身は `train: {seed: 1001}`（1002）だけ |
| docs\stage4\seed_replication.md | このファイル |

## 1. しくみ

- 40_h.py の `cmd_train`（61・140 行付近）は、起動器に渡す設定の seed を `CFG["train"]["seed"]`（default.yaml の 90 行、値 1000）から取る。`CFG` は import 時に `config.load()` で決まるので、40_h.py を書き換えずに種だけ変えるには、設定側を重ねるのが筋になる。
- config の重ね方（`recovla.common.config.load(*overrides)`）は、default.yaml の上に configs\<名前>.yaml を辞書として再帰的に重ねる。そこで `configs\s4_seed_<種>.yaml` を足し、包みが `config.load("s4_seed_<種>")` で読む。
- 包みは 40_h.py の `cmd_train`（R1v3・N1v3 の分）と同じ起動器の設定を作り、`train_launcher.run` に渡す。違うのは (a) seed の出所、(b) note の末尾に種の標識が付くこと、(c) 出力先を outputs\s4\ 以下にしていること、の 3 つだけ。
- 毎回の確かめ（包みが自分でする）
  1. 重ねた設定と default.yaml の違いが `train.seed` だけであること。ほかも変える yaml なら止まる。
  2. データセットの conversion.json に埋まったマニフェストが、outputs\manifests の同名ファイルと一致すること。
  3. 起動器の検査（データセット、スナップショット、バッチ上限、出力先が未使用）。
- 種 1000 を指定すると断る（既存の R1v3・N1v3）。configs\s4_seed_<種>.yaml がない種も断る。

### 事前に決める項目と、後で決める項目

- 結果を見る前に決める（固定済み）: 種は 1001・1002、名前は R1v3・N1v3、ステップ 20000、バッチ 32、保存 5000 手ごと、num_workers 3、cue_augment なし（outputs\h\cue_aug_decision.json の採否のまま）、fast_query あり。すべて R1v3・N1v3 と同じ。
- 結果を見た後に決める: なし。この包みは評価を含まない。「loss が R1v3 と同程度」の判定は、`--post-check` が出す数字を人が見て決める（閾値は事前に決めていないので、数字をそのまま報告に載せる）。

## 2. 使い方

すべて C:\PAI\recovery_vla で、`.venv\Scripts\python.exe scripts\99_s4_train_seed.py ...`。

```
# 学習しないで、渡す設定を出す（既定の動作。--dry-run と同じ）
... R1v3 --seed 1001 --dry-run
# 同じ + LeRobot が組み立てる train_config.json 相当も出す（CPU だけ、数十秒。GPU は CUDA_VISIBLE_DEVICES を空にして隠す）
... R1v3 --seed 1001 --dry-run --resolve
# 既存の学習（種 1000）との差分一覧（--resolve を含む）
... R1v3 --seed 1001 --compare outputs\train\train_R1v3_20261005-180404_20261005-180404
... N1v3 --seed 1002 --compare outputs\train\train_N1v3_20261005-202158_20261005-202158
# 種 1000 のとき 40_h.py と同じ設定になるか（R1v3・N1v3、smoke の有無の 4 通り）
... --check-vs-40h
# 学習の起動（運用役だけ。GPU を使う。前景で、切り離さない）
... R1v3 --seed 1001 --start [--smoke]
# 学習が終わった後の確認（CPU・読み取りだけ）
... R1v3 --post-check outputs\s4\train\<実行名>
```

- `--start` を付けない限り、学習は始まらない。`--start` と `--dry-run` は同時に使えない。
- `--smoke` は 40_h.py と同じ 1000 手（保存は 1000 手に 1 回）。種の包みの smoke は、運用役が `--start --smoke` で 1 本だけ通す想定（10/09 午前）。
- `--num-workers` の既定は 3。R1v3・N1v3 は実際に `--num-workers 3` で起動されている（train_run.json の command で確認）ので、default.yaml の 6 ではなくこちらに合わせた。値は試料の並びを変えない（40_h.py の説明どおり、並びは主プロセスの sampler が決める）。
- 出力（すべて outputs\s4\ 以下）

| 動作 | 書く場所 |
|---|---|
| dry-run・compare | outputs\s4\seed_wrap\dryrun_<名前>_s<種>.json、compare_<名前>_s<種>.json、_resolve_*.json、dryrun_cfg\ |
| --start | 設定 outputs\s4\train_cfg\、学習の出力 outputs\s4\train\train_<名前>s<種>_<日時>_<日時>\、ログ outputs\s4\train_logs\、実行の記録 outputs\s4\seed_wrap\train_<名前>s<種>.json |
| --post-check | outputs\s4\seed_wrap\postcheck_<実行名>.json |

## 3. 実行前に確かめる項目（運用役）

学習は GPU を占有し、評価と同時に走らせるとメモリが足りなくなる。次を満たしてから `--start` する。

1. **学習は 1 本ずつ。** 4 本は順に回す（R1v3 1001 → N1v3 1001 → R1v3 1002 → N1v3 1002 など、順番は運用役が決めてよい。結果は順番に依存しない）。包みの `--start` は、ほかの lerobot 学習（train_wrapped・lerobot-train）が走っていれば断る。
2. **学習中の評価は 1 本まで。** 10/05 の記録では、学習＋評価 2 本＋別作業で 12:06 に cv2 が OOM を起こし、評価（R2v3_P1b）が 22/66 で落ちた。final.md の日程どおり、学習中は物差し K（約 2 h）などの評価 1 本だけにする。
3. **メモリ。**
   - GPU: 学習の GPU 割り当て最大は R1v3 7.69 GiB・N1v3 7.70 GiB（train_run.json の log_summary）。上限 14 GiB の約半分。包みは空き 12 GiB 未満なら起動しない（nvidia-smi。確認時点は空き 14.7 GiB）。
   - 主記憶・仮想領域: 評価 3 本と学習の同時実行は仮想領域不足で落ちた前例がある（段階 3 の引き継ぎ）。num_workers は 3 のまま変えない。ユーザーの別作業が重い間は始めない。
   - ディスク: 1 本あたり約 4.9 GiB（R1v3 の出力フォルダで計測）。4 本で約 20 GiB。C: の空きは約 1.2 TiB。
4. **所要時間。** 1 本あたり約 2.3 h（R1v3 は 18:04:05〜20:21:57 で 2 h 17.9 min、N1v3 は 20:21:58〜22:40:52 で 2 h 18.9 min）。4 本で約 9.2 h。final.md の日程（10/09 12:00〜21:30、9.5 h）に余裕は約 0.3 h しかないので、1 本ごとの前後に待ちを入れないこと。遅れたら最後の 1 本が 21:30 を越える。
5. **再起動の危険。** 学習は途中から再開できない（包みは `--resume` を使わない。起動器は毎回新しい出力先を作る）。強制再起動が起きたら、その 1 本は最初からやり直す（新しい実行名になる。壊れた出力フォルダは消さずに残す）。更新による再起動は 02:00 ごろなので、12:00〜21:30 の枠は外れるが、起動前に RebootRequired・RebootPending が立っていないことを確かめる（reference_pc_restart_schedule.md）。
6. **監視役。** 切り離さず前景で回す場合も、終了・エラーを通知する監視役を背景で付ける（メモリの運用規則）。
7. **起動前の dry-run。** `--dry-run --resolve` を 1 回通し、出た `--seed=` の値と output_dir が意図どおりであること、マニフェストの SHA-256 が下の表と同じであることを見る。

| マニフェスト | SHA-256（確認時点） |
|---|---|
| outputs\manifests\R1v3_20261005-133737.json | 85dd9dbe951b8244ed2606f9aacb3ab5d635d3ade4dcc86a36b39519bc1d810f |
| outputs\manifests\N1v3_20261005-133737.json | cebcb4a3684cffe3b8a23ec29c823cbcef4e608cb344e8b7de2bb34cac31ddaa |

どちらも 330 エントリで、データセットの conversion.json に埋まったマニフェストの写しと一致する（包みが毎回確かめる）。出所は outputs\f\data_v3.json の datasets.R1.manifest・datasets.N1.manifest で、既存の学習が使ったものと同じ。データセットの指紋（train_launcher.dataset_fingerprint）も既存の train_run.json と同じ（R1v3 c717c12f…、N1v3 bdf18633…）。

## 4. 確かめた結果（2026-10-08、CPU だけ。学習は起動していない）

### 4.1 種 1000 で 40_h.py と同じ設定になる（`--check-vs-40h`）

40_h.py の `cmd_train` を、起動器の `run` だけ差し替えて（起動せず設定の JSON を拾う）走らせ、包みの設定と比べた。R1v3・N1v3、smoke の有無の 4 通りすべてで、note を含めて完全に一致した。記録: outputs\s4\seed_wrap\check_vs_40h.json。

### 4.2 差分の一覧（`--compare`）

既存の学習（種 1000）に対して、4 通り（R1v3 と N1v3 の各 1001・1002）とも同じ結果だった。想定外の違いは 0 件。下は R1v3・種 1001 の例（N1v3 も名前が変わるだけで同じ形）。記録: outputs\s4\seed_wrap\compare_<名前>_s<種>.json。

比べた 4 つの層と、違った項目:

| 層 | 項目 | 既存（種 1000） | 新（種 1001） | 分類 |
|---|---|---|---|---|
| 起動器の設定 | seed | 1000 | 1001 | seed（意図した違い） |
| 起動器の設定 | note | Stage 3 R1v3: ... | 同じ文 + 「; S4 seed replication (train.seed=1001, ...)」 | 自由記述。学習に渡らない |
| lerobot へ渡す引数 | --seed | 1000 | 1001 | seed |
| lerobot へ渡す引数 | --output_dir | outputs\train\train_R1v3_20261005-180404_20261005-180404 | outputs\s4\train\train_R1v3s1001_<日時>_<日時> | 出力先・実行名 |
| lerobot へ渡す引数 | --job_name | train_R1v3_20261005-180404_20261005-180404 | train_R1v3s1001_<日時>_<日時> | 出力先・実行名 |
| train_config.json（LeRobot の構文解析 + validate の結果） | seed | 1000 | 1001 | seed |
| train_config.json | output_dir | 上と同じ | 上と同じ | 出力先・実行名 |
| train_config.json | job_name | 上と同じ | 上と同じ | 出力先・実行名 |

- 上の 4 層のほかの項目（データセット、スナップショット、rename_map、バッチ 32、ステップ 20000、保存 5000、num_workers 3、方策の設定、最適化・スケジューラ、fast_query、前置き `python -m recovla.policy.train_wrapped --reference --fast-query --`）は、すべて一致した。
- **train_config.json の作り方。** 既存の学習が保存した `checkpoints\020000\pretrained_model\train_config.json` を、同じ引数（種 1000）を LeRobot の構文解析（`parser.wrap` + `cfg.validate()`、`CUDA_VISIBLE_DEVICES` を空にして CPU だけ）に通して作り直したところ、output_dir 以外は完全に一致した。つまりこの方法で出した種 1001・1002 の train_config.json 相当は、実際に学習が保存するものと同じ形になる。比べる相手として十分と判断した。
- **コードの同一性。** train_launcher.py の SHA-256 が既存の train_run.json の記録と同じ（8cc07ad8…）。convert.py と default.yaml も記録と同じ。train_wrapped.py は既存の記録に SHA-256 がない（記録は files_sha256 の一部だけ）ので、v3-s3-freeze からの変更がないことは `git diff --name-status v3-s3-freeze -- src` で確かめた（src の変更は 0 件）。
- 実際に学習を始めたあとの確認は `--post-check`（train_config.json の違いが seed と出力先だけか、指紋、終了コード、チェックポイント 4 つ、loss の要約）。自己照合（既存の R1v3 を新として渡す）は通り、別のデータ（N1v3 を R1v3 として渡す）は「想定外」と不一致で検出されることまで確かめた。

### 4.3 マニフェスト

上の表のとおり。既存と同じファイル、同じ SHA-256、conversion.json の写しと一致。

## 5. 学習が終わった後に見る数字（R1v3・N1v3 の種 1000、train_run.json の log_summary）

`--post-check` が新旧を並べて出す。事前に閾値は決めていない。人が見て「同程度」かどうかを報告する。

| 項目 | R1v3（種 1000） | N1v3（種 1000） |
|---|---|---|
| 記録点 | 400 | 400 |
| loss 最初 | 1.968 | 1.887 |
| loss 最後 | 0.031 | 0.032 |
| loss 最後の 10% の平均 | 0.032025 | 0.03205 |
| GPU 割り当て最大 [GiB] | 7.69 | 7.70 |
| 学習時間 | 2 h 17.9 min | 2 h 18.9 min |

## 6. 残る危険・限界

- 学習は `cudnn_deterministic=false`（train_config.json）なので、同じ種で回し直してもビット一致は保証されない。ここで複製するのは「種を変えたときのぶれ」で、同じ種の再現ではない。
- 種が 2 つなので、ぶれの大きさまでは言えない（final.md 束 3 の限界）。
- seed は LeRobot では乱数の初期化（set_seed）と、サンプラの並び（EpisodeAwareSampler のようなシード）に使われる。cue_augment は使っていないので、train_wrapped の手がかりずらしの乱数は関係しない。
- 包みの `--start` は、このセッションでは一度も実行していない（禁止）。起動器の dry-run までと、precheck の関数単体（nvidia-smi の読み取りと psutil の走査）は確かめたが、実際の起動の経路（`tl.run` を dry-run なしで呼ぶ部分と、実行の記録の書き出し）は 10/09 午前の smoke が初めての実行になる。smoke で、outputs\s4\train\ に出力が作られ、outputs\s4\seed_wrap\train_R1v3s1001_smoke.json が書かれることを確かめる。
- 40_h.py は学習後に outputs\h\train_<名前>.json を書くが、包みは書かず、代わりに outputs\s4\seed_wrap\train_<名前>s<種>.json を書く。既存のスクリプト（54_results_s3.py など）がこの記録を拾うことはない。評価側は run_dir（outputs\s4\train\...）を明示して使うこと。
