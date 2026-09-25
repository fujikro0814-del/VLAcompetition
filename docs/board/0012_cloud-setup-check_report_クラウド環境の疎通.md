# クラウド環境の疎通確認

- 番号: 0012 / 差出: cloud-setup-check / 宛先: review、main / 日時: 2026-09-25 17:51（日本時間）
- 返信先: なし（監督の指示書「支線 cloud/setup-check」、チャットで受領）
- 状態: 未読
- 関係する枝・PR・コミット: 枝 `cloud/setup-check`（起点 main 1a6ce6d）、PR「cloud/setup-check: クラウド環境の疎通確認」

## 要点（3行以内）

環境は作れた（Python 3.12.14、固定一覧 104 件は 6 秒・約 1.1 GB）。ただし **torch の CPU 版は入らない**（download.pytorch.org がこの環境の回線の方針で 403）。
検査は 102 件中 合格 70・飛ばし 1・不合格 13・エラー 18。GPU・Windows・Git 管理外のもの以外で落ちたのは「torch が無い」5 件と「場面の SHA-256 の不一致」1 件。
物理の速さ（10 秒分、3 回の中央値）: 物理だけ 実時間の 48.3 倍、制御器あり 5.9 倍（4 CPU）。

## 本文

### 1. 計算機

| 項目 | 値 | 出所 |
|---|---|---|
| CPU の数 | 4（Intel Xeon 2.80GHz） | `nproc`、`/proc/cpuinfo` |
| 主記憶 | 15.7 GiB（16,481,980 kB）、スワップなし | `/proc/meminfo`、`free -h` |
| OS | Linux 6.18（glibc 2.39）、x86_64 | `platform.platform()` |
| GPU・画面 | なし（`DISPLAY`・`MUJOCO_GL` とも未設定） | `env` |

### 2. 環境の作成

`env/requirements-cloud.txt` の冒頭の注記どおりに、リポジトリ直下の `.venv`（Git 管理外）に入れた。キャッシュは `UV_CACHE_DIR=.cache/uv`、Python は `UV_PYTHON_INSTALL_DIR=.python`（どちらも Git 管理外）。

| 手順 | 可否 | 所要時間 | 取得量 | 備考 |
|---|---|---|---|---|
| Python 3.12.14 | 可 | 約 3 s | 110 MB（展開後） | コンテナにある uv 0.8.17 は 3.12.14 を知らない（`No download found`）。§3.2 と同じ uv 0.12.15 を PyPI から作業用の場所（リポジトリの外）に入れて使った |
| torch==2.11.0・torchvision==0.26.0（CPU 版、`--index-url https://download.pytorch.org/whl/cpu`） | **不可** | — | — | 回線の中継が `download.pytorch.org:443` への接続を 403 で拒否（この環境の通信の方針）。迂回はしていない。版も変えていない |
| `--no-deps -r env/requirements-cloud.txt`（104 件） | 可 | 6 s | 約 1.1 GB（キャッシュ）、`.venv` は 961 MB | `uv pip check` は torch・torchvision が無いことだけを指摘（lerobot・accelerate が要求） |
| `--no-deps -e .` | 可 | 1 s | — | `recovla==0.1.0` |

**import と版**（`.venv`、torch なし）

| import | 版 |
|---|---|
| `recovla` | 0.1.0（`importlib.metadata`。`__version__` は持たない） |
| `lerobot` | 0.6.1（トップの import は torch なしで通る） |
| `mujoco` | 3.2.3 |

**参考（固定一覧の外。`.venv` には入れていない）**: 同じ版の torch を PyPI から別の venv（作業用の場所、Git 管理外）に入れると入った。ただし中身は CPU 版ではなく `2.11.0+cu130` で、依存の nvidia の部品を含めて **37 s・約 4.7 GB**。`torch.cuda.is_available()` は False、`lerobot.policies.smolvla.modeling_smolvla` の import は通る。この形を支線の環境として採るかは、監督・本線の判断に任せる（下の「求めること」）。

### 3. 検査

命令: `.venv/bin/python -m pytest -o faulthandler_timeout=300 tests/legacy/ tests/`（`PYTHONDONTWRITEBYTECODE=1`、`HF_HUB_OFFLINE=1`）

- **一括で回すと途中で止まる**: `tests/legacy/test_replay.py` の `mujoco.Renderer` の作成で Python ごと `Fatal Python error: Aborted`（直前の `test_closed_loop_eval.py` では同じ原因が `mujoco.FatalError` として ERROR で済んでいる）。そのため、ファイルごとに別の process で回し直した。下の数字はファイルごとの結果の合計
- 描画は指示書の「やらないこと」なので、`MUJOCO_GL=egl`・`osmesa` などの設定は試していない

| ファイル | 合格 | 飛ばし | 不合格 | エラー |
|---|---|---|---|---|
| `tests/legacy/test_closed_loop_eval.py` | 6 | 1 | 0 | 3 |
| `tests/legacy/test_controller_ik.py` | 3 | 0 | 0 | 0 |
| `tests/legacy/test_convert_to_lerobot.py` | 2 | 0 | 4 | 0 |
| `tests/legacy/test_replay.py` | 1 | 0 | 0 | 5 |
| `tests/legacy/test_train_launcher.py` | 27 | 0 | 0 | 0 |
| `tests/legacy/test_vla_image_spec.py` | 17 | 0 | 0 | 0 |
| `tests/legacy/test_vla_observation.py` | 7 | 0 | 1 | 0 |
| `tests/legacy/test_vla_state.py` | 6 | 0 | 0 | 0 |
| `tests/test_c_port.py` | 1 | 0 | 8 | 0 |
| `tests/test_push_check.py` | 0 | 0 | 0 | 10 |
| **計（102 件）** | **70** | **1** | **13** | **18** |

**落ちたもの（飛ばし 1 を含む 32 件）の原因の分類**。直していない。

| 分類 | 件数 | 検査 | 出たもの |
|---|---|---|---|
| GPU がない（描画の文脈が作れない） | 8 | `test_closed_loop_eval.py` の `test_action_moves_x_des_by_exactly_its_xyz`・`test_gripper_sign_and_single_press`・`test_trial_record_has_every_field`（`EvalRig` が `mujoco.Renderer` を作る）、`test_replay.py` の 5 件（`renderer` の fixture） | `mujoco.FatalError: an OpenGL platform library has not been loaded into this process` |
| Windows 専用 | 11 | `test_push_check.py` の 10 件、`test_c_port.py::test_environment_matches_the_lock` | `.tools/git/cmd/git.exe`・`.tools/uv/uv.exe` が無い（検査本体も `powershell` で `check_before_push.ps1` を呼ぶ） |
| モデルやデータが無い（Git 管理外） | 6 | `test_c_port.py` の `test_controller_and_recording_values_equal_the_legacy_raw_meta`（`outputs/legacy/raw/ep_800000/meta.json`）、`test_training_placements_equal_the_legacy_evaluation`（`outputs/legacy/eval_stage2_closed_loop` に trial が 0 件、`assert 0 == 20`）、`test_condition_1`〜`4`（`outputs/g0/*.json` が無い。作るには本線の `02_g0_check.py`・`03_vla_listing.ps1`＝GPU・モデル・Windows が要る） | `FileNotFoundError`、`pytest.fail(... がない。先に ... を実行する)` |
| それ以外: torch が無い | 6 | `test_convert_to_lerobot.py` の 4 件、`test_vla_observation.py::test_evaluation_entry_equals_training_dataset_images`、`test_closed_loop_eval.py::test_input_check_catches_swapped_or_missing_views`（飛ばし） | `ModuleNotFoundError: No module named 'torch'`。§2 の参考の venv（PyPI の torch）では、この 6 件は合格した（convert 6/6、observation 8/8、closed_loop の input check も合格） |
| それ以外: 場面の SHA-256 の不一致 | 1 | `test_c_port.py::test_scene_compiles_to_the_same_model_as_the_source` | `scene_g0.xml` を `mj_saveModel` した 32,077,350 バイトの SHA-256 が `8470e871…` で、検査の期待値 `9a328745…`（本線が Windows で測った値）と違う。この環境の中では 2 回とも同じ値。原因（OS・CPU・コンパイラによる計算の差か、場面そのものの違いか）は、流用元の XML も Windows の結果も無いので、ここでは切り分けられない。§15.1 の「MuJoCo の計算は OS・CPU で末尾の桁が変わりうる」に当たる可能性がある |

分類の合計は 8＋11＋6＋6＋1＝32（不合格 13＋エラー 18＋飛ばし 1）。

### 4. MuJoCo の物理の速さ（描画なし）

スクリプト: `scripts/cloud/bench_physics.py`（この PR で足した 1 本）。命令: `.venv/bin/python scripts/cloud/bench_physics.py`（既定で 5000 手 × 3 回）

- 場面 `assets/mjcf/scene_g0.xml`、timestep 0.002 s、5000 手＝模擬 10 秒。開始状態を落ち着かせる処理（`settle_start_state`）と読み込み（0.95 s）は時間に含めない。開始は `reset_episode`（学習配置の 1 番目）
- **制御器あり**: 評価器（`closed_loop.execute_action`）と同じ並びで、毎周期 `ScriptPad` → `VelocityCommandIntegrator.refresh` → `controller.update`（DLS の IK）→ `mj_step`。指令は xy の円（0.05 m/s、0.1 s ごとに向きを変え 4 s で 1 周）。10 秒で手先が 6.3 cm 動いたことを別に確かめた（円 2.5 周＝直径ぶん）
- **制御器なし**: 落ち着かせた開始状態の `ctrl` のまま `mj_step` だけ

| 方式 | 3 回の経過時間 [s] | 中央値 [s] | 実時間に対する倍率（中央値） | 1 手あたり |
|---|---|---|---|---|
| 制御器なし | 0.2069、0.1940、0.2139 | 0.2069 | **48.3 倍** | 41 µs |
| 制御器あり | 1.6929、1.7380、1.6845 | 1.6929 | **5.9 倍** | 339 µs |

1 process・1 スレッドの値。4 CPU あるので、試行を process で並べればおおよそ CPU の数だけ伸びる見込み（未計測）。制御器の Python の部分が 1 手の約 88% を占める。

### 5. 枝・PR・掲示板の運用

- 枝 `cloud/setup-check` を main（1a6ce6d）から切り、この報告とスクリプトの 2 ファイルだけを足した。`main` と他の枝には push していない
- 製品のコード（`src/`、`configs/`、`tests/`）・固定一覧は変えていない
- push の前にステージを見て、足したのが上の 2 ファイルだけであること、鍵・証明書・回線の設定の値・個人の情報を含まないことを確かめた。本線の `scripts/check_before_push.ps1` は PowerShell と Windows 用の git が要るので、ここでは回せない（§3 の Windows 専用と同じ理由）
- git の作者はこのセッションの既定（`Claude <noreply@anthropic.com>`）のまま

## 求めること

判断を求めるものではない（報告）。次の支線を始める前に、監督・本線に見てほしい点:

1. **torch の入れ方**: この環境では CPU 版の配布元に出られない。支線で torch が要る場面（lerobot の方策まわりの import、`rtc.py` との照合）は、(a) torch なしで済む範囲に絞る、(b) PyPI の同じ版（`2.11.0+cu130`、約 4.7 GB、GPU なしでも import・上の 6 件は通る）を支線だけで使う、(c) 環境の回線の方針に `download.pytorch.org` を足してもらう（クラウドの環境の設定。人が行う）、のどれにするか
2. **場面の SHA-256 の検査**: クラウドでは期待値と一致しない。支線の検査の対象から外すか、OS ごとに期待値を持つか
3. 検査を一括で回すと描画のところで process ごと止まる。支線では、描画を使う検査を除いて回す（`-m` や `--deselect`）決まりがあるとよい
