# Step B 提案書 — 最終課題「生成復帰 VLA」

作成: 2026-09-25 / 対象: `手順書_最終課題_生成復帰VLA_v2.md` §3 / 上位文書: `最終課題_計画書_v2.md`
状態: **2026-09-25 承認（修正 4 点つき、掲示板 0005）。** 修正 1（§8 の settle）・2（§15 の planner）・4（§5・§14 R3 の置き場所の条件）と §15.4 の支線の順序は反映済み。修正 3（非公開の recovery_vla）は 0007 で取り消し、0002・0004 のとおり公開の VLAcompetition を使う（§3.2 の 10）
前提: `docs\A_調査報告.md`（Step A）と、Step A への判断（IK は流用元のまま、G0 は種 100000〜100019、最終評価は 110000〜、RTC は `rollout/inference/rtc.py` に合わせる、環境はネットワークから入れ直す）

この Step ではコードを書いていない。Step B のために追加で確かめた事実（すべて C:\VLA を読むだけ）:

- 学生用 README（`C:\VLA\02_環境\lerobot\README.md`）: 学内のプロキシは TLS を中継している。uv は `UV_SYSTEM_CERTS=1`（Windows の証明書ストアを使う）で通り、Python から HF に出るときは `SSL_CERT_FILE`・`REQUESTS_CA_BUNDLE` に `ca_bundle_proxy.pem` を指定する。uv は GitHub の 0.12.15、Python 本体は 3.12.14。この PC では `HTTP(S)_PROXY=<学内のプロキシ>` が計算機全体に設定済み
- MuJoCo 3.2.3 に `mj_getState`・`mj_setState`・`mjSTATE_INTEGRATION`、`mj_geomDistance(m, d, g1, g2, distmax, fromto)` がある（`mujoco.h:411, 414, 499`、`mjdata.h:50`）
- LeRobot 0.6.1: データセットの統計量は `meta/stats.json` から読まれる（`datasets/io_utils.py:161-174`）。`use_policy_training_preset=True`（既定）かつ resume でないときは、最適化器と学習率の予定を方策の設定から作る（`configs/train.py:272-276`）ので、`--policy.optimizer_lr` などで変えられる
- LeRobot 0.6.1 の VLM の順伝播は勾配を止めていない（`requires_grad=False` にするだけ。`smolvlm_with_expert.py:152-200`）。VLM の層に LoRA を足せば、行動エキスパートの交差注意を通って勾配が届く
- この PC: Core i7-14700KF（論理 28）、主記憶 32 GB、C: の空き 1.6 TB。git と Python は PATH にない（`python` は Windows ストアの入口だけ）
- Claude API: 軽量モデルは `claude-haiku-4-5`（入力 $1、出力 $5 / 100 万トークン、200K 文脈）。Haiku 4.5 では温度を指定できる

---

## 1. ディレクトリ構成（確定案）

```
C:\PAI\recovery_vla\                ＝ <ROOT>、Git のリポジトリ
  README.md
  pyproject.toml                    パッケージ recovla の定義（依存は書かない。版の固定は env\ 側）
  .gitignore
  env\                              実行環境の定義（Git 管理）
    requirements-lock.txt           版の固定一覧（§3）
    setup_env.ps1                   環境を作る手順（§3）
    session_env.example.ps1         1 回の作業で設定する環境変数の例（鍵は書かない）
  configs\
    default.yaml                    すべての設定値（§5）
    g0.yaml                         G0 用の上書き（旧場面、旧評価）
    demo\                           デモの場面ごとの種と設定（Step J）
  assets\mjcf\
    panda\                          panda.xml、メッシュ、LICENSE（流用元のまま）
    scene_g0.xml                    流用元の teleop_scene.xml そのまま（立方体 1 個、G0 専用）
    scene_3cube.xml                 3 色の場面（Step D で作る。発表用カメラもここに置く）
  src\recovla\
    common\       設定の読み込み、乱数の流れ、コードの版、パス
    sim\          場面の生成、座標と幾何の集約（frames.py）、制御ループ、IK（流用）、
                  指令の入口、接触と最小距離の計測、安全フィルタ（条件付き）、カメラ
    expert\       台本（段階の判定と続き）、失敗の注入、引き継ぎ
    record\       記録、途中状態の保存と復元、分割、メタデータ、再生確認
    data\         変換、検証、マニフェスト、vla_image_spec・vla_state・vla_observation（流用）
    policy\       読み込み、実行器（sync / naive / rtc）、学習の起動器
    eval\         閉ループ評価、誘発、指標、統計、表と図
    planner\      LLM、色の検出、完了判定、手順の実行、音声（任意）
    viz\          発表用の描画、重ね表示、波形
  scripts\        00_setup_env.ps1、01_import_from_vla.ps1、02_g0_check.py、
                  10_gen_normal.py、11_gen_recovery.py、20_convert.py、30_train.py、
                  40_eval.py、41_results.py、50_planner.py、60_render_demo.py …（番号順に使う）
  tests\          test_c_port.py、test_d_scene.py、… と、流用した部品の回帰検査
  docs\           調査報告、提案書、各 Step の報告、provenance.csv（複写の記録）
    board\        掲示板（README.md の運用に従う。STATUS.md は本線が更新）
    interfaces\   支線と本線の境目の約束（§15。本線が書き、支線は変えない）

  （Git 管理外）
  .tools\         uv.exe、MinGit
  .python\        uv が入れる Python 本体
  .venv\          実行環境
  .cache\         uv のキャッシュ
  .local\         プロキシの証明書（公開情報）。鍵は置かない
  models\hf_home\ 出発点モデルと SmolVLM2（HF のキャッシュの構成のまま）
  outputs\        legacy\（流用元から複写した記録・チェックポイント）、gen\、datasets\、
                  train\、eval\、results\、videos\、llm_cache\
  docs\local\     公開しない手元の資料（C:\VLA の全ファイル一覧、公開版で伏せた記述の原文）
```

叩き台からの変更は4点。`assets\mjcf\` を独立させる（場面の XML は設定からも台本からも参照するため）。`src\` の下を 1 つのパッケージ `recovla` にまとめる（`pip install -e . --no-deps` で入れ、`sys.path` を書き換えない）。`env\` を足す（環境の定義を Git で固定するため）。Git 管理外の置き場所を明示する。

---

## 2. 流用の一覧と、複写元の記録方法

### 2.1 複写の方法

- **コード**は、流用元のリポジトリの**コミットから**取り出す。作業ツリーからは取らない。`git --no-optional-locks -c safe.directory=C:/VLA/pytools/panda_teleop -C C:\VLA\pytools\panda_teleop archive dbb2c3b4db64c7e6db4a997e087549e67cd776c6 <パス…> -o <ROOT>\.cache\import\vla_dbb2c3b.tar` で tar に書き出して `<ROOT>` 側で展開する。`git archive` は流用元のリポジトリに書き込まない（索引も参照も触らない）。git は自前の MinGit（`<ROOT>\.tools\git`）を使う。Step A の時点で、対象ファイルの作業ツリーとコミットは一致していた（未コミットの変更なし）
- **コード以外**（HF のスナップショット、予備実験のチェックポイント、raw の記録 1 本、評価の json）は `Copy-Item` で複写し、複写の前後で SHA-256 を比べる
- **記録**: `docs\provenance.csv` に 1 ファイル 1 行で書く（複写元のパス、コミット（コードのみ）、git の blob id（コードのみ）、複写元の SHA-256、複写先のパス、複写先の SHA-256（手を入れる前）、扱い）。手を入れたファイルは、その後の変更を `<ROOT>` の Git 履歴で追う。最初のコミットを「流用元をそのまま取り込んだ状態」にして、手を入れた差分が 2 番目以降のコミットで見えるようにする
- 取り込みは `scripts\01_import_from_vla.ps1` 1 本にまとめ、取り込み後に C:\VLA の全ファイル一覧を取り直して Step A の開始時と比べる（Step C 完了条件4）

### 2.2 流用の一覧（Step A の表を確定させたもの）

| 流用元（`pytools\panda_teleop\panda_teleop\` からの相対。他は明記） | 複写先 | 扱い | Step C で行う変更（動作は変えない） |
|---|---|---|---|
| `assets/panda/panda.xml`、`assets/panda/assets/*`、`LICENSE`、`README.md` | `assets/mjcf/panda/` | そのまま | なし |
| `assets/panda/teleop_scene.xml` | `assets/mjcf/scene_g0.xml` | そのまま | `include` の相対パスだけ直す（バイト差は provenance に記録） |
| `teleop/controller_ik.py` | `src/recovla/sim/controller_ik.py` | そのまま | なし（診断用の print もそのまま。呼ぶ側で標準出力を捨てる、流用元と同じ扱い） |
| `teleop/device.py` | `src/recovla/sim/device.py` | そのまま | import のパスだけ |
| `teleop/app.py` の `make_controller`・`apply_home_pose`・`HOME_QPOS` | `src/recovla/sim/control.py` | 手を入れて | 調整値を `default.yaml` の `controller:` から読む。環境変数 `TELEOP_CONTROLLER`・`TELEOP_DEVICE` への依存を外す（常に DLS）。値は流用元と同じで、同一性は G0 で確かめる |
| `teleop/collect.py` の `VelocityCommandIntegrator`・`make_collect_controller`・`make_integrator`・`StartState`・`settle_start_state`・`reset_episode`・`close_renderer`・`render_cameras`・定数 | `src/recovla/sim/control.py`、`sim/render.py` | 手を入れて | 定数を設定へ。`CollectSession`・`OperatorView`・`main`・台帳を外す |
| `scripted_demo.py` の `ScriptPad`・`pad_state` | `src/recovla/sim/device.py` | 手を入れて | 評価器が使う部分だけ |
| `teleop/recorder.py` | `src/recovla/record/recorder.py` | 手を入れて | `DEFAULT_RAW_DIR`・`scan_raw`・`SessionLog` を外す。Step C では立方体 1 個のまま |
| `teleop/replay.py`、`replay_check.py` | `src/recovla/record/replay.py`、`scripts/` | 手を入れて | import のパスだけ（途中状態からの再生は Step F で足す） |
| `vla_image_spec.py`、`vla_state.py`、`vla_observation.py` | `src/recovla/data/` | そのまま | import のパスだけ。ファイル名は変えない（卒研と同一であることを追いやすくするため） |
| `convert_to_lerobot.py` | `src/recovla/data/convert.py` | 手を入れて | `recorder.DEFAULT_RAW_DIR` への依存を外す。ラベルを要求するマニフェストは外す（§6）。`episode_arrays`・`--verify` はそのまま |
| `closed_loop_eval.py` | `src/recovla/eval/closed_loop.py` | 手を入れて | Step C では import とパスだけ（G0 の同一性のため）。実行器の分離は Step G |
| `train_launcher.py` | `src/recovla/policy/train_launcher.py` | 手を入れて | パス（HF_HOME、出力先、ログ先）を設定経由に。起動器の検査はそのまま |
| `code_version.py` | `src/recovla/common/code_version.py` | 手を入れて | git の場所と対象ファイルの一覧 |
| `gui/gui_core.py` の `wilson_interval()` | `src/recovla/eval/stats.py` | 関数だけ | なし |
| `tests/test_vla_image_spec.py`・`test_vla_state.py`・`test_vla_observation.py`・`test_convert_to_lerobot.py`・`test_replay.py`・`test_closed_loop_eval.py`・`test_controller_ik.py`・`test_train_launcher.py` | `tests/legacy/` | 手を入れて | 台帳・03_収録 への依存を外す。流用した部品の回帰検査として Step C で全部通す |
| `02_環境\lerobot\scripted_check\analysis\route_eval_summary.py` | 参照のみ | 持ち込まない | 持ち上げの高さなどの考え方だけを段階の判定に使う |
| `teleop/ledger.py`、`episode_ledger.py`、`gui/*`（上記の関数以外）、`teleop/dualsense_device.py`、`collect_main.py`、`main.py`、各 `.bat`、`start_training.bat`、`train_config_example.json`、`teleop/{filters,frames,cameras,game,shelf_task,pad_feedback,touch_device,probe_touch}.py`、`probe_*.py`、`controller_qp.py` | — | 持ち込まない | スコープ外、または使われていない。DualSense の速度上限（0.20・0.10 m/s）は値だけを設定へ |
| `02_環境\hf_home\hub\models--lerobot--smolvla_libero\`（`refs\main`、`snapshots\31d453f…\*`） | `models/hf_home/hub/…` | そのまま | なし |
| `02_環境\hf_home\hub\models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct\`（`refs\main`、`snapshots\7b375e1…\*`） | `models/hf_home/hub/…` | そのまま | なし |
| `02_環境\lerobot\scripted_check\train\stage2_20260917-221354\` の `conversion.json`・`train_run.json`・`train_launch_config.json`・`checkpoints\019000\pretrained_model\*` | `outputs/legacy/stage2_20260917-221354/`（同じ構成） | そのまま | なし（`training_state\` は持ち込まない。約 413 MB 減） |
| `02_環境\lerobot\scripted_check\raw\stage2\2026-09-17\ep_800000\` | `outputs/legacy/raw/ep_800000/` | そのまま | なし（G0 の再生確認用。学習に使われた 50 本のうちの 1 本） |
| `02_環境\lerobot\scripted_check\eval\stage2_closed_loop\` の `summary.json`・`trial_*.json` | `outputs/legacy/eval_stage2_closed_loop/` | そのまま | なし（動画と npz は持ち込まない） |
| `C:\VLA\02_環境\lerobot\ca_bundle_proxy.pem` | `.local/ca_bundle_proxy.pem`（Git 管理外） | そのまま | なし（プロキシのルート証明書＋certifi。公開情報で秘密ではない。LLM の呼び出しと、必要なら HF で使う） |

---

## 3. 実行環境

### 3.1 方針

- **1 つの環境で、生成・学習・評価をすべて回す。** Python 3.12.14 に lerobot 0.6.1 と mujoco 3.2.3 を入れる。根拠: 流用元の評価器はすでにこの組み合わせで物理と描画を回している。開始時の描画は python311 で記録した PNG と画素単位で一致し、10 fps の行動の再生も 1.9 mm で通っている（Step A §2-13）。台本の生成に要る opencv は headless 版で足りる（画面は出さない）。python311 側は使わない
- 流用元の raw の記録（python311 で作られたもの）と新しい環境とでは、開始状態のビット一致は期待しない。G0 の再生確認は流用元と同じ 5 mm の許容で判定する。新しく作るデータの決定性（組の最初のこまの一致など）は、すべて新しい環境の中で確かめる
- **管理者権限は使わない。** すべて `<ROOT>` の下に入れ、PATH・レジストリ・計算機全体の設定は変えない

### 3.2 手順（`env\setup_env.ps1`。Step C で実行）

1. **uv**: GitHub の astral-sh/uv の 0.12.15（学生用と同じ）の `uv-x86_64-pc-windows-msvc.zip` を `<ROOT>\.tools\uv\` に取得し、配布元の `.sha256` と照合する。C:\VLA の `uv.exe` は使わない
2. **MinGit**: Git for Windows の MinGit（64-bit、portable）を `<ROOT>\.tools\git\` に置く。`safe.directory` は設定ファイルに書かず、コマンドごとの `-c` で渡す
3. このセッションだけの環境変数: `UV_PYTHON_INSTALL_DIR=<ROOT>\.python`、`UV_CACHE_DIR=<ROOT>\.cache\uv`、`UV_SYSTEM_CERTS=1`（README と同じ。プロキシは計算機全体の設定がそのまま効く）
4. `uv python install 3.12.14` → `uv venv <ROOT>\.venv --python 3.12.14`
5. torch・torchvision: `torch==2.11.0+cu126`、`torchvision==0.26.0+cu126` を `https://download.pytorch.org/whl/cu126` から入れる
6. それ以外: `env\requirements-lock.txt`（Step A の付録 A の全 87 パッケージを `==` で固定したもの。当初「95」と書いたのは数え違い。掲示板 0008）を `uv pip install --no-deps -r` で入れる。依存を解き直さないので、学生用と同じ組み合わせになる
7. 追加分: `uv pip install scipy matplotlib anthropic --constraint env\requirements-lock.txt`（既存の固定を動かさずに解く）。入った版を lock に追記して固定する。**peft は入れない**（K1 で LoRA の分岐を採ったときに、同じ方法で足す）。mink は入れない
8. `uv pip check` と、入った一覧が lock と一致することの検査（`tests\test_c_port.py` の一項目）
9. `uv pip install -e <ROOT> --no-deps`（`recovla` を登録する。C:\VLA は指さない）
10. **GitHub**（掲示板 0002・0004 で改めた）: リポジトリは作らず、既存の**公開**リポジトリ `https://github.com/fujikro0814-del/VLAcompetition` を `remote add origin` する。リモートの `main` には `Initial commit`（README.md 1 行）があるので、取得してその上に積む（取り込みのコミットはその次になる。§2.1 の目的は変わらない）。GitHub CLI（gh）の portable 版を `<ROOT>\.tools\gh\` に置き、認証だけは人が行う（`! .tools\gh\bin\gh.exe auth login`。ブラウザでの確認。鍵やトークンをファイルに書かない）。共同作業者は追加しない。git の HTTPS は `-c http.sslBackend=schannel`（Windows の証明書ストアを使う。TLS の中継対策）で通す。コミットの作者は、このリポジトリの設定だけに `user.name=fujikro0814-del`、`user.email=327317808+fujikro0814-del@users.noreply.github.com`（GitHub の非公開用のアドレス）とする
    - **push の前の検査**（`scripts\check_before_push.ps1`、push のたびに必ず通す）: ステージに次のどれかがあれば push せずに止まって報告する。(a) `.gitignore` の対象（`.tools`、`.python`、`.venv`、`.cache`、`.local`、`models`、`outputs`、`docs\local`）の下のファイル、(b) 鍵の形をした文字列（`sk-ant-`、`ghp_`・`gho_`・`github_pat_`、`hf_` の後に英数字などが 20 字以上続くもの、`BEGIN … PRIVATE KEY` の行）、(c) 証明書（`BEGIN CERTIFICATE` の行、拡張子 `.pem`・`.crt`・`.cer`）、(d) 学内のプロキシのアドレスと、共有フォルダのパスの実際の値。(d) の値はリポジトリに書かず、Git 管理外の `.local\` に置いて読む。文書に出てくる前置きの名前だけ（例: 「`sk-ant-` など」「`Y:\` へ複写」）は引っかからないようにし、そのことを検査の単体検査で確かめる
    - C:\VLA の記録・データ・卒研の資料（本冊・別冊）はリポジトリに入れない。流用したコードと場面の定義は入れてよい（0002 の 2）。`outputs\legacy\` は Git 管理外なので、ここに複写する記録・チェックポイントは公開されない

- 支線（クラウド）のための CPU 用の固定一覧 `env\requirements-cloud.txt` も作る（lock から torch・torchvision・CUDA 関係を除いた部分集合＋scipy・matplotlib。版は lock と同じ）。支線が要る場合だけ torch の CPU 版を同じ版で入れる

- 回線で詰まったときは、学生用 README の記述だけを参照する（例: Python から HF に出るなら `SSL_CERT_FILE`）。計算機全体の設定を変える前に止まって報告する
- 取得の量: torch（cu126）が約 2.5 GB、その他が約 1 GB

### 3.3 毎回の環境変数（`env\session_env.example.ps1`）

`PYTHONDONTWRITEBYTECODE=1`、`HF_HOME=<ROOT>\models\hf_home`、`HF_HUB_OFFLINE=1`、`PYTHONIOENCODING=utf-8`。LLM を使うときだけ `ANTHROPIC_API_KEY`（手で設定する。ファイルに書かない）と、証明書の場所（§12）を足す。

---

## 4. 出発点モデルと SmolVLM2 をネットワークなしで読む方法

- §2.2 のとおり、HF のキャッシュの構成（`hub\models--<org>--<name>\refs\main` と `snapshots\<hash>\…`）をそのまま `<ROOT>\models\hf_home\hub\` に複写する。流用元の `blobs\` は空で、スナップショットの中は実体ファイルなので、構成を変えずに複写できる。`smolvla_base` と LIBERO のデータセットは持ち込まない
- `HF_HOME=<ROOT>\models\hf_home`、`HF_HUB_OFFLINE=1` で起動する。出発点は `--policy.path=<ROOT>\models\hf_home\hub\models--lerobot--smolvla_libero\snapshots\31d453f7edd78c839a8bbc39744a292686daf0de`（Windows ではリポジトリ名で渡すと失敗するため、フォルダのパスを渡す）。SmolVLM2 の重み・設定・字句解析器は、方策の中から名前で呼ばれるので、同じ `HF_HOME` から解決される
- 確かめ方（Step C 完了条件1）: 上の環境変数だけで、(a) 出発点モデルと前処理・後処理の読み込み、(b) 予備実験のチェックポイントの読み込みと 1 回の推論、(c) 学習の起動器の空打ち（`--dry-run`）と 10 手の学習、が通ること。あわせて、`HF_HOME` の下に `C:\VLA` を指すパスが残っていないことを検査する

---

## 5. 設定ファイル（`configs/default.yaml` の確定案）

数値はすべてここに置く。流用元の値は出典を併記した。`null` は Step D 以降で測って決める値。

```yaml
paths:                       # すべて <ROOT> からの相対
  models_home: models/hf_home
  policy_snapshot: models/hf_home/hub/models--lerobot--smolvla_libero/snapshots/31d453f7edd78c839a8bbc39744a292686daf0de
  outputs: outputs
  scene_g0: assets/mjcf/scene_g0.xml
  scene: assets/mjcf/scene_3cube.xml

sim:                         # 流用元 collect.py / closed_loop_eval.py
  timestep: 0.002            # モデルの値と照合する（食い違えば止める）
  steps_per_pad_read: 10     # scripted_demo.STEPS_PER_LOOP
  record_every: 25           # 20 Hz
  stride: 2                  # 10 fps
  image_size: 256
  cameras: [overhead, wrist]
  settle_s: 3.0
  start_pos: [0.307, 0.0, 0.35]
  home_qpos: [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
  finger_open: 0.04
  fingertip_offset: 0.1034
  workspace: {x: [0.28, 0.62], y: [-0.30, 0.40], z: [0.115, 0.35]}

controller:                  # 流用元 app.make_controller（DLS）と collect の上書き
  ik_gain: 0.08
  max_joint_step: 0.03
  rot_weight: 0.5
  lock_orientation: true
  null_gain: 0.05
  ik_damping: 0.05
  tracker_omega: 20.0
  max_target_vel: 0.8
  q_des_leash: 0.10
  max_dx_norm: 0.10
  scale: 1.0

pad:                         # 流用元 dualsense_device.PROFILES["collect"]
  xy_speed: 0.20
  z_speed: 0.10

scene:
  cube_size: 0.04
  cube_mass: 0.05
  cube_friction: [1.0, 0.005, 0.0001]
  colors: {red: [1, 0, 0, 1], green: [0, 1, 0, 1], blue: [0, 0, 1, 1]}
  box: {pos: [0.45, 0.25, 0.0], inner_half: 0.06, wall_top_z: 0.06, success_inner_half: 0.05}
  region: {x: [0.33, 0.57], y: [-0.18, 0.08]}     # Step D で視野と到達性を見て確定
  min_center_dist: 0.09
  yaw_range_deg: 30
  layout_kinds: {empty: 0.57, prefilled_1: 0.14, prefilled_2: 0.29}
  start_at_retreat_prob: 0.33
  box_slots: null            # 箱の中の置き場所の候補（Step D で決める。§14 の R3）
  box_slot_clearance_m: {cube: 0.02, wall: 0.01}   # 着地した立方体どうしの隙間・壁との隙間の下限（mj_geomDistance）。指と壁の隙間は条件にしない（壁の上端より上で放すため）
  settle_prefilled_s: 1.0
  presentation_camera: {name: presentation, width: 1280, height: 720, pos: null, xyaxes: null}

expert:
  speed_scale: [0.9, 1.1]    # ±10%
  gain_per_s: 4.0
  grasp_jitter_m: 0.003
  approach_height: null      # 流用元は作業空間の上限 0.35 を経由。Step D で決める
  grasp_z: 0.115             # 流用元 LOW_Z（作業空間の下限）
  lift_m: 0.05
  carry_z: 0.25
  release_z: 0.18            # 流用元 RELEASE_Z
  retreat_pose: null         # 箱の中が俯瞰カメラから見える位置（Step D）
  idle_after_place_s: 1.5
  move_tol_m: 0.001
  close_settle_s: 0.3        # 閉じ切りの判定に使う指の速さの持続時間
  slip_retry_max: 3
  phase: {grasped_lift_min_m: 0.01, drop_dist_m: 0.035, rest_speed: 0.01, rest_hold_s: 0.3}

inject:
  A: {lateral_offset_m: [0.025, 0.035], raise_close_m: [0.015, 0.025], raise_ratio: 0.5, lift_m: [0.03, 0.05],
      confirm: {hand_rise_min_m: 0.03, cube_rise_max_m: 0.01}}
  B: {min_dist_from_box_m: 0.15}
  C: {min_clearance_m: 0.03}
  landing: {max_tilt_deg: 10, min_clearance_m: 0.03, max_invalid_ratio: 0.2,
            rest_speed: 0.01, rest_hold_s: 0.3, finger_gap_margin_m: 0.005}

contact:
  robot_bodies: [hand, left_finger, right_finger]
  obstacles: [table_cubes_except_target, box_walls]
  distmax_m: 0.10            # これより遠い組は distmax を返す（計算の打ち切り）

safety_filter:               # 条件付き（接触率 5% 以上のときだけ作る）
  enabled: false
  d_min_m: null
  d_detect_m: null
  gamma: 0.85
  tracking_lag_margin_m: 0.017   # 参照から手先までの遅れ 5〜17mm（落とし穴13）

convert:
  fps: 10
  instruction: "put the {color} cube in the box"

train:
  scope: expert
  batch_size: 32
  seed: 1000
  num_workers: 6
  log_freq: 50
  gpu_limit_gib: 14.0
  rename_map: {observation.images.image: observation.images.camera1,
               observation.images.image2: observation.images.camera2}
  runs:
    smoke: {steps: 1000}
    K1: {steps: 8000, save_freq: 8000}
    R1: {steps: 30000, save_freq: 5000}
    N1: {steps: 30000, save_freq: 5000}
    second_round: {steps: 10000, save_freq: 5000, optimizer_lr: 3.0e-5,
                   warmup_steps: 200, decay_steps: 10000, decay_lr: 2.5e-6}   # §10

runtime:
  mode: rtc                  # sync | naive | rtc
  exec_interval: 10
  delay_steps: 2
  rtc_guidance_horizon: 10   # 10 と 40 を比べる
  rtc_schedule: EXP          # LINEAR と EXP を比べる。必ず明示（落とし穴16）
  rtc_max_guidance_weight: 10.0

eval:
  time_limit_s: 30
  success: {rest_speed: 0.01, rest_hold_s: 1.0}
  error_lift_m: 0.02
  collateral_move_m: 0.02
  reaction_close_m: 0.02
  recovery_lift_m: 0.02
  P1: {delay_s: 0.3, lateral_offset_m: [0.025, 0.035], confirm_window_s: 2.0, confirm_lift_m: 0.02}
  P2: {min_dist_from_box_m: 0.15, min_lift_m: 0.05}
  P3: {min_clearance_m: 0.03}
  induction_min_rate: 0.8

seeds: {}                    # §9 の表（帯と用途）をそのまま書く

planner:
  model: claude-haiku-4-5
  temperature: 0.0
  max_tokens: 1024
  timeout_s: 30
  step_timeout_s: 30
  retry: 1
  completion_hold_s: 1.0
  color_detect: null         # 色ごとの閾値（Step I）

viz: {fps: 30}
```

---

## 6. エピソード番号と保存の方式

学生用の台帳は使わない。**番号の数え上げをやめ、中身から決まる名前にする。**

- 1 回の生成を 1 つの「回」として `outputs\gen\<目的>_<日時>\` に書く。フォルダが既にあれば拒否する（流用元の出力先の決まりと同じ）
- エピソードの名前は `<種類>_<配置の種>_<色>_r<作り直しの回数>`（種類 `n` 通常、`A`・`B`・`C` 復帰、`h` 引き継ぎ）。回の中では一意に決まり、同じ設定で作り直せば同じ名前になる。保存中は `.partial`、保存が済んだら名前を変える（流用元の方式）
- 回ごとに `generation.jsonl` を書く。試みた全件（捨てたものを含む）について、名前、種、結果の区分、所要時間、コードの版を 1 行ずつ残す
- 整数の番号は変換のときに初めて付ける（LeRobot の `episode_index`）。`conversion.json` の `sources` に、回・名前・メタデータの旗との対応を残す
- **マニフェスト**（`outputs\manifests\<名前>.json`）: `{name, created, base_commit, rule, entries: [{run, key}], composition}`。ラベルは要求しない。変換器はマニフェストに載った回と名前だけを読み、見つからなければ変換の前に止まる（流用元の検査の考え方は継ぐ）

---

## 7. 部品ごとのインターフェース

型は概略。すべての数値は `cfg`（§5）から取る。

**場面（`sim/scene.py`、`sim/frames.py`）**
- `build_model(cfg, variant: "g0"|"3cube") -> MjModel`: 3 色の場面は `scene_3cube.xml` から作る（XML を 1 つにし、コードで生成しない。色は XML に書く）
- `Layout`（dataclass）: `seed, kind, cubes: {color: (x, y, yaw)}, prefilled: {color: slot}, start: "home"|"retreat"`
- `sample_layout(rng, cfg, kind) -> Layout`（範囲・中心間距離・向きの棄却抽選）、`apply_layout(model, data, layout)`（先客を置いて静止させるまで）
- `frames.py` に座標と幾何を集約する: 世界→カメラへの投影（`project(camera, point, size)`、流用元の `OperatorView.project` から）、指先の中心、箱の内寸・成功の体積・壁の幾何、画像の反転は `vla_image_spec` を参照するだけ

**制御ループと指令の入口（`sim/control.py`、`sim/device.py`）**
- `SimRig(cfg, variant)`: `model, data, controller, integrator, pad, renderer, sampler`。`reset(layout)`、`step(n)`（`controller.update(integrator)` → `mj_step` → フック）
- 指令の入口は流用元と同じ: `pad.state = pad_state(vel=…, button_grip=…)` → `integrator.refresh()`。台本・方策・誘発はこれ以外の経路で腕を動かさない
- **安全フィルタの差し込み口**: `VelocityCommandIntegrator.read()` の `x_cmd += vel·dt` の直前に `command_filter(x_cmd, dx, data) -> dx` を置く。既定は恒等関数。`safety_filter.enabled` のときだけ差し替える。台本・方策・誘発の全部が同じ場所を通る

**接触と最小距離（`sim/contact.py`）**
- `ContactMeter(model, cfg)`: `set_obstacles(target_color, cubes_in_box)`（手順・目標が変わるたびに組を作り直す）、`measure(data) -> {min_dist: {class: float}, closest: …, contact: {class: bool}}`
- 距離は `mj_geomDistance(m, d, g_robot, g_obs, distmax, fromto)`。対象はロボット側の衝突形状（hand の衝突メッシュ、両指の衝突メッシュと当て板の箱）× 障害物（机上の目標外の立方体、箱の壁 4 枚）。接触は物理ステップごとの `d.contact` を窓（20 Hz）の中で OR する（流用元の `finger_contacts` の考え方）

**台本（`expert/script.py`）→ §8**
- `phase_of(truth, target, cfg) -> Phase`（純関数。評価器の段階の判定と共用）、`Expert.act(truth, target) -> Command(vel, grip_closed)`

**注入（`expert/inject.py`）**
- `Injector(kind, params)`: `modify(command, truth) -> command`（A: 把持目標のずらし、B・C: 開く指令）、`save_start_reached(truth) -> bool`（手順書 Step F の 2 の物理条件）。パラメータは種から前もって決める

**記録・分割・途中状態（`record/`）**
- `SimSnapshot.capture(rig) -> SimSnapshot` / `restore(rig, snap)`: `mj_getState(mjSTATE_INTEGRATION)`（qpos、qvel、act、ctrl、mocap、warmstart など）、制御器の内部状態（`q_des, target_pos, target_vel, desired_pos, target_quat, gripper_closed, prev_grip, prev_clutch`、クラッチの基準 4 つ）、積分器（`x_cmd, latest, enabled`）、パッドの状態、物理ステップの番号（20 Hz・10 Hz の位相）
- `EpisodeWriter`（流用元を拡張）: フレームごとに立方体 3 個の姿勢と速度、最小距離、接触、段階、安全フィルタの作動を足す。`start(snapshot)` で途中から保存を始められる
- `replay(episode, mode, from_snapshot=None)`: 途中状態から再生する（Step F の 6）

**変換（`data/convert.py`）**: `convert(manifest, out)`、`verify(out)`（流用元）。`task` はエピソードの `instruction`。メタデータの旗は `conversion.json` の `sources` に持ち越す

**方策の実行器（`policy/runner.py`）→ §9 と合わせて**
- `PolicyRunner(checkpoint, cfg.runtime)`: `start_trial(seed_streams)`、`reset_chunk()`（指示の切り替え・やり直しの直後）、`action(k, frame, raw_images, task) -> action7`
  - sync: `k % s == 0` で推論し、物理は止まっている
  - naive / rtc: 推論は直前の推論の開始から `s` 手ごと（k の時点の観測）。結果は `k + d` から有効で、新しい塊の `d` 手目以降を使う。それまでは前の塊を続ける
  - rtc: 読み込み後に `policy.config.rtc_config = RTCConfig(enabled=True, prefix_attention_schedule=…, max_guidance_weight=…, execution_horizon=…)` を入れて `policy.init_rtc_processor()` を呼ぶ。前の塊の残りは**後処理の前（正規化された空間）**の、推論開始時点で未実行の部分を渡し、`execution_horizon` の長さに切り詰め・ゼロ詰めする。`predict_action_chunk(batch, noise=…, inference_delay=d, prev_chunk_left_over=…)`。同梱の `rollout/inference/rtc.py`（`ActionQueue.get_left_over`、`_normalize_prev_actions_length`、`merge`）と同じ扱い。`select_action` は使わない
  - 推論の実時間（`torch.cuda.synchronize` 込み）、塊の境目、塊そのもの（手先の予測経路を積算したもの）を記録する

**誘発（`eval/induce.py`）**: `Inducer(kind, params)`: `filter(command, truth, t) -> command`、`status -> {fired, established, t_fire, t_confirm}`。パラメータは試行の開始時に誘発の乱数列から全部引く（§9）

**集計（`eval/metrics.py`、`eval/stats.py`）**: 指標は試行の配列を受け取る純関数（手順書 Step G の 5）。統計は `wilson_interval`（流用）、`mcnemar_exact`（`scipy.stats.binomtest`）、対応ありの差の 95% 区間（Newcombe の方法 10）、`wilcoxon`（`scipy.stats.wilcoxon`）

**LLM・完了判定（`planner/`）**: `decompose(text, table_colors, box_colors) -> {"steps": [...], "reply": str}`、`ColorDetector.table_colors(img) / box_colors(img)`（箱の内寸を `frames.project` で投影）、`CompletionJudge.update(obs) -> bool`、`TaskExecutor.run(text)`

**描画（`viz/`）**: `PresentationRenderer`（1280×720、30 fps、シミュレーション時刻に同期）、重ね表示の関数群。発表用カメラは `scene_3cube.xml` に最初から入れておく（カメラを足しても記録用カメラの描画は変わらないことを Step D で確かめる）

---

## 8. 台本の作り

**内部に段階を持たない。毎周期（入力の読み取りごと、20 ms）、真値から今の段階を判定し、その段階の目標へ向かう指令を 1 回分だけ出す。**

真値: 目標の立方体の位置・姿勢・速さ、指の開き（2 本）と速さ、グリッパの指令の状態（`controller.gripper_closed`）、手先の位置、x_cmd、箱の位置、障害物との距離。

段階の判定（`phase_of`、上から順に最初に当てはまるもの）:

| 段階 | 条件（数値は設定） | 出す指令 |
|---|---|---|
| done | 目標が箱の中で静止し、手が待機位置にあり、グリッパが開いている | 静止（待機の 1.5 秒を数える） |
| retreat | 目標が箱の中で静止し、グリッパが開いている | 待機位置へ（まず真上へ上がってから） |
| settle | 把持していない（下の carry・lift の条件に当たらない）うえで、目標の速さが `rest_speed` 以上、または目標が箱の中にあってまだ静止していない | その場で静止（グリッパの指令は変えない）。立方体が止まるのを待つ |
| release | 把持中で、手が箱の上の解放位置にある | 開く |
| carry | 把持中（閉じていて、目標が指先の中心から `drop_dist` 以内、机から `grasped_lift_min` 以上浮いている） | 搬送高さ → 解放位置へ |
| lift | 閉じていて目標が指の間にあるが、まだ浮いていない | 真上へ `lift_m` |
| close | 開いていて、手が目標の把持点（±ずれ）の把持高さにあり、止まっている | 閉じる |
| descend | 開いていて、手が目標の真上にある | 把持高さへ下ろす |
| reopen | 閉じているのに目標が指の間にない（掴み損ね、落下の後） | 開いて真上へ上がる |
| approach | 上のどれでもない | 接近の高さを保ったまま目標の真上へ |

- **静止**の判定は、目標の速さが `expert.phase.rest_speed` 未満の状態が `expert.phase.rest_hold_s` 続くこと。done・retreat・settle で同じ判定を使う。settle があるので、転がっている立方体や、箱の中でまだ落ち着いていない立方体に向かって descend に入ることはない。把持中は立方体が手と一緒に動くので、settle の条件から外す
- **settle の単体検査**（`tests/expert/test_phase.py`）: B の注入の直後、C の注入の直後、箱の上で放した直後の 3 つの場面で、`phase_of` の列に settle が現れ、その間に descend が現れないこと。記録から切り出した真値の列と、作った真値の両方で確かめる
- **待ち時間は状態から数え直す。** 「閉じてから 0.3 秒」は「指の速さが一定以下になってから 0.3 秒」に置き換える。待機の 1.5 秒は、done に入ってからの経過で数える。数えた値は引き継ぐと失われるので、引き継いだ直後は最初から数え直す（長めに待つ側に倒れるだけで、動作は変わらない）
- 把持点のずれ（±3 mm）・速さのばらつき（±10%）・接近の経由点のずれは、**台本の乱数列から、エピソードの開始時に一度だけ**引いて固定する。これらは段階ではなくパラメータなので、引き継いでも同じ値を使える（引き継ぎのときは、新しい台本の種から引き直す）
- 立方体の向き（±30°）に合わせて手首を回すことはしない（ヨーは固定。スコープ外）
- 引き継ぎ（Step D 完了条件3、Step F、R2）: 任意の時点で `Expert` を作り直し、同じ真値から段階を判定して続ける。専用の復帰の分岐を作らず、上の表の reopen → approach → … がそのまま立て直しになる
- 自然に滑ったときの扱い（Step D の 3）は、台本の外（生成の枠）で決める。carry 中に目標が指から外れたら、その試みを「滑り」として打ち切り、作り直しの対象にする

---

## 9. 乱数の作り

**1 つの試行の種から、名前つきの独立した乱数列を作る。**

```
streams(seed) = { name: np.random.Generator(PCG64(SeedSequence(seed, spawn_key=(ID[name],)))) }
ID = {layout: 0, induce: 1, noise: 2, script: 3, inject: 4, order: 5}   # 番号は固定し、変えない
```

- **配置**（`layout`）と**誘発**（`induce`）は、試行の開始時に必要な値を全部引いてしまう（P1 のずらす量と向き、P2 の割合 u、P3 のずらし先）。途中で引かないので、推論の回数や方策の動きが違っても、同じ種なら同じ配置と同じ誘発になる
- P2 の「搬送経路のうち、箱の中心から 15 cm 以上離れた部分で一様に選んだ地点」は、方策の経路が前もって分からない。そこで、持ち上げ（5 cm）を検出した時点の箱までの水平距離を r0 とし、距離が `0.15 + u·(r0 − 0.15)` を切った時点で開く（経路の進み具合について一様）。r0 < 0.15 なら不成立として記録する
- **方策の雑音**（`noise`）: k 回目の推論の雑音は `SeedSequence(seed, spawn_key=(2, k))` から作る `torch.Generator` で引く。推論の回数が違う条件でも k 回目どうしは同じ雑音になり、naive で d=0 のときは sync と一致する（Step G 完了条件2）
- 生成では、配置の種から `layout`、台本の種から `script`・`inject` を作る。台本の種は配置の種・色・作り直しの回数から `SeedSequence(layout_seed, spawn_key=(3, color_index, retry))` で決まり、記録には `(layout_seed, color, retry)` を残す

**種の帯**（学習用は 100000 未満、評価は 100000 以上。流用元の決まりを継ぐ）

| 帯 | 用途 |
|---|---|
| 0〜2999 | 流用元の学習配置（G0 の `training_placements` だけ）。新しい場面では使わない |
| 10000〜10999 | K1 の配置 |
| 20000〜29999 | 通常デモの配置（R1・N1 共通） |
| 30000〜39999 | 復帰の配置（A・B・C） |
| 40000〜49999 | R2 の引き継ぎの配置 |
| 50000〜59999 | Step D・F の試験（台本の成功率、引き継ぎ試験、再生確認） |
| 100000〜100019 | G0 の照合だけ（予備実験と同じ種） |
| 100020〜109999 | 使わない（予備実験で使用済み） |
| 110000〜110032 | 最終: E1・E3・E5 の自然試行（33 配置 × 3 色） |
| 111000〜111049 / 112000〜112049 / 113000〜113049 | 最終: P1 / P2（E2・E3・E4 共通） / P3 |
| 114000〜114032 | 最終: E6 |
| 115000〜115019 | 最終: E7 の複数手順の通し |
| 190000〜190032 | 選択: K1 の E6 |
| 191000〜191029 | 選択: K1 の閉ループ 30 回 |
| 192000〜192029 | 選択: RTC の設定の 2×2（P2 誘発） |
| 193000〜193029 | 選択: 保存点の選択 |
| 194000〜194032 / 195000〜195049 | 選択: G2 の自然 99 回 / P2 50 回 |
| 196000〜196199 | 選択: Step I の完了判定の照合データ |
| 197000〜197019 | 選択: G3・Step I の複数手順 |
| 198000〜198999 | 選択: Step G の検査（誘発の発生、実行器の一致など） |

最終評価用の帯（110000〜189999）を Step J より前に使っていないことは、評価の記録から機械的に検査する（Step J の最初に実行）。

---

## 10. 2 周目の学習の作り（R2・R1+）

**統計量の固定**: `--policy.path=<R1 の選んだ保存点>/pretrained_model` で始めると、統計量は学習データの `meta/stats.json` で上書きされる（Step A §2-17）。そこで、**学習に渡すデータセットの `meta/stats.json` を R1 のデータセットのものに置き換える**。

1. R2 のデータ（R1 のデータ＋引き継ぎ区間）を普通に変換し、`--verify` を通す（検証は自分の統計量で行う）
2. 学習用の写し（`outputs\datasets\<名前>_statsR1\`）を作り、`meta/stats.json` だけを R1 のデータセットのものに置き換える。`conversion.json` に `stats_source`（R1 のデータセットの指紋）を書く
3. R1+ は R1 と同じデータなので統計量は同じになるはずだが、同じ手順で置き換えて扱いを揃える
4. **検査**: 学習後の各保存点の `policy_preprocessor_step_5_normalizer_processor.safetensors` と `policy_postprocessor_step_0_unnormalizer_processor.safetensors` の中身（observation.state と action の mean・std）が、R1 の保存点のものとビット単位で一致すること。起動器が学習の後に自動で比べ、結果を `train_run.json` に書く

`--resume` は使わない。データセットと手数が変わる続きの学習では、resume は保存時の設定を読み直すので扱いにくい。ここで使う仕組み（`meta/stats.json` から統計量を読むこと、`--policy.path` で上書きされること）は、Step C の 10 手の学習で先に確かめる。

**学習率の予定**: 最適化器は作り直しになる（AdamW の状態は引き継がない）。R1 は 25000 手で 2.5e-6 まで下がり、最後の 5000 手はそのまま（Step A §2-18）。1e-4 から立ち上げ直すと R1 の性質を崩す恐れがあるので、`--policy.optimizer_lr=3e-5 --policy.scheduler_warmup_steps=200 --policy.scheduler_decay_steps=10000 --policy.scheduler_decay_lr=2.5e-6` とする（ピーク 3e-5 は、R1 の cosine（25000 手で 1e-4 → 2.5e-6）の約 1 万 6 千手目の学習率に相当）。**R2 と R1+ で同じ予定を使う**ので、比べたときの差はデータだけから来る。起動器の設定キーとして持たせ、学習率の値は `train_run.json` と学習ログから検査する。

---

## 11. K1 の分岐の費用

分岐は K1 が不合格のときだけ発動する。以下は計算での見積もりで、分岐を採るときに 50 手の試し学習で実測してから確定させる（1 回あたり数分）。

**LoRA（指示依存度 30% 未満のとき）**
- 対象: SmolVLM2 のテキスト部の上位の層（16 層のうち 12〜15 の 4 層、または 8〜15 の 8 層）の `q_proj`・`v_proj`、rank 16。**行動エキスパートと射影層（`state_proj`、`action_in_proj`、`action_out_proj`、`action_time_mlp_in/out`）は `--peft.full_training_modules` に入れて、これまでどおり全部学習する**（入れないと PEFT がそれらを凍結し、既定の対象（エキスパートの q・v だけの LoRA）より弱くなる）。`train_expert_only=true`・`freeze_vision_encoder=true` のまま（VLM 本体は凍結、LoRA だけが増える）
- 映像用メモリ: 現状（バッチ 32、エキスパートのみ）7.69 GiB。LoRA を入れた層より上の VLM の活性化を逆伝播のために持つ分が増える。1 画像あたり約 64 トークン（SigLIP 512 px とピクセルシャッフルからの推定）× 2 画像＋言語 48＋状態 1 ≈ 177 トークン、隠れ 960 次元、bf16 で、1 層あたり約 0.25 GiB（バッチ 32）。4 層で +1 GiB 前後、8 層で +2 GiB 前後 → **9〜10 GiB で 14 GiB に収まる見込み**
- 1 手あたりの時間: VLM の上位層の逆伝播が増え、+10〜30%（0.45〜0.55 秒/手）。K1 の 8000 手は 1 時間 → 1.1〜1.3 時間、R1・N1 の 3 万手は 3.5 → 4〜4.5 時間ずつ
- 追加: peft を入れる（§3 の 7 と同じ方法）。推論の時間はほぼ変わらない（LoRA を重みに畳み込めるため）

**CFG（指示依存度 30% 以上だが正答率 90% 未満、または中間落ち率 20% 以上のとき）**
- 学習: 変換のときに約 1 割のエピソードを、指示を空（`task=""`）にした写しとして追加する（データ側だけで済み、学習コードは変えない）
- 推論: 条件つき・条件なしの 2 つの速度場が要る。バッチを 2（同じ画像・状態、指示だけ違う）にして 1 回で計算し、`v = v_u + w·(v_c − v_u)` にする。前半（VLM の KV キャッシュ）も後半（10 段のオイラー）もバッチ 2 になるので、推論時間は 1.3〜2 倍（GPU の並列でバッチ 2 は 2 倍より小さい）
- RTC への組み込み: 同梱の `RTCProcessor.denoise_step` は「x_t から v_t を返す関数」を包む作りなので、その関数を CFG で合成した速度場にすれば、RTC の誘導はそのまま合成後の速度場に掛かる。SmolVLA の `VLAFlowMatching.sample_actions` を `recovla` 側の小さな派生クラスで上書きして実現する（LeRobot のファイルは書き換えない）。誘導の勾配も合成後の速度場を通るので、RTC ありの推論は RTC なしの 2 倍強
- 推論時間は模擬の中では物理を止めて計るので、成否には影響しない。ただし実測が d×0.1 秒（d=2 で 0.2 秒）を超えたら報告する（手順書 Step G の 1）

---

## 12. LLM の選択と、鍵の置き場所

- **Claude API の `claude-haiku-4-5`** を使う（計画書 §4.5 の「軽量モデル」。入力 $1、出力 $5 / 100 万トークン）。1 回の呼び出しは入力 1 千トークン弱・出力 100 トークン程度で、Step I の検査（30 文＋20 回の通し）とデモを合わせても 1 ドルに届かない。温度 0
- 出力は JSON（`{"steps": [...], "reply": "..."}`）。Haiku 4.5 が構造化出力（`output_config.format` / `messages.parse`）に対応しているかは、Step I の最初に Models API の `capabilities` で確かめる。対応していなければ `strict: true` のツール 1 つで同じ形を受け取る。どちらの場合も、受け取った後に自前で検査する（色は机上の色の部分集合か、箱の中の色を含まないか、重複がないか）。検査に落ちたら手順を空にして人に尋ねる側に倒す
- **鍵**: 環境変数 `ANTHROPIC_API_KEY` で渡す。リポジトリ・設定ファイル・ログに書かない。起動時に鍵の有無だけを確かめ、値は表示しない
- **回線**: 計算機全体のプロキシ（`HTTPS_PROXY`）はそのまま効く。TLS の中継に対しては、`.local\ca_bundle_proxy.pem`（§2.2）から `ssl.create_default_context(cafile=…)` を作り、SDK の `DefaultHttpxClient(verify=…)` に渡す（環境変数の読み方に頼らない）。Step I の最初に 1 回呼んで確かめる
- **再現性**: 呼び出しと応答を `outputs\llm_cache\` に保存する（キーはモデル名・プロンプト・入力の SHA-256）。デモの作り直し（Step J の 4）と、ネットワークのない場所での再現はこの記録から行う
- ローカルの小型モデルは、Claude API が使えないと分かったときだけ用意する（その時点で候補と取得量を報告する）

---

## 13. 生成の並列化

- 見込み: 流用元の台本は 1 本 2.8 秒（16 秒のエピソード）。3 色の場面は待機を含めて 20〜25 秒になり、距離の計測も加わるので、1 本 4〜5 秒と見込む。R1・N1 の生成は、捨てる分を含めて 500 回前後の試みで、1 プロセスでも 40 分程度。**並列化は必須ではないが、Step D で本/時を測るために作る**
- 方式: `multiprocessing` の spawn（Windows の既定）。`Pool(initializer=…)` で、各プロセスが自分の `MjModel`・`MjData`・`Renderer`（描画の文脈）を 1 回だけ作る。仕事の単位は「1 エピソードの指定」（配置の種、色、種類、作り直しの回数）という純粋なデータ。各プロセスは自分のエピソードのフォルダにだけ書き、親が `generation.jsonl` を指定の順に書く（書く順を並列数に依存させない）
- 決定性: エピソードの中身は指定だけで決まり、どのプロセスで作っても同じになるはずである。Step D の完了条件2（組の最初のこまの一致）を、並列数 1 と 8 の両方で確かめる
- 並列数は 1・4・8・12 で本/時を測り、主記憶（1 プロセス 0.3〜0.5 GB の見込み）と GPU の描画の競合を見て決める。学習とは同時に走らせない（Step F と G の並行の規則どおり）

---

## 14. 危険と未決事項

| # | 内容 | 影響 | 対応 |
|---|---|---|---|
| R1 | 台本の書き直し（状態から段階を判定する方式）が 95% の成功率に届くまでの調整時間 | G1 前半（10/3）の遅れ | 流用元の経由点・高さ・速さの値から始める。段階の判定を純関数にして、記録から段階の列を見て調整する |
| R2 | 3 個・中心間 9 cm・向き ±30° を満たす配置が、広げた範囲の中で取りにくい、または手が届かない・写らない | 配置の抽選の失敗、学習外の配置の偏り | Step D で棄却率・視野・到達性を測り、範囲を確定する（設定の `region`） |
| R3 | **箱が 3 個には狭い。** 内寸 12 cm に一辺 4 cm を 3 個。流用元は壁の上端より上（指先 z≈0.077）で放しているので指と壁は当たらないが、先客の上に落ちて傾く・縁に乗る恐れがある | 先客 2 個の配置の成功率、完了判定 | 置き場所の条件は「**着地した**立方体どうしの隙間 2 cm 以上、壁との隙間 1 cm 以上」（`box_slot_clearance_m`、形状間の最短距離で測る）。放すのは壁の上端より上なので、指と壁の隙間は条件にしない。候補は 2×2 の格子（中心 ±3 cm）の空きから選ぶ。**注意**: 内寸 12 cm では、向き 0° で格子の中心にぴったり着地したときにちょうど条件を満たす（隙間 2 cm・1 cm）だけで余裕がない。向きが 30° だと壁との隙間は約 3 mm になる。Step D で、着地の位置と向きのばらつきと、条件を満たした割合を先客 0・1・2 個ごとに示す。足りなければ、放す高さを下げる、または箱を広げる（課題の定義の変更になるので相談） |
| R4 | 台本の把持は手首の向きが固定。向き 30° 近くでは滑る（予備実験の失敗 8 本はすべて搬送中、向き 40° 超） | 作り直しの増加、配置の偏り | 範囲は ±30° に絞った（計画書のとおり）。Step D で向きごとの成功率を示す |
| R5 | `mj_geomDistance` がメッシュの衝突形状（hand、指）で遅い、または期待と違う値を返す | 生成の速さ、最小距離の信頼性 | Step D の最初に、既知の配置で解析値と比べる。問題があれば、手・指を箱とカプセルで近似した距離を自前で計算する |
| R6 | 描画の非決定性（プロセス・GPU の状態） | 反実仮想の揃え（組の最初のこまの一致） | Step D の完了条件2を並列数を変えて確かめる。流用元の `close_renderer` の注意（文脈の破棄順）を継ぐ |
| R7 | RTC の誘導は推論のたびに勾配を計算するので、推論時間が伸びる | d の実測が 0.2 秒を超える | 比較は固定の d で行い、実測を記録して報告する（計画書どおり） |
| R8 | 統計量の固定（§10）と学習率の上書きが、この版で意図どおり効かない | 2 周目の比較の妥当性 | Step C の 10 手の学習で先に確かめる |
| R9 | TLS の中継のため、uv・pip・Anthropic SDK のどれかが証明書で止まる | 環境の構築、LLM | 学生用 README の方法（`UV_SYSTEM_CERTS`、証明書の束）に従う。計算機全体の設定は変えずに止まって報告する |
| R10 | SmolVLM2 の読み込みが、ネットワークなしで何かを取りに行く（処理器の設定など） | Step C の完了条件1 | HF のキャッシュの構成ごと複写する。Step C で `HF_HUB_OFFLINE=1` のまま読み込みと推論を通す |
| R11 | 学習は開発者モード（シンボリックリンク）と、ページファイル 16 GB（num_workers 6）に依存している。どちらもこの PC では設定済み | 別の PC での再現（Step J の完了条件3） | README の前提として書く。この PC の設定は変えない |
| R12 | 日程が詰まっている（G0 9/30、G1 前半 10/3、K1 10/5） | 後ろの Step の圧迫 | 並行してよい Step（E の学習中の F、F の生成中の G）を最大限に使う |

**未決事項（承認をお願いしたいこと）**

1. **ディレクトリ構成**（§1）と、パッケージ名 `recovla`
2. **コードの取り込みを `git archive`（コミット dbb2c3b から）で行うこと**（§2.1）と、自前の MinGit・uv を GitHub から取得すること（§3.2）
3. **`ca_bundle_proxy.pem` を `.local\` に複写すること**（§2.2。公開の証明書で、Git 管理外）
4. **2 周目の学習率**: ピーク 3e-5・立ち上がり 200 手・1 万手で 2.5e-6 まで（§10）
5. **種の帯の割り当て**（§9）
6. **LLM は `claude-haiku-4-5`**（§12）。鍵の準備（`ANTHROPIC_API_KEY`）は Step I の前にお願いする
7. ~~`<ROOT>` の Git のコミットの作者名とメールアドレスと、GitHub のリポジトリ名・アカウント~~ → 掲示板 0002・0004 で決定済み（公開の VLAcompetition、作者は GitHub の非公開用のアドレス。§3.2 の 10）。gh の認証は人の操作が要る
8. R3（箱の広さ）: まず 2×2 の置き場所で進め、先客 2 個の組の成功率が足りなければ相談する、という進め方でよいか
9. **支線の分け方**（§15）: 枝の数と名前、境目、それぞれの開始の条件

---

## 15. クラウドに回す支線の候補

GPU を使わない部品を、クラウドの Claude Code セッション（支線）に回す。本線（この PC）は GPU を使う作業と統合、判断点を受け持つ。

### 15.1 共通の約束

- **枝の名前**は `cloud/<名前>`、掲示板の差出は `cloud-<名前>`。支線は `main` に直接 push せず、PR で返す。取り込むのは本線
- **境目**: 各支線が書いてよいのは、下の表の「持ち場」のパスだけ。共有の部分（`configs/default.yaml`、`src/recovla/common/`、`src/recovla/sim/` の既存のインターフェース、`docs/interfaces/`）は本線が持つ。変えたいときは掲示板に ask を出す
- **インターフェースは本線が先に決める**: 各支線の開始前に、本線が `docs/interfaces/<名前>.md`（データの形、関数の型、単位、座標系）と、必要なら型だけの空の実装（スタブ）を main に入れる。支線はそれに合わせて作る
- **支線の検査**は、GPU・学習済みモデル・C:\VLA・ネットワークなしで通ること（LLM の支線は応答の模擬で通す）。環境は `env\requirements-cloud.txt`
- **数字の扱い**: 支線が出した数字（台本の成功率など）は参考とし、完了条件の判定は本線がこの PC で測り直した値で行う（MuJoCo の計算は OS・CPU で末尾の桁が変わりうるため）
- 支線に C:\VLA の中身は渡さない。リポジトリに入っているもの（流用して `<ROOT>` に取り込んだコード・場面を含む）だけで作業する

### 15.2 候補

| 枝 | 持ち場（書いてよいパス） | 中身 | 境目（本線が持つもの） | 開始の条件 | 返すもの（完了条件） |
|---|---|---|---|---|---|
| `cloud/expert` | `src/recovla/expert/`、`tests/expert/` | 台本の段階の判定（§8）と注入（A・B・C）の調整。画像を描かずに物理だけで回し、成功率・作り直しの内訳・着地の区分・最小距離を測る | 場面（`scene_3cube.xml`）、`SimRig`・指令の入口・`ContactMeter` のインターフェース、本番のデータ生成（描画と決定性の確認）、Step D・F の完了条件の判定 | Step D の最初に本線が場面と `SimRig` の骨組みと `docs/interfaces/expert.md` を main に入れた後 | 配置の種類ごと 100 本以上の成功率（物理のみ）、引き継ぎ試験 50 回、注入ごとの 4 区分の表、`phase_of` の単体検査 |
| `cloud/metrics` | `src/recovla/eval/metrics.py`、`src/recovla/eval/stats.py`、`tests/eval/` | 指標（成功・誤り・巻き添え・段階・復帰・反応時間・復帰時間・継ぎ目の跳び・躍度・接触回数）と統計（Wilson、McNemar の正確検定、対応ありの差の区間、Wilcoxon） | 試行の記録の形（`docs/interfaces/trial_record.md`）、評価器本体 | Step C の後（記録の形を本線が決めた時点） | 跳びと時刻が既知の合成軌跡での検算（Step G 完了条件5）、既知の数値例での統計の照合 |
| `cloud/runner` | `src/recovla/policy/schedule.py`、`tests/policy/` | 実行器の時間の流れ（sync・naive・rtc の切り替え、推論の開始、d 手の遅延、塊の差し替え、前の塊の残りの切り出しと長さの揃え、持ち越さない条件、塊の境目の記録）。方策は決まった塊を返す模擬で置き換える | LeRobot の方策の呼び出し（`predict_action_chunk`、`RTCConfig` の有効化）の部分と、実モデルでの確認（Step G 完了条件6） | Step C の後 | naive で d=0 が sync と一致、同じ種で配置と誘発の条件が方式によらず一致（Step G 完了条件2・3 の模擬版）、残りの切り出しが `rollout/inference/rtc.py` と同じ扱いであることの検査 |
| `cloud/planner` | `src/recovla/planner/decompose.py`、`tests/planner/`（固定の試験データを除く） | LLM による分解のプロンプト、JSON の形と検査（色の部分集合、箱の中の色の除外、重複）、応答の保存 | 実 API での採点（鍵は本線だけが持つ）、色の検出と完了判定（描画した画像が要る）、手順の実行器。**日本語の指示 30 文と期待する出力は監督が書き、固定の試験データとして main に入れる**（`tests/planner/fixtures/`。支線は変えない） | Step G の頃（固定の試験データが main に入った後） | 検査ロジックの単体検査（模擬応答。固定の試験データを使う）、プロンプトの確定案。正解率は本線が実 API で測る（Step I 完了条件2） |
| `cloud/figures` | `src/recovla/eval/report.py`、`src/recovla/viz/overlay.py`、`tests/viz/` | 結果の表（CSV・md）と図、動画の重ね表示の部品（手順の一覧、出来事の名前、塊の予測経路の色分け、手先速度の波形）。合成データで作る | 結果の CSV の形（`docs/interfaces/results.md`）、MuJoCo での発表用の描画、動画の書き出し | Step G の頃（`results.md` が main に入った後）。本番の数字が出る Step J までに | 合成データでの表・図・重ね表示の見本と、その検査 |

### 15.3 本線に残すもの

データの本番の生成（描画、決定性の確認、R1・N1 のデータ）、学習、閉ループ評価（実モデル）、RTC の実モデルでの確認、色の検出と完了判定、デモの描画、各 Step の完了条件の判定と判断点、支線の PR の取り込み。

### 15.4 進め方

（承認時に決定。掲示板 0005）

1. **支線は同時に 2 本まで。** 順序は、Step C の後に `cloud/metrics` と `cloud/runner`、Step D に `cloud/expert`、Step G の頃に `cloud/planner` と `cloud/figures`。前の組が終わって枠が空いてから次を始める
2. 本線は、各支線の開始の前に `docs/interfaces/` の文書（`trial_record.md`・`runner.md` → `expert.md` → `planner.md`・`results.md`）を main に入れる。**各支線の指示書は、その支線の最初の interfaces の文書が main に入った後に、監督が掲示板に書く**（本線は叩き台を出さない）
3. 本線は、支線の PR を取り込む前に、この PC で検査を回し直す。取り込んだら STATUS.md を更新する
