# Step A 調査報告 — 最終課題「生成復帰 VLA」

作成: 2026-09-25 / 対象: `手順書_最終課題_生成復帰VLA_v1.md` §2 / 作業場所 `<ROOT>` = `C:\PAI\recovery_vla`

## 調べ方

- C:\VLA 以下は**読むだけ**にした。使ったのはファイルの読み取り（Read）、文字列検索（ripgrep）、PowerShell の `Get-ChildItem`（一覧）と、`C:\VLA\02_環境\git\cmd\git.exe --no-optional-locks`（`log` / `diff` / `status` / `rev-parse` のみ）。C:\VLA の Python（python311、lerobot の .venv）は一度も起動していない。パッケージの版は site-packages の `*.dist-info` のフォルダ名から読んだ。
- 台帳（C:\VLA\99_台帳）の中身は開いていない。全ファイル一覧（パス・大きさ・更新日時）には、手順どおり 99_台帳 以下のファイル名も含まれる。台帳を扱う**コード**（pytools の `teleop/ledger.py`）は読んだ。
- 調査の開始時と終了時の全ファイル一覧: `docs\local\A_vla_listing_before.csv` / `docs\local\A_vla_listing_after.csv`（比較の結果は末尾の §6。C:\VLA のファイル名を含むので Git 管理外に置く。掲示板 0004）。
- 補足: Step A の開始前（この会話の最初の質問「VLA のファイルがある PC か」への回答時）に、別の調査エージェントが C:\VLA を読み取りで一覧している。開始時の一覧はその後に取ったもの。

---

## 0. 先に判断が要る食い違い（要約）

| # | 食い違い | 根拠 | 影響と選択肢 |
|---|---|---|---|
| **D1** | **収録系の IK は mink ではない。** 自前の減衰最小二乗（DLS）IK（`teleop/controller_ik.py`、`TeleopControllerIK`）で、二次計画でもない。**mink はどちらの環境にも入っていない**（dist-info なし）。mink を使っているのは旧システム `01_既存システム\code\scripts\12_closed_loop_rollout.py`（VLA 以前の独自モデル）だけ | §2-3 参照 | 計画書 §3・§4.4、手順書 §0.1-3・§5-4・落とし穴13 の「mink の CollisionAvoidanceLimit を IK に入れる」はそのままでは実現できない。選択肢は Step B で出す: (a) 新環境に mink を入れ、IK を mink に置き換える（行動＝x_des の定義は変わらないが追従特性が変わり、予備実験のチェックポイントで G0 を再現する前提が崩れうる）、(b) G0 は既存 DLS で通してから mink に切り替え、切替後に G0 相当を再確認、(c) 既存 IK に衝突回避の不等式を自作で足す（`controller_qp.py` に拡張口はあるが、箱型制約の自作ソルバしかない） |
| **D2** | LeRobot 0.6.1 の RTC の**既定の重みは線形**（`prefix_attention_schedule = LINEAR`）。計画書 §4.3 は「既定（…減衰は指数）」と書いている | `lerobot/policies/rtc/configuration_rtc.py:42`（`# Todo change to exp` の注記つき） | 手順書 §3 の設定の叩き台は `rtc_schedule: EXP` を明示しているので、実装上の問題はない。計画書の記述を直すか、既定（LINEAR）と EXP のどちらを使うか Step B で決める |
| **D3** | **`examples/rtc` は導入物に含まれない**（site-packages には examples がない。C:\VLA 内に LeRobot のソースの複製もない） | §2-14 参照 | 手順書 §2-14・§8-1 は「examples/rtc に合わせる」としている。代わりに同梱の `lerobot/rollout/inference/rtc.py`（`RTCInferenceEngine`）と `lerobot/rollout/context.py` が同じ呼び方をしているので、これを参照実装とすることを提案する |
| **D4** | **予備実験の評価は、評価用の種 100000–100099 を使っている**（段階 2: 100000–100019、段階 3: 100000–100099）。新しい決まりでは 100000〜 は「最終評価用」で、Step J より前に使えない | §2-22 参照 | G0（Step C 完了条件3）で「流用元と同じ種の結果を試行ごとに並べる」には 100000–100019 を使う必要がある。旧課題（立方体1個）なので新課題の配置とは無関係で、情報の漏れはないと考えるが、決まりの例外になるので承認が要る。代案: G0 だけ選択用の種（190000〜）で回し、試行ごとの比較は諦める |
| **D5** | 台本（`scripted_demo.py`）は**開ループ**で、内部の段階（逐次のプログラム）を持ち、失敗を検出して止める仕組みはない | §2-4 参照 | 計画書 §4.1(6) の方式とは別物。手順書どおり書き直しになる（想定内。確認のみ） |
| **D6** | 学習環境の Python 本体（uv の cpython 3.12.14）は `C:\VLA\02_環境\lerobot\python\` にあり、.venv はそこを指している（`pyvenv.cfg` の `home`）。uv.exe も `02_環境\lerobot\tools\` にある | §2-13 | .venv を複写しても使えない。新しい環境は Python 本体ごと `<ROOT>` 側に用意する必要がある。uv.exe を C:\VLA から直接実行するのは「C:\VLA 以下で何も実行しない」に触れるので、複写してから使うか別途入手するかを Step B で決める |
| **D7** | 新しい環境に必要で、どちらの環境にも**入っていない**もの: mink、peft（LoRA）、scipy（McNemar・Wilcoxon）、matplotlib、Anthropic SDK、Whisper 系 | §2-13・§2-20 | 入れる版と入手経路（ネットワーク）を Step B で示す |

---

## 1. 収録の仕組み（C:\VLA\pytools\panda_teleop\panda_teleop）

Git: `pytools\panda_teleop` がリポジトリ。HEAD = `dbb2c3b4db64c7e6db4a997e087549e67cd776c6`、追跡中のファイルに未コミットの変更なし。予備実験の学習・評価は `1db17a03393c949b7b7fa71d3782ddf2caac9a27`（clean）で行われた。その後の 19 コミットのうち、下の中核ファイルに入った変更は `closed_loop_eval.py`（出力先の既存チェックのみ）、`teleop/collect.py`・`teleop/recorder.py`（meta.json の記録項目の追加のみ）、`convert_to_lerobot.py`（マニフェスト対応の追加。`episode_arrays` は差分なし）で、**制御・観測・行動・評価の動作は変わっていない**。

### 1. モジュールの一覧と役割、依存関係

実行環境は 2 つに分かれている。

- **python311**（組み込み版 Python 3.11、`pytools\python311`）: 収録、台本、再生確認。mujoco 3.2.3、numpy 1.26.4、opencv-python 4.10
- **LeRobot の .venv**（Python 3.12、`02_環境\lerobot\.venv`）: 変換、学習、閉ループ評価。mujoco 3.2.3 も入っている（2026-09-17 に評価用に追加）

| モジュール | 役割 | 主な依存 |
|---|---|---|
| `teleop/app.py` | デモ用の入口。流用されるのは `make_controller()`（IK の調整値の唯一の定義）、`apply_home_pose()`、`HOME_QPOS`、`SCENE_PATH` | controller_ik、cameras、game |
| `teleop/collect.py` | 収録モード。`VelocityCommandIntegrator`（速度→x_des）、`make_collect_controller()`、`settle_start_state()`、`reset_episode()`、`CollectSession`（収録の状態機械・台帳）、`OperatorView`（操作者用画面）、`close_renderer()`、定数（作業空間、`START_POS`、`RECORD_EVERY=25`、`IMAGE_SIZE=256`、`CAMERAS`、`FINGERTIP_OFFSET=0.1034`） | app、ledger、recorder、device、cv2 |
| `teleop/recorder.py` | 配置（`Placement`、`placement_from_seed`、`training_placements`、`apply_placement`）、成功の体積（`cube_in_box`）、同時刻の画像と状態の取得（`FrameSampler`）、書き出し（`EpisodeWriter`）、`SessionLog`、定数（`INSTRUCTION`、`CUBE_BODY="cube_green"`、種の帯） | cv2、mujoco |
| `teleop/controller_ik.py` | DLS の IK と、目標の 2 次追従器（平滑化）、グリッパの切り替え | mujoco |
| `teleop/controller_qp.py` | 関節の速度・位置の箱型制約つき IK（自作の有効制約法）。`TELEOP_CONTROLLER=qp` のときだけ使われる。収録・評価では使われていない | controller_ik |
| `teleop/device.py` | `DeviceInput` / `DeviceState` の定義 | — |
| `teleop/dualsense_device.py` | DualSense。台本が参照するのは `PROFILES["collect"]`（xy 0.20 m/s、z 0.10 m/s）だけ | （pydualsense は遅延 import） |
| `teleop/replay.py` | 再生確認の中身 | collect、recorder |
| `teleop/ledger.py` | エピソード番号の台帳 | — |
| `scripted_demo.py` | 台本（DualSense の位置に入る `ScriptPad`） | app、collect、ledger、recorder、device、dualsense_device、code_version |
| `convert_to_lerobot.py` | raw → LeRobot v3.0 | vla_image_spec、vla_state、recorder（`DEFAULT_RAW_DIR` のみ）、lerobot |
| `replay_check.py` | 再生確認の入口 | collect、replay |
| `closed_loop_eval.py` | 閉ループ評価 | scripted_demo、vla_image_spec、vla_observation、vla_state、app、collect、recorder、code_version、lerobot |
| `train_launcher.py` / `start_training.bat` / `train_config_example.json` | 学習の起動器 | code_version、vla_image_spec、vla_observation |
| `vla_image_spec.py` | 画像の変換の唯一の定義（numpy のみ） | — |
| `vla_state.py` | 状態 15 次元の唯一の定義（numpy のみ） | — |
| `vla_observation.py` | 評価時の観測の組み立て。学習出力先の `conversion.json` と照合 | vla_image_spec、vla_state |
| `code_version.py` | Git のコミットとファイルの SHA-256 | （02_環境\git の git.exe） |
| `episode_ledger.py` | 台帳の CLI | ledger |
| `gui/gui_core.py`・`gui/screen.py` | 操作画面（tkinter）。`wilson_interval()` はここ | scripted_demo ほか |
| その他（`filters.py`、`frames.py`、`cameras.py`、`game.py`、`shelf_task.py`、`pad_feedback.py`、`touch_device.py`、`probe_*.py`、`main.py`） | デモ・機器用。本課題では使わない | — |
| `tests/` | pytest 一式（27 ファイル） | — |

### 2. 場面の定義

- 定義場所: `assets/panda/teleop_scene.xml`（机・立方体・箱・俯瞰カメラ・操作者用カメラ・mocap）と `assets/panda/panda.xml`（ロボット、手首カメラ、操作者用のグリッパカメラ、アクチュエータ）
- 物理: `timestep=0.002`（毎秒 500 回）、積分器 `implicitfast`
- 机: 上面 z=0、半幅 0.8×0.6 m、摩擦 1
- 立方体: `cube_green` の **1 個だけ**（2026-09-16 に赤・青を削除）。一辺 4 cm（半幅 0.02）、質量 0.05 kg、摩擦 `1 0.005 0.0001`、freejoint。keyframe `home` の qpos は nq=16（腕 7＋指 2＋立方体 7）で、立方体が増えると書き直しになる
- 箱: `goal_box`、原点 (0.45, 0.25, 0)、静的物体。壁の内面は ±0.06（物理的な内寸 12 cm）、壁の上端 z=0.06。成功判定の体積は内側 ±0.05・0<z<0.06（`recorder.cube_in_box`）
- 初期範囲（`recorder.py`）: x 0.35〜0.55、y −0.15〜0.05、向き ±45°、学習配置は種 0〜2999 から中心間 5 cm 以上で 10 個
- **カメラの向きはすべて `xyaxes` で指定**（落とし穴4に該当。quat への代入は効かない）
  - 俯瞰 `overhead`: pos (1.407555, 0.05, 0.853485)、xyaxes `0 1 0 -0.642788 0 0.766044`、fovy 45（(0.45, 0.05, 0.05) を前方から、距離 1.25 m、仰角 −40°）
  - 手首 `wrist`（W6m）: hand 体に固定、pos (−0.09, 0, 0.02)、xyaxes `0 -1 0 -0.8575 0 0.5145`、fovy 60
  - 記録しないカメラ: `operator_top`、`operator_back`（正射影）、`operator_gripper`
- **テクスチャ**: 画像ファイルのテクスチャは使っていない（背景は `builtin="flat"` の skybox、色は rgba）。メッシュは `meshdir="assets"` の相対パスで、実体は `...\panda_teleop\assets\panda\assets\*.obj/*.stl`（英数字のみのパス）
- mocap の目標球は geom group 5（画像に写らない）。記録時は mocap を複写しない（影に目標位置が漏れるのを防ぐため。`FrameSampler.capture` の注記）

### 3. 制御ループ

毎秒 500 回、物理 1 ステップごとに次を行う（`collect.py` の `main()` と `scripted_demo.Rig.loop()` で同じ順）。

1. **入力**: `VelocityCommandIntegrator.read()` が `x_cmd += vel × 0.002`（作業空間でクランプ）。作業空間 x 0.28〜0.62、y −0.30〜0.40、z 0.115〜0.35（hand 原点の高さ。指先は 0.1034 m 下）
2. **x_des**: クラッチ（倍率 1.0、常時接続）で `desired_pos = x_cmd` そのもの。**これが記録される行動の元（x_des）**
3. **平滑化**: `_track_target()`。臨界減衰の 2 次系 `a = ω²(x_des − target) − 2ω v`（ω=20 rad/s、速さ上限 0.8 m/s、半陰的オイラー、作業空間の壁で速度を 0）
4. **IK**: `TeleopControllerIK._solve_ik_step()`、**DLS（閉形式）で二次計画ではない**。位置誤差＋姿勢誤差（`rot_weight=0.5`）、誤差の大きさ上限 0.10、`mj_jacBody`（hand 体）のうち**腕 7 自由度の列だけ**を使う、減衰 0.05、`ik_gain=0.08`（1 ステップあたり）、零空間の姿勢項 `null_gain=0.05`（`q_neutral`＝構築時の関節角＝HOME）、`dq` を ±0.03 rad/ステップでクリップ、関節範囲でクリップ、実測関節角の ±0.10 rad に制限（巻き上がり防止）
5. **関節指令**: `data.ctrl[腕] = q_des`（位置制御アクチュエータ、剛性 4500/3500/2000）

- **手先姿勢の固定**: `lock_orientation=True` で、姿勢の目標は構築時（HOME）の hand の姿勢から変わらない。IK は `rot_weight=0.5` でそれを保つ
- グリッパ: 立ち上がりでの切り替え（□を 1 回押すと全閉、もう 1 回で全開）。指令値 0＝閉、255＝開
- 定常誤差: IK の零空間項のため x_des と hand の間に 5〜17 mm（主に +z）の静的なずれが残る（`settle_start_state` の注記、2026-09-16 実測）
- **IK の模型に立方体の自由度が含まれているか**: IK は MuJoCo の全体模型の上で計算するが、ヤコビアンは腕 7 自由度の列だけを使うので、**立方体の自由度は IK の変数に入っていない**（mink を使う場合は `Configuration` が模型の全自由度を持つので、別途固定が要る。落とし穴13）
- `TELEOP_CONTROLLER` 環境変数が `qp` だと QP 版に切り替わる（既定は `dls`）。新しい場所では環境変数に依存しないようにすべき
- **mink の版と CollisionAvoidanceLimit の有無**: **mink は未導入**（python311 にも .venv にもない）。したがって CollisionAvoidanceLimit も使えない（→ D1）

### 4. 台本（800000 番台を作ったもの）

`scripted_demo.py`（2026-09-17、`02_環境\lerobot\wrist_mount\stepd` の Step D 台本が元）。

- **入口**: DualSense と同じ。`ScriptPad` が `VelocityCommandIntegrator` の内側の機器になり、速度 `vel` と □（把持）を出す。○（収録開始）・△長押し（保存）も同じ `CollectSession` を通るので、x_des・グリッパ・保存物は人の収録と同一の経路。1 回の入力読み取りごとに物理 10 ステップ（20 ms）
- **開ループ**: 収録開始 0.5 秒後に立方体の位置を**一度だけ**読み、そこから経由点を決めて順に動く（経由点の上→真上→降下→□→5 cm 持ち上げ→搬送高さ→箱の上→解放高さ 0.18→□→待機→上昇→△で保存）。移動は `velocity_to()`（距離×ゲイン、collect の速度上限×0.8〜1.0 でクリップ）で、1 mm 以内に入ったら次へ。**立方体の向き（ヨー）は無視**して手首を固定のまま掴む
- **内部の段階**: 関数内の逐次呼び出しという形で持つ（状態から段階を判定する作りではない）
- 真値を読むのは経由点の決定と判定材料だけ。搬送中は `carry_monitor` で指先中心と立方体の距離を毎ループ記録する（3.5 cm 超で落下とみなす）が、**記録するだけで止めない**
- 失敗の検出・停止（経路確認の段階 4 用）: **実装されていない**。段階 4 はタスク文書（Downloads の `タスク_台本データによる経路確認と復帰入りの予備確認_v2.md` §4）に設計の叩き台があるだけで、コードには入っていない
- 台本の種のばらつき（`sample_params`、種は 3000〜99999）: 経由点のずれ ±1 cm、速さ倍率 0.8〜1.0、ゲイン 3〜5 /s、待ち時間 ±0.1 s、搬送高さ 0.22〜0.27 m、解放点のずれ ±1 cm
- 全エピソードを保存し、台本の判定（grasp / carry / place のどこで失敗したか）を生成記録（jsonl）に書く。学習用は判定 success かつ recorder の success のものだけ（`training_list`）
- 実績（段階 2、`generation\stage2.jsonl`）: 58 本中 50 本成功、失敗 8 本はすべて搬送中（向き −41.3° の配置で 11 本中 6 本、+42.9° で 7 本中 2 本）。平均 326 フレーム（16.3 秒）
- 台帳に結合: `Rig` は `CollectSession` を台帳つきで作る（番号帯 script 800000〜）。出力先が 03_収録 の中なら拒否

### 5. 記録

- **20 Hz の窓**: `RECORD_EVERY=25` 物理ステップ。フレーム i は物理ステップ i×25 の時点（フレーム 0 はリセット直後、ステップ前）
- **同時刻の取得**: `FrameSampler.capture()` が qpos・qvel・ctrl・time を作業用の MjData に写して `mj_forward` し、そこから状態と 2 画像を取る（生の MjData を `mj_forward` すると次の IK が変わるため）。mocap は写さない
- フレームごとの記録（`FRAME_FIELDS`）: step、sim_time、ee_pos、ee_quat、ee_axisangle、fingers、joints、joint_vel、x_des、tracker_target、gripper_cmd、gripper_closed、cube_pos、cube_quat。さらに物理ステップごとの `step_x_des`・`step_gripper_cmd`、開始状態 `start_qpos`・`start_ctrl`・`start_q_des`、`reset_qpos`
- **視点ごとの画像変換**: 記録時には**何もしない**（生の描画を PNG で保存）。変換は変換時と評価時に `vla_image_spec.py` だけが行う（俯瞰＝左右反転、手首＝上下反転）
- メタデータ（meta.json）: schema_version、episode_id、episode_id_scheme、storage_kind、host、ledger_seq、session、operator、recorded_at、saved_at、**instruction（固定文 "put the cube in the box"）**、placement_id、placement_seed、cube_init、retakes、success、final_cube_pos、control_hz、timestep、record_hz、record_every、duration_s、truncated_at_step、timing（RTF、停止、固まったこま）、cameras、gripper、frame/action の定義文、state_layout、controller の全調整値、pad の設定、software（mujoco の版と主要ファイルの SHA-256）
- 保存形式: `<raw>/<日付>/ep_NNNNNN/{overhead,wrist}/NNNNNN.png`（256×256、PNG 圧縮 1、cv2.imencode＋Python で書き出し＝日本語パス対策）、`data.npz`、`meta.json`。保存中は `.partial`。△を押した時点で打ち切る
- **途中の状態は保存していない**（開始状態だけ）。分割（Step F）に要る qvel・act・warmstart、追従器の速度 `target_vel`、積分器の `x_cmd`、グリッパの切り替え状態（`prev_grip`・`gripper_closed`）、20 Hz/10 Hz の位相は、任意の時点では残らない

### 6. 変換（convert_to_lerobot.py）

- 指示文: `task = meta["instruction"]` を**エピソードごと**に入れている。変換器はエピソードごとの指示文に対応済みだが、記録側（`recorder.INSTRUCTION`）が固定文なので、現状は全件同じ
- **10 fps への間引き**: 10 fps のフレーム k = raw フレーム 2k。行動 k の xyz = `x_des[2k+2] − x_des[2k]`（その後 0.1 秒の x_des の変化の合計）、回転 3 = 0、グリッパ = raw フレーム 2k+2 の `gripper_closed` が真なら +1、偽なら −1（指令値ではなく制御器の状態）。後ろに 0.1 秒が続かない最後の raw フレームは捨てる（k の数 = (n−1)//2）
- 状態 15 次元は `vla_state.policy_state`（ee 位置 3、真下からの姿勢のずれ 3、指 2（2 本目は符号反転）、関節 7）
- LeRobot v3.0、画像は動画ではなく埋め込み（`use_videos=False`）、`robot_type="panda"`、`repo_id=local/<名前>`。出力先が既にあれば拒否
- **conversion.json**（`<dataset>/meta/`）: converter_version 2、変換時刻、fps・raw_hz・stride、画像仕様（`image_spec_version` 2、視点ごとの変換、画像のキー）、state/action の名前と定義文、マニフェストの記録、`sources`（エピソード番号、raw のパス、フレーム数、placement_id、operator、session、success、retakes、meta.json と data.npz の SHA-256）
- **メタデータの旗の持ち越し**: 上の `sources` の項目だけ。新しい旗（配置の種類、目標の色、組の番号、注入の種類など）は持ち越されない
- `--verify`: 画像仕様の照合、fps、件数、統計量の有限性と float32 分散の相殺の検出、全フレームの状態・行動と raw の一致、画像の一致（抜き取り）、グリッパが ±1、回転が 0、姿勢の連続性。変換の検証に**そのまま使える**
- **マニフェスト**: `--manifest` に対応しているが、**ラベル（`03_収録\labels\*.jsonl`）が全件にあること**と「失敗」のラベルがないことを要求し、既定の raw の場所は `03_収録\raw`。ラベル付けの仕組み（本課題のスコープ外）と結合しているので、新しい場所では手を入れる必要がある

### 7. 再生確認（replay_check.py、teleop/replay.py）

- **途中の状態からの再生はできない。** `replay()` は必ず `reset_episode()`（保存された開始状態＋配置）から始める。モードは `step`（物理ステップごとの x_des、ビット一致を要求）、`window`（20 Hz の行動、許容 5 mm）、`last_step`・`shifted`（検出できることの陰性対照）、`actions`（LeRobot から読み戻した 10 fps 行動、`--lerobot-actions`）
- 許容: 手先 5 mm（`WINDOW_TOL_M`）、画像 ±2（GPU の揺らぎ）
- 保存された状態から再生するには、§5 の不足分を保存し、`reset_episode` に相当する「任意時点の復元」を足す必要がある（Step F の 6）

### 8. 評価（closed_loop_eval.py）

- 方策の呼び方: `run` は **`select_action`**（方策内部の待ち行列を使う）。`open-loop` サブコマンドだけ `predict_action_chunk`
- **実行する手数**: チェックポイントの設定 `n_action_steps=50` がそのまま効き、**50 手を実行してから次を推論**する（コマンドから変えられない）
- **同期**: 推論は物理を進めないループの中で呼ばれる（`run_trial` が `act()` を呼んでから `execute_action()` で 50 物理ステップ進める）。**推論の間は物理が止まっている同期実行**（計画書 §4.3 の推測どおり）
- 行動の入口: DualSense と同じ（`ScriptPad` を積分器の内側に置き、0.1 秒間 `vel = action_xyz / 0.1` を保つ。グリッパは状態と違えば窓の最初のステップで □）
- 雑音: 試行の種で `torch.Generator` を作り、推論ごとに `(1, 50, 32)` の標準正規を引く（配置・誘発・雑音を分けた乱数列にはなっていない）
- **段階の判定: ない**（「材料だけ記録し、段階や失敗の分類はしない」と明記）。成功だけを判定: 30 秒以内に立方体が箱の体積内で速さ 1 cm/s 未満を 1 秒保つ（物理ステップごと）
- **入力照合**: 最初の推論で、前処理後のバッチの 2 つのカメラの枠それぞれが、対応する視点の画像と値で一致し、互いに異なることを確かめる（rename_map の落とし穴2 の対策）。状態の形と正規化後の最大絶対値も記録
- 種: 100000 未満を拒否。`--eval-seeds A B` は `placement_from_seed` で新しい配置、`--placement-ids` は学習配置に種を順に割り当てる
- 出力: `trial_NNNN.npz`（20 Hz のフレーム、実行した行動、窓ごとの指の接触と立方体の最大速さ）、`trial_NNNN.json`（種、配置、チェックポイント、コードの版、方策の設定、入力照合、成否と時刻）、`_raw.mp4`、`_policy_input.mp4`、`summary.json`。出力先が既にあれば拒否
- 段階ごとの材料（持ち上げの最大高さ、把持の回数など）は評価器の外の `02_環境\lerobot\scripted_check\analysis\route_eval_summary.py` が後から集計している

### 9. 学習の起動器（start_training.bat、train_launcher.py、train_config_example.json）

`start_training.bat` は `02_*\lerobot\.venv` の python で `train_launcher.py CONFIG --confirm` を起動するだけ。起動器が自動で付けるもの:

- `lerobot-train.exe`（`python -m lerobot.scripts.train` は 0.6.1 にない）
- 環境変数 `HF_HOME=C:\VLA\02_環境\hf_home`、`HF_HUB_OFFLINE=1`、`PYTHONIOENCODING=utf-8`、`PYTHONUNBUFFERED=1`
- `--policy.path=<smolvla_libero のスナップショットのフォルダ>`（リポジトリ名だと Windows で失敗するため）
- `--dataset.repo_id=local/<フォルダ名>`、`--dataset.root=<データセット>`
- `--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}`（2 視点とも）
- 範囲 `expert`: `--policy.train_expert_only=true --policy.freeze_vision_encoder=true`（`full` もある）
- `--batch_size`・`--steps`・`--save_freq`（既定 = steps）・`--seed`（既定 1000）・`--log_freq`（既定 50）・`--num_workers`（任意）
- `--output_dir=<04_学習\checkpoints\<設定名>_<日時>>`、`--job_name`
- **外部送信の停止**: `--policy.push_to_hub=false`、`--wandb.enable=false`。`--policy.device=cuda`
- バッチの上限（14 GiB で測った値）: expert 62、full 12。超えると拒否
- 起動器が握る引数（`--resume` を含む）は `extra_args` で渡せない → **続けて学習（resume）には対応していない**
- **正規化**: 起動器は何も指定しない（lerobot-train がデータセットの統計量で上書き。§2-17）
- 出力先に `conversion.json`（データセットの写し）、`train_launch_config.json`、`train_run.json`（コマンド、データセットの指紋、コードの版、ログの要約、GPU の最大使用量）、`loss.csv`・`loss.png` を残す。シンボリックリンクが作れるか（開発者モード）を事前に確かめる
- 設定例: `train_scope "expert"`、`batch_size 32`（7.9 GiB）、`steps 1000`

### 10. 集計（Wilson 区間など）

- `gui/gui_core.py:411` `wilson_interval(successes, trials, z=1.959963984540054) -> (lo, hi)`。入力は成功数と試行数。`EvalResults`（`trial_NNNN.json` と `summary.json` を読む）が率・区間・入力照合の合否・向きの帯（0–15、15–30、30–45°）ごとの成功率を出す。テスト（`tests/test_gui_core.py`）で 61/100、18/20 の報告値と照合済み
- 関数自体は数行で依存なし。gui_core は tkinter を含まないが `scripted_demo` を import するので、**関数だけ**を持ち込むのがよい
- 段階別の到達: pytools には**ない**（上の route_eval_summary.py が材料を出すのみ）
- McNemar、Wilcoxon、対応ありの差の区間: **ない**。scipy もない

### 11. 台帳と結合している箇所

- `teleop/collect.py`: `CollectSession.__init__` が `episode_ledger.open_session()` を必須とし（番号帯・保存先の登録・ロック）、保存時に `allocate()`・`record_saved()`、終了時に `close(backup)`（Y:\ へ複写）。`parse_args` の `--ledger-dir` 既定 `99_台帳\episode_ledger`。meta.json の `episode_id_scheme`・`storage_kind`・`ledger_seq`
- `scripted_demo.py`: `Rig` が `CollectSession` を台帳つきで作る（`--ledger-dir`）
- `episode_ledger.py`（CLI）、`teleop/ledger.py` の既定値 `DEFAULT_LEDGER_DIR`・`DEFAULT_BACKUP_DIR = <共有フォルダ>\99_台帳\episode_ledger_控え`
- `gui/gui_core.py`: `RUN_RECORD_PATH = 99_台帳\操作記録\gui_runs.jsonl`
- ハッシュの対象一覧に `teleop/ledger.py` を含む（`collect.HASHED_FILES`、`code_version.CODE_FILES`）
- 結合して**いない**: `closed_loop_eval.py`（`EvalRig` は `CollectSession` を作らない）、`convert_to_lerobot.py`、`teleop/replay.py`
- `start_training.bat` は終了時に「99_* の台帳に記録せよ」と表示するだけ

### 12. C:\VLA の絶対パス、日本語を含むパスの直書き

| 場所 | 内容 |
|---|---|
| `teleop/recorder.py:52` | `DEFAULT_RAW_DIR = parents[4] / "03_収録" / "raw"` |
| `teleop/collect.py:549` | `PRACTICE_DIR = DEFAULT_RAW_DIR.parent / "practice"` |
| `teleop/ledger.py:43-45, 539-542` | `VLA_ROOT/99_台帳/episode_ledger`、`<共有フォルダ>\99_台帳\episode_ledger_控え`、登録済みの保存先（03_収録、02_環境\lerobot 以下） |
| `train_launcher.py:70-73` | `VLA_ROOT = parents[2]`、`HF_HOME = VLA_ROOT/02_環境/hf_home`、`OUTPUT_ROOT = 04_学習/checkpoints`、`LOG_ROOT = 04_学習/logs` |
| `code_version.py:17` | `PORTABLE_GIT = parents[2]/02_環境/git/cmd/git.exe` |
| `gui/gui_core.py:33-77, 916-917` | `VLA_ROOT`、`HF_HOME`、`99_台帳`、03_収録・04_学習・05_評価・scripted_check の各所 |
| `scripted_demo.py:131-136` | 03_収録 の中への出力を拒否（`DEFAULT_RAW_DIR` 経由） |
| `start_training.bat:8` | `..\..\..\02_*\lerobot\.venv` を探す |
| `train_config_example.json:3` | `C:\VLA\03_収録\lerobot\CHANGE_ME` |
| docstring の使い方（多数） | `C:\VLA\pytools\python311\python.exe ...`、`C:\VLA\02_環境\lerobot\.venv\...`（動作には影響しない） |
| 学習済みの記録 | チェックポイントの `train_config.json`、`train_run.json`、`conversion.json` の `raw_path`、評価の `trial_*.json` に `C:\VLA\02_環境\...` の絶対パスが残る（`vla_observation` は出力先の `conversion.json` の**中身**を照合するだけで、パスは使わない） |
| .venv の `pyvenv.cfg` | `home = C:\VLA\02_環境\lerobot\python\cpython-3.12-windows-x86_64-none`（D6） |

MuJoCo のモデルとメッシュのパス自体は `C:\VLA\pytools\panda_teleop\panda_teleop\assets\panda\` で英数字のみ（落とし穴1 は現状では踏んでいない）。

---

## 2. 学習用の実行環境（C:\VLA\02_環境\lerobot\.venv、読むだけ）

### 13. 版

| パッケージ | .venv | python311 |
|---|---|---|
| Python | 3.12（uv 管理の cpython-3.12.14、本体は `02_環境\lerobot\python\`） | 3.11（組み込み版） |
| torch | 2.11.0+cu126 | — |
| transformers | 5.5.4 | — |
| lerobot | 0.6.1 | — |
| mujoco | 3.2.3 | 3.2.3 |
| mink | **なし** | **なし** |
| numpy | 2.2.6 | 1.26.4 |
| opencv | opencv-python-headless 4.13.0.92 | opencv-python 4.10.0.84 |

全一覧は付録 A。**2 つの環境は mujoco の版が同じ**で、評価器は .venv で物理と描画を回している。予備実験の `check-observation`（`scripted_check\eval\stage0_observation\observation_check.json`）では、.venv で描いた開始時の画像が python311 で記録した PNG と**画素単位で一致**した（最大差 0）。ただし `settle_start_state` の結果は 2 環境でビット一致しない（`evaluator_start_state_equals_raw: false`）。10 fps 行動の再生は手先の最大誤差 1.9 mm（許容 5 mm）で通っている。→ 生成・学習・評価を **1 つの環境で回すのは可能な見込み**。ただし環境をまたぐとビット一致は期待できないので、組の最初のこまの一致（Step D 完了条件2）などは同じ環境の中で確かめる必要がある。

### 14. RTC

- **`lerobot.policies.rtc` はある**（`RTCConfig`、`RTCProcessor`、`ActionQueue`、`LatencyTracker`、`ActionInterpolator`、`reanchor_relative_rtc_prefix`）
- `SmolVLAPolicy.predict_action_chunk(batch, noise=None, **kwargs)` は `inference_delay`・`prev_chunk_left_over`・`execution_horizon` を受ける（`ActionSelectKwargs`、`modeling_smolvla.py:82-85, 232-241`）。`supports_rtc()` は True
- `select_action` は RTC が有効だと `assert` で止まる（`modeling_smolvla.py:254-256`）。**RTC は `predict_action_chunk` で使う**
- **`RTCConfig` の項目と既定値**（`configuration_rtc.py:37-48`）: `enabled=True`、`prefix_attention_schedule=LINEAR`（ZEROS / ONES / LINEAR / EXP）、`max_guidance_weight=10.0`、`execution_horizon=10`、`debug=False`、`debug_maxlen=100`
- **RTC を有効にする場所**: 方策の設定の `rtc_config`。smolvla_libero とすべての学習済みチェックポイントは `rtc_config: null`。有効にするには読み込み後に `policy.config.rtc_config = RTCConfig(...)` を入れて `policy.init_rtc_processor()` を呼ぶ（同梱の `lerobot/rollout/context.py:259-268` がこの手順）。**`rtc_config` が None のままだと `euler_integrate` は RTC を通らず、`inference_delay`・`prev_chunk_left_over` は黙って無視される**（`flow_matching.py:105-115`、`modeling_smolvla.py:539-540, 759-773`）。落とし穴7 のとおり
- **`execution_horizon` が決めるもの**: 前の塊に合わせる範囲の**終わり**。重みは `get_prefix_weights(start=inference_delay, end=execution_horizon, total=50)` で、先頭 `min(d, H)` 手が 1、そこから H 手目まで減衰（LINEAR は直線、EXP は `w·expm1(w)/(e−1)`）、H 手目以降は 0（`modeling_rtc.py:251-298`）。前の塊の残りが H より短ければ H を残りの長さに縮める（同 187-190）。**原論文の実行間隔 s ではない**。同梱の推論器では、推論を始める頻度は別の引数（`rtc_queue_threshold`、既定 30。待ち行列の残りがこれ以下になったら推論）で決まる
- 誘導の強さ: 流れの時刻に応じた `c·inv_r2` を `max_guidance_weight` で頭打ち（同 221-227）。誘導の勾配は `torch.enable_grad()` で方策の速度場を通して求める（同 212-219）→ **推論 1 回の時間とメモリは RTC なしより増える**（数値は未測定）
- **前の塊がないとき**: `prev_chunk_left_over is None` なら誘導なしで通常の生成（同 167-170）
- **前の塊の残りを渡す空間**: `examples/rtc` は導入物にない（D3）。同梱の `lerobot/rollout/inference/rtc.py:288-342` では、`ActionQueue.get_left_over()`＝**後処理の前（正規化された、方策の出力のまま）の行動**の、推論開始時点で未実行の部分を渡す（相対行動の方策だけは絶対値から付け直す。SmolVLA は該当しない）。長さは `execution_horizon` に切り詰め・ゼロ詰め（`_normalize_prev_actions_length`）。新しい塊は、推論中に消費した手数（`real_delay`）だけ先頭を捨てて待ち行列を**置き換える**（`action_queue.py:175-194`）。`denoise_step` 内で前の塊は `(B, 50, 32)` にゼロ詰めされ、7 次元より後ろ（パディングの次元）にも重みが掛かる

### 15. 損失とパディング

`SmolVLAPolicy.forward`（`modeling_smolvla.py`）:

```python
299        actions_is_pad = batch.get("action_is_pad")
...
306        if actions_is_pad is not None:
307            in_episode_bound = ~actions_is_pad
308            losses = losses * in_episode_bound.unsqueeze(-1)
...
326            if actions_is_pad is None:
327                loss = losses.mean()
328            else:
329                num_valid = ((~actions_is_pad).sum() * losses.shape[-1]).clamp_min(1)
330                loss = losses.sum() / num_valid
```

データセット側（`lerobot/datasets/dataset_reader.py:222-231`）: 取得位置は `[ep_start, ep_end−1]` にクランプし、`abs_idx + delta >= ep_end`（または `< ep_start`）の位置に `action_is_pad = True` を立てる。SmolVLA の `action_delta_indices = range(50)`（`configuration_smolvla.py:153-155`）なので、**塊がエピソードの末尾を越える分は損失から外れる**。予備実験の学習ログにも `losses_after_in_ep_bound` が出ている。→ 計画書 §4.1(2) の前提は成り立つ

### 16. SmolVLA の設定（smolvla_libero と予備実験のチェックポイント）

- `n_obs_steps=1`、`chunk_size=50`、`n_action_steps=50`、`max_action_dim=32`、`max_state_dim=32`、`num_steps=10`（オイラー）、`tokenizer_max_length=48`、`pad_language_to="max_length"`
- 画像: 入力 256×256 を `resize_imgs_with_padding=(512, 512)` で詰め物つき拡大、[−1, 1] に
- `vlm_model_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct"`、`load_vlm_weights=True`（smolvla_libero の設定）
- **SmolVLM2 を名前で読みに行く**: `SmolVLMWithExpertModel.__init__` が `AutoModelForImageTextToText.from_pretrained(model_id)`（重み）と `AutoProcessor.from_pretrained(model_id)` を呼ぶ（`smolvlm_with_expert.py:90-101`）。前処理の `tokenizer_processor` も `tokenizer_name: "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"` を名前で読む（チェックポイントの `policy_preprocessor.json`）。→ 落とし穴15 のとおり、**SmolVLM2 のスナップショットも HF キャッシュ構成で必要**
- 注意: チェックポイントの `input_features` は smolvla_libero から引き継いだ `observation.state: [6]`・`camera1〜3` のままで、実データの状態は 15 次元（32 までゼロ詰めされるので動作に問題はない）。camera3 はバッチになく、`empty_cameras=0` なので単に使われない
- `use_amp=False`、`compile_model=False`

### 17. 正規化の統計量

- **保存場所**: チェックポイントの `pretrained_model/policy_preprocessor_step_5_normalizer_processor.safetensors`（正規化）と `policy_postprocessor_step_0_unnormalizer_processor.safetensors`（逆正規化）。設定は `policy_preprocessor.json`・`policy_postprocessor.json`（`norm_map`: STATE・ACTION は MEAN_STD、VISUAL は IDENTITY、`eps=1e-8`）
- **`--policy.path` で別のデータセットの学習を始めると上書きされる**: `lerobot_train.py:350-351` で `dataset_stats=dataset.meta.stats`、`375-377` で正規化・逆正規化の両方の `stats` を新しいデータセットの統計量に差し替える
- **`--resume` の場合**: どちらも行わず、チェックポイントに保存された統計量を使う（`lerobot_train.py:371-377` の注記）。最適化器と学習率の予定も `training_state/` から復元される（`422-428`）
- → 2 周目（R2・R1+）を `--policy.path=<R1>` で始めると統計量は新しいデータセットのもの（R1+引き継ぎの合併）になる。落とし穴10 のとおり固定が要る。方法は Step B で出す（候補: 新しいデータセットの `meta/stats.json` を R1 の統計量で置き換える、`--resume` を使う、学習後に差し替える、など）

### 18. 学習率の予定

- 種類: `cosine_decay_with_warmup`（`lerobot/optim/schedulers.py:130-182`）。最適化器は AdamW（betas 0.9/0.95、eps 1e-8、weight_decay 1e-10、勾配クリップ 10）
- SmolVLA の既定: 立ち上がり 1000 手、減衰 30000 手、ピーク 1e-4、最終 2.5e-6。**smolvla_libero の設定では減衰 25000 手**（予備実験のログ）
- **学習の手数が減衰の手数より短いと自動で縮める**: 立ち上がり ×(steps/decay)、減衰 = steps（同 149-161）。予備実験（19000 手）は立ち上がり 760、減衰 19000 だった
- したがって、本課題の 3 万手（R1・N1）では縮めず、立ち上がり 1000、25000 手で 2.5e-6 まで下がり、**最後の 5000 手は最終の学習率のまま**になる。8000 手（K1）は立ち上がり 320・減衰 8000、1 万手（R2・R1+）は立ち上がり 400・減衰 10000
- `--policy.path` から始めると最適化器と予定は最初から（立ち上がりから）やり直しになる

### 19. 外部への送信の既定値

- `policy.push_to_hub`: 既定 **True**（`lerobot/configs/policies.py:70`）。smolvla_libero の `config.json` も `push_to_hub: true`、`repo_id: "pepijn223/smolvla_libero"`。明示的に `--policy.push_to_hub=false` が要る（起動器は付けている）
- 実験管理サービス: `wandb.enable` 既定 **False**（`lerobot/configs/default.py:82`）。起動器は念のため `--wandb.enable=false`
- `job.target`: 既定 None（手元で実行）。値を入れると HF Jobs に送られる（`default.py:163-168`）。設定しないこと
- `DatasetRecordConfig.push_to_hub=True`（`configs/dataset.py:43`）は収録用の設定クラスで、学習の経路では使われない

### 20. PEFT（LoRA）

- LeRobot 側には対応がある: `--peft.*`（`PeftConfig`: `target_modules`（文字列・一覧・正規表現）、`full_training_modules`（PEFT の `modules_to_save`）、`method_type="LORA"`、`init_type`、`r=16`、`lora_alpha`）。`lerobot_train.py:331-341` で `policy.wrap_with_peft()`
- **`peft` パッケージは .venv に入っていない**（`require_package("peft")` で止まる）
- SmolVLA の既定の対象（`modeling_smolvla.py:420-429`）: 行動エキスパートの `q_proj`・`v_proj` と、`state_proj`・`action_in_proj`・`action_out_proj`・`action_time_mlp_in/out`。**VLM の層は既定の対象に入らない**ので、`--peft.target_modules` に VLM の層（例: `model.vlm_with_expert.vlm.model.text_model.layers.<i>.self_attn.(q|v)_proj`）を明示する必要がある
- 注意: PEFT で包むと、対象外の重み（行動エキスパート本体を含む）は凍結される。行動エキスパートも学習し続けるには `full_training_modules` で指定する必要がある（Step B の K1 の分岐の費用見積もりで扱う）

### 21. 出発点モデルと SmolVLM2 の置き場所

`C:\VLA\02_環境\hf_home\`（HF キャッシュの構成）:

- `hub\models--lerobot--smolvla_libero\snapshots\31d453f7edd78c839a8bbc39744a292686daf0de\`: config.json、model.safetensors（906,712,520 B）、policy_pre/postprocessor の json と safetensors、train_config.json、README.md
- `hub\models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct\snapshots\7b375e1b73b11138ff12fe22c8f2822d8fe03467\`: config.json、model.safetensors（2,029,990,624 B）、tokenizer 一式、processor・preprocessor の設定、chat_template.json ほか
- `hub\models--lerobot--smolvla_base\...`（本課題では使わない）、`hub\datasets--HuggingFaceVLA--libero\...`（437,591,868 B、使わない）
- `refs\`・`trees\` あり。**`blobs\` は空で、スナップショットの中が実体ファイル**（シンボリックリンクではない）→ そのまま複写できる
- `datasets\parquet\...`（HF datasets のキャッシュ、22 件）と `xet\` もある。HF_HOME を C:\VLA に向けて学習するとここに書き込まれる

---

## 3. 予備実験の資産

### 22. 台本 50 本で学習したチェックポイントと評価結果

場所: `C:\VLA\02_環境\lerobot\scripted_check\`

- データ: raw `raw\stage2\2026-09-17\ep_8000xx`（58 本、学習に使ったのは 50 本: 800000〜800057 のうち成功分）、生成記録 `generation\stage2.jsonl`、変換済み `datasets\stage2`（50 エピソード、8008 フレーム）、設定 `configs\stage2.json`
- 学習: `train\stage2_20260917-221354\`（expert、バッチ 32、19000 手、種 1000、num_workers 6）。保存点 005000・010000・015000・019000（`last` → 019000）。**評価に使ったのは `checkpoints\019000\pretrained_model`**（model.safetensors 906,712,520 B）。学習記録 `train_run.json`、ログ `train_logs\stage2_20260917-221354.log`、コミット `1db17a0`（clean）
- **学習配置 20 回で 18 回成功**: `eval\stage2_closed_loop\`（種 100000〜100019、学習配置 0〜9 を 2 周。失敗は種 100000（配置 0、向き −41.3°）と 100013（配置 3、+42.9°））。`summary.json` と `trial_0000〜0019.*`
- **学習外 100 回で 61 回成功**: `eval\stage3_100trials\`（種 100000〜100099、`placement_from_seed` の新しい配置）。同じ条件の最初の 20 回は `eval\stage3_closed_loop\`（13/20）
- 集計と図: `analysis\stage2_*`、`analysis\stage3_*`、`analysis\stage3_100\*`（`route_eval_summary.py`）
- 実行は `closed_loop_eval.py run`（sync、50 手、`select_action`）

### 23. 実測値

| 項目 | 値 | 根拠 |
|---|---|---|
| 台本 1 本の生成時間 | **約 2.8 秒/本**（平均 16.3 秒のエピソード。並列なし） | 段階 2 の 58 本の保存時刻（22:08:37〜22:11:17）の間隔。起動時間は含まない |
| 学習の速さ | **約 0.41 秒/手**（`updt_s` 0.40、`data_s` 0.005、num_workers 6、バッチ 32）。映像用メモリ最大 7.69 GiB | 学習ログ。保存点の間隔（5000 手で約 35 分）とも一致 |
| 学習の所要時間 | 19000 手で実質約 2 時間 10 分。ログ上は 22:13〜06:17 だが、途中で PC がスリープしている（`analysis\powercfg_note.txt`: 04:13 にスリープを無効化、06:25 に戻した） | 同上 |
| 評価 100 試行の時間 | **約 6 分 35 秒（約 4.0 秒/試行、動画書き出し込み）** | `stage3_100trials` の試行ファイルの時刻（09:54:08〜10:00:43） |
| 推論 1 回の時間 | **不明**。評価器は推論時間を記録していない | 確かめ方: Step C で複写したチェックポイントを新しい評価器で動かすときに、`predict_action_chunk` の前後の時刻（`torch.cuda.synchronize()` 込み）を記録する。RTC あり（勾配計算が入る）となしの両方を測る |

その他: num_workers 6 で学習するにはページファイル 16 GB の固定が要った（`99_調査\gpu_util\report_stageC.md`）。

---

## 4. 本書（手順書・計画書）と実態の食い違いの一覧

§0 の D1〜D7 に加えて、記録として残す。

1. 計画書 §9 の予備実験の数字「1 万 9 千手で 2 時間 10 分」はログ上の経過時間（8 時間 4 分）とは違うが、スリープを除いた実質の時間としては一致する（§3-23）
2. 箱の「内寸 12 cm」（計画書 §2）は物理的な内寸と一致。ただし成功判定の体積は内側 10 cm（±0.05）で、壁から 1 cm 内側まで。計画書 §4.4 の「指と壁・先客の間に 1.5 cm 以上の隙間」を決めるときはこの差に注意
3. 評価器の雑音は試行の種 1 本から推論ごとに引いているので、推論の回数が違う条件間では雑音の列がずれる。手順書 §8-2（配置・誘発・雑音を別々の乱数列に）は新規の実装になる
4. 変換器は指示文をエピソードごとに扱えるので、手順書 §5-6 の「task に入れる」は変換器側の変更は小さい（記録側と meta の持ち越しが主）

---

## 5. 流用候補の表

扱い: **そのまま複写** / **手を入れて複写** / **書き直し** / **持ち込まない**。コードは `pytools\panda_teleop` の HEAD `dbb2c3b` から複写し、そのコミット番号を記録する想定（方法は Step B）。

| ファイル | 役割 | 扱い | 理由 |
|---|---|---|---|
| `assets/panda/panda.xml`、`assets/panda/assets/*`（メッシュ）、`LICENSE`・`README.md` | ロボット・手首カメラ W6m | そのまま複写 | カメラと行動の定義を卒研と揃えるため変えない。同梱の LICENSE を保持（出所の確認は Step B で README 用に行う） |
| `assets/panda/teleop_scene.xml` | 机・立方体・箱・俯瞰カメラ | 手を入れて複写 | 立方体を 3 色に、keyframe の qpos を更新。俯瞰カメラ・箱・机は変えない。G0 用に 1 個版もそのまま残す |
| `teleop/controller_ik.py` | DLS IK、追従器、グリッパ | そのまま複写（G0 用） | 行動の意味と予備実験の再現に必須。衝突回避をどう入れるかは D1 の判断による |
| `teleop/controller_qp.py` | 箱型制約の QP 版 | 持ち込まない | 使われていない。衝突回避に使うなら (c) 案として再検討 |
| `teleop/app.py` | `make_controller`、`apply_home_pose`、`HOME_QPOS`、`SCENE_PATH` | 手を入れて複写 | 必要な関数・定数だけを抜き出す（デモ・機器選択・環境変数 `TELEOP_CONTROLLER` への依存を外す）。調整値は設定ファイルへ |
| `teleop/collect.py` | 積分器、制御器の生成、開始状態、リセット、`close_renderer` | 手を入れて複写 | 上記だけを残し、`CollectSession`・`OperatorView`・`main`・台帳を外す |
| `teleop/recorder.py` | 配置、成功判定、`FrameSampler`、`EpisodeWriter` | 手を入れて複写 | 立方体 3 個、指示文をエピソードごとに、途中状態の保存、`DEFAULT_RAW_DIR`・`scan_raw`・`SessionLog` を外す |
| `teleop/device.py` | `DeviceInput`・`DeviceState` | そのまま複写 | 入口を DualSense と同じに保つための型 |
| `teleop/dualsense_device.py` | DualSense | 持ち込まない | 使うのは `PROFILES["collect"]` の速度上限（0.20・0.10 m/s）だけなので、値を設定ファイルに移す |
| `teleop/replay.py`、`replay_check.py` | 再生確認 | 手を入れて複写 | 途中の状態からの再生を足す。陰性対照は残す |
| `teleop/ledger.py`、`episode_ledger.py` | 台帳 | 持ち込まない | 手順書 §1-1 |
| `scripted_demo.py` | 台本 | 書き直し | 開ループ・内部段階・台帳結合（D5）。`ScriptPad`・`pad_state`・`velocity_to` の考え方と `Rig.loop` の順序は継ぐ |
| `convert_to_lerobot.py` | 変換・検証 | 手を入れて複写 | 旗の持ち越し、ラベルなしのマニフェスト、`recorder.DEFAULT_RAW_DIR` への依存を外す。`episode_arrays` と `--verify` はそのまま |
| `vla_image_spec.py`、`vla_state.py` | 画像変換・状態の唯一の定義 | そのまま複写 | 卒研と同一でなければならない |
| `vla_observation.py` | 評価時の観測 | そのまま複写 | 出力先の `conversion.json` との照合を継ぐ |
| `closed_loop_eval.py` | 閉ループ評価 | 手を入れて複写 | `EvalRig`・`execute_action`・`SuccessJudge`・`TrialLog`・`PolicyActions` の入力照合を継ぎ、実行器（sync / naive / rtc）、乱数の分離、多物体の判定、誘発、指標を足す。sync・s=50 で元と一致することを Step G で確かめる |
| `train_launcher.py` | 学習の起動器 | 手を入れて複写 | パスを設定ファイル経由に、HF_HOME を `<ROOT>` に、統計量の固定と続けて学習（R2・R1+）を足す。外部送信の停止・rename_map・conversion.json の写しはそのまま |
| `start_training.bat`、`train_config_example.json` | 起動の入口・設定例 | 書き直し | `02_*` の探索と C:\VLA のパスを含む |
| `code_version.py` | コードの版 | 手を入れて複写 | git の場所と対象ファイルの一覧を新しい構成に |
| `gui/gui_core.py` の `wilson_interval()` | Wilson 区間 | 手を入れて複写（関数だけ） | gui_core 全体は tkinter 画面・台帳・03_収録 と結合 |
| `gui/*` の残り、`collect_main.py`、`main.py`、`*.bat`（収録・練習・画面）、`teleop/{filters,frames,cameras,game,shelf_task,pad_feedback,touch_device,probe_touch}.py`、`probe_*.py`、`test*.py`（直下） | 収録画面・デモ・機器 | 持ち込まない | スコープ外（手順書 §0.3） |
| `tests/test_vla_image_spec.py`、`test_vla_state.py`、`test_vla_observation.py`、`test_convert_to_lerobot.py`、`test_replay.py`、`test_closed_loop_eval.py`、`test_controller_ik.py`、`test_train_launcher.py` | 既存の検査 | 手を入れて複写 | 流用する部品の回帰検査として。台帳・03_収録 への依存を外す |
| `02_環境\lerobot\scripted_check\analysis\route_eval_summary.py` | 段階の材料の集計 | 手を入れて複写（一部） | 持ち上げ高さなどの材料の計算を段階判定の参考にする |
| `01_既存システム\code\scripts\12_closed_loop_rollout.py` | 旧システム（mink 使用） | 持ち込まない | mink の `Configuration`・`FrameTask`・`solve_ik(solver="daqp")` の使用例として参照のみ |
| `hf_home\hub\models--lerobot--smolvla_libero\snapshots\31d453f...` | 出発点モデル | そのまま複写（`<ROOT>\models\`） | HF キャッシュの構成（refs・snapshots）ごと |
| `hf_home\hub\models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct\snapshots\7b375e1...` | SmolVLM2 の重み・設定・字句解析器 | そのまま複写（`<ROOT>\models\`） | 名前で読みに行くため（§2-16） |
| `scripted_check\train\stage2_20260917-221354\checkpoints\019000\pretrained_model` と同フォルダの `conversion.json`・`train_run.json` | 予備実験のチェックポイント | そのまま複写（`<ROOT>\outputs\` の下） | G0。`vla_observation` は出力先の `conversion.json` を要求するので、フォルダ構成（`<out>/checkpoints/<step>/pretrained_model`）ごと写す |
| `scripted_check\raw\stage2\2026-09-17\ep_8000xx` から 1 本 | 流用元の記録 | そのまま複写 | G0 の再生確認 |
| `scripted_check\eval\stage2_closed_loop\`（json・summary のみ） | 流用元の評価結果 | そのまま複写（動画は除く） | G0 の試行ごとの比較（D4） |

---

## 6. 調査の前後の全ファイル一覧の比較

| | 開始時 | 終了時 |
|---|---|---|
| ファイル | `docs\local\A_vla_listing_before.csv` | `docs\local\A_vla_listing_after.csv` |
| 件数 | 148,972 | 148,972 |
| CSV の SHA-256 | `C668C68990D95C236B681140DF0ED4F324E81699B8E437B9385765AABBBF7E7B` | `C668C68990D95C236B681140DF0ED4F324E81699B8E437B9385765AABBBF7E7B` |

`Compare-Object`（パス、大きさ、更新日時 UTC）の差: **0 件**。C:\VLA 以下に変更も新しいファイルもない。

---

## 7. 完了条件の確認

- 23 項目すべてに回答した。「不明」は 1 件（23 の推論 1 回の時間。理由と確かめ方を記載）
- 調査の前後で C:\VLA 以下の全ファイル一覧が一致した（§6）

## 8. 承認をお願いしたい点（Step B に進む前）

1. **D1（IK と衝突回避）**: Step B でどの案を主に提案するかの方向づけ（mink を導入して置き換える／G0 は既存 IK で通してから切り替える／既存 IK に自作で足す）
2. **D4（評価の種）**: G0 で予備実験と同じ種 100000–100019 を使ってよいか
3. **D3（RTC の参照実装）**: `examples/rtc` の代わりに同梱の `lerobot/rollout/inference/rtc.py` に合わせてよいか
4. **D6・D7（環境）**: 新しい環境の Python 本体と、不足パッケージ（mink、peft、scipy、matplotlib、Anthropic SDK）をネットワークから入れることを前提に Step B を書いてよいか

---

## 付録 A. パッケージの一覧（dist-info のフォルダ名から）

**.venv（Python 3.12）**: absl_py-2.5.0, accelerate-1.15.0, aiohappyeyeballs-2.7.1, aiohttp-3.14.3, aiosignal-1.4.0, annotated_doc-0.0.5, anyio-4.15.1, attrs-26.1.0, av-15.1.0, certifi-2026.7.22, charset_normalizer-3.5.1, click-8.5.0, cloudpickle-3.1.2, cmake-4.1.3, colorama-0.4.6, datasets-4.8.5, dill-0.4.1, docopt-0.6.2, draccus-0.11.6, einops-0.8.2, etils-1.14.0, farama_notifications-0.0.6, filelock-3.32.3, frozenlist-1.8.0, fsspec-2026.2.0, glfw-2.10.2, gymnasium-1.3.0, h11-0.16.0, hf_xet-1.6.0, httpcore-1.0.9, httpx-0.28.1, huggingface_hub-1.31.0, idna-3.19, iniconfig-2.3.0, jinja2-3.1.6, jsonlines-4.0.0, lerobot-0.6.1, markdown_it_py-4.2.0, markupsafe-3.0.3, mdurl-0.1.2, mergedeep-1.3.4, mpmath-1.3.0, mujoco-3.2.3, multidict-6.8.0, multiprocess-0.70.19, mypy_extensions-1.1.0, networkx-3.6.1, num2words-0.5.14, numpy-2.2.6, opencv_python_headless-4.13.0.92, packaging-25.0, pandas-2.3.3, pillow-12.3.0, pluggy-1.6.0, propcache-0.5.4, psutil-7.2.2, pyarrow-25.0.1, pygments-2.21.0, pyopengl-3.1.10, pytest-9.1.1, python_dateutil-2.9.0.post0, pytz-2026.3.post1, pyyaml-6.0.3, regex-2026.9.10, requests-2.34.2, rich-15.0.0, safetensors-0.8.0, setuptools-78.1.0, shellingham-1.5.4, six-1.17.0, sympy-1.14.0, termcolor-3.3.0, tokenizers-0.22.2, toml-0.10.2, torch-2.11.0+cu126, torchcodec-0.11.1, torchvision-0.26.0+cu126, tqdm-4.70.1, transformers-5.5.4, typer-0.27.2, typing_extensions-4.16.0, typing_inspect-0.9.0, tzdata-2026.4, urllib3-2.8.0, xxhash-4.0.1, yarl-1.25.1, zipp-4.1.0

（torchcodec は FFmpeg の DLL が見つからず読み込めない警告が学習ログに毎回出ているが、画像は埋め込み形式なので学習には影響していない）

**python311（Python 3.11 組み込み版）**: absl_py-2.5.0, cffi-2.1.1, colorama-0.4.6, etils-1.14.0, fsspec-2026.6.0, gitdb-4.0.12, gitpython-3.1.50, glfw-2.10.0, hidapi_usb-0.3.2, iniconfig-2.3.0, mujoco-3.2.3, numpy-1.26.4, opencv_python-4.10.0.84, packaging-26.2, pip-26.1.2, pluggy-1.6.0, pycparser-3.0, pydualsense-0.7.5, pygments-2.20.0, pyopengl-3.1.10, pyopenhaptics-1.0.1, pytest-9.1.1, robot_descriptions-1.12.0, setuptools-83.0.0, smmap-5.0.3, tqdm-4.68.4, typing_extensions-4.16.0, wheel-0.47.0, zipp-4.1.0
