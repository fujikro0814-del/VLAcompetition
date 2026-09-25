# 評価の 1 試行の記録（trial record）

版: 1（2026-09-25）/ 書く: 本線の評価器（`src/recovla/eval/closed_loop.py`、Step G で拡張）/ 読む: `src/recovla/eval/metrics.py`（cloud/metrics）、`src/recovla/eval/report.py`・`src/recovla/viz/overlay.py`（cloud/figures）

単位・座標系・時間の格子・色の並びは `README.md` の共通の決まりに従う。

## 1. ファイル

1 回の評価の実行は 1 つのフォルダ（`outputs/eval/<名前>_<日時>/`）。その中に試行ごとに次のファイルを書く。

| ファイル | 中身 |
|---|---|
| `trial_NNNN.json` | 試行の属性と出来事（§2） |
| `trial_NNNN.npz` | こまごとの配列（§3） |
| `trial_NNNN_raw.mp4` など | 動画。指標は使わない |

実行全体の `summary.json` は `{"run": 名前, "config": 実行の設定, "trials": [{"trial", "seed", "success"}...]}`。

指標の関数は json と npz だけから計算する。MuJoCo も方策も呼ばない。

## 2. `trial_NNNN.json`

```jsonc
{
  "record_version": 1,
  "trial": 3, "seed": 110003,
  "experiment": "E2",                  // E1〜E8、または選択用の名前（"K1_closed_loop" など）
  "condition": "R1_rtc_s10_d2",        // 条件の名前（表の列の見出しになる）
  "model": {"name": "R1", "checkpoint": "…/pretrained_model", "step": 30000},
  "runtime": {"mode": "rtc", "exec_interval": 10, "delay_steps": 2, "rtc_guidance_horizon": 10,
              "rtc_schedule": "EXP", "safety_filter": false},
  "layout": {"kind": "empty",          // empty | prefilled_1 | prefilled_2
             "start": "home",          // home | retreat
             "cubes": {"red": [x, y, yaw], "green": [x, y, yaw], "blue": [x, y, yaw]},   // m, m, rad（初期）
             "prefilled": ["blue"]},   // 箱に最初から入っている色
  "steps": [                           // 手順（単一手順の試行では 1 つ）
    {"target": "red", "instruction": "put the red cube in the box",
     "t_start": 0.0, "t_end": 14.52,
     "success": true, "t_success": 14.52}          // 成功の判定は物理ステップごと（§4）
  ],
  "success": true,                     // 全手順が成功
  "time_limit_s": 30.0,
  "induce": {"kind": "P2",             // P1 | P2 | P3 | null
             "params": {"u": 0.37},    // 試行の開始時に誘発の乱数列から引いた値（P1: offset_m, direction_rad ／ P2: u, r0_m ／ P3: offset_xy）
             "fired": true, "t_fire": 6.84,         // 誘発の操作をした時刻（発生しなければ fired=false, t_fire=null）
             "established": true, "t_established": 7.40,
             "t_failure": 7.40,        // 反応時間の起点: P2・P3 は立方体が着地した時刻、P1 は失敗が確定した時刻
             "reason": null},          // 成立しなかった理由（"lifted" "landing_tilt" "landing_in_box" など）
  "obstacles": ["cube_red", "cube_green", "cube_blue", "wall_xp", "wall_xn", "wall_yp", "wall_yn"],
  "inference": [                       // 推論ごと（runner.md §4 の記録と同じ）
    {"i": 0, "k_obs": 0, "k_valid": 0, "offset": 0, "left_over_len": 0, "reset": true,
     "wall_s": 0.271, "wall_breakdown_s": {"preprocess": 0.012, "vlm_prefix": 0.081, "denoise": 0.170}}
  ],
  "input_check": {"overhead": {…}, "wrist": {…}, "state_shape": [1, 15], "state_normalized_max_abs": 3.4},
  "code_version": {…}
}
```

## 3. `trial_NNNN.npz`（N = こまの数、20 Hz）

| キー | 形 | 型 | 中身 |
|---|---|---|---|
| `step` | (N,) | int64 | 物理ステップの番号（25 の倍数） |
| `sim_time` | (N,) | float64 | [s] |
| `ee_pos` | (N, 3) | float64 | 手先（hand 原点） |
| `ee_quat` | (N, 4) | float64 | 手先の姿勢（w ≥ 0） |
| `fingertip` | (N, 3) | float64 | 指先の中心 |
| `fingers` | (N, 2) | float64 | finger_joint1・2 [m]（開き 0〜0.04 ずつ） |
| `x_des` | (N, 3) | float64 | 手先参照位置 |
| `gripper_closed` | (N,) | bool | 制御器のグリッパの状態 |
| `cube_pos` | (N, 3, 3) | float64 | [こま, 色, xyz] |
| `cube_quat` | (N, 3, 4) | float64 | [こま, 色, wxyz] |
| `cube_linvel` | (N, 3, 3) | float64 | [こま, 色, xyz] 世界座標 [m/s] |
| `cube_in_box` | (N, 3) | bool | 成功の体積の中か（静止は問わない） |
| `target` | (N,) | int8 | そのこまの手順の目標の色の添字（0〜2）。手順の外は −1 |
| `phase` | (N,) | int8 | 目標に対する段階（§5 の符号）。台本の `phase_of` と同じ純関数で真値から判定する |
| `action` | (N, 7) | float64 | このこまの後に実行した 10 Hz の行動（同じ行動が 2 こまに並ぶ）。なければ NaN |
| `chunk_id` | (N,) | int32 | このこまの後の行動を出した塊の番号（inference の i）。なければ −1 |
| `chunk_index` | (N,) | int32 | その塊の中の行動の添字（0〜49）。なければ −1 |
| `chunk_switch` | (N,) | bool | このこまの後の行動から新しい塊に切り替わった |
| `contact_robot` | (N, K) | bool | 直前の 25 物理ステップの間に、手・指（hand・left_finger・right_finger）が障害物 k に 1 度でも触れた。K と並びは json の `obstacles` |
| `contact_cube_cube` | (N, 3, 3) | bool | 同じ窓の中で、色 a の立方体が色 b の立方体に触れた（対称） |
| `min_dist` | (N, K) | float64 | 手・指と障害物 k の最短距離 [m]（こまの時点。`contact.distmax_m` より遠ければその値） |
| `safety_active` | (N,) | bool | 安全フィルタが指令を変えた（フィルタなしは常に偽） |
| `induce_active` | (N,) | bool | 誘発が指令を変えていた |

塊の予測経路（描画用）は別のキーに入れる（M = 推論の回数）。

| キー | 形 | 中身 |
|---|---|---|
| `chunk_k_valid` | (M,) int32 | 塊が有効になった行動の番号 k |
| `chunk_xdes_pred` | (M, 50, 3) float64 | 塊の各行動を、有効になった時点の x_des から積算した手先参照位置の予測経路 |
| `chunk_grip_pred` | (M, 50) float64 | 塊のグリッパの行動（±1 の連続値のまま） |

## 4. 判定の決まり（指標の関数が使う定義。数値は `configs/default.yaml` の `eval`）

- **成功**（評価器が判定し、json に書く）: 目標の立方体が箱の成功の体積の中で、速さ 0.01 m/s 未満の状態が 1.0 s 続く。物理ステップごとに判定する（流用元と同じ）。制限時間は 1 手順 30 s
- **持ち上げ**: 立方体の z がその試行の最初のこまの z から `eval.error_lift_m`（0.02）以上高い
- **誤り**: 目標でない色を持ち上げた（箱の中の先客を含む）
- **巻き添え**: 目標でない立方体が、`contact_robot` が真の窓を含む区間で、初期位置から水平に 0.02 以上動いた
  - 同じ移動のうち、誘発後に落ちた目標の立方体との `contact_cube_cube` が先にあるものは「誘発による移動」として別に数える
- **復帰**: 誘発が成立した試行のうち、制限時間内に成功した
- **反応時間**: `induce.t_failure` の時点の「指先の中心と目標の立方体の距離」を d0 とする。そこから d0 − 0.02 以下になった最初のこまの時刻までの秒数
- **復帰時間**: `induce.t_fire` から、目標が再び持ち上げ（0.02）の状態になった最初のこまの時刻までの秒数
- **継ぎ目の跳び**: 行動 k の手先参照速度を v_k = action[k, :3] / 0.1 とする
  - 塊が切り替わった k（`chunk_switch`）での |v_k − v_{k−1}| [m/s] を跳びとする
  - 切り替わりでない k の同じ量を対照として並べる
  - 手先速度 (ee_pos の差分 / 0.05) の、同じ時点の差も出す
- **躍度**: x_des の 10 Hz の 3 階差分 / 0.1³ の大きさの二乗平均の平方根 [m/s³]
- **接触回数**: `contact_robot` の、目標以外の立方体と壁の列について、偽から真に変わった回数

## 5. 段階の符号（`phase`）

台本の段階の判定（B_提案書 §8）と同じ表を使う。

| 符号 | 段階 | 段階別の到達の区分 |
|---|---|---|
| 0 | approach | 接近 |
| 1 | descend | 接近 |
| 2 | close | 把持 |
| 3 | reopen | （掴み損ね） |
| 4 | lift | 把持 |
| 5 | carry | 搬送 |
| 6 | release | 設置 |
| 7 | settle | （待ち） |
| 8 | retreat | 退避 |
| 9 | done | 完了 |

段階別の到達（表の列）は、試行の中で到達した区分の最も先のもの（接近 < 把持 < 搬送 < 設置 < 退避 < 完了）。

## 6. 指標の関数の型（cloud/metrics が作る）

```python
# src/recovla/eval/metrics.py — 純関数。入力は読み込んだ記録だけ
@dataclass(frozen=True)
class TrialRecord:
    meta: dict              # trial_NNNN.json
    arrays: dict            # trial_NNNN.npz の {キー: ndarray}

def load_trial(json_path) -> TrialRecord
def trial_metrics(rec: TrialRecord, eval_cfg: dict) -> dict
    # 返り値（1 試行 1 行。results.md の trials.csv の列と同じ名前）:
    # success, error, collateral, collateral_induced, stage_reached, induce_fired, induce_established,
    # recovered, reaction_time_s, recovery_time_s, seam_jump_mean, seam_jump_max, nonseam_jump_mean,
    # seam_ee_speed_jump_mean, jerk_rms, contacts_n, inference_mean_s, inference_p95_s
def seam_jumps(action: np.ndarray, chunk_switch: np.ndarray, dt: float = 0.1) -> tuple[np.ndarray, np.ndarray]
    # (切り替わりの跳び, 切り替わりでない時点の同じ量) [m/s]
def reaction_time(t, fingertip, target_pos, t_failure, close_m=0.02) -> float   # 見つからなければ NaN
def recovery_time(t, target_z, z0, t_fire, lift_m=0.02) -> float
def jerk_rms(x_des_10hz: np.ndarray, dt: float = 0.1) -> float
```

合成の軌跡で検算する（手順書 Step G 完了条件 5）。跳びと時刻を前もって決めた記録を作り、上の関数が同じ値を返すこと。
