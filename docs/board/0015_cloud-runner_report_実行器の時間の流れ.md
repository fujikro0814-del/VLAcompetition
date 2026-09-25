# 実行器の時間の流れ（cloud/runner）

- 番号: 0015 / 差出: cloud-runner / 宛先: review、main / 日時: 2026-09-25 18:12（日本時間）
- 返信先: 0014（監督の指示書。受け取ったときの題は「返信先: 掲示板 0011」）
- 状態: 未読
- 関係する枝・PR・コミット: 枝 `cloud/runner`（起点 main 1a6ce6d）、PR「cloud/runner: 実行器の時間の流れ」

## 要点（3行以内）

`src/recovla/policy/schedule.py`（`RuntimeConfig`・`Decision`・`Schedule`・`noise_generator`）と検査を作った。numpy だけで済む検査 64 件はすべて合格。
torch が要る検査（指示書 §5 の 2 と、§5 の 3 の `rtc.py` との照合。計 60 件）は、**torch の CPU 版が入らないので実行していない**（download.pytorch.org が 403。0017 の ask）。
仕様どおりに作ると、SmolVLA の `predict_action_chunk` が返す塊は `(H, 7)` で、runner.md の `(H, 32)` と合わない（0018 の ask）。`Schedule` は A を問わないので、この部品には影響しない。

## 本文

### 0. 番号

受け取った指示書は、指示どおり `docs/board/` に保存した。main の最大番号は 0011。ただし、まだ取り込まれていない PR（cloud/setup-check、https://github.com/fujikro0814-del/VLAcompetition/pull/1）が 0012・0013 を使っている。ぶつからないように 0014 から使った。

| 番号 | 中身 |
|---|---|
| 0014 | 監督の指示書（そのまま） |
| 0015 | この報告 |
| 0016 | ask: 乱数列の関数の型の案（指示書 §6） |
| 0017 | ask: torch が入らず、`rtc.py` との照合と雑音の検査を実行できない（指示書 §3・§5 の 3「import できなければ ask」） |
| 0018 | ask: 塊の次元（runner.md の `(H, 32)` と、lerobot 0.6.1 の `predict_action_chunk` の出力 `(H, 7)`） |

### 1. 作ったもの

| パス | 中身 |
|---|---|
| `src/recovla/policy/schedule.py` | `RuntimeConfig`（生成時に検査）、`Decision`、`Schedule`（`reset`・`decide`・`deliver`・`action`・`log`）、`normalize_left_over`（`_normalize_prev_actions_length` を numpy で書いたもの）、`noise_generator(seed, i)` |
| `tests/policy/mock_policy.py` | 模擬の方策。塊[添字, a] = 1000·(i+1) + 添字 + a/64（float32、(50, 32)）。+1 は 0 詰めの行と区別するため。後処理の模擬は −塊[:, :7]。本線の呼び方（decide → deliver → action）をなぞる `run(cfg, n, resets)` |
| `tests/policy/test_schedule.py` | numpy だけの検査（指示書 §5 の 1・3 の行の中身・4・5・6）。64 件 |
| `tests/policy/test_rtc_compare.py` | torch が要る検査（§5 の 2、§5 の 3 の `rtc.py` との照合）。60 件。torch が無ければ飛ばす（`pytest.importorskip`） |

torch の import は `noise_generator` の中だけにした。時間の流れの部分は torch なしで動く。

### 2. 検査の結果

環境: `.venv`（Python 3.12.3、`uv pip install --no-deps -r env/requirements-cloud.txt` と `-e .`）。torch・torchvision は入っていない（§5）。

命令: `.venv/bin/python -m pytest -q -p no:cacheprovider tests/policy`

| 指示書 §5 | ファイル | 件数 | 結果 |
|---|---|---|---|
| 1 naive・d=0 が sync と一致（s = 5・10・25・50、200 手。rtc・d=0 も同じ） | test_schedule.py | 4 | 合格 |
| 2 乱数（`noise_generator`） | test_rtc_compare.py | 4 | **未実行**（torch なし） |
| 3 left_over の行: j の定義どおり、E = 10・40、(s, d) = (10,0)(10,2)(10,4)(25,3)(40,10)(50,0)。切り詰め・0 詰め・長さ 0 | test_schedule.py | 15 | 合格 |
| 3 `rtc.py` の `_normalize_prev_actions_length` との照合（関数を import して直接比べる） | test_rtc_compare.py | 56 | **未実行**（torch なし） |
| 4 naive と rtc の時間の流れ（d = 0〜4、s = 10、各 200 手）: v_i、o_i、塊ごとの実行範囲、切り替えの直前まで前の塊の続き。記録・`execute`・実行した行の中身の 3 つで確かめる | test_schedule.py | 11 | 合格 |
| 5 持ち越さない時点（最初と `reset()` の直後。3 方式 × d = 0・3。reset の位置は、推論の直後で塊が有効になる前・推論の間・推論の予定の手・続けて 2 回） | test_schedule.py | 7 | 合格 |
| 6 不正な設定・k を飛ばした `decide`・`deliver` の呼び方（ほかに、記録の枠に本線が書き込めることの 1 件） | test_schedule.py | 27 | 合格 |
| 計 | | 124 | **合格 64・未実行 60** |

**7 既存の検査**: ファイルごとに別の process で回した（一括で回すと描画のところで process ごと止まる。0012 の setup-check と同じ）。

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
| 計（102 件） | 70 | 1 | 13 | 18 |

cloud/setup-check の報告（PR 1 の 0012 §3）とファイルごとに同じ数。落ちたものの原因も同じ（描画の文脈がない、Windows 専用、Git 管理外の outputs/ がない、torch がない、場面の SHA-256）。この枝で増えた不合格はない。

### 3. `rtc.py` との照合の方法と結果

**方法 1（検査。未実行）**: `test_rtc_compare.py` で `from lerobot.rollout.inference import rtc` を import し、次の 2 つを要素ごとに完全一致で比べる。
- `normalize_left_over(L, E)` と `rtc._normalize_prev_actions_length(torch.from_numpy(L), E)`。E = 10・40、長さ 0〜50 の 11 通り、float32・float64（44 件）
- `Schedule` が渡す `left_over` と、前の塊[j:H] を `rtc._normalize_prev_actions_length` で揃えたもの（12 件）

`rtc.py` は冒頭で `import torch` するので、この環境では import できない。**結果はまだない**（0017）。

**方法 2（コードの読み合わせ。済み）**: `.venv` に入った lerobot 0.6.1 のソースで確かめた。
- `_normalize_prev_actions_length`（rtc.py）: 2 次元でなければ ValueError。長さが同じならそのまま、長ければ `[:target_steps]`、短ければ同じ dtype の 0 の配列の先頭に入れる。`normalize_left_over` は同じ手順を numpy で書いたもの
- j の定義: `ActionQueue.merge`（`policies/rtc/action_queue.py`、RTC が有効のとき）は、新しい塊の先頭の `delay` 行を捨てて差し替え、消費位置を 0 に戻す。`get_left_over` は差し替えた後の塊[消費位置 :]を返す。元の塊の添字で言えば、残りの先頭は `delay + 消費した手数` で、runner.md の j = o_{i−1} + (k_i − v_{i−1}) と同じ（o = d、k_i − v_{i−1} = その塊から消費した手数）
- 新しい塊を添字 d から実行するのも、`merge` が先頭の d 行を捨てるのと同じ
- 違い（仕様どおり）: LeRobot の実行器は d を実測の遅れから毎回決める。ここでは設定の d を固定で使う

**方法 2 で分かった、今の設定での値**: 時間の流れの式から、持ち越す推論では常に j = s になる（i = 1 は o_0 = 0・v_0 = 0・k_1 = s。i ≥ 2 は o = d・k_i − v_{i−1} = s − d）。そのため残りの長さは H − s。s = 10 なら 40 行で、**E = 10 は切り詰め、E = 40 はちょうど 40 行（0 詰めは起きない）**。0 詰めは E > H − s のときだけ起きる（検査では s = 25・40・50 で確かめた）。

### 4. 仕様（runner.md）で曖昧だった点と、どう解釈したか

結果が変わりうる点のうち、仕様や他の文書から一つに決まると判断したものは下のように作った。決まらないものは ask にした（0016 の 2、0018）。

| 点 | 解釈 | 理由 |
|---|---|---|
| `reset()` の後、いつ推論するか | 次の `decide(k)` のその k で推論する（block=True、v = k、o = 0、L なし）。以後はそこから s 手ごと | §2「そこを i = 0 と同じに扱う」。指示の切り替えの後に古い塊を続けない |
| reset の後の塊の番号 i | 0 に戻さず、試行の中で通しのまま続ける。雑音も通しの i で引く | trial_record.md の `chunk_id` は「inference の i」で、1 試行の中で一意でないと、こまと推論を対応づけられない |
| reset で、有効になる前の塊を捨てたときの記録 | 記録（`k_valid`）は予定の値のまま残す。実行されなかったことは、こまの `chunk_id` にその i が現れないことで分かる | 記録のキーは §4 のものだけにした |
| sync の d | 使わない。検査も d = 0 として行う（sync では `delay_steps` が None でも、s = 50 と d > 0 の組でもよい） | §3「sync では無視して 0 とみなす」。`configs/default.yaml` の `delay_steps` は 1 つの値で、sync の s = 50 と組むと s + d > H になるため |
| E の範囲 | rtc のときだけ 1 ≤ E ≤ H を検査して ValueError（仕様にない検査を 1 つ足した） | E > H だと LeRobot の `modeling_rtc.py` で塊と形が合わず assert で落ちる。正しい設定の結果は変わらない |
| 持ち越さない推論の `inference_delay` | rtc でも 0 | `left_over` が None なら RTC の誘導は働かない（`modeling_rtc.py` は `prev_chunk_left_over is None` で普通の生成）。§3「rtc のとき」を、持ち越す推論のときと読んだ |
| naive の `left_over_len` | 0 | §4「渡した残りの長さ」。naive は何も渡さない |
| 推論しない手の `Decision` | `block=False`、`left_over=None`、`inference_delay=0` | — |
| naive・rtc の d = 0 の推論の `block` | False（物理を止めるのは最初・reset 後と sync だけ） | §3 の `block` の注記どおり。模擬の中では推論は常に瞬時 |
| 呼び方の誤り | `decide` の k を飛ばす・戻す・整数でない → ValueError（状態は変えない）。`deliver` を忘れて次の `decide`、待っていない `deliver`、`deliver` の前の `reset` → RuntimeError | §3「その k のうちに deliver() を呼ぶ」 |
| SeedSequence から torch の種への変換 | `SeedSequence(seed, spawn_key=(2, i)).generate_state(1, dtype=uint64)[0]` を `torch.Generator("cpu").manual_seed` に入れる | runner.md に変換の方法がない。性質（同じ (seed, i) で同じ、i が違えば違う）はどの変換でも満たす。確かめてほしい（0016 の 2） |

### 5. 本線が実物につなぐときの注意

1. **呼ぶ順**: 1 試行に 1 つ `Schedule` を作る（`reset()` は k を 0 に戻さない）。毎手 `dec = schedule.decide(k)` → `dec.infer` なら、その手のうちに推論して `schedule.deliver(塊)` → `schedule.action({i: 後処理の後の塊})[:7]` を実行する
2. **雑音の i**: `decide` が infer=True を返した時点で、その推論の記録が `schedule.log` の末尾に入っている。`i = schedule.log[-1]["i"]`、`noise = torch.randn((1, 50, 32), generator=noise_generator(seed, i)).to(device)`。生成器は CPU のもの（device によらず同じ雑音にするため）
3. **`deliver` に渡す塊**: `predict_action_chunk` の出力（**後処理の前**、正規化されたまま）を `squeeze(0)` して numpy にしたもの。`postprocessor` を通したものではない。bfloat16 は numpy にできないので float32 にしてから渡す。`Schedule` は受け取った塊を複写して持つ
4. **形**: lerobot 0.6.1 の SmolVLA の `predict_action_chunk` は、`_get_action_chunk` の中で詰め物を外して `(1, 50, 7)` を返す（`actions[:, :, :original_action_dim]`、`modeling_smolvla.py` 216〜218 行）。runner.md の「`(H, 32)` の詰め物つき」とは違う。`Schedule` は A を問わないので、どちらでも動く。`(E, 7)` の残りを渡しても、`modeling_rtc.py` が 32 次元まで 0 で詰める。LeRobot 自身の実行器も 7 次元の塊を残りとして渡している。どちらにするかは 0018
5. **`left_over` を方策に渡す**: `prev_chunk_left_over=torch.from_numpy(dec.left_over).to(device)`、`inference_delay=dec.inference_delay`。持ち越さない推論では `left_over` が None なので、そのまま None を渡す。`RTCConfig.execution_horizon` は `RuntimeConfig.execution_horizon`（E）と同じ値にする（`modeling_rtc.py` は、kwargs に `execution_horizon` が無ければ設定の値を使う）
6. **`action` に渡す塊**: 後処理の後の塊を、塊の番号 → `(H, A)` の辞書で渡す。使うのは、いま実行している塊と、推論済みで有効になる前の塊の 2 つだけ。`action` は行をそのまま返すので、先頭 7 次元を取るのは呼び出し側
7. **記録**: `wall_s`・`wall_breakdown_s` は None の枠。`schedule.log` は中の dict をそのまま返すので、測った値は `schedule.log[-1]["wall_s"] = …` と書き込めば残る。こまの `chunk_id`・`chunk_index` は `dec.execute`。`chunk_switch` は、`execute[0]` が前の手と違う手（k = 0 を真とするかは本線が決める）。1 手が 2 こまに並ぶので、同じ値を 2 こまに書く
8. **`block`**: 模擬では推論は常に物理を止めて計算するので、`block` は物理の進め方を変えない。記録と、実時間の扱い（待つ推論かどうか）にだけ使う
9. **reset**: 指示の切り替えとやり直しの直後、次の `decide` の前に `schedule.reset()` を呼ぶ。後処理の後の塊の辞書も、その時点で捨ててよい

### 6. 変更の範囲

足したファイル: `src/recovla/policy/schedule.py`、`tests/policy/`（`__init__.py`・`mock_policy.py`・`test_schedule.py`・`test_rtc_compare.py`）、`docs/board/` の 0014〜0018。`docs/interfaces/`・`configs/`・`src/recovla/common/`・`env/`・`pyproject.toml` は変えていない。依存は足していない。push の前にステージを見て、鍵・証明書・回線の設定の値・個人の情報を含まないことを確かめた。

## 求めること

報告。判断は 0016・0017・0018 で求める。0017 の答えが出るまで、§5 の 2 と 3（`rtc.py` との照合）は完了にしない。
