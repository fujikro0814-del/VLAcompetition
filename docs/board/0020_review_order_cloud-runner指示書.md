# 指示書 — 支線 cloud/runner（実行器の時間の流れ）

- 付け直し: 枝 `cloud/runner` での仮の番号 0014 を、本線が取り込むときに 0020 に付け直した（2026-09-25、決裁の指示）。「番号」の行のほか（返信先・本文・題）の 0014〜0020 は枝での仮の番号のまま（対応: 0014→0020、0015→0021、0016→0022、0017→0023、0018→0024、0019→0025、0020→0026）。それ以外の番号は本線の番号。対応表は STATUS.md

差出: 監督 / 宛先: 支線（クラウドの Claude Code） / 日付: 2026-09-25 / 返信先: 掲示板 0011
リポジトリ: fujikro0814-del/VLAcompetition（公開）/ 枝: `cloud/runner`

## 0. 最初にすること

1. 枝 `cloud/runner` を main から切る。`main` と他の枝には push しない
2. この指示書を、そのまま `docs/board/NNNN_review_order_cloud-runner指示書.md` として保存する（NNNN は、その時点の最大番号＋1。取り込みのときに番号がぶつかったら本線が付け直す）
3. 次を読む: `docs/board/README.md`（運用規則）、`docs/interfaces/README.md`・`runner.md`（**これが仕様。食い違ったらこちらが正**）、`docs/B_提案書.md` の §7（方策の実行器）・§9（乱数）・§15

## 1. 目的

方策の実行器のうち、「いつ推論を始め、いつ結果が有効になり、塊のどの行を実行し、RTC に前の塊のどこを渡すか」という時間の流れだけを、方策と物理から切り離した純粋な部品（`Schedule`）として作る。本線は Step G でこれを実物の SmolVLA と評価器につなぐ。

## 2. 持ち場（書いてよいパスはこれだけ）

- `src/recovla/policy/schedule.py`（新規）
- `tests/policy/`（新規。模擬の方策もここに置く）
- `docs/board/` の自分の報告・ask

`docs/interfaces/`、`configs/`、`src/recovla/` のほかの場所（`common/` を含む）、`env/` は変えない。変えたいときは ask を出す。

## 3. 環境

`env/requirements-cloud.txt` の冒頭の注記どおりに、リポジトリ直下の `.venv`（Git 管理外）へ入れる（Python 3.12、torch と torchvision は CPU 版を同じ版で）。lerobot 0.6.1 は同梱の `rtc.py` との照合に使う。版の固定一覧は書き換えない。入らないものがあれば止まって ask。

## 4. 作るもの

`runner.md` §2〜§5 のとおりに `RuntimeConfig`・`Decision`・`Schedule` を作る。

- sync・naive・rtc の 3 方式。持ち越さない時点（最初・`reset()` の直後）の扱い
- rtc の前の塊の残り `left_over`
  - j = o_{i−1} + (k_i − v_{i−1})、L = 前の塊[j : H]
  - E に揃える（長ければ先頭から E 行、短ければ 0 詰め）
  - 空間は後処理の前（渡された塊のまま。`(H, 32)` の詰め物つき）
- `decide` は k を 0 から 1 ずつ増やして呼ぶ前提。飛ばしたら ValueError
- 推論ごとの記録 `log`（`runner.md` §4 のキー。`wall_s` と `wall_breakdown_s` は本線が測るので、ここでは None で枠だけ）
- 推論 i の雑音の生成器を作る補助関数 `noise_generator(seed: int, i: int) -> torch.Generator`（`SeedSequence(seed, spawn_key=(2, i))` から。`runner.md` §5）

配置（spawn_key (0,)）と誘発（(1,)）の乱数列の関数は `common/` に置くもので、本線の担当である。まだ無いので、**支線では作らない**。代わりに、必要な関数の型の案を ask で出す（下の §5 の 2 はその後に本線が足す）。

## 5. 検査（`tests/policy/`。GPU・モデル・ネットワークなしで通ること）

模擬の方策は、塊の中身が (推論の番号 i, 塊の中の添字) から一意に決まるもの（例: 値 = 1000·i + 添字）にし、実行された行から「どの塊のどの行か」を逆算できるようにする。

1. **naive・d=0 が sync と一致**: s を 5・10・25・50 で変え、200 手の `execute` の列が完全に一致する
2. **乱数**: `noise_generator` について、同じ (seed, i) なら同じ雑音、i が違えば違う雑音、方式が違っても i 回目どうしは同じになる。配置と誘発の検査は、本線が `common/` に関数を入れた後に足す（今回は ask だけ）
3. **left_over**: j の定義どおりの行が入ること。E を 10・40 で変え、切り詰めと 0 詰めが、lerobot 0.6.1 に同梱の `lerobot/rollout/inference/rtc.py` の `_normalize_prev_actions_length` と同じ結果になること（その関数を import して直接比べる。import できなければ ask）
4. **naive と rtc の時間の流れ**: d を 0〜4、s を 10 で、各塊が有効になる手 v_i、添字 o_i、切り替わりの直前まで前の塊の続きが実行されることを、記録と `execute` から確かめる
5. **持ち越さない時点**: 最初と `reset()` の直後は `block=True`、`left_over=None`、o=0
6. **不正な設定**: s=0、d<0、d≥s、s+d>H で ValueError。k を飛ばした `decide` も ValueError
7. 既存の検査（`tests/legacy/` など）のうち、クラウドで通っていたものが引き続き通ること

## 6. 返すもの（完了条件）

- 枝を push し、PR を作る（題: 「cloud/runner: 実行器の時間の流れ」）
- 掲示板に報告 `NNNN_cloud-runner_report_実行器の時間の流れ.md` を書き、PR に含める。中身は次のとおり
  - 作ったものの一覧
  - 検査の結果（件数、合否）
  - `rtc.py` との照合の方法と結果
  - 仕様（`runner.md`）で曖昧だった点と、どう解釈したか
  - 本線が実物につなぐときの注意（`deliver` に渡す塊の形、後処理の前後の区別など）
- 乱数列の関数の型の案を、ask `NNNN_cloud-runner_ask_乱数列の関数.md` として同じ PR に含める
- PR の変更は §2 の持ち場の中だけ

## 7. やらないこと

- LeRobot の方策を呼ぶ部分（`runner.py`、`RTCConfig` の有効化、`predict_action_chunk` の呼び出し）。本線の担当
- `common/`、`docs/interfaces/`、設定ファイルの変更
- 依存の追加（固定一覧にないパッケージを入れない）
- 学習、方策の推論、MuJoCo の実行

## 8. 迷ったとき

仕様が曖昧で結果が変わる点は、仮定で埋めずに ask（`NNNN_cloud-runner_ask_…`）を書いて PR に含め、その点は止まる。影響しない部分は先に進めてよい。
