# torch が入らず、rtc.py との照合と雑音の検査を実行できない

- 番号: 0023（仮 0017） / 差出: cloud-runner / 宛先: human、main（写し: review） / 日時: 2026-09-25 18:12（日本時間）
- 返信先: 0014（指示書 §3「入らないものがあれば止まって ask」、§5 の 3「import できなければ ask」）。関係: PR 1 の 0012・0013
- 状態: 未読
- 関係する枝・PR・コミット: 枝 `cloud/runner`、PR「cloud/runner: 実行器の時間の流れ」
- 付け直し: 枝 `cloud/runner` での仮の番号 0017 を、本線が取り込むときに 0023 に付け直した（2026-09-25、決裁の指示）。「番号」の行のほか（返信先・本文・題）の 0014〜0020 は枝での仮の番号のまま（対応: 0014→0020、0015→0021、0016→0022、0017→0023、0018→0024、0019→0025、0020→0026）。それ以外の番号は本線の番号。対応表は STATUS.md

## 要点（3行以内）

この枝の環境でも download.pytorch.org への接続が 403 で、torch・torchvision の CPU 版が入らない（0013 の 2 の許可先の追加は、この時点ではまだ効いていない）。
そのため `tests/policy/test_rtc_compare.py` の 60 件（雑音 4、`rtc.py` との照合 56）は飛ばしになり、結果がない。
どちらで確かめるかを決めてほしい: (a) 許可先に足した後にこの枝で回す、(b) 本線が取り込む前に Windows で回す。

## 本文

- 手順: `uv pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu` → `tunnel error: unsuccessful`。回線の中継の状態では `download.pytorch.org:443` への CONNECT が 403（2026-09-25 18:05 ごろ、日本時間）。迂回はしていない。PyPI の `2.11.0+cu130` も入れていない（0013 の 2 で採らなかった方法なので）
- 固定一覧の残り（104 件）と `-e .` は入った。lerobot 0.6.1 のソースは読めるが、`lerobot/rollout/inference/rtc.py` は冒頭で `import torch` するので import できない
- 実行していない検査: `test_rtc_compare.py`
  - §5 の 2: `noise_generator` の 4 件（同じ (seed, i) で同じ、i・種が違えば違う、CPU の生成器で SeedSequence から作った種、方式が違っても i 回目どうしは同じ）
  - §5 の 3: `_normalize_prev_actions_length` を import して直接比べる 56 件
- 代わりにしたこと（照合の代わりにはならない）: numpy だけの検査で、j の定義どおりの行と、切り詰め・0 詰めを確かめた（合格）。`rtc.py`・`action_queue.py` のソースとの読み合わせ（0015 §3）
- 飛ばしは `pytest.importorskip("torch")`。torch がある環境では、そのまま実行される

| 選択肢 | 中身 | 後戻り |
|---|---|---|
| (a) | 人が環境の許可先に `download.pytorch.org` を足す（0013 の 2）。その後、この枝で torch を入れて 60 件を回し、結果を掲示板に足す | 容易。枝に結果の報告が 1 通増えるだけ |
| (b) | 本線が PR を取り込む前に、Windows の `.venv`（torch あり）で `python -m pytest tests/policy` を回す。落ちたらこの枝に戻す | 容易。ただし落ちたときの直しが 1 往復遅れる |

## 求めること

(a) か (b) か。答えが来るまで、§5 の 2 と 3（`rtc.py` との照合）は完了にしない。ほかの部分は完了している（0015）。
