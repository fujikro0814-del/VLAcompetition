# 取り込みと番号の付け直し（PR #1・#2・#3、0022・0024・0019 への対応）

- 番号: 0027 / 差出: human / 宛先: main / 日時: 2026-09-25 18:45（日本時間。チャットでの指示を本線が書き写した。指示の時刻は 18:35〜18:40 ごろ）
- 返信先: 0022、0024、0019（いずれも付け直し後の番号。指示の文面では枝での仮の番号 0016、0018、0014 と書かれていた）
- 状態: 対応中
- 関係する枝・PR・コミット: PR #1（cloud/setup-check）、PR #2（cloud/metrics）、PR #3（cloud/runner）

## 要点（3行以内）

PR #1・metrics・#3（runner）を取り込む。取り込む前に Windows で tests/eval・tests/policy（torch ありで全件）と check_before_push.ps1 を通す。
枝どうしでぶつかった掲示板の番号は、取り込む順に本線が付け直し、STATUS に対応表を残す。以後、支線は仮の番号 cNN で書く。
common/seeds.py を入れる（0022）。runner.md を「(H, 7)、後処理の前」に直す（0024）。results.md に phi_correction と trial_metrics の全列を書き足す（0019）。

## 本文

（決裁の指示をそのまま書き写す。括弧の中は本線が付け直し後の番号を添えたもの）

PR #1（setup-check）・metrics・#3（runner）を取り込む。取り込む前に Windows で pytest tests/eval tests/policy
（torch ありで全件）と scripts/check_before_push.ps1 を通す。
掲示板の番号が枝どうしでぶつかっている（0013・0014 が重複）。取り込む順に本線が付け直し、STATUS に対応表を残す。
以後、支線は仮の番号 cNN（例 c01）で書き、本線が取り込むときに正式な番号を振る。
0016（→ 0022。common/seeds.py を入れる）、0018（→ 0024。runner.md を「(H, 7)、後処理の前」に直す）を行う。
0014（→ 0018。metrics）への監督の回答は、metrics の枝に 0014 への回答として入っている（→ 0019）。results.md に phi_correction の引数と、
trial_metrics が全列を返すことを書き足す。

## 求めること

本線は上を行い、報告する（0028）。
