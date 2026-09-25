# 0012 への判断（クラウド環境の疎通確認）

- 番号: 0013 / 差出: human / 宛先: main / 日時: 2026-09-25 17:58（日本時間。チャットでの指示を cloud-setup-check が書き写した）
- 返信先: 0012
- 状態: 未読
- 関係する枝・PR・コミット: 枝 `cloud/setup-check`、https://github.com/fujikro0814-del/VLAcompetition/pull/1

## 要点（3行以内）

PR「cloud/setup-check」を取り込む。torch は人がクラウドの許可先に download.pytorch.org を足して解決する（0012 の (c)）。
検査に pytest の目印（render・windows・needs_outputs・torch）を付け、支線の既定の実行から render・windows・needs_outputs を除く。
場面のバイト一致の検査は windows の目印を付けて本線だけで回し、OS によらない検査を足す。物理の速さは STATUS に残す。

## 本文

（決裁の回答をそのまま書き写す。支線は 1〜5 のどれも行っていない。行うのは本線）

1. PR「cloud/setup-check」を取り込む（変更は報告と scripts/cloud/bench_physics.py だけ）。
2. torch：支線の環境は、人がクラウドの環境の許可先に download.pytorch.org を足して解決する（0012 の (c)）。
   固定一覧と env/requirements-cloud.txt の手順は変えない。
3. 検査に pytest の目印を付ける：render（描画の文脈が要る）、windows（Windows 専用）、
   needs_outputs（Git 管理外の outputs/ が要る）、torch（torch が要る）。pyproject.toml に登録し、
   支線の既定の実行を -m "not render and not windows and not needs_outputs" とする。これを docs/interfaces/README.md の
   「支線の検査」に書き足す。
4. test_scene_compiles_to_the_same_model_as_the_source：バイト一致は windows の目印を付けて本線だけで回す。
   別に、OS によらない検査（nq・nv・nbody・ngeom の一致と、主な配列の許容誤差内の一致）を足す。
5. 物理の速さ（制御器ありで実時間の 5.9 倍、4 CPU）は cloud/expert の計画に使うので STATUS に残す。

## 求めること

本線は 1・3・4・5 を行い、終わったら報告する。2 は決裁（人）が行う。
