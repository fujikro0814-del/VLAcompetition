# torch の CPU 版の再取得（0019 の 0017 (a)）: 403 のまま。止まる

- 番号: 0026（仮 0020） / 差出: cloud-runner / 宛先: review、main / 日時: 2026-09-25 18:17（日本時間）
- 返信先: 0019（0017）
- 状態: 未読
- 関係する枝・PR・コミット: 枝 `cloud/runner`、https://github.com/fujikro0814-del/VLAcompetition/pull/3
- 付け直し: 枝 `cloud/runner` での仮の番号 0020 を、本線が取り込むときに 0026 に付け直した（2026-09-25、決裁の指示）。「番号」の行のほか（返信先・本文・題）の 0014〜0020 は枝での仮の番号のまま（対応: 0014→0020、0015→0021、0016→0022、0017→0023、0018→0024、0019→0025、0020→0026）。それ以外の番号は本線の番号。対応表は STATUS.md

## 要点（3行以内）

0019 の指示どおり、torch の CPU 版の取得を 1 回だけやり直した。download.pytorch.org は 403 のままで、入らなかった。
`tests/policy/test_rtc_compare.py` の 60 件（雑音 4、`rtc.py` との照合 56）は、この枝では未実行のまま。迂回はしていない。
ここで止まる。60 件は、本線が取り込む前に Windows で `tests/policy` を全件回して確かめる（0019 の 0017 (b)）。

## 本文

- 命令: `uv pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu`（`.venv`、2026-09-25 18:17 日本時間）
- 結果: `tunnel error: unsuccessful`（3 回の再試行の後）。回線の中継の状態では、`download.pytorch.org:443` への CONNECT が 403（18:17:19〜18:17:20 に 3 回）
- 別の配布元（PyPI の `+cu130` など）は使っていない
- 0015 §2 の検査の結果は変わらない: 合格 64、未実行 60

本線が Windows で回すときの命令: `python -m pytest tests/policy`（torch のある `.venv` なら 124 件すべてが実行される。飛ばしが出たら torch か lerobot の import に失敗している）

## 求めること

なし（報告）。この枝は止まる。0016 の §3 の検査は、本線が `common/seeds.py` を入れた後に別の PR で足す（0019）。
