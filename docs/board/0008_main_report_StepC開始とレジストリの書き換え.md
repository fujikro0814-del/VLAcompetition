# Step C 開始と、uv によるレジストリの書き換え

- 番号: 0008 / 差出: main / 宛先: human、review / 日時: 2026-09-25 18:10（日本時間）
- 返信先: 0005、0007
- 状態: 完了
- 関係する枝・PR・コミット: main（このファイルを含む最初のコミット）

## 要点（3行以内）

Step C を始めた。道具（uv・MinGit・gh）と実行環境（Python 3.12.14、105 パッケージ）を <ROOT> の下に作った。
uv が Python を HKCU のレジストリに登録し、学生用の Python を指していた項目を書き換えた。人の判断で学生用の値に戻した。
以後の `uv python install` は `--no-bin --no-registry` で行う（`env/setup_env.ps1` を修正済み）。

## 本文

**道具**（`env/setup_env.ps1 -Stage tools`）: 配布元の SHA-256 と一致したものだけを展開した。

| 道具 | 版 | 照合 |
|---|---|---|
| uv | 0.12.15 | 配布元の `.sha256` |
| MinGit | 2.55.0.windows.5 | GitHub の API の資産の digest |
| gh | 2.101.0 | 配布元の `checksums.txt` |

**実行環境**（`-Stage python・torch・lock・extras・check`）: torch 2.11.0+cu126・torchvision 0.26.0+cu126、固定一覧 87 件、追加 18 件（scipy 1.18.1、matplotlib 3.11.2、anthropic 1.8.0 とその依存）。`uv pip check` 合格、入った一覧は `env/requirements-lock.txt` と一致（105 件）。

- 訂正: 提案書 §3.2 の 6 の「全 95 パッケージ」は誤り。付録 A も学生用の環境の dist-info も **87 件**で、固定一覧は dist-info の一覧と完全に一致した

**レジストリの書き換え（手違い）**

- `uv python install 3.12.14` が、既定の動作で `HKCU\Software\Python\Astral\CPython3.12.14` に <ROOT> の Python を登録した（PEP 514）。提案書 §3.1 の「レジストリは変えない」に反する
- 学生用の README も同じ版を登録つきで入れているので、学生用の Python を指していた項目を上書きした可能性が高い。以前の値の記録はない
- 人の判断（チャット）で、`InstallPath` の 3 つの値（既定・`ExecutablePath`・`WindowedExecutablePath`）を学生用の Python の場所に戻した。C:\VLA には書き込んでいない。ほかの値（配布元の URL など）は uv が書いたまま
- 学生用の .venv はこの項目を使わない（`pyvenv.cfg` の `home` で Python の場所を持つ）ので、学生用の環境の動作は変わっていない
- あわせて uv は `~\.local\bin\python3.12.exe` を置こうとしたが、同じ名前のファイルがすでにあったので置かなかった（そのファイルには触れていない）
- 対策: `env/setup_env.ps1` の Python の段を `--no-bin --no-registry` に直した

**Git**: <ROOT> を Git のリポジトリにし、公開の VLAcompetition を `origin` にした。リモートの `Initial commit` の上に積む。作者はこのリポジトリの設定だけ（0004）。`core.autocrlf=false`（複写したファイルのバイトを変えないため）、`http.sslBackend=schannel`。

**push の前の検査**: `scripts/check_before_push.ps1`（`scripts/push.ps1` から必ず呼ぶ）。検査用の一時リポジトリで、きれいな文書（「`sk-ant-` など」「`hf_home`」の記述）は通り、鍵・学内のプロキシ・管理外の置き場所・証明書は、索引にある場合もコミットの履歴だけにある場合も止まることを確かめた。

## 求めること

なし（報告のみ）。
