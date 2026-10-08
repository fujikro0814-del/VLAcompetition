# 提出の手順（課題の 10/8 版）

課題の 10/8 版に合わせた提出の手順と、各設定の確かめ表。作業場所 `C:\PAI\recovery_vla`（開発用）と `C:\PAI\recovery-vla-panda`（提出用、書き出し先）、Python は `.venv\Scripts\python.exe`。
道具は 3 つ: `scripts\70_export_submission.py`（書き出し・YouTube の下書き）、`scripts\check_submission.py`（検査）、`scripts\push_submission.ps1`（履歴の作り直しと push）。

## 10/8 版で変わったこと

| 項目 | 10/8 版 | 前の版との違い・この手順での扱い |
|---|---|---|
| 動画 | 1〜3 分。2 倍速まで可。倍速は動画の中か YouTube の概要欄に明記 | 動画の中（各映像の場面の右上）と概要欄の両方に書く。`70_export_submission.py youtube` が長さと倍速を確かめる |
| 提出物 | omnicampus に 2 つ: (1) レポート PDF「説明資料_PAI最終課題_omnicampusアカウント名.pdf」、(2) GitHub のリポジトリの URL（必ず public） | Zip はなくなった。`70_export_submission.py package` は消さずに残し、既定では使わない（`--legacy` のときだけ動く） |
| YouTube | 自分の YouTube に上げて URL を出す。タイトル「PAI最終課題_omnicampusアカウント名」。限定公開が最低条件、全体公開は加点。「子ども向け」「年齢制限」は「いいえ」。インスタントプレミア公開はしない | 5 の設定の一覧で確かめる |
| 個人情報 | PDF・コード・動画に氏名・所属・メールアドレスを載せない（omnicampus のアカウント名は可） | `check_submission.py` の (9) で調べる。LICENSE の著作権者はアカウント名。コミットの作者・コミッタはアカウント名と GitHub の noreply アドレス |
| 日程 | 提出開始 10/19、締切 11/2 AM10:00 | |

## 0. 作者が提出の前にすること

1. **omnicampus のアカウント名を確かめる**。omnicampus にログインし、アカウント（プロフィール）の画面に出るアカウント名を写す。氏名の欄ではなくアカウント名を使う（どれがアカウント名か迷ったら、課題の案内の「omnicampusアカウント名」の説明に従う）。
2. **アカウント名をスクリプトに入れる**。`scripts\60_paper.py` と `scripts\62_video.py` の `ACCOUNT = "アカウント名"` を、同じアカウント名に直す（ファイル名・YouTube のタイトル・LICENSE の著作権者に入る。本文と画面の文字は変わらない）。仮の値のままだと `70_export_submission.py export`・`youtube` は止まり、LICENSE に仮の値が残っていれば `check_submission.py` が止める。
3. **説明資料と動画を作り直す**（ファイル名が変わるため）。
   ```
   .venv\Scripts\python.exe scripts\60_paper.py build
   .venv\Scripts\python.exe scripts\60_paper.py check
   .venv\Scripts\python.exe scripts\62_video.py build
   .venv\Scripts\python.exe scripts\62_video.py check
   ```
4. **GitHub の noreply アドレスを調べる**。形は `<ID>+<ユーザー名>@users.noreply.github.com`（ID は数字）。
   - GitHub の Settings → Emails を開き、「Keep my email addresses private」に印を付ける。その説明の文の中に、この形のアドレスが出る。同じ画面の「Block command line pushes that expose my email」にも印を付けておくと、本当のアドレスのコミットを GitHub が受け取らない。
   - または `gh api user --jq '.id, .login'` の id と login から `<id>+<login>@users.noreply.github.com` を組む。
5. **手元の git の設定を確かめる**。`git config --get user.name`・`git config --get user.email` に本名・本当のアドレスが入っていても、作り直したコミットには使わない（`push_submission.ps1` が -AuthorName・-AuthorEmail に固定する）。ただし `check_submission.py` は、この 2 つの値が提出物に出ていないかを探す。
6. （必要なら）**許す語を足す**。個人情報の検査で正当な語が当たるときは、`.local\pii_allow.txt`（git の外。1 行 1 語、`#` で始まる行は注釈）に書くか、`--pii-allow <語>` で渡す。アカウント名（`60_paper.py` の ACCOUNT）・GitHub の noreply・`noreply@anthropic.com`（共同作成者の行）・例示用のドメイン（example.com など）は最初から許している。

## 1. 書き出し

```
.venv\Scripts\python.exe scripts\70_export_submission.py export --dest C:\PAI\recovery-vla-panda
.venv\Scripts\python.exe scripts\70_export_submission.py scan --dest C:\PAI\recovery-vla-panda
```

- LICENSE（MIT）の著作権者は ACCOUNT（omnicampus のアカウント名）。本名を入れる欄はない。
- 履歴の作り直し（3 の (b)）でも同じ書き出しを行う。ここは下見として回す。

## 2. 検査

```
.venv\Scripts\python.exe scripts\check_submission.py --repo C:\PAI\recovery-vla-panda --worktree
.venv\Scripts\python.exe scripts\check_submission.py --repo C:\PAI\recovery-vla-panda --pii-only
```

- 個人情報の検査（(9)）の対象: 書き出し先の全テキストのファイル、説明資料の PDF の本文とメタデータ、動画の文字の一覧（`paper\build\video_texts.json`）、送るコミットのコミット文。
- 探すもの: メールアドレスの形、個人のパス（`C:\Users\<名前>`・`/home/<名前>`・`/Users/<名前>`）、個人・端末の名前（`check_submission.py` の PERSONAL）、git の設定の user.name・user.email の値（許すメールアドレス、たとえば GitHub の noreply の中に入っている所は数えない）。送るコミットの作者・コミッタの名前も調べ、ACCOUNT が決まっていれば名前がアカウント名でないコミットで止める。見つけたら exit 1 で、場所（ファイル:行、PDF のページ、動画の文字の番号）を出す。当たった文字は一部を伏せて出す。
- PDF は pypdf で読む。開発用の環境に入っていない（10/09 の時点で入っていない。.venv に pip はない）ので、提出の前に `.tools\uv\uv.exe pip install --system-certs --python .venv\Scripts\python.exe pypdf` で入れる。読めないときはその旨を出し、組み上げた HTML（`paper\build\paper.html`）を代わりに調べる。
- 動画の画面の文字は `video_texts.json`（62_video.py が画面に書いた文字の一覧）で調べる。映像の部分に写っている文字はこの一覧に入らないので、最後に動画を目でも通して見る。

## 3. 履歴の作り直し

`push_submission.ps1` の使い方 2。3 段に分け、(c) は作者が (b) の表示を確かめて明示したときだけ行う。

```
# (a) 下見（何も変えない）
powershell -ExecutionPolicy Bypass -File scripts\push_submission.ps1 -RebuildHistory
# (b) 作る（手元だけ。送らない）
powershell -ExecutionPolicy Bypass -File scripts\push_submission.ps1 -RebuildHistory -Execute -AuthorName <アカウント名> -AuthorEmail <ID>+<ユーザー名>@users.noreply.github.com
# (c) 送る
powershell -ExecutionPolicy Bypass -File scripts\push_submission.ps1 -RebuildHistory -ForcePush -ConfirmHash <(b) の短いハッシュ> -AuthorName <同じ> -AuthorEmail <同じ> [-Tag pai-final-v1]
```

- -AuthorName・-AuthorEmail は既定が空で、空なら (b)(c) は止まる。-AuthorEmail が noreply の形でなければ止まる。
- (b) はコミットの作者とコミッタを環境変数（GIT_AUTHOR_* と GIT_COMMITTER_*）でその値に固定し、作った後に `git log --format='%an %ae %cn %ce'` が全部「アカウント名 noreply アカウント名 noreply」かを確かめる。(c) も送る前に同じことを確かめる。違えば止まる。
- 作り直した後に提出用リポジトリでさらにコミットするときは、先に `git -C C:\PAI\recovery-vla-panda config user.name <アカウント名>`・`git -C C:\PAI\recovery-vla-panda config user.email <noreply>` を設定しておく（`check_submission.py` の (7) は、名前がアカウント名でないコミットで止める）。
- タグを付けるときは、注釈付きタグの作成者（tagger）も公開されるので、同じ値で付ける: `git -c user.name=<アカウント名> -c user.email=<noreply> tag -a pai-final-v1 -m "提出版"`。(c) はタグの作成者が違えば止まる。

## 4. public の確認（スクリプトは GitHub の設定を変えない）

1. GitHub でリポジトリを開き、名前の横の札が「Public」であることを見る（「Private」なら Settings → General → 一番下の Danger Zone →「Change repository visibility」で Public にする。作者が自分で行う）。
2. または `gh repo view <ユーザー名>/recovery-vla-panda --json visibility` が `"PUBLIC"` を返すことを見る。
3. ログインしていないブラウザ（シークレットウィンドウ）で URL を開き、中身が見えることを確かめる。
4. 送った版を見る: 履歴が 1 コミットだけで、作者がアカウント名になっていること。

## 5. YouTube に上げる

```
.venv\Scripts\python.exe scripts\70_export_submission.py youtube --github-url https://github.com/<ユーザー名>/recovery-vla-panda
```

- 動画 `paper\build\動画_PAI最終課題_<アカウント名>.mp4` の長さ（ffprobe、なければ OpenCV で読む）が 1〜3 分で、場面の表の合計と合うこと、`configs\demo\video_s3.yaml` の映像の場面の倍速 (t1 − t0) / dur_s がすべて 2 倍以下であることを確かめる。外れたら止まる。
- タイトルと概要欄の下書きを、画面と `outputs\submission\youtube_title.txt`・`youtube_description.txt`・`youtube.json` に出す。概要欄には、倍速の明記（どの場面が何倍速か）、1 文のアイデア、GitHub の URL の欄、説明資料の要点 3 行が入る。数字は説明資料と同じ値の一覧（`paper\build\values.json`）から差し込む（概要欄に < と > は使えないので、値の中の < > は全角の ＜ ＞ にする。例: p ＜ 0.001）。`60_paper.py` の FORBIDDEN の語に当たれば止まる。

上げるときの設定:

| 設定 | 値 | 済 |
|---|---|---|
| ファイル | `paper\build\動画_PAI最終課題_<アカウント名>.mp4` | |
| タイトル | `PAI最終課題_<アカウント名>`（`youtube_title.txt`） | |
| 説明（概要欄） | `youtube_description.txt` を貼る。GitHub の URL の欄が埋まっていること | |
| 視聴者 | 「いいえ、子ども向けではありません」 | |
| 年齢制限（詳細） | 「いいえ、18 歳以上の視聴者のみに制限しません」 | |
| 公開設定 | 限定公開以上（最低条件）。全体公開なら加点。非公開は不可 | |
| プレミア公開 | しない（「インスタントプレミア公開」の印を付けない） | |
| 上げた後 | ログインしていないブラウザで URL を開き、再生できること・タイトル・概要欄を見る | |

## 6. omnicampus に出すもの

| 出すもの | 中身 | 済 |
|---|---|---|
| (1) レポート PDF | `paper\build\説明資料_PAI最終課題_<アカウント名>.pdf`（ファイル名がこの形であること） | |
| (2) GitHub のリポジトリの URL | `https://github.com/<ユーザー名>/recovery-vla-panda`（public であること。4 で確かめる） | |
| YouTube の URL | 5 で上げた動画の URL。出す場所は課題の案内に従う（omnicampus の提出欄にあればそこへ） | |

- 提出開始は 10/19、締切は 11/2 AM10:00。

## 確かめ表

| 項目 | 決まり | 確かめ方 | 済 |
|---|---|---|---|
| アカウント名 | 2 つのスクリプトの ACCOUNT が同じで、仮の値でない | `70_export_submission.py export`・`youtube` が止まらない | |
| 動画の長さ | 1〜3 分 | `70_export_submission.py youtube` | |
| 倍速 | すべての場面が 2 倍以下で、動画の中か概要欄に明記 | `youtube`（概要欄の「■ 倍速」）と `62_video.py check` | |
| 説明資料 | 照合が合格、ファイル名「説明資料_PAI最終課題_<アカウント名>.pdf」 | `60_paper.py check` | |
| 使わない語 | 説明資料・動画・概要欄・提出用リポジトリに 0 | `60_paper.py check`・`62_video.py check`・`youtube`・`check_submission.py` | |
| 個人情報 | PDF・コード・動画に氏名・所属・メールアドレスがない | `check_submission.py`（(9)、`--pii-only` でも）。最後に目でも見る | |
| LICENSE | 著作権者がアカウント名 | `check_submission.py`（仮の値なら止まる） | |
| コミットの作者・コミッタ | アカウント名と GitHub の noreply アドレスだけ | `push_submission.ps1` (b)(c) の確かめと、`check_submission.py` の (7) | |
| タグの作成者 | 同上（注釈付きタグのとき） | `push_submission.ps1` (c) | |
| リポジトリ | public | 4 | |
| YouTube | タイトル・限定公開以上・子ども向けいいえ・年齢制限いいえ・プレミア公開しない | 5 の表 | |
| omnicampus | PDF と GitHub の URL の 2 つ（Zip はなし） | 6 の表 | |
| 日程 | 10/19〜11/2 AM10:00 | | |

## 既知の食い違い（作者の判断を待つもの）

- README の原稿 `paper\README_submission.md` の「動画と説明資料」の段落は「omnicampus に提出した Zip に、動画（MP4）と説明資料（PDF）が入っている」と書いており、10/8 版（Zip なし、動画は YouTube）と合わない。原稿は説明資料と同じ扱いなので、この手順では変えていない。直すなら、動画は YouTube（概要欄に倍速）・説明資料は omnicampus に出した PDF、と書き換える。
