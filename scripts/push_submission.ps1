# 提出用リポジトリ（recovery-vla-panda）の push の入口（束 5 の 1）。検査（scripts\check_submission.py）に通ったときだけ push する。
# 開発用のリポジトリの push.ps1 とは別。目標書・G1 の検査はしない（提出用リポジトリには目標書がない）。
# 検査の中身: 作業ツリーがきれい、鍵・証明書・手元の文字列・個人の名前が送るコミットすべてにない、60_paper.py の FORBIDDEN の語が
# HEAD の全ファイルにない、入れないもの（段階 4・作業記録・outputs など）のパスがない、100 MB を超えるファイルがない、JSON が読める。
#
# 使い方 1（ふだんの push。履歴をそのまま積む）:
#   powershell -ExecutionPolicy Bypass -File scripts\push_submission.ps1 [-Repo C:\PAI\recovery-vla-panda] [-Branch main] [-Tag <タグ>] [-DryRun]
#   -Tag   : 枝を送った後、そのタグも送る。タグは HEAD（検査した版）を指していること。同じ名前のタグが GitHub に別の版であれば送らない
#   -DryRun: 検査と git push --dry-run だけ（何も送らない）
#
# 使い方 2（履歴の作り直し。10/08 の作者の決定。既定では何も変えない）:
#   GitHub の origin/main の履歴（最初のコミット 4b5ed59 に以前の道具の場所「C:\…VLA」の文字が残る）と、まだ送っていない途中の
#   コミット（今の決まりで使わない語を含む版）を公開しないため、提出の版だけの 1 コミットの履歴に作り直す。3 段に分けて行う。
#   (a) 下見（何も変えない）:
#         ... push_submission.ps1 -RebuildHistory
#       消える履歴（origin/main の全コミットと未送信のコミット）の一覧と、作業ツリーの検査の結果を出すだけ
#   (b) 作る（手元だけ。送らない）:
#         ... push_submission.ps1 -RebuildHistory -Execute [-Message "<コミット文>"]
#       書き出し（70_export_submission.py export）→ 作業ツリーの検査 → 孤立した新しい枝 $CleanBranch に 1 コミット → HEAD の検査
#       → 確認の表示（新しいコミットの短いハッシュ・ファイル数・消える履歴）。元の main の枝は手元にそのまま残る
#   (c) 送る（作者が (b) の表示を確かめた後だけ）:
#         ... push_submission.ps1 -RebuildHistory -ForcePush -ConfirmHash <(b) で出た短いハッシュ> [-Tag pai-final-v1]
#       今の枝が $CleanBranch で、HEAD が -ConfirmHash と同じで、検査に通り、origin/main が (a)(b) のときから動いていないときだけ、
#       origin の main を強制で置き換える（--force-with-lease）。送った後、手元の古い main は $BackupBranch に名前を変えて残し
#       （送らない）、$CleanBranch を main にする
#   実行はメインの担当と作者が最終の段階で行う。-ForcePush は作者の明示の確認なしには使わない。
param(
    [string]$Repo = 'C:\PAI\recovery-vla-panda',
    [string]$Branch = 'main',
    [string]$Tag = '',
    [switch]$DryRun,
    [switch]$RebuildHistory,
    [switch]$Execute,
    [switch]$ForcePush,
    [string]$ConfirmHash = '',
    [string]$Message = 'PAI 最終課題 提出版: 失敗からの復帰を学習する VLA（MuJoCo 上の Franka Panda）のコード・設定・テスト・説明資料と動画の原稿・評価結果の表'
)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$env:PYTHONIOENCODING = 'utf-8'
$Root = Split-Path -Parent $PSScriptRoot
$git = Join-Path $Root '.tools\git\cmd\git.exe'
$py = Join-Path $Root '.venv\Scripts\python.exe'
$ExpectedRemote = 'github.com/fujikro0814-del/recovery-vla-panda'
$CleanBranch = 'submission-clean'
$BackupBranch = 'pre-rebuild-main'

# 送り先を取り違えない: 名前・開発用のリポジトリでないこと・origin の URL
if ((Split-Path -Leaf $Repo) -ne 'recovery-vla-panda') { throw "提出用リポジトリではない: $Repo" }
if ((Resolve-Path $Repo).Path -eq (Resolve-Path $Root).Path) { throw '開発用のリポジトリには使わない（scripts\push.ps1 を使う）' }
$url = & $git -C $Repo remote get-url origin
if ($LASTEXITCODE -ne 0 -or $url -notmatch [regex]::Escape($ExpectedRemote)) { throw "origin が提出用の GitHub リポジトリではない: $url" }
$cur = & $git -C $Repo rev-parse --abbrev-ref HEAD

& $git -C $Repo fetch origin
if ($LASTEXITCODE -ne 0) { throw 'fetch に失敗' }

$out = Join-Path $Root 'outputs\submission'
New-Item -ItemType Directory -Force $out | Out-Null

function Show-LostHistory {
    # 作り直しで GitHub から消える履歴（origin/main の全コミット）と、送らずに捨てる未送信のコミット（手元の古い main には残る）
    Write-Host ''
    Write-Host '=== 作り直しで GitHub から消える履歴（origin/main の全コミット） ===' -ForegroundColor Yellow
    & $git -C $Repo log --oneline "origin/$Branch"
    Write-Host '（最初のコミット 4b5ed59 の docs/interfaces/README.md に、以前の道具の場所の文字が 1 か所ある）'
    Write-Host ''
    Write-Host "=== 送らずに捨てる未送信のコミット（$Branch にあって origin/$Branch にないもの。手元の $BackupBranch には残る） ===" -ForegroundColor Yellow
    & $git -C $Repo log --oneline "origin/$Branch..$Branch"
    Write-Host ''
    Write-Host 'GitHub のタグ（作り直しても消えない。古い版を指すタグがあれば、作者が別に消すか決める）:' -ForegroundColor Yellow
    & $git -C $Repo ls-remote --tags origin
}

function Invoke-Check([string[]]$extra) {
    & $py (Join-Path $PSScriptRoot 'check_submission.py') --repo $Repo @extra | Out-Host      # 出力は画面へ（戻り値に混ぜない）
    return [int]$LASTEXITCODE
}

function Push-TagIfAny([string]$headSha) {
    if ($Tag -eq '') { return }
    $tc = & $git -C $Repo rev-parse --verify --quiet "$Tag^{commit}"
    if (-not $tc) { Write-Host "タグ $Tag がない（先に git tag -a $Tag -m ... で HEAD に付ける）" -ForegroundColor Red; exit 1 }
    if ($tc -ne $headSha) { Write-Host "タグ $Tag が HEAD（検査した版）を指していない" -ForegroundColor Red; exit 1 }
    $remoteTag = & $git -C $Repo ls-remote --tags origin "refs/tags/$Tag^{}" "refs/tags/$Tag"
    if ($remoteTag) {
        $remoteCommit = (($remoteTag | Select-Object -Last 1) -split '\s+')[0]
        if ($remoteCommit -ne $headSha) { Write-Host "GitHub に同じ名前のタグ $Tag が別の版である。名前を変える" -ForegroundColor Red; exit 1 }
    }
}

if ($RebuildHistory) {
    $originSha = (& $git -C $Repo rev-parse "origin/$Branch").Trim()

    if ($ForcePush) {
        # (c) 送る: 作者が (b) の表示を確かめた後だけ
        if ($Execute) { throw '-Execute と -ForcePush は同時に使わない（(b) で作って表示を確かめてから (c)）' }
        if ($cur -ne $CleanBranch) { throw "今の枝 $cur が $CleanBranch ではない（先に -RebuildHistory -Execute）" }
        $head = (& $git -C $Repo rev-parse HEAD).Trim()
        $short = (& $git -C $Repo rev-parse --short HEAD).Trim()
        if ($ConfirmHash -eq '' -or -not $head.StartsWith($ConfirmHash.Trim())) {
            throw "-ConfirmHash（$ConfirmHash）が HEAD $short と違う。(b) の表示を確かめ、その短いハッシュを渡す"
        }
        $count = (& $git -C $Repo rev-list --count HEAD).Trim()
        if ($count -ne '1') { throw "$CleanBranch のコミットが 1 つではない（$count）" }
        if ((Invoke-Check @('--upstream', "origin/$Branch", '--json', (Join-Path $out 'check_submission.json'))) -ne 0) {
            Write-Host '検査に通らないので送らない。' -ForegroundColor Red; exit 1
        }
        Push-TagIfAny $head
        Show-LostHistory
        Write-Host ''
        Write-Host "origin の $Branch を $short（1 コミット）で強制的に置き換える（--force-with-lease、origin/$Branch = $($originSha.Substring(0,7)) のときだけ）" -ForegroundColor Yellow
        $pushArgs = @('push', "--force-with-lease=refs/heads/${Branch}:$originSha")
        if ($DryRun) { $pushArgs += '--dry-run' }
        & $git -C $Repo @pushArgs origin "HEAD:refs/heads/$Branch"
        if ($LASTEXITCODE -ne 0) { Write-Host '強制 push に失敗（origin/main が動いていれば、(a) からやり直す）' -ForegroundColor Red; exit $LASTEXITCODE }
        if ($DryRun) { Write-Host '-DryRun なので送っていない。'; exit 0 }
        # 手元の枝の名前を整える: 古い main は送らずに残し、作り直した枝を main にする
        & $git -C $Repo branch -m $Branch $BackupBranch
        if ($LASTEXITCODE -ne 0) { throw "古い $Branch の名前を $BackupBranch に変えられない（送るのは済んでいる）" }
        & $git -C $Repo branch -m $CleanBranch $Branch
        & $git -C $Repo branch --set-upstream-to "origin/$Branch" $Branch
        if ($Tag -ne '') {
            & $git -C $Repo push origin "refs/tags/$Tag"
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
            Write-Host "提出の URL: https://$ExpectedRemote/tree/$Tag"
        }
        Write-Host "送った。手元の古い履歴は枝 $BackupBranch に残っている（送らない。要らなくなったら作者が消す）。" -ForegroundColor Green
        exit 0
    }

    if (-not $Execute) {
        # (a) 下見: 何も変えない
        Write-Host '履歴の作り直しの下見（何も変えない）。' -ForegroundColor Cyan
        Show-LostHistory
        Write-Host ''
        Write-Host '作業ツリーの検査（書き出した中身が提出の版になる）:' -ForegroundColor Cyan
        $rc = Invoke-Check @('--worktree')
        Write-Host ''
        Write-Host "次の段: -RebuildHistory -Execute で、手元に孤立した枝 $CleanBranch を作って 1 コミットにする（送らない）。"
        exit $rc
    }

    # (b) 作る: 手元だけ。送らない
    if ($cur -ne $Branch) { throw "今の枝 $cur が $Branch ではない（作り直しは $Branch の作業ツリーから始める）" }
    $exists = & $git -C $Repo rev-parse --verify --quiet "refs/heads/$CleanBranch"
    if ($exists) { throw "枝 $CleanBranch がすでにある（前の試しの残り。中身を確かめてから git branch -D $CleanBranch）" }
    & $py (Join-Path $PSScriptRoot '70_export_submission.py') export --dest $Repo
    if ($LASTEXITCODE -ne 0) { throw '書き出しに失敗' }
    if ((Invoke-Check @('--worktree')) -ne 0) { Write-Host '作業ツリーの検査に通らないので作らない。' -ForegroundColor Red; exit 1 }
    & $git -C $Repo checkout --orphan $CleanBranch
    if ($LASTEXITCODE -ne 0) { throw '孤立した枝を作れない' }
    & $git -C $Repo add -A
    & $git -C $Repo commit -q -m $Message
    if ($LASTEXITCODE -ne 0) { throw 'コミットに失敗' }
    $short = (& $git -C $Repo rev-parse --short HEAD).Trim()
    $nfiles = (& $git -C $Repo ls-files | Measure-Object -Line).Lines
    $rc = Invoke-Check @('--upstream', "origin/$Branch", '--json', (Join-Path $out 'check_submission_rebuild.json'))
    Show-LostHistory
    Write-Host ''
    Write-Host '=== 作り直した履歴（まだ送っていない） ===' -ForegroundColor Cyan
    & $git -C $Repo log --format='%h %an <%ae> %s' -n 3
    Write-Host "枝 $CleanBranch、コミット 1 つ（$short）、ファイル $nfiles。検査: $(if ($rc -eq 0) { '合格' } else { '不合格' })"
    Write-Host ''
    Write-Host '作者が上の表示（消える履歴と新しいコミット）を確かめて、送ってよいと明示したときだけ、次を実行する:' -ForegroundColor Yellow
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\push_submission.ps1 -RebuildHistory -ForcePush -ConfirmHash $short [-Tag pai-final-v1] [-DryRun]"
    Write-Host "やめるとき: git checkout $Branch のあと git branch -D $CleanBranch（$Branch の履歴は変わっていない）"
    exit $rc
}

# ---- 使い方 1: ふだんの push
if ($cur -ne $Branch) { throw "今の枝 $cur が $Branch ではない" }

# 検査（不合格なら送らない）
& $py (Join-Path $PSScriptRoot 'check_submission.py') --repo $Repo --upstream "origin/$Branch" --json (Join-Path $out 'check_submission.json')
if ($LASTEXITCODE -ne 0) { Write-Host 'push を中止した。' -ForegroundColor Red; exit 1 }

# タグ: HEAD（検査した版）を指していること。GitHub に同じ名前で別の版があれば送らない（上書きしない）
$headSha = (& $git -C $Repo rev-parse HEAD).Trim()
Push-TagIfAny $headSha

$pushArgs = @('push')
if ($DryRun) { $pushArgs += '--dry-run' }
& $git -C $Repo @pushArgs origin $Branch
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($Tag -ne '') {
    & $git -C $Repo @pushArgs origin "refs/tags/$Tag"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    if (-not $DryRun) { Write-Host "提出の URL: https://$ExpectedRemote/tree/$Tag" }
}
exit 0
