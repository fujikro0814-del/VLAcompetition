# push の前の検査（B_提案書 §3.2 の 10、掲示板 0002・0004）。公開リポジトリなので、push のたびに必ず通す。
# 対象: 索引（ステージ）と、まだリモートにないコミットのすべて。どれか 1 つでも当たれば exit 1（push しない）。
#  (a) Git 管理外の置き場所の下のファイル
#  (b) 鍵の形をした文字列  (c) 証明書  (d) .local\push_check_patterns.txt の文字列（学内の回線の設定・共有フォルダ）
#  (e) docs/目標書.md が goals-v* タグの版と一致しない（0106）  (f) G1 の実機境界の検査・目標書の版の照合が不合格（0106）
# 当たった箇所は「ファイル名と件数」だけを表示し、中身は表示しない（鍵をログに残さないため）。
# 使い方: powershell -ExecutionPolicy Bypass -File scripts\check_before_push.ps1 [-Repo <パス>] [-Upstream origin/main] [-PatternFile <パス>]
param(
    [string]$Repo = (Split-Path -Parent $PSScriptRoot),
    [string]$Upstream = 'origin/main',
    [string]$PatternFile = ''
)
$ErrorActionPreference = 'Stop'
if (-not $PatternFile) { $PatternFile = Join-Path $Repo '.local\push_check_patterns.txt' }
$git = Join-Path (Split-Path -Parent $PSScriptRoot) '.tools\git\cmd\git.exe'
if (-not (Test-Path $git)) { $git = 'git' }

$IgnoredDirs = @('.tools/', '.python/', '.venv/', '.cache/', '.local/', 'models/', 'outputs/', 'docs/local/')
# 目印の文字列は、この検査自身に当たらないように分けて書く
$Dashes = '-' * 5
$KeyPatterns = [ordered]@{
    'Anthropic の鍵'   = 'sk-ant-[A-Za-z0-9_-]{20,}'
    'GitHub の鍵'      = 'gh[pousr]_[A-Za-z0-9]{20,}'
    'GitHub の鍵(PAT)' = 'github_pat_[A-Za-z0-9_]{20,}'
    'Hugging Face の鍵' = 'hf_[A-Za-z0-9]{30,}'
    '秘密鍵'           = "${Dashes}BEGIN [A-Z ]*PRIVATE KEY${Dashes}"
    '証明書'           = "${Dashes}BEGIN CERTIFICATE${Dashes}"
}
$CertExt = '\.(pem|crt|cer|der|p12|pfx|key)$'

$findings = New-Object System.Collections.Generic.List[string]

function Invoke-Git([string[]]$ArgList) {
    $out = & $git -C $Repo -c core.quotepath=false @ArgList
    return , @($out)
}

# 検査の対象: 索引（--cached）と、上流にないコミット
$targets = @(@{ Name = '索引'; Grep = @('--cached'); Files = @('ls-files') })
$hasUpstream = $true
& $git -C $Repo rev-parse --verify --quiet $Upstream | Out-Null
if ($LASTEXITCODE -ne 0) { $hasUpstream = $false }
$range = if ($hasUpstream) { "$Upstream..HEAD" } else { 'HEAD' }
& $git -C $Repo rev-parse --verify --quiet HEAD | Out-Null
if ($LASTEXITCODE -eq 0) {
    foreach ($c in (Invoke-Git @('rev-list', $range))) {
        if ($c) { $targets += @{ Name = "コミット $($c.Substring(0, 7))"; Grep = @($c); Files = @('ls-tree', '-r', '--name-only', $c) } }
    }
}

foreach ($t in $targets) {
    $files = Invoke-Git $t.Files
    # (a) Git 管理外の置き場所
    foreach ($f in $files) {
        foreach ($d in $IgnoredDirs) { if ($f.StartsWith($d)) { $findings.Add("$($t.Name): 管理外の置き場所のファイル: $f") } }
        if ($f -match $CertExt) { $findings.Add("$($t.Name): 証明書・鍵の拡張子: $f") }
    }
    # (b)(c) 鍵・証明書の形
    foreach ($k in $KeyPatterns.Keys) {
        $hits = Invoke-Git (@('grep', '-I', '-c', '-E', '-e', $KeyPatterns[$k]) + $t.Grep)
        foreach ($h in $hits) { if ($h) { $findings.Add("$($t.Name): ${k}: $h 件") } }
    }
    # (d) 手元にだけ置いた文字列
    if (-not (Test-Path $PatternFile)) {
        $findings.Add("照合用の文字列のファイルがない: $PatternFile（作ってからやり直す）")
    } else {
        $pats = Get-Content $PatternFile -Encoding UTF8 | Where-Object { $_ -and -not $_.StartsWith('#') }
        $i = 0
        foreach ($p in $pats) {
            $i++
            $hits = Invoke-Git (@('grep', '-I', '-c', '-F', '-e', $p) + $t.Grep)
            foreach ($h in $hits) { if ($h) { $findings.Add("$($t.Name): 手元の照合文字列 $i 行目: $h 件") } }
        }
    }
}

# (e) 目標書（0106）: 送るコミットと索引の docs/目標書.md が、goals-v* タグのどれかの版と同じ。HEAD はいちばん新しい版と同じ。
#     版の登録（タグ・変更履歴の行）は recovla.common.goals でも確かめる
$GoalsPath = 'docs/目標書.md'
$goalTags = @(Invoke-Git @('tag', '--list', 'goals-v*') | ForEach-Object { $_ } | Where-Object { $_ -match '^goals-v\d+$' } |
    Sort-Object { [int]($_ -replace '^goals-v', '') })
if ($goalTags.Count -eq 0) {
    $findings.Add('目標書: goals-v* のタグがない')
} else {
    $tagBlobs = @{}
    foreach ($t in $goalTags) { $tagBlobs[(& $git -C $Repo rev-parse "${t}:$GoalsPath")] = $t }
    $latestBlob = & $git -C $Repo rev-parse "$($goalTags[-1]):$GoalsPath"
    foreach ($t in $targets) {
        $spec = if ($t.Grep[0] -eq '--cached') { ":$GoalsPath" } else { "$($t.Grep[0]):$GoalsPath" }
        $blob = & $git -C $Repo rev-parse --verify --quiet $spec
        if (-not $blob) { $findings.Add("$($t.Name): 目標書がない"); continue }
        if (-not $tagBlobs.ContainsKey($blob)) { $findings.Add("$($t.Name): 目標書が goals-v* のどの版とも違う") }
    }
    $headBlob = & $git -C $Repo rev-parse --verify --quiet "HEAD:$GoalsPath"
    if ($headBlob -ne $latestBlob) { $findings.Add("HEAD の目標書が最新の版（$($goalTags[-1])）と違う") }
}

# (f) G1 の実機境界（0106）: 送る HEAD の実行系のコードに、真値に触れる新しい書き方がない（scripts/check_g1_boundary.py）
#     と、目標書の版の登録（recovla.common.goals）
$ToolRoot = Split-Path -Parent $PSScriptRoot
$py = Join-Path $ToolRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) {
    $findings.Add("Python がない: $py")
} else {
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $g1 = & $py (Join-Path $PSScriptRoot 'check_g1_boundary.py') --rev HEAD --root $Repo
    if ($LASTEXITCODE -ne 0) { $findings.Add('G1 の境界の検査が不合格'); $g1 | ForEach-Object { Write-Host "  $_" } } else { Write-Host ($g1 -join ' ') }
    $gv = & $py -m recovla.common.goals $Repo
    if ($LASTEXITCODE -ne 0) { $findings.Add('目標書の版の照合が不合格'); Write-Host "  $gv" }
}

if ($findings.Count -gt 0) {
    Write-Host "push 前の検査: 不合格（$($findings.Count) 件）。push しない。" -ForegroundColor Red
    $findings | Sort-Object -Unique | ForEach-Object { Write-Host "  $_" }
    exit 1
}
Write-Host "push 前の検査: 合格（対象: $(($targets | ForEach-Object { $_.Name }) -join '、')）"
exit 0
