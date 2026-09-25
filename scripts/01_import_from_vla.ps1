# 流用元（C:\VLA）から <ROOT> への取り込み（B_提案書 §2）。C:\VLA には書き込まない。
#   code   : 流用元のコミットから git archive で取り出し、承認された一覧の複写先へ「そのまま」置く（最初の取り込みのコミット用）。
#            そのまま置けない 2 か所だけ、決まった置き換えをしてバイト差を provenance に書く。
#   assets : コード以外（HF のスナップショット、予備実験のチェックポイント、raw 1 本、評価の json）を
#            Copy-Item で複写し、複写の前後で SHA-256 を比べる（Git 管理外の models\・outputs\legacy\）。
# 使い方: powershell -ExecutionPolicy Bypass -File scripts\01_import_from_vla.ps1 -Part code|assets
# 記録: docs\provenance.csv（1 ファイル 1 行。code の行を先に、assets の行を後に書く）
param(
    [Parameter(Mandatory = $true)][ValidateSet('code', 'assets')][string]$Part
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Git = Join-Path $Root '.tools\git\cmd\git.exe'
$SrcRepo = 'C:\VLA\pytools\panda_teleop'
$Commit = 'dbb2c3b4db64c7e6db4a997e087549e67cd776c6'
$VlaEnv = 'C:\VLA\02_環境'
$Provenance = Join-Path $Root 'docs\provenance.csv'
$Utf8 = New-Object Text.UTF8Encoding $false

function Get-Sha([string]$Path) { (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLower() }

function Invoke-SrcGit([string[]]$ArgList) {
    # 流用元のリポジトリは読むだけ（索引を書き換えない、safe.directory は設定ファイルに書かない）
    $out = & $Git --no-optional-locks -c safe.directory=C:/VLA/pytools/panda_teleop -c core.quotepath=false -C $SrcRepo @ArgList
    if ($LASTEXITCODE -ne 0) { throw "git 失敗: $($ArgList -join ' ')" }
    return , @($out)
}

$Rows = New-Object System.Collections.Generic.List[object]
function Add-Row($src, $commit, $blob, $srcSha, $dst, $dstSha, $treatment, $note) {
    $Rows.Add([pscustomobject]@{
            source = $src; commit = $commit; blob = $blob; source_sha256 = $srcSha
            dest = $dst; dest_sha256_at_import = $dstSha; treatment = $treatment; note = $note
        })
}

function Write-Provenance([string]$Kind) {
    # 同じ Part の行を置き換え、他の Part の行は残す
    $keep = @()
    if (Test-Path $Provenance) {
        $keep = Import-Csv -LiteralPath $Provenance -Encoding UTF8 | Where-Object {
            if ($Kind -eq 'code') { $_.commit -eq '' } else { $_.commit -ne '' }
        }
    }
    $all = New-Object System.Collections.ArrayList
    if ($Kind -eq 'code') { [void]$all.AddRange($Rows.ToArray()); foreach ($r in $keep) { [void]$all.Add($r) } }
    else { foreach ($r in $keep) { [void]$all.Add($r) }; [void]$all.AddRange($Rows.ToArray()) }
    $csv = $all | ConvertTo-Csv -NoTypeInformation
    [IO.File]::WriteAllLines($Provenance, [string[]]$csv, $Utf8)
    Write-Host "provenance: $($Rows.Count) 行（$Kind）→ $Provenance"
}

if ($Part -eq 'code') {
    # 流用元の HEAD とコミットが一致し、対象に未コミットの変更がないこと（Step A と同じ状態）
    $head = (Invoke-SrcGit @('rev-parse', 'HEAD'))[0].Trim()
    if ($head -ne $Commit) { throw "流用元の HEAD が $head（期待 $Commit）。止まって報告する" }

    # 複写元（パッケージ内の相対）→ 複写先（<ROOT> からの相対）。そのまま置く
    $Map = [ordered]@{
        'assets/panda/panda.xml'           = 'assets/mjcf/panda/panda.xml'
        'assets/panda/LICENSE'             = 'assets/mjcf/panda/LICENSE'
        'assets/panda/README.md'           = 'assets/mjcf/panda/README.md'
        'assets/panda/teleop_scene.xml'    = 'assets/mjcf/scene_g0.xml'
        'teleop/controller_ik.py'          = 'src/recovla/sim/controller_ik.py'
        'teleop/device.py'                 = 'src/recovla/sim/device.py'
        'teleop/app.py'                    = 'src/recovla/sim/app.py'
        'teleop/collect.py'                = 'src/recovla/sim/collect.py'
        'scripted_demo.py'                 = 'src/recovla/sim/scripted_demo.py'
        'teleop/recorder.py'               = 'src/recovla/record/recorder.py'
        'teleop/replay.py'                 = 'src/recovla/record/replay.py'
        'replay_check.py'                  = 'scripts/replay_check.py'
        'vla_image_spec.py'                = 'src/recovla/data/vla_image_spec.py'
        'vla_state.py'                     = 'src/recovla/data/vla_state.py'
        'vla_observation.py'               = 'src/recovla/data/vla_observation.py'
        'convert_to_lerobot.py'            = 'src/recovla/data/convert.py'
        'closed_loop_eval.py'              = 'src/recovla/eval/closed_loop.py'
        'train_launcher.py'                = 'src/recovla/policy/train_launcher.py'
        'code_version.py'                  = 'src/recovla/common/code_version.py'
        'tests/test_vla_image_spec.py'     = 'tests/legacy/test_vla_image_spec.py'
        'tests/test_vla_state.py'          = 'tests/legacy/test_vla_state.py'
        'tests/test_vla_observation.py'    = 'tests/legacy/test_vla_observation.py'
        'tests/test_convert_to_lerobot.py' = 'tests/legacy/test_convert_to_lerobot.py'
        'tests/test_replay.py'             = 'tests/legacy/test_replay.py'
        'tests/test_closed_loop_eval.py'   = 'tests/legacy/test_closed_loop_eval.py'
        'tests/test_controller_ik.py'      = 'tests/legacy/test_controller_ik.py'
        'tests/test_train_launcher.py'     = 'tests/legacy/test_train_launcher.py'
    }
    $pkg = 'panda_teleop'
    $meshes = (Invoke-SrcGit @('ls-tree', '-r', '--name-only', $Commit, '--', "$pkg/assets/panda/assets")) | Where-Object { $_ }
    foreach ($m in $meshes) { $Map[$m.Substring($pkg.Length + 1)] = 'assets/mjcf/panda/assets/' + (Split-Path -Leaf $m) }
    $Map['gui/gui_core.py'] = 'src/recovla/eval/stats.py'      # 関数 wilson_interval だけ（下で切り出す）

    # git archive で書き出して展開（流用元のリポジトリには書き込まない）
    $work = Join-Path $Root '.cache\import'
    $tar = Join-Path $work 'vla_dbb2c3b.tar'
    $x = Join-Path $work 'vla_dbb2c3b'
    New-Item -ItemType Directory -Force $work | Out-Null
    if (Test-Path $x) { Remove-Item -LiteralPath $x -Recurse -Force }
    New-Item -ItemType Directory $x | Out-Null
    Invoke-SrcGit (@('archive', $Commit, '-o', $tar, '--') + @($Map.Keys | ForEach-Object { "$pkg/$_" })) | Out-Null
    tar -xf $tar -C $x
    if ($LASTEXITCODE -ne 0) { throw 'tar の展開に失敗' }

    # blob id の一覧
    $blobs = @{}
    foreach ($line in (Invoke-SrcGit @('ls-tree', '-r', $Commit, '--', $pkg))) {
        if ($line -match '^\d+ blob ([0-9a-f]+)\t(.+)$') { $blobs[$matches[2]] = $matches[1] }
    }

    foreach ($rel in $Map.Keys) {
        $src = Join-Path $x "$pkg\$rel"
        $dst = Join-Path $Root $Map[$rel]
        New-Item -ItemType Directory -Force (Split-Path -Parent $dst) | Out-Null
        $srcSha = Get-Sha $src
        $treatment = 'そのまま'
        $note = ''
        if ($rel -eq 'assets/panda/teleop_scene.xml') {
            # 場面を panda\ の外に置くので、include の場所と、メッシュの場所（meshdir は主ファイルからの相対）だけ直す。
            # コンパイル後のモデル（mj_saveModel のバイト列）は流用元と一致する（tests/test_c_port.py で検査）
            $t = [IO.File]::ReadAllText($src, $Utf8)
            $old = '<include file="panda.xml"/>'
            if (-not $t.Contains($old)) { throw "teleop_scene.xml に $old がない" }
            $nl = if ($t.Contains("`r`n")) { "`r`n" } else { "`n" }
            $t = $t.Replace($old, '<include file="panda/panda.xml"/>' + $nl + '  <compiler meshdir="panda/assets"/>')
            [IO.File]::WriteAllText($dst, $t, $Utf8)
            $treatment = '置き換え'
            $note = 'include の相対パス panda.xml -> panda/panda.xml と、compiler meshdir="panda/assets" の 1 行を追加。コンパイル後のモデルは同一'
        } elseif ($rel -eq 'train_launcher.py') {
            # docstring にある学内の共有フォルダのパスを伏せる（公開リポジトリのため。掲示板 0002・0004）
            $t = [IO.File]::ReadAllText($src, $Utf8)
            $n = [regex]::Matches($t, '\(Y:\\\\[^,\r\n]*?\\\\03_').Count
            if ($n -ne 1) { throw "train_launcher.py の共有フォルダのパスが $n か所（期待 1）" }
            $t = [regex]::Replace($t, '\(Y:\\\\[^,\r\n]*?\\\\03_', '(<共有フォルダ>\\03_')
            [IO.File]::WriteAllText($dst, $t, $Utf8)
            $treatment = '伏せ字'
            $note = 'docstring の学内の共有フォルダのパス 1 か所を <共有フォルダ> に置き換え（動作に関係しない）'
        } elseif ($rel -eq 'gui/gui_core.py') {
            # 関数 wilson_interval だけを切り出す（本体の行はそのまま）
            $lines = [IO.File]::ReadAllLines($src, $Utf8)
            $i0 = -1
            for ($i = 0; $i -lt $lines.Length; $i++) {
                if ($lines[$i] -eq 'def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054):') { $i0 = $i; break }
            }
            if ($i0 -lt 0) { throw 'gui_core.py に wilson_interval がない' }
            $i1 = $i0 + 1
            while ($i1 -lt $lines.Length -and ($lines[$i1] -match '^\s' -or $lines[$i1] -eq '')) { $i1++ }
            while ($lines[$i1 - 1] -eq '') { $i1-- }
            $body = @('"""流用元 gui/gui_core.py の wilson_interval() だけを切り出したもの（B_提案書 §2.2）。"""', 'import math', '', '') + $lines[$i0..($i1 - 1)]
            [IO.File]::WriteAllLines($dst, [string[]]$body, $Utf8)
            $treatment = '関数だけ'
            $note = "gui_core.py の $($i0 + 1)〜$i1 行目（wilson_interval）と import math だけ"
        } else {
            Copy-Item -LiteralPath $src -Destination $dst -Force
        }
        Add-Row "pytools/panda_teleop/$pkg/$rel" $Commit $blobs["$pkg/$rel"] $srcSha $Map[$rel] (Get-Sha $dst) $treatment $note
    }
    # パッケージの目印（空）
    foreach ($d in @('src/recovla', 'src/recovla/common', 'src/recovla/sim', 'src/recovla/record', 'src/recovla/data', 'src/recovla/eval', 'src/recovla/policy', 'tests', 'tests/legacy')) {
        $f = Join-Path $Root "$d/__init__.py"
        if (-not (Test-Path $f)) { [IO.File]::WriteAllText($f, '', $Utf8) }
    }
    Write-Provenance 'code'
}

if ($Part -eq 'assets') {
    # 複写元 → 複写先（<ROOT> からの相対）。フォルダはその下の全ファイル
    $Items = @(
        @{ src = "$VlaEnv\hf_home\hub\models--lerobot--smolvla_libero"; dst = 'models/hf_home/hub/models--lerobot--smolvla_libero'; only = @('refs', 'snapshots') },
        @{ src = "$VlaEnv\hf_home\hub\models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct"; dst = 'models/hf_home/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct'; only = @('refs', 'snapshots') },
        @{ src = "$VlaEnv\lerobot\scripted_check\train\stage2_20260917-221354"; dst = 'outputs/legacy/stage2_20260917-221354'; files = @('conversion.json', 'train_run.json', 'train_launch_config.json'); dirs = @('checkpoints\019000\pretrained_model') },
        @{ src = "$VlaEnv\lerobot\scripted_check\raw\stage2\2026-09-17\ep_800000"; dst = 'outputs/legacy/raw/ep_800000' },
        @{ src = "$VlaEnv\lerobot\scripted_check\eval\stage2_closed_loop"; dst = 'outputs/legacy/eval_stage2_closed_loop'; pattern = '^(summary|trial_\d+)\.json$' }
    )
    $total = 0
    foreach ($it in $Items) {
        $files = @()
        if ($it.only) {
            foreach ($sub in $it.only) { $files += Get-ChildItem -LiteralPath (Join-Path $it.src $sub) -Recurse -File }
        } elseif ($it.files -or $it.dirs) {
            foreach ($f in $it.files) { $files += Get-Item -LiteralPath (Join-Path $it.src $f) }
            foreach ($d in $it.dirs) { $files += Get-ChildItem -LiteralPath (Join-Path $it.src $d) -Recurse -File }
        } elseif ($it.pattern) {
            $files = Get-ChildItem -LiteralPath $it.src -File | Where-Object { $_.Name -match $it.pattern }
        } else {
            $files = Get-ChildItem -LiteralPath $it.src -Recurse -File
        }
        foreach ($f in $files) {
            $rel = $f.FullName.Substring($it.src.Length).TrimStart('\')
            $dst = Join-Path $Root (Join-Path $it.dst $rel)
            New-Item -ItemType Directory -Force (Split-Path -Parent $dst) | Out-Null
            $srcSha = Get-Sha $f.FullName
            if (-not (Test-Path -LiteralPath $dst) -or (Get-Sha $dst) -ne $srcSha) {
                Copy-Item -LiteralPath $f.FullName -Destination $dst -Force
            }
            $dstSha = Get-Sha $dst
            if ($dstSha -ne $srcSha) { throw "複写の前後で SHA-256 が違う: $($f.FullName)" }
            Add-Row $f.FullName '' '' $srcSha (($it.dst + '/' + ($rel -replace '\\', '/'))) $dstSha 'そのまま' ''
            $total += $f.Length
        }
        Write-Host ("{0}: {1} ファイル" -f $it.dst, $files.Count)
    }
    Write-Host ("合計 {0:N1} MB" -f ($total / 1MB))
    Write-Provenance 'assets'
}
