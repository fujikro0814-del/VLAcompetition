# push は必ずこの入口から行う。検査（check_before_push.ps1）に通ったときだけ push する。
# 使い方: powershell -ExecutionPolicy Bypass -File scripts\push.ps1 [枝の名前（既定 main）]
param([string]$Branch = 'main')
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$git = Join-Path $Root '.tools\git\cmd\git.exe'

& $git -C $Root fetch origin
if ($LASTEXITCODE -ne 0) { throw 'fetch に失敗' }
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'check_before_push.ps1') -Repo $Root -Upstream "origin/$Branch"
if ($LASTEXITCODE -ne 0) { Write-Host 'push を中止した。' -ForegroundColor Red; exit 1 }
& $git -C $Root push origin $Branch
exit $LASTEXITCODE
