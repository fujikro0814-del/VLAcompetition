# C:\VLA 以下の全ファイルの一覧を取り、Step A の開始時の一覧と比べる（手順書 Step C 完了条件 4）。C:\VLA は読むだけ。
# 一覧の形は Step A と同じ（Path, Length, LastWriteTimeUtc）。一覧は Git 管理外の docs\local\ に置く（ファイル名を含むため）。
# 結果の要約（件数・差の件数・差の内訳。パスは書かない）を outputs\g0\vla_listing_check.json に書く。
# 使い方: powershell -ExecutionPolicy Bypass -File scripts\03_vla_listing.ps1
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Before = Join-Path $Root 'docs\local\A_vla_listing_before.csv'
$Stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$Now = Join-Path $Root "docs\local\C_vla_listing_$Stamp.csv"
$Report = Join-Path $Root 'outputs\g0\vla_listing_check.json'

Get-ChildItem -LiteralPath 'C:\VLA' -Recurse -File -Force |
    ForEach-Object { [pscustomobject]@{ Path = $_.FullName; Length = $_.Length; LastWriteTimeUtc = $_.LastWriteTimeUtc.ToString('o') } } |
    Export-Csv -LiteralPath $Now -NoTypeInformation -Encoding UTF8

$a = Import-Csv -LiteralPath $Before -Encoding UTF8
$b = Import-Csv -LiteralPath $Now -Encoding UTF8
$diff = @(Compare-Object $a $b -Property Path, Length, LastWriteTimeUtc)
$onlyBefore = @($diff | Where-Object SideIndicator -eq '<=')
$onlyNow = @($diff | Where-Object SideIndicator -eq '=>')
# 差の内訳は C:\VLA の直下のフォルダごとの件数だけ（パスそのものは docs\local\ の一覧で見る）
function Get-Top($rows) { $rows | ForEach-Object { ($_.Path -replace '^C:\\VLA\\', '').Split('\')[0] } | Group-Object | ForEach-Object { @{ top = $_.Name; n = $_.Count } } }
$result = [ordered]@{
    at = (Get-Date).ToString('s'); before = $Before; now = $Now
    count_before = $a.Count; count_now = $b.Count
    only_before = $onlyBefore.Count; only_now = $onlyNow.Count
    only_before_by_top = @(Get-Top $onlyBefore); only_now_by_top = @(Get-Top $onlyNow)
    ok = ($diff.Count -eq 0)
}
New-Item -ItemType Directory -Force (Split-Path -Parent $Report) | Out-Null
[IO.File]::WriteAllText($Report, ($result | ConvertTo-Json -Depth 5), (New-Object Text.UTF8Encoding $false))
Write-Host ("一覧: 開始時 {0} 件、今 {1} 件、差 {2} 件（開始時にだけ {3}、今だけ {4}）" -f $a.Count, $b.Count, $diff.Count, $onlyBefore.Count, $onlyNow.Count)
if ($diff.Count -gt 0) { $diff | Select-Object -First 20 | Format-Table -AutoSize | Out-String -Width 250 | Write-Host }
exit ([int](-not $result.ok))
