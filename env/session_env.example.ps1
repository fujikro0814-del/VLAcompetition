# 1 回の作業で設定する環境変数の例（B_提案書 §3.3）。このセッションだけに効く。鍵は書かない。
# 使い方: . .\env\session_env.example.ps1   （先頭のドットで、今の PowerShell に読み込む）
$Root = Split-Path -Parent $PSScriptRoot
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:HF_HOME = Join-Path $Root 'models\hf_home'
$env:HF_HUB_OFFLINE = '1'
$env:PYTHONIOENCODING = 'utf-8'
# LLM を使うとき（Step I）だけ、鍵は手で設定する（ファイルに書かない）:
#   $env:ANTHROPIC_API_KEY = '...'
Write-Host "HF_HOME=$env:HF_HOME  HF_HUB_OFFLINE=$env:HF_HUB_OFFLINE"
