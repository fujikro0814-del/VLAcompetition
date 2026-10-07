# Stage 4 bundle 1 (no-training diagnostics): runs the plan's jobs in N lanes (default 3), resumable.
# Owner: ops role (D). ASCII only (PowerShell 5.1).
#
# Usage (from any directory):
#   powershell -NoProfile -ExecutionPolicy Bypass -File outputs\s4\runs\run_bundle1.ps1                 # production plan, 3 lanes
#   powershell -NoProfile -ExecutionPolicy Bypass -File outputs\s4\runs\run_bundle1.ps1 -DryRun         # order and commands only
#   powershell -NoProfile -ExecutionPolicy Bypass -File outputs\s4\runs\run_bundle1.ps1 -Plan outputs\s4\bundle1_smoke_plan.json -Lanes 1 -NoEnvRecord
# Same command again = resume (finished jobs are skipped; each child resumes its own trials).
# Stop after the current block: create outputs\s4\runs\STOP_BUNDLE1 (smoke plan: STOP_BUNDLE1_SMOKE). Delete it before resuming.
# If launched detached, ALWAYS attach the watcher in the background:
#   .venv\Scripts\python.exe scripts\96_s4_ops.py wait --progress outputs\s4\runs\bundle1_status.json --appear-min 30
# Reads: the plan JSON. Writes: outputs\s4\runs\bundle1_status.json, bundle1.log, bundle1_logs\<job>.log (paths come from the plan),
#   and the children write outputs\v2eval\<experiment>\<condition>\ and outputs\s4\<diag>\.
# Exit code = the bundle's exit code (0 all done, 1 stopped, 2 some job failed, 3 precondition such as a second bundle running).
param(
    [string]$Plan = 'outputs\s4\bundle1_plan_queue.json',   # same as 98_s4_d_plan.py PLAN_PATH ("queue" in the name: 97 does not count its seeds)
    [int]$Lanes = 3,
    [switch]$DryRun,
    [switch]$NoInterleave,
    [switch]$NoEnvRecord,
    [string]$Only = '',
    [string]$Extra = ''
)
$ErrorActionPreference = 'Continue'
Set-Location 'C:\PAI\recovery_vla'
. .\env\session_env.example.ps1 *> $null
$env:PYTHONIOENCODING = 'utf-8'

$py = '.\.venv\Scripts\python.exe'
$a = @('scripts\98_s4_d_plan.py', 'bundle', '--plan', $Plan, '--lanes', "$Lanes")
if ($DryRun) { $a += '--dry-run' }
if ($NoInterleave) { $a += '--no-interleave' }
if ((-not $NoEnvRecord) -and (-not $DryRun)) { $a += '--env-record' }
if ($Only -ne '') { $a += @('--only', $Only) }
if ($Extra -ne '') { $a += ($Extra -split ' ' | Where-Object { $_ -ne '' }) }

# E7 jobs call the planner (Haiku 5.5) when the cache misses: warn if no API key is visible (process or user environment).
$key = $env:ANTHROPIC_API_KEY
if (-not $key) { $key = [Environment]::GetEnvironmentVariable('ANTHROPIC_API_KEY', 'User') }
if (-not $key) { Write-Host 'WARNING: ANTHROPIC_API_KEY not set (process/user). E7 jobs fail if the planner cache misses.' }

& $py @a
$code = $LASTEXITCODE
Write-Host "run_bundle1.ps1: EXIT $code $(Get-Date -Format s)"
exit $code
