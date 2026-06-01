# Launch PATROL (room-switching) engine LIVE (real betting), elevated.
# 巡房真下注：換桌找對象，跟著下注。
$Root = $PSScriptRoot
if (-not $Root) { $Root = Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $Root) { $Root = (Get-Location).Path }
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Error "Missing .venv. Create venv in project folder first."
    exit 1
}
$Cmd = "Set-Location -LiteralPath '$Root'; `$env:PYTHONIOENCODING='utf-8'; & '$Py' -m star_follow.tools.run --live --patrol"
Start-Process powershell.exe -Verb RunAs -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $Cmd
