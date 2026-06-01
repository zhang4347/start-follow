# Launch DRY-RUN engine elevated (opens/closes stats + OCR, no chip clicks).
# 用於量測時序，不會真的下注。
$Root = $PSScriptRoot
if (-not $Root) { $Root = Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $Root) { $Root = (Get-Location).Path }
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Error "Missing .venv. Create venv in project folder first."
    exit 1
}
$Cmd = "Set-Location -LiteralPath '$Root'; `$env:PYTHONIOENCODING='utf-8'; & '$Py' -m star_follow.tools.run"
Start-Process powershell.exe -Verb RunAs -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $Cmd
