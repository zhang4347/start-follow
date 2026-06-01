# Launch PATROL (room-switching) engine in DRY-RUN, elevated.
# 巡房 dry-run：會換桌、開關統計、OCR、判斷跟注，但不真的點籌碼。
$Root = $PSScriptRoot
if (-not $Root) { $Root = Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $Root) { $Root = (Get-Location).Path }
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Error "Missing .venv. Create venv in project folder first."
    exit 1
}
$Cmd = "Set-Location -LiteralPath '$Root'; `$env:PYTHONIOENCODING='utf-8'; & '$Py' -m star_follow.tools.run --patrol"
Start-Process powershell.exe -Verb RunAs -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $Cmd
