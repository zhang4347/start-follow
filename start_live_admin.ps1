# Launch LIVE engine in an elevated PowerShell window (UAC prompt).
$Root = $PSScriptRoot
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Error "Missing .venv. Create venv in project folder first."
    exit 1
}
$Cmd = "Set-Location -LiteralPath '$Root'; `$env:PYTHONIOENCODING='utf-8'; & '$Py' -m star_follow.tools.run --live"
Start-Process powershell.exe -Verb RunAs -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $Cmd
