# 一鍵打包交付版：產生免裝 Python 的可執行資料夾並壓成 zip。
# 用法（在專案根目錄、PowerShell）：  .\build_release.ps1
# 註：用 Continue 而非 Stop，避免 pip/PyInstaller 把訊息寫到 stderr 時被
#     誤判為致命錯誤；關鍵步驟改用明確的 Test-Path 檢查把關。
$ErrorActionPreference = "Continue"
$proj = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $proj

$py = ".\.venv\Scripts\python.exe"
$distApp = ".\dist\StarFollow"
$stageRoot = ".\release"
$stageName = "星城跟注"
$stage = Join-Path $stageRoot $stageName

Write-Host "[0/5] 關閉殘留程式並確認相依套件..." -ForegroundColor Cyan
Get-Process StarFollow -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
& $py -m pip install -q gspread google-auth cachetools 2>&1 | Out-Null
# 先刪 build/dist 與所有 __pycache__，避免 PyInstaller 中途失敗時沿用「舊的」
# dist\StarFollow.exe（會把過期 bytecode 打包進去，造成已修好的程式碼沒生效）。
Remove-Item .\build -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item .\dist -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path .\star_follow -Recurse -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "[1/5] PyInstaller 打包中..." -ForegroundColor Cyan
$buildStart = Get-Date
& $py -m PyInstaller --noconfirm --clean star_follow.spec 2>&1 | Out-Null
if (-not (Test-Path "$distApp\StarFollow.exe")) { Write-Host "打包失敗：找不到 StarFollow.exe" -ForegroundColor Red; exit 1 }
# 確認 exe 是「這次」剛建出來的（不是殘留舊檔），否則視為打包失敗。
if ((Get-Item "$distApp\StarFollow.exe").LastWriteTime -lt $buildStart) {
  Write-Host "打包失敗：dist\StarFollow.exe 不是本次新建（可能 PyInstaller 中途失敗、沿用舊檔）。請重跑。" -ForegroundColor Red
  exit 1
}

Write-Host "[2/5] 準備交付資料夾..." -ForegroundColor Cyan
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Force -Path $stage | Out-Null
Copy-Item "$distApp\*" $stage -Recurse -Force

Write-Host "[3/5] 放入可編輯的設定/名單/啟動檔/說明..." -ForegroundColor Cyan
Copy-Item ".\star_follow\config.yaml" "$stage\config.yaml" -Force
New-Item -ItemType Directory -Force -Path "$stage\data" | Out-Null
Copy-Item ".\star_follow\data\follow_list.json" "$stage\data\follow_list.json" -Force
New-Item -ItemType Directory -Force -Path "$stage\logs" | Out-Null
Copy-Item ".\release_files\*" $stage -Recurse -Force
if (-not (Test-Path "$stage\config.yaml")) { Write-Host "打包失敗：交付資料夾缺 config.yaml" -ForegroundColor Red; exit 1 }
if (-not (Test-Path "$stage\StarFollow.exe")) { Write-Host "打包失敗：交付資料夾缺 StarFollow.exe" -ForegroundColor Red; exit 1 }

Write-Host "[4/5] 壓縮成 zip..." -ForegroundColor Cyan
# 讀版本號（用於命名與 version.json）
$verMatch = Select-String -Path ".\star_follow\version.py" -Pattern '__version__\s*=\s*"([^"]+)"'
$ver = if ($verMatch) { $verMatch.Matches[0].Groups[1].Value } else { "0.0.0" }
# 更新包檔名用 ASCII，避免上傳 GitHub 後下載網址中文編碼問題
$zipName = "StarFollow_$ver.zip"
$zip = Join-Path $stageRoot $zipName
if (Test-Path $zip) { Remove-Item $zip -Force }
# 直接在 OneDrive 資料夾壓縮常被 OneDrive 鎖住 base_library.zip，
# 故先複製到 TEMP 外部資料夾壓縮，再把 zip 搬回來。
$tmpStage = Join-Path $env:TEMP "sf_stage"
$tmpZip = Join-Path $env:TEMP $zipName
Remove-Item $tmpStage -Recurse -Force -ErrorAction SilentlyContinue
if (Test-Path $tmpZip) { Remove-Item $tmpZip -Force -ErrorAction SilentlyContinue }
Copy-Item $stage $tmpStage -Recurse -Force
Compress-Archive -Path $tmpStage -DestinationPath $tmpZip -Force
Copy-Item $tmpZip $zip -Force
Remove-Item $tmpStage -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item $tmpZip -Force -ErrorAction SilentlyContinue
if (-not (Test-Path $zip)) { Write-Host "打包失敗：zip 未產生" -ForegroundColor Red; exit 1 }

# 產生 version.json 範本（上傳更新時用；URL 換成實際的 zip 下載網址）
$hash = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
$verJson = [ordered]@{
  version = $ver
  url     = "https://github.com/zhang4347/start-follow/releases/download/v$ver/$zipName"
  sha256  = $hash
  notes   = "v$ver 更新"
}
$verJsonPath = Join-Path $stageRoot "version.json"
($verJson | ConvertTo-Json) | Set-Content -Path $verJsonPath -Encoding UTF8

$size = [math]::Round((Get-Item $zip).Length / 1MB, 1)
Write-Host "[5/5] 完成！(v$ver)" -ForegroundColor Green
Write-Host "  交付資料夾：$stage"
Write-Host "  交付壓縮檔：$zip  ($size MB)"
Write-Host "  更新清單：  $verJsonPath"
Write-Host ""
Write-Host "首次交付：把 zip 給客戶。客戶解壓後："
Write-Host "  1) 編輯〔啟動設定.txt〕選模式/下注  2) 雙擊〔StarFollow.exe〕按 UAC 是"
Write-Host ""
Write-Host "發佈更新（GitHub）："
Write-Host "  1) 在 GitHub 開一個 Release（tag 用 v$ver），把 $zipName 當附件上傳。"
Write-Host "  2) 把 release\version.json 的內容覆蓋到 repo 根目錄的 version.json"
Write-Host "     （url 已是 zhang4347/start-follow，確認 sha256 是這支 zip 的）。"
Write-Host "  3) git add version.json; git commit -m \"release v$ver\"; git push"
Write-Host "     客戶 config.yaml 的 manifest_url 指向（repo 必須是 Public）："
Write-Host "     https://raw.githubusercontent.com/zhang4347/start-follow/main/version.json"

