r"""自動更新：啟動時偵測雲端版本，較新就下載更新包並換檔重啟。

運作方式（Windows 不能覆蓋執行中的 exe，所以用一支腳本接手換檔）：
  1. 抓 manifest（version.json）→ 取得最新版本號與更新包 zip 網址。
  2. 比對版本；雲端較新才繼續。
  3. 下載 zip → （可選）驗證 sha256 → 解壓到暫存資料夾。
  4. 產生 apply_update.ps1，等本程式關閉後用 robocopy 把新檔蓋上去，
     但「跳過使用者自己的檔案」（config.yaml / 啟動設定.txt /
     service_account.json / data\ / logs\），再自動重開程式。
  5. 主程式立刻結束，交給腳本完成換檔。

manifest（version.json）格式範例：
  {
    "version": "1.1.0",
    "url": "https://.../星城跟注_1.1.0.zip",
    "sha256": "可省略",
    "notes": "更新說明（可省略）"
  }
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

from star_follow import paths
from star_follow.config import UpdateConfig
from star_follow.version import __version__ as CURRENT_VERSION

logger = logging.getLogger(__name__)

# 換檔時要保留、不被覆蓋的使用者檔案／資料夾（相對安裝資料夾）
_PRESERVE_FILES = ["config.yaml", "啟動設定.txt", "service_account.json"]
_PRESERVE_DIRS = ["data", "logs"]

# 更新暫存用固定 ASCII 目錄（避免提權後 %TEMP% 路徑不一致、或中文路徑問題）。
_STAGE_DIR = Path(r"C:\ProgramData\StarFollow\update")


def _ulog(msg: str) -> None:
    """更新流程自有 log，直接寫到安裝資料夾的 logs\\update.log。

    （maybe_update 在主程式 logging 建立前就執行，且換檔會 os._exit，所以這裡
    自己寫檔，確保即使換檔失敗或子程序被殺，也留得下 Python 端的軌跡。）
    """
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    try:
        logger.info("[update] %s", msg)
    except Exception:  # noqa: BLE001
        pass
    try:
        p = paths.app_dir() / "logs" / "update.log"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:  # noqa: BLE001
        pass


def _parse_ver(s: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in str(s).strip().lstrip("vV").split("."):
        num = "".join(ch for ch in chunk if ch.isdigit())
        parts.append(int(num) if num else 0)
    return tuple(parts) or (0,)


def is_newer(remote: str, local: str) -> bool:
    return _parse_ver(remote) > _parse_ver(local)


def fetch_manifest(url: str, timeout: float) -> dict | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "StarFollow-Updater"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        if isinstance(data, dict) and data.get("version") and data.get("url"):
            return data
        logger.warning("更新 manifest 格式不正確（需含 version 與 url）")
    except Exception as exc:  # noqa: BLE001
        logger.info("檢查更新失敗（略過，照常啟動）：%s", exc)
    return None


def _download(url: str, dest: Path, timeout: float) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "StarFollow-Updater"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, dest.open("wb") as f:
            while True:
                buf = resp.read(1 << 16)
                if not buf:
                    break
                f.write(buf)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("下載更新包失敗：%s", exc)
        return False


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda: f.read(1 << 16), b""):
            h.update(buf)
    return h.hexdigest()


def _find_program_root(extracted: Path) -> Path | None:
    """在解壓內容裡找出含 StarFollow.exe 的資料夾。"""
    if (extracted / "StarFollow.exe").is_file():
        return extracted
    for p in extracted.rglob("StarFollow.exe"):
        return p.parent
    return None


def _write_apply_script(work: Path) -> Path:
    """產生換檔用 PowerShell 腳本。

    重點：全程寫 log 到 $Dst\\logs\\update_apply.log（方便事後查為何沒換成功），
    並做到：等舊程式完全結束 → 把舊 exe 先改名讓位（避免被鎖） → robocopy 蓋上
    新檔（保留使用者檔）→ 檢查 robocopy 退出碼 → 重開程式 → 清暫存。
    """
    script = r"""
param(
  [int]$ProcId,
  [string]$Src,
  [string]$Dst,
  [string]$Exe
)
$ErrorActionPreference = "Continue"
$log = Join-Path $Dst "logs\update_apply.log"
try { New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null } catch {}
function Log($m) {
  $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m
  try { Add-Content -Path $log -Value $line -Encoding UTF8 } catch {}
}
Log "=== 換檔開始 ProcId=$ProcId ==="
Log "Src=$Src"
Log "Dst=$Dst"
Log "Exe=$Exe"

# 1) 等主程式完全結束（最多 30 秒輪詢，避免檔案還被鎖）
if ($ProcId -gt 0) {
  for ($i = 0; $i -lt 30; $i++) {
    $p = Get-Process -Id $ProcId -ErrorAction SilentlyContinue
    if (-not $p) { break }
    Start-Sleep -Milliseconds 500
  }
}
Start-Sleep -Seconds 1
Log "主程式已結束，開始換檔"

# 2) 舊 exe 先改名讓位（即使仍被短暫鎖住，新檔也能寫入）
$old = "$Exe.old"
try { if (Test-Path $old) { Remove-Item $old -Force -ErrorAction SilentlyContinue } } catch {}
try { if (Test-Path $Exe) { Rename-Item -Path $Exe -NewName ([System.IO.Path]::GetFileName($old)) -Force -ErrorAction SilentlyContinue } } catch { Log "改名舊 exe 失敗：$_" }

# 3) 蓋上新檔，但保留使用者檔案
$xf = @('config.yaml','啟動設定.txt','service_account.json')
$xd = @('data','logs')
$roboLog = Join-Path $Dst "logs\update_robocopy.log"
robocopy $Src $Dst /E /XF $xf /XD $xd /R:3 /W:2 /NP /LOG:$roboLog | Out-Null
$code = $LASTEXITCODE
Log "robocopy 退出碼=$code（<8 視為成功）"

if ($code -ge 8) {
  Log "robocopy 失敗，嘗試把舊 exe 還原以免無法啟動"
  try { if ((Test-Path $old) -and -not (Test-Path $Exe)) { Rename-Item -Path $old -NewName ([System.IO.Path]::GetFileName($Exe)) -Force } } catch {}
} else {
  try { if (Test-Path $old) { Remove-Item $old -Force -ErrorAction SilentlyContinue } } catch {}
}

# 4) 重開程式
if (Test-Path $Exe) {
  Log "重新啟動：$Exe"
  try { Start-Process -FilePath $Exe -WorkingDirectory $Dst } catch { Log "重啟失敗：$_" }
} else {
  Log "找不到 $Exe，無法重啟"
}

# 5) 清掉暫存
try { Remove-Item $Src -Recurse -Force -ErrorAction SilentlyContinue } catch {}
Log "=== 換檔結束 ==="
"""
    sp = work / "apply_update.ps1"
    # 用 UTF-8 with BOM 讓 PowerShell 正確讀中文
    sp.write_text(script, encoding="utf-8-sig")
    return sp


def _launch_helper(args: list[str]) -> bool:
    """啟動換檔用 powershell 子程序，並確保它能脫離父程序的 job 而存活。

    之前的問題：父程序 os._exit 後，detached 子程序仍可能因為同屬一個 job
    （kill-on-close）被一起殺掉，導致換檔腳本一行都沒跑。這裡加上
    CREATE_BREAKAWAY_FROM_JOB；若該旗標不被允許（不在 job 內或不允許脫離），
    退回不帶該旗標再試。
    """
    # 注意：千萬別用 DETACHED_PROCESS——實測它會讓子程序在父程序 os._exit 後
    # 一起死掉（換檔腳本一行都跑不到）。改用 CREATE_NO_WINDOW（隱藏但存活）。
    NO_WINDOW = 0x08000000
    NEW_GROUP = 0x00000200
    BREAKAWAY = 0x01000000
    sysroot = os.environ.get("SystemRoot", r"C:\Windows")
    ps_exe = os.path.join(sysroot, "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
    if not os.path.isfile(ps_exe):
        ps_exe = "powershell"
    cmd = [
        ps_exe, "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-WindowStyle", "Hidden", "-File", *args,
    ]
    for flags, tag in (
        (NO_WINDOW | NEW_GROUP | BREAKAWAY, "nowindow+breakaway"),
        (NO_WINDOW | NEW_GROUP, "nowindow"),
    ):
        try:
            p = subprocess.Popen(cmd, creationflags=flags, close_fds=True)
            _ulog(f"換檔子程序已啟動（{tag}）pid={p.pid}")
            return True
        except Exception as exc:  # noqa: BLE001
            _ulog(f"以 {tag} 啟動換檔子程序失敗：{exc}")
    return False


def _apply(zip_path: Path, install_dir: Path) -> bool:
    """解壓、產生腳本、啟動腳本並回傳是否成功啟動換檔流程。"""
    # 只清「解壓暫存子資料夾」，千萬別清整個 _STAGE_DIR——下載好的 zip 就放在
    # 那裡，清掉會把要解壓的檔案一起刪掉。
    extracted = _STAGE_DIR / "extracted"
    try:
        if extracted.exists():
            shutil.rmtree(extracted, ignore_errors=True)
        extracted.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        _ulog(f"建立解壓暫存資料夾失敗 {extracted}：{exc}")
        return False

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extracted)
    except Exception as exc:  # noqa: BLE001
        _ulog(f"解壓更新包失敗：{exc}")
        return False
    root = _find_program_root(extracted)
    if root is None:
        _ulog("更新包裡找不到 StarFollow.exe，放棄更新")
        return False

    ps = _write_apply_script(_STAGE_DIR)
    exe = str(install_dir / "StarFollow.exe")
    _ulog(f"準備換檔：Src={root} Dst={install_dir}")
    _ulog(f"換檔腳本：{ps}")
    if not _launch_helper([
        str(ps),
        "-ProcId", str(os.getpid()),
        "-Src", str(root),
        "-Dst", str(install_dir),
        "-Exe", exe,
    ]):
        return False
    # 給子程序一點起跑時間，避免父程序立刻 os._exit 造成搶跑
    time.sleep(1.5)
    return True


def _marker_path() -> Path:
    return paths.app_dir() / "data" / ".update_attempt.json"


def _read_marker() -> dict:
    try:
        return json.loads(_marker_path().read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _write_marker(target: str) -> None:
    try:
        import time as _t

        p = _marker_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"target": target, "ts": _t.time()}),
                     encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def _clear_marker() -> None:
    try:
        _marker_path().unlink()
    except Exception:  # noqa: BLE001
        pass


def maybe_update(cfg: UpdateConfig) -> bool:
    """啟動時檢查並（若設定 auto_apply）套用更新。

    回傳 True 代表已啟動換檔流程、呼叫端應立刻結束程式。
    """
    if not cfg.enabled or not cfg.check_on_start or not cfg.manifest_url:
        return False

    print(f"目前版本 v{CURRENT_VERSION}，檢查更新中…")
    man = fetch_manifest(cfg.manifest_url, cfg.timeout_s)
    if not man:
        return False
    remote = str(man["version"])
    if not is_newer(remote, CURRENT_VERSION):
        # 已是最新：清掉「上次嘗試」標記（代表先前若有更新已成功）
        _clear_marker()
        print(f"已是最新版 v{CURRENT_VERSION}")
        return False

    # 防無限更新迴圈：若上次才剛嘗試更新到同一版、但這次啟動還是舊版，
    # 代表換檔沒成功；不要再自動下載換檔（會一直重開），改為提示後照常啟動。
    import time as _t

    mk = _read_marker()
    if mk.get("target") == remote and (_t.time() - float(mk.get("ts", 0))) < 1800:
        print(f"偵測到上次自動更新到 v{remote} 未成功（仍為 v{CURRENT_VERSION}）。")
        print("已略過這次自動更新以免一直重開；請改用手動更新（見使用說明）。")
        logger.warning("自動更新疑似失敗：上次已嘗試 v%s，仍停在 v%s", remote, CURRENT_VERSION)
        return False

    notes = man.get("notes") or ""
    print(f"發現新版本 v{remote}（目前 v{CURRENT_VERSION}）{('：' + notes) if notes else ''}")
    _ulog(f"發現新版本 v{remote}（目前 v{CURRENT_VERSION}）")

    if not cfg.auto_apply:
        print("（auto_apply=false，僅提示，不自動更新）")
        return False
    if not paths.is_frozen():
        logger.info("開發模式不執行自動換檔（請改用 git pull）")
        return False

    # 下載更新包到固定 ASCII 暫存目錄
    try:
        _STAGE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001
        pass
    tmp_zip = _STAGE_DIR / f"StarFollow_{remote}.zip"
    print("下載更新包中…")
    _ulog(f"下載更新包：{man['url']} -> {tmp_zip}")
    # 下載逾時放寬，更新包通常較大
    if not _download(str(man["url"]), tmp_zip, max(cfg.timeout_s, 120.0)):
        _ulog("下載更新包失敗")
        return False
    _ulog(f"下載完成，大小={tmp_zip.stat().st_size if tmp_zip.exists() else 0}")

    want_hash = str(man.get("sha256") or "").strip().lower()
    if want_hash:
        got = _sha256(tmp_zip).lower()
        if got != want_hash:
            _ulog(f"sha256 不符：預期 {want_hash} 實得 {got}")
            logger.warning("更新包校驗碼不符（預期 %s，實得 %s），放棄更新", want_hash, got)
            return False
        _ulog("sha256 校驗通過")

    install_dir = paths.app_dir()
    print("套用更新並重新啟動…")
    _ulog(f"套用更新，install_dir={install_dir}")
    # 先記下「這次要更新到哪一版」；若換檔失敗、下次啟動還是舊版，就靠這個標記
    # 避免無限重開。
    _write_marker(remote)
    if _apply(tmp_zip, install_dir):
        # 立刻硬結束，讓換檔腳本能蓋掉檔案（略過 frozen 的按 Enter 暫停）
        os._exit(0)
    # 啟動換檔腳本失敗：清掉標記，照常啟動
    _clear_marker()
    return False
