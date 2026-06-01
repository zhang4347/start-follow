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
import subprocess
import sys
import tempfile
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
    """產生換檔用 PowerShell 腳本。"""
    script = r"""
param(
  [int]$ProcId,
  [string]$Src,
  [string]$Dst,
  [string]$Exe
)
$ErrorActionPreference = "SilentlyContinue"
# 等主程式關閉
if ($ProcId -gt 0) { Wait-Process -Id $ProcId -Timeout 60 }
Start-Sleep -Seconds 1
# 蓋上新檔，但保留使用者檔案
$xf = @('config.yaml','啟動設定.txt','service_account.json')
$xd = @('data','logs')
robocopy $Src $Dst /E /XF $xf /XD $xd /R:2 /W:1 | Out-Null
# 清掉暫存並重開程式
Start-Process -FilePath $Exe
Remove-Item $Src -Recurse -Force
"""
    sp = work / "apply_update.ps1"
    # 用 UTF-8 with BOM 讓 PowerShell 正確讀中文
    sp.write_text(script, encoding="utf-8-sig")
    return sp


def _apply(zip_path: Path, install_dir: Path) -> bool:
    """解壓、產生腳本、啟動腳本並回傳是否成功啟動換檔流程。"""
    work = Path(tempfile.mkdtemp(prefix="StarFollowUpd_"))
    extracted = work / "extracted"
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extracted)
    except Exception as exc:  # noqa: BLE001
        logger.warning("解壓更新包失敗：%s", exc)
        return False
    root = _find_program_root(extracted)
    if root is None:
        logger.warning("更新包裡找不到 StarFollow.exe，放棄更新")
        return False

    ps = _write_apply_script(work)
    exe = str(install_dir / "StarFollow.exe")
    DETACHED = 0x00000008
    NEW_GROUP = 0x00000200
    try:
        subprocess.Popen(
            [
                "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                "-WindowStyle", "Hidden", "-File", str(ps),
                "-ProcId", str(os.getpid()),
                "-Src", str(root),
                "-Dst", str(install_dir),
                "-Exe", exe,
            ],
            creationflags=DETACHED | NEW_GROUP,
            close_fds=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("啟動換檔腳本失敗：%s", exc)
        return False
    return True


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
        print(f"已是最新版 v{CURRENT_VERSION}")
        return False

    notes = man.get("notes") or ""
    print(f"發現新版本 v{remote}（目前 v{CURRENT_VERSION}）{('：' + notes) if notes else ''}")

    if not cfg.auto_apply:
        print("（auto_apply=false，僅提示，不自動更新）")
        return False
    if not paths.is_frozen():
        logger.info("開發模式不執行自動換檔（請改用 git pull）")
        return False

    # 下載更新包
    tmp_zip = Path(tempfile.gettempdir()) / f"StarFollow_{remote}.zip"
    print("下載更新包中…")
    # 下載逾時放寬，更新包通常較大
    if not _download(str(man["url"]), tmp_zip, max(cfg.timeout_s, 120.0)):
        return False

    want_hash = str(man.get("sha256") or "").strip().lower()
    if want_hash:
        got = _sha256(tmp_zip).lower()
        if got != want_hash:
            logger.warning("更新包校驗碼不符（預期 %s，實得 %s），放棄更新", want_hash, got)
            return False

    install_dir = paths.app_dir()
    print("套用更新並重新啟動…")
    if _apply(tmp_zip, install_dir):
        # 立刻硬結束，讓換檔腳本能蓋掉檔案（略過 frozen 的按 Enter 暫停）
        os._exit(0)
    return False
