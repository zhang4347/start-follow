"""Windows 是否以系統管理員執行。"""

from __future__ import annotations

import ctypes
import sys


def is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def require_admin(*, live: bool = False) -> None:
    if is_admin():
        return
    msg = (
        "目前 PowerShell / Python 不是「以系統管理員身分執行」。\n"
        "星城若也是管理員開的，一般權限的點擊會進不了遊戲（你看不到任何動作）。\n"
    )
    if live:
        msg += (
            "\n請改為：\n"
            "  1. 右鍵 PowerShell -> 以系統管理員身分執行\n"
            "  2. cd 到專案目錄後執行：.\\start_live_admin.ps1\n"
            "     或：.\\.venv\\Scripts\\python -m star_follow.tools.run --live\n"
        )
        print(msg, file=sys.stderr)
        raise SystemExit(2)
    print(msg + "（dry-run 可繼續，但 LIVE 建議用管理員）")
