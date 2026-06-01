"""PyInstaller 打包進入點。

雙擊 StarFollow.exe 即可啟動；模式與是否下注由同資料夾的「啟動設定.txt」決定。
也支援命令列參數（會覆寫設定檔）：
  --live    真正下注
  --patrol  換房巡房模式
  --stay    掛房單桌模式
  --selftest 自我檢查
"""

import sys

# 凍結（PyInstaller）環境下，先在「乾淨的啟動環境」就把 OpenCV 初始化好。
# OpenCV 的 __init__ 會在首次 import 時把 native 綁定（putText、matchTemplate…）
# 接到 cv2 命名空間；若拖到主流程後段（自動更新、OCR 暫存搬移之後）才首次
# import，偶發會載入不全而出現「module 'cv2' has no attribute 'putText'」。
# 提早 import 可避免此問題；失敗也不擋啟動（後續仍會再 import 一次）。
try:
    import cv2  # noqa: F401
except Exception:  # noqa: BLE001
    pass

from star_follow.tools.run import main

if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except SystemExit as exc:
        code = int(exc.code) if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except KeyboardInterrupt:
        code = 0
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        code = 1
    # 打包後雙擊執行時，結束前暫停，讓使用者能看到訊息/錯誤
    if getattr(sys, "frozen", False):
        try:
            input("\n程式已結束，按 Enter 關閉視窗...")
        except Exception:  # noqa: BLE001
            pass
    sys.exit(code)
