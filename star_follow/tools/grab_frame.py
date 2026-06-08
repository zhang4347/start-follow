"""擷取整個遊戲 client 畫面存成 PNG（供校正 ROI／診斷餘額辨識用）。

用法（先進 venv 或直接用 venv 的 python）：
    python -m star_follow.tools.grab_frame

不需管理員、不會點任何東西，只截圖。會把畫面存到 logs\\grab\\ 並印出餘額框現在
框到的位置（紅框）方便核對。把存出的 PNG 傳回即可。
"""

from __future__ import annotations

import time

import numpy as np
from PIL import Image

from star_follow.capture.screen import capture_client
from star_follow.capture.window import find_game_window, refresh_game_window
from star_follow.config import load_config
from star_follow.paths import logs_dir
from star_follow.vision.roi import scale_rect


def main() -> int:
    cfg = load_config()
    win = find_game_window(
        cfg.window.title_substring, title_aliases=cfg.window.title_aliases or None
    )
    if win is None:
        print("找不到遊戲視窗（請先開啟星城 Online，餘額顯示在左下角）。")
        return 1
    win = refresh_game_window(win.hwnd, win.title)
    frame = capture_client(win)
    h, w = frame.shape[:2]
    out = logs_dir() / "grab"
    out.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    full_path = out / f"frame_{ts}.png"
    Image.fromarray(np.asarray(frame)).save(str(full_path))
    print(f"已存全畫面：{full_path}  ({w}x{h})")

    rect_ref = cfg.roi.get("balance")
    if rect_ref:
        x, y, bw, bh = scale_rect(
            list(rect_ref), cfg.window.reference_width, cfg.window.reference_height, w, h
        )
        crop = frame[y : y + bh, x : x + bw]
        if crop.size:
            crop_path = out / f"balance_crop_{ts}.png"
            Image.fromarray(np.asarray(crop)).save(str(crop_path))
            print(f"已存餘額框裁切：{crop_path}  目前 roi.balance={list(rect_ref)}")
        # 在全畫面上把餘額框畫紅框另存，方便看框位準不準
        marked = np.asarray(frame).copy()
        marked[y : y + 2, x : x + bw] = [255, 0, 0]
        marked[y + bh - 2 : y + bh, x : x + bw] = [255, 0, 0]
        marked[y : y + bh, x : x + 2] = [255, 0, 0]
        marked[y : y + bh, x + bw - 2 : x + bw] = [255, 0, 0]
        marked_path = out / f"frame_marked_{ts}.png"
        Image.fromarray(marked).save(str(marked_path))
        print(f"已存標記圖（紅框=餘額框）：{marked_path}")
    else:
        print("config 沒有 roi.balance。")

    print("\n請把 logs\\grab 這個資料夾（或裡面的 PNG）傳回，我會據此調框位與前處理。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
