"""等待進入百家樂牌桌後自動擷圖。用法: python -m star_follow.tools.wait_table"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from PIL import Image

from star_follow.capture.screen import capture_client
from star_follow.capture.window import find_game_window
from star_follow.config import load_config, save_config
from star_follow.vision.game_detect import is_baccarat_table, is_lobby


def main() -> int:
    cfg = load_config()
    out_dir = Path(__file__).resolve().parents[1] / "reference"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("等待進入百家樂牌桌…（請在遊戲裡點進牌桌）")
    print("提示：請把 Steam 等視窗移開，避免擋住遊戲右側選單。")

    last_lobby = False
    for i in range(600):
        win = find_game_window(cfg.window.title_substring)
        if not win:
            if i % 10 == 0:
                print("  找不到星城視窗…")
            time.sleep(1)
            continue

        frame = capture_client(win)
        h, w = frame.shape[:2]
        cfg.window.reference_width = w
        cfg.window.reference_height = h

        if is_lobby(frame) or not is_baccarat_table(frame, cfg):
            if not last_lobby:
                print(f"  目前在大廳/入口 ({w}x{h})，請點進牌桌…")
                lobby_path = out_dir / "current_lobby.png"
                Image.fromarray(frame).save(lobby_path)
            last_lobby = True
            time.sleep(1)
            continue

        last_lobby = False
        if is_baccarat_table(frame, cfg):
            path = out_dir / "current_table.png"
            Image.fromarray(frame).save(path)
            save_config(cfg)
            print(f"已偵測牌桌！截圖: {path}")
            print(f"視窗客戶區: {w} x {h}（已寫入 config.yaml）")
            print("下一步: python -m star_follow.tools.calibrate --image", path)
            return 0

        time.sleep(0.5)

    print("逾時（10 分鐘）仍未進入牌桌。", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
