r"""用滑鼠拉框標出「餘額」位置，存回 config.yaml 的 roi.balance（不需 PyQt，用 OpenCV）。

用法（先進 venv 或直接用 venv 的 python）：
    # 直接抓現在的遊戲畫面來框：
    python -m star_follow.tools.mark_balance
    # 或對一張存好的全畫面截圖框：
    python -m star_follow.tools.mark_balance --image logs\grab\frame_xxx.png

操作：
    1) 跳出視窗顯示畫面 → 用滑鼠在「餘額數字」上拉一個框（框緊一點，左邊不要切到第一個數字）
    2) 按 Enter 或空白鍵確認；按 c 取消
    3) 會印出換算後的座標、存裁切圖、用目前 OCR 試讀一次給你看
    4) 問你要不要寫回 config.yaml（y 寫入 / n 只顯示不存）
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
from PIL import Image

from star_follow.config import load_config, save_config
from star_follow.paths import logs_dir


def _load_frame(cfg, image: str | None) -> np.ndarray | None:
    if image:
        bgr = cv2.imread(image)
        if bgr is None:
            print(f"讀不到圖檔：{image}")
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    from star_follow.capture.screen import capture_client
    from star_follow.capture.window import find_game_window, refresh_game_window

    win = find_game_window(
        cfg.window.title_substring, title_aliases=cfg.window.title_aliases or None
    )
    if win is None:
        print("找不到遊戲視窗（請先開好星城 Online，或改用 --image 指定截圖）。")
        return None
    win = refresh_game_window(win.hwnd, win.title)
    return np.asarray(capture_client(win))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="改用存好的全畫面截圖")
    args = parser.parse_args()

    cfg = load_config()
    frame = _load_frame(cfg, args.image)
    if frame is None:
        return 1
    h, w = frame.shape[:2]
    ref_w = cfg.window.reference_width or 1280
    ref_h = cfg.window.reference_height or 720

    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # 顯示目前的 roi.balance（黃框）當參考
    cur = cfg.roi.get("balance")
    if cur:
        from star_follow.vision.roi import scale_rect

        cx, cy, cw, ch = scale_rect(list(cur), ref_w, ref_h, w, h)
        cv2.rectangle(bgr, (cx, cy), (cx + cw, cy + ch), (0, 215, 255), 1)

    print("拉框：用滑鼠框住『餘額數字』，按 Enter/空白確認，按 c 取消。")
    r = cv2.selectROI("mark balance (Enter=OK, c=cancel)", bgr, showCrosshair=True)
    cv2.destroyAllWindows()
    x, y, bw, bh = (int(v) for v in r)
    if bw <= 0 or bh <= 0:
        print("沒有框選（已取消）。")
        return 1

    # 像素 → 參考座標（scale_rect 的反運算）
    ref_rect = [
        int(round(x * ref_w / w)),
        int(round(y * ref_h / h)),
        int(round(bw * ref_w / w)),
        int(round(bh * ref_h / h)),
    ]
    print(f"\n框選像素 = [{x}, {y}, {bw}, {bh}]（畫面 {w}x{h}）")
    print(f"換算 roi.balance（參考 {ref_w}x{ref_h}）= {ref_rect}")

    out = logs_dir() / "grab"
    out.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    crop = frame[y : y + bh, x : x + bw]
    if crop.size:
        crop_path = out / f"mark_balance_{ts}.png"
        Image.fromarray(crop).save(str(crop_path))
        print(f"已存框選裁切：{crop_path}")
        try:
            from star_follow.vision.ocr import ocr_balance

            val, conf = ocr_balance(crop)
            print(f"用目前 OCR 試讀這個框 → {val:,}（conf={conf}）")
        except Exception as exc:  # noqa: BLE001
            print(f"試讀失敗：{exc}")

    try:
        ans = input("\n要把這個框寫回 config.yaml 的 roi.balance 嗎？(y/n) ").strip().lower()
    except Exception:  # noqa: BLE001
        ans = "n"
    if ans == "y":
        cfg.raw.setdefault("roi", {})["balance"] = ref_rect
        cfg.roi["balance"] = ref_rect
        p = save_config(cfg)
        print(f"已寫入 {p} 的 roi.balance = {ref_rect}")
    else:
        print("未寫入。可把上面的座標與裁切圖傳回，我幫你設定。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
