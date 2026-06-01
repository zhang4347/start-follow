"""依參考解析度縮放 config ROI，或從牌桌截圖偵測籌碼列。"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
from PIL import Image

from star_follow.config import load_config, save_config


def scale_config(cfg, from_w: int, from_h: int) -> None:
    tw, th = cfg.window.reference_width, cfg.window.reference_height
    sx, sy = tw / from_w, th / from_h

    def sc_rect(r: list[int]) -> list[int]:
        return [int(r[0] * sx), int(r[1] * sy), int(r[2] * sx), int(r[3] * sy)]

    def sc_pt(p: list[int]) -> list[int]:
        return [int(p[0] * sx), int(p[1] * sy)]

    for k in list(cfg.roi.keys()):
        if isinstance(cfg.roi[k], list) and len(cfg.roi[k]) == 4:
            cfg.roi[k] = sc_rect(cfg.roi[k])
    for k in cfg.chips:
        cfg.chips[k] = sc_pt(cfg.chips[k])
    for k in cfg.bet_areas:
        cfg.bet_areas[k] = sc_pt(cfg.bet_areas[k])


def detect_chips(frame: np.ndarray) -> dict[str, list[int]]:
    h, w = frame.shape[:2]
    strip = frame[int(h * 0.78) : int(h * 0.92), int(w * 0.22) : int(w * 0.78)]
    hsv = cv2.cvtColor(strip, cv2.COLOR_RGB2HSV)
    centers: list[tuple[int, int, int]] = []
    for hue_range in [
        (35, 90),   # 1K 綠
        (90, 130),  # 5K 藍
        (130, 165), # 10K 紫
        (165, 180), # 50K 粉
        (0, 12),    # 100K 紅
        (12, 30),   # 500K 金
    ]:
        lo, hi = hue_range
        mask = cv2.inRange(hsv, np.array([lo, 60, 60]), np.array([hi, 255, 255]))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < 200:
                continue
            m = cv2.moments(c)
            if m["m00"] == 0:
                continue
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            centers.append((cx + int(w * 0.22), cy + int(h * 0.78), cv2.contourArea(c)))

    centers.sort(key=lambda t: t[0])
    values = ["1000", "5000", "10000", "50000", "100000", "500000"]
    out: dict[str, list[int]] = {}
    if len(centers) >= 4:
        picked = centers[:6] if len(centers) >= 6 else centers
        for val, (x, y, _) in zip(values[: len(picked)], picked):
            out[val] = [x, y]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--from-w", type=int, default=1024)
    parser.add_argument("--from-h", type=int, default=597)
    args = parser.parse_args()

    cfg = load_config()
    scale_config(cfg, args.from_w, args.from_h)

    if args.image:
        frame = np.array(Image.open(args.image).convert("RGB"))
        chips = detect_chips(frame)
        if len(chips) >= 4:
            cfg.chips.update(chips)
            print("detected chips:", chips)

    path = save_config(cfg)
    print("saved", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
