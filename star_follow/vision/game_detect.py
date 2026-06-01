from __future__ import annotations

import cv2
import numpy as np

from star_follow.config import AppConfig
from star_follow.vision.roi import scale_rect


def is_lobby(frame: np.ndarray) -> bool:
    """百家樂入口／大廳：中央大片黃字說明、尚無牌桌籌碼列。"""
    h, w = frame.shape[:2]
    if _has_chip_bar(frame):
        return False
    center = frame[int(h * 0.25) : int(h * 0.72), int(w * 0.18) : int(w * 0.72)]
    hsv = cv2.cvtColor(center, cv2.COLOR_RGB2HSV)
    yellow = cv2.inRange(hsv, np.array([15, 70, 100]), np.array([45, 255, 255]))
    ratio = yellow.sum() / 255 / max(1, center.shape[0] * center.shape[1])
    return ratio > 0.018


def _has_chip_bar(frame: np.ndarray) -> bool:
    """牌桌底部 1K~500K 籌碼列（多種高飽和色）。"""
    h, w = frame.shape[:2]
    strip = frame[int(h * 0.76) : int(h * 0.92), int(w * 0.22) : int(w * 0.78)]
    hsv = cv2.cvtColor(strip, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = (sat > 80) & (val > 80)
    if mask.sum() < strip.shape[0] * strip.shape[1] * 0.04:
        return False
    # 籌碼列橫向分佈較廣
    col = mask.sum(axis=0)
    active = (col > strip.shape[0] * 0.15).sum()
    return active >= 4


def _has_countdown_area(frame: np.ndarray, cfg: AppConfig) -> bool:
    h, w = frame.shape[:2]
    cd = cfg.roi.get("countdown")
    if not cd:
        cx0, cy0, cw, ch = int(w * 0.42), int(h * 0.02), int(w * 0.16), int(h * 0.14)
    else:
        cx0, cy0, cw, ch = scale_rect(
            cd, cfg.window.reference_width, cfg.window.reference_height, w, h
        )
    top = frame[cy0 : cy0 + ch, cx0 : cx0 + cw]
    if top.size == 0:
        return False
    gray = cv2.cvtColor(top, cv2.COLOR_RGB2GRAY)
    # 圓形倒數：高對比圓形區域
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th.std() > 35


def is_baccarat_table(frame: np.ndarray, cfg: AppConfig) -> bool:
    if is_lobby(frame):
        return False
    return _has_chip_bar(frame) or _has_countdown_area(frame, cfg)
