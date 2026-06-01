from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from .ocr import ocr_digits


class CountdownColor(str, Enum):
    GREEN = "green"
    RED = "red"
    OTHER = "other"


@dataclass
class CountdownState:
    color: CountdownColor
    seconds: int | None
    confidence: float
    status_text: str | None = None


def _dominant_color_hue(roi: np.ndarray) -> CountdownColor:
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    mask = (s > 60) & (v > 60)
    if int(mask.sum()) < 20:
        return CountdownColor.OTHER
    hues = h[mask]
    mean_h = float(np.mean(hues))
    # 綠 ≈ 35–85；紅 ≈ 0–15 或 165–180
    if 35 <= mean_h <= 90:
        return CountdownColor.GREEN
    if mean_h <= 15 or mean_h >= 165:
        return CountdownColor.RED
    return CountdownColor.OTHER


def read_countdown(roi_bgr: np.ndarray) -> CountdownState:
    color = _dominant_color_hue(roi_bgr)
    if color not in (CountdownColor.GREEN, CountdownColor.RED):
        return CountdownState(color=color, seconds=None, confidence=0.0)

    digits, conf = ocr_digits(roi_bgr, psm=8)
    seconds: int | None = None
    if digits.isdigit():
        val = int(digits)
        if 0 <= val <= 30:
            seconds = val
    return CountdownState(color=color, seconds=seconds, confidence=conf)
