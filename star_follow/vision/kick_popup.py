"""「超過五局未押注」居中提示窗偵測（OCR 為主，避免牌桌 UI 誤觸）。"""

from __future__ import annotations

import cv2
import numpy as np

from star_follow.config import AppConfig
from star_follow.vision.menu_match import match_template_in_region
from star_follow.vision.nav_text import region_has_keywords
from star_follow.vision.roi import scale_point, scale_rect

_T_CONFIRM = "lobby_confirm_button.png"
# 1280×720：居中提示窗本體（含標題列與確定鈕）
_REF_DIALOG = (380, 370, 520, 290)
_REF_MSG_BODY = (400, 400, 480, 120)
_MSG_KEYS = ("五局", "未押注", "退出遊戲", "退出", "未押")
_THR_TEMPLATE = 0.62


def _dialog_rect(frame: np.ndarray, cfg: AppConfig) -> tuple[int, int, int, int]:
    h, w = frame.shape[:2]
    return scale_rect(
        list(_REF_DIALOG),
        cfg.window.reference_width or 1280,
        cfg.window.reference_height or 720,
        w,
        h,
    )


def kick_popup_message_ocr(frame: np.ndarray, cfg: AppConfig) -> tuple[bool, str]:
    h, w = frame.shape[:2]
    body = scale_rect(
        list(_REF_MSG_BODY),
        cfg.window.reference_width or 1280,
        cfg.window.reference_height or 720,
        w,
        h,
    )
    hit, text, _conf = region_has_keywords(frame, body, _MSG_KEYS)
    return hit, text


def is_kick_idle_popup(
    frame: np.ndarray,
    cfg: AppConfig,
    *,
    win: object | None = None,
) -> bool:
    """是否為五局未押注提示窗（須 OCR 到關鍵字，或確定鈕模板高分）。"""
    hit, _text = kick_popup_message_ocr(frame, cfg)
    if hit:
        return True
    x, y, bw, bh = _dialog_rect(frame, cfg)
    tpl = match_template_in_region(frame, _T_CONFIRM, (x, y, bw, bh), threshold=0.0)
    if tpl and float(tpl[2]) >= _THR_TEMPLATE:
        return True
    return False


def find_kick_confirm_xy(
    frame: np.ndarray,
    cfg: AppConfig,
    win: object,
) -> tuple[int, int] | None:
    """僅在 is_kick_idle_popup 為真時呼叫。"""
    pt = cfg.click_points.get("kick_idle_confirm")
    if pt and len(pt) == 2:
        ref_w = cfg.window.reference_width or 1280
        ref_h = cfg.window.reference_height or 720
        cw = int(getattr(win, "client_width", frame.shape[1]))
        ch = int(getattr(win, "client_height", frame.shape[0]))
        return scale_point(list(pt), ref_w, ref_h, cw, ch)
    x, y, bw, bh = _dialog_rect(frame, cfg)
    tpl = match_template_in_region(frame, _T_CONFIRM, (x, y, bw, bh), threshold=0.0)
    if tpl and float(tpl[2]) >= _THR_TEMPLATE:
        return int(tpl[0]), int(tpl[1])
    return None
