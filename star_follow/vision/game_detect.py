from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from star_follow.config import AppConfig
from star_follow.vision.menu_match import match_template_in_region
from star_follow.vision.roi import scale_point, scale_rect
from star_follow.vision.state import CountdownColor, CountdownState, read_countdown

_T_ROOM_SWITCH = "room_switch.png"
_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
# 換桌圖示只會出現在牌桌右上角這一帶（1280×720）
_REF_SWITCH_SEARCH = (1000, 4, 275, 108)
# 右側棋牌分頁（用來排除首頁大廳）
_REF_QIPAI_TAB = (1130, 335, 150, 95)


def is_lobby(frame: np.ndarray, cfg: AppConfig | None = None) -> bool:
    """百家樂入口：中央黃字說明，且尚無牌桌籌碼列／倒數區特徵。"""
    h, w = frame.shape[:2]
    if _has_chip_bar(frame):
        return False
    if cfg is not None and _has_countdown_area(frame, cfg):
        return False
    center = frame[int(h * 0.25) : int(h * 0.72), int(w * 0.18) : int(w * 0.72)]
    hsv = cv2.cvtColor(center, cv2.COLOR_RGB2HSV)
    yellow = cv2.inRange(hsv, np.array([15, 70, 100]), np.array([45, 255, 255]))
    ratio = yellow.sum() / 255 / max(1, center.shape[0] * center.shape[1])
    return ratio > 0.032


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


def qipai_sidebar_tab_state(
    frame: np.ndarray,
    tab_rect: tuple[int, int, int, int],
) -> tuple[str, float, float]:
    """右側『棋牌』分頁：未選中偏紫、選中會變亮/金/青。

    回傳 (狀態, 紫色占比, 高亮占比)。狀態：unselected | selected | unclear
    """
    x, y, w, h = tab_rect
    if w <= 0 or h <= 0:
        return "unclear", 0.0, 0.0
    roi = frame[y : y + h, x : x + w]
    if roi.size == 0:
        return "unclear", 0.0, 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    purple = cv2.inRange(hsv, np.array([108, 45, 55]), np.array([158, 255, 255]))
    warm = cv2.inRange(hsv, np.array([5, 70, 110]), np.array([38, 255, 255]))
    cool = cv2.inRange(hsv, np.array([78, 45, 110]), np.array([102, 255, 255]))
    n = max(1, roi.shape[0] * roi.shape[1])
    purple_r = float(purple.sum() / 255 / n)
    highlight_r = float((warm | cool).sum() / 255 / n)
    if purple_r >= 0.18 and highlight_r < 0.09:
        return "unselected", purple_r, highlight_r
    if highlight_r >= 0.08 and purple_r < 0.14:
        return "selected", purple_r, highlight_r
    if highlight_r >= 0.12 and highlight_r > purple_r * 1.2:
        return "selected", purple_r, highlight_r
    return "unclear", purple_r, highlight_r


def room_switch_template_path() -> Path:
    return _TEMPLATES_DIR / _T_ROOM_SWITCH


def has_room_switch_template() -> bool:
    return room_switch_template_path().is_file()


class _ClientGeom:
    """僅供 ROI 縮放，無需真實 hwnd。"""

    def __init__(self, client_width: int, client_height: int) -> None:
        self.client_width = client_width
        self.client_height = client_height


def _win_from_frame(frame: np.ndarray, win: _ClientGeom | None) -> _ClientGeom:
    h, w = frame.shape[:2]
    return win if win is not None else _ClientGeom(w, h)


def read_countdown_from_frame(frame: np.ndarray, cfg: AppConfig) -> CountdownState:
    cd = cfg.roi.get("countdown")
    if not cd or len(cd) != 4:
        return CountdownState(color=CountdownColor.OTHER, seconds=None, confidence=0.0)
    h, w = frame.shape[:2]
    x, y, bw, bh = scale_rect(
        cd, cfg.window.reference_width or 1280, cfg.window.reference_height or 720, w, h
    )
    patch = frame[y : y + bh, x : x + bw]
    if patch.size == 0:
        return CountdownState(color=CountdownColor.OTHER, seconds=None, confidence=0.0)
    return read_countdown(patch)


def countdown_ocr_in_room(frame: np.ndarray, cfg: AppConfig) -> tuple[bool, dict]:
    """中央倒數 OCR 讀到 0~30 秒且為紅/綠 → 在牌桌內（大廳不會有）。"""
    st = read_countdown_from_frame(frame, cfg)
    ok = (
        st.seconds is not None
        and st.color in (CountdownColor.GREEN, CountdownColor.RED)
        and st.confidence >= 0.15
    )
    return ok, {
        "seconds": st.seconds,
        "color": st.color.value if st.color else None,
        "ocr_conf": st.confidence,
    }


def _read_valid_table_no(
    frame: np.ndarray, cfg: AppConfig, win: _ClientGeom | None
) -> int | None:
    from star_follow.automation.room_nav import read_current_table

    return read_current_table(frame, cfg, _win_from_frame(frame, win))


def _sidebar_looks_like_qipai_hall(frame: np.ndarray, cfg: AppConfig) -> bool:
    """右側棋牌分頁明顯亮起（金/亮占比夠高），且非入口黃字畫面。"""
    if is_lobby(frame, cfg):
        return False
    h, w = frame.shape[:2]
    ref_w = cfg.window.reference_width or 1280
    ref_h = cfg.window.reference_height or 720
    tab_rect = scale_rect(list(_REF_QIPAI_TAB), ref_w, ref_h, w, h)
    tab, purple_r, highlight_r = qipai_sidebar_tab_state(frame, tab_rect)
    if tab != "selected":
        return False
    return highlight_r >= 0.10 and purple_r < 0.16


def _table_hud_without_ocr(frame: np.ndarray, cfg: AppConfig) -> bool:
    """倒數區有圓形對比 + 底部籌碼列（載入中 OCR 尚未讀到秒數時用）。"""
    if not (_has_countdown_area(frame, cfg) and _has_chip_bar(frame)):
        return False
    return not _sidebar_looks_like_qipai_hall(frame, cfg)


def detect_in_baccarat_room(
    frame: np.ndarray,
    cfg: AppConfig,
    win: _ClientGeom | None = None,
) -> tuple[bool, dict]:
    """是否在百家樂牌桌內（與舊版 is_baccarat_table 類似，但排除明確大廳／入口）。"""
    meta: dict = {"method": "none"}
    from star_follow.vision.kick_popup import is_kick_idle_popup

    if is_kick_idle_popup(frame, cfg, win=win):
        meta["reason"] = "kick_popup"
        return False, meta

    if is_lobby(frame, cfg):
        if _table_hud_without_ocr(frame, cfg):
            meta = {"method": "table_hud"}
            return True, meta
        meta["reason"] = "entry_yellow"
        return False, meta

    # 先確認牌桌特徵；牌桌畫面右側常誤判「棋牌已選中」，不可先否決
    sw_ok, sw_meta = detect_room_switch_button(frame, cfg)
    if sw_ok:
        meta = {"method": "room_switch", **sw_meta}
        return True, meta

    cd_ok, cd_meta = countdown_ocr_in_room(frame, cfg)
    if cd_ok:
        meta = {"method": "countdown_ocr", **cd_meta}
        return True, meta

    table_no = _read_valid_table_no(frame, cfg, win)
    if table_no is not None:
        meta = {"method": "table_no_ocr", "table_no": table_no}
        return True, meta

    if _has_chip_bar(frame):
        h, w = frame.shape[:2]
        ref_w = cfg.window.reference_width or 1280
        ref_h = cfg.window.reference_height or 720
        tab_rect = scale_rect(list(_REF_QIPAI_TAB), ref_w, ref_h, w, h)
        tab, purple_r, _ = qipai_sidebar_tab_state(frame, tab_rect)
        # 首頁底部商城列也會觸發籌碼列特徵
        if tab == "unselected" and purple_r >= 0.15:
            pass
        elif not _sidebar_looks_like_qipai_hall(frame, cfg):
            meta = {"method": "chip_bar"}
            return True, meta

    if _sidebar_looks_like_qipai_hall(frame, cfg):
        meta["reason"] = "qipai_hall"
        return False, meta

    return False, meta


def detect_room_switch_button(frame: np.ndarray, cfg: AppConfig) -> tuple[bool, dict]:
    """僅用 room_switch.png 模板（在右上角帶搜尋）。無模板或比不中 → 不算牌桌。

    已停用灰階對比 fallback（大廳右上按鈕也會觸發）。
    """
    h, w = frame.shape[:2]
    ref_w = cfg.window.reference_width or 1280
    ref_h = cfg.window.reference_height or 720
    meta: dict = {"switch_score": 0.0, "method": "none"}

    if not has_room_switch_template():
        meta["method"] = "no_template"
        return False, meta

    search = scale_rect(list(_REF_SWITCH_SEARCH), ref_w, ref_h, w, h)
    hit = match_template_in_region(frame, _T_ROOM_SWITCH, search, threshold=0.48)
    if not hit or hit[2] < 0.48:
        return False, meta

    meta["switch_score"] = float(hit[2])
    meta["method"] = "template"
    meta["hit_xy"] = (hit[0], hit[1])

    rs = cfg.click_points.get("room_switch_button")
    if rs and len(rs) == 2:
        cx, cy = scale_point(rs, ref_w, ref_h, w, h)
        tol_x = max(55, int(70 * w / ref_w))
        tol_y = max(40, int(55 * h / ref_h))
        if abs(hit[0] - cx) > tol_x or abs(hit[1] - cy) > tol_y:
            meta["method"] = "template_far"
            return False, meta

    return True, meta


def is_at_table_for_nav(
    frame: np.ndarray, cfg: AppConfig, win: _ClientGeom | None = None
) -> tuple[bool, dict]:
    """回桌／引擎：換桌模板、倒數 OCR、左下桌號、籌碼列等任一成立即在房內。"""
    return detect_in_baccarat_room(frame, cfg, win)


def detect_table_hud(frame: np.ndarray, cfg: AppConfig) -> tuple[bool, dict]:
    """相容舊名稱：等同 detect_room_switch_button。"""
    return detect_room_switch_button(frame, cfg)


def is_qipai_hall(frame: np.ndarray, cfg: AppConfig | None = None) -> bool:
    """棋牌大廳：分頁亮起且確定不在牌桌內。"""
    if cfg is None:
        from star_follow.config import load_config

        cfg = load_config()
    if detect_in_baccarat_room(frame, cfg)[0]:
        return False
    return _sidebar_looks_like_qipai_hall(frame, cfg)


def is_baccarat_table(frame: np.ndarray, cfg: AppConfig, win: _ClientGeom | None = None) -> bool:
    return detect_in_baccarat_room(frame, cfg, win)[0]


def is_baccarat_table_strict(frame: np.ndarray, cfg: AppConfig, win: _ClientGeom | None = None) -> bool:
    return is_at_table_for_nav(frame, cfg, win)[0]
