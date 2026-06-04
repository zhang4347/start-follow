"""☰ 選單圖示：柱狀圖 = 押注統計。"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

@lru_cache(maxsize=8)
def _load_template(name: str) -> np.ndarray | None:
    from star_follow.paths import templates_dir

    path = templates_dir() / name
    if not path.is_file():
        return None
    try:
        return np.array(Image.open(path).convert("RGB"))
    except OSError:
        return None


def dropdown_strip_rect(menu_button_rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """
    ☰ 正下方窄條（不含右側玩家頭像列）。
    1280×720 實測：柱狀圖在按鈕下緣 ~10–40px，玩家頭像從 y≈134 起。
    """
    bx, by, bw, bh = menu_button_rect
    y0 = by + bh - 12
    h = min(50, max(36, bh + 8))
    return bx - 6, y0, bw + 12, h


def find_menu_icon_centers(
    frame: np.ndarray,
    panel_rect: tuple[int, int, int, int],
) -> list[tuple[int, int, float]]:
    """回傳 [(client_x, client_y, area), ...] 由上到下。"""
    x, y, w, h = panel_rect
    if w <= 0 or h <= 0:
        return []
    roi = frame[y : y + h, x : x + w]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw: list[tuple[int, int, float]] = []
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < 80 or area > 6000:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw < 10 or bh < 10 or bw > 70 or bh > 70:
            continue
        if bw / max(bh, 1) > 2.5:
            continue
        raw.append((x + bx + bw // 2, y + by + bh // 2, area))
    raw.sort(key=lambda t: t[1])

    merged: list[tuple[int, int, float]] = []
    for cx, cy, area in raw:
        if merged and abs(cy - merged[-1][1]) < 28:
            if area > merged[-1][2]:
                merged[-1] = (cx, cy, area)
        else:
            merged.append((cx, cy, area))
    return merged


def find_dropdown_icons(
    frame: np.ndarray,
    menu_button_rect: tuple[int, int, int, int],
) -> list[tuple[int, int, float]]:
    """僅在 ☰ 正下方窄條找圖示，排除右側玩家頭像。"""
    bx, by, bw, bh = menu_button_rect
    cx = bx + bw // 2
    y_max = by + bh + 44
    strip = dropdown_strip_rect(menu_button_rect)
    icons = find_menu_icon_centers(frame, strip)
    out: list[tuple[int, int, float]] = []
    for ix, iy, area in icons:
        if abs(ix - cx) > 28:
            continue
        if iy > y_max:
            continue
        if iy < by - 4:
            continue
        out.append((ix, iy, area))
    out.sort(key=lambda t: t[1])
    return out


def match_template_in_region(
    frame: np.ndarray,
    template_name: str,
    search_rect: tuple[int, int, int, int],
    *,
    threshold: float = 0.55,
    base_scale: float = 1.0,
) -> tuple[int, int, float] | None:
    """在 search_rect 內多尺度比對模板。

    base_scale：模板是以參考解析度（1280×720）擷取的；若實際視窗較大/較小，傳入
    實際/參考的比例（例如 0.8），讓模板先依視窗縮放，再做 ±微調，避免不同解析度比不中。
    """
    tpl = _load_template(template_name)
    if tpl is None or tpl.size == 0:
        return None
    sx, sy, sw, sh = search_rect
    if sw <= 0 or sh <= 0:
        return None
    hay = frame[sy : sy + sh, sx : sx + sw]
    th, tw = tpl.shape[:2]

    best: tuple[int, int, float] | None = None
    for rel in (1.0, 0.85, 1.15, 0.7, 1.3):
        scale = base_scale * rel
        nw, nh = max(8, int(tw * scale)), max(8, int(th * scale))
        tpl_s = (
            tpl if (nw == tw and nh == th)
            else cv2.resize(tpl, (nw, nh), interpolation=cv2.INTER_LINEAR)
        )
        ths, tws = tpl_s.shape[:2]
        if hay.shape[0] < ths or hay.shape[1] < tws:
            continue
        res = cv2.matchTemplate(hay, tpl_s, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val >= threshold and (best is None or max_val > best[2]):
            x, y = max_loc
            best = (sx + x + tws // 2, sy + y + ths // 2, float(max_val))
    return best


def effective_menu_button_rect(
    roi_rect: tuple[int, int, int, int],
    *,
    menu_button_pt: tuple[int, int] | None = None,
    menu_chart_pt: tuple[int, int] | None = None,
) -> tuple[int, int, int, int]:
    """依手動標記推估 ☰ 區域（config ROI 常與實際 UI 不符）。"""
    if menu_button_pt:
        x, y = menu_button_pt
        return x - 20, y - 20, 40, 40
    if menu_chart_pt:
        cx, cy = menu_chart_pt
        return cx - 20, cy - 78, 40, 40
    return roi_rect


def hamburger_click_candidates(
    roi_rect: tuple[int, int, int, int],
    *,
    menu_button_pt: tuple[int, int] | None = None,
    menu_chart_pt: tuple[int, int] | None = None,
) -> list[tuple[int, int, str]]:
    """☰ 點擊候選（含標記附近微調、由柱狀圖反推）。"""
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int, str]] = []

    def add(x: int, y: int, tag: str) -> None:
        k = (x // 2, y // 2)
        if k in seen:
            return
        seen.add(k)
        out.append((x, y, tag))

    if menu_button_pt:
        mx, my = menu_button_pt
        add(mx, my, "click_menu_button")
        for dy in (8, 16, 24, 32, -6, -12):
            add(mx, my + dy, f"menu_button_dy{dy:+d}")

    if menu_chart_pt:
        cx, cy = menu_chart_pt
        for gap in (50, 58, 66, 72, 80):
            add(cx, cy - gap, f"above_chart_{gap}")

    bx, by, bw, bh = roi_rect
    add(bx + bw // 2, by + bh // 2, "roi_center")
    return out


def chart_icon_candidates(
    frame: np.ndarray,
    menu_button_rect: tuple[int, int, int, int],
    menu_panel_rect: tuple[int, int, int, int] | None = None,
    *,
    fixed_chart: tuple[int, int] | None = None,
    stats_option_rect: tuple[int, int, int, int] | None = None,
) -> list[tuple[int, int, float, str]]:
    """
    柱狀圖點擊候選。若 config 有 menu_chart 手動標記，只使用該點。
    """
    del menu_panel_rect, stats_option_rect

    if fixed_chart:
        return [(fixed_chart[0], fixed_chart[1], 10.0, "config_click_point")]

    bx, by, bw, bh = menu_button_rect
    cx = bx + bw // 2
    tiers: list[tuple[int, int, float, str]] = []
    seen: set[tuple[int, int]] = set()

    def add(x: int, y: int, score: float, tag: str) -> None:
        key = (x // 4, y // 4)
        if key in seen:
            return
        seen.add(key)
        tiers.append((x, y, score, tag))

    # 以下為自動推算（無手動標記時）
    add(cx, by + bh + 12, 3.0, "geom_primary")
    add(cx, by + bh + 22, 2.9, "geom_secondary")
    add(cx, by + bh + 32, 2.8, "geom_tertiary")

    strip = dropdown_strip_rect(menu_button_rect)
    for name, th in (("menu_chart.png", 0.38), ("menu_chart_crop.png", 0.35)):
        hit = match_template_in_region(frame, name, strip, threshold=th)
        if hit:
            add(hit[0], hit[1], 2.5 + hit[2], f"template_{name}")

    dropdown = find_dropdown_icons(frame, menu_button_rect)
    for i, (ix, iy, _area) in enumerate(dropdown):
        add(ix, iy, 2.4 - i * 0.1, f"strip_icon_{i}")

    tiers.sort(key=lambda t: -t[2])
    return tiers


def menu_dropdown_open(frame: np.ndarray, menu_button_rect: tuple[int, int, int, int]) -> bool:
    """☰ 展開後，按鈕正下方窄條會出現 2 個以上小圖示。"""
    bx, by, bw, bh = menu_button_rect
    icons = find_dropdown_icons(frame, menu_button_rect)
    below = [t for t in icons if t[1] >= by + bh - 8]
    return len(below) >= 2


def find_chart_icon_click(
    frame: np.ndarray,
    menu_button_rect: tuple[int, int, int, int],
    menu_panel_rect: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, float] | None:
    cands = chart_icon_candidates(frame, menu_button_rect, menu_panel_rect)
    if not cands:
        return None
    x, y, score, _ = cands[0]
    return x, y, score
