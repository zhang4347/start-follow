"""玩家暱稱「影像比對」：用事先做好的名字樣板，在統計表各欄表頭裡找出對象那一欄。

動機：少數暱稱（藝術字／反白）Tesseract 怎麼加強都讀不準，且逐字辨識又慢。
對象名單已知且每台只追 1 隻，與其「辨識文字」，不如「比對長相」——比的是名字的
影像樣式，純 OpenCV（毫秒級、不開子行程），而且免疫 OCR 認錯（噓→嚏、整組崩）。

樣板由開發端用名字截圖製作，放在 data/name_templates/：
  - index.json：{ "暱稱": "檔名.png", ... }（用 ASCII 檔名避免中文路徑讀檔問題）
  - 對應的圖檔（名字那一格的裁切圖）

沒有任何樣板時，本模組所有查詢都回傳「無」，呼叫端照舊走 OCR，行為不變。
"""

from __future__ import annotations

import json
import logging
import re

import cv2
import numpy as np

from star_follow import paths

logger = logging.getLogger(__name__)

# TM_CCOEFF_NORMED 分數門檻與「領先第二名」的最小差距（沿用名字比對的精準優先精神）。
# 可由設定檔覆寫（config.yaml / 啟動設定.txt → set_match_params）。樣板與實機渲染有差
# 時可調低門檻；若出現跟錯人就調高。
_MATCH_THRESHOLD = 0.55
_MATCH_MARGIN = 0.04


def set_match_params(threshold: float | None = None, margin: float | None = None) -> None:
    """以設定檔的值覆寫影像比對門檻／領先差（啟動時呼叫一次）。"""
    global _MATCH_THRESHOLD, _MATCH_MARGIN
    if threshold is not None:
        _MATCH_THRESHOLD = float(threshold)
    if margin is not None:
        _MATCH_MARGIN = float(margin)
    logger.info("影像比對門檻=%.2f 領先差=%.2f", _MATCH_THRESHOLD, _MATCH_MARGIN)
# 比對前統一縮放到的高度（樣板與各欄表頭都先正規化到同一高度，吸收些微縮放差異）。
_NORM_H = 40

_cache: dict[str, np.ndarray] | None = None
_cache_sig: str | None = None


def _norm_name(s: str) -> str:
    """與 stats_parser 一致的暱稱正規化：全形轉半形、去非中英數雜訊。"""
    if not s:
        return ""
    out = []
    for ch in s:
        o = ord(ch)
        if o == 0x3000:
            out.append(" ")
        elif 0xFF01 <= o <= 0xFF5E:
            out.append(chr(o - 0xFEE0))
        else:
            out.append(ch)
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", "".join(out))


def _imread_unicode(path) -> np.ndarray | None:
    """支援中文／非 ASCII 路徑讀圖（cv2.imread 在 Windows 對中文路徑會回 None）。"""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _text_mask(img_bgr_or_rgb: np.ndarray) -> np.ndarray | None:
    """把一格名字圖轉成「文字遮罩」：白/亮字 → 1，背景 → 0，再正規化到固定高度。

    比對亮字遮罩（而非原色），可吸收背景顏色、反白高亮、輕微亮度差異。
    """
    if img_bgr_or_rgb is None or img_bgr_or_rgb.size == 0:
        return None
    if img_bgr_or_rgb.ndim == 3:
        gray = cv2.cvtColor(img_bgr_or_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_bgr_or_rgb
    if gray.size == 0 or gray.shape[0] < 4 or gray.shape[1] < 4:
        return None
    # 亮字遮罩：高於 (平均+一點) 視為文字。Otsu 在純背景會亂切，故用相對門檻。
    m = float(gray.mean())
    thr = max(m + 25.0, 130.0)
    mask = (gray >= thr).astype(np.uint8) * 255
    # 先「裁到文字外框」：不同來源的截圖（樣板 vs 實際表頭格）左右/上下留白不一樣，
    # 直接比會因為文字佔比、位置不同而相關係數偏低（實測同一個名字只有 ~0.45）。
    # 裁掉純背景邊界、只留文字方塊，再正規化高度，能讓同名分數大幅拉高、與別名分得開。
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    mask = mask[y0:y1, x0:x1]
    if mask.shape[0] < 4 or mask.shape[1] < 4:
        return None
    # 正規化高度，等比例縮放寬度（保留字數帶來的長寬比，利於區分不同名字）。
    h = mask.shape[0]
    if h != _NORM_H:
        scale = _NORM_H / float(h)
        new_w = max(8, int(round(mask.shape[1] * scale)))
        mask = cv2.resize(mask, (new_w, _NORM_H), interpolation=cv2.INTER_AREA)
    return mask


def _templates_signature() -> str:
    d = paths.name_templates_dir()
    idx = d / "index.json"
    try:
        return f"{d}|{idx.stat().st_mtime_ns}" if idx.is_file() else f"{d}|none"
    except Exception:
        return f"{d}|err"


def load_name_templates() -> dict[str, np.ndarray]:
    """載入所有暱稱樣板（正規化暱稱 → 文字遮罩）。無樣板時回傳空 dict。"""
    global _cache, _cache_sig
    sig = _templates_signature()
    if _cache is not None and _cache_sig == sig:
        return _cache

    out: dict[str, np.ndarray] = {}
    d = paths.name_templates_dir()
    idx = d / "index.json"
    if idx.is_file():
        try:
            mapping = json.loads(idx.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("名字樣板 index.json 讀取失敗：%s", exc)
            mapping = {}
        for name, fname in (mapping or {}).items():
            key = _norm_name(str(name))
            if not key:
                continue
            img = _imread_unicode(d / str(fname))
            mask = _text_mask(img) if img is not None else None
            if mask is None:
                logger.warning("名字樣板載入失敗：%s → %s", name, fname)
                continue
            out[key] = mask
        if out:
            logger.info("已載入 %d 個暱稱影像樣板：%s", len(out), "、".join(mapping.keys()))
    _cache = out
    _cache_sig = sig
    return out


def has_template(name: str) -> bool:
    return _norm_name(name) in load_name_templates()


def _score_cell(cell_mask: np.ndarray, tpl_mask: np.ndarray) -> float:
    """單格 vs 樣板的相似度（0~1）。樣板在格內滑動取最高相關係數。"""
    ch, cw = cell_mask.shape[:2]
    th, tw = tpl_mask.shape[:2]
    # matchTemplate 要求樣板 <= 影像；樣板較寬就縮到略小於格寬。
    if tw > cw:
        scale = (cw - 1) / float(tw)
        tpl_mask = cv2.resize(
            tpl_mask, (max(4, int(tw * scale)), th), interpolation=cv2.INTER_AREA
        )
        th, tw = tpl_mask.shape[:2]
    if th > ch:
        scale = (ch - 1) / float(th)
        tpl_mask = cv2.resize(
            tpl_mask, (tw, max(4, int(th * scale))), interpolation=cv2.INTER_AREA
        )
        th, tw = tpl_mask.shape[:2]
    if th < 4 or tw < 4 or ch < th or cw < tw:
        return 0.0
    try:
        res = cv2.matchTemplate(cell_mask, tpl_mask, cv2.TM_CCOEFF_NORMED)
    except cv2.error:
        return 0.0
    return float(res.max()) if res.size else 0.0


def match_name_column(
    header_cells: list[tuple[int, np.ndarray]],
    name: str,
    *,
    threshold: float | None = None,
    margin: float | None = None,
) -> tuple[int | None, float]:
    """在各欄表頭裡用樣板找出 `name` 的欄位索引。

    回傳 (欄位索引或 None, 最佳分數)。比照名字比對的精準優先：最佳分數需達門檻，
    且明顯勝過第二名才採用（避免比到長相相近的路人）。沒有該名字的樣板 → (None, 0)。
    threshold/margin 留空時用模組目前值（可由 set_match_params 從設定檔覆寫）。
    """
    threshold = _MATCH_THRESHOLD if threshold is None else threshold
    margin = _MATCH_MARGIN if margin is None else margin
    templates = load_name_templates()
    tpl = templates.get(_norm_name(name))
    if tpl is None:
        return None, 0.0
    scored: list[tuple[float, int]] = []
    for idx, cell in header_cells:
        mask = _text_mask(cell)
        if mask is None:
            continue
        scored.append((_score_cell(mask, tpl), idx))
    if not scored:
        return None, 0.0
    scored.sort(key=lambda x: x[0], reverse=True)
    best_s, best_idx = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    if best_s < threshold:
        logger.info(
            "影像比對「%s」最佳分數=%.2f（欄 %d，第二名 %.2f）< 門檻 %.2f → 未採用",
            name, best_s, best_idx, second, threshold,
        )
        return None, best_s
    if second >= best_s - margin:
        logger.info(
            "影像比對「%s」最佳 %.2f 與第二名 %.2f 太接近（差 %.2f<%.2f）→ 不採用避免認錯",
            name, best_s, second, best_s - second, margin,
        )
        return None, best_s
    logger.info("影像比對「%s」命中欄 %d 分數=%.2f（第二名 %.2f）", name, best_idx, best_s, second)
    return best_idx, best_s
