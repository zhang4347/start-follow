from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image

from star_follow import paths as _paths

_PKG = Path(__file__).resolve().parents[1]
_TESS_EXE, _TESSDATA = _paths.ocr_runtime()
if _TESSDATA.is_dir():
    os.environ["TESSDATA_PREFIX"] = str(_TESSDATA)
if _TESS_EXE.is_file():
    pytesseract.pytesseract.tesseract_cmd = str(_TESS_EXE)

_DIGITS = re.compile(r"[^\d]")
_OCR_SCALE = 1
_FAST = True


def set_ocr_options(*, fast: bool = True, scale: int = 1) -> None:
    global _FAST, _OCR_SCALE
    _FAST = fast
    _OCR_SCALE = max(1, scale)


def preprocess_stats_cell(img: np.ndarray, scale: int | None = None) -> np.ndarray:
    sc = scale if scale is not None else _OCR_SCALE
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if sc > 1:
        gray = cv2.resize(gray, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
    _, th = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
    return th


def preprocess_text(img: np.ndarray, scale: int = 2) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if scale > 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    return gray


def ocr_digits(img: np.ndarray, psm: int = 8) -> tuple[str, float]:
    proc = preprocess_stats_cell(img, 2 if not _FAST else 1)
    config = f"--psm {psm} -c tessedit_char_whitelist=0123456789"
    raw = pytesseract.image_to_string(Image.fromarray(proc), config=config).strip()
    digits = _DIGITS.sub("", raw)
    return digits, 0.8 if digits else 0.0


def _cell_may_have_digits(img: np.ndarray) -> bool:
    """快速判斷格子是否可能有數字，跳過空白格不呼叫 Tesseract。"""
    if img.size == 0:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return int((gray < 115).sum()) >= 25


def ocr_amount(img: np.ndarray) -> tuple[int, float]:
    if img.size == 0:
        return 0, 0.0
    if not _cell_may_have_digits(img):
        return 0, 0.0
    proc = preprocess_stats_cell(img)
    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(Image.fromarray(proc), config=config).strip()
    text = _DIGITS.sub("", text.replace(",", "").replace("，", ""))
    if not text:
        return 0, 0.0
    try:
        return int(text), 0.85
    except ValueError:
        return 0, 0.0


def _ocr_digit_string(th: np.ndarray) -> str:
    """對「黑字白底」二值圖，用單行 psm 7（純數字白名單）讀整串數字。"""
    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    try:
        raw = pytesseract.image_to_string(Image.fromarray(th), config=config)
    except pytesseract.TesseractError:
        return ""
    return _DIGITS.sub("", raw)


def _normalize_glyph(sub: np.ndarray) -> np.ndarray | None:
    """單一數字（白字黑底）正規化：緊裁字形 → 統一高度 64 → 加粗筆畫 → 留白 → 反白。

    Tesseract 對小且細的字辨識不穩（7→1、5/8/9 混淆）。把每個字裁緊、放大到固定高度
    再稍微加粗，單字元辨識可大幅變準（實測這步讓逐位辨識全對）。
    """
    ys, xs = np.where(sub > 0)
    if len(xs) == 0:
        return None
    sub = sub[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h = 64
    w = max(1, int(round(sub.shape[1] * 64 / sub.shape[0])))
    sub = cv2.resize(sub, (w, h), interpolation=cv2.INTER_CUBIC)
    sub = cv2.dilate(sub, np.ones((3, 3), np.uint8), iterations=1)
    sub = cv2.copyMakeBorder(sub, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=0)
    return cv2.bitwise_not(sub)  # 白字黑底 → 黑字白底


def _ocr_per_digit(clean: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> str:
    """把每個數字塊正規化後單獨送 psm 10（單字元）辨識，逐位組回字串。

    clean：白字黑底的乾淨數字遮罩（已濾掉逗號）。boxes：由左到右的數字外框。
    每塊正規化高度＋加粗，沒有版面/逗號干擾，對小字、5/8/9/7 混淆更穩。
    """
    config = "--psm 10 -c tessedit_char_whitelist=0123456789"
    out = []
    for (x, y, w, h) in boxes:
        sub = clean[y:y + h, x:x + w]
        th = _normalize_glyph(sub)
        if th is None:
            out.append("?")
            continue
        try:
            raw = pytesseract.image_to_string(Image.fromarray(th), config=config)
        except pytesseract.TesseractError:
            raw = ""
        d = _DIGITS.sub("", raw)
        out.append(d[0] if d else "?")  # 單字元只取第一個數字；讀不到記 ?
    return "".join(out)


def _ocr_balance_gray(img: np.ndarray) -> tuple[int, float]:
    """退路：黃色遮罩抓不到時，用舊的灰階二值法（含逗號白名單）。"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
    th = cv2.bitwise_not(th)
    best = _ocr_digit_string(th)
    if not best:
        return 0, 0.0
    try:
        return int(best), 0.6
    except ValueError:
        return 0, 0.0


def ocr_balance(img: np.ndarray) -> tuple[int, float]:
    """讀左下角餘額（亮黃數字、深色底，格式像 7,273,414）。

    做法（針對固定字體遊戲 UI）：
      1) 用黃色遮罩只抽出數字，背景全部清掉（比轉灰階乾淨非常多）。
      2) 連通元件依「高度」濾掉逗號與雜訊 → 根本不讓逗號進辨識（解決多一位/少一位）。
      3) 整串 psm 7 讀一次；若位數和逐位辨識對不上，採逐位 psm 10 的結果。
    黃色抓不到時退回灰階法。回傳 (金額, 信心)。讀不到回 (0, 0.0)。
    """
    if img is None or img.size == 0:
        return 0, 0.0
    rgb = img if (img.ndim == 3 and img.shape[2] == 3) else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # 黃色：H≈15~45、S/V 偏高（涵蓋亮黃到偏橘黃）
    mask = cv2.inRange(hsv, np.array([15, 60, 90], np.uint8), np.array([45, 255, 255], np.uint8))
    if int(cv2.countNonZero(mask)) < 20:
        return _ocr_balance_gray(img)

    scale = 3
    mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    # 補小縫，避免一個數字被切成兩塊
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = []  # (x, y, w, h, label)
    for i in range(1, n):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 8:
            continue
        comps.append((x, y, w, h, i))
    if not comps:
        return _ocr_balance_gray(img)

    heights = sorted(c[3] for c in comps)
    med_h = heights[len(heights) // 2]
    # 數字高度接近中位數；逗號/小數點又矮又小（高度遠小於中位數）→ 丟掉
    digits = [c for c in comps if c[3] >= 0.6 * med_h]
    if not digits:
        return _ocr_balance_gray(img)
    digits.sort(key=lambda c: c[0])  # 由左到右

    keep_labels = {c[4] for c in digits}
    clean = np.where(np.isin(labels, list(keep_labels)), np.uint8(255), np.uint8(0))
    th = cv2.bitwise_not(clean)
    th = cv2.copyMakeBorder(th, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)

    whole = _ocr_digit_string(th)
    boxes = [(c[0], c[1], c[2], c[3]) for c in digits]
    per = _ocr_per_digit(clean, boxes)  # 長度 == 塊數，讀不到的位置記 '?'
    n_digits = len(digits)

    # 逐位辨識（已正規化＋加粗）最可靠，且位數由切割保證；整串只當交叉驗證/補洞。
    best = ""
    conf = 0.0
    if "?" not in per:
        best = per
        conf = 0.95 if (whole == per) else 0.9
    elif whole and len(whole) == n_digits and whole.isdigit():
        # 逐位有讀不到的洞，但整串位數對得上 → 用整串
        best = whole
        conf = 0.8
    elif per.count("?") == 1 and whole and len(whole) == n_digits:
        # 只有一個洞 → 用整串對應位置補洞
        idx = per.index("?")
        repaired = per[:idx] + whole[idx] + per[idx + 1:]
        if repaired.isdigit():
            best = repaired
            conf = 0.75
    if not best or not best.isdigit():
        # 仍不可靠 → 回 0，交由上層多次取多數決重試（不要硬猜）
        return _ocr_balance_gray(img) if not per.strip("?") else (0, 0.0)
    try:
        return int(best), conf
    except ValueError:
        return 0, 0.0


def ocr_account_name(img: np.ndarray) -> tuple[str, float]:
    """讀左下角遊戲帳號名（淺色字、深底，中文）。

    回傳 (名稱, 信心)。讀不到回 ("", 0.0)。
    """
    if img is None or img.size == 0:
        return "", 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    th = cv2.bitwise_not(th)  # 亮字深底 → 黑字白底
    config = "--psm 7 -l chi_tra"
    try:
        raw = pytesseract.image_to_string(Image.fromarray(th), config=config)
    except pytesseract.TesseractError:
        return "", 0.0
    name = "".join(raw.split())  # 去掉所有空白
    return name, 0.7 if name else 0.0


def preprocess_name_cell(img: np.ndarray, scale: int = 3) -> np.ndarray:
    """玩家暱稱（黑字、淺色背景）：放大 + Otsu 二值化，必要時反白。"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if scale > 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(th.mean()) < 127:
        th = cv2.bitwise_not(th)
    return th


def preprocess_name_cell_wtext(img: np.ndarray, scale: int = 3) -> np.ndarray:
    """統計表表頭：白/淺字 + 深褐底（Otsu 常切壞，需抓亮筆畫）。"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if scale > 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    m = float(gray.mean())
    thr = int(min(max(m + 20, 110), 155))
    _, bright = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k, iterations=1)
    return cv2.bitwise_not(bright)


def _polarize(th: np.ndarray) -> np.ndarray:
    """Tesseract 偏好黑字白底；若前景偏黑（暱稱黑字）維持，否則反白。"""
    if float(th.mean()) < 127:
        return cv2.bitwise_not(th)
    return th


def _variant_image(gray: np.ndarray, scale: int, mode: str) -> np.ndarray:
    g = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gb = cv2.GaussianBlur(g, (3, 3), 0)
    if mode == "adaptive":
        try:
            th = cv2.adaptiveThreshold(
                gb, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
            )
        except cv2.error:
            _, th = cv2.threshold(gb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, th = cv2.threshold(gb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _polarize(th)



def _clean_name_ocr(raw: str) -> str:
    """去掉空白與常見雜訊符號，只留可比對的字。"""
    s = "".join(raw.split())
    for ch in "|;:.,，。、_-~`'\"[](){}（）【】<>《》":
        s = s.replace(ch, "")
    # Tesseract 常多讀出 ˉˍ 等假字；只留中英數與 CJK
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", s)


def _has_ink(gray: np.ndarray) -> bool:
    """這欄看起來是否「有字」：統計表表頭是白/淺字深底，深字淺底也會出現，兩種都要算。"""
    if gray.size == 0:
        return False
    m = float(gray.mean())
    dark = int((gray < (m - 35)).sum())
    bright = int((gray > (m + 35)).sum())
    ink = max(dark, bright)
    return (ink / gray.size) >= 0.004


def _try_name_ocr(proc: np.ndarray, *, step: str, config: str) -> dict:
    """跑一次表頭 OCR，回傳 raw/cleaned/是否採用/原因（供診斷）。"""
    entry: dict = {"step": step, "raw": "", "cleaned": "", "accepted": False, "reason": ""}
    try:
        entry["raw"] = pytesseract.image_to_string(
            Image.fromarray(proc), config=config
        ).strip()
    except pytesseract.TesseractError as exc:
        entry["reason"] = f"tesseract_error:{exc}"
        return entry
    cleaned = _clean_name_ocr(entry["raw"])
    entry["cleaned"] = cleaned
    if len(cleaned) < 2:
        entry["reason"] = "too_short_or_empty"
    elif cleaned.isdigit():
        entry["reason"] = "digits_only"
    else:
        entry["accepted"] = True
        entry["reason"] = "ok"
    return entry


def _ink_stats(gray: np.ndarray) -> dict:
    if gray.size == 0:
        return {"mean": 0.0, "dark_pct": 0.0, "bright_pct": 0.0, "has_ink": False}
    m = float(gray.mean())
    dark = int((gray < (m - 35)).sum()) / gray.size
    bright = int((gray > (m + 35)).sum()) / gray.size
    return {
        "mean": round(m, 1),
        "dark_pct": round(dark, 5),
        "bright_pct": round(bright, 5),
        "has_ink": max(dark, bright) >= 0.004,
    }


def ocr_name_trace(img: np.ndarray, *, deep: bool = True) -> dict:
    """表頭暱稱 OCR 完整追蹤：每步 raw 輸出、過濾原因、最終候選（供客戶端離線診斷）。

    deep=False（淺掃）：只跑最便宜的 legacy_s3。空白／裝飾欄即使 has_ink=True 也不再
    把白字補救 7 步全跑完（那是首局整排表頭 ~26s 的元凶）。深掃（白字補救）只在呼叫端
    確認「對象還沒對到」時，才針對需要的欄補跑，避免在空欄上空轉。
    """
    trace: dict = {
        "shape": list(img.shape) if img.size else [],
        "ink": {},
        "steps": [],
        "candidates": [],
    }
    if img.size == 0:
        trace["ink"] = _ink_stats(np.array([]))
        return trace
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    trace["ink"] = _ink_stats(gray)
    seen: set[str] = set()

    def _run(step: str, proc: np.ndarray, psm: int = 7) -> bool:
        ent = _try_name_ocr(proc, step=step, config=f"--psm {psm} -l chi_tra")
        trace["steps"].append(ent)
        if ent["accepted"] and ent["cleaned"] not in seen:
            seen.add(ent["cleaned"])
            trace["candidates"].append(ent["cleaned"])
            return True
        return False

    # 速度關鍵：每欄「讀到就停」。每次 OCR 都要另起一個 tesseract 子行程（客戶端約 0.6s），
    # 跑滿全部步驟 ×7 欄會吃掉整個下注窗（實測首局表頭曾耗 ~28s 而被看門狗收尾）。
    # 比對門檻已降到 0.7，最便宜的 legacy_s3 即可讓多數暱稱過關，故只在「讀空」時才往下補。
    if _run("legacy_s3", preprocess_name_cell(img, scale=3)):
        return trace
    if not trace["ink"]["has_ink"]:
        return trace
    if not deep:
        return trace
    # 白字表頭：打包版 Tesseract 對 Otsu 常讀空，逐一補跑專用前處理，任一步讀到就停。
    if _run("wtext_s3", preprocess_name_cell_wtext(img, scale=3)):
        return trace
    if _run("wtext_s3_psm6", preprocess_name_cell_wtext(img, scale=3), psm=6):
        return trace
    if _run("adapt_s3", _variant_image(gray, 3, "adaptive")):
        return trace
    if img.shape[1] > 0 and img.shape[1] < 155:
        if _run("legacy_s4", preprocess_name_cell(img, scale=4)):
            return trace
        if _run("wtext_s4", preprocess_name_cell_wtext(img, scale=4)):
            return trace
        _run("adapt_s4", _variant_image(gray, 4, "adaptive"))
    return trace


def ocr_name_candidates(img: np.ndarray, *, deep: bool = True) -> list[str]:
    """讀玩家暱稱，回傳候選字串清單（去重、長度>=2）。"""
    return list(ocr_name_trace(img, deep=deep)["candidates"])


def _consensus_name(cands: list[str]) -> str:
    """從候選中挑「跟其他候選最相似」者（雜訊多為離群值，會被排除）。"""
    if not cands:
        return ""
    if len(cands) == 1:
        return cands[0]
    import difflib as _dl

    def score(c: str) -> tuple[float, int]:
        sim = sum(_dl.SequenceMatcher(None, c, o).ratio() for o in cands if o is not c)
        return sim, len(c)

    return max(cands, key=score)


def ocr_name_cell(img: np.ndarray) -> tuple[str, float]:
    """讀玩家暱稱：多倍率＋多種二值化容錯，psm 7 單行 chi_tra，取共識結果。"""
    cands = ocr_name_candidates(img)
    name = _consensus_name(cands)
    return name, 0.7 if name else 0.0


def warmup_ocr() -> None:
    """啟動時先載入 Tesseract（中文+數字），避免第一局冷啟動卡 ~4 秒。"""
    dummy = np.full((40, 120, 3), 255, dtype=np.uint8)
    # 用純 numpy 畫幾條黑色直條當假字即可暖機 Tesseract；不呼叫 cv2 繪圖函式，
    # 避免凍結環境下 cv2 native 綁定偶發載入不全（缺 putText）而讓整支程式崩潰。
    dummy[8:32, 20:28] = 0
    dummy[8:32, 44:52] = 0
    dummy[8:32, 68:76] = 0
    try:
        ocr_digits(dummy)
        ocr_name_cell(dummy)
    except Exception:
        pass


def ocr_chinese_line(img: np.ndarray, *, stats: bool = False) -> tuple[str, float]:
    proc = preprocess_stats_cell(img) if stats else preprocess_text(img)
    config = "--psm 7 -l chi_tra"
    try:
        raw = pytesseract.image_to_string(Image.fromarray(proc), config=config).strip()
        return raw, 0.7 if raw else 0.0
    except pytesseract.TesseractError:
        return "", 0.0


@lru_cache(maxsize=1)
def paddle_available() -> bool:
    """是否真的能用 PaddleOCR（打包版未內含 paddle 時為 False，避免誤走退回路徑）。"""
    try:
        import paddleocr  # noqa: F401

        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def _paddle_reader():
    from paddleocr import PaddleOCR

    return PaddleOCR(use_angle_cls=False, lang="ch", show_log=False)


def ocr_chinese_paddle(img: np.ndarray) -> tuple[str, float]:
    try:
        reader = _paddle_reader()
    except Exception:
        return ocr_chinese_line(img)
    result = reader.ocr(img, cls=False)
    if not result or not result[0]:
        return "", 0.0
    texts: list[str] = []
    confs: list[float] = []
    for line in result[0]:
        texts.append(line[1][0])
        confs.append(float(line[1][1]))
    return "".join(texts), sum(confs) / len(confs)
