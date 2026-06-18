"""簡繁正規化。

神經網路中文 OCR（RapidOCR/PaddleOCR 的 PP-OCR 模型）多半輸出簡體字，
而追蹤名單通常是繁體。比對前把兩邊都轉成簡體當共同基準，簡繁差異就消失，
短名（<=3 字要求 0.95 相似度）也不會因為一個簡繁字就配不上。
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _zhconv():
    try:
        import zhconv

        return zhconv
    except Exception:
        return None


def to_hans(s: str) -> str:
    """轉簡體；zhconv 不可用時原樣回傳（不致命，僅退化成不做簡繁正規化）。"""
    if not s:
        return s
    z = _zhconv()
    if z is None:
        return s
    try:
        return z.convert(s, "zh-hans")
    except Exception:
        return s
