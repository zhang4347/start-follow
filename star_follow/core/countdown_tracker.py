"""錨定 OCR 讀到的開盤秒數後，以本機時鐘每秒遞減（減少漏拍 T=8）。"""

from __future__ import annotations

import logging
import math
import time

from star_follow.vision.state import CountdownColor, CountdownState

logger = logging.getLogger(__name__)


class CountdownTracker:
    def __init__(
        self,
        *,
        anchor_at: int = 20,
        resync_tolerance: int = 1,
    ) -> None:
        self.anchor_at = anchor_at
        self.resync_tolerance = resync_tolerance
        self._anchor_t: int | None = None
        self._anchor_mono: float | None = None

    @property
    def anchored(self) -> bool:
        return self._anchor_t is not None

    def reset(self) -> None:
        self._anchor_t = None
        self._anchor_mono = None

    def try_anchor(self, ocr: CountdownState) -> bool:
        """錨定開局倒數（預設僅 T=19～20），掛機用。"""
        if self._anchor_t is not None:
            return False
        if ocr.color != CountdownColor.GREEN or ocr.seconds is None:
            return False
        t = ocr.seconds
        lo = self.anchor_at - 1
        if t > self.anchor_at or t < lo:
            return False
        self._anchor_t = t
        self._anchor_mono = time.monotonic()
        logger.info("倒數錨定 T=%s（之後由軟體每秒遞減）", t)
        return True

    def _estimate(self) -> int | None:
        if self._anchor_t is None or self._anchor_mono is None:
            return None
        elapsed = time.monotonic() - self._anchor_mono
        # ceil：軟體略偏快，避免比實際慢 1～2 秒以為還有時間
        return max(0, self._anchor_t - math.ceil(elapsed))

    def _resync_to(self, ocr_t: int, est: int) -> int:
        if ocr_t != est:
            logger.info("倒數校正 OCR=%s 軟體=%s", ocr_t, est)
        self._anchor_t = ocr_t
        self._anchor_mono = time.monotonic()
        return ocr_t

    def effective(self, ocr: CountdownState) -> tuple[int | None, CountdownColor]:
        """
        錨定後在軟體 T>0 期間一律用本機倒數，不受 OCR 紅字／誤讀干擾。
        OCR 讀到比軟體小 → 立刻採較小值（遊戲較快，防掛表後來不及下注）。
        T 歸零後才交還 OCR（封盤）。
        """
        if self._anchor_t is None:
            return ocr.seconds, ocr.color

        est = self._estimate()
        if est is None:
            return ocr.seconds, ocr.color

        if est > 0:
            if ocr.color == CountdownColor.GREEN and ocr.seconds is not None:
                diff = ocr.seconds - est
                if diff < 0 and -diff <= 6:
                    est = self._resync_to(ocr.seconds, est)
                elif diff > self.resync_tolerance and diff <= 4:
                    est = self._resync_to(ocr.seconds, est)
            elif ocr.color != CountdownColor.GREEN and ocr.seconds is not None:
                if abs(ocr.seconds - est) > 5:
                    logger.debug(
                        "忽略 OCR 倒數 color=%s T=%s，沿用軟體 T=%s",
                        ocr.color.name,
                        ocr.seconds,
                        est,
                    )
            return est, CountdownColor.GREEN

        return ocr.seconds, ocr.color
