"""離線驗證 logs/lobby_mark 標記截圖能否正確分段（不需遊戲視窗）。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from star_follow.automation.lobby_nav import (
    PHASE_ENTRY,
    PHASE_HOME,
    PHASE_QIPAI_READY,
    PHASE_QIPAI_SCROLL,
    PHASE_TABLE,
    classify_nav_screen,
)
from star_follow.config import load_config
from star_follow.paths import logs_dir

_EXPECT: dict[int, set[str]] = {
    1: {PHASE_HOME},
    2: {PHASE_QIPAI_SCROLL, PHASE_QIPAI_READY},
    3: {PHASE_ENTRY},
}


@dataclass
class _FakeWin:
    client_width: int = 1280
    client_height: int = 720


def main() -> int:
    mark_dir = logs_dir() / "lobby_mark"
    points_path = mark_dir / "points.json"
    if not points_path.is_file():
        print(f"缺少 {points_path}")
        return 1
    rows = json.loads(points_path.read_text(encoding="utf-8"))
    cfg = load_config()
    win = _FakeWin()
    ok_all = True
    for row in rows:
        scene = int(row["scene"])
        pngs = [p for p in mark_dir.glob(f"scene_{scene}_*.png") if "_marked" not in p.name]
        if not pngs:
            print(f"scene {scene}: 找不到 PNG")
            ok_all = False
            continue
        png = sorted(pngs)[-1]
        frame = np.array(Image.open(png).convert("RGB"))
        nav = classify_nav_screen(frame, cfg, win, use_ocr=False)  # type: ignore[arg-type]
        expect = _EXPECT.get(scene, set())
        hit = nav.phase in expect
        mark = "OK" if hit else "FAIL"
        if not hit:
            ok_all = False
        print(
            f"[{mark}] scene {scene} {row.get('label', '')} → {nav.label} ({nav.phase}) "
            f"信心={nav.confidence:.0%} tab={nav.qipai_tab} table={nav.table_hud}"
        )
        for r in nav.reasons[:4]:
            print(f"      {r}")
    return 0 if ok_all else 2


if __name__ == "__main__":
    raise SystemExit(main())
