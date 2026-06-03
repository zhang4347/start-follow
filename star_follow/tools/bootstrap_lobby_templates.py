"""從 logs/lobby_mark 標記截圖裁切大廳導覽模板（離線，不需遊戲視窗）。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from star_follow.paths import logs_dir
_MARK_DIR = logs_dir() / "lobby_mark"
_OUT = Path(__file__).resolve().parents[1] / "vision" / "templates"

# scene → (檔名關鍵, 輸出模板, 裁切半寬, 裁切半高)
_SPECS = [
    (1, "lobby_qipai_menu.png", 58, 42),
    (2, "lobby_baccarat_card.png", 95, 130),
    (3, "lobby_random_select.png", 118, 38),
]


def _scene_png(row: dict) -> Path | None:
    name = row.get("file")
    if name:
        p = _MARK_DIR / name
        if p.is_file():
            return p
    scene = int(row["scene"])
    for p in sorted(_MARK_DIR.glob(f"scene_{scene}_*.png")):
        if "_marked" not in p.name:
            return p
    return None


def _crop_center(rgb: np.ndarray, cx: int, cy: int, hw: int, hh: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    x0 = max(0, cx - hw)
    y0 = max(0, cy - hh)
    x1 = min(w, cx + hw)
    y1 = min(h, cy + hh)
    return rgb[y0:y1, x0:x1]


def main() -> None:
    points_path = _MARK_DIR / "points.json"
    if not points_path.is_file():
        raise SystemExit(f"找不到 {points_path}，請先用 mark_lobby 標點並存檔。")
    rows = json.loads(points_path.read_text(encoding="utf-8"))
    by_scene = {int(r["scene"]): r for r in rows}
    _OUT.mkdir(parents=True, exist_ok=True)
    for scene, out_name, hw, hh in _SPECS:
        row = by_scene.get(scene)
        if not row:
            print(f"略過 scene {scene}：points.json 無此場景")
            continue
        png = _scene_png(row)
        if not png:
            print(f"略過 scene {scene}：找不到 scene_{scene}_*.png")
            continue
        rgb = np.array(Image.open(png).convert("RGB"))
        patch = _crop_center(rgb, int(row["x"]), int(row["y"]), hw, hh)
        dest = _OUT / out_name
        Image.fromarray(patch).save(dest)
        print(f"已寫入 {dest.name} ← {png.name} ({patch.shape[1]}×{patch.shape[0]})")


if __name__ == "__main__":
    main()
