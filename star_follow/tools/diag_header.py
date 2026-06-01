"""離線診斷表頭 OCR。用法:
  python -m star_follow.tools.diag_header star_follow/logs/ocr_T7_1780126155.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from star_follow.config import load_config
from star_follow.vision import stats_parser as sp

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: python -m star_follow.tools.diag_header <截圖路徑>")
        return 1
    img_path = Path(sys.argv[1])
    frame = np.array(Image.open(img_path).convert("RGB"))
    print(f"影像尺寸: {frame.shape[1]}x{frame.shape[0]}")

    cfg = load_config()
    panel_rect = cfg.roi.get("stats_panel")
    table_rect_cfg = cfg.roi.get("stats_table")
    print(f"config stats_panel={panel_rect}  stats_table={table_rect_cfg}")

    from star_follow.vision.stats_table_roi import find_stats_table_in_panel

    detected = find_stats_table_in_panel(frame, list(panel_rect)) if panel_rect else None
    print(f"偵測到的表格框: {detected}")

    table, used_rect = sp.extract_stats_table(frame, cfg)
    print(f"實際使用的表格框: {used_rect}  table={table.shape[1]}x{table.shape[0]}")

    layout = cfg.raw.get("stats_layout", {})
    band = layout.get("header_band", [0.0, 0.0, 1.0, 0.12])
    h, w = table.shape[:2]
    hy0, hy1 = int(h * band[1]), int(h * band[3])
    header = table[hy0:hy1, :]
    Image.fromarray(table).save(LOG_DIR / "diag_table.png")
    Image.fromarray(header).save(LOG_DIR / "diag_header_band.png")
    print(f"表頭帶 y={hy0}..{hy1}（已存 diag_header_band.png）")

    label_w, cols = sp._columns_for(table, layout)
    det = sp.detect_column_bounds(table, layout)
    print(f"格線偵測: {det}")
    print(f"採用 label_w={label_w}  欄位數={len(cols)} 欄x範圍={cols}")
    for i, (x0, x1) in enumerate(cols):
        Image.fromarray(header[:, x0:x1]).save(LOG_DIR / f"diag_col{i}.png")

    print("\n=== 讀表頭 ===")
    sp.set_header_ocr(cfg.vision.header_use_paddle)
    for idx, name in sp.read_column_headers(table, layout):
        print(f"  欄{idx}: {name!r}")

    targets = [("人類學教授", None), ("速趴賽亞人", None)]
    print("\n=== 解析跟注（含金額）===")
    res = sp.parse_stats_table(frame, cfg, follow_targets=targets)
    print(f"  resolved_columns={res.resolved_columns}")
    for col, bets in res.bets_by_column.items():
        print(f"  欄{col} 金額={bets}")

    print("\n=== 全表逐欄逐列金額（核對對位）===")
    from star_follow.vision.ocr import ocr_amount

    _, cols = sp._columns_for(table, layout)
    rows = cfg.stats_rows or []
    _, _, vis = sp._layout_metrics(layout)
    for ci, (x0, x1) in enumerate(cols):
        line = []
        for ri, rname in enumerate(rows[:vis]):
            ry0, ry1 = sp._row_band(layout, ri)
            cell = sp._crop_rel(table, 0.0, ry0, 1.0, ry1)[:, x0:x1]
            amt, _ = ocr_amount(cell)
            if amt > 0:
                line.append(f"{rname}={amt}")
        print(f"  欄{ci} x=({x0},{x1}): {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
