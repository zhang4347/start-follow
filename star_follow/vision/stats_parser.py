from __future__ import annotations

import difflib
import re
import time
from dataclasses import dataclass, field

import numpy as np

from star_follow.config import AppConfig

from . import ocr as ocr_mod
from .ocr import (
    ocr_amount,
    ocr_chinese_line,
    ocr_chinese_paddle,
    ocr_name_candidates,
    ocr_name_cell,
)
from .roi import scale_rect
from .stats_table_roi import find_stats_table_in_panel


@dataclass
class StatsParseResult:
    players: list[str]
    bets_by_player: dict[str, dict[str, int]]
    bets_by_column: dict[int, dict[str, int]]
    header_columns: list[tuple[int, str]]
    confidences: dict[str, float]
    raw_header: list[tuple[str, float]]
    elapsed_ms: float = 0.0
    resolved_columns: dict[str, int] = field(default_factory=dict)


_SKIP_NAMES = {"注區", "玩家", "押注統計", "總計", ""}
_MAX_PLAYER_COLS = 8

# 統計表診斷：偵測不到任何追蹤對象時，存「表格＋逐欄表頭裁切＋OCR 結果」供校正。
_STATS_DEBUG = True
import logging as _logging

_sp_logger = _logging.getLogger(__name__)


def _save_stats_debug(
    table: np.ndarray,
    header: np.ndarray,
    cols: list[tuple[int, int]],
    names: list[str],
    targets: list[str],
) -> None:
    """存統計表診斷圖：整張表 + 表頭那排，檔名帶讀到的名字，供事後對照欄位是否切歪／
    名字是否被切掉。只在偵測不到對象時呼叫，並自動只留最近數十組。"""
    if not _STATS_DEBUG:
        return
    try:
        from PIL import Image

        from star_follow.paths import logs_dir

        d = logs_dir() / "stats_debug"
        d.mkdir(parents=True, exist_ok=True)
        files = sorted(d.glob("*.png"), key=lambda p: p.stat().st_mtime)
        for p in files[: max(0, len(files) - 60)]:
            p.unlink(missing_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        seen = "_".join(n for n in names if n)[:40] or "none"
        Image.fromarray(np.asarray(table)).save(str(d / f"{ts}_table_seen-{seen}.png"))
        # 逐欄表頭裁切（看每一欄到底框到什麼、名字有沒有被切掉）
        for idx, (x0, x1) in enumerate(cols):
            cell = header[:, x0:x1]
            if getattr(cell, "size", 0):
                nm = names[idx] if idx < len(names) else ""
                Image.fromarray(np.asarray(cell)).save(
                    str(d / f"{ts}_col{idx}-{nm or 'empty'}.png")
                )
        _sp_logger.info(
            "統計表診斷已存 logs\\stats_debug：欄數=%d 讀到名字=%s 目標=%s",
            len(cols), [n for n in names if n], targets,
        )
    except Exception as exc:  # noqa: BLE001
        _sp_logger.debug("存統計表診斷圖失敗：%s", exc)


def _crop_rel(img: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    h, w = img.shape[:2]
    return img[int(h * y0) : int(h * y1), int(w * x0) : int(w * x1)]


def _layout_metrics(layout: dict) -> tuple[float, float, int]:
    """依可見列數計算列高（固定比例，不 OCR 列名）。"""
    data_top = float(layout.get("data_top", 0.155))
    data_bottom = float(layout.get("data_bottom", 0.88))
    visible_rows = int(layout.get("visible_rows", 7))
    if layout.get("row_height") is not None:
        row_h = float(layout["row_height"])
    else:
        row_h = (data_bottom - data_top) / max(1, visible_rows)
    return data_top, row_h, visible_rows


def _row_band(layout: dict, row_index: int) -> tuple[float, float]:
    data_top, row_h, visible_rows = _layout_metrics(layout)
    if row_index < 0 or row_index >= visible_rows:
        return 0.0, 0.0
    y0 = data_top + row_index * row_h
    return y0, min(float(layout.get("data_bottom", 0.88)), y0 + row_h)


def extract_stats_table(frame: np.ndarray, cfg: AppConfig) -> tuple[np.ndarray, list[int]]:
    layout = cfg.raw.get("stats_layout", {})
    h_frame, w_frame = frame.shape[:2]
    # 手動固定表格範圍：關掉會把最右欄收窄的自動偵測，直接用標好的 stats_table（參考座標縮放）。
    if layout.get("fixed_table") and cfg.roi.get("stats_table"):
        rx, ry, rw, rh = scale_rect(
            list(cfg.roi["stats_table"]),
            cfg.window.reference_width,
            cfg.window.reference_height,
            w_frame,
            h_frame,
        )
        return frame[ry : ry + rh, rx : rx + rw].copy(), [rx, ry, rw, rh]
    panel_rect = cfg.roi.get("stats_panel", [118, 86, 1043, 548])
    detected = find_stats_table_in_panel(frame, list(panel_rect))
    rect = detected or cfg.roi.get("stats_table", [110, 118, 805, 400])
    x, y, w, h = rect
    return frame[y : y + h, x : x + w].copy(), list(rect)


def _player_cols(layout: dict) -> int:
    """百家樂統計表的玩家座位數（固定 7）。可由 config stats_layout.player_cols 覆寫。"""
    try:
        n = int(layout.get("player_cols", 7))
    except (TypeError, ValueError):
        n = 7
    return max(1, min(_MAX_PLAYER_COLS, n))


def _column_layout(table_w: int, layout: dict) -> tuple[int, list[tuple[int, int]]]:
    label_frac = float(layout.get("label_col_frac", 0.11))
    label_w = max(40, int(table_w * label_frac))
    remain = table_w - label_w
    n = _player_cols(layout)
    col_w = max(50, remain // n)
    cols = []
    for i in range(n):
        x0 = label_w + i * col_w
        x1 = min(table_w, x0 + col_w)
        if x1 - x0 < 30:
            break
        cols.append((x0, x1))
    return label_w, cols


def _pad_columns_to(
    cols: list[tuple[int, int]], table_w: int, target: int
) -> list[tuple[int, int]]:
    """格線偵測常因最右欄外框線偵測不到而少抓 1～2 欄（7 人桌只讀到 6 欄，對象在
    最右就被誤判「不在房」）。用偵測到的欄寬中位數，往右補到 target 欄（補到表格右緣
    為止）。若裁切範圍本身太窄裝不下，補出的欄會被夾到右緣（仍比完全沒有好）。
    """
    if not cols:
        return cols
    if len(cols) >= target:
        return cols[:target]
    widths = sorted(b - a for a, b in cols)
    col_w = widths[len(widths) // 2]
    if col_w < 30:
        return cols
    out = list(cols)
    x = out[-1][1]
    while len(out) < target and x + 32 < table_w:
        x0 = x + 2
        x1 = min(table_w, x0 + col_w)
        if x1 - x0 < 32:
            break
        out.append((x0, x1))
        x = x1
    return out


def detect_column_bounds(
    table: "np.ndarray",
    layout: dict,
) -> tuple[int, list[tuple[int, int]]] | None:
    """偵測表格垂直格線取得真實欄界；失敗回 None 由等分法接手。"""
    import cv2
    import numpy as np

    h, w = table.shape[:2]
    if w < 200 or h < 80:
        return None
    # 取資料列區域（避開上方標題/下方總計），對格線最敏感
    y0 = int(h * 0.12)
    y1 = int(h * 0.90)
    body = table[y0:y1, :]
    gray = cv2.cvtColor(body, cv2.COLOR_RGB2GRAY).astype("float32")
    col_mean = gray.mean(axis=0)  # 每個 x 的平均亮度；格線較暗 → 局部極小
    # 與局部背景（中位數平滑）比較，找明顯偏暗的縱向溝槽
    k = max(9, w // 60) | 1
    bg = cv2.medianBlur(col_mean.reshape(1, -1).astype("uint8"), 1).flatten().astype("float32")
    bg = np.convolve(col_mean, np.ones(k) / k, mode="same")
    dip = bg - col_mean  # 越大代表越暗（線）
    thr = max(8.0, float(dip.max()) * 0.4)
    xs = [x for x in range(2, w - 2) if dip[x] >= thr]
    if len(xs) < 3:
        return None
    # 把連續的 x 併成一條線，取中心
    lines: list[int] = []
    group = [xs[0]]
    for x in xs[1:]:
        if x - group[-1] <= 8:
            group.append(x)
        else:
            lines.append(sum(group) // len(group))
            group = [x]
    lines.append(sum(group) // len(group))
    if len(lines) < 3:
        return None
    # lines[0] 很小 = 表格左外框，其後第一段才是 label 欄需丟棄；
    # 否則 label 欄在 (0, lines[0])，各線之間皆為玩家欄。
    drop_first = lines[0] < 25
    label_w = lines[1] if drop_first and len(lines) > 1 else lines[0]
    cols: list[tuple[int, int]] = []
    for a, b in zip(lines, lines[1:]):
        if b - a < 40:
            continue
        cols.append((a + 2, b - 2))
    if drop_first and cols:
        cols = cols[1:]
    if not cols:
        return None
    return label_w, cols[:_MAX_PLAYER_COLS]


def _manual_columns(
    table_w: int, layout: dict
) -> tuple[int, list[tuple[int, int]]] | None:
    """手動標記的玩家欄範圍（player_band=[x0_frac, x1_frac]，相對表格寬度的比例）。

    手動框住「所有玩家名字那一排」的左右範圍後，等分成 player_cols（預設 7）欄。
    欄寬是等寬的，等分比會歪掉的自動格線偵測更穩，也不會把名字邊緣切掉。
    """
    band = layout.get("player_band")
    if not (isinstance(band, (list, tuple)) and len(band) == 2):
        return None
    x0f, x1f = float(band[0]), float(band[1])
    if not (0.0 <= x0f < x1f <= 1.0):
        return None
    x0 = int(round(x0f * table_w))
    x1 = int(round(x1f * table_w))
    n = _player_cols(layout)
    col_w = (x1 - x0) / n
    if col_w < 20:
        return None
    cols: list[tuple[int, int]] = []
    for i in range(n):
        a = int(round(x0 + i * col_w))
        b = int(round(x0 + (i + 1) * col_w))
        cols.append((a + 1, b - 1))
    return x0, cols


_COL_MODE_LOGGED = False


def _columns_for(table: "np.ndarray", layout: dict) -> tuple[int, list[tuple[int, int]]]:
    """決定 7 個玩家欄的左右界。

    固定 7 座位的 UI：一律等分成 target 欄（中間沒人就讀空白），不用會在空欄斷掉的
    格線偵測（那會造成「第 N 欄之後全當沒人」、對象在右側卻被判不在房）。
    優先序：手動 player_band > 等分法 _column_layout。只有明確 detect_grid:true 時才
    試格線，且結果必須補滿到 target 欄才採用，否則仍退回等分法。
    """
    global _COL_MODE_LOGGED
    target = _player_cols(layout)
    manual = _manual_columns(table.shape[1], layout)
    if manual is not None:
        if not _COL_MODE_LOGGED:
            _sp_logger.info("欄位偵測：手動 player_band 等分 %d 欄", len(manual[1]))
            _COL_MODE_LOGGED = True
        return manual
    if layout.get("detect_grid", False):
        try:
            det = detect_column_bounds(table, layout)
        except Exception:
            det = None
        if det and len(det[1]) >= 1:
            label_w, cols = det
            cols = _pad_columns_to(cols, table.shape[1], target)
            if len(cols) >= target:
                if not _COL_MODE_LOGGED:
                    _sp_logger.info("欄位偵測：格線+補欄 共 %d 欄", len(cols))
                    _COL_MODE_LOGGED = True
                return label_w, cols
    fallback = _column_layout(table.shape[1], layout)
    if not _COL_MODE_LOGGED:
        _sp_logger.info("欄位偵測：等分法（label_col_frac）共 %d 欄", len(fallback[1]))
        _COL_MODE_LOGGED = True
    return fallback


def _clean_name(name: str) -> str:
    name = re.sub(r"\s+", "", name)
    for ch in "注區玩家|;[]()（）":
        name = name.replace(ch, "")
    return name.strip()


def _read_header_columns(
    header: np.ndarray,
    cols: list[tuple[int, int]],
    *,
    use_paddle: bool = False,
) -> list[tuple[str, float, int, int]]:
    results: list[tuple[str, float, int, int]] = []
    for x0, x1 in cols:
        cell = header[:, x0:x1]
        if use_paddle:
            name, conf = ocr_chinese_paddle(cell)
        else:
            name, conf = ocr_chinese_line(cell, stats=True)
        name = _clean_name(name)
        if not name or name in _SKIP_NAMES or len(name) < 2 or name.isdigit():
            continue
        results.append((name, conf, x0, x1))
    return results


_USE_PADDLE_HEADER = True


def set_header_ocr(use_paddle: bool) -> None:
    global _USE_PADDLE_HEADER
    _USE_PADDLE_HEADER = use_paddle


def _ocr_header_cands(cell: np.ndarray) -> list[str]:
    """玩家暱稱候選（中文）：PaddleOCR 可用就用它；否則用多倍率＋多二值化的 Tesseract。

    回傳「多個候選」而非單一字串：比對時拿目標去比所有候選取最佳，只要任一種前處理讀對
    （或接近）就配得到，大幅降低單一門檻把整欄切壞而落空的風險。
    """
    if _USE_PADDLE_HEADER:
        try:
            name, _ = ocr_chinese_paddle(cell)
            name = _clean_name(name)
            if name:
                return [name]
        except Exception:
            pass
    return [_clean_name(c) for c in ocr_name_candidates(cell) if _clean_name(c)]


def _pick_consensus(cands: list[str]) -> str:
    """從候選中挑「跟其他候選最相似」者（雜訊離群值會被排除）；供顯示/記錄用。"""
    if not cands:
        return ""
    if len(cands) == 1:
        return cands[0]
    return max(
        cands,
        key=lambda c: sum(
            difflib.SequenceMatcher(None, c, o).ratio() for o in cands if o is not c
        ),
    )


def _ocr_header_name(cell: np.ndarray) -> str:
    """單一暱稱（共識挑選），供顯示/記錄用。"""
    return _pick_consensus(_ocr_header_cands(cell))


def _header_cells(table: np.ndarray, layout: dict) -> list[tuple[int, np.ndarray]]:
    band = layout.get("header_band", [0.0, 0.0, 1.0, 0.12])
    if isinstance(band, list) and len(band) == 4:
        header = _crop_rel(table, band[0], band[1], band[2], band[3])
    else:
        header = table[: max(20, int(table.shape[0] * 0.12)), :]
    _, cols = _columns_for(table, layout)
    return [(idx, header[:, x0:x1]) for idx, (x0, x1) in enumerate(cols)]


def read_column_header_cands(
    table: np.ndarray, layout: dict
) -> list[tuple[int, list[str]]]:
    """讀各欄表頭暱稱「候選清單」，回傳 (欄位索引, [候選名...])。"""
    return [(idx, _ocr_header_cands(cell)) for idx, cell in _header_cells(table, layout)]


def read_column_headers(table: np.ndarray, layout: dict) -> list[tuple[int, str]]:
    """讀取各玩家欄表頭暱稱（共識單名），回傳 (欄位索引, 名稱)。"""
    return [
        (idx, _pick_consensus(cands))
        for idx, cands in read_column_header_cands(table, layout)
    ]


# 跟注是真金白銀，比對要「精準優先」：寧可錯過也不要跟錯路人。
_NAME_MATCH_MARGIN = 0.08  # 最佳與第二名差距需大於此值，否則視為無法分辨→不跟


def _name_cutoff(target: str) -> float:
    """依名字長度算「容許剛好錯 1 個字」的相似度門檻。

    SequenceMatcher 對長度 L、差 1 字的字串相似度 = (L-1)/L。短名（<=3 字）要求幾乎
    完全相同，避免「錯 1 字 = 相似度太低」反而把路人也放進來。
    """
    L = len(target)
    if L <= 3:
        return 0.95
    return (L - 1) / L - 0.02


def _norm_name(s: str) -> str:
    """正規化暱稱：全形轉半形、去空白、去除非中英數雜訊符號，利於精準比對。"""
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
    t = "".join(out)
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", t)


def find_column_for_player_cands(
    cand_headers: list[tuple[int, list[str]]],
    player_name: str,
    hint: int | None = None,
) -> int | None:
    """用「每欄多個候選」比對：每欄取候選對目標的最佳相似度，再套精準+領先守則。

    比對相似度用該欄所有候選的最大值（任一種前處理讀對就算數），但仍要求最佳欄明顯勝過
    第二名，避免跟到名字相近的路人。寧可錯過也不跟錯。
    """
    target = _norm_name(player_name)
    if not target:
        return None
    scored: list[tuple[float, int]] = []
    for idx, cands in cand_headers:
        best = 0.0
        for c in cands:
            n = _norm_name(c)
            if not n:
                continue
            r = 1.0 if n == target else difflib.SequenceMatcher(None, target, n).ratio()
            if r > best:
                best = r
        if best > 0:
            scored.append((best, idx))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)

    exact = [idx for r, idx in scored if r >= 0.999]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        return hint if (hint is not None and hint in exact) else None

    best_r, best_idx = scored[0]
    if best_r < _name_cutoff(target):
        return None
    if len(scored) > 1 and scored[1][0] >= best_r - _NAME_MATCH_MARGIN:
        return None
    return best_idx


def find_column_for_player(
    headers: list[tuple[int, str]],
    player_name: str,
    hint: int | None = None,
) -> int | None:
    """相容介面：單一名字版，轉成候選版比對。"""
    return find_column_for_player_cands(
        [(idx, [name]) for idx, name in headers], player_name, hint=hint
    )


def _parse_single_column(
    table: np.ndarray,
    col_index: int,
    rows: list[str],
    layout: dict,
) -> dict[str, int]:
    """只 OCR 一欄金額（固定座標；表必須已捲到頂，首列=莊家）。"""
    _, cols = _columns_for(table, layout)
    if col_index < 0 or col_index >= len(cols):
        return {}
    x0, x1 = cols[col_index]
    _, _, visible_rows = _layout_metrics(layout)
    out: dict[str, int] = {}
    for i, row_name in enumerate(rows[:visible_rows]):
        y0, y1 = _row_band(layout, i)
        if y1 <= y0:
            continue
        row_img = _crop_rel(table, 0.0, y0, 1.0, y1)
        cell = row_img[:, x0:x1]
        amount, _ = ocr_amount(cell)
        if amount > 0:
            out[row_name] = amount
    return out


def parse_bottom_row_amount(
    frame: np.ndarray,
    cfg: AppConfig,
    col_index: int,
) -> int:
    """捲到底後讀最後一格金額（有數字 = 閒龍寶）。"""
    table, _ = extract_stats_table(frame, cfg)
    layout = cfg.raw.get("stats_layout", {})
    _, cols = _columns_for(table, layout)
    if col_index < 0 or col_index >= len(cols):
        return 0
    _, _, visible_rows = _layout_metrics(layout)
    y0, y1 = _row_band(layout, visible_rows - 1)
    if y1 <= y0:
        return 0
    x0, x1 = cols[col_index]
    cell = _crop_rel(table, 0.0, y0, 1.0, y1)[:, x0:x1]
    amount, _ = ocr_amount(cell)
    return amount if amount > 0 else 0


def match_player_name(players: list[str], target: str) -> str | None:
    """精準比對暱稱：正規化後完全相同優先，否則高相似度且明顯勝過第二名才算。
    比對不到夠像的就回 None（寧可錯過也不要跟錯路人）。"""
    target_n = _norm_name(target)
    if not target_n:
        return None
    named = [(p, _norm_name(p)) for p in players]
    named = [(p, n) for p, n in named if n]
    if not named:
        return None
    for p, n in named:
        if n == target_n:
            return p
    scored = sorted(
        ((difflib.SequenceMatcher(None, target_n, n).ratio(), p) for p, n in named),
        key=lambda x: x[0],
        reverse=True,
    )
    best_r, best_p = scored[0]
    if best_r < _NAME_MATCH_CUTOFF:
        return None
    if len(scored) > 1 and scored[1][0] >= best_r - _NAME_MATCH_MARGIN:
        return None
    return best_p


def resolve_follow_columns(
    frame: np.ndarray,
    cfg: AppConfig,
    follow_targets: list[tuple[str, int | None]],
) -> StatsParseResult:
    """只解析跟注對象欄位（不讀金額），供 T=15 預定位。"""
    t0 = time.perf_counter()
    ocr_mod.set_ocr_options(fast=cfg.vision.fast_ocr, scale=cfg.vision.ocr_scale)
    set_header_ocr(cfg.vision.header_use_paddle)
    table, _ = extract_stats_table(frame, cfg)
    layout = cfg.raw.get("stats_layout", {})

    # 一律讀表頭重新確認欄位；hint 只當提示，不直接採用（換桌後舊欄位會指到路人）。
    resolved: dict[str, int] = {}
    cand_headers = read_column_header_cands(table, layout)
    headers: list[tuple[int, str]] = [(idx, _pick_consensus(c)) for idx, c in cand_headers]
    for name, hint in follow_targets:
        col = find_column_for_player_cands(cand_headers, name, hint=hint)
        if col is not None:
            resolved[name] = col

    elapsed = (time.perf_counter() - t0) * 1000
    return StatsParseResult(
        players=[n for _, n in headers if n],
        bets_by_player={},
        bets_by_column={},
        header_columns=headers,
        confidences={},
        raw_header=[(n, 0.0) for _, n in headers if n],
        elapsed_ms=elapsed,
        resolved_columns=resolved,
    )


def verify_follow_columns(
    frame: np.ndarray,
    cfg: AppConfig,
    verify: list[tuple[str, int]],
) -> StatsParseResult:
    """跨局快取驗證：只 OCR 指定欄位的表頭，確認該欄名字仍是同一位對象。

    全部對得上 → resolved_columns 帶回全部對象（省掉整排表頭 OCR，最大宗的慢動作）。
    任一對不上（對象換座位／已換桌）→ 該對象不在 resolved_columns，呼叫端應退回整排重掃。
    名字比對沿用 find_column_for_player 的精準+模糊規則，故不會把路人當成對象。
    """
    t0 = time.perf_counter()
    ocr_mod.set_ocr_options(fast=cfg.vision.fast_ocr, scale=cfg.vision.ocr_scale)
    set_header_ocr(cfg.vision.header_use_paddle)
    table, _ = extract_stats_table(frame, cfg)
    layout = cfg.raw.get("stats_layout", {})

    band = layout.get("header_band", [0.0, 0.0, 1.0, 0.12])
    if isinstance(band, list) and len(band) == 4:
        header = _crop_rel(table, band[0], band[1], band[2], band[3])
    else:
        header = table[: max(20, int(table.shape[0] * 0.12)), :]
    _, cols = _columns_for(table, layout)

    resolved: dict[str, int] = {}
    for name, col in verify:
        if col is None or col < 0 or col >= len(cols):
            continue
        x0, x1 = cols[col]
        cands = _ocr_header_cands(header[:, x0:x1])
        if not cands:
            continue
        if find_column_for_player_cands([(col, cands)], name, hint=col) == col:
            resolved[name] = col

    elapsed = (time.perf_counter() - t0) * 1000
    return StatsParseResult(
        players=[],
        bets_by_player={},
        bets_by_column={},
        header_columns=[],
        confidences={},
        raw_header=[],
        elapsed_ms=elapsed,
        resolved_columns=resolved,
    )


def parse_stats_table(
    frame: np.ndarray,
    cfg: AppConfig,
    *,
    use_paddle: bool = False,
    only_columns: list[int] | None = None,
    follow_targets: list[tuple[str, int | None]] | None = None,
    known_columns: dict[str, int] | None = None,
    rows_override: list[str] | None = None,
) -> StatsParseResult:
    t0 = time.perf_counter()
    ocr_mod.set_ocr_options(fast=cfg.vision.fast_ocr, scale=cfg.vision.ocr_scale)
    set_header_ocr(cfg.vision.header_use_paddle)

    table, _ = extract_stats_table(frame, cfg)

    layout = cfg.raw.get("stats_layout", {})
    # rows_override：統計表「滑到最底」時，畫面可見列名與滑到頂不同（少了莊家、
    # 多了最後一列），用這份對應表讀，一次截圖就抓到所有邊注。
    rows = rows_override if rows_override is not None else (cfg.stats_rows or [])

    if follow_targets is not None and cfg.vision.fast_ocr:
        if known_columns:
            headers = []
            players = []
            bets_by_column: dict[int, dict[str, int]] = {}
            resolved = dict(known_columns)
            for name, col in known_columns.items():
                bets_by_column[col] = _parse_single_column(table, col, rows, layout)
        else:
            cand_headers = read_column_header_cands(table, layout)
            headers = [(idx, _pick_consensus(c)) for idx, c in cand_headers]
            players = [n for _, n in headers if n]
            bets_by_column = {}
            resolved = {}
            for name, hint in follow_targets:
                col = find_column_for_player_cands(cand_headers, name, hint=hint)
                if col is None:
                    continue
                resolved[name] = col
                bets_by_column[col] = _parse_single_column(table, col, rows, layout)
            if not resolved and follow_targets:
                # 偵測不到任何對象：存診斷圖（整張表＋逐欄表頭裁切）供校正欄位/名字切位
                band = layout.get("header_band", [0.0, 0.0, 1.0, 0.12])
                if isinstance(band, list) and len(band) == 4:
                    hdr = _crop_rel(table, band[0], band[1], band[2], band[3])
                else:
                    hdr = table[: max(20, int(table.shape[0] * 0.12)), :]
                _, dcols = _columns_for(table, layout)
                _save_stats_debug(
                    table, hdr, dcols,
                    [n for _, n in headers],
                    [n for n, _ in follow_targets],
                )
        elapsed = (time.perf_counter() - t0) * 1000
        return StatsParseResult(
            players=players,
            bets_by_player={},
            bets_by_column=bets_by_column,
            header_columns=headers,
            confidences={},
            raw_header=[(n, 0.0) for _, n in headers if n],
            elapsed_ms=elapsed,
            resolved_columns=resolved,
        )

    if only_columns is not None and cfg.vision.fast_ocr:
        headers = read_column_headers(table, layout)
        bets_by_column: dict[int, dict[str, int]] = {}
        for col in only_columns:
            bets_by_column[col] = _parse_single_column(table, col, rows, layout)
        elapsed = (time.perf_counter() - t0) * 1000
        return StatsParseResult(
            players=[n for _, n in headers if n],
            bets_by_player={},
            bets_by_column=bets_by_column,
            header_columns=headers,
            confidences={},
            raw_header=[],
            elapsed_ms=elapsed,
        )

    band = layout.get("header_band", [0.0, 0.0, 1.0, 0.12])
    if isinstance(band, list) and len(band) == 4:
        header = _crop_rel(table, band[0], band[1], band[2], band[3])
    else:
        header = table[: max(20, int(table.shape[0] * 0.12)), :]

    _, _, visible_rows = _layout_metrics(layout)

    _, cols = _columns_for(table, layout)
    header_info = _read_header_columns(header, cols, use_paddle=use_paddle)
    players = [h[0] for h in header_info]

    bets: dict[str, dict[str, int]] = {p: {} for p in players}
    bets_by_column: dict[int, dict[str, int]] = {}
    confs: dict[str, float] = {}

    for col_idx, (name, _, x0, x1) in enumerate(header_info):
        bets_by_column[col_idx] = {}

    for i, row_name in enumerate(rows[:visible_rows]):
        y0, y1 = _row_band(layout, i)
        row_img = _crop_rel(table, 0.0, y0, 1.0, y1)
        for col_idx, (name, _, x0, x1) in enumerate(header_info):
            cell = row_img[:, x0:x1]
            amount, conf = ocr_amount(cell)
            if amount > 0:
                bets[name][row_name] = amount
                bets_by_column[col_idx][row_name] = amount
            confs[f"{name}:{row_name}"] = conf

    elapsed = (time.perf_counter() - t0) * 1000
    return StatsParseResult(
        players=players,
        bets_by_player=bets,
        bets_by_column=bets_by_column,
        header_columns=[(col_idx, name) for col_idx, (name, _, _, _) in enumerate(header_info)],
        confidences=confs,
        raw_header=[(n, c) for n, c, _, _ in header_info],
        elapsed_ms=elapsed,
    )


def extract_player_bets(
    result: StatsParseResult,
    player_name: str,
    stats_to_bet: dict[str, str],
    *,
    column_index: int | None = None,
) -> dict[str, int]:
    raw: dict[str, int] = {}
    if column_index is not None and column_index in result.bets_by_column:
        raw = result.bets_by_column[column_index]
    if not raw:
        matched = match_player_name(result.players, player_name)
        if matched is not None:
            raw = result.bets_by_player.get(matched, {})
    out: dict[str, int] = {}
    for stats_name, amount in raw.items():
        bet_key = stats_to_bet.get(stats_name, stats_name)
        out[bet_key] = out.get(bet_key, 0) + amount
    return out
