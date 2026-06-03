"""主引擎入口。用法: python -m star_follow.tools.run [--live]"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from star_follow.core.engine import FollowEngine
from star_follow.core.follow_list import FollowList
from star_follow.paths import logs_dir as _logs_dir
from star_follow.util.admin import is_admin, require_admin


def _setup_logging() -> Path:
    """同時輸出到主控台與 logs/run_*.log，方便事後/跨視窗檢視。"""
    log_dir = _logs_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    root.handlers = [sh, fh]
    return log_path


def _read_launch_settings() -> dict:
    """讀取 exe 旁的「啟動設定.txt」，決定模式與是否真下注。

    回傳 dict：{"mode": "patrol"/"stay"/None, "live": True/False/None, "selftest": bool}
    讀不到或格式不符時對應值為 None / False，由上層套用預設。
    """
    from star_follow import paths

    out: dict = {
        "mode": None,
        "live": None,
        "selftest": False,
        "balance_test": False,
        "list_windows": False,
        "stay_table": None,
        "stay_targets": [],
        "account_name": None,
    }
    p = paths.launch_settings_path()
    if not p.is_file():
        return out

    raw = b""
    try:
        raw = p.read_bytes()
    except Exception:  # noqa: BLE001
        return out
    text = None
    for enc in ("utf-8-sig", "utf-8", "cp950", "big5", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except Exception:  # noqa: BLE001
            continue
    if text is None:
        return out

    for line in text.splitlines():
        s = line.strip()
        if not s or s[0] in "#；;/":
            continue
        s = s.replace("＝", "=").replace("：", "=").replace(":", "=")
        if "=" not in s:
            continue
        key, val = s.split("=", 1)
        key = key.strip().replace(" ", "")
        val = val.strip().replace(" ", "")
        if any(k in key for k in ("模式", "mode")):
            if any(v in val for v in ("換房", "巡房", "patrol")):
                out["mode"] = "patrol"
            elif any(v in val for v in ("掛房", "單桌", "stay")):
                out["mode"] = "stay"
            elif any(v in val for v in ("視窗", "視窗診斷", "list-windows", "windows")):
                out["list_windows"] = True
            elif any(v in val for v in ("餘額測試", "餘額", "balance")):
                out["balance_test"] = True
            elif any(v in val for v in ("檢查", "檢測", "自我", "selftest", "test")):
                out["selftest"] = True
        elif any(k in key for k in ("下注", "實戰", "bet", "live")):
            if any(v in val for v in ("開", "啟", "是", "on", "true", "yes", "1")):
                out["live"] = True
            elif any(v in val for v in ("關", "閉", "否", "off", "false", "no", "0", "測試")):
                out["live"] = False
        elif any(k in key for k in ("桌號", "桌号", "桌", "table")):
            import re as _re

            m = _re.search(r"\d+", val)
            if m:
                out["stay_table"] = int(m.group())
        elif any(k in key for k in ("對象", "对象", "跟注對象", "target")):
            # 一行可填多個（用 , 、 ， 空白分隔），也可多行累加
            parts = [p for p in _split_targets(val) if p]
            out["stay_targets"].extend(parts)
        elif any(k in key for k in ("帳號", "帐号", "帳戶", "account", "玩家")):
            if val:
                out["account_name"] = val
    return out


def _split_targets(val: str) -> list[str]:
    import re as _re

    return [p.strip() for p in _re.split(r"[,，、;；\s]+", val) if p.strip()]


def _balance_test() -> int:
    """讀一次左下角餘額與帳號名，存下截圖供校正 roi.balance / roi.account_name。"""
    import time

    import numpy as np
    from PIL import Image

    from star_follow.config import load_config
    from star_follow.monitor.sheet_uploader import _read_roi
    from star_follow.paths import logs_dir
    from star_follow.vision.ocr import ocr_account_name, ocr_balance

    cfg = load_config()
    out = logs_dir()
    out.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    print("=== 餘額／帳號讀取測試 ===")

    def _do(roi_key: str, label: str, ocr_fn):
        rect = cfg.roi.get(roi_key)
        print(f"\nroi.{roi_key} = {rect}")
        img = _read_roi(cfg, roi_key)
        if img is None:
            print(f"找不到遊戲視窗或未設定 roi.{roi_key}，請先把星城開好再測。")
            return
        crop_path = out / f"{roi_key}_{ts}.png"
        try:
            Image.fromarray(np.asarray(img)).save(crop_path)
            print(f"已存擷取畫面：{crop_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"存圖失敗：{exc}")
        value, _ = ocr_fn(img)
        if isinstance(value, int):
            print(f"OCR 讀到的{label}：{value:,}" if value else f"OCR 讀不到{label}（請校正 roi.{roi_key}）")
        else:
            print(f"OCR 讀到的{label}：{value!r}" if value else f"OCR 讀不到{label}（請校正 roi.{roi_key}）")

    _do("balance", "餘額", ocr_balance)
    _do("account_name", "帳號名", ocr_account_name)
    print(f"\nsheet.account_name 設定值：{cfg.sheet.account_name!r}（有填就用這個，不靠 OCR）")
    try:
        input("\n按 Enter 關閉本視窗...")
    except Exception:  # noqa: BLE001
        pass
    return 0


def _list_windows() -> int:
    """列出符合設定的星城視窗；找不到時列出其他大視窗供對照。"""
    from star_follow.capture.window import (
        find_game_window,
        list_candidate_windows,
        list_large_windows,
    )
    from star_follow.config import load_config

    cfg = load_config()
    wcfg = cfg.window
    print("=== 星城視窗診斷 ===")
    print(f"設定 title_substring: {wcfg.title_substring!r}")
    if wcfg.title_aliases:
        print(f"額外 title_aliases: {wcfg.title_aliases}")

    cands = list_candidate_windows(
        wcfg.title_substring,
        title_aliases=wcfg.title_aliases or None,
    )
    if cands:
        print("\n找到以下符合的視窗（程式會用最大那個）：")
        for w, h, hwnd, vis, mini, title in cands:
            print(f"  {w}x{h}  hwnd={hwnd}  可見={vis}  最小化={mini}")
            print(f"    標題: {title}")
        win = find_game_window(wcfg.title_substring, title_aliases=wcfg.title_aliases or None)
        if win:
            print(f"\n目前會使用: {win.client_width}x{win.client_height}  標題={win.title!r}")
        print("\n若程式仍說找不到，請確認星城沒被最小化，且建議已進入百家樂牌桌。")
        return 0

    print("\n[找不到] 沒有任何視窗標題符合「星城Online / 星城」等關鍵字。")
    print("\n請業主確認：")
    print("  1. 星城已開啟（不要只開登入器就關掉）")
    print("  2. 視窗不要最小化到工作列")
    print("  3. 建議已點進「百家樂」牌桌（不要只停在大廳選單）")
    print("  4. 星城與本程式都要用「系統管理員」執行（右鍵 → 以系統管理員身分執行）")
    print("\n以下是目前畫面上較大的視窗標題，請找哪一個是星城：")
    print("（找到後把完整標題填進 config.yaml 的 title_substring 或 title_aliases）\n")
    for w, h, vis, mini, title, cls in list_large_windows():
        print(f"  {w}x{h}  可見={vis}  最小化={mini}  類別={cls}  |  {title}")
    return 1


def _selftest() -> int:
    """打包後自我檢查：確認設定檔、語言檔、Tesseract、OCR 都能正確載入。"""
    import os

    from star_follow import paths
    from star_follow.config import load_config

    print("=== StarFollow 自我檢查 ===")
    print(f"打包模式 frozen: {paths.is_frozen()}")
    print(f"程式資料夾 app_dir: {paths.app_dir()}")
    print(f"設定檔 config.yaml: {paths.config_path()}  存在={paths.config_path().is_file()}")
    print(f"名單 follow_list.json: {paths.follow_list_path()}  存在={paths.follow_list_path().is_file()}")
    print(f"Tesseract 執行檔: {paths.tesseract_exe()}  存在={paths.tesseract_exe().is_file()}")
    print(f"語言檔資料夾 tessdata: {paths.tessdata_dir()}  存在={paths.tessdata_dir().is_dir()}")
    print(f"TESSDATA_PREFIX: {os.environ.get('TESSDATA_PREFIX')}")

    ok = True
    try:
        cfg = load_config()
        print(f"設定載入成功：room.mode={cfg.room.mode} tables={cfg.room.tables}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"[失敗] 設定載入：{exc}")

    try:
        fl = FollowList.load()
        print(f"跟注名單：{[e.name for e in fl.entries]}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"[失敗] 名單載入：{exc}")

    try:
        import numpy as np
        import pytesseract
        from PIL import Image

        print(f"Tesseract 命令：{pytesseract.pytesseract.tesseract_cmd}")
        blank = Image.fromarray(np.full((48, 160, 3), 255, dtype=np.uint8))
        # 載入 chi_tra+eng 做一次實際辨識（空白圖回空字串，但會驗證語言檔可被讀取）
        pytesseract.image_to_string(blank, lang="chi_tra+eng")
        print("Tesseract OCR（chi_tra+eng）載入成功")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"[失敗] Tesseract OCR：{exc}")

    print("=== 結果：", "全部通過 OK" if ok else "有錯誤 FAIL", "===")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live",
        action="store_true",
        help="連下注也真點擊（預設 dry-run：會開關統計表，但不點籌碼）",
    )
    parser.add_argument("--add", type=str, help="新增跟注暱稱後結束（不啟動引擎）")
    parser.add_argument("--patrol", action="store_true", help="換房模式（巡房跟注），覆寫 config 的 room.mode")
    parser.add_argument("--stay", action="store_true", help="掛房模式（單桌跟注），覆寫 config 的 room.mode")
    parser.add_argument("--selftest", action="store_true", help="自我檢查：確認設定檔/語言檔/OCR 可正常載入後結束")
    parser.add_argument("--balance-test", action="store_true", help="測試讀取左下角餘額與帳號名（校正 roi.balance / roi.account_name）")
    parser.add_argument("--list-windows", action="store_true", help="診斷：列出星城視窗（找不到時列出其他大視窗）")
    args = parser.parse_args()

    if getattr(args, "list_windows", False):
        return _list_windows()

    if getattr(args, "balance_test", False):
        return _balance_test()

    # 啟動設定檔（雙擊 exe 時用；命令列參數優先）
    settings = _read_launch_settings()

    if args.selftest or (settings["selftest"] and not (args.patrol or args.stay)):
        return _selftest()

    if settings["list_windows"] and not (args.patrol or args.stay):
        return _list_windows()

    if settings["balance_test"] and not (args.patrol or args.stay):
        return _balance_test()

    # 自動更新：偵測到新版會下載換檔並重啟（此行之後就不會再往下執行）
    try:
        from star_follow.config import load_config
        from star_follow.update.updater import maybe_update

        maybe_update(load_config().update)
    except Exception:  # noqa: BLE001  更新流程任何錯誤都不應擋住正常啟動
        pass

    # 模式：CLI > 設定檔 > config 預設
    if args.patrol:
        mode_override = "patrol"
    elif args.stay:
        mode_override = "stay"
    else:
        mode_override = settings["mode"]

    # 下注：CLI --live > 設定檔；預設為測試(不下注)以策安全
    if args.live:
        live = True
    elif settings["live"] is not None:
        live = settings["live"]
    else:
        live = False

    log_path = _setup_logging()
    fl = FollowList.load()
    if args.add:
        fl.add(args.add)
        fl.save()
        print(f"已加入跟注: {args.add}")
        return 0

    require_admin(live=live)
    engine = FollowEngine(follow=fl, dry_run=not live)
    if mode_override:
        engine.cfg.room.mode = mode_override
    # 掛房：套用啟動設定的「桌號 / 對象」
    if settings.get("stay_table") is not None:
        engine.cfg.room.stay_table = settings["stay_table"]
    if settings.get("stay_targets"):
        engine.cfg.room.stay_targets = list(settings["stay_targets"])
    # 掛房且有指定對象 → 用指定對象覆寫跟注名單（不動 follow_list.json）
    if engine.cfg.room.mode == "stay" and engine.cfg.room.stay_targets:
        engine.follow = FollowList.from_names(engine.cfg.room.stay_targets)
    # 餘額上傳用的帳號名：以啟動設定手動輸入為準（不用 OCR 讀名字，避免誤判）
    if settings.get("account_name"):
        engine.cfg.sheet.account_name = settings["account_name"]
    t = engine.cfg.timing
    mode = "LIVE（含下注）" if not engine.dry_run else "dry-run（開關統計+OCR，不下注）"
    room_mode = "換房巡房" if engine.cfg.room.mode == "patrol" else "掛房（單桌）"
    admin_tag = "管理員" if is_admin() else "一般使用者"
    from star_follow.version import __version__ as _ver

    print(f"StarFollow v{_ver}")
    print("引擎啟動", mode, f"[{room_mode}]", f"({admin_tag})")
    print(f"記錄檔：{log_path}")
    from star_follow.vision.ocr import warmup_ocr

    warmup_ocr()
    if engine.cfg.room.mode == "patrol":
        print(f"換房：進房綠燈 T>={engine.cfg.room.min_enter_t} 才開統計跟注；下注後等開牌再換桌")
        print(f"桌號序列：{engine.cfg.room.tables}")
    else:
        print(
            f"時間軸：T={t.open_stats_at_t} 開統計 -> T<={t.prefetch_at_t} 預定位 -> "
            f"T={t.finalize_at_t} 定稿（僅 T>={t.min_round_start_t} 且錨定成功才開局）"
        )
        if engine.cfg.room.stay_table:
            print(f"掛房固定桌號：No.{engine.cfg.room.stay_table}（被踢/當機會自動回此桌）")
        else:
            print("掛房未指定桌號（待在當前桌；回大廳時回任一桌）。可在啟動設定填『桌號=5』")
        tgt = engine.cfg.room.stay_targets or fl.active()
        print(f"掛房追蹤對象：{('、'.join(tgt) if tgt else '（無）')}")
        if engine.cfg.room.stay_pause_when_targets_absent:
            print(
                f"對象全離桌偵測：連續 {engine.cfg.room.stay_absent_rounds_to_pause} 局都讀不到任何對象 → 暫停跟注、不回桌"
            )
    if engine.dry_run:
        print("真下注：python -m star_follow.tools.run --live")
        print("標記籌碼：python -m star_follow.tools.mark_chips")

    uploader = None
    if engine.cfg.sheet.enabled:
        from star_follow.monitor.sheet_uploader import SheetUploader

        uploader = SheetUploader(engine.cfg)
        uploader.start()
        print("餘額上傳：每整點上傳到 Google 試算表")

    try:
        engine.run_loop()
    except KeyboardInterrupt:
        engine.stop()
    finally:
        if uploader is not None:
            uploader.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
