from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from star_follow.paths import config_path as _config_path

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = _config_path()


@dataclass
class WindowConfig:
    title_substring: str = "星城Online"
    title_aliases: list[str] = field(default_factory=list)  # 額外標題關鍵字，比對不到時可填
    reference_width: int = 1024
    reference_height: int = 640


@dataclass
class TimingConfig:
    poll_ms: int = 250
    open_stats_at_t: int = 20
    open_stats_latest_t: int = 15
    min_round_start_t: int = 19
    prefetch_at_t: int = 15
    finalize_at_t: int = 13
    ocr_at_t: int = 8
    ocr_retry_at_t: int = 6
    min_bet_t: int = 4
    max_bet_t: int = 10
    save_screenshot_on_ocr: bool = True
    click_delay_ms_min: int = 30
    click_delay_ms_max: int = 80
    bet_click_delay_ms_min: int = 15
    bet_click_delay_ms_max: int = 35
    software_countdown: bool = True
    countdown_anchor_t: int = 20
    countdown_resync_tolerance: int = 1


@dataclass
class StatsConfig:
    manual_panel: bool = False
    open_retries: int = 3
    close_retries: int = 5
    menu_delay_s: float = 0.65
    stats_open_wait_s: float = 2.5
    close_click_wait_s: float = 0.35
    force_clean_on_round_start: bool = True
    idle_cleanup: bool = False
    idle_cleanup_cooldown_s: float = 45.0


@dataclass
class VisionConfig:
    fast_ocr: bool = True
    ocr_scale: int = 1
    header_use_paddle: bool = True
    # 整排表頭 OCR 的總時間預算（秒）。超時就停止掃描剩餘欄位，避免慢機器卡在
    # 「開表→整排 OCR 20s+→逾時→重來」的死循環。對象沒對到時改走防踢路徑。
    header_ocr_budget_s: float = 8.0


@dataclass
class AutomationConfig:
    menu_click_backend: str = "postmessage"
    ui_click_backend: str = "win32"


@dataclass
class SafetyConfig:
    max_ui_fail_streak: int = 5      # 連續這麼多局 UI 異常就停止引擎（0=不停）
    max_rounds: int = 0              # 本次最多下注局數（0=不限）
    max_total_stake: int = 0         # 本次累計下注金額上限（0=不限）
    max_stake_per_round: int = 0     # 單局下注金額上限（0=不限，超過則跳過該局）
    audit_screenshot: bool = True    # 下注後存截圖供事後核對


@dataclass
class RoomConfig:
    """換房（巡房）模式設定。座標點/矩形放在 click_points / roi：
    click_points: room_switch_button（開桌號清單的中間圖示）、room_list_scroll（清單內捲動點）
    roi: room_no_col（清單左側桌號數字欄，用來 OCR 各列桌號+取得列 y）、room_current_table（左下角目前桌號）
    """

    mode: str = "stay"  # "stay"=掛房, "patrol"=換房
    tables: list[int] = field(
        default_factory=lambda: [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19]
    )
    min_enter_t: int = 8  # 進房後綠燈 T 至少這麼大才開統計跟注，否則換下一桌
    goto_x: int = 0  # 「前往」鈕欄位中心 x（參考座標）
    switch_confirm_s: float = 1.6  # 點前往後等多久確認左下桌號是否改變
    result_wait_timeout_s: float = 75.0  # 下注後等開牌可換房的最長等待
    result_retry_s: float = 3.0  # 等開牌期間每隔多久重試換桌
    scroll_clicks: int = 1  # 清單往下捲一次的滾輪格數（細捲以免漏中間列）
    max_scroll_pages: int = 14  # 清單最多往下捲幾次找桌號
    # 換房模式專用定稿時間（與 timing 區段分開；掛房改 timing 不影響換房）
    patrol_finalize_at_t: int = 14
    patrol_ocr_at_t: int = 11
    patrol_ocr_retry_at_t: int = 9
    # 黏桌：連續幾局沒跟到邊注（含 OCR 暫時漏讀）才換桌。設 >1 可容忍偶發漏讀，
    # 避免對象其實還在下、卻因一次讀失敗就提早換桌。
    patrol_leave_after_idle: int = 2
    # 掛房（stay）專用：指定要固定待著的桌號。被踢/當機回大廳時，會自動回到這一桌。
    # 0 = 不指定（待在當前桌，回大廳時回任一桌）。
    stay_table: int = 0
    # 掛房指定追蹤對象（覆寫 follow_list.json）。空 = 沿用 follow_list.json。
    stay_targets: list[str] = field(default_factory=list)
    # 掛房：是否偵測「指定對象全部都不在這桌」。
    stay_pause_when_targets_absent: bool = True
    # 連續這麼多局都讀不到任何指定對象，才判定「對象全離桌」（容忍偶發漏讀）。
    stay_absent_rounds_to_pause: int = 1
    # 對象全離桌時的處置（stay_on_absent）：
    #   keep  = 只發 TG 通知，但程式照常跑（繼續防踢補注、被踢照樣自動回桌），
    #           待對象某局又出現就自動恢復跟注（推薦，最不會卡）。
    #   pause = 發 TG 後暫停跟注（不防踢、被踢也不回桌），但程式不結束。
    #   stop  = 發 TG 後直接停止程式。
    stay_on_absent: str = "keep"
    # 舊設定（相容用）：對象全離桌→停止程式。僅在未設定 stay_on_absent 時生效。
    stay_stop_when_targets_absent: bool = True


@dataclass
class BettingConfig:
    """下注規則：哪些邊不跟、掛房防踢補注。"""

    # 兩模式都「不跟」這些下注區（跟注計畫會濾掉）。預設不跟莊、閒。
    follow_exclude: list[str] = field(default_factory=lambda: ["莊", "閒"])
    # 掛房防踢：連續這麼多局我方都沒下注，就自己補一手最小注，避免被系統踢出。
    anti_kick_enabled: bool = True
    anti_kick_idle_rounds: int = 3   # 連續幾局沒下注就補注
    anti_kick_side: str = "閒"        # 補注下哪一邊（莊/閒）
    anti_kick_amount: int = 0         # 補注金額，0=用最小籌碼
    # 掛房：首次 OCR 追到對象時先補一手閒（抵消進桌/OCR 暖機的閒置），之後才累計未下注局數
    anti_kick_first_track_bet: bool = True
    # 跟注金額擬真：避免每把都跟對方一模一樣太明顯。
    # 下注額 = 對方金額 × 隨機比例(min~max)，再「無條件進位」到 round_to 的倍數（最低一注）。
    # 例：對方 10000、比例 0.83 → 8300 → 進位到 9000。範圍可在「啟動設定.txt」調整。
    follow_ratio_enabled: bool = True
    follow_ratio_min: float = 0.8
    follow_ratio_max: float = 0.99
    follow_ratio_round_to: int = 1000


# 內建 Telegram 通知預設：放在「程式碼」裡，才會被打包進 exe，並隨自動更新送到
# 每一台（含自動更新者——他們的 config.yaml 會被保留、拿不到新 token，所以不能只放
# config.yaml）。config.yaml 的 bot_token/chat_id 留空就用這組；要換 bot 改這裡即可。
DEFAULT_BOT_TOKEN = "8893028334:AAGNJQt5djgMjjmCswxfwTUdh2h7VdvoXMM"
DEFAULT_CHAT_ID = "-5153638599"


@dataclass
class TelegramConfig:
    """Telegram 機器人通知（掛桌對象全離桌等）。"""

    enabled: bool = False
    bot_token: str = DEFAULT_BOT_TOKEN  # 找 @BotFather 建 bot 取得；留空用內建預設
    chat_id: str = DEFAULT_CHAT_ID      # 你的聊天室/個人 chat id；留空用內建預設
    interval_min: float = 30.0  # 每隔幾分鐘回報一次
    report_on_start: bool = True  # 啟動後先報一次（建立基準）


@dataclass
class UpdateConfig:
    """自動更新：啟動時偵測雲端版本，較新就下載更新包並換檔。"""

    enabled: bool = True
    manifest_url: str = ""        # version.json 的網址；留空則不檢查
    auto_apply: bool = True       # true=自動換檔重啟；false=只提示有新版
    check_on_start: bool = True   # 啟動時檢查
    timeout_s: float = 8.0        # 連線逾時（離線時不卡太久）


@dataclass
class SheetConfig:
    """每整點把本機帳號餘額上傳到 Google 試算表（集中統計用）。"""

    enabled: bool = False
    service_account_file: str = "service_account.json"  # 服務帳戶金鑰檔（放 exe 旁）
    spreadsheet_id: str = ""   # 試算表 ID（網址 /d/ 後那段）
    worksheet: str = "餘額"     # 工作表名稱
    account_name: str = ""     # 本機帳號名稱；留空則用 OCR 讀桌內帳號名
    upload_on_start: bool = True  # 啟動先上傳一次


@dataclass
class AppConfig:
    window: WindowConfig = field(default_factory=WindowConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    automation: AutomationConfig = field(default_factory=AutomationConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    room: RoomConfig = field(default_factory=RoomConfig)
    betting: BettingConfig = field(default_factory=BettingConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    sheet: SheetConfig = field(default_factory=SheetConfig)
    update: UpdateConfig = field(default_factory=UpdateConfig)
    roi: dict[str, list[int]] = field(default_factory=dict)
    click_points: dict[str, list[int]] = field(default_factory=dict)
    chips: dict[str, list[int]] = field(default_factory=dict)
    bet_areas: dict[str, list[int]] = field(default_factory=dict)
    stats_rows: list[str] = field(default_factory=list)
    stats_to_bet: dict[str, str] = field(default_factory=dict)
    chip_values: list[int] = field(default_factory=lambda: [1000, 5000, 10000, 50000, 100000, 500000])
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def config_path(self) -> Path | None:
        p = self.raw.get("_path")
        return Path(p) if p else None


def _rect(d: dict[str, Any], key: str, default: list[int]) -> list[int]:
    v = d.get(key, default)
    if isinstance(v, dict):
        return [int(v["x"]), int(v["y"]), int(v["w"]), int(v["h"])]
    return [int(x) for x in v]


def _resolve_stay_on_absent(rm: dict[str, Any]) -> str:
    """決定對象全離桌的處置模式。優先用 stay_on_absent；沒設才從舊旗標推導，
    確保舊 config（只有 stay_stop_when_targets_absent）行為不變。"""
    raw = rm.get("stay_on_absent")
    if raw:
        v = str(raw).strip().lower()
        if v in ("keep", "pause", "stop"):
            return v
    # 舊設定相容：有停止旗標→stop；否則→pause（舊「暫停」語意）
    if bool(rm.get("stay_stop_when_targets_absent", True)):
        return "stop"
    return "pause"


def load_config(path: Path | str | None = None) -> AppConfig:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.is_file():
        return AppConfig(raw={"_path": str(path)})

    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    w = data.get("window", {})
    t = data.get("timing", {})
    s = data.get("stats", {})
    v = data.get("vision", {})
    a = data.get("automation", {})
    sf = data.get("safety", {})
    rm = data.get("room", {})
    bt = data.get("betting", {})
    tg = data.get("telegram", {})
    sh = data.get("sheet", {})
    up = data.get("update", {})
    cfg = AppConfig(
        window=WindowConfig(
            title_substring=w.get("title_substring", "星城Online"),
            title_aliases=[str(x) for x in (w.get("title_aliases") or [])],
            reference_width=int(w.get("reference_width", 1024)),
            reference_height=int(w.get("reference_height", 640)),
        ),
        timing=TimingConfig(
            poll_ms=int(t.get("poll_ms", 250)),
            open_stats_at_t=int(t.get("open_stats_at_t", 20)),
            open_stats_latest_t=int(t.get("open_stats_latest_t", 15)),
            min_round_start_t=int(t.get("min_round_start_t", 19)),
            prefetch_at_t=int(t.get("prefetch_at_t", 15)),
            finalize_at_t=int(t.get("finalize_at_t", 13)),
            ocr_at_t=int(t.get("ocr_at_t", 11)),
            ocr_retry_at_t=int(t.get("ocr_retry_at_t", 9)),
            min_bet_t=int(t.get("min_bet_t", 4)),
            max_bet_t=int(t.get("max_bet_t", 9)),
            save_screenshot_on_ocr=bool(t.get("save_screenshot_on_ocr", True)),
            click_delay_ms_min=int(t.get("click_delay_ms_min", 30)),
            click_delay_ms_max=int(t.get("click_delay_ms_max", 80)),
            bet_click_delay_ms_min=int(t.get("bet_click_delay_ms_min", 15)),
            bet_click_delay_ms_max=int(t.get("bet_click_delay_ms_max", 35)),
            software_countdown=bool(t.get("software_countdown", True)),
            countdown_anchor_t=int(t.get("countdown_anchor_t", 20)),
            countdown_resync_tolerance=int(t.get("countdown_resync_tolerance", 1)),
        ),
        stats=StatsConfig(
            manual_panel=bool(s.get("manual_panel", False)),
            open_retries=int(s.get("open_retries", 3)),
            close_retries=int(s.get("close_retries", 5)),
            menu_delay_s=float(s.get("menu_delay_s", 0.65)),
            stats_open_wait_s=float(s.get("stats_open_wait_s", 2.5)),
            close_click_wait_s=float(s.get("close_click_wait_s", 0.35)),
            force_clean_on_round_start=bool(s.get("force_clean_on_round_start", True)),
            idle_cleanup=bool(s.get("idle_cleanup", False)),
            idle_cleanup_cooldown_s=float(s.get("idle_cleanup_cooldown_s", 45.0)),
        ),
        vision=VisionConfig(
            fast_ocr=bool(v.get("fast_ocr", True)),
            ocr_scale=int(v.get("ocr_scale", 1)),
            header_use_paddle=bool(v.get("header_use_paddle", True)),
            header_ocr_budget_s=float(v.get("header_ocr_budget_s", 8.0)),
        ),
        automation=AutomationConfig(
            menu_click_backend=str(
                a.get("menu_click_backend", a.get("click_backend", "postmessage"))
            ),
            ui_click_backend=str(
                a.get("ui_click_backend", a.get("click_backend", "win32"))
            ),
        ),
        safety=SafetyConfig(
            max_ui_fail_streak=int(sf.get("max_ui_fail_streak", 5)),
            max_rounds=int(sf.get("max_rounds", 0)),
            max_total_stake=int(sf.get("max_total_stake", 0)),
            max_stake_per_round=int(sf.get("max_stake_per_round", 0)),
            audit_screenshot=bool(sf.get("audit_screenshot", True)),
        ),
        room=RoomConfig(
            mode=str(rm.get("mode", "stay")),
            tables=[int(x) for x in (rm.get("tables") or [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19])],
            min_enter_t=int(rm.get("min_enter_t", 8)),
            goto_x=int(rm.get("goto_x", 0)),
            switch_confirm_s=float(rm.get("switch_confirm_s", 1.6)),
            result_wait_timeout_s=float(rm.get("result_wait_timeout_s", 75.0)),
            result_retry_s=float(rm.get("result_retry_s", 3.0)),
            scroll_clicks=int(rm.get("scroll_clicks", 1)),
            max_scroll_pages=int(rm.get("max_scroll_pages", 14)),
            patrol_finalize_at_t=int(rm.get("patrol_finalize_at_t", 14)),
            patrol_ocr_at_t=int(rm.get("patrol_ocr_at_t", 11)),
            patrol_ocr_retry_at_t=int(rm.get("patrol_ocr_retry_at_t", 9)),
            patrol_leave_after_idle=int(rm.get("patrol_leave_after_idle", 2)),
            stay_table=int(rm.get("stay_table", 0) or 0),
            stay_targets=[str(x) for x in (rm.get("stay_targets") or [])],
            stay_pause_when_targets_absent=bool(rm.get("stay_pause_when_targets_absent", True)),
            stay_absent_rounds_to_pause=int(rm.get("stay_absent_rounds_to_pause", 1)),
            stay_stop_when_targets_absent=bool(rm.get("stay_stop_when_targets_absent", True)),
            stay_on_absent=_resolve_stay_on_absent(rm),
        ),
        betting=BettingConfig(
            follow_exclude=[str(x) for x in (bt.get("follow_exclude") or ["莊", "閒"])],
            anti_kick_enabled=bool(bt.get("anti_kick_enabled", True)),
            anti_kick_idle_rounds=int(bt.get("anti_kick_idle_rounds", 3)),
            anti_kick_side=str(bt.get("anti_kick_side", "閒")),
            anti_kick_amount=int(bt.get("anti_kick_amount", 0)),
            anti_kick_first_track_bet=bool(bt.get("anti_kick_first_track_bet", True)),
            follow_ratio_enabled=bool(bt.get("follow_ratio_enabled", True)),
            follow_ratio_min=float(bt.get("follow_ratio_min", 0.8)),
            follow_ratio_max=float(bt.get("follow_ratio_max", 0.99)),
            follow_ratio_round_to=int(bt.get("follow_ratio_round_to", 1000)),
        ),
        telegram=TelegramConfig(
            enabled=bool(tg.get("enabled", False)),
            bot_token=str(tg.get("bot_token") or DEFAULT_BOT_TOKEN),
            chat_id=str(tg.get("chat_id") or DEFAULT_CHAT_ID),
            interval_min=float(tg.get("interval_min", 30.0)),
            report_on_start=bool(tg.get("report_on_start", True)),
        ),
        sheet=SheetConfig(
            enabled=bool(sh.get("enabled", False)),
            service_account_file=str(sh.get("service_account_file", "service_account.json")),
            spreadsheet_id=str(sh.get("spreadsheet_id", "")),
            worksheet=str(sh.get("worksheet", "餘額")),
            account_name=str(sh.get("account_name", "")),
            upload_on_start=bool(sh.get("upload_on_start", True)),
        ),
        update=UpdateConfig(
            enabled=bool(up.get("enabled", True)),
            manifest_url=str(up.get("manifest_url", "")),
            auto_apply=bool(up.get("auto_apply", True)),
            check_on_start=bool(up.get("check_on_start", True)),
            timeout_s=float(up.get("timeout_s", 8.0)),
        ),
        roi={
            k: _rect(data.get("roi", {}), k, v)
            for k, v in (data.get("roi") or {}).items()
            if isinstance(v, list) and len(v) == 4
        },
        click_points={
            k: [int(x) for x in v]
            for k, v in (data.get("click_points") or {}).items()
            if isinstance(v, list) and len(v) == 2
        },
        chips={str(k): list(v) for k, v in (data.get("chips") or {}).items()},
        bet_areas={k: list(v) for k, v in (data.get("bet_areas") or {}).items()},
        stats_rows=list(data.get("stats_rows") or []),
        stats_to_bet=dict(data.get("stats_to_bet") or {}),
        chip_values=[int(x) for x in (data.get("chip_values") or [1000, 5000, 10000, 50000, 100000, 500000])],
        raw={"_path": str(path), **data},
    )
    return cfg


def save_config(cfg: AppConfig, path: Path | str | None = None) -> Path:
    path = Path(path) if path else (cfg.config_path or DEFAULT_CONFIG_PATH)
    data = dict(cfg.raw)
    data.pop("_path", None)
    data["window"] = {
        "title_substring": cfg.window.title_substring,
        "title_aliases": cfg.window.title_aliases,
        "reference_width": cfg.window.reference_width,
        "reference_height": cfg.window.reference_height,
    }
    data["timing"] = {
        "poll_ms": cfg.timing.poll_ms,
        "open_stats_at_t": cfg.timing.open_stats_at_t,
        "open_stats_latest_t": cfg.timing.open_stats_latest_t,
        "min_round_start_t": cfg.timing.min_round_start_t,
        "prefetch_at_t": cfg.timing.prefetch_at_t,
        "finalize_at_t": cfg.timing.finalize_at_t,
        "ocr_at_t": cfg.timing.ocr_at_t,
        "ocr_retry_at_t": cfg.timing.ocr_retry_at_t,
        "min_bet_t": cfg.timing.min_bet_t,
        "max_bet_t": cfg.timing.max_bet_t,
        "save_screenshot_on_ocr": cfg.timing.save_screenshot_on_ocr,
        "click_delay_ms_min": cfg.timing.click_delay_ms_min,
        "click_delay_ms_max": cfg.timing.click_delay_ms_max,
        "bet_click_delay_ms_min": cfg.timing.bet_click_delay_ms_min,
        "bet_click_delay_ms_max": cfg.timing.bet_click_delay_ms_max,
        "software_countdown": cfg.timing.software_countdown,
        "countdown_anchor_t": cfg.timing.countdown_anchor_t,
        "countdown_resync_tolerance": cfg.timing.countdown_resync_tolerance,
    }
    data["stats"] = {
        "manual_panel": cfg.stats.manual_panel,
        "open_retries": cfg.stats.open_retries,
        "close_retries": cfg.stats.close_retries,
        "menu_delay_s": cfg.stats.menu_delay_s,
        "stats_open_wait_s": cfg.stats.stats_open_wait_s,
        "close_click_wait_s": cfg.stats.close_click_wait_s,
        "force_clean_on_round_start": cfg.stats.force_clean_on_round_start,
        "idle_cleanup": cfg.stats.idle_cleanup,
        "idle_cleanup_cooldown_s": cfg.stats.idle_cleanup_cooldown_s,
    }
    data["vision"] = {
        "fast_ocr": cfg.vision.fast_ocr,
        "ocr_scale": cfg.vision.ocr_scale,
        "header_use_paddle": cfg.vision.header_use_paddle,
        "header_ocr_budget_s": cfg.vision.header_ocr_budget_s,
    }
    data["automation"] = {
        "menu_click_backend": cfg.automation.menu_click_backend,
        "ui_click_backend": cfg.automation.ui_click_backend,
    }
    data["safety"] = {
        "max_ui_fail_streak": cfg.safety.max_ui_fail_streak,
        "max_rounds": cfg.safety.max_rounds,
        "max_total_stake": cfg.safety.max_total_stake,
        "max_stake_per_round": cfg.safety.max_stake_per_round,
        "audit_screenshot": cfg.safety.audit_screenshot,
    }
    data["room"] = {
        "mode": cfg.room.mode,
        "tables": cfg.room.tables,
        "min_enter_t": cfg.room.min_enter_t,
        "goto_x": cfg.room.goto_x,
        "switch_confirm_s": cfg.room.switch_confirm_s,
        "result_wait_timeout_s": cfg.room.result_wait_timeout_s,
        "result_retry_s": cfg.room.result_retry_s,
        "scroll_clicks": cfg.room.scroll_clicks,
        "max_scroll_pages": cfg.room.max_scroll_pages,
        "patrol_finalize_at_t": cfg.room.patrol_finalize_at_t,
        "patrol_ocr_at_t": cfg.room.patrol_ocr_at_t,
        "patrol_ocr_retry_at_t": cfg.room.patrol_ocr_retry_at_t,
        "patrol_leave_after_idle": cfg.room.patrol_leave_after_idle,
        "stay_table": cfg.room.stay_table,
        "stay_targets": cfg.room.stay_targets,
        "stay_pause_when_targets_absent": cfg.room.stay_pause_when_targets_absent,
        "stay_absent_rounds_to_pause": cfg.room.stay_absent_rounds_to_pause,
        "stay_stop_when_targets_absent": cfg.room.stay_stop_when_targets_absent,
        "stay_on_absent": cfg.room.stay_on_absent,
    }
    data["betting"] = {
        "follow_exclude": cfg.betting.follow_exclude,
        "anti_kick_enabled": cfg.betting.anti_kick_enabled,
        "anti_kick_idle_rounds": cfg.betting.anti_kick_idle_rounds,
        "anti_kick_side": cfg.betting.anti_kick_side,
        "anti_kick_amount": cfg.betting.anti_kick_amount,
        "anti_kick_first_track_bet": cfg.betting.anti_kick_first_track_bet,
        "follow_ratio_enabled": cfg.betting.follow_ratio_enabled,
        "follow_ratio_min": cfg.betting.follow_ratio_min,
        "follow_ratio_max": cfg.betting.follow_ratio_max,
        "follow_ratio_round_to": cfg.betting.follow_ratio_round_to,
    }
    data["telegram"] = {
        "enabled": cfg.telegram.enabled,
        "bot_token": cfg.telegram.bot_token,
        "chat_id": cfg.telegram.chat_id,
        "interval_min": cfg.telegram.interval_min,
        "report_on_start": cfg.telegram.report_on_start,
    }
    data["sheet"] = {
        "enabled": cfg.sheet.enabled,
        "service_account_file": cfg.sheet.service_account_file,
        "spreadsheet_id": cfg.sheet.spreadsheet_id,
        "worksheet": cfg.sheet.worksheet,
        "account_name": cfg.sheet.account_name,
        "upload_on_start": cfg.sheet.upload_on_start,
    }
    data["update"] = {
        "enabled": cfg.update.enabled,
        "manifest_url": cfg.update.manifest_url,
        "auto_apply": cfg.update.auto_apply,
        "check_on_start": cfg.update.check_on_start,
        "timeout_s": cfg.update.timeout_s,
    }
    data["roi"] = cfg.roi
    data["click_points"] = cfg.click_points
    data["chips"] = cfg.chips
    data["bet_areas"] = cfg.bet_areas
    data["stats_rows"] = cfg.stats_rows
    data["stats_to_bet"] = cfg.stats_to_bet
    data["chip_values"] = cfg.chip_values

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    return path
