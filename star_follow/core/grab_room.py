"""搶房模式：人已經在某一間百家樂房內，程式一直搶「指定房號」，搶到後守房不被踢。

流程（模仿人工）：
  1. 使用者自己先進任一間房，再開搶房程式（程式不判斷大廳、不會自己進房）。
  2. 開桌號清單，一直點指定房號的「前往」；滿房點不進去就繼續點。
  3. 搶房期間每 2 局（或太久沒下）就關清單、在「目前這間」補一手最小注，
     免得搶著搶著被這間踢回大廳。
  4. 左上角「No.X」＝指定房號 → 發 TG「已搶到 No.X」。
  5. 守房：不再換房，照樣定時補最小注維持在房內，直到使用者自己關掉程式
     （再改開原本的跟注程式）。
  ＊ 左上角房號持續讀不到 → 視為離開房間：發 TG、暫停（不下注、不搶、不自己回去），
    等使用者手動回到任一間房後自動繼續。
  每小時發一則 TG 心跳，讓人知道程式還在跑。

只用現成元件（換桌／倒數 OCR／防踢下注），不讀統計表、不辨識名字，CPU 很輕。
"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum

from star_follow.automation import lobby_nav
from star_follow.automation.room_nav import (
    click_goto_at,
    close_room_list,
    confirm_switch,
    find_target_row,
    goto_is_lit,
    list_is_open,
    open_room_list,
    read_room_label,
)
from star_follow.capture.screen import CaptureUnavailable, capture_client
from star_follow.capture.window import focus_window
from star_follow.core.engine import FollowEngine, RoundContext
from star_follow.notify.telegram import send_message
from star_follow.vision.state import CountdownColor

logger = logging.getLogger(__name__)

_GRAB_MAX_IDLE_S = 90.0      # 搶房/守房：距上次下注超過這麼久就補（遊戲 5 局≈200 秒踢人，留足緩衝）
_BET_WAIT_S = 60.0           # 要補注時最多等多久的綠燈下注窗
_OFF_ROOM_CONFIRM_S = 20.0   # 左上角房號連續這麼久讀不到，才當作已離開房間
_HOLD_VERIFY_S = 10.0        # 守房時多久核對一次左上角房號
_LOST_CONFIRM_READS = 3      # 守房時連續幾次讀到「別的房號」才算離開指定房
_HEARTBEAT_S = 3600.0        # TG 心跳間隔（每小時）
_ROW_RECHECK_S = 5.0         # 清單開著時多久重新 OCR 一次目標列位置（防清單飄移點錯列）
_GO_CLICK_GAP_S = 0.5        # 兩次點 GO 的最短間隔
_NOWIN_CONFIRM_S = 30.0      # 找不到遊戲視窗多久才發 TG


class GrabPhase(Enum):
    GRABBING = "搶房中"
    HOLDING = "守房中"
    AWAY = "已離開房間（暫停）"


class GrabRoomRunner:
    """搶房主迴圈。借用 FollowEngine 的倒數 OCR 與防踢下注，不跑跟注流程。"""

    def __init__(self, engine: FollowEngine, target: int) -> None:
        self.eng = engine
        self.cfg = engine.cfg
        self.target = int(target)
        self.phase = GrabPhase.GRABBING
        self._running = False
        self._started = time.monotonic()
        self._attempts = 0
        self._got_at: float | None = None
        # 防踢：靠倒數 OCR 數局（看到新一局綠燈就 +1），再加一條時間保險
        self._last_bet_mono = time.monotonic()
        self._rounds_since_bet = 0
        self._last_cd_t: int | None = None
        self._last_cd_green = False
        # 在不在房內（左上角房號）
        self._cur_room: int | None = None
        self._label_miss_since: float | None = None
        self._away_since: float | None = None
        self._nowin_since: float | None = None
        self._nowin_notified = False
        # 守房核對
        self._last_verify_mono = 0.0
        self._lost_reads = 0
        self._last_heartbeat = time.monotonic()
        self._last_popup_check = 0.0
        self._last_log_mono = 0.0
        self._row_y: int | None = None
        self._row_checked = 0.0

    # ---------------- 通知 ----------------
    def _tg(self, text: str) -> None:
        tg = self.cfg.telegram
        if not tg.bot_token or not tg.chat_id:
            return
        threading.Thread(target=send_message, args=(tg.bot_token, tg.chat_id, text), daemon=True).start()

    def _heartbeat(self) -> None:
        now = time.monotonic()
        if now - self._last_heartbeat < _HEARTBEAT_S:
            return
        self._last_heartbeat = now
        hours = (now - self._started) / 3600.0
        if self.phase == GrabPhase.HOLDING:
            held = (now - (self._got_at or now)) / 3600.0
            self._tg(f"🟢 搶房程式運行中：守在 No.{self.target}（已守 {held:.1f} 小時），持續補注防踢")
        elif self.phase == GrabPhase.AWAY:
            away = (now - (self._away_since or now)) / 60.0
            self._tg(f"🔴 搶房程式運行中但已離開房間（約 {away:.0f} 分鐘），請手動回到房間，回去後會自動繼續")
        else:
            where = f"No.{self._cur_room}" if self._cur_room else "（讀不到房號）"
            self._tg(
                f"🟡 搶房程式運行中：還在搶 No.{self.target}（已跑 {hours:.1f} 小時、嘗試 {self._attempts} 次），"
                f"目前待在 {where}"
            )

    # ---------------- 倒數／防踢 ----------------
    def _read_cd(self) -> tuple[int | None, CountdownColor]:
        try:
            cd = self.eng._read_cd(capture_client(self.eng._win))
        except CaptureUnavailable:
            raise
        except Exception:  # noqa: BLE001
            return None, CountdownColor.OTHER
        return cd.seconds, cd.color

    def _track_round(self, t: int | None, color: CountdownColor) -> None:
        """看到「新一局開盤」（綠燈且秒數往上跳）就算一局。讀不到倒數就不動，交給時間保險。"""
        if t is None:
            return
        green = color == CountdownColor.GREEN
        if green and (not self._last_cd_green or (self._last_cd_t is not None and t > self._last_cd_t + 3)):
            self._rounds_since_bet += 1
            logger.info("搶房：新一局開盤 T=%s（距上次下注 %d 局）", t, self._rounds_since_bet)
        self._last_cd_t = t
        self._last_cd_green = green

    def _bet_due(self) -> bool:
        idle_rounds = max(1, int(self.cfg.betting.anti_kick_idle_rounds))
        if self._rounds_since_bet >= idle_rounds:
            return True
        return time.monotonic() - self._last_bet_mono >= _GRAB_MAX_IDLE_S

    def _keepalive_bet(self, win) -> bool:
        """關清單 → 等綠燈下注窗 → 補一手最小注。回傳是否有下到。"""
        cap = lambda: capture_client(win)
        close_room_list(win, self.cfg, cap)
        self._row_y = None  # 清單重開後列位置可能變，重新找
        deadline = time.monotonic() + _BET_WAIT_S
        while self._running and time.monotonic() < deadline:
            t, color = self._read_cd()
            self._track_round(t, color)
            # 比一般補注保守一點：至少留 5 秒給點籌碼，免得點到一半封盤
            if t is not None and t >= max(5, self.cfg.timing.min_bet_t) and self.eng._can_bet_now(t, color):
                self.eng.ctx = RoundContext()
                reason = f"搶房防踢補注（{self.phase.value}，距上次 {time.monotonic() - self._last_bet_mono:.0f} 秒）"
                if self.eng._place_anti_kick_bet(cap(), t, color, reason=reason):
                    self._last_bet_mono = time.monotonic()
                    self._rounds_since_bet = 0
                    return True
                return False
            time.sleep(0.5)
        logger.warning("搶房：%.0f 秒內等不到可下注的綠燈，稍後再試", _BET_WAIT_S)
        return False

    # ---------------- 在不在房內（左上角 No.X） ----------------
    def _update_presence(self, frame, win) -> bool:
        """讀左上角房號，更新 _cur_room；回傳「目前是否算在房內」。

        讀不到時不立刻判離開：清單開著（可能擋到）或剛換房動畫中都會短暫讀不到，
        要連續 _OFF_ROOM_CONFIRM_S 秒都讀不到才算離開。
        """
        cur = read_room_label(frame, self.cfg, win)
        now = time.monotonic()
        if cur is not None:
            self._cur_room = cur
            self._label_miss_since = None
            if self.phase == GrabPhase.AWAY:
                self._on_back_in_room(cur)
            return True
        if self.phase == GrabPhase.AWAY:
            return False
        if list_is_open(frame, self.cfg, win):
            self._label_miss_since = None
            return True  # 清單開著擋到標籤也算在房內
        if self._label_miss_since is None:
            self._label_miss_since = now
            return True
        if now - self._label_miss_since < _OFF_ROOM_CONFIRM_S:
            return True
        self._on_left_room(now - self._label_miss_since)
        return False

    def _on_left_room(self, gone_s: float) -> None:
        was = self.phase
        self.phase = GrabPhase.AWAY
        self._away_since = time.monotonic()
        logger.warning("搶房：左上角房號已 %.0f 秒讀不到 → 判定離開房間，暫停（不會自己回房）", gone_s)
        if was == GrabPhase.HOLDING:
            head = f"⚠ 已離開搶到的 No.{self.target} 房間！"
        else:
            head = f"⚠ 搶房：已離開房間（搶 No.{self.target} 中）"
        self._tg(
            f"{head}\n程式不會自動回房，已暫停下注與搶房；"
            "請手動進入任一間房（或直接進 No.{0}），回去後程式會自動繼續。".format(self.target)
        )

    def _on_back_in_room(self, cur: int) -> None:
        away = time.monotonic() - (self._away_since or time.monotonic())
        self._away_since = None
        self._reset_room_state()
        if cur == self.target:
            self.phase = GrabPhase.HOLDING
            self._got_at = time.monotonic()
            msg = f"✅ 已回到 No.{cur}（離開約 {away:.0f} 秒），就是目標房，繼續補注守房"
        else:
            self.phase = GrabPhase.GRABBING
            msg = f"✅ 已回到 No.{cur}（離開約 {away:.0f} 秒），繼續搶 No.{self.target}"
        logger.info(msg)
        self._tg(msg)

    def _reset_room_state(self) -> None:
        """剛換到新房／回到房內：遊戲端的未押注計數重來。"""
        self._label_miss_since = None
        self._last_bet_mono = time.monotonic()
        self._rounds_since_bet = 0
        self._last_cd_t = None
        self._last_cd_green = False
        self._lost_reads = 0

    # ---------------- 搶房／守房 ----------------
    def _confirm_got_target(self, win, timeout_s: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            cur = read_room_label(capture_client(win), self.cfg, win)
            if cur is not None:
                self._cur_room = cur
            if cur == self.target:
                return True
            if cur is not None:
                return False
            time.sleep(0.3)
        return False

    def _on_got_target(self) -> None:
        self.phase = GrabPhase.HOLDING
        self._got_at = time.monotonic()
        self._reset_room_state()
        mins = (self._got_at - self._started) / 60.0
        logger.info("搶房成功：已進入 No.%d（嘗試 %d 次、%.1f 分鐘）", self.target, self._attempts, mins)
        self._tg(
            f"✅ 已搶到 No.{self.target} 房間！（嘗試 {self._attempts} 次、花 {mins:.0f} 分鐘）\n"
            "程式會持續補注守住房間，請盡快關掉搶房程式、改開跟注程式。"
        )

    def _grab_attempt(self, win) -> None:
        """清單是即時更新的：開著清單、對著目標列的 GO 一直點（滿房時是灰的，點了沒反應）。

        GO 變亮（有空位）才按「確定」——確定鈕在清單外，滿房時亂按會把清單關掉。
        """
        cap = lambda: capture_client(win)
        if not open_room_list(win, self.cfg, cap):
            logger.warning("搶房：開桌號清單失敗，稍後再試")
            self._row_y = None
            time.sleep(1.0)
            return
        now = time.monotonic()
        if self._row_y is None or now - self._row_checked >= _ROW_RECHECK_S:
            self._row_y = find_target_row(win, self.cfg, cap, self.target)
            self._row_checked = time.monotonic()
            if self._row_y is None:
                logger.info("搶房：清單裡找不到 No.%d，下一輪再找", self.target)
                time.sleep(0.5)
                return
        frame = cap()
        lit = goto_is_lit(frame, self.cfg, win, self._row_y)
        click_goto_at(win, self.cfg, self._row_y)
        self._attempts += 1
        time.sleep(0.4)
        if not lit and list_is_open(cap(), self.cfg, win):
            # 灰色 GO、清單還開著 → 仍滿房，繼續點
            pc = time.perf_counter()
            if pc - self._last_log_mono >= 30.0:
                logger.info("搶房：No.%d 仍滿房，持續點 GO（已點 %d 次）", self.target, self._attempts)
                self._last_log_mono = pc
            time.sleep(_GO_CLICK_GAP_S)
            return
        # GO 是亮的（或點完清單被確認視窗擋住）→ 按確定
        logger.info("搶房：No.%d 的 GO 亮了（lit=%s），按確定進房", self.target, lit)
        confirm_switch(win, self.cfg)
        self._row_y = None
        deadline = time.monotonic() + self.cfg.room.switch_confirm_s + 2.0
        while time.monotonic() < deadline:
            cur = read_room_label(cap(), self.cfg, win)
            if cur == self.target:
                self._cur_room = cur
                self._on_got_target()
                return
            time.sleep(0.3)
        logger.info("搶房：按了確定但房號還不是 No.%d（可能被別人搶先），繼續搶", self.target)

    def _hold_verify(self) -> None:
        """守房：定時核對左上角房號；連續讀到別的房號才判定離開指定房 → 回到搶房。"""
        now = time.monotonic()
        if now - self._last_verify_mono < _HOLD_VERIFY_S:
            return
        self._last_verify_mono = now
        cur = self._cur_room
        if cur is None or cur == self.target:
            self._lost_reads = 0
            return
        self._lost_reads += 1
        if self._lost_reads < _LOST_CONFIRM_READS:
            return
        logger.warning("守房：目前在 No.%s，已不在 No.%d → 重新搶房", cur, self.target)
        self._tg(f"⚠ 已不在 No.{self.target}（目前在 No.{cur}），重新開始搶房")
        self.phase = GrabPhase.GRABBING
        self._reset_room_state()

    # ---------------- 主迴圈 ----------------
    def _check_window(self):
        win = self.eng._find_game()
        now = time.monotonic()
        if not win:
            if self._nowin_since is None:
                self._nowin_since = now
                logger.warning("找不到星城視窗「%s」", self.cfg.window.title_substring)
            if not self._nowin_notified and now - self._nowin_since >= _NOWIN_CONFIRM_S:
                self._nowin_notified = True
                self._tg("⚠ 搶房：找不到星城遊戲視窗（可能已關閉或當機），請手動開啟遊戲")
            self.eng._win = None
            return None
        if self._nowin_since is not None and self._nowin_notified:
            self._tg("✅ 搶房：已重新找到星城遊戲視窗，繼續執行")
        self._nowin_since = None
        self._nowin_notified = False
        if self.eng._win is None:
            try:
                focus_window(win.hwnd)
            except Exception:  # noqa: BLE001
                pass
        self.eng._win = win
        return win

    def run_once(self) -> None:
        win = self._check_window()
        if win is None:
            time.sleep(1.0)
            return
        cap = lambda: capture_client(win)
        now = time.monotonic()
        if self.phase != GrabPhase.AWAY and now - self._last_popup_check >= 2.0:
            self._last_popup_check = now
            if lobby_nav.dismiss_popup_if_any(win, self.cfg, cap):
                return
        frame = cap()
        in_room = self._update_presence(frame, win)
        self._heartbeat()
        if not in_room:
            time.sleep(1.0)  # 已離開房間：暫停，只等使用者自己回房
            return

        t, color = self._read_cd()
        self._track_round(t, color)
        if self._bet_due():
            self._keepalive_bet(win)
            return

        if self.phase == GrabPhase.GRABBING:
            if self._cur_room == self.target:
                close_room_list(win, self.cfg, cap)
                if self._confirm_got_target(win):
                    self._on_got_target()  # 一開始就在指定房（或剛搶到）
                return
            self._grab_attempt(win)
        else:
            self._hold_verify()
            time.sleep(1.0)  # 守房沒事做，放慢輪詢

    def run_loop(self) -> None:
        self._running = True
        self.eng._running = True
        logger.info("搶房模式啟動：目標 No.%d（%s）", self.target, "dry-run" if self.eng.dry_run else "LIVE")
        self._tg(f"🔵 搶房程式啟動：目標 No.{self.target}，搶到會通知")
        while self._running:
            if not self.eng._running:  # 安全停損（單局／累計上限）觸發
                logger.error("搶房：引擎安全停損（%s），停止", self.eng._stopped_reason)
                self._tg(f"⛔ 搶房程式已停止：{self.eng._stopped_reason}")
                break
            try:
                self.run_once()
            except CaptureUnavailable as exc:
                logger.warning("畫面暫時無法擷取（%s），1 秒後重試", exc)
                time.sleep(1.0)
            except KeyboardInterrupt:
                break
            except Exception:  # noqa: BLE001
                logger.exception("搶房迴圈錯誤")
                time.sleep(1.0)
        self._running = False

    def stop(self) -> None:
        self._running = False
