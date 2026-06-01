/**
 * 星城跟注 — 餘額彙總並發 Telegram（放在 Google 試算表裡執行，跑在 Google 雲端）
 *
 * 分工：
 *   - 每台主機只把「自己帳號的最新餘額」寫到「餘額」分頁（一帳號一列、每小時覆蓋）。
 *   - 這支腳本每小時讀總額、算與上次差額、寫一筆到「歷史」分頁（自動修剪），再發 TG。
 *
 * 安裝步驟（只做一次）：
 *   1. 開你的試算表 → 上方「擴充功能」→「Apps Script」，把本檔內容整段貼進去、存檔。
 *   2. 左側「專案設定（齒輪）」→ 最下面「指令碼屬性」→ 新增兩個：
 *        TG_TOKEN = 你的 Telegram bot token（找 @BotFather 取得）
 *        TG_CHAT  = 要收訊息的 chat id
 *   3. 上方函式選「createHourlyTrigger」按「執行」一次（會要你授權，按允許）。
 *      → 之後每小時自動跑 hourlyReport。
 *   4. 想立刻測：函式選「hourlyReport」按「執行」，看 TG 有沒有收到。
 *
 * 備註：Apps Script 的「每小時」觸發器不保證剛好在整點，但主機是在整點上傳，
 *       腳本稍後在該小時內執行時資料已是最新，影響不大。
 */

var LATEST_SHEET = '餘額';   // 主機上傳的分頁（欄位：帳號 | 餘額 | 更新時間）
var HISTORY_SHEET = '歷史';  // 本腳本寫入的歷史快照
var HISTORY_KEEP = 800;      // 歷史最多保留筆數（超過自動刪最舊的）

function _props() {
  return PropertiesService.getScriptProperties();
}

function _toInt(v) {
  var n = parseInt(String(v).replace(/[^0-9-]/g, ''), 10);
  return isNaN(n) ? 0 : n;
}

function sendTelegram(text) {
  var p = _props();
  var token = p.getProperty('TG_TOKEN');
  var chat = p.getProperty('TG_CHAT');
  if (!token || !chat) {
    Logger.log('尚未設定 TG_TOKEN / TG_CHAT，略過發送');
    return;
  }
  var url = 'https://api.telegram.org/bot' + token + '/sendMessage';
  UrlFetchApp.fetch(url, {
    method: 'post',
    payload: { chat_id: chat, text: text },
    muteHttpExceptions: true,
  });
}

function hourlyReport() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var latest = ss.getSheetByName(LATEST_SHEET);
  if (!latest) {
    Logger.log('找不到「' + LATEST_SHEET + '」分頁');
    return;
  }

  var values = latest.getDataRange().getValues(); // 含表頭
  var total = 0;
  var lines = [];
  for (var i = 1; i < values.length; i++) {
    var name = values[i][0];
    if (!name) continue;
    var bal = _toInt(values[i][1]);
    total += bal;
    lines.push('・' + name + '：' + bal.toLocaleString());
  }

  var p = _props();
  var lastStr = p.getProperty('lastTotal');
  var diffText, diffVal;
  if (lastStr !== null && lastStr !== '') {
    diffVal = total - parseInt(lastStr, 10);
    diffText = (diffVal >= 0 ? '＋' : '') + diffVal.toLocaleString() +
      (diffVal > 0 ? '（賺）' : diffVal < 0 ? '（虧）' : '（持平）');
  } else {
    diffVal = '';
    diffText = '（首次，建立基準）';
  }

  var now = Utilities.formatDate(new Date(), Session.getScriptTimeZone(), 'yyyy-MM-dd HH:mm:ss');

  // 寫歷史
  var hist = ss.getSheetByName(HISTORY_SHEET);
  if (!hist) {
    hist = ss.insertSheet(HISTORY_SHEET);
    hist.appendRow(['時間', '總額', '與上次差額', '帳號數']);
  }
  hist.appendRow([now, total, diffVal, lines.length]);

  // 修剪歷史：超過 HISTORY_KEEP 筆就刪最舊的（保留表頭在第 1 列）
  var n = hist.getLastRow();
  if (n > HISTORY_KEEP + 1) {
    hist.deleteRows(2, n - (HISTORY_KEEP + 1));
  }

  // 發 TG
  var msg = '💰 餘額彙總　' + now +
    '\n帳號數：' + lines.length +
    '\n總額：' + total.toLocaleString() +
    '\n與上次：' + diffText +
    '\n\n' + lines.join('\n');
  sendTelegram(msg);

  p.setProperty('lastTotal', String(total));
}

/** 執行一次即可：建立「每小時」觸發器。 */
function createHourlyTrigger() {
  // 先移除舊的同名觸發器，避免重複
  var triggers = ScriptApp.getProjectTriggers();
  for (var i = 0; i < triggers.length; i++) {
    if (triggers[i].getHandlerFunction() === 'hourlyReport') {
      ScriptApp.deleteTrigger(triggers[i]);
    }
  }
  ScriptApp.newTrigger('hourlyReport').timeBased().everyHours(1).create();
  Logger.log('已建立每小時觸發器');
}
