"""Telegram 機器人通知（只用 Python 內建 urllib，避免增加打包相依）。"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

_API = "https://api.telegram.org/bot{token}/sendMessage"


def send_message(bot_token: str, chat_id: str, text: str, timeout: float = 15.0) -> bool:
    """送出一則文字訊息，成功回 True。"""
    if not bot_token or not chat_id:
        logger.warning("Telegram 未設定 bot_token / chat_id，略過發送")
        return False
    url = _API.format(token=bot_token)
    data = urllib.parse.urlencode(
        {"chat_id": chat_id, "text": text, "disable_web_page_preview": "true"}
    ).encode("utf-8")
    try:
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
        ok = json.loads(body).get("ok", False)
        if not ok:
            logger.warning("Telegram 回應非 ok：%s", body[:200])
        return bool(ok)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:200]
        except Exception:  # noqa: BLE001
            pass
        logger.warning("Telegram HTTP 錯誤 %s：%s", exc.code, detail)
        return False
    except Exception as exc:  # noqa: BLE001
        logger.warning("Telegram 發送失敗：%s", exc)
        return False
