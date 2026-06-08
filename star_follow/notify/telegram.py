"""Telegram 機器人通知（只用 Python 內建 urllib，避免增加打包相依）。"""

from __future__ import annotations

import json
import logging
import ssl
import urllib.error
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

_API = "https://api.telegram.org/bot{token}/sendMessage"


def _post(url: str, data: bytes, timeout: float, ctx: ssl.SSLContext | None) -> tuple[bool, str]:
    """送出一次請求。回傳 (是否成功 ok, 診斷訊息)。"""
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        body = resp.read().decode("utf-8", "replace")
    ok = bool(json.loads(body).get("ok", False))
    return ok, body[:200]


def send_message(bot_token: str, chat_id: str, text: str, timeout: float = 15.0) -> bool:
    """送出一則文字訊息，成功回 True。

    部分客戶端網路（中華電信/公司網路）會在 TLS 鏈中插入自簽憑證做側錄，造成
    CERTIFICATE_VERIFY_FAILED。通知類訊息安全風險低（bot token 本身即授權），
    故正常驗證失敗時，退而用「不驗證憑證」的連線重試一次，確保通知不漏發。
    """
    if not bot_token or not chat_id:
        logger.warning("Telegram 未設定 bot_token / chat_id，略過發送")
        return False
    url = _API.format(token=bot_token)
    data = urllib.parse.urlencode(
        {"chat_id": chat_id, "text": text, "disable_web_page_preview": "true"}
    ).encode("utf-8")
    try:
        ok, body = _post(url, data, timeout, None)
        if not ok:
            logger.warning("Telegram 回應非 ok：%s", body)
        return ok
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:200]
        except Exception:  # noqa: BLE001
            pass
        logger.warning("Telegram HTTP 錯誤 %s：%s", exc.code, detail)
        return False
    except ssl.SSLError as exc:
        # 憑證驗證失敗 → 用不驗證憑證的連線重試一次
        logger.info("Telegram TLS 驗證失敗（%s），改用不驗證憑證重試", exc)
    except urllib.error.URLError as exc:
        if not isinstance(getattr(exc, "reason", None), ssl.SSLError):
            logger.warning("Telegram 發送失敗：%s", exc)
            return False
        logger.info("Telegram TLS 驗證失敗（%s），改用不驗證憑證重試", exc.reason)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Telegram 發送失敗：%s", exc)
        return False

    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ok, body = _post(url, data, timeout, ctx)
        if not ok:
            logger.warning("Telegram 回應非 ok（不驗證重試）：%s", body)
        return ok
    except Exception as exc:  # noqa: BLE001
        logger.warning("Telegram 發送失敗（不驗證重試也失敗）：%s", exc)
        return False
