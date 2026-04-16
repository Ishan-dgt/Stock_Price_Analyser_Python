"""
Alert Manager
Handles price threshold alerts, MA crossover alerts,
and delivery via email (SMTP) or Telegram Bot API.
"""

import smtplib
import logging
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Dict, List, Any

try:
    import requests as _requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Checks live conditions against thresholds and emits alerts.

    Alert dict schema:
      {
        "title": str,
        "message": str,
        "severity": "positive" | "negative" | "neutral",
        "time": str (ISO timestamp),
      }
    """

    def __init__(
        self,
        email_config: Optional[Dict] = None,
        telegram_config: Optional[Dict] = None,
    ):
        self.email_config = email_config  # keys: to, host, user, password
        self.telegram_config = telegram_config  # keys: token, chat_id
        self._sent: List[str] = []  # deduplicate recent alerts

    # ─────────────────────────────────────────────────────────────────────────
    # Condition Checks
    # ─────────────────────────────────────────────────────────────────────────

    def check_price_alerts(
        self,
        ticker: str,
        current_price: float,
        high_threshold: float = 0.0,
        low_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Trigger alerts if price crosses configured thresholds.
        Thresholds of 0.0 are treated as "not configured".
        """
        alerts = []
        now = datetime.now().strftime("%H:%M:%S")

        if high_threshold > 0 and current_price >= high_threshold:
            alert = {
                "title": f"{ticker} High Alert 🚀",
                "message": f"Price ${current_price:.2f} reached/exceeded threshold ${high_threshold:.2f}",
                "severity": "positive",
                "time": now,
            }
            key = f"high_{ticker}_{high_threshold}"
            if key not in self._sent:
                alerts.append(alert)
                self._sent.append(key)
                self._dispatch(alert)

        if low_threshold > 0 and current_price <= low_threshold:
            alert = {
                "title": f"{ticker} Low Alert ⚠️",
                "message": f"Price ${current_price:.2f} dropped to/below threshold ${low_threshold:.2f}",
                "severity": "negative",
                "time": now,
            }
            key = f"low_{ticker}_{low_threshold}"
            if key not in self._sent:
                alerts.append(alert)
                self._sent.append(key)
                self._dispatch(alert)

        return alerts

    def check_ma_crossover_alerts(
        self,
        ticker: str,
        crossovers: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Issue alerts for any crossover events detected on the *latest* date.
        """
        if not crossovers:
            return []

        today = datetime.now().strftime("%Y-%m-%d")
        alerts = []

        for c in crossovers:
            if c["date"] != today:
                continue
            is_golden = c["type"] == "GOLDEN_CROSS"
            alert = {
                "title": f"{ticker} {'Golden Cross 🌟' if is_golden else 'Death Cross 💀'}",
                "message": c["description"] + f" at ${c['price']:.2f}",
                "severity": "positive" if is_golden else "negative",
                "time": today,
            }
            key = f"cross_{ticker}_{c['type']}_{today}"
            if key not in self._sent:
                alerts.append(alert)
                self._sent.append(key)
                self._dispatch(alert)

        return alerts

    # ─────────────────────────────────────────────────────────────────────────
    # Delivery
    # ─────────────────────────────────────────────────────────────────────────

    def _dispatch(self, alert: Dict[str, Any]) -> None:
        """Send alert via all configured channels."""
        body = f"{alert['title']}\n\n{alert['message']}\nTime: {alert['time']}"
        if self.email_config and self.email_config.get("to"):
            self._send_email(alert["title"], body)
        if self.telegram_config and self.telegram_config.get("token"):
            self._send_telegram(body)

    def _send_email(self, subject: str, body: str) -> bool:
        """Send an alert email via SMTP."""
        cfg = self.email_config
        try:
            msg = MIMEMultipart()
            msg["From"] = cfg.get("user", "")
            msg["To"] = cfg["to"]
            msg["Subject"] = f"[Stock Alert] {subject}"
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(cfg.get("host", "smtp.gmail.com"), 587) as server:
                server.ehlo()
                server.starttls()
                server.login(cfg["user"], cfg["password"])
                server.sendmail(cfg["user"], cfg["to"], msg.as_string())
            logger.info(f"Email alert sent: {subject}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False

    def _send_telegram(self, text: str) -> bool:
        """Send an alert message via Telegram Bot API."""
        if not HAS_REQUESTS:
            logger.warning("requests library not available. Cannot send Telegram alert.")
            return False
        cfg = self.telegram_config
        try:
            url = f"https://api.telegram.org/bot{cfg['token']}/sendMessage"
            payload = {"chat_id": cfg["chat_id"], "text": text, "parse_mode": "Markdown"}
            response = _requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            logger.info("Telegram alert sent.")
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            return False

    def send_custom_alert(
        self,
        title: str,
        message: str,
        severity: str = "neutral",
    ) -> Dict[str, Any]:
        """Manually fire a custom alert through all delivery channels."""
        alert = {
            "title": title,
            "message": message,
            "severity": severity,
            "time": datetime.now().strftime("%H:%M:%S"),
        }
        self._dispatch(alert)
        return alert
