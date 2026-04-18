"""
news_service.py — Unified news feed for all traders.

Sources: news_monitor (Redis pub/sub) + ForexFactory calendar polling.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 300  # 5 minutes


class NewsService:
    """Unified news feed with ForexFactory polling and Redis feed integration."""

    def __init__(self, redis_client=None, news_api_key: str = ""):
        self._redis = redis_client
        self._api_key = news_api_key
        self._events: List[dict] = []   # list of {name, time, currencies, impact, actual, forecast, previous}
        self._lock = threading.Lock()
        self._last_poll = 0.0
        self._poll_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start background polling thread."""
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

    def _poll_loop(self) -> None:
        while True:
            try:
                self._fetch_calendar()
            except Exception as exc:
                logger.debug("NewsService poll error: %s", exc)
            time.sleep(_POLL_INTERVAL)

    def _fetch_calendar(self) -> None:
        """Fetch upcoming high-impact events from ForexFactory (lightweight scrape)."""
        try:
            import requests
            url = "https://www.forexfactory.com/calendar/day"
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                return
            events = self._parse_ff_response(resp.text)
            with self._lock:
                self._events = events
        except Exception as exc:
            logger.debug("ForexFactory fetch failed: %s", exc)

    @staticmethod
    def _parse_ff_response(html: str) -> List[dict]:
        """Minimal HTML parser for ForexFactory calendar rows."""
        import re
        events = []
        pattern = re.compile(
            r'<td class="calendar__impact".*?impact-icon-red.*?</tr>',
            re.DOTALL,
        )
        for match in pattern.finditer(html):
            row = match.group(0)
            try:
                time_match = re.search(r'data-hm="(\d+:\d+[ap]m)"', row)
                name_match = re.search(r'class="calendar__event-title">(.*?)</a>', row)
                curr_match = re.search(r'class="calendar__currency">(.*?)</td>', row)
                if name_match and curr_match:
                    events.append({
                        "name": name_match.group(1).strip(),
                        "currencies": [curr_match.group(1).strip()],
                        "impact": "high",
                        "time": datetime.now(timezone.utc),
                    })
            except Exception:
                continue
        return events

    def ingest_event(self, event: dict) -> None:
        """Called by news_monitor when a Redis event arrives."""
        with self._lock:
            self._events.append(event)
            # Keep only last 100
            if len(self._events) > 100:
                self._events = self._events[-100:]

    def get_upcoming_events(
        self,
        within_minutes: int = 30,
        currencies: Optional[List[str]] = None,
    ) -> List[dict]:
        """Returns high-impact events within the next `within_minutes`."""
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(minutes=within_minutes)
        with self._lock:
            events = [
                e for e in self._events
                if now <= self._to_dt(e.get("time")) <= cutoff
                and self._matches_currencies(e, currencies)
            ]
        return events

    def get_recent_events(
        self,
        within_minutes: int = 15,
        currencies: Optional[List[str]] = None,
    ) -> List[dict]:
        """Returns recently released high-impact events."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=within_minutes)
        with self._lock:
            events = [
                e for e in self._events
                if cutoff <= self._to_dt(e.get("time")) <= now
                and self._matches_currencies(e, currencies)
            ]
        return events

    def get_active_events(self, window_minutes: int = 15) -> List[dict]:
        """Convenience: events within the last window_minutes."""
        return self.get_recent_events(within_minutes=window_minutes)

    def get_pre_news_block(
        self,
        symbol: str,
        df: pd.DataFrame,
        news_time: datetime,
    ) -> Optional[dict]:
        """Compute pre-news consolidation zone for a symbol."""
        if df is None or len(df) < 5:
            return None
        # Last 30 bars before news_time
        if hasattr(df.index, "tz"):
            cutoff = news_time.astimezone(df.index.tz) if df.index.tz else news_time.replace(tzinfo=None)
        else:
            cutoff = news_time.replace(tzinfo=None)

        window = df[df.index <= cutoff].tail(30)
        if len(window) < 5:
            window = df.tail(30)

        from indicators.market_structure import compute_atr
        atr = compute_atr(window, 14)
        atr_val = float(atr.iloc[-1]) if len(atr.dropna()) > 0 else 0.001

        high = float(window["high"].max())
        low = float(window["low"].min())
        news_range = high - low

        if news_range > atr_val * 2.0:
            return None  # Not a consolidation

        return {
            "symbol": symbol,
            "news_time": news_time.isoformat(),
            "high": high,
            "low": low,
            "mid": (high + low) / 2,
            "range": news_range,
        }

    def is_blocked(self, symbol: str, block_minutes: int = 30) -> bool:
        """True if high-impact news for symbol's currency within block_minutes."""
        currencies = self._symbol_to_currencies(symbol)
        upcoming = self.get_upcoming_events(within_minutes=block_minutes, currencies=currencies)
        recent = self.get_recent_events(within_minutes=block_minutes // 2, currencies=currencies)
        return len(upcoming) > 0 or len(recent) > 0

    @staticmethod
    def _to_dt(t) -> datetime:
        if isinstance(t, datetime):
            return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc)

    @staticmethod
    def _matches_currencies(event: dict, currencies: Optional[List[str]]) -> bool:
        if currencies is None:
            return True
        ev_curr = event.get("currencies", [])
        return any(c in ev_curr for c in currencies)

    @staticmethod
    def _symbol_to_currencies(symbol: str) -> List[str]:
        mapping = {
            "EURUSD": ["EUR", "USD"],
            "GBPUSD": ["GBP", "USD"],
            "USDJPY": ["USD", "JPY"],
            "AUDUSD": ["AUD", "USD"],
            "USDCAD": ["USD", "CAD"],
            "XAUUSD": ["USD", "XAU"],
            "EURJPY": ["EUR", "JPY"],
            "AUDJPY": ["AUD", "JPY"],
            "GBPJPY": ["GBP", "JPY"],
        }
        return mapping.get(symbol, [symbol[:3], symbol[3:]])
