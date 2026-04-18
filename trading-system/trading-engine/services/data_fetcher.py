"""
data_fetcher.py — Capital.com REST primary, yfinance fallback.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# yfinance symbol mapping
_YF_MAP = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "XAUUSD": "GC=F",
    "EURJPY": "EURJPY=X",
    "AUDJPY": "AUDJPY=X",
    "GBPJPY": "GBPJPY=X",
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
}

_TF_MAP = {
    "1M": "1m", "5M": "5m", "15M": "15m",
    "1H": "1h", "4H": "4h", "1D": "1d",
}
_CAPITAL_TF = {
    "1M": "MINUTE", "5M": "MINUTE_5", "15M": "MINUTE_15",
    "1H": "HOUR", "4H": "HOUR_4", "1D": "DAY",
}


class DataFetcher:
    """Fetches OHLCV data from Capital.com (primary) with yfinance fallback."""

    def __init__(self, capital_session=None, settings=None):
        self._session = capital_session
        self._settings = settings

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "15M",
        bars: int = 300,
    ) -> Optional[pd.DataFrame]:
        """
        Returns DataFrame with columns: open, high, low, close, volume
        and a DatetimeIndex (UTC).
        """
        df = None

        if self._session is not None:
            df = self._fetch_capital(symbol, timeframe, bars)

        if df is None or len(df) < 10:
            df = self._fetch_yfinance(symbol, timeframe, bars)

        return df

    def _fetch_capital(
        self, symbol: str, timeframe: str, bars: int
    ) -> Optional[pd.DataFrame]:
        try:
            tf = _CAPITAL_TF.get(timeframe, "MINUTE_15")
            path = f"/api/v1/prices/{symbol}"
            params = {"resolution": tf, "max": bars}
            data = self._session.get(path, params)
            if data is None:
                return None

            prices = data.get("prices", [])
            if not prices:
                return None

            rows = []
            for p in prices:
                ts = p.get("snapshotTime") or p.get("snapshotTimeUTC")
                if ts is None:
                    continue
                try:
                    dt = pd.to_datetime(ts, utc=True)
                except Exception:
                    continue
                bid = p.get("openPrice", {})
                ask = p.get("closePrice", {})
                rows.append({
                    "timestamp": dt,
                    "open": (float(bid.get("bid", 0)) + float(bid.get("ask", 0))) / 2,
                    "high": max(float(p.get("highPrice", {}).get("bid", 0)), float(p.get("highPrice", {}).get("ask", 0))),
                    "low": min(float(p.get("lowPrice", {}).get("bid", 0)), float(p.get("lowPrice", {}).get("ask", 0))),
                    "close": (float(ask.get("bid", 0)) + float(ask.get("ask", 0))) / 2,
                    "volume": float(p.get("lastTradedVolume", 0)),
                })

            if not rows:
                return None

            df = pd.DataFrame(rows).set_index("timestamp").sort_index()
            df = df[df["close"] > 0]
            return df
        except Exception as exc:
            logger.debug("Capital.com fetch %s/%s failed: %s", symbol, timeframe, exc)
            return None

    def _fetch_yfinance(
        self, symbol: str, timeframe: str, bars: int
    ) -> Optional[pd.DataFrame]:
        try:
            import yfinance as yf
            yf_sym = _YF_MAP.get(symbol, symbol)
            interval = _TF_MAP.get(timeframe, "15m")

            # yfinance bar limits per interval
            period_map = {"1m": "7d", "5m": "60d", "15m": "60d", "1h": "730d", "4h": "730d", "1d": "max"}
            period = period_map.get(interval, "60d")

            ticker = yf.Ticker(yf_sym)
            hist = ticker.history(period=period, interval=interval, auto_adjust=True)

            if hist is None or len(hist) == 0:
                return None

            df = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.columns = ["open", "high", "low", "close", "volume"]
            df.index = pd.to_datetime(df.index, utc=True)
            df = df.sort_index().tail(bars)
            df = df[df["close"] > 0]
            return df
        except Exception as exc:
            logger.debug("yfinance fetch %s/%s failed: %s", symbol, timeframe, exc)
            return None

    def get_spread_pips(self, symbol: str) -> float:
        """Returns current bid-ask spread in pips."""
        if self._session is None:
            raise RuntimeError(f"DataFetcher: no active session — cannot fetch spread for {symbol}")
        data = self._session.get(f"/api/v1/markets/{symbol}")
        if data is None:
            raise RuntimeError(f"DataFetcher: empty response fetching spread for {symbol}")
        instrument = data.get("instrument", {})
        bid = float(instrument.get("bid", 0))
        offer = float(instrument.get("offer", 0))
        spread = offer - bid
        if "JPY" in symbol or symbol == "XAUUSD":
            return round(spread * 100, 2)
        return round(spread * 10000, 2)
