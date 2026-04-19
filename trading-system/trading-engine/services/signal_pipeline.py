"""
signal_pipeline.py — ML-native signal generator for the live/paper engine.

Mirrors run_backtest._compute_backtest_signal exactly (that is the source of truth).
Called by main.py on every MARKET_DATA event.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from services.event_bus import EventBus, EventType

logger = logging.getLogger(__name__)


def _first_present_frame(df_htf: dict, *keys: str, default=None):
    """Return the first non-None HTF frame without truth-testing pandas DataFrames."""
    if not isinstance(df_htf, dict):
        return default
    for key in keys:
        frame = df_htf.get(key)
        if frame is not None:
            return frame
    return default


class SignalPipeline:
    """
    Per-bar pipeline: ML inference → _compute_ml_signal → ensemble gate → publish.

    Signal logic mirrors run_backtest._compute_backtest_signal (source of truth).
    QualityScorer / EV gate runs after PM enrichment in main.py (needs rr_ratio).
    """

    def __init__(
        self,
        ml_models: dict,
        feature_engine,
        session_manager,
        news_service,
        settings,
        event_bus: EventBus,
    ):
        self._ml = ml_models
        self._fe = feature_engine
        self._session = session_manager
        self._news = news_service
        self._settings = settings
        self._bus = event_bus

        # Per-symbol OHLCV store: {symbol: {tf: df}}
        self._ohlcv: Dict[str, Dict[str, pd.DataFrame]] = {}
        self._bar_count = 0

    def update_ohlcv(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        self._ohlcv.setdefault(symbol, {})[timeframe] = df

    def get_ohlcv(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        return self._ohlcv.get(symbol, {}).get(timeframe)

    async def process_bar(
        self, symbol: str, df: pd.DataFrame, df_htf: dict
    ) -> List[dict]:
        """
        Returns list of approved signals (usually 0 or 1).
        df_htf: {timeframe: DataFrame} for HTF context.
        """
        if df is None or len(df) < 20:
            return []

        self._bar_count += 1

        # Step 1: ML inference
        if self._settings.ML_ENABLED:
            ml_preds = self._run_ml_inference(symbol, df, df_htf)
        else:
            ml_preds = self._neutral_ml_preds(symbol, df)

        # Augment with news service data
        if self._news is not None:
            ml_preds["news_in_30min"] = self._news.is_blocked(symbol, block_minutes=30)
            ml_preds["news_in_15min"] = self._news.is_blocked(symbol, block_minutes=15)
            ml_preds["news_in_90min"] = self._news.is_blocked(symbol, block_minutes=90)
            recent = self._news.get_active_events(window_minutes=15)
            ml_preds["active_news_events"] = recent
        else:
            ml_preds.update({
                "news_in_30min": False,
                "news_in_15min": False,
                "news_in_90min": False,
                "active_news_events": [],
            })

        # Step 2: Dead zone 12:00–13:00 UTC (mirrors backtest _backtest_trader)
        now = datetime.now(timezone.utc)
        if now.hour == 12:
            return []

        # Step 3: Unified ML signal (mirrors _compute_backtest_signal exactly)
        # QualityScorer/EV gate runs in main.py after PM enrichment (needs rr_ratio).
        raw_signal = self._compute_ml_signal(symbol, df, ml_preds)
        if raw_signal is None:
            return []

        # Step 4: Confidence gate (≥ 0.55) — same as backtest MIN_CONFIDENCE
        if float(raw_signal.get("confidence", 0)) < 0.55:
            return []

        # Enrich metadata with sentiment and macro context
        meta = raw_signal.setdefault("signal_metadata", {})
        meta["sentiment_label"] = ml_preds.get("sentiment_label", "neutral")
        meta["sentiment_backend"] = ml_preds.get("sentiment_backend", "neutral")
        meta["sentiment_confidence"] = ml_preds.get("sentiment_confidence", 0.0)
        from services.feature_engine import MACRO_FEATURES
        meta["macro"] = {k: float(ml_preds.get(k, 0.0)) for k in MACRO_FEATURES}

        logger.info(
            "Signal APPROVED ml_trader %s %s — conf=%.3f regime=%s p_bull=%.3f p_bear=%.3f",
            symbol, raw_signal.get("side"),
            raw_signal.get("confidence", 0),
            ml_preds.get("regime", "?"),
            ml_preds.get("p_bull", 0.5),
            ml_preds.get("p_bear", 0.5),
        )

        self._publish_signal(raw_signal)
        return [raw_signal]

    def _run_ml_inference(
        self, symbol: str, df: pd.DataFrame, df_htf: dict
    ) -> dict:
        """
        Returns ml_predictions dict with all model outputs.
        df_htf: full {tf: DataFrame} dict — both models receive all TFs and extract
                what they need internally. Keys expected: "5M", "1H", "4H", "1D".
        """
        preds: dict = {}
        htf = df_htf if isinstance(df_htf, dict) else {}

        # Hot-reload check
        for name, model in self._ml.items():
            if hasattr(model, "reload_if_updated"):
                model.reload_if_updated()

        # GRU-LSTM: 15M base sequence + MTF cross-TF features from 5M/1H/4H/1D
        gru = self._ml.get("gru_lstm")
        if gru:
            try:
                r = gru.predict(df, symbol=symbol, df_htf=htf)
                preds.update(r)
                preds["p_bull_gru"] = r.get("p_bull")
                preds["p_bear_gru"] = r.get("p_bear")
            except RuntimeError as exc:
                logger.error("GRU-LSTM not trained — ML signals disabled. %s", exc)
                raise
            except Exception as exc:
                logger.error("GRU-LSTM inference error: %s", exc)
                raise

        # RegimeClassifier: base structural features + MTF adx/ema_stack/atr/bb from 5M/15M/1H/4H/1D
        regime_model = self._ml.get("regime")
        if regime_model:
            try:
                r = regime_model.predict(df, symbol=symbol, df_htf=htf)
                preds["regime"] = r.get("regime")
                preds["regime_id"] = r.get("regime_id")
                preds["regime_proba"] = r.get("proba")
            except RuntimeError as exc:
                logger.error("RegimeClassifier not trained — ML signals disabled. %s", exc)
                raise
            except Exception as exc:
                logger.error("Regime inference error: %s", exc)
                raise

        # Sentiment
        sent_model = self._ml.get("sentiment")
        if sent_model:
            try:
                headline = f"{symbol} market update"
                r = sent_model.analyze(headline, instrument=symbol)
                preds["sentiment_score"] = r.get("score")
                preds["sentiment_label"] = r.get("label")
                preds["sentiment_backend"] = r.get("backend", "neutral")
                preds["sentiment_confidence"] = r.get("confidence", 0.0)
            except Exception as exc:
                logger.error("Sentiment inference error: %s", exc)
                raise

        # Spread
        preds["spread_pips"] = 1.0  # default; updated by DataFetcher in main.py

        # Macro snapshot (indices + fundamentals) for model/strategy context
        ts = df.index[-1] if len(df.index) else None
        if ts is not None:
            preds.update(self._fe.get_macro_snapshot(symbol, ts))

        # ATR history ratios (for RL state)
        from indicators.market_structure import compute_atr
        atr = compute_atr(df, 14)
        from services.feature_engine import _ATR_LAGS
        atr_current = float(atr.iloc[-1]) if len(atr.dropna()) > 0 else 0.001
        for lag in _ATR_LAGS:
            if len(atr) > lag:
                past = float(atr.iloc[-lag - 1]) if not pd.isna(atr.iloc[-lag - 1]) else atr_current
                preds[f"atr_lag_{lag}"] = float(np.clip(atr_current / (past + 1e-9), 0, 5))
            else:
                preds[f"atr_lag_{lag}"] = 1.0

        # QualityScorer (EV) runs post-signal in main.py after PM enrichment.
        # It requires actual rr_ratio from the enriched signal, so it cannot run here.

        return preds

    def _neutral_ml_preds(self, symbol: str, df: pd.DataFrame) -> dict:
        """Rule-only predictions when ML_ENABLED=false (no ML scoring applied)."""
        return {
            "spread_pips": 1.0,
            **{f"atr_lag_{l}": 1.0 for l in [1, 4, 8, 24, 48, 96, 168, 336]},
        }

    def _compute_ml_signal(
        self, symbol: str, df: pd.DataFrame, ml_preds: dict
    ) -> dict | None:
        """
        Mirrors run_backtest._compute_backtest_signal exactly (source of truth).

        Gates (in order):
          1. ATR sanity
          2. GRU variance ≤ MAX_UNCERTAINTY (default 2.0 bootstrap; set 0.80 post-training)
          3. GRU direction ≥ ML_DIRECTION_THRESHOLD (default 0.52; EURUSD=0.85)
        EV gate runs in main.py after PM enrichment (needs actual rr_ratio).
        """
        if df is None or len(df) == 0:
            return None

        bar = df.iloc[-1]
        close = float(bar["close"])
        atr = float(bar.get("atr_14", close * 0.001))
        if atr < 1e-9:
            return None

        if not ml_preds:
            return None

        # GRU uncertainty gate
        _uncertainty = float(ml_preds.get("expected_variance", 0.0))
        if _uncertainty > float(os.getenv("MAX_UNCERTAINTY", "2.0")):
            return None

        # Direction gate
        p_bull = float(ml_preds.get("p_bull", 0.5))
        p_bear = float(ml_preds.get("p_bear", 0.5))
        _dir_thresh = float(os.getenv("ML_DIRECTION_THRESHOLD", "0.52"))
        if symbol == "EURUSD":
            _dir_thresh = float(os.getenv("EURUSD_DIRECTION_THRESHOLD", "0.85"))

        if p_bull >= p_bear and p_bull >= _dir_thresh:
            side = "buy"
            conf = p_bull
        elif p_bear > p_bull and p_bear >= _dir_thresh:
            side = "sell"
            conf = p_bear
        else:
            return None

        # Regime context (metadata only — regime is encoded in SEQUENCE_FEATURES)
        _regime = str(ml_preds.get("regime", "BIAS_NEUTRAL"))
        _regime_dur = float(bar.get("regime_duration", 0.5))

        # Regime confidence multiplier (light bias — GRU already encodes regime in features)
        _regime_mult_map = {
            # HTF bias classes
            "BIAS_UP":      1.10,
            "BIAS_DOWN":    1.10,
            "BIAS_NEUTRAL": 1.00,
            # LTF behaviour classes
            "TRENDING":     1.10,
            "RANGING":      0.95,
            "CONSOLIDATING": 1.05,
            "VOLATILE":     0.85,
            # Legacy compat
            "TRENDING_UP": 1.10, "TRENDING_DOWN": 1.10, "CONSOLIDATION": 1.05,
        }
        _regime_weight = _regime_mult_map.get(_regime, 1.0)

        # Weak regime-age prior (mirrors backtest exactly)
        _age_weight = 1.0 + 0.08 * (1.0 - min(_regime_dur, 1.0))
        conf = float(conf * _age_weight)

        # ATR-based entry / SL / TP (mirrors backtest exactly)
        _sl_mult = float(os.getenv("SL_ATR_MULT", "1.5"))
        _rr_default = float(os.getenv("RR_DEFAULT", "2.0"))
        sl_dist = atr * _sl_mult
        if side == "buy":
            stop_loss = close - sl_dist
            take_profit = close + sl_dist * _rr_default
        else:
            stop_loss = close + sl_dist
            take_profit = close - sl_dist * _rr_default

        return {
            "side": side,
            "entry": close,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "confidence": round(conf, 3),
            "trader_id": "ml_trader",
            "symbol": symbol,
            "signal_metadata": {
                "regime": _regime,
                "expected_variance": _uncertainty,
                "p_bull": p_bull,
                "p_bear": p_bear,
                "atr": atr,
                "atr_at_entry": atr,
                "session_weight": 1.0,
                "regime_weight": round(_regime_weight, 3),
                "age_weight": round(_age_weight, 3),
                "strategy": "ml_native",
            },
        }

    def _publish_signal(self, sig: dict) -> None:
        """Publish SIGNAL_GENERATED event (Contract 1)."""
        meta = sig.get("signal_metadata", {}) or {}
        event = {
            "trader_id": sig.get("trader_id", ""),
            "symbol": sig.get("symbol", ""),
            "side": sig.get("side", "buy"),
            "confidence": float(sig.get("confidence", 0.6)),
            "stop_loss": float(sig.get("stop_loss", 0)),
            "take_profit": float(sig.get("take_profit", 0)),
            "correlation_id": str(sig.get("correlation_id", "")),
            "signal_metadata": {
                "strategy": str(meta.get("strategy", "")),
                "session": str(meta.get("session", "")),
                "rl_action": int(meta.get("rl_action", 0)),
                "quality_score": float(meta.get("quality_score", 0.5)),
                "p_bull": float(meta.get("p_bull", 0.5)),
                "p_bear": float(meta.get("p_bear", 0.5)),
                "regime": str(meta.get("regime", "")),
                "sentiment_score": float(meta.get("sentiment_score", 0.0)),
                "rr_ratio": float(sig.get("rr_ratio", 1.5)),
            },
        }
        self._bus.publish(EventType.SIGNAL_GENERATED, event)
