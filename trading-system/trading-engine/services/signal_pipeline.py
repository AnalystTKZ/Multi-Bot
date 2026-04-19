"""
signal_pipeline.py — Replaces signal_engine.py + strategy_manager.py.

Orchestrates: ML inference → strategy scan → RL selection → execution gate.
Called by main.py on every MARKET_DATA event.

v2 additions:
  - CandidateLogger injected into all traders (logs pre-gate candidates)
  - EVFilter injected into all traders (EV gate before execution)
  - EVFilter.refresh() called every 100 bars to update historical averages
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from services.event_bus import EventBus, EventType
from services.candidate_logger import CandidateLogger
from services.quant_analytics import EVFilter

logger = logging.getLogger(__name__)

_EV_REFRESH_INTERVAL = 100  # refresh EVFilter every N bars


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
    Full pipeline per bar:
      1. ML inference (if ML_ENABLED)
      2. Run all session-appropriate traders
      3. Collect valid rule-based signals
      4. RL action selection (if ML_ENABLED)
      5. Confidence gate
      6. Return approved signals
    """

    def __init__(
        self,
        traders: list,
        ml_models: dict,
        feature_engine,
        session_manager,
        news_service,
        settings,
        event_bus: EventBus,
    ):
        self._traders = traders
        self._ml = ml_models
        self._fe = feature_engine
        self._session = session_manager
        self._news = news_service
        self._settings = settings
        self._bus = event_bus

        # Per-symbol OHLCV store: {symbol: {tf: df}}
        self._ohlcv: Dict[str, Dict[str, pd.DataFrame]] = {}

        # Quant analytics layer
        self._candidate_logger = CandidateLogger()
        self._ev_filter = EVFilter()
        self._ev_filter.refresh()   # load historical averages at startup
        self._bar_count = 0         # for periodic EV refresh

        # Inject analytics into all traders
        for t in self._traders:
            t._candidate_logger = self._candidate_logger
            t._ev_filter = self._ev_filter

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
          Expects keys: "5M", "1H", "4H". 5M is used by T3/T5 for precision entry.
        """
        if df is None or len(df) < 20:
            return []

        # Periodically refresh EV averages from candidate log
        self._bar_count += 1
        if self._bar_count % _EV_REFRESH_INTERVAL == 0:
            self._ev_filter.refresh()

        # Extract per-TF dataframes from the HTF dict passed by main.py.
        # Each trader requires a specific timeframe as its HTF context:
        #   T1 (NY EMA):        1H — EMA21/50/200 stack, ADX, London open detection
        #   T2 (FVG+BOS):       4H — BOS detection, structure swing high/low
        #   T3 (Judas Swing):   4H — trend bias ADX + BOS
        #   T4 (News Momentum): 4H — H4 EMA21/50 trend filter
        #   T5 (Asian MR):      1H — ADX ranging filter (must be <= 27)
        htf_1h = _first_present_frame(df_htf, "1H", "H1", default=df)
        htf_4h = _first_present_frame(df_htf, "4H", "H4", default=df)
        df_5m  = df_htf.get("5M")   # precision execution TF for T3/T5 (None = fall back to 15M)

        _HTF_MAP = {
            "trader_1": htf_1h,
            "trader_2": htf_4h,
            "trader_3": htf_4h,
            "trader_4": htf_4h,
            "trader_5": htf_1h,
        }

        now = datetime.now(timezone.utc)

        # Step 1: ML inference — pass full df_htf dict so each model extracts its TF.
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

        # Step 2: Run all session-appropriate traders.
        # Each trader receives its designated HTF timeframe.
        # T3 and T5 also receive df_5m for precision execution entry.
        raw_signals = {}
        for trader in self._traders:
            try:
                tid = trader.TRADER_ID
                trader_htf = _HTF_MAP.get(tid, htf_1h)
                trader_5m  = df_5m if tid in ("trader_3", "trader_5") else None
                sig = trader.analyze_market(symbol, df, trader_htf, ml_preds, df_5m=trader_5m)
                if sig is not None:
                    raw_signals[tid] = sig
            except Exception as exc:
                logger.error("Trader %s error: %s", trader.TRADER_ID, exc)

        if not raw_signals:
            return []

        # Step 3 (ML): Refine quality score per signal (now has full signal context).
        # Overwrites the bar-level quality_score in ml_preds with the signal-specific score
        # so _compute_ensemble_score uses the most accurate value for each signal.
        if self._settings.ML_ENABLED:
            for tid, sig in raw_signals.items():
                quality = self._get_ml_score_for_signal(sig, ml_preds, df.iloc[-1])
                if quality is not None:
                    sig.setdefault("signal_metadata", {})["quality_score"] = quality
                    # Update ml_preds so ensemble score uses this signal's quality
                    ml_preds["quality_score"] = quality

        # Step 4: RL action selection (done inside BaseTrader.analyze_market already)
        # Confirm signals with ensemble score >= 0.5
        approved = []
        for tid, sig in raw_signals.items():
            confidence = float(sig.get("confidence", 0.60))
            if self._settings.ML_ENABLED:
                meta = sig.get("signal_metadata", {})
                ict_score = confidence
                ensemble = self._compute_ensemble_score(
                    ict_score, ml_preds, sig.get("side", "buy"), sig.get("symbol", symbol)
                )
                if ensemble < 0.5:
                    logger.debug(
                        "Signal %s filtered — ensemble=%.3f ict=%.3f regime=%s "
                        "p_dir=%.3f quality=%.3f sentiment=%.3f(%s/%s)",
                        tid, ensemble, ict_score,
                        ml_preds.get("regime", "?"),
                        ml_preds.get("p_bull" if sig.get("side") == "buy" else "p_bear", 0.5),
                        ml_preds.get("quality_score", 0.5),
                        ml_preds.get("sentiment_score", 0.0),
                        ml_preds.get("sentiment_label", "?"),
                        ml_preds.get("sentiment_backend", "?"),
                    )
                    continue
                sig["confidence"] = round(float(np.clip(ensemble, 0, 1)), 4)
                meta["ensemble_score"] = round(ensemble, 4)
                meta["sentiment_label"] = ml_preds.get("sentiment_label", "neutral")
                meta["sentiment_backend"] = ml_preds.get("sentiment_backend", "neutral")
                meta["sentiment_confidence"] = ml_preds.get("sentiment_confidence", 0.0)
                from services.feature_engine import MACRO_FEATURES
                meta["macro"] = {k: float(ml_preds.get(k, 0.0)) for k in MACRO_FEATURES}
                logger.info(
                    "Signal APPROVED %s %s %s — ensemble=%.3f ict=%.3f regime=%s "
                    "p_dir=%.3f quality=%.3f sentiment=%.3f(%s/%s)",
                    tid, sig.get("symbol"), sig.get("side"),
                    ensemble, ict_score,
                    ml_preds.get("regime", "?"),
                    ml_preds.get("p_bull" if sig.get("side") == "buy" else "p_bear", 0.5),
                    ml_preds.get("quality_score", 0.5),
                    ml_preds.get("sentiment_score", 0.0),
                    ml_preds.get("sentiment_label", "?"),
                    ml_preds.get("sentiment_backend", "?"),
                )

            # Confidence gate — minimum 0.55
            if float(sig.get("confidence", 0)) < 0.55:
                continue

            # Publish SIGNAL_GENERATED event (Contract 1)
            self._publish_signal(sig)
            approved.append(sig)

        return approved

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

        # Bar-level quality score — computed before traders run so they can gate on it.
        # Uses a neutral signal scaffold (no trader_id, neutral rr_ratio) to get a
        # regime/ATR/volume-based score from QualityScorer independent of signal details.
        qs = self._ml.get("quality")
        if qs is not None:
            try:
                bar = df.iloc[-1]
                neutral_signal = {"trader_id": "", "side": "buy", "rr_ratio": 1.5}
                features = self._fe.get_quality_features(neutral_signal, preds, bar)
                from services.feature_engine import QUALITY_FEATURES
                feat_dict = dict(zip(QUALITY_FEATURES, features))
                preds["quality_score"] = float(qs.predict(feat_dict))
            except Exception as exc:
                logger.error("QualityScorer bar-level inference failed: %s", exc)
                raise

        return preds

    def _neutral_ml_preds(self, symbol: str, df: pd.DataFrame) -> dict:
        """Rule-only predictions when ML_ENABLED=false (no ML scoring applied)."""
        return {
            "spread_pips": 1.0,
            **{f"atr_lag_{l}": 1.0 for l in [1, 4, 8, 24, 48, 96, 168, 336]},
        }

    def _get_ml_score_for_signal(
        self, signal: dict, ml_base: dict, bar: pd.Series
    ) -> float:
        """Compute per-signal quality score using XGBoost. Raises if model untrained."""
        qs = self._ml.get("quality")
        if qs is None:
            return None
        from services.feature_engine import QUALITY_FEATURES
        features = self._fe.get_quality_features(signal, ml_base, bar)
        feat_dict = dict(zip(QUALITY_FEATURES, features))
        return float(qs.predict(feat_dict))

    def _compute_ensemble_score(
        self,
        ict_score: float,
        ml_preds: dict,
        signal_direction: str,
        symbol: str = "",
    ) -> float:
        """
        New ensemble formula:
          gru_score = (p_bull - 0.5) * 2.0  if long
                    = (p_bear - 0.5) * 2.0  if short
          ml_score  = gru_score * 0.5 + quality_score * 0.5
          sentiment_bonus = sign_match * abs(sentiment) * 0.1 else -0.05
          regime_mult = {TRENDING_UP: 1.2 long, TRENDING_DOWN: 1.2 short,
                         RANGING: 0.9, VOLATILE: 0.85}
          score = (ict_score * 0.5 + ml_score * 0.5 + sentiment_bonus) * regime_mult
        """
        if signal_direction == "buy":
            p_dir = float(ml_preds.get("p_bull", 0.5))
        else:
            p_dir = float(ml_preds.get("p_bear", 0.5))

        gru_score = (p_dir - 0.5) * 2.0  # maps [0,1] → [-1,1]
        quality_score = float(ml_preds.get("quality_score", 0.5))
        ml_score = gru_score * 0.5 + quality_score * 0.5

        sentiment = float(ml_preds.get("sentiment_score", 0.0))
        if (signal_direction == "buy" and sentiment > 0) or \
           (signal_direction == "sell" and sentiment < 0):
            sentiment_bonus = abs(sentiment) * 0.1
        else:
            sentiment_bonus = -0.05

        regime = ml_preds.get("regime", "RANGING")
        regime_mult_map = {
            "TRENDING_UP": 1.2 if signal_direction == "buy" else 0.9,
            "TRENDING_DOWN": 1.2 if signal_direction == "sell" else 0.9,
            "RANGING": 0.9,
            "VOLATILE": 0.85,
            "CONSOLIDATION": 1.05,  # pre-breakout compression — slight positive bias for breakout traders
        }
        regime_mult = regime_mult_map.get(regime, 1.0)

        # Macro bias (risk-on/off + USD strength)
        macro_bias = 0.0
        vix = float(ml_preds.get("macro_vix_level", 0.0))
        if vix > 0.6:  # > ~30 VIX (scaled /50)
            macro_bias -= 0.05
        if float(ml_preds.get("macro_yield_spread", 0.0)) < 0:
            macro_bias -= 0.03
        spx_ret = float(ml_preds.get("idx_spx_ret", 0.0))
        if spx_ret < -0.01:
            macro_bias -= 0.02

        dxy_ret = float(ml_preds.get("idx_dxy_ret", 0.0))
        if symbol:
            usd_base = symbol.startswith("USD") and symbol != "XAUUSD"
            usd_quote = symbol.endswith("USD") and not usd_base and symbol != "XAUUSD"
            if usd_base:
                if signal_direction == "buy" and dxy_ret > 0:
                    macro_bias += 0.03
                elif signal_direction == "sell" and dxy_ret > 0:
                    macro_bias -= 0.03
            elif usd_quote or symbol == "XAUUSD":
                if signal_direction == "buy" and dxy_ret > 0:
                    macro_bias -= 0.03
                elif signal_direction == "sell" and dxy_ret > 0:
                    macro_bias += 0.03

        raw = (ict_score * 0.5 + ml_score * 0.5 + sentiment_bonus + macro_bias) * regime_mult
        return float(np.clip(raw, 0.0, 1.0))

    def record_trade_outcome(
        self,
        candidate_id: str,
        tp_hit: bool,
        sl_hit: bool,
        pnl: float,
        exit_reason: str = "",
    ) -> None:
        """
        Called by TradeJournal/ExecutionEngine when a trade closes.
        Updates the CandidateLogger with the actual outcome for statistical analysis.
        """
        if candidate_id and self._candidate_logger is not None:
            self._candidate_logger.mark_outcome(
                candidate_id=candidate_id,
                tp_hit=tp_hit,
                sl_hit=sl_hit,
                pnl=pnl,
                exit_reason=exit_reason,
            )

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
