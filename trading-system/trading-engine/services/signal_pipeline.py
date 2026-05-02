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
from services.market_decision import combined_market_decision

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
    Per-bar pipeline: ML inference → RL threshold → market-decision gate.

    Signal logic mirrors run_backtest._compute_backtest_signal (source of truth).
    PortfolioManager and QualityScorer run after this in main.py; only the final
    enriched signal is published.
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
        self, symbol: str, df: pd.DataFrame, df_htf: dict,
        portfolio: Optional[dict] = None,
    ) -> List[dict]:
        """
        Returns list of approved signals (usually 0 or 1).
        df_htf: {timeframe: DataFrame} for HTF context.
        portfolio: current portfolio state for RL state vector.
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

        # Dead zone 12:00–13:00 UTC (mirrors backtest _backtest_trader)
        now = datetime.now(timezone.utc)
        if now.hour == 12:
            return []

        # Step 2: RL agent selects dynamic confidence threshold.
        # decide() returns (trader_id=1, threshold) or (0, 0.0) for NoTrade.
        # Uses trained PPO when available; session-aware heuristic before training.
        rl_agent = self._ml.get("rl")
        bar = df.iloc[-1]
        session = self._detect_session(now)
        if rl_agent is not None:
            rl_state = self._build_rl_state(symbol, ml_preds, bar, portfolio)
            _trader_id, rl_threshold = rl_agent.decide(
                rl_state, {"ml_trader": True}, session
            )
            if _trader_id == 0:
                logger.debug("RL NoTrade %s (session=%s)", symbol, session)
                return []
        else:
            rl_threshold = float(getattr(self._settings, "ML_DIRECTION_THRESHOLD", 0.62))
            from services.feature_engine import RL_STATE_DIM
            rl_state = np.zeros(RL_STATE_DIM, dtype=np.float32)

        from models.rl_agent import _encode_action
        rl_action = _encode_action(rl_threshold) if rl_threshold > 0 else 0

        # Step 3: Generate signal gated by RL-determined threshold.
        raw_signal = self._compute_ml_signal(symbol, df, ml_preds, threshold=rl_threshold)
        if raw_signal is None:
            return []

        # Attach RL metadata so TradeJournal can reconstruct episodes for RL training.
        raw_signal["rl_action"] = rl_action
        raw_signal["state_at_entry"] = rl_state.tolist()
        meta = raw_signal.setdefault("signal_metadata", {})
        meta["rl_action"] = rl_action
        meta["rl_threshold"] = rl_threshold

        meta["session"] = session

        logger.info(
            "Signal APPROVED ml_trader %s %s — conf=%.3f rl_thresh=%.2f htf=%s ltf=%s "
            "p_bull=%.3f p_bear=%.3f",
            symbol, raw_signal.get("side"),
            raw_signal.get("confidence", 0),
            rl_threshold,
            ml_preds.get("regime", "?"),
            ml_preds.get("regime_ltf", "?"),
            ml_preds.get("p_bull", 0.5),
            ml_preds.get("p_bear", 0.5),
        )

        return [raw_signal]

    @staticmethod
    def _detect_session(now: datetime) -> str:
        h = now.hour
        if 2 <= h < 7:
            return "ASIAN"
        if 7 <= h < 12:
            return "LONDON"
        if 13 <= h < 18:
            return "NY"
        return "INACTIVE"

    def _build_rl_state(
        self, symbol: str, ml_preds: dict, bar, portfolio: Optional[dict]
    ) -> np.ndarray:
        """
        Build the canonical 43-dim RL state vector from the shared FeatureEngine
        contract. This keeps live RL, journal replay, and retraining aligned.
        """
        ml_preds = dict(ml_preds or {})
        ml_preds["session"] = self._detect_session(datetime.now(timezone.utc))
        return self._fe.get_rl_state(
            bar=bar,
            portfolio=portfolio or {},
            signals={"ml_trader": True, "session": ml_preds["session"]},
            ml_preds=ml_preds,
            symbol=symbol,
        )

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

        # HTF RegimeClassifier (4H bias: BIAS_UP / BIAS_DOWN / BIAS_NEUTRAL)
        regime_htf = self._ml.get("regime_htf") or self._ml.get("regime_4h") or self._ml.get("regime")
        if regime_htf:
            try:
                r = regime_htf.predict(df, symbol=symbol, df_htf=htf)
                preds["regime"]       = r.get("regime")
                preds["regime_id"]    = r.get("regime_id")
                preds["regime_proba"] = r.get("proba")
                preds["regime_conf"]  = float(r.get("regime_confidence", 1.0 / 3.0))
            except RuntimeError as exc:
                logger.error("HTF RegimeClassifier not trained — ML signals disabled. %s", exc)
                raise
            except Exception as exc:
                logger.error("HTF Regime inference error: %s", exc)
                raise

        # LTF RegimeClassifier (1H behaviour: TRENDING / RANGING / CONSOLIDATING / VOLATILE)
        regime_ltf = self._ml.get("regime_ltf") or self._ml.get("regime_1h")
        if regime_ltf:
            try:
                r = regime_ltf.predict(df, symbol=symbol, df_htf=htf)
                preds["regime_ltf"]      = r.get("regime")
                preds["regime_ltf_id"]   = r.get("regime_id")
                preds["regime_ltf_conf"] = r.get("proba")
                if r.get("regime_scores"):
                    preds["regime_scores"] = r.get("regime_scores")
                    preds.update(r.get("regime_scores") or {})
                if r.get("trade_regime"):
                    preds["trade_regime"] = r.get("trade_regime")
            except RuntimeError as exc:
                logger.error("LTF RegimeClassifier not trained — ML signals disabled. %s", exc)
                raise
            except Exception as exc:
                logger.error("LTF Regime inference error: %s", exc)
                raise

        # Spread
        preds["spread_pips"] = 1.0  # default; updated by DataFetcher in main.py

        # QualityScorer (EV) runs post-signal in main.py after PM enrichment.
        # It requires actual rr_ratio from the enriched signal, so it cannot run here.

        return preds

    def _neutral_ml_preds(self, symbol: str, df: pd.DataFrame) -> dict:
        """Rule-only predictions when ML_ENABLED=false (no ML scoring applied)."""
        return {
            "spread_pips": 1.0,
        }

    def _compute_ml_signal(
        self, symbol: str, df: pd.DataFrame, ml_preds: dict,
        threshold: float | None = None,
    ) -> dict | None:
        """
        Mirrors run_backtest._compute_backtest_signal (source of truth).
        threshold: RL-determined confidence threshold; falls back to settings.

        Gate order:
          1. ATR sanity
          2. GRU variance ≤ MAX_UNCERTAINTY
          3. GRU direction ≥ threshold  →  determines side
          4. HTF bias must agree with GRU side
          5. LTF behaviour must permit entry:
               TRENDING   → optional pullback filter (REQUIRE_TRENDING_PULLBACK=1 is strict)
               VOLATILE   → high-conviction GRU only
               RANGING    → significant range + price at correct boundary
               CONSOLIDATING → blocked
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

        # Gate 2: GRU uncertainty — use settings (default 0.25, not the old env-var default of 2.0)
        _uncertainty = float(ml_preds.get("expected_variance", 0.0))
        _max_unc = float(getattr(self._settings, "MAX_UNCERTAINTY", 0.25))
        if _uncertainty > _max_unc:
            return None

        # Gate 3: GRU direction — use RL-determined threshold (or settings default)
        p_bull = float(ml_preds.get("p_bull", 0.5))
        p_bear = float(ml_preds.get("p_bear", 0.5))
        _dir_thresh = threshold if threshold is not None else float(
            getattr(self._settings, "ML_DIRECTION_THRESHOLD", 0.62)
        )
        if p_bull >= p_bear and p_bull >= _dir_thresh:
            side = "buy"
            conf = p_bull
        elif p_bear > p_bull and p_bear >= _dir_thresh:
            side = "sell"
            conf = p_bear
        else:
            return None

        # Gate 4: HTF regime classifier confidence
        _htf_bias = str(ml_preds.get("regime", "BIAS_NEUTRAL"))
        _htf_regime_conf = float(ml_preds.get("regime_conf", 1.0 / 3.0))
        _htf_min_conf = float(os.getenv("HTF_MIN_REGIME_CONFIDENCE", "0.55"))
        if _htf_bias != "BIAS_NEUTRAL" and _htf_regime_conf < _htf_min_conf:
            logger.debug(
                "Signal rejected %s — htf_low_regime_confidence htf=%s conf=%.3f",
                symbol, _htf_bias, _htf_regime_conf,
            )
            return None

        # Gate 5: combined HTF/LTF market-decision matrix
        _ltf_behaviour = str(ml_preds.get("regime_ltf", "TRENDING"))
        _trade_regime = str(ml_preds.get("trade_regime", "") or "").upper()
        if _trade_regime in {"TRADEABLE_TREND", "TRADEABLE_TREND_HIGH_VOL"}:
            _ltf_behaviour = "TRENDING"
        elif _trade_regime == "RANGE":
            _ltf_behaviour = "RANGING"
        elif _trade_regime == "CONSOLIDATION":
            _ltf_behaviour = "CONSOLIDATING"
        elif _trade_regime == "NO_TRADE_EXTREME_VOL":
            _ltf_behaviour = "VOLATILE"
        _range_valid    = bool(bar.get("range_valid", False))
        _pullback_valid = bool(bar.get("pullback_valid", False))
        _neutral_thresh = float(os.getenv("NEUTRAL_BIAS_THRESHOLD", "0.60"))
        _volatile_thresh = float(os.getenv("VOLATILE_ENTRY_THRESHOLD", "0.70"))
        _block_consol = str(os.getenv("BLOCK_LTF_CONSOLIDATING", "1")).lower() in ("1", "true", "yes")
        _require_range = str(os.getenv("RANGING_REQUIRE_RANGE", "1")).lower() in ("1", "true", "yes")
        _allowed, _reason = combined_market_decision(
            htf_bias=_htf_bias,
            ltf_behaviour=_ltf_behaviour,
            side=side,
            confidence=conf,
            bar=bar,
            neutral_threshold=_neutral_thresh,
            volatile_threshold=_volatile_thresh,
            block_consolidating=_block_consol,
            require_range=_require_range,
            htf_confidence=_htf_regime_conf,
            regime_scores=ml_preds.get("regime_scores"),
            trade_regime=ml_preds.get("trade_regime"),
        )
        if not _allowed:
            logger.debug(
                "Signal rejected %s %s — %s htf=%s ltf=%s conf=%.3f",
                symbol, side, _reason, _htf_bias, _ltf_behaviour, conf,
            )
            return None

        # ATR-based entry / SL / TP — use settings multipliers, with stricter values for XAUUSD
        _is_gold = (symbol == "XAUUSD")
        _sl_mult = float(
            getattr(self._settings, "GOLD_ATR_STOP_MULTIPLIER", 2.0)
            if _is_gold else getattr(self._settings, "ATR_STOP_MULTIPLIER", 1.5)
        )
        _tp_mult = float(
            getattr(self._settings, "GOLD_ATR_TARGET_MULTIPLIER", 3.5)
            if _is_gold else getattr(self._settings, "ATR_TARGET_MULTIPLIER", 2.5)
        )
        _min_rr = float(getattr(self._settings, "MIN_REWARD_TO_RISK", 1.50))
        sl_dist = atr * _sl_mult

        if _ltf_behaviour == "RANGING" and _range_valid:
            if side == "buy":
                stop_loss   = float(bar.get("range_support", close - sl_dist)) - atr * 0.3
                take_profit = float(bar.get("range_resist",  close + sl_dist * _tp_mult))
            else:
                stop_loss   = float(bar.get("range_resist",  close + sl_dist)) + atr * 0.3
                take_profit = float(bar.get("range_support", close - sl_dist * _tp_mult))
            actual_rr = abs(take_profit - close) / (abs(close - stop_loss) + 1e-9)
            if actual_rr < _min_rr:
                stop_loss   = (close - sl_dist) if side == "buy" else (close + sl_dist)
                take_profit = (close + sl_dist * _tp_mult) if side == "buy" else (close - sl_dist * _tp_mult)
        else:
            if side == "buy":
                stop_loss   = close - sl_dist
                take_profit = close + sl_dist * _tp_mult
            else:
                stop_loss   = close + sl_dist
                take_profit = close - sl_dist * _tp_mult

        # Gate: minimum reward-to-risk ratio (hard geometric check at entry)
        actual_rr = abs(take_profit - close) / (abs(close - stop_loss) + 1e-9)
        if actual_rr < _min_rr:
            logger.debug(
                "Signal rejected %s %s — rr_ratio=%.2f < min_rr=%.2f",
                symbol, side, actual_rr, _min_rr,
            )
            return None

        # Gate: probability-weighted expected R must exceed MIN_EXPECTED_R.
        # E[R] = P(win) × RR − P(loss) × 1.0
        p_win = p_bull if side == "buy" else p_bear
        expected_r = p_win * actual_rr - (1.0 - p_win) * 1.0
        _min_er = float(getattr(self._settings, "MIN_EXPECTED_R", 1.30))
        if expected_r < _min_er:
            logger.debug(
                "Signal rejected %s %s — expected_R=%.3f < min_expected_R=%.2f "
                "(p_win=%.3f rr=%.2f)",
                symbol, side, expected_r, _min_er, p_win, actual_rr,
            )
            return None

        return {
            "side":        side,
            "entry":       close,
            "stop_loss":   stop_loss,
            "take_profit": take_profit,
            "confidence":  round(float(conf), 3),
            "trader_id":   "ml_trader",
            "symbol":      symbol,
            "signal_metadata": {
                "regime":            _htf_bias,
                "regime_ltf":        _ltf_behaviour,
                "trade_regime":      _trade_regime or "",
                "regime_scores":     ml_preds.get("regime_scores", {}),
                "expected_variance": _uncertainty,
                "p_bull":            p_bull,
                "p_bear":            p_bear,
                "atr":               atr,
                "atr_at_entry":      atr,
                "strategy":          "ml_native",
                "pullback_valid":    _pullback_valid,
                "pullback_level":    float(bar.get("pullback_level", float("nan"))),
                "adx_at_signal":     float(ml_preds.get("adx_14", ml_preds.get("adx", 20.0))),
                "atr_ratio_at_signal": float(ml_preds.get("atr_normalized", ml_preds.get("atr_ratio", 1.0))),
                "spread_at_signal":  float(ml_preds.get("spread_pips", 1.0)),
                "news_in_30min":     int(ml_preds.get("news_in_30min", 0)),
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
                "ev": float(meta.get("ev", 0.0)),
                "p_bull": float(meta.get("p_bull", 0.5)),
                "p_bear": float(meta.get("p_bear", 0.5)),
                "regime": str(meta.get("regime", "")),
                "regime_ltf": str(meta.get("regime_ltf", "")),
                "trade_regime": str(meta.get("trade_regime", "")),
                "rr_ratio": float(sig.get("rr_ratio", 1.5)),
            },
        }
        self._bus.publish(EventType.SIGNAL_GENERATED, event)
