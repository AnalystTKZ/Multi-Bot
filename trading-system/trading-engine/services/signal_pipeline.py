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
from services.regime_scores import classify_tradeability_directional

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
    Per-bar pipeline: Regime → GRU → Quality → Decision Engine.

    Decision hierarchy (no RL in active path):
      1. Risk limits (checked in main.py before this)
      2. Regime tradeability (TRADEABLE_UP / TRADEABLE_DOWN / NO_TRADE_*)
      3. GRU confidence ≥ ML_DIRECTION_THRESHOLD
      4. GRU expected_R ≥ MIN_EXPECTED_R
      5. Quality EV gate (runs post-PM-enrichment in main.py)
      6. Optional RL refinement (only when RL_ENABLED=true)

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

        # Per-symbol/side GRU R thresholds calibrated from training label distributions.
        # Shape: {symbol: {side: {q25, q50, n}}}. Loaded once at init; falls back to the
        # global GRU_MIN_EXPECTED_R_MULTIPLE env var when a symbol/side is not present.
        self._sym_r_thresholds: dict = {}
        _thresh_candidates = [
            os.path.join(os.path.dirname(__file__), "..", "weights", "gru_lstm", "symbol_r_thresholds.json"),
            os.path.join(os.path.dirname(__file__), "..", "..", "weights", "gru_lstm", "symbol_r_thresholds.json"),
        ]
        import json as _init_json
        for _tp in _thresh_candidates:
            _tp = os.path.normpath(_tp)
            if os.path.exists(_tp):
                try:
                    with open(_tp) as _tf_in:
                        self._sym_r_thresholds = _init_json.load(_tf_in)
                    logger.info("Loaded per-symbol GRU R thresholds from %s (%d symbols)",
                                _tp, len(self._sym_r_thresholds))
                except Exception as _te:
                    logger.warning("Failed to load GRU R thresholds from %s: %s", _tp, _te)
                break

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

        bar = df.iloc[-1]
        session = self._detect_session(now)

        # Step 2: Generate signal via Regime → GRU → EV decision hierarchy.
        raw_signal = self._compute_ml_signal(symbol, df, ml_preds)
        if raw_signal is None:
            return []

        # Optional Step 3: RL refinement (dormant unless RL_ENABLED=true).
        # RL can block a trade but never creates one; it refines, not gates.
        from services.feature_engine import RL_STATE_DIM
        rl_action = 0
        rl_state = np.zeros(RL_STATE_DIM, dtype=np.float32)
        _rl_enabled = str(os.getenv("RL_ENABLED", "0")).lower() in ("1", "true", "yes")
        if _rl_enabled:
            rl_agent = self._ml.get("rl")
            if rl_agent is not None:
                rl_state = self._build_rl_state(symbol, ml_preds, bar, portfolio)
                from models.rl_agent import _encode_action
                _trader_id, _rl_threshold = rl_agent.decide(
                    rl_state, {"ml_trader": True}, session
                )
                rl_action = _encode_action(_rl_threshold) if _rl_threshold > 0 else 0
                if _trader_id == 0:
                    logger.debug("RL refinement blocked %s (session=%s)", symbol, session)
                    return []

        meta = raw_signal.setdefault("signal_metadata", {})
        meta["session"] = session
        meta["rl_action"] = rl_action
        meta["rl_threshold"] = 0.0   # kept for journal schema compatibility
        raw_signal["rl_action"] = rl_action
        raw_signal["state_at_entry"] = rl_state.tolist()

        logger.info(
            "Signal APPROVED ml_trader %s %s — conf=%.3f tradeability=%s htf=%s ltf=%s "
            "p_bull=%.3f p_bear=%.3f",
            symbol, raw_signal.get("side"),
            raw_signal.get("confidence", 0),
            meta.get("tradeability", "?"),
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

        unified = self._ml.get("unified_direction_regime")
        if unified:
            try:
                preds.update(unified.predict(df, symbol=symbol, df_htf=htf))
                preds["spread_pips"] = 1.0
                return preds
            except RuntimeError as exc:
                logger.error("UnifiedDirectionRegime not trained — ML signals disabled. %s", exc)
                raise
            except Exception as exc:
                logger.error("UnifiedDirectionRegime inference error: %s", exc)
                raise

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
        threshold: optional override for direction confidence (defaults to settings).

        Gate order:
          1. ATR sanity
          2. GRU variance ≤ MAX_UNCERTAINTY
          3. GRU direction ≥ ML_DIRECTION_THRESHOLD  →  determines side
          4. Directional tradeability: TRADEABLE_UP (buy) / TRADEABLE_DOWN (sell)
             Derived from classify_tradeability_directional(ltf_trade_regime, htf_bias)
          5. HTF regime confidence ≥ HTF_MIN_REGIME_CONFIDENCE
          6. Per-bar structural validation (range_valid, pullback, BOS, FVG)
          7. Geometric RR ≥ MIN_REWARD_TO_RISK
          8. Probability-weighted expected_R ≥ MIN_EXPECTED_R
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

        # Gate 2: GRU uncertainty — inference variance is clamped to the same
        # scale used by the training objective.
        _uncertainty = float(ml_preds.get("expected_variance", 0.0))
        if not np.isfinite(_uncertainty):
            return None
        _max_unc = float(getattr(self._settings, "MAX_UNCERTAINTY", 1.0))
        if _uncertainty > _max_unc:
            return None

        # Gate 3: GRU direction — use settings threshold (RL no longer sets this)
        p_bull = float(ml_preds.get("p_bull", 0.5))
        p_bear = float(ml_preds.get("p_bear", 0.5))
        _dir_thresh = threshold if threshold is not None else float(
            getattr(self._settings, "ML_DIRECTION_THRESHOLD", 0.65)
        )
        if p_bull >= p_bear and p_bull >= _dir_thresh:
            side = "buy"
            conf = p_bull
        elif p_bear > p_bull and p_bear >= _dir_thresh:
            side = "sell"
            conf = p_bear
        else:
            return None

        # GRU side-conditioned expected R gate — uses the head for the predicted side.
        # Per-symbol/side thresholds loaded from weights/gru_lstm/symbol_r_thresholds.json
        # (calibrated from training label Q25 per symbol × side after each GRU training run).
        # Falls back to global GRU_MIN_EXPECTED_R_MULTIPLE when no per-symbol data exists.
        _side_r_key = "expected_r_long" if side == "buy" else "expected_r_short"
        _gru_expected_r = ml_preds.get(_side_r_key)
        if _gru_expected_r is None:
            _gru_expected_r = ml_preds.get("expected_r_gru")
        if _gru_expected_r is not None:
            _gru_expected_r = float(_gru_expected_r)
            if np.isfinite(_gru_expected_r):
                _global_min_r = float(os.getenv("GRU_MIN_EXPECTED_R_MULTIPLE", "0.50"))
                # Look up per-symbol/side threshold from calibrated JSON; use Q25 as gate.
                _sym_side_data = self._sym_r_thresholds.get(symbol, {}).get(side, {})
                if _sym_side_data:
                    _min_gru_r = float(_sym_side_data.get("q25", _global_min_r))
                    # Don't allow per-symbol threshold to go below a hard floor of 0.20
                    # (prevents degenerate symbols from disabling the gate entirely).
                    _min_gru_r = max(_min_gru_r, 0.20)
                else:
                    _min_gru_r = _global_min_r
                if _gru_expected_r < _min_gru_r:
                    logger.debug(
                        "Signal rejected %s %s — GRU side_R=%.3f < min=%.3f (per_sym=%s)",
                        symbol, side, _gru_expected_r, _min_gru_r, bool(_sym_side_data),
                    )
                    return None

        # Gate 4: Directional tradeability — replaces old HTF bias + LTF routing
        # TRADEABLE_UP → buy only; TRADEABLE_DOWN → sell only; NO_TRADE_* → block
        _htf_bias = str(ml_preds.get("regime", "BIAS_NEUTRAL"))
        _htf_regime_conf = float(ml_preds.get("regime_conf", 1.0 / 3.0))
        _htf_min_conf = float(os.getenv("HTF_MIN_REGIME_CONFIDENCE", "0.70"))
        _ltf_trade_regime = str(ml_preds.get("trade_regime") or "").upper()

        _tradeability = classify_tradeability_directional(_ltf_trade_regime, _htf_bias)

        if _tradeability in ("NO_TRADE_CHOP", "NO_TRADE_EXTREME_VOL", "NO_TRADE_UNCERTAIN"):
            logger.debug("Signal rejected %s — tradeability=%s", symbol, _tradeability)
            return None
        if side == "buy" and _tradeability != "TRADEABLE_UP":
            logger.debug("Signal rejected %s buy — tradeability=%s", symbol, _tradeability)
            return None
        if side == "sell" and _tradeability != "TRADEABLE_DOWN":
            logger.debug("Signal rejected %s sell — tradeability=%s", symbol, _tradeability)
            return None

        # Gate 5: HTF regime classifier confidence
        if _htf_regime_conf < _htf_min_conf:
            logger.debug(
                "Signal rejected %s — htf_low_regime_confidence conf=%.3f",
                symbol, _htf_regime_conf,
            )
            return None

        # Gate 6: Per-bar structural validation (range, pullback, BOS, FVG)
        _ltf_behaviour = str(ml_preds.get("regime_ltf", "TRENDING")).upper()
        if _ltf_trade_regime in {"TRADEABLE_TREND", "TRADEABLE_TREND_HIGH_VOL"}:
            _ltf_behaviour = "TRENDING"
        elif _ltf_trade_regime == "RANGE":
            _ltf_behaviour = "RANGING"
        elif _ltf_trade_regime == "CONSOLIDATION":
            _ltf_behaviour = "CONSOLIDATING"
        elif _ltf_trade_regime == "NO_TRADE_EXTREME_VOL":
            _ltf_behaviour = "VOLATILE"
        _range_valid = bool(bar.get("range_valid", False))
        _pullback_valid = bool(bar.get("pullback_valid", False))
        _volatile_thresh = float(os.getenv("VOLATILE_ENTRY_THRESHOLD", "0.70"))
        _block_consol = str(os.getenv("BLOCK_LTF_CONSOLIDATING", "1")).lower() in ("1", "true", "yes")
        _require_range = str(os.getenv("RANGING_REQUIRE_RANGE", "1")).lower() in ("1", "true", "yes")
        _allowed, _reason = combined_market_decision(
            htf_bias=_htf_bias,
            ltf_behaviour=_ltf_behaviour,
            side=side,
            confidence=conf,
            bar=bar,
            neutral_threshold=0.60,
            volatile_threshold=_volatile_thresh,
            block_consolidating=_block_consol,
            require_range=_require_range,
            htf_confidence=_htf_regime_conf,
            regime_scores=ml_preds.get("regime_scores"),
            trade_regime=_tradeability,   # pass directional state for structural routing
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
        _min_er = float(getattr(self._settings, "MIN_EXPECTED_R", 1.20))
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
                "trade_regime":      _ltf_trade_regime,
                "tradeability":      _tradeability,
                "regime_scores":     ml_preds.get("regime_scores", {}),
                "expected_variance": _uncertainty,
                "p_bull":            p_bull,
                "p_bear":            p_bear,
                "expected_r_gru":    float(ml_preds.get("expected_r_gru", float("nan"))),
                "atr":               atr,
                "atr_at_entry":      atr,
                "expected_r":        float(expected_r),
                "strategy":          "ml_native",
                "pullback_valid":    _pullback_valid,
                "pullback_level":    float(bar.get("pullback_level", float("nan"))),
                "mss_bull":          int(bool(bar.get("mss_bull", False))),
                "mss_bear":          int(bool(bar.get("mss_bear", False))),
                "adx_at_signal":     float(ml_preds.get("adx_14", ml_preds.get("adx", 20.0))),
                "atr_ratio_at_signal": float(ml_preds.get("atr_normalized", ml_preds.get("atr_ratio", 1.0))),
                "spread_at_signal":  float(ml_preds.get("spread_pips", 1.0)),
                "spread_pips":       float(ml_preds.get("spread_pips", 1.0)),
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
                "tradeability": str(meta.get("tradeability", "")),
                "rr_ratio": float(sig.get("rr_ratio", 1.5)),
            },
        }
        self._bus.publish(EventType.SIGNAL_GENERATED, event)
