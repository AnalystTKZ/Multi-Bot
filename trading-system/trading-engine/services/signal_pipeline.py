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
            rl_state = self._build_rl_state(ml_preds, bar, portfolio)
            _trader_id, rl_threshold = rl_agent.decide(
                rl_state, {"ml_trader": True}, session
            )
            if _trader_id == 0:
                logger.debug("RL NoTrade %s (session=%s)", symbol, session)
                return []
        else:
            rl_threshold = float(getattr(self._settings, "ML_DIRECTION_THRESHOLD", 0.62))
            rl_state = np.zeros(43, dtype=np.float32)

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

        # Enrich metadata with sentiment and macro context
        meta["session"] = session
        meta["sentiment_label"] = ml_preds.get("sentiment_label", "neutral")
        meta["sentiment_backend"] = ml_preds.get("sentiment_backend", "neutral")
        meta["sentiment_confidence"] = ml_preds.get("sentiment_confidence", 0.0)
        from services.feature_engine import MACRO_FEATURES
        meta["macro"] = {k: float(ml_preds.get(k, 0.0)) for k in MACRO_FEATURES}

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

        self._publish_signal(raw_signal)
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
        self, ml_preds: dict, bar, portfolio: Optional[dict]
    ) -> np.ndarray:
        """
        Build the 43-dim RL state vector from inference outputs + portfolio context.

        Dims 0–2:   GRU direction (p_bull, p_bear, uncertainty)
        Dims 3–5:   HTF regime probabilities [BIAS_UP, BIAS_DOWN, BIAS_NEUTRAL]
        Dims 6–9:   LTF regime probabilities [TRENDING, RANGING, CONSOLIDATING, VOLATILE]
        Dims 10–11: Sentiment (score, confidence)
        Dims 12–19: ATR history ratios (8 lags)
        Dim  20:    Spread normalized
        Dims 21–23: Time (sin_hour, cos_hour, session_enc)
        Dims 24–28: Portfolio (win_rate, drawdown, trades_today, open_pos, daily_pnl)
        Dims 29–32: Market structure (adx, vol_slope, vix, yield_spread)
        Dims 33–36: Auction indicators (vwap_dist, vol_delta, wick_ratio, cum_delta)
        Dims 37–39: Regime quality (ema_stack, htf_conf, ltf_conf)
        Dims 40–42: Extra context (atr_ratio, volume_ratio, direction_strength)
        """
        from services.feature_engine import _ATR_LAGS

        port = portfolio or {}

        # HTF / LTF regime probability arrays
        htf_p = ml_preds.get("regime_proba") or [1/3, 1/3, 1/3]
        if len(htf_p) < 3:
            htf_p = list(htf_p) + [0.0] * (3 - len(htf_p))

        ltf_p = ml_preds.get("regime_ltf_conf")
        if ltf_p is None or isinstance(ltf_p, float):
            # Scalar confidence — rebuild as one-hot-ish
            ltf_id = int(ml_preds.get("regime_ltf_id", 0))
            c = float(ltf_p or 0.4)
            ltf_p = [(c if i == ltf_id else (1 - c) / 3) for i in range(4)]
        if len(ltf_p) < 4:
            ltf_p = list(ltf_p) + [0.0] * (4 - len(ltf_p))

        atr_lags = [
            float(np.clip(ml_preds.get(f"atr_lag_{lag}", 1.0), 0.0, 5.0))
            for lag in _ATR_LAGS
        ]

        now_h = datetime.now(timezone.utc).hour
        _session_enc = {"INACTIVE": 0.0, "ASIAN": 1/3, "LONDON": 2/3, "NY": 1.0}
        session_str = self._detect_session(datetime.now(timezone.utc))
        session_enc = _session_enc.get(session_str, 0.0)

        state = np.array([
            # GRU direction
            float(np.clip(ml_preds.get("p_bull", 0.5), 0.0, 1.0)),          # 0
            float(np.clip(ml_preds.get("p_bear", 0.5), 0.0, 1.0)),          # 1
            float(np.clip(ml_preds.get("expected_variance", 0.1), 0.0, 1.0)), # 2
            # HTF regime
            float(np.clip(htf_p[0], 0.0, 1.0)),                             # 3
            float(np.clip(htf_p[1], 0.0, 1.0)),                             # 4
            float(np.clip(htf_p[2], 0.0, 1.0)),                             # 5
            # LTF regime
            float(np.clip(ltf_p[0], 0.0, 1.0)),                             # 6
            float(np.clip(ltf_p[1], 0.0, 1.0)),                             # 7
            float(np.clip(ltf_p[2], 0.0, 1.0)),                             # 8
            float(np.clip(ltf_p[3], 0.0, 1.0)),                             # 9
            # Sentiment
            float(np.clip(ml_preds.get("sentiment_score", 0.0), -1.0, 1.0)), # 10
            float(np.clip(ml_preds.get("sentiment_confidence", 0.5), 0.0, 1.0)), # 11
            # ATR history ratios (8 lags)
            *atr_lags,                                                        # 12–19
            # Spread
            float(np.clip(ml_preds.get("spread_pips", 1.0) / 5.0, 0.0, 1.0)), # 20
            # Time cyclical + session
            float(np.sin(2 * np.pi * now_h / 24)),                          # 21
            float(np.cos(2 * np.pi * now_h / 24)),                          # 22
            float(session_enc),                                               # 23
            # Portfolio context
            float(np.clip(port.get("win_rate_10", 0.5), 0.0, 1.0)),         # 24
            float(np.clip(port.get("drawdown_pct", 0.0) / 0.20, 0.0, 1.0)), # 25
            float(np.clip(port.get("trades_today", 0) / 5.0, 0.0, 1.0)),    # 26
            float(np.clip(port.get("open_positions", 0) / 5.0, 0.0, 1.0)),  # 27
            float(np.clip(
                port.get("daily_pnl", 0.0) / (port.get("equity", 1000.0) + 1e-9),
                -0.10, 0.10,
            )),                                                               # 28
            # Market structure
            float(np.clip(float(bar.get("adx_14", 20.0)) / 50.0, 0.0, 1.0)), # 29
            float(np.clip(float(ml_preds.get("vol_slope", 0.0)), -1.0, 1.0)), # 30
            float(np.clip(ml_preds.get("macro_vix_level", 0.2), 0.0, 1.0)), # 31
            float(np.clip(
                ml_preds.get("macro_yield_spread", 0.0) / 0.02, -1.0, 1.0
            )),                                                               # 32
            # Auction indicators (from bar computed in feature engine)
            float(np.clip(float(bar.get("vwap_dist_atr", 0.0)) / 3.0, -1.0, 1.0)), # 33
            float(np.clip(float(bar.get("volume_delta_pct", 0.0)), -1.0, 1.0)),     # 34
            float(np.clip(float(bar.get("wick_auction_ratio", 0.5)), 0.0, 1.0)),    # 35
            float(np.clip(float(ml_preds.get("cum_delta_norm", 0.0)), -1.0, 1.0)),  # 36
            # Regime quality signals
            float(np.clip(float(bar.get("ema_stack", 0)) / 2.0, -1.0, 1.0)), # 37
            float(np.clip(float(max(htf_p)), 0.0, 1.0)),                    # 38
            float(np.clip(float(max(ltf_p)), 0.0, 1.0)),                    # 39
            # Extra context
            float(np.clip(
                float(ml_preds.get("atr_normalized", ml_preds.get("atr_ratio", 1.0)))
                / 3.0, 0.0, 1.0
            )),                                                               # 40
            float(np.clip(float(ml_preds.get("volume_ratio", 1.0)) / 3.0, 0.0, 1.0)), # 41
            float(np.clip(
                max(ml_preds.get("p_bull", 0.5), ml_preds.get("p_bear", 0.5)) * 2.0 - 1.0,
                -1.0, 1.0,
            )),                                                               # 42
        ], dtype=np.float32)

        return np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)

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
            except RuntimeError as exc:
                logger.error("LTF RegimeClassifier not trained — ML signals disabled. %s", exc)
                raise
            except Exception as exc:
                logger.error("LTF Regime inference error: %s", exc)
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
                "volume_ratio":      float(ml_preds.get("volume_ratio", 1.0)),
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
                "p_bull": float(meta.get("p_bull", 0.5)),
                "p_bear": float(meta.get("p_bear", 0.5)),
                "regime": str(meta.get("regime", "")),
                "sentiment_score": float(meta.get("sentiment_score", 0.0)),
                "rr_ratio": float(sig.get("rr_ratio", 1.5)),
            },
        }
        self._bus.publish(EventType.SIGNAL_GENERATED, event)
