"""
Unit tests for score-based regime detection.

Run from trading-engine/:
    python -m pytest tests/test_regime_scores.py -v
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _ohlcv(close: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(close), freq="h")
    close_s = pd.Series(close, index=idx)
    spread = np.maximum(close_s.abs() * 0.001, 0.0001)
    return pd.DataFrame(
        {
            "open": close_s.shift(1).fillna(close_s.iloc[0]),
            "high": close_s + spread,
            "low": close_s - spread,
            "close": close_s,
            "volume": 1000.0,
        },
        index=idx,
    )


def test_efficiency_ratio_clean_trend_is_high_and_chop_is_low():
    from services.regime_scores import efficiency_ratio

    trend_close = pd.Series(np.arange(1.0, 41.0))
    chop_close = pd.Series(np.tile([1.0, 2.0], 20))

    assert efficiency_ratio(trend_close, 20).iloc[-1] > 0.95
    assert efficiency_ratio(chop_close, 20).iloc[-1] < 0.10


def test_rolling_percentile_is_causal():
    from services.regime_scores import rolling_percentile

    values = pd.Series([1.0, 2.0, 3.0, 100.0])
    pct = rolling_percentile(values, window=4, min_periods=2)

    assert pct.iloc[-1] == 1.0
    assert pct.iloc[1] == 1.0


def test_build_regime_score_frame_detects_trend_structure():
    from services.regime_scores import LTF_SCORE_COLUMNS, build_regime_score_frame

    close = np.linspace(1.0, 1.30, 240)
    scores = build_regime_score_frame(_ohlcv(close), symbol="EURUSD")
    last = scores.iloc[-1]

    assert LTF_SCORE_COLUMNS == [
        "trend_score",
        "range_score",
        "chop_score",
        "volatility_percentile",
        "consolidation_score",
    ]
    assert set(LTF_SCORE_COLUMNS).issubset(scores.columns)
    assert last["trend_score"] > last["range_score"]
    assert last["bias_up_score"] > last["bias_down_score"]
    assert last["efficiency_ratio_20"] > 0.80


def test_score_frame_does_not_look_ahead():
    from services.regime_scores import build_regime_score_frame

    close = np.ones(180)
    close[:120] = np.linspace(1.00, 1.10, 120)
    close[120:] = np.linspace(1.10, 1.80, 60)
    full = _ohlcv(close)
    truncated = full.iloc[:120].copy()

    full_scores = build_regime_score_frame(full, symbol="EURUSD")
    truncated_scores = build_regime_score_frame(truncated, symbol="EURUSD")

    cols = ["efficiency_ratio_20", "trend_score", "atr_percentile_500", "bb_width_percentile"]
    np.testing.assert_allclose(
        full_scores.loc[truncated.index[-1], cols].to_numpy(dtype=float),
        truncated_scores.iloc[-1][cols].to_numpy(dtype=float),
        rtol=1e-6,
        atol=1e-6,
    )


def test_final_regime_decision_blocks_chop_and_allows_trend():
    from services.regime_scores import classify_trade_regime

    assert classify_trade_regime(
        {
            "trend_score": 0.20,
            "range_score": 0.30,
            "chop_score": 0.80,
            "volatility_percentile": 0.40,
            "atr_percentile_500": 0.50,
            "efficiency_ratio_20": 0.10,
            "consolidation_score": 0.10,
        }
    ) == "NO_TRADE_CHOP"

    assert classify_trade_regime(
        {
            "trend_score": 0.80,
            "range_score": 0.20,
            "chop_score": 0.10,
            "volatility_percentile": 0.40,
            "atr_percentile_500": 0.55,
            "efficiency_ratio_20": 0.70,
            "consolidation_score": 0.10,
        }
    ) == "TRADEABLE_TREND"


def test_market_decision_uses_score_regime_as_filter():
    from services.market_decision import combined_market_decision

    allowed, reason = combined_market_decision(
        htf_bias="BIAS_UP",
        ltf_behaviour="TRENDING",
        side="buy",
        confidence=0.90,
        bar={"bos_bull": True},
        trade_regime="NO_TRADE_CHOP",
    )
    assert not allowed
    assert reason == "no_trade_chop"

    allowed, reason = combined_market_decision(
        htf_bias="BIAS_UP",
        ltf_behaviour="RANGING",
        side="buy",
        confidence=0.90,
        bar={"bos_bull": True},
        trade_regime="TRADEABLE_TREND",
    )
    assert allowed
    assert reason == "trend_structure_entry"


def test_ltf_model_score_alignment_keeps_native_source_history():
    cwd = os.getcwd()
    try:
        from scripts.run_backtest import _align_ltf_score_frame_complete
    finally:
        os.chdir(cwd)
    from services.regime_scores import LTF_SCORE_COLUMNS

    source_idx = pd.date_range("2020-01-01", periods=6, freq="h", tz="UTC")
    target_idx = pd.date_range("2020-01-01 03:00", periods=8, freq="15min", tz="UTC")
    score_frame = pd.DataFrame(
        {
            name: np.linspace(0.1, 0.9, len(source_idx), dtype=np.float32)
            for name in LTF_SCORE_COLUMNS
        },
        index=source_idx,
    )

    source_aligned = _align_ltf_score_frame_complete(
        score_frame, source_idx, LTF_SCORE_COLUMNS, "GBPUSD", "model source"
    )
    target_aligned = _align_ltf_score_frame_complete(
        source_aligned, target_idx, LTF_SCORE_COLUMNS, "GBPUSD", "alignment"
    )

    assert not source_aligned[LTF_SCORE_COLUMNS].isna().any().any()
    assert not target_aligned[LTF_SCORE_COLUMNS].isna().any().any()

    with pytest.raises(RuntimeError, match="LTF score frame has gaps"):
        _align_ltf_score_frame_complete(
            target_aligned, source_idx, LTF_SCORE_COLUMNS, "GBPUSD", "model source"
        )
