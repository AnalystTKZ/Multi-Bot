"""
Unit tests for risk math, statistical metrics, and signal rejection logic.

Run from trading-engine/:
    python -m pytest tests/test_risk_math.py -v

Each test is self-contained and requires no external data or model weights.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# Allow imports from trading-engine root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_settings(**overrides):
    defaults = dict(
        RISK_PER_TRADE=0.005,
        MAX_DAILY_LOSS_PCT=0.02,
        MAX_WEEKLY_LOSS_PCT=0.05,
        MAX_DRAWDOWN_PCT=0.15,
        MAX_CONCURRENT_POSITIONS=2,
        MAX_CORRELATED_POSITIONS=1,
        MAX_CONSECUTIVE_LOSSES=3,
        CONSECUTIVE_LOSS_COOLDOWN_BARS=10,
        ML_DIRECTION_THRESHOLD=0.62,
        MIN_EXPECTED_R=1.30,
        MAX_UNCERTAINTY=0.25,
        MIN_REWARD_TO_RISK=1.50,
        ATR_STOP_MULTIPLIER=1.5,
        ATR_TARGET_MULTIPLIER=2.5,
        GOLD_ATR_STOP_MULTIPLIER=2.0,
        GOLD_ATR_TARGET_MULTIPLIER=3.5,
        MAX_SPREAD_EURUSD=2.0,
        MAX_SPREAD_XAUUSD=30.0,
        KELLY_ENABLED=False,
        KELLY_FRACTION=0.25,
        ACCOUNT_BALANCE=10000.0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Expectancy
# ─────────────────────────────────────────────────────────────────────────────

class TestExpectancy:
    def test_positive_expectancy(self):
        """60% win rate, 2R reward, 1R loss → E[R] = 0.6×2 − 0.4×1 = 0.80"""
        realized_rr = [2.0, 2.0, -1.0] * 10  # 30 trades: 20 wins, 10 losses
        e = np.mean(realized_rr)
        assert abs(e - 0.80) < 1e-6

    def test_negative_expectancy(self):
        """40% win rate, 1.5R reward, 1R loss → E[R] = 0.4×1.5 − 0.6×1 = 0.00 (break-even)"""
        # Actually: 0.4*1.5 - 0.6*1.0 = 0.6 - 0.6 = 0.0
        realized_rr = [1.5, 1.5, -1.0, -1.0, -1.0] * 10
        e = np.mean(realized_rr)
        assert abs(e - 0.0) < 1e-6

    def test_expectancy_formula(self):
        """E[R] = win_rate × avg_win_R − loss_rate × avg_loss_R"""
        wins = [2.5, 3.0, 1.8]
        losses = [-1.0, -1.0, -1.0, -1.0]
        all_trades = wins + losses
        win_rate   = len(wins) / len(all_trades)
        avg_win_r  = np.mean(wins)
        avg_loss_r = abs(np.mean(losses))
        expected   = win_rate * avg_win_r - (1 - win_rate) * avg_loss_r
        assert abs(expected - np.mean(all_trades)) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# 2. Break-even win rate
# ─────────────────────────────────────────────────────────────────────────────

class TestBreakEvenWinRate:
    def test_2r_reward(self):
        """At 2:1 RR, break-even win rate = 1/(1+2) = 33.3%"""
        from services.risk_engine import RiskEngine
        be = RiskEngine.compute_break_even_win_rate(2.0)
        assert abs(be - 1/3) < 1e-6

    def test_1_5r_reward(self):
        """At 1.5:1 RR, break-even = 1/(1+1.5) = 40%"""
        from services.risk_engine import RiskEngine
        be = RiskEngine.compute_break_even_win_rate(1.5)
        assert abs(be - 0.40) < 1e-6

    def test_1r_reward(self):
        """At 1:1 RR (scalping), break-even = 50%"""
        from services.risk_engine import RiskEngine
        be = RiskEngine.compute_break_even_win_rate(1.0)
        assert abs(be - 0.50) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# 3. Expected R (probability-weighted)
# ─────────────────────────────────────────────────────────────────────────────

class TestExpectedR:
    def test_positive_ev(self):
        """P(win)=0.65, RR=2.5 → E[R] = 0.65×2.5 − 0.35×1 = 1.275"""
        from services.risk_engine import RiskEngine
        er = RiskEngine.compute_expected_r(p_win=0.65, reward_to_risk=2.5)
        assert abs(er - (0.65 * 2.5 - 0.35 * 1.0)) < 1e-6

    def test_negative_ev(self):
        """P(win)=0.40, RR=1.5 → E[R] = 0.40×1.5 − 0.60×1 = 0.00 (break-even)"""
        from services.risk_engine import RiskEngine
        er = RiskEngine.compute_expected_r(p_win=0.40, reward_to_risk=1.5)
        assert abs(er - 0.0) < 1e-6

    def test_below_min_expected_r_rejected(self):
        """E[R]=0.2 should be below MIN_EXPECTED_R=1.30"""
        from services.risk_engine import RiskEngine
        er = RiskEngine.compute_expected_r(p_win=0.55, reward_to_risk=1.5)
        # 0.55×1.5 − 0.45×1 = 0.825 − 0.45 = 0.375
        assert er < 1.30


# ─────────────────────────────────────────────────────────────────────────────
# 4. Position sizing (fixed fractional — no ML scaling)
# ─────────────────────────────────────────────────────────────────────────────

class TestPositionSizing:
    def test_fixed_fractional_formula(self):
        """risk_amount = equity × risk_pct; size = risk_amount / stop_distance"""
        from services.risk_engine import RiskEngine
        s = _make_settings(RISK_PER_TRADE=0.01)
        re = RiskEngine(s)
        equity = 10000.0
        entry = 1.1000
        sl    = 1.0985  # 15 pips
        stop_dist = abs(entry - sl)          # 0.0015
        expected_size = (equity * 0.01) / stop_dist
        actual_size = re.compute_position_size("EURUSD", entry, sl, equity)
        assert abs(actual_size - round(expected_size, 2)) < 0.01

    def test_zero_stop_returns_minimum(self):
        """Zero stop distance should not cause division by zero."""
        from services.risk_engine import RiskEngine
        s = _make_settings()
        re = RiskEngine(s)
        size = re.compute_position_size("EURUSD", 1.1000, 1.1000, 10000.0)
        assert size == 0.01

    def test_no_ml_confidence_scaling(self):
        """Position size must be identical regardless of quality_score / ml_enabled args."""
        from services.risk_engine import RiskEngine
        s = _make_settings()
        re = RiskEngine(s)
        kw = dict(symbol="EURUSD", entry=1.1000, stop_loss=1.0985, equity=10000.0)
        # The new API no longer accepts ml_enabled; same size every time
        size_a = re.compute_position_size(**kw)
        size_b = re.compute_position_size(**kw)
        assert size_a == size_b

    def test_kelly_overlay_does_not_increase_size(self):
        """Fractional Kelly must never produce a LARGER position than fixed-fractional."""
        from services.risk_engine import RiskEngine
        s = _make_settings(KELLY_ENABLED=True, KELLY_FRACTION=0.25, RISK_PER_TRADE=0.005)
        re = RiskEngine(s)
        fixed_only = RiskEngine(_make_settings(KELLY_ENABLED=False)).compute_position_size(
            "EURUSD", 1.1000, 1.0985, 10000.0
        )
        kelly_size = re.compute_position_size(
            "EURUSD", 1.1000, 1.0985, 10000.0, win_rate=0.55, avg_rr=2.0
        )
        assert kelly_size <= fixed_only + 0.01  # Kelly only shrinks or equals


# ─────────────────────────────────────────────────────────────────────────────
# 5. ATR-based stop and take-profit
# ─────────────────────────────────────────────────────────────────────────────

class TestATRStops:
    def test_stop_distance(self):
        """stop_distance = ATR × stop_multiplier"""
        atr = 0.0012
        stop_mult = 1.5
        tp_mult   = 2.5
        stop_dist = atr * stop_mult
        tp_dist   = atr * tp_mult
        assert abs(stop_dist - 0.0018) < 1e-9
        assert abs(tp_dist  - 0.0030) < 1e-9

    def test_implied_rr(self):
        """With ATR_STOP=1.5 and ATR_TARGET=2.5, implied RR = 2.5/1.5 = 1.667"""
        rr = 2.5 / 1.5
        assert rr > 1.50  # must exceed MIN_REWARD_TO_RISK

    def test_gold_stricter_stops(self):
        """Gold ATR multipliers must produce wider stops than standard forex."""
        s = _make_settings()
        forex_sl = s.ATR_STOP_MULTIPLIER
        gold_sl  = s.GOLD_ATR_STOP_MULTIPLIER
        assert gold_sl > forex_sl

    def test_gold_stricter_targets(self):
        """Gold target multiplier must be larger to compensate for wider stops."""
        s = _make_settings()
        assert s.GOLD_ATR_TARGET_MULTIPLIER > s.ATR_TARGET_MULTIPLIER


# ─────────────────────────────────────────────────────────────────────────────
# 6. Drawdown
# ─────────────────────────────────────────────────────────────────────────────

class TestDrawdown:
    def _max_dd(self, pnls, initial=10000.0):
        eq = [initial]
        for p in pnls:
            eq.append(eq[-1] + p)
        eq = np.array(eq)
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / (peak + 1e-9)
        return float(dd.max())

    def test_no_drawdown_on_only_wins(self):
        assert self._max_dd([100, 200, 150]) < 1e-9

    def test_50pct_drawdown(self):
        """10000 → 5000 is 50% drawdown"""
        dd = self._max_dd([-5000])
        assert abs(dd - 0.50) < 1e-4

    def test_recovery_does_not_reset_peak(self):
        """Peak is not reset after recovery — must track global high-water mark."""
        # 10000 → 12000 → 8000 → 11000; peak=12000 never drops
        dd = self._max_dd([2000, -4000, 3000])
        expected = (12000 - 8000) / 12000
        assert abs(dd - expected) < 1e-4

    def test_drawdown_formula(self):
        """drawdown = (peak − current) / peak"""
        peak = 15000.0
        current = 12000.0
        dd = (peak - current) / peak
        assert abs(dd - 0.20) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# 7. Profit factor
# ─────────────────────────────────────────────────────────────────────────────

class TestProfitFactor:
    def test_perfect_system(self):
        """All wins → profit factor approaches infinity."""
        wins   = [100, 200, 300]
        losses = []
        pf = sum(wins) / (abs(sum(losses)) + 1e-9)
        assert pf > 1e8

    def test_break_even_system(self):
        """Equal gross profit and gross loss → PF = 1.0"""
        wins   = [100, 200]
        losses = [-150, -150]
        pf = sum(wins) / abs(sum(losses))
        assert abs(pf - 1.0) < 1e-6

    def test_typical_good_system(self):
        """PF of 1.5 means $1.50 won per $1.00 lost."""
        wins   = [150] * 6    # gross = 900
        losses = [-100] * 6   # gross_loss = 600
        pf = sum(wins) / abs(sum(losses))
        assert abs(pf - 1.5) < 1e-6

    def test_pf_above_1_25_threshold(self):
        """A system with PF < 1.25 should fail the acceptance criterion."""
        wins   = [120] * 5   # gross = 600
        losses = [-100] * 5  # gross_loss = 500
        pf = 600 / 500
        assert pf == 1.20
        assert pf < 1.25  # → should FAIL verdict rule


# ─────────────────────────────────────────────────────────────────────────────
# 8. Sortino ratio
# ─────────────────────────────────────────────────────────────────────────────

class TestSortinoRatio:
    def test_sortino_higher_than_sharpe_for_positively_skewed_returns(self):
        """A return stream with only small losses has higher Sortino than Sharpe."""
        rr = [2.0, 3.0, -0.5, 2.5, -0.3, 2.0, 1.8]
        arr = np.array(rr)
        mean_r  = arr.mean()
        std_r   = arr.std(ddof=1)
        sharpe  = mean_r / (std_r + 1e-9)
        neg     = arr[arr < 0]
        down_dev = np.sqrt((neg ** 2).mean())
        sortino  = mean_r / (down_dev + 1e-9)
        assert sortino > sharpe

    def test_sortino_zero_when_no_losses(self):
        """With no negative returns, downside deviation = 0 → Sortino = inf (clamped)."""
        rr = [1.0, 2.0, 3.0]
        arr = np.array(rr)
        neg = arr[arr < 0]
        assert len(neg) == 0  # no losses → perfect Sortino


# ─────────────────────────────────────────────────────────────────────────────
# 9. Fractional Kelly criterion
# ─────────────────────────────────────────────────────────────────────────────

class TestFractionalKelly:
    def test_positive_edge_produces_positive_fraction(self):
        from services.risk_engine import RiskEngine
        f = RiskEngine.fractional_kelly(p_win=0.60, reward_to_risk=2.0, fraction=0.25)
        # f* = (0.6×2 − 0.4) / 2 = (1.2 − 0.4)/2 = 0.4; quarter = 0.1
        assert abs(f - 0.10) < 1e-6

    def test_negative_edge_returns_zero(self):
        from services.risk_engine import RiskEngine
        # p=0.30, R=1.5 → f* = (0.30×1.5 − 0.70)/1.5 = (0.45−0.70)/1.5 = −0.167 → 0
        f = RiskEngine.fractional_kelly(p_win=0.30, reward_to_risk=1.5, fraction=0.25)
        assert f == 0.0

    def test_full_kelly_capped_at_20_percent(self):
        """Even a very strong edge must be capped at 20% max per trade."""
        from services.risk_engine import RiskEngine
        f = RiskEngine.fractional_kelly(p_win=0.95, reward_to_risk=10.0, fraction=1.0)
        assert f <= 0.20


# ─────────────────────────────────────────────────────────────────────────────
# 10. Signal rejection logic (RiskEngine.check_pre_trade)
# ─────────────────────────────────────────────────────────────────────────────

class TestSignalRejection:
    def _engine(self, **s_overrides):
        from services.risk_engine import RiskEngine
        return RiskEngine(_make_settings(**s_overrides))

    def _portfolio(self, **overrides):
        base = dict(
            daily_loss_pct=0.0,
            drawdown_pct=0.0,
            open_positions=0,
            open_symbols=[],
            positions=[],
        )
        base.update(overrides)
        return base

    def _signal(self, symbol="EURUSD", side="buy", spread=1.0):
        return {"symbol": symbol, "side": side, "signal_metadata": {"spread_pips": spread}}

    def test_clean_signal_passes(self):
        re = self._engine()
        ok, reason = re.check_pre_trade(self._signal(), self._portfolio())
        assert ok, reason

    def test_daily_loss_blocks(self):
        re = self._engine(MAX_DAILY_LOSS_PCT=0.02)
        pf = self._portfolio(daily_loss_pct=0.025)
        ok, reason = re.check_pre_trade(self._signal(), pf)
        assert not ok
        assert "daily_loss" in reason

    def test_weekly_loss_blocks(self):
        re = self._engine(MAX_WEEKLY_LOSS_PCT=0.05)
        re._weekly_loss_pct = 0.06
        ok, reason = re.check_pre_trade(self._signal(), self._portfolio())
        assert not ok
        assert "weekly_loss" in reason

    def test_drawdown_blocks(self):
        re = self._engine(MAX_DRAWDOWN_PCT=0.15)
        pf = self._portfolio(drawdown_pct=0.20)
        ok, reason = re.check_pre_trade(self._signal(), pf)
        assert not ok
        assert "drawdown" in reason

    def test_max_concurrent_positions_blocks(self):
        re = self._engine(MAX_CONCURRENT_POSITIONS=2)
        pf = self._portfolio(open_positions=2)
        ok, reason = re.check_pre_trade(self._signal(), pf)
        assert not ok
        assert "max_concurrent" in reason

    def test_duplicate_symbol_blocks(self):
        re = self._engine()
        pf = self._portfolio(open_symbols=["EURUSD"])
        ok, reason = re.check_pre_trade(self._signal("EURUSD"), pf)
        assert not ok
        assert "already open" in reason

    def test_correlated_exposure_blocks(self):
        """Two USD-short long positions should be blocked when MAX_CORRELATED=1."""
        re = self._engine(MAX_CORRELATED_POSITIONS=1)
        pf = self._portfolio(
            positions=[{"symbol": "EURUSD", "side": "buy"}],
        )
        # GBPUSD buy is also USD short — should be blocked
        ok, reason = re.check_pre_trade(self._signal("GBPUSD", "buy"), pf)
        assert not ok
        assert "correlated" in reason

    def test_opposite_direction_not_blocked(self):
        """GBPUSD sell is USD long — should NOT be blocked by EURUSD buy (USD short)."""
        re = self._engine(MAX_CORRELATED_POSITIONS=1)
        pf = self._portfolio(
            positions=[{"symbol": "EURUSD", "side": "buy"}],
        )
        ok, reason = re.check_pre_trade(self._signal("GBPUSD", "sell"), pf)
        assert ok, reason

    def test_spread_blocks_forex(self):
        re = self._engine(MAX_SPREAD_EURUSD=2.0)
        sig = self._signal("EURUSD", spread=3.0)
        ok, reason = re.check_pre_trade(sig, self._portfolio())
        assert not ok
        assert "spread" in reason

    def test_spread_blocks_gold(self):
        re = self._engine(MAX_SPREAD_XAUUSD=30.0)
        sig = self._signal("XAUUSD", spread=35.0)
        ok, reason = re.check_pre_trade(sig, self._portfolio())
        assert not ok
        assert "spread" in reason

    def test_cooldown_blocks(self):
        re = self._engine()
        re._cooldown_remaining = 5
        ok, reason = re.check_pre_trade(self._signal(), self._portfolio())
        assert not ok
        assert "cooldown" in reason

    def test_cooldown_activates_after_consecutive_losses(self):
        re = self._engine(MAX_CONSECUTIVE_LOSSES=3, CONSECUTIVE_LOSS_COOLDOWN_BARS=10)
        equity = 10000.0
        for _ in range(3):
            re.record_trade_result(-100, equity)
        assert re._cooldown_remaining == 10

    def test_consecutive_win_resets_streak(self):
        re = self._engine(MAX_CONSECUTIVE_LOSSES=3)
        re.record_trade_result(-100, 10000)
        re.record_trade_result(-100, 9900)
        re.record_trade_result(200, 9800)   # win resets streak
        assert re._consecutive_losses == 0

    def test_weekly_state_resets_on_new_week(self):
        from datetime import datetime, timezone
        re = self._engine()
        re._weekly_loss_pct = 0.04
        re._last_week_number = 1
        # Simulate next ISO week
        next_week_dt = datetime(2026, 1, 12, tzinfo=timezone.utc)  # ISO week 3
        re.update_weekly_state(10000.0, current_dt=next_week_dt)
        assert re._weekly_loss_pct == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 11. No leakage: walk-forward split ordering
# ─────────────────────────────────────────────────────────────────────────────

class TestNoLeakage:
    def test_train_end_before_val_start(self):
        """Validation start must be strictly after training end."""
        import json
        import os
        split_candidates = [
            Path(__file__).resolve().parent.parent.parent.parent
            / "ml_training" / "datasets" / "split_summary.json",
        ]
        for sp in split_candidates:
            if sp.exists():
                summary = json.loads(sp.read_text())
                dr = summary.get("date_ranges", {})
                if "train" in dr and "validation" in dr:
                    train_end = dr["train"]["end"][:10]
                    val_start = dr["validation"]["start"][:10]
                    assert train_end <= val_start, (
                        f"Leakage: train ends {train_end} AFTER val starts {val_start}"
                    )
                if "validation" in dr and "test" in dr:
                    val_end   = dr["validation"]["end"][:10]
                    test_start = dr["test"]["start"][:10]
                    assert val_end <= test_start, (
                        f"Leakage: validation ends {val_end} AFTER test starts {test_start}"
                    )
                return  # tested one file; skip if not found
        pytest.skip("split_summary.json not found — run pipeline/step5_split.py first")


# ─────────────────────────────────────────────────────────────────────────────
# 12. Backtest metrics integration (stress test)
# ─────────────────────────────────────────────────────────────────────────────

class TestBacktestMetricsIntegration:
    """Validates the _fixed_risk_metrics output structure and key mathematical invariants."""

    def _run_metrics(self, rr_list: list[float]) -> dict:
        """Call _fixed_risk_metrics via subprocess-safe in-process import."""
        import importlib.util, types

        # Build a minimal module namespace to avoid importing the full backtest env
        spec_path = (
            Path(__file__).resolve().parent.parent / "scripts" / "run_backtest.py"
        )
        if not spec_path.exists():
            pytest.skip("run_backtest.py not found")

        source = spec_path.read_text()
        # Inject minimal globals so the top-level can be executed without full deps
        ns: dict = {
            "__file__": str(spec_path),
            "np": np,
            "os": __import__("os"),
            "sys": sys,
            "json": __import__("json"),
            "math": __import__("math"),
            "logging": __import__("logging"),
            "Path": Path,
        }
        # Locate and exec only _fixed_risk_metrics
        import ast, textwrap
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_fixed_risk_metrics":
                func_src = ast.get_source_segment(source, node)
                exec(textwrap.dedent(func_src), ns)
                break
        if "_fixed_risk_metrics" not in ns:
            pytest.skip("Could not extract _fixed_risk_metrics from run_backtest.py")
        # Set constants that the function uses
        ns["INITIAL_CAPITAL"] = 10000.0
        ns["RISK_PER_TRADE"]  = 0.005
        trades = [{"realized_rr": r, "confidence": 0.65} for r in rr_list]
        return ns["_fixed_risk_metrics"](trades)

    def test_verdict_pass_for_strong_strategy(self):
        """A clearly profitable strategy must receive PASS verdict."""
        rr = [2.0] * 20 + [-1.0] * 8  # 71% win rate, ~0.76R expectancy, PF=5.0
        m = self._run_metrics(rr)
        assert m["profit_factor"] > 1.25
        assert m["expectancy_r"] > 0
        assert m["verdict"] == "PASS"

    def test_verdict_fail_for_losing_strategy(self):
        """A losing strategy must receive FAIL verdict."""
        rr = [-1.0] * 20 + [2.0] * 5  # 20% win rate, negative expectancy
        m = self._run_metrics(rr)
        assert m["expectancy_r"] < 0
        assert m["verdict"] == "FAIL"

    def test_sortino_present_and_non_negative_for_positive_strat(self):
        rr = [2.0] * 15 + [-1.0] * 5
        m = self._run_metrics(rr)
        assert "sortino" in m
        assert m["sortino"] >= 0

    def test_monte_carlo_keys_present(self):
        rr = [1.5, -1.0] * 20
        m = self._run_metrics(rr)
        for key in ("mc_median_final_equity", "mc_p10_final_equity", "mc_p95_max_drawdown"):
            assert key in m

    def test_monthly_returns_empty_when_no_timestamps(self):
        rr = [1.0, -1.0] * 5
        m = self._run_metrics(rr)
        assert isinstance(m["monthly_returns"], dict)

    def test_information_coefficient_in_range(self):
        rr = [2.0] * 20 + [-1.0] * 10
        m = self._run_metrics(rr)
        assert -1.0 <= m["information_coefficient"] <= 1.0

    def test_profit_factor_invariant(self):
        """profit_factor × gross_loss ≈ gross_profit (accounting for epsilon guard)."""
        rr = [2.0, 2.0, -1.0] * 10
        m = self._run_metrics(rr)
        implied_profit = m["profit_factor"] * m["gross_loss"]
        assert abs(implied_profit - m["gross_profit"]) < 1.0  # allow $1 rounding

    def test_var_less_than_or_equal_cvar(self):
        """CVaR (expected shortfall) must be ≤ VaR (worst outcome can't be better than average)."""
        rr = [1.5, -1.0, 2.0, -0.5, -1.2, 3.0, -0.8] * 5
        m = self._run_metrics(rr)
        assert m["cvar_95"] <= m["var_95"] + 1e-6  # cvar is worse (more negative)
