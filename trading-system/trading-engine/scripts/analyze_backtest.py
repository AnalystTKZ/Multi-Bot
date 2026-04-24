"""
Backtest diagnostics: reads the most recent (or specified) backtest JSON and
outputs critical health checks to stdout + logs/backtest_diagnostics.csv.

Usage:
    python scripts/analyze_backtest.py
    python scripts/analyze_backtest.py --file backtest_results/backtest_20260418_123456.json
    python scripts/analyze_backtest.py --min-trades 5

Checks:
  1. Trade frequency per symbol per day (overtrading / undertrading)
  2. Regime distribution vs trade distribution (collapse detection)
  3. EV calibration — Pearson + Spearman + top-20% + regime breakdown
  4. Direction confidence calibration (GRU bins vs win rate, ECE)
  5. GRU↔EV consistency (corr(p_bull, ev_pred))
"""

import argparse
import json
import sys
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_ENGINE_DIR  = Path(__file__).resolve().parent.parent
_RESULTS_DIR = _ENGINE_DIR / "backtest_results"
_LOGS_DIR    = _ENGINE_DIR / "logs"

_MIN_TRADES_PER_DAY  = 0.05
_MAX_TRADES_PER_DAY  = 1.5
_REGIME_CONC_WARN    = 0.70
_MIN_EV_PEARSON      = 0.10
_MIN_EV_SPEARMAN     = 0.15
_CALIBRATION_MAX_ERR = 0.15
_MIN_GRU_EV_CORR     = 0.10   # corr(p_bull/p_bear, ev_pred) below this → architecture misaligned


def _latest_json() -> Path | None:
    if not _RESULTS_DIR.exists():
        return None
    jsons = sorted(_RESULTS_DIR.glob("backtest_*.json"))
    return jsons[-1] if jsons else None


def _load(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    trades = data.get("trade_log", [])
    if not trades:
        for v in data.get("results", {}).values():
            if isinstance(v, dict) and v.get("trade_log"):
                trades.extend(v["trade_log"])
    return trades


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation — more appropriate than Pearson for trading outcomes."""
    n = len(x)
    if n < 3:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


# ── Check 1: Trade frequency ─────────────────────────────────────────────────

def check_frequency(trades: list[dict], min_trades: int) -> dict:
    per_symbol: dict[str, list[str]] = defaultdict(list)
    for t in trades:
        per_symbol[t["symbol"]].append(t["timestamp"][:10])

    results = {}
    flags   = []
    for sym, dates in sorted(per_symbol.items()):
        n_trades = len(dates)
        n_days   = len(set(dates))
        rate     = n_trades / (n_days + 1e-9)
        status   = "OK"
        if rate > _MAX_TRADES_PER_DAY:
            status = "OVERTRADE"
            flags.append(f"{sym}: {rate:.2f}/day (>{_MAX_TRADES_PER_DAY})")
        elif rate < _MIN_TRADES_PER_DAY and n_trades >= min_trades:
            status = "UNDERTRADE"
            flags.append(f"{sym}: {rate:.2f}/day (<{_MIN_TRADES_PER_DAY})")
        results[sym] = {
            "trades": n_trades, "active_days": n_days,
            "rate_per_day": round(rate, 3), "status": status,
        }
    return {"by_symbol": results, "flags": flags}


# ── Check 2: Regime distribution ─────────────────────────────────────────────

def check_regime_distribution(trades: list[dict]) -> dict:
    regime_counts: dict[str, int]        = defaultdict(int)
    regime_wins:   dict[str, list[int]]  = defaultdict(list)
    regime_evs:    dict[str, list[float]]= defaultdict(list)
    for t in trades:
        r = t.get("regime", "UNKNOWN")
        regime_counts[r] += 1
        regime_wins[r].append(1 if t["pnl"] > 0 else 0)
        regime_evs[r].append(float(t.get("ev", 0.0)))

    total = sum(regime_counts.values()) or 1
    distribution = {}
    for r, c in sorted(regime_counts.items()):
        wins = regime_wins[r]
        evs  = regime_evs[r]
        distribution[r] = {
            "count":    c,
            "pct":      round(c / total * 100, 1),
            "win_rate": round(float(np.mean(wins)), 3) if wins else None,
            "avg_ev":   round(float(np.mean(evs)),  3) if evs  else None,
        }

    flags = []
    dominant = max(regime_counts, key=regime_counts.get)
    if regime_counts[dominant] / total >= _REGIME_CONC_WARN:
        flags.append(
            f"{dominant} = {regime_counts[dominant]/total*100:.0f}% of trades — regime collapse?"
        )
    _EXPECTED_REGIMES = {
        "BIAS_UP", "BIAS_DOWN", "BIAS_NEUTRAL",
        "TRENDING", "RANGING", "CONSOLIDATING", "VOLATILE",
    }
    missing = [r for r in sorted(_EXPECTED_REGIMES) if r not in regime_counts]
    if missing:
        flags.append(f"Regimes never traded: {missing}")

    return {"distribution": distribution, "dominant": dominant, "flags": flags}


# ── Check 3: EV calibration (Pearson + Spearman + top-20% + per-regime) ──────

def check_ev_calibration(trades: list[dict], min_trades: int) -> dict:
    ev_vals  = [float(t.get("ev", 0.0))          for t in trades]
    rr_vals  = [float(t.get("realized_rr", 0.0)) for t in trades]
    regimes  = [t.get("regime", "UNKNOWN")        for t in trades]

    if len(ev_vals) < min_trades:
        return {"error": f"Insufficient trades ({len(ev_vals)}) for EV calibration", "flags": []}

    ev_arr = np.array(ev_vals)
    rr_arr = np.array(rr_vals)

    pearson  = float(np.corrcoef(ev_arr, rr_arr)[0, 1]) if len(ev_arr) > 1 else 0.0
    spearman = _spearman(ev_arr, rr_arr)

    # Quartile buckets
    quartiles = np.percentile(ev_arr, [25, 50, 75])
    labels    = ["Q1 (low EV)", "Q2", "Q3", "Q4 (high EV)"]
    edges     = [-np.inf, quartiles[0], quartiles[1], quartiles[2], np.inf]
    buckets   = {}
    for k, label in enumerate(labels):
        mask = (ev_arr >= edges[k]) & (ev_arr < edges[k + 1])
        s_ev = ev_arr[mask]
        s_rr = rr_arr[mask]
        buckets[label] = {
            "n":           int(mask.sum()),
            "avg_ev_pred": round(float(s_ev.mean()), 3) if len(s_ev) else None,
            "avg_rr_real": round(float(s_rr.mean()), 3) if len(s_rr) else None,
            "win_rate":    round(float((s_rr > 0).mean()), 3) if len(s_rr) else None,
        }

    # Top-20% EV trades: the subset the system will lean on most
    thresh_80 = np.percentile(ev_arr, 80)
    top_mask  = ev_arr >= thresh_80
    top20     = {
        "n":           int(top_mask.sum()),
        "avg_ev_pred": round(float(ev_arr[top_mask].mean()), 3) if top_mask.any() else None,
        "avg_rr_real": round(float(rr_arr[top_mask].mean()), 3) if top_mask.any() else None,
        "win_rate":    round(float((rr_arr[top_mask] > 0).mean()), 3) if top_mask.any() else None,
    }

    # Per-regime EV correlation
    regime_breakdown = {}
    for r in sorted(set(regimes)):
        idx = [j for j, reg in enumerate(regimes) if reg == r]
        if len(idx) < 5:
            continue
        r_ev = ev_arr[idx]
        r_rr = rr_arr[idx]
        regime_breakdown[r] = {
            "n":        len(idx),
            "pearson":  round(float(np.corrcoef(r_ev, r_rr)[0, 1]), 4) if len(idx) > 2 else None,
            "spearman": round(_spearman(r_ev, r_rr), 4),
            "win_rate": round(float((r_rr > 0).mean()), 3),
            "avg_ev":   round(float(r_ev.mean()), 3),
        }

    flags = []
    if pearson < _MIN_EV_PEARSON:
        flags.append(
            f"EV↔RR Pearson={pearson:.3f} < {_MIN_EV_PEARSON} — EV model weak, check training labels"
        )
    if spearman < _MIN_EV_SPEARMAN:
        flags.append(
            f"EV↔RR Spearman={spearman:.3f} < {_MIN_EV_SPEARMAN} — EV rankings don't predict outcomes"
        )
    q1_rr = buckets["Q1 (low EV)"]["avg_rr_real"]
    q4_rr = buckets["Q4 (high EV)"]["avg_rr_real"]
    if q1_rr is not None and q4_rr is not None and q4_rr <= q1_rr:
        flags.append(
            f"Non-monotonic bins: Q4 avg_rr={q4_rr:.3f} ≤ Q1 avg_rr={q1_rr:.3f} — EV not predictive"
        )
    if top20["win_rate"] is not None and top20["win_rate"] < 0.50:
        flags.append(
            f"Top-20% EV trades win_rate={top20['win_rate']:.1%} — high-EV selection not working"
        )
    for r, rb in regime_breakdown.items():
        if rb["spearman"] is not None and rb["spearman"] < 0.05 and rb["n"] >= 20:
            flags.append(
                f"EV↔RR Spearman in {r} = {rb['spearman']:.3f} — EV useless in this regime"
            )

    return {
        "pearson_corr":     round(pearson, 4),
        "spearman_corr":    round(spearman, 4),
        "quartile_buckets": buckets,
        "top20_ev":         top20,
        "regime_breakdown": regime_breakdown,
        "flags":            flags,
    }


# ── Check 4: Confidence calibration ──────────────────────────────────────────

def check_confidence_calibration(trades: list[dict], min_trades: int, n_bins: int = 5) -> dict:
    confidences = []
    outcomes    = []
    for t in trades:
        side = t.get("side", "buy")
        conf = float(t.get("p_bull", 0.5) if side == "buy" else t.get("p_bear", 0.5))
        confidences.append(conf)
        outcomes.append(1 if t["pnl"] > 0 else 0)

    if len(confidences) < min_trades:
        return {"error": f"Insufficient trades ({len(confidences)}) for calibration", "flags": []}

    conf_arr = np.array(confidences)
    out_arr  = np.array(outcomes)
    edges    = np.linspace(conf_arr.min(), conf_arr.max(), n_bins + 1)

    bins  = {}
    flags = []
    for k in range(n_bins):
        lo, hi = edges[k], edges[k + 1]
        mask = (conf_arr >= lo) & (conf_arr <= hi)
        n    = int(mask.sum())
        if n == 0:
            continue
        mid   = round(float((lo + hi) / 2), 3)
        wr    = float(out_arr[mask].mean())
        error = abs(mid - wr)
        label = f"[{lo:.2f}-{hi:.2f}]"
        bins[label] = {"n": n, "midpoint": mid, "win_rate": round(wr, 3), "error": round(error, 3)}
        if error > _CALIBRATION_MAX_ERR:
            flags.append(
                f"Bin {label}: midpoint={mid:.2f} win_rate={wr:.2f} "
                f"(err={error:.2f} > {_CALIBRATION_MAX_ERR}) — GRU miscalibrated"
            )

    total_n = len(conf_arr)
    ece = sum(b["n"] / total_n * b["error"] for b in bins.values())

    # Monotonicity: each bin's win_rate should rise with midpoint
    wr_seq = [b["win_rate"] for b in bins.values()]
    if len(wr_seq) >= 3 and not all(wr_seq[i] <= wr_seq[i+1] for i in range(len(wr_seq)-1)):
        flags.append("Win rate non-monotonic across confidence bins — GRU confidence unreliable")

    return {"bins": bins, "ece": round(ece, 4), "flags": flags}


# ── Check 5: GRU↔EV consistency ──────────────────────────────────────────────

def check_gru_ev_consistency(trades: list[dict], min_trades: int) -> dict:
    """
    Correlation between GRU direction confidence and predicted EV.
    If GRU says p_bull=0.8 but EV=0.05, the two models disagree — likely
    architecture misalignment or training distribution mismatch.
    """
    confidences = []
    evs         = []
    for t in trades:
        side = t.get("side", "buy")
        conf = float(t.get("p_bull", 0.5) if side == "buy" else t.get("p_bear", 0.5))
        ev   = float(t.get("ev", 0.0))
        confidences.append(conf)
        evs.append(ev)

    if len(confidences) < min_trades:
        return {"error": f"Insufficient trades ({len(confidences)}) for GRU↔EV check", "flags": []}

    c_arr = np.array(confidences)
    e_arr = np.array(evs)

    pearson  = float(np.corrcoef(c_arr, e_arr)[0, 1]) if len(c_arr) > 1 else 0.0
    spearman = _spearman(c_arr, e_arr)

    # Quadrant analysis: high_conf vs low_conf × high_ev vs low_ev
    conf_med = np.median(c_arr)
    ev_med   = np.median(e_arr)
    hc_he = int(((c_arr >= conf_med) & (e_arr >= ev_med)).sum())
    hc_le = int(((c_arr >= conf_med) & (e_arr <  ev_med)).sum())
    lc_he = int(((c_arr <  conf_med) & (e_arr >= ev_med)).sum())
    lc_le = int(((c_arr <  conf_med) & (e_arr <  ev_med)).sum())
    total  = len(c_arr)
    agree_pct = round((hc_he + lc_le) / total * 100, 1)  # both high or both low

    flags = []
    if pearson < _MIN_GRU_EV_CORR:
        flags.append(
            f"GRU↔EV Pearson={pearson:.3f} < {_MIN_GRU_EV_CORR} — "
            f"direction model and EV model disagree (architecture misaligned?)"
        )
    if agree_pct < 55:
        flags.append(
            f"GRU and EV agree on only {agree_pct}% of trades — "
            f"models pulling in opposite directions"
        )

    return {
        "pearson":   round(pearson, 4),
        "spearman":  round(spearman, 4),
        "agree_pct": agree_pct,
        "quadrants": {
            "high_conf_high_ev": hc_he,
            "high_conf_low_ev":  hc_le,
            "low_conf_high_ev":  lc_he,
            "low_conf_low_ev":   lc_le,
        },
        "flags": flags,
    }


# ── CSV export ────────────────────────────────────────────────────────────────

def _write_csv(trades: list[dict], out_path: Path) -> None:
    fields = [
        "timestamp", "symbol", "side", "regime", "ev", "realized_rr",
        "p_bull", "p_bear", "rr_ratio", "pnl", "bars_held",
        "session_hour", "regime_duration", "session_weight", "regime_weight",
        "age_weight", "expected_variance", "exit_reason",
    ]
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(",".join(fields) + "\n")
        for t in trades:
            row = [str(t.get(field, "")) for field in fields]
            f.write(",".join(row) + "\n")
    logger.info("Diagnostics CSV → %s (%d rows)", out_path, len(trades))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backtest diagnostics")
    parser.add_argument("--file", help="Path to backtest JSON (default: most recent)")
    parser.add_argument("--min-trades", type=int, default=10)
    args = parser.parse_args()

    path = Path(args.file) if args.file else _latest_json()
    if path is None or not path.exists():
        logger.error("No backtest JSON found. Run run_backtest.py first.")
        sys.exit(1)

    logger.info("Loading: %s", path)
    trades = _load(path)
    logger.info("Total trades: %d", len(trades))

    if not trades:
        logger.error("trade_log is empty.")
        sys.exit(1)

    sep = "─" * 62
    all_flags: list[str] = []

    # ── 1. Trade frequency ────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("CHECK 1: TRADE FREQUENCY  (trades/day/symbol)")
    print(sep)
    freq = check_frequency(trades, args.min_trades)
    for sym, info in freq["by_symbol"].items():
        tag = f"  [{info['status']}]" if info["status"] != "OK" else ""
        print(f"  {sym:<10}  {info['trades']:>5} trades  "
              f"{info['active_days']:>4} days  {info['rate_per_day']:>5.2f}/day{tag}")
    if freq["flags"]:
        for f in freq["flags"]:
            print(f"  ⚠  {f}")
        all_flags += freq["flags"]
    else:
        print("  ✓  All symbols within normal range.")

    # ── 2. Regime distribution ────────────────────────────────────────────────
    print(f"\n{sep}")
    print("CHECK 2: REGIME DISTRIBUTION  (% of trades, WR, avgEV)")
    print(sep)
    regime_check = check_regime_distribution(trades)
    for r, info in regime_check["distribution"].items():
        wr  = f"WR={info['win_rate']:.1%}" if info["win_rate"] is not None else "WR=n/a"
        ev  = f"avgEV={info['avg_ev']:.3f}" if info["avg_ev"] is not None else "avgEV=n/a"
        print(f"  {r:<16}  {info['count']:>5} trades  {info['pct']:>5.1f}%  {wr}  {ev}")
    if regime_check["flags"]:
        for f in regime_check["flags"]:
            print(f"  ⚠  {f}")
        all_flags += regime_check["flags"]

    # ── 3. EV calibration ─────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("CHECK 3: EV PREDICTED vs REALIZED RR")
    print(sep)
    ev_check = check_ev_calibration(trades, args.min_trades)
    if "error" in ev_check:
        print(f"  {ev_check['error']}")
    else:
        print(f"  Pearson  = {ev_check['pearson_corr']:+.4f}   "
              f"Spearman = {ev_check['spearman_corr']:+.4f}")
        print(f"\n  {'Bucket':<18}  {'N':>5}  {'AvgEV':>8}  {'AvgRR':>8}  {'WinRate':>8}")
        for label, b in ev_check["quartile_buckets"].items():
            av = f"{b['avg_ev_pred']:.3f}" if b["avg_ev_pred"] is not None else "  n/a"
            ar = f"{b['avg_rr_real']:.3f}" if b["avg_rr_real"] is not None else "  n/a"
            wr = f"{b['win_rate']:.1%}"    if b["win_rate"]    is not None else "  n/a"
            print(f"  {label:<18}  {b['n']:>5}  {av:>8}  {ar:>8}  {wr:>8}")
        t20 = ev_check["top20_ev"]
        print(f"\n  Top-20% EV trades: n={t20['n']}  "
              f"avgEV={t20['avg_ev_pred']}  avgRR={t20['avg_rr_real']}  "
              f"WR={t20['win_rate']:.1%}" if t20["win_rate"] is not None else "  Top-20%: n/a")
        if ev_check["regime_breakdown"]:
            print(f"\n  Per-regime EV↔RR correlation:")
            print(f"  {'Regime':<16}  {'N':>5}  {'Pearson':>9}  {'Spearman':>9}  {'WR':>7}  {'AvgEV':>8}")
            for r, rb in ev_check["regime_breakdown"].items():
                p = f"{rb['pearson']:+.4f}"   if rb["pearson"]   is not None else "   n/a"
                s = f"{rb['spearman']:+.4f}"  if rb["spearman"]  is not None else "   n/a"
                print(f"  {r:<16}  {rb['n']:>5}  {p:>9}  {s:>9}  "
                      f"{rb['win_rate']:>6.1%}  {rb['avg_ev']:>8.3f}")
        if ev_check["flags"]:
            for f in ev_check["flags"]:
                print(f"  ⚠  {f}")
            all_flags += ev_check["flags"]
        else:
            print("  ✓  EV model is predictive.")

    # ── 4. Confidence calibration ─────────────────────────────────────────────
    print(f"\n{sep}")
    print("CHECK 4: GRU CONFIDENCE CALIBRATION  (p_bull/bear vs win rate)")
    print(sep)
    cal = check_confidence_calibration(trades, args.min_trades)
    if "error" in cal:
        print(f"  {cal['error']}")
    else:
        print(f"  ECE = {cal['ece']:.4f}  (target < 0.10)")
        print(f"  {'Bin':<16}  {'N':>5}  {'Midpoint':>9}  {'WinRate':>9}  {'Error':>7}")
        for label, b in cal["bins"].items():
            print(f"  {label:<16}  {b['n']:>5}  {b['midpoint']:>9.3f}  "
                  f"{b['win_rate']:>9.3f}  {b['error']:>7.3f}")
        if cal["flags"]:
            for f in cal["flags"]:
                print(f"  ⚠  {f}")
            all_flags += cal["flags"]
        else:
            print("  ✓  GRU confidence well-calibrated.")

    # ── 5. GRU↔EV consistency ─────────────────────────────────────────────────
    print(f"\n{sep}")
    print("CHECK 5: GRU ↔ EV MODEL CONSISTENCY  (direction × value agreement)")
    print(sep)
    gru_ev = check_gru_ev_consistency(trades, args.min_trades)
    if "error" in gru_ev:
        print(f"  {gru_ev['error']}")
    else:
        q = gru_ev["quadrants"]
        print(f"  Pearson={gru_ev['pearson']:+.4f}  Spearman={gru_ev['spearman']:+.4f}  "
              f"Agree={gru_ev['agree_pct']:.0f}%")
        print(f"\n  Quadrants  (conf_threshold=median, ev_threshold=median):")
        print(f"  high_conf + high_ev: {q['high_conf_high_ev']:>5}  ← ideal")
        print(f"  high_conf + low_ev:  {q['high_conf_low_ev']:>5}  ← GRU overconfident")
        print(f"  low_conf  + high_ev: {q['low_conf_high_ev']:>5}  ← EV optimistic")
        print(f"  low_conf  + low_ev:  {q['low_conf_low_ev']:>5}  ← correct abstention")
        if gru_ev["flags"]:
            for f in gru_ev["flags"]:
                print(f"  ⚠  {f}")
            all_flags += gru_ev["flags"]
        else:
            print("  ✓  GRU and EV models aligned.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    if all_flags:
        print(f"SUMMARY — {len(all_flags)} flag(s):")
        for f in all_flags:
            print(f"  ⚠  {f}")
    else:
        print("SUMMARY — all checks passed.")
    print(sep)

    _write_csv(trades, _LOGS_DIR / "backtest_diagnostics.csv")


if __name__ == "__main__":
    sys.path.insert(0, str(_ENGINE_DIR))
    main()
