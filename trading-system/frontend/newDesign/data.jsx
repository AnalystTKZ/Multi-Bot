(function(){
// data.jsx — mock data + small helpers shared across pages

const fmtUSD = (n, decimals = 2) => {
  const sign = n < 0 ? "-" : "";
  const abs = Math.abs(n);
  return sign + "$" + abs.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
};
const fmtPct = (n, decimals = 2) => (n >= 0 ? "+" : "") + n.toFixed(decimals) + "%";
const fmtNum = (n, decimals = 0) => n.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
const fmtCompact = (n) => {
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return n.toFixed(0);
};
const timeAgo = (d) => {
  const s = Math.floor((Date.now() - d) / 1000);
  if (s < 60) return s + "s";
  if (s < 3600) return Math.floor(s / 60) + "m";
  if (s < 86400) return Math.floor(s / 3600) + "h";
  return Math.floor(s / 86400) + "d";
};

// Seeded random so the page is deterministic between renders/refreshes
function mulberry32(seed) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const KPI = {
  balance: 184_273.48,
  balanceDelta: 1842.13,
  monthlyPnl: 12_845.22,
  monthlyPnlPct: 7.48,
  winRate: 64.2,
  winRateDelta: 1.4,
  maxDrawdown: -8.7,
  maxDrawdownDelta: -0.3,
  activeTrades: 7,
  totalTrades30d: 213,
};

// Single ML trader
const ML_TRADER = {
  id: "ML-1",
  name: "ML Trader",
  strategy: "Hybrid LSTM + RL Policy",
  session: "24/7 (NYSE / Crypto)",
  symbols: ["BTC-USD", "ETH-USD", "SPY", "QQQ", "AAPL", "NVDA", "TSLA", "MSFT"],
  status: "running",
  pnl: 12_845.22,
  winRate: 64.2,
  trades: 213,
  avgRR: 1.84,
  sharpe: 2.31,
  sortino: 3.12,
  exposure: 0.42,
  modelVersion: "v0.7.3-rc2",
  lastTrained: Date.now() - 1000 * 60 * 60 * 8,
};

const POSITIONS = [
  { id: 1, sym: "BTC-USD", side: "long", qty: 0.42, entry: 67_240.5, mark: 68_512.3, pnl: 534.18, pnlPct: 1.89, opened: Date.now() - 1000 * 60 * 47 },
  { id: 2, sym: "NVDA", side: "long", qty: 80, entry: 902.14, mark: 911.6, pnl: 756.8, pnlPct: 1.05, opened: Date.now() - 1000 * 60 * 92 },
  { id: 3, sym: "ETH-USD", side: "short", qty: 4.5, entry: 3540.2, mark: 3502.7, pnl: 168.75, pnlPct: 1.06, opened: Date.now() - 1000 * 60 * 18 },
  { id: 4, sym: "AAPL", side: "long", qty: 120, entry: 211.32, mark: 209.18, pnl: -256.8, pnlPct: -1.01, opened: Date.now() - 1000 * 60 * 134 },
  { id: 5, sym: "TSLA", side: "short", qty: 50, entry: 184.6, mark: 181.42, pnl: 159.0, pnlPct: 1.72, opened: Date.now() - 1000 * 60 * 22 },
  { id: 6, sym: "QQQ", side: "long", qty: 40, entry: 482.18, mark: 484.6, pnl: 96.8, pnlPct: 0.5, opened: Date.now() - 1000 * 60 * 250 },
  { id: 7, sym: "SPY", side: "long", qty: 25, entry: 552.4, mark: 554.12, pnl: 43.0, pnlPct: 0.31, opened: Date.now() - 1000 * 60 * 8 },
];

const SIGNALS = [
  { id: 1, time: Date.now() - 1000 * 60 * 2, kind: "entry", sym: "NVDA", msg: "Entered long position", detail: "LSTM confidence 0.82 · momentum aligned · vol regime stable", severity: "ok" },
  { id: 2, time: Date.now() - 1000 * 60 * 6, kind: "alert", sym: "BTC-USD", msg: "Volatility spike detected", detail: "ATR(14) crossed 2.1× 30d baseline — reducing exposure 25%", severity: "med" },
  { id: 3, time: Date.now() - 1000 * 60 * 14, kind: "exit", sym: "MSFT", msg: "Take-profit triggered", detail: "Closed @ 432.18 · +$412.40 (+1.23%) · R:R 2.1", severity: "ok" },
  { id: 4, time: Date.now() - 1000 * 60 * 23, kind: "news", sym: "AAPL", msg: "Negative news sentiment", detail: "FT: 'Supply chain concerns' · sentiment -0.71 · weighting reduced", severity: "med" },
  { id: 5, time: Date.now() - 1000 * 60 * 41, kind: "alert", sym: "ETH-USD", msg: "Correlation regime shift", detail: "ETH-BTC 30m corr fell to 0.42 (from 0.81)", severity: "low" },
  { id: 6, time: Date.now() - 1000 * 60 * 58, kind: "entry", sym: "TSLA", msg: "Entered short position", detail: "Mean-reversion signal · RSI 78 · pivot rejection", severity: "ok" },
];

const TRADES = (() => {
  const rand = mulberry32(7);
  const symbols = ["BTC-USD", "ETH-USD", "SPY", "QQQ", "AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "META"];
  const arr = [];
  for (let i = 0; i < 60; i++) {
    const sym = symbols[Math.floor(rand() * symbols.length)];
    const side = rand() > 0.5 ? "long" : "short";
    const qty = +(rand() * 100 + 1).toFixed(2);
    const entry = +(80 + rand() * 600).toFixed(2);
    const exit = +(entry * (1 + (rand() - 0.42) * 0.04)).toFixed(2);
    const pnl = +((side === "long" ? exit - entry : entry - exit) * qty).toFixed(2);
    arr.push({
      id: i + 1,
      time: Date.now() - 1000 * 60 * 30 * i - Math.floor(rand() * 60_000),
      sym,
      side,
      qty,
      entry,
      exit,
      pnl,
      rr: +(Math.abs(pnl) / (Math.abs(pnl) + 80 * rand() + 30)).toFixed(2) * 3,
      reason: ["take-profit", "stop-loss", "trailing-stop", "signal-exit", "time-stop"][Math.floor(rand() * 5)],
    });
  }
  return arr;
})();

const ALERTS = [
  { id: 1, time: Date.now() - 1000 * 60 * 3, severity: "high", title: "Drawdown approaching threshold", msg: "Daily DD at 4.8% (threshold 5.0%). Position sizing reduced.", source: "Risk Engine" },
  { id: 2, time: Date.now() - 1000 * 60 * 11, severity: "med", title: "Model confidence dropped", msg: "LSTM ensemble confidence below 0.55 for 22 mins on NVDA.", source: "ML Monitor" },
  { id: 3, time: Date.now() - 1000 * 60 * 24, severity: "low", title: "Latency anomaly", msg: "WebSocket round-trip exceeded 240ms (Coinbase). Auto-recovered.", source: "System Health" },
  { id: 4, time: Date.now() - 1000 * 60 * 67, severity: "med", title: "News sentiment alert", msg: "AAPL: 4 negative items in 30 min. Exposure reduced 15%.", source: "News Monitor" },
  { id: 5, time: Date.now() - 1000 * 60 * 134, severity: "ok", title: "Daily training complete", msg: "Nightly retrain finished. Val loss 0.0231 (-3.4% vs prior).", source: "Training Pipeline" },
  { id: 6, time: Date.now() - 1000 * 60 * 250, severity: "high", title: "Stop-loss triggered", msg: "AAPL long stopped out at 209.18. -$256.80 (-1.01%).", source: "Execution" },
  { id: 7, time: Date.now() - 1000 * 60 * 360, severity: "low", title: "Backtest queued", msg: "Run #2026-04-02-A queued: 6 symbols, 90d.", source: "Backtest" },
];

// Equity curve — 90 daily points trending up with volatility
function makeEquityCurve(days = 90, start = 150_000) {
  const rand = mulberry32(42);
  const arr = [];
  let v = start;
  for (let i = 0; i < days; i++) {
    const drift = 0.0014;
    const vol = 0.012;
    v = v * (1 + drift + (rand() - 0.5) * vol);
    arr.push({
      day: i,
      date: new Date(Date.now() - (days - i) * 86400000),
      value: v,
    });
  }
  return arr;
}
const EQUITY = makeEquityCurve();

// Monthly returns (last 12 months)
const MONTHLY = (() => {
  const rand = mulberry32(11);
  const months = ["May","Jun","Jul","Aug","Sep","Oct","Nov","Dec","Jan","Feb","Mar","Apr"];
  return months.map((m) => ({ month: m, ret: +(((rand() - 0.32) * 14).toFixed(2)) }));
})();

// Per-symbol breakdown
const SYM_BREAKDOWN = [
  { sym: "BTC-USD", trades: 42, winRate: 66.7, pnl: 4_213.5, share: 0.18 },
  { sym: "ETH-USD", trades: 38, winRate: 60.5, pnl: 2_147.2, share: 0.15 },
  { sym: "NVDA", trades: 28, winRate: 71.4, pnl: 3_842.6, share: 0.21 },
  { sym: "SPY", trades: 31, winRate: 58.1, pnl: 1_124.0, share: 0.12 },
  { sym: "AAPL", trades: 24, winRate: 54.2, pnl: -312.8, share: 0.09 },
  { sym: "TSLA", trades: 19, winRate: 63.2, pnl: 1_530.4, share: 0.13 },
  { sym: "MSFT", trades: 16, winRate: 68.8, pnl: 800.3, share: 0.07 },
  { sym: "QQQ", trades: 15, winRate: 60.0, pnl: -500.0, share: 0.05 },
];

// ML models powering the trader
const ML_MODELS = [
  { id: "lstm-price", name: "Price LSTM", role: "Sequence forecast", status: "active", enabled: true, accuracy: 0.624, loss: 0.0231, lastTrain: Date.now() - 1000*60*60*8, params: "2.4M", weights: "lstm_price_v0.7.3.pt" },
  { id: "rl-policy", name: "RL Policy (PPO)", role: "Action selection", status: "active", enabled: true, accuracy: 0.712, loss: 0.184, lastTrain: Date.now() - 1000*60*60*22, params: "860K", weights: "ppo_policy_v0.7.3.pt" },
  { id: "sent-bert", name: "Sentiment BERT", role: "News sentiment", status: "active", enabled: true, accuracy: 0.812, loss: 0.0894, lastTrain: Date.now() - 1000*60*60*72, params: "110M", weights: "sent_bert_v0.6.0.pt" },
  { id: "vol-garch", name: "Volatility GARCH", role: "Vol forecast", status: "warming", enabled: true, accuracy: 0.572, loss: 0.412, lastTrain: Date.now() - 1000*60*60*4, params: "12K", weights: "garch_v0.4.1.pkl" },
  { id: "regime-cls", name: "Regime Classifier", role: "Market regime", status: "active", enabled: false, accuracy: 0.681, loss: 0.341, lastTrain: Date.now() - 1000*60*60*120, params: "340K", weights: "regime_cls_v0.5.2.pt" },
];

// RL agent
const RL_AGENT = {
  episodes: 14_283,
  avgReward: 0.347,
  recentReward: 0.412,
  exploration: 0.06,
  bestEpisode: 17.42,
};

// Backtest results
const BACKTESTS = [
  { id: "2026-04-02-A", date: "2026-04-02", symbols: 8, period: "90d", capital: 100_000, finalPnl: 14_281.4, winRate: 67.2, maxDD: -6.8, sharpe: 2.34, status: "complete" },
  { id: "2026-04-01-B", date: "2026-04-01", symbols: 6, period: "180d", capital: 100_000, finalPnl: 22_412.0, winRate: 62.1, maxDD: -9.2, sharpe: 1.98, status: "complete" },
  { id: "2026-03-30-A", date: "2026-03-30", symbols: 4, period: "30d", capital: 50_000, finalPnl: 1_842.0, winRate: 58.3, maxDD: -3.1, sharpe: 1.42, status: "complete" },
  { id: "2026-03-28-C", date: "2026-03-28", symbols: 10, period: "365d", capital: 250_000, finalPnl: 51_283.0, winRate: 64.0, maxDD: -12.4, sharpe: 1.78, status: "complete" },
  { id: "2026-03-25-B", date: "2026-03-25", symbols: 8, period: "90d", capital: 100_000, finalPnl: -1_240.0, winRate: 51.0, maxDD: -8.4, sharpe: -0.32, status: "complete" },
];

// Risk overview
const RISK = {
  exposure: 0.42,
  exposureLimit: 0.7,
  dailyDD: 0.018,
  dailyDDLimit: 0.05,
  varDay: -1284.0,
  beta: 0.62,
  sharpe: 2.31,
  sortino: 3.12,
  longShort: { long: 0.71, short: 0.29 },
  byClass: [
    { name: "Crypto", value: 0.34, color: "var(--info)" },
    { name: "Tech (US)", value: 0.42, color: "var(--gain)" },
    { name: "Index ETF", value: 0.18, color: "var(--warn)" },
    { name: "Cash", value: 0.06, color: "var(--text-2)" },
  ],
};

// Monitors live feed mock
const MONITOR_FEED = [
  { t: Date.now() - 2_000, src: "MARKET", msg: "BTC-USD tick 68,512.30 (vol+1.8%)" },
  { t: Date.now() - 5_000, src: "SIGNAL", msg: "NVDA: long-entry confidence 0.82" },
  { t: Date.now() - 8_000, src: "EXEC", msg: "Filled 80 NVDA @ 902.14 ($72,171)" },
  { t: Date.now() - 12_000, src: "RISK", msg: "Exposure 42% / limit 70% — OK" },
  { t: Date.now() - 18_000, src: "NEWS", msg: "AAPL: -0.71 sentiment (FT)" },
  { t: Date.now() - 24_000, src: "MARKET", msg: "ETH-USD tick 3,502.70 (vol-0.4%)" },
  { t: Date.now() - 31_000, src: "SIGNAL", msg: "TSLA: short-entry confidence 0.74" },
  { t: Date.now() - 38_000, src: "MARKET", msg: "SPY tick 554.12 (vol+0.2%)" },
  { t: Date.now() - 45_000, src: "ML", msg: "LSTM forecast batch 224 done (38ms)" },
  { t: Date.now() - 51_000, src: "EXEC", msg: "Closed 22 MSFT @ 432.18 (+$412)" },
  { t: Date.now() - 58_000, src: "RISK", msg: "Daily DD 1.8% / limit 5.0%" },
  { t: Date.now() - 63_000, src: "MARKET", msg: "QQQ tick 484.60 (vol+0.7%)" },
];

// Hand-rolled icons (so we don't depend on external libs)
const Icon = ({ name, size = 16, stroke = 1.6 }) => {
  const props = {
    width: size, height: size, viewBox: "0 0 24 24",
    fill: "none", stroke: "currentColor", strokeWidth: stroke,
    strokeLinecap: "round", strokeLinejoin: "round",
  };
  switch (name) {
    case "dashboard": return <svg {...props}><rect x="3" y="3" width="7" height="9"/><rect x="14" y="3" width="7" height="5"/><rect x="14" y="12" width="7" height="9"/><rect x="3" y="16" width="7" height="5"/></svg>;
    case "trader":    return <svg {...props}><circle cx="12" cy="8" r="4"/><path d="M4 21c0-4 4-7 8-7s8 3 8 7"/></svg>;
    case "monitor":   return <svg {...props}><path d="M3 12h4l2-7 4 14 2-7h6"/></svg>;
    case "analytics": return <svg {...props}><path d="M3 3v18h18"/><path d="M7 14l4-4 4 4 5-7"/></svg>;
    case "history":   return <svg {...props}><path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 4v5h5"/><path d="M12 7v5l3 2"/></svg>;
    case "alerts":    return <svg {...props}><path d="M6 8a6 6 0 1 1 12 0c0 7 3 8 3 8H3s3-1 3-8"/><path d="M10 21a2 2 0 0 0 4 0"/></svg>;
    case "backtest":  return <svg {...props}><path d="M5 3h14v18l-7-3-7 3z"/><path d="M9 9h6M9 13h6"/></svg>;
    case "training":  return <svg {...props}><path d="M12 3l9 5-9 5-9-5 9-5z"/><path d="M3 13l9 5 9-5"/></svg>;
    case "ml":        return <svg {...props}><circle cx="6" cy="6" r="2"/><circle cx="6" cy="18" r="2"/><circle cx="18" cy="12" r="2"/><circle cx="12" cy="4" r="2"/><circle cx="12" cy="20" r="2"/><path d="M8 6h2.5M8 18h2.5M14 4l2 6.5M14 20l2-6.5M13.5 5l-1 13"/></svg>;
    case "settings":  return <svg {...props}><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3h0a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8v0a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/></svg>;
    case "chevron-l": return <svg {...props}><polyline points="15 18 9 12 15 6"/></svg>;
    case "chevron-r": return <svg {...props}><polyline points="9 18 15 12 9 6"/></svg>;
    case "chevron-d": return <svg {...props}><polyline points="6 9 12 15 18 9"/></svg>;
    case "search":    return <svg {...props}><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/></svg>;
    case "play":      return <svg {...props}><polygon points="6 4 20 12 6 20 6 4"/></svg>;
    case "pause":     return <svg {...props}><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>;
    case "stop":      return <svg {...props}><rect x="6" y="6" width="12" height="12" rx="1"/></svg>;
    case "x":         return <svg {...props}><path d="M6 6l12 12M6 18L18 6"/></svg>;
    case "plus":      return <svg {...props}><path d="M12 5v14M5 12h14"/></svg>;
    case "refresh":   return <svg {...props}><path d="M3 12a9 9 0 0 1 15.5-6.4L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-15.5 6.4L3 16"/><path d="M3 21v-5h5"/></svg>;
    case "up":        return <svg {...props}><polyline points="18 15 12 9 6 15"/></svg>;
    case "down":      return <svg {...props}><polyline points="6 9 12 15 18 9"/></svg>;
    case "filter":    return <svg {...props}><path d="M3 5h18l-7 9v6l-4-2v-4z"/></svg>;
    case "download":  return <svg {...props}><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><path d="M12 15V3"/></svg>;
    case "lightning": return <svg {...props}><path d="M13 2L3 14h7l-1 8 10-12h-7z"/></svg>;
    case "brain":     return <svg {...props}><path d="M9 4a3 3 0 0 0-3 3v0a3 3 0 0 0-2 5v0a3 3 0 0 0 2 5v0a3 3 0 0 0 3 3"/><path d="M9 4a3 3 0 0 1 3 3v10a3 3 0 0 1-3 3"/><path d="M15 4a3 3 0 0 1 3 3v0a3 3 0 0 1 2 5v0a3 3 0 0 1-2 5v0a3 3 0 0 1-3 3"/><path d="M15 4a3 3 0 0 0-3 3"/></svg>;
    case "shield":    return <svg {...props}><path d="M12 2l9 4v6c0 5-4 9-9 10-5-1-9-5-9-10V6z"/></svg>;
    default: return null;
  }
};

window.DB = {
  fmtUSD, fmtPct, fmtNum, fmtCompact, timeAgo, mulberry32,
  KPI, ML_TRADER, POSITIONS, SIGNALS, TRADES, ALERTS, EQUITY, MONTHLY,
  SYM_BREAKDOWN, ML_MODELS, RL_AGENT, BACKTESTS, RISK, MONITOR_FEED,
  Icon,
};

})();