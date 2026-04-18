import api from './api'

const normalizePerformance = (payload = {}) => {
  const metrics = payload.metrics || payload
  const totalTrades = Number(metrics.total_trades || 0)
  const winRate = Number(metrics.win_rate || 0)
  const winningTrades = Math.round(totalTrades * winRate)
  const losingTrades = Math.max(totalTrades - winningTrades, 0)

  return {
    ...payload,
    ...metrics,
    balance: Number(metrics.portfolio_value || metrics.balance || 0),
    portfolio_value: Number(metrics.portfolio_value || metrics.balance || 0),
    net_pnl: Number(metrics.total_pnl || metrics.net_pnl || 0),
    total_pnl: Number(metrics.total_pnl || metrics.net_pnl || 0),
    monthly_pnl: Number(metrics.total_pnl || metrics.monthly_pnl || 0),
    win_rate: winRate,
    total_trades: totalTrades,
    winning_trades: Number(metrics.winning_trades || winningTrades),
    losing_trades: Number(metrics.losing_trades || losingTrades),
    avg_rr: Number(metrics.avg_rr || 0),
    max_drawdown: Number(metrics.max_drawdown || 0),
    profit_factor: Number(metrics.profit_factor || 0),
    sharpe_ratio: Number(metrics.sharpe_ratio || 0),
    by_trader: payload.by_trader || {},
  }
}

const normalizeEquityCurve = (payload = {}) => {
  const curve = payload.equity_curve || payload
  const dates = Array.isArray(curve.dates) ? curve.dates : []
  const values = Array.isArray(curve.equity) ? curve.equity : []

  return dates.map((date, index) => ({
    date,
    equity: Number(values[index] || 0),
  }))
}

const normalizeTrades = (payload = {}) => {
  const trades = Array.isArray(payload) ? payload : payload.trades || []
  return {
    ...payload,
    trades: trades.map((trade) => ({
      ...trade,
      timestamp: trade.timestamp || trade.closed_at || trade.opened_at,
      side: trade.side || trade.type || trade.direction,
      pnl: Number(trade.pnl ?? trade.profit ?? trade.net_pnl ?? 0),
    })),
    total: payload.total ?? trades.length,
  }
}

const normalizeDashboardOverview = (payload = {}) => ({
  symbol: payload.symbol || '',
  symbols: Array.isArray(payload.symbols) ? payload.symbols : [],
  portfolio_overview: payload.portfolio_overview || {},
  portfolio_curve: Array.isArray(payload.portfolio_curve)
    ? payload.portfolio_curve.map((point) => ({
        timestamp: point.timestamp,
        equity: Number(point.equity || 0),
        drawdown: Number(point.drawdown || 0),
      }))
    : [],
  regime_history: Array.isArray(payload.regime_history) ? payload.regime_history : [],
  prediction_history: Array.isArray(payload.prediction_history) ? payload.prediction_history : [],
  latest_prediction: payload.latest_prediction || {},
  pair_regimes: Array.isArray(payload.pair_regimes) ? payload.pair_regimes : [],
  current_pair_regime: payload.current_pair_regime || null,
  exposure_by_symbol: Array.isArray(payload.exposure_by_symbol) ? payload.exposure_by_symbol : [],
  exposure_by_trader: Array.isArray(payload.exposure_by_trader) ? payload.exposure_by_trader : [],
  pattern_snapshot: Array.isArray(payload.pattern_snapshot) ? payload.pattern_snapshot : [],
})

const analyticsService = {
  getPerformance: (period = '30d', traderId = null) => {
    const params = new URLSearchParams({ period })
    if (traderId) params.append('trader_id', traderId)
    return api.get(`/analytics/performance?${params}`).then(normalizePerformance)
  },
  getEquityCurve: (period = '30d', resolution = '1d') =>
    api.get(`/analytics/equity-curve?period=${period}&resolution=${resolution}`).then(normalizeEquityCurve),
  getTrades: (limitOrOpts = 50, offset = 0, traderId = null) => {
    let limit = limitOrOpts
    if (limitOrOpts && typeof limitOrOpts === 'object') {
      limit = limitOrOpts.limit ?? 50
      offset = limitOrOpts.offset ?? 0
      traderId = limitOrOpts.traderId ?? null
    }
    const params = new URLSearchParams({ limit, offset })
    if (traderId) params.append('trader_id', traderId)
    return api.get(`/analytics/trades?${params}`).then(normalizeTrades)
  },
  getMonthlyReturns: () => api.get('/analytics/monthly-returns'),
  getDashboardOverview: (symbol = null, limit = 60) => {
    const params = new URLSearchParams({ limit })
    if (symbol) params.append('symbol', symbol)
    return api.get(`/analytics/dashboard?${params}`).then(normalizeDashboardOverview)
  },
}

export default analyticsService
