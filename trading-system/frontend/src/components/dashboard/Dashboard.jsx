import { useCallback, useEffect, useMemo, useState } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import { useNavigate } from 'react-router-dom'
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  Analytics as AnalyticsIcon,
  AutoGraph as AutoGraphIcon,
  Bolt as BoltIcon,
  Download as DownloadIcon,
  History as HistoryIcon,
  Psychology as PsychologyIcon,
  Refresh as RefreshIcon,
  Shield as ShieldIcon,
} from '@mui/icons-material'
import analyticsService from '@services/analyticsService'
import { fetchOpenPositions, closePosition } from '@store/slices/positionsSlice'
import { fetchAllTraders } from '@store/slices/tradersSlice'
import { formatCurrency } from '@utils/formatters'

const CHART_COLORS = {
  gain: '#26e0b8',
  loss: '#ef4444',
  warn: '#f59e0b',
  info: '#60a5fa',
  muted: '#64748b',
}

const chartTooltip = {
  contentStyle: {
    background: '#0f1421',
    border: '1px solid #232b42',
    borderRadius: 6,
    color: '#e8ecf5',
    fontSize: 12,
  },
  labelStyle: { color: '#aab1c4' },
}

const asNumber = (value, fallback = 0) => {
  const numeric = Number(value)
  return Number.isFinite(numeric) ? numeric : fallback
}

const asPercent = (value, decimals = 1) => {
  const numeric = asNumber(value)
  const pct = Math.abs(numeric) <= 1 ? numeric * 100 : numeric
  return `${pct.toFixed(decimals)}%`
}

const formatSignedCurrency = (value) => {
  const numeric = asNumber(value)
  const formatted = formatCurrency(Math.abs(numeric))
  return `${numeric >= 0 ? '+' : '-'}${formatted}`
}

const compactCurrency = (value) => {
  const numeric = asNumber(value)
  if (Math.abs(numeric) >= 1_000_000) return `$${(numeric / 1_000_000).toFixed(1)}M`
  if (Math.abs(numeric) >= 1_000) return `$${Math.round(numeric / 1_000)}k`
  return `$${Math.round(numeric)}`
}

const timeLabel = (value) => {
  if (!value) return '--'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '--'
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

const ageLabel = (value) => {
  if (!value) return '--'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '--'
  const seconds = Math.max(0, Math.floor((Date.now() - date.getTime()) / 1000))
  if (seconds < 60) return `${seconds}s`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h`
  return `${Math.floor(seconds / 86400)}d`
}

const positionId = (position, index) =>
  position.id || position.ticket || position.position_id || position.order_id || `${position.symbol || 'position'}-${index}`

const positionSide = (position) =>
  String(position.side || position.type || position.direction || 'long').toLowerCase()

const positionSymbol = (position) => position.symbol || position.sym || position.instrument || 'UNKNOWN'

const positionPnl = (position) => asNumber(position.pnl ?? position.profit ?? position.unrealized_pnl)

const positionQty = (position) => asNumber(position.quantity ?? position.volume ?? position.qty)

const DashboardCard = ({ title, icon: Icon, action, children }) => (
  <section className="dash-card">
    <div className="dash-card-head">
      <h2 className="dash-card-title">
        {Icon ? <Icon sx={{ fontSize: 15 }} /> : null}
        <span>{title}</span>
      </h2>
      {action}
    </div>
    {children}
  </section>
)

const KpiCard = ({ label, value, detail, tone = 'neutral', data = [] }) => (
  <div className="dash-kpi">
    <div className="dash-kpi-label">{label}</div>
    <div className="dash-kpi-value">{value}</div>
    <div className={`dash-kpi-delta ${tone}`}>{detail}</div>
    <div className="dash-kpi-spark">
      <ResponsiveContainer width="100%" height={36}>
        <AreaChart data={data}>
          <Area
            type="monotone"
            dataKey="value"
            stroke={tone === 'loss' ? CHART_COLORS.loss : tone === 'warn' ? CHART_COLORS.warn : CHART_COLORS.gain}
            fill={tone === 'loss' ? CHART_COLORS.loss : tone === 'warn' ? CHART_COLORS.warn : CHART_COLORS.gain}
            fillOpacity={0.1}
            strokeWidth={1.4}
            dot={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  </div>
)

const EmptyState = ({ children }) => <div className="dash-empty">{children}</div>

const Dashboard = () => {
  const dispatch = useDispatch()
  const navigate = useNavigate()
  const positions = useSelector((state) => state.positions.open)
  const traders = useSelector((state) => state.traders.list)
  const traderSignals = useSelector((state) => state.traders.signals)
  const connection = useSelector((state) => state.ui.connectionStatus)
  const [dashboard, setDashboard] = useState(null)
  const [selectedSymbol, setSelectedSymbol] = useState('XAUUSD')
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState(null)

  const loadDashboard = useCallback(async () => {
    setLoading(true)
    try {
      const [overview] = await Promise.all([
        analyticsService.getDashboardOverview(selectedSymbol, 72),
        dispatch(fetchOpenPositions()),
        dispatch(fetchAllTraders()),
      ])
      setDashboard(overview)
      setLoadError(null)
      if (overview.symbol && overview.symbol !== selectedSymbol) {
        setSelectedSymbol(overview.symbol)
      }
    } catch (error) {
      setLoadError(error?.message || error?.detail || 'Failed to load dashboard data')
    } finally {
      setLoading(false)
    }
  }, [dispatch, selectedSymbol])

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      await loadDashboard()
      if (cancelled) return
    }
    load()
    const interval = setInterval(loadDashboard, 15000)
    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [loadDashboard])

  const portfolio = dashboard?.portfolio_overview || {}
  const curve = dashboard?.portfolio_curve || []
  const predictions = dashboard?.prediction_history || []
  const latestPrediction = dashboard?.latest_prediction || {}
  const exposures = dashboard?.exposure_by_symbol || []
  const symbols = dashboard?.symbols?.length ? dashboard.symbols : [selectedSymbol]
  const openPositions = Array.isArray(positions) ? positions : []
  const activeTraders = Array.isArray(traders) ? traders.filter((trader) => trader.status !== 'inactive') : []

  const recentSignals = useMemo(
    () =>
      Object.entries(traderSignals || {})
        .flatMap(([traderId, signals]) =>
          (signals || []).map((signal) => ({
            ...signal,
            traderId,
            timestamp: signal.timestamp || new Date().toISOString(),
          }))
        )
        .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
        .slice(0, 6),
    [traderSignals]
  )

  const totalPnl = asNumber(portfolio.total_pnl)
  const portfolioValue = asNumber(portfolio.portfolio_value || portfolio.balance || portfolio.total_equity)
  const openPnl = openPositions.reduce((sum, position) => sum + positionPnl(position), 0)
  const totalTrades = asNumber(portfolio.total_trades)
  const winRate = asNumber(portfolio.win_rate)
  const drawdown = asNumber(portfolio.max_drawdown || portfolio.drawdown)
  const sparkSource = curve.length ? curve : [{ equity: portfolioValue || 0 }, { equity: (portfolioValue || 0) + totalPnl }]
  const sparkData = sparkSource.map((point, index) => ({ name: index, value: asNumber(point.equity) }))
  const exposureRows = exposures.length
    ? exposures
    : openPositions.map((position) => ({
        label: positionSymbol(position),
        symbol: positionSymbol(position),
        pnl: positionPnl(position),
        volume: positionQty(position),
      }))

  const riskPie = [
    { name: 'Open P&L', value: Math.max(openPnl, 0), color: CHART_COLORS.gain },
    { name: 'Drawdown', value: Math.abs(drawdown), color: CHART_COLORS.warn },
    { name: 'Cash', value: Math.max(asNumber(portfolio.cash_balance), 0), color: CHART_COLORS.info },
  ].filter((item) => item.value > 0)

  return (
    <div className="dash-page">
      <div className="dash-page-head">
        <div>
          <h1>Command Dashboard</h1>
          <p>
            Portfolio pulse, live model state, and backend-routed market intelligence.
            {loading ? ' Refreshing...' : ''}
          </p>
        </div>
        <div className="dash-actions">
          <select value={selectedSymbol} onChange={(event) => setSelectedSymbol(event.target.value)} aria-label="Symbol">
            {symbols.map((symbol) => (
              <option key={symbol} value={symbol}>
                {symbol}
              </option>
            ))}
          </select>
          <button className="dash-btn ghost" type="button" onClick={loadDashboard}>
            <RefreshIcon sx={{ fontSize: 15 }} />
            Refresh
          </button>
          <button className="dash-btn" type="button" onClick={() => navigate('/analytics')}>
            <DownloadIcon sx={{ fontSize: 15 }} />
            Export
          </button>
        </div>
      </div>

      {loadError ? <div className="dash-alert">{loadError}</div> : null}

      <div className="dash-kpi-grid">
        <KpiCard label="Portfolio Value" value={formatCurrency(portfolioValue)} detail="backend equity state" data={sparkData} />
        <KpiCard label="Total P&L" value={formatSignedCurrency(totalPnl)} detail="realized + unrealized" tone={totalPnl >= 0 ? 'gain' : 'loss'} data={sparkData} />
        <KpiCard label="Win Rate" value={asPercent(winRate)} detail={`${totalTrades} recorded trades`} tone="info" data={sparkData} />
        <KpiCard label="Drawdown" value={asPercent(drawdown)} detail="current backend metric" tone="warn" data={sparkData} />
        <KpiCard label="Open Trades" value={String(openPositions.length)} detail={`${activeTraders.length} active traders`} tone="info" data={sparkData} />
        <KpiCard
          label="WebSocket"
          value={connection === 'online' ? 'Online' : 'Offline'}
          detail={connection === 'online' ? 'Live stream' : 'Reconnecting'}
          tone={connection === 'online' ? 'gain' : 'loss'}
          data={sparkData}
        />
      </div>

      <div className="dash-layout">
        <div className="dash-col">
          <DashboardCard
            title="Open Positions"
            icon={BoltIcon}
            action={<span className="dash-muted">{openPositions.length} open · {formatSignedCurrency(openPnl)}</span>}
          >
            <div className="dash-list">
              {openPositions.length === 0 ? (
                <EmptyState>No open positions.</EmptyState>
              ) : (
                openPositions.slice(0, 7).map((position, index) => {
                  const id = positionId(position, index)
                  const pnl = positionPnl(position)
                  const side = positionSide(position)
                  return (
                    <div className="dash-position-row" key={id}>
                      <div>
                        <div className="dash-row-main">
                          <span className="dash-symbol">{positionSymbol(position)}</span>
                          <span className={`dash-chip ${side.includes('short') || side === 'sell' ? 'loss' : 'gain'}`}>{side}</span>
                          <span className="dash-muted">{positionQty(position)} @ {asNumber(position.price_open ?? position.entry_price ?? position.entry).toFixed(2)}</span>
                        </div>
                        <div className="dash-sub">mark {asNumber(position.price_current ?? position.current_price ?? position.mark).toFixed(2)} · trader {position.trader_id || position.strategy_id || 'system'}</div>
                      </div>
                      <div className="dash-row-end">
                        <strong className={pnl >= 0 ? 'gain' : 'loss'}>{formatSignedCurrency(pnl)}</strong>
                        <button className="dash-btn sm" type="button" onClick={() => dispatch(closePosition({ id, reason: 'manual' }))}>
                          Close
                        </button>
                      </div>
                    </div>
                  )
                })
              )}
            </div>
          </DashboardCard>

          <DashboardCard title="Bot Status" icon={PsychologyIcon} action={<span className="dash-chip gain">running</span>}>
            <div className="dash-status-grid">
              {activeTraders.length === 0 ? (
                <EmptyState>No trader allocation data.</EmptyState>
              ) : (
                activeTraders.slice(0, 6).map((trader) => (
                  <div className="dash-status-cell" key={trader.id || trader.strategy_id || trader.name}>
                    <span>{trader.name || trader.strategy_id || 'Trader'}</span>
                    <strong>{asPercent(trader.win_rate)}</strong>
                    <small>{formatSignedCurrency(trader.total_pnl || trader.pnl || 0)}</small>
                  </div>
                ))
              )}
            </div>
          </DashboardCard>

          <DashboardCard title="Signal Alerts" icon={AutoGraphIcon} action={<button className="dash-link" type="button" onClick={() => navigate('/alerts')}>View all</button>}>
            <div className="dash-list compact">
              {recentSignals.length === 0 ? (
                <EmptyState>No recent signal events.</EmptyState>
              ) : (
                recentSignals.map((signal, index) => (
                  <div className="dash-signal-row" key={`${signal.traderId}-${signal.timestamp}-${index}`}>
                    <span className="dash-time">{ageLabel(signal.timestamp)}</span>
                    <div>
                      <div className="dash-row-main">
                        <span className="dash-symbol">{signal.symbol || 'SYSTEM'}</span>
                        <span>{String(signal.direction || signal.side || 'signal').toUpperCase()}</span>
                      </div>
                      <div className="dash-sub">confidence {asPercent(signal.confidence)} · {signal.traderId}</div>
                    </div>
                    <span className="dash-chip info">signal</span>
                  </div>
                ))
              )}
            </div>
          </DashboardCard>
        </div>

        <div className="dash-col wide">
          <DashboardCard
            title="Equity Curve"
            icon={AnalyticsIcon}
            action={<span className={totalPnl >= 0 ? 'gain' : 'loss'}>{formatSignedCurrency(totalPnl)}</span>}
          >
            <div className="dash-chart">
              {curve.length === 0 ? (
                <EmptyState>No portfolio snapshots yet.</EmptyState>
              ) : (
                <ResponsiveContainer width="100%" height={280}>
                  <AreaChart data={curve}>
                    <defs>
                      <linearGradient id="dashEquity" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={CHART_COLORS.gain} stopOpacity={0.35} />
                        <stop offset="95%" stopColor={CHART_COLORS.gain} stopOpacity={0.02} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1a2034" />
                    <XAxis dataKey="timestamp" tickFormatter={timeLabel} tick={{ fill: '#6b7493', fontSize: 11 }} minTickGap={28} />
                    <YAxis tickFormatter={compactCurrency} tick={{ fill: '#6b7493', fontSize: 11 }} width={54} />
                    <Tooltip {...chartTooltip} />
                    <Area type="monotone" dataKey="equity" stroke={CHART_COLORS.gain} fill="url(#dashEquity)" strokeWidth={2} dot={false} />
                  </AreaChart>
                </ResponsiveContainer>
              )}
            </div>
          </DashboardCard>

          <div className="dash-grid-2">
            <DashboardCard title="Prediction Confidence" icon={PsychologyIcon}>
              <div className="dash-chart small">
                {predictions.length === 0 ? (
                  <EmptyState>Prediction traces will render when backend signal metadata is available.</EmptyState>
                ) : (
                  <ResponsiveContainer width="100%" height={230}>
                    <LineChart data={predictions}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1a2034" />
                      <XAxis dataKey="timestamp" tickFormatter={timeLabel} tick={{ fill: '#6b7493', fontSize: 11 }} minTickGap={28} />
                      <YAxis domain={[0, 1]} tickFormatter={(value) => `${Math.round(value * 100)}%`} tick={{ fill: '#6b7493', fontSize: 11 }} />
                      <Tooltip {...chartTooltip} />
                      <Line type="monotone" dataKey="p_bull" name="Bull" stroke={CHART_COLORS.gain} strokeWidth={2} dot={false} />
                      <Line type="monotone" dataKey="p_bear" name="Bear" stroke={CHART_COLORS.loss} strokeWidth={2} dot={false} />
                      <Line type="monotone" dataKey="quality_score" name="Quality" stroke={CHART_COLORS.warn} strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </div>
            </DashboardCard>

            <DashboardCard title="Risk Overview" icon={ShieldIcon}>
              <div className="dash-risk">
                <div className="dash-risk-bars">
                  <div>
                    <span>Predicted move</span>
                    <strong className={asNumber(latestPrediction.predicted_move) >= 0 ? 'gain' : 'loss'}>
                      {formatSignedCurrency(latestPrediction.predicted_move || 0)}
                    </strong>
                  </div>
                  <div>
                    <span>Quality score</span>
                    <strong>{asPercent(latestPrediction.quality_score)}</strong>
                  </div>
                  <div>
                    <span>Current regime</span>
                    <strong>{latestPrediction.regime || dashboard?.current_pair_regime?.regime || 'Unknown'}</strong>
                  </div>
                </div>
                {riskPie.length ? (
                  <ResponsiveContainer width="100%" height={150}>
                    <PieChart>
                      <Pie data={riskPie} dataKey="value" nameKey="name" innerRadius={44} outerRadius={68} paddingAngle={2}>
                        {riskPie.map((entry) => (
                          <Cell key={entry.name} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip {...chartTooltip} />
                    </PieChart>
                  </ResponsiveContainer>
                ) : null}
              </div>
            </DashboardCard>
          </div>

          <DashboardCard title="Exposure By Symbol" icon={HistoryIcon}>
            <div className="dash-chart small">
              {exposureRows.length === 0 ? (
                <EmptyState>Exposure appears after backend position state is populated.</EmptyState>
              ) : (
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={exposureRows.slice(0, 10)}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1a2034" vertical={false} />
                    <XAxis dataKey={(row) => row.symbol || row.label} tick={{ fill: '#6b7493', fontSize: 11 }} />
                    <YAxis tick={{ fill: '#6b7493', fontSize: 11 }} />
                    <Tooltip {...chartTooltip} />
                    <Bar dataKey={(row) => asNumber(row.pnl || row.volume || row.exposure)} radius={[4, 4, 0, 0]}>
                      {exposureRows.slice(0, 10).map((row, index) => (
                        <Cell key={`${row.symbol || row.label}-${index}`} fill={asNumber(row.pnl) < 0 ? CHART_COLORS.loss : CHART_COLORS.gain} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              )}
            </div>
          </DashboardCard>
        </div>
      </div>
    </div>
  )
}

export default Dashboard
