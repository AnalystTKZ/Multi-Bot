import { useEffect, useState } from 'react'
import {
  Box,
  Button,
  Chip,
  FormControl,
  Grid,
  MenuItem,
  Paper,
  Select,
  Stack,
  Typography,
} from '@mui/material'
import { useNavigate } from 'react-router-dom'
import { useDispatch } from 'react-redux'
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import analyticsService from '@services/analyticsService'
import { fetchOpenPositions } from '@store/slices/positionsSlice'
import { fetchAllTraders } from '@store/slices/tradersSlice'
import { formatCurrency, formatPercent } from '@utils/formatters'
import KeyMetrics from './KeyMetrics'
import OpenPositions from './OpenPositions'
import BotStatus from './BotStatus'
import RiskMetrics from './RiskMetrics'
import SignalFeed from './SignalFeed'
import RecentTrades from './RecentTrades'

const CHART_TOOLTIP_STYLE = {
  contentStyle: {
    background: '#101826',
    border: '1px solid rgba(148,163,184,0.18)',
    borderRadius: '12px',
    fontSize: '0.75rem',
  },
  labelStyle: {
    color: '#cbd5f5',
  },
}

const REGIME_LABELS = {
  '-1': 'Trend Down',
  0: 'Range',
  1: 'Trend Up',
  2: 'Volatile',
}

const SectionHeader = ({ title, subtitle, action, onAction, controls = null }) => (
  <Box
    sx={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: { xs: 'flex-start', md: 'center' },
      gap: 1.5,
      flexWrap: 'wrap',
      mb: 2,
    }}
  >
    <Box>
      <Typography variant="subtitle1" fontWeight={700}>
        {title}
      </Typography>
      {subtitle ? (
        <Typography variant="body2" color="text.secondary">
          {subtitle}
        </Typography>
      ) : null}
    </Box>
    <Stack direction="row" spacing={1} alignItems="center" sx={{ flexWrap: 'wrap' }}>
      {controls}
      {action ? (
        <Button size="small" variant="text" onClick={onAction} sx={{ color: '#26e0b8' }}>
          {action}
        </Button>
      ) : null}
    </Stack>
  </Box>
)

const EmptyState = ({ message, height = 240 }) => (
  <Box
    sx={{
      height,
      display: 'grid',
      placeItems: 'center',
      border: '1px dashed rgba(148,163,184,0.16)',
      borderRadius: 3,
      color: 'text.secondary',
      textAlign: 'center',
      px: 2,
    }}
  >
    <Typography variant="body2">{message}</Typography>
  </Box>
)

const chartTimeLabel = (value) => {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '—'
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

const ModelSnapshot = ({ latestPrediction, patterns }) => {
  const predictedMove = Number(latestPrediction?.predicted_move || 0)
  const predictedMovePct = Number(latestPrediction?.predicted_move_pct || 0)
  const direction = String(latestPrediction?.direction || 'buy').toUpperCase()

  return (
    <Grid container spacing={2}>
      <Grid item xs={12} sm={6}>
        <Stack spacing={1.2}>
          <Box>
            <Typography variant="caption" color="text.secondary">
              Active Bias
            </Typography>
            <Typography variant="h5" fontWeight={700}>
              {direction}
            </Typography>
          </Box>
          <Grid container spacing={1}>
            <Grid item xs={6}>
              <Paper className="theme-panel" sx={{ p: 1.25 }}>
                <Typography variant="caption" color="text.secondary">
                  Bull Prob.
                </Typography>
                <Typography variant="body1" fontWeight={700} sx={{ color: '#26e0b8' }}>
                  {formatPercent(Number(latestPrediction?.p_bull || 0))}
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={6}>
              <Paper className="theme-panel" sx={{ p: 1.25 }}>
                <Typography variant="caption" color="text.secondary">
                  Bear Prob.
                </Typography>
                <Typography variant="body1" fontWeight={700} sx={{ color: '#ef4444' }}>
                  {formatPercent(Number(latestPrediction?.p_bear || 0))}
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={6}>
              <Paper className="theme-panel" sx={{ p: 1.25 }}>
                <Typography variant="caption" color="text.secondary">
                  Quality
                </Typography>
                <Typography variant="body1" fontWeight={700}>
                  {formatPercent(Number(latestPrediction?.quality_score || 0))}
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={6}>
              <Paper className="theme-panel" sx={{ p: 1.25 }}>
                <Typography variant="caption" color="text.secondary">
                  Predicted Move
                </Typography>
                <Typography
                  variant="body1"
                  fontWeight={700}
                  sx={{ color: predictedMove >= 0 ? '#26e0b8' : '#ef4444' }}
                >
                  {predictedMove >= 0 ? '+' : ''}
                  {formatCurrency(predictedMove)}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {predictedMove >= 0 ? '+' : ''}
                  {formatPercent(predictedMovePct)}
                </Typography>
              </Paper>
            </Grid>
          </Grid>
        </Stack>
      </Grid>
      <Grid item xs={12} sm={6}>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
          Pattern Pressure
        </Typography>
        {patterns.length === 0 ? (
          <EmptyState message="Pattern detections will appear here once signal events stream through the backend." height={198} />
        ) : (
          <ResponsiveContainer width="100%" height={198}>
            <BarChart data={patterns}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" vertical={false} />
              <XAxis dataKey="pattern" tick={{ fontSize: 10, fill: '#94a3b8' }} interval={0} angle={-15} textAnchor="end" height={56} />
              <YAxis tick={{ fontSize: 10, fill: '#94a3b8' }} allowDecimals={false} />
              <Tooltip {...CHART_TOOLTIP_STYLE} />
              <Bar dataKey="count" fill="#26e0b8" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        )}
      </Grid>
    </Grid>
  )
}

const Dashboard = () => {
  const dispatch = useDispatch()
  const navigate = useNavigate()
  const [dashboard, setDashboard] = useState(null)
  const [selectedSymbol, setSelectedSymbol] = useState('XAUUSD')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false

    const load = async () => {
      try {
        const [overview] = await Promise.all([
          analyticsService.getDashboardOverview(selectedSymbol, 72),
          dispatch(fetchOpenPositions()),
          dispatch(fetchAllTraders()),
        ])
        if (!cancelled) {
          setDashboard(overview)
          if (overview.symbol && overview.symbol !== selectedSymbol) {
            setSelectedSymbol(overview.symbol)
          }
        }
      } catch {
        if (!cancelled) {
          setDashboard(null)
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    load()
    const interval = setInterval(load, 15000)
    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [dispatch, selectedSymbol])

  const portfolioOverview = dashboard?.portfolio_overview || {}
  const portfolioCurve = dashboard?.portfolio_curve || []
  const regimeHistory = dashboard?.regime_history || []
  const predictionHistory = dashboard?.prediction_history || []
  const latestPrediction = dashboard?.latest_prediction || null
  const patternSnapshot = dashboard?.pattern_snapshot || []
  const availableSymbols = dashboard?.symbols?.length ? dashboard.symbols : ['XAUUSD']

  const control = (
    <FormControl size="small" sx={{ minWidth: 140 }}>
      <Select
        value={selectedSymbol}
        onChange={(event) => setSelectedSymbol(event.target.value)}
        sx={{
          color: 'text.primary',
          '& .MuiOutlinedInput-notchedOutline': { borderColor: 'rgba(148,163,184,0.18)' },
        }}
      >
        {availableSymbols.map((symbol) => (
          <MenuItem key={symbol} value={symbol}>
            {symbol}
          </MenuItem>
        ))}
      </Select>
    </FormControl>
  )

  return (
    <Box sx={{ display: 'grid', gap: 2, width: '100%', maxWidth: '100%', overflow: 'hidden' }}>
      <SectionHeader
        title="Command Dashboard"
        subtitle="Portfolio pulse, live model state, and backend-routed market intelligence."
        controls={control}
      />

      <KeyMetrics />

      <Grid container spacing={2}>
        <Grid item xs={12} xl={7}>
          <Paper className="theme-panel" sx={{ p: 2.25, height: '100%' }}>
            <SectionHeader
              title="Portfolio Curve"
              subtitle={`${Number(portfolioOverview.open_positions || 0)} open positions across ${Number(portfolioOverview.total_trades || 0)} recorded trades`}
              action="Analytics"
              onAction={() => navigate('/analytics')}
            />
            {loading && portfolioCurve.length === 0 ? (
              <EmptyState message="Loading portfolio history..." />
            ) : portfolioCurve.length === 0 ? (
              <EmptyState message="Portfolio snapshots will render here once the backend receives equity updates." />
            ) : (
              <Box sx={{ display: 'grid', gap: 1.5 }}>
                <Grid container spacing={1}>
                  <Grid item xs={6} sm={3}>
                    <Chip label={`Equity ${formatCurrency(Number(portfolioOverview.portfolio_value || 0))}`} sx={{ width: '100%' }} />
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <Chip label={`P&L ${formatCurrency(Number(portfolioOverview.total_pnl || 0))}`} sx={{ width: '100%' }} />
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <Chip label={`Cash ${formatCurrency(Number(portfolioOverview.cash_balance || 0))}`} sx={{ width: '100%' }} />
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <Chip label={`${portfolioCurve.length} points`} sx={{ width: '100%' }} />
                  </Grid>
                </Grid>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={portfolioCurve}>
                    <defs>
                      <linearGradient id="portfolioEquity" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#26e0b8" stopOpacity={0.45} />
                        <stop offset="95%" stopColor="#26e0b8" stopOpacity={0.02} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                    <XAxis dataKey="timestamp" tickFormatter={chartTimeLabel} tick={{ fontSize: 11, fill: '#94a3b8' }} minTickGap={24} />
                    <YAxis yAxisId="equity" tick={{ fontSize: 11, fill: '#94a3b8' }} tickFormatter={(value) => `${Math.round(value / 1000)}k`} />
                    <YAxis yAxisId="drawdown" orientation="right" tick={{ fontSize: 11, fill: '#94a3b8' }} tickFormatter={(value) => `${Math.round(value * 100)}%`} />
                    <Tooltip {...CHART_TOOLTIP_STYLE} />
                    <ReferenceLine yAxisId="drawdown" y={0} stroke="rgba(148,163,184,0.3)" />
                    <Area yAxisId="equity" type="monotone" dataKey="equity" stroke="#26e0b8" fill="url(#portfolioEquity)" strokeWidth={2} />
                    <Line yAxisId="drawdown" type="monotone" dataKey="drawdown" stroke="#f59e0b" strokeWidth={2} dot={false} />
                  </AreaChart>
                </ResponsiveContainer>
              </Box>
            )}
          </Paper>
        </Grid>
        <Grid item xs={12} xl={5}>
          <Paper className="theme-panel" sx={{ p: 2.25, height: '100%' }}>
            <SectionHeader
              title="Model Snapshot"
              subtitle={latestPrediction?.regime ? `${selectedSymbol} regime: ${latestPrediction.regime}` : `${selectedSymbol} model state`}
              action="ML / AI"
              onAction={() => navigate('/ml')}
            />
            <ModelSnapshot latestPrediction={latestPrediction} patterns={patternSnapshot} />
          </Paper>
        </Grid>
      </Grid>

      <Grid container spacing={2}>
        <Grid item xs={12} lg={6}>
          <Paper className="theme-panel" sx={{ p: 2.25, height: '100%' }}>
            <SectionHeader
              title="Regime Timeline"
              subtitle={`Latest regime changes for ${selectedSymbol}`}
              action="Monitors"
              onAction={() => navigate('/monitors')}
            />
            {regimeHistory.length === 0 ? (
              <EmptyState message="No regime events yet. Once signals are emitted through the backend, this timeline will populate." />
            ) : (
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={regimeHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                  <XAxis dataKey="timestamp" tickFormatter={chartTimeLabel} tick={{ fontSize: 11, fill: '#94a3b8' }} minTickGap={24} />
                  <YAxis yAxisId="regime" domain={[-1, 2]} ticks={[-1, 0, 1, 2]} tickFormatter={(value) => REGIME_LABELS[String(value)] || value} tick={{ fontSize: 10, fill: '#94a3b8' }} width={86} />
                  <YAxis yAxisId="score" orientation="right" domain={[0, 1]} tickFormatter={(value) => `${Math.round(value * 100)}%`} tick={{ fontSize: 10, fill: '#94a3b8' }} />
                  <Tooltip {...CHART_TOOLTIP_STYLE} />
                  <ReferenceLine yAxisId="regime" y={0} stroke="rgba(148,163,184,0.25)" />
                  <Line yAxisId="regime" type="stepAfter" dataKey="regime_score" stroke="#1976d2" strokeWidth={2.5} dot={false} />
                  <Line yAxisId="score" type="monotone" dataKey="quality_score" stroke="#26e0b8" strokeWidth={2} dot={false} />
                  <Line yAxisId="score" type="monotone" dataKey="confidence" stroke="#f59e0b" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            )}
          </Paper>
        </Grid>
        <Grid item xs={12} lg={6}>
          <Paper className="theme-panel" sx={{ p: 2.25, height: '100%' }}>
            <SectionHeader
              title="Prediction Confidence"
              subtitle={`Bull/bear probabilities and quality score for ${selectedSymbol}`}
            />
            {predictionHistory.length === 0 ? (
              <EmptyState message="Prediction traces will render here when the backend receives signal metadata from the engine." />
            ) : (
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={predictionHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                  <XAxis dataKey="timestamp" tickFormatter={chartTimeLabel} tick={{ fontSize: 11, fill: '#94a3b8' }} minTickGap={24} />
                  <YAxis domain={[0, 1]} tickFormatter={(value) => `${Math.round(value * 100)}%`} tick={{ fontSize: 10, fill: '#94a3b8' }} />
                  <Tooltip {...CHART_TOOLTIP_STYLE} />
                  <Line type="monotone" dataKey="p_bull" name="Bull" stroke="#26e0b8" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="p_bear" name="Bear" stroke="#ef4444" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="quality_score" name="Quality" stroke="#f59e0b" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            )}
          </Paper>
        </Grid>
      </Grid>

      <Grid container spacing={2}>
        <Grid item xs={12} lg={6}>
          <Paper className="theme-panel" sx={{ p: 2.25, height: '100%' }}>
            <SectionHeader title="Open Positions" action="History" onAction={() => navigate('/history')} />
            <OpenPositions compact />
          </Paper>
        </Grid>
        <Grid item xs={12} lg={3}>
          <Paper className="theme-panel" sx={{ p: 2.25, height: '100%' }}>
            <SectionHeader title="Bot Status" action="Traders" onAction={() => navigate('/traders')} />
            <BotStatus compact />
          </Paper>
        </Grid>
        <Grid item xs={12} lg={3}>
          <Paper className="theme-panel" sx={{ p: 2.25, height: '100%' }}>
            <SectionHeader title="Risk Overview" action="Analytics" onAction={() => navigate('/analytics')} />
            <RiskMetrics compact />
          </Paper>
        </Grid>
      </Grid>

      <Grid container spacing={2}>
        <Grid item xs={12} lg={7}>
          <Paper className="theme-panel" sx={{ p: 2.25, height: '100%' }}>
            <SectionHeader title="Recent Trades" action="Full History" onAction={() => navigate('/history')} />
            <RecentTrades />
          </Paper>
        </Grid>
        <Grid item xs={12} lg={5}>
          <Paper className="theme-panel" sx={{ p: 2.25, height: '100%' }}>
            <SectionHeader title="Signal Feed" action="Alerts" onAction={() => navigate('/alerts')} />
            <SignalFeed />
          </Paper>
        </Grid>
      </Grid>
    </Box>
  )
}

export default Dashboard
