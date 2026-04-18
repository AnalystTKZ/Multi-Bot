import { useEffect, useState } from 'react'
import { Grid, Paper, Typography, Skeleton, Chip } from '@mui/material'
import { useSelector } from 'react-redux'
import analyticsService from '@services/analyticsService'
import { formatCurrency, formatPercent } from '@utils/formatters'

const MetricCard = ({ label, value, sub, subColor, loading, highlight }) => (
  <Paper
    className="theme-panel"
    sx={{
      p: { xs: 1.25, sm: 1.5 },
      borderRadius: 2,
      borderLeft: highlight ? `3px solid ${highlight}` : undefined,
    }}
  >
    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', fontSize: '0.68rem' }}>
      {label}
    </Typography>
    {loading ? (
      <Skeleton variant="text" width="70%" height={28} />
    ) : (
      <Typography sx={{ fontWeight: 700, mt: 0.25, lineHeight: 1.2, fontSize: { xs: '0.95rem', sm: '1.05rem' } }}>
        {value}
      </Typography>
    )}
    {sub && !loading && (
      <Typography variant="caption" sx={{ color: subColor || 'text.secondary', fontSize: '0.65rem' }}>
        {sub}
      </Typography>
    )}
  </Paper>
)

const KeyMetrics = () => {
  const positions = useSelector((state) => state.positions.open)
  const tradingMode = useSelector((state) => state.ui.tradingMode)
  const [metrics, setMetrics] = useState(null)
  const [loading, setLoading] = useState(true)
  const portfolioValue = metrics?.portfolio_value || metrics?.balance || 10000

  useEffect(() => {
    analyticsService
      .getPerformance('30d')
      .then(setMetrics)
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  const modeColor =
    tradingMode === 'live' ? '#D32F2F' : tradingMode === 'paper' ? '#00C853' : '#1976D2'

  return (
    <Grid container spacing={1.5}>
      <Grid item xs={6} sm={4} md={2}>
        <MetricCard
          label="Balance"
          value={formatCurrency(metrics?.portfolio_value ?? metrics?.balance ?? 0)}
          sub={
            <Chip
              label={tradingMode.toUpperCase()}
              size="small"
              sx={{ height: 16, fontSize: '0.6rem', backgroundColor: `${modeColor}20`, color: modeColor }}
            />
          }
          loading={loading}
        />
      </Grid>
      <Grid item xs={6} sm={4} md={2}>
        <MetricCard
          label="Monthly P&L"
          value={metrics ? formatCurrency(metrics.total_pnl ?? 0) : '--'}
          sub={metrics?.total_return != null ? formatPercent(metrics.total_return) : undefined}
          subColor={(metrics?.total_pnl ?? 0) >= 0 ? '#00C853' : '#D32F2F'}
          highlight={(metrics?.total_pnl ?? 0) >= 0 ? '#00C853' : '#D32F2F'}
          loading={loading}
        />
      </Grid>
      <Grid item xs={6} sm={4} md={2}>
        <MetricCard
          label="Win Rate"
          value={metrics ? formatPercent(metrics.win_rate ?? 0) : '--'}
          sub={metrics ? `${metrics.winning_trades ?? '—'}W / ${metrics.losing_trades ?? '—'}L` : undefined}
          loading={loading}
        />
      </Grid>
      <Grid item xs={6} sm={4} md={2}>
        <MetricCard
          label="Max Drawdown"
          value={metrics ? formatCurrency(Math.abs(metrics.max_drawdown ?? 0)) : '--'}
          sub={metrics?.max_drawdown != null ? formatPercent(Math.abs(metrics.max_drawdown) / portfolioValue) : undefined}
          subColor="#FF6F00"
          highlight={Math.abs(metrics?.max_drawdown ?? 0) > 1000 ? '#FF6F00' : undefined}
          loading={loading}
        />
      </Grid>
      <Grid item xs={6} sm={4} md={2}>
        <MetricCard
          label="Active Trades"
          value={positions.length}
          sub={positions.length > 0 ? 'positions open' : 'no open positions'}
          loading={false}
        />
      </Grid>
      <Grid item xs={6} sm={4} md={2}>
        <MetricCard
          label="Total Trades (30d)"
          value={metrics?.total_trades ?? '--'}
          sub={metrics ? `Avg R:R ${(metrics.avg_rr ?? 0).toFixed(2)}` : undefined}
          loading={loading}
        />
      </Grid>
    </Grid>
  )
}

export default KeyMetrics
