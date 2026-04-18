import React, { startTransition, useEffect, useState } from 'react'
import {
  Box,
  Container,
  Grid,
  Paper,
  Typography,
  ToggleButtonGroup,
  ToggleButton,
  Skeleton,
  Divider,
  Chip,
  Stack,
} from '@mui/material'
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from 'recharts'
import analyticsService from '@services/analyticsService'
import traderService from '@services/traderService'
import { formatCurrency, formatPercent } from '@utils/formatters'

const PERIODS = ['7d', '30d', '90d', 'all']

const TRADER_COLORS = {
  trader_1: '#26e0b8',
  trader_2: '#1976D2',
  trader_3: '#FF6F00',
  trader_4: '#9C27B0',
}

const TRADER_LABELS = {
  trader_1: 'T1',
  trader_2: 'T2',
  trader_3: 'T3',
  trader_4: 'T4',
}

const ChartTooltipStyle = {
  contentStyle: { background: '#111a2e', border: '1px solid rgba(148,163,184,0.2)', fontSize: '0.75rem' },
  labelStyle: { color: '#94a3b8' },
}

const StatCard = ({ label, value, sub, color, loading }) => (
  <Paper className="theme-panel" sx={{ p: 2, textAlign: 'center' }}>
    <Typography variant="caption" color="text.secondary" display="block">
      {label}
    </Typography>
    {loading ? (
      <Skeleton variant="text" width="60%" sx={{ mx: 'auto' }} />
    ) : (
      <Typography variant="h6" fontWeight={700} sx={{ color: color || 'text.primary', mt: 0.5 }}>
        {value}
      </Typography>
    )}
    {sub && !loading && (
      <Typography variant="caption" color="text.secondary">
        {sub}
      </Typography>
    )}
  </Paper>
)

const AnalyticsPage = () => {
  const [period, setPeriod] = useState('30d')
  const [perf, setPerf] = useState(null)
  const [equity, setEquity] = useState([])
  const [monthly, setMonthly] = useState([])
  const [traders, setTraders] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.allSettled([
      analyticsService.getPerformance(period),
      analyticsService.getEquityCurve(period),
      analyticsService.getMonthlyReturns(),
      traderService.getAllTraders(),
    ]).then(([perfRes, equityRes, monthlyRes, tradersRes]) => {
      if (perfRes.status === 'fulfilled') setPerf(perfRes.value)
      if (equityRes.status === 'fulfilled') {
        setEquity(Array.isArray(equityRes.value) ? equityRes.value : equityRes.value?.data || [])
      }
      if (monthlyRes.status === 'fulfilled') {
        const raw = monthlyRes.value
        setMonthly(Array.isArray(raw) ? raw : raw?.data || [])
      }
      if (tradersRes.status === 'fulfilled') {
        setTraders(tradersRes.value?.traders || tradersRes.value || [])
      }
      startTransition(() => {
        setLoading(false)
      })
    })
  }, [period])

  const traderStats = traders.map((trader) => ({
    name: trader.name || TRADER_LABELS[trader.trader_id] || trader.trader_id,
    id: trader.trader_id || trader.id,
    win_rate: trader.win_rate ?? trader.metrics?.win_rate ?? 0,
    net_pnl: trader.total_pnl ?? trader.metrics?.total_pnl ?? 0,
    avg_rr: trader.metrics?.avg_rr ?? 0,
    total_trades: trader.total_trades ?? trader.metrics?.total_trades ?? 0,
    max_drawdown: trader.metrics?.max_drawdown ?? 0,
  }))

  return (
    <Container maxWidth="xl" className="fade-in" sx={{ py: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 3 }}>
        <Box>
          <Typography variant="h5" fontWeight={700}>
            Analytics
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Portfolio and per-trader performance
          </Typography>
        </Box>
        <ToggleButtonGroup
          value={period}
          exclusive
          onChange={(_, v) => {
            if (!v) return
            setLoading(true)
            setPeriod(v)
          }}
          size="small"
        >
          {PERIODS.map((p) => (
            <ToggleButton key={p} value={p} sx={{ fontSize: '0.72rem', px: 1.5 }}>
              {p}
            </ToggleButton>
          ))}
        </ToggleButtonGroup>
      </Box>

      {/* KPI row */}
      <Grid container spacing={1.5} sx={{ mb: 3 }}>
        {[
          {
            label: 'Net P&L',
            value: formatCurrency(perf?.net_pnl ?? 0),
            color: (perf?.net_pnl ?? 0) >= 0 ? '#00C853' : '#D32F2F',
          },
          { label: 'Win Rate', value: formatPercent(perf?.win_rate ?? 0) },
          { label: 'Total Trades', value: perf?.total_trades ?? 0 },
          { label: 'Avg R:R', value: (perf?.avg_rr ?? 0).toFixed(2) },
          {
            label: 'Max Drawdown',
            value: formatCurrency(Math.abs(perf?.max_drawdown ?? 0)),
            color: '#FF6F00',
          },
          { label: 'Profit Factor', value: (perf?.profit_factor ?? 0).toFixed(2) },
        ].map(({ label, value, color }) => (
          <Grid item xs={6} sm={4} md={2} key={label}>
            <StatCard label={label} value={value} color={color} loading={loading} />
          </Grid>
        ))}
      </Grid>

      <Grid container spacing={2.5} sx={{ mb: 2.5 }}>
        {/* Equity curve */}
        <Grid item xs={12} lg={8}>
          <Paper className="theme-panel" sx={{ p: 2.5 }}>
            <Typography variant="subtitle2" fontWeight={700} gutterBottom>
              Equity Curve
            </Typography>
            {equity.length === 0 ? (
              <Box sx={{ height: 240, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Typography variant="body2" color="text.secondary">
                  No equity data for this period
                </Typography>
              </Box>
            ) : (
              <ResponsiveContainer width="100%" height={240}>
                <AreaChart data={equity}>
                  <defs>
                    <linearGradient id="equityGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#26e0b8" stopOpacity={0.2} />
                      <stop offset="95%" stopColor="#26e0b8" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                  <XAxis dataKey="date" tick={{ fontSize: 10, fill: '#94a3b8' }} />
                  <YAxis tick={{ fontSize: 10, fill: '#94a3b8' }} />
                  <Tooltip {...ChartTooltipStyle} />
                  <ReferenceLine y={0} stroke="rgba(148,163,184,0.3)" />
                  <Area
                    type="monotone"
                    dataKey="equity"
                    stroke="#26e0b8"
                    strokeWidth={2}
                    fill="url(#equityGrad)"
                    dot={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </Paper>
        </Grid>

        {/* Monthly returns */}
        <Grid item xs={12} lg={4}>
          <Paper className="theme-panel" sx={{ p: 2.5 }}>
            <Typography variant="subtitle2" fontWeight={700} gutterBottom>
              Monthly Returns
            </Typography>
            {monthly.length === 0 ? (
              <Box sx={{ height: 240, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Typography variant="body2" color="text.secondary">
                  No monthly data yet
                </Typography>
              </Box>
            ) : (
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={monthly}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                  <XAxis dataKey="month" tick={{ fontSize: 10, fill: '#94a3b8' }} />
                  <YAxis tick={{ fontSize: 10, fill: '#94a3b8' }} />
                  <Tooltip {...ChartTooltipStyle} />
                  <ReferenceLine y={0} stroke="rgba(148,163,184,0.3)" />
                  <Bar
                    dataKey="return"
                    fill="#26e0b8"
                    radius={[3, 3, 0, 0]}
                    // color each bar based on sign
                    label={false}
                  />
                </BarChart>
              </ResponsiveContainer>
            )}
          </Paper>
        </Grid>
      </Grid>

      {/* Per-trader comparison */}
      {traderStats.length > 0 && (
        <Paper className="theme-panel" sx={{ p: 2.5, mb: 2.5 }}>
          <Typography variant="subtitle2" fontWeight={700} gutterBottom>
            Trader Performance Comparison
          </Typography>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={traderStats} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 10, fill: '#94a3b8' }} />
              <YAxis type="category" dataKey="name" tick={{ fontSize: 11, fill: '#94a3b8' }} width={30} />
              <Tooltip {...ChartTooltipStyle} />
              <Bar dataKey="net_pnl" name="Net P&L" fill="#26e0b8" radius={[0, 3, 3, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Paper>
      )}

      {/* Trader stats table */}
      {traderStats.length > 0 && (
        <Paper className="theme-panel" sx={{ p: 2.5 }}>
          <Typography variant="subtitle2" fontWeight={700} gutterBottom>
            Trader Breakdown
          </Typography>
          <Grid container spacing={1.5}>
            {traderStats.map((t) => (
              <Grid item xs={12} sm={6} md={3} key={t.id}>
                <Box
                  sx={{
                    p: 2,
                    borderRadius: 2,
                    border: `1px solid ${TRADER_COLORS[t.id] || 'rgba(148,163,184,0.2)'}30`,
                    backgroundColor: `${TRADER_COLORS[t.id] || '#94a3b8'}08`,
                  }}
                >
                  <Typography variant="subtitle2" fontWeight={700} sx={{ color: TRADER_COLORS[t.id] }}>
                    {t.name}
                  </Typography>
                  <Divider sx={{ my: 1, borderColor: 'rgba(148,163,184,0.1)' }} />
                  <Stack spacing={0.5}>
                    {[
                      { label: 'Net P&L', value: formatCurrency(t.net_pnl ?? 0), color: (t.net_pnl ?? 0) >= 0 ? '#00C853' : '#D32F2F' },
                      { label: 'Win Rate', value: formatPercent(t.win_rate ?? 0) },
                      { label: 'Trades', value: t.total_trades ?? 0 },
                      { label: 'Avg R:R', value: (t.avg_rr ?? 0).toFixed(2) },
                    ].map(({ label, value, color }) => (
                      <Box key={label} sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Typography variant="caption" color="text.secondary">{label}</Typography>
                        <Typography variant="caption" fontWeight={700} sx={{ color: color || 'text.primary' }}>{value}</Typography>
                      </Box>
                    ))}
                  </Stack>
                </Box>
              </Grid>
            ))}
          </Grid>
        </Paper>
      )}

      {!perf && !loading && (
        <Box sx={{ textAlign: 'center', py: 6 }}>
          <Typography color="text.secondary">No data available — start paper trading to populate analytics</Typography>
        </Box>
      )}
    </Container>
  )
}

export default AnalyticsPage
