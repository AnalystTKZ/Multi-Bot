import { useEffect, useState } from 'react'
import {
  Box,
  Container,
  Grid,
  Paper,
  Typography,
  Chip,
  Button,
  Stack,
  Alert,
  CircularProgress,
} from '@mui/material'
import {
  PlayArrow as ResumeIcon,
  Pause as PauseIcon,
  Stop as StopIcon,
  TrendingUp,
  TrendingDown,
} from '@mui/icons-material'
import { useSelector, useDispatch } from 'react-redux'
import { fetchAllTraders, fetchTraderPerformance } from '@store/slices/tradersSlice'
import traderService from '@services/traderService'
import { formatCurrency, formatPercent } from '@utils/formatters'

const TRADER_META = {
  trader_1: {
    name: 'T1 — MACD + ICT',
    fullName: 'Trader 1: MACD + Price Action + ICT',
    timeframe: '4H',
    strategy: 'Trend following — MACD crossover + growing histogram + ≥1 PA filter + ICT confluence bonus',
    color: '#26e0b8',
    symbols: ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD'],
  },
  trader_2: {
    name: 'T2 — SMC Confluence',
    fullName: 'Trader 2: SMC/ICT Confluence Engine',
    timeframe: '1H',
    strategy: '10+10 SMC/ICT condition scoring, fires on ≥2 — CHoCH, BOS, OB, FVG, sweeps',
    color: '#1976D2',
    symbols: ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'XAUUSD'],
  },
  trader_3: {
    name: 'T3 — Breakout Fade',
    fullName: 'Trader 3: Breakout Failure + Liquidity Grab',
    timeframe: '15m',
    strategy: 'False breakout fade — liquidity sweep ≥0.3×ATR + rejection close + direction confirmation',
    color: '#FF6F00',
    symbols: ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],
  },
  trader_4: {
    name: 'T4 — News Fade',
    fullName: 'Trader 4: News Overreaction Fade',
    timeframe: '5m',
    strategy: 'Post-release fade (3–30m window) — spike >2×ATR, ≥3/5 overreaction checklist, USD + XAU',
    color: '#9C27B0',
    symbols: ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD'],
  },
  trader_5: {
    name: 'T5 — Asian MR',
    fullName: 'Trader 5: Asian Range Mean Reversion',
    timeframe: '1H',
    strategy: 'Asian session range fade — regime=RANGING + ML quality gate + EV>0, hard close 06:45 UTC',
    color: '#E91E63',
    symbols: ['USDJPY', 'EURJPY', 'AUDJPY', 'EURUSD', 'AUDUSD'],
  },
}

const FALLBACK_TRADERS = Object.entries(TRADER_META).map(([id, meta]) => ({
  trader_id: id,
  name: meta.name,
  status: 'running',
  win_rate: null,
  trades_today: 0,
  session_pnl: 0,
  max_concurrent: 3,
}))

const TraderCard = ({ trader }) => {
  const dispatch = useDispatch()
  const meta = TRADER_META[trader.trader_id] || {}
  const performance = useSelector((state) => state.traders.performance[trader.trader_id])
  const signals = useSelector((state) => state.traders.signals[trader.trader_id] || [])
  const [actionLoading, setActionLoading] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    dispatch(fetchTraderPerformance({ id: trader.trader_id, period: '30d' }))
  }, [dispatch, trader.trader_id])

  const sendAction = async (action) => {
    // Map UI actions to backend-accepted control verbs
    const actionMap = { pause: 'stop', resume: 'start', stop: 'stop' }
    const backendAction = actionMap[action] || action
    setActionLoading(action)
    setError(null)
    try {
      await traderService.updateTraderStatus(trader.trader_id, backendAction)
      dispatch(fetchAllTraders())
    } catch {
      setError(`${action} failed`)
    } finally {
      setActionLoading(null)
    }
  }

  const isRunning = trader.status === 'running' || trader.status === 'active'
  const isPaused = trader.status === 'paused'
  const winRate = performance?.win_rate ?? trader.win_rate
  const pnl = performance?.monthly_pnl ?? performance?.daily_pnl ?? trader.session_pnl ?? 0
  const totalTrades = performance?.trader?.total_trades ?? trader.total_trades ?? 0
  const avgRR = performance?.profit_factor ?? 0

  return (
    <Paper
      className="theme-panel"
      sx={{
        p: 0,
        overflow: 'hidden',
        border: `1px solid ${meta.color ? `${meta.color}30` : 'rgba(148,163,184,0.12)'}`,
      }}
    >
      {/* Card header */}
      <Box
        sx={{
          p: 2,
          borderBottom: '1px solid rgba(148,163,184,0.1)',
          background: meta.color ? `linear-gradient(90deg, ${meta.color}10, transparent)` : 'none',
        }}
      >
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <Box>
            <Typography variant="subtitle1" fontWeight={700}>
              {meta.name || trader.name}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {meta.timeframe} · {meta.symbols?.join(', ')}
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Box
              sx={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                backgroundColor: isRunning ? '#00C853' : isPaused ? '#FF6F00' : '#94a3b8',
                boxShadow: isRunning ? '0 0 6px #00C853' : 'none',
              }}
            />
            <Chip
              label={trader.status || 'unknown'}
              size="small"
              sx={{
                height: 20,
                fontSize: '0.68rem',
                backgroundColor: isRunning ? 'rgba(0,200,83,0.15)' : 'rgba(148,163,184,0.15)',
                color: isRunning ? '#00C853' : 'text.secondary',
              }}
            />
          </Box>
        </Box>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.75 }}>
          {meta.strategy}
        </Typography>
      </Box>

      {/* Performance stats */}
      <Grid container sx={{ borderBottom: '1px solid rgba(148,163,184,0.1)' }}>
        {[
          { label: 'Net P&L', value: formatCurrency(pnl), color: pnl >= 0 ? '#00C853' : '#D32F2F' },
          { label: 'Win Rate', value: winRate != null ? formatPercent(winRate) : '—' },
          { label: 'Trades', value: totalTrades || trader.trades_today || 0 },
          { label: 'Avg R:R', value: avgRR ? avgRR.toFixed(2) : '—' },
        ].map(({ label, value, color }) => (
          <Grid item xs={3} key={label} sx={{ p: 1.5, textAlign: 'center', borderRight: '1px solid rgba(148,163,184,0.1)', '&:last-child': { borderRight: 'none' } }}>
            <Typography variant="caption" color="text.secondary" display="block">
              {label}
            </Typography>
            <Typography variant="body2" fontWeight={700} sx={{ color: color || 'text.primary' }}>
              {value}
            </Typography>
          </Grid>
        ))}
      </Grid>

      {/* Recent signals */}
      {signals.length > 0 && (
        <Box sx={{ p: 1.5, borderBottom: '1px solid rgba(148,163,184,0.1)' }}>
          <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 0.75 }}>
            RECENT SIGNALS
          </Typography>
          <Stack spacing={0.5}>
            {signals.slice(0, 3).map((sig, i) => {
              const isBuy = sig.direction === 'buy' || sig.side === 'buy'
              return (
                <Box key={i} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Box sx={{ color: isBuy ? '#00C853' : '#D32F2F', lineHeight: 0 }}>
                    {isBuy ? <TrendingUp sx={{ fontSize: 14 }} /> : <TrendingDown sx={{ fontSize: 14 }} />}
                  </Box>
                  <Typography variant="caption" fontWeight={600}>
                    {sig.symbol}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {isBuy ? 'BUY' : 'SELL'}
                  </Typography>
                  {sig.confidence != null && (
                    <Typography variant="caption" color="text.disabled">
                      conf: {(sig.confidence * 100).toFixed(0)}%
                    </Typography>
                  )}
                </Box>
              )
            })}
          </Stack>
        </Box>
      )}

      {/* Controls */}
      <Box sx={{ p: 1.5 }}>
        {error && (
          <Alert severity="error" sx={{ mb: 1, py: 0, fontSize: '0.72rem' }}>
            {error}
          </Alert>
        )}
        <Stack direction="row" spacing={1}>
          {isPaused ? (
            <Button
              size="small"
              variant="outlined"
              startIcon={actionLoading === 'resume' ? <CircularProgress size={12} color="inherit" /> : <ResumeIcon />}
              disabled={!!actionLoading}
              onClick={() => sendAction('resume')}
              sx={{ fontSize: '0.72rem', flex: 1, color: '#00C853', borderColor: '#00C853' }}
            >
              Resume
            </Button>
          ) : (
            <Button
              size="small"
              variant="outlined"
              startIcon={actionLoading === 'pause' ? <CircularProgress size={12} color="inherit" /> : <PauseIcon />}
              disabled={!!actionLoading}
              onClick={() => sendAction('pause')}
              sx={{ fontSize: '0.72rem', flex: 1 }}
            >
              Pause
            </Button>
          )}
          <Button
            size="small"
            variant="outlined"
            color="error"
            startIcon={actionLoading === 'stop' ? <CircularProgress size={12} color="inherit" /> : <StopIcon />}
            disabled={!!actionLoading}
            onClick={() => sendAction('stop')}
            sx={{ fontSize: '0.72rem', flex: 1 }}
          >
            Stop
          </Button>
        </Stack>
      </Box>
    </Paper>
  )
}

const TradersPage = () => {
  const dispatch = useDispatch()
  const { list, loading } = useSelector((state) => state.traders)
  const traders = list.length > 0 ? list : FALLBACK_TRADERS

  useEffect(() => {
    dispatch(fetchAllTraders())
    const interval = setInterval(() => dispatch(fetchAllTraders()), 15000)
    return () => clearInterval(interval)
  }, [dispatch])

  return (
    <Container maxWidth="xl" className="fade-in" sx={{ py: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 3 }}>
        <Box>
          <Typography variant="h5" fontWeight={700}>
            Traders
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {traders.filter((t) => t.status === 'running' || t.status === 'active').length} of {traders.length} bots running
          </Typography>
        </Box>
        <Button
          size="small"
          variant="outlined"
          onClick={() => dispatch(fetchAllTraders())}
          disabled={loading}
          startIcon={loading ? <CircularProgress size={14} /> : null}
        >
          Refresh
        </Button>
      </Box>

      <Grid container spacing={2.5}>
        {traders.map((trader) => (
          <Grid item xs={12} sm={6} xl={3} key={trader.trader_id || trader.id}>
            <TraderCard trader={trader} />
          </Grid>
        ))}
      </Grid>
    </Container>
  )
}

export default TradersPage
