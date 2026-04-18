import React, { useEffect } from 'react'
import { Box, Container, Grid, Paper, Typography, Chip, Stack, Divider, LinearProgress } from '@mui/material'
import {
  ShowChart as ChartIcon,
  Newspaper as NewsIcon,
  Security as RiskIcon,
  Memory as SystemIcon,
  CheckCircle as OkIcon,
  Warning as WarnIcon,
  Error as ErrIcon,
} from '@mui/icons-material'
import { useDispatch, useSelector } from 'react-redux'
import { fetchMonitorStatus } from '@store/slices/monitorsSlice'
import { formatDistanceToNow } from 'date-fns'

const MONITOR_META = {
  chart_monitor: { label: 'Chart Signal Detector', icon: <ChartIcon />, color: '#26e0b8' },
  news_monitor: { label: 'News Sentiment Tracker', icon: <NewsIcon />, color: '#1976D2' },
  risk_monitor: { label: 'Risk Monitor', icon: <RiskIcon />, color: '#FF6F00' },
  drawdown_monitor: { label: 'Drawdown Monitor', icon: <RiskIcon />, color: '#D32F2F' },
  system_monitor: { label: 'System Monitor', icon: <SystemIcon />, color: '#9C27B0' },
}

const FALLBACK_MONITORS = [
  { monitor_id: 'chart_monitor', name: 'Chart Signal Detector', status: 'running', last_check: new Date().toISOString() },
  { monitor_id: 'news_monitor', name: 'News Sentiment Tracker', status: 'running', last_check: new Date().toISOString() },
  { monitor_id: 'risk_monitor', name: 'Risk Monitor', status: 'running', last_check: new Date().toISOString() },
  { monitor_id: 'drawdown_monitor', name: 'Drawdown Monitor', status: 'running', last_check: new Date().toISOString() },
  { monitor_id: 'system_monitor', name: 'System Monitor', status: 'running', last_check: new Date().toISOString() },
]

const StatusIcon = ({ status }) => {
  if (status === 'running' || status === 'ok') return <OkIcon sx={{ fontSize: 16, color: '#00C853' }} />
  if (status === 'warn' || status === 'degraded') return <WarnIcon sx={{ fontSize: 16, color: '#FF6F00' }} />
  return <ErrIcon sx={{ fontSize: 16, color: '#D32F2F' }} />
}

const MonitorCard = ({ monitor }) => {
  const meta = MONITOR_META[monitor.monitor_id] || {}
  const timeAgo = (() => {
    try {
      return formatDistanceToNow(new Date(monitor.last_check), { addSuffix: true })
    } catch {
      return '—'
    }
  })()

  return (
    <Paper
      className="theme-panel"
      sx={{
        p: 2.5,
        borderLeft: `3px solid ${meta.color || '#94a3b8'}`,
        display: 'flex',
        flexDirection: 'column',
        gap: 1,
      }}
    >
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box sx={{ color: meta.color || 'text.secondary' }}>{meta.icon}</Box>
          <Box>
            <Typography variant="subtitle2" fontWeight={700}>
              {meta.label || monitor.name}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Last check: {timeAgo}
            </Typography>
          </Box>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}>
          <StatusIcon status={monitor.status} />
          <Chip
            label={monitor.status || 'unknown'}
            size="small"
            sx={{
              height: 20,
              fontSize: '0.68rem',
              backgroundColor:
                monitor.status === 'running' || monitor.status === 'ok'
                  ? 'rgba(0,200,83,0.15)'
                  : 'rgba(255,111,0,0.15)',
              color:
                monitor.status === 'running' || monitor.status === 'ok' ? '#00C853' : '#FF6F00',
            }}
          />
        </Box>
      </Box>

      {/* Extra fields if present */}
      {monitor.details && (
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75, mt: 0.5 }}>
          {Object.entries(monitor.details).map(([k, v]) => (
            <Box
              key={k}
              sx={{ px: 1, py: 0.25, borderRadius: 1, backgroundColor: 'rgba(148,163,184,0.08)' }}
            >
              <Typography variant="caption" color="text.secondary">
                {k}:{' '}
              </Typography>
              <Typography variant="caption" fontWeight={600}>
                {String(v)}
              </Typography>
            </Box>
          ))}
        </Box>
      )}
    </Paper>
  )
}

const MonitorsPage = () => {
  const dispatch = useDispatch()
  const { status, loading } = useSelector((state) => state.monitors)
  const monitors = status?.length > 0 ? status : FALLBACK_MONITORS

  useEffect(() => {
    dispatch(fetchMonitorStatus())
    const interval = setInterval(() => dispatch(fetchMonitorStatus()), 15000)
    return () => clearInterval(interval)
  }, [dispatch])

  const runningCount = monitors.filter((m) => m.status === 'running' || m.status === 'ok').length

  return (
    <Container maxWidth="xl" className="fade-in" sx={{ py: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 3 }}>
        <Box>
          <Typography variant="h5" fontWeight={700}>
            Monitors
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {runningCount}/{monitors.length} monitors healthy
          </Typography>
        </Box>
      </Box>

      {loading && <LinearProgress sx={{ mb: 2, borderRadius: 1 }} />}

      <Grid container spacing={2}>
        {monitors.map((monitor) => (
          <Grid item xs={12} sm={6} md={4} key={monitor.monitor_id}>
            <MonitorCard monitor={monitor} />
          </Grid>
        ))}
      </Grid>

      <Divider sx={{ my: 3, borderColor: 'rgba(148,163,184,0.1)' }} />

      {/* System info */}
      <Grid container spacing={2}>
        <Grid item xs={12} md={6}>
          <Paper className="theme-panel" sx={{ p: 2.5 }}>
            <Typography variant="subtitle2" fontWeight={700} gutterBottom>
              Event Bus
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Redis pub/sub — SIGNAL_GENERATED → TRADE_REQUESTED → TRADE_EXECUTED
            </Typography>
            <Stack direction="row" spacing={1} sx={{ mt: 1.5, flexWrap: 'wrap', gap: 0.75 }}>
              {['MARKET_DATA', 'SIGNAL_GENERATED', 'TRADE_REQUESTED', 'TRADE_EXECUTED'].map((ch) => (
                <Chip
                  key={ch}
                  label={ch}
                  size="small"
                  sx={{ fontSize: '0.65rem', height: 20, backgroundColor: 'rgba(38,224,184,0.08)', color: '#26e0b8' }}
                />
              ))}
            </Stack>
          </Paper>
        </Grid>
        <Grid item xs={12} md={6}>
          <Paper className="theme-panel" sx={{ p: 2.5 }}>
            <Typography variant="subtitle2" fontWeight={700} gutterBottom>
              Trade Journal
            </Typography>
            <Typography variant="body2" color="text.secondary">
              CSV (clean) + JSONL (detailed) — written on every TRADE_EXECUTED event
            </Typography>
            <Stack direction="row" spacing={1} sx={{ mt: 1.5, flexWrap: 'wrap', gap: 0.75 }}>
              {['trade_journal.csv', 'trade_journal_detailed.jsonl'].map((f) => (
                <Chip
                  key={f}
                  label={f}
                  size="small"
                  sx={{ fontSize: '0.65rem', height: 20, backgroundColor: 'rgba(148,163,184,0.08)' }}
                />
              ))}
            </Stack>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  )
}

export default MonitorsPage
