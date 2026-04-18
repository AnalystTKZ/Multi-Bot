import React, { useEffect, useState } from 'react'
import { Box, Typography, LinearProgress, Stack, Grid, Divider } from '@mui/material'
import { useSelector } from 'react-redux'
import analyticsService from '@services/analyticsService'
import { formatCurrency, formatPercent } from '@utils/formatters'

const RiskBar = ({ label, value, max, color, format }) => {
  const pct = Math.min((value / max) * 100, 100)
  const displayVal = format === 'pct' ? formatPercent(value) : formatCurrency(value)
  const barColor = pct > 80 ? '#D32F2F' : pct > 60 ? '#FF6F00' : color || '#26e0b8'
  return (
    <Box sx={{ mb: 1.5 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
        <Typography variant="caption" color="text.secondary">
          {label}
        </Typography>
        <Typography variant="caption" fontWeight={700} sx={{ color: barColor }}>
          {displayVal}
        </Typography>
      </Box>
      <LinearProgress
        variant="determinate"
        value={pct}
        sx={{
          height: 5,
          borderRadius: 3,
          backgroundColor: 'rgba(148,163,184,0.1)',
          '& .MuiLinearProgress-bar': { backgroundColor: barColor, borderRadius: 3 },
        }}
      />
    </Box>
  )
}

const RiskMetrics = ({ compact }) => {
  const positions = useSelector((state) => state.positions.open)
  const [perf, setPerf] = useState(null)

  useEffect(() => {
    analyticsService
      .getPerformance('30d')
      .then(setPerf)
      .catch(() => {})
  }, [])

  const positionCount = positions.length
  const maxPositions = 12
  const openPnl = positions.reduce((sum, p) => sum + (p.pnl || 0), 0)
  const drawdown = perf?.max_drawdown ?? 0
  const portfolioValue = perf?.portfolio_value || perf?.balance || 10000
  const drawdownPct = portfolioValue > 0 ? Math.abs(drawdown / portfolioValue) : 0
  const exposure = positionCount / maxPositions

  return (
    <Box>
      <Grid container spacing={2} sx={{ mb: compact ? 1.5 : 2 }}>
        <Grid item xs={4}>
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="caption" color="text.secondary" display="block">
              Open Positions
            </Typography>
            <Typography variant="h6" fontWeight={700}>
              {positionCount}
            </Typography>
          </Box>
        </Grid>
        <Grid item xs={4}>
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="caption" color="text.secondary" display="block">
              Open P&L
            </Typography>
            <Typography
              variant="h6"
              fontWeight={700}
              sx={{ color: openPnl >= 0 ? '#00C853' : '#D32F2F' }}
            >
              {openPnl >= 0 ? '+' : ''}
              {formatCurrency(openPnl)}
            </Typography>
          </Box>
        </Grid>
        <Grid item xs={4}>
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="caption" color="text.secondary" display="block">
              Max Drawdown
            </Typography>
            <Typography
              variant="h6"
              fontWeight={700}
              sx={{ color: Math.abs(drawdown) > 500 ? '#FF6F00' : 'text.primary' }}
            >
              {formatCurrency(drawdown)}
            </Typography>
          </Box>
        </Grid>
      </Grid>

      <Divider sx={{ borderColor: 'rgba(148,163,184,0.1)', mb: 1.5 }} />

      <RiskBar
        label="Portfolio Exposure"
        value={exposure}
        max={1}
        format="pct"
        color="#26e0b8"
      />
      <RiskBar
        label="Drawdown (vs 20% limit)"
        value={drawdownPct}
        max={0.2}
        format="pct"
        color="#26e0b8"
      />
    </Box>
  )
}

export default RiskMetrics
