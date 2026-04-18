import React from 'react'
import { Box, Typography, Stack, Chip, LinearProgress, Grid } from '@mui/material'
import { useSelector } from 'react-redux'

const TRADER_NAMES = {
  trader_1: 'T1 — MACD+ICT',
  trader_2: 'T2 — SMC Confluence',
  trader_3: 'T3 — Breakout Retest',
  trader_4: 'T4 — News Fade',
}

const TRADER_TF = {
  trader_1: '4H',
  trader_2: '1H',
  trader_3: '15m',
  trader_4: '5m',
}

const BotStatus = ({ compact }) => {
  const traders = useSelector((state) => state.traders.list)
  const tradingMode = useSelector((state) => state.ui.tradingMode)

  const displayTraders =
    traders.length > 0
      ? traders
      : [
          { trader_id: 'trader_1', name: 'T1 — MACD+ICT', status: 'initializing' },
          { trader_id: 'trader_2', name: 'T2 — SMC Confluence', status: 'initializing' },
          { trader_id: 'trader_3', name: 'T3 — Breakout Retest', status: 'initializing' },
          { trader_id: 'trader_4', name: 'T4 — News Fade', status: 'initializing' },
        ]

  return (
    <Stack spacing={compact ? 1 : 1.5}>
      {displayTraders.map((trader) => {
        const id = trader.trader_id || trader.id
        const name = TRADER_NAMES[id] || trader.name || id
        const tf = TRADER_TF[id] || trader.timeframe || '—'
        const status = trader.status || 'unknown'
        const isRunning = status === 'running' || status === 'active'
        const winRate = trader.win_rate ?? null
        const tradesToday = trader.trades_today ?? trader.session_trades ?? null

        return (
          <Box
            key={id}
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 1.5,
              p: 1,
              borderRadius: 1.5,
              border: '1px solid rgba(148,163,184,0.1)',
              backgroundColor: isRunning ? 'rgba(0,200,83,0.04)' : 'rgba(148,163,184,0.03)',
            }}
          >
            {/* Status dot */}
            <Box
              sx={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                backgroundColor: isRunning ? '#00C853' : status === 'paused' ? '#FF6F00' : '#94a3b8',
                flexShrink: 0,
                boxShadow: isRunning ? '0 0 6px #00C853' : 'none',
              }}
            />
            <Box sx={{ flex: 1, minWidth: 0 }}>
              <Typography variant="body2" fontWeight={600} noWrap>
                {name}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {tf} · {status}
              </Typography>
            </Box>
            <Stack direction="row" spacing={0.75} alignItems="center">
              {tradesToday != null && (
                <Typography variant="caption" color="text.secondary">
                  {tradesToday}T
                </Typography>
              )}
              {winRate != null && (
                <Typography
                  variant="caption"
                  sx={{ color: winRate >= 0.5 ? '#00C853' : '#FF6F00', fontWeight: 600 }}
                >
                  {(winRate * 100).toFixed(0)}%W
                </Typography>
              )}
              <Chip
                label={tradingMode === 'paper' ? 'PAPER' : tradingMode === 'live' ? 'LIVE' : tradingMode.toUpperCase()}
                size="small"
                sx={{
                  height: 16,
                  fontSize: '0.6rem',
                  backgroundColor:
                    tradingMode === 'live' ? 'rgba(211,47,47,0.15)' : 'rgba(0,200,83,0.15)',
                  color: tradingMode === 'live' ? '#D32F2F' : '#00C853',
                }}
              />
            </Stack>
          </Box>
        )
      })}
    </Stack>
  )
}

export default BotStatus
