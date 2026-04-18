import React from 'react'
import { Box, Typography, Stack, Chip } from '@mui/material'
import { TrendingUp, TrendingDown, RadioButtonChecked } from '@mui/icons-material'
import { useSelector } from 'react-redux'
import { formatDistanceToNow } from 'date-fns'

const TRADER_LABELS = {
  trader_1: 'T1',
  trader_2: 'T2',
  trader_3: 'T3',
  trader_4: 'T4',
}

const SignalFeed = () => {
  const signals = useSelector((state) => state.traders.signals)
  const allSignals = Object.entries(signals)
    .flatMap(([tid, sigs]) => sigs.map((s) => ({ ...s, traderId: tid })))
    .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
    .slice(0, 8)

  if (allSignals.length === 0) {
    return (
      <Box sx={{ py: 2, textAlign: 'center' }}>
        <RadioButtonChecked sx={{ fontSize: 28, color: 'rgba(148,163,184,0.2)', mb: 1 }} />
        <Typography variant="body2" color="text.secondary">
          Waiting for signals…
        </Typography>
      </Box>
    )
  }

  return (
    <Stack spacing={0.75}>
      {allSignals.map((sig, i) => {
        const isBuy = sig.direction === 'buy' || sig.side === 'buy'
        const timeAgo = (() => {
          try {
            return formatDistanceToNow(new Date(sig.timestamp), { addSuffix: true })
          } catch {
            return '—'
          }
        })()
        return (
          <Box
            key={i}
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 1,
              p: 1,
              borderRadius: 1.5,
              border: '1px solid rgba(148,163,184,0.1)',
              backgroundColor: isBuy ? 'rgba(0,200,83,0.04)' : 'rgba(211,47,47,0.04)',
            }}
          >
            <Box sx={{ color: isBuy ? '#00C853' : '#D32F2F', lineHeight: 0 }}>
              {isBuy ? <TrendingUp sx={{ fontSize: 16 }} /> : <TrendingDown sx={{ fontSize: 16 }} />}
            </Box>
            <Chip
              label={TRADER_LABELS[sig.traderId] || sig.traderId}
              size="small"
              sx={{ height: 18, fontSize: '0.65rem', backgroundColor: 'rgba(148,163,184,0.1)' }}
            />
            <Typography variant="body2" fontWeight={600} sx={{ flex: 1 }}>
              {sig.symbol}
            </Typography>
            <Typography
              variant="caption"
              sx={{ color: isBuy ? '#00C853' : '#D32F2F', fontWeight: 700 }}
            >
              {isBuy ? 'BUY' : 'SELL'}
            </Typography>
            {sig.confidence != null && (
              <Typography variant="caption" color="text.secondary">
                {(sig.confidence * 100).toFixed(0)}%
              </Typography>
            )}
            <Typography variant="caption" color="text.disabled" sx={{ minWidth: 60, textAlign: 'right' }}>
              {timeAgo}
            </Typography>
          </Box>
        )
      })}
    </Stack>
  )
}

export default SignalFeed
