import React, { useEffect, useState } from 'react'
import {
  Box,
  Table,
  TableContainer,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  Chip,
  Skeleton,
  Typography,
} from '@mui/material'
import analyticsService from '@services/analyticsService'
import { formatCurrency } from '@utils/formatters'

const RecentTrades = () => {
  const [trades, setTrades] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    analyticsService
      .getTrades({ limit: 10 })
      .then((data) => setTrades(Array.isArray(data) ? data : data.trades || []))
      .catch(() => setTrades([]))
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <Box>
        {[...Array(5)].map((_, i) => (
          <Skeleton key={i} variant="rectangular" height={28} sx={{ mb: 0.5, borderRadius: 1 }} />
        ))}
      </Box>
    )
  }

  if (!trades.length) {
    return (
      <Typography variant="body2" color="text.secondary" sx={{ py: 2 }}>
        No trades recorded yet.
      </Typography>
    )
  }

  return (
    <TableContainer sx={{ overflowX: 'auto' }}>
      <Table size="small" sx={{ minWidth: 560 }}>
      <TableHead>
        <TableRow>
          {['Time', 'Trader', 'Symbol', 'Side', 'R:R', 'P&L'].map((h) => (
            <TableCell key={h} sx={{ color: 'text.secondary', fontSize: '0.7rem', py: 0.75 }}>
              {h}
            </TableCell>
          ))}
        </TableRow>
      </TableHead>
      <TableBody>
        {trades.map((t, i) => {
          const pnl = t.pnl ?? t.profit ?? t.net_pnl ?? 0
          const isBuy = (t.side || t.direction || '').toLowerCase() === 'buy'
          const timeStr = t.timestamp || t.closed_at || t.opened_at
            ? new Date(t.timestamp || t.closed_at || t.opened_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            : '—'
          return (
            <TableRow key={t.id || i} sx={{ '&:last-child td': { border: 0 } }}>
              <TableCell sx={{ fontSize: '0.75rem', color: 'text.secondary' }}>{timeStr}</TableCell>
              <TableCell sx={{ fontSize: '0.75rem' }}>{t.trader || t.trader_id || '—'}</TableCell>
              <TableCell sx={{ fontSize: '0.75rem', fontWeight: 600 }}>{t.symbol || t.pair}</TableCell>
              <TableCell>
                <Chip
                  label={isBuy ? 'BUY' : 'SELL'}
                  size="small"
                  sx={{
                    height: 18,
                    fontSize: '0.65rem',
                    fontWeight: 700,
                    backgroundColor: isBuy ? 'rgba(0,200,83,0.15)' : 'rgba(211,47,47,0.15)',
                    color: isBuy ? '#00C853' : '#D32F2F',
                  }}
                />
              </TableCell>
              <TableCell sx={{ fontSize: '0.75rem' }}>
                {t.rr_ratio != null ? Number(t.rr_ratio).toFixed(2) : '—'}
              </TableCell>
              <TableCell
                sx={{
                  fontSize: '0.75rem',
                  fontWeight: 700,
                  color: pnl >= 0 ? '#00C853' : '#D32F2F',
                }}
              >
                {pnl >= 0 ? '+' : ''}
                {formatCurrency(pnl)}
              </TableCell>
            </TableRow>
          )
        })}
      </TableBody>
      </Table>
    </TableContainer>
  )
}

export default RecentTrades
