import React from 'react'
import {
  Table,
  TableContainer,
  TableHead,
  TableRow,
  TableCell,
  TableBody,
  Chip,
  Tooltip,
  IconButton,
  Typography,
} from '@mui/material'
import { Close as CloseIcon } from '@mui/icons-material'
import { useSelector, useDispatch } from 'react-redux'
import { closePosition } from '@store/slices/positionsSlice'
import { formatCurrency } from '@utils/formatters'

const OpenPositions = ({ compact }) => {
  const dispatch = useDispatch()
  const positions = useSelector((state) => state.positions.open)

  if (!positions.length) {
    return (
      <Typography variant="body2" color="text.secondary" sx={{ py: compact ? 1 : 2 }}>
        No open positions.
      </Typography>
    )
  }

  return (
    <TableContainer sx={{ overflowX: 'auto' }}>
      <Table size="small" sx={{ minWidth: 520 }}>
        <TableHead>
          <TableRow>
            {['Symbol', 'Side', 'Entry', 'P&L', 'R:R', ''].map((h) => (
              <TableCell key={h} sx={{ color: 'text.secondary', fontSize: '0.7rem', py: 0.75, px: compact ? 0.75 : 1 }}>
                {h}
              </TableCell>
            ))}
          </TableRow>
        </TableHead>
        <TableBody>
          {positions.map((p) => {
            const side = (p.direction || p.side || p.type || '').toLowerCase()
            const isBuy = side === 'buy'
            const pnl = p.pnl ?? p.profit ?? 0
            return (
              <TableRow key={p.id} sx={{ '&:last-child td': { border: 0 } }}>
                <TableCell sx={{ fontWeight: 700, fontSize: '0.8rem', px: compact ? 0.75 : 1 }}>
                  {p.symbol || p.pair}
                </TableCell>
                <TableCell sx={{ px: compact ? 0.75 : 1 }}>
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
                <TableCell sx={{ fontSize: '0.78rem', px: compact ? 0.75 : 1 }}>
                  {formatCurrency(p.price_open ?? p.entry_price ?? 0)}
                </TableCell>
                <TableCell
                  sx={{
                    fontSize: '0.78rem',
                    fontWeight: 700,
                    color: pnl >= 0 ? '#00C853' : '#D32F2F',
                    px: compact ? 0.75 : 1,
                  }}
                >
                  {pnl >= 0 ? '+' : ''}
                  {formatCurrency(pnl)}
                </TableCell>
                <TableCell sx={{ fontSize: '0.78rem', color: 'text.secondary', px: compact ? 0.75 : 1 }}>
                  {p.rr_ratio ? Number(p.rr_ratio).toFixed(1) : '—'}
                </TableCell>
                <TableCell sx={{ px: 0 }}>
                  <Tooltip title="Close position">
                    <IconButton
                      size="small"
                      onClick={() => dispatch(closePosition({ id: p.id, reason: 'manual' }))}
                      sx={{ opacity: 0.5, '&:hover': { opacity: 1, color: '#D32F2F' } }}
                    >
                      <CloseIcon sx={{ fontSize: 14 }} />
                    </IconButton>
                  </Tooltip>
                </TableCell>
              </TableRow>
            )
          })}
        </TableBody>
      </Table>
    </TableContainer>
  )
}

export default OpenPositions
