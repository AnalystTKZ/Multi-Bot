import React, { useEffect, useState, useCallback } from 'react'
import {
  Box,
  Container,
  Paper,
  Typography,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  Chip,
  Stack,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Collapse,
  IconButton,
  Divider,
  Skeleton,
  Pagination,
  Grid,
  Tooltip,
} from '@mui/material'
import {
  ExpandMore as ExpandIcon,
  ExpandLess as CollapseIcon,
  Download as ExportIcon,
  FilterList as FilterIcon,
  TrendingUp,
  TrendingDown,
} from '@mui/icons-material'
import analyticsService from '@services/analyticsService'
import traderService from '@services/traderService'
import { formatCurrency, formatPercent } from '@utils/formatters'

const PAGE_SIZE = 20

const TRADERS = [
  { id: '', label: 'All Traders' },
  { id: 'trader_1', label: 'T1 — MACD+ICT' },
  { id: 'trader_2', label: 'T2 — SMC Confluence' },
  { id: 'trader_3', label: 'T3 — Breakout Retest' },
  { id: 'trader_4', label: 'T4 — News Fade' },
]

const DetailRow = ({ trade }) => {
  const payload = trade.payload || trade
  const meta = payload.metadata || payload.signal_metadata || trade.signal_metadata || trade.metadata || {}
  const fields = [
    { label: 'Trader', value: payload.trader || payload.trader_id || trade.trader || trade.trader_id },
    { label: 'Strategy', value: meta.strategy },
    { label: 'Confidence', value: payload.confidence != null ? formatPercent(payload.confidence) : meta.confidence != null ? formatPercent(meta.confidence) : '—' },
    { label: 'SMC Score', value: meta.smc_score ?? '—' },
    { label: 'ICT Conditions', value: meta.ict_conditions?.join(', ') || '—' },
    { label: 'Pattern', value: meta.pattern || meta.entry_reason || '—' },
    { label: 'Entry Reason', value: meta.entry_reason || '—' },
    { label: 'Exit Reason', value: meta.exit_reason || '—' },
    { label: 'ML Scores', value: meta.ml_scores ? JSON.stringify(meta.ml_scores) : '—' },
    { label: 'Correlation ID', value: payload.correlation_id || trade.correlation_id || meta.correlation_id || '—' },
    { label: 'Stop Loss', value: payload.stop_loss ? formatCurrency(payload.stop_loss) : '—' },
    { label: 'Take Profit', value: payload.take_profit ? formatCurrency(payload.take_profit) : '—' },
    { label: 'Commission', value: payload.commission != null ? formatCurrency(payload.commission) : '—' },
  ]

  return (
    <Box
      sx={{
        p: 2,
        backgroundColor: 'rgba(148,163,184,0.04)',
        borderTop: '1px solid rgba(148,163,184,0.1)',
      }}
    >
      <Grid container spacing={1.5}>
        {fields
          .filter((f) => f.value && f.value !== '—')
          .map(({ label, value }) => (
            <Grid item xs={6} sm={4} md={3} key={label}>
              <Typography variant="caption" color="text.secondary" display="block">
                {label}
              </Typography>
              <Typography
                variant="caption"
                fontWeight={500}
                sx={{ wordBreak: 'break-word', fontFamily: label.includes('ID') ? 'monospace' : 'inherit' }}
              >
                {value}
              </Typography>
            </Grid>
          ))}
      </Grid>
    </Box>
  )
}

const TradeRow = ({ trade }) => {
  const [expanded, setExpanded] = useState(false)
  const payload = trade.payload || trade
  const isBuy = (payload.side || payload.direction || payload.type || trade.side || trade.direction || '').toLowerCase() === 'buy'
  const pnl = payload.pnl ?? payload.profit ?? trade.pnl ?? trade.profit ?? trade.net_pnl ?? 0
  const timeStr = payload.timestamp || trade.timestamp || payload.closed_at || payload.opened_at
    ? new Date(payload.timestamp || trade.timestamp || payload.closed_at || payload.opened_at).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' })
    : '—'

  return (
    <>
      <TableRow
        sx={{
          '& td': { py: 1, fontSize: '0.8rem' },
          cursor: 'pointer',
          '&:hover': { backgroundColor: 'rgba(148,163,184,0.04)' },
        }}
        onClick={() => setExpanded((e) => !e)}
      >
        <TableCell sx={{ color: 'text.secondary', fontSize: '0.72rem !important' }}>{timeStr}</TableCell>
        <TableCell>
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
            {payload.trader || payload.trader_id || trade.trader || trade.trader_id || '—'}
          </Typography>
        </TableCell>
        <TableCell sx={{ fontWeight: 700 }}>{payload.symbol || trade.symbol || payload.pair || trade.pair}</TableCell>
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
        <TableCell sx={{ color: 'text.secondary' }}>{formatCurrency(payload.entry_price || payload.entry || payload.average_price || payload.price)}</TableCell>
        <TableCell sx={{ color: 'text.secondary' }}>{payload.size || payload.quantity || payload.volume || '—'}</TableCell>
        <TableCell sx={{ color: 'text.secondary' }}>{payload.rr_ratio != null ? Number(payload.rr_ratio).toFixed(2) : '—'}</TableCell>
        <TableCell
          sx={{ fontWeight: 700, color: pnl >= 0 ? '#00C853' : '#D32F2F' }}
        >
          {pnl >= 0 ? '+' : ''}{formatCurrency(pnl)}
        </TableCell>
        <TableCell>
          <IconButton size="small" sx={{ p: 0.25 }}>
            {expanded ? <CollapseIcon sx={{ fontSize: 16 }} /> : <ExpandIcon sx={{ fontSize: 16 }} />}
          </IconButton>
        </TableCell>
      </TableRow>
      {expanded && (
        <TableRow>
          <TableCell colSpan={9} sx={{ p: 0, border: 0 }}>
            <DetailRow trade={trade} />
          </TableCell>
        </TableRow>
      )}
    </>
  )
}

const TradeHistoryPage = () => {
  const [trades, setTrades] = useState([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(true)
  const [filters, setFilters] = useState({ trader: '', symbol: '', side: '' })
  const [showFilters, setShowFilters] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const res = filters.trader
        ? await traderService.getTraderTrades(filters.trader, PAGE_SIZE)
        : await analyticsService.getTrades({
            limit: PAGE_SIZE,
            offset: (page - 1) * PAGE_SIZE,
          })
      const arr = Array.isArray(res) ? res : res?.trades || []
      const filtered = arr.filter((t) => {
        const payload = t.payload || t
        if (filters.symbol && !(payload.symbol || payload.pair || '').toLowerCase().includes(filters.symbol.toLowerCase())) return false
        if (filters.side && (payload.side || payload.direction || payload.type || '').toLowerCase() !== filters.side) return false
        return true
      })
      setTrades(filtered)
      setTotal(res?.total ?? arr.length)
    } catch {
      setTrades([])
    } finally {
      setLoading(false)
    }
  }, [page, filters])

  useEffect(() => {
    load()
  }, [load])

  const exportCSV = async () => {
    const headers = ['timestamp', 'trader', 'symbol', 'side', 'entry_price', 'size', 'rr_ratio', 'pnl']
    const escapeCell = (value) => `"${String(value ?? '').replaceAll('"', '""')}"`
    const rows = trades.map((trade) => {
      const payload = trade.payload || trade
      return [
        payload.timestamp || trade.timestamp || payload.closed_at || payload.opened_at || '',
        payload.trader || payload.trader_id || trade.trader || trade.trader_id || '',
        payload.symbol || trade.symbol || payload.pair || trade.pair || '',
        payload.side || payload.direction || payload.type || trade.side || trade.direction || trade.type || '',
        payload.entry_price || payload.entry || payload.average_price || payload.price || '',
        payload.size || payload.quantity || payload.volume || '',
        payload.rr_ratio ?? trade.rr_ratio ?? '',
        payload.pnl ?? payload.profit ?? trade.pnl ?? trade.profit ?? trade.net_pnl ?? '',
      ]
        .map(escapeCell)
        .join(',')
    })
    const blob = new Blob([`${headers.map(escapeCell).join(',')}\n${rows.join('\n')}`], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'trade_journal.csv'
    a.click()
    URL.revokeObjectURL(url)
  }

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE))

  return (
    <Container maxWidth="xl" className="fade-in" sx={{ py: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 3 }}>
        <Box>
          <Typography variant="h5" fontWeight={700}>
            Trade History
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {total > 0 ? `${total} trades` : 'No trades recorded yet'}
          </Typography>
        </Box>
        <Stack direction="row" spacing={1}>
          <Button
            size="small"
            variant="outlined"
            startIcon={<FilterIcon />}
            onClick={() => setShowFilters((f) => !f)}
          >
            Filters
          </Button>
          <Button size="small" variant="outlined" startIcon={<ExportIcon />} onClick={exportCSV}>
            Export CSV
          </Button>
        </Stack>
      </Box>

      {/* Filters */}
      <Collapse in={showFilters}>
        <Paper className="theme-panel" sx={{ p: 2, mb: 2 }}>
          <Grid container spacing={1.5} alignItems="center">
            <Grid item xs={12} sm={4}>
              <FormControl fullWidth size="small">
                <InputLabel>Trader</InputLabel>
                <Select
                  value={filters.trader}
                  label="Trader"
                  onChange={(e) => { setFilters((f) => ({ ...f, trader: e.target.value })); setPage(1) }}
                >
                  {TRADERS.map((t) => (
                    <MenuItem key={t.id} value={t.id}>{t.label}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={6} sm={3}>
              <TextField
                label="Symbol"
                size="small"
                fullWidth
                placeholder="e.g. EURUSD"
                value={filters.symbol}
                onChange={(e) => { setFilters((f) => ({ ...f, symbol: e.target.value })); setPage(1) }}
              />
            </Grid>
            <Grid item xs={6} sm={3}>
              <FormControl fullWidth size="small">
                <InputLabel>Side</InputLabel>
                <Select
                  value={filters.side}
                  label="Side"
                  onChange={(e) => { setFilters((f) => ({ ...f, side: e.target.value })); setPage(1) }}
                >
                  <MenuItem value="">All</MenuItem>
                  <MenuItem value="buy">Buy</MenuItem>
                  <MenuItem value="sell">Sell</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={2}>
              <Button
                fullWidth
                size="small"
                variant="outlined"
                onClick={() => { setFilters({ trader: '', symbol: '', side: '' }); setPage(1) }}
              >
                Reset
              </Button>
            </Grid>
          </Grid>
        </Paper>
      </Collapse>

      <Paper className="theme-panel" sx={{ overflow: 'hidden' }}>
        <Table size="small">
          <TableHead>
            <TableRow sx={{ '& th': { fontWeight: 600, color: 'text.secondary', fontSize: '0.72rem', py: 1 } }}>
              <TableCell>Time</TableCell>
              <TableCell>Trader</TableCell>
              <TableCell>Symbol</TableCell>
              <TableCell>Side</TableCell>
              <TableCell>Entry</TableCell>
              <TableCell>Size</TableCell>
              <TableCell>R:R</TableCell>
              <TableCell>P&L</TableCell>
              <TableCell />
            </TableRow>
          </TableHead>
          <TableBody>
            {loading
              ? [...Array(8)].map((_, i) => (
                  <TableRow key={i}>
                    {[...Array(9)].map((_, j) => (
                      <TableCell key={j}>
                        <Skeleton variant="text" width={j === 0 ? 90 : 60} />
                      </TableCell>
                    ))}
                  </TableRow>
                ))
              : trades.length === 0
              ? (
                  <TableRow>
                    <TableCell colSpan={9} sx={{ textAlign: 'center', py: 4, color: 'text.secondary' }}>
                      No trades found
                    </TableCell>
                  </TableRow>
                )
              : trades.map((t, i) => <TradeRow key={t.id || i} trade={t} />)}
          </TableBody>
        </Table>
        {totalPages > 1 && (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 2, borderTop: '1px solid rgba(148,163,184,0.1)' }}>
            <Pagination
              count={totalPages}
              page={page}
              onChange={(_, v) => setPage(v)}
              size="small"
              color="primary"
            />
          </Box>
        )}
      </Paper>

      <Typography variant="caption" color="text.disabled" sx={{ display: 'block', mt: 1, textAlign: 'right' }}>
        Click any row to expand strategy details from trade_journal_detailed.jsonl
      </Typography>
    </Container>
  )
}

export default TradeHistoryPage
