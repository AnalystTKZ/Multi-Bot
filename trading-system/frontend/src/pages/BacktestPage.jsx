import React, { useState, useEffect } from 'react'
import {
  Box,
  Container,
  Grid,
  Paper,
  Typography,
  Button,
  Chip,
  Divider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Checkbox,
  FormControlLabel,
  FormGroup,
  CircularProgress,
  Alert,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  LinearProgress,
} from '@mui/material'
import {
  ExpandMore as ExpandMoreIcon,
  PlayArrow as RunIcon,
  Assessment as ResultsIcon,
  Download as DownloadIcon,
} from '@mui/icons-material'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, Legend } from 'recharts'
import systemService from '@services/systemService'
import { formatCurrency, formatPercent } from '@utils/formatters'

const TRADERS = [
  { id: 1, name: 'T1 — MACD + ICT', description: '4H trend following' },
  { id: 2, name: 'T2 — SMC Confluence', description: '1H OTE + SMC scoring' },
  { id: 3, name: 'T3 — Breakout Retest', description: '15m breakout retest' },
  { id: 4, name: 'T4 — News Fade', description: '5m post-release fade' },
]

const SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'XAUUSD', 'USDCHF', 'NZDUSD']

const DEFAULT_CONFIG = {
  traders: [1, 2, 3],
  symbols: ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD'],
  start_date: '2024-01-01',
  end_date: '2025-01-01',
  initial_capital: 10000,
  commission_pct: 0.0002,
  slippage_pct: 0.0001,
}

const StatCard = ({ label, value, color, small }) => (
  <Box sx={{ textAlign: 'center', p: small ? 1 : 1.5 }}>
    <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
      {label}
    </Typography>
    <Typography
      variant={small ? 'body2' : 'h6'}
      sx={{ fontWeight: 700, color: color || 'text.primary', mt: 0.25 }}
    >
      {value}
    </Typography>
  </Box>
)

const BacktestPage = () => {
  const [config, setConfig] = useState(DEFAULT_CONFIG)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState(null)
  const [results, setResults] = useState(null)
  const [pastResults, setPastResults] = useState([])
  const [selectedResult, setSelectedResult] = useState(null)

  useEffect(() => {
    systemService.listBacktestResults().then((r) => setPastResults(r.results || [])).catch(() => {})
  }, [])

  const toggleTrader = (id) => {
    setConfig((c) => ({
      ...c,
      traders: c.traders.includes(id) ? c.traders.filter((t) => t !== id) : [...c.traders, id],
    }))
  }

  const toggleSymbol = (sym) => {
    setConfig((c) => ({
      ...c,
      symbols: c.symbols.includes(sym) ? c.symbols.filter((s) => s !== sym) : [...c.symbols, sym],
    }))
  }

  const runBacktest = async () => {
    if (!config.traders.length || !config.symbols.length) {
      setError('Select at least one trader and one symbol.')
      return
    }
    setRunning(true)
    setError(null)
    setResults(null)
    try {
      const res = await systemService.runBacktest(config)
      setResults(res)
      setPastResults((p) => [res, ...p])
    } catch (e) {
      setError(e?.message || 'Backtest failed — check that the engine container is running.')
    } finally {
      setRunning(false)
    }
  }

  const displayResults = selectedResult || results

  return (
    <Container maxWidth="xl" className="fade-in" sx={{ py: 3 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
        <Box>
          <Typography variant="h5" fontWeight={700}>
            Backtesting
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Run strategy simulations on historical OHLCV data
          </Typography>
        </Box>
        {pastResults.length > 0 && (
          <FormControl size="small" sx={{ minWidth: 220 }}>
            <InputLabel>Load past result</InputLabel>
            <Select
              value=""
              label="Load past result"
              onChange={(e) => {
                const r = pastResults.find((x) => x.id === e.target.value)
                setSelectedResult(r || null)
              }}
            >
              {pastResults.map((r, i) => (
                <MenuItem key={r.id || i} value={r.id || i}>
                  {r.timestamp || r.id || `Result ${i + 1}`}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        )}
      </Box>

      <Grid container spacing={3}>
        {/* Config panel */}
        <Grid item xs={12} lg={4}>
          <Paper className="theme-panel" sx={{ p: 3 }}>
            <Typography variant="subtitle1" fontWeight={700} gutterBottom>
              Configuration
            </Typography>
            <Divider sx={{ mb: 2, borderColor: 'rgba(148,163,184,0.1)' }} />

            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
              TRADERS
            </Typography>
            <FormGroup sx={{ mb: 2 }}>
              {TRADERS.map((t) => (
                <FormControlLabel
                  key={t.id}
                  control={
                    <Checkbox
                      checked={config.traders.includes(t.id)}
                      onChange={() => toggleTrader(t.id)}
                      size="small"
                      sx={{ '&.Mui-checked': { color: '#26e0b8' } }}
                    />
                  }
                  label={
                    <Box>
                      <Typography variant="body2" sx={{ fontWeight: 500 }}>
                        {t.name}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {t.description}
                      </Typography>
                    </Box>
                  }
                />
              ))}
            </FormGroup>

            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
              SYMBOLS
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75, mb: 2 }}>
              {SYMBOLS.map((sym) => (
                <Chip
                  key={sym}
                  label={sym}
                  size="small"
                  clickable
                  onClick={() => toggleSymbol(sym)}
                  variant={config.symbols.includes(sym) ? 'filled' : 'outlined'}
                  sx={
                    config.symbols.includes(sym)
                      ? { backgroundColor: 'rgba(38,224,184,0.15)', color: '#26e0b8', borderColor: '#26e0b8' }
                      : { color: 'text.secondary' }
                  }
                />
              ))}
            </Box>

            <Grid container spacing={1.5} sx={{ mb: 2 }}>
              <Grid item xs={6}>
                <TextField
                  label="Start Date"
                  type="date"
                  size="small"
                  fullWidth
                  value={config.start_date}
                  onChange={(e) => setConfig((c) => ({ ...c, start_date: e.target.value }))}
                  InputLabelProps={{ shrink: true }}
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  label="End Date"
                  type="date"
                  size="small"
                  fullWidth
                  value={config.end_date}
                  onChange={(e) => setConfig((c) => ({ ...c, end_date: e.target.value }))}
                  InputLabelProps={{ shrink: true }}
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  label="Initial Capital ($)"
                  type="number"
                  size="small"
                  fullWidth
                  value={config.initial_capital}
                  onChange={(e) => setConfig((c) => ({ ...c, initial_capital: Number(e.target.value) }))}
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  label="Commission (%)"
                  type="number"
                  size="small"
                  fullWidth
                  value={(config.commission_pct * 100).toFixed(3)}
                  onChange={(e) =>
                    setConfig((c) => ({ ...c, commission_pct: Number(e.target.value) / 100 }))
                  }
                  inputProps={{ step: 0.001 }}
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  label="Slippage (%)"
                  type="number"
                  size="small"
                  fullWidth
                  value={(config.slippage_pct * 100).toFixed(3)}
                  onChange={(e) =>
                    setConfig((c) => ({ ...c, slippage_pct: Number(e.target.value) / 100 }))
                  }
                  inputProps={{ step: 0.001 }}
                />
              </Grid>
            </Grid>

            {error && (
              <Alert severity="error" sx={{ mb: 2, fontSize: '0.78rem' }} onClose={() => setError(null)}>
                {error}
              </Alert>
            )}

            <Button
              fullWidth
              variant="contained"
              startIcon={running ? <CircularProgress size={16} color="inherit" /> : <RunIcon />}
              disabled={running}
              onClick={runBacktest}
              sx={{ backgroundColor: '#1976D2', '&:hover': { backgroundColor: '#1565c0' } }}
            >
              {running ? 'Running Backtest…' : 'Run Backtest'}
            </Button>
            {running && <LinearProgress sx={{ mt: 1, borderRadius: 1 }} />}
          </Paper>
        </Grid>

        {/* Results panel */}
        <Grid item xs={12} lg={8}>
          {!displayResults ? (
            <Paper
              className="theme-panel"
              sx={{ p: 4, textAlign: 'center', height: 400, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}
            >
              <ResultsIcon sx={{ fontSize: 56, color: 'rgba(148,163,184,0.2)', mb: 2 }} />
              <Typography color="text.secondary">
                Configure and run a backtest to see results
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                Most recent result: T2 (SMC) +43% aggregate, USDJPY +251%
              </Typography>
            </Paper>
          ) : (
            <Box>
              {/* Summary stats */}
              <Paper className="theme-panel" sx={{ p: 2.5, mb: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="subtitle1" fontWeight={700}>
                    Results — {displayResults.timestamp || 'Latest'}
                  </Typography>
                  <Button
                    size="small"
                    startIcon={<DownloadIcon />}
                    variant="outlined"
                    onClick={() => {
                      const blob = new Blob([JSON.stringify(displayResults, null, 2)], {
                        type: 'application/json',
                      })
                      const url = URL.createObjectURL(blob)
                      const a = document.createElement('a')
                      a.href = url
                      a.download = `backtest_${displayResults.timestamp || 'result'}.json`
                      a.click()
                    }}
                    sx={{ fontSize: '0.72rem' }}
                  >
                    Export JSON
                  </Button>
                </Box>
                <Grid container>
                  <Grid item xs={6} sm={3}>
                    <StatCard
                      label="Total Return"
                      value={formatPercent(displayResults.total_return ?? 0)}
                      color={(displayResults.total_return ?? 0) >= 0 ? '#00C853' : '#D32F2F'}
                    />
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <StatCard label="Win Rate" value={formatPercent(displayResults.win_rate ?? 0)} />
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <StatCard
                      label="Max Drawdown"
                      value={formatPercent(displayResults.max_drawdown ?? 0)}
                      color="#FF6F00"
                    />
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <StatCard label="Profit Factor" value={(displayResults.profit_factor ?? 0).toFixed(2)} />
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <StatCard label="Total Trades" value={displayResults.total_trades ?? 0} small />
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <StatCard label="Avg R:R" value={(displayResults.avg_rr ?? 0).toFixed(2)} small />
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <StatCard
                      label="Net P&L"
                      value={formatCurrency(displayResults.net_pnl ?? 0)}
                      color={(displayResults.net_pnl ?? 0) >= 0 ? '#00C853' : '#D32F2F'}
                      small
                    />
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <StatCard label="Sharpe Ratio" value={(displayResults.sharpe_ratio ?? 0).toFixed(2)} small />
                  </Grid>
                </Grid>
              </Paper>

              {/* Equity curve chart */}
              {displayResults.equity_curve?.length > 0 && (
                <Paper className="theme-panel" sx={{ p: 2.5, mb: 2 }}>
                  <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                    Equity Curve
                  </Typography>
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={displayResults.equity_curve}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                      <XAxis dataKey="date" tick={{ fontSize: 10, fill: '#94a3b8' }} />
                      <YAxis tick={{ fontSize: 10, fill: '#94a3b8' }} />
                      <Tooltip
                        contentStyle={{ background: '#111a2e', border: '1px solid rgba(148,163,184,0.2)' }}
                        labelStyle={{ color: '#94a3b8' }}
                      />
                      <Line type="monotone" dataKey="equity" stroke="#26e0b8" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </Paper>
              )}

              {/* Per-trader breakdown */}
              {displayResults.traders && (
                <Paper className="theme-panel" sx={{ p: 2.5 }}>
                  <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                    Per-Trader Breakdown
                  </Typography>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell sx={{ color: 'text.secondary', fontSize: '0.72rem' }}>Trader</TableCell>
                        <TableCell align="right" sx={{ color: 'text.secondary', fontSize: '0.72rem' }}>Return</TableCell>
                        <TableCell align="right" sx={{ color: 'text.secondary', fontSize: '0.72rem' }}>Win %</TableCell>
                        <TableCell align="right" sx={{ color: 'text.secondary', fontSize: '0.72rem' }}>Trades</TableCell>
                        <TableCell align="right" sx={{ color: 'text.secondary', fontSize: '0.72rem' }}>Max DD</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {Object.entries(displayResults.traders).map(([tid, td]) => (
                        <TableRow key={tid} sx={{ '&:last-child td': { border: 0 } }}>
                          <TableCell sx={{ fontWeight: 600, fontSize: '0.8rem' }}>{tid}</TableCell>
                          <TableCell
                            align="right"
                            sx={{ color: (td.return ?? 0) >= 0 ? '#00C853' : '#D32F2F', fontSize: '0.8rem', fontWeight: 600 }}
                          >
                            {formatPercent(td.return ?? 0)}
                          </TableCell>
                          <TableCell align="right" sx={{ fontSize: '0.8rem' }}>
                            {formatPercent(td.win_rate ?? 0)}
                          </TableCell>
                          <TableCell align="right" sx={{ fontSize: '0.8rem' }}>
                            {td.total_trades ?? 0}
                          </TableCell>
                          <TableCell align="right" sx={{ color: '#FF6F00', fontSize: '0.8rem' }}>
                            {formatPercent(td.max_drawdown ?? 0)}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </Paper>
              )}
            </Box>
          )}
        </Grid>
      </Grid>
    </Container>
  )
}

export default BacktestPage
