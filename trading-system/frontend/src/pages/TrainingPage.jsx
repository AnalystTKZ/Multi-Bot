import React, { useState, useRef, useEffect } from 'react'
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
  Slider,
  CircularProgress,
  Alert,
  LinearProgress,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  Stack,
  IconButton,
  Tooltip,
} from '@mui/material'
import {
  CloudUpload as UploadIcon,
  Delete as DeleteIcon,
  PlayArrow as TrainIcon,
  CheckCircle as DoneIcon,
  Info as InfoIcon,
} from '@mui/icons-material'
import systemService from '@services/systemService'

const MODELS = [
  {
    id: 'price_predictor',
    name: 'Price Predictor',
    algo: 'LightGBM',
    description: 'Short-term direction from OHLCV features',
  },
  {
    id: 'pattern_recognizer',
    name: 'Pattern Recognizer',
    algo: 'LightGBM',
    description: 'Candlestick & SMC structure patterns',
  },
  {
    id: 'ml_signal_filter',
    name: 'Signal Filter',
    algo: 'LightGBM',
    description: 'Binary gate trained on paper trade outcomes',
  },
  {
    id: 'anomaly_detector',
    name: 'Anomaly Detector',
    algo: 'IsolationForest',
    description: 'Regime anomaly detection',
  },
]

const TRAINING_METHODS = [
  { id: 'full', label: 'Full Retrain', description: 'Discard existing weights and train from scratch' },
  { id: 'incremental', label: 'Incremental', description: 'Warm-start from existing weights (LightGBM only)' },
  { id: 'journal', label: 'Journal Labels', description: 'Retrain signal filter using trade journal outcomes only' },
]

const SYMBOLS_EXAMPLE = 'EURUSD,GBPUSD,USDJPY,USDCHF,USDCAD,AUDUSD,NZDUSD,EURGBP,EURJPY,GBPJPY,XAUUSD'

const formatBytes = (bytes) => {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

const TrainingPage = () => {
  const fileInputRef = useRef(null)
  const [files, setFiles] = useState([])
  const [uploadProgress, setUploadProgress] = useState({})
  const [uploading, setUploading] = useState(false)
  const [training, setTraining] = useState(false)
  const [status, setStatus] = useState(null) // { type: 'success'|'error'|'info', message }
  const [trainingEvents, setTrainingEvents] = useState([])
  const [config, setConfig] = useState({
    models: ['price_predictor', 'pattern_recognizer', 'ml_signal_filter'],
    method: 'incremental',
    train_split: 70,
    val_split: 15,
    // test_split is derived: 100 - train - val
    lookback_days: 180,
  })

  const testSplit = Math.max(0, 100 - config.train_split - config.val_split)

  useEffect(() => {
    let active = true
    const loadEvents = async () => {
      try {
        const data = await systemService.getTrainingStatus()
        if (active) setTrainingEvents(Array.isArray(data?.events) ? data.events : [])
      } catch {
        if (active) setTrainingEvents([])
      }
    }
    loadEvents()
    const id = setInterval(loadEvents, 10000)
    return () => {
      active = false
      clearInterval(id)
    }
  }, [])

  const handleFileSelect = (e) => {
    const selected = Array.from(e.target.files).filter((f) => f.name.endsWith('.csv'))
    if (selected.length === 0) {
      setStatus({ type: 'warning', message: 'Only CSV files are accepted.' })
      return
    }
    setFiles((prev) => {
      const names = new Set(prev.map((f) => f.name))
      return [...prev, ...selected.filter((f) => !names.has(f.name))]
    })
    e.target.value = ''
  }

  const removeFile = (name) => setFiles((f) => f.filter((x) => x.name !== name))

  const uploadFiles = async () => {
    if (!files.length) return
    setUploading(true)
    setStatus(null)
    const results = []
    for (const file of files) {
      const fd = new FormData()
      fd.append('file', file)
      try {
        await systemService.uploadTrainingData(fd, (pct) =>
          setUploadProgress((p) => ({ ...p, [file.name]: pct }))
        )
        results.push({ file: file.name, ok: true })
      } catch {
        results.push({ file: file.name, ok: false })
      }
    }
    setUploading(false)
    const failed = results.filter((r) => !r.ok)
    if (failed.length === 0) {
      setStatus({ type: 'success', message: `${files.length} file(s) uploaded successfully.` })
    } else {
      setStatus({
        type: 'warning',
        message: `${results.length - failed.length} uploaded, ${failed.length} failed: ${failed.map((r) => r.file).join(', ')}`,
      })
    }
  }

  const startTraining = async () => {
    if (!config.models.length) {
      setStatus({ type: 'error', message: 'Select at least one model to train.' })
      return
    }
    setTraining(true)
    setStatus(null)
    try {
      const payload = {
        ...config,
        test_split: testSplit,
      }
      await systemService.startTraining(payload)
      setStatus({ type: 'success', message: 'Training job started. Monitor progress in the ML/AI page.' })
    } catch (e) {
      setStatus({ type: 'error', message: e?.message || 'Failed to start training.' })
    } finally {
      setTraining(false)
    }
  }

  const toggleModel = (id) => {
    setConfig((c) => ({
      ...c,
      models: c.models.includes(id) ? c.models.filter((m) => m !== id) : [...c.models, id],
    }))
  }

  return (
    <Container maxWidth="xl" className="fade-in" sx={{ py: 3 }}>
      <Box sx={{ mb: 3 }}>
        <Typography variant="h5" fontWeight={700}>
          Training
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Upload historical data and retrain ML models
        </Typography>
      </Box>

      <Grid container spacing={3}>
        {/* Data upload */}
        <Grid item xs={12} lg={5}>
          <Paper className="theme-panel" sx={{ p: 3, mb: 3 }}>
            <Typography variant="subtitle1" fontWeight={700} gutterBottom>
              Training Data
            </Typography>
            <Divider sx={{ mb: 2, borderColor: 'rgba(148,163,184,0.1)' }} />
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 2 }}>
              Upload OHLCV CSV files. Expected columns: timestamp, open, high, low, close, volume. One file per
              symbol (e.g. EURUSD.csv).
            </Typography>

            <Box
              onClick={() => fileInputRef.current?.click()}
              sx={{
                border: '2px dashed rgba(38,224,184,0.3)',
                borderRadius: 2,
                p: 3,
                textAlign: 'center',
                cursor: 'pointer',
                transition: 'border-color 0.2s',
                '&:hover': { borderColor: '#26e0b8' },
                mb: 2,
              }}
            >
              <UploadIcon sx={{ fontSize: 36, color: 'rgba(38,224,184,0.5)', mb: 1 }} />
              <Typography variant="body2" color="text.secondary">
                Click to select CSV files
              </Typography>
              <Typography variant="caption" color="text.secondary">
                or drag & drop
              </Typography>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".csv"
                style={{ display: 'none' }}
                onChange={handleFileSelect}
              />
            </Box>

            {files.length > 0 && (
              <Box sx={{ mb: 2 }}>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell sx={{ color: 'text.secondary', fontSize: '0.7rem' }}>File</TableCell>
                      <TableCell sx={{ color: 'text.secondary', fontSize: '0.7rem' }}>Size</TableCell>
                      <TableCell sx={{ color: 'text.secondary', fontSize: '0.7rem' }}>Progress</TableCell>
                      <TableCell />
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {files.map((f) => (
                      <TableRow key={f.name} sx={{ '&:last-child td': { border: 0 } }}>
                        <TableCell sx={{ fontSize: '0.78rem' }}>{f.name}</TableCell>
                        <TableCell sx={{ fontSize: '0.78rem', color: 'text.secondary' }}>
                          {formatBytes(f.size)}
                        </TableCell>
                        <TableCell sx={{ minWidth: 80 }}>
                          {uploadProgress[f.name] != null ? (
                            uploadProgress[f.name] === 100 ? (
                              <DoneIcon sx={{ fontSize: 16, color: '#00C853' }} />
                            ) : (
                              <LinearProgress
                                variant="determinate"
                                value={uploadProgress[f.name]}
                                sx={{ height: 4, borderRadius: 2 }}
                              />
                            )
                          ) : (
                            <Typography variant="caption" color="text.disabled">
                              —
                            </Typography>
                          )}
                        </TableCell>
                        <TableCell align="right" sx={{ p: 0.5 }}>
                          <IconButton size="small" onClick={() => removeFile(f.name)}>
                            <DeleteIcon sx={{ fontSize: 14 }} />
                          </IconButton>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </Box>
            )}

            <Button
              variant="outlined"
              size="small"
              fullWidth
              startIcon={uploading ? <CircularProgress size={14} color="inherit" /> : <UploadIcon />}
              disabled={!files.length || uploading}
              onClick={uploadFiles}
            >
              {uploading ? 'Uploading…' : `Upload ${files.length} file${files.length !== 1 ? 's' : ''}`}
            </Button>
          </Paper>
        </Grid>

        {/* Training config */}
        <Grid item xs={12} lg={7}>
          <Paper className="theme-panel" sx={{ p: 3, mb: 3 }}>
            <Typography variant="subtitle1" fontWeight={700} gutterBottom>
              Training Configuration
            </Typography>
            <Divider sx={{ mb: 2.5, borderColor: 'rgba(148,163,184,0.1)' }} />

            {/* Models */}
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1.5 }}>
              MODELS TO TRAIN
            </Typography>
            <Grid container spacing={1} sx={{ mb: 3 }}>
              {MODELS.map((m) => {
                const active = config.models.includes(m.id)
                return (
                  <Grid item xs={12} sm={6} key={m.id}>
                    <Box
                      onClick={() => toggleModel(m.id)}
                      sx={{
                        p: 1.5,
                        borderRadius: 2,
                        border: `1px solid ${active ? 'rgba(38,224,184,0.4)' : 'rgba(148,163,184,0.15)'}`,
                        backgroundColor: active ? 'rgba(38,224,184,0.07)' : 'transparent',
                        cursor: 'pointer',
                        transition: 'all 0.15s',
                        '&:hover': { borderColor: 'rgba(38,224,184,0.4)' },
                      }}
                    >
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Typography variant="body2" fontWeight={600} sx={{ color: active ? '#26e0b8' : 'text.primary' }}>
                          {m.name}
                        </Typography>
                        <Chip
                          label={m.algo}
                          size="small"
                          sx={{ fontSize: '0.65rem', height: 18, backgroundColor: 'rgba(148,163,184,0.1)' }}
                        />
                      </Box>
                      <Typography variant="caption" color="text.secondary">
                        {m.description}
                      </Typography>
                    </Box>
                  </Grid>
                )
              })}
            </Grid>

            {/* Training method */}
            <FormControl fullWidth size="small" sx={{ mb: 3 }}>
              <InputLabel>Training Method</InputLabel>
              <Select
                value={config.method}
                label="Training Method"
                onChange={(e) => setConfig((c) => ({ ...c, method: e.target.value }))}
              >
                {TRAINING_METHODS.map((m) => (
                  <MenuItem key={m.id} value={m.id}>
                    <Box>
                      <Typography variant="body2" fontWeight={600}>
                        {m.label}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {m.description}
                      </Typography>
                    </Box>
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            {/* Splits */}
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
              DATA SPLITS
            </Typography>
            <Stack direction="row" spacing={2} alignItems="center" sx={{ mb: 1.5 }}>
              {[
                { label: 'Train', key: 'train_split', color: '#26e0b8' },
                { label: 'Validation', key: 'val_split', color: '#1976D2' },
              ].map(({ label, key, color }) => (
                <Box key={key} sx={{ flex: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="caption" color="text.secondary">
                      {label}
                    </Typography>
                    <Typography variant="caption" sx={{ color }}>
                      {config[key]}%
                    </Typography>
                  </Box>
                  <Slider
                    size="small"
                    value={config[key]}
                    min={10}
                    max={85}
                    step={5}
                    onChange={(_, v) => setConfig((c) => ({ ...c, [key]: v }))}
                    sx={{ color, mt: 0.5 }}
                  />
                </Box>
              ))}
              <Box sx={{ minWidth: 80, textAlign: 'center' }}>
                <Typography variant="caption" color="text.secondary">
                  Test
                </Typography>
                <Typography
                  variant="body2"
                  fontWeight={700}
                  sx={{ color: testSplit >= 10 ? '#FF6F00' : '#D32F2F' }}
                >
                  {testSplit}%
                </Typography>
              </Box>
            </Stack>
            {testSplit < 5 && (
              <Alert severity="warning" sx={{ mb: 2, fontSize: '0.75rem' }}>
                Test split is very small ({testSplit}%). Reduce train or validation.
              </Alert>
            )}

            {/* Lookback */}
            <Grid container spacing={1.5} sx={{ mb: 3 }}>
              <Grid item xs={12} sm={6}>
                <TextField
                  label="Lookback Window (days)"
                  type="number"
                  size="small"
                  fullWidth
                  value={config.lookback_days}
                  onChange={(e) => setConfig((c) => ({ ...c, lookback_days: Number(e.target.value) }))}
                  helperText="Rolling window for training data"
                />
              </Grid>
            </Grid>

            {status && (
              <Alert severity={status.type} sx={{ mb: 2, fontSize: '0.78rem' }} onClose={() => setStatus(null)}>
                {status.message}
              </Alert>
            )}

            <Button
              variant="contained"
              fullWidth
              startIcon={training ? <CircularProgress size={16} color="inherit" /> : <TrainIcon />}
              disabled={training || !config.models.length}
              onClick={startTraining}
              sx={{ backgroundColor: '#FF6F00', '&:hover': { backgroundColor: '#e65100' } }}
            >
              {training ? 'Starting Training Job…' : 'Start Training'}
            </Button>
          </Paper>

          <Paper className="theme-panel" sx={{ p: 3 }}>
            <Typography variant="subtitle1" fontWeight={700} gutterBottom>
              Training Progress
            </Typography>
            <Divider sx={{ mb: 2.5, borderColor: 'rgba(148,163,184,0.1)' }} />
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 2 }}>
              Example symbol scope: {`RETRAIN_SYMBOLS_GRU="${SYMBOLS_EXAMPLE}"`} and
              {`RETRAIN_SYMBOLS_REGIME="${SYMBOLS_EXAMPLE}"`}
            </Typography>

            {trainingEvents.length === 0 ? (
              <Typography variant="caption" color="text.secondary">
                No training events yet. Start a retrain to see per-symbol progress here.
              </Typography>
            ) : (
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ color: 'text.secondary', fontSize: '0.7rem' }}>Time</TableCell>
                    <TableCell sx={{ color: 'text.secondary', fontSize: '0.7rem' }}>Model</TableCell>
                    <TableCell sx={{ color: 'text.secondary', fontSize: '0.7rem' }}>Symbol</TableCell>
                    <TableCell sx={{ color: 'text.secondary', fontSize: '0.7rem' }}>Status</TableCell>
                    <TableCell sx={{ color: 'text.secondary', fontSize: '0.7rem' }}>Samples</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {trainingEvents.slice(-8).reverse().map((evt, idx) => (
                    <TableRow key={`${evt.timestamp || 't'}-${idx}`} sx={{ '&:last-child td': { border: 0 } }}>
                      <TableCell sx={{ fontSize: '0.75rem', color: 'text.secondary' }}>
                        {evt.timestamp ? new Date(evt.timestamp).toLocaleString() : '—'}
                      </TableCell>
                      <TableCell sx={{ fontSize: '0.78rem' }}>{evt.model || evt.model_name || '—'}</TableCell>
                      <TableCell sx={{ fontSize: '0.78rem' }}>{evt.symbol || '—'}</TableCell>
                      <TableCell sx={{ fontSize: '0.78rem' }}>{evt.status || '—'}</TableCell>
                      <TableCell sx={{ fontSize: '0.78rem' }}>
                        {evt.samples != null ? evt.samples.toLocaleString() : '—'}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Container>
  )
}

export default TrainingPage
