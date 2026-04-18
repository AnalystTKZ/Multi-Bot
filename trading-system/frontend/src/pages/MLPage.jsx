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
  CircularProgress,
  LinearProgress,
  Alert,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  Tooltip,
  IconButton,
} from '@mui/material'
import {
  Refresh as RefreshIcon,
  PlayArrow as RetrainIcon,
  CheckCircle as OkIcon,
  Warning as WarnIcon,
  Error as ErrIcon,
  Psychology as RLIcon,
} from '@mui/icons-material'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RTooltip, ResponsiveContainer, LineChart, Line } from 'recharts'
import systemService from '@services/systemService'
import { formatPercent } from '@utils/formatters'

const MODEL_DEFS = [
  { id: 'price_predictor', name: 'Price Predictor', algo: 'LightGBM', role: 'Short-term direction (OHLCV features)' },
  { id: 'pattern_recognizer', name: 'Pattern Recognizer', algo: 'LightGBM', role: 'Candlestick & SMC structure' },
  { id: 'ml_signal_filter', name: 'Signal Filter', algo: 'LightGBM', role: 'Quality gate — trade outcome labels' },
  { id: 'anomaly_detector', name: 'Anomaly Detector', algo: 'IsolationForest', role: 'Regime anomaly flag (5% contamination)' },
  { id: 'sentiment_analyzer', name: 'Sentiment Analyzer', algo: 'VADER + Lexicon', role: 'Macro/news sentiment scores' },
]

const StatusIcon = ({ status }) => {
  if (status === 'ok') return <OkIcon sx={{ fontSize: 16, color: '#00C853' }} />
  if (status === 'warn') return <WarnIcon sx={{ fontSize: 16, color: '#FF6F00' }} />
  return <ErrIcon sx={{ fontSize: 16, color: '#D32F2F' }} />
}

const ModelCard = ({ model, onRetrain, retraining }) => {
  const accuracy = model.accuracy != null ? model.accuracy : null
  const passRate = model.pass_rate != null ? model.pass_rate : null
  const age = model.last_retrain ? model.last_retrain : 'Never'

  return (
    <Paper
      className="theme-panel"
      sx={{ p: 2, border: '1px solid rgba(148,163,184,0.12)', height: '100%' }}
    >
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
        <Box>
          <Typography variant="body2" fontWeight={700}>
            {model.name || model.id}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {MODEL_DEFS.find((d) => d.id === model.id)?.algo || 'Unknown'}
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
          <StatusIcon status={model.status || 'ok'} />
          <Tooltip title="Trigger retrain">
            <span>
              <IconButton
                size="small"
                disabled={retraining === model.id}
                onClick={() => onRetrain(model.id)}
              >
                {retraining === model.id ? (
                  <CircularProgress size={14} />
                ) : (
                  <RetrainIcon sx={{ fontSize: 14 }} />
                )}
              </IconButton>
            </span>
          </Tooltip>
        </Box>
      </Box>

      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1.5 }}>
        {MODEL_DEFS.find((d) => d.id === model.id)?.role}
      </Typography>

      <Divider sx={{ borderColor: 'rgba(148,163,184,0.1)', mb: 1.5 }} />

      <Grid container spacing={1}>
        {accuracy != null && (
          <Grid item xs={6}>
            <Typography variant="caption" color="text.secondary" display="block">
              Accuracy
            </Typography>
            <Typography variant="body2" fontWeight={700} sx={{ color: accuracy >= 0.6 ? '#00C853' : '#FF6F00' }}>
              {formatPercent(accuracy)}
            </Typography>
          </Grid>
        )}
        {passRate != null && (
          <Grid item xs={6}>
            <Typography variant="caption" color="text.secondary" display="block">
              Pass Rate
            </Typography>
            <Typography variant="body2" fontWeight={700}>
              {formatPercent(passRate)}
            </Typography>
          </Grid>
        )}
        <Grid item xs={12}>
          <Typography variant="caption" color="text.secondary" display="block">
            Last Retrain
          </Typography>
          <Typography variant="caption" fontWeight={500}>
            {age}
          </Typography>
        </Grid>
        {model.feature_count != null && (
          <Grid item xs={6}>
            <Typography variant="caption" color="text.secondary" display="block">
              Features
            </Typography>
            <Typography variant="caption" fontWeight={500}>
              {model.feature_count}
            </Typography>
          </Grid>
        )}
        {model.training_samples != null && (
          <Grid item xs={6}>
            <Typography variant="caption" color="text.secondary" display="block">
              Samples
            </Typography>
            <Typography variant="caption" fontWeight={500}>
              {model.training_samples.toLocaleString()}
            </Typography>
          </Grid>
        )}
      </Grid>
    </Paper>
  )
}

const MOCK_MODELS = MODEL_DEFS.map((d) => ({
  id: d.id,
  name: d.name,
  status: 'ok',
  accuracy: d.id === 'anomaly_detector' || d.id === 'sentiment_analyzer' ? null : 0.62,
  pass_rate: d.id === 'ml_signal_filter' ? 0.41 : null,
  last_retrain: 'Never (ML_ENABLED=false)',
  feature_count: d.id === 'price_predictor' ? 24 : d.id === 'pattern_recognizer' ? 31 : null,
  training_samples: null,
}))

const MOCK_RL = {
  episodes: 0,
  epsilon: 0.30,
  q_table_size: 0,
  win_rate: null,
  avg_reward: null,
  last_update: 'Never',
  recent_rewards: [],
}

const MLPage = () => {
  const [models, setModels] = useState(MOCK_MODELS)
  const [rl, setRL] = useState(MOCK_RL)
  const [loading, setLoading] = useState(false)
  const [retraining, setRetraining] = useState(null)
  const [error, setError] = useState(null)
  const [ensembleInfo] = useState({
    formula: 'score = ict × 0.6 + ml × 0.4 + sentiment_bonus − anomaly_penalty',
    mlEnabled: false,
  })

  const fetchData = async () => {
    setLoading(true)
    setError(null)
    try {
      const [modelsData, rlData] = await Promise.allSettled([
        systemService.getMLModels(),
        systemService.getRLAgentStats(),
      ])
      if (modelsData.status === 'fulfilled' && modelsData.value?.models) {
        setModels(modelsData.value.models)
      }
      if (rlData.status === 'fulfilled') {
        setRL(rlData.value)
      }
    } catch {
      setError('Could not reach backend — showing cached state')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 30000)
    return () => clearInterval(interval)
  }, [])

  const handleRetrain = async (modelId) => {
    setRetraining(modelId)
    try {
      await systemService.triggerRetrain(modelId)
      setError(null)
    } catch {
      setError(`Failed to trigger retrain for ${modelId}`)
    } finally {
      setRetraining(null)
    }
  }

  const recentRewardData = (rl.recent_rewards || []).map((r, i) => ({ i, reward: r }))

  return (
    <Container maxWidth="xl" className="fade-in" sx={{ py: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 3 }}>
        <Box>
          <Typography variant="h5" fontWeight={700}>
            ML / AI Layer
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Model status, RL agent, and ensemble configuration
          </Typography>
        </Box>
        <Button
          size="small"
          startIcon={loading ? <CircularProgress size={14} color="inherit" /> : <RefreshIcon />}
          onClick={fetchData}
          disabled={loading}
          variant="outlined"
        >
          Refresh
        </Button>
      </Box>

      {error && (
        <Alert severity="warning" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Ensemble summary */}
      <Paper className="theme-panel" sx={{ p: 2.5, mb: 3, borderLeft: `3px solid ${ensembleInfo.mlEnabled ? '#26e0b8' : '#FF6F00'}` }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 1 }}>
          <Box>
            <Typography variant="subtitle2" fontWeight={700} gutterBottom>
              Ensemble Formula
            </Typography>
            <Typography
              variant="body2"
              sx={{ fontFamily: 'monospace', color: '#26e0b8', fontSize: '0.8rem' }}
            >
              {ensembleInfo.formula}
            </Typography>
          </Box>
          <Chip
            label={ensembleInfo.mlEnabled ? 'ML ENABLED' : 'ML DISABLED (ML_ENABLED=false)'}
            color={ensembleInfo.mlEnabled ? 'success' : 'warning'}
            size="small"
            variant="outlined"
          />
        </Box>
      </Paper>

      {/* Model cards */}
      <Typography variant="subtitle1" fontWeight={700} sx={{ mb: 2 }}>
        Models ({models.length})
      </Typography>
      <Grid container spacing={2} sx={{ mb: 3 }}>
        {models.map((m) => (
          <Grid item xs={12} sm={6} md={4} key={m.id}>
            <ModelCard model={m} onRetrain={handleRetrain} retraining={retraining} />
          </Grid>
        ))}
      </Grid>

      {/* RL Agent */}
      <Typography variant="subtitle1" fontWeight={700} sx={{ mb: 2 }}>
        RL Agent — Online Q-Learning
      </Typography>
      <Grid container spacing={3}>
        <Grid item xs={12} md={5}>
          <Paper className="theme-panel" sx={{ p: 2.5 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
              <RLIcon sx={{ color: '#26e0b8' }} />
              <Typography variant="subtitle2" fontWeight={700}>
                Q-Learning Agent
              </Typography>
            </Box>
            <Grid container spacing={1.5}>
              {[
                { label: 'Episodes (Trades)', value: rl.episodes ?? 0 },
                {
                  label: 'Epsilon (ε)',
                  value: (rl.epsilon ?? 0.3).toFixed(3),
                  note: 'Exploration rate — decreases with episodes',
                },
                { label: 'Q-Table States', value: (rl.q_table_size ?? 0).toLocaleString() },
                { label: 'Last Update', value: rl.last_update || 'Never' },
                {
                  label: 'Win Rate',
                  value: rl.win_rate != null ? formatPercent(rl.win_rate) : '—',
                  color: rl.win_rate != null ? (rl.win_rate >= 0.5 ? '#00C853' : '#FF6F00') : undefined,
                },
                {
                  label: 'Avg Reward',
                  value: rl.avg_reward != null ? rl.avg_reward.toFixed(3) : '—',
                  color: rl.avg_reward != null ? (rl.avg_reward >= 0 ? '#00C853' : '#D32F2F') : undefined,
                },
              ].map(({ label, value, note, color }) => (
                <Grid item xs={6} key={label}>
                  <Typography variant="caption" color="text.secondary" display="block">
                    {label}
                  </Typography>
                  <Tooltip title={note || ''}>
                    <Typography variant="body2" fontWeight={700} sx={{ color: color || 'text.primary' }}>
                      {value}
                    </Typography>
                  </Tooltip>
                </Grid>
              ))}
            </Grid>

            {rl.episodes < 50 && (
              <Alert severity="info" sx={{ mt: 2, fontSize: '0.72rem' }}>
                Agent is in exploration phase (ε=high). Needs ~500 trades to converge.
              </Alert>
            )}

            {rl.episodes >= 500 && (
              <Alert severity="success" sx={{ mt: 2, fontSize: '0.72rem' }}>
                Agent has sufficient trade history. Q-table quality should be evaluated.
              </Alert>
            )}
          </Paper>
        </Grid>

        <Grid item xs={12} md={7}>
          <Paper className="theme-panel" sx={{ p: 2.5, height: '100%' }}>
            <Typography variant="subtitle2" fontWeight={600} gutterBottom>
              Recent Rewards
            </Typography>
            {recentRewardData.length === 0 ? (
              <Box
                sx={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
              >
                <Typography variant="body2" color="text.secondary">
                  No trades recorded yet — rewards will appear after first paper trades
                </Typography>
              </Box>
            ) : (
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={recentRewardData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                  <XAxis dataKey="i" hide />
                  <YAxis tick={{ fontSize: 10, fill: '#94a3b8' }} />
                  <RTooltip
                    contentStyle={{ background: '#111a2e', border: '1px solid rgba(148,163,184,0.2)' }}
                    labelStyle={{ color: '#94a3b8' }}
                  />
                  <Line
                    type="monotone"
                    dataKey="reward"
                    stroke="#26e0b8"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}

            <Divider sx={{ my: 2, borderColor: 'rgba(148,163,184,0.1)' }} />

            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
              HYPERPARAMETERS
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              {[
                { k: 'α (learning rate)', v: '0.15' },
                { k: 'γ (discount)', v: '0.95' },
                { k: 'ε_max', v: '0.30' },
                { k: 'ε_min', v: '0.05' },
                { k: 'ε_decay', v: '0.995' },
                { k: 'State features', v: '12' },
                { k: 'Actions', v: '3 (BUY/SELL/HOLD)' },
              ].map(({ k, v }) => (
                <Box
                  key={k}
                  sx={{
                    px: 1.5,
                    py: 0.5,
                    borderRadius: 1,
                    border: '1px solid rgba(148,163,184,0.15)',
                    backgroundColor: 'rgba(148,163,184,0.04)',
                  }}
                >
                  <Typography variant="caption" color="text.secondary">
                    {k}:{' '}
                  </Typography>
                  <Typography variant="caption" fontWeight={700}>
                    {v}
                  </Typography>
                </Box>
              ))}
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  )
}

export default MLPage
