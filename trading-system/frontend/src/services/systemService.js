import api from './api'

const systemService = {
  getSystemStatus: () => api.get('/system/status'),

  setMode: (mode) => api.post('/system/mode', { mode }),

  getHealth: () => api.get('/health'),

  getBacktestConfig: () => api.get('/backtest/config'),

  runBacktest: (config) => api.post('/backtest/run', config),

  getBacktestResults: (id) => api.get(`/backtest/results/${id}`),

  listBacktestResults: () => api.get('/backtest/results'),

  uploadTrainingData: (formData, onProgress) =>
    api.post('/training/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (e) => {
        if (onProgress) onProgress(Math.round((e.loaded * 100) / e.total))
      },
    }),

  startTraining: (config) => api.post('/training/start', config),

  getTrainingStatus: () => api.get('/training/status'),

  getMLModels: () => api.get('/ml/models'),

  getMLModelDetail: (modelId) => api.get(`/ml/models/${modelId}`),

  getRLAgentStats: () => api.get('/ml/rl-agent'),

  triggerRetrain: (modelId) => api.post(`/ml/models/${modelId}/retrain`),
}

export default systemService
