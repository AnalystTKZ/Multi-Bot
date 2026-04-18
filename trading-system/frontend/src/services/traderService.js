import api from './api'

const traderService = {
  getAllTraders: () => api.get('/traders/'),
  getTraderById: (id) => api.get(`/traders/${id}`),
  getTraderPerformance: (id, period = '30d') =>
    api.get(`/traders/${id}/performance?period=${period}`),
  getTraderSignals: (id, limit = 10) => api.get(`/traders/${id}/signals?limit=${limit}`),
  getTraderTrades: (id, limit = 50) => api.get(`/traders/${id}/trades?limit=${limit}`),
  updateTraderStatus: (id, action) => api.patch(`/traders/${id}/status`, { action }),
  getTraderHistory: (id, period = '30d') => api.get(`/traders/${id}/history?period=${period}`),
}

export default traderService
