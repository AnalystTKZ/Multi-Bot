import api from './api'

const positionService = {
  getOpenPositions: (limit = 50) => api.get(`/positions?status=open&limit=${limit}`),
  getClosedPositions: (limit = 50) => api.get(`/positions?status=closed&limit=${limit}`),
  getAllPositions: (limit = 50) => api.get(`/positions?status=all&limit=${limit}`),
  closePosition: (id, reason = 'manual') => api.post(`/positions/${id}/close`, { reason }),
  getLockedAssets: () => api.get('/positions/locked-assets'),
  getPositionMetrics: () => api.get('/positions/metrics'),
}

export default positionService
