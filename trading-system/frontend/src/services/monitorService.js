import api from './api'

const monitorService = {
  getMonitorStatus: () => api.get('/monitors'),
  getAlerts: () => api.get('/monitors/alerts'),
  getIctSignal: (payload) => api.post('/monitors/ict-signal', payload),
  getFilteredIctSignal: (payload) => api.post('/monitors/ict-signal/filtered', payload),
  trainIctFilter: (payload) => api.post('/monitors/ict-signal/train', payload),
}

export default monitorService
