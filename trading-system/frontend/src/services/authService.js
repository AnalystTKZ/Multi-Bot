import api from './api'

const authService = {
  login: (payload) => api.post('/auth/login', payload),
  profile: () => api.get('/auth/profile'),
  logout: () => api.post('/auth/logout'),
}

export default authService
