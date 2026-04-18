import axios from 'axios'
import { apiConfig } from '@/config/api.config'
import { store } from '@store/store'
import { logout } from '@store/slices/authSlice'

const api = axios.create({
  baseURL: apiConfig.baseURL,
  timeout: apiConfig.timeout,
  withCredentials: apiConfig.withCredentials,
  headers: {
    'Content-Type': 'application/json',
  },
})

api.interceptors.request.use(
  (config) => {
    config.withCredentials = true
    return config
  },
  (error) => Promise.reject(error)
)

api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const status = error.response?.status
    const data = error.response?.data
    console.error('[API]', error.config?.method?.toUpperCase(), error.config?.url, status, data)
    // Only redirect on 401 when NOT on the login page (avoid swallowing login errors)
    if (status === 401 && !window.location.pathname.startsWith('/login')) {
      store.dispatch(logout())
      window.location.href = '/login'
    }
    return Promise.reject(data || error.message)
  }
)

export default api
