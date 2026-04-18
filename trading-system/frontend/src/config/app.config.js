export const appConfig = {
  name: import.meta.env.VITE_APP_NAME || 'Trading Bot System',
  version: import.meta.env.VITE_APP_VERSION || '1.0.0',
  refreshInterval: Number(import.meta.env.VITE_REFRESH_INTERVAL || 5000),
}
