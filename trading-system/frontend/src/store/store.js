import { configureStore } from '@reduxjs/toolkit'
import authReducer from './slices/authSlice'
import positionsReducer from './slices/positionsSlice'
import tradersReducer from './slices/tradersSlice'
import monitorsReducer from './slices/monitorsSlice'
import alertsReducer from './slices/alertsSlice'
import analyticsReducer from './slices/analyticsSlice'
import settingsReducer from './slices/settingsSlice'
import uiReducer from './slices/uiSlice'
import websocketMiddleware from './middleware/websocketMiddleware'

export const store = configureStore({
  reducer: {
    auth: authReducer,
    positions: positionsReducer,
    traders: tradersReducer,
    monitors: monitorsReducer,
    alerts: alertsReducer,
    analytics: analyticsReducer,
    settings: settingsReducer,
    ui: uiReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['websocket/messageReceived'],
        ignoredPaths: ['websocket.socket'],
      },
    }).concat(websocketMiddleware),
})

export default store
