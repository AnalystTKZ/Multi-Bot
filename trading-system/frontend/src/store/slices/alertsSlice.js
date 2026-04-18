import { createSlice } from '@reduxjs/toolkit'

const alertsSlice = createSlice({
  name: 'alerts',
  initialState: {
    list: [],
    unreadCount: 0,
  },
  reducers: {
    addAlert: (state, action) => {
      const alert = {
        id: Date.now(),
        timestamp: new Date().toISOString(),
        read: false,
        ...action.payload,
      }
      state.list.unshift(alert)
      state.unreadCount += 1
      if (state.list.length > 100) {
        state.list.pop()
      }
    },
    markAlertAsRead: (state, action) => {
      const alert = state.list.find((a) => a.id === action.payload)
      if (alert && !alert.read) {
        alert.read = true
        state.unreadCount -= 1
      }
    },
    clearAlerts: (state) => {
      state.list = []
      state.unreadCount = 0
    },
    removeAlert: (state, action) => {
      const index = state.list.findIndex((a) => a.id === action.payload)
      if (index !== -1) {
        if (!state.list[index].read) {
          state.unreadCount -= 1
        }
        state.list.splice(index, 1)
      }
    },
  },
})

export const { addAlert, markAlertAsRead, clearAlerts, removeAlert } = alertsSlice.actions

export default alertsSlice.reducer
