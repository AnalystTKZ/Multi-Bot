import { createSlice } from '@reduxjs/toolkit'

const uiSlice = createSlice({
  name: 'ui',
  initialState: {
    connectionStatus: 'offline',
    sidebarOpen: true,
    tradingMode: localStorage.getItem('tradingMode') || 'paper',
    modeChanging: false,
  },
  reducers: {
    setConnectionStatus: (state, action) => {
      state.connectionStatus = action.payload
    },
    toggleSidebar: (state) => {
      state.sidebarOpen = !state.sidebarOpen
    },
    setTradingMode: (state, action) => {
      state.tradingMode = action.payload
      localStorage.setItem('tradingMode', action.payload)
    },
    setModeChanging: (state, action) => {
      state.modeChanging = action.payload
    },
  },
})

export const { setConnectionStatus, toggleSidebar, setTradingMode, setModeChanging } = uiSlice.actions

export default uiSlice.reducer
