import { createSlice } from '@reduxjs/toolkit'

const settingsSlice = createSlice({
  name: 'settings',
  initialState: {
    theme: 'dark',
    refreshInterval: 5000,
    notifications: true,
  },
  reducers: {
    updateSettings: (state, action) => {
      return { ...state, ...action.payload }
    },
  },
})

export const { updateSettings } = settingsSlice.actions

export default settingsSlice.reducer
