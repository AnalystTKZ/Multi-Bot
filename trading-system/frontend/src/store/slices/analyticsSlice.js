import { createSlice } from '@reduxjs/toolkit'

// Performance data is fetched directly by components via analyticsService.
// This slice exists to hold any analytics state that needs to be shared globally.
const analyticsSlice = createSlice({
  name: 'analytics',
  initialState: {
    performance: null,
    loading: false,
    error: null,
  },
  reducers: {},
})

export default analyticsSlice.reducer
