import { createSlice, createAsyncThunk } from '@reduxjs/toolkit'
import analyticsService from '@services/analyticsService'

export const fetchPerformance = createAsyncThunk(
  'analytics/fetchPerformance',
  async (_, { rejectWithValue }) => {
    try {
      const response = await analyticsService.getPerformance()
      return response
    } catch (error) {
      return rejectWithValue(error.message || error)
    }
  }
)

const analyticsSlice = createSlice({
  name: 'analytics',
  initialState: {
    performance: null,
    loading: false,
    error: null,
  },
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(fetchPerformance.pending, (state) => {
        state.loading = true
      })
      .addCase(fetchPerformance.fulfilled, (state, action) => {
        state.performance = action.payload
        state.loading = false
      })
      .addCase(fetchPerformance.rejected, (state, action) => {
        state.error = action.payload
        state.loading = false
      })
  },
})

export default analyticsSlice.reducer
