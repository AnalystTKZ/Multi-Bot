import { createSlice, createAsyncThunk } from '@reduxjs/toolkit'
import monitorService from '@services/monitorService'

export const fetchMonitorStatus = createAsyncThunk(
  'monitors/fetchStatus',
  async (_, { rejectWithValue }) => {
    try {
      const response = await monitorService.getMonitorStatus()
      return response
    } catch (error) {
      return rejectWithValue(error.message || error)
    }
  }
)

const monitorsSlice = createSlice({
  name: 'monitors',
  initialState: {
    status: [],
    loading: false,
    error: null,
  },
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(fetchMonitorStatus.pending, (state) => {
        state.loading = true
      })
      .addCase(fetchMonitorStatus.fulfilled, (state, action) => {
        state.status = action.payload
        state.loading = false
      })
      .addCase(fetchMonitorStatus.rejected, (state, action) => {
        state.error = action.payload
        state.loading = false
      })
  },
})

export default monitorsSlice.reducer
