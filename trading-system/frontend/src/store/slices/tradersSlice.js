import { createSlice, createAsyncThunk } from '@reduxjs/toolkit'
import traderService from '@services/traderService'

export const fetchAllTraders = createAsyncThunk(
  'traders/fetchAllTraders',
  async (_, { rejectWithValue }) => {
    try {
      const response = await traderService.getAllTraders()
      return response.traders || response
    } catch (error) {
      return rejectWithValue(error.message || error)
    }
  }
)

export const fetchTraderPerformance = createAsyncThunk(
  'traders/fetchPerformance',
  async ({ id, period }, { rejectWithValue }) => {
    try {
      const response = await traderService.getTraderPerformance(id, period)
      return response
    } catch (error) {
      return rejectWithValue(error.message || error)
    }
  }
)

const tradersSlice = createSlice({
  name: 'traders',
  initialState: {
    list: [],
    performance: {},
    signals: {},
    loading: false,
    error: null,
  },
  reducers: {
    updateTraderSignal: (state, action) => {
      const { traderId, signal } = action.payload
      if (!state.signals[traderId]) {
        state.signals[traderId] = []
      }
      state.signals[traderId].unshift(signal)
      if (state.signals[traderId].length > 10) {
        state.signals[traderId].pop()
      }
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchAllTraders.pending, (state) => {
        state.loading = true
      })
      .addCase(fetchAllTraders.fulfilled, (state, action) => {
        state.list = action.payload
        state.loading = false
      })
      .addCase(fetchAllTraders.rejected, (state, action) => {
        state.error = action.payload
        state.loading = false
      })
      .addCase(fetchTraderPerformance.fulfilled, (state, action) => {
        const traderId = action.payload.trader_id || action.payload.trader?.trader_id
        if (traderId) {
          state.performance[traderId] = action.payload
        }
      })
  },
})

export const { updateTraderSignal } = tradersSlice.actions

export default tradersSlice.reducer
