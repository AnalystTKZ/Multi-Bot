import { createSlice, createAsyncThunk } from '@reduxjs/toolkit'
import positionService from '@services/positionService'

export const fetchOpenPositions = createAsyncThunk(
  'positions/fetchOpenPositions',
  async (_, { rejectWithValue }) => {
    try {
      const response = await positionService.getOpenPositions()
      return response.positions || response
    } catch (error) {
      return rejectWithValue(error.message || error)
    }
  }
)

export const closePosition = createAsyncThunk(
  'positions/closePosition',
  async ({ id, reason }, { rejectWithValue }) => {
    try {
      const response = await positionService.closePosition(id, reason)
      return response
    } catch (error) {
      return rejectWithValue(error.message || error)
    }
  }
)

export const fetchLockedAssets = createAsyncThunk(
  'positions/fetchLockedAssets',
  async (_, { rejectWithValue }) => {
    try {
      const response = await positionService.getLockedAssets()
      return response.locked_assets || response
    } catch (error) {
      return rejectWithValue(error.message || error)
    }
  }
)

const positionsSlice = createSlice({
  name: 'positions',
  initialState: {
    open: [],
    closed: [],
    lockedAssets: [],
    loading: false,
    error: null,
    lastUpdate: null,
  },
  reducers: {
    addPosition: (state, action) => {
      state.open.push(action.payload)
      state.lastUpdate = new Date().toISOString()
    },
    removePosition: (state, action) => {
      state.open = state.open.filter((p) => p.id !== action.payload)
      state.lastUpdate = new Date().toISOString()
    },
    updatePosition: (state, action) => {
      const index = state.open.findIndex((p) => p.id === action.payload.id)
      if (index !== -1) {
        state.open[index] = { ...state.open[index], ...action.payload }
        state.lastUpdate = new Date().toISOString()
      }
    },
    updatePrice: (state, action) => {
      const { positionId, currentPrice, pnl, pnlPercent } = action.payload
      const position = state.open.find((p) => p.id === positionId)
      if (position) {
        position.current_price = currentPrice
        position.pnl = pnl
        position.pnl_percent = pnlPercent
      }
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchOpenPositions.pending, (state) => {
        state.loading = true
      })
      .addCase(fetchOpenPositions.fulfilled, (state, action) => {
        state.open = action.payload
        state.loading = false
        state.lastUpdate = new Date().toISOString()
      })
      .addCase(fetchOpenPositions.rejected, (state, action) => {
        state.error = action.payload
        state.loading = false
      })
      .addCase(closePosition.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(closePosition.fulfilled, (state, action) => {
        // Use the original thunk arg (the id we sent) — backend returns { ticket, message }
        const closedId = action.meta.arg.id
        state.open = state.open.filter((p) => p.id !== closedId)
        state.loading = false
        state.lastUpdate = new Date().toISOString()
      })
      .addCase(closePosition.rejected, (state, action) => {
        state.error = action.payload
        state.loading = false
      })
      .addCase(fetchLockedAssets.fulfilled, (state, action) => {
        state.lockedAssets = action.payload
      })
  },
})

export const { addPosition, removePosition, updatePosition, updatePrice } =
  positionsSlice.actions

export default positionsSlice.reducer
