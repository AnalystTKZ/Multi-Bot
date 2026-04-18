import { createSlice, createAsyncThunk } from '@reduxjs/toolkit'
import authService from '@services/authService'

export const restoreSession = createAsyncThunk(
  'auth/restoreSession',
  async (_, { rejectWithValue }) => {
    try {
      const response = await authService.profile()
      return {
        user: response.user || response,
      }
    } catch (error) {
      const detail = error?.detail
      const msg = detail
        ? Array.isArray(detail) ? detail.map((e) => e.msg).join(', ') : String(detail)
        : error?.message || 'Session restore failed'
      return rejectWithValue(msg)
    }
  }
)

export const loginUser = createAsyncThunk(
  'auth/login',
  async (payload, { rejectWithValue }) => {
    try {
      const response = await authService.login(payload)
      return response
    } catch (error) {
      // Normalise to a plain string so Redux doesn't store a non-serialisable object
      const detail = error?.detail
      const msg = detail
        ? Array.isArray(detail) ? detail.map((e) => e.msg).join(', ') : String(detail)
        : error?.message || 'Login failed'
      return rejectWithValue(msg)
    }
  }
)

export const logoutUser = createAsyncThunk(
  'auth/logout',
  async (_, { rejectWithValue }) => {
    try {
      await authService.logout()
      return true
    } catch (error) {
      const detail = error?.detail
      const msg = detail
        ? Array.isArray(detail) ? detail.map((e) => e.msg).join(', ') : String(detail)
        : error?.message || 'Logout failed'
      return rejectWithValue(msg)
    }
  }
)

const authSlice = createSlice({
  name: 'auth',
  initialState: {
    token: null,
    user: null,
    initialized: false,
    loading: false,
    error: null,
  },
  reducers: {
    logout: (state) => {
      state.token = null
      state.user = null
      state.initialized = true
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(restoreSession.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(restoreSession.fulfilled, (state, action) => {
        state.loading = false
        state.initialized = true
        state.user = action.payload.user || null
        state.token = null
      })
      .addCase(restoreSession.rejected, (state, action) => {
        state.loading = false
        state.initialized = true
        state.error = action.payload
      })
      .addCase(loginUser.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(loginUser.fulfilled, (state, action) => {
        state.loading = false
        state.initialized = true
        state.token = null
        state.user = action.payload.user || null
      })
      .addCase(loginUser.rejected, (state, action) => {
        state.loading = false
        state.error = action.payload
        state.initialized = true
      })
      .addCase(logoutUser.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(logoutUser.fulfilled, (state) => {
        state.loading = false
        state.token = null
        state.user = null
        state.initialized = true
      })
      .addCase(logoutUser.rejected, (state, action) => {
        state.loading = false
        state.error = action.payload
      })
  },
})

export const { logout } = authSlice.actions

export default authSlice.reducer
