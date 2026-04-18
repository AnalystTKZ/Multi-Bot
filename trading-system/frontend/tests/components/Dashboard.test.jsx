import React from 'react'
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { Provider } from 'react-redux'
import { ThemeProvider } from '@mui/material/styles'
import { MemoryRouter } from 'react-router-dom'
import { store } from '@/store/store'
import { theme } from '@/config/theme.config'
import Dashboard from '@/components/dashboard/Dashboard'

const renderWithProviders = (ui) =>
  render(
    <Provider store={store}>
      <ThemeProvider theme={theme}>
        <MemoryRouter>{ui}</MemoryRouter>
      </ThemeProvider>
    </Provider>
  )

describe('Dashboard', () => {
  it('renders key sections', () => {
    renderWithProviders(<Dashboard />)
    expect(screen.getByRole('heading', { name: /Command Dashboard/i })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: /Open Positions/i })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: /Bot Status/i })).toBeInTheDocument()
  })
})
