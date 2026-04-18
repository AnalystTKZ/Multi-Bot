import { createTheme } from '@mui/material/styles'

export const theme = createTheme({
  palette: {
    mode: 'dark',
    background: {
      default: '#0b1220',
      paper: '#111a2e',
    },
    primary: {
      main: '#26e0b8',
    },
    secondary: {
      main: '#7c3aed',
    },
    error: {
      main: '#ef4444',
    },
    warning: {
      main: '#f59e0b',
    },
    success: {
      main: '#10b981',
    },
    text: {
      primary: '#f8fafc',
      secondary: '#cbd5f5',
    },
  },
  typography: {
    fontFamily: 'IBM Plex Sans, sans-serif',
    h1: { fontFamily: 'Space Grotesk, sans-serif' },
    h2: { fontFamily: 'Space Grotesk, sans-serif' },
    h3: { fontFamily: 'Space Grotesk, sans-serif' },
    h4: { fontFamily: 'Space Grotesk, sans-serif' },
  },
  shape: {
    borderRadius: 14,
  },
})
