import React from 'react'
import { Box, CircularProgress, Typography } from '@mui/material'

const LoadingSpinner = ({ label = 'Loading data...' }) => (
  <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2, py: 4 }}>
    <CircularProgress color="primary" />
    <Typography variant="body2" color="text.secondary">
      {label}
    </Typography>
  </Box>
)

export default LoadingSpinner
