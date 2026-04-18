import React from 'react'
import { Box, Typography } from '@mui/material'

const TradeHistory = () => (
  <Box>
    <Typography variant="h6" sx={{ mb: 2 }}>
      Trade History
    </Typography>
    <Typography variant="body2" color="text.secondary">
      Historical trade logs will populate here.
    </Typography>
  </Box>
)

export default TradeHistory
