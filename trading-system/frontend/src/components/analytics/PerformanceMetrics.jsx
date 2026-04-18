import React from 'react'
import { Box, Typography } from '@mui/material'
import { useSelector } from 'react-redux'

const PerformanceMetrics = () => {
  const performance = useSelector((state) => state.analytics.performance)

  return (
    <Box>
      <Typography variant="subtitle1" sx={{ mb: 1 }}>
        Performance Metrics
      </Typography>
      <Typography variant="body2" color="text.secondary">
        Sharpe: {performance?.sharpe_ratio ?? '--'}
      </Typography>
      <Typography variant="body2" color="text.secondary">
        Win Rate: {performance?.win_rate ?? '--'}
      </Typography>
      <Typography variant="body2" color="text.secondary">
        Max Drawdown: {performance?.max_drawdown ?? '--'}
      </Typography>
    </Box>
  )
}

export default PerformanceMetrics
