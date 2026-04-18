import React, { useEffect } from 'react'
import { Box, Grid, Typography, Paper } from '@mui/material'
import { useDispatch } from 'react-redux'
import { fetchPerformance } from '@store/slices/analyticsSlice'
import PerformanceMetrics from './PerformanceMetrics'
import EquityCurve from './EquityCurve'

const Analytics = () => {
  const dispatch = useDispatch()

  useEffect(() => {
    dispatch(fetchPerformance())
  }, [dispatch])

  return (
    <Box>
      <Typography variant="h6" sx={{ mb: 2 }}>
        Portfolio Analytics
      </Typography>
      <Grid container spacing={2}>
        <Grid item xs={12} lg={4}>
          <Paper className="theme-panel" sx={{ p: 2 }}>
            <PerformanceMetrics />
          </Paper>
        </Grid>
        <Grid item xs={12} lg={8}>
          <Paper className="theme-panel" sx={{ p: 2 }}>
            <EquityCurve />
          </Paper>
        </Grid>
      </Grid>
    </Box>
  )
}

export default Analytics
