import React, { useEffect } from 'react'
import { Box, Typography, Stack, Chip } from '@mui/material'
import { useDispatch, useSelector } from 'react-redux'
import { fetchMonitorStatus } from '@store/slices/monitorsSlice'

const MonitorsPanel = () => {
  const dispatch = useDispatch()
  const { status } = useSelector((state) => state.monitors)

  useEffect(() => {
    dispatch(fetchMonitorStatus())
  }, [dispatch])

  return (
    <Box>
      <Typography variant="h6" sx={{ mb: 2 }}>
        Monitor Stack
      </Typography>
      <Stack spacing={2}>
        {status.length === 0 && (
          <Typography variant="body2" color="text.secondary">
            No monitor updates yet.
          </Typography>
        )}
        {status.map((monitor) => (
          <Box key={monitor.monitor_id} sx={{ display: 'flex', justifyContent: 'space-between' }}>
            <Box>
              <Typography variant="subtitle2">{monitor.name}</Typography>
              <Typography variant="caption" color="text.secondary">
                Last check: {monitor.last_check}
              </Typography>
            </Box>
            <Chip
              label={monitor.status}
              size="small"
              color={monitor.status === 'running' || monitor.status === 'ok' ? 'success' : monitor.status === 'warn' || monitor.status === 'degraded' ? 'warning' : 'error'}
            />
          </Box>
        ))}
      </Stack>
    </Box>
  )
}

export default MonitorsPanel
