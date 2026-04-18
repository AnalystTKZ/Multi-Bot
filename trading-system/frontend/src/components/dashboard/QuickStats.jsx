import React from 'react'
import { Box, Typography, Stack, Chip } from '@mui/material'
import { useSelector } from 'react-redux'

const QuickStats = () => {
  const lockedAssets = useSelector((state) => state.positions.lockedAssets)
  const alerts = useSelector((state) => state.alerts.list)

  return (
    <Box>
      <Typography variant="h6" sx={{ mb: 2 }}>
        Execution Pulse
      </Typography>
      <Stack spacing={2}>
        <Box>
          <Typography variant="caption" color="text.secondary">
            Locked Assets
          </Typography>
          <Stack direction="row" spacing={1} flexWrap="wrap" sx={{ mt: 1 }}>
            {lockedAssets.length === 0 && (
              <Typography variant="body2" color="text.secondary">
                None
              </Typography>
            )}
            {lockedAssets.map((asset) => (
              <Chip key={asset} label={asset} size="small" />
            ))}
          </Stack>
        </Box>
        <Box>
          <Typography variant="caption" color="text.secondary">
            Latest Alert
          </Typography>
          <Typography variant="body2" sx={{ mt: 1 }}>
            {alerts[0]?.message || 'System stable. Monitoring live liquidity.'}
          </Typography>
        </Box>
      </Stack>
    </Box>
  )
}

export default QuickStats
