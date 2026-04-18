import React from 'react'
import { Box, Typography, Stack, Chip } from '@mui/material'
import { useSelector } from 'react-redux'

const TradersList = () => {
  const { list } = useSelector((state) => state.traders)

  return (
    <Box>
      <Typography variant="h6" sx={{ mb: 2 }}>
        Traders Overview
      </Typography>
      <Stack spacing={2}>
        {list.map((trader) => (
          <Box key={trader.trader_id} sx={{ display: 'flex', justifyContent: 'space-between' }}>
            <Box>
              <Typography variant="subtitle2">{trader.name}</Typography>
              <Typography variant="caption" color="text.secondary">
                {trader.strategy}
              </Typography>
            </Box>
            <Chip label={`${Math.round(trader.win_rate * 100)}% win`} size="small" />
          </Box>
        ))}
        {list.length === 0 && (
          <Typography variant="body2" color="text.secondary">
            No trader data yet.
          </Typography>
        )}
      </Stack>
    </Box>
  )
}

export default TradersList
