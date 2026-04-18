import React from 'react'
import { Box, Typography } from '@mui/material'

const Settings = () => (
  <Box>
    <Typography variant="h6" sx={{ mb: 2 }}>
      System Settings
    </Typography>
    <Typography variant="body2" color="text.secondary">
      Configure broker connections, notifications, and UI preferences.
    </Typography>
  </Box>
)

export default Settings
