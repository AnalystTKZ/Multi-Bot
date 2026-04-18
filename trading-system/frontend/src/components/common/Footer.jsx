import React from 'react'
import { Box, Typography } from '@mui/material'
import { appConfig } from '@/config/app.config'

const Footer = () => (
  <Box sx={{ mt: 4, textAlign: 'center', color: 'text.secondary' }}>
    <Typography variant="caption">
      {appConfig.name} v{appConfig.version} • Built for institutional-grade execution
    </Typography>
  </Box>
)

export default Footer
