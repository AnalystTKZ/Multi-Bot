import React from 'react'
import { Box, Typography } from '@mui/material'
import { Link } from 'react-router-dom'

const NotFoundPage = () => (
  <Box sx={{ textAlign: 'center', mt: 10 }}>
    <Typography variant="h3" sx={{ mb: 2 }}>
      404
    </Typography>
    <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
      Route not found. Return to the command center.
    </Typography>
    <Link to="/">Back to Dashboard</Link>
  </Box>
)

export default NotFoundPage
