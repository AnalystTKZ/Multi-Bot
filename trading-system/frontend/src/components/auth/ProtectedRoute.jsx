import React from 'react'
import { Navigate } from 'react-router-dom'
import { useSelector } from 'react-redux'
import { Box } from '@mui/material'
import LoadingSpinner from '@components/common/LoadingSpinner'

const ProtectedRoute = ({ children }) => {
  const user = useSelector((state) => state.auth.user)
  const initialized = useSelector((state) => state.auth.initialized)

  if (!initialized) {
    return (
      <Box sx={{ minHeight: '60vh', display: 'grid', placeItems: 'center' }}>
        <LoadingSpinner />
      </Box>
    )
  }

  if (!user) {
    return <Navigate to="/login" replace />
  }
  return children
}

export default ProtectedRoute
