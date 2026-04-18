import { useEffect } from 'react'
import { Box } from '@mui/material'
import { useNavigate } from 'react-router-dom'
import { useSelector } from 'react-redux'
import Login from '@components/auth/Login'

const LoginPage = () => {
  const user = useSelector((state) => state.auth.user)
  const navigate = useNavigate()

  useEffect(() => {
    if (user) navigate('/', { replace: true })
  }, [user, navigate])

  return (
    <Box className="section-card" sx={{ maxWidth: 420, mx: 'auto', mt: 8 }}>
      <Login />
    </Box>
  )
}

export default LoginPage
