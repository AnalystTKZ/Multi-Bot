import { useState } from 'react'
import { Box, Button, TextField, Typography, Alert } from '@mui/material'
import { useForm } from 'react-hook-form'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '@hooks/useAuth'

const Login = () => {
  const { register, handleSubmit, formState: { isSubmitting } } = useForm()
  const { login } = useAuth()
  const navigate = useNavigate()
  const [error, setError] = useState(null)

  const onSubmit = async (data) => {
    setError(null)
    const isEmail = data.identifier?.includes('@')
    const payload = isEmail
      ? { email: data.identifier, password: data.password }
      : { username: data.identifier, password: data.password }
    try {
      await login(payload)
      navigate('/', { replace: true })
    } catch (err) {
      const detail = err?.detail ?? err
      setError(
        Array.isArray(detail)
          ? detail.map((e) => e.msg).join(', ')
          : String(detail ?? 'Login failed')
      )
    }
  }

  return (
    <Box component="form" onSubmit={handleSubmit(onSubmit)} sx={{ display: 'grid', gap: 2 }}>
      <Typography variant="h5" fontWeight={700}>
        Welcome back
      </Typography>
      <Typography variant="body2" color="text.secondary">
        Authenticate to access the trading command center.
      </Typography>
      <TextField
        label="Username or Email"
        autoComplete="username"
        {...register('identifier', { required: true })}
        placeholder="admin  or  admin@admin.com"
      />
      <TextField
        label="Password"
        type="password"
        autoComplete="current-password"
        {...register('password', { required: true })}
      />
      {error && (
        <Alert severity="error" sx={{ py: 0.5, fontSize: '0.82rem' }}>
          {error}
        </Alert>
      )}
      <Button type="submit" variant="contained" disabled={isSubmitting} size="large">
        {isSubmitting ? 'Signing in…' : 'Sign In'}
      </Button>
    </Box>
  )
}

export default Login
