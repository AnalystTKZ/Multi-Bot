import React from 'react'
import { Box, Button, TextField, Typography } from '@mui/material'
import { useForm } from 'react-hook-form'

const Register = () => {
  const { register, handleSubmit } = useForm()

  const onSubmit = () => {
    // Registration not yet wired to backend
  }

  return (
    <Box component="form" onSubmit={handleSubmit(onSubmit)} sx={{ display: 'grid', gap: 2 }}>
      <Typography variant="h5">Create an account</Typography>
      <TextField label="Full Name" {...register('name')} />
      <TextField label="Email" type="email" {...register('email')} />
      <TextField label="Password" type="password" {...register('password')} />
      <Button type="submit" variant="contained">
        Register
      </Button>
    </Box>
  )
}

export default Register
