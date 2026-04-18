import { AppBar, Box, Toolbar, Typography, Chip, Stack, IconButton, Badge } from '@mui/material'
import { Notifications as NotifIcon, Circle as DotIcon } from '@mui/icons-material'
import { useSelector } from 'react-redux'
import { useNavigate } from 'react-router-dom'
import { appConfig } from '@/config/app.config'
import ModeSelector from './ModeSelector'

const Header = () => {
  const alerts = useSelector((state) => state.alerts.unreadCount)
  const connection = useSelector((state) => state.ui.connectionStatus)
  const navigate = useNavigate()

  return (
    <AppBar
      position="sticky"
      elevation={0}
      sx={{
        background: 'rgba(11, 18, 32, 0.97)',
        borderBottom: '1px solid rgba(148, 163, 184, 0.12)',
        backdropFilter: 'blur(8px)',
      }}
    >
      <Toolbar sx={{ display: 'flex', justifyContent: 'space-between', minHeight: '48px !important', px: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <DotIcon sx={{ fontSize: 10, color: '#26e0b8' }} />
          <Typography variant="h6" sx={{ fontWeight: 800, letterSpacing: '-0.02em', fontSize: '1rem' }}>
            {appConfig.name ?? 'Multi-Bot'}
          </Typography>
          <Typography
            variant="caption"
            color="text.secondary"
            sx={{ display: { xs: 'none', sm: 'block' }, ml: 0.5 }}
          >
            ICT/SMC Command Center
          </Typography>
        </Box>

        <ModeSelector />

        <Stack direction="row" spacing={1.5} alignItems="center">
          <Chip
            icon={<DotIcon sx={{ fontSize: '10px !important' }} />}
            label={connection === 'online' ? 'Connected' : 'Offline'}
            color={connection === 'online' ? 'success' : 'warning'}
            size="small"
            variant="outlined"
            sx={{ height: 26, fontSize: '0.7rem' }}
          />
          <IconButton size="small" onClick={() => navigate('/alerts')} sx={{ color: 'text.secondary' }}>
            <Badge badgeContent={alerts > 0 ? alerts : null} color="error" max={99}>
              <NotifIcon sx={{ fontSize: 20 }} />
            </Badge>
          </IconButton>
        </Stack>
      </Toolbar>
    </AppBar>
  )
}

export default Header
