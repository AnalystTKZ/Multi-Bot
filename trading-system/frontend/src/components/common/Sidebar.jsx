import { useState } from 'react'
import { NavLink } from 'react-router-dom'
import {
  Box, Typography, List, ListItemButton, ListItemText, ListItemIcon,
  Divider, IconButton, Tooltip, useMediaQuery, useTheme,
} from '@mui/material'
import {
  Dashboard as DashboardIcon,
  SmartToy as TradersIcon,
  Monitor as MonitorsIcon,
  BarChart as AnalyticsIcon,
  History as HistoryIcon,
  Settings as SettingsIcon,
  PlayCircle as BacktestIcon,
  ModelTraining as TrainingIcon,
  Psychology as MLIcon,
  NotificationsActive as AlertsIcon,
  ChevronLeft as CollapseIcon,
  ChevronRight as ExpandIcon,
} from '@mui/icons-material'

const navGroups = [
  {
    label: 'Main',
    items: [
      { label: 'Dashboard', to: '/', icon: <DashboardIcon sx={{ fontSize: 18 }} /> },
      { label: 'Traders', to: '/traders', icon: <TradersIcon sx={{ fontSize: 18 }} /> },
      { label: 'Monitors', to: '/monitors', icon: <MonitorsIcon sx={{ fontSize: 18 }} /> },
    ],
  },
  {
    label: 'Analysis',
    items: [
      { label: 'Analytics', to: '/analytics', icon: <AnalyticsIcon sx={{ fontSize: 18 }} /> },
      { label: 'Trade History', to: '/history', icon: <HistoryIcon sx={{ fontSize: 18 }} /> },
      { label: 'Alerts', to: '/alerts', icon: <AlertsIcon sx={{ fontSize: 18 }} /> },
    ],
  },
  {
    label: 'Engine',
    items: [
      { label: 'Backtesting', to: '/backtest', icon: <BacktestIcon sx={{ fontSize: 18 }} /> },
      { label: 'Training', to: '/training', icon: <TrainingIcon sx={{ fontSize: 18 }} /> },
      { label: 'ML / AI', to: '/ml', icon: <MLIcon sx={{ fontSize: 18 }} /> },
    ],
  },
  {
    label: 'System',
    items: [
      { label: 'Settings', to: '/settings', icon: <SettingsIcon sx={{ fontSize: 18 }} /> },
    ],
  },
]

const Sidebar = () => {
  const theme = useTheme()
  const isMd = useMediaQuery(theme.breakpoints.down('lg'))
  const [collapsed, setCollapsed] = useState(isMd)

  return (
    <Box
      className="sidebar"
      sx={{
        width: collapsed ? 52 : 200,
        minWidth: collapsed ? 52 : 200,
        transition: 'width 0.2s ease, min-width 0.2s ease',
        background: 'rgba(10, 15, 28, 0.98)',
        borderRight: '1px solid rgba(148, 163, 184, 0.1)',
        display: 'flex',
        flexDirection: 'column',
        overflowY: 'auto',
        overflowX: 'hidden',
        py: 1,
        flexShrink: 0,
      }}
    >
      {/* Collapse toggle */}
      <Box sx={{ display: 'flex', justifyContent: collapsed ? 'center' : 'flex-end', px: collapsed ? 0 : 1, mb: 0.5 }}>
        <IconButton
          size="small"
          onClick={() => setCollapsed((c) => !c)}
          sx={{ color: 'rgba(148,163,184,0.5)', '&:hover': { color: '#26e0b8' } }}
        >
          {collapsed ? <ExpandIcon sx={{ fontSize: 16 }} /> : <CollapseIcon sx={{ fontSize: 16 }} />}
        </IconButton>
      </Box>

      {navGroups.map((group, gi) => (
        <Box key={group.label}>
          {gi > 0 && <Divider sx={{ borderColor: 'rgba(148,163,184,0.08)', my: 0.5, mx: collapsed ? 0.5 : 1.5 }} />}
          {!collapsed && (
            <Typography
              variant="caption"
              sx={{
                px: 1.5,
                py: 0.25,
                display: 'block',
                color: 'rgba(148,163,184,0.4)',
                letterSpacing: '0.1em',
                textTransform: 'uppercase',
                fontSize: '0.6rem',
                fontWeight: 600,
              }}
            >
              {group.label}
            </Typography>
          )}
          <List dense disablePadding>
            {group.items.map((item) => (
              <Tooltip key={item.to} title={collapsed ? item.label : ''} placement="right">
                <ListItemButton
                  component={NavLink}
                  to={item.to}
                  end={item.to === '/'}
                  sx={{
                    borderRadius: 1.5,
                    mx: 0.5,
                    mb: 0.25,
                    py: 0.6,
                    px: collapsed ? 1 : 1.25,
                    minWidth: 0,
                    justifyContent: collapsed ? 'center' : 'flex-start',
                    color: 'rgba(148,163,184,0.75)',
                    '&.active': {
                      backgroundColor: 'rgba(38, 224, 184, 0.12)',
                      color: '#26e0b8',
                      '& .MuiListItemIcon-root': { color: '#26e0b8' },
                    },
                    '&:hover': {
                      backgroundColor: 'rgba(38, 224, 184, 0.07)',
                      color: '#94a3b8',
                    },
                  }}
                >
                  <ListItemIcon sx={{ minWidth: collapsed ? 0 : 28, color: 'inherit', justifyContent: 'center' }}>
                    {item.icon}
                  </ListItemIcon>
                  {!collapsed && (
                    <ListItemText
                      primary={item.label}
                      slotProps={{ primary: { fontSize: '0.8rem', fontWeight: 500, noWrap: true } }}
                    />
                  )}
                </ListItemButton>
              </Tooltip>
            ))}
          </List>
        </Box>
      ))}
    </Box>
  )
}

export default Sidebar
