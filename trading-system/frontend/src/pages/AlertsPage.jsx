import React, { useState } from 'react'
import {
  Box,
  Container,
  Paper,
  Typography,
  Button,
  Chip,
  Divider,
  Stack,
  IconButton,
  ToggleButtonGroup,
  ToggleButton,
  Alert,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Tooltip,
} from '@mui/material'
import {
  Error as CriticalIcon,
  Warning as WarnIcon,
  Info as InfoIcon,
  Delete as DeleteIcon,
  DoneAll as ReadAllIcon,
  NotificationsOff as MuteIcon,
} from '@mui/icons-material'
import { useSelector, useDispatch } from 'react-redux'
import { markAlertAsRead, clearAlerts, removeAlert } from '@store/slices/alertsSlice'
import { formatDistanceToNow } from 'date-fns'

const SEVERITY_CONFIG = {
  critical: { label: 'Critical', color: '#D32F2F', icon: <CriticalIcon sx={{ fontSize: 18 }} /> },
  warning: { label: 'Warning', color: '#FF6F00', icon: <WarnIcon sx={{ fontSize: 18 }} /> },
  info: { label: 'Info', color: '#1976D2', icon: <InfoIcon sx={{ fontSize: 18 }} /> },
}

const AlertRow = ({ alert, onRead, onDelete }) => {
  const cfg = SEVERITY_CONFIG[alert.severity] || SEVERITY_CONFIG.info
  const timeAgo = (() => {
    try {
      return formatDistanceToNow(new Date(alert.timestamp), { addSuffix: true })
    } catch {
      return alert.timestamp
    }
  })()

  return (
    <ListItem
      onClick={() => !alert.read && onRead(alert.id)}
      sx={{
        borderRadius: 2,
        mb: 0.75,
        border: `1px solid ${alert.read ? 'rgba(148,163,184,0.1)' : `${cfg.color}30`}`,
        backgroundColor: alert.read ? 'transparent' : `${cfg.color}08`,
        cursor: alert.read ? 'default' : 'pointer',
        transition: 'all 0.15s',
        '&:hover': { backgroundColor: `${cfg.color}10` },
      }}
    >
      <ListItemIcon sx={{ minWidth: 36, color: alert.read ? 'text.disabled' : cfg.color }}>
        {cfg.icon}
      </ListItemIcon>
      <ListItemText
        primary={
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography
              variant="body2"
              fontWeight={alert.read ? 400 : 700}
              sx={{ color: alert.read ? 'text.secondary' : 'text.primary' }}
            >
              {alert.message || alert.title || 'Alert'}
            </Typography>
            {!alert.read && (
              <Box
                sx={{
                  width: 6,
                  height: 6,
                  borderRadius: '50%',
                  backgroundColor: cfg.color,
                  flexShrink: 0,
                }}
              />
            )}
          </Box>
        }
        secondary={
          <Stack direction="row" spacing={1} alignItems="center" sx={{ mt: 0.25 }}>
            {alert.source && (
              <Chip label={alert.source} size="small" sx={{ height: 16, fontSize: '0.65rem' }} />
            )}
            <Typography variant="caption" color="text.disabled">
              {timeAgo}
            </Typography>
          </Stack>
        }
      />
      <Tooltip title="Dismiss">
        <IconButton
          size="small"
          onClick={(e) => {
            e.stopPropagation()
            onDelete(alert.id)
          }}
          sx={{ ml: 1, opacity: 0.4, '&:hover': { opacity: 1 } }}
        >
          <DeleteIcon sx={{ fontSize: 14 }} />
        </IconButton>
      </Tooltip>
    </ListItem>
  )
}

const AlertsPage = () => {
  const dispatch = useDispatch()
  const { list, unreadCount } = useSelector((state) => state.alerts)
  const [filter, setFilter] = useState('all')

  const filtered = list.filter((a) => {
    if (filter === 'unread') return !a.read
    if (filter === 'critical') return a.severity === 'critical'
    if (filter === 'warning') return a.severity === 'warning'
    return true
  })

  const counts = {
    critical: list.filter((a) => a.severity === 'critical').length,
    warning: list.filter((a) => a.severity === 'warning').length,
    info: list.filter((a) => a.severity === 'info').length,
  }

  return (
    <Container maxWidth="lg" className="fade-in" sx={{ py: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 3 }}>
        <Box>
          <Typography variant="h5" fontWeight={700}>
            Alerts & Notifications
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {unreadCount > 0 ? `${unreadCount} unread` : 'All caught up'}
          </Typography>
        </Box>
        <Stack direction="row" spacing={1}>
          <Button
            size="small"
            variant="outlined"
            startIcon={<ReadAllIcon />}
            onClick={() => list.filter((a) => !a.read).forEach((a) => dispatch(markAlertAsRead(a.id)))}
            disabled={unreadCount === 0}
          >
            Mark All Read
          </Button>
          <Button
            size="small"
            variant="outlined"
            color="error"
            startIcon={<DeleteIcon />}
            onClick={() => dispatch(clearAlerts())}
            disabled={list.length === 0}
          >
            Clear All
          </Button>
        </Stack>
      </Box>

      {/* Summary chips */}
      <Stack direction="row" spacing={1.5} sx={{ mb: 3 }}>
        {Object.entries(SEVERITY_CONFIG).map(([key, cfg]) => (
          <Paper
            key={key}
            className="theme-panel"
            sx={{ px: 2.5, py: 1.5, display: 'flex', alignItems: 'center', gap: 1, minWidth: 100 }}
          >
            <Box sx={{ color: cfg.color }}>{cfg.icon}</Box>
            <Box>
              <Typography variant="h6" fontWeight={700} sx={{ lineHeight: 1 }}>
                {counts[key] || 0}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {cfg.label}
              </Typography>
            </Box>
          </Paper>
        ))}
      </Stack>

      <Paper className="theme-panel" sx={{ p: 2.5 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <ToggleButtonGroup
            value={filter}
            exclusive
            onChange={(_, v) => v && setFilter(v)}
            size="small"
          >
            <ToggleButton value="all" sx={{ fontSize: '0.72rem', px: 1.5 }}>All</ToggleButton>
            <ToggleButton value="unread" sx={{ fontSize: '0.72rem', px: 1.5 }}>Unread</ToggleButton>
            <ToggleButton value="critical" sx={{ fontSize: '0.72rem', px: 1.5, color: '#D32F2F' }}>
              Critical
            </ToggleButton>
            <ToggleButton value="warning" sx={{ fontSize: '0.72rem', px: 1.5, color: '#FF6F00' }}>
              Warning
            </ToggleButton>
          </ToggleButtonGroup>
          <Typography variant="caption" color="text.secondary">
            {filtered.length} alert{filtered.length !== 1 ? 's' : ''}
          </Typography>
        </Box>

        <Divider sx={{ mb: 1.5, borderColor: 'rgba(148,163,184,0.1)' }} />

        {filtered.length === 0 ? (
          <Box sx={{ py: 6, textAlign: 'center' }}>
            <MuteIcon sx={{ fontSize: 48, color: 'rgba(148,163,184,0.2)', mb: 2 }} />
            <Typography color="text.secondary">No alerts</Typography>
          </Box>
        ) : (
          <List disablePadding>
            {filtered.map((alert) => (
              <AlertRow
                key={alert.id}
                alert={alert}
                onRead={(id) => dispatch(markAlertAsRead(id))}
                onDelete={(id) => dispatch(removeAlert(id))}
              />
            ))}
          </List>
        )}
      </Paper>

      {/* Alert tier legend */}
      <Paper className="theme-panel" sx={{ p: 2, mt: 3 }}>
        <Typography variant="caption" color="text.secondary" fontWeight={600} display="block" gutterBottom>
          ALERT TIERS
        </Typography>
        <Stack spacing={0.75}>
          {[
            { icon: <CriticalIcon sx={{ fontSize: 14, color: '#D32F2F' }} />, label: 'Critical', desc: 'Risk violation, connection loss, position breach — full-screen overlay' },
            { icon: <WarnIcon sx={{ fontSize: 14, color: '#FF6F00' }} />, label: 'Warning', desc: 'Drawdown approaching limit, loss streak, high exposure' },
            { icon: <InfoIcon sx={{ fontSize: 14, color: '#1976D2' }} />, label: 'Info', desc: 'Trade executed, signal generated, model retrained' },
          ].map(({ icon, label, desc }) => (
            <Box key={label} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              {icon}
              <Typography variant="caption" fontWeight={600} sx={{ minWidth: 60 }}>
                {label}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {desc}
              </Typography>
            </Box>
          ))}
        </Stack>
      </Paper>
    </Container>
  )
}

export default AlertsPage
