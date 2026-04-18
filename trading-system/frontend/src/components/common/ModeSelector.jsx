import React, { useState } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import {
  Box,
  ButtonGroup,
  Button,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Typography,
  CircularProgress,
} from '@mui/material'
import {
  PlayArrow as LiveIcon,
  Science as PaperIcon,
  Assessment as BacktestIcon,
  Psychology as TrainIcon,
} from '@mui/icons-material'
import { setTradingMode, setModeChanging } from '@store/slices/uiSlice'
import systemService from '@services/systemService'

const MODES = [
  {
    id: 'live',
    label: 'Live',
    icon: <LiveIcon sx={{ fontSize: 16 }} />,
    color: '#D32F2F',
    tooltip: 'Live trading — real capital at risk',
    confirmRequired: true,
  },
  {
    id: 'paper',
    label: 'Paper',
    icon: <PaperIcon sx={{ fontSize: 16 }} />,
    color: '#00C853',
    tooltip: 'Paper trading — simulated execution',
    confirmRequired: false,
  },
  {
    id: 'backtest',
    label: 'Backtest',
    icon: <BacktestIcon sx={{ fontSize: 16 }} />,
    color: '#1976D2',
    tooltip: 'Backtesting — run strategies on historical data',
    confirmRequired: false,
  },
  {
    id: 'training',
    label: 'Training',
    icon: <TrainIcon sx={{ fontSize: 16 }} />,
    color: '#FF6F00',
    tooltip: 'Training — retrain ML models',
    confirmRequired: false,
  },
]

const MODE_COLORS = {
  live: '#D32F2F',
  paper: '#00C853',
  backtest: '#1976D2',
  training: '#FF6F00',
}

const ModeSelector = () => {
  const dispatch = useDispatch()
  const currentMode = useSelector((state) => state.ui.tradingMode)
  const modeChanging = useSelector((state) => state.ui.modeChanging)
  const [confirmMode, setConfirmMode] = useState(null)

  const handleModeClick = (mode) => {
    if (mode.id === currentMode) return
    if (mode.confirmRequired) {
      setConfirmMode(mode)
    } else {
      applyMode(mode.id)
    }
  }

  const applyMode = async (modeId) => {
    dispatch(setModeChanging(true))
    try {
      await systemService.setMode(modeId)
    } catch {
      // Backend may not be connected; still update UI mode locally
    } finally {
      dispatch(setTradingMode(modeId))
      dispatch(setModeChanging(false))
    }
    setConfirmMode(null)
  }

  const activeColor = MODE_COLORS[currentMode] || '#00C853'

  return (
    <>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <Typography
          variant="caption"
          sx={{ color: 'text.secondary', textTransform: 'uppercase', letterSpacing: '0.08em', mr: 0.5 }}
        >
          Mode
        </Typography>
        <ButtonGroup size="small" variant="outlined" disabled={modeChanging}>
          {MODES.map((mode) => {
            const isActive = currentMode === mode.id
            return (
              <Tooltip key={mode.id} title={mode.tooltip} placement="bottom">
                <Button
                  onClick={() => handleModeClick(mode)}
                  startIcon={modeChanging && isActive ? <CircularProgress size={12} /> : mode.icon}
                  sx={{
                    borderColor: isActive ? mode.color : 'rgba(148,163,184,0.3)',
                    color: isActive ? mode.color : 'text.secondary',
                    backgroundColor: isActive ? `${mode.color}18` : 'transparent',
                    fontWeight: isActive ? 700 : 400,
                    textTransform: 'none',
                    px: 1.5,
                    '&:hover': {
                      borderColor: mode.color,
                      backgroundColor: `${mode.color}18`,
                      color: mode.color,
                    },
                  }}
                >
                  {mode.label}
                </Button>
              </Tooltip>
            )
          })}
        </ButtonGroup>
        <Box
          sx={{
            width: 8,
            height: 8,
            borderRadius: '50%',
            backgroundColor: activeColor,
            boxShadow: `0 0 6px ${activeColor}`,
            animation: currentMode === 'live' ? 'pulse 1.5s infinite' : 'none',
            '@keyframes pulse': {
              '0%, 100%': { opacity: 1 },
              '50%': { opacity: 0.3 },
            },
          }}
        />
      </Box>

      <Dialog
        open={!!confirmMode}
        onClose={() => setConfirmMode(null)}
        PaperProps={{ sx: { background: '#111a2e', border: '1px solid #D32F2F' } }}
      >
        <DialogTitle sx={{ color: '#D32F2F' }}>Enable Live Trading?</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            This will switch to <strong style={{ color: '#D32F2F' }}>live trading mode</strong> using your Capital.com
            account with real funds.
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Ensure the engine is configured correctly and you have reviewed the risk settings. This action can be
            reversed by switching back to Paper mode.
          </Typography>
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button onClick={() => setConfirmMode(null)} variant="outlined" size="small">
            Cancel
          </Button>
          <Button
            onClick={() => applyMode(confirmMode.id)}
            variant="contained"
            size="small"
            sx={{ backgroundColor: '#D32F2F', '&:hover': { backgroundColor: '#b71c1c' } }}
          >
            Confirm Live
          </Button>
        </DialogActions>
      </Dialog>
    </>
  )
}

export default ModeSelector
