import { Badge, IconButton } from '@mui/material'
import {
  Circle as DotIcon,
  Notifications as NotificationsIcon,
  Search as SearchIcon,
} from '@mui/icons-material'
import { useSelector } from 'react-redux'
import { useNavigate } from 'react-router-dom'
import { appConfig } from '@/config/app.config'
import ModeSelector from './ModeSelector'

const MARKET_TICKERS = ['XAUUSD', 'EURUSD', 'GBPUSD', 'USDJPY', 'BTCUSD', 'US30']

const Header = () => {
  const alerts = useSelector((state) => state.alerts.unreadCount)
  const connection = useSelector((state) => state.ui.connectionStatus)
  const navigate = useNavigate()
  const connected = connection === 'online'

  return (
    <header className="dash-header">
      <div className="dash-brand">
        <span className="dash-brand-mark" />
        <span>{appConfig.name ?? 'Multi-Bot'}</span>
      </div>

      <div className="dash-header-divider" />

      <ModeSelector />

      <div className={`dash-connection ${connected ? 'online' : 'offline'}`}>
        <DotIcon sx={{ fontSize: 10 }} />
        <span>{connected ? 'WS connected' : 'WS offline'}</span>
      </div>

      <div className="dash-ticker-strip" aria-label="Watched symbols">
        {MARKET_TICKERS.map((symbol, index) => (
          <span className="dash-ticker-item" key={symbol}>
            <span className="dash-ticker-symbol">{symbol}</span>
            <span>{index % 2 === 0 ? 'active' : 'watch'}</span>
          </span>
        ))}
      </div>

      <div className="dash-header-actions">
        <IconButton size="small" className="dash-icon-button" aria-label="Search">
          <SearchIcon sx={{ fontSize: 18 }} />
        </IconButton>
        <IconButton size="small" className="dash-icon-button" aria-label="Alerts" onClick={() => navigate('/alerts')}>
          <Badge badgeContent={alerts > 0 ? alerts : null} color="error" max={99}>
            <NotificationsIcon sx={{ fontSize: 18 }} />
          </Badge>
        </IconButton>
        <div className="dash-user-avatar">T</div>
      </div>
    </header>
  )
}

export default Header
