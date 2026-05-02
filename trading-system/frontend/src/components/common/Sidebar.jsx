import { useEffect, useState } from 'react'
import { NavLink } from 'react-router-dom'
import {
  Analytics as AnalyticsIcon,
  ChevronLeft as CollapseIcon,
  ChevronRight as ExpandIcon,
  Dashboard as DashboardIcon,
  History as HistoryIcon,
  ModelTraining as TrainingIcon,
  Monitor as MonitorIcon,
  NotificationsActive as AlertsIcon,
  PlayCircle as BacktestIcon,
  Psychology as MLIcon,
  Settings as SettingsIcon,
  SmartToy as TradersIcon,
} from '@mui/icons-material'

const navItems = [
  { section: 'Main' },
  { label: 'Dashboard', to: '/', icon: DashboardIcon, end: true },
  { label: 'ML Trader', to: '/traders', icon: TradersIcon },
  { label: 'Monitors', to: '/monitors', icon: MonitorIcon },
  { label: 'Analytics', to: '/analytics', icon: AnalyticsIcon },
  { label: 'Trade History', to: '/history', icon: HistoryIcon },
  { label: 'Alerts', to: '/alerts', icon: AlertsIcon },
  { section: 'ML / Research' },
  { label: 'ML / AI', to: '/ml', icon: MLIcon },
  { label: 'Backtesting', to: '/backtest', icon: BacktestIcon },
  { label: 'Training', to: '/training', icon: TrainingIcon },
  { section: 'System' },
  { label: 'Settings', to: '/settings', icon: SettingsIcon },
]

const Sidebar = () => {
  const [collapsed, setCollapsed] = useState(false)

  useEffect(() => {
    const syncCollapsed = () => {
      if (window.innerWidth < 1280) setCollapsed(true)
    }
    syncCollapsed()
    window.addEventListener('resize', syncCollapsed)
    return () => window.removeEventListener('resize', syncCollapsed)
  }, [])

  return (
    <aside className={`dash-sidebar${collapsed ? ' collapsed' : ''}`}>
      <nav className="dash-nav-list" aria-label="Main navigation">
        {navItems.map((item, index) => {
          if (item.section) {
            return collapsed ? (
              <div className="dash-nav-spacer" key={`${item.section}-${index}`} />
            ) : (
              <div className="dash-nav-section" key={item.section}>
                {item.section}
              </div>
            )
          }
          const Icon = item.icon
          return (
            <NavLink
              className="dash-nav-item"
              end={item.end}
              key={item.to}
              title={collapsed ? item.label : undefined}
              to={item.to}
            >
              <span className="dash-nav-icon">
                <Icon sx={{ fontSize: 17 }} />
              </span>
              <span>{item.label}</span>
            </NavLink>
          )
        })}
      </nav>

      <div className="dash-sidebar-footer">
        <div className="dash-user-avatar small">T</div>
        {!collapsed ? (
          <div className="dash-sidebar-user">
            <span>Trader</span>
            <small>admin@admin.com</small>
          </div>
        ) : null}
        <button
          className="dash-collapse-button"
          type="button"
          title={collapsed ? 'Expand' : 'Collapse'}
          onClick={() => setCollapsed((value) => !value)}
        >
          {collapsed ? <ExpandIcon sx={{ fontSize: 15 }} /> : <CollapseIcon sx={{ fontSize: 15 }} />}
        </button>
      </div>
    </aside>
  )
}

export default Sidebar
