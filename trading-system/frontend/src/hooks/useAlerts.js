import { useSelector } from 'react-redux'

export const useAlerts = () => {
  const alerts = useSelector((state) => state.alerts.list)
  const unreadCount = useSelector((state) => state.alerts.unreadCount)

  return { alerts, unreadCount }
}
