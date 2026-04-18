import { useEffect, useCallback } from 'react'
import { useSelector } from 'react-redux'
import { initWebSocket, closeWebSocket, on, off } from '@services/websocket'

export const useWebSocket = () => {
  const user = useSelector((state) => state.auth.user)

  useEffect(() => {
    if (user) {
      initWebSocket()
    }

    return () => {
      closeWebSocket()
    }
  }, [user])

  const subscribe = useCallback((event, callback) => {
    on(event, callback)
    return () => off(event, callback)
  }, [])

  return { subscribe }
}
