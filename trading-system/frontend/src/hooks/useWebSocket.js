import { useCallback } from 'react'
import { on, off } from '@services/websocket'

// The global WebSocket lifecycle (connect/disconnect) is managed by
// websocketMiddleware in response to auth events. This hook only exposes
// a stable subscribe/unsubscribe API for components that need to react
// to specific WS event types.
export const useWebSocket = () => {
  const subscribe = useCallback((event, callback) => {
    on(event, callback)
    return () => off(event, callback)
  }, [])

  return { subscribe }
}
