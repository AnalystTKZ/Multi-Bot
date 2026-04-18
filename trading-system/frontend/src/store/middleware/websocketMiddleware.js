import { initWebSocket, closeWebSocket } from '@services/websocket'
import { loginUser, logout, logoutUser, restoreSession } from '@store/slices/authSlice'

const websocketMiddleware = () => (next) => (action) => {
  if (loginUser.fulfilled.match(action) || restoreSession.fulfilled.match(action)) {
    initWebSocket()
  }

  if (logout.match(action) || logoutUser.fulfilled.match(action)) {
    closeWebSocket()
  }

  return next(action)
}

export default websocketMiddleware
