import { store } from '@store/store'
import { addPosition, updatePosition, updatePrice } from '@store/slices/positionsSlice'
import { addAlert } from '@store/slices/alertsSlice'
import { setConnectionStatus } from '@store/slices/uiSlice'
import { updateTraderSignal } from '@store/slices/tradersSlice'

let socket = null
const handlers = new Map()

const getWsUrl = () => {
  const origin = window.location.origin
  const base = origin.startsWith('https://') ? origin.replace('https://', 'wss://') : origin.replace('http://', 'ws://')
  return `${base}/ws`
}

const emitHandlers = (eventName, payload) => {
  const set = handlers.get(eventName)
  if (!set) return
  for (const cb of set) {
    cb(payload)
  }
}

const handleEvent = (event) => {
  const eventType = event.event_type || event.type
  const payload = event.payload || event.data || {}
  const correlationId = event.correlation_id || payload.correlation_id || null
  const strategyId = payload.strategy_id || null
  console.info(
    JSON.stringify({
      timestamp: new Date().toISOString(),
      level: 'INFO',
      service: 'frontend',
      module: 'websocket',
      correlation_id: correlationId,
      strategy_id: strategyId,
      event_type: eventType,
      message: 'Frontend received websocket event',
      data: payload,
    })
  )
  if (eventType === 'trade_executed') {
    const position = {
      id: payload.order_id || payload.request_id,
      symbol: payload.symbol,
      type: payload.side,
      volume: parseFloat(payload.filled_quantity || payload.quantity || 0),
      price_open: parseFloat(payload.average_price || payload.price || 0),
      price_current: parseFloat(payload.average_price || payload.price || 0),
      profit: parseFloat(payload.pnl || 0),
      trader_id: payload.strategy_id,
    }
    store.dispatch(addPosition(position))
    store.dispatch(
      addAlert({
        type: 'position',
        severity: 'info',
        title: 'Trade Executed',
        message: `${position.symbol} ${position.type} @ ${position.price_open}`,
      })
    )
  }

  if (eventType === 'position_updated') {
    store.dispatch(updatePosition(payload))
  }

  if (eventType === 'price_update') {
    store.dispatch(updatePrice(payload))
  }

  if (eventType === 'system_alert') {
    store.dispatch(
      addAlert({
        type: 'system',
        severity: 'warning',
        title: payload.alert_type || 'System Alert',
        message: payload.message || 'System event received',
      })
    )
  }

  if (eventType === 'trade_failed') {
    store.dispatch(
      addAlert({
        type: 'trade',
        severity: 'error',
        title: 'Trade Failed',
        message: payload.error || 'Trade execution failed',
      })
    )
  }

  if (eventType === 'risk_violation') {
    store.dispatch(
      addAlert({
        type: 'risk',
        severity: 'warning',
        title: 'Risk Violation',
        message: payload.message || 'Risk limit breached',
      })
    )
  }

  if (eventType === 'signal_generated' || eventType === 'SIGNAL_GENERATED') {
    const traderId = payload.trader_id || payload.strategy_id
    if (traderId) {
      store.dispatch(
        updateTraderSignal({
          traderId,
          signal: {
            symbol: payload.symbol,
            direction: payload.direction || payload.side,
            confidence: payload.confidence,
            timestamp: payload.timestamp || new Date().toISOString(),
            correlation_id: payload.correlation_id || correlationId,
          },
        })
      )
    }
    store.dispatch(
      addAlert({
        type: 'signal',
        severity: 'info',
        source: traderId || 'engine',
        message: `${payload.symbol || ''} ${(payload.direction || payload.side || '').toUpperCase()} signal — conf: ${((payload.confidence || 0) * 100).toFixed(0)}%`,
      })
    )
  }

  if (eventType === 'drawdown_alert' || eventType === 'DRAWDOWN_ALERT') {
    store.dispatch(
      addAlert({
        type: 'drawdown',
        severity: 'critical',
        source: 'risk_monitor',
        message: payload.message || `Drawdown alert: ${payload.drawdown_pct || ''}`,
      })
    )
  }

  emitHandlers(eventType, payload)
}

export const initWebSocket = () => {
  if (socket) return socket

  socket = new WebSocket(getWsUrl())

  socket.onopen = () => {
    store.dispatch(setConnectionStatus('online'))
  }

  socket.onclose = () => {
    store.dispatch(setConnectionStatus('offline'))
  }

  socket.onerror = () => {
    store.dispatch(setConnectionStatus('error'))
  }

  socket.onmessage = (message) => {
    try {
      const data = JSON.parse(message.data)
      handleEvent(data)
    } catch {
      // ignore invalid messages
    }
  }

  return socket
}

export const closeWebSocket = () => {
  if (socket) {
    socket.close()
    socket = null
  }
}

export const getSocket = () => socket

export const emit = (event, data) => {
  if (socket && socket.readyState === WebSocket.OPEN) {
    const payload = { event, data, correlation_id: data?.correlation_id || null }
    console.info(
      JSON.stringify({
        timestamp: new Date().toISOString(),
        level: 'INFO',
        service: 'frontend',
        module: 'websocket',
        correlation_id: payload.correlation_id,
        strategy_id: data?.strategy_id || null,
        event_type: event,
        message: 'Frontend sent websocket event',
        data,
      })
    )
    socket.send(JSON.stringify(payload))
  }
}

export const on = (event, callback) => {
  if (!handlers.has(event)) handlers.set(event, new Set())
  handlers.get(event).add(callback)
}

export const off = (event, callback) => {
  const set = handlers.get(event)
  if (!set) return
  set.delete(callback)
}
