# Frontend Design — Multi-Bot Trading Dashboard

Updated 2026-04-02. Implementation: `trading-system/frontend/` (React 18 + Vite, port 3001)

---

## Purpose

A real-time trading dashboard surfacing live positions, bot performance, chart signals, news alerts, and risk controls with minimal latency. Designed for fast situational awareness — a trader should be able to assess system health in under 5 seconds.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | React 18 (JSX, not TypeScript) |
| UI Library | Material UI (MUI) v5 |
| Charts | Recharts |
| Real-time | Native WebSocket (no socket.io) via `services/websocket.js` |
| State | Redux Toolkit + Redux slices |
| Routing | React Router v6 |
| HTTP | Axios (with request/response interceptors; auto `Authorization: Bearer`) |
| Build | Vite |
| Path aliases | `@`, `@components`, `@pages`, `@hooks`, `@services`, `@store`, `@utils` |
| Container | Nginx 1.27-alpine (serves static build in Docker) |
| API base | `VITE_API_URL=http://localhost:3000/api` (baked in at build time) |
| WS base | `VITE_WS_URL=ws://localhost:3000` (baked in at build time) |

---

## Layout

```
┌──────────────────────────────────────────────────────────────────┐
│ HEADER (48px): Logo | Mode Selector | Connection Status | Alerts │
├──────────┬───────────────────────────────────────────────────────┤
│ SIDEBAR  │ CONTENT AREA (scrollable)                             │
│ 200px    │                                                        │
│ (collapses│                                                       │
│ to 52px) │                                                        │
└──────────┴───────────────────────────────────────────────────────┘
```

**Sidebar** is collapsible — click `<` to shrink to icon-only mode (52px). Tooltips on hover in collapsed state. Auto-collapses on screens < 1280px.

---

## Pages (all implemented)

### Dashboard (`/`)
- **KPI bar**: 6 metric cards — Balance, Monthly P&L, Win Rate, Max Drawdown, Active Trades, Total Trades (30d)
- **Left col (5/12)**: Open Positions, Bot Status (T1–T5), Signal Alerts
- **Right col (7/12)**: Risk Overview, Recent Trades
- 2-col layout collapses to single col at md (960px) breakpoint

### Traders (`/traders`)
- Card per trader (T1–T5) with: strategy name, session window, symbols, status chip, P&L, win rate, trade count, avg R:R
- Recent signals per trader (from Redis state)
- Pause / Stop controls per trader

### Monitors (`/monitors`)
- Real-time monitoring feeds from trading engine
- Chart monitor, news monitor, risk monitor, system health

### Analytics (`/analytics`)
- Period selector: 7d / 30d / 90d / all
- KPI row: Net P&L, Win Rate, Total Trades, Avg R:R, Max Drawdown, Profit Factor
- Equity curve (AreaChart)
- Monthly returns bar chart (from `/api/analytics/monthly-returns`)
- Per-trader comparison bar chart and breakdown cards

### Trade History (`/history`)
- Sortable/filterable table from analytics trades API
- Pagination

### Alerts (`/alerts`)
- System alerts and notifications

### Backtesting (`/backtest`)
- Config form: symbols, capital, commission, slippage
- Run backtest → queued result
- Results list from `/api/backtest/results`
- Detail view per result

### Training (`/training`)
- Training status cards
- Upload training data
- Start training job trigger

### ML / AI (`/ml`)
- ML model status cards (weight file, status, enabled state)
- RL agent stats (episodes, avg reward)
- Retrain trigger per model

### Settings (`/settings`)
- Placeholder (not yet implemented)

---

## Redux Store Slices

| Slice | State |
|---|---|
| `authSlice` | token, user, loading, error |
| `tradersSlice` | list, performance keyed by `trader_id` (not `id`), signals |
| `positionsSlice` | open positions; `closePosition` filters by `meta.arg.id` |
| `alertsSlice` | unreadCount, alerts list |
| `analyticsSlice` | performance data, equity curve |
| `monitorsSlice` | monitor feeds |
| `settingsSlice` | user settings |
| `uiSlice` | tradingMode, connectionStatus, sidebarOpen |

---

## API Endpoints Used

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/auth/login` | Login (username or email + password) |
| POST | `/api/auth/logout` | Logout |
| GET | `/api/traders/` | List all traders (trailing slash required) |
| GET | `/api/traders/{id}/performance` | Per-trader performance metrics |
| PATCH | `/api/traders/{id}/status` | Pause/resume trader |
| GET | `/api/positions?status=open` | Open positions |
| GET | `/api/analytics/performance?period=30d` | System performance |
| GET | `/api/analytics/equity-curve` | Equity curve data |
| GET | `/api/analytics/trades` | Trade history |
| GET | `/api/analytics/monthly-returns` | Monthly P&L buckets |
| GET | `/api/system/status` | System mode and engine status |
| POST | `/api/system/mode` | Set mode (paper/live) |
| GET | `/api/backtest/results` | List backtest JSON results |
| POST | `/api/backtest/run` | Queue a backtest |
| GET | `/api/training/status` | Training status + model weights |
| POST | `/api/training/start` | Start training job |
| GET | `/api/ml/models` | ML model list (5 models) |
| GET | `/api/ml/rl-agent` | RL agent state (episodes, avg_reward) |
| POST | `/api/ml/models/{id}/retrain` | Trigger model retrain |
| GET | `/api/monitors` | Monitors data |

**Trailing slash**: always use trailing slash for collection endpoints — FastAPI 307 redirect strips `Authorization` header on some clients.

WebSocket: `ws://localhost:3000/ws` — real-time TRADE_EXECUTED, SIGNAL_GENERATED, MARKET_DATA, alert events.

---

## Authentication

- JWT token stored in `localStorage`
- Login accepts `username` (`admin`) OR `email` (`admin@admin.com`)
- `ProtectedRoute` wraps all non-login routes — redirects to `/login` if no token
- On successful login, `useEffect` watching token triggers `navigate('/')` (dashboard)
- 401 responses auto-logout and redirect to `/login`, **except** on the login page itself (guard: `!pathname.startsWith('/login')`)
- Axios request interceptor attaches `Authorization: Bearer <token>` header to all requests
- `authSlice` clears `state.error = null` in the pending case (so previous error doesn't persist on new attempt)

---

## Mode Selector

Header ModeSelector supports 2 operational modes (paper vs live):

| Mode | Colour | Confirm Required |
|---|---|---|
| Paper | Green | No |
| Live | Red | Yes (dialog) |

Mode is stored in Redux `uiSlice.tradingMode`. Change calls `POST /api/system/mode`.

---

## Known Gotchas

| Issue | Fix |
|---|---|
| `GET /traders` 307 redirect strips Auth header | Use `/traders/` (trailing slash) |
| `closePosition.fulfilled` filters by wrong field | Filter by `meta.arg.id` (original dispatch arg), not `payload.position_id` |
| `tradersSlice` stored performance by `trader.id` | Store by `action.payload.trader_id` — backend field is `trader_id` |
| Login page shows old error on new attempt | `state.error = null` in `loginUser.pending` case |
| No redirect after login | `useEffect` on token change → `navigate('/')` |
| 401 interceptor fired on login page itself | Guard `if (!pathname.startsWith('/login'))` before redirect |

---

## UX Principles

- **Responsive**: fully responsive from mobile (480px) to 4K — sidebar collapses, grid stacks
- **Fast scan**: critical info (P&L, drawdown, positions) above the fold without scrolling
- **Risk-first**: drawdown and exposure indicators are visually prominent
- **Real-time**: positions and journal entries update via WebSocket push and polling
- **Dark theme**: default deep navy — easy on the eyes during extended sessions
- **Error resilience**: all API calls use `Promise.allSettled` or try/catch — partial failures don't break the page
- **5-bot aware**: Traders page and Dashboard BotStatus show T1–T5 (not the old T1–T4)
