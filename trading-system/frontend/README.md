# Trading Bot Frontend

Production-grade React dashboard for a multi-bot ICT/Smart Money trading system. The UI is optimized for real-time monitoring, WebSocket updates, and fast portfolio decisioning.

## Features
- Multi-panel dashboard with bot status, positions, and risk overview
- Redux state management and WebSocket middleware
- Real-time updates with Socket.IO
- MUI + Tailwind styling with custom themes
- Modular component architecture
- Vitest + Testing Library setup

## Quick Start

```bash
npm install
npm run dev
```

## Environment

Use the shared environment template at `trading-system/.env.example`.
Copy it to `trading-system/.env` and update values as needed.

## Project Structure

See `src/` for components, hooks, services, store, and styles. The structure mirrors the requested architecture from the trading system documentation.

## Testing

```bash
npm run test
```

> Note: add a `test` script in `package.json` if you want to run `vitest` directly.

## WebSocket Channels
- `positions`
- `traders`
- `alerts`
- `monitors`

## Backend Integration
The API proxy points to `http://localhost:3000` and expects `/api/*` endpoints.
