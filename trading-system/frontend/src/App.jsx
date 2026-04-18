import React, { lazy, Suspense, useEffect } from 'react'
import { Routes, Route, useLocation } from 'react-router-dom'
import { useDispatch, useSelector } from 'react-redux'
import Header from '@components/common/Header'
import Sidebar from '@components/common/Sidebar'
import Footer from '@components/common/Footer'
import ErrorBoundary from '@components/common/ErrorBoundary'
import LoadingSpinner from '@components/common/LoadingSpinner'
import ProtectedRoute from '@components/auth/ProtectedRoute'
import { restoreSession } from '@store/slices/authSlice'
import './App.css'

// Eager-load pages used on startup
import DashboardPage from '@pages/DashboardPage'
import LoginPage from '@pages/LoginPage'
import NotFoundPage from '@pages/NotFoundPage'

// Lazy-load the rest
const TradersPage = lazy(() => import('@pages/TradersPage'))
const MonitorsPage = lazy(() => import('@pages/MonitorsPage'))
const AnalyticsPage = lazy(() => import('@pages/AnalyticsPage'))
const TradeHistoryPage = lazy(() => import('@pages/TradeHistoryPage'))
const SettingsPage = lazy(() => import('@pages/SettingsPage'))
const BacktestPage = lazy(() => import('@pages/BacktestPage'))
const TrainingPage = lazy(() => import('@pages/TrainingPage'))
const MLPage = lazy(() => import('@pages/MLPage'))
const AlertsPage = lazy(() => import('@pages/AlertsPage'))

const ProtectedLazy = ({ children }) => (
  <ProtectedRoute>
    <Suspense fallback={<LoadingSpinner />}>{children}</Suspense>
  </ProtectedRoute>
)

const App = () => {
  const location = useLocation()
  const dispatch = useDispatch()
  const initialized = useSelector((state) => state.auth.initialized)
  const isAuthRoute = location.pathname.startsWith('/login')

  useEffect(() => {
    if (!initialized) {
      dispatch(restoreSession())
    }
  }, [dispatch, initialized])

  return (
    <ErrorBoundary>
      {isAuthRoute ? (
        <main className="content-area auth-area">
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route path="*" element={<NotFoundPage />} />
          </Routes>
        </main>
      ) : (
        <div className="app-shell">
          <Sidebar />
          <div className="layout-shell">
            <Header />
            <main className="content-area">
              <Routes>
                <Route
                  path="/"
                  element={
                    <ProtectedRoute>
                      <DashboardPage />
                    </ProtectedRoute>
                  }
                />
                <Route path="/traders" element={<ProtectedLazy><TradersPage /></ProtectedLazy>} />
                <Route path="/monitors" element={<ProtectedLazy><MonitorsPage /></ProtectedLazy>} />
                <Route path="/analytics" element={<ProtectedLazy><AnalyticsPage /></ProtectedLazy>} />
                <Route path="/history" element={<ProtectedLazy><TradeHistoryPage /></ProtectedLazy>} />
                <Route path="/alerts" element={<ProtectedLazy><AlertsPage /></ProtectedLazy>} />
                <Route path="/backtest" element={<ProtectedLazy><BacktestPage /></ProtectedLazy>} />
                <Route path="/training" element={<ProtectedLazy><TrainingPage /></ProtectedLazy>} />
                <Route path="/ml" element={<ProtectedLazy><MLPage /></ProtectedLazy>} />
                <Route path="/settings" element={<ProtectedLazy><SettingsPage /></ProtectedLazy>} />
                <Route path="*" element={<NotFoundPage />} />
              </Routes>
              <Footer />
            </main>
          </div>
        </div>
      )}
    </ErrorBoundary>
  )
}

export default App
