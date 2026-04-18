import { useEffect } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import { fetchOpenPositions, fetchLockedAssets } from '@store/slices/positionsSlice'
import { appConfig } from '@/config/app.config'

export const usePositions = () => {
  const dispatch = useDispatch()
  const { open, loading, error } = useSelector((state) => state.positions)

  useEffect(() => {
    dispatch(fetchOpenPositions())
    dispatch(fetchLockedAssets())
    const interval = setInterval(
      () => dispatch(fetchOpenPositions()),
      appConfig.refreshInterval
    )

    return () => clearInterval(interval)
  }, [dispatch])

  return { positions: open, loading, error }
}
