import { useEffect } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import { fetchAllTraders } from '@store/slices/tradersSlice'

export const useTraders = () => {
  const dispatch = useDispatch()
  const { list, loading, error } = useSelector((state) => state.traders)

  useEffect(() => {
    dispatch(fetchAllTraders())
  }, [dispatch])

  return { traders: list, loading, error }
}
