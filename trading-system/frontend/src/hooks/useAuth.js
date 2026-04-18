import { useDispatch, useSelector } from 'react-redux'
import { loginUser, logoutUser } from '@store/slices/authSlice'

export const useAuth = () => {
  const dispatch = useDispatch()
  const auth = useSelector((state) => state.auth)

  const login = (payload) => dispatch(loginUser(payload)).unwrap()
  const signOut = () => dispatch(logoutUser()).unwrap()

  return { ...auth, login, signOut }
}
