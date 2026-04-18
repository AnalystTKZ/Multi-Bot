import { describe, it, expect } from 'vitest'
import { useAuth } from '@/hooks/useAuth'

describe('useAuth', () => {
  it('exports auth hook', () => {
    expect(typeof useAuth).toBe('function')
  })
})
