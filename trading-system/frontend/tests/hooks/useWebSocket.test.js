import { describe, it, expect } from 'vitest'

import { useWebSocket } from '@/hooks/useWebSocket'

describe('useWebSocket', () => {
  it('exports subscribe handler', () => {
    expect(typeof useWebSocket).toBe('function')
  })
})
