import { describe, it, expect } from 'vitest'
import { calculateDrawdown } from '@/utils/calculations'

describe('calculations', () => {
  it('calculates drawdown', () => {
    const dd = calculateDrawdown([100, 110, 90, 120])
    expect(dd).toBeGreaterThan(0)
  })
})
