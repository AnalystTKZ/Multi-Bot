import { describe, it, expect } from 'vitest'
import { formatCurrency, formatPercent } from '@/utils/formatters'

describe('formatters', () => {
  it('formats currency', () => {
    expect(formatCurrency(1000)).toContain('$')
  })

  it('formats percent', () => {
    expect(formatPercent(0.5)).toBe('50.00%')
  })
})
