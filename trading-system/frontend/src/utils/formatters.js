import { format } from 'date-fns'

export const formatCurrency = (value, currency = 'USD') => {
  if (value === null || value === undefined) return '--'
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    maximumFractionDigits: 2,
  }).format(value)
}

export const formatPercent = (value) => {
  if (value === null || value === undefined) return '--'
  return `${(value * 100).toFixed(2)}%`
}

export const formatDate = (value) => {
  if (!value) return '--'
  return format(new Date(value), 'MMM dd, yyyy HH:mm')
}
