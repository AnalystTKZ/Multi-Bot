export const mapCandlesToSeries = (candles) =>
  candles.map((c) => ({ time: c.time, open: c.open, high: c.high, low: c.low, close: c.close }))
