export const calculateExpectancy = (winRate, avgWin, avgLoss) =>
  winRate * avgWin + (1 - winRate) * avgLoss

export const calculateDrawdown = (equityCurve) => {
  let peak = equityCurve[0] || 0
  let maxDrawdown = 0
  equityCurve.forEach((value) => {
    if (value > peak) peak = value
    const drawdown = peak ? (peak - value) / peak : 0
    if (drawdown > maxDrawdown) maxDrawdown = drawdown
  })
  return maxDrawdown
}
