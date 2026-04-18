export const clamp = (value, min, max) => Math.min(Math.max(value, min), max)

export const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))
