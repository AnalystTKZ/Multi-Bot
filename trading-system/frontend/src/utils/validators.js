export const isEmail = (value) => /\S+@\S+\.\S+/.test(value)

export const required = (value) => value !== null && value !== undefined && value !== ''
