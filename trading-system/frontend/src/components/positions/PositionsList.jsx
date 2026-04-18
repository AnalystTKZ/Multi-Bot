import React from 'react'
import {
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
} from '@mui/material'
import { useSelector } from 'react-redux'
import { formatCurrency, formatPercent } from '@utils/formatters'

const PositionsList = () => {
  const positions = useSelector((state) => state.positions.open)

  return (
    <Box>
      <TableContainer component={Paper} className="theme-panel">
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Symbol</TableCell>
              <TableCell>Direction</TableCell>
              <TableCell>Entry</TableCell>
              <TableCell>Current</TableCell>
              <TableCell>P&L</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {positions.map((position) => (
              <TableRow key={position.id}>
                <TableCell>{position.symbol || position.pair}</TableCell>
                <TableCell>
                  <Chip
                    size="small"
                    label={position.direction}
                    color={position.direction === 'buy' ? 'success' : 'error'}
                  />
                </TableCell>
                <TableCell>{formatCurrency(position.entry_price)}</TableCell>
                <TableCell>{formatCurrency(position.current_price || position.entry_price)}</TableCell>
                <TableCell>
                  {formatCurrency(position.pnl || 0)} ({formatPercent(position.pnl_percent || 0)})
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  )
}

export default PositionsList
