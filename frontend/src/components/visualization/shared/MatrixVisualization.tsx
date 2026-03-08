import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import type { TensorData } from '../../../types/transformer';
import { Card } from '../../ui/card';

interface MatrixVisualizationProps {
  data: TensorData;
  title?: string | React.ReactNode;
  color?: string;
  showLabels?: boolean;
  rowLabels?: string[];
  colLabels?: string[];
  highlightedCells?: Array<[number, number]>;
  highlightedRows?: number[];
  highlightedCols?: number[];
  cellSize?: number;
  maxValue?: number;
  minValue?: number;
  showValues?: boolean;
  formatValue?: (value: number) => string;
  className?: string;
}

/**
 * MatrixVisualization Component
 *
 * Displays a matrix as an interactive heatmap with color-coded values.
 * Supports highlighting specific cells, rows, and columns.
 */
export const MatrixVisualization: React.FC<MatrixVisualizationProps> = ({
  data,
  title,
  color = '#3b82f6',
  showLabels = false,
  rowLabels,
  colLabels,
  highlightedCells = [],
  highlightedRows = [],
  highlightedCols = [],
  cellSize = 40,
  maxValue,
  minValue,
  showValues = false,
  formatValue = (v) => {
    if (typeof v !== 'number') return String(v);

    // Handle very large numbers with scientific notation
    if (Math.abs(v) >= 10000) {
      return v.toExponential(2);
    }

    // Handle very small numbers
    if (Math.abs(v) < 0.01 && v !== 0) {
      return v.toExponential(2);
    }

    // Default: 2 decimal places
    return v.toFixed(2);
  },
  className = '',
}) => {
  const matrix = useMemo(() => {
    // Handle empty or invalid data
    if (!data || !data.data) {
      return [[]];
    }

    // If data is already a 2D array
    if (Array.isArray(data.data)) {
      // Check if it's actually a 2D array
      if (data.data.length > 0 && Array.isArray(data.data[0])) {
        return data.data as number[][];
      } else if (data.data.length > 0) {
        // Convert 1D array to 2D (single row)
        return [data.data as number[]];
      }
    }

    // Handle nested data structures
    if (data.data && typeof data.data === 'object') {
      // Try to extract array from object
      const values = Object.values(data.data);
      if (values.length > 0 && Array.isArray(values[0])) {
        return values as number[][];
      }
    }

    return [[]];
  }, [data]);

  const rows = matrix.length;
  const cols = matrix[0]?.length || 0;

  // Check if matrix is empty
  const isEmptyMatrix = rows === 0 || cols === 0;

  // Compute min/max values for color scaling
  const flatMatrix = matrix.flat().filter((v): v is number => typeof v === 'number' && !isNaN(v));
  const computedMinValue = minValue ?? (flatMatrix.length > 0 ? Math.min(...flatMatrix) : 0);
  const computedMaxValue = maxValue ?? (flatMatrix.length > 0 ? Math.max(...flatMatrix) : 1);
  const valueRange = computedMaxValue - computedMinValue || 1;

  // Parse color to RGB for interpolation
  const parseColor = (hex: string) => {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
      r: parseInt(result[1], 16),
      g: parseInt(result[2], 16),
      b: parseInt(result[3], 16),
    } : { r: 59, g: 130, b: 246 };
  };

  const baseColor = parseColor(color);

  // Get background color for a cell based on its value
  const getCellColor = (value: number) => {
    // Clamp value to prevent NaN or Infinity
    const clampedValue = Math.max(-Infinity, Math.min(Infinity, value));
    const normalizedValue = (clampedValue - computedMinValue) / valueRange;
    const opacity = Math.max(0, Math.min(1, 0.1 + normalizedValue * 0.9));
    return `rgba(${baseColor.r}, ${baseColor.g}, ${baseColor.b}, ${opacity})`;
  };

  // Get text color based on background
  const getTextColor = (value: number) => {
    const clampedValue = Math.max(-Infinity, Math.min(Infinity, value));
    const normalizedValue = (clampedValue - computedMinValue) / valueRange;
    return normalizedValue > 0.5 ? 'white' : 'black';
  };

  // Check if a cell should be highlighted
  const isHighlighted = (row: number, col: number) => {
    return highlightedCells.some(([r, c]) => r === row && c === col) ||
           highlightedRows.includes(row) ||
           highlightedCols.includes(col);
  };

  return (
    <div className={`matrix-visualization ${className}`}>
      {title && (
        <h4 className="text-sm font-medium mb-3 text-gray-700">{title}</h4>
      )}

      {/* Empty state */}
      {isEmptyMatrix ? (
        <div className="bg-white rounded-lg border border-gray-200 p-8 text-center text-gray-500">
          <div className="text-sm">No data available</div>
          {data.shape && (
            <div className="text-xs text-gray-400 mt-1">
              Expected shape: {data.shape.join(' × ')}
            </div>
          )}
        </div>
      ) : (
        <div className="inline-block bg-white rounded-lg border border-gray-200 p-4 overflow-auto max-w-full">
          <div className="flex flex-col gap-1">
            {/* Column labels */}
            {showLabels && colLabels && (
              <div className="flex gap-1 ml-12">
                {colLabels.map((label, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-center text-xs text-gray-500 font-medium"
                    style={{ width: cellSize, height: 20 }}
                  >
                    {label}
                  </div>
                ))}
              </div>
            )}

            {/* Matrix rows */}
            {matrix.map((row, rowIndex) => (
            <div key={rowIndex} className="flex gap-1">
              {/* Row label */}
              {showLabels && rowLabels && (
                <div
                  className="flex items-center justify-end text-xs text-gray-500 font-medium pr-2"
                  style={{ width: 40, height: cellSize }}
                >
                  {rowLabels[rowIndex]}
                </div>
              )}

              {/* Cells */}
              {row.map((value, colIndex) => {
                const cellKey = `${rowIndex}-${colIndex}`;
                const highlighted = isHighlighted(rowIndex, colIndex);

                return (
                  <motion.div
                    key={cellKey}
                    className="relative flex items-center justify-center border border-gray-300 rounded"
                    style={{
                      width: cellSize,
                      height: cellSize,
                      backgroundColor: getCellColor(value),
                      color: getTextColor(value),
                      fontSize: Math.max(9, cellSize / 5),
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: (rowIndex * cols + colIndex) * 0.01 }}
                    whileHover={{ scale: 1.1, zIndex: 10 }}
                  >
                    {showValues && (
                      <span className="font-mono font-medium">
                        {formatValue(value)}
                      </span>
                    )}

                    {/* Highlight border */}
                    {highlighted && (
                      <motion.div
                        className="absolute inset-0 border-2 border-yellow-400 rounded pointer-events-none"
                        initial={{ scale: 0.8 }}
                        animate={{ scale: 1 }}
                        transition={{ type: 'spring', stiffness: 300, damping: 20 }}
                      />
                    )}
                  </motion.div>
                );
              })}
            </div>
          ))}
        </div>

        {/* Color scale legend */}
        <div className="mt-4 flex items-center gap-2">
          <span className="text-xs text-gray-500">
            {isNaN(computedMinValue) ? 'N/A' : computedMinValue.toFixed(3)}
          </span>
          <div
            className="flex-1 h-3 rounded"
            style={{
              background: `linear-gradient(to right,
                rgba(${baseColor.r}, ${baseColor.g}, ${baseColor.b}, 0.1),
                rgba(${baseColor.r}, ${baseColor.g}, ${baseColor.b}, 1))`
            }}
          />
          <span className="text-xs text-gray-500">
            {isNaN(computedMaxValue) ? 'N/A' : computedMaxValue.toFixed(3)}
          </span>
        </div>
      </div>
      )}
    </div>
  );
};

/**
 * MatrixComparison Component
 *
 * Displays two matrices side by side for comparison.
 */
export const MatrixComparison: React.FC<{
  left: MatrixVisualizationProps;
  right: MatrixVisualizationProps;
  leftTitle?: string;
  rightTitle?: string;
  operator?: string | React.ReactNode;
}> = ({ left, right, leftTitle = 'Before', rightTitle = 'After', operator = '→' }) => {
  return (
    <Card className="p-6">
      <div className="flex items-center gap-4">
        <div className="flex-1">
          {leftTitle && <h4 className="text-sm font-medium mb-3 text-center">{leftTitle}</h4>}
          <MatrixVisualization {...left} />
        </div>

        <div className="flex items-center justify-center text-2xl text-gray-400 font-light">
          {operator}
        </div>

        <div className="flex-1">
          {rightTitle && <h4 className="text-sm font-medium mb-3 text-center">{rightTitle}</h4>}
          <MatrixVisualization {...right} />
        </div>
      </div>
    </Card>
  );
};

/**
 * VectorVisualization Component
 *
 * Displays a 1D vector as a horizontal or vertical bar chart.
 */
export const VectorVisualization: React.FC<{
  data: number[];
  labels?: string[];
  color?: string;
  orientation?: 'horizontal' | 'vertical';
  title?: string;
  showValues?: boolean;
  highlightedIndices?: number[];
}> = ({
  data,
  labels,
  color = '#3b82f6',
  orientation = 'horizontal',
  title,
  showValues = false,
  highlightedIndices = [],
}) => {
  const maxValue = Math.max(...data.map(Math.abs));
  const minValue = Math.min(...data);

  const parseColor = (hex: string) => {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
      r: parseInt(result[1], 16),
      g: parseInt(result[2], 16),
      b: parseInt(result[3], 16),
    } : { r: 59, g: 130, b: 246 };
  };

  const baseColor = parseColor(color);

  const getBarColor = (value: number) => {
    if (value < 0 && minValue < 0) {
      // Negative values - use red
      const intensity = Math.abs(value) / Math.abs(minValue);
      return `rgba(239, 68, 68, ${0.3 + intensity * 0.7})`;
    } else {
      // Positive values - use base color
      const intensity = value / maxValue;
      return `rgba(${baseColor.r}, ${baseColor.g}, ${baseColor.b}, ${0.3 + intensity * 0.7})`;
    }
  };

  return (
    <div className="vector-visualization">
      {title && <h4 className="text-sm font-medium mb-3 text-gray-700">{title}</h4>}

      <div className={`flex ${orientation === 'vertical' ? 'flex-col' : 'flex-row'} gap-1`}>
        {data.map((value, index) => {
          const isHighlighted = highlightedIndices.includes(index);
          const size = orientation === 'horizontal' ? 'height' : 'width';
          const length = orientation === 'horizontal' ? 'width' : 'height';

          return (
            <motion.div
              key={index}
              className={`relative flex items-center justify-center rounded ${
                orientation === 'horizontal' ? 'flex-1' : ''
              }`}
              style={{
                [size]: 30,
                [length]: `${(Math.abs(value) / maxValue) * 100}%`,
                backgroundColor: getBarColor(value),
                minWidth: orientation === 'vertical' ? 30 : undefined,
              }}
              initial={{ opacity: 0, [length]: 0 }}
              animate={{ opacity: 1, [length]: `${(Math.abs(value) / maxValue) * 100}%` }}
              transition={{ delay: index * 0.05 }}
              whileHover={{ scale: 1.05 }}
            >
              {showValues && (
                <span className="text-xs font-medium text-white">
                  {value.toFixed(2)}
                </span>
              )}
              {labels && orientation === 'horizontal' && (
                <div className="absolute -bottom-5 text-xs text-gray-500 truncate max-w-full">
                  {labels[index]}
                </div>
              )}
              {isHighlighted && (
                <div className="absolute inset-0 border-2 border-yellow-400 rounded" />
              )}
            </motion.div>
          );
        })}
      </div>
    </div>
  );
};
