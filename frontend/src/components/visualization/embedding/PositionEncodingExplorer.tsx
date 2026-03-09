/**
 * PositionEncodingExplorer Component
 *
 * Interactive exploration of positional encoding in Transformers:
 * - Sinusoidal positional encoding visualization
 * - Learned positional encoding
 * - Rotary Position Embedding (RoPE)
 * - Heat map of encoding patterns
 * - Dimension-wise analysis
 * - Interactive position/dimension exploration
 */

import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../ui/card';
import { Button } from '../../ui/button';
import { Slider } from '../../ui/slider';
import { Select } from '../../ui/select';
import {
  Activity,
  Settings,
  Waves,
  Zap,
  Eye,
  RotateCcw,
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

type EncodingType = 'sinusoidal' | 'learned' | 'rope';

interface PositionEncodingProps {
  className?: string;
  maxSequenceLength?: number;
  dModel?: number;
}

// Generate sinusoidal positional encoding
function generateSinusoidalEncoding(
  pos: number,
  dModel: number,
  dim: number
): number {
  // Apply positional encoding formula
  const k = Math.floor(dim / 2);
  const divTerm = Math.pow(10000, (2 * k) / dModel);

  if (dim % 2 === 0) {
    // Even dimensions: sin
    return Math.sin(pos / divTerm);
  } else {
    // Odd dimensions: cos
    return Math.cos(pos / divTerm);
  }
}

// Generate sinusoidal encoding matrix
function generateSinusoidalMatrix(
  maxPos: number,
  dModel: number
): number[][] {
  const matrix: number[][] = [];

  for (let pos = 0; pos < maxPos; pos++) {
    const row: number[] = [];
    for (let dim = 0; dim < dModel; dim++) {
      row.push(generateSinusoidalEncoding(pos, dModel, dim));
    }
    matrix.push(row);
  }

  return matrix;
}

// Generate learned positional encoding (simulated)
function generateLearnedEncoding(
  maxPos: number,
  dModel: number
): number[][] {
  const matrix: number[][] = [];

  // Simulate learned embeddings with smooth patterns
  for (let pos = 0; pos < maxPos; pos++) {
    const row: number[] = [];
    for (let dim = 0; dim < dModel; dim++) {
      // Create smooth learned-like patterns
      const freq = Math.pow(2, Math.floor(dim / 2) / (dModel / 4));
      const phase = (dim % 2) * (Math.PI / 2);
      row.push(Math.sin((pos / 10) * freq + phase) * (0.5 + (dim / dModel) * 0.5));
    }
    matrix.push(row);
  }

  return matrix;
}

// Generate Rotary Position Embedding
function generateRoPEEncoding(
  maxPos: number,
  dModel: number
): number[][] {
  const matrix: number[][] = [];

  for (let pos = 0; pos < maxPos; pos++) {
    const row: number[] = [];
    for (let dim = 0; dim < dModel; dim++) {
      // RoPE uses rotation matrices
      const k = Math.floor(dim / 2);
      const theta = pos / Math.pow(10000, (2 * k) / dModel);

      if (dim % 2 === 0) {
        row.push(Math.cos(theta));
      } else {
        row.push(Math.sin(theta));
      }
    }
    matrix.push(row);
  }

  return matrix;
}

// Get encoding matrix based on type
function getEncodingMatrix(
  type: EncodingType,
  maxPos: number,
  dModel: number
): number[][] {
  switch (type) {
    case 'sinusoidal':
      return generateSinusoidalMatrix(maxPos, dModel);
    case 'learned':
      return generateLearnedEncoding(maxPos, dModel);
    case 'rope':
      return generateRoPEEncoding(maxPos, dModel);
    default:
      return generateSinusoidalMatrix(maxPos, dModel);
  }
}

// Calculate frequency statistics
function calculateFrequencyStats(matrix: number[][], dimension: number) {
  const values = matrix.map((row) => row[dimension] || 0);

  // Find peaks
  const peaks: number[] = [];
  for (let i = 1; i < values.length - 1; i++) {
    if (values[i] > values[i - 1] && values[i] > values[i + 1]) {
      peaks.push(i);
    }
  }

  // Estimate wavelength (distance between peaks)
  let avgWavelength = 0;
  if (peaks.length > 1) {
    const distances = [];
    for (let i = 1; i < peaks.length; i++) {
      distances.push(peaks[i] - peaks[i - 1]);
    }
    avgWavelength = distances.reduce((a, b) => a + b, 0) / distances.length;
  }

  return {
    min: Math.min(...values),
    max: Math.max(...values),
    mean: values.reduce((a, b) => a + b, 0) / values.length,
    wavelength: avgWavelength || 0,
    peaks: peaks.length,
  };
}

export const PositionEncodingExplorer: React.FC<PositionEncodingProps> = ({
  className = '',
  maxSequenceLength = 50,
  dModel = 64,
}) => {
  const [encodingType, setEncodingType] = useState<EncodingType>('sinusoidal');
  const [selectedPosition, setSelectedPosition] = useState(0);
  const [selectedDimension, setSelectedDimension] = useState(0);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [showWaveform, setShowWaveform] = useState(true);
  const [show3DView, setShow3DView] = useState(false);

  // Generate encoding matrix
  const encodingMatrix = useMemo(
    () => getEncodingMatrix(encodingType, maxSequenceLength, dModel),
    [encodingType, maxSequenceLength, dModel]
  );

  // Get dimension across all positions
  const dimensionWaveform = useMemo(
    () => encodingMatrix.map((row) => row[selectedDimension] || 0),
    [encodingMatrix, selectedDimension]
  );

  // Calculate frequency statistics
  const freqStats = useMemo(
    () => calculateFrequencyStats(encodingMatrix, selectedDimension),
    [encodingMatrix, selectedDimension]
  );

  // Prepare heatmap data
  const heatmapData = useMemo(() => {
    const data: Array<{ pos: number; [key: string]: number }> = [];

    for (let pos = 0; pos < Math.min(maxSequenceLength, 50); pos++) {
      const row: { pos: number; [key: string]: number } = { pos };

      for (let dim = 0; dim < Math.min(dModel, 20); dim++) {
        row[`dim${dim}`] = encodingMatrix[pos]?.[dim] || 0;
      }

      data.push(row);
    }

    return data;
  }, [encodingMatrix, maxSequenceLength, dModel]);

  // Prepare waveform data
  const waveformChartData = useMemo(
    () =>
      dimensionWaveform.map((value, pos) => ({
        position: pos,
        value,
      })),
    [dimensionWaveform]
  );

  // Prepare multi-dimension comparison data
  const multiDimData = useMemo(() => {
    const dims = [0, 2, 4, 8, 16, 32];
    const data: Array<{ position: number; [key: string]: number }> = [];

    for (let pos = 0; pos < Math.min(maxSequenceLength, 50); pos++) {
      const row: { position: number; [key: string]: number } = { position: pos };

      dims.forEach((dim) => {
        if (dim < dModel) {
          row[`Dim ${dim}`] = encodingMatrix[pos]?.[dim] || 0;
        }
      });

      data.push(row);
    }

    return data;
  }, [encodingMatrix, maxSequenceLength, dModel]);

  // Color scale for heatmap
  const getHeatmapColor = (value: number) => {
    const intensity = Math.abs(value);
    const hue = value > 0 ? 210 : 0; // Blue for positive, red for negative
    return `hsla(${hue}, 70%, ${50 + intensity * 40}%, ${intensity * 0.8})`;
  };

  // Reset view
  const resetView = () => {
    setSelectedPosition(0);
    setSelectedDimension(0);
    setShowHeatmap(true);
    setShowWaveform(true);
    setShow3DView(false);
  };

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Waves className="h-5 w-5 text-primary" />
              Position Encoding Explorer
            </CardTitle>
            <CardDescription>
              Understand how Transformers encode position information
            </CardDescription>
          </div>
          <Button variant="outline" size="sm" onClick={resetView}>
            <RotateCcw className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>

      <CardContent className="space-y-6" key={`content-${encodingType}`}>
        {/* Controls */}
        <div className="space-y-4 p-4 bg-muted rounded-lg">
          {/* Encoding Type Selection */}
          <div className="flex items-center gap-4">
            <label className="text-sm font-medium min-w-[120px]">Encoding Type:</label>
            <Select
              value={encodingType}
              onChange={(e) => setEncodingType(e.target.value as EncodingType)}
              className="w-[200px]"
            >
              <option value="sinusoidal">Sinusoidal</option>
              <option value="learned">Learned</option>
              <option value="rope">RoPE</option>
            </Select>
            <div className="text-xs text-muted-foreground ml-4">
              {encodingType === 'sinusoidal' && 'Fixed sinusoidal functions with different frequencies'}
              {encodingType === 'learned' && 'Learned embedding vectors (simulated)'}
              {encodingType === 'rope' && 'Rotary Position Embedding with rotation'}
            </div>
          </div>

          {/* Position Selection */}
          <div className="space-y-2">
            <div className="flex items-center gap-4">
              <label className="text-sm font-medium min-w-[120px]">Position:</label>
              <div className="flex-1">
                <Slider
                  value={[selectedPosition]}
                  onValueChange={([v]) => setSelectedPosition(v)}
                  min={0}
                  max={maxSequenceLength - 1}
                  step={1}
                  className="w-full"
                />
              </div>
              <span className="text-sm font-mono w-16 text-right">{selectedPosition}</span>
            </div>
          </div>

          {/* Dimension Selection */}
          <div className="space-y-2">
            <div className="flex items-center gap-4">
              <label className="text-sm font-medium min-w-[120px]">Dimension:</label>
              <div className="flex-1">
                <Slider
                  value={[selectedDimension]}
                  onValueChange={([v]) => setSelectedDimension(v)}
                  min={0}
                  max={dModel - 1}
                  step={1}
                  className="w-full"
                />
              </div>
              <span className="text-sm font-mono w-16 text-right">{selectedDimension}</span>
            </div>
          </div>

          {/* View Toggles */}
          <div className="flex items-center gap-2 flex-wrap">
            <Button
              variant={showHeatmap ? 'default' : 'outline'}
              size="sm"
              onClick={() => setShowHeatmap(!showHeatmap)}
            >
              <Eye className="h-4 w-4 mr-1" />
              Heatmap
            </Button>
            <Button
              variant={showWaveform ? 'default' : 'outline'}
              size="sm"
              onClick={() => setShowWaveform(!showWaveform)}
            >
              <Activity className="h-4 w-4 mr-1" />
              Waveform
            </Button>
            <Button
              variant={show3DView ? 'default' : 'outline'}
              size="sm"
              onClick={() => setShow3DView(!show3DView)}
            >
              <Zap className="h-4 w-4 mr-1" />
              3D View
            </Button>
          </div>
        </div>

        {/* Statistics Card */}
        <div className="grid md:grid-cols-4 gap-4">
          <div className="p-4 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-lg hover:scale-[1.02] transition-transform">
            <div className="text-xs text-blue-600 dark:text-blue-400 mb-1">Dimension Value</div>
            <div className="text-2xl font-bold font-mono">
              {encodingMatrix[selectedPosition]?.[selectedDimension]?.toFixed(4) || '0.0000'}
            </div>
          </div>

          <div className="p-4 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-lg hover:scale-[1.02] transition-transform">
            <div className="text-xs text-green-600 dark:text-green-400 mb-1">Frequency Range</div>
            <div className="text-2xl font-bold">
              [{freqStats.min.toFixed(2)}, {freqStats.max.toFixed(2)}]
            </div>
          </div>

          <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 rounded-lg hover:scale-[1.02] transition-transform">
            <div className="text-xs text-purple-600 dark:text-purple-400 mb-1">Est. Wavelength</div>
            <div className="text-2xl font-bold">
              {freqStats.wavelength > 0 ? freqStats.wavelength.toFixed(1) : 'N/A'}
            </div>
          </div>

          <div className="p-4 bg-gradient-to-br from-orange-50 to-orange-100 dark:from-orange-900/20 dark:to-orange-800/20 rounded-lg hover:scale-[1.02] transition-transform">
            <div className="text-xs text-orange-600 dark:text-orange-400 mb-1">Peaks Detected</div>
            <div className="text-2xl font-bold">{freqStats.peaks}</div>
          </div>
        </div>

        {/* Heatmap View */}
        {showHeatmap && (
          <div className="space-y-3">
            <h3 className="text-sm font-medium flex items-center gap-2">
              <Waves className="h-4 w-4" />
              Position Encoding Heatmap
            </h3>
            <div className="bg-muted rounded-lg p-4">
              {/* Heatmap Grid */}
              <div className="space-y-1">
                {heatmapData.slice(0, 25).map((row, posIdx) => (
                  <div key={posIdx} className="flex gap-1">
                    <div className="w-8 h-6 flex items-center justify-center text-xs text-muted-foreground">
                      {posIdx}
                    </div>
                    {Object.keys(row)
                      .filter((key) => key.startsWith('dim'))
                      .slice(0, 20)
                      .map((dimKey) => {
                        const value = row[dimKey];
                        const bgColor = getHeatmapColor(value);
                        return (
                          <div
                            key={dimKey}
                            className="w-6 h-6 rounded-sm flex items-center justify-center text-[8px] font-mono"
                            style={{ backgroundColor: bgColor }}
                            title={`Pos ${posIdx}, ${dimKey}: ${value.toFixed(3)}`}
                          >
                            {value > 0 ? '+' : '-'}
                          </div>
                        );
                      })}
                  </div>
                ))}
              </div>

              {/* Dimension Labels */}
              <div className="flex ml-9 mt-1">
                {Array.from({ length: Math.min(20, dModel) }, (_, i) => (
                  <div key={i} className="w-6 h-4 flex items-center justify-center text-[8px] text-muted-foreground">
                    {i}
                  </div>
                ))}
              </div>

              {/* Legend */}
              <div className="flex items-center gap-4 mt-3 text-xs">
                <span className="text-muted-foreground">Legend:</span>
                <div className="flex items-center gap-1">
                  <div className="w-4 h-4 rounded-sm" style={{ backgroundColor: 'hsla(210, 70%, 90%, 0.8)' }} />
                  <span>-1.0</span>
                </div>
                <div className="flex items-center gap-1">
                  <div className="w-4 h-4 rounded-sm" style={{ backgroundColor: 'hsla(210, 70%, 50%, 0.5)' }} />
                  <span>0.0</span>
                </div>
                <div className="flex items-center gap-1">
                  <div className="w-4 h-4 rounded-sm" style={{ backgroundColor: 'hsla(210, 70%, 90%, 0.8)' }} />
                  <span>+1.0</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Waveform View */}
        {showWaveform && (
          <div className="space-y-3">
            <h3 className="text-sm font-medium flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Dimension {selectedDimension} Across Positions
            </h3>
            <div key={`waveform-${encodingType}-${selectedDimension}`}>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={waveformChartData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis
                    dataKey="position"
                    label={{ value: 'Position', position: 'insideBottom', offset: -5 }}
                    className="text-xs"
                  />
                  <YAxis
                    label={{ value: 'Encoding Value', angle: -90, position: 'insideLeft' }}
                    className="text-xs"
                    domain={[-1.1, 1.1]}
                  />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const value = payload[0].value;
                        const numericValue = typeof value === 'number' ? value : parseFloat(String(value));
                        return (
                          <div className="bg-background border rounded-lg p-2 shadow-lg">
                            <p className="text-xs">Position: {payload[0].payload.position}</p>
                            <p className="text-xs font-mono">Value: {isNaN(numericValue) ? 'N/A' : numericValue.toFixed(4)}</p>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Multi-Dimension Comparison */}
        {showWaveform && (
          <div className="space-y-3">
            <h3 className="text-sm font-medium flex items-center gap-2">
              <Settings className="h-4 w-4" />
              Multi-Dimension Comparison
            </h3>
            <div key={`multidim-${encodingType}`}>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={multiDimData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis
                    dataKey="position"
                    label={{ value: 'Position', position: 'insideBottom', offset: -5 }}
                    className="text-xs"
                  />
                  <YAxis
                    label={{ value: 'Encoding Value', angle: -90, position: 'insideLeft' }}
                    className="text-xs"
                    domain={[-1.1, 1.1]}
                  />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="Dim 0" stroke="#ef4444" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="Dim 2" stroke="#f97316" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="Dim 4" stroke="#eab308" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="Dim 8" stroke="#22c55e" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="Dim 16" stroke="#3b82f6" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="Dim 32" stroke="#8b5cf6" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* 3D Visualization */}
        {show3DView && (
          <div className="space-y-3">
            <h3 className="text-sm font-medium flex items-center gap-2">
              <Zap className="h-4 w-4" />
              3D Encoding Space Visualization
            </h3>
            <div className="p-4 bg-muted rounded-lg">
              <div className="relative w-full h-[300px] bg-background rounded-lg border overflow-hidden">
                {/* 3D-style scatter plot */}
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="relative w-full h-full p-4">
                    {/* Y-axis */}
                    <div className="absolute left-8 top-4 bottom-8 w-px bg-muted-foreground"></div>
                    <div className="absolute left-2 top-1/2 -translate-y-1/2 -rotate-90 text-xs text-muted-foreground">
                      Dim 2
                    </div>

                    {/* X-axis */}
                    <div className="absolute left-8 right-4 bottom-8 h-px bg-muted-foreground"></div>
                    <div className="absolute bottom-2 left-1/2 -translate-x-1/2 text-xs text-muted-foreground">
                      Dim 0
                    </div>

                    {/* Z-axis (diagonal) */}
                    <div className="absolute left-8 bottom-8 w-32 h-px bg-muted-foreground origin-left -rotate-45"></div>
                    <div className="absolute left-1/3 bottom-16 -rotate-45 text-xs text-muted-foreground">
                      Position
                    </div>

                    {/* Data points */}
                    <div className="absolute inset-12">
                      {encodingMatrix.slice(0, 30).map((row, pos) => {
                        const x = 50 + row[0] * 40;
                        const y = 50 - row[2] * 40;
                        const z = pos * 2;
                        const scale = 0.5 + (pos / 30) * 0.5;
                        const opacity = 0.3 + (pos / 30) * 0.7;
                        const color = pos === selectedPosition ? '#ec4899' : '#3b82f6';

                        return (
                          <div
                            key={pos}
                            className="absolute w-3 h-3 rounded-full cursor-pointer transition-all hover:scale-150"
                            style={{
                              left: `${x + z * 0.3}%`,
                              top: `${y - z * 0.2}%`,
                              backgroundColor: color,
                              opacity,
                              transform: `scale(${pos === selectedPosition ? 1.5 : scale})`,
                              zIndex: Math.floor(pos),
                            }}
                            title={`Pos ${pos}: [${row[0].toFixed(2)}, ${row[1].toFixed(2)}, ${row[2].toFixed(2)}]`}
                            onClick={() => setSelectedPosition(pos)}
                          />
                        );
                      })}
                    </div>

                    {/* Selected position indicator */}
                    {selectedPosition < 30 && (
                      <div className="absolute top-4 right-4 bg-background/90 backdrop-blur border rounded p-2 text-xs">
                        <div className="font-medium">Position {selectedPosition}</div>
                        <div className="text-muted-foreground font-mono">
                          [{encodingMatrix[selectedPosition]?.[0]?.toFixed(2) || '0.00'}, {encodingMatrix[selectedPosition]?.[1]?.toFixed(2) || '0.00'}, {encodingMatrix[selectedPosition]?.[2]?.toFixed(2) || '0.00'}]
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* 3D controls legend */}
              <div className="mt-3 text-xs text-muted-foreground text-center">
                Shows first 30 positions across 3 dimensions (Dim 0, Dim 1, Dim 2). Click points to select position.
              </div>
            </div>
          </div>
        )}

        {/* Encoding Explanation */}
        <div className="p-4 bg-muted rounded-lg space-y-2">
          <h3 className="text-sm font-medium">Encoding Formula</h3>
          {encodingType === 'sinusoidal' && (
            <div className="space-y-2 text-xs font-mono">
              <p>
                For even dimensions: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
              </p>
              <p>
                For odd dimensions: PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
              </p>
              <p className="text-muted-foreground mt-2">
                Dimension {selectedDimension}: Frequency = 10000^({Math.floor(selectedDimension / 2) * 2}/{dModel})
              </p>
            </div>
          )}
          {encodingType === 'learned' && (
            <div className="space-y-2 text-xs text-muted-foreground">
              <p>Learned positional embeddings are learned parameters during training.</p>
              <p>Each position has a unique embedding vector optimized for the task.</p>
              <p className="mt-2">(This visualization shows simulated learned patterns)</p>
            </div>
          )}
          {encodingType === 'rope' && (
            <div className="space-y-2 text-xs font-mono">
              <p>RoPE applies rotation to query and key vectors based on position:</p>
              <p>
                θ(pos, 2i) = pos / 10000^(2i/d_model)
              </p>
              <p className="text-muted-foreground mt-2">
                Queries and keys are rotated by θ instead of adding positional embeddings.
              </p>
            </div>
          )}
        </div>

        {/* Key Insights */}
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
          <h3 className="text-sm font-medium mb-2 flex items-center gap-2">
            <Zap className="h-4 w-4 text-blue-600 dark:text-blue-400" />
            Key Insights
          </h3>
          <ul className="space-y-1 text-xs text-blue-800 dark:text-blue-200">
            <li>• Lower dimensions have lower frequencies (capture long-range patterns)</li>
            <li>• Higher dimensions have higher frequencies (capture fine-grained position info)</li>
            <li>• Each dimension creates a unique wavelength that helps distinguish positions</li>
            <li>• The combination of all dimensions creates unique encodings for each position</li>
            {encodingType === 'sinusoidal' && (
              <li>• Sinusoidal encoding generalizes to longer sequences than seen during training</li>
            )}
            {encodingType === 'rope' && (
              <li>• RoPE preserves relative positional information through rotation</li>
            )}
          </ul>
        </div>
      </CardContent>
    </Card>
  );
};
