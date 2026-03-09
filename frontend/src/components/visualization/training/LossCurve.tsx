/**
 * LossCurve Component
 *
 * Detailed loss curve visualization with:
 * - Train/validation loss comparison
 * - Multiple runs comparison
 * - Smoothing options
 * - Logarithmic scale
 * - Zoom and pan
 * - Statistical analysis
 */

import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../ui/card';
import { Button } from '../../ui/button';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Brush,
  Area,
  AreaChart,
} from 'recharts';
import { motion } from 'framer-motion';
import {
  TrendingDown,
  TrendingUp,
  Minus,
  ZoomOut,
  Activity,
  Eye,
  EyeOff,
} from 'lucide-react';

interface LossCurveProps {
  trainLoss: Array<{ epoch: number; value: number }>;
  validationLoss?: Array<{ epoch: number; value: number }>;
  className?: string;
}

interface ComparisonRun {
  id: string;
  name: string;
  trainLoss: Array<{ epoch: number; value: number }>;
  validationLoss?: Array<{ epoch: number; value: number }>;
  color: string;
}

// Generate mock comparison runs
function generateMockComparisonRuns(): ComparisonRun[] {
  return [
    {
      id: 'baseline',
      name: 'Baseline (Adam)',
      trainLoss: Array.from({ length: 50 }, (_, i) => ({
        epoch: i + 1,
        value: 2.5 * Math.exp(-i * 0.05) + 0.2 + (Math.random() - 0.5) * 0.1,
      })),
      validationLoss: Array.from({ length: 50 }, (_, i) => ({
        epoch: i + 1,
        value: 2.5 * Math.exp(-i * 0.05) + 0.35 + (Math.random() - 0.5) * 0.12,
      })),
      color: '#3b82f6',
    },
    {
      id: 'sgd',
      name: 'SGD + Momentum',
      trainLoss: Array.from({ length: 50 }, (_, i) => ({
        epoch: i + 1,
        value: 2.5 * Math.exp(-i * 0.03) + 0.25 + (Math.random() - 0.5) * 0.15,
      })),
      validationLoss: Array.from({ length: 50 }, (_, i) => ({
        epoch: i + 1,
        value: 2.5 * Math.exp(-i * 0.03) + 0.4 + (Math.random() - 0.5) * 0.18,
      })),
      color: '#ef4444',
    },
    {
      id: 'adamw',
      name: 'AdamW + Decay',
      trainLoss: Array.from({ length: 50 }, (_, i) => ({
        epoch: i + 1,
        value: 2.5 * Math.exp(-i * 0.055) + 0.18 + (Math.random() - 0.5) * 0.08,
      })),
      validationLoss: Array.from({ length: 50 }, (_, i) => ({
        epoch: i + 1,
        value: 2.5 * Math.exp(-i * 0.055) + 0.3 + (Math.random() - 0.5) * 0.1,
      })),
      color: '#22c55e',
    },
  ];
}

// Apply exponential moving average smoothing
function smoothLoss(
  data: Array<{ epoch: number; value: number }>,
  alpha: number = 0.3
): Array<{ epoch: number; value: number }> {
  const smoothed: Array<{ epoch: number; value: number }> = [];
  data.forEach((point, index) => {
    if (index === 0) {
      smoothed.push(point);
    } else {
      const prevValue = smoothed[index - 1].value;
      smoothed.push({
        epoch: point.epoch,
        value: alpha * point.value + (1 - alpha) * prevValue,
      });
    }
  });
  return smoothed;
}

// Calculate statistics
function calculateStats(data: Array<{ epoch: number; value: number }>) {
  if (data.length === 0) return null;

  const values = data.map((d) => d.value);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
  const std = Math.sqrt(variance);

  // Find convergence epoch (when loss stops improving significantly)
  let convergenceEpoch = data.length;
  const improvementThreshold = 0.01;
  for (let i = 5; i < data.length - 5; i++) {
    const recentImprovement =
      (data[i - 5].value - data[i].value) / data[i - 5].value;
    if (recentImprovement < improvementThreshold) {
      const nextImprovement =
        (data[i].value - data[i + 5].value) / data[i].value;
      if (nextImprovement < improvementThreshold) {
        convergenceEpoch = data[i].epoch;
        break;
      }
    }
  }

  return { min, max, mean, std, convergenceEpoch };
}

export const LossCurve: React.FC<LossCurveProps> = ({
  trainLoss,
  validationLoss,
  className = '',
}) => {
  // State
  const [showValidation, setShowValidation] = useState(true);
  const [showSmoothed, setShowSmoothed] = useState(false);
  const [smoothingFactor, setSmoothingFactor] = useState(0.3);
  const [useLogScale, setUseLogScale] = useState(false);
  const [zoomRange, setZoomRange] = useState<{ startIndex?: number; endIndex?: number }>({});
  const [showComparison, setShowComparison] = useState(false);
  const [comparisonRuns] = useState<ComparisonRun[]>(generateMockComparisonRuns);
  const [selectedRuns, setSelectedRuns] = useState<Set<string>>(new Set(['baseline']));

  // Calculate smoothed data
  const smoothedTrainLoss = useMemo(
    () => smoothLoss(trainLoss, smoothingFactor),
    [trainLoss, smoothingFactor]
  );

  const smoothedValidationLoss = useMemo(
    () => (validationLoss ? smoothLoss(validationLoss, smoothingFactor) : undefined),
    [validationLoss, smoothingFactor]
  );

  // Filter data based on zoom range
  const filteredTrainLoss = useMemo(() => {
    if (!zoomRange.startIndex && !zoomRange.endIndex) return trainLoss;
    return trainLoss.slice(zoomRange.startIndex || 0, zoomRange.endIndex || trainLoss.length);
  }, [trainLoss, zoomRange]);

  // Calculate statistics
  const trainStats = useMemo(() => calculateStats(trainLoss), [trainLoss]);
  const valStats = useMemo(
    () => (validationLoss ? calculateStats(validationLoss) : null),
    [validationLoss]
  );

  // Handle zoom
  const handleZoom = (newRange: { startIndex?: number; endIndex?: number }) => {
    setZoomRange(newRange);
  };

  const resetZoom = () => {
    setZoomRange({});
  };

  // Get trend icon
  const getTrendIcon = (current: number, previous: number) => {
    if (current < previous * 0.99) return <TrendingDown className="h-3 w-3 text-green-500" />;
    if (current > previous * 1.01) return <TrendingUp className="h-3 w-3 text-red-500" />;
    return <Minus className="h-3 w-3 text-gray-500" />;
  };

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload || !payload.length) return null;

    return (
      <div className="bg-background border rounded-lg p-3 shadow-lg">
        <p className="text-sm font-medium mb-2">Epoch {label}</p>
        {payload.map((entry: any, index: number) => (
          <div key={index} className="flex items-center gap-2 text-xs">
            <div
              className="w-3 h-3 rounded-sm"
              style={{ backgroundColor: entry.color }}
            />
            <span className="font-medium">{entry.name}:</span>
            <span>{entry.value.toFixed(4)}</span>
          </div>
        ))}
      </div>
    );
  };

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-primary" />
              Loss Curve
            </CardTitle>
            <CardDescription>
              Training and validation loss over time
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowComparison(!showComparison)}
            >
              {showComparison ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              Compare
            </Button>
            <Button variant="outline" size="sm" onClick={resetZoom}>
              <ZoomOut className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Statistics Cards */}
        <div className="grid md:grid-cols-2 gap-4">
          <motion.div
            className="p-4 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-lg"
            whileHover={{ scale: 1.02 }}
          >
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm font-medium text-blue-700 dark:text-blue-300">
                Train Loss
              </span>
              {trainStats && trainLoss.length > 1 && (
                getTrendIcon(trainLoss[trainLoss.length - 1].value, trainLoss[trainLoss.length - 2].value)
              )}
            </div>
            <div className="grid grid-cols-3 gap-2 text-xs">
              <div>
                <div className="text-blue-600 dark:text-blue-400">Min</div>
                <div className="font-mono font-bold">
                  {trainStats ? trainStats.min.toFixed(4) : 'N/A'}
                </div>
              </div>
              <div>
                <div className="text-blue-600 dark:text-blue-400">Mean</div>
                <div className="font-mono font-bold">
                  {trainStats ? trainStats.mean.toFixed(4) : 'N/A'}
                </div>
              </div>
              <div>
                <div className="text-blue-600 dark:text-blue-400">Std</div>
                <div className="font-mono font-bold">
                  {trainStats ? trainStats.std.toFixed(4) : 'N/A'}
                </div>
              </div>
            </div>
            {trainStats && (
              <div className="mt-2 pt-2 border-t border-blue-200 dark:border-blue-800">
                <div className="text-blue-600 dark:text-blue-400">Convergence</div>
                <div className="font-bold">Epoch {trainStats.convergenceEpoch}</div>
              </div>
            )}
          </motion.div>

          {validationLoss && valStats && (
            <motion.div
              className="p-4 bg-gradient-to-br from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-800/20 rounded-lg"
              whileHover={{ scale: 1.02 }}
            >
              <div className="flex items-center justify-between mb-3">
                <span className="text-sm font-medium text-red-700 dark:text-red-300">
                  Validation Loss
                </span>
                {getTrendIcon(
                  validationLoss[validationLoss.length - 1].value,
                  validationLoss[validationLoss.length - 2].value
                )}
              </div>
              <div className="grid grid-cols-3 gap-2 text-xs">
                <div>
                  <div className="text-red-600 dark:text-red-400">Min</div>
                  <div className="font-mono font-bold">
                    {valStats.min.toFixed(4)}
                  </div>
                </div>
                <div>
                  <div className="text-red-600 dark:text-red-400">Mean</div>
                  <div className="font-mono font-bold">
                    {valStats.mean.toFixed(4)}
                  </div>
                </div>
                <div>
                  <div className="text-red-600 dark:text-red-400">Std</div>
                  <div className="font-mono font-bold">
                    {valStats.std.toFixed(4)}
                  </div>
                </div>
              </div>
              {valStats && (
                <div className="mt-2 pt-2 border-t border-red-200 dark:border-red-800">
                  <div className="text-red-600 dark:text-red-400">Convergence</div>
                  <div className="font-bold">Epoch {valStats.convergenceEpoch}</div>
                </div>
              )}
            </motion.div>
          )}
        </div>

        {/* Controls */}
        <div className="flex flex-wrap items-center gap-3 p-3 bg-muted rounded-lg">
          <div className="flex items-center gap-2">
            <Button
              variant={showValidation ? 'default' : 'outline'}
              size="sm"
              onClick={() => setShowValidation(!showValidation)}
            >
              Validation
            </Button>
            <Button
              variant={showSmoothed ? 'default' : 'outline'}
              size="sm"
              onClick={() => setShowSmoothed(!showSmoothed)}
            >
              Smoothed
            </Button>
            <Button
              variant={useLogScale ? 'default' : 'outline'}
              size="sm"
              onClick={() => setUseLogScale(!useLogScale)}
            >
              Log Scale
            </Button>
          </div>

          {showSmoothed && (
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground">Smoothing:</span>
              <input
                type="range"
                min="0.05"
                max="0.95"
                step="0.05"
                value={smoothingFactor}
                onChange={(e) => setSmoothingFactor(Number(e.target.value))}
                className="w-24"
              />
              <span className="text-xs font-mono">{smoothingFactor.toFixed(2)}</span>
            </div>
          )}
        </div>

        {/* Main Chart */}
        <div className="space-y-2">
          <ResponsiveContainer width="100%" height={350}>
            <AreaChart data={filteredTrainLoss}>
              <defs>
                <linearGradient id="trainGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis
                dataKey="epoch"
                label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }}
                className="text-xs"
              />
              <YAxis
                label={{
                  value: useLogScale ? 'Log Loss' : 'Loss',
                  angle: -90,
                  position: 'insideLeft',
                }}
                className="text-xs"
                scale={useLogScale ? 'log' : 'auto'}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend />

              <Area
                type="monotone"
                dataKey="value"
                stroke="#3b82f6"
                fill="url(#trainGradient)"
                name="Train Loss"
              />

              {showSmoothed && (
                <Line
                  type="monotone"
                  dataKey="value"
                  data={smoothedTrainLoss}
                  stroke="#8b5cf6"
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  name="Smoothed Train"
                  dot={false}
                />
              )}

              {showValidation && validationLoss && (
                <Line
                  type="monotone"
                  dataKey="value"
                  data={validationLoss}
                  stroke="#ef4444"
                  strokeWidth={2}
                  name="Validation Loss"
                  dot={false}
                />
              )}

              {showValidation && showSmoothed && smoothedValidationLoss && (
                <Line
                  type="monotone"
                  dataKey="value"
                  data={smoothedValidationLoss}
                  stroke="#f97316"
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  name="Smoothed Val"
                  dot={false}
                />
              )}
            </AreaChart>
          </ResponsiveContainer>

          {/* Brush for zooming */}
          <ResponsiveContainer width="100%" height={60}>
            <LineChart data={trainLoss}>
              <Line
                type="monotone"
                dataKey="value"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={false}
              />
              {showValidation && validationLoss && (
                <Line
                  type="monotone"
                  dataKey="value"
                  data={validationLoss}
                  stroke="#ef4444"
                  strokeWidth={2}
                  dot={false}
                />
              )}
              <Brush
                onChange={(e: any) => {
                  if (e?.startIndex !== undefined && e?.endIndex !== undefined) {
                    handleZoom({ startIndex: e.startIndex, endIndex: e.endIndex });
                  }
                }}
                stroke="#3b82f6"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Comparison View */}
        {showComparison && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="space-y-3"
          >
            <h4 className="text-sm font-medium">Compare Runs</h4>
            <div className="flex flex-wrap gap-2">
              {comparisonRuns.map((run) => (
                <Button
                  key={run.id}
                  variant={selectedRuns.has(run.id) ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => {
                    const newSelected = new Set(selectedRuns);
                    if (newSelected.has(run.id)) {
                      newSelected.delete(run.id);
                    } else {
                      newSelected.add(run.id);
                    }
                    setSelectedRuns(newSelected);
                  }}
                  style={{
                    borderColor: run.color,
                    backgroundColor: selectedRuns.has(run.id) ? run.color : undefined,
                  }}
                >
                  {run.name}
                </Button>
              ))}
            </div>

            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={comparisonRuns[0].trainLoss}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis dataKey="epoch" className="text-xs" />
                <YAxis className="text-xs" />
                <Tooltip />
                <Legend />

                {comparisonRuns
                  .filter((run) => selectedRuns.has(run.id))
                  .map((run) => (
                    <React.Fragment key={run.id}>
                      <Line
                        type="monotone"
                        dataKey="value"
                        data={run.trainLoss}
                        stroke={run.color}
                        strokeWidth={2}
                        name={`${run.name} (Train)`}
                        dot={false}
                      />
                      {run.validationLoss && (
                        <Line
                          type="monotone"
                          dataKey="value"
                          data={run.validationLoss}
                          stroke={run.color}
                          strokeWidth={2}
                          strokeDasharray="5 5"
                          name={`${run.name} (Val)`}
                          dot={false}
                          opacity={0.7}
                        />
                      )}
                    </React.Fragment>
                  ))}
              </LineChart>
            </ResponsiveContainer>
          </motion.div>
        )}

        {/* Overfitting Detection */}
        {trainStats && valStats && (
          <div
            className={`p-4 rounded-lg border ${
              valStats.mean - trainStats.mean > 0.1
                ? 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800'
                : 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800'
            }`}
          >
            <div className="flex items-start gap-3">
              <Activity
                className={`h-5 w-5 ${
                  valStats.mean - trainStats.mean > 0.1
                    ? 'text-yellow-600 dark:text-yellow-400'
                    : 'text-green-600 dark:text-green-400'
                }`}
              />
              <div className="flex-1">
                <h4
                  className={`text-sm font-medium ${
                    valStats.mean - trainStats.mean > 0.1
                      ? 'text-yellow-800 dark:text-yellow-200'
                      : 'text-green-800 dark:text-green-200'
                  }`}
                >
                  {valStats.mean - trainStats.mean > 0.1
                    ? 'Potential Overfitting Detected'
                    : 'Training Healthy'}
                </h4>
                <p
                  className={`text-xs mt-1 ${
                    valStats.mean - trainStats.mean > 0.1
                      ? 'text-yellow-700 dark:text-yellow-300'
                      : 'text-green-700 dark:text-green-300'
                  }`}
                >
                  {valStats.mean - trainStats.mean > 0.1
                    ? `Validation loss is ${((valStats.mean - trainStats.mean) / trainStats.mean * 100).toFixed(1)}% higher than training loss. Consider regularization or early stopping.`
                    : 'Validation and training losses are well balanced. Model generalization appears good.'}
                </p>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};
