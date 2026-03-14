/**
 * WeightDistribution Component
 *
 * Visualization of neural network weight distributions:
 * - Histogram visualization of weight distributions
 * - Layer-by-layer weight distribution comparison
 * - Statistical analysis (mean, std, min, max)
 * - Normal distribution fitting
 * - Dead neuron detection
 - Weight initialization comparison
 * - Interactive layer selection
 * - Outlier detection
 * - Weight evolution over training
 */

import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../ui/card';
import { Button } from '../../ui/button';
import { Slider } from '../../ui/slider';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  LineChart,
  Line,
  AreaChart,
  Area,
} from 'recharts';
import {
  TrendingUp,
  Activity,
  AlertTriangle,
  CheckCircle,
  Zap,
  Layers,
  BarChart3,
  Eye,
} from 'lucide-react';

// Layer types
type LayerType = 'embedding' | 'encoder' | 'decoder' | 'ffn' | 'output';

// Weight statistics interface
interface WeightStats {
  layerName: string;
  layerType: LayerType;
  mean: number;
  std: number;
  min: number;
  max: number;
  median: number;
  deadNeurons: number;
  totalNeurons: number;
  outlierCount: number;
  skewness: number;
  kurtosis: number;
}

// Histogram bin interface
interface HistogramBin {
  range: string;
  count: number;
  expected: number;
}

type EvolutionPoint = {
  epoch: number;
} & Record<string, number | { mean: number; std: number }>;

// Generate mock weight data for a layer
function generateWeightData(
  layerName: string,
  layerType: LayerType,
  initMethod: 'xavier' | 'he' | 'uniform',
  size: number = 1000
): { weights: number[]; stats: WeightStats } {
  const weights: number[] = [];
  const fanIn = layerType === 'embedding' ? 100 : layerType === 'ffn' ? 512 : 768;
  const fanOut = layerType === 'embedding' ? 512 : layerType === 'ffn' ? 768 : 512;

  let std = 0;
  if (initMethod === 'xavier') {
    std = Math.sqrt(2 / (fanIn + fanOut));
  } else if (initMethod === 'he') {
    std = Math.sqrt(2 / fanIn);
  } else {
    std = 0.1;
  }

  // Generate weights with some outliers and potential dead neurons
  for (let i = 0; i < size; i++) {
    const u1 = Math.random();
    const u2 = Math.random();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    const weight = z * std;

    // Add some outliers
    if (Math.random() < 0.01) {
      weights.push(weight * (Math.random() > 0.5 ? 3 : -3));
    } else {
      weights.push(weight);
    }
  }

  // Calculate statistics
  const mean = weights.reduce((a, b) => a + b, 0) / weights.length;
  const variance = weights.reduce((sum, w) => sum + Math.pow(w - mean, 2), 0) / weights.length;
  const stdCalc = Math.sqrt(variance);
  const min = Math.min(...weights);
  const max = Math.max(...weights);
  const sorted = [...weights].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];

  // Count dead neurons (weights very close to zero)
  const deadNeurons = weights.filter((w) => Math.abs(w) < 0.001).length;

  // Count outliers (beyond 3 standard deviations)
  const outlierCount = weights.filter((w) => Math.abs(w - mean) > 3 * stdCalc).length;

  // Calculate skewness and kurtosis
  const skewness =
    weights.reduce((sum, w) => sum + Math.pow((w - mean) / stdCalc, 3), 0) / weights.length;
  const kurtosis =
    weights.reduce((sum, w) => sum + Math.pow((w - mean) / stdCalc, 4), 0) / weights.length - 3;

  return {
    weights,
    stats: {
      layerName,
      layerType,
      mean,
      std: stdCalc,
      min,
      max,
      median,
      deadNeurons,
      totalNeurons: size,
      outlierCount,
      skewness,
      kurtosis,
    },
  };
}

// Create histogram bins
function createHistogram(weights: number[], binCount: number = 50): HistogramBin[] {
  const min = Math.min(...weights);
  const max = Math.max(...weights);
  const binWidth = (max - min) / binCount;

  const bins: HistogramBin[] = [];
  for (let i = 0; i < binCount; i++) {
    const binStart = min + i * binWidth;
    const binEnd = binStart + binWidth;
    const count = weights.filter((w) => w >= binStart && w < binEnd).length;

    bins.push({
      range: `${binStart.toFixed(3)}`,
      count,
      expected: 0,
    });
  }

  // Calculate expected normal distribution
  const mean = weights.reduce((a, b) => a + b, 0) / weights.length;
  const std = Math.sqrt(
    weights.reduce((sum, w) => sum + Math.pow(w - mean, 2), 0) / weights.length
  );

  bins.forEach((bin) => {
    const binStart = parseFloat(bin.range);
    const binCenter = binStart + binWidth / 2;
    const z = (binCenter - mean) / std;
    const pdf = (1 / (std * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * z * z);
    const expected = pdf * binWidth * weights.length;
    bin.expected = expected;
  });

  return bins;
}

// Generate data for all layers
function generateAllLayerData(): { weights: number[]; stats: WeightStats }[] {
  const layers: LayerType[] = ['embedding', 'encoder', 'encoder', 'ffn', 'ffn', 'output'];
  const initMethods: Array<'xavier' | 'he' | 'uniform'> = ['xavier', 'he', 'he', 'xavier', 'xavier', 'xavier'];

  return layers.map((layerType, index) => {
    const layerName =
      layerType === 'embedding'
        ? 'Embedding Layer'
        : layerType === 'encoder'
        ? `Encoder Layer ${index + 1}`
        : layerType === 'ffn'
        ? `FFN Layer ${index - 1}`
        : 'Output Layer';

    const initMethod = initMethods[index];
    return generateWeightData(layerName, layerType, initMethod);
  });
}

interface WeightDistributionProps {
  className?: string;
}

export const WeightDistribution: React.FC<WeightDistributionProps> = ({ className = '' }) => {
  const [selectedLayerIndex, setSelectedLayerIndex] = useState(0);
  const [showNormalFit, setShowNormalFit] = useState(true);
  const [showOutliers, setShowOutliers] = useState(true);
  const [binCount, setBinCount] = useState(50);
  const [viewMode, setViewMode] = useState<'histogram' | 'comparison' | 'evolution'>('histogram');

  // Generate all layer data
  const allLayerData = useMemo(() => generateAllLayerData(), []);

  // Get selected layer data
  const selectedLayerData = allLayerData[selectedLayerIndex];
  const selectedStats = selectedLayerData.stats;

  // Create histogram for selected layer
  const histogram = useMemo(
    () => createHistogram(selectedLayerData.weights, binCount),
    [selectedLayerData.weights, binCount]
  );

  // Prepare comparison data
  const comparisonData = useMemo(() => {
    return allLayerData.map((layer) => ({
      name: layer.stats.layerName.split(' ')[0],
      mean: layer.stats.mean,
      std: layer.stats.std,
      deadNeurons: layer.stats.deadNeurons,
      outlierCount: layer.stats.outlierCount,
      layerType: layer.stats.layerType,
    }));
  }, [allLayerData]);

  // Prepare evolution data (simulated weight changes over training)
  const evolutionData = useMemo(() => {
    const epochs = [0, 1, 2, 3, 5, 10, 20];
    return epochs.map((epoch) => {
      const data: EvolutionPoint = { epoch };
      allLayerData.forEach((layer) => {
        const name = layer.stats.layerName.split(' ')[0];
        // Simulate weight statistics changing over training
        const decay = Math.exp(-epoch * 0.1);
        data[name] = {
          mean: layer.stats.mean * decay,
          std: layer.stats.std * (1 - epoch * 0.02),
        };
      });
      return data;
    });
  }, [allLayerData]);

  // Get health status
  const getHealthStatus = (stats: WeightStats) => {
    const deadRatio = stats.deadNeurons / stats.totalNeurons;
    const outlierRatio = stats.outlierCount / stats.totalNeurons;

    if (deadRatio > 0.1 || outlierRatio > 0.05) {
      return {
        status: 'warning',
        message: 'High number of dead neurons or outliers detected',
        icon: AlertTriangle,
        color: 'text-yellow-600 dark:text-yellow-400',
      };
    } else if (deadRatio > 0.05 || outlierRatio > 0.02) {
      return {
        status: 'caution',
        message: 'Some dead neurons or outliers present',
        icon: Activity,
        color: 'text-orange-600 dark:text-orange-400',
      };
    } else {
      return {
        status: 'healthy',
        message: 'Weight distribution looks healthy',
        icon: CheckCircle,
        color: 'text-green-600 dark:text-green-400',
      };
    }
  };

  const healthStatus = getHealthStatus(selectedStats);

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-primary" />
              Weight Distribution Analysis
            </CardTitle>
            <CardDescription>
              Analyze weight distributions across all layers
            </CardDescription>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Layer Selection */}
        <div className="space-y-3 p-4 bg-muted rounded-lg">
          <div className="flex items-center gap-4">
            <label className="text-sm font-medium min-w-[100px]">Layer:</label>
            <div className="flex-1">
              <Slider
                value={[selectedLayerIndex]}
                onValueChange={([v]) => setSelectedLayerIndex(v)}
                min={0}
                max={allLayerData.length - 1}
                step={1}
                className="w-full"
              />
            </div>
            <span className="text-sm font-mono w-48 text-right">{selectedStats.layerName}</span>
          </div>
        </div>

        {/* View Mode Toggle */}
        <div className="flex gap-2">
          <Button
            variant={viewMode === 'histogram' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setViewMode('histogram')}
          >
            <BarChart3 className="h-4 w-4 mr-1" />
            Histogram
          </Button>
          <Button
            variant={viewMode === 'comparison' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setViewMode('comparison')}
          >
            <Layers className="h-4 w-4 mr-1" />
            Comparison
          </Button>
          <Button
            variant={viewMode === 'evolution' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setViewMode('evolution')}
          >
            <TrendingUp className="h-4 w-4 mr-1" />
            Evolution
          </Button>
        </div>

        {/* Statistics Cards */}
        <div className="grid md:grid-cols-4 gap-4">
          <div className="p-4 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-lg">
            <div className="text-xs text-blue-600 dark:text-blue-400 mb-1">Mean</div>
            <div className="text-2xl font-bold font-mono">{selectedStats.mean.toFixed(4)}</div>
          </div>
          <div className="p-4 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-lg">
            <div className="text-xs text-green-600 dark:text-green-400 mb-1">Std Dev</div>
            <div className="text-2xl font-bold font-mono">{selectedStats.std.toFixed(4)}</div>
          </div>
          <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 rounded-lg">
            <div className="text-xs text-purple-600 dark:text-purple-400 mb-1">Dead Neurons</div>
            <div className="text-2xl font-bold">
              {selectedStats.deadNeurons} / {selectedStats.totalNeurons}
            </div>
          </div>
          <div className="p-4 bg-gradient-to-br from-orange-50 to-orange-100 dark:from-orange-900/20 dark:to-orange-800/20 rounded-lg">
            <div className="text-xs text-orange-600 dark:text-orange-400 mb-1">Outliers</div>
            <div className="text-2xl font-bold">{selectedStats.outlierCount}</div>
          </div>
        </div>

        {/* Health Status */}
        <div className={`p-4 rounded-lg border ${
          healthStatus.status === 'healthy'
            ? 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800'
            : healthStatus.status === 'caution'
            ? 'bg-orange-50 dark:bg-orange-900/20 border-orange-200 dark:border-orange-800'
            : 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800'
        }`}>
          <div className="flex items-start gap-3">
            <healthStatus.icon className={`h-5 w-5 ${healthStatus.color} flex-shrink-0 mt-0.5`} />
            <div>
              <div className={`text-sm font-medium ${healthStatus.color}`}>
                {healthStatus.status === 'healthy' ? 'Healthy' : healthStatus.status === 'caution' ? 'Caution' : 'Warning'}
              </div>
              <div className="text-xs text-muted-foreground mt-1">{healthStatus.message}</div>
            </div>
          </div>
        </div>

        {/* Histogram View */}
        {viewMode === 'histogram' && (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-medium">Weight Distribution Histogram</h3>
              <div className="flex gap-2">
                <Button
                  variant={showNormalFit ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setShowNormalFit(!showNormalFit)}
                >
                  <Eye className="h-4 w-4 mr-1" />
                  Normal Fit
                </Button>
                <Button
                  variant={showOutliers ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setShowOutliers(!showOutliers)}
                >
                  <AlertTriangle className="h-4 w-4 mr-1" />
                  Outliers
                </Button>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-4">
                <label className="text-sm font-medium">Bins:</label>
                <Slider
                  value={[binCount]}
                  onValueChange={([v]) => setBinCount(v)}
                  min={20}
                  max={100}
                  step={5}
                  className="flex-1"
                />
                <span className="text-sm font-mono w-12">{binCount}</span>
              </div>
            </div>

            <div className="bg-background rounded-lg p-4 border">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={histogram}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis
                    dataKey="range"
                    label={{ value: 'Weight Value', position: 'insideBottom', offset: -5 }}
                    className="text-xs"
                    tick={false}
                  />
                  <YAxis
                    label={{ value: 'Count', angle: -90, position: 'insideLeft' }}
                    className="text-xs"
                  />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        return (
                          <div className="bg-background border rounded-lg p-2 shadow-lg">
                            <p className="text-xs">Range: {payload[0].payload.range}</p>
                            <p className="text-xs">Count: {payload[0].value}</p>
                            {showNormalFit && (
                              <p className="text-xs text-muted-foreground">
                                Expected: {payload[0].payload.expected.toFixed(1)}
                              </p>
                            )}
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Legend />
                  <Bar dataKey="count" fill="#3b82f6" name="Actual Distribution" />
                  {showNormalFit && (
                    <Bar dataKey="expected" fill="#ef4444" fillOpacity={0.3} name="Normal Distribution" />
                  )}
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Comparison View */}
        {viewMode === 'comparison' && (
          <div className="space-y-4">
            <h3 className="text-sm font-medium">Layer Comparison</h3>

            <div className="bg-background rounded-lg p-4 border">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={comparisonData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis dataKey="name" className="text-xs" angle={-45} textAnchor="end" height={100} />
                  <YAxis className="text-xs" />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="mean" fill="#3b82f6" name="Mean" />
                  <Bar dataKey="std" fill="#22c55e" name="Std Dev" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-background rounded-lg p-4 border">
                <h4 className="text-sm font-medium mb-3">Dead Neurons by Layer</h4>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={comparisonData}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                    <XAxis dataKey="name" className="text-xs" angle={-45} textAnchor="end" height={80} />
                    <YAxis className="text-xs" />
                    <Tooltip />
                    <Bar dataKey="deadNeurons" fill="#ef4444" />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="bg-background rounded-lg p-4 border">
                <h4 className="text-sm font-medium mb-3">Outliers by Layer</h4>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={comparisonData}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                    <XAxis dataKey="name" className="text-xs" angle={-45} textAnchor="end" height={80} />
                    <YAxis className="text-xs" />
                    <Tooltip />
                    <Bar dataKey="outlierCount" fill="#f97316" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}

        {/* Evolution View */}
        {viewMode === 'evolution' && (
          <div className="space-y-4">
            <h3 className="text-sm font-medium">Weight Statistics Evolution Over Training</h3>

            <div className="bg-background rounded-lg p-4 border">
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={evolutionData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis
                    dataKey="epoch"
                    label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }}
                    className="text-xs"
                  />
                  <YAxis className="text-xs" />
                  <Tooltip />
                  <Legend />
                  {allLayerData.slice(0, 4).map((layer, idx) => {
                    const name = layer.stats.layerName.split(' ')[0];
                    return (
                      <Line
                        key={name}
                        type="monotone"
                        dataKey={`${name}.mean`}
                        stroke={['#3b82f6', '#22c55e', '#ef4444', '#f97316'][idx]}
                        strokeWidth={2}
                        dot={false}
                        name={`${name} Mean`}
                      />
                    );
                  })}
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-background rounded-lg p-4 border">
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={evolutionData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis
                    dataKey="epoch"
                    label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }}
                    className="text-xs"
                  />
                  <YAxis className="text-xs" />
                  <Tooltip />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="Embedding.std"
                    stroke="#3b82f6"
                    fill="#3b82f6"
                    fillOpacity={0.3}
                    name="Embedding Std"
                  />
                  <Area
                    type="monotone"
                    dataKey="Encoder.std"
                    stroke="#22c55e"
                    fill="#22c55e"
                    fillOpacity={0.3}
                    name="Encoder Std"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Detailed Statistics */}
        <div className="space-y-3">
          <h3 className="text-sm font-medium">Detailed Statistics</h3>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="p-4 bg-muted rounded-lg space-y-2">
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div>Min:</div>
                <div className="font-mono text-right">{selectedStats.min.toFixed(6)}</div>
                <div>Max:</div>
                <div className="font-mono text-right">{selectedStats.max.toFixed(6)}</div>
                <div>Median:</div>
                <div className="font-mono text-right">{selectedStats.median.toFixed(6)}</div>
                <div>Range:</div>
                <div className="font-mono text-right">{(selectedStats.max - selectedStats.min).toFixed(6)}</div>
              </div>
            </div>

            <div className="p-4 bg-muted rounded-lg space-y-2">
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div>Skewness:</div>
                <div className="font-mono text-right">{selectedStats.skewness.toFixed(4)}</div>
                <div>Kurtosis:</div>
                <div className="font-mono text-right">{selectedStats.kurtosis.toFixed(4)}</div>
                <div>Dead Neuron Ratio:</div>
                <div className="font-mono text-right">
                  {((selectedStats.deadNeurons / selectedStats.totalNeurons) * 100).toFixed(2)}%
                </div>
                <div>Outlier Ratio:</div>
                <div className="font-mono text-right">
                  {((selectedStats.outlierCount / selectedStats.totalNeurons) * 100).toFixed(2)}%
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Interpretation Guide */}
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
          <h3 className="text-sm font-medium mb-2 flex items-center gap-2">
            <Zap className="h-4 w-4 text-blue-600 dark:text-blue-400" />
            Interpretation Guide
          </h3>
          <ul className="space-y-1 text-xs text-blue-800 dark:text-blue-200">
            <li>• <strong>Healthy distribution:</strong> Bell-shaped curve centered near zero with low dead neuron count</li>
            <li>• <strong>Dead neurons:</strong> Weights very close to zero may indicate dying ReLU units</li>
            <li>• <strong>Outliers:</strong> Weights beyond 3σ may indicate training instability</li>
            <li>• <strong>Skewness:</strong> Non-zero values indicate asymmetric distribution</li>
            <li>• <strong>Kurtosis:</strong> High values indicate heavy tails (more outliers)</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  );
};
