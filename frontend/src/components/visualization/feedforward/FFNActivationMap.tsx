/**
 * FFNActivationMap Component
 *
 * Heatmap visualization showing activation patterns in feed-forward networks
 * across different layers and positions. Includes comparison of different
 * activation functions (ReLU, GELU, Swish) and their effect on sparsity.
 */

import React, { useState, useCallback, useEffect, useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../ui/card';
import { Badge } from '../../ui/badge';
import { Button } from '../../ui/button';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Activity,
  GitCompare,
  Grid3x3,
  Eye,
  Info,
  Layers,
  RefreshCw,
} from 'lucide-react';

// Types
interface ActivationData {
  position: number;
  layer: number;
  activations: number[];
  statistics: {
    mean: number;
    std: number;
    sparsity: number;
    maxActivation: number;
    deadNeurons: number;
  };
}

interface LayerActivationSummary {
  layerIndex: number;
  meanActivation: number;
  sparsity: number;
  deadNeuronRatio: number;
  activationVariance: number;
}

interface FFNActivationMapProps {
  sequenceLength?: number;
  dimFeedforward?: number;
  numLayers?: number;
  className?: string;
}

// Mock activation data generator
function generateMockActivations(
  sequenceLength: number,
  dimFeedforward: number,
  layer: number,
  activationFn: 'relu' | 'gelu' | 'swish' = 'gelu'
): ActivationData[] {
  const data: ActivationData[] = [];

  for (let pos = 0; pos < sequenceLength; pos++) {
    // Generate activations based on activation function
    const activations: number[] = [];
    let deadNeurons = 0;

    for (let i = 0; i < Math.min(dimFeedforward, 128); i++) {
      let activation: number;

      // Base random values
      const baseValue = Math.random() * 2 - 1;

      switch (activationFn) {
        case 'relu':
          activation = Math.max(0, baseValue + (layer * 0.1));
          // ReLU tends to produce more zeros (sparse)
          if (Math.random() < 0.3 + layer * 0.05) {
            activation = 0;
            deadNeurons++;
          }
          break;
        case 'gelu':
          // GELU: x * Φ(x) where Φ is CDF of standard normal
          const gelu = baseValue * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (baseValue + 0.044715 * Math.pow(baseValue, 3))));
          activation = gelu * (1 + layer * 0.05);
          break;
        case 'swish':
          // Swish: x * sigmoid(βx)
          const swish = baseValue * (1 / (1 + Math.exp(-baseValue)));
          activation = swish * (1 + layer * 0.05);
          break;
        default:
          activation = baseValue;
      }

      activations.push(activation);
    }

    const mean = activations.reduce((sum, a) => sum + a, 0) / activations.length;
    const variance = activations.reduce((sum, a) => sum + Math.pow(a - mean, 2), 0) / activations.length;
    const sparsity = activations.filter(a => Math.abs(a) < 0.01).length / activations.length;
    const maxActivation = Math.max(...activations.map(a => Math.abs(a)));

    data.push({
      position: pos,
      layer,
      activations,
      statistics: {
        mean,
        std: Math.sqrt(variance),
        sparsity,
        maxActivation,
        deadNeurons: activationFn === 'relu' ? deadNeurons : 0,
      },
    });
  }

  return data;
}

function generateLayerSummary(
  activationFn: string,
  layerIndex: number,
  sequenceLength: number,
  dimFeedforward: number
): LayerActivationSummary {
  const mockData = generateMockActivations(sequenceLength, dimFeedforward, layerIndex, activationFn as any);

  const meanActivation = mockData.reduce((sum, d) => sum + d.statistics.mean, 0) / mockData.length;
  const avgSparsity = mockData.reduce((sum, d) => sum + d.statistics.sparsity, 0) / mockData.length;
  const avgDeadNeurons = mockData.reduce((sum, d) => sum + d.statistics.deadNeurons, 0) / mockData.length;
  const activationVariance = mockData.reduce((sum, d) => sum + d.statistics.std, 0) / mockData.length;

  return {
    layerIndex,
    meanActivation,
    sparsity: avgSparsity,
    deadNeuronRatio: avgDeadNeurons / dimFeedforward,
    activationVariance,
  };
}

const ACTIVATION_COLORS: Record<string, string> = {
  relu: '#3b82f6',      // blue
  gelu: '#22c55e',     // green
  swish: '#f59e0b',    // orange
  sigmoid: '#ef4444',  // red
  tanh: '#8b5cf6',     // purple
};

export const FFNActivationMap: React.FC<FFNActivationMapProps> = ({
  sequenceLength = 8,
  dimFeedforward = 2048,
  numLayers = 6,
  className = '',
}) => {
  // State
  const [activationData, setActivationData] = useState<ActivationData[]>([]);
  const [layerSummaries, setLayerSummaries] = useState<Record<string, LayerActivationSummary[]>>({});
  const [loading, setLoading] = useState(false);
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [selectedActivation, setSelectedActivation] = useState<'relu' | 'gelu' | 'swish'>('gelu');
  const [selectedPosition, setSelectedPosition] = useState<number | null>(null);
  const [viewMode, setViewMode] = useState<'heatmap' | 'neurons' | 'comparison'>('heatmap');
  const [colorScale, setColorScale] = useState<'viridis' | 'plasma' | 'coolwarm'>('viridis');

  // Load data
  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 500));

      // Generate activations for selected layer and activation function
      const data = generateMockActivations(sequenceLength, dimFeedforward, selectedLayer, selectedActivation);
      setActivationData(data);

      // Generate layer summaries for all activation functions
      const summaries: Record<string, LayerActivationSummary[]> = {};
      ['relu', 'gelu', 'swish'].forEach(fn => {
        summaries[fn] = Array.from({ length: Math.min(numLayers, 6) }, (_, i) =>
          generateLayerSummary(fn, i, sequenceLength, dimFeedforward)
        );
      });
      setLayerSummaries(summaries);
    } catch (error) {
      console.error('Failed to load activation data:', error);
    } finally {
      setLoading(false);
    }
  }, [sequenceLength, dimFeedforward, numLayers, selectedLayer, selectedActivation]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Get color for value
  const getColorForValue = useCallback((value: number, max: number): string => {
    const normalized = Math.min(Math.abs(value) / max, 1);

    if (colorScale === 'viridis') {
      // Purple (low) -> Blue -> Green -> Yellow (high)
      if (normalized < 0.25) {
        return `rgb(68, 1, 84)`;
      } else if (normalized < 0.5) {
        return `rgb(59, 82, 139)`;
      } else if (normalized < 0.75) {
        return `rgb(33, 154, 143)`;
      } else {
        return `rgb(253, 231, 37)`;
      }
    } else if (colorScale === 'plasma') {
      // Blue (low) -> Purple -> Red -> Yellow (high)
      if (normalized < 0.25) {
        return `rgb(13, 8, 135)`;
      } else if (normalized < 0.5) {
        return `rgb(147, 39, 143)`;
      } else if (normalized < 0.75) {
        return `rgb(222, 113, 87)`;
      } else {
        return `rgb(240, 249, 33)`;
      }
    } else {
      // Coolwarm: Blue (low) -> White -> Red (high)
      if (normalized < 0.5) {
        const blue = Math.floor(255 * (1 - normalized * 2));
        return `rgb(${blue}, ${blue}, 255)`;
      } else {
        const red = Math.floor(255 * ((normalized - 0.5) * 2));
        return `rgb(255, ${red}, ${red})`;
      }
    }
  }, [colorScale]);

  // Current layer statistics
  const currentStats = useMemo(() => {
    if (activationData.length === 0) return null;
    return {
      meanActivation: activationData.reduce((sum, d) => sum + d.statistics.mean, 0) / activationData.length,
      avgSparsity: activationData.reduce((sum, d) => sum + d.statistics.sparsity, 0) / activationData.length,
      avgStd: activationData.reduce((sum, d) => sum + d.statistics.std, 0) / activationData.length,
      totalDeadNeurons: activationData.reduce((sum, d) => sum + d.statistics.deadNeurons, 0),
    };
  }, [activationData]);

  // Comparison data
  const comparisonData = useMemo(() => {
    if (!layerSummaries[selectedActivation]) return [];
    return ['relu', 'gelu', 'swish'].map(fn => ({
      activationFunction: fn,
      summary: layerSummaries[fn]?.[selectedLayer] || layerSummaries[selectedActivation][selectedLayer],
      color: ACTIVATION_COLORS[fn] || '#888888',
    }));
  }, [layerSummaries, selectedActivation, selectedLayer]);

  if (loading) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center py-12">
          <RefreshCw className="h-8 w-8 animate-spin text-primary" />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-primary" />
              FFN Activation Map
            </CardTitle>
            <CardDescription>
              Visualize feed-forward network activation patterns
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={loadData}>
              <RefreshCw className="h-4 w-4 mr-1" />
              Refresh
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Controls */}
        <div className="flex flex-wrap gap-4">
          {/* View Mode */}
          <div className="flex gap-2">
            <Button
              variant={viewMode === 'heatmap' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setViewMode('heatmap')}
            >
              <Grid3x3 className="h-4 w-4 mr-1" />
              Heatmap
            </Button>
            <Button
              variant={viewMode === 'neurons' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setViewMode('neurons')}
            >
              <Eye className="h-4 w-4 mr-1" />
              Neurons
            </Button>
            <Button
              variant={viewMode === 'comparison' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setViewMode('comparison')}
            >
              <GitCompare className="h-4 w-4 mr-1" />
              Comparison
            </Button>
          </div>

          {/* Activation Function Selector */}
          <div className="flex items-center gap-2">
            {(['relu', 'gelu', 'swish'] as const).map(fn => (
              <Button
                key={fn}
                variant={selectedActivation === fn ? 'default' : 'outline'}
                size="sm"
                onClick={() => setSelectedActivation(fn)}
                style={{
                  backgroundColor: selectedActivation === fn ? ACTIVATION_COLORS[fn] : undefined,
                }}
              >
                {fn.toUpperCase()}
              </Button>
            ))}
          </div>
        </div>

        {/* Layer Selector */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium flex items-center gap-2">
              <Layers className="h-4 w-4" />
              Layer
            </label>
            <Badge variant="secondary">Layer {selectedLayer}</Badge>
          </div>
          <div className="flex gap-2 flex-wrap">
            {Array.from({ length: Math.min(numLayers, 6) }, (_, i) => (
              <Button
                key={i}
                variant={selectedLayer === i ? 'default' : 'outline'}
                size="sm"
                onClick={() => setSelectedLayer(i)}
              >
                {i + 1}
              </Button>
            ))}
          </div>
        </div>

        {/* Statistics Cards */}
        {currentStats && (
          <div className="grid grid-cols-4 gap-4">
            <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <div className="text-xs text-gray-600 dark:text-gray-400">Mean Activation</div>
              <div className="text-lg font-bold text-blue-700 dark:text-blue-300">
                {currentStats.meanActivation.toFixed(3)}
              </div>
            </div>
            <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
              <div className="text-xs text-gray-600 dark:text-gray-400">Avg Sparsity</div>
              <div className="text-lg font-bold text-purple-700 dark:text-purple-300">
                {(currentStats.avgSparsity * 100).toFixed(1)}%
              </div>
            </div>
            <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
              <div className="text-xs text-gray-600 dark:text-gray-400">Std Deviation</div>
              <div className="text-lg font-bold text-green-700 dark:text-green-300">
                {currentStats.avgStd.toFixed(3)}
              </div>
            </div>
            <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
              <div className="text-xs text-gray-600 dark:text-gray-400">Dead Neurons</div>
              <div className="text-lg font-bold text-orange-700 dark:text-orange-300">
                {currentStats.totalDeadNeurons}
              </div>
            </div>
          </div>
        )}

        {viewMode === 'heatmap' && (
          <>
            {/* Heatmap Visualization */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-medium">Activation Heatmap</h3>
                <select
                  value={colorScale}
                  onChange={(e) => setColorScale(e.target.value as any)}
                  className="text-xs border rounded px-2 py-1 bg-white dark:bg-gray-700"
                >
                  <option value="viridis">Viridis</option>
                  <option value="plasma">Plasma</option>
                  <option value="coolwarm">Cool-Warm</option>
                </select>
              </div>

              <div className="space-y-2">
                <AnimatePresence>
                  {activationData.map((data, posIdx) => {
                    const maxActivation = Math.max(...data.activations.map(a => Math.abs(a)));
                    const isSelected = selectedPosition === posIdx;

                    return (
                      <motion.div
                        key={posIdx}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        transition={{ delay: posIdx * 0.02 }}
                        className={`space-y-1 ${isSelected ? 'ring-2 ring-primary rounded' : ''}`}
                        onClick={() => setSelectedPosition(isSelected ? null : posIdx)}
                      >
                        <div className="flex items-center gap-2 text-xs">
                          <Badge variant="outline">Pos {posIdx}</Badge>
                          <span className="text-gray-500">
                            Mean: {data.statistics.mean.toFixed(3)} | Sparsity: {(data.statistics.sparsity * 100).toFixed(0)}%
                          </span>
                        </div>

                        {/* Activation bar */}
                        <div className="h-8 flex gap-px bg-gray-200 dark:bg-gray-800 rounded overflow-hidden">
                          {data.activations.slice(0, 100).map((activation, i) => (
                            <div
                              key={i}
                              className="flex-1 transition-all hover:opacity-80"
                              style={{
                                backgroundColor: getColorForValue(activation, maxActivation || 1),
                              }}
                              title={`Neuron ${i}: ${activation.toFixed(3)}`}
                            />
                          ))}
                        </div>
                      </motion.div>
                    );
                  })}
                </AnimatePresence>
              </div>
            </div>

            {/* Color Scale Legend */}
            <div className="flex items-center gap-2 text-xs">
              <span>Low</span>
              <div className="flex-1 h-4 rounded bg-gradient-to-r from-purple-900 via-blue-500 via-green-500 to-yellow-400 dark:from-purple-900 dark:via-blue-600 dark:via-green-600 dark:to-yellow-400" />
              <span>High</span>
            </div>
          </>
        )}

        {viewMode === 'neurons' && (
          <>
            {/* Individual Neuron Analysis */}
            <div className="space-y-3">
              <h3 className="text-sm font-medium">Neuron-wise Analysis (Top 20)</h3>

              <div className="grid grid-cols-10 gap-2">
                {Array.from({ length: 20 }, (_, neuronIdx) => {
                  const neuronActivations = activationData.map(d => d.activations[neuronIdx] || 0);
                  const max = Math.max(...neuronActivations.map(a => Math.abs(a)));
                  const avg = neuronActivations.reduce((sum, a) => sum + a, 0) / neuronActivations.length;

                  return (
                    <motion.div
                      key={neuronIdx}
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: neuronIdx * 0.02 }}
                      className="space-y-1"
                    >
                      <div className="text-xs text-center text-gray-500">N{neuronIdx}</div>
                      <div
                        className="h-20 rounded"
                        style={{ backgroundColor: getColorForValue(avg, max || 1) }}
                        title={`Avg: ${avg.toFixed(3)}, Max: ${max.toFixed(3)}`}
                      />
                      <div className="text-xs text-center text-gray-400">{avg.toFixed(2)}</div>
                    </motion.div>
                  );
                })}
              </div>
            </div>
          </>
        )}

        {viewMode === 'comparison' && (
          <>
            {/* Activation Function Comparison */}
            <div className="space-y-4">
              <h3 className="text-sm font-medium">Activation Function Comparison</h3>

              <div className="space-y-4">
                {comparisonData.map((comp, idx) => (
                  <motion.div
                    key={comp.activationFunction}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: idx * 0.1 }}
                    className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg"
                  >
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <div
                          className="w-3 h-3 rounded-full"
                          style={{ backgroundColor: comp.color }}
                        />
                        <h4 className="font-medium">{comp.activationFunction.toUpperCase()}</h4>
                      </div>
                    </div>

                    <div className="grid grid-cols-4 gap-3 text-sm">
                      <div>
                        <span className="text-gray-600 dark:text-gray-400">Mean Act:</span>
                        <div className="font-medium">{comp.summary.meanActivation.toFixed(3)}</div>
                      </div>
                      <div>
                        <span className="text-gray-600 dark:text-gray-400">Sparsity:</span>
                        <div className="font-medium">{(comp.summary.sparsity * 100).toFixed(1)}%</div>
                      </div>
                      <div>
                        <span className="text-gray-600 dark:text-gray-400">Variance:</span>
                        <div className="font-medium">{comp.summary.activationVariance.toFixed(3)}</div>
                      </div>
                      <div>
                        <span className="text-gray-600 dark:text-gray-400">Dead Neurons:</span>
                        <div className="font-medium">{(comp.summary.deadNeuronRatio * 100).toFixed(1)}%</div>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>

            {/* Characteristics */}
            <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <h4 className="text-sm font-medium mb-3">Activation Function Characteristics</h4>
              <div className="space-y-2 text-sm">
                <div className="flex items-start gap-2">
                  <div className="w-3 h-3 rounded-full bg-blue-500 mt-0.5" />
                  <div>
                    <span className="font-medium">ReLU:</span>
                    <span className="text-gray-600 dark:text-gray-400 ml-2">
                      Simple, produces sparse activations, prone to dead neurons
                    </span>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <div className="w-3 h-3 rounded-full bg-green-500 mt-0.5" />
                  <div>
                    <span className="font-medium">GELU:</span>
                    <span className="text-gray-600 dark:text-gray-400 ml-2">
                      Smooth, better gradient flow, used in BERT/GPT
                    </span>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <div className="w-3 h-3 rounded-full bg-orange-500 mt-0.5" />
                  <div>
                    <span className="font-medium">Swish:</span>
                    <span className="text-gray-600 dark:text-gray-400 ml-2">
                      Self-gated, smooth, performs well on deep networks
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}

        {/* Info Box */}
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
          <div className="flex items-start gap-3">
            <Info className="h-5 w-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
            <div className="text-sm text-blue-900 dark:text-blue-100">
              <h4 className="font-medium mb-1">Understanding FFN Activations</h4>
              <p className="text-blue-800 dark:text-blue-200">
                The feed-forward network applies two linear transformations with a non-linear activation.
                Visualizing activations helps understand information flow, identify dead neurons,
                and compare different activation functions. Sparse activations (many zeros) are common
                with ReLU, while smoother functions like GELU maintain gradient flow better.
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
