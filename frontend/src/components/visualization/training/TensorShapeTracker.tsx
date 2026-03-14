/**
 * TensorShapeTracker Component
 *
 * Interactive debugging tool for tracking tensor shapes throughout Transformer:
 * - Visual representation of tensor shapes at each layer
 * - Interactive data flow exploration
 * - Dimension tracking through the network
 * - Memory usage estimation per tensor
 * - Shape change highlighting
 * - Batch size and sequence length impact visualization
 * - Common shape error detection
 * - Dimension compatibility checking
 */

import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../ui/card';
import { Button } from '../../ui/button';
import { Slider } from '../../ui/slider';
import {
  Box,
  ArrowRight,
  Layers,
  Database,
  AlertTriangle,
  CheckCircle,
  Info,
  ChevronDown,
  ChevronRight,
  Search,
  Zap,
  Eye,
  EyeOff,
  RotateCcw,
  Settings,
} from 'lucide-react';

// Tensor shape information
interface TensorShape {
  name: string;
  shape: number[];
  dtype: string;
  memoryMB: number;
  isIntermediate: boolean;
}

// Layer information
interface LayerInfo {
  name: string;
  type: string;
  inputShapes: TensorShape[];
  outputShapes: TensorShape[];
  parameters?: number;
  description: string;
}

// Configuration
interface TrackerConfig {
  batchSize: number;
  sequenceLength: number;
  d_model: number;
  nhead: number;
  numLayers: number;
  dimFeedforward: number;
  vocabSize: number;
}

// Generate layer shapes for a Transformer encoder
const generateEncoderShapes = (config: TrackerConfig): LayerInfo[] => {
  const { batchSize, sequenceLength, d_model, nhead, numLayers, dimFeedforward, vocabSize } =
    config;
  const layers: LayerInfo[] = [];

  // Input embedding layer
  layers.push({
    name: 'Input Embedding',
    type: 'embedding',
    inputShapes: [
      {
        name: 'input_ids',
        shape: [batchSize, sequenceLength],
        dtype: 'int64',
        memoryMB: (batchSize * sequenceLength * 8) / (1024 * 1024),
        isIntermediate: false,
      },
    ],
    outputShapes: [
      {
        name: 'embeddings',
        shape: [batchSize, sequenceLength, d_model],
        dtype: 'float32',
        memoryMB: (batchSize * sequenceLength * d_model * 4) / (1024 * 1024),
        isIntermediate: true,
      },
    ],
    parameters: vocabSize * d_model,
    description: 'Convert token IDs to dense embeddings and add positional encoding',
  });

  // Encoder layers
  for (let layer = 0; layer < numLayers; layer++) {
    // Self-Attention
    layers.push({
      name: `Encoder Layer ${layer + 1} - Self-Attention`,
      type: 'attention',
      inputShapes: [
        {
          name: 'hidden_states',
          shape: [batchSize, sequenceLength, d_model],
          dtype: 'float32',
          memoryMB: (batchSize * sequenceLength * d_model * 4) / (1024 * 1024),
          isIntermediate: true,
        },
      ],
      outputShapes: [
        {
          name: 'attention_output',
          shape: [batchSize, sequenceLength, d_model],
          dtype: 'float32',
          memoryMB: (batchSize * sequenceLength * d_model * 4) / (1024 * 1024),
          isIntermediate: true,
        },
        {
          name: 'attention_weights',
          shape: [batchSize, nhead, sequenceLength, sequenceLength],
          dtype: 'float32',
          memoryMB: (batchSize * nhead * sequenceLength * sequenceLength * 4) / (1024 * 1024),
          isIntermediate: true,
        },
      ],
      parameters: 4 * d_model * d_model,
      description: 'Multi-head self-attention with Q, K, V projections and output projection',
    });

    // Feed-Forward Network
    layers.push({
      name: `Encoder Layer ${layer + 1} - Feed-Forward`,
      type: 'ffn',
      inputShapes: [
        {
          name: 'attention_output',
          shape: [batchSize, sequenceLength, d_model],
          dtype: 'float32',
          memoryMB: (batchSize * sequenceLength * d_model * 4) / (1024 * 1024),
          isIntermediate: true,
        },
      ],
      outputShapes: [
        {
          name: 'ffn_output',
          shape: [batchSize, sequenceLength, d_model],
          dtype: 'float32',
          memoryMB: (batchSize * sequenceLength * d_model * 4) / (1024 * 1024),
          isIntermediate: true,
        },
        {
          name: 'intermediate',
          shape: [batchSize, sequenceLength, dimFeedforward],
          dtype: 'float32',
          memoryMB: (batchSize * sequenceLength * dimFeedforward * 4) / (1024 * 1024),
          isIntermediate: true,
        },
      ],
      parameters: d_model * dimFeedforward * 2,
      description: 'Two-layer feed-forward network with expansion and projection',
    });

    // Add & Norm
    layers.push({
      name: `Encoder Layer ${layer + 1} - Add & Norm`,
      type: 'norm',
      inputShapes: [
        {
          name: 'residual_input',
          shape: [batchSize, sequenceLength, d_model],
          dtype: 'float32',
          memoryMB: (batchSize * sequenceLength * d_model * 4) / (1024 * 1024),
          isIntermediate: true,
        },
        {
          name: 'ffn_output',
          shape: [batchSize, sequenceLength, d_model],
          dtype: 'float32',
          memoryMB: (batchSize * sequenceLength * d_model * 4) / (1024 * 1024),
          isIntermediate: true,
        },
      ],
      outputShapes: [
        {
          name: 'layer_output',
          shape: [batchSize, sequenceLength, d_model],
          dtype: 'float32',
          memoryMB: (batchSize * sequenceLength * d_model * 4) / (1024 * 1024),
          isIntermediate: layer < numLayers - 1,
        },
      ],
      parameters: d_model * 4,
      description: 'Residual connection followed by layer normalization',
    });
  }

  return layers;
};

export const TensorShapeTracker: React.FC<{ className?: string }> = ({ className = '' }) => {
  // State
  const [config, setConfig] = useState<TrackerConfig>({
    batchSize: 32,
    sequenceLength: 128,
    d_model: 512,
    nhead: 8,
    numLayers: 6,
    dimFeedforward: 2048,
    vocabSize: 50000,
  });
  const [expandedLayers, setExpandedLayers] = useState<Set<number>>(new Set([0]));
  const [searchQuery, setSearchQuery] = useState('');
  const [showIntermediate, setShowIntermediate] = useState(true);

  // Generate layer shapes
  const layerShapes = useMemo(() => {
    return generateEncoderShapes(config);
  }, [config]);

  // Filter layers by search
  const filteredLayers = useMemo(() => {
    if (!searchQuery) return layerShapes;
    return layerShapes.filter((layer) =>
      layer.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      layer.type.toLowerCase().includes(searchQuery.toLowerCase())
    );
  }, [layerShapes, searchQuery]);

  // Calculate total memory
  const totalMemory = useMemo(() => {
    let total = 0;
    layerShapes.forEach((layer) => {
      layer.outputShapes.forEach((tensor) => {
        if (tensor.isIntermediate || !tensor.name.includes('layer_output')) {
          total += tensor.memoryMB;
        }
      });
    });
    return total;
  }, [layerShapes]);

  // Detect potential issues
  const detectIssues = useMemo(() => {
    const issues: string[] = [];

    if (config.sequenceLength > 512) {
      issues.push(
        `Long sequence length (${config.sequenceLength}) may cause memory issues with attention`
      );
    }

    if (config.batchSize * config.sequenceLength * config.d_model > 50000000) {
      issues.push('Large intermediate tensors - consider gradient checkpointing');
    }

    if (config.d_model % config.nhead !== 0) {
      issues.push(`d_model (${config.d_model}) must be divisible by nhead (${config.nhead})`);
    }

    const attentionMemory =
      (config.batchSize *
        config.nhead *
        config.sequenceLength *
        config.sequenceLength *
        4) /
      (1024 * 1024);
    if (attentionMemory > 1000) {
      issues.push(`Attention weights memory: ${attentionMemory.toFixed(0)}MB - very large`);
    }

    return issues;
  }, [config]);

  // Toggle layer expansion
  const toggleLayer = (index: number) => {
    setExpandedLayers((prev) => {
      const next = new Set(prev);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  };

  // Expand/collapse all
  const expandAll = () => setExpandedLayers(new Set(filteredLayers.map((_, i) => i)));
  const collapseAll = () => setExpandedLayers(new Set());

  // Reset config
  const resetConfig = () => {
    setConfig({
      batchSize: 32,
      sequenceLength: 128,
      d_model: 512,
      nhead: 8,
      numLayers: 6,
      dimFeedforward: 2048,
      vocabSize: 50000,
    });
  };

  // Get layer type color
  const getLayerTypeColor = (type: string): string => {
    const colors: Record<string, string> = {
      embedding: 'from-blue-500 to-cyan-500',
      attention: 'from-purple-500 to-pink-500',
      ffn: 'from-green-500 to-emerald-500',
      norm: 'from-yellow-500 to-orange-500',
    };
    return colors[type] || 'from-gray-500 to-gray-600';
  };

  // Get layer type icon
  const getLayerTypeIcon = (type: string) => {
    const icons: Record<string, React.ReactNode> = {
      embedding: <Database className="h-4 w-4" />,
      attention: <Zap className="h-4 w-4" />,
      ffn: <Layers className="h-4 w-4" />,
      norm: <CheckCircle className="h-4 w-4" />,
    };
    return icons[type] || <Box className="h-4 w-4" />;
  };

  // Format shape
  const formatShape = (shape: number[]): string => {
    return `[${shape.join(', ')}]`;
  };

  // Format memory
  const formatMemory = (mb: number): string => {
    if (mb < 1) return `${(mb * 1024).toFixed(1)}KB`;
    return `${mb.toFixed(2)}MB`;
  };

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Box className="h-5 w-5 text-primary" />
              Tensor Shape Tracker
            </CardTitle>
            <CardDescription>
              Debug tensor shapes throughout the model
            </CardDescription>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={expandAll}>
              Expand All
            </Button>
            <Button variant="outline" size="sm" onClick={collapseAll}>
              Collapse All
            </Button>
            <Button variant="outline" size="sm" onClick={resetConfig}>
              <RotateCcw className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Configuration */}
        <div className="space-y-4 p-4 bg-muted rounded-lg">
          <h3 className="text-sm font-medium flex items-center gap-2">
            <Settings className="h-4 w-4" />
            Model Configuration
          </h3>

          <div className="grid md:grid-cols-2 gap-4">
            {/* Batch Size */}
            <div className="space-y-2">
              <div className="flex items-center gap-4">
                <label className="text-sm font-medium min-w-[100px]">Batch Size:</label>
                <Slider
                  value={[config.batchSize]}
                  onValueChange={([v]) => setConfig((prev) => ({ ...prev, batchSize: v }))}
                  min={1}
                  max={128}
                  step={1}
                  className="flex-1"
                />
                <span className="text-sm font-mono w-12 text-right">{config.batchSize}</span>
              </div>
            </div>

            {/* Sequence Length */}
            <div className="space-y-2">
              <div className="flex items-center gap-4">
                <label className="text-sm font-medium min-w-[100px]">Seq Length:</label>
                <Slider
                  value={[config.sequenceLength]}
                  onValueChange={([v]) => setConfig((prev) => ({ ...prev, sequenceLength: v }))}
                  min={32}
                  max={1024}
                  step={32}
                  className="flex-1"
                />
                <span className="text-sm font-mono w-12 text-right">{config.sequenceLength}</span>
              </div>
            </div>

            {/* d_model */}
            <div className="space-y-2">
              <div className="flex items-center gap-4">
                <label className="text-sm font-medium min-w-[100px]">d_model:</label>
                <Slider
                  value={[config.d_model]}
                  onValueChange={([v]) => setConfig((prev) => ({ ...prev, d_model: v }))}
                  min={128}
                  max={1024}
                  step={128}
                  className="flex-1"
                />
                <span className="text-sm font-mono w-12 text-right">{config.d_model}</span>
              </div>
            </div>

            {/* nhead */}
            <div className="space-y-2">
              <div className="flex items-center gap-4">
                <label className="text-sm font-medium min-w-[100px]">nhead:</label>
                <Slider
                  value={[config.nhead]}
                  onValueChange={([v]) => setConfig((prev) => ({ ...prev, nhead: v }))}
                  min={1}
                  max={16}
                  step={1}
                  className="flex-1"
                />
                <span className="text-sm font-mono w-12 text-right">{config.nhead}</span>
              </div>
            </div>

            {/* numLayers */}
            <div className="space-y-2">
              <div className="flex items-center gap-4">
                <label className="text-sm font-medium min-w-[100px]">Num Layers:</label>
                <Slider
                  value={[config.numLayers]}
                  onValueChange={([v]) => setConfig((prev) => ({ ...prev, numLayers: v }))}
                  min={1}
                  max={12}
                  step={1}
                  className="flex-1"
                />
                <span className="text-sm font-mono w-12 text-right">{config.numLayers}</span>
              </div>
            </div>

            {/* dimFeedforward */}
            <div className="space-y-2">
              <div className="flex items-center gap-4">
                <label className="text-sm font-medium min-w-[100px]">FFN Dim:</label>
                <Slider
                  value={[config.dimFeedforward]}
                  onValueChange={([v]) => setConfig((prev) => ({ ...prev, dimFeedforward: v }))}
                  min={512}
                  max={4096}
                  step={512}
                  className="flex-1"
                />
                <span className="text-sm font-mono w-16 text-right">{config.dimFeedforward}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Search and Filter */}
        <div className="flex gap-4 items-center">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search layers..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border rounded-md text-sm"
            />
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowIntermediate(!showIntermediate)}
          >
            {showIntermediate ? <Eye className="h-4 w-4 mr-2" /> : <EyeOff className="h-4 w-4 mr-2" />}
            {showIntermediate ? 'Hide' : 'Show'} Intermediate
          </Button>
        </div>

        {/* Statistics */}
        <div className="grid md:grid-cols-4 gap-4">
          <div className="p-4 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-lg">
            <div className="text-xs text-blue-600 dark:text-blue-400 mb-1">Total Layers</div>
            <div className="text-2xl font-bold">{layerShapes.length}</div>
            <div className="text-xs text-muted-foreground">operations</div>
          </div>

          <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 rounded-lg">
            <div className="text-xs text-purple-600 dark:text-purple-400 mb-1">Tensors</div>
            <div className="text-2xl font-bold">
              {layerShapes.reduce((sum, layer) => sum + layer.outputShapes.length, 0)}
            </div>
            <div className="text-xs text-muted-foreground">output tensors</div>
          </div>

          <div className="p-4 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-lg">
            <div className="text-xs text-green-600 dark:text-green-400 mb-1">Total Memory</div>
            <div className="text-2xl font-bold">{formatMemory(totalMemory)}</div>
            <div className="text-xs text-muted-foreground">activations</div>
          </div>

          <div className="p-4 bg-gradient-to-br from-orange-50 to-orange-100 dark:from-orange-900/20 dark:to-orange-800/20 rounded-lg">
            <div className="text-xs text-orange-600 dark:text-orange-400 mb-1">Output Shape</div>
            <div className="text-lg font-bold">
              [{config.batchSize}, {config.sequenceLength}, {config.d_model}]
            </div>
            <div className="text-xs text-muted-foreground">final output</div>
          </div>
        </div>

        {/* Issues */}
        {detectIssues.length > 0 && (
          <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
            <h3 className="text-sm font-medium mb-2 flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-yellow-600 dark:text-yellow-400" />
              Potential Issues Detected
            </h3>
            <ul className="space-y-1 text-xs text-yellow-800 dark:text-yellow-200">
              {detectIssues.map((issue, idx) => (
                <li key={idx}>• {issue}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Layer Shapes */}
        <div className="space-y-2">
          {filteredLayers.map((layer) => {
            const originalIndex = layerShapes.indexOf(layer);
            const isExpanded = expandedLayers.has(originalIndex);

            return (
              <div
                key={layer.name}
                className="border rounded-lg overflow-hidden transition-all border-border"
              >
                {/* Layer Header */}
                <div
                  className="p-3 bg-muted/50 cursor-pointer hover:bg-muted transition-colors"
                  onClick={() => toggleLayer(originalIndex)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      {isExpanded ? (
                        <ChevronDown className="h-4 w-4 text-muted-foreground" />
                      ) : (
                        <ChevronRight className="h-4 w-4 text-muted-foreground" />
                      )}
                      <div
                        className={`p-2 rounded-lg bg-gradient-to-br ${getLayerTypeColor(layer.type)}`}
                      >
                        {getLayerTypeIcon(layer.type)}
                      </div>
                      <div>
                        <div className="text-sm font-medium">{layer.name}</div>
                        <div className="text-xs text-muted-foreground">{layer.description}</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-4">
                      {layer.parameters && (
                        <div className="text-xs text-muted-foreground">
                          Params: {(layer.parameters / 1000000).toFixed(2)}M
                        </div>
                      )}
                      <div className="text-xs text-muted-foreground capitalize">{layer.type}</div>
                    </div>
                  </div>
                </div>

                {/* Layer Details */}
                {isExpanded && (
                  <div className="p-4 space-y-4 bg-background">
                    {/* Input Shapes */}
                    <div className="space-y-2">
                      <h4 className="text-xs font-medium text-muted-foreground">Input Tensors</h4>
                      <div className="grid gap-2">
                        {layer.inputShapes.map((tensor, idx) => (
                          <div
                            key={idx}
                            className="p-3 bg-muted rounded-lg flex items-center justify-between"
                          >
                            <div className="flex items-center gap-3">
                              <ArrowRight className="h-4 w-4 text-muted-foreground" />
                              <div>
                                <div className="text-sm font-medium">{tensor.name}</div>
                                <div className="text-xs text-muted-foreground">{tensor.dtype}</div>
                              </div>
                            </div>
                            <div className="text-right">
                              <div className="text-sm font-mono">{formatShape(tensor.shape)}</div>
                              <div className="text-xs text-muted-foreground">
                                {formatMemory(tensor.memoryMB)}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Operation Arrow */}
                    <div className="flex items-center justify-center">
                      <ArrowRight className="h-5 w-5 text-muted-foreground" />
                    </div>

                    {/* Output Shapes */}
                    <div className="space-y-2">
                      <h4 className="text-xs font-medium text-muted-foreground">Output Tensors</h4>
                      <div className="grid gap-2">
                        {layer.outputShapes
                          .filter((t) => showIntermediate || !t.isIntermediate)
                          .map((tensor, idx) => (
                            <div
                              key={idx}
                              className={`p-3 rounded-lg flex items-center justify-between ${
                                tensor.isIntermediate
                                  ? 'bg-muted'
                                  : 'bg-gradient-to-r from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 border border-blue-200 dark:border-blue-800'
                              }`}
                            >
                              <div className="flex items-center gap-3">
                                <Box className="h-4 w-4 text-primary" />
                                <div>
                                  <div className="text-sm font-medium">{tensor.name}</div>
                                  <div className="text-xs text-muted-foreground">
                                    {tensor.dtype}
                                    {tensor.isIntermediate && (
                                      <span className="ml-2 text-orange-600">
                                        (intermediate)
                                      </span>
                                    )}
                                  </div>
                                </div>
                              </div>
                              <div className="text-right">
                                <div className="text-sm font-mono">{formatShape(tensor.shape)}</div>
                                <div className="text-xs text-muted-foreground">
                                  {formatMemory(tensor.memoryMB)}
                                </div>
                              </div>
                            </div>
                          ))}
                      </div>
                    </div>

                    {/* Shape Changes */}
                    {layer.inputShapes.length > 0 && layer.outputShapes.length > 0 && (
                      <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
                        <div className="text-xs text-blue-800 dark:text-blue-200">
                          <strong>Shape transformation:</strong>{' '}
                          {formatShape(layer.inputShapes[0].shape)} →{' '}
                          {formatShape(layer.outputShapes[0].shape)}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Help */}
        <div className="p-4 bg-muted rounded-lg">
          <div className="flex items-start gap-2 text-xs text-muted-foreground">
            <Info className="h-4 w-4 mt-0.5 flex-shrink-0" />
            <div>
              <strong>Tip:</strong> Click on any layer to see detailed tensor shapes. Use the sliders to
              adjust model configuration and see how shapes change. Intermediate tensors are shown
              in gray, final outputs in blue. Memory estimates are for float32 tensors.
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
