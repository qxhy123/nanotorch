import React from 'react';
import { motion } from 'framer-motion';
import { ArrowDown, ChevronRight } from 'lucide-react';
import { Card } from '../../ui/card';
import { Badge } from '../../ui/badge';
import { Latex } from '../../ui/Latex';
import type { TransformerConfig } from '../../../types/transformer';

interface LayerNode {
  id: string;
  name: string;
  type: 'input' | 'embedding' | 'positional' | 'attention' | 'ffn' | 'output';
  description: string;
  inputShape: number[];
  outputShape: number[];
  parameters?: number;
  formula?: string;
}

interface ArchitectureFlowProps {
  config: TransformerConfig;
}

/**
 * ArchitectureFlow Component
 *
 * Displays Transformer architecture as an intuitive vertical flow diagram.
 * Each layer is shown as a card with detailed information.
 */
export const ArchitectureFlow: React.FC<ArchitectureFlowProps> = ({ config }) => {
  const [expandedLayers, setExpandedLayers] = React.useState<Set<string>>(new Set());

  const toggleLayer = (id: string) => {
    const newExpanded = new Set(expandedLayers);
    if (newExpanded.has(id)) {
      newExpanded.delete(id);
    } else {
      newExpanded.add(id);
    }
    setExpandedLayers(newExpanded);
  };

  // Format numbers for display
  const formatNumber = (num: number) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toLocaleString();
  };

  // Get color for layer type
  const getLayerColor = (type: string) => {
    const colors: Record<string, string> = {
      input: '#22c55e',
      embedding: '#8b5cf6',
      positional: '#06b6d4',
      attention: '#ec4899',
      ffn: '#f59e0b',
      output: '#ef4444',
    };
    return colors[type] || '#6366f1';
  };

  // Build architecture flow
  const buildLayers = (): LayerNode[] => {
    const layers: LayerNode[] = [];

    // Input
    layers.push({
      id: 'input',
      name: 'Input',
      type: 'input',
      description: 'Raw token IDs from vocabulary',
      inputShape: [config.max_seq_len || 128],
      outputShape: [config.max_seq_len || 128],
    });

    // Embedding
    const embeddingParams = (config.vocab_size || 10000) * (config.d_model || 512);
    layers.push({
      id: 'embedding',
      name: 'Token Embedding',
      type: 'embedding',
      description: 'Convert token IDs to dense vectors',
      inputShape: [config.max_seq_len || 128],
      outputShape: [config.max_seq_len || 128, config.d_model || 512],
      parameters: embeddingParams,
      formula: 'X \\cdot W_E',
    });

    // Positional Encoding
    layers.push({
      id: 'positional',
      name: 'Positional Encoding',
      type: 'positional',
      description: 'Add position information to embeddings',
      inputShape: [config.max_seq_len || 128, config.d_model || 512],
      outputShape: [config.max_seq_len || 128, config.d_model || 512],
      parameters: (config.max_seq_len || 128) * (config.d_model || 512),
      formula: 'PE(pos, 2i) = \\sin(\\frac{pos}{10000^{\\frac{2i}{d}}})',
    });

    // Encoder Layers
    const numLayersToShow = Math.min(config.num_encoder_layers || 6, 2);

    for (let i = 0; i < numLayersToShow; i++) {
      // Attention
      const attnParams = 4 * (config.d_model || 512) * (config.d_model || 512);
      layers.push({
        id: `layer-${i}-attn`,
        name: `Encoder Layer ${i + 1}: Multi-Head Attention`,
        type: 'attention',
        description: 'Self-attention mechanism allows each token to attend to all other tokens',
        inputShape: [config.max_seq_len || 128, config.d_model || 512],
        outputShape: [config.max_seq_len || 128, config.d_model || 512],
        parameters: attnParams,
        formula: '\\text{Attention}(Q, K, V) = \\text{softmax}(\\frac{QK^T}{\\sqrt{d_k}})V',
      });

      // Feed-Forward
      const ffnParams = 2 * (config.d_model || 512) * (config.dim_feedforward || 2048);
      layers.push({
        id: `layer-${i}-ffn`,
        name: `Encoder Layer ${i + 1}: Feed-Forward Network`,
        type: 'ffn',
        description: 'Two-layer neural network with ReLU activation',
        inputShape: [config.max_seq_len || 128, config.d_model || 512],
        outputShape: [config.max_seq_len || 128, config.d_model || 512],
        parameters: ffnParams,
        formula: '\\text{FFN}(x) = \\max(0, xW_1 + b_1)W_2 + b_2',
      });
    }

    // Output
    layers.push({
      id: 'output',
      name: 'Output',
      type: 'output',
      description: 'Final transformer output',
      inputShape: [config.max_seq_len || 128, config.d_model || 512],
      outputShape: [config.max_seq_len || 128, config.d_model || 512],
    });

    return layers;
  };

  const layers = buildLayers();

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="text-center mb-6">
        <h3 className="text-xl font-semibold mb-2">Transformer Architecture</h3>
        <p className="text-sm text-gray-600">
          Click on any layer to see detailed information
        </p>
      </div>

      {/* Architecture Flow */}
      <div className="space-y-3">
        {layers.map((layer, index) => {
          const isExpanded = expandedLayers.has(layer.id);
          const color = getLayerColor(layer.type);

          return (
            <motion.div
              key={layer.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              {/* Layer Card */}
              <Card
                className="cursor-pointer transition-all hover:shadow-lg"
                style={{ borderLeft: `4px solid ${color}` }}
                onClick={() => toggleLayer(layer.id)}
              >
                {/* Compact View */}
                <div className="flex items-center justify-between p-4">
                  <div className="flex items-center gap-3">
                    <div
                      className="w-10 h-10 rounded-lg flex items-center justify-center text-white"
                      style={{ backgroundColor: color }}
                    >
                      {layer.type === 'input' && '📝'}
                      {layer.type === 'embedding' && '🔤'}
                      {layer.type === 'positional' && '📍'}
                      {layer.type === 'attention' && '🔍'}
                      {layer.type === 'ffn' && '🧠'}
                      {layer.type === 'output' && '📤'}
                    </div>
                    <div>
                      <h4 className="font-semibold text-gray-900">{layer.name}</h4>
                      <p className="text-sm text-gray-600">{layer.description}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {layer.parameters && (
                      <Badge variant="outline" className="text-xs">
                        {formatNumber(layer.parameters)} params
                      </Badge>
                    )}
                    <motion.div
                      animate={{ rotate: isExpanded ? 90 : 0 }}
                      transition={{ duration: 0.2 }}
                    >
                      <ChevronRight className="w-5 h-5 text-gray-400" />
                    </motion.div>
                  </div>
                </div>

                {/* Expanded View */}
                {isExpanded && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.3 }}
                    className="border-t"
                  >
                    <div className="p-4 space-y-3 bg-gray-50">
                      {/* Shapes */}
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <div className="text-xs text-gray-500 mb-1">Input Shape</div>
                          <div className="font-mono text-sm bg-white px-2 py-1 rounded border">
                            [{layer.inputShape.join(', ')}]
                          </div>
                        </div>
                        <div>
                          <div className="text-xs text-gray-500 mb-1">Output Shape</div>
                          <div className="font-mono text-sm bg-white px-2 py-1 rounded border">
                            [{layer.outputShape.join(', ')}]
                          </div>
                        </div>
                      </div>

                      {/* Formula */}
                      {layer.formula && (
                        <div>
                          <div className="text-xs text-gray-500 mb-1">Formula</div>
                          <div className="bg-white px-3 py-2 rounded border">
                            <Latex display={false}>{layer.formula}</Latex>
                          </div>
                        </div>
                      )}

                      {/* Parameters breakdown */}
                      {layer.parameters && (
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-gray-600">Total Parameters:</span>
                          <span className="font-semibold" style={{ color }}>
                            {formatNumber(layer.parameters)}
                          </span>
                        </div>
                      )}
                    </div>
                  </motion.div>
                )}
              </Card>

              {/* Arrow to next layer */}
              {index < layers.length - 1 && (
                <div className="flex justify-center py-1">
                  <ArrowDown className="w-5 h-5 text-gray-400" />
                </div>
              )}
            </motion.div>
          );
        })}
      </div>

      {/* Legend */}
      <Card className="p-4 bg-gradient-to-r from-blue-50 to-purple-50 border-blue-200">
        <h4 className="text-sm font-medium mb-3">Layer Types</h4>
        <div className="grid grid-cols-3 md:grid-cols-6 gap-3">
          {[
            { type: 'input', name: 'Input', emoji: '📝' },
            { type: 'embedding', name: 'Embedding', emoji: '🔤' },
            { type: 'positional', name: 'Positional', emoji: '📍' },
            { type: 'attention', name: 'Attention', emoji: '🔍' },
            { type: 'ffn', name: 'Feed-Forward', emoji: '🧠' },
            { type: 'output', name: 'Output', emoji: '📤' },
          ].map((item) => (
            <div key={item.type} className="flex items-center gap-2">
              <span className="text-xl">{item.emoji}</span>
              <span className="text-xs text-gray-600">{item.name}</span>
            </div>
          ))}
        </div>
      </Card>

      {/* Summary Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="p-3 text-center">
          <div className="text-2xl font-bold" style={{ color: getLayerColor('input') }}>
            {config.max_seq_len || 128}
          </div>
          <div className="text-xs text-gray-500">Max Sequence</div>
        </Card>
        <Card className="p-3 text-center">
          <div className="text-2xl font-bold" style={{ color: getLayerColor('embedding') }}>
            {config.d_model || 512}
          </div>
          <div className="text-xs text-gray-500">Model Dimension</div>
        </Card>
        <Card className="p-3 text-center">
          <div className="text-2xl font-bold" style={{ color: getLayerColor('attention') }}>
            {config.nhead || 8}
          </div>
          <div className="text-xs text-gray-500">Attention Heads</div>
        </Card>
        <Card className="p-3 text-center">
          <div className="text-2xl font-bold" style={{ color: getLayerColor('ffn') }}>
            {config.dim_feedforward || 2048}
          </div>
          <div className="text-xs text-gray-500">FFN Dimension</div>
        </Card>
      </div>
    </div>
  );
};
