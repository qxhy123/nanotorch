import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { ArrowRight, Check, Circle } from 'lucide-react';
import { Card } from '../../ui/card';
import { Badge } from '../../ui/badge';
import { Button } from '../../ui/button';
import { Latex } from '../../ui/Latex';
import type { AttentionData } from '../../../types/transformer';

interface QKVFlowDiagramProps {
  attentionData: AttentionData;
  tokens?: string[];
}

interface FlowNode {
  id: string;
  label: string;
  description: string;
  formula: string;
  color: string;
  shape: number[];
  icon: string;
}

interface FlowConnection {
  from: string;
  to: string;
  label: string;
  operation: string;
}

/**
 * QKVFlowDiagram Component
 *
 * Visualizes the flow of Q, K, V matrices through the attention computation
 * using an intuitive left-to-right flow diagram.
 */
export const QKVFlowDiagram: React.FC<QKVFlowDiagramProps> = ({ attentionData, tokens = [] }) => {
  const [currentStage, setCurrentStage] = useState<string>('all');
  const [activeNode, setActiveNode] = useState<string | null>(null);

  const { queries, keys, values, scale } = attentionData;

  // Get matrix dimensions
  const getShape = (tensor?: { shape?: number[] } | null): number[] => {
    if (!Array.isArray(tensor?.shape)) {
      return [];
    }
    return tensor.shape;
  };

  const qShape = getShape(queries);
  const kShape = getShape(keys);
  const vShape = getShape(values);

  // Define flow nodes
  const flowNodes: FlowNode[] = [
    {
      id: 'input',
      label: 'Input X',
      description: `Sequence of tokens [${tokens.join(', ') || 'tokens'}]`,
      formula: 'X',
      color: '#22c55e',
      shape: [qShape[0] ?? 128],
      icon: '📝',
    },
    {
      id: 'q',
      label: 'Query Q',
      description: `What each token is looking for\nShape: [${qShape.join(' × ') || 'seq_len × d_model'}]`,
      formula: 'Q = X · W_Q',
      color: '#3b82f6',
      shape: qShape,
      icon: '🔍',
    },
    {
      id: 'k',
      label: 'Key K',
      description: `What each token contains\nShape: [${kShape.join(' × ') || 'seq_len × d_model'}]`,
      formula: 'K = X · W_K',
      color: '#ef4444',
      shape: kShape,
      icon: '🔑',
    },
    {
      id: 'v',
      label: 'Value V',
      description: `Information to be extracted\nShape: [${vShape.join(' × ') || 'seq_len × d_model'}]`,
      formula: 'V = X · W_V',
      color: '#22c55e',
      shape: vShape,
      icon: '💎',
    },
    {
      id: 'dot',
      label: 'Dot Product',
      description: 'Measure similarity between queries and keys',
      formula: 'Q · K^T',
      color: '#8b5cf6',
      shape: [qShape[0] ?? 128, kShape[0] ?? 128],
      icon: '✖️',
    },
    {
      id: 'scale',
      label: 'Scale',
      description: `Divide by √dₖ to prevent large values\nScale factor: 1/√dₖ = ${(1 / scale).toFixed(4)}`,
      formula: '\\frac{QK^T}{\\sqrt{d_k}}',
      color: '#a855f7',
      shape: [qShape[0] ?? 128, kShape[0] ?? 128],
      icon: '⚖️',
    },
    {
      id: 'softmax',
      label: 'Softmax',
      description: 'Convert to probabilities (sum to 1)',
      formula: '\\text{softmax}(\\frac{QK^T}{\\sqrt{d_k}})',
      color: '#ec4899',
      shape: [qShape[0] ?? 128, kShape[0] ?? 128],
      icon: '📊',
    },
    {
      id: 'weighted',
      label: 'Weighted Sum',
      description: 'Final output = attention weights × values',
      formula: '\\text{softmax}(...) · V',
      color: '#f59e0b',
      shape: [qShape[0] ?? 128, vShape[1] ?? 512],
      icon: '⚖️',
    },
  ];

  // Define connections
  const connections: FlowConnection[] = [
    { from: 'input', to: 'q', label: '× W_Q', operation: 'Matrix multiplication' },
    { from: 'input', to: 'k', label: '× W_K', operation: 'Matrix multiplication' },
    { from: 'input', to: 'v', label: '× W_V', operation: 'Matrix multiplication' },
    { from: 'q', to: 'dot', label: 'with K', operation: 'Dot product' },
    { from: 'k', to: 'dot', label: 'with Q', operation: 'Dot product' },
    { from: 'dot', to: 'scale', label: '÷ √dₖ', operation: 'Division' },
    { from: 'scale', to: 'softmax', label: 'softmax', operation: 'Normalization' },
    { from: 'softmax', to: 'weighted', label: '× V', operation: 'Matrix multiplication' },
    { from: 'v', to: 'weighted', label: 'values', operation: 'Data' },
  ];

  const stages = [
    { id: 'all', label: 'All Stages', color: '#6366f1' },
    { id: 'qkv', label: 'Q, K, V Generation', color: '#3b82f6' },
    { id: 'attention', label: 'Attention Computation', color: '#ec4899' },
    { id: 'output', label: 'Output', color: '#f59e0b' },
  ];

  const getNodeOpacity = (nodeId: string) => {
    if (currentStage === 'all') return 1;
    if (currentStage === 'qkv' && ['input', 'q', 'k', 'v'].includes(nodeId)) return 1;
    if (currentStage === 'attention' && ['dot', 'scale', 'softmax'].includes(nodeId)) return 1;
    if (currentStage === 'output' && ['weighted', 'v'].includes(nodeId)) return 1;
    return 0.2;
  };

  const getConnectionOpacity = (conn: FlowConnection) => {
    if (currentStage === 'all') return 1;
    const fromOpacity = getNodeOpacity(conn.from);
    const toOpacity = getNodeOpacity(conn.to);
    return Math.min(fromOpacity, toOpacity);
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold">QKV Attention Flow</h3>
          <p className="text-sm text-gray-500">
            Interactive visualization of Query, Key, Value flow
          </p>
        </div>
      </div>

      {/* Stage Selector */}
      <div className="flex gap-2 mb-6">
        {stages.map((stage) => (
          <Button
            key={stage.id}
            variant={currentStage === stage.id ? 'default' : 'outline'}
            size="sm"
            onClick={() => setCurrentStage(stage.id)}
            style={{
              backgroundColor: currentStage === stage.id ? stage.color : undefined,
              borderColor: currentStage !== stage.id ? stage.color : undefined,
            }}
          >
            {stage.label}
          </Button>
        ))}
      </div>

      {/* Flow Diagram */}
      <Card className="p-6">
        <div className="relative">
          {/* Horizontal Flow */}
          <div className="flex items-start justify-between gap-2 overflow-x-auto pb-8">
            {flowNodes.map((node, index) => {
              const opacity = getNodeOpacity(node.id);
              const isActive = activeNode === node.id;

              return (
                <div key={node.id} className="flex flex-col items-center">
                  {/* Node Card */}
                  <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity, scale: 1 }}
                    transition={{ delay: index * 0.1 }}
                    className="relative"
                  >
                    <Card
                      className={`p-4 cursor-pointer transition-all ${
                        isActive ? 'ring-2 ring-offset-2' : ''
                      }`}
                      style={{
                        borderColor: isActive ? node.color : undefined,
                        borderWidth: isActive ? '2px' : '1px',
                        minWidth: '140px',
                        opacity: currentStage === 'all' ? 1 : getNodeOpacity(node.id),
                      }}
                      onClick={() => setActiveNode(activeNode === node.id ? null : node.id)}
                    >
                      {/* Icon */}
                      <div className="text-3xl mb-2 text-center">{node.icon}</div>

                      {/* Label */}
                      <h4 className="text-sm font-semibold text-center mb-2" style={{ color: node.color }}>
                        {node.label}
                      </h4>

                      {/* Description */}
                      <p className="text-xs text-gray-600 text-center whitespace-pre-line mb-2">
                        {node.description}
                      </p>

                      {/* Shape Badge */}
                      <Badge variant="outline" className="text-xs w-full justify-center">
                        {node.shape.length > 0 ? `[${node.shape.join(', ')}]` : 'N/A'}
                      </Badge>

                      {/* Formula */}
                      {node.formula && (
                        <div className="mt-2 pt-2 border-t">
                          <Latex display={false} className="text-xs">
                            {node.formula}
                          </Latex>
                        </div>
                      )}
                    </Card>

                    {/* Highlight indicator */}
                    {isActive && (
                      <div className="absolute -bottom-1 left-1/2 transform -translate-x-1/2">
                        <div
                          className="w-2 h-2 rounded-full"
                          style={{ backgroundColor: node.color }}
                        />
                      </div>
                    )}
                  </motion.div>

                  {/* Connection Arrow */}
                  {index < flowNodes.length - 1 && (
                    <div className="flex items-center self-stretch py-8">
                      {/* Find connections originating from this node */}
                      {connections
                        .filter(conn => conn.from === node.id)
                        .map((conn) => {
                          const targetNode = flowNodes.find(n => n.id === conn.to);
                          if (!targetNode) return null;

                          const connOpacity = getConnectionOpacity(conn);
                          const isHighlighted = activeNode === node.id || activeNode === targetNode.id;

                          return (
                            <div key={`${conn.from}-${conn.to}`} className="flex flex-col items-center gap-1">
                              {/* Connection line */}
                              <div
                                className="h-0.5 bg-gray-300"
                                style={{
                                  width: '60px',
                                  opacity: connOpacity,
                                  backgroundColor: isHighlighted ? node.color : undefined,
                                }}
                              />

                              {/* Operation label */}
                              {connOpacity > 0.3 && (
                                <div
                                  className="text-xs text-gray-500 whitespace-nowrap bg-white px-2 py-1 rounded border"
                                  style={{ opacity: connOpacity }}
                                >
                                  <Latex display={false}>{conn.label}</Latex>
                                </div>
                              )}
                            </div>
                          );
                        })}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Legend */}
          <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-white via-white to-transparent">
            <div className="flex items-center justify-center gap-6 text-sm">
              <div className="flex items-center gap-2">
                <Circle className="w-3 h-3" fill={getNodeOpacity('input') > 0.5 ? '#22c55e' : '#e5e7eb'} />
                <span className="text-gray-600">Input</span>
              </div>
              <div className="flex items-center gap-2">
                <Circle className="w-3 h-3" fill={getNodeOpacity('q') > 0.5 ? '#3b82f6' : '#e5e7eb'} />
                <span className="text-gray-600">Q</span>
              </div>
              <div className="flex items-center gap-2">
                <Circle className="w-3 h-3" fill={getNodeOpacity('k') > 0.5 ? '#ef4444' : '#e5e7eb'} />
                <span className="text-gray-600">K</span>
              </div>
              <div className="flex items-center gap-2">
                <Circle className="w-3 h-3" fill={getNodeOpacity('v') > 0.5 ? '#22c55e' : '#e5e7eb'} />
                <span className="text-gray-600">V</span>
              </div>
              <ArrowRight className="w-4 h-4 text-gray-400" />
              <div className="flex items-center gap-2">
                <Circle className="w-3 h-3" fill={getNodeOpacity('softmax') > 0.5 ? '#ec4899' : '#e5e7eb'} />
                <span className="text-gray-600">Attention</span>
              </div>
              <ArrowRight className="w-4 h-4 text-gray-400" />
              <div className="flex items-center gap-2">
                <Circle className="w-3 h-3" fill={getNodeOpacity('weighted') > 0.5 ? '#f59e0b' : '#e5e7eb'} />
                <span className="text-gray-600">Output</span>
              </div>
            </div>
          </div>
        </div>
      </Card>

      {/* Detailed Info Panel */}
      {activeNode && (
        <Card className="p-4 bg-blue-50 border-blue-200">
          <div className="flex items-start justify-between mb-3">
            <div>
              <h4 className="font-semibold flex items-center gap-2">
                {flowNodes.find(n => n.id === activeNode)?.icon}
                {flowNodes.find(n => n.id === activeNode)?.label}
              </h4>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setActiveNode(null)}
            >
              ×
            </Button>
          </div>
          <div className="space-y-2 text-sm">
            <div>
              <span className="font-medium text-gray-700">Description: </span>
              <span className="text-gray-600">
                {flowNodes.find(n => n.id === activeNode)?.description}
              </span>
            </div>
            <div>
              <span className="font-medium text-gray-700">Formula: </span>
              <Latex display={false} className="ml-2">
                {flowNodes.find(n => n.id === activeNode)?.formula || ''}
              </Latex>
            </div>
          </div>
        </Card>
      )}

      {/* Key Insights */}
      <Card className="p-4 bg-gradient-to-r from-purple-50 to-blue-50 border border-purple-200">
        <h4 className="font-semibold mb-3 flex items-center gap-2">
          <Check className="w-5 h-5 text-purple-600" />
          Key Insights
        </h4>
        <ul className="space-y-2 text-sm text-gray-700">
          <li className="flex items-start gap-2">
            <div className="w-1.5 h-1.5 rounded-full bg-purple-500 mt-1.5 flex-shrink-0" />
            <span>
              <strong>Parallel Generation:</strong> Q, K, V are computed independently from the same input X
            </span>
          </li>
          <li className="flex items-start gap-2">
            <div className="w-1.5 h-1.5 rounded-full bg-purple-500 mt-1.5 flex-shrink-0" />
            <span>
              <strong>Key-Query Similarity:</strong> The dot product Q·K^T measures how relevant each key is to each query
            </span>
          </li>
          <li className="flex items-start gap-2">
            <div className="w-1.5 h-1.5 rounded-full bg-purple-500 mt-1.5 flex-shrink-0" />
            <span>
              <strong>Scaled Attention:</strong> Scaling by 1/√dₖ prevents gradients from vanishing in deep networks
            </span>
          </li>
          <li className="flex items-start gap-2">
            <div className="w-1.5 h-1.5 rounded-full bg-purple-500 mt-1.5 flex-shrink-0" />
            <span>
              <strong>Value Aggregation:</strong> Final output is a weighted sum where weights come from attention
            </span>
          </li>
        </ul>
      </Card>
    </div>
  );
};
