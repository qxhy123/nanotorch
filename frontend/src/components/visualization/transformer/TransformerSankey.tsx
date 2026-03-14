/* eslint-disable react-refresh/only-export-components, @typescript-eslint/no-explicit-any */
import React, { useEffect, useRef, useState } from 'react';
import { sankey, sankeyLinkHorizontal } from 'd3-sankey';
import * as d3 from 'd3';
import type { SankeyData } from '../../../types/transformer';
import { Card } from '../../ui/card';
import { Badge } from '../../ui/badge';
import { Info } from 'lucide-react';

interface TransformerSankeyProps {
  data: SankeyData;
  width?: number;
  height?: number;
  margin?: { top: number; right: number; bottom: number; left: number };
  interactive?: boolean;
}

/**
 * TransformerSankey Component
 *
 * Displays a Sankey diagram showing data flow through the Transformer.
 */
export const TransformerSankey: React.FC<TransformerSankeyProps> = ({
  data,
  width = 800,
  height = 500,
  margin = { top: 20, right: 20, bottom: 20, left: 20 },
  interactive = true,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; content: string } | null>(null);

  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  const getNodeColor = (node: { type: string }): string => {
    const colorMap: Record<string, string> = {
      input: '#22c55e',
      query: '#3b82f6',
      key: '#ef4444',
      value: '#22c55e',
      dot_product: '#8b5cf6',
      scaled: '#a855f7',
      attention_weights: '#ec4899',
      output: '#f59e0b',
      // Legacy types
      embedding: '#8b5cf6',
      positional: '#06b6d4',
      attention: '#ec4899',
      ffn: '#f59e0b',
      normalization: '#3b82f6',
      operation: '#6b7280',
      layer: '#8b5cf6',
    };
    return colorMap[node.type] || colorMap.layer;
  };

  useEffect(() => {
    if (!svgRef.current || !data.nodes.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Create sankey generator with optimized spacing
    const sankeyGenerator = sankey()
      .nodeWidth(40)           // Wider nodes for better visibility
      .nodePadding(50)          // More vertical space between nodes
      .extent([[0, 0], [innerWidth, innerHeight]]);

    // Process the data - sankey expects node objects to be referenced
    const nodeMap = new Map(data.nodes.map(n => [n.id, n]));
    const graph = {
      nodes: data.nodes,
      links: data.links.map(l => {
        const sourceNode = typeof l.source === 'string' ? nodeMap.get(l.source) : l.source;
        const targetNode = typeof l.target === 'string' ? nodeMap.get(l.target) : l.target;
        return {
          ...l,
          source: sourceNode || l.source,
          target: targetNode || l.target,
        };
      })
    };

    sankeyGenerator(graph);

    // CRITICAL FIX: d3-sankey modifies source/target in place
    // After processing, they may be indices instead of node objects
    // We need to ensure they reference the actual node objects with all properties
    graph.links.forEach((link: any) => {
      if (typeof link.source === 'number') {
        // d3-sankey converted to index - map back to node object
        link.source = graph.nodes[link.source];
      }
      if (typeof link.target === 'number') {
        // d3-sankey converted to index - map back to node object
        link.target = graph.nodes[link.target];
      }

      // Ensure the node objects have all required properties
      // Sometimes d3-sankey creates minimal node objects
      if (link.source && typeof link.source === 'object' && !link.source.name) {
        const sourceNode = graph.nodes.find((n: any) => n.id === link.source.id || n === link.source);
        if (sourceNode) link.source = sourceNode;
      }
      if (link.target && typeof link.target === 'object' && !link.target.name) {
        const targetNode = graph.nodes.find((n: any) => n.id === link.target.id || n === link.target);
        if (targetNode) link.target = targetNode;
      }
    });

    // Create gradient definitions
    const defs = svg.append('defs');
    graph.links.forEach((link: any, i: number) => {
      const sourceColor = getNodeColor(link.source);
      const targetColor = getNodeColor(link.target);
      const gradientId = `gradient-${i}`;

      const gradient = defs
        .append('linearGradient')
        .attr('id', gradientId)
        .attr('gradientUnits', 'userSpaceOnUse')
        .attr('x1', link.source.x1)
        .attr('x2', link.target.x0);

      gradient
        .append('stop')
        .attr('offset', '0%')
        .attr('stop-color', sourceColor);

      gradient
        .append('stop')
        .attr('offset', '100%')
        .attr('stop-color', targetColor);

      (link as any).gradientId = gradientId;
    });

    // Draw links
    g.append('g')
      .selectAll('path')
      .data(graph.links)
      .join('path')
      .attr('d', sankeyLinkHorizontal())
      .attr('stroke', (d: any) => `url(#${d.gradientId})`)
      .attr('stroke-width', (d: any) => Math.max(2, d.width || 2))
      .attr('fill', 'none')
      .attr('opacity', 0.5)
      .style('cursor', interactive ? 'pointer' : 'default')
      .on('mouseenter', function(event: any, d: any) {
        void event;
        if (interactive) {
          // Calculate tooltip position based on link coordinates
          const linkX = (d.source.x1 + d.target.x0) / 2;
          const linkY = (d.source.y0 + d.source.y1 + d.target.y0 + d.target.y1) / 4;

          // Robust name extraction with multiple fallbacks
          const getName = (n: any): string => {
            if (!n) return 'Unknown';
            if (typeof n === 'string') return n;
            if (n.name) return n.name;
            if (n.id) return n.id;
            return 'Unknown';
          };

          const sourceName = getName(d.source);
          const targetName = getName(d.target);
          const value = d.value !== undefined ? d.value.toLocaleString() : 'N/A';

          setTooltip({
            x: linkX + margin.left,
            y: linkY + margin.top,
            content: `${sourceName} → ${targetName}: ${value}`,
          });
          d3.select(this).attr('opacity', 0.8);
        }
      })
      .on('mouseleave', function(event: any) {
        void event;
        if (interactive) {
          setTooltip(null);
          d3.select(this).attr('opacity', 0.5);
        }
      });

    // Draw nodes
    g.append('g')
      .selectAll('rect')
      .data(graph.nodes)
      .join('rect')
      .attr('x', (d: any) => d.x0)
      .attr('y', (d: any) => d.y0)
      .attr('height', (d: any) => Math.max(1, d.y1 - d.y0))
      .attr('width', (d: any) => Math.max(1, d.x1 - d.x0))
      .attr('fill', (d: any) => getNodeColor(d))
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .attr('rx', 3)
      .style('cursor', interactive ? 'pointer' : 'default')
      .on('mouseenter', function(event: any, d: any) {
        void event;
        if (interactive) {
          // Calculate tooltip position based on node coordinates
          const nodeX = (d.x0 + d.x1) / 2;
          const nodeY = d.y0;

          const name = d.name || 'Unknown';
          let content = name;

          // Build detailed tooltip
          if (d.metadata) {
            const lines = [name];

            // Add description if available
            if (d.metadata.description) {
              lines.push(d.metadata.description);
            }

            // Add formula if available
            if (d.metadata.formula) {
              lines.push(`Formula: ${d.metadata.formula}`);
            }

            // Add shape information
            if (d.metadata.shape) {
              lines.push(`Shape: [${d.metadata.shape.join(', ')}]`);
            }

            // Add parameters
            if (d.metadata.parameters) {
              const params = d.metadata.parameters;
              const formattedParams = params >= 1000000
                ? `${(params / 1000000).toFixed(1)}M`
                : params >= 1000
                ? `${(params / 1000).toFixed(1)}K`
                : params.toLocaleString();
              lines.push(`Parameters: ${formattedParams}`);
            }

            // Add layer number if applicable
            if (d.metadata.layerNum) {
              lines.push(`Layer: ${d.metadata.layerNum}`);
            }

            content = lines.join('\n');
          }

          setTooltip({
            x: nodeX + margin.left,
            y: nodeY + margin.top,
            content,
          });
          d3.select(this).attr('opacity', 0.8);
        }
      })
      .on('mouseleave', function(event: any) {
        void event;
        if (interactive) {
          setTooltip(null);
          d3.select(this).attr('opacity', 1);
        }
      });

    // Add node labels
    g.append('g')
      .selectAll('text')
      .data(graph.nodes)
      .join('text')
      .attr('x', (d: any) => d.x0 < innerWidth / 2 ? d.x1 + 6 : d.x0 - 6)
      .attr('y', (d: any) => (d.y1 + d.y0) / 2)
      .attr('dy', '0.35em')
      .attr('text-anchor', (d: any) => d.x0 < innerWidth / 2 ? 'start' : 'end')
      .text((d: any) => d.name)
      .attr('fill', '#374151')
      .attr('font-size', '12px')
      .attr('font-weight', '500')
      .style('pointer-events', 'none')
      .style('text-shadow', '0 1px 2px rgba(255,255,255,0.8)');

  }, [data, innerWidth, innerHeight, margin, interactive]);

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold">QKV Attention Flow</h3>
          <p className="text-sm text-gray-500 mt-1">
            Visualizes the Query-Key-Value attention mechanism
          </p>
        </div>
        <Badge variant="outline">{data.nodes.length} nodes, {data.links.length} connections</Badge>
      </div>

      {/* Information Panel */}
      <div className="mb-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-blue-500 mt-0.5 flex-shrink-0" />
          <div className="text-sm text-gray-700">
            <p className="font-medium mb-2">What does this diagram show?</p>
            <ul className="space-y-1 text-xs">
              <li>• <strong>Node width</strong> = Amount of data/parameters</li>
              <li>• <strong>Connection width</strong> = Data flow magnitude</li>
              <li>• <strong>Hover over nodes</strong> to see formulas and shapes</li>
              <li>• <strong>Colors</strong>: Query (blue), Key (red), Value (green), Attention (pink)</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="relative" style={{ width, height }}>
        <svg
          ref={svgRef}
          width={width}
          height={height}
          className="sankey-diagram"
        />

        {tooltip && (
          <div
            className="absolute z-10 px-3 py-2 bg-gray-900 text-white text-sm rounded-lg shadow-lg pointer-events-none whitespace-pre-line"
            style={{
              left: tooltip.x,
              top: tooltip.y - 10,
              transform: 'translate(-50%, -100%)',
            }}
          >
            {tooltip.content}
          </div>
        )}
      </div>

      <div className="mt-4 p-4 bg-gray-50 rounded-lg">
        <h4 className="text-sm font-medium mb-3">Layer Types</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {Array.from(new Set(data.nodes.map((n) => n.type))).map((type) => {
            const typeInfo: Record<string, { description: string; example: string }> = {
              input: { description: 'Input embeddings', example: 'X ∈ [seq_len, d_model]' },
              query: { description: 'Query vectors', example: 'Q = X · W_Q (what to look for)' },
              key: { description: 'Key vectors', example: 'K = X · W_K (what is contained)' },
              value: { description: 'Value vectors', example: 'V = X · W_V (information to extract)' },
              dot_product: { description: 'Attention scores', example: 'Q · K^T (similarity)' },
              scaled: { description: 'Scaled scores', example: '(Q · K^T) / √dₖ' },
              attention_weights: { description: 'Attention probabilities', example: 'softmax(...) → [0, 1]' },
              output: { description: 'Contextual output', example: 'Attention · V' },
              // Legacy types
              embedding: { description: 'Token → Vector', example: 'Output: [128, 512]' },
              positional: { description: 'Position information', example: 'Sin/Cos encoding' },
              attention: { description: 'Self-attention mechanism', example: 'Query·Key^T·Value' },
              ffn: { description: 'Feed-forward network', example: 'Linear → ReLU → Linear' },
              normalization: { description: 'Layer normalization', example: 'Normalize across features' },
            };

            const info = typeInfo[type as string] || { description: type, example: '' };

            return (
              <div key={type} className="flex items-start gap-2">
                <div
                  className="w-3 h-3 rounded mt-0.5 flex-shrink-0"
                  style={{ backgroundColor: getNodeColor({ type: type as string }) }}
                />
                <div className="flex-1 min-w-0">
                  <div className="text-xs font-medium text-gray-700 capitalize">{type}</div>
                  <div className="text-xs text-gray-500 truncate">{info.description}</div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </Card>
  );
};

/**
 * Hook to generate Sankey data for QKV flow
 */
export const useTransformerSankeyData = (
  _attentionData: any,
  config: any
): SankeyData | null => {
  if (!config) return null;

  const nodes: any[] = [];
  const links: any[] = [];

  // Calculate values based on actual config for realistic representation
  const dModel = config.d_model || 512;
  const seqLen = config.max_seq_len || 128;

  // Base value scaling
  const baseValue = Math.sqrt(dModel * seqLen);

  // Node values for QKV components
  const getNodeValue = (type: string) => {
    const multipliers: Record<string, number> = {
      input: 1.0,
      query: 2.2,
      key: 2.2,
      value: 2.2,
      dot_product: 1.8,
      scaled: 1.8,
      attention_weights: 2.5,
      output: 1.5,
    };
    return Math.round(baseValue * (multipliers[type] || 2));
  };

  // Calculate link value as percentage of source flow
  const getLinkValue = (sourceType: string) => {
    const sourceNode = nodes.find(n => n.type === sourceType);
    if (!sourceNode) return 50;

    // Flow ratios for different connection types
    const flowRatios: Record<string, number> = {
      'input-query': 0.95,
      'input-key': 0.95,
      'input-value': 0.95,
      'query-dot_product': 0.85,
      'key-dot_product': 0.85,
      'dot_product-scaled': 0.90,
      'scaled-attention_weights': 0.88,
      'value-output': 0.82,
      'attention_weights-output': 0.80,
    };

    return Math.round(sourceNode.value * (flowRatios[sourceType] || 0.85));
  };

  // === Build QKV Flow ===

  // 1. Input
  nodes.push({
    id: 'input',
    name: 'Input X',
    type: 'input',
    depth: 0,
    value: getNodeValue('input'),
    layerIndex: 0,
    metadata: {
      description: 'Token embeddings',
      shape: [seqLen, dModel],
      parameters: 0,
    },
  });

  // 2. Query, Key, Value (parallel projection)
  nodes.push({
    id: 'query',
    name: 'Query Q',
    type: 'query',
    depth: 1,
    value: getNodeValue('query'),
    layerIndex: 0,
    metadata: {
      description: 'What to look for',
      shape: [seqLen, dModel],
      parameters: dModel * dModel,
      formula: 'Q = X · W_Q',
    },
  });

  nodes.push({
    id: 'key',
    name: 'Key K',
    type: 'key',
    depth: 1,
    value: getNodeValue('key'),
    layerIndex: 0,
    metadata: {
      description: 'What is contained',
      shape: [seqLen, dModel],
      parameters: dModel * dModel,
      formula: 'K = X · W_K',
    },
  });

  nodes.push({
    id: 'value',
    name: 'Value V',
    type: 'value',
    depth: 1,
    value: getNodeValue('value'),
    layerIndex: 0,
    metadata: {
      description: 'Information to extract',
      shape: [seqLen, dModel],
      parameters: dModel * dModel,
      formula: 'V = X · W_V',
    },
  });

  // Links from input to QKV
  links.push({ source: 'input', target: 'query', value: getLinkValue('input-query') });
  links.push({ source: 'input', target: 'key', value: getLinkValue('input-key') });
  links.push({ source: 'input', target: 'value', value: getLinkValue('input-value') });

  // 3. Dot Product (Q · K^T)
  nodes.push({
    id: 'dot_product',
    name: 'Q · K^T',
    type: 'dot_product',
    depth: 2,
    value: getNodeValue('dot_product'),
    layerIndex: 0,
    metadata: {
      description: 'Similarity scores',
      shape: [seqLen, seqLen],
      parameters: 0,
      formula: 'Q · K^T',
    },
  });
  links.push({ source: 'query', target: 'dot_product', value: getLinkValue('query-dot_product') });
  links.push({ source: 'key', target: 'dot_product', value: getLinkValue('key-dot_product') });

  // 4. Scaled
  nodes.push({
    id: 'scaled',
    name: 'Scaled',
    type: 'scaled',
    depth: 3,
    value: getNodeValue('scaled'),
    layerIndex: 0,
    metadata: {
      description: 'Divide by √dₖ',
      shape: [seqLen, seqLen],
      parameters: 0,
      formula: '(Q · K^T) / √dₖ',
    },
  });
  links.push({ source: 'dot_product', target: 'scaled', value: getLinkValue('dot_product-scaled') });

  // 5. Attention Weights (Softmax)
  nodes.push({
    id: 'attention_weights',
    name: 'Attention Weights',
    type: 'attention_weights',
    depth: 4,
    value: getNodeValue('attention_weights'),
    layerIndex: 0,
    metadata: {
      description: 'Softmax probabilities',
      shape: [seqLen, seqLen],
      parameters: 0,
      formula: 'softmax((Q · K^T) / √dₖ)',
    },
  });
  links.push({ source: 'scaled', target: 'attention_weights', value: getLinkValue('scaled-attention_weights') });

  // 6. Output (Weighted Sum)
  nodes.push({
    id: 'output',
    name: 'Output',
    type: 'output',
    depth: 5,
    value: getNodeValue('output'),
    layerIndex: 0,
    metadata: {
      description: 'Weighted sum of values',
      shape: [seqLen, dModel],
      parameters: 0,
      formula: 'Attention · V',
    },
  });
  links.push({ source: 'attention_weights', target: 'output', value: getLinkValue('attention_weights-output') });
  links.push({ source: 'value', target: 'output', value: getLinkValue('value-output') });

  return {
    nodes,
    links,
    layers: nodes.length,
  };
};
