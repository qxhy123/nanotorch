import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import katex from 'katex';
import { Card } from '../../ui/card';
import { Badge } from '../../ui/badge';
import { Info } from 'lucide-react';
import { Latex } from '../../ui/Latex';

interface FlowNode {
  id: string;
  label: string;
  type: 'input' | 'query' | 'key' | 'value' | 'operation' | 'output';
  x: number;
  y: number;
  width: number;
  height: number;
  color: string;
  data?: any;
  description?: string;
  formula?: string;
}

interface FlowLink {
  source: string | FlowNode;
  target: string | FlowNode;
  value: number;
  label?: string;
  color?: string;
}

interface FlowDirectionGraphProps {
  nodes: FlowNode[];
  links: FlowLink[];
  width?: number;
  height?: number;
  title?: string;
  description?: string;
  interactive?: boolean;
}

const SEMANTIC_COLORS = {
  input: '#22c55e',
  query: '#3b82f6',
  key: '#ef4444',
  value: '#22c55e',
  operation: '#8b5cf6',
  output: '#f59e0b',
};

/**
 * FlowDirectionGraph Component
 *
 * A D3.js-based directed flow graph for visualizing data flow through layers.
 * Uses semantic colors and smooth animations.
 */
export const FlowDirectionGraph: React.FC<FlowDirectionGraphProps> = ({
  nodes,
  links,
  width = 1100,
  height = 500,
  title,
  description,
  interactive = true,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    content: string;
    details?: string;
  } | null>(null);

  useEffect(() => {
    if (!svgRef.current || nodes.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Create main group with margin
    const margin = { top: 40, right: 120, bottom: 40, left: 120 };

    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Create node map for link resolution
    const nodeMap = new Map(nodes.map(n => [n.id, n]));
    const resolvedLinks = links.map(link => ({
      ...link,
      source: nodeMap.get(link.source as string) || link.source,
      target: nodeMap.get(link.target as string) || link.target,
    }));

    // Create simple arrow marker
    const defs = svg.append('defs');
    const markerId = 'arrowhead';
    defs
      .append('marker')
      .attr('id', markerId)
      .attr('viewBox', '0 0 10 10')
      .attr('refX', 9)
      .attr('refY', 5)
      .attr('markerWidth', 4)
      .attr('markerHeight', 4)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M 0 0 L 10 5 L 0 10 z')
      .attr('fill', '#cbd5e1');

    // Draw links with clean simple style
    const linkGroup = g.append('g').attr('class', 'links').style('opacity', 0);

    resolvedLinks.forEach((link) => {
      const sourceNode = link.source as FlowNode;
      const targetNode = link.target as FlowNode;

      // Simple connection points
      const sourceX = sourceNode.x + sourceNode.width;
      const sourceY = sourceNode.y + sourceNode.height / 2;
      const targetX = targetNode.x;
      const targetY = targetNode.y + targetNode.height / 2;

      // Simple curved path
      const dx = targetX - sourceX;
      const curve = dx * 0.3;

      const pathData = `M ${sourceX} ${sourceY} C ${sourceX + curve} ${sourceY}, ${targetX - curve} ${targetY}, ${targetX} ${targetY}`;

      const linkPath = linkGroup
        .append('path')
        .attr('d', pathData)
        .attr('stroke', '#cbd5e1')
        .attr('stroke-width', 2)
        .attr('fill', 'none')
        .attr('marker-end', `url(#${markerId})`)
        .style('cursor', interactive ? 'pointer' : 'default');

      // Add label with LaTeX rendering
      if (link.label && link.label.trim()) {
        const labelX = (sourceX + targetX) / 2;
        const labelY = (sourceY + targetY) / 2;

        // Create foreignObject for LaTeX
        const fo = linkGroup
          .append('foreignObject')
          .attr('x', labelX - 20)
          .attr('y', labelY - 10)
          .attr('width', 40)
          .attr('height', 20)
          .attr('style', 'overflow: visible; pointer-events: none;');

        const div = fo
          .append('xhtml:div')
          .attr('style', 'display: flex; align-items: center; justify-content: center; width: 100%; height: 100%; font-size: 11px; color: #64748b;');

        // Render LaTeX
        try {
          const html = katex.renderToString(link.label, {
            throwOnError: false,
            displayMode: false,
            strict: false,
            trust: true,
          });
          div.html(html);
        } catch (e) {
          div.text(link.label);
        }
      }

      // Interactive events
      if (interactive) {
        linkPath
          .on('mouseenter', function () {
            d3.select(this)
              .attr('stroke', '#94a3b8')
              .attr('stroke-width', 2.5);

            const tooltipX = (sourceX + targetX) / 2;
            const tooltipY = (sourceY + targetY) / 2;

            setTooltip({
              x: tooltipX + 120,
              y: tooltipY + 40,
              content: `${sourceNode.label} → ${targetNode.label}`,
              details: `Flow: ${link.value}`,
            });
          })
          .on('mouseleave', function () {
            d3.select(this)
              .attr('stroke', '#cbd5e1')
              .attr('stroke-width', 2);
            setTooltip(null);
          });
      }
    });

    // Fade in links
    linkGroup.transition().duration(600).style('opacity', 1);

    // Draw nodes with improved styling
    const nodeGroup = g.append('g').attr('class', 'nodes');

    nodes.forEach((node) => {
      const nodeG = nodeGroup
        .append('g')
        .attr('transform', `translate(${node.x},${node.y})`)
        .style('cursor', interactive ? 'pointer' : 'default');

      // Node rectangle with gradient and shadow
      nodeG
        .append('rect')
        .attr('width', node.width)
        .attr('height', node.height)
        .attr('fill', node.color)
        .attr('stroke', '#fff')
        .attr('stroke-width', 2.5)
        .attr('rx', 10)
        .attr('filter', 'drop-shadow(0 3px 6px rgba(0,0,0,0.15))')
        .style('cursor', interactive ? 'pointer' : 'default');

      // Node label with better styling
      nodeG
        .append('text')
        .attr('x', node.width / 2)
        .attr('y', node.height / 2 - 3)
        .attr('text-anchor', 'middle')
        .attr('dy', '0.35em')
        .attr('font-weight', '600')
        .attr('font-size', '13px')
        .attr('fill', 'white')
        .style('text-shadow', '0 1px 2px rgba(0,0,0,0.2)')
        .style('pointer-events', 'none')
        .text(node.label);

      // Node type/subtitle
      if (node.type) {
        nodeG
          .append('text')
          .attr('x', node.width / 2)
          .attr('y', node.height / 2 + 13)
          .attr('text-anchor', 'middle')
          .attr('dy', '0.35em')
          .attr('font-size', '9px')
          .attr('fill', 'rgba(255,255,255,0.85)')
          .attr('font-weight', '400')
          .style('text-shadow', '0 1px 2px rgba(0,0,0,0.2)')
          .style('pointer-events', 'none')
          .text(node.type.charAt(0).toUpperCase() + node.type.slice(1));
      }

      // Interactive events
      if (interactive) {
        nodeG
          .on('mouseenter', function () {
            d3.select(this)
              .select('rect')
              .attr('filter', 'drop-shadow(0 4px 8px rgba(0,0,0,0.2))')
              .transition()
              .duration(200);

            // Calculate position based on node coordinates
            const tooltipX = node.x + node.width / 2;
            const tooltipY = node.y;

            let details = '';

            if (node.formula) {
              details = `Formula: ${node.formula}`;
            }
            if (node.data?.shape) {
              details += details ? '\n' : '';
              details += `Shape: [${node.data.shape.join(' × ')}]`;
            }

            setTooltip({
              x: tooltipX + 120, // Add left margin offset
              y: tooltipY + 40,  // Add top margin offset
              content: node.label,
              details: details || node.description,
            });
          })
          .on('mouseleave', function () {
            d3.select(this)
              .select('rect')
              .attr('filter', 'drop-shadow(0 2px 4px rgba(0,0,0,0.1))');
            setTooltip(null);
          });
      }
    });

    // Add title
    if (title) {
      svg
        .append('text')
        .attr('x', width / 2)
        .attr('y', 24)
        .attr('text-anchor', 'middle')
        .attr('font-size', '18px')
        .attr('font-weight', '600')
        .attr('fill', '#1e293b')
        .text(title);
    }
  }, [nodes, links, width, height, interactive, title]);

  return (
    <Card className="p-6">
      {title && (
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-semibold">{title}</h3>
            {description && <p className="text-sm text-gray-500 mt-1">{description}</p>}
          </div>
          <Badge variant="outline">{nodes.length} nodes</Badge>
        </div>
      )}

      {/* Information Panel */}
      <div className="mb-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-blue-500 mt-0.5 flex-shrink-0" />
          <div className="text-sm text-gray-700">
            <p className="font-medium mb-2">Data Flow Visualization</p>
            <ul className="space-y-1 text-xs">
              <li>• <strong>Left to right</strong> shows computation flow</li>
              <li>• <strong>Node colors</strong> indicate component type</li>
              <li>• <strong>Connection thickness</strong> represents data magnitude</li>
              {interactive && <li>• <strong>Hover</strong> over nodes and links for details</li>}
            </ul>
          </div>
        </div>
      </div>

      <div className="relative" style={{ width, height }}>
        <svg ref={svgRef} width={width} height={height} className="flow-graph" />

        {tooltip && (
          <div
            className="absolute z-10 px-4 py-3 bg-gray-900 text-white text-sm rounded-lg shadow-xl pointer-events-none whitespace-pre-line"
            style={{
              left: `${tooltip.x}px`,
              top: `${tooltip.y}px`,
              transform: 'translate(-50%, -100%)',
              maxWidth: '300px',
            }}
          >
            <div className="font-semibold mb-1">{tooltip.content}</div>
            {tooltip.details && (
              <div className="text-gray-300 text-xs mt-2">
                {tooltip.details.includes('Formula') ? (
                  <Latex display={false}>{tooltip.details.replace('Formula: ', '')}</Latex>
                ) : (
                  tooltip.details
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="mt-4 p-4 bg-gray-50 rounded-lg">
        <h4 className="text-sm font-medium mb-3">Component Types</h4>
        <div className="flex flex-wrap gap-4">
          {Object.entries(SEMANTIC_COLORS).map(([type, color]) => (
            <div key={type} className="flex items-center gap-2">
              <div className="w-4 h-4 rounded" style={{ backgroundColor: color }} />
              <span className="text-xs text-gray-600 capitalize">{type}</span>
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
};

/**
 * Hook to generate QKV flow data
 */
export const useQKVFlowData = (attentionData: any, tokens: string[] = []) => {
  if (!attentionData || !attentionData.queries) {
    return { nodes: [], links: [] };
  }

  const { queries, keys, values } = attentionData;
  const getShape = (tensor: any) => (tensor?.shape ? tensor.shape : ['seq_len', 'd_model']);

  const qShape = getShape(queries);
  const kShape = getShape(keys);
  const vShape = getShape(values);

  // Calculate layout - optimized for better spacing
  const baseY = 140;
  const ySpacing = 85;
  const xSpacing = 150;
  const startX = 40;

  const nodes: FlowNode[] = [
    {
      id: 'input',
      label: 'Input X',
      type: 'input',
      x: startX,
      y: baseY + ySpacing * 1.5,
      width: 100,
      height: 60,
      color: SEMANTIC_COLORS.input,
      description: `Tokens: ${tokens.slice(0, 3).join(', ')}${tokens.length > 3 ? '...' : ''}`,
      formula: 'X',
      data: { shape: [tokens.length] },
    },
    {
      id: 'query',
      label: 'Query Q',
      type: 'query',
      x: startX + xSpacing,
      y: baseY,
      width: 110,
      height: 60,
      color: SEMANTIC_COLORS.query,
      description: 'What to look for',
      formula: 'Q = X \\cdot W_Q',
      data: { shape: qShape },
    },
    {
      id: 'key',
      label: 'Key K',
      type: 'key',
      x: startX + xSpacing,
      y: baseY + ySpacing,
      width: 110,
      height: 60,
      color: SEMANTIC_COLORS.key,
      description: 'What is contained',
      formula: 'K = X \\cdot W_K',
      data: { shape: kShape },
    },
    {
      id: 'value',
      label: 'Value V',
      type: 'value',
      x: startX + xSpacing,
      y: baseY + ySpacing * 2,
      width: 110,
      height: 60,
      color: SEMANTIC_COLORS.value,
      description: 'Information to extract',
      formula: 'V = X \\cdot W_V',
      data: { shape: vShape },
    },
    {
      id: 'dot',
      label: 'Q · K^T',
      type: 'operation',
      x: startX + xSpacing * 2,
      y: baseY + ySpacing * 0.5,
      width: 100,
      height: 60,
      color: SEMANTIC_COLORS.operation,
      description: 'Similarity scores',
      formula: 'Q \\cdot K^T',
      data: { shape: [qShape[0], kShape[0]] },
    },
    {
      id: 'scale',
      label: 'Scale',
      type: 'operation',
      x: startX + xSpacing * 3,
      y: baseY + ySpacing * 0.5,
      width: 100,
      height: 60,
      color: SEMANTIC_COLORS.operation,
      description: 'Scale by 1/√dₖ',
      formula: '\\frac{QK^T}{\\sqrt{d_k}}',
      data: { shape: [qShape[0], kShape[0]] },
    },
    {
      id: 'softmax',
      label: 'Softmax',
      type: 'operation',
      x: startX + xSpacing * 4,
      y: baseY + ySpacing * 0.5,
      width: 110,
      height: 60,
      color: SEMANTIC_COLORS.operation,
      description: 'Attention weights',
      formula: '\\text{softmax}(\\cdot)',
      data: { shape: [qShape[0], kShape[0]] },
    },
    {
      id: 'output',
      label: 'Output',
      type: 'output',
      x: startX + xSpacing * 5,
      y: baseY + ySpacing * 1.5,
      width: 100,
      height: 60,
      color: SEMANTIC_COLORS.output,
      description: 'Final attention output',
      formula: '\\text{softmax} \\cdot V',
      data: { shape: [qShape[0], vShape[1]] },
    },
  ];

  const links: FlowLink[] = [
    { source: 'input', target: 'query', value: 50, label: '\\times W_Q' },
    { source: 'input', target: 'key', value: 50, label: '\\times W_K' },
    { source: 'input', target: 'value', value: 50, label: '\\times W_V' },
    { source: 'query', target: 'dot', value: 40, label: '+ K' },
    { source: 'key', target: 'dot', value: 40, label: '+ Q' },
    { source: 'dot', target: 'scale', value: 35, label: '\\div \\sqrt{d_k}' },
    { source: 'scale', target: 'softmax', value: 35, label: '\\sigma' },
    { source: 'softmax', target: 'output', value: 45, label: '\\times V' },
    { source: 'value', target: 'output', value: 45, label: '' },
  ];

  return { nodes, links };
};
