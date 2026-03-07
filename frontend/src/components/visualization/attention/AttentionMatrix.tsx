import React, { useMemo, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../ui/card';
import { Badge } from '../../ui/badge';
import { Button } from '../../ui/button';
import { useTransformerStore } from '../../../stores/transformerStore';

interface AttentionMatrixProps {
  className?: string;
}

export const AttentionMatrix: React.FC<AttentionMatrixProps> = ({ className }) => {
  const {
    attentionWeights,
    config,
    visualizationState,
    setSelectedHead,
    setColorScheme,
  } = useTransformerStore();

  const { selectedHead, colorScheme } = visualizationState;
  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number } | null>(null);

  // Debug: log attentionWeights when it changes
  React.useEffect(() => {
    if (attentionWeights) {
      console.log('Attention weights loaded:', {
        count: attentionWeights.length,
        firstItem: attentionWeights[0],
        hasWeights: !!(attentionWeights[0] as any)?.weights
      });
    }
  }, [attentionWeights]);

  const currentAttentionData = useMemo(() => {
    if (!attentionWeights || attentionWeights.length === 0) {
      console.log('No attention weights available');
      return null;
    }

    const data = attentionWeights[0];
    console.log('Processing attention data:', data);

    // Handle the structure returned by backend
    if (!data || !data.weights) {
      console.log('Invalid data structure:', data);
      return null;
    }

    const weights = data.weights;
    const weightsShape = weights.shape || [];

    console.log('Weights shape:', weightsShape);
    console.log('Selected head:', selectedHead, 'Total heads:', config.nhead);

    // Ensure selectedHead is within valid range
    const safeSelectedHead = Math.min(selectedHead, config.nhead - 1);
    if (safeSelectedHead !== selectedHead) {
      console.log('Adjusting selectedHead from', selectedHead, 'to', safeSelectedHead);
      setSelectedHead(safeSelectedHead);
    }

    console.log('Weights data type:', typeof weights.data);

    // Try to extract the actual array data
    let weightsArray: any;
    if (Array.isArray(weights.data)) {
      weightsArray = weights.data;
    } else if (typeof weights.data === 'object' && weights.data !== null) {
      // It might be a nested structure
      weightsArray = weights.data;
    }

    if (!weightsArray) {
      console.log('Could not extract weights array');
      return null;
    }

    // Handle different dimensionalities
    if (weightsShape.length === 4) {
      // Shape: (batch, heads, seq_len, seq_len)
      // weightsArray should be nested arrays
      const batchData = weightsArray[0]; // First batch
      console.log('Batch data:', batchData, 'length:', batchData?.length);
      console.log('Accessing head index:', safeSelectedHead);

      if (Array.isArray(batchData) && batchData.length > safeSelectedHead && Array.isArray(batchData[safeSelectedHead])) {
        const headData = batchData[safeSelectedHead];
        console.log('Head data shape:', [headData.length, headData[0]?.length]);
        return {
          weights: headData,
          scale: data.scale || (1 / Math.sqrt(config.d_model / config.nhead)),
        };
      } else {
        console.log('Failed to access head data. Batch length:', batchData?.length, 'Selected head:', safeSelectedHead);
      }
    } else if (weightsShape.length === 3) {
      // Shape: (batch, seq_len, seq_len) - averaged heads
      const batchData = weightsArray[0];
      if (Array.isArray(batchData)) {
        return {
          weights: batchData,
          scale: data.scale || (1 / Math.sqrt(config.d_model / config.nhead)),
        };
      }
    }

    // Fallback: try to use the data as-is
    if (Array.isArray(weightsArray) && weightsArray.length > 0) {
      return {
        weights: weightsArray,
        scale: data.scale || (1 / Math.sqrt(config.d_model / config.nhead)),
      };
    }

    console.log('Could not process weights data');
    return null;
  }, [attentionWeights, selectedHead, config, setSelectedHead]);

  const getColor = (value: number): string => {
    const normalized = Math.max(0, Math.min(1, value));

    switch (colorScheme) {
      case 'blues':
        return `rgba(59, 130, 246, ${0.2 + normalized * 0.8})`;
      case 'reds':
        return `rgba(239, 68, 68, ${0.2 + normalized * 0.8})`;
      case 'viridis':
        if (normalized < 0.25) {
          return `rgba(68, 1, 84, ${0.2 + normalized * 4 * 0.8})`;
        } else if (normalized < 0.5) {
          return `rgba(59, 82, 139, ${0.4 + (normalized - 0.25) * 4 * 0.4})`;
        } else if (normalized < 0.75) {
          return `rgba(33, 154, 131, ${0.4 + (normalized - 0.5) * 4 * 0.4})`;
        } else {
          return `rgba(94, 201, 98, ${0.4 + (normalized - 0.75) * 4 * 0.4})`;
        }
      case 'plasma':
        if (normalized < 0.33) {
          return `rgba(13, 8, 135, ${0.2 + normalized * 3 * 0.8})`;
        } else if (normalized < 0.66) {
          return `rgba(204, 70, 120, ${0.2 + (normalized - 0.33) * 3 * 0.8})`;
        } else {
          return `rgba(240, 249, 33, ${0.2 + (normalized - 0.66) * 3 * 0.8})`;
        }
      case 'inferno':
        if (normalized < 0.33) {
          return `rgba(0, 0, 4, ${0.2 + normalized * 3 * 0.8})`;
        } else if (normalized < 0.66) {
          return `rgba(159, 36, 40, ${0.2 + (normalized - 0.33) * 3 * 0.8})`;
        } else {
          return `rgba(252, 253, 97, ${0.2 + (normalized - 0.66) * 3 * 0.8})`;
        }
      default:
        return `rgba(59, 130, 246, ${0.2 + normalized * 0.8})`;
    }
  };

  const stats = useMemo(() => {
    if (!currentAttentionData?.weights) return null;

    const allValues = currentAttentionData.weights.flat();
    const min = Math.min(...allValues);
    const max = Math.max(...allValues);
    const mean = allValues.reduce((a, b) => a + b, 0) / allValues.length;

    return { min, max, mean };
  }, [currentAttentionData]);

  if (!currentAttentionData) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle>Attention Weights</CardTitle>
          <CardDescription>
            {attentionWeights && attentionWeights.length > 0
              ? `Data loaded but could not be processed. Check console for details.`
              : 'Run a forward pass to see attention patterns'}
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }

  const seqLen = currentAttentionData.weights.length;

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div>
            <CardTitle>Attention Weights</CardTitle>
            <CardDescription>
              Head {selectedHead + 1} / {config.nhead} | Scale: {currentAttentionData.scale.toFixed(4)}
            </CardDescription>
          </div>
          <div className="flex gap-2">
            {Array.from({ length: Math.min(config.nhead, 8) }, (_, i) => (
              <Button
                key={i}
                variant={selectedHead === i ? 'default' : 'outline'}
                size="sm"
                onClick={() => setSelectedHead(i)}
                className="transition-all duration-200"
              >
                H{i + 1}
              </Button>
            ))}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Statistics */}
        {stats && (
          <div className="flex gap-4 text-sm">
            <Badge variant="secondary">Min: {stats.min.toFixed(4)}</Badge>
            <Badge variant="secondary">Max: {stats.max.toFixed(4)}</Badge>
            <Badge variant="secondary">Mean: {stats.mean.toFixed(4)}</Badge>
          </div>
        )}

        {/* Attention Matrix */}
        <div className="overflow-x-auto">
          <div className="inline-block min-w-full">
            {/* Column labels (top) */}
            <div className="flex ml-8 mb-1">
              {Array.from({ length: Math.min(seqLen, 32) }, (_, i) => (
                <div
                  key={i}
                  className="w-6 h-4 flex items-center justify-center text-xs text-muted-foreground"
                >
                  {i}
                </div>
              ))}
            </div>

            {/* Matrix */}
            <div className="flex flex-col">
              {currentAttentionData.weights.slice(0, 32).map((row, rowIdx) => (
                <div key={rowIdx} className="flex items-center">
                  {/* Row label */}
                  <div className="w-8 h-6 flex items-center justify-center text-xs text-muted-foreground shrink-0">
                    {rowIdx}
                  </div>

                  {/* Cells */}
                  <div className="flex">
                    {row.slice(0, 32).map((value: number, colIdx: number) => (
                      <div
                        key={colIdx}
                        className="w-6 h-6 border border-border cursor-pointer relative group"
                        style={{
                          backgroundColor: getColor(value),
                        }}
                        onMouseEnter={() => setHoveredCell({ row: rowIdx, col: colIdx })}
                        onMouseLeave={() => setHoveredCell(null)}
                      >
                        {/* Tooltip */}
                        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 px-2 py-1 bg-background border rounded text-xs whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none z-10">
                          [{rowIdx}, {colIdx}]: {typeof value === 'number' ? value.toFixed(4) : value}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Hover info */}
        {hoveredCell && (
          <div className="text-sm text-muted-foreground">
            Attention from token <Badge variant="outline">{hoveredCell.row}</Badge> to token{' '}
            <Badge variant="outline">{hoveredCell.col}</Badge>:{' '}
            {typeof currentAttentionData.weights[hoveredCell.row]?.[hoveredCell.col] === 'number'
              ? currentAttentionData.weights[hoveredCell.row][hoveredCell.col].toFixed(6)
              : 'N/A'}
          </div>
        )}

        {/* Labels */}
        <div className="text-xs text-muted-foreground">
          Rows = Query positions, Columns = Key positions
        </div>

        {/* Color Scale Legend */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">0</span>
          <div className="flex-1 h-3 rounded" style={{
            background: `linear-gradient(to right, ${getColor(0)}, ${getColor(1)})`
          }} />
          <span className="text-xs text-muted-foreground">1</span>
        </div>

        {/* Color Scheme Options */}
        <div className="flex gap-2 flex-wrap">
          {(['blues', 'reds', 'viridis', 'plasma', 'inferno'] as const).map((scheme) => (
            <Button
              key={scheme}
              variant={colorScheme === scheme ? 'default' : 'outline'}
              size="sm"
              onClick={() => setColorScheme(scheme)}
              className="transition-all duration-200"
            >
              <span className={`w-3 h-3 rounded-full mr-2 inline-block ${
                scheme === 'blues' ? 'bg-blue-500' :
                scheme === 'reds' ? 'bg-red-500' :
                scheme === 'viridis' ? 'bg-green-500' :
                scheme === 'plasma' ? 'bg-purple-500' :
                'bg-orange-500'
              }`} />
              {scheme}
            </Button>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};
