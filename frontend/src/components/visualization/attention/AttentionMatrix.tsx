import React, { useEffect, useMemo, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../ui/card';
import { Badge } from '../../ui/badge';
import { Button } from '../../ui/button';
import { useTransformerStore } from '../../../stores/transformerStore';
import type { AttentionData } from '../../../types/transformer';

interface AttentionMatrixProps {
  className?: string;
}

interface ProcessedAttentionData {
  weights: number[][];
  scale: number;
  selectedHead: number;
}

function isNumberArray(value: unknown): value is number[] {
  return Array.isArray(value) && value.every((entry) => typeof entry === 'number');
}

function isNumberMatrix(value: unknown): value is number[][] {
  return Array.isArray(value) && value.every((entry) => isNumberArray(entry));
}

function isNumberTensor3D(value: unknown): value is number[][][] {
  return Array.isArray(value) && value.every((entry) => isNumberMatrix(entry));
}

function isNumberTensor4D(value: unknown): value is number[][][][] {
  return Array.isArray(value) && value.every((entry) => isNumberTensor3D(entry));
}

function extractAttentionMatrix(
  attention: AttentionData | null,
  selectedHead: number,
  configHeads: number,
  defaultScale: number
): ProcessedAttentionData | null {
  if (!attention) {
    return null;
  }

  const safeSelectedHead = Math.min(selectedHead, Math.max(0, configHeads - 1));
  const { weights } = attention;

  if (weights.shape.length === 4 && isNumberTensor4D(weights.data) && weights.data.length > 0) {
    const batchData = weights.data[0];
    if (batchData.length === 0) {
      return null;
    }

    const resolvedHead = Math.min(safeSelectedHead, batchData.length - 1);
    return {
      weights: batchData[resolvedHead],
      scale: attention.scale || defaultScale,
      selectedHead: resolvedHead,
    };
  }

  if (weights.shape.length === 3 && isNumberTensor3D(weights.data) && weights.data.length > 0) {
    return {
      weights: weights.data[0],
      scale: attention.scale || defaultScale,
      selectedHead: safeSelectedHead,
    };
  }

  if (isNumberMatrix(weights.data)) {
    return {
      weights: weights.data,
      scale: attention.scale || defaultScale,
      selectedHead: safeSelectedHead,
    };
  }

  return null;
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

  const currentAttentionData = useMemo(
    () =>
      extractAttentionMatrix(
        attentionWeights?.[0] || null,
        selectedHead,
        config.nhead,
        1 / Math.sqrt(config.d_model / config.nhead)
      ),
    [attentionWeights, selectedHead, config]
  );

  useEffect(() => {
    if (
      currentAttentionData &&
      currentAttentionData.selectedHead !== selectedHead
    ) {
      setSelectedHead(currentAttentionData.selectedHead);
    }
  }, [currentAttentionData, selectedHead, setSelectedHead]);

  const getColor = (value: number): string => {
    const normalized = Math.max(0, Math.min(1, value));

    switch (colorScheme) {
      case 'blues':
        return `rgba(59, 130, 246, ${0.2 + normalized * 0.8})`;
      case 'reds':
        return `rgba(239, 68, 68, ${0.2 + normalized * 0.8})`;
      case 'viridis':
        if (normalized < 0.25) {
          return `rgba(68, 1, 84, ${0.2 + normalized * 3.2})`;
        }
        if (normalized < 0.5) {
          return `rgba(59, 82, 139, ${0.4 + (normalized - 0.25) * 1.6})`;
        }
        if (normalized < 0.75) {
          return `rgba(33, 154, 131, ${0.4 + (normalized - 0.5) * 1.6})`;
        }
        return `rgba(94, 201, 98, ${0.4 + (normalized - 0.75) * 1.6})`;
      case 'plasma':
        if (normalized < 0.33) {
          return `rgba(13, 8, 135, ${0.2 + normalized * 2.4})`;
        }
        if (normalized < 0.66) {
          return `rgba(204, 70, 120, ${0.2 + (normalized - 0.33) * 2.4})`;
        }
        return `rgba(240, 249, 33, ${0.2 + (normalized - 0.66) * 2.4})`;
      case 'inferno':
        if (normalized < 0.33) {
          return `rgba(0, 0, 4, ${0.2 + normalized * 2.4})`;
        }
        if (normalized < 0.66) {
          return `rgba(159, 36, 40, ${0.2 + (normalized - 0.33) * 2.4})`;
        }
        return `rgba(252, 253, 97, ${0.2 + (normalized - 0.66) * 2.4})`;
      default:
        return `rgba(59, 130, 246, ${0.2 + normalized * 0.8})`;
    }
  };

  const stats = useMemo(() => {
    if (!currentAttentionData?.weights) return null;

    const allValues = currentAttentionData.weights.flat();
    const min = Math.min(...allValues);
    const max = Math.max(...allValues);
    const mean = allValues.reduce((sum, value) => sum + value, 0) / allValues.length;

    return { min, max, mean };
  }, [currentAttentionData]);

  if (!currentAttentionData) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle>Attention Weights</CardTitle>
          <CardDescription>
            Run a forward pass to see attention patterns.
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
              Head {currentAttentionData.selectedHead + 1} / {config.nhead} | Scale:{' '}
              {currentAttentionData.scale.toFixed(4)}
            </CardDescription>
          </div>
          <div className="flex gap-2">
            {Array.from({ length: Math.min(config.nhead, 8) }, (_, headIndex) => (
              <Button
                key={headIndex}
                variant={currentAttentionData.selectedHead === headIndex ? 'default' : 'outline'}
                size="sm"
                onClick={() => setSelectedHead(headIndex)}
                className="transition-all duration-200 head-selector"
              >
                H{headIndex + 1}
              </Button>
            ))}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {stats && (
          <div className="flex gap-4 text-sm">
            <Badge variant="secondary">Min: {stats.min.toFixed(4)}</Badge>
            <Badge variant="secondary">Max: {stats.max.toFixed(4)}</Badge>
            <Badge variant="secondary">Mean: {stats.mean.toFixed(4)}</Badge>
          </div>
        )}

        <div className="overflow-x-auto">
          <div className="inline-block min-w-full">
            <div className="flex ml-8 mb-1">
              {Array.from({ length: Math.min(seqLen, 32) }, (_, index) => (
                <div
                  key={index}
                  className="w-6 h-4 flex items-center justify-center text-xs text-muted-foreground"
                >
                  {index}
                </div>
              ))}
            </div>

            <div className="flex flex-col">
              {currentAttentionData.weights.slice(0, 32).map((row, rowIdx) => (
                <div key={rowIdx} className="flex items-center">
                  <div className="w-8 h-6 flex items-center justify-center text-xs text-muted-foreground shrink-0">
                    {rowIdx}
                  </div>

                  <div className="flex">
                    {row.slice(0, 32).map((value, colIdx) => (
                      <div
                        key={colIdx}
                        className="w-6 h-6 border border-border cursor-pointer relative group"
                        style={{
                          backgroundColor: getColor(value),
                        }}
                        onMouseEnter={() => setHoveredCell({ row: rowIdx, col: colIdx })}
                        onMouseLeave={() => setHoveredCell(null)}
                      >
                        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 px-2 py-1 bg-background border rounded text-xs whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none z-10">
                          [{rowIdx}, {colIdx}]: {value.toFixed(4)}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {hoveredCell && (
          <div className="text-sm text-muted-foreground">
            Attention from token <Badge variant="outline">{hoveredCell.row}</Badge> to token{' '}
            <Badge variant="outline">{hoveredCell.col}</Badge>:{' '}
            {currentAttentionData.weights[hoveredCell.row]?.[hoveredCell.col]?.toFixed(6) || 'N/A'}
          </div>
        )}

        <div className="text-xs text-muted-foreground">
          Rows = query positions, columns = key positions.
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">0</span>
          <div
            className="flex-1 h-3 rounded"
            style={{
              background: `linear-gradient(to right, ${getColor(0)}, ${getColor(1)})`,
            }}
          />
          <span className="text-xs text-muted-foreground">1</span>
        </div>

        <div className="flex gap-2 flex-wrap">
          {(['blues', 'reds', 'viridis', 'plasma', 'inferno'] as const).map((scheme) => (
            <Button
              key={scheme}
              variant={colorScheme === scheme ? 'default' : 'outline'}
              size="sm"
              onClick={() => setColorScheme(scheme)}
              className="transition-all duration-200"
            >
              <span
                className={`w-3 h-3 rounded-full mr-2 inline-block ${
                  scheme === 'blues'
                    ? 'bg-blue-500'
                    : scheme === 'reds'
                    ? 'bg-red-500'
                    : scheme === 'viridis'
                    ? 'bg-green-500'
                    : scheme === 'plasma'
                    ? 'bg-purple-500'
                    : 'bg-orange-500'
                }`}
              />
              {scheme}
            </Button>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};
