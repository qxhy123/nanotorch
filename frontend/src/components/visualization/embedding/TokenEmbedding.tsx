import React, { useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../ui/card';
import { Badge } from '../../ui/badge';
import { Button } from '../../ui/button';
import { useTransformerStore } from '../../../stores/transformerStore';
import type { TensorData } from '../../../types/transformer';

interface TokenEmbeddingProps {
  className?: string;
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

function extractEmbeddingMatrix(tensor: TensorData | null | undefined): number[][] | null {
  if (!tensor) {
    return null;
  }

  if (tensor.shape.length === 3 && isNumberTensor3D(tensor.data) && tensor.data.length > 0) {
    return tensor.data[0];
  }

  if (tensor.shape.length === 2 && isNumberMatrix(tensor.data)) {
    return tensor.data;
  }

  if (isNumberMatrix(tensor.data)) {
    return tensor.data;
  }

  return null;
}

export const TokenEmbedding: React.FC<TokenEmbeddingProps> = ({ className }) => {
  const { embeddings, visualizationState } = useTransformerStore();
  const { showValues, colorScheme } = visualizationState;

  const embeddingsData = useMemo(
    () => extractEmbeddingMatrix(embeddings?.output),
    [embeddings]
  );

  const tokens = useMemo(() => {
    if (!embeddingsData) {
      return [];
    }

    return embeddingsData.map((_, index) => `Token ${index}`);
  }, [embeddingsData]);

  const getColor = (value: number, min: number, max: number): string => {
    if (max === min) return 'rgb(200, 200, 200)';
    const normalized = (value - min) / (max - min);

    switch (colorScheme) {
      case 'blues': {
        const blue = Math.round(255 * normalized);
        return `rgb(200, 200, ${255 - blue})`;
      }
      case 'reds': {
        const red = Math.round(255 * normalized);
        return `rgb(${255 - red}, 200, 200)`;
      }
      case 'viridis':
        if (normalized < 0.25) {
          return `rgb(68, 1, ${Math.round(84 + normalized * 400)})`;
        }
        if (normalized < 0.5) {
          return `rgb(${Math.round(33 + (normalized - 0.25) * 800)}, ${Math.round(144 + (normalized - 0.25) * 200)}, 140)`;
        }
        if (normalized < 0.75) {
          return `rgb(${Math.round(233 - (normalized - 0.5) * 400)}, ${Math.round(196 - (normalized - 0.5) * 200)}, ${Math.round(106 + (normalized - 0.5) * 320)})`;
        }
        return `rgb(${Math.round(253 - (normalized - 0.75) * 400)}, ${Math.round(231 - (normalized - 0.75) * 400)}, ${Math.round(37 + (normalized - 0.75) * 160)})`;
      default: {
        const shade = Math.round(255 * (1 - normalized));
        return `rgb(${shade}, ${shade}, ${shade})`;
      }
    }
  };

  const stats = useMemo(() => {
    if (!embeddingsData || embeddingsData.length === 0) return null;

    const allValues = embeddingsData.flat();
    const min = Math.min(...allValues);
    const max = Math.max(...allValues);
    const mean = allValues.reduce((sum, value) => sum + value, 0) / allValues.length;

    return { min, max, mean };
  }, [embeddingsData]);

  if (!embeddingsData) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle>Token Embeddings</CardTitle>
          <CardDescription>Run a forward pass to see token embeddings.</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Token Embeddings</CardTitle>
            <CardDescription>
              Shape: ({embeddingsData.length}, {embeddingsData[0]?.length || 0})
            </CardDescription>
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
            {embeddingsData.map((tokenEmbedding, tokenIdx) => (
              <div key={tokenIdx} className="flex items-center gap-2 mb-1">
                <div className="w-16 text-xs text-muted-foreground shrink-0">
                  {tokens[tokenIdx] || `T${tokenIdx}`}
                </div>
                <div className="flex flex-nowrap">
                  {tokenEmbedding.slice(0, 64).map((value, dimIdx) => (
                    <div
                      key={dimIdx}
                      className="w-2 h-6 border-r border-r-background/20 relative group cursor-pointer"
                      style={{
                        backgroundColor: getColor(
                          value,
                          stats?.min || 0,
                          stats?.max || 1
                        ),
                      }}
                    >
                      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 px-2 py-1 bg-background border rounded text-xs whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none z-10 shadow-lg">
                        [{tokenIdx}, {dimIdx}]: {value.toFixed(4)}
                      </div>
                      {showValues && (
                        <div className="absolute inset-0 flex items-center justify-center text-[8px] font-mono text-white mix-blend-difference overflow-hidden">
                          {value.toFixed(2)}
                        </div>
                      )}
                    </div>
                  ))}
                  {tokenEmbedding.length > 64 && (
                    <div className="w-16 h-6 flex items-center justify-center text-xs text-muted-foreground bg-muted/20">
                      +{tokenEmbedding.length - 64}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">Min</span>
          <div
            className="flex-1 h-3 rounded"
            style={{
              background: `linear-gradient(to right, ${
                getColor(stats?.min || 0, stats?.min || 0, stats?.max || 1)
              }, ${
                getColor(stats?.max || 1, stats?.min || 0, stats?.max || 1)
              })`,
            }}
          />
          <span className="text-xs text-muted-foreground">Max</span>
        </div>

        <div className="flex gap-2">
          <Button
            variant={showValues ? 'default' : 'outline'}
            size="sm"
            onClick={() => useTransformerStore.getState().setShowValues(!showValues)}
            className="transition-all"
          >
            {showValues ? (
              <>
                <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
                </svg>
                Hide Inline Values
              </>
            ) : (
              <>
                <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                </svg>
                Show Inline Values
              </>
            )}
          </Button>
        </div>
        <p className="text-xs text-muted-foreground">
          {showValues
            ? 'Values are displayed inline on each cell. Hover any cell for a more precise tooltip.'
            : 'Hover over any cell to see its value in a tooltip.'}
        </p>
      </CardContent>
    </Card>
  );
};
