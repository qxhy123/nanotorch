import React, { useMemo, useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../ui/card';
import { Badge } from '../../ui/badge';
import { Button } from '../../ui/button';
import { useTransformerStore } from '../../../stores/transformerStore';
import { useSemanticColors } from '../../../hooks/useSemanticColors';
import { Brain, Play, Pause, ChevronDown, ChevronUp, Info } from 'lucide-react';
import { Latex } from '../../ui/Latex';

interface MultiHeadAttentionProps {
  className?: string;
}

export const MultiHeadAttention: React.FC<MultiHeadAttentionProps> = ({ className }) => {
  const {
    attentionWeights,
    config,
    visualizationState,
    inputText,
  } = useTransformerStore();

  const { selectedHead, selectedLayer } = visualizationState;
  const [showComputationDetails, setShowComputationDetails] = useState(true);
  const { query, key, value, attention } = useSemanticColors();

  const attentionData = useMemo(() => {
    if (!attentionWeights || attentionWeights.length === 0) return null;
    const data = attentionWeights[Math.min(selectedLayer, attentionWeights.length - 1)];

    if (data?.queries?.data && data?.keys?.data && data?.values?.data) {
      const headDim = Math.floor(config.d_model / config.nhead);
      const queries = data.queries.data as any;
      const keys = data.keys.data as any;
      const values = data.values.data as any;

      if (Array.isArray(queries) && queries.length > 0) {
        const batch = queries[0];
        if (Array.isArray(batch) && batch.length > selectedHead) {
          const headQ = batch[selectedHead];
          const headK = keys[0][selectedHead];
          const headV = values[0][selectedHead];

          if (Array.isArray(headQ) && Array.isArray(headQ[0])) {
            return {
              ...data,
              headData: {
                queries: headQ as number[][],
                keys: headK as number[][],
                values: headV as number[][],
                headDim,
              },
            };
          }
        }
      }
    }

    return data;
  }, [attentionWeights, selectedLayer, selectedHead, config]);

  const headDim = Math.floor(config.d_model / config.nhead);
  const tokens = useMemo(() => inputText.split('').slice(0, 16), [inputText]);

  if (!attentionData) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle>Multi-Head Attention</CardTitle>
          <CardDescription>Run a forward pass to see multi-head attention details</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5 text-primary" />
              Multi-Head Attention
            </CardTitle>
            <CardDescription>
              Layer {selectedLayer + 1} | Head {selectedHead + 1} / {config.nhead}
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Quick Stats */}
        <div className="grid grid-cols-4 gap-3">
          <div className="p-3 rounded-lg border" style={{ backgroundColor: `${query.primary}10`, borderColor: `${query.primary}40` }}>
            <div className="text-xs font-medium" style={{ color: query.primary }}>Queries</div>
            <div className="text-lg font-bold">Q</div>
            <div className="text-[10px] text-muted-foreground">{headDim}d</div>
          </div>
          <div className="p-3 rounded-lg border" style={{ backgroundColor: `${key.primary}10`, borderColor: `${key.primary}40` }}>
            <div className="text-xs font-medium" style={{ color: key.primary }}>Keys</div>
            <div className="text-lg font-bold">K</div>
            <div className="text-[10px] text-muted-foreground">{headDim}d</div>
          </div>
          <div className="p-3 rounded-lg border" style={{ backgroundColor: `${value.primary}10`, borderColor: `${value.primary}40` }}>
            <div className="text-xs font-medium" style={{ color: value.primary }}>Values</div>
            <div className="text-lg font-bold">V</div>
            <div className="text-[10px] text-muted-foreground">{headDim}d</div>
          </div>
          <div className="p-3 rounded-lg border" style={{ backgroundColor: `${attention.primary}10`, borderColor: `${attention.primary}40` }}>
            <div className="text-xs font-medium" style={{ color: attention.primary }}>Output</div>
            <div className="text-lg font-bold">O</div>
            <div className="text-[10px] text-muted-foreground">{config.d_model}d</div>
          </div>
        </div>

        {/* Toggle Computation Details */}
        <button
          onClick={() => setShowComputationDetails(!showComputationDetails)}
          className="flex items-center gap-2 text-sm font-medium hover:text-primary transition-colors w-full"
        >
          <Info className="h-4 w-4" />
          <span>Computation Details</span>
          {showComputationDetails ? (
            <ChevronUp className="h-4 w-4 ml-auto" />
          ) : (
            <ChevronDown className="h-4 w-4 ml-auto" />
          )}
        </button>

        {showComputationDetails && (
          <div className="space-y-3 animate-in fade-in slide-in-from-top-2 duration-300">
            <div className="p-4 bg-muted/50 rounded-lg">
              <div className="text-center">
                <Latex display>{`Attention(Q, K, V) = softmax(\\frac{QK^T}{\\sqrt{d_k}}) \\cdot V`}</Latex>
              </div>
              <div className="text-xs text-center text-muted-foreground mt-2">
                Scale factor: <Latex>{`\\frac{1}{\\sqrt{${headDim}}} ≈ ${(1 / Math.sqrt(headDim)).toFixed(4)}`}</Latex>
              </div>
            </div>

            <div className="space-y-2">
              <div className="text-sm font-medium flex items-center gap-2">
                <span className="w-6 h-6 bg-blue-500/20 rounded-full flex items-center justify-center text-xs font-bold">1</span>
                Linear Projections
              </div>
              <div className="text-xs text-muted-foreground pl-8">
                Transform input into Q, K, V matrices
              </div>
            </div>

            <div className="space-y-2">
              <div className="text-sm font-medium flex items-center gap-2">
                <span className="w-6 h-6 bg-purple-500/20 rounded-full flex items-center justify-center text-xs font-bold">2</span>
                Split into Heads
              </div>
              <div className="text-xs text-muted-foreground pl-8">
                {config.nhead} heads × {headDim} dimensions each
              </div>
            </div>

            <div className="space-y-2">
              <div className="text-sm font-medium flex items-center gap-2">
                <span className="w-6 h-6 bg-orange-500/20 rounded-full flex items-center justify-center text-xs font-bold">3</span>
                Scaled Dot-Product Attention
              </div>
              <div className="text-xs text-muted-foreground pl-8">
                Compute Q·K^T, scale, apply softmax, multiply by V
              </div>
            </div>

            <div className="space-y-2">
              <div className="text-sm font-medium flex items-center gap-2">
                <span className="w-6 h-6 bg-green-500/20 rounded-full flex items-center justify-center text-xs font-bold">4</span>
                Concatenate & Project
              </div>
              <div className="text-xs text-muted-foreground pl-8">
                Merge all heads and apply output projection
              </div>
            </div>
          </div>
        )}

        {(attentionData as any)?.headData && (
          <InteractiveQKVVisualization
            queries={(attentionData as any).headData.queries}
            keys={(attentionData as any).headData.keys}
            values={(attentionData as any).headData.values}
            headDim={(attentionData as any).headData.headDim}
            seqLen={(attentionData as any).headData.queries.length}
            tokens={tokens}
            scale={1 / Math.sqrt(headDim)}
          />
        )}
      </CardContent>
    </Card>
  );
};

interface InteractiveQKVVisualizationProps {
  queries: number[][];
  keys: number[][];
  values: number[][];
  headDim: number;
  seqLen: number;
  tokens: string[];
  scale: number;
}

type ViewMode = 'attention' | 'qkv' | 'computation' | 'flow';

const InteractiveQKVVisualization: React.FC<InteractiveQKVVisualizationProps> = ({
  queries,
  keys,
  values,
  headDim,
  seqLen,
  tokens,
  scale,
}) => {
  const [viewMode, setViewMode] = React.useState<ViewMode>('attention');
  const [hoveredToken, setHoveredToken] = useState<number | null>(null);
  const [selectedToken, setSelectedToken] = useState<number | null>(null);
  const [selectedDim, setSelectedDim] = useState<number | null>(null);
  const [computationStep, setComputationStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [hoveredCell, setHoveredCell] = useState<{ pos: number; dim: number } | null>(null);

  // Calculate attention scores
  const attentionScores = useMemo(() => {
    const scores: number[][] = [];
    for (let i = 0; i < seqLen; i++) {
      scores[i] = [];
      for (let j = 0; j < seqLen; j++) {
        let dotProduct = 0;
        for (let d = 0; d < headDim; d++) {
          dotProduct += queries[i][d] * keys[j][d];
        }
        scores[i][j] = dotProduct * scale;
      }
    }
    return scores;
  }, [queries, keys, seqLen, headDim, scale]);

  // Calculate softmax probabilities
  const attentionProbs = useMemo(() => {
    return attentionScores.map(row => {
      const maxScore = Math.max(...row);
      const expScores = row.map(s => Math.exp(s - maxScore));
      const sumExp = expScores.reduce((a, b) => a + b, 0);
      return expScores.map(s => s / sumExp);
    });
  }, [attentionScores]);

  // Get heatmap color
  const getHeatmapColor = useCallback((value: number, min: number, max: number) => {
    if (max === min) return 'rgb(220, 220, 220)';
    const normalized = (value - min) / (max - min);

    if (normalized < 0.5) {
      const blue = Math.round(255 * (0.5 + normalized));
      const red = Math.round(255 * (0.5 - normalized));
      return `rgb(${red}, 240, ${blue})`;
    } else {
      const red = Math.round(255 * (0.5 + (normalized - 0.5)));
      const blue = Math.round(255 * (1 - (normalized - 0.5)));
      return `rgb(${red}, 240, ${blue})`;
    }
  }, []);

  // Computation animation
  useEffect(() => {
    if (isPlaying && viewMode === 'computation') {
      const interval = setInterval(() => {
        setComputationStep(prev => {
          if (prev >= 4) {
            setIsPlaying(false);
            return 0;
          }
          return prev + 1;
        });
      }, 1500);
      return () => clearInterval(interval);
    }
  }, [isPlaying, viewMode]);

  // Attention Matrix View - 使用 HTML div 而不是 SVG 以获得更好的交互
  const renderAttentionView = () => {
    const flatProbs = attentionProbs.flat();
    const minProb = Math.min(...flatProbs);
    const maxProb = Math.max(...flatProbs);

    const cellSize = 20;

    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h4 className="text-sm font-medium">Attention Pattern</h4>
          <Badge variant="secondary">Token {selectedToken ?? '?'}</Badge>
        </div>

        {/* Attention Matrix Container */}
        <div className="flex gap-2">
          {/* Row labels (left) */}
          <div className="flex flex-col justify-around flex-shrink-0" style={{ width: '20px' }}>
            {tokens.map((token, i) => (
              <div
                key={i}
                className="text-[8px] text-muted-foreground text-right pr-1"
                style={{ height: cellSize, display: 'flex', alignItems: 'center' }}
              >
                {token || `T${i}`}
              </div>
            ))}
          </div>

          {/* Matrix with column labels */}
          <div className="flex-1 overflow-x-auto">
            <div className="inline-block">
              {/* Column labels (top) */}
              <div className="flex mb-1" style={{ marginLeft: '0px' }}>
                {tokens.map((token, i) => (
                  <div
                    key={i}
                    className="text-center text-[8px] text-muted-foreground flex-shrink-0"
                    style={{ width: `${cellSize}px` }}
                  >
                    {token || `T${i}`}
                  </div>
                ))}
              </div>

              {/* Matrix */}
              <div className="relative">
                <div
                  className="grid gap-px bg-border p-px rounded"
                  style={{
                    gridTemplateColumns: `repeat(${seqLen}, ${cellSize}px)`,
                    gridTemplateRows: `repeat(${seqLen}, ${cellSize}px)`,
                  }}
                >
                  {attentionProbs.map((row, i) =>
                    row.map((prob, j) => {
                      const isRowHovered = hoveredToken === i;
                      const isColHovered = hoveredToken === j;
                      const isRowSelected = selectedToken === i;
                      const isColSelected = selectedToken === j;

                      return (
                        <div
                          key={`${i}-${j}`}
                          className="relative cursor-pointer transition-all duration-200"
                          style={{
                            backgroundColor: getHeatmapColor(prob, minProb, maxProb),
                            opacity: (isRowHovered || isColHovered) ? 1 : 0.85,
                            border: (isRowSelected || isColSelected) ? '2px solid hsl(var(--primary))' : 'none',
                            transform: (isRowHovered || isColHovered) ? 'scale(1.05)' : 'scale(1)',
                            zIndex: (isRowHovered || isColHovered) ? 10 : 1,
                            boxShadow: (isRowHovered || isColHovered) ? '0 4px 12px rgba(0,0,0,0.3)' : 'none',
                          }}
                          onMouseEnter={() => setHoveredToken(i)}
                          onMouseLeave={() => setHoveredToken(null)}
                          onClick={() => setSelectedToken(i)}
                          title={`[${i}][${j}]: ${(prob * 100).toFixed(1)}%`}
                        >
                          <span className="absolute inset-0 flex items-center justify-center text-[6px] font-mono mix-blend-difference text-white hidden">
                            {prob.toFixed(2)}
                          </span>
                        </div>
                      );
                    })
                  )}
                </div>

                {/* Highlight overlay for hovered row/column */}
                {hoveredToken !== null && (
                  <div
                    className="pointer-events-none absolute inset-0"
                    style={{
                      background: `linear-gradient(to right, transparent 20px, hsla(var(--primary) / 0.1) 20px, hsla(var(--primary) / 0.1) calc(100% - 20px), transparent calc(100% - 20px)),
                              linear-gradient(to bottom, transparent 20px, hsla(var(--primary) / 0.1) 20px, hsla(var(--primary) / 0.1) calc(100% - 20px), transparent calc(100% - 20px))`,
                    }}
                  />
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Selected token details */}
        {selectedToken !== null && (
          <div className="p-4 bg-muted/50 rounded-lg space-y-3 animate-in fade-in slide-in-from-bottom-2">
            <h5 className="text-sm font-medium">
              Token "{tokens[selectedToken] || `T${selectedToken}`}" Attention
            </h5>

            <div className="space-y-2">
              <div className="text-xs text-muted-foreground">Top Attended Tokens</div>
              {attentionProbs[selectedToken]
                .map((prob, j) => ({ token: j, prob, text: tokens[j] || `T${j}` }))
                .sort((a, b) => b.prob - a.prob)
                .slice(0, 5)
                .map(({ token, prob, text }) => (
                  <div key={token} className="flex items-center gap-2">
                    <span className="text-xs font-medium w-12">{text}</span>
                    <div className="flex-1 h-2 bg-background rounded-full overflow-hidden">
                      <div
                        className="h-full bg-primary transition-all duration-500"
                        style={{ width: `${prob * 100}%` }}
                      />
                    </div>
                    <Badge variant="secondary" className="text-xs">
                      {(prob * 100).toFixed(1)}%
                    </Badge>
                  </div>
                ))}
            </div>
          </div>
        )}

        {/* Color legend */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">Low attention</span>
          <div className="flex-1 h-3 rounded" style={{
            background: `linear-gradient(to right, ${getHeatmapColor(minProb, minProb, maxProb)}, ${getHeatmapColor(maxProb, minProb, maxProb)})`
          }} />
          <span className="text-xs text-muted-foreground">High attention</span>
        </div>
      </div>
    );
  };

  // QKV Matrix View
  const renderQKVView = () => {
    const cellSize = 12;

    return (
      <div className="space-y-6">
        {[
          { name: 'Queries (Q)', data: queries, color: 'blue', desc: 'What to look for' },
          { name: 'Keys (K)', data: keys, color: 'green', desc: 'What to match' },
          { name: 'Values (V)', data: values, color: 'orange', desc: 'Content to extract' },
        ].map(({ name, data, color, desc }) => {
          const flat = data.flat();
          const min = Math.min(...flat);
          const max = Math.max(...flat);
          const displayData = data.slice(0, Math.min(seqLen, 16));
          const displayDim = Math.min(headDim, 32);

          return (
            <div key={name} className="space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <h5 className={`text-sm font-medium text-${color}-500`}>{name}</h5>
                  <p className="text-xs text-muted-foreground">{desc}</p>
                </div>
                <Badge variant="secondary">({seqLen}, {headDim})</Badge>
              </div>

              {/* Interactive heatmap */}
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-muted-foreground">
                    {hoveredCell ? `[${hoveredCell.pos}, ${hoveredCell.dim}]` : 'Hover over cells'}
                  </span>
                  {hoveredCell && (
                    <Badge variant="outline" className="text-xs">
                      {data[hoveredCell.pos]?.[hoveredCell.dim]?.toFixed(4) ?? 'N/A'}
                    </Badge>
                  )}
                </div>

                <div className="overflow-x-auto">
                  <div className="inline-block">
                    {/* Column labels */}
                    <div className="flex" style={{ marginLeft: '16px' }}>
                      {Array.from({ length: displayDim }).map((_, j) => (
                        <div
                          key={j}
                          className="text-[8px] text-muted-foreground text-center flex-shrink-0"
                          style={{ width: cellSize }}
                        >
                          {j}
                        </div>
                      ))}
                    </div>

                    <div className="flex">
                      {/* Row labels */}
                      <div className="flex flex-col flex-shrink-0" style={{ width: '14px' }}>
                        {displayData.map((_, i) => (
                          <div
                            key={i}
                            className="text-[8px] text-muted-foreground text-right pr-1"
                            style={{ height: cellSize, display: 'flex', alignItems: 'center' }}
                          >
                            {i}
                          </div>
                        ))}
                      </div>

                      {/* Matrix */}
                      <div
                        className="grid gap-px bg-border p-px rounded"
                        style={{
                          gridTemplateColumns: `repeat(${displayDim}, ${cellSize}px)`,
                          gridTemplateRows: `repeat(${displayData.length}, ${cellSize}px)`,
                        }}
                      >
                        {displayData.map((row, i) =>
                          row.slice(0, displayDim).map((value, j) => {
                            const isHovered = hoveredCell?.pos === i && hoveredCell?.dim === j;
                            const isRowHovered = hoveredCell?.pos === i;
                            const isColHovered = hoveredCell?.dim === j;
                            const isRowSelected = selectedToken === i;
                            const isColSelected = selectedDim === j;

                            return (
                              <div
                                key={`${i}-${j}`}
                                className="relative cursor-pointer transition-all duration-150"
                                style={{
                                  backgroundColor: getHeatmapColor(value, min, max),
                                  opacity: (isRowHovered || isColHovered) ? 1 : 0.8,
                                  border: (isRowSelected || isColSelected) ? '2px solid hsl(var(--primary))' : 'none',
                                  transform: isHovered ? 'scale(1.3)' : 'scale(1)',
                                  zIndex: isHovered ? 10 : 1,
                                  boxShadow: isHovered ? '0 2px 8px rgba(0,0,0,0.3)' : 'none',
                                }}
                                onMouseEnter={() => setHoveredCell({ pos: i, dim: j })}
                                onMouseLeave={() => setHoveredCell(null)}
                                onClick={() => {
                                  setSelectedToken(i);
                                  setSelectedDim(j);
                                }}
                                title={`[${i}, ${j}]: ${value.toFixed(6)}`}
                              >
                                {/* Value tooltip on hover */}
                                {isHovered && (
                                  <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 px-2 py-1 bg-background border rounded shadow-lg whitespace-nowrap z-50">
                                    <div className="text-[10px] font-mono">
                                      {value.toFixed(6)}
                                    </div>
                                  </div>
                                )}
                              </div>
                            );
                          })
                        )}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Color legend */}
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-muted-foreground">{min.toFixed(2)}</span>
                  <div className="flex-1 h-2 rounded" style={{
                    background: `linear-gradient(to right, ${getHeatmapColor(min, min, max)}, ${getHeatmapColor(max, min, max)})`
                  }} />
                  <span className="text-[10px] text-muted-foreground">{max.toFixed(2)}</span>
                </div>

                {/* Stats */}
                <div className="flex gap-2 flex-wrap">
                  <Badge variant="outline" className="text-xs">
                    Min: {min.toFixed(3)}
                  </Badge>
                  <Badge variant="outline" className="text-xs">
                    Max: {max.toFixed(3)}
                  </Badge>
                  <Badge variant="outline" className="text-xs">
                    Mean: {(flat.reduce((a, b) => a + b, 0) / flat.length).toFixed(3)}
                  </Badge>
                  <Badge variant="outline" className="text-xs">
                    Std: {Math.sqrt(flat.reduce((sum, v) => sum + Math.pow(v - flat.reduce((a, b) => a + b, 0) / flat.length, 2), 0) / flat.length).toFixed(3)}
                  </Badge>
                </div>

                {/* Selected cell details */}
                {(selectedToken !== null || selectedDim !== null) && (
                  <div className="p-3 bg-muted/50 rounded-lg space-y-2">
                    <div className="text-xs font-medium">
                      {selectedToken !== null && selectedDim !== null
                        ? `Position ${selectedToken}, Dimension ${selectedDim}`
                        : selectedToken !== null
                        ? `Position ${selectedToken} (all dimensions)`
                        : `Dimension ${selectedDim} (all positions)`}
                    </div>

                    {selectedToken !== null && selectedDim !== null && (
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-muted-foreground">Value:</span>
                        <Badge variant="secondary">{data[selectedToken]?.[selectedDim]?.toFixed(6) ?? 'N/A'}</Badge>
                      </div>
                    )}

                    {selectedToken !== null && (
                      <div className="space-y-1">
                        <div className="text-[10px] text-muted-foreground">Vector values (first 8 dims):</div>
                        <div className="flex gap-1 flex-wrap">
                          {data[selectedToken]?.slice(0, 8).map((v, idx) => (
                            <div
                              key={idx}
                              className="text-[10px] font-mono px-1 py-0.5 rounded"
                              style={{
                                backgroundColor: getHeatmapColor(v, min, max),
                                color: Math.abs(v) > (max - min) / 2 ? 'white' : 'black',
                              }}
                            >
                              {v.toFixed(2)}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  // Step-by-step computation with real data
  const renderComputationView = () => {
    const cellSize = 28;

    return (
      <div className="space-y-4">
        {/* Animation controls */}
        <div className="flex items-center gap-2">
          <Button
            size="sm"
            variant={isPlaying ? 'destructive' : 'default'}
            onClick={() => setIsPlaying(!isPlaying)}
          >
            {isPlaying ? <Pause className="h-4 w-4 mr-1" /> : <Play className="h-4 w-4 mr-1" />}
            {isPlaying ? 'Pause' : 'Play Demo'}
          </Button>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            Step: {computationStep + 1} / 4
          </div>
          <div className="flex gap-1">
            {[0, 1, 2, 3].map((i) => (
              <div
                key={i}
                className={`w-6 h-1 rounded transition-colors duration-300 ${
                  computationStep >= i ? 'bg-primary' : 'bg-muted'
                }`}
              />
            ))}
          </div>
        </div>

        {/* Step 1: Q · K^T */}
        <div className={`p-4 rounded-lg border-2 transition-all duration-300 ${
          computationStep >= 0 ? 'border-primary bg-primary/5' : 'border-border bg-muted/30'
        }`}>
          <div className="flex items-center gap-2 mb-3">
            <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
              computationStep >= 0 ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground'
            }`}>1</div>
            <span className="text-sm font-medium"><Latex>{`Q \\cdot K^T`}</Latex> (Dot Product)</span>
            <Badge variant="outline" className="text-xs ml-auto">
              Shape: <Latex>{`(${seqLen}, ${seqLen})`}</Latex>
            </Badge>
          </div>

          <div className="text-xs text-muted-foreground mb-2">
            Compute similarity between each query and key: <Latex>{`scores[i,j] = Q[i] \\cdot K[j]`}</Latex>
          </div>

          {/* Show matrix for selected token */}
          {selectedToken !== null && (
            <div className="space-y-2">
              <div className="text-xs font-medium"><Latex>{`Q[${selectedToken}] \\cdot K^T:`}</Latex></div>
              <div className="overflow-x-auto">
                <div className="inline-grid gap-1 p-2 bg-background rounded" style={{
                  gridTemplateColumns: `repeat(${seqLen}, ${cellSize}px)`,
                }}>
                  <div className="contents">
                    <div></div>
                    {Array.from({ length: seqLen }).map((_, j) => (
                      <div key={j} className="text-[8px] text-center text-muted-foreground">K{j}</div>
                    ))}
                  </div>
                  {attentionScores[selectedToken].map((score, j) => (
                    <div key={j} className="contents">
                      <div className="text-[8px] text-right pr-1 text-muted-foreground">Q{selectedToken}</div>
                      <div
                        className="relative cursor-pointer hover:ring-2 hover:ring-primary transition-all"
                        style={{
                          width: cellSize,
                          height: cellSize,
                          backgroundColor: getHeatmapColor(score, Math.min(...attentionScores.flat()), Math.max(...attentionScores.flat())),
                        }}
                        title={`Raw: ${(queries[selectedToken] || []).reduce((sum: number, v: number, idx: number) => {
                          const kVal = (keys[j] || [])[idx] || 0;
                          return sum + v * kVal;
                        }, 0).toFixed(4)}`}
                      >
                        <span className="absolute inset-0 flex items-center justify-center text-[8px] font-mono mix-blend-difference text-white">
                          {score.toFixed(1)}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Show all scores as small matrix */}
          <div className="overflow-x-auto">
            <div className="text-[10px] text-muted-foreground mb-1">Full Score Matrix:</div>
            <div className="inline-grid gap-px bg-border p-px rounded" style={{
              gridTemplateColumns: `repeat(${Math.min(seqLen, 12)}, 14px)`,
              gridTemplateRows: `repeat(${Math.min(seqLen, 8)}, 14px)`,
            }}>
              {attentionScores.slice(0, 8).map((row, i) =>
                row.slice(0, 12).map((score, j) => (
                  <div
                    key={`${i}-${j}`}
                    className="cursor-pointer hover:ring-1 hover:ring-primary transition-all"
                    style={{
                      backgroundColor: getHeatmapColor(score, Math.min(...attentionScores.flat()), Math.max(...attentionScores.flat())),
                    }}
                    onClick={() => setSelectedToken(i)}
                    title={`[${i},${j}]: ${score.toFixed(4)}`}
                  />
                ))
              )}
            </div>
          </div>
        </div>

        {/* Step 2: Scale */}
        <div className={`p-4 rounded-lg border-2 transition-all duration-300 ${
          computationStep >= 1 ? 'border-primary bg-primary/5' : 'border-border bg-muted/30'
        }`}>
          <div className="flex items-center gap-2 mb-3">
            <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
              computationStep >= 1 ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground'
            }`}>2</div>
            <span className="text-sm font-medium">Scale by <Latex>{`\\frac{1}{\\sqrt{d_k}}`}</Latex></span>
            <Badge variant="outline" className="text-xs">
              Factor: <Latex>{`\\frac{1}{\\sqrt{${headDim}}}`}</Latex> = {scale.toFixed(4)}
            </Badge>
          </div>

          <div className="text-xs text-muted-foreground mb-2">
            Divide scores to prevent vanishing gradients: <Latex>{`scaled = scores / \\sqrt{d_k}`}</Latex>
          </div>

          {/* Example calculation */}
          {selectedToken !== null && computationStep >= 1 && (
            <div className="space-y-2">
              <div className="text-xs font-medium">Example: <Latex>{`Q[${selectedToken}]`}</Latex> scores after scaling:</div>
              <div className="flex gap-1 flex-wrap">
                {attentionScores[selectedToken].slice(0, 6).map((score, j) => (
                  <div key={j} className="text-xs">
                    <span className="text-muted-foreground">K{j}:</span>
                    <Badge variant="secondary" className="ml-1">{score.toFixed(3)}</Badge>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Step 3: Softmax */}
        <div className={`p-4 rounded-lg border-2 transition-all duration-300 ${
          computationStep >= 2 ? 'border-primary bg-primary/5' : 'border-border bg-muted/30'
        }`}>
          <div className="flex items-center gap-2 mb-3">
            <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
              computationStep >= 2 ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground'
            }`}>3</div>
            <span className="text-sm font-medium">Softmax</span>
          </div>

          <div className="text-xs text-muted-foreground mb-2">
            Convert scores to probabilities: <Latex>{`attn[i,j] = \\frac{\\exp(scaled[i,j])}{\\sum_k \\exp(scaled[i,k])}`}</Latex>
          </div>

          {/* Softmax formula explanation */}
          <div className="p-3 bg-background rounded text-xs space-y-1 mb-3">
            <div>1. Subtract max: <Latex>{`s'_{ij} = s_{ij} - \\max_k(s_{ik})`}</Latex></div>
            <div>2. Apply exponential: <Latex>{`\\tilde{a}_{ij} = \\exp(s'_{ij})`}</Latex></div>
            <div>3. Normalize: <Latex>{`a_{ij} = \\frac{\\tilde{a}_{ij}}{\\sum_k \\tilde{a}_{ik}}`}</Latex></div>
          </div>

          {/* Show softmax result for selected token */}
          {selectedToken !== null && computationStep >= 2 && (
            <div className="space-y-2">
              <div className="text-xs font-medium">Q[{selectedToken}] Attention Weights:</div>
              <div className="flex gap-1 flex-wrap">
                {attentionProbs[selectedToken].map((prob, j) => (
                  <div key={j} className="flex items-center gap-1">
                    <span className="text-[10px] text-muted-foreground">K{j}:</span>
                    <div className="w-16 h-2 bg-muted rounded-full overflow-hidden">
                      <div
                        className="h-full bg-primary"
                        style={{ width: `${prob * 100}%` }}
                      />
                    </div>
                    <Badge variant="secondary" className="text-[10px]">
                      {(prob * 100).toFixed(1)}%
                    </Badge>
                  </div>
                ))}
              </div>
              <div className="text-[10px] text-muted-foreground">
                Sum: {(attentionProbs[selectedToken].reduce((a, b) => a + b, 0)).toFixed(6)} (should be 1.0)
              </div>
            </div>
          )}
        </div>

        {/* Step 4: Weighted Sum */}
        <div className={`p-4 rounded-lg border-2 transition-all duration-300 ${
          computationStep >= 3 ? 'border-primary bg-primary/5' : 'border-border bg-muted/30'
        }`}>
          <div className="flex items-center gap-2 mb-3">
            <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
              computationStep >= 3 ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground'
            }`}>4</div>
            <span className="text-sm font-medium">Weighted Sum (<Latex>{`\\cdot V`}</Latex>)</span>
          </div>

          <div className="text-xs text-muted-foreground mb-2">
            Get final output: <Latex>{`output[i] = \\sum_j attn[i,j] \\cdot V[j]`}</Latex>
          </div>

          {/* Show weighted sum calculation */}
          {selectedToken !== null && computationStep >= 3 && (
            <div className="space-y-2">
              <div className="text-xs font-medium">Output[{selectedToken}] = <Latex>{`\\sum_j attn[${selectedToken},j] \\cdot V[j]`}</Latex>:</div>
              <div className="p-3 bg-background rounded text-xs space-y-1">
                {attentionProbs[selectedToken].slice(0, 4).map((prob, j) => (
                  <div key={j} className="flex items-center gap-2">
                    <span className="w-8 text-muted-foreground">V{j}:</span>
                    <span className="font-mono">({(values[j] || [])[0]?.toFixed(2) ?? 'N/A'}, ...)</span>
                    <span className="text-muted-foreground">×</span>
                    <Badge variant="secondary">{prob.toFixed(3)}</Badge>
                  </div>
                ))}
                <div className="text-muted-foreground pt-1 border-t">
                  = weighted sum of all {seqLen} value vectors
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Quick reference */}
        <div className="grid grid-cols-3 gap-2 text-xs">
          <div className="p-2 bg-blue-500/10 rounded">
            <div className="font-medium text-blue-500">Q (Query)</div>
            <div className="font-mono">({seqLen}, {headDim})</div>
            <div className="text-muted-foreground">What to look for</div>
          </div>
          <div className="p-2 bg-green-500/10 rounded">
            <div className="font-medium text-green-500">K (Key)</div>
            <div className="font-mono">({seqLen}, {headDim})</div>
            <div className="text-muted-foreground">What to match</div>
          </div>
          <div className="p-2 bg-orange-500/10 rounded">
            <div className="font-medium text-orange-500">V (Value)</div>
            <div className="font-mono">({seqLen}, {headDim})</div>
            <div className="text-muted-foreground">Content to extract</div>
          </div>
        </div>
      </div>
    );
  };

  // Data flow view
  const renderFlowView = () => {
    return (
      <div className="space-y-4">
        <div className="flex items-center justify-center gap-4">
          <div className="flex flex-col items-center gap-2">
            <div className="w-16 h-16 bg-muted rounded-lg flex items-center justify-center">
              <span className="text-xs font-medium">Input</span>
            </div>
            <span className="text-xs text-muted-foreground">({seqLen}, {headDim})</span>
          </div>

          <div className="text-muted-foreground">→</div>

          <div className="flex flex-col items-center gap-2">
            <div className="flex gap-1">
              <div className="w-10 h-10 bg-blue-500/20 rounded flex items-center justify-center text-xs font-medium">Q</div>
              <div className="w-10 h-10 bg-green-500/20 rounded flex items-center justify-center text-xs font-medium">K</div>
              <div className="w-10 h-10 bg-orange-500/20 rounded flex items-center justify-center text-xs font-medium">V</div>
            </div>
            <span className="text-xs text-muted-foreground">Projections</span>
          </div>

          <div className="text-muted-foreground">→</div>

          <div className="flex flex-col items-center gap-2">
            <div className="w-20 h-16 bg-purple-500/20 rounded-lg flex items-center justify-center">
              <span className="text-xs font-medium">Attention</span>
            </div>
            <span className="text-xs text-muted-foreground">Softmax(Q·K^T)</span>
          </div>

          <div className="text-muted-foreground">→</div>

          <div className="flex flex-col items-center gap-2">
            <div className="w-16 h-16 bg-primary/20 rounded-lg flex items-center justify-center">
              <span className="text-xs font-medium">Output</span>
            </div>
            <span className="text-xs text-muted-foreground">({seqLen}, {headDim})</span>
          </div>
        </div>

        <div className="space-y-2">
          <h5 className="text-sm font-medium">Data Dimensions</h5>
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div className="p-2 bg-blue-500/10 rounded">
              <div className="font-medium text-blue-500">Q</div>
              <div className="font-mono">({seqLen}, {headDim})</div>
            </div>
            <div className="p-2 bg-green-500/10 rounded">
              <div className="font-medium text-green-500">K</div>
              <div className="font-mono">({seqLen}, {headDim})</div>
            </div>
            <div className="p-2 bg-orange-500/10 rounded">
              <div className="font-medium text-orange-500">V</div>
              <div className="font-mono">({seqLen}, {headDim})</div>
            </div>
          </div>
          <div className="p-2 bg-purple-500/10 rounded text-xs">
            <div className="font-medium text-purple-500">Attention Weights</div>
            <div className="font-mono">({seqLen}, {seqLen})</div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-4">
      <div className="flex gap-2 flex-wrap">
        <Button
          variant={viewMode === 'attention' ? 'default' : 'outline'}
          size="sm"
          onClick={() => setViewMode('attention')}
          className="transition-all"
        >
          🎯 Attention
        </Button>
        <Button
          variant={viewMode === 'qkv' ? 'default' : 'outline'}
          size="sm"
          onClick={() => setViewMode('qkv')}
          className="transition-all"
        >
          📊 QKV Matrices
        </Button>
        <Button
          variant={viewMode === 'computation' ? 'default' : 'outline'}
          size="sm"
          onClick={() => setViewMode('computation')}
          className="transition-all"
        >
          🧮 Computation
        </Button>
        <Button
          variant={viewMode === 'flow' ? 'default' : 'outline'}
          size="sm"
          onClick={() => setViewMode('flow')}
          className="transition-all"
        >
          🔄 Data Flow
        </Button>
      </div>

      <div className="min-h-[300px]">
        {viewMode === 'attention' && renderAttentionView()}
        {viewMode === 'qkv' && renderQKVView()}
        {viewMode === 'computation' && renderComputationView()}
        {viewMode === 'flow' && renderFlowView()}
      </div>

      <div className="text-xs text-muted-foreground bg-muted/50 p-2 rounded">
        💡 <strong>Hover</strong> over cells to highlight relationships. <strong>Click</strong> to select a token and see its attention pattern.
      </div>
    </div>
  );
};
