/**
 * Attention Mechanism Visualization
 *
 * Visualizes self-attention and cross-attention maps with multi-head comparison
 */

import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Slider } from '../components/ui/slider';
import { Grid3x3, Target, Play, Pause } from 'lucide-react';
import { InlineMath } from 'react-katex';
import 'katex/dist/katex.min.css';

type AttentionType = 'self' | 'cross';

// Generate synthetic attention pattern
const generateAttentionPattern = (headId: number, size: number = 16): number[][] => {
  const pattern: number[][] = [];
  const centerX = Math.floor(size / 2);
  const centerY = Math.floor(size / 2);

  for (let i = 0; i < size; i++) {
    pattern[i] = [];
    for (let j = 0; j < size; j++) {
      // Different patterns for different heads
      const dx = i - centerX;
      const dy = j - centerY;
      const dist = Math.sqrt(dx * dx + dy * dy);

      let value = 0;
      switch (headId % 4) {
        case 0: // Radial pattern
          value = Math.max(0, 1 - dist / (size / 2));
          break;
        case 1: // Horizontal stripe
          value = Math.exp(-Math.abs(dy) / 2);
          break;
        case 2: // Vertical stripe
          value = Math.exp(-Math.abs(dx) / 2);
          break;
        case 3: // Diagonal
          value = Math.exp(-Math.abs(dx - dy) / 3);
          break;
      }

      // Add some noise
      value += (Math.random() - 0.5) * 0.1;
      pattern[i][j] = Math.max(0, Math.min(1, value));
    }
  }
  return pattern;
};

// Generate cross-attention token influence
const generateTokenInfluence = (): Array<{ token: string; influence: number; color: string }> => {
  const tokens = [
    { token: 'a', influence: 0.85, color: 'rgb(59, 130, 246)' },
    { token: 'beautiful', influence: 0.72, color: 'rgb(168, 85, 247)' },
    { token: 'landscape', influence: 0.68, color: 'rgb(236, 72, 153)' },
    { token: 'with', influence: 0.25, color: 'rgb(234, 179, 8)' },
    { token: 'mountains', influence: 0.91, color: 'rgb(34, 197, 94)' },
    { token: 'and', influence: 0.15, color: 'rgb(249, 115, 22)' },
    { token: 'sky', influence: 0.58, color: 'rgb(6, 182, 212)' },
  ];
  return tokens.sort((a, b) => b.influence - a.influence);
};

const GRID_SIZE = 16;

export const AttentionView = () => {
  
  const [attentionType, setAttentionType] = useState<AttentionType>('self');
  const [selectedHead, setSelectedHead] = useState(0);
  const [timestep, setTimestep] = useState(250);
  const [isPlaying, setIsPlaying] = useState(false);
  const [hoveredCell, setHoveredCell] = useState<{ x: number; y: number } | null>(null);

  // Generate attention patterns for all heads
  const attentionPatterns = useMemo(() => {
    return Array.from({ length: 8 }, (_, i) => ({
      id: i,
      name: `Head ${i + 1}`,
      pattern: generateAttentionPattern(i, GRID_SIZE)
  }));
  }, [timestep]);

  // Get token influence for cross-attention
  const tokenInfluence = useMemo(() => generateTokenInfluence(), [timestep]);

  // Get current pattern
  const currentPattern = attentionPatterns[selectedHead]?.pattern || [];

  // Auto-play animation
  React.useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      setTimestep(prev => {
        if (prev >= 1000) return 0;
        return prev + 10;
      });
    }, 50);

    return () => clearInterval(interval);
  }, [isPlaying]);

  const getAttentionColor = (value: number): string => {
    // Color scale from blue (low) to red (high)
    const hue = 240 - value * 240;
    return `hsl(${hue}, 70%, 50%)`;
  };

  const getAttentionValue = (x: number, y: number): number => {
    if (currentPattern[y] && currentPattern[y][x] !== undefined) {
      return currentPattern[y][x];
    }
    return 0;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center py-8 bg-gradient-to-br from-primary/10 to-primary/5 rounded-lg">
        <h1 className="text-3xl font-bold mb-2">
          {'Attention Mechanism Visualization'}
        </h1>
        <p className="text-muted-foreground">
          {'Explore self-attention and cross-attention mechanisms in UNet'
          }
        </p>
      </div>

      {/* Mode Switcher */}
      <div className="flex justify-center gap-4">
        <Button
          variant={attentionType === 'self' ? 'default' : 'outline'}
          onClick={() => { setAttentionType('self'); setSelectedHead(0); }}
          className="gap-2"
        >
          <Grid3x3 className="h-4 w-4" />
          {'Self-Attention'}
        </Button>
        <Button
          variant={attentionType === 'cross' ? 'default' : 'outline'}
          onClick={() => { setAttentionType('cross'); setSelectedHead(0); }}
          className="gap-2"
        >
          <Target className="h-4 w-4" />
          {'Cross-Attention'}
        </Button>
      </div>

      {/* Timestep Control */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex items-center gap-4">
            <Button
              variant="outline"
              size="icon"
              onClick={() => setIsPlaying(!isPlaying)}
            >
              {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            </Button>
            <div className="flex-1">
              <div className="flex justify-between text-sm mb-2">
                <span className="text-muted-foreground">
                  {'Timestep'}
                </span>
                <span className="font-mono">{timestep}</span>
              </div>
              <Slider
                value={timestep}
                onValueChange={(v) => setTimestep(v)}
                min={0}
                max={1000}
                step={10}
                className="w-full"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {attentionType === 'self' ? (
        <>
          {/* Multi-Head Comparison */}
          <Card>
            <CardHeader>
              <CardTitle>
                {'Multi-Head Attention Comparison'}
              </CardTitle>
              <CardDescription>
                {'Spatial attention patterns across 8 attention heads'
                }
              </CardDescription>
            </CardHeader>
            <CardContent>
              {/* Head Selection Grid */}
              <div className="grid grid-cols-4 md:grid-cols-8 gap-2 mb-6">
                {attentionPatterns.map((head) => (
                  <button
                    key={head.id}
                    className={`
                      aspect-square p-2 rounded-lg border-2 transition-all
                      ${selectedHead === head.id
                        ? 'border-primary bg-primary/10 scale-105'
                        : 'border-border hover:border-primary/50'
                      }
                    `}
                    onClick={() => setSelectedHead(head.id)}
                  >
                    <svg viewBox="0 0 16 16" className="w-full h-full">
                      {head.pattern.slice(0, 16).map((row, y) =>
                        row.slice(0, 16).map((val, x) => (
                          <rect
                            key={`${x}-${y}`}
                            x={x}
                            y={y}
                            width="1"
                            height="1"
                            fill={getAttentionColor(val)}
                            opacity="0.8"
                          />
                        ))
                      )}
                    </svg>
                  </button>
                ))}
              </div>

              {/* Selected Head Detail */}
              {selectedHead !== null && (
                <div className="space-y-4">
                  <h3 className="text-sm font-medium">
                    {'Attention Head'} {selectedHead + 1}
                  </h3>

                  {/* Large Attention Map */}
                  <div className="relative aspect-square max-w-md mx-auto">
                    <svg
                      viewBox={`0 0 ${GRID_SIZE} ${GRID_SIZE}`}
                      className="w-full h-full rounded-lg overflow-hidden border"
                    >
                      {currentPattern.map((row, y) =>
                        row.map((value, x) => (
                          <rect
                            key={`${x}-${y}`}
                            x={x}
                            y={y}
                            width="1"
                            height="1"
                            fill={getAttentionColor(value)}
                            className={`
                              cursor-pointer transition-opacity
                              ${hoveredCell?.x === x && hoveredCell?.y === y ? 'opacity-100' : 'opacity-80'}
                            `}
                            onMouseEnter={() => setHoveredCell({ x, y })}
                            onMouseLeave={() => setHoveredCell(null)}
                          />
                        ))
                      )}
                    </svg>

                    {/* Hover Info */}
                    {hoveredCell && (
                      <div className="absolute top-2 right-2 bg-black/80 text-white px-2 py-1 rounded text-xs font-mono">
                        [{hoveredCell.x}, {hoveredCell.y}]: {getAttentionValue(hoveredCell.x, hoveredCell.y).toFixed(3)}
                      </div>
                    )}
                  </div>

                  {/* Color Legend */}
                  <div className="flex items-center justify-center gap-2 text-xs">
                    <span>0.0</span>
                    <div className="w-48 h-3 rounded" style={{
                      background: 'linear-gradient(to right, hsl(240, 70%, 50%), hsl(0, 70%, 50%))'
                    }} />
                    <span>1.0</span>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </>
      ) : (
        <>
          {/* Cross-Attention Visualization */}
          <Card>
            <CardHeader>
              <CardTitle>
                {'Cross-Attention Matrix'}
              </CardTitle>
              <CardDescription>
                {'Token influence on spatial positions'
                }
              </CardDescription>
            </CardHeader>
            <CardContent>
              {/* Token Influence Bar Chart */}
              <div className="space-y-3">
                {tokenInfluence.map((item, idx) => (
                  <div key={idx} className="flex items-center gap-4">
                    <div className="w-20 text-sm font-mono text-right">
                      "{item.token}"
                    </div>
                    <div className="flex-1">
                      <div className="h-8 rounded bg-muted overflow-hidden">
                        <div
                          className="h-full transition-all duration-300 flex items-center justify-end px-2 text-white text-xs font-medium"
                          style={{
                            width: `${item.influence * 100}%`,
                            backgroundColor: item.color
  }}
                        >
                          {(item.influence * 100).toFixed(0)}%
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Spatial Heatmap for top token */}
              <div className="mt-6">
                <h3 className="text-sm font-medium mb-4">
                  {'Spatial distribution of most influential token'}
                  : "{tokenInfluence[0]?.token}"
                </h3>
                <div className="aspect-square max-w-md mx-auto">
                  <svg
                    viewBox={`0 0 ${GRID_SIZE} ${GRID_SIZE}`}
                    className="w-full h-full rounded-lg overflow-hidden border"
                  >
                    {currentPattern.map((row, y) =>
                      row.map((value, x) => (
                        <rect
                          key={`${x}-${y}`}
                          x={x}
                          y={y}
                          width="1"
                          height="1"
                          fill={tokenInfluence[0]?.color || '#3b82f6'}
                          fillOpacity={value}
                          className="cursor-pointer hover:fill-opacity-100 transition-all"
                        />
                      ))
                    )}
                  </svg>
                </div>
              </div>
            </CardContent>
          </Card>
        </>
      )}

      {/* Learn More */}
      <Card>
        <CardHeader>
          <CardTitle>
            {'About Attention'}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-sm">
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-medium mb-2 flex items-center gap-2">
                <Grid3x3 className="h-4 w-4" />
                {'Self-Attention'}
              </h4>
              <p className="text-muted-foreground">
                {'Self-attention allows each spatial position to attend to all other positions, capturing long-range dependencies in images. Each head learns different attention patterns.'
                }
              </p>
            </div>
            <div>
              <h4 className="font-medium mb-2 flex items-center gap-2">
                <Target className="h-4 w-4" />
                {'Cross-Attention'}
              </h4>
              <p className="text-muted-foreground">
                {'Cross-attention injects text conditioning into the image generation process, with each spatial position attending to relevant text tokens.'
                }
              </p>
            </div>
          </div>

          <div className="p-4 bg-muted rounded-lg">
            <h4 className="font-medium mb-2">
              {'Key Features'}
            </h4>
            <ul className="space-y-2 text-xs">
              <li>• {'8 attention heads provide diverse representations'}</li>
              <li>• {'Spatial self-attention captures global and local features'}</li>
              <li>• {'Cross-attention enables text-to-image generation'}</li>
              <li>• {'Attention weights evolve dynamically across timesteps'}</li>
            </ul>
          </div>

          <div className="p-4 bg-muted rounded-lg">
            <h4 className="font-medium mb-2">
              {'Computation'}
            </h4>
            <div className="font-mono text-xs space-y-1">
              <div><InlineMath style={{ color: 'inherit' }}>{"\\text{Attention}(Q, K, V) = \\text{softmax}(\\frac{QK^T}{\\sqrt{d_k}})V"}</InlineMath></div>
              <div className="text-muted-foreground">
                {'Where Q=Query, K=Key, V=Value,'} <InlineMath style={{ color: 'inherit' }}>{"d_k"}</InlineMath> {'=dimension'
                }
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
