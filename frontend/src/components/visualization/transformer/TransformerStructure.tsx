import React, { useMemo, useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../ui/card';
import { Badge } from '../../ui/badge';
import { Button } from '../../ui/button';
import { useTransformerStore } from '../../../stores/transformerStore';
import { Box, Layers, ArrowRight, ArrowDown, Play, Pause, RotateCw, Eye } from 'lucide-react';
import { Latex } from '../../ui/Latex';

interface TransformerStructureProps {
  className?: string;
}

type LayerType = 'embedding' | 'positional' | 'attention' | 'feedforward' | 'norm' | 'output';

interface LayerNode {
  id: string;
  name: string;
  type: LayerType;
  description: string;
  shape: string[];
  formula?: string;
  config?: Record<string, any>;
}

const getTransformerLayers = (config: any, hasDecoder: boolean): LayerNode[] => {
  const { d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward } = config;

  const layers: LayerNode[] = [
    {
      id: 'input',
      name: 'Input Tokens',
      type: 'output' as LayerType,
      description: 'Token indices from tokenizer',
      shape: ['batch_size', 'seq_len'],
    },
    {
      id: 'embedding',
      name: 'Token Embedding',
      type: 'embedding' as LayerType,
      description: 'Convert tokens to dense vectors',
      shape: ['batch', 'seq_len', `d_model (${d_model})`],
      formula: 'Token IDs → W_embed[token_id]',
      config: { vocab_size: config.vocab_size },
    },
    {
      id: 'positional',
      name: 'Positional Encoding',
      type: 'positional' as LayerType,
      description: 'Add position information',
      shape: ['batch', 'seq_len', `d_model (${d_model})`],
      formula: 'PE(pos, 2i) = sin(pos/10000^(2i/d_model))',
    },
    {
      id: 'add-embed',
      name: 'Add: Token + Positional',
      type: 'embedding' as LayerType,
      description: 'Combine embeddings',
      shape: ['batch', 'seq_len', `d_model (${d_model})`],
    },
  ];

  // Encoder layers
  for (let i = 0; i < num_encoder_layers; i++) {
    layers.push({
      id: `encoder-${i}-norm1`,
      name: `Encoder Layer ${i + 1}: LayerNorm 1`,
      type: 'norm' as LayerType,
      description: 'Pre-attention normalization',
      shape: ['batch', 'seq_len', `d_model (${d_model})`],
    });

    layers.push({
      id: `encoder-${i}-attention`,
      name: `Encoder Layer ${i + 1}: Multi-Head Attention`,
      type: 'attention' as LayerType,
      description: `${nhead} heads × ${Math.floor(d_model / nhead)}d each`,
      shape: ['batch', 'seq_len', `d_model (${d_model})`],
      config: { nhead, head_dim: Math.floor(d_model / nhead) },
    });

    layers.push({
      id: `encoder-${i}-norm2`,
      name: `Encoder Layer ${i + 1}: LayerNorm 2`,
      type: 'norm' as LayerType,
      description: 'Post-attention normalization',
      shape: ['batch', 'seq_len', `d_model (${d_model})`],
    });

    layers.push({
      id: `encoder-${i}-ffn`,
      name: `Encoder Layer ${i + 1}: Feed Forward`,
      type: 'feedforward' as LayerType,
      description: `Two linear transformations with ${config.activation}`,
      shape: ['batch', 'seq_len', `d_model (${d_model})`, `→ ${dim_feedforward}`, `→ d_model (${d_model})`],
      config: { hidden: dim_feedforward, activation: config.activation },
    });
  }

  // Decoder layers (if enabled)
  if (hasDecoder && num_decoder_layers > 0) {
    layers.push({
      id: 'decoder-bridge',
      name: 'Cross Connection',
      type: 'attention' as LayerType,
      description: 'Encoder output to decoder',
      shape: [`d_model (${d_model})`],
    });

    for (let i = 0; i < num_decoder_layers; i++) {
      layers.push({
        id: `decoder-${i}-norm1`,
        name: `Decoder Layer ${i + 1}: LayerNorm 1`,
        type: 'norm' as LayerType,
        description: 'Pre-attention normalization',
        shape: ['batch', 'seq_len', `d_model (${d_model})`],
      });

      layers.push({
        id: `decoder-${i}-masked-attention`,
        name: `Decoder Layer ${i + 1}: Masked Self-Attention`,
        type: 'attention' as LayerType,
        description: `${nhead} heads with causal mask`,
        shape: ['batch', 'seq_len', `d_model (${d_model})`],
        config: { nhead, head_dim: Math.floor(d_model / nhead), masked: true },
      });

      layers.push({
        id: `decoder-${i}-cross-attention`,
        name: `Decoder Layer ${i + 1}: Cross-Attention`,
        type: 'attention' as LayerType,
        description: `Attend to encoder output`,
        shape: ['batch', 'seq_len', `d_model (${d_model})`],
        config: { nhead, head_dim: Math.floor(d_model / nhead) },
      });

      layers.push({
        id: `decoder-${i}-norm2`,
        name: `Decoder Layer ${i + 1}: LayerNorm 2`,
        type: 'norm' as LayerType,
        description: 'Post-attention normalization',
        shape: ['batch', 'seq_len', `d_model (${d_model})`],
      });

      layers.push({
        id: `decoder-${i}-ffn`,
        name: `Decoder Layer ${i + 1}: Feed Forward`,
        type: 'feedforward' as LayerType,
        description: `Two linear transformations with ${config.activation}`,
        shape: ['batch', 'seq_len', `d_model (${d_model})`, `→ ${dim_feedforward}`, `→ d_model (${d_model})`],
        config: { hidden: dim_feedforward, activation: config.activation },
      });
    }
  }

  layers.push({
    id: 'final-norm',
    name: 'Final LayerNorm',
    type: 'norm' as LayerType,
    description: 'Final normalization before output',
    shape: ['batch', 'seq_len', `d_model (${d_model})`],
  });

  layers.push({
    id: 'output',
    name: 'Output Projection',
    type: 'output' as LayerType,
    description: 'Project to vocabulary',
    shape: ['batch', 'seq_len', `vocab_size (${config.vocab_size})`],
    config: { vocab_size: config.vocab_size },
  });

  return layers;
};

const getLayerColor = (type: LayerType): string => {
  const colors = {
    embedding: 'bg-blue-500 border-blue-500 text-blue-500',
    positional: 'bg-purple-500 border-purple-500 text-purple-500',
    attention: 'bg-green-500 border-green-500 text-green-500',
    feedforward: 'bg-orange-500 border-orange-500 text-orange-500',
    norm: 'bg-pink-500 border-pink-500 text-pink-500',
    output: 'bg-red-500 border-red-500 text-red-500',
  };
  return colors[type] || 'bg-gray-500';
};

export const TransformerStructure: React.FC<TransformerStructureProps> = ({ className }) => {
  const { config, output, animationState, setIsPlaying } = useTransformerStore();
  const { isPlaying, speed } = animationState;
  const [currentLayer, setCurrentLayer] = useState(0);
  const [autoPlay, setAutoPlay] = useState(false);
  const [selectedLayer, setSelectedLayer] = useState<string | null>(null);
  const [showDetails] = useState(true);

  const { num_encoder_layers, num_decoder_layers } = config;
  const hasDecoder = config.num_decoder_layers > 0;
  const layers = useMemo(() => getTransformerLayers(config, hasDecoder), [config, hasDecoder]);

  // Auto-play animation
  useEffect(() => {
    if (!autoPlay || !isPlaying) return;

    const interval = setInterval(() => {
      setCurrentLayer((prev) => {
        const next = (prev + 1) % layers.length;
        return next;
      });
    }, speed);

    return () => clearInterval(interval);
  }, [autoPlay, isPlaying, layers.length, speed]);

  // Get current layer data
  const getCurrentLayerData = () => {
    if (!output?.data) return null;

    const layerId = selectedLayer || layers[currentLayer]?.id;

    if (layerId?.includes('attention') && output.data?.attention_weights) {
      return {
        type: 'attention',
        data: output.data.attention_weights,
        description: 'Attention weights for all heads and layers',
      };
    }

    if (layerId?.includes('embedding') && output.data?.embeddings) {
      return {
        type: 'embedding',
        data: output.data.embeddings,
        description: 'Token and positional embeddings',
      };
    }

    if (layerId?.includes('ffn') && output.data?.layer_outputs) {
      return {
        type: 'ffn',
        data: output.data.layer_outputs,
        description: 'Feed-forward network outputs',
      };
    }

    return null;
  };

  const currentData = getCurrentLayerData();

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Box className="h-5 w-5 text-primary" />
              Transformer Architecture
            </CardTitle>
            <CardDescription>
              {layers.length} layers | {config.d_model} dimensions | {config.nhead} heads
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="icon"
              onClick={() => {
                setCurrentLayer(0);
                setSelectedLayer(null);
              }}
              title="Reset to first layer"
            >
              <RotateCw className="h-4 w-4" />
            </Button>
            <Button
              size="icon"
              variant={isPlaying && autoPlay ? 'destructive' : 'default'}
              onClick={() => {
                if (isPlaying && autoPlay) {
                  setAutoPlay(false);
                  setIsPlaying(false);
                } else {
                  setAutoPlay(true);
                  setIsPlaying(true);
                }
              }}
              title={autoPlay ? 'Pause auto-play' : 'Start auto-play'}
            >
              {isPlaying && autoPlay ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Layer navigation */}
        <div className="space-y-3">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Layer Navigation</span>
            <Badge variant="secondary">
              {currentLayer + 1} / {layers.length}
            </Badge>
          </div>

          {/* Layer list */}
          <div className="flex gap-1 overflow-x-auto pb-2">
            {layers.map((layer, index) => (
              <button
                key={layer.id}
                onClick={() => {
                  setCurrentLayer(index);
                  setSelectedLayer(layer.id);
                }}
                className={`flex-shrink-0 px-3 py-2 rounded-lg border-2 text-xs font-medium transition-all ${
                  currentLayer === index || selectedLayer === layer.id
                    ? getLayerColor(layer.type) + ' scale-105'
                    : 'bg-muted border-border hover:bg-muted/80'
                }`}
              >
                <div className="flex items-center gap-1">
                  <div className={`w-2 h-2 rounded ${
                    layer.type === 'embedding' ? 'bg-blue-500' :
                    layer.type === 'positional' ? 'bg-purple-500' :
                    layer.type === 'attention' ? 'bg-green-500' :
                    layer.type === 'feedforward' ? 'bg-orange-500' :
                    layer.type === 'norm' ? 'bg-pink-500' :
                    layer.type === 'output' ? 'bg-red-500' :
                    'bg-gray-500'
                  }`} />
                  <span className="whitespace-nowrap">{layer.name}</span>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Current layer details */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <Eye className="h-4 w-4 text-primary" />
              {layers[currentLayer]?.name}
            </h4>
            <Badge variant="outline" className="text-xs">
              {layers[currentLayer]?.id}
            </Badge>
          </div>

          {showDetails && (
            <div className="p-4 bg-muted/50 rounded-lg space-y-3 animate-in fade-in">
              {/* Description */}
              <p className="text-xs text-muted-foreground">
                {layers[currentLayer]?.description}
              </p>

              {/* Shape */}
              {layers[currentLayer]?.shape && (
                <div className="flex items-center gap-2 text-xs">
                  <span className="text-muted-foreground">Output shape:</span>
                  <code className="px-2 py-1 bg-background rounded">
                    {layers[currentLayer]?.shape.join(' → ')}
                  </code>
                </div>
              )}

              {/* Formula */}
              {layers[currentLayer]?.formula && (
                <div className="p-2 bg-background rounded">
                  <Latex className="text-sm">{layers[currentLayer]?.formula}</Latex>
                </div>
              )}

              {/* Config */}
              {layers[currentLayer]?.config && (
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {Object.entries(layers[currentLayer].config).map(([key, value]) => (
                    <div key={key} className="flex items-center justify-between">
                      <span className="text-muted-foreground">{key}:</span>
                      <Badge variant="secondary">{String(value)}</Badge>
                    </div>
                  ))}
                </div>
              )}

              {/* Real data if available */}
              {currentData && (
                <div className="pt-2 border-t">
                  <div className="text-xs font-medium text-muted-foreground mb-2">Live Data</div>
                  <div className="text-xs text-muted-foreground">
                    {currentData.type} ({Array.isArray(currentData.data) ? `${currentData.data.length} items` : 'available'})
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Architecture diagram */}
        <div className="space-y-3">
          <h4 className="text-sm font-medium flex items-center gap-2">
            <Layers className="h-4 w-4 text-primary" />
            Architecture Overview
          </h4>

          {/* Simplified architecture view */}
          <div className="p-4 bg-muted/50 rounded-lg space-y-4">
            {/* Input to Embedding */}
            <div className="flex items-center gap-2 text-sm">
              <div className="flex-shrink-0 w-20 text-center p-2 bg-gray-500/20 rounded border border-gray-500 text-xs">
                Input
              </div>
              <ArrowRight className="h-4 w-4 text-muted-foreground" />
              <div className="flex-1 text-center p-2 bg-blue-500/20 rounded border border-blue-500 text-xs">
                Embedding
              </div>
            </div>

            {/* Encoder block */}
            <div className="space-y-2">
              <div className="text-xs text-center font-medium text-blue-500">
                Encoder ({num_encoder_layers} layers)
              </div>
              <div className="space-y-1">
                {Array.from({ length: Math.min(num_encoder_layers, 3) }).map((_, i) => (
                  <div key={i} className="flex items-center gap-2 text-xs">
                    <div className="flex-shrink-0 w-16 text-center p-1 bg-green-500/20 rounded border border-green-500">
                      Attn
                    </div>
                    <ArrowRight className="h-3 w-3 text-muted-foreground flex-shrink-0" />
                    <div className="flex-shrink-0 w-16 text-center p-1 bg-orange-500/20 rounded border border-orange-500">
                      FFN
                    </div>
                    {i < num_encoder_layers - 1 && (
                      <ArrowRight className="h-3 w-3 text-muted-foreground flex-shrink-0" />
                    )}
                  </div>
                ))}
                {num_encoder_layers > 3 && (
                  <div className="text-xs text-center text-muted-foreground">
                    ... {num_encoder_layers - 3} more layers
                  </div>
                )}
              </div>
            </div>

            {/* Decoder block (if enabled) */}
            {hasDecoder && num_decoder_layers > 0 && (
              <>
                <div className="flex justify-center">
                  <ArrowDown className="h-4 w-4 text-muted-foreground" />
                </div>
                <div className="space-y-2">
                  <div className="text-xs text-center font-medium text-orange-500">
                    Decoder ({num_decoder_layers} layers)
                  </div>
                  <div className="space-y-1">
                    {Array.from({ length: Math.min(num_decoder_layers, 3) }).map((_, i) => (
                      <div key={i} className="flex items-center gap-2 text-xs">
                        <div className="flex-shrink-0 w-12 text-center p-1 bg-purple-500/20 rounded border border-purple-500 text-[10px]">
                          Masked
                        </div>
                        <ArrowRight className="h-3 w-3 text-muted-foreground flex-shrink-0" />
                        <div className="flex-shrink-0 w-12 text-center p-1 bg-pink-500/20 rounded border border-pink-500 text-[10px]">
                          Cross
                        </div>
                        <ArrowRight className="h-3 w-3 text-muted-foreground flex-shrink-0" />
                        <div className="flex-shrink-0 w-16 text-center p-1 bg-orange-500/20 rounded border border-orange-500">
                          FFN
                        </div>
                        {i < num_decoder_layers - 1 && (
                          <ArrowRight className="h-3 w-3 text-muted-foreground flex-shrink-0" />
                        )}
                      </div>
                    ))}
                    {num_decoder_layers > 3 && (
                      <div className="text-xs text-center text-muted-foreground">
                        ... {num_decoder_layers - 3} more layers
                      </div>
                    )}
                  </div>
                </div>
              </>
            )}

            {/* To Output */}
            <div className="flex items-center gap-2 text-sm">
              <ArrowRight className="h-4 w-4 text-muted-foreground" />
              <div className="flex-1 text-center p-2 bg-red-500/20 rounded border border-red-500 text-xs">
                Output
              </div>
            </div>
          </div>
        </div>

        {/* Animation speed control */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Animation Speed</span>
            <Badge variant="secondary">{speed}ms</Badge>
          </div>
          <input
            type="range"
            min={100}
            max={3000}
            step={100}
            value={speed}
            onChange={(e) => useTransformerStore.getState().setAnimationSpeed(parseInt(e.target.value))}
            className="w-full h-2 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary"
          />
        </div>

        {/* Stats summary */}
        <div className="grid grid-cols-4 gap-2 text-xs">
          <div className="p-2 bg-blue-500/10 rounded">
            <div className="font-medium text-blue-500">Model Dim</div>
            <div className="font-mono">{config.d_model}</div>
          </div>
          <div className="p-2 bg-green-500/10 rounded">
            <div className="font-medium text-green-500">Heads</div>
            <div className="font-mono">{config.nhead}</div>
          </div>
          <div className="p-2 bg-orange-500/10 rounded">
            <div className="font-medium text-orange-500">FFN Dim</div>
            <div className="font-mono">{config.dim_feedforward}</div>
          </div>
          <div className="p-2 bg-purple-500/10 rounded">
            <div className="font-medium text-purple-500">Layers</div>
            <div className="font-mono">{config.num_encoder_layers + config.num_decoder_layers}</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
