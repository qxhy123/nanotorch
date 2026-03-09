/**
 * ModelProfiler Component
 *
 * Performance analysis and profiling for Transformer models:
 * - FLOPs computation and visualization
 * - Memory usage estimation
 * - Layer-wise timing breakdown
 * - Parameter count analysis
 * - Batch size impact analysis
 * - Sequence length scaling
 * - Comparison across model sizes
 * - Bottleneck identification
 */

import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../ui/card';
import { Button } from '../../ui/button';
import { Slider } from '../../ui/slider';
import {
  Activity,
  Cpu,
  MemoryStick,
  Zap,
  Gauge,
  TrendingUp,
  Database,
  Layers,
  Play,
  Settings,
  BarChart3,
  Clock,
  AlertTriangle,
  CheckCircle,
  Info,
} from 'lucide-react';

// Layer types
type LayerType = 'embedding' | 'attention' | 'ffn' | 'layer_norm' | 'output';

// Profile data for a single layer
interface LayerProfile {
  name: string;
  type: LayerType;
  parameters: number;
  flops: number;
  memoryMB: number;
  timeMs: number;
  percentage: number;
}

// Model configuration
interface ModelConfig {
  d_model: number;
  nhead: number;
  num_encoder_layers: number;
  num_decoder_layers: number;
  dim_feedforward: number;
  vocab_size: number;
  max_seq_length: number;
}

// Default model configurations
const MODEL_PRESETS: Record<string, ModelConfig> = {
  tiny: {
    d_model: 128,
    nhead: 4,
    num_encoder_layers: 3,
    num_decoder_layers: 0,
    dim_feedforward: 512,
    vocab_size: 10000,
    max_seq_length: 128,
  },
  small: {
    d_model: 256,
    nhead: 8,
    num_encoder_layers: 6,
    num_decoder_layers: 0,
    dim_feedforward: 1024,
    vocab_size: 30000,
    max_seq_length: 256,
  },
  base: {
    d_model: 512,
    nhead: 8,
    num_encoder_layers: 6,
    num_decoder_layers: 0,
    dim_feedforward: 2048,
    vocab_size: 50000,
    max_seq_length: 512,
  },
  large: {
    d_model: 768,
    nhead: 12,
    num_encoder_layers: 12,
    num_decoder_layers: 0,
    dim_feedforward: 3072,
    vocab_size: 50000,
    max_seq_length: 512,
  },
  'x-large': {
    d_model: 1024,
    nhead: 16,
    num_encoder_layers: 24,
    num_decoder_layers: 0,
    dim_feedforward: 4096,
    vocab_size: 50000,
    max_seq_length: 512,
  },
};

export const ModelProfiler: React.FC<{ className?: string }> = ({ className = '' }) => {
  // State
  const [modelConfig, setModelConfig] = useState(MODEL_PRESETS.base);
  const [batchSize, setBatchSize] = useState(32);
  const [sequenceLength, setSequenceLength] = useState(128);
  const [selectedPreset, setSelectedPreset] = useState<keyof typeof MODEL_PRESETS>('base');
  const [showDetails, setShowDetails] = useState(false);
  const [isProfiling, setIsProfiling] = useState(false);

  // Compute FLOPs for different operations
  const computeFLOPs = useMemo(() => {
    const { d_model, nhead, num_encoder_layers, dim_feedforward, vocab_size } = modelConfig;
    const seq_len = sequenceLength;
    const batch = batchSize;
    const d_k = d_model / nhead;

    // Embedding layer FLOPs
    const embeddingFLOPs = batch * seq_len * d_model * vocab_size * 2; // lookup and add

    // Position encoding FLOPs (negligible, but included)
    const posEncFLOPs = batch * seq_len * d_model * 10;

    // Single attention head FLOPs
    const qkvFLOPs = batch * seq_len * d_model * 3 * d_model * 2; // Q, K, V projections
    const scoresFLOPs = batch * nhead * seq_len * seq_len * d_k * 2; // QK^T
    const softmaxFLOPs = batch * nhead * seq_len * seq_len * 3; // softmax
    const valueFLOPs = batch * nhead * seq_len * seq_len * d_k * 2; // Attention * V
    const outputProjFLOPs = batch * seq_len * d_model * d_model * 2; // Output projection
    const attentionFLOPs =
      qkvFLOPs + scoresFLOPs + softmaxFLOPs + valueFLOPs + outputProjFLOPs;

    // FFN FLOPs
    const ffnFLOPs =
      batch * seq_len * d_model * dim_feedforward * 2 * 2 + // Two linear layers
      batch * seq_len * d_model * 2; // Activation and bias

    // Layer norm FLOPs
    const layerNormFLOPs = batch * seq_len * d_model * 8 * 2; // 2 layer norms per layer

    // Total encoder FLOPs
    const encoderFLOPs = num_encoder_layers * (attentionFLOPs + ffnFLOPs + layerNormFLOPs);

    // Total FLOPs
    const totalFLOPs = embeddingFLOPs + posEncFLOPs + encoderFLOPs;

    return {
      embedding: embeddingFLOPs,
      posEnc: posEncFLOPs,
      attention: attentionFLOPs * num_encoder_layers,
      ffn: ffnFLOPs * num_encoder_layers,
      layerNorm: layerNormFLOPs * num_encoder_layers,
      total: totalFLOPs,
    };
  }, [modelConfig, batchSize, sequenceLength]);

  // Compute parameter counts
  const computeParameters = useMemo(() => {
    const { d_model, num_encoder_layers, dim_feedforward, vocab_size } = modelConfig;

    // Embedding parameters
    const embeddingParams = vocab_size * d_model;

    // Position encoding parameters (fixed, no learning)
    const posEncParams = 0;

    // Single attention layer parameters
    const qkvParams = 3 * d_model * d_model; // Q, K, V projections
    const outputProjParams = d_model * d_model; // Output projection
    const attentionParams = qkvParams + outputProjParams;

    // FFN parameters
    const ffnParams = d_model * dim_feedforward + dim_feedforward * d_model;

    // Layer norm parameters (2 sets per layer)
    const layerNormParams = 4 * d_model * 2; // 2 layer norms, each with scale and bias

    // Total encoder parameters
    const encoderParams = num_encoder_layers * (attentionParams + ffnParams + layerNormParams);

    // Total parameters
    const totalParams = embeddingParams + posEncParams + encoderParams;

    return {
      embedding: embeddingParams,
      posEnc: posEncParams,
      attention: attentionParams * num_encoder_layers,
      ffn: ffnParams * num_encoder_layers,
      layerNorm: layerNormParams * num_encoder_layers,
      total: totalParams,
    };
  }, [modelConfig]);

  // Estimate memory usage (MB)
  const computeMemory = useMemo(() => {
    const { d_model, num_encoder_layers, dim_feedforward } = modelConfig;
    const seq_len = sequenceLength;
    const batch = batchSize;

    // Activation memory (assuming float32)
    const bytesPerFloat = 4;

    // Input embeddings
    const inputMemory = batch * seq_len * d_model * bytesPerFloat;

    // Attention activations (Q, K, V, scores, attention, output)
    const attentionMemory =
      batch * num_encoder_layers * seq_len * d_model * 6 * bytesPerFloat;

    // FFN activations
    const ffnMemory = batch * num_encoder_layers * seq_len * dim_feedforward * 2 * bytesPerFloat;

    // Parameter memory
    const paramsMemory = computeParameters.total * bytesPerFloat;

    // Gradient memory (same as parameters)
    const gradientMemory = paramsMemory;

    // Optimizer state (2x for Adam)
    const optimizerMemory = paramsMemory * 2;

    // Total memory
    const totalMemory =
      inputMemory + attentionMemory + ffnMemory + paramsMemory + gradientMemory + optimizerMemory;

    return {
      activations: (inputMemory + attentionMemory + ffnMemory) / (1024 * 1024),
      parameters: paramsMemory / (1024 * 1024),
      gradients: gradientMemory / (1024 * 1024),
      optimizer: optimizerMemory / (1024 * 1024),
      total: totalMemory / (1024 * 1024),
    };
  }, [modelConfig, batchSize, sequenceLength, computeParameters.total]);

  // Estimate timing (ms) based on FLOPs
  const computeTiming = useMemo(() => {
    const flopsPerMs = 1000000; // Assume 1 GFLOPS for simplicity

    const embeddingTime = computeFLOPs.embedding / flopsPerMs;
    const attentionTime = computeFLOPs.attention / flopsPerMs;
    const ffnTime = computeFLOPs.ffn / flopsPerMs;
    const layerNormTime = computeFLOPs.layerNorm / flopsPerMs;
    const totalTime = embeddingTime + attentionTime + ffnTime + layerNormTime;

    const total = totalTime;
    const percentage = (val: number) => ((val / total) * 100).toFixed(1);

    return {
      embedding: { time: embeddingTime, percentage: parseFloat(percentage(embeddingTime)) },
      attention: { time: attentionTime, percentage: parseFloat(percentage(attentionTime)) },
      ffn: { time: ffnTime, percentage: parseFloat(percentage(ffnTime)) },
      layerNorm: { time: layerNormTime, percentage: parseFloat(percentage(layerNormTime)) },
      total: totalTime,
    };
  }, [computeFLOPs]);

  // Layer-wise profiles
  const layerProfiles = useMemo<LayerProfile[]>(() => {
    const profiles: LayerProfile[] = [];
    const { num_encoder_layers } = modelConfig;

    // Embedding layer
    profiles.push({
      name: 'Embedding',
      type: 'embedding',
      parameters: computeParameters.embedding,
      flops: computeFLOPs.embedding,
      memoryMB: computeMemory.activations * 0.1,
      timeMs: computeTiming.embedding.time,
      percentage: computeTiming.embedding.percentage,
    });

    // Encoder layers
    for (let i = 0; i < num_encoder_layers; i++) {
      // Attention
      profiles.push({
        name: `Encoder Layer ${i + 1} - Attention`,
        type: 'attention',
        parameters: computeParameters.attention / num_encoder_layers,
        flops: computeFLOPs.attention / num_encoder_layers,
        memoryMB: computeMemory.activations * 0.3 / num_encoder_layers,
        timeMs: computeTiming.attention.time / num_encoder_layers,
        percentage: computeTiming.attention.percentage / num_encoder_layers,
      });

      // FFN
      profiles.push({
        name: `Encoder Layer ${i + 1} - FFN`,
        type: 'ffn',
        parameters: computeParameters.ffn / num_encoder_layers,
        flops: computeFLOPs.ffn / num_encoder_layers,
        memoryMB: computeMemory.activations * 0.5 / num_encoder_layers,
        timeMs: computeTiming.ffn.time / num_encoder_layers,
        percentage: computeTiming.ffn.percentage / num_encoder_layers,
      });

      // Layer Norm
      profiles.push({
        name: `Encoder Layer ${i + 1} - LayerNorm`,
        type: 'layer_norm',
        parameters: computeParameters.layerNorm / num_encoder_layers / 2,
        flops: computeFLOPs.layerNorm / num_encoder_layers / 2,
        memoryMB: computeMemory.activations * 0.05 / num_encoder_layers,
        timeMs: computeTiming.layerNorm.time / num_encoder_layers / 2,
        percentage: computeTiming.layerNorm.percentage / num_encoder_layers / 2,
      });
    }

    return profiles.sort((a, b) => b.flops - a.flops);
  }, [
    modelConfig,
    computeParameters,
    computeFLOPs,
    computeMemory,
    computeTiming,
  ]);

  // Identify bottlenecks
  const bottlenecks = useMemo(() => {
    const results: string[] = [];
    const avgFLOPs = layerProfiles.reduce((sum, p) => sum + p.flops, 0) / layerProfiles.length;
    const avgTime = computeTiming.total / layerProfiles.length;

    layerProfiles.forEach((profile) => {
      if (profile.flops > avgFLOPs * 1.5) {
        results.push(`${profile.name} has high computational cost (${profile.percentage.toFixed(1)}% of total)`);
      }
      if (profile.timeMs > avgTime * 1.5 && profile.timeMs > 10) {
        results.push(`${profile.name} may be a timing bottleneck`);
      }
    });

    if (computeMemory.total > 8000) {
      results.push(`High memory usage: ${computeMemory.total.toFixed(0)}MB - consider gradient checkpointing`);
    }

    if (sequenceLength > 512 && modelConfig.num_encoder_layers > 12) {
      results.push('Long sequences with deep models - consider using efficient attention variants');
    }

    if (batchSize * sequenceLength * modelConfig.d_model > 10000000) {
      results.push('Large activation memory - consider reducing batch size or sequence length');
    }

    return results;
  }, [layerProfiles, computeMemory, computeTiming, modelConfig, sequenceLength, batchSize]);

  // Format numbers
  const formatNumber = (num: number): string => {
    if (num >= 1e9) return `${(num / 1e9).toFixed(2)}B`;
    if (num >= 1e6) return `${(num / 1e6).toFixed(2)}M`;
    if (num >= 1e3) return `${(num / 1e3).toFixed(2)}K`;
    return num.toFixed(0);
  };

  const formatTime = (ms: number): string => {
    if (ms >= 1000) return `${(ms / 1000).toFixed(2)}s`;
    if (ms >= 1) return `${ms.toFixed(2)}ms`;
    return `${(ms * 1000).toFixed(2)}μs`;
  };

  // Layer type colors
  const getLayerColor = (type: LayerType): string => {
    const colors = {
      embedding: 'from-blue-500 to-cyan-500',
      attention: 'from-purple-500 to-pink-500',
      ffn: 'from-green-500 to-emerald-500',
      layer_norm: 'from-yellow-500 to-orange-500',
      output: 'from-red-500 to-rose-500',
    };
    return colors[type];
  };

  // Run profiling animation
  const runProfiling = () => {
    setIsProfiling(true);
    setTimeout(() => setIsProfiling(false), 2000);
  };

  // Load preset
  const loadPreset = (preset: keyof typeof MODEL_PRESETS) => {
    setSelectedPreset(preset);
    setModelConfig(MODEL_PRESETS[preset]);
  };

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Gauge className="h-5 w-5 text-primary" />
              Model Profiler
            </CardTitle>
            <CardDescription>
              Performance analysis and computational profiling
            </CardDescription>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={runProfiling} disabled={isProfiling}>
              <Play className="h-4 w-4 mr-1" />
              {isProfiling ? 'Profiling...' : 'Profile'}
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowDetails(!showDetails)}
            >
              {showDetails ? <CheckCircle className="h-4 w-4" /> : <BarChart3 className="h-4 w-4" />}
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Model Presets */}
        <div className="flex flex-wrap gap-2">
          <span className="text-sm font-medium">Model Size:</span>
          {(Object.keys(MODEL_PRESETS) as Array<keyof typeof MODEL_PRESETS>).map((preset) => (
            <Button
              key={preset}
              variant={selectedPreset === preset ? 'default' : 'outline'}
              size="sm"
              onClick={() => loadPreset(preset)}
            >
              {preset.charAt(0).toUpperCase() + preset.slice(1)}
            </Button>
          ))}
        </div>

        {/* Input Parameters */}
        <div className="space-y-4 p-4 bg-muted rounded-lg">
          <h3 className="text-sm font-medium flex items-center gap-2">
            <Settings className="h-4 w-4" />
            Input Parameters
          </h3>

          <div className="grid md:grid-cols-2 gap-4">
            {/* Batch Size */}
            <div className="space-y-2">
              <div className="flex items-center gap-4">
                <label className="text-sm font-medium min-w-[100px]">Batch Size:</label>
                <Slider
                  value={[batchSize]}
                  onValueChange={([v]) => setBatchSize(v)}
                  min={1}
                  max={128}
                  step={1}
                  className="flex-1"
                />
                <span className="text-sm font-mono w-12 text-right">{batchSize}</span>
              </div>
            </div>

            {/* Sequence Length */}
            <div className="space-y-2">
              <div className="flex items-center gap-4">
                <label className="text-sm font-medium min-w-[100px]">Seq Length:</label>
                <Slider
                  value={[sequenceLength]}
                  onValueChange={([v]) => setSequenceLength(v)}
                  min={32}
                  max={1024}
                  step={32}
                  className="flex-1"
                />
                <span className="text-sm font-mono w-12 text-right">{sequenceLength}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Overview Statistics */}
        <div className="grid md:grid-cols-4 gap-4">
          <div className="p-4 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <Database className="h-4 w-4 text-blue-600 dark:text-blue-400" />
              <div className="text-xs text-blue-600 dark:text-blue-400">Parameters</div>
            </div>
            <div className="text-2xl font-bold">{formatNumber(computeParameters.total)}</div>
            <div className="text-xs text-muted-foreground">Total parameters</div>
          </div>

          <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <Cpu className="h-4 w-4 text-purple-600 dark:text-purple-400" />
              <div className="text-xs text-purple-600 dark:text-purple-400">FLOPs</div>
            </div>
            <div className="text-2xl font-bold">{formatNumber(computeFLOPs.total)}</div>
            <div className="text-xs text-muted-foreground">Per forward pass</div>
          </div>

          <div className="p-4 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <MemoryStick className="h-4 w-4 text-green-600 dark:text-green-400" />
              <div className="text-xs text-green-600 dark:text-green-400">Memory</div>
            </div>
            <div className="text-2xl font-bold">{computeMemory.total.toFixed(0)}MB</div>
            <div className="text-xs text-muted-foreground">Estimated total</div>
          </div>

          <div className="p-4 bg-gradient-to-br from-orange-50 to-orange-100 dark:from-orange-900/20 dark:to-orange-800/20 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <Clock className="h-4 w-4 text-orange-600 dark:text-orange-400" />
              <div className="text-xs text-orange-600 dark:text-orange-400">Time</div>
            </div>
            <div className="text-2xl font-bold">{formatTime(computeTiming.total)}</div>
            <div className="text-xs text-muted-foreground">Est. forward pass</div>
          </div>
        </div>

        {/* FLOPs Breakdown */}
        <div className="space-y-3">
          <h3 className="text-sm font-medium flex items-center gap-2">
            <Zap className="h-4 w-4" />
            FLOPs Breakdown by Component
          </h3>
          <div className="space-y-2">
            {[
              { name: 'Embedding', value: computeFLOPs.embedding, color: 'from-blue-500 to-cyan-500' },
              { name: 'Attention', value: computeFLOPs.attention, color: 'from-purple-500 to-pink-500' },
              { name: 'Feed-Forward', value: computeFLOPs.ffn, color: 'from-green-500 to-emerald-500' },
              { name: 'Layer Norm', value: computeFLOPs.layerNorm, color: 'from-yellow-500 to-orange-500' },
            ].map((item) => {
              const percentage = ((item.value / computeFLOPs.total) * 100).toFixed(1);
              return (
                <div key={item.name} className="space-y-1">
                  <div className="flex items-center justify-between text-sm">
                    <span>{item.name}</span>
                    <span className="text-muted-foreground">
                      {formatNumber(item.value)} ({percentage}%)
                    </span>
                  </div>
                  <div className="h-3 bg-muted rounded-full overflow-hidden">
                    <div
                      className={`h-full bg-gradient-to-r ${item.color} transition-all duration-500`}
                      style={{ width: `${percentage}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Parameters Breakdown */}
        <div className="space-y-3">
          <h3 className="text-sm font-medium flex items-center gap-2">
            <Layers className="h-4 w-4" />
            Parameters Breakdown
          </h3>
          <div className="grid md:grid-cols-4 gap-3">
            {[
              { name: 'Embedding', value: computeParameters.embedding, icon: Database },
              { name: 'Attention', value: computeParameters.attention, icon: Activity },
              { name: 'Feed-Forward', value: computeParameters.ffn, icon: TrendingUp },
              { name: 'Layer Norm', value: computeParameters.layerNorm, icon: CheckCircle },
            ].map((item) => (
              <div key={item.name} className="p-3 bg-muted rounded-lg">
                <div className="flex items-center gap-2 mb-1">
                  <item.icon className="h-3 w-3 text-primary" />
                  <div className="text-xs text-muted-foreground">{item.name}</div>
                </div>
                <div className="text-lg font-bold">{formatNumber(item.value)}</div>
                <div className="text-xs text-muted-foreground">
                  {((item.value / computeParameters.total) * 100).toFixed(1)}%
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Memory Breakdown */}
        <div className="space-y-3">
          <h3 className="text-sm font-medium flex items-center gap-2">
            <MemoryStick className="h-4 w-4" />
            Memory Usage Breakdown
          </h3>
          <div className="grid md:grid-cols-4 gap-3">
            {[
              { name: 'Activations', value: computeMemory.activations, color: 'blue' },
              { name: 'Parameters', value: computeMemory.parameters, color: 'purple' },
              { name: 'Gradients', value: computeMemory.gradients, color: 'green' },
              { name: 'Optimizer', value: computeMemory.optimizer, color: 'orange' },
            ].map((item) => (
              <div key={item.name} className="p-3 bg-muted rounded-lg">
                <div className="text-xs text-muted-foreground mb-1">{item.name}</div>
                <div className="text-lg font-bold">{item.value.toFixed(0)}MB</div>
                <div className="text-xs text-muted-foreground">
                  {((item.value / computeMemory.total) * 100).toFixed(1)}%
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Layer-wise Details */}
        {showDetails && (
          <div className="space-y-3">
            <h3 className="text-sm font-medium flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              Layer-wise Profile
            </h3>
            <div className="space-y-2 max-h-[400px] overflow-y-auto">
              {layerProfiles.map((profile, idx) => (
                <div
                  key={`${profile.name}-${idx}`}
                  className="p-3 bg-muted rounded-lg space-y-2"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">{profile.name}</span>
                    <span className="text-xs text-muted-foreground">
                      {profile.percentage.toFixed(1)}% of total
                    </span>
                  </div>
                  <div className="grid grid-cols-4 gap-2 text-xs">
                    <div>
                      <div className="text-muted-foreground">Params</div>
                      <div className="font-mono">{formatNumber(profile.parameters)}</div>
                    </div>
                    <div>
                      <div className="text-muted-foreground">FLOPs</div>
                      <div className="font-mono">{formatNumber(profile.flops)}</div>
                    </div>
                    <div>
                      <div className="text-muted-foreground">Memory</div>
                      <div className="font-mono">{profile.memoryMB.toFixed(2)}MB</div>
                    </div>
                    <div>
                      <div className="text-muted-foreground">Time</div>
                      <div className="font-mono">{formatTime(profile.timeMs)}</div>
                    </div>
                  </div>
                  <div className="h-2 bg-background rounded-full overflow-hidden">
                    <div
                      className={`h-full bg-gradient-to-r ${getLayerColor(profile.type)}`}
                      style={{ width: `${profile.percentage}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Bottlenecks and Warnings */}
        {bottlenecks.length > 0 && (
          <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
            <h3 className="text-sm font-medium mb-2 flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-yellow-600 dark:text-yellow-400" />
              Performance Bottlenecks & Recommendations
            </h3>
            <ul className="space-y-1 text-xs text-yellow-800 dark:text-yellow-200">
              {bottlenecks.map((bottleneck, idx) => (
                <li key={idx}>• {bottleneck}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Scaling Analysis */}
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
          <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
            <TrendingUp className="h-4 w-4 text-blue-600 dark:text-blue-400" />
            Scaling Analysis
          </h3>
          <div className="grid md:grid-cols-3 gap-4 text-xs text-blue-800 dark:text-blue-200">
            <div>
              <div className="font-medium mb-1">Sequence Length Scaling</div>
              <ul className="space-y-1 text-blue-700 dark:text-blue-300">
                <li>• FLOPs scale as O(n²) for attention</li>
                <li>• Memory scales as O(n²) for attention maps</li>
                <li>• Consider gradient checkpointing for long sequences</li>
              </ul>
            </div>
            <div>
              <div className="font-medium mb-1">Batch Size Scaling</div>
              <ul className="space-y-1 text-blue-700 dark:text-blue-300">
                <li>• Linear scaling for FLOPs and memory</li>
                <li>• Larger batches = better GPU utilization</li>
                <li>• Trade-off: batch size vs sequence length</li>
              </ul>
            </div>
            <div>
              <div className="font-medium mb-1">Model Depth Scaling</div>
              <ul className="space-y-1 text-blue-700 dark:text-blue-300">
                <li>• Linear scaling for FLOPs and parameters</li>
                <li>• Deeper models = more sequential computation</li>
                <li>• Consider model parallelism for very deep models</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Info */}
        <div className="p-4 bg-muted rounded-lg">
          <div className="flex items-start gap-2 text-xs text-muted-foreground">
            <Info className="h-4 w-4 mt-0.5 flex-shrink-0" />
            <div>
              <strong>Note:</strong> These are estimated values based on theoretical computations.
              Actual performance may vary depending on hardware, implementation details,
              optimization level, and other factors. Use this profiler as a guideline for
              understanding relative costs and identifying potential bottlenecks.
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
