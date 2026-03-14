/**
 * LayerVisualization Component
 *
 * Main component for visualizing Encoder/Decoder Layer computation.
 * Displays step-by-step computation with interactive navigation.
 */

import React, { useState, useCallback, useEffect, useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../ui/card';
import { Badge } from '../../ui/badge';
import { Button } from '../../ui/button';
import { useTransformerStore } from '../../../stores/transformerStore';
import { layerApi } from '../../../services/layerApi';
import type {
  LayerType,
  EncoderLayerResult,
  DecoderLayerResult,
  LayerStep,
  SublayerComputation,
} from '../../../types/layer';
import { ChevronLeft, ChevronRight, Play, Pause, RotateCw, BarChart3 } from 'lucide-react';
import { InputStep } from './steps/InputStep';
import { LayerNormStep } from './steps/LayerNormStep';
import { AttentionStep } from './steps/AttentionStep';
import { FeedForwardStep } from './steps/FeedForwardStep';
import { ResidualStep } from './steps/ResidualStep';
import { OutputStep } from './steps/OutputStep';
import { LayerStatistics } from './LayerStatistics';

interface LayerVisualizationProps {
  layerType: LayerType;
  layerIndex?: number;
  className?: string;
}

type LayerComputationResult = EncoderLayerResult | DecoderLayerResult;

function getSublayerData(
  result: LayerComputationResult,
  sublayerIndex: number
): SublayerComputation {
  if (sublayerIndex <= 0) {
    return result.sublayer1;
  }

  if (sublayerIndex === 1) {
    return result.sublayer2;
  }

  if ('sublayer3' in result) {
    return result.sublayer3;
  }

  return result.sublayer2;
}

export const LayerVisualization: React.FC<LayerVisualizationProps> = ({
  layerType,
  layerIndex = 0,
  className = '',
}) => {
  const { config } = useTransformerStore();

  // State
  const [computationResult, setComputationResult] = useState<LayerComputationResult | null>(null);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(2000);
  const [isComputing, setIsComputing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Generate steps based on layer type and norm_first setting
  const steps = useMemo((): LayerStep[] => {
    const normFirst = config.norm_first;
    const baseSteps: LayerStep[] = [
      { id: 'input', name: 'input', title: 'Input', description: 'Layer input tensor', index: 0, isActive: false, isCompleted: false },
    ];

    if (normFirst) {
      // Pre-Norm architecture
      baseSteps.push(
        { id: 'norm1', name: 'norm', title: 'LayerNorm 1', description: 'Normalize before self-attention', index: 1, isActive: false, isCompleted: false, sublayer: 0 },
        { id: 'attn', name: 'attention', title: 'Self-Attention', description: 'Multi-head self-attention mechanism', index: 2, isActive: false, isCompleted: false, sublayer: 0 },
        { id: 'res1', name: 'residual', title: 'Residual 1', description: 'Add residual connection', index: 3, isActive: false, isCompleted: false, sublayer: 0 },
      );

      if (layerType === 'decoder') {
        baseSteps.push(
          { id: 'norm2', name: 'norm', title: 'LayerNorm 2', description: 'Normalize before cross-attention', index: 4, isActive: false, isCompleted: false, sublayer: 1 },
          { id: 'cross_attn', name: 'attention', title: 'Cross-Attention', description: 'Cross-attention with encoder output', index: 5, isActive: false, isCompleted: false, sublayer: 1 },
          { id: 'res2', name: 'residual', title: 'Residual 2', description: 'Add residual connection', index: 6, isActive: false, isCompleted: false, sublayer: 1 },
        );
      } else {
        baseSteps.push(
          { id: 'norm2', name: 'norm', title: 'LayerNorm 2', description: 'Normalize before feed-forward', index: 4, isActive: false, isCompleted: false, sublayer: 1 },
        );
      }

      baseSteps.push(
        { id: 'ffn', name: 'feedforward', title: 'Feed-Forward', description: 'Position-wise feed-forward network', index: baseSteps.length, isActive: false, isCompleted: false, sublayer: layerType === 'decoder' ? 2 : 1 },
        { id: 'res3', name: 'residual', title: 'Residual 3', description: 'Add residual connection', index: baseSteps.length + 1, isActive: false, isCompleted: false, sublayer: layerType === 'decoder' ? 2 : 1 },
      );
    } else {
      // Post-Norm architecture
      baseSteps.push(
        { id: 'attn', name: 'attention', title: 'Self-Attention', description: 'Multi-head self-attention mechanism', index: 1, isActive: false, isCompleted: false, sublayer: 0 },
        { id: 'res1', name: 'residual', title: 'Residual 1', description: 'Add residual connection', index: 2, isActive: false, isCompleted: false, sublayer: 0 },
        { id: 'norm1', name: 'norm', title: 'LayerNorm 1', description: 'Normalize after residual', index: 3, isActive: false, isCompleted: false, sublayer: 0 },
      );

      if (layerType === 'decoder') {
        baseSteps.push(
          { id: 'cross_attn', name: 'attention', title: 'Cross-Attention', description: 'Cross-attention with encoder output', index: 4, isActive: false, isCompleted: false, sublayer: 1 },
          { id: 'res2', name: 'residual', title: 'Residual 2', description: 'Add residual connection', index: 5, isActive: false, isCompleted: false, sublayer: 1 },
          { id: 'norm2', name: 'norm', title: 'LayerNorm 2', description: 'Normalize after residual', index: 6, isActive: false, isCompleted: false, sublayer: 1 },
        );
      }

      baseSteps.push(
        { id: 'ffn', name: 'feedforward', title: 'Feed-Forward', description: 'Position-wise feed-forward network', index: baseSteps.length, isActive: false, isCompleted: false, sublayer: layerType === 'decoder' ? 2 : 1 },
        { id: 'res3', name: 'residual', title: 'Residual 3', description: 'Add residual connection', index: baseSteps.length + 1, isActive: false, isCompleted: false, sublayer: layerType === 'decoder' ? 2 : 1 },
        { id: 'norm3', name: 'norm', title: 'LayerNorm 3', description: 'Normalize after residual', index: baseSteps.length + 2, isActive: false, isCompleted: false, sublayer: layerType === 'decoder' ? 2 : 1 },
      );
    }

    baseSteps.push(
      { id: 'output', name: 'output', title: 'Output', description: 'Layer output tensor', index: baseSteps.length, isActive: false, isCompleted: false }
    );

    // Update active state
    return baseSteps.map((step, i) => ({
      ...step,
      isActive: i === currentStepIndex,
      isCompleted: i < currentStepIndex,
    }));
  }, [layerType, config.norm_first, currentStepIndex]);

  // Compute layer
  const computeLayer = useCallback(async () => {
    setIsComputing(true);
    setError(null);

    try {
      // Create sample input
      const seqLen = Math.min(8, config.max_seq_len);
      const dModel = config.d_model;
      const inputData = layerApi.createSampleInput(seqLen, dModel, 42);

      let result: EncoderLayerResult | DecoderLayerResult;

      if (layerType === 'encoder') {
        result = await layerApi.computeEncoderLayer(config, inputData);
      } else {
        // For decoder, create encoder output as well
        const encoderOutput = layerApi.createSampleInput(seqLen, dModel, 43);
        result = await layerApi.computeDecoderLayer(config, inputData, encoderOutput);
      }

      if (result.success) {
        setComputationResult(result);
        setCurrentStepIndex(0);
      } else {
        setError(result.error || 'Computation failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setIsComputing(false);
    }
  }, [layerType, config]);

  // Auto-play functionality
  useEffect(() => {
    if (isPlaying && currentStepIndex < steps.length - 1) {
      const timer = setTimeout(() => {
        setCurrentStepIndex(prev => prev + 1);
      }, playbackSpeed);
      return () => clearTimeout(timer);
    } else if (isPlaying && currentStepIndex >= steps.length - 1) {
      setIsPlaying(false);
    }
  }, [isPlaying, currentStepIndex, steps.length, playbackSpeed]);

  // Initial computation
  useEffect(() => {
    computeLayer();
  }, [computeLayer]);

  const handlePlayPause = () => {
    if (currentStepIndex >= steps.length - 1) {
      setCurrentStepIndex(0);
    }
    setIsPlaying(!isPlaying);
  };

  const handleReset = () => {
    setIsPlaying(false);
    setCurrentStepIndex(0);
  };

  const handleStepChange = (newIndex: number) => {
    setCurrentStepIndex(newIndex);
  };

  const currentStep = steps[currentStepIndex];

  // Render current step content
  const renderStepContent = () => {
    if (!computationResult || !currentStep) {
      return (
        <div className="flex items-center justify-center py-12 text-gray-500">
          {isComputing ? 'Computing...' : 'No computation result'}
        </div>
      );
    }

    switch (currentStep.name) {
      case 'input':
        return <InputStep data={computationResult.input} config={computationResult.config} />;
      case 'norm': {
        const sublayerIndex = currentStep.sublayer ?? 0;
        const sublayerData = getSublayerData(computationResult, sublayerIndex);

        return (
          <LayerNormStep
            normInput={sublayerData.norm_input}
            normOutput={sublayerData.norm_output}
            config={computationResult.config}
          />
        );
      }
      case 'attention': {
        const sublayerIndex = currentStep.sublayer ?? 0;
        const sublayerData = getSublayerData(computationResult, sublayerIndex);

        return (
          <AttentionStep
            attentionData={sublayerData.attention}
            config={computationResult.config}
          />
        );
      }
      case 'residual': {
        const sublayerIndex = currentStep.sublayer ?? 0;
        const sublayerData = getSublayerData(computationResult, sublayerIndex);

        return (
          <ResidualStep
            residualData={sublayerData}
            stepIndex={sublayerIndex}
          />
        );
      }
      case 'feedforward':
        return (
          <FeedForwardStep
            ffnData={
              'sublayer3' in computationResult
                ? (computationResult as DecoderLayerResult).sublayer3
                : computationResult.sublayer2
            }
            config={computationResult.config}
          />
        );
      case 'output':
        return <OutputStep data={computationResult.output} config={computationResult.config} />;
      default:
        return <div className="text-gray-500">Unknown step: {currentStep.name}</div>;
    }
  };

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-primary" />
              {layerType === 'encoder' ? 'Encoder' : 'Decoder'} Layer Visualization
            </CardTitle>
            <CardDescription>
              Layer {layerIndex + 1} • {config.norm_first ? 'Pre-Norm' : 'Post-Norm'} Architecture
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={computeLayer} disabled={isComputing}>
              <RotateCw className={`h-4 w-4 mr-1 ${isComputing ? 'animate-spin' : ''}`} />
              Recompute
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Error Display */}
        {error && (
          <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
            <p className="text-sm text-red-800 dark:text-red-200">{error}</p>
          </div>
        )}

        {/* Progress Bar */}
        <div className="space-y-2">
          <div className="relative h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 transition-all duration-300"
              style={{ width: `${((currentStepIndex + 1) / steps.length) * 100}%` }}
            />
          </div>
          <div className="flex justify-between text-xs text-gray-600 dark:text-gray-400">
            <span>Step {currentStepIndex + 1} of {steps.length}</span>
            <span>{Math.round(((currentStepIndex + 1) / steps.length) * 100)}%</span>
          </div>
        </div>

        {/* Step Navigation */}
        <div className="flex flex-wrap gap-2">
          {steps.map((step, index) => (
            <Button
              key={step.id}
              variant={step.isActive ? 'default' : 'outline'}
              size="sm"
              onClick={() => handleStepChange(index)}
              className="relative"
              disabled={!computationResult}
            >
              {step.isCompleted && (
                <span className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full flex items-center justify-center">
                  <span className="text-white text-xs">✓</span>
                </span>
              )}
              {step.index + 1}. {step.title}
            </Button>
          ))}
        </div>

        {/* Playback Controls */}
        <div className="flex items-center gap-2 p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <Button
            variant="outline"
            size="sm"
            onClick={handlePlayPause}
            disabled={!computationResult || (currentStepIndex >= steps.length - 1 && !isPlaying)}
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleReset}
            disabled={!computationResult}
          >
            <RotateCw className="w-4 h-4" />
          </Button>
          <div className="flex items-center gap-2 ml-4">
            <span className="text-xs text-gray-600 dark:text-gray-400">Speed:</span>
            <select
              value={playbackSpeed}
              onChange={(e) => setPlaybackSpeed(Number(e.target.value))}
              className="text-xs border rounded px-2 py-1 bg-white dark:bg-gray-700"
            >
              <option value={1000}>Fast (1s)</option>
              <option value={2000}>Normal (2s)</option>
              <option value={4000}>Slow (4s)</option>
            </select>
          </div>
          <div className="ml-auto flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => handleStepChange(Math.max(0, currentStepIndex - 1))}
              disabled={!computationResult || currentStepIndex === 0}
            >
              <ChevronLeft className="w-4 h-4 mr-1" />
              Previous
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => handleStepChange(Math.min(steps.length - 1, currentStepIndex + 1))}
              disabled={!computationResult || currentStepIndex === steps.length - 1}
            >
              Next
              <ChevronRight className="w-4 h-4 ml-1" />
            </Button>
          </div>
        </div>

        {/* Current Step Content */}
        {computationResult && currentStep && (
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <Badge className="px-3 py-1">
                Step {currentStep.index + 1}
              </Badge>
              <h3 className="text-xl font-bold">{currentStep.title}</h3>
            </div>
            <p className="text-gray-600 dark:text-gray-400">{currentStep.description}</p>
            {renderStepContent()}
          </div>
        )}

        {/* Layer Statistics */}
        {computationResult && (
          <LayerStatistics
            layerType={layerType}
            config={computationResult.config}
          />
        )}
      </CardContent>
    </Card>
  );
};
