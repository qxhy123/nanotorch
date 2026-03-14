/**
 * DropoutMaskVisualization Component
 *
 * Interactive demonstration of Dropout regularization:
 * - Visual representation of dropout masks
 * - Before/after activation comparison
 * - Interactive dropout rate control
 * - Animated mask generation
 * - Effect on gradient flow
 * - Training vs inference mode
 */

import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../ui/card';
import { Button } from '../../ui/button';
import { Slider } from '../../ui/slider';
import {
  Play,
  RotateCcw,
  Settings,
  Eye,
  EyeOff,
  Zap,
  Shield,
  Brain,
  TrendingDown,
  AlertTriangle,
} from 'lucide-react';

// Neuron data type
interface Neuron {
  id: number;
  value: number;
  isDropped: boolean;
  isHighlighted: boolean;
}

// Generate mock layer activations
function generateLayerActivations(
  size: number,
  dropoutRate: number,
  isTraining: boolean
): { beforeDropout: Neuron[]; afterDropout: Neuron[]; mask: boolean[] } {
  const beforeDropout: Neuron[] = [];
  const afterDropout: Neuron[] = [];
  const mask: boolean[] = [];

  // Generate base activations
  const baseValues = Array.from({ length: size }, (_, i) => {
    const x = (i - size / 2) / (size / 4);
    return Math.exp(-x * x) * (0.5 + Math.random() * 0.5);
  });

  // Generate mask (only in training mode)
  const keepProbability = isTraining ? 1 - dropoutRate : 1;
  const randomMask = Array.from({ length: size }, () => Math.random() < keepProbability);

  // Apply mask and scale
  for (let i = 0; i < size; i++) {
    const isDropped = isTraining && !randomMask[i];
    const scale = isTraining ? 1 / keepProbability : 1;

    beforeDropout.push({
      id: i,
      value: baseValues[i],
      isDropped: false,
      isHighlighted: false,
    });

    afterDropout.push({
      id: i,
      value: isDropped ? 0 : baseValues[i] * scale,
      isDropped,
      isHighlighted: false,
    });

    mask.push(!isDropped);
  }

  return { beforeDropout, afterDropout, mask };
}

interface DropoutMaskVisualizationProps {
  className?: string;
}

interface NeuronDisplayProps {
  neurons: Neuron[];
  title: string;
  showMask?: boolean;
}

const NeuronDisplay: React.FC<NeuronDisplayProps> = ({
  neurons,
  title,
  showMask = true,
}) => {
  const maxValue = Math.max(...neurons.map((neuron) => neuron.value));
  const displayedNeurons = showMask ? neurons : neurons.filter((neuron) => !neuron.isDropped);

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-medium">{title}</h3>
      <div className="flex flex-wrap gap-2 justify-center min-h-[100px] p-2 bg-muted/30 rounded-lg">
        {displayedNeurons.map((neuron) => {
          const isActive = !neuron.isDropped;
          const opacity = neuron.value / maxValue;
          const size = 30 + opacity * 20;

          return (
            <div
              key={neuron.id}
              className="relative flex items-center justify-center rounded-full transition-all duration-300"
              style={{
                width: `${size}px`,
                height: `${size}px`,
                backgroundColor: isActive
                  ? `rgba(59, 130, 246, ${opacity})`
                  : 'rgba(239, 68, 68, 0.3)',
                border: neuron.isDropped ? '2px dashed #ef4444' : '2px solid #3b82f6',
                opacity: neuron.isDropped ? 0.5 : 1,
              }}
              title={`Neuron ${neuron.id}: ${neuron.isDropped ? 'DROPPED' : 'Active'} (${neuron.value.toFixed(3)})`}
            >
              {neuron.isDropped && showMask && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="w-full h-px bg-red-500 rotate-45" />
                  <div className="w-full h-px bg-red-500 -rotate-45" />
                </div>
              )}
              {!neuron.isDropped && (
                <span className="text-xs font-medium text-white">
                  {neuron.value.toFixed(2)}
                </span>
              )}
            </div>
          );
        })}
        {!showMask && neurons.filter((neuron) => neuron.isDropped).length > 0 && (
          <div className="w-full text-center text-xs text-muted-foreground">
            <span className="inline-flex items-center gap-1">
              <EyeOff className="h-3 w-3" />
              {neurons.filter((neuron) => neuron.isDropped).length} dropped neurons hidden
            </span>
          </div>
        )}
      </div>
      <div className="text-xs text-center text-muted-foreground">
        {showMask ? (
          <>
            Showing all {neurons.length} neurons ({neurons.filter((neuron) => !neuron.isDropped).length} active, {neurons.filter((neuron) => neuron.isDropped).length} dropped)
          </>
        ) : (
          <>Showing {displayedNeurons.length} active neurons only</>
        )}
      </div>
    </div>
  );
};

export const DropoutMaskVisualization: React.FC<DropoutMaskVisualizationProps> = ({
  className = '',
}) => {
  const [layerSize, setLayerSize] = useState(12);
  const [dropoutRate, setDropoutRate] = useState(0.3);
  const [isTraining, setIsTraining] = useState(true);
  const [isAnimating, setIsAnimating] = useState(false);
  const [showMask, setShowMask] = useState(true);
  const [animationStep, setAnimationStep] = useState(0);

  // Generate activations
  const { beforeDropout, afterDropout } = useMemo(() => {
    return generateLayerActivations(layerSize, dropoutRate, isTraining);
  }, [layerSize, dropoutRate, isTraining]);

  // Calculate statistics
  const stats = useMemo(() => {
    const activeCount = afterDropout.filter(n => !n.isDropped).length;
    const droppedCount = afterDropout.filter(n => n.isDropped).length;
    const beforeMean = beforeDropout.reduce((sum, n) => sum + n.value, 0) / beforeDropout.length;
    const afterMean = afterDropout.reduce((sum, n) => sum + n.value, 0) / afterDropout.length;

    return {
      activeCount,
      droppedCount,
      activePercentage: (activeCount / layerSize) * 100,
      beforeMean,
      afterMean,
      expectedDropout: dropoutRate * 100,
    };
  }, [beforeDropout, afterDropout, layerSize, dropoutRate]);

  // Run animation
  const runAnimation = () => {
    setIsAnimating(true);
    setAnimationStep(0);

    setTimeout(() => setAnimationStep(1), 500);
    setTimeout(() => setAnimationStep(2), 1200);
    setTimeout(() => setAnimationStep(3), 1900);
    setTimeout(() => {
      setIsAnimating(false);
      setAnimationStep(0);
    }, 2500);
  };

  // Reset
  const reset = () => {
    setIsAnimating(false);
    setAnimationStep(0);
  };

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Shield className="h-5 w-5 text-primary" />
              Dropout Mask Visualization
            </CardTitle>
            <CardDescription>
              Interactive demonstration of dropout regularization
            </CardDescription>
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowMask(!showMask)}
            >
              {showMask ? <EyeOff className="h-4 w-4 mr-1" /> : <Eye className="h-4 w-4 mr-1" />}
              {showMask ? 'Hide' : 'Show'} Mask
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={runAnimation}
              disabled={isAnimating}
            >
              <Play className="h-4 w-4 mr-1" />
              Animate
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={reset}
              disabled={isAnimating}
            >
              <RotateCcw className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Parameters Control */}
        <div className="space-y-4 p-4 bg-muted rounded-lg">
          <h3 className="text-sm font-medium flex items-center gap-2">
            <Settings className="h-4 w-4" />
            Dropout Parameters
          </h3>

          {/* Layer Size */}
          <div className="space-y-2">
            <div className="flex items-center gap-4">
              <label className="text-sm font-medium min-w-[100px]">Layer Size:</label>
              <Slider
                value={[layerSize]}
                onValueChange={([v]) => setLayerSize(v)}
                min={6}
                max={24}
                step={2}
                className="flex-1"
              />
              <span className="text-sm font-mono w-12 text-right">{layerSize}</span>
            </div>
            <p className="text-xs text-muted-foreground">
              Number of neurons in the layer
            </p>
          </div>

          {/* Dropout Rate */}
          <div className="space-y-2">
            <div className="flex items-center gap-4">
              <label className="text-sm font-medium min-w-[100px]">Dropout Rate:</label>
              <Slider
                value={[dropoutRate]}
                onValueChange={([v]) => setDropoutRate(v)}
                min={0}
                max={0.8}
                step={0.05}
                className="flex-1"
              />
              <span className="text-sm font-mono w-12 text-right">{(dropoutRate * 100).toFixed(0)}%</span>
            </div>
            <p className="text-xs text-muted-foreground">
              Probability of dropping each neuron during training
            </p>
          </div>

          {/* Mode Toggle */}
          <div className="flex items-center gap-4">
            <label className="text-sm font-medium min-w-[100px]">Mode:</label>
            <div className="flex gap-2">
              <Button
                variant={isTraining ? 'default' : 'outline'}
                size="sm"
                onClick={() => setIsTraining(true)}
              >
                <Brain className="h-4 w-4 mr-1" />
                Training
              </Button>
              <Button
                variant={!isTraining ? 'default' : 'outline'}
                size="sm"
                onClick={() => setIsTraining(false)}
              >
                <Eye className="h-4 w-4 mr-1" />
                Inference
              </Button>
            </div>
          </div>
        </div>

        {/* Animation Status */}
        {isAnimating && (
          <div className="p-4 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
            <div className="flex items-center gap-3">
              <Zap className="h-5 w-5 text-blue-600 dark:text-blue-400 animate-pulse" />
              <div className="flex-1">
                <div className="text-sm font-medium text-blue-800 dark:text-blue-200">
                  {animationStep === 0 && 'Initializing forward pass...'}
                  {animationStep === 1 && 'Computing activations...'}
                  {animationStep === 2 && 'Applying dropout mask...'}
                  {animationStep === 3 && 'Scaling remaining activations...'}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Statistics Cards */}
        <div className="grid md:grid-cols-4 gap-4">
          <div className="p-4 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-lg">
            <div className="text-xs text-blue-600 dark:text-blue-400 mb-1">Active Neurons</div>
            <div className="text-2xl font-bold">{stats.activeCount}</div>
            <div className="text-xs text-muted-foreground">/ {layerSize} total</div>
          </div>

          <div className="p-4 bg-gradient-to-br from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-800/20 rounded-lg">
            <div className="text-xs text-red-600 dark:text-red-400 mb-1">Dropped</div>
            <div className="text-2xl font-bold">{stats.droppedCount}</div>
            <div className="text-xs text-muted-foreground">
              {stats.activePercentage.toFixed(0)}% kept
            </div>
          </div>

          <div className="p-4 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-lg">
            <div className="text-xs text-green-600 dark:text-green-400 mb-1">Before Mean</div>
            <div className="text-2xl font-bold">{stats.beforeMean.toFixed(3)}</div>
            <div className="text-xs text-muted-foreground">avg activation</div>
          </div>

          <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 rounded-lg">
            <div className="text-xs text-purple-600 dark:text-purple-400 mb-1">After Mean</div>
            <div className="text-2xl font-bold">{stats.afterMean.toFixed(3)}</div>
            <div className="text-xs text-muted-foreground">scaled avg</div>
          </div>
        </div>

        {/* Visualization */}
        <div className="grid md:grid-cols-2 gap-6">
          <div className="p-4 bg-background rounded-lg border">
            <NeuronDisplay neurons={beforeDropout} title="Before Dropout" showMask={false} />
          </div>

          <div className="p-4 bg-background rounded-lg border">
            <NeuronDisplay neurons={afterDropout} title="After Dropout" showMask={showMask} />
          </div>
        </div>

        {/* Inverted Dropout Explanation */}
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
          <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
            <TrendingDown className="h-4 w-4 text-blue-600 dark:text-blue-400" />
            Understanding Inverted Dropout
          </h3>
          <div className="grid md:grid-cols-2 gap-4 text-xs text-blue-800 dark:text-blue-200">
            <div>
              <div className="font-medium mb-2">Why Scale During Training?</div>
              <ul className="space-y-1 text-blue-700 dark:text-blue-300">
                <li>• Scale factor = 1 / (1 - dropout_rate)</li>
                <li>• Ensures expected value remains the same</li>
                <li>• No scaling needed during inference</li>
                <li>• Makes model predictions consistent</li>
              </ul>
            </div>
            <div>
              <div className="font-medium mb-2">Benefits</div>
              <ul className="space-y-1 text-blue-700 dark:text-blue-300">
                <li>• Prevents co-adaptation of neurons</li>
                <li>• Forces network to learn redundant features</li>
                <li>• Reduces overfitting significantly</li>
                <li>• Simple but very effective</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Training vs Inference */}
        <div className="grid md:grid-cols-2 gap-4">
          <div className={`p-4 rounded-lg border-2 ${isTraining ? 'border-primary bg-primary/5' : 'border-border'}`}>
            <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
              <Brain className="h-4 w-4" />
              Training Mode
            </h4>
            <ul className="space-y-1 text-xs text-muted-foreground">
              <li>• Randomly drop neurons with probability p</li>
              <li>• Scale remaining neurons by 1/(1-p)</li>
              <li>• Different mask each forward pass</li>
              <li>• Helps prevent overfitting</li>
            </ul>
            {isTraining && (
              <div className="mt-2 p-2 bg-primary/10 rounded text-xs text-primary">
                Currently active
              </div>
            )}
          </div>

          <div className={`p-4 rounded-lg border-2 ${!isTraining ? 'border-primary bg-primary/5' : 'border-border'}`}>
            <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
              <Eye className="h-4 w-4" />
              Inference Mode
            </h4>
            <ul className="space-y-1 text-xs text-muted-foreground">
              <li>• Use all neurons (no dropout)</li>
              <li>• No scaling applied</li>
              <li>• Deterministic predictions</li>
              <li>• Full model capacity used</li>
            </ul>
            {!isTraining && (
              <div className="mt-2 p-2 bg-primary/10 rounded text-xs text-primary">
                Currently active
              </div>
            )}
          </div>
        </div>

        {/* Common Dropout Rates */}
        <div className="space-y-3">
          <h3 className="text-sm font-medium">Common Dropout Rates</h3>
          <div className="grid md:grid-cols-4 gap-3">
            {[
              { rate: 0.2, label: 'Low', use: 'RNNs, shallow networks' },
              { rate: 0.3, label: 'Medium', use: 'Most feedforward networks' },
              { rate: 0.5, label: 'High', use: 'Large networks, regularization' },
              { rate: 0.8, label: 'Very High', use: 'Extreme cases, research' },
            ].map((item) => (
              <div
                key={item.rate}
                className="p-3 bg-muted rounded-lg cursor-pointer hover:bg-muted/80 transition-colors"
                onClick={() => setDropoutRate(item.rate)}
              >
                <div className="text-sm font-medium">{item.label}</div>
                <div className="text-lg font-bold">{(item.rate * 100).toFixed(0)}%</div>
                <div className="text-xs text-muted-foreground mt-1">{item.use}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Tips */}
        <div className="p-4 bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
          <h3 className="text-sm font-medium mb-2 flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 text-yellow-600 dark:text-yellow-400" />
            Practical Tips
          </h3>
          <ul className="space-y-1 text-xs text-yellow-800 dark:text-yellow-200">
            <li>• Use dropout on fully-connected layers, not usually on convolutional layers</li>
            <li>• Typical dropout rates: 0.2-0.5 for hidden layers, 0.5-0.8 for input layers</li>
            <li>• Combine with other regularization techniques (L1/L2, batch norm)</li>
            <li>• Monitor validation loss to tune dropout rate</li>
            <li>• Higher dropout rates may require more training epochs</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  );
};
