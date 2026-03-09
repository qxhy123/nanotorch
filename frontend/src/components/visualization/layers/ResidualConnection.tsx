/**
 * ResidualConnection Component
 *
 * Interactive visualization of residual (skip) connections in neural networks.
 * Demonstrates how residual connections help with gradient flow and training stability.
 *
 * Features:
 * - Visual representation of residual connection architecture
 * - Interactive comparison with/without residual connections
 * - Gradient flow visualization
 * - Network depth exploration
 * - Step-by-step computation demo
 */

import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../ui/card';
import { Button } from '../../ui/button';
import { Slider } from '../../ui/slider';
import { Badge } from '../../ui/badge';
import { Latex } from '../../ui/Latex';
import {
  Plus,
  ArrowRight,
  Zap,
  Network,
  TrendingUp,
  Eye,
  EyeOff,
  Play,
  RotateCcw,
  Layers,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';

type NetworkDepth = 2 | 5 | 10 | 20 | 50;
type ViewMode = 'architecture' | 'gradient-flow' | 'comparison' | 'interactive';

// Network layer simulation
interface LayerData {
  index: number;
  input: number[];
  output: number[];
  sublayerOutput: number[];
  residualOutput: number[];
  gradientWithoutResidual: number;
  gradientWithResidual: number;
}

export const ResidualConnection: React.FC<{ className?: string }> = ({ className = '' }) => {
  const [networkDepth, setNetworkDepth] = useState<NetworkDepth>(10);
  const [showResidual, setShowResidual] = useState(true);
  const [viewMode, setViewMode] = useState<ViewMode>('architecture');
  const [isAnimating, setIsAnimating] = useState(false);
  const [animationStep, setAnimationStep] = useState(0);
  const [expandedSection, setExpandedSection] = useState<string | null>(null);

  // Simulate forward pass through network
  const simulateNetwork = useMemo(() => {
    const layers: LayerData[] = [];
    const initialInput = Array.from({ length: 10 }, (_, i) => Math.sin(i * 0.5) * 2);

    let currentInput = [...initialInput];
    let gradientWithout = 1.0;
    let gradientWith = 1.0;

    for (let i = 0; i < networkDepth; i++) {
      // Simulate sublayer computation (e.g., attention or FFN)
      const sublayerOutput = currentInput.map((x, idx) =>
        Math.tanh(x * 1.5 + Math.sin(idx + i) * 0.5)
      );

      // Residual connection: output = input + sublayer(input)
      const residualOutput = currentInput.map((x, idx) => x + sublayerOutput[idx]);

      // Simulate gradient decay
      gradientWithout *= 0.85; // Vanishes without residual
      gradientWith *= 0.98; // Maintained with residual

      layers.push({
        index: i,
        input: [...currentInput],
        output: [...residualOutput],
        sublayerOutput,
        residualOutput,
        gradientWithoutResidual: gradientWithout,
        gradientWithResidual: gradientWith,
      });

      currentInput = residualOutput;
    }

    return { layers, initialInput };
  }, [networkDepth, showResidual]);

  // Interactive demo data
  const [inputValues, setInputValues] = useState<number[]>([1.0, 2.0, 1.5, 0.5]);
  const [sublayerStrength, setSublayerStrength] = useState(1.0);

  const interactiveResult = useMemo(() => {
    const sublayerOutput = inputValues.map((x, idx) =>
      Math.tanh(x * sublayerStrength + idx * 0.3) * sublayerStrength
    );
    const residualOutput = inputValues.map((x, idx) => x + sublayerOutput[idx]);
    return { sublayerOutput, residualOutput };
  }, [inputValues, sublayerStrength]);

  // Gradient flow data for visualization
  const gradientFlowData = useMemo(() => {
    const depths = [2, 5, 10, 20, 50];
    return depths.map(depth => {
      let gradWithout = 1.0;
      let gradWith = 1.0;
      for (let i = 0; i < depth; i++) {
        gradWithout *= 0.85;
        gradWith *= 0.98;
      }
      return { depth, gradWithout, gradWith };
    });
  }, []);

  // Comparison statistics
  const comparisonStats = useMemo(() => {
    const finalLayer = simulateNetwork.layers[simulateNetwork.layers.length - 1];
    return {
      finalGradientWithout: finalLayer.gradientWithoutResidual,
      finalGradientWith: finalLayer.gradientWithResidual,
      gradientRatio: finalLayer.gradientWithResidual / finalLayer.gradientWithoutResidual,
      signalRetentionWithout: Math.pow(0.85, networkDepth) * 100,
      signalRetentionWith: Math.pow(0.98, networkDepth) * 100,
    };
  }, [simulateNetwork, networkDepth]);

  // Animation handler
  const runAnimation = () => {
    setIsAnimating(true);
    setAnimationStep(0);
    const steps = ['forward', 'sublayer', 'residual', 'backward', 'gradient'];
    let currentStep = 0;

    const interval = setInterval(() => {
      if (currentStep < steps.length) {
        setAnimationStep(currentStep);
        currentStep++;
      } else {
        clearInterval(interval);
        setIsAnimating(false);
      }
    }, 1000);
  };

  const reset = () => {
    setInputValues([1.0, 2.0, 1.5, 0.5]);
    setSublayerStrength(1.0);
    setAnimationStep(0);
    setIsAnimating(false);
  };

  const renderArchitectureView = () => (
    <div className="space-y-6">
      {/* Residual Block Diagram */}
      <div className="p-6 bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg border">
        <h3 className="text-lg font-semibold mb-4">Residual Block Architecture</h3>
        <div className="flex flex-col items-center gap-4">
          {/* Input */}
          <div className="flex items-center gap-3">
            <Badge variant="outline" className="px-4 py-2">
              <span className="font-mono">x</span>
            </Badge>
            <span className="text-sm text-muted-foreground">Input</span>
          </div>

          <ArrowRight className="w-5 h-5 text-muted-foreground" />

          {/* Split */}
          <div className="flex items-center gap-8">
            {/* Main path */}
            <div className="flex flex-col items-center gap-3">
              <Badge variant="outline" className="px-4 py-2">
                <span className="font-mono">F(x)</span>
              </Badge>
              <span className="text-xs text-muted-foreground">Sublayer</span>
            </div>

            {/* Skip connection */}
            <div className="flex items-center gap-2">
              <div className="w-12 h-0.5 bg-gray-400" />
              <span className="text-xs text-muted-foreground">Identity</span>
              <div className="w-12 h-0.5 bg-gray-400" />
            </div>
          </div>

          <ArrowRight className="w-5 h-5 text-muted-foreground" />

          {/* Addition */}
          <div className="flex items-center gap-3">
            <div className="w-16 h-16 rounded-full bg-blue-500 flex items-center justify-center">
              <Plus className="w-8 h-8 text-white" />
            </div>
            <span className="text-sm text-muted-foreground">Add</span>
          </div>

          <ArrowRight className="w-5 h-5 text-muted-foreground" />

          {/* Output */}
          <div className="flex items-center gap-3">
            <Badge variant="outline" className="px-4 py-2 bg-green-100 dark:bg-green-900/30">
              <span className="font-mono">x + F(x)</span>
            </Badge>
            <span className="text-sm text-muted-foreground">Output</span>
          </div>
        </div>
      </div>

      {/* Formula */}
      <div className="p-4 bg-muted rounded-lg">
        <h4 className="text-sm font-medium mb-2">Mathematical Formulation</h4>
        <div className="space-y-2 text-sm">
          <div className="flex items-center justify-center gap-4">
            <Latex display>{`\\text{Output} = x + F(x)`}</Latex>
          </div>
          <div className="text-xs text-muted-foreground text-center">
            Where x is the input and F(x) is the sublayer computation
          </div>
        </div>
      </div>

      {/* Key Properties */}
      <div className="grid md:grid-cols-3 gap-4">
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
          <Zap className="w-6 h-6 text-blue-500 mb-2" />
          <h4 className="text-sm font-medium mb-1">Gradient Highway</h4>
          <p className="text-xs text-muted-foreground">
            Unimpeded gradient flow through identity mapping
          </p>
        </div>
        <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
          <TrendingUp className="w-6 h-6 text-green-500 mb-2" />
          <h4 className="text-sm font-medium mb-1">Identity Learning</h4>
          <p className="text-xs text-muted-foreground">
            Easy to learn identity functions
          </p>
        </div>
        <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
          <Network className="w-6 h-6 text-purple-500 mb-2" />
          <h4 className="text-sm font-medium mb-1">Deep Networks</h4>
          <p className="text-xs text-muted-foreground">
            Enables training of very deep networks
          </p>
        </div>
      </div>
    </div>
  );

  const renderGradientFlowView = () => (
    <div className="space-y-6">
      <div className="p-4 bg-muted rounded-lg">
        <h4 className="text-sm font-medium mb-3">Network Depth Configuration</h4>
        <div className="flex items-center gap-4">
          <span className="text-sm">Depth:</span>
          <div className="flex gap-2">
            {[2, 5, 10, 20, 50].map((depth) => (
              <Button
                key={depth}
                variant={networkDepth === depth ? 'default' : 'outline'}
                size="sm"
                onClick={() => setNetworkDepth(depth as NetworkDepth)}
              >
                {depth}
              </Button>
            ))}
          </div>
        </div>
      </div>

      {/* Gradient Flow Comparison Chart */}
      <div className="p-6 bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-lg border">
        <h3 className="text-lg font-semibold mb-4">Gradient Flow Comparison</h3>
        <div className="space-y-4">
          {gradientFlowData.map(({ depth, gradWithout, gradWith }) => (
            <div key={depth} className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="font-medium">Depth {depth}</span>
                <div className="flex gap-4">
                  <span className="text-red-600 dark:text-red-400">
                    Without: {(gradWithout * 100).toFixed(1)}%
                  </span>
                  <span className="text-green-600 dark:text-green-400">
                    With: {(gradWith * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
              <div className="flex gap-1 h-8">
                {/* Without residual */}
                <div
                  className="bg-red-400 dark:bg-red-600 rounded-l transition-all"
                  style={{ width: `${gradWithout * 100}%` }}
                  title={`Without residual: ${(gradWithout * 100).toFixed(2)}%`}
                />
                {/* With residual */}
                <div
                  className="bg-green-400 dark:bg-green-600 rounded-r transition-all"
                  style={{ width: `${gradWith * 100}%` }}
                  title={`With residual: ${(gradWith * 100).toFixed(2)}%`}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Statistics */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
          <h4 className="text-sm font-medium mb-3 text-red-800 dark:text-red-200">
            Without Residual Connection
          </h4>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span>Final Gradient:</span>
              <span className="font-mono">{(comparisonStats.finalGradientWithout * 100).toFixed(3)}%</span>
            </div>
            <div className="flex justify-between">
              <span>Signal Retention:</span>
              <span className="font-mono">{comparisonStats.signalRetentionWithout.toFixed(1)}%</span>
            </div>
            <div className="text-xs text-muted-foreground mt-2">
              Gradient vanishes exponentially with depth
            </div>
          </div>
        </div>

        <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
          <h4 className="text-sm font-medium mb-3 text-green-800 dark:text-green-200">
            With Residual Connection
          </h4>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span>Final Gradient:</span>
              <span className="font-mono">{(comparisonStats.finalGradientWith * 100).toFixed(3)}%</span>
            </div>
            <div className="flex justify-between">
              <span>Signal Retention:</span>
              <span className="font-mono">{comparisonStats.signalRetentionWith.toFixed(1)}%</span>
            </div>
            <div className="text-xs text-muted-foreground mt-2">
              Gradient flows through identity path
            </div>
          </div>
        </div>
      </div>

      {/* Improvement Ratio */}
      <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
        <div className="flex items-center justify-between">
          <div>
            <h4 className="text-sm font-medium text-blue-800 dark:text-blue-200">
              Gradient Improvement
            </h4>
            <p className="text-xs text-blue-600 dark:text-blue-400 mt-1">
              Residual connections improve gradient flow by
            </p>
          </div>
          <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">
            {comparisonStats.gradientRatio.toFixed(1)}x
          </div>
        </div>
      </div>
    </div>
  );

  const renderComparisonView = () => (
    <div className="space-y-6">
      {/* Network visualization */}
      <div className="p-4 bg-muted rounded-lg">
        <div className="flex items-center justify-between mb-4">
          <h4 className="text-sm font-medium">Network Visualization ({networkDepth} layers)</h4>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowResidual(!showResidual)}
          >
            {showResidual ? <EyeOff className="w-4 h-4 mr-1" /> : <Eye className="w-4 h-4 mr-1" />}
            {showResidual ? 'Hide' : 'Show'} Residual
          </Button>
        </div>

        <div className="flex gap-4 overflow-x-auto pb-2">
          {simulateNetwork.layers.slice(0, 10).map((layer) => (
            <div key={layer.index} className="flex-shrink-0">
              <div
                className={`w-12 h-32 rounded-lg flex flex-col items-center justify-center transition-all ${
                  showResidual
                    ? 'bg-gradient-to-t from-blue-500 to-green-500'
                    : 'bg-gradient-to-t from-red-500 to-orange-500'
                }`}
                style={{
                  opacity: Math.max(0.2, layer.gradientWithResidual),
                }}
              >
                <span className="text-white text-xs font-mono">
                  {layer.index + 1}
                </span>
              </div>
              <div className="text-xs text-center mt-1 text-muted-foreground">
                {(layer.gradientWithResidual * 100).toFixed(0)}%
              </div>
            </div>
          ))}
          {networkDepth > 10 && (
            <div className="flex items-center text-muted-foreground text-sm">
              ... +{networkDepth - 10} more layers
            </div>
          )}
        </div>
      </div>

      {/* Layer-wise statistics */}
      <div className="p-4 bg-muted rounded-lg">
        <h4 className="text-sm font-medium mb-3">Layer-wise Analysis</h4>
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {simulateNetwork.layers.map((layer) => (
            <div
              key={layer.index}
              className="flex items-center gap-4 p-2 bg-background rounded"
            >
              <span className="text-xs font-mono w-8">L{layer.index + 1}</span>
              <div className="flex-1 h-4 bg-gray-200 dark:bg-gray-700 rounded overflow-hidden">
                <div
                  className={`h-full transition-all ${
                    showResidual ? 'bg-green-500' : 'bg-red-500'
                  }`}
                  style={{
                    width: `${
                      showResidual
                        ? layer.gradientWithResidual * 100
                        : layer.gradientWithoutResidual * 100
                    }%`,
                  }}
                />
              </div>
              <span className="text-xs font-mono w-16 text-right">
                {showResidual
                  ? `${(layer.gradientWithResidual * 100).toFixed(1)}%`
                  : `${(layer.gradientWithoutResidual * 100).toFixed(1)}%`}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Benefits explanation */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
          <h4 className="text-sm font-medium mb-2 text-green-800 dark:text-green-200">
            ✓ With Residual Connections
          </h4>
          <ul className="text-xs text-green-700 dark:text-green-300 space-y-1">
            <li>• Gradients flow freely to early layers</li>
            <li>• Stable training of deep networks</li>
            <li>• Better optimization landscape</li>
            <li>• Higher accuracy on deep architectures</li>
          </ul>
        </div>

        <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
          <h4 className="text-sm font-medium mb-2 text-red-800 dark:text-red-200">
            ✗ Without Residual Connections
          </h4>
          <ul className="text-xs text-red-700 dark:text-red-300 space-y-1">
            <li>• Gradient vanishes in early layers</li>
            <li>• Difficult to train deep networks</li>
            <li>• Optimization problems</li>
            <li>• Performance degrades with depth</li>
          </ul>
        </div>
      </div>
    </div>
  );

  const renderInteractiveView = () => {
    const stepDescriptions = [
      { title: 'Input', desc: 'Original input tensor x', highlight: 'input' },
      { title: 'Sublayer', desc: 'Apply transformation F(x) = tanh(x)', highlight: 'sublayer' },
      { title: 'Residual', desc: 'Add input to sublayer output: x + F(x)', highlight: 'residual' },
      { title: 'Backward', desc: 'Gradients flow back through network', highlight: 'gradient' },
      { title: 'Complete', desc: 'Full residual connection cycle', highlight: 'complete' },
    ];

    const currentStep = animationStep >= stepDescriptions.length ? 4 : animationStep;

    return (
      <div className="space-y-6">
        {/* Step-by-step animation */}
        <div className="p-4 bg-muted rounded-lg">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h4 className="text-sm font-medium">Interactive Demo</h4>
              {isAnimating && (
                <p className="text-xs text-muted-foreground mt-1">
                  {stepDescriptions[currentStep].desc}
                </p>
              )}
            </div>
            <div className="flex gap-2">
              <Button variant="outline" size="sm" onClick={runAnimation} disabled={isAnimating}>
                <Play className="w-4 h-4 mr-1" />
                Animate
              </Button>
              <Button variant="outline" size="sm" onClick={reset} disabled={isAnimating}>
                <RotateCcw className="w-4 h-4" />
              </Button>
            </div>
          </div>

          {/* Animation steps */}
          <div className="flex gap-2 mb-4 flex-wrap">
            {stepDescriptions.map((step, idx) => (
              <Badge
                key={step.title}
                variant={currentStep === idx ? 'default' : 'outline'}
                className={currentStep === idx ? 'animate-pulse' : ''}
              >
                {idx + 1}. {step.title}
              </Badge>
            ))}
          </div>

          {/* Visual representation with animation */}
          <div className="p-6 bg-background rounded-lg border-2 transition-all duration-500"
               style={{
                 borderColor: currentStep === 4 ? '#22c55e' :
                             currentStep === 3 ? '#f59e0b' :
                             currentStep === 2 ? '#8b5cf6' :
                             currentStep === 1 ? '#ec4899' : '#3b82f6'
               }}
          >
            {/* Step 0: Input */}
            <div className={`transition-all duration-500 ${currentStep >= 0 ? 'opacity-100 scale-100' : 'opacity-50 scale-95'}`}>
              <div className="text-center mb-4">
                <div className={`text-xs font-semibold mb-2 ${currentStep === 0 ? 'text-blue-600 dark:text-blue-400' : 'text-muted-foreground'}`}>
                  Step 1: Input (x)
                </div>
                <div className="flex gap-2 justify-center">
                  {inputValues.map((val, idx) => (
                    <div
                      key={`input-${idx}`}
                      className={`w-12 h-12 rounded-lg flex items-center justify-center text-sm font-mono font-bold transition-all duration-300 ${
                        currentStep === 0 ? 'bg-blue-500 text-white scale-110 shadow-lg' : 'bg-blue-100 dark:bg-blue-900/30'
                      }`}
                      style={{
                        transform: currentStep === 0 ? `scale(${1 + Math.abs(val) * 0.1})` : 'scale(1)',
                      }}
                    >
                      {val.toFixed(1)}
                    </div>
                  ))}
                </div>
                <div className="text-xs text-muted-foreground mt-2">
                  x = [{inputValues.map(v => v.toFixed(1)).join(', ')}]
                </div>
              </div>
            </div>

            {/* Arrow and operator */}
            {currentStep >= 1 && (
              <div className="flex items-center justify-center gap-2 my-4 transition-all duration-500">
                <ArrowRight className="w-6 h-6 text-purple-500 animate-bounce" />
                <span className="text-sm font-medium text-purple-600 dark:text-purple-400">Apply F(x)</span>
                <ArrowRight className="w-6 h-6 text-purple-500 animate-bounce" />
              </div>
            )}

            {/* Step 1: Sublayer */}
            {currentStep >= 1 && (
              <div className={`transition-all duration-500 ${currentStep >= 1 ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-4'}`}>
                <div className="text-center mb-4">
                  <div className={`text-xs font-semibold mb-2 ${currentStep === 1 ? 'text-purple-600 dark:text-purple-400' : 'text-muted-foreground'}`}>
                    Step 2: Sublayer Output F(x)
                  </div>
                  <div className="flex gap-2 justify-center">
                    {interactiveResult.sublayerOutput.map((val, idx) => (
                      <div
                        key={`sublayer-${idx}`}
                        className={`w-12 h-12 rounded-lg flex items-center justify-center text-sm font-mono font-bold transition-all duration-300 ${
                          currentStep === 1 ? 'bg-purple-500 text-white scale-110 shadow-lg' : 'bg-purple-100 dark:bg-purple-900/30'
                        }`}
                        style={{
                          transform: currentStep === 1 ? `scale(${1 + Math.abs(val) * 0.1})` : 'scale(1)',
                        }}
                      >
                        {val.toFixed(1)}
                      </div>
                    ))}
                  </div>
                  <div className="text-xs text-muted-foreground mt-2">
                    <Latex>{`F(x) = \\tanh(x \\times ${sublayerStrength.toFixed(1)})`}</Latex>
                  </div>
                </div>
              </div>
            )}

            {/* Addition operator */}
            {currentStep >= 2 && (
              <div className="flex items-center justify-center gap-2 my-4 transition-all duration-500">
                <div className="w-16 h-16 rounded-full bg-green-500 flex items-center justify-center animate-pulse">
                  <Plus className="w-8 h-8 text-white" />
                </div>
                <span className="text-sm font-medium text-green-600 dark:text-green-400">Add Residual</span>
              </div>
            )}

            {/* Step 2: Residual Output */}
            {currentStep >= 2 && (
              <div className={`transition-all duration-500 ${currentStep >= 2 ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}`}>
                <div className="text-center mb-4">
                  <div className={`text-xs font-semibold mb-2 ${currentStep === 2 ? 'text-green-600 dark:text-green-400' : 'text-muted-foreground'}`}>
                    Step 3: Residual Output (x + F(x))
                  </div>
                  <div className="flex gap-2 justify-center">
                    {interactiveResult.residualOutput.map((val, idx) => (
                      <div
                        key={`output-${idx}`}
                        className={`w-12 h-12 rounded-lg flex items-center justify-center text-sm font-mono font-bold transition-all duration-300 ${
                          currentStep === 2 ? 'bg-green-500 text-white scale-110 shadow-lg' : 'bg-green-100 dark:bg-green-900/30'
                        }`}
                        style={{
                          transform: currentStep === 2 ? `scale(${1 + Math.abs(val) * 0.1})` : 'scale(1)',
                        }}
                      >
                        {val.toFixed(1)}
                      </div>
                    ))}
                  </div>
                  <div className="text-xs text-muted-foreground mt-2">
                    Output = x + F(x) = [{interactiveResult.residualOutput.map(v => v.toFixed(1)).join(', ')}]
                  </div>
                </div>
              </div>
            )}

            {/* Step 3: Gradient Flow */}
            {currentStep >= 3 && (
              <div className={`mt-4 p-4 rounded-lg transition-all duration-500 ${currentStep >= 3 ? 'opacity-100' : 'opacity-0'}`}
                   style={{
                     background: 'linear-gradient(135deg, rgba(251, 191, 36, 0.2) 0%, rgba(245, 158, 11, 0.2) 100%)',
                     border: '2px solid #f59e0b'
                   }}
              >
                <div className="text-center">
                  <div className="text-sm font-semibold text-amber-700 dark:text-amber-300 mb-2">
                    Step 4: Backward Pass - Gradient Flow
                  </div>
                  <div className="flex items-center justify-center gap-4">
                    <div className="text-center">
                      <Latex display>{"\\frac{\\partial L}{\\partial y}"}</Latex>
                      <div className="text-xs text-muted-foreground mt-1">Output Grad</div>
                    </div>
                    <ArrowRight className="w-6 h-6 text-amber-500" />
                    <div className="text-center">
                      <Latex display>{"\\times \\left(1 + \\frac{\\partial F}{\\partial x}\\right)"}</Latex>
                      <div className="text-xs text-muted-foreground mt-1">Chain Rule</div>
                    </div>
                    <ArrowRight className="w-6 h-6 text-amber-500" />
                    <div className="text-center">
                      <Latex display>{"\\frac{\\partial L}{\\partial x}"}</Latex>
                      <div className="text-xs text-muted-foreground mt-1">Input Grad</div>
                    </div>
                  </div>
                  <div className="mt-3 p-2 bg-amber-100 dark:bg-amber-900/30 rounded text-xs text-amber-800 dark:text-amber-200">
                    ✓ Gradient flows through +1 identity path - no vanishing!
                  </div>
                </div>
              </div>
            )}

            {/* Step 4: Complete */}
            {currentStep >= 4 && (
              <div className={`mt-4 p-4 rounded-lg transition-all duration-500 ${currentStep >= 4 ? 'opacity-100 scale-100' : 'opacity-0 scale-95'}`}
                   style={{
                     background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(22, 163, 74, 0.2) 100%)',
                     border: '2px solid #22c55e'
                   }}
              >
                <div className="text-center">
                  <div className="text-lg font-bold text-green-700 dark:text-green-300 mb-2">
                    ✓ Complete!
                  </div>
                  <div className="text-sm text-green-600 dark:text-green-400">
                    Residual connection successfully preserves information and enables gradient flow
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Interactive parameters */}
        <div className="space-y-4">
          <div className="p-4 bg-muted rounded-lg">
            <h4 className="text-sm font-medium mb-3">Input Values (Real-time)</h4>
            <div className="flex gap-2">
              {inputValues.map((val, idx) => (
                <div key={idx} className="flex-1">
                  <label className="text-xs text-muted-foreground">x{idx}</label>
                  <Slider
                    value={[val]}
                    onValueChange={([v]) => {
                      const newValues = [...inputValues];
                      newValues[idx] = v;
                      setInputValues(newValues);
                    }}
                    min={-3}
                    max={3}
                    step={0.1}
                    className="my-2"
                  />
                  <div className="text-center">
                    <span className={`text-sm font-mono font-bold ${
                      val > 0 ? 'text-green-600 dark:text-green-400' :
                      val < 0 ? 'text-red-600 dark:text-red-400' :
                      'text-gray-600 dark:text-gray-400'
                    }`}>
                      {val.toFixed(2)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="p-4 bg-muted rounded-lg">
            <h4 className="text-sm font-medium mb-3">Sublayer Strength: {sublayerStrength.toFixed(1)}</h4>
            <div className="flex items-center gap-4">
              <Slider
                value={[sublayerStrength]}
                onValueChange={([v]) => setSublayerStrength(v)}
                min={0}
                max={2}
                step={0.1}
                className="flex-1"
              />
            </div>
            <div className="mt-2 p-2 bg-background rounded text-xs">
              <div className="text-purple-600 dark:text-purple-400">
                <Latex>{`F(x) = \\tanh(x \\times ${sublayerStrength.toFixed(1)})`}</Latex>
              </div>
              <div className="text-green-600 dark:text-green-400 mt-1">
                <Latex>{`\\text{Output} = [${interactiveResult.residualOutput.map(v => v.toFixed(2)).join(', ')}]`}</Latex>
              </div>
            </div>
          </div>
        </div>

        {/* Comparison panel */}
        <div className="grid md:grid-cols-2 gap-4">
          <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
            <h4 className="text-sm font-medium mb-2 text-blue-800 dark:text-blue-200">✓ With Residual</h4>
            <div className="space-y-1 text-xs text-blue-700 dark:text-blue-300">
              <div>Input preserved: <span className="font-mono font-bold">{inputValues.map(v => v.toFixed(1)).join(', ')}</span></div>
              <div>Output: <span className="font-mono font-bold">{interactiveResult.residualOutput.map(v => v.toFixed(1)).join(', ')}</span></div>
              <div className="text-xs text-muted-foreground mt-2">Original signal flows through unchanged</div>
            </div>
          </div>

          <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
            <h4 className="text-sm font-medium mb-2 text-gray-800 dark:text-gray-200">✗ Without Residual</h4>
            <div className="space-y-1 text-xs text-gray-700 dark:text-gray-300">
              <div>Input: <span className="font-mono">{inputValues.map(v => v.toFixed(1)).join(', ')}</span></div>
              <div>Output: <span className="font-mono">{interactiveResult.sublayerOutput.map(v => v.toFixed(1)).join(', ')}</span></div>
              <div className="text-xs text-muted-foreground mt-2">Signal transformed, may lose information</div>
            </div>
          </div>
        </div>

        {/* Interactive explanation */}
        <div className="p-4 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg border">
          <h4 className="text-sm font-medium mb-2">🎮 How to Use</h4>
          <ul className="text-xs text-muted-foreground space-y-1">
            <li>• <strong>Sliders:</strong> Adjust input values (x₀-x₃) and sublayer strength in real-time</li>
            <li>• <strong>Animate:</strong> Click to see step-by-step forward and backward pass</li>
            <li>• <strong>Observe:</strong> Watch how the +1 in gradient (1 + ∂F/∂x) preserves gradients</li>
            <li>• <strong>Compare:</strong> See how residual connections preserve input information</li>
          </ul>
        </div>
      </div>
    );
  };

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Layers className="h-5 w-5 text-primary" />
              Residual Connection Visualization
            </CardTitle>
            <CardDescription>
              Explore how residual connections enable training of deep neural networks
            </CardDescription>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* View mode selector */}
        <div className="flex flex-wrap gap-2">
          <Button
            variant={viewMode === 'architecture' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setViewMode('architecture')}
          >
            <Network className="w-4 h-4 mr-1" />
            Architecture
          </Button>
          <Button
            variant={viewMode === 'gradient-flow' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setViewMode('gradient-flow')}
          >
            <TrendingUp className="w-4 h-4 mr-1" />
            Gradient Flow
          </Button>
          <Button
            variant={viewMode === 'comparison' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setViewMode('comparison')}
          >
            <Eye className="w-4 h-4 mr-1" />
            Comparison
          </Button>
          <Button
            variant={viewMode === 'interactive' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setViewMode('interactive')}
          >
            <Zap className="w-4 h-4 mr-1" />
            Interactive
          </Button>
        </div>

        {/* Content based on view mode */}
        {viewMode === 'architecture' && renderArchitectureView()}
        {viewMode === 'gradient-flow' && renderGradientFlowView()}
        {viewMode === 'comparison' && renderComparisonView()}
        {viewMode === 'interactive' && renderInteractiveView()}

        {/* Educational section */}
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
          <button
            onClick={() => setExpandedSection(
              expandedSection === 'why-it-works' ? null : 'why-it-works'
            )}
            className="flex items-center justify-between w-full"
          >
            <h4 className="text-sm font-medium text-blue-800 dark:text-blue-200">
              Why Residual Connections Work
            </h4>
            {expandedSection === 'why-it-works' ? (
              <ChevronUp className="w-4 h-4" />
            ) : (
              <ChevronDown className="w-4 h-4" />
            )}
          </button>
          {expandedSection === 'why-it-works' && (
            <div className="mt-3 text-xs text-blue-700 dark:text-blue-300 space-y-2">
              <p>
                Residual connections add the input directly to the output: <Latex>{`y = x + F(x)`}</Latex>
              </p>
              <p>
                During backpropagation, gradients flow through the identity path:
                <Latex display>{`\\frac{\\partial L}{\\partial x} = \\frac{\\partial L}{\\partial y} \\cdot (1 + \\frac{\\partial F}{\\partial x})`}</Latex>
              </p>
              <p>
                The <span className="font-semibold">+1</span> term ensures gradients can always flow
                directly, preventing vanishing gradients in deep networks.
              </p>
              <p className="mt-2">
                This allows networks to learn identity mappings easily and enables training of
                networks with hundreds of layers.
              </p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};
