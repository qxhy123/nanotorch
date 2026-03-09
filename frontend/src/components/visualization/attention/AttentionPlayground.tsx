/**
 * AttentionPlayground Component
 *
 * Interactive playground for exploring attention mechanism:
 * - Custom text input for Q, K, V
 * - Real-time attention computation
 * - Interactive matrix manipulation
 * - Visualize attention flow step by step
 * - Compare different attention patterns
 * - Explore scale factor effects
 * - Masking visualization
 */

import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../ui/card';
import { Button } from '../../ui/button';
import { Slider } from '../../ui/slider';
import { Latex } from '../../ui/Latex';
import {
  Play,
  RotateCcw,
  Settings,
  Sparkles,
  Eye,
  EyeOff,
  Maximize2,
  Minimize2,
  Copy,
  Check,
} from 'lucide-react';

// Matrix data type
interface MatrixData {
  name: string;
  data: number[][];
  rows: number;
  cols: number;
  label?: string;
}

// Attention step type
type AttentionStep = 'input' | 'qk' | 'scaled' | 'softmax' | 'output';

// Default example texts
const EXAMPLES = {
  simple: {
    query: 'The cat sat on the mat',
    key: 'cat',
    value: 'feline',
  },
  translation: {
    query: 'The quick brown fox',
    key: 'quick fox',
    value: 'rapide renard',
  },
  selfAttention: {
    query: 'I love machine learning',
    key: 'I love machine learning',
    value: 'I love machine learning',
  },
};

export const AttentionPlayground: React.FC<{ className?: string }> = ({ className = '' }) => {
  // State
  const [queryText, setQueryText] = useState(EXAMPLES.simple.query);
  const [keyText, setKeyText] = useState(EXAMPLES.simple.key);
  const [valueText, setValueText] = useState(EXAMPLES.simple.value);
  const [scaleFactor, setScaleFactor] = useState(1.0);
  const [temperature, setTemperature] = useState(1.0);
  const [currentStep, setCurrentStep] = useState<AttentionStep>('input');
  const [showMask, setShowMask] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [copied, setCopied] = useState(false);
  const [isAnimating, setIsAnimating] = useState(false);

  // Text to vectors (simple character-based embedding)
  const textToVector = (text: string, dim: number = 8): number[] => {
    const vector = new Array(dim).fill(0);
    const chars = text.toLowerCase().replace(/\s/g, '');
    for (let i = 0; i < chars.length; i++) {
      const charCode = chars.charCodeAt(i);
      vector[i % dim] += charCode / 255;
    }
    // Normalize
    const norm = Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0));
    return norm > 0 ? vector.map(v => v / norm) : vector;
  };

  // Generate Q, K, V matrices from text
  const { Q, K, V } = useMemo(() => {
    const qTokens = queryText.split(' ').filter(t => t);
    const kTokens = keyText.split(' ').filter(t => t);
    const vTokens = valueText.split(' ').filter(t => t);

    const dim = 8;
    const Q = qTokens.map(token => textToVector(token, dim));
    const K = kTokens.map(token => textToVector(token, dim));
    const V = vTokens.map(token => textToVector(token, dim));

    return { Q, K, V };
  }, [queryText, keyText, valueText]);

  // Compute attention step by step
  const attentionSteps = useMemo(() => {
    // Step 1: QK^T
    const QK_T = Q.map((q) =>
      K.map((k) => q.reduce((sum, qVal, idx) => sum + qVal * k[idx], 0))
    );

    // Step 2: Scaled
    const scaled = QK_T.map(row =>
      row.map(val => val / Math.sqrt(K[0]?.length || 1) * scaleFactor)
    );

    // Step 3: Softmax
    const softmax = (scaled: number[][], temp: number) => {
      return scaled.map(row => {
        const max = Math.max(...row);
        const exp = row.map(v => Math.exp((v - max) / temp));
        const sum = exp.reduce((a, b) => a + b, 0);
        return exp.map(v => v / sum);
      });
    };

    const attentionWeights = softmax(scaled, temperature);

    // Step 4: Output = Attention * V
    const output = attentionWeights.map((row) => {
      return V[0]?.map((_, colIdx) =>
        row.reduce((sum, _, k) => sum + row[k] * (V[k]?.[colIdx] || 0), 0)
      ) || [];
    });

    return {
      input: { Q, K, V },
      qk: QK_T,
      scaled,
      attentionWeights,
      output,
    };
  }, [Q, K, V, scaleFactor, temperature]);

  // Get current display data
  const getDisplayData = (): MatrixData[] => {
    switch (currentStep) {
      case 'input':
        return [
          { name: 'Q', data: attentionSteps.input.Q, rows: Q.length, cols: Q[0]?.length || 0, label: 'Query' },
          { name: 'K', data: attentionSteps.input.K, rows: K.length, cols: K[0]?.length || 0, label: 'Key' },
          { name: 'V', data: attentionSteps.input.V, rows: V.length, cols: V[0]?.length || 0, label: 'Value' },
        ];
      case 'qk':
        return [
          {
            name: 'QK^T',
            data: attentionSteps.qk,
            rows: Q.length,
            cols: K.length,
            label: 'Query-Key Similarity',
          },
        ];
      case 'scaled':
        return [
          {
            name: 'Scaled',
            data: attentionSteps.scaled,
            rows: Q.length,
            cols: K.length,
            label: 'Scaled Scores',
          },
        ];
      case 'softmax':
        return [
          {
            name: 'Attention',
            data: attentionSteps.attentionWeights,
            rows: Q.length,
            cols: K.length,
            label: 'Attention Weights',
          },
        ];
      case 'output':
        return [
          {
            name: 'Output',
            data: attentionSteps.output,
            rows: Q.length,
            cols: V[0]?.length || 0,
            label: 'Final Output',
          },
        ];
      default:
        return [];
    }
  };

  // Run animation through all steps
  const runAnimation = () => {
    setIsAnimating(true);
    const steps: AttentionStep[] = ['input', 'qk', 'scaled', 'softmax', 'output'];
    let stepIdx = 0;

    const animate = () => {
      if (stepIdx < steps.length) {
        setCurrentStep(steps[stepIdx]);
        stepIdx++;
        setTimeout(animate, 1500);
      } else {
        setIsAnimating(false);
      }
    };

    animate();
  };

  // Reset
  const reset = () => {
    setCurrentStep('input');
    setIsAnimating(false);
  };

  // Copy configuration
  const copyConfig = () => {
    const config = {
      query: queryText,
      key: keyText,
      value: valueText,
      scaleFactor,
      temperature,
    };
    navigator.clipboard.writeText(JSON.stringify(config, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Load example
  const loadExample = (name: keyof typeof EXAMPLES) => {
    const example = EXAMPLES[name];
    setQueryText(example.query);
    setKeyText(example.key);
    setValueText(example.value);
    setCurrentStep('input');
  };

  // Matrix cell component
  const MatrixCell: React.FC<{
    value: number;
    maxValue: number;
    showValue: boolean;
  }> = ({ value, maxValue, showValue }) => {
    const intensity = Math.abs(value) / maxValue;
    const isPositive = value >= 0;
    const bgColor = isPositive
      ? `rgba(59, 130, 246, ${intensity})`
      : `rgba(239, 68, 68, ${intensity})`;

    return (
      <div
        className="relative flex items-center justify-center border border-border transition-all hover:scale-110 hover:z-10"
        style={{
          backgroundColor: bgColor,
          width: isExpanded ? '60px' : '45px',
          height: isExpanded ? '60px' : '45px',
        }}
        title={`Value: ${value.toFixed(4)}`}
      >
        {showValue && (
          <span
            className={`text-xs font-mono font-medium ${
              intensity > 0.5 ? 'text-white' : 'text-foreground'
            }`}
          >
            {value.toFixed(2)}
          </span>
        )}
      </div>
    );
  };

  const displayData = getDisplayData();
  const stepNames: Record<AttentionStep, string> = {
    input: 'Input Matrices',
    qk: 'QK^T',
    scaled: 'Scaled Scores',
    softmax: 'Attention Weights',
    output: 'Final Output',
  };

  const stepFormulas: Record<AttentionStep, string> = {
    input: '\\text{Input: } Q, K, V',
    qk: '\\text{Scores} = QK^{\\top}',
    scaled: '\\text{Scaled} = \\frac{QK^{\\top}}{\\sqrt{d_k}}',
    softmax: '\\text{Attention} = \\text{softmax}(\\text{Scaled})',
    output: '\\text{Output} = \\text{Attention} \\cdot V',
  };

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-primary" />
              Attention Playground
            </CardTitle>
            <CardDescription>
              Interactive exploration of attention mechanism
            </CardDescription>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={copyConfig}>
              {copied ? <Check className="h-4 w-4 mr-1" /> : <Copy className="h-4 w-4 mr-1" />}
              {copied ? 'Copied!' : 'Copy'}
            </Button>
            <Button variant="outline" size="sm" onClick={runAnimation} disabled={isAnimating}>
              <Play className="h-4 w-4 mr-1" />
              Animate
            </Button>
            <Button variant="outline" size="sm" onClick={reset} disabled={isAnimating}>
              <RotateCcw className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setIsExpanded(!isExpanded)}
            >
              {isExpanded ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Example Selection */}
        <div className="flex flex-wrap gap-2">
          <span className="text-sm font-medium">Load Example:</span>
          <Button variant="outline" size="sm" onClick={() => loadExample('simple')}>
            Simple
          </Button>
          <Button variant="outline" size="sm" onClick={() => loadExample('translation')}>
            Translation
          </Button>
          <Button variant="outline" size="sm" onClick={() => loadExample('selfAttention')}>
            Self-Attention
          </Button>
        </div>

        {/* Text Inputs */}
        <div className="grid md:grid-cols-3 gap-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Query Text:</label>
            <textarea
              value={queryText}
              onChange={(e) => setQueryText(e.target.value)}
              className="w-full p-2 border rounded-md text-sm min-h-[60px] font-mono"
              placeholder="Enter query text..."
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium">Key Text:</label>
            <textarea
              value={keyText}
              onChange={(e) => setKeyText(e.target.value)}
              className="w-full p-2 border rounded-md text-sm min-h-[60px] font-mono"
              placeholder="Enter key text..."
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium">Value Text:</label>
            <textarea
              value={valueText}
              onChange={(e) => setValueText(e.target.value)}
              className="w-full p-2 border rounded-md text-sm min-h-[60px] font-mono"
              placeholder="Enter value text..."
            />
          </div>
        </div>

        {/* Parameters */}
        <div className="space-y-4 p-4 bg-muted rounded-lg">
          <h3 className="text-sm font-medium flex items-center gap-2">
            <Settings className="h-4 w-4" />
            Attention Parameters
          </h3>

          {/* Scale Factor */}
          <div className="space-y-2">
            <div className="flex items-center gap-4">
              <label className="text-sm font-medium min-w-[120px]">Scale Factor:</label>
              <Slider
                value={[scaleFactor]}
                onValueChange={([v]) => setScaleFactor(v)}
                min={0.1}
                max={3}
                step={0.1}
                className="flex-1"
              />
              <span className="text-sm font-mono w-12 text-right">{scaleFactor.toFixed(1)}</span>
            </div>
            <p className="text-xs text-muted-foreground">
              Controls the scaling of QK^T before softmax
            </p>
          </div>

          {/* Temperature */}
          <div className="space-y-2">
            <div className="flex items-center gap-4">
              <label className="text-sm font-medium min-w-[120px]">Temperature:</label>
              <Slider
                value={[temperature]}
                onValueChange={([v]) => setTemperature(v)}
                min={0.1}
                max={2}
                step={0.1}
                className="flex-1"
              />
              <span className="text-sm font-mono w-12 text-right">{temperature.toFixed(1)}</span>
            </div>
            <p className="text-xs text-muted-foreground">
              Controls the sharpness of attention distribution
            </p>
          </div>

          {/* Show Values Toggle */}
          <div className="flex items-center gap-4">
            <label className="text-sm font-medium min-w-[120px]">Show Values:</label>
            <Button
              variant={showMask ? 'default' : 'outline'}
              size="sm"
              onClick={() => setShowMask(!showMask)}
            >
              {showMask ? <Eye className="h-4 w-4 mr-1" /> : <EyeOff className="h-4 w-4 mr-1" />}
              {showMask ? 'On' : 'Off'}
            </Button>
          </div>
        </div>

        {/* Step Navigation */}
        <div className="flex flex-wrap gap-2">
          {(Object.keys(stepNames) as AttentionStep[]).map((step) => (
            <Button
              key={step}
              variant={currentStep === step ? 'default' : 'outline'}
              size="sm"
              onClick={() => setCurrentStep(step)}
              disabled={isAnimating}
            >
              {stepNames[step]}
            </Button>
          ))}
        </div>

        {/* Current Formula */}
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
          <div className="text-sm text-center">
            <Latex display>{stepFormulas[currentStep]}</Latex>
          </div>
        </div>

        {/* Matrix Visualization */}
        <div className="space-y-4">
          {displayData.map((matrix) => {
            const flatData = matrix.data.flat();
            const maxValue = Math.max(...flatData.map(Math.abs), 1);

            return (
              <div key={matrix.name} className="space-y-2">
                <h3 className="text-sm font-medium flex items-center gap-2">
                  <Latex>{matrix.name}</Latex>
                  {matrix.label && <span className="text-muted-foreground">- {matrix.label}</span>}
                  <span className="text-xs text-muted-foreground">
                    ({matrix.rows}×{matrix.cols})
                  </span>
                </h3>
                <div className="overflow-x-auto">
                  <div className="inline-flex flex-col gap-1">
                    {matrix.data.map((row, i) => (
                      <div key={i} className="flex gap-1">
                        {row.map((cell, j) => (
                          <MatrixCell
                            key={`${i}-${j}`}
                            value={cell}
                            maxValue={maxValue}
                            showValue={showMask}
                          />
                        ))}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Statistics */}
        <div className="grid md:grid-cols-4 gap-4">
          <div className="p-4 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-lg">
            <div className="text-xs text-blue-600 dark:text-blue-400 mb-1">Query Tokens</div>
            <div className="text-2xl font-bold">{Q.length}</div>
          </div>
          <div className="p-4 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-lg">
            <div className="text-xs text-green-600 dark:text-green-400 mb-1">Key Tokens</div>
            <div className="text-2xl font-bold">{K.length}</div>
          </div>
          <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 rounded-lg">
            <div className="text-xs text-purple-600 dark:text-purple-400 mb-1">Embedding Dim</div>
            <div className="text-2xl font-bold">{Q[0]?.length || 0}</div>
          </div>
          <div className="p-4 bg-gradient-to-br from-orange-50 to-orange-100 dark:from-orange-900/20 dark:to-orange-800/20 rounded-lg">
            <div className="text-xs text-orange-600 dark:text-orange-400 mb-1">Output Dim</div>
            <div className="text-2xl font-bold">{V[0]?.length || 0}</div>
          </div>
        </div>

        {/* Explanation */}
        <div className="p-4 bg-muted rounded-lg">
          <h3 className="text-sm font-medium mb-2">How to Use</h3>
          <ul className="space-y-1 text-xs text-muted-foreground">
            <li>• Enter your own text for Query, Key, and Value</li>
            <li>• Adjust scale factor and temperature to see effects</li>
            <li>• Click through steps to see attention computation</li>
            <li>• Use Animate to see the full attention flow</li>
            <li>• Load examples to explore different attention patterns</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  );
};
