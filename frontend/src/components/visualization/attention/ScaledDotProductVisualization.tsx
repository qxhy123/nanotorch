/**
 * ScaledDotProductVisualization Component
 *
 * Interactive demonstration of Scaled Dot-Product Attention:
 * - Step-by-step computation flow
 * - Query, Key, Value matrix visualization
 * - Attention score calculation (QK^T)
 * - Scaling operation (divide by sqrt(d_k))
 * - Softmax normalization
 * - Final weighted sum with Value matrix
 * - Interactive dimension controls
 * - Color-coded heatmaps
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
  ArrowRight,
  Zap,
  Grid3x3,
  Calculator,
  Layers,
  Eye,
} from 'lucide-react';

// Matrix types
interface MatrixData {
  name: string;
  data: number[][];
  rows: number;
  cols: number;
  label?: string;
}

// Generate mock Q, K, V matrices
function generateAttentionMatrices(seqLen: number, dModel: number): {
  Q: MatrixData;
  K: MatrixData;
  V: MatrixData;
} {
  // Generate random-like but deterministic matrices
  const Q: number[][] = [];
  const K: number[][] = [];
  const V: number[][] = [];

  for (let i = 0; i < seqLen; i++) {
    const qRow: number[] = [];
    const kRow: number[] = [];
    const vRow: number[] = [];
    for (let j = 0; j < dModel; j++) {
      qRow.push(Math.sin(i * 0.5 + j * 0.3) * 2 + Math.cos(j * 0.2));
      kRow.push(Math.cos(i * 0.4 + j * 0.5) * 2 + Math.sin(i * 0.3));
      vRow.push(Math.sin(i * 0.3 + j * 0.4) * 1.5 + Math.cos(j * 0.1));
    }
    Q.push(qRow);
    K.push(kRow);
    V.push(vRow);
  }

  return {
    Q: { name: 'Q', data: Q, rows: seqLen, cols: dModel, label: 'Query' },
    K: { name: 'K', data: K, rows: seqLen, cols: dModel, label: 'Key' },
    V: { name: 'V', data: V, rows: seqLen, cols: dModel, label: 'Value' },
  };
}

// Matrix multiplication
function matrixMultiply(A: number[][], B: number[][]): number[][] {
  const result: number[][] = [];
  for (let i = 0; i < A.length; i++) {
    result[i] = [];
    for (let j = 0; j < B[0].length; j++) {
      let sum = 0;
      for (let k = 0; k < B.length; k++) {
        sum += A[i][k] * B[k][j];
      }
      result[i][j] = sum;
    }
  }
  return result;
}

// Transpose matrix
function transpose(A: number[][]): number[][] {
  return A[0].map((_, c) => A.map(r => r[c]));
}

// Apply softmax
function softmax(A: number[][]): number[][] {
  return A.map(row => {
    const maxVal = Math.max(...row);
    const exps = row.map(x => Math.exp(x - maxVal));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sum);
  });
}

// Get color for value
function getValueColor(value: number, min: number, max: number): string {
  const normalized = (value - min) / (max - min + 0.001);
  const hue = (1 - normalized) * 240; // Blue to red
  return `hsl(${hue}, 70%, 50%)`;
}

// Matrix display component
const MatrixDisplay: React.FC<{
  matrix: MatrixData;
  highlightIndices?: Array<{row: number, col: number}>;
  showValues?: boolean;
  size?: 'small' | 'medium' | 'large';
}> = ({ matrix, highlightIndices = [], showValues = true, size = 'medium' }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  // 更大的单元格尺寸
  const cellSize = size === 'small' ? 36 : size === 'medium' ? 45 : 55;
  const displayRows = isExpanded ? matrix.rows : Math.min(matrix.rows, 4);
  const displayCols = isExpanded ? matrix.cols : Math.min(matrix.cols, 8);

  const allValues = matrix.data.flat();
  const minVal = Math.min(...allValues);
  const maxVal = Math.max(...allValues);

  const isHighlighted = (row: number, col: number) => {
    return highlightIndices.some(h => h.row === row && h.col === col);
  };

  // 根据单元格大小决定是否显示数字
  const shouldShowValue = showValues && cellSize >= 36;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-sm font-medium">
            {matrix.label || matrix.name === 'Q' && <Latex>Q</Latex> ||
             matrix.name === 'K' && <Latex>K</Latex> ||
             matrix.name === 'V' && <Latex>V</Latex> ||
             matrix.name === 'QK^T' && <Latex>{'QK^{\\top}'}</Latex> ||
             matrix.label || matrix.name}
          </div>
          <div className="text-xs text-muted-foreground">
            {matrix.rows} × {matrix.cols}
          </div>
        </div>
        {matrix.rows > 4 || matrix.cols > 8 ? (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setIsExpanded(!isExpanded)}
          >
            {isExpanded ? <Eye className="h-3 w-3" /> : <Layers className="h-3 w-3" />}
          </Button>
        ) : null}
      </div>

      <div className="inline-block border border-border rounded-lg p-3 bg-background overflow-x-auto">
        <div
          className="grid gap-0.5 bg-border"
          style={{
            gridTemplateColumns: `repeat(${displayCols}, ${cellSize}px)`,
          }}
        >
          {matrix.data.slice(0, displayRows).map((row, i) =>
            row.slice(0, displayCols).map((val, j) => {
              const bgColor = getValueColor(val, minVal, maxVal);
              const isBright = Math.abs(val) > (maxVal - minVal) / 2;
              const textColor = isBright ? 'white' : 'black';

              return (
                <div
                  key={`${i}-${j}`}
                  className="relative flex items-center justify-center font-mono font-medium"
                  style={{
                    backgroundColor: bgColor,
                    color: textColor,
                    height: `${cellSize}px`,
                    fontSize: cellSize >= 45 ? '12px' : '10px',
                    outline: isHighlighted(i, j) ? '3px solid orange' : 'none',
                    outlineOffset: '-2px',
                  }}
                  title={`${val.toFixed(4)}`}
                >
                  {shouldShowValue ? (
                    <span className="truncate px-0.5" style={{ maxWidth: '100%' }}>
                      {val.toFixed(2)}
                    </span>
                  ) : null}
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
};

interface ScaledDotProductVisualizationProps {
  className?: string;
}

export const ScaledDotProductVisualization: React.FC<ScaledDotProductVisualizationProps> = ({
  className = '',
}) => {
  const [seqLen, setSeqLen] = useState(4);
  const [dModel, setDModel] = useState(6);
  const [currentStep, setCurrentStep] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);
  const [showHeatmaps, setShowHeatmaps] = useState(true);

  // Generate matrices
  const { Q, K, V } = useMemo(() => generateAttentionMatrices(seqLen, dModel), [seqLen, dModel]);

  // Compute attention step by step
  const attentionSteps = useMemo(() => {
    const steps: Array<{
      name: string;
      description: string;
      matrix: MatrixData;
      formula?: string;
    }> = [];

    // Step 1: QK^T
    const KTranspose = transpose(K.data);
    const QKt = matrixMultiply(Q.data, KTranspose);
    steps.push({
      name: 'Q × K^T',
      description: 'Compute raw attention scores by multiplying Query with transposed Key',
      matrix: {
        name: 'QK^T',
        data: QKt,
        rows: seqLen,
        cols: seqLen,
        label: 'Raw Scores',
      },
      formula: '\\text{Scores} = QK^{\\top}',
    });

    // Step 2: Scale
    const scaleFactor = 1 / Math.sqrt(dModel);
    const scaled = QKt.map(row => row.map(v => v * scaleFactor));
    steps.push({
      name: 'Scaled Scores',
      description: `Scale by 1/√d_k where d_k = ${dModel}, so scale factor = ${scaleFactor.toFixed(4)}`,
      matrix: {
        name: 'Scaled',
        data: scaled,
        rows: seqLen,
        cols: seqLen,
        label: 'Scaled Scores',
      },
      formula: `\\text{Scaled} = \\frac{QK^{\\top}}{\\sqrt{d_k}}`,
    });

    // Step 3: Softmax
    const attentionWeights = softmax(scaled);
    steps.push({
      name: 'Attention Weights',
      description: 'Apply softmax to get probability distribution (each row sums to 1)',
      matrix: {
        name: 'Softmax',
        data: attentionWeights,
        rows: seqLen,
        cols: seqLen,
        label: 'Attention Weights',
      },
      formula: '\\text{Attention} = \\text{softmax}(\\text{Scaled})',
    });

    // Step 4: Output
    const output = matrixMultiply(attentionWeights, V.data);
    steps.push({
      name: 'Output',
      description: 'Weighted sum of Value vectors using attention weights',
      matrix: {
        name: 'Output',
        data: output,
        rows: seqLen,
        cols: dModel,
        label: 'Attention Output',
      },
      formula: '\\text{Output} = \\text{Attention} \\cdot V',
    });

    return steps;
  }, [Q, K, V, seqLen, dModel]);

  // Run animation
  const runAnimation = () => {
    setIsAnimating(true);
    setCurrentStep(0);

    for (let i = 0; i <= attentionSteps.length; i++) {
      setTimeout(() => {
        setCurrentStep(i);
        if (i === attentionSteps.length) {
          setIsAnimating(false);
        }
      }, i * 1500);
    }
  };

  // Reset
  const reset = () => {
    setCurrentStep(0);
    setIsAnimating(false);
  };

  // Calculate statistics
  const stats = useMemo(() => {
    const attentionWeights = attentionSteps[2]?.matrix.data || [];
    const flatWeights = attentionWeights.flat();

    return {
      scaleFactor: (1 / Math.sqrt(dModel)).toFixed(4),
      maxAttention: Math.max(...flatWeights).toFixed(3),
      minAttention: Math.min(...flatWeights).toFixed(3),
      totalRowsSum: attentionWeights.map(row => row.reduce((a, b) => a + b, 0)),
    };
  }, [attentionSteps, dModel]);

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Calculator className="h-5 w-5 text-primary" />
              Scaled Dot-Product Attention
            </CardTitle>
            <CardDescription>
              Step-by-step visualization of the attention mechanism
            </CardDescription>
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowHeatmaps(!showHeatmaps)}
            >
              <Grid3x3 className="h-4 w-4 mr-1" />
              {showHeatmaps ? 'Hide' : 'Show'} Heatmaps
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
            Matrix Dimensions
          </h3>

          {/* Sequence Length */}
          <div className="space-y-2">
            <div className="flex items-center gap-4">
              <label className="text-sm font-medium min-w-[120px]">Sequence Length:</label>
              <Slider
                value={[seqLen]}
                onValueChange={([v]) => setSeqLen(v)}
                min={2}
                max={6}
                step={1}
                className="flex-1"
              />
              <span className="text-sm font-mono w-12 text-right">{seqLen}</span>
            </div>
          </div>

          {/* Model Dimension */}
          <div className="space-y-2">
            <div className="flex items-center gap-4">
              <label className="text-sm font-medium min-w-[120px]">Model Dimension:</label>
              <Slider
                value={[dModel]}
                onValueChange={([v]) => setDModel(v)}
                min={4}
                max={12}
                step={2}
                className="flex-1"
              />
              <span className="text-sm font-mono w-12 text-right">{dModel}</span>
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
                  {currentStep === 0 && 'Initializing attention computation...'}
                  {currentStep > 0 && currentStep <= attentionSteps.length && attentionSteps[currentStep - 1].name}
                </div>
                {currentStep > 0 && currentStep <= attentionSteps.length && (
                  <div className="text-xs text-blue-600 dark:text-blue-400 mt-1">
                    {attentionSteps[currentStep - 1].description}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Input Matrices */}
        <div className="space-y-3">
          <h3 className="text-sm font-medium">Input Matrices (Q, K, V)</h3>
          <div className="grid md:grid-cols-3 gap-4">
            {showHeatmaps ? (
              <>
                <MatrixDisplay matrix={Q} size="small" />
                <MatrixDisplay matrix={K} size="small" />
                <MatrixDisplay matrix={V} size="small" />
              </>
            ) : (
              <>
                <div className="p-4 bg-muted rounded-lg text-center">
                  <div className="text-sm font-medium mb-2">Query (Q)</div>
                  <div className="text-xs text-muted-foreground">{seqLen} × {dModel}</div>
                </div>
                <div className="p-4 bg-muted rounded-lg text-center">
                  <div className="text-sm font-medium mb-2">Key (K)</div>
                  <div className="text-xs text-muted-foreground">{seqLen} × {dModel}</div>
                </div>
                <div className="p-4 bg-muted rounded-lg text-center">
                  <div className="text-sm font-medium mb-2">Value (V)</div>
                  <div className="text-xs text-muted-foreground">{seqLen} × {dModel}</div>
                </div>
              </>
            )}
          </div>
        </div>

        {/* Computation Steps */}
        <div className="space-y-4">
          <h3 className="text-sm font-medium">Computation Steps</h3>

          {attentionSteps.map((step, idx) => {
            const isCurrentStep = currentStep === idx + 1;
            const isPastStep = currentStep > idx + 1;

            return (
              <div
                key={idx}
                className={`p-4 rounded-lg border-2 transition-all ${
                  isCurrentStep
                    ? 'border-primary bg-primary/5'
                    : isPastStep
                    ? 'border-green-500 bg-green-50 dark:bg-green-900/20'
                    : 'border-border bg-background opacity-50'
                }`}
              >
                <div className="flex items-center gap-3 mb-3">
                  <div
                    className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                      isCurrentStep
                        ? 'bg-primary text-primary-foreground'
                        : isPastStep
                        ? 'bg-green-500 text-white'
                        : 'bg-muted text-muted-foreground'
                    }`}
                  >
                    {idx + 1}
                  </div>
                  <div className="flex-1">
                    <div className="text-sm font-medium">{step.name}</div>
                    <div className="text-xs text-muted-foreground">{step.description}</div>
                  </div>
                  {isPastStep && !isCurrentStep && (
                    <div className="text-xs text-green-600 dark:text-green-400 font-medium">
                      ✓ Complete
                    </div>
                  )}
                </div>

                {(isCurrentStep || isPastStep) && showHeatmaps && (
                  <MatrixDisplay
                    matrix={step.matrix}
                    size="small"
                  />
                )}

                {step.formula && (
                  <div className="mt-2 p-3 bg-muted rounded text-center">
                    <Latex>{step.formula}</Latex>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Statistics */}
        <div className="grid md:grid-cols-4 gap-4">
          <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 rounded-lg">
            <div className="text-xs text-purple-600 dark:text-purple-400 mb-1">Scale Factor</div>
            <div className="text-2xl font-bold">{stats.scaleFactor}</div>
            <div className="text-xs text-muted-foreground">1/√d_k</div>
          </div>

          <div className="p-4 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-lg">
            <div className="text-xs text-blue-600 dark:text-blue-400 mb-1">Max Attention</div>
            <div className="text-2xl font-bold">{stats.maxAttention}</div>
            <div className="text-xs text-muted-foreground">Highest weight</div>
          </div>

          <div className="p-4 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-lg">
            <div className="text-xs text-green-600 dark:text-green-400 mb-1">Min Attention</div>
            <div className="text-2xl font-bold">{stats.minAttention}</div>
            <div className="text-xs text-muted-foreground">Lowest weight</div>
          </div>

          <div className="p-4 bg-gradient-to-br from-orange-50 to-orange-100 dark:from-orange-900/20 dark:to-orange-800/20 rounded-lg">
            <div className="text-xs text-orange-600 dark:text-orange-400 mb-1">Output Shape</div>
            <div className="text-2xl font-bold">{seqLen} × {dModel}</div>
            <div className="text-xs text-muted-foreground">Final dimensions</div>
          </div>
        </div>

        {/* Explanation */}
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
          <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
            <Zap className="h-4 w-4 text-blue-600 dark:text-blue-400" />
            Understanding Scaled Dot-Product Attention
          </h3>
          <div className="grid md:grid-cols-2 gap-4 text-xs text-blue-800 dark:text-blue-200">
            <div>
              <div className="font-medium mb-2">Why Scaling Matters</div>
              <ul className="space-y-1 text-blue-700 dark:text-blue-300">
                <li>• Prevents softmax saturation when <Latex>{'d_k'}</Latex> is large</li>
                <li>• Keeps gradients stable during training</li>
                <li>• Scale factor = <Latex>{'1/\\sqrt{d_k}'}</Latex> ensures variance ≈ 1</li>
                <li>• Without scaling, attention becomes too "sharp"</li>
              </ul>
            </div>
            <div>
              <div className="font-medium mb-2">Key Insights</div>
              <ul className="space-y-1 text-blue-700 dark:text-blue-300">
                <li>• <Latex>Q</Latex> determines what to look for</li>
                <li>• <Latex>K</Latex> determines what's available</li>
                <li>• <Latex>V</Latex> contains the actual information</li>
                <li>• Attention weights are learned patterns</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Formula Summary */}
        <div className="p-4 bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-lg border border-indigo-200 dark:border-indigo-800">
          <h3 className="text-sm font-medium mb-3">Complete Formula</h3>
          <div className="bg-background p-4 rounded-lg text-center">
            <div className="mb-2 text-lg">
              <Latex display>{'\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^{\\top}}{\\sqrt{d_k}}\\right)V'}</Latex>
            </div>
            <div className="text-xs text-muted-foreground">
              Where: <Latex>Q</Latex> = Query, <Latex>K</Latex> = Key, <Latex>V</Latex> = Value, <Latex>{'d_k'}</Latex> = dimension of keys
            </div>
          </div>
        </div>

        {/* Visual Flow */}
        <div className="flex items-center justify-center gap-4 flex-wrap p-4 bg-muted rounded-lg">
          <div className="text-center">
            <div className="text-xs font-medium mb-1"><Latex>Q</Latex></div>
            <div className="w-12 h-8 bg-blue-500 rounded flex items-center justify-center text-white text-xs">
              {seqLen}×{dModel}
            </div>
          </div>
          <ArrowRight className="h-4 w-4 text-muted-foreground" />
          <div className="text-center">
            <div className="text-xs font-medium mb-1">Scores</div>
            <div className="w-16 h-8 bg-purple-500 rounded flex items-center justify-center text-white text-xs">
              {seqLen}×{seqLen}
            </div>
          </div>
          <ArrowRight className="h-4 w-4 text-muted-foreground" />
          <div className="text-center">
            <div className="text-xs font-medium mb-1">Weights</div>
            <div className="w-16 h-8 bg-green-500 rounded flex items-center justify-center text-white text-xs">
              {seqLen}×{seqLen}
            </div>
          </div>
          <ArrowRight className="h-4 w-4 text-muted-foreground" />
          <div className="text-center">
            <div className="text-xs font-medium mb-1"><Latex>V</Latex></div>
            <div className="w-12 h-8 bg-orange-500 rounded flex items-center justify-center text-white text-xs">
              {seqLen}×{dModel}
            </div>
          </div>
          <ArrowRight className="h-4 w-4 text-muted-foreground" />
          <div className="text-center">
            <div className="text-xs font-medium mb-1">Output</div>
            <div className="w-12 h-8 bg-red-500 rounded flex items-center justify-center text-white text-xs">
              {seqLen}×{dModel}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
