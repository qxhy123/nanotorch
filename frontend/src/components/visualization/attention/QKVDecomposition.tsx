/**
 * QKVDecomposition Component
 *
 * Step-by-step visualization showing how Q, K, V matrices are computed,
 * how attention scores are calculated, and how the final attention output is produced.
 * Includes interactive controls for matrix dimensions and attention head count.
 */

import React, { useState, useCallback, useEffect, useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../ui/card';
import { Badge } from '../../ui/badge';
import { Button } from '../../ui/button';
import { Latex } from '../../ui/Latex';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ArrowRight,
  Grid3x3,
  Play,
  Pause,
  SkipForward,
  SkipBack,
  RefreshCw,
  Info,
  Zap,
} from 'lucide-react';

// Types
interface MatrixData {
  name: string;
  latexName?: string;
  data: number[][];
  shape: number[];
  description: string;
}

interface ComputationStep {
  id: string;
  title: string;
  description: string;
  formula: string;
  inputMatrices: MatrixData[];
  outputMatrix: MatrixData;
  notes: string[];
}

interface QKVDecompositionProps {
  seqLen?: number;
  dModel?: number;
  numHeads?: number;
  headDim?: number;
  className?: string;
}

// Mock matrix generator
function generateMatrix(rows: number, cols: number, scale: number = 1): number[][] {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => (Math.random() * 2 - 1) * scale)
  );
}

function matrixMultiply(A: number[][], B: number[][]): number[][] {
  const rows = A.length;
  const cols = B[0].length;
  const inner = A[0].length;
  const result = Array.from({ length: rows }, () => Array(cols).fill(0));

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      for (let k = 0; k < inner; k++) {
        result[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return result;
}

function transposeMatrix(A: number[][]): number[][] {
  return A[0].map((_, colIndex) => A.map(row => row[colIndex]));
}

function softmax(arr: number[]): number[] {
  const max = Math.max(...arr);
  const exp = arr.map(x => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map(x => x / sum);
}

function generateComputationSteps(
  seqLen: number,
  dModel: number,
  numHeads: number,
  headDim: number
): ComputationStep[] {
  const steps: ComputationStep[] = [];

  // Input X
  const X = generateMatrix(seqLen, dModel, 0.1);
  steps.push({
    id: 'input',
    title: 'Input X',
    description: 'Input sequence embeddings',
    formula: 'X ∈ ℝ^{n×d_model}',
    inputMatrices: [],
    outputMatrix: {
      name: 'X',
      latexName: '\\mathbf{X}',
      data: X,
      shape: [seqLen, dModel],
      description: 'Input embeddings',
    },
    notes: [`Shape: [${seqLen}, ${dModel}]`, 'Each row is a token embedding'],
  });

  // Weight matrices
  const WQ = generateMatrix(dModel, dModel, 0.1);
  const WK = generateMatrix(dModel, dModel, 0.1);
  const WV = generateMatrix(dModel, dModel, 0.1);

  // Step 1: Compute Q, K, V
  const Q = matrixMultiply(X, WQ);
  const K = matrixMultiply(X, WK);
  const V = matrixMultiply(X, WV);

  steps.push({
    id: 'qkv',
    title: 'Compute Q, K, V',
    description: 'Project input into Query, Key, Value matrices',
    formula: 'Q = XW_Q, K = XW_K, V = XW_V',
    inputMatrices: [
      { name: 'X', latexName: '\\mathbf{X}', data: X, shape: [seqLen, dModel], description: 'Input' },
      { name: 'W_Q', latexName: '\\mathbf{W}_Q', data: WQ, shape: [dModel, dModel], description: 'Query weights' },
      { name: 'W_K', latexName: '\\mathbf{W}_K', data: WK, shape: [dModel, dModel], description: 'Key weights' },
      { name: 'W_V', latexName: '\\mathbf{W}_V', data: WV, shape: [dModel, dModel], description: 'Value weights' },
    ],
    outputMatrix: {
      name: 'Q, K, V',
      latexName: '\\mathbf{Q}, \\mathbf{K}, \\mathbf{V}',
      data: Q,
      shape: [seqLen, dModel],
      description: 'Projections',
    },
    notes: [
      'Each projection learns different aspects',
      'Dimensions preserved: [seq_len, d_model]',
    ],
  });

  // Step 2: Split into heads
  const Q_heads = [];
  const K_heads = [];
  const V_heads = [];

  for (let h = 0; h < numHeads; h++) {
    const startCol = h * headDim;
    const endCol = startCol + headDim;

    Q_heads.push(Q.map(row => row.slice(startCol, endCol)));
    K_heads.push(K.map(row => row.slice(startCol, endCol)));
    V_heads.push(V.map(row => row.slice(startCol, endCol)));
  }

  steps.push({
    id: 'split',
    title: 'Split into Heads',
    description: `Split Q, K, V into ${numHeads} heads`,
    formula: `Q_h = Q[:, h·d_k:(h+1)·d_k]`,
    inputMatrices: [
      { name: 'Q', latexName: '\\mathbf{Q}', data: Q, shape: [seqLen, dModel], description: 'Full Query' },
    ],
    outputMatrix: {
      name: 'Q_0',
      latexName: '\\mathbf{Q}_0',
      data: Q_heads[0],
      shape: [seqLen, headDim],
      description: 'Head 0 Query',
    },
    notes: [
      `Each head has dimension: d_k = ${headDim}`,
      `Total heads: ${numHeads}`,
      `Heads process representations independently`,
    ],
  });

  // Step 3: Compute scores
  const headIndex = 0;
  const Q_h = Q_heads[headIndex];
  const K_h = K_heads[headIndex];
  const K_h_T = transposeMatrix(K_h);
  const scores = matrixMultiply(Q_h, K_h_T);

  steps.push({
    id: 'scores',
    title: 'Compute Attention Scores',
    description: 'Calculate raw attention scores between all pairs',
    formula: 'Scores = Q · K^T',
    inputMatrices: [
      { name: 'Q_h', latexName: '\\mathbf{Q}_h', data: Q_h, shape: [seqLen, headDim], description: 'Head Query' },
      { name: 'K_h^T', latexName: '\\mathbf{K}_h^{\\top}', data: K_h_T, shape: [headDim, seqLen], description: 'Head Key (transposed)' },
    ],
    outputMatrix: {
      name: 'Scores',
      latexName: '\\mathbf{S}',
      data: scores,
      shape: [seqLen, seqLen],
      description: 'Raw attention scores',
    },
    notes: [
      'Measures similarity between queries and keys',
      `Output shape: [${seqLen}, ${seqLen}] (attention map)`,
    ],
  });

  // Step 4: Scale scores
  const scale = Math.sqrt(headDim);
  const scaledScores = scores.map(row => row.map(s => s / scale));

  steps.push({
    id: 'scale',
    title: 'Scale Scores',
    description: 'Divide by √d_k to prevent large values',
    formula: `Scaled = Scores / √d_k = Scores / ${scale.toFixed(2)}`,
    inputMatrices: [
      { name: 'Scores', latexName: '\\mathbf{S}', data: scores, shape: [seqLen, seqLen], description: 'Raw scores' },
    ],
    outputMatrix: {
      name: 'Scaled',
      latexName: '\\mathbf{S}_{\\text{scaled}}',
      data: scaledScores,
      shape: [seqLen, seqLen],
      description: 'Scaled scores',
    },
    notes: [
      'Prevents gradients from vanishing/exploding',
      `Scale factor: 1/√d_k = ${(1/scale).toFixed(4)}`,
    ],
  });

  // Step 5: Apply softmax
  const attentionWeights = scaledScores.map(row => softmax(row));

  steps.push({
    id: 'softmax',
    title: 'Apply Softmax',
    description: 'Convert scores to probabilities (sum to 1)',
    formula: 'Attention = softmax(Scaled)',
    inputMatrices: [
      { name: 'Scaled', latexName: '\\mathbf{S}_{\\text{scaled}}', data: scaledScores, shape: [seqLen, seqLen], description: 'Scaled scores' },
    ],
    outputMatrix: {
      name: 'Attention',
      latexName: '\\mathbf{A}',
      data: attentionWeights,
      shape: [seqLen, seqLen],
      description: 'Attention weights',
    },
    notes: [
      'Each row sums to 1',
      'Represents how much each token attends to others',
      'This is the attention visualization!',
    ],
  });

  // Step 6: Weighted sum of values
  const V_h = V_heads[headIndex];
  const output = matrixMultiply(attentionWeights, V_h);

  steps.push({
    id: 'output',
    title: 'Weighted Sum of Values',
    description: 'Multiply attention weights by values',
    formula: 'Output = Attention · V',
    inputMatrices: [
      { name: 'Attention', latexName: '\\mathbf{A}', data: attentionWeights, shape: [seqLen, seqLen], description: 'Attention weights' },
      { name: 'V_h', latexName: '\\mathbf{V}_h', data: V_h, shape: [seqLen, headDim], description: 'Head values' },
    ],
    outputMatrix: {
      name: 'Output',
      latexName: '\\mathbf{O}_h',
      data: output,
      shape: [seqLen, headDim],
      description: 'Head output',
    },
    notes: [
      'Aggregates information based on attention weights',
      `Output shape: [${seqLen}, ${headDim}]`,
      'Will be concatenated with other heads',
    ],
  });

  return steps;
}

export const QKVDecomposition: React.FC<QKVDecompositionProps> = ({
  seqLen = 4,
  dModel = 8,
  numHeads = 2,
  headDim = 4,
  className = '',
}) => {
  // State
  const [steps, setSteps] = useState<ComputationStep[]>([]);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(2000);

  // Load data
  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 500));
      const computationSteps = generateComputationSteps(seqLen, dModel, numHeads, headDim);
      setSteps(computationSteps);
      setCurrentStepIndex(0);
    } catch (error) {
      console.error('Failed to load QKV decomposition:', error);
    } finally {
      setLoading(false);
    }
  }, [seqLen, dModel, numHeads, headDim]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Auto-play
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

  // Current step
  const currentStep = useMemo(() => {
    if (steps.length === 0 || currentStepIndex >= steps.length) return null;
    return steps[currentStepIndex];
  }, [steps, currentStepIndex]);

  // Render matrix cell
  const renderMatrixCell = (value: number, max: number = 1) => {
    const intensity = Math.min(Math.abs(value) / max, 1);
    const bgColor = value >= 0
      ? `rgba(59, 130, 246, ${intensity})`  // blue for positive
      : `rgba(239, 68, 68, ${intensity})`;   // red for negative

    // Simplified formatting: just show 1 decimal place for cleaner display
    const latexValue = value.toFixed(1);

    return (
      <div
        className="w-12 h-12 flex items-center justify-center text-sm"
        style={{ backgroundColor: bgColor }}
        title={`Value: ${value.toFixed(4)}`}
      >
        {latexValue}
      </div>
    );
  };

  // Render matrix
  const renderMatrix = (matrix: MatrixData, maxSize = 8) => {
    const data = matrix.data.slice(0, maxSize).map(row => row.slice(0, maxSize));
    const max = Math.max(...data.flat().map(Math.abs));
    const shape = matrix.shape;
    const shapeLatex = `\\mathbb{R}^{${shape[0]} \\times ${shape[1]}}`;

    return (
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">
            {matrix.latexName ? <Latex>{matrix.latexName}</Latex> : matrix.name}
          </span>
          <Badge variant="secondary" className="font-mono">
            <Latex>{shapeLatex}</Latex>
          </Badge>
        </div>
        <p className="text-xs text-gray-600 dark:text-gray-400">{matrix.description}</p>

        <div className="inline-grid gap-px bg-gray-300 dark:bg-gray-700 p-1 rounded">
          {data.map((row, i) => (
            <div key={i} className="flex gap-px">
              {row.map((val, j) => (
                <div key={j}>
                  {renderMatrixCell(val, max)}
                </div>
              ))}
            </div>
          ))}
        </div>

        {matrix.data.length > maxSize && (
          <p className="text-xs text-gray-500 italic">
            Showing first <Latex>{`${maxSize} \\times ${maxSize}`}</Latex> of{' '}
            <Latex>{`${matrix.data.length} \\times ${matrix.data[0]?.length || 0}`}</Latex>
          </p>
        )}
      </div>
    );
  };

  // Control handlers
  const handlePlayPause = () => {
    if (currentStepIndex >= steps.length - 1) {
      setCurrentStepIndex(0);
    }
    setIsPlaying(!isPlaying);
  };

  const handleNext = () => {
    if (currentStepIndex < steps.length - 1) {
      setCurrentStepIndex(prev => prev + 1);
    }
  };

  const handlePrev = () => {
    if (currentStepIndex > 0) {
      setCurrentStepIndex(prev => prev - 1);
    }
  };

  if (loading) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center py-12">
          <RefreshCw className="h-8 w-8 animate-spin text-primary" />
        </CardContent>
      </Card>
    );
  }

  if (steps.length === 0) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center py-12">
          <div className="text-center text-gray-500">
            <Grid3x3 className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p className="text-lg font-medium">No QKV decomposition data available</p>
            <Button onClick={loadData} className="mt-4">
              Load Data
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Grid3x3 className="h-5 w-5 text-primary" />
              QKV Decomposition
            </CardTitle>
            <CardDescription>
              Step-by-step attention computation breakdown
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={loadData}>
              <RefreshCw className="h-4 w-4 mr-1" />
              Refresh
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Parameters */}
        <div className="grid grid-cols-4 gap-4">
          <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
            <div className="text-xs text-gray-600 dark:text-gray-400">Sequence Length</div>
            <div className="text-lg font-bold text-blue-700 dark:text-blue-300">{seqLen}</div>
          </div>
          <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
            <div className="text-xs text-gray-600 dark:text-gray-400">Model Dimension</div>
            <div className="text-lg font-bold text-purple-700 dark:text-purple-300">{dModel}</div>
          </div>
          <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
            <div className="text-xs text-gray-600 dark:text-gray-400">Attention Heads</div>
            <div className="text-lg font-bold text-green-700 dark:text-green-300">{numHeads}</div>
          </div>
          <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
            <div className="text-xs text-gray-600 dark:text-gray-400">Head Dimension</div>
            <div className="text-lg font-bold text-orange-700 dark:text-orange-300">{headDim}</div>
          </div>
        </div>

        {/* Step Navigation */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium">Computation Steps</h3>
            <Badge variant="secondary">
              Step {currentStepIndex + 1} of {steps.length}
            </Badge>
          </div>

          <div className="flex flex-wrap gap-2">
            {steps.map((step, index) => (
              <Button
                key={step.id}
                variant={index === currentStepIndex ? 'default' : 'outline'}
                size="sm"
                onClick={() => setCurrentStepIndex(index)}
                className="relative"
              >
                {index + 1}. {step.title}
                {index < currentStepIndex && (
                  <span className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full flex items-center justify-center">
                    <span className="text-white text-xs">✓</span>
                  </span>
                )}
              </Button>
            ))}
          </div>

          {/* Progress bar */}
          <div className="relative h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500"
              initial={{ width: 0 }}
              animate={{ width: `${((currentStepIndex + 1) / steps.length) * 100}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
        </div>

        {/* Playback Controls */}
        <div className="flex items-center gap-2 p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <Button variant="outline" size="sm" onClick={handlePrev} disabled={currentStepIndex === 0}>
            <SkipBack className="w-4 h-4" />
          </Button>
          <Button variant="default" size="sm" onClick={handlePlayPause}>
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          </Button>
          <Button variant="outline" size="sm" onClick={handleNext} disabled={currentStepIndex === steps.length - 1}>
            <SkipForward className="w-4 h-4" />
          </Button>
          <div className="ml-auto flex items-center gap-2">
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
        </div>

        {/* Current Step Content */}
        <AnimatePresence mode="wait">
          {currentStep && (
            <motion.div
              key={currentStep.id}
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -50 }}
              transition={{ duration: 0.3 }}
              className="space-y-4"
            >
              {/* Step Header */}
              <div className="p-4 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 bg-primary text-white rounded-full flex items-center justify-center font-bold">
                    {currentStepIndex + 1}
                  </div>
                  <div>
                    <h3 className="text-lg font-bold">{currentStep.title}</h3>
                    <p className="text-sm text-gray-700 dark:text-gray-300">{currentStep.description}</p>
                  </div>
                </div>
              </div>

              {/* Formula */}
              <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <Zap className="h-4 w-4 text-purple-600 dark:text-purple-400" />
                  <span className="text-sm font-medium">Formula:</span>
                </div>
                <div className="bg-white dark:bg-gray-900 px-3 py-2 rounded">
                  <Latex className="text-lg">{currentStep.formula}</Latex>
                </div>
              </div>

              {/* Input Matrices */}
              {currentStep.inputMatrices.length > 0 && (
                <div className="space-y-3">
                  <h4 className="text-sm font-medium">Input Matrices</h4>
                  <div className="grid md:grid-cols-2 gap-4">
                    {currentStep.inputMatrices.map((matrix) => renderMatrix(matrix))}
                  </div>
                </div>
              )}

              {/* Operation Arrow */}
              {currentStep.inputMatrices.length > 0 && (
                <div className="flex items-center justify-center">
                  <ArrowRight className="h-6 w-6 text-gray-400" />
                  <span className="ml-2 text-sm text-gray-600 dark:text-gray-400">
                    {currentStep.id === 'scores' ? 'Matrix Multiply' :
                     currentStep.id === 'scale' ? 'Divide by √d_k' :
                     currentStep.id === 'softmax' ? 'Softmax' :
                     currentStep.id === 'output' ? 'Matrix Multiply' :
                     'Matrix Multiply'}
                  </span>
                </div>
              )}

              {/* Output Matrix */}
              <div className="space-y-3">
                <h4 className="text-sm font-medium">Output Matrix</h4>
                {renderMatrix(currentStep.outputMatrix)}
              </div>

              {/* Notes */}
              {currentStep.notes.length > 0 && (
                <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
                  <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
                    <Info className="h-4 w-4 text-blue-600 dark:text-blue-400" />
                    Key Insights
                  </h4>
                  <ul className="space-y-1 text-sm text-blue-800 dark:text-blue-200">
                    {currentStep.notes.map((note) => (
                      <li key={note} className="flex items-start gap-2">
                        <div className="w-1.5 h-1.5 rounded-full bg-blue-500 mt-1.5 flex-shrink-0" />
                        <span>{note}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Color Legend */}
        <div className="flex items-center gap-4 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-blue-500 rounded" />
            <span>Positive</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-red-500 rounded" />
            <span>Negative</span>
          </div>
          <div className="flex items-center gap-2 ml-auto">
            <span>Light = Low magnitude</span>
            <span>Dark = High magnitude</span>
          </div>
        </div>

        {/* Info Box */}
        <div className="p-4 bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-lg">
          <div className="flex items-start gap-3">
            <Info className="h-5 w-5 text-purple-600 dark:text-purple-400 flex-shrink-0 mt-0.5" />
            <div className="text-sm text-purple-900 dark:text-purple-100">
              <h4 className="font-medium mb-1">Understanding QKV Decomposition</h4>
              <p className="text-purple-800 dark:text-purple-200">
                The attention mechanism computes three projections of the input: Query (what to look for),
                Key (what's contained), and Value (what to extract). The similarity between Query and Key
                determines attention weights, which are then used to aggregate Value information. This decomposition
                allows the model to attend to different representation subspaces simultaneously.
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
