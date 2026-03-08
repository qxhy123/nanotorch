import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAttentionStages } from '../../../hooks/useAttentionStages';
import { useSemanticColors } from '../../../hooks/useSemanticColors';
import { Latex } from '../../ui/Latex';
import { MatrixVisualization, MatrixComparison } from '../shared/MatrixVisualization';
import type { AttentionData } from '../../../types/transformer';
import { Button } from '../../ui/button';
import { Badge } from '../../ui/badge';
import { Card } from '../../ui/card';
import { ChevronLeft, ChevronRight, Play, Pause, RotateCcw, Info } from 'lucide-react';

interface StagedAttentionVisualizationProps {
  attentionData: AttentionData;
  tokens: string[];
}

/**
 * StagedAttentionVisualization Component
 *
 * Displays the multi-step attention computation process:
 * Q · K^T → Scale → Mask → Softmax → Weighted Sum
 */
export const StagedAttentionVisualization: React.FC<StagedAttentionVisualizationProps> = ({
  attentionData,
  tokens,
}) => {
  const {
    currentStage,
    currentStageInfo,
    hasNextStage,
    hasPreviousStage,
    goToNextStage,
    goToPreviousStage,
    goToStage,
    resetToFirstStage,
    currentStageIndex,
    allStages,
    computationStep,
  } = useAttentionStages(attentionData);

  const { getQKVColors } = useSemanticColors();
  const [isPlaying, setIsPlaying] = useState(false);
  const [autoPlaySpeed, setAutoPlaySpeed] = useState(2000);
  const containerRef = useRef<HTMLDivElement>(null);

  const qkvColors = getQKVColors();

  // Auto-play functionality
  useEffect(() => {
    if (isPlaying && hasNextStage) {
      const timer = setTimeout(() => {
        goToNextStage();
      }, autoPlaySpeed);
      return () => clearTimeout(timer);
    } else if (isPlaying && !hasNextStage) {
      setIsPlaying(false);
    }
  }, [isPlaying, currentStage, hasNextStage, goToNextStage, autoPlaySpeed]);

  const handlePlayPause = () => {
    if (isPlaying) {
      setIsPlaying(false);
    } else {
      setIsPlaying(true);
    }
  };

  const handleReset = () => {
    setIsPlaying(false);
    resetToFirstStage();
  };

  const getStageColor = (stage: string) => {
    switch (stage) {
      case 'queries':
        return qkvColors.query.primary;
      case 'keys':
        return qkvColors.key.primary;
      case 'values':
        return qkvColors.value.primary;
      case 'dot_product':
      case 'scaled':
      case 'masked':
      case 'softmax':
        return qkvColors.query.gradient[1];
      case 'weighted_sum':
        return qkvColors.value.primary;
      default:
        return '#6366f1';
    }
  };

  const renderComputationStep = () => {
    if (!currentStageInfo || !computationStep) return null;

    const { title, description, formula } = currentStageInfo;
    const stageColor = getStageColor(currentStage);
    const { data, metadata } = computationStep;

    // Get matrix data
    const getMatrixData = () => {
      const matrixData = Array.isArray(data.data)
        ? Array.isArray(data.data[0])
          ? data.data as number[][]
          : [data.data as number[]]
        : [[]];
      return matrixData;
    };

    const matrixData = getMatrixData();

    // Render matrix visualization for each stage
    const renderStageVisualization = () => {
      const cellSize = matrixData.length > 8 ? 45 : 60;

      switch (currentStage) {
        case 'queries':
          return (
            <div className="space-y-4">
              <div className="grid lg:grid-cols-2 gap-6">
                <Card className="p-4">
                  <h4 className="text-sm font-medium mb-3" style={{ color: qkvColors.query.primary }}>
                    Query Matrix (Q)
                  </h4>
                  <p className="text-xs text-gray-500 mb-3">
                    Each token projects a query vector representing what it's looking for.
                  </p>
                  <MatrixVisualization
                    data={data}
                    color={qkvColors.query.primary}
                    cellSize={cellSize}
                    showLabels={tokens.length > 0}
                    rowLabels={tokens.length > 0 ? tokens : undefined}
                    showValues={matrixData.length <= 6}
                  />
                </Card>

                <Card className="p-4 bg-blue-50">
                  <h4 className="text-sm font-medium mb-2">Understanding Queries</h4>
                  <p className="text-sm text-gray-600 mb-2">
                    In attention, each token creates a Query vector that represents what information
                    it is looking for from other tokens.
                  </p>
                  <ul className="text-xs text-gray-600 space-y-1 list-disc list-inside">
                    <li><Latex>{'Q = X \\cdot W_Q'}</Latex> (input × weight matrix)</li>
                    <li>Dimension: (<Latex>{'\\text{seq}_{\\text{len}}, d_{\\text{model}}'}</Latex>)</li>
                    <li>Each row is a token's query vector</li>
                  </ul>
                </Card>
              </div>
            </div>
          );

        case 'keys':
          return (
            <div className="space-y-4">
              <div className="grid lg:grid-cols-2 gap-6">
                <Card className="p-4">
                  <h4 className="text-sm font-medium mb-3" style={{ color: qkvColors.key.primary }}>
                    Key Matrix (K)
                  </h4>
                  <p className="text-xs text-gray-500 mb-3">
                    Each token projects a key vector representing what it contains.
                  </p>
                  <MatrixVisualization
                    data={data}
                    color={qkvColors.key.primary}
                    cellSize={cellSize}
                    showLabels={tokens.length > 0}
                    rowLabels={tokens.length > 0 ? tokens : undefined}
                    showValues={matrixData.length <= 6}
                  />
                </Card>

                <Card className="p-4 bg-red-50">
                  <h4 className="text-sm font-medium mb-2">Understanding Keys</h4>
                  <p className="text-sm text-gray-600 mb-2">
                    Key vectors represent what information each token contains for others to query.
                    They act like "labels" or "tags" on tokens.
                  </p>
                  <ul className="text-xs text-gray-600 space-y-1 list-disc list-inside">
                    <li><Latex>{'K = X \\cdot W_K'}</Latex> (input × weight matrix)</li>
                    <li>Same dimension as Q: (<Latex>{'\\text{seq}_{\\text{len}}, d_{\\text{model}}'}</Latex>)</li>
                    <li>Used to compute attention scores</li>
                  </ul>
                </Card>
              </div>
            </div>
          );

        case 'values':
          return (
            <div className="space-y-4">
              <div className="grid lg:grid-cols-2 gap-6">
                <Card className="p-4">
                  <h4 className="text-sm font-medium mb-3" style={{ color: qkvColors.value.primary }}>
                    Value Matrix (V)
                  </h4>
                  <p className="text-xs text-gray-500 mb-3">
                    Each token's value contains the actual information to be aggregated.
                  </p>
                  <MatrixVisualization
                    data={data}
                    color={qkvColors.value.primary}
                    cellSize={cellSize}
                    showLabels={tokens.length > 0}
                    rowLabels={tokens.length > 0 ? tokens : undefined}
                    showValues={matrixData.length <= 6}
                  />
                </Card>

                <Card className="p-4 bg-green-50">
                  <h4 className="text-sm font-medium mb-2">Understanding Values</h4>
                  <p className="text-sm text-gray-600 mb-2">
                    Value vectors contain the actual information that will be extracted and combined
                    based on the attention weights.
                  </p>
                  <ul className="text-xs text-gray-600 space-y-1 list-disc list-inside">
                    <li><Latex>{'V = X \\cdot W_V'}</Latex> (input × weight matrix)</li>
                    <li>Same dimension as Q, K: (<Latex>{'\\text{seq}_{\\text{len}}, d_{\\text{model}}'}</Latex>)</li>
                    <li>Output is weighted sum of values</li>
                  </ul>
                </Card>
              </div>
            </div>
          );

        case 'dot_product':
          return (
            <div className="space-y-4">
              <Card className="p-4 bg-gradient-to-r from-blue-50 to-purple-50">
                <div className="flex items-start gap-3">
                  <Info className="w-5 h-5 text-blue-500 mt-0.5 flex-shrink-0" />
                  <div>
                    <h4 className="text-sm font-medium mb-2">Computing Attention Scores</h4>
                    <p className="text-sm text-gray-600 mb-2">
                      The dot product measures similarity between queries and keys. A higher value
                      means the query token should pay more attention to that key token.
                    </p>
                    <Latex display={true}>{'\\text{Score}[i,j] = Q_i \\cdot K_j^T'}</Latex>
                  </div>
                </div>
              </Card>

              <MatrixVisualization
                data={data}
                color="#8b5cf6"
                cellSize={cellSize}
                showLabels={tokens.length > 0}
                rowLabels={tokens.length > 0 ? tokens : undefined}
                colLabels={tokens.length > 0 ? tokens : undefined}
                showValues={matrixData.length <= 6}
                title="Attention Scores (Q·K^T)"
              />
            </div>
          );

        case 'scaled':
          return (
            <div className="space-y-4">
              <Card className="p-4 bg-purple-50">
                <h4 className="text-sm font-medium mb-2">Why Scale?</h4>
                <p className="text-sm text-gray-600 mb-2">
                  We divide by <Latex>{'\\sqrt{d_k}'}</Latex> to prevent the dot products from growing too large. Large values
                  push softmax into regions with extremely small gradients.
                </p>
                <div className="text-sm font-mono bg-white p-2 rounded flex items-center gap-2">
                  <span>Scale factor: </span>
                  <Latex display={false}>{'\\frac{1}{\\sqrt{d_k}}'}</Latex>
                  <span> = {metadata?.scale ? (1 / metadata.scale).toFixed(4) : 'N/A'}</span>
                </div>
              </Card>

              <MatrixComparison
                left={{
                  data: {
                    ...data,
                    data: matrixData.map(row => row.map(v => v * (metadata?.scale || 1))),
                  },
                  color: '#ef4444',
                  cellSize: Math.max(25, cellSize - 10),
                  showValues: false,
                }}
                right={{
                  data: data,
                  color: '#3b82f6',
                  cellSize: Math.max(25, cellSize - 10),
                  showValues: false,
                }}
                leftTitle="Before Scaling (Large values)"
                rightTitle="After Scaling (Normalized)"
                operator={<span>÷ <Latex>{'\\sqrt{d_k}'}</Latex> →</span>}
              />
            </div>
          );

        case 'masked':
          return (
            <div className="space-y-4">
              <Card className="p-4 bg-yellow-50 border-yellow-200">
                <div className="flex items-start gap-3">
                  <Info className="w-5 h-5 text-yellow-600 mt-0.5 flex-shrink-0" />
                  <div>
                    <h4 className="text-sm font-medium mb-2">Causal Masking</h4>
                    <p className="text-sm text-gray-600 mb-2">
                      In decoder attention, we mask future positions to prevent tokens from
                      "seeing" the future. This is shown as very negative values (<Latex>{'-10^9'}</Latex>).
                    </p>
                  </div>
                </div>
              </Card>

              <MatrixVisualization
                data={data}
                color="#f59e0b"
                cellSize={cellSize}
                showLabels={tokens.length > 0}
                rowLabels={tokens.length > 0 ? tokens : undefined}
                colLabels={tokens.length > 0 ? tokens : undefined}
                showValues={matrixData.length <= 6}
                highlightedCells={Array.from({ length: matrixData.length }, (_, i) =>
                  Array.from({ length: matrixData[0]?.length || 0 }, (_, j) => [i, j] as [number, number])
                ).flat().filter(([i, j]) => j > i)}
                title={<span>Masked Attention Scores (Future = <Latex>{'-\\infty'}</Latex>)</span>}
              />
            </div>
          );

        case 'softmax':
          return (
            <div className="space-y-4">
              <Card className="p-4 bg-blue-50">
                <h4 className="text-sm font-medium mb-2">Softmax Normalization</h4>
                <p className="text-sm text-gray-600 mb-2">
                  Softmax converts scores to probabilities that sum to 1. Each row now shows how much
                  each token should attend to all other tokens.
                </p>
                <Latex display={true}>{'\\text{softmax}(x_i) = \\frac{e^{x_i}}{\\sum_j e^{x_j}}'}</Latex>
              </Card>

              <div className="grid lg:grid-cols-2 gap-6">
                <Card className="p-4">
                  <h4 className="text-sm font-medium mb-3 text-gray-700">
                    Attention Weights (Probability Distribution)
                  </h4>
                  <p className="text-xs text-gray-500 mb-3">
                    Each row sums to 1.0. Darker cells = higher attention.
                  </p>
                  <MatrixVisualization
                    data={data}
                    color="#ec4899"
                    cellSize={cellSize}
                    showLabels={tokens.length > 0}
                    rowLabels={tokens.length > 0 ? tokens : undefined}
                    colLabels={tokens.length > 0 ? tokens : undefined}
                    showValues={matrixData.length <= 6}
                  />
                </Card>

                <Card className="p-4 bg-pink-50">
                  <h4 className="text-sm font-medium mb-2">Key Properties</h4>
                  <ul className="text-sm text-gray-600 space-y-2">
                    <li className="flex items-start gap-2">
                      <span className="text-pink-500 font-bold">✓</span>
                      <span>Each row sums to 1.0 (valid probability distribution)</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-pink-500 font-bold">✓</span>
                      <span>Values are in range [0, 1]</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-pink-500 font-bold">✓</span>
                      <span>Higher values = stronger attention</span>
                    </li>
                  </ul>

                  <div className="mt-4 p-3 bg-white rounded border">
                    <div className="text-xs text-gray-500 mb-1">Row sums (should all be ~1.0):</div>
                    {matrixData.slice(0, 5).map((row, i) => (
                      <div key={i} className="text-xs font-mono">
                        Row {i}: {row.reduce((a, b) => a + b, 0).toFixed(6)}
                      </div>
                    ))}
                  </div>
                </Card>
              </div>
            </div>
          );

        case 'weighted_sum':
          return (
            <div className="space-y-4">
              <Card className="p-4 bg-green-50 border-green-200">
                <div className="flex items-start gap-3">
                  <Info className="w-5 h-5 text-green-600 mt-0.5 flex-shrink-0" />
                  <div>
                    <h4 className="text-sm font-medium mb-2">Final Output</h4>
                    <p className="text-sm text-gray-600 mb-2">
                      The output is a weighted sum of value vectors, where the weights are the
                      attention probabilities. This aggregates information from all tokens based on
                      relevance.
                    </p>
                    <Latex display={true}>{'\\text{Output}_i = \\sum_j \\text{Attention}[i,j] \\cdot V_j'}</Latex>
                  </div>
                </div>
              </Card>

              <div className="grid lg:grid-cols-2 gap-6">
                <Card className="p-4">
                  <h4 className="text-sm font-medium mb-3" style={{ color: qkvColors.value.primary }}>
                    Output Matrix
                  </h4>
                  <p className="text-xs text-gray-500 mb-3">
                    Final attention output for each token.
                  </p>
                  <MatrixVisualization
                    data={data}
                    color={qkvColors.value.primary}
                    cellSize={cellSize}
                    showLabels={tokens.length > 0}
                    rowLabels={tokens.length > 0 ? tokens : undefined}
                    showValues={matrixData.length <= 6}
                  />
                </Card>

                <Card className="p-4 bg-gray-50">
                  <h4 className="text-sm font-medium mb-2">Summary</h4>
                  <div className="space-y-3 text-sm text-gray-600">
                    <div className="p-3 bg-white rounded border">
                      <div className="font-medium text-gray-700 mb-1">Input Dimension</div>
                      <div className="font-mono text-xs">
                        ({attentionData.queries.shape.join(', ')})
                      </div>
                    </div>
                    <div className="p-3 bg-white rounded border">
                      <div className="font-medium text-gray-700 mb-1">Output Dimension</div>
                      <div className="font-mono text-xs">({data.shape.join(', ')})</div>
                    </div>
                    <div className="p-3 bg-white rounded border">
                      <div className="font-medium text-gray-700 mb-1">Operation</div>
                      <div className="text-xs">
                        <Latex>{'\\text{Output} = \\text{AttentionWeights} \\cdot V'}</Latex>
                      </div>
                    </div>
                  </div>
                </Card>
              </div>
            </div>
          );

        default:
          return null;
      }
    };

    return (
      <motion.div
        key={currentStage}
        className="stage-content space-y-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        transition={{ duration: 0.3 }}
      >
        {/* Stage Header */}
        <div className="flex items-center gap-3">
          <Badge
            className="px-3 py-1 text-white"
            style={{ backgroundColor: stageColor }}
          >
            Stage {currentStageIndex + 1}
          </Badge>
          <h2 className="text-2xl font-bold">{title}</h2>
        </div>

        {/* Stage Description */}
        <p className="text-gray-600">{description}</p>

        {/* Mathematical Formula */}
        <Card className="p-4 bg-gradient-to-r from-blue-50 to-purple-50 border-blue-200">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-sm font-medium text-gray-500">Formula:</span>
          </div>
          <Latex display={true}>{formula}</Latex>
        </Card>

        {/* Stage-specific Visualization */}
        {renderStageVisualization()}
      </motion.div>
    );
  };

  return (
    <div ref={containerRef} className="staged-attention-visualization space-y-6">
      {/* Progress Bar */}
      <div className="relative">
        <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500"
            initial={{ width: 0 }}
            animate={{ width: `${((currentStageIndex + 1) / allStages.length) * 100}%` }}
            transition={{ duration: 0.3 }}
          />
        </div>
        <div className="flex justify-between mt-2">
          <span className="text-xs text-gray-500">
            Stage {currentStageIndex + 1} of {allStages.length}
          </span>
          <span className="text-xs text-gray-500">
            {Math.round(((currentStageIndex + 1) / allStages.length) * 100)}%
          </span>
        </div>
      </div>

      {/* Stage Navigation */}
      <div className="flex flex-wrap gap-2">
        {allStages.map((stage, index) => (
          <Button
            key={stage.stage}
            variant={stage.isActive ? 'default' : 'outline'}
            size="sm"
            onClick={() => goToStage(stage.stage)}
            className="relative"
            style={{
              backgroundColor: stage.isActive ? getStageColor(stage.stage) : undefined,
              borderColor: stage.isCompleted ? getStageColor(stage.stage) : undefined,
            }}
          >
            {stage.isCompleted && (
              <span className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full flex items-center justify-center">
                <span className="text-white text-xs">✓</span>
              </span>
            )}
            {index + 1}. {stage.stage}
          </Button>
        ))}
      </div>

      {/* Playback Controls */}
      <div className="flex items-center gap-2 p-3 bg-gray-50 rounded-lg">
        <Button
          variant="outline"
          size="sm"
          onClick={handlePlayPause}
          disabled={!hasNextStage && !isPlaying}
        >
          {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={handleReset}
        >
          <RotateCcw className="w-4 h-4" />
        </Button>
        <div className="flex items-center gap-2 ml-4">
          <span className="text-xs text-gray-500">Speed:</span>
          <select
            value={autoPlaySpeed}
            onChange={(e) => setAutoPlaySpeed(Number(e.target.value))}
            className="text-xs border rounded px-2 py-1"
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
            onClick={goToPreviousStage}
            disabled={!hasPreviousStage}
          >
            <ChevronLeft className="w-4 h-4 mr-1" />
            Previous
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={goToNextStage}
            disabled={!hasNextStage}
          >
            Next
            <ChevronRight className="w-4 h-4 ml-1" />
          </Button>
        </div>
      </div>

      {/* Stage Content */}
      <Card className="p-6">
        <AnimatePresence mode="wait">
          {renderComputationStep()}
        </AnimatePresence>
      </Card>

      {/* Token Context */}
      {tokens.length > 0 && (
        <Card className="p-4 bg-gray-50">
          <h4 className="text-sm font-medium mb-2">Input Tokens</h4>
          <div className="flex flex-wrap gap-2">
            {tokens.map((token, index) => (
              <span
                key={index}
                className="px-2 py-1 bg-white border rounded text-sm"
              >
                {token}
              </span>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
};
