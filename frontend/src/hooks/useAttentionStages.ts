import { useMemo } from 'react';
import { useTransformerStore } from '../stores/transformerStore';
import type {
  AttentionStage,
  AttentionComputationStep,
  TensorData,
  AttentionData,
} from '../types/transformer';

/**
 * Attention computation stages in order
 */
const ATTENTION_STAGES: AttentionStage[] = [
  'queries',
  'keys',
  'values',
  'dot_product',
  'scaled',
  'masked',
  'softmax',
  'weighted_sum',
];

/**
 * Stage information for each attention computation step
 */
const STAGE_INFO: Record<AttentionStage, {
  title: string;
  description: string;
  formula: string;
  nextStage?: AttentionStage;
  prevStage?: AttentionStage;
}> = {
  queries: {
    title: 'Query Matrices (Q)',
    description: 'Query vectors represent what each token is "looking for" in other tokens.',
    formula: 'Q = X W_Q',
    nextStage: 'keys',
  },
  keys: {
    title: 'Key Matrices (K)',
    description: 'Key vectors represent what each token "contains" for other tokens to query.',
    formula: 'K = X W_K',
    nextStage: 'values',
    prevStage: 'queries',
  },
  values: {
    title: 'Value Matrices (V)',
    description: 'Value vectors contain the actual information that will be aggregated.',
    formula: 'V = X W_V',
    nextStage: 'dot_product',
    prevStage: 'keys',
  },
  dot_product: {
    title: 'Dot Product (Q·K^T)',
    description: 'Compute raw attention scores by taking the dot product of queries and keys.',
    formula: 'Score = Q K^T',
    nextStage: 'scaled',
    prevStage: 'values',
  },
  scaled: {
    title: 'Scaled Dot Product',
    description: 'Scale the scores to prevent vanishing gradients in softmax.',
    formula: '\\text{Scaled} = \\frac{Q K^T}{\\sqrt{d_k}}',
    nextStage: 'masked',
    prevStage: 'dot_product',
  },
  masked: {
    title: 'Masked Attention',
    description: 'Apply causal mask to prevent attending to future tokens (in decoder).',
    formula: '\\text{Masked} = \\text{Scaled} \\odot M',
    nextStage: 'softmax',
    prevStage: 'scaled',
  },
  softmax: {
    title: 'Softmax Normalization',
    description: 'Normalize scores to create attention weights that sum to 1.',
    formula: '\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}}\\right)',
    nextStage: 'weighted_sum',
    prevStage: 'masked',
  },
  weighted_sum: {
    title: 'Weighted Sum (Output)',
    description: 'Compute final output as weighted sum of value vectors.',
    formula: '\\text{Output} = \\text{Attention} \\cdot V',
    prevStage: 'softmax',
  },
};

/**
 * Hook for managing attention computation stages
 *
 * Provides:
 * - Current stage information
 * - Navigation between stages
 * - Computation steps data
 * - Stage progression utilities
 */
export const useAttentionStages = (attentionData?: AttentionData) => {
  const { attentionComputationStage, setAttentionComputationStage } = useTransformerStore();

  /**
   * Get current stage information
   */
  const currentStageInfo = useMemo(() => {
    return STAGE_INFO[attentionComputationStage];
  }, [attentionComputationStage]);

  /**
   * Get current stage index
   */
  const currentStageIndex = useMemo(() => {
    return ATTENTION_STAGES.indexOf(attentionComputationStage);
  }, [attentionComputationStage]);

  /**
   * Check if we can go to next stage
   */
  const hasNextStage = useMemo(() => {
    return currentStageInfo?.nextStage !== undefined;
  }, [currentStageInfo]);

  /**
   * Check if we can go to previous stage
   */
  const hasPreviousStage = useMemo(() => {
    return currentStageInfo?.prevStage !== undefined;
  }, [currentStageInfo]);

  /**
   * Navigate to next stage
   */
  const goToNextStage = () => {
    if (hasNextStage && currentStageInfo?.nextStage) {
      setAttentionComputationStage(currentStageInfo.nextStage);
    }
  };

  /**
   * Navigate to previous stage
   */
  const goToPreviousStage = () => {
    if (hasPreviousStage && currentStageInfo?.prevStage) {
      setAttentionComputationStage(currentStageInfo.prevStage);
    }
  };

  /**
   * Navigate to specific stage
   */
  const goToStage = (stage: AttentionStage) => {
    setAttentionComputationStage(stage);
  };

  /**
   * Reset to first stage
   */
  const resetToFirstStage = () => {
    setAttentionComputationStage('queries');
  };

  /**
   * Compute step data for current stage
   */
  const computationStep = useMemo<AttentionComputationStep | null>(() => {
    if (!attentionData) return null;

    let { queries, keys, values, scale } = attentionData;

    // Validate data - if queries/keys/values are invalid, generate synthetic data
    const qData = queries.data as number[][];
    const hasValidData = qData && qData.length > 0 && qData[0] && qData[0].length > 0 &&
                         Array.isArray(qData[0]) && typeof qData[0][0] === 'number';

    if (!hasValidData) {
      // Generate synthetic data for visualization
      const seqLen = 5; // 5 tokens
      const headDim = 8; // 8 dimensions per head
      const synthetic = generateSyntheticAttentionData(seqLen, headDim);
      queries = synthetic.queries;
      keys = synthetic.keys;
      values = synthetic.values;
      scale = Math.sqrt(headDim);
    }

    // Pre-compute all values once to avoid scope issues
    let dotProductData: TensorData | null = null;
    let scaledData: TensorData | null = null;
    let maskedData: TensorData | null = null;

    // Only compute if needed (for stages that depend on previous computations)
    if (attentionComputationStage === 'dot_product' ||
        attentionComputationStage === 'scaled' ||
        attentionComputationStage === 'masked' ||
        attentionComputationStage === 'softmax' ||
        attentionComputationStage === 'weighted_sum') {
      dotProductData = computeDotProduct(queries, keys);
      scaledData = computeScaled(dotProductData, scale);
      maskedData = computeMask(scaledData);
    }

    switch (attentionComputationStage) {
      case 'queries':
        return {
          stage: 'queries',
          title: STAGE_INFO.queries.title,
          description: STAGE_INFO.queries.description,
          formula: STAGE_INFO.queries.formula,
          data: queries,
        };

      case 'keys':
        return {
          stage: 'keys',
          title: STAGE_INFO.keys.title,
          description: STAGE_INFO.keys.description,
          formula: STAGE_INFO.keys.formula,
          data: keys,
        };

      case 'values':
        return {
          stage: 'values',
          title: STAGE_INFO.values.title,
          description: STAGE_INFO.values.description,
          formula: STAGE_INFO.values.formula,
          data: values,
        };

      case 'dot_product':
        return {
          stage: 'dot_product',
          title: STAGE_INFO.dot_product.title,
          description: STAGE_INFO.dot_product.description,
          formula: STAGE_INFO.dot_product.formula,
          data: dotProductData!,
        };

      case 'scaled':
        return {
          stage: 'scaled',
          title: STAGE_INFO.scaled.title,
          description: STAGE_INFO.scaled.description,
          formula: STAGE_INFO.scaled.formula,
          data: scaledData!,
          metadata: { scale },
        };

      case 'masked':
        return {
          stage: 'masked',
          title: STAGE_INFO.masked.title,
          description: STAGE_INFO.masked.description,
          formula: STAGE_INFO.masked.formula,
          data: maskedData!,
          metadata: { maskValue: -1e9 },
        };

      case 'softmax': {
        const softmaxData = computeSoftmax(maskedData!);
        return {
          stage: 'softmax',
          title: STAGE_INFO.softmax.title,
          description: STAGE_INFO.softmax.description,
          formula: STAGE_INFO.softmax.formula,
          data: softmaxData,
          metadata: { temperature: 1.0 },
        };
      }

      case 'weighted_sum': {
        const finalSoftmax = computeSoftmax(maskedData!);
        const weightedSum = computeWeightedSum(finalSoftmax, values);
        return {
          stage: 'weighted_sum',
          title: STAGE_INFO.weighted_sum.title,
          description: STAGE_INFO.weighted_sum.description,
          formula: STAGE_INFO.weighted_sum.formula,
          data: weightedSum,
        };
      }

      default:
        return null;
    }
  }, [attentionData, attentionComputationStage]);

  /**
   * Get all stages for navigation UI
   */
  const allStages = useMemo(() => {
    return ATTENTION_STAGES.map((stage) => ({
      stage,
      ...STAGE_INFO[stage],
      isActive: stage === attentionComputationStage,
      isCompleted: ATTENTION_STAGES.indexOf(stage) < currentStageIndex,
    }));
  }, [attentionComputationStage, currentStageIndex]);

  return {
    // Current state
    currentStage: attentionComputationStage,
    currentStageInfo,
    currentStageIndex,
    computationStep,

    // Navigation
    hasNextStage,
    hasPreviousStage,
    goToNextStage,
    goToPreviousStage,
    goToStage,
    resetToFirstStage,

    // All stages
    allStages,
    stages: ATTENTION_STAGES,
  };
};

// ============================================
// HELPER FUNCTIONS FOR ATTENTION COMPUTATION
// ============================================

/**
 * Compute dot product Q·K^T
 */
function computeDotProduct(queries: TensorData, keys: TensorData): TensorData {
  const qData = queries.data as number[][];
  const kData = keys.data as number[][];

  const seqLen = qData.length;
  const headDim = qData[0]?.length || 0;

  const result: number[][] = [];
  for (let i = 0; i < seqLen; i++) {
    result[i] = [];
    const qRow = qData[i];
    if (!qRow) {
      // If query row is undefined, fill with zeros
      result[i] = new Array(seqLen).fill(0);
      continue;
    }

    for (let j = 0; j < seqLen; j++) {
      const kRow = kData[j];
      if (!kRow) {
        // If key row is undefined, use 0
        result[i][j] = 0;
        continue;
      }

      let sum = 0;
      for (let k = 0; k < headDim; k++) {
        const qVal = qRow[k];
        const kVal = kRow[k];

        // Only add to sum if both values are valid numbers
        if (typeof qVal === 'number' && typeof kVal === 'number' && !isNaN(qVal) && !isNaN(kVal)) {
          sum += qVal * kVal;
        }
      }
      result[i][j] = sum;
    }
  }

  return {
    shape: [seqLen, seqLen],
    data: result,
    dtype: 'float32',
    name: 'dot_product',
  };
}

/**
 * Generate synthetic attention data for visualization when real data is not available
 */
function generateSyntheticAttentionData(seqLen: number, headDim: number): {
  queries: TensorData;
  keys: TensorData;
  values: TensorData;
} {
  // Generate random but deterministic Q, K, V matrices
  const queries: number[][] = [];
  const keys: number[][] = [];
  const values: number[][] = [];

  for (let i = 0; i < seqLen; i++) {
    const qRow: number[] = [];
    const kRow: number[] = [];
    const vRow: number[] = [];

    for (let j = 0; j < headDim; j++) {
      // Use deterministic pseudo-random values based on position
      qRow.push(Math.sin((i * 0.5 + j * 0.3)) * 2 + Math.cos(j * 0.7));
      kRow.push(Math.cos((i * 0.3 + j * 0.5)) * 2 + Math.sin(j * 0.4));
      vRow.push(Math.sin((i * 0.4 + j * 0.6)) * 1.5 + Math.cos(j * 0.5));
    }

    queries.push(qRow);
    keys.push(kRow);
    values.push(vRow);
  }

  return {
    queries: {
      shape: [seqLen, headDim],
      data: queries,
      dtype: 'float32',
      name: 'queries',
    },
    keys: {
      shape: [seqLen, headDim],
      data: keys,
      dtype: 'float32',
      name: 'keys',
    },
    values: {
      shape: [seqLen, headDim],
      data: values,
      dtype: 'float32',
      name: 'values',
    },
  };
}

/**
 * Compute scaled dot product (Q·K^T) / sqrt(d_k)
 */
function computeScaled(dotProduct: TensorData | null, scale: number): TensorData {
  if (!dotProduct) {
    throw new Error('dotProduct is required for computeScaled');
  }
  const data = dotProduct.data as number[][];
  const scaled: number[][] = data.map(row =>
    row.map(val => val / scale)
  );

  return {
    ...dotProduct,
    data: scaled,
    name: 'scaled',
  };
}

/**
 * Apply causal mask to attention scores
 */
function computeMask(scaled: TensorData | null): TensorData {
  if (!scaled) {
    throw new Error('scaled is required for computeMask');
  }
  const data = scaled.data as number[][];

  const masked: number[][] = data.map((row, i) =>
    row.map((val, j) => j > i ? -1e9 : val)
  );

  return {
    ...scaled,
    data: masked,
    name: 'masked',
  };
}

/**
 * Apply softmax normalization
 */
function computeSoftmax(scores: TensorData): TensorData {
  const data = scores.data as number[][];

  const softmaxData: number[][] = data.map(row => {
    // Filter out invalid values
    const validRow = row.filter(val => typeof val === 'number' && !isNaN(val) && isFinite(val));

    if (validRow.length === 0) {
      // If no valid values, return uniform distribution
      return row.map(() => 1 / row.length);
    }

    // Find max for numerical stability
    const max = Math.max(...validRow);

    // Subtract max and exponentiate
    const exp = validRow.map(val => Math.exp(val - max));

    // Sum and normalize
    const sum = exp.reduce((a, b) => a + b, 0);

    if (sum === 0) {
      // If sum is zero, return uniform distribution
      return row.map(() => 1 / row.length);
    }

    const normalizedExp = exp.map(val => val / sum);

    // Map back to original row structure
    let validIndex = 0;
    return row.map(val => {
      if (typeof val === 'number' && !isNaN(val) && isFinite(val)) {
        return normalizedExp[validIndex++];
      }
      return 0; // Invalid values get 0 probability
    });
  });

  return {
    ...scores,
    data: softmaxData,
    name: 'softmax',
  };
}

/**
 * Compute weighted sum: attention_weights · V
 */
function computeWeightedSum(attentionWeights: TensorData, values: TensorData): TensorData {
  const weights = attentionWeights.data as number[][];
  const vData = values.data as number[][];

  const seqLen = weights.length;
  const headDim = vData[0]?.length || 0;

  // Compute weighted sum
  const result: number[][] = [];
  for (let i = 0; i < seqLen; i++) {
    result[i] = [];
    const weightRow = weights[i];

    if (!weightRow) {
      // If weight row is undefined, fill with zeros
      result[i] = new Array(headDim).fill(0);
      continue;
    }

    for (let k = 0; k < headDim; k++) {
      let sum = 0;
      for (let j = 0; j < seqLen; j++) {
        const weight = weightRow[j];
        const vRow = vData[j];

        // Skip if weight is invalid or value row is undefined
        if (typeof weight !== 'number' || isNaN(weight) || !vRow) {
          continue;
        }

        const vVal = vRow[k];

        // Only add if value is also a valid number
        if (typeof vVal === 'number' && !isNaN(vVal)) {
          sum += weight * vVal;
        }
      }
      result[i][k] = sum;
    }
  }

  return {
    shape: [seqLen, headDim],
    data: result,
    dtype: 'float32',
    name: 'weighted_sum',
  };
}
