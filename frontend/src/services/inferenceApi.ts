/**
 * API service for inference and probability distribution
 */

import api from './api';
import type {
  ProbabilityDistributionData,
  SamplingOptions,
  SamplingStrategy,
  StrategyComparison,
  PositionDistribution,
  GenerationStep,
  TopKFilterResult,
  TopPFilterResult,
  TemperatureScalingResult,
  BeamSearchStep,
} from '../types/inference';
import type { TransformerConfig, TransformerInput } from '../types/transformer';

const INFERENCE_API_BASE = '/api/v1/inference';

/**
 * Convert camelCase to snake_case for backend
 */
function toSnakeCase(obj: any): any {
  if (obj === null || typeof obj !== 'object') {
    return obj;
  }

  if (Array.isArray(obj)) {
    return obj.map(toSnakeCase);
  }

  const result: any = {};
  for (const key in obj) {
    if (obj.hasOwnProperty(key)) {
      const snakeKey = key.replace(/([A-Z])/g, '_$1').toLowerCase();
      result[snakeKey] = toSnakeCase(obj[key]);
    }
  }
  return result;
}

/**
 * Generate probability distribution for input sequence
 */
export async function generateProbabilityDistribution(
  config: TransformerConfig,
  input: TransformerInput,
  options: Partial<SamplingOptions> = {}
): Promise<ProbabilityDistributionData> {
  const response = await api.post(`${INFERENCE_API_BASE}/probability-distribution`, {
    config: toSnakeCase(config),
    input_data: toSnakeCase(input),
    options: toSnakeCase(options),
  });
  return response.data;
}

/**
 * Sample a token using specified strategy
 */
export async function sampleToken(
  config: TransformerConfig,
  input: TransformerInput,
  options: SamplingOptions
): Promise<GenerationStep> {
  const response = await api.post(`${INFERENCE_API_BASE}/sample`, {
    config: toSnakeCase(config),
    input_data: toSnakeCase(input),
    options: toSnakeCase(options),
  });
  return response.data;
}

/**
 * Generate sequence autoregressively
 */
export async function generateSequence(
  config: TransformerConfig,
  input: TransformerInput,
  options: Partial<SamplingOptions> = {},
  maxLength: number = 50
): Promise<{
  steps: GenerationStep[];
  finalSequence: string;
  totalTime: number;
}> {
  const response = await api.post(`${INFERENCE_API_BASE}/generate`, {
    config: toSnakeCase(config),
    input_data: toSnakeCase(input),
    options: toSnakeCase(options),
    max_length: maxLength,
  });
  return response.data;
}

/**
 * Compare different sampling strategies
 */
export async function compareStrategies(
  config: TransformerConfig,
  input: TransformerInput,
  strategies: SamplingStrategy[],
  maxLength: number = 20
): Promise<StrategyComparison[]> {
  const response = await api.post(`${INFERENCE_API_BASE}/compare-strategies`, {
    config: toSnakeCase(config),
    input_data: toSnakeCase(input),
    strategies,
    max_length: maxLength,
  });
  return response.data;
}

/**
 * Apply top-k filtering to distribution
 */
export async function applyTopKFilter(
  distribution: PositionDistribution,
  k: number
): Promise<TopKFilterResult> {
  const response = await api.post(`${INFERENCE_API_BASE}/filter/top-k`, {
    distribution,
    k,
  });
  return response.data;
}

/**
 * Apply top-p (nucleus) filtering to distribution
 */
export async function applyTopPFilter(
  distribution: PositionDistribution,
  p: number
): Promise<TopPFilterResult> {
  const response = await api.post(`${INFERENCE_API_BASE}/filter/top-p`, {
    distribution,
    p,
  });
  return response.data;
}

/**
 * Apply temperature scaling to distribution
 */
export async function applyTemperatureScaling(
  distribution: PositionDistribution,
  temperature: number
): Promise<TemperatureScalingResult> {
  const response = await api.post(`${INFERENCE_API_BASE}/scale/temperature`, {
    distribution,
    temperature,
  });
  return response.data;
}

/**
 * Perform beam search
 */
export async function beamSearch(
  config: TransformerConfig,
  input: TransformerInput,
  beamWidth: number,
  maxLength: number = 20
): Promise<{
  steps: BeamSearchStep[];
  finalSequence: string;
  bestScore: number;
}> {
  const response = await api.post(`${INFERENCE_API_BASE}/beam-search`, {
    config: toSnakeCase(config),
    input_data: toSnakeCase(input),
    beam_width: beamWidth,
    max_length: maxLength,
  });
  return response.data;
}

/**
 * Get distribution statistics
 */
export async function getDistributionStatistics(
  distribution: PositionDistribution
): Promise<{
  entropy: number;
  perplexity: number;
  effectiveVocabSize: number;
  topKCoverage: Map<number, number>;
}> {
  // Calculate client-side for now
  const tokens = distribution.tokens;
  const probs = tokens.map(t => t.probability);

  // Entropy: H = -sum(p * log(p))
  const entropy = -probs.reduce((sum, p) => {
    return p > 0 ? sum + p * Math.log2(p) : sum;
  }, 0);

  // Perplexity: 2^H
  const perplexity = Math.pow(2, entropy);

  // Effective vocab size: exp(entropy) for natural log, or 2^entropy for log base 2
  const effectiveVocabSize = Math.pow(2, entropy);

  // Top-k coverage
  const topKCoverage = new Map<number, number>();
  for (const k of [1, 5, 10, 50, 100, 500, 1000]) {
    const topK = tokens.slice(0, Math.min(k, tokens.length));
    const cumulativeProb = topK.reduce((sum, t) => sum + t.probability, 0);
    topKCoverage.set(k, cumulativeProb);
  }

  return {
    entropy,
    perplexity,
    effectiveVocabSize,
    topKCoverage,
  };
}

/**
 * Demo/mock data generator for development
 */
export function generateMockProbabilityDistribution(
  sequence: string = 'The quick brown fox',
  vocabSize: number = 10000
): ProbabilityDistributionData {
  const tokens = sequence.split(' ');
  const distributions: PositionDistribution[] = [];
  const steps: GenerationStep[] = [];

  let currentContext = '<sos>';

  tokens.forEach((token, position) => {
    // Generate mock probabilities
    const mockTokens: any[] = [];
    const actualTokenIndex = Math.floor(Math.random() * 50); // Put actual token in top 50

    for (let i = 0; i < Math.min(vocabSize, 100); i++) {
      const isActualToken = i === actualTokenIndex;
      // Higher probability for actual token
      const baseProb = isActualToken ? 0.3 : Math.random() * 0.01;
      mockTokens.push({
        tokenId: i,
        token: i === actualTokenIndex ? token : `token_${i}`,
        probability: baseProb,
        logProbability: Math.log(baseProb + 1e-10),
        rank: i + 1,
      });
    }

    // Sort by probability
    mockTokens.sort((a, b) => b.probability - a.probability);

    // Renormalize
    const sum = mockTokens.reduce((s, t) => s + t.probability, 0);
    mockTokens.forEach(t => t.probability /= sum);

    // Update ranks
    mockTokens.forEach((t, i) => t.rank = i + 1);

    const entropy = -mockTokens.reduce((sum, t) => {
      return sum + t.probability * Math.log2(t.probability + 1e-10);
    }, 0);

    const distribution: PositionDistribution = {
      position,
      tokens: mockTokens,
      entropy,
      topToken: mockTokens[0],
      topKTokens: mockTokens.slice(0, 10),
      cumulativeProbability: mockTokens.slice(0, 10).reduce((s, t) => s + t.probability, 0),
    };

    distributions.push(distribution);

    // Create generation step
    const step: GenerationStep = {
      stepIndex: position,
      position,
      generatedToken: {
        tokenId: actualTokenIndex,
        token,
        probability: mockTokens[actualTokenIndex]?.probability || mockTokens[0].probability,
        chosenStrategy: 'greedy',
        alternatives: mockTokens.slice(0, 5),
      },
      distribution,
      context: currentContext,
      timeTaken: Math.random() * 100 + 50,
    };

    steps.push(step);
    currentContext += ` ${token}`;
  });

  return {
    sequence,
    tokens,
    distributions,
    samplingOptions: {
      strategy: 'greedy',
      temperature: 1.0,
      topK: 50,
      topP: 0.9,
      beamWidth: 5,
    },
    generatedSequence: sequence,
    steps,
  };
}

export const inferenceApi = {
  generateProbabilityDistribution,
  sampleToken,
  generateSequence,
  compareStrategies,
  applyTopKFilter,
  applyTopPFilter,
  applyTemperatureScaling,
  beamSearch,
  getDistributionStatistics,
  generateMockProbabilityDistribution,
};
