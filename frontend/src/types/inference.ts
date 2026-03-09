/**
 * Types for inference and probability distribution visualization
 */

import type { TensorData } from './transformer';

/**
 * Sampling strategy types
 */
export type SamplingStrategy = 'greedy' | 'multinomial' | 'top-k' | 'top-p' | 'beam-search';

/**
 * Token probability data
 */
export interface TokenProbability {
  tokenId: number;
  token: string;
  probability: number;
  logProbability: number;
  rank: number;
  embedding?: TensorData;  // Optional embedding vector for visualization
}

/**
 * Probability distribution at a specific position
 */
export interface PositionDistribution {
  position: number;
  tokens: TokenProbability[];
  entropy: number;
  topToken: TokenProbability;
  topKTokens: TokenProbability[];
  cumulativeProbability: number;
}

/**
 * Sampling options
 */
export interface SamplingOptions {
  strategy: SamplingStrategy;
  temperature: number;
  topK: number;
  topP: number;
  beamWidth: number;
  seed?: number;
}

/**
 * Sampling result
 */
export interface SamplingResult {
  tokenId: number;
  token: string;
  probability: number;
  chosenStrategy: SamplingStrategy;
  alternatives: TokenProbability[];
}

/**
 * Generation step for autoregressive visualization
 */
export interface GenerationStep {
  stepIndex: number;
  position: number;
  generatedToken: SamplingResult;
  distribution: PositionDistribution;
  context: string;
  timeTaken: number;
}

/**
 * Probability distribution visualization data
 */
export interface ProbabilityDistributionData {
  sequence: string;
  tokens: string[];
  distributions: PositionDistribution[];
  samplingOptions: SamplingOptions;
  generatedSequence: string;
  steps: GenerationStep[];
}

/**
 * Comparison data for different sampling strategies
 */
export interface StrategyComparison {
  strategy: SamplingStrategy;
  options: SamplingOptions;
  result: string;
  steps: GenerationStep[];
  timeMs: number;
}

/**
 * Top-K filter result
 */
export interface TopKFilterResult {
  k: number;
  originalCount: number;
  filteredCount: number;
  filteredTokens: TokenProbability[];
  remainingProbability: number;
}

/**
 * Top-P (nucleus) filter result
 */
export interface TopPFilterResult {
  p: number;
  originalCount: number;
  filteredCount: number;
  filteredTokens: TokenProbability[];
  remainingProbability: number;
  threshold: number;
}

/**
 * Temperature scaling result
 */
export interface TemperatureScalingResult {
  temperature: number;
  originalDistribution: TokenProbability[];
  scaledDistribution: TokenProbability[];
  entropyChange: number;
}

/**
 * Beam search candidate
 */
export interface BeamCandidate {
  sequence: number[];
  tokens: string[];
  logProbability: number;
  score: number; // normalized by length
  completed: boolean;
}

/**
 * Beam search step
 */
export interface BeamSearchStep {
  step: number;
  candidates: BeamCandidate[];
  pruned: BeamCandidate[];
}

/**
 * Interactive visualization state
 */
export interface ProbabilityVizState {
  selectedPosition: number;
  showTopK: number;
  showTopP: number;
  temperature: number;
  highlightTokens: number[];
  showLogProbs: boolean;
  sortBy: 'probability' | 'token' | 'id';
}

/**
 * Distribution statistics
 */
export interface DistributionStatistics {
  entropy: number;
  perplexity: number;
  maxProbability: number;
  minProbability: number;
  avgProbability: number;
  topKCoverage: Map<number, number>; // k -> cumulative probability
  effectiveVocabSize: number;
}

/**
 * Visualization color scheme
 */
export interface ProbabilityColorScheme {
  highProb: string; // color for high probability tokens
  mediumProb: string;
  lowProb: string;
  selectedToken: string;
  highlightedToken: string;
  gradient: string[]; // for continuous probability visualization
}
