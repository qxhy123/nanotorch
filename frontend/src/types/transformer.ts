/**
 * Core types for Transformer visualization
 */

export interface TensorData {
  shape: number[];
  data: number[] | number[][] | number[][][] | number[][][][];
  dtype: string;
  name?: string;
}

export interface AttentionData {
  weights: TensorData; // (batch, heads, seq_len, seq_len) or (batch, seq_len, seq_len)
  queries: TensorData; // (batch, heads, seq_len, head_dim)
  keys: TensorData;
  values: TensorData;
  scale: number;
}

export interface LayerOutputData {
  layerName: string;
  layerType: 'embedding' | 'positional_encoding' | 'attention' | 'feedforward' | 'normalization' | 'dropout' | 'linear';
  inputShape: number[];
  outputShape: number[];
  output: TensorData;
  parameters?: Record<string, TensorData>;
  attentionData?: AttentionData;
}

export interface TransformerConfig {
  d_model: number;              // 512
  nhead: number;                // 8
  num_encoder_layers: number;   // 6
  num_decoder_layers: number;   // 6
  dim_feedforward: number;      // 2048
  dropout: number;              // 0.1
  activation: 'relu' | 'gelu';
  max_seq_len: number;          // 128
  vocab_size: number;           // 10000
  layer_norm_eps: number;       // 1e-5
  batch_first: boolean;         // true
  norm_first: boolean;          // false
}

export interface TransformerInput {
  text: string;
  tokens?: number[];
  targetText?: string;
  targetTokens?: number[];
  src?: TensorData;
  tgt?: TensorData;
}

export interface TransformerForwardOptions {
  returnAttention?: boolean;
  returnAllLayers?: boolean;
  returnEmbeddings?: boolean;
}

export interface TransformerOutput {
  success: boolean;
  data?: {
    finalOutput: TensorData;
    final_output?: TensorData;
    steps?: LayerOutputData[];
    attentionWeights?: AttentionData[];
    attention_weights?: AttentionData[];
    layerOutputs?: LayerOutputData[];
    layer_outputs?: LayerOutputData[];
    metadata?: Record<string, unknown>;
    embeddings?: {
      tokenEmbeddings: TensorData;
      positionalEncodings: TensorData;
      combined: TensorData;
    };
    targetEmbeddings?: {
      tokenEmbeddings: TensorData;
      positionalEncodings: TensorData;
      combined: TensorData;
    };
  };
  error?: string;
}

export interface ComponentVisualizationState {
  selectedLayer: number;
  selectedHead: number;
  selectedComponent: string;
  showValues: boolean;
  colorScheme: 'viridis' | 'plasma' | 'inferno' | 'blues' | 'reds';
}

export interface PositionalEncodingData {
  encodings: TensorData;
  maxSeqLen: number;
  dModel: number;
}

export interface EmbeddingVisualizationData {
  tokens: string[];
  tokenIds: number[];
  embeddings: TensorData;
  positionalEncodings: TensorData;
  combinedEmbeddings: TensorData;
}

export interface AttentionVisualizationData {
  weights: number[][][]; // [heads][seq_len][seq_len]
  queries: number[][][]; // [heads][seq_len][head_dim]
  keys: number[][][];
  values: number[][][];
  tokens: string[];
  scale: number;
}

export interface FeedForwardData {
  input: TensorData;
  afterLinear1: TensorData;
  afterActivation: TensorData;
  afterLinear2: TensorData;
  output: TensorData;
  intermediateDim: number;
  outputDim: number;
  activation: string;
}

export interface NormalizationData {
  input: TensorData;
  mean: number[];
  std: number[];
  normalized: TensorData;
  afterScale: TensorData;
  afterShift: TensorData;
  gamma: number[];
  beta: number[];
  eps: number;
}

export type StepType = 'embedding' | 'positional_encoding' | 'self_attention' | 'cross_attention' | 'feed_forward' | 'layer_norm' | 'output';

export interface TransformerStep {
  id: string;
  type: StepType;
  name: string;
  description: string;
  layerIndex: number;
  data: LayerOutputData;
}

export interface AnimationState {
  isPlaying: boolean;
  currentStep: number;
  speed: number; // milliseconds per step
}

// ============================================
// NEW TYPES FOR TRANSFORMER VISUALIZATION 2.0
// ============================================

/**
 * Attention computation stages for step-by-step visualization
 */
export type AttentionStage =
  | 'queries'        // Q matrices
  | 'keys'           // K matrices
  | 'values'         // V matrices
  | 'dot_product'    // Q·K^T raw scores
  | 'scaled'         // Q·K^T / sqrt(d_k)
  | 'masked'         // After causal mask applied
  | 'softmax'        // After softmax normalization
  | 'weighted_sum';  // Final attention output (softmax · V)

/**
 * Individual computation step data for staged attention visualization
 */
export interface AttentionComputationStep {
  stage: AttentionStage;
  title: string;
  description: string;
  formula: string;
  data: TensorData;
  highlightedElements?: {
    rows?: number[];
    cols?: number[];
    cells?: Array<[number, number]>;
  };
  metadata?: {
    scale?: number;
    maskValue?: number;
    temperature?: number;
  };
}

/**
 * Progress levels for progressive disclosure
 */
export type DisclosureLevel = 'overview' | 'intermediate' | 'detailed' | 'math';

/**
 * Disclosure level configuration
 */
export interface DisclosureLevelConfig {
  level: DisclosureLevel;
  showMath: boolean;
  showImplementation: boolean;
  showAllParameters: boolean;
  interactiveElements: string[];
}

/**
 * Tutorial system types
 */
export interface Tutorial {
  id: string;
  title: string;
  description: string;
  steps: TutorialStep[];
  targetAudience: 'beginner' | 'intermediate' | 'advanced';
  estimatedTime: number; // minutes
  prerequisites?: string[];
}

export interface TutorialStep {
  id: string;
  title: string;
  content: string;
  target?: string; // CSS selector for target element
  requiredDisclosureLevel?: DisclosureLevel;
  position?: 'top' | 'bottom' | 'left' | 'right' | 'center';
  action?: {
    type: 'click' | 'hover' | 'input' | 'wait';
    selector?: string;
    timeout?: number;
  };
  onNext?: () => void | Promise<void>;
  onPrev?: () => void | Promise<void>;
  highlightElements?: string[]; // CSS selectors
  dismissOnAction?: boolean;
}

export interface TutorialState {
  activeTutorial: string | null;
  currentStep: number;
  isTutorialActive: boolean;
  completedTutorials: string[];
  skippedTutorials: string[];
}

/**
 * Sankey diagram types for data flow visualization
 */
export interface SankeyNode {
  id: string;
  name: string;
  type:
    | 'input'
    | 'operation'
    | 'output'
    | 'layer'
    | 'embedding'
    | 'positional'
    | 'attention'
    | 'ffn'
    | 'normalization'
    | 'query'
    | 'key'
    | 'value'
    | 'dot_product'
    | 'scaled'
    | 'attention_weights';
  depth: number;
  value?: number;
  color?: string;
  metadata?: {
    shape?: number[];
    parameters?: number;
    computationCost?: number;
    description?: string;
    formula?: string;
    layerNum?: number;
  };
}

export interface SankeyLink {
  source: string;
  target: string;
  value: number;
  color?: string;
  opacity?: number;
  metadata?: {
    tensorShape?: number[];
    dataSize?: number;
  };
}

export interface SankeyData {
  nodes: SankeyNode[];
  links: SankeyLink[];
  layers: number;
}

/**
 * GSAP animation timeline types
 */
export interface AnimationTimeline {
  id: string;
  label: string;
  duration: number;
  progress: number;
  isPlaying: boolean;
  steps: AnimationStep[];
}

export interface AnimationStep {
  id: string;
  label: string;
  timestamp: number;
  description?: string;
  onStart?: () => void;
  onComplete?: () => void;
}

/**
 * Vector visualization data for Canvas renderer
 */
export interface VectorVisualizationData {
  vectors: number[][];
  labels?: string[];
  colors?: string[];
  dimensions: {
    rows: number;
    cols: number;
  };
  metadata?: {
    magnitude?: number[][];
    direction?: number[][];
  };
}

/**
 * Color coded math component types
 */
export interface ColorCodedMathProps {
  latex: string;
  colorMap?: Record<string, string>; // variable -> color mapping
  highlightVariables?: string[];
  interactive?: boolean;
  onVariableClick?: (variable: string) => void;
}

/**
 * Math explanation levels
 */
export type MathExplanationLevel = 'intuitive' | 'formal' | 'rigorous';

export interface MathExplanation {
  level: MathExplanationLevel;
  title: string;
  content: string;
  formula?: string;
  visualAid?: React.ReactNode;
  examples?: MathExample[];
}

export type MathExampleValue =
  | boolean
  | number
  | string
  | null
  | MathExampleValue[]
  | { [key: string]: MathExampleValue };

export interface MathExample {
  description: string;
  input: MathExampleValue;
  output: MathExampleValue;
  steps?: string[];
}

/**
 * Progressive reveal section states
 */
export interface RevealSectionState {
  id: string;
  isExpanded: boolean;
  level: DisclosureLevel;
  content: React.ReactNode;
}

/**
 * Performance monitoring types
 */
export interface PerformanceMetrics {
  fps: number;
  renderTime: number;
  memoryUsage: number;
  tensorSize: number;
}

/**
 * Cache entry for memoization
 */
export interface CacheEntry<T> {
  data: T;
  timestamp: number;
  hits: number;
  size: number;
}
