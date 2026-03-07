/**
 * Core types for Transformer visualization
 */

export interface TensorData {
  shape: number[];
  data: number[] | number[][];
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
    steps?: LayerOutputData[];
    attentionWeights?: AttentionData[];
    layerOutputs?: LayerOutputData[];
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
