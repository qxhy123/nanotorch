/**
 * TypeScript types for layer computation visualization.
 *
 * This module defines all types related to Transformer encoder and decoder layer
 * computation visualization, including intermediate results, step-by-step processing,
 * and layer statistics.
 */

/**
 * Supported layer types.
 */
export type LayerType = 'encoder' | 'decoder';

/**
 * Sublayer types within a Transformer layer.
 */
export type SublayerType = 'self_attention' | 'cross_attention' | 'feedforward';

/**
 * Tensor data representation.
 */
export interface TensorData {
  /** Shape of the tensor (e.g., [batch_size, seq_len, d_model]) */
  shape: number[];
  /** Tensor data (flattened or nested depending on dimensionality) */
  data: number[] | number[][] | number[][][] | number[][][][];
  /** Data type (e.g., 'float32') */
  dtype: string;
  /** Optional tensor name */
  name?: string;
}

/**
 * Attention computation results.
 */
export interface AttentionComputation {
  /** Output from attention computation */
  output: TensorData;
  /** Query tensor (optional, if returned) */
  queries?: TensorData;
  /** Key tensor (optional, if returned) */
  keys?: TensorData;
  /** Value tensor (optional, if returned) */
  values?: TensorData;
  /** Attention scores before softmax */
  scores?: TensorData;
  /** Attention weights (after softmax) */
  weights?: TensorData;
  /** Scale factor used in attention */
  scale?: number;
}

/**
 * Results from a single sublayer computation.
 */
export interface SublayerComputation {
  /** Input to layer normalization */
  norm_input?: TensorData;
  /** Output from layer normalization */
  norm_output?: TensorData;
  /** Attention computation results (for attention sublayers) */
  attention?: AttentionComputation;
  /** Output from first linear layer (FFN) */
  linear1_output?: TensorData;
  /** Output after activation function (FFN) */
  activation_output?: TensorData;
  /** Output from second linear layer (FFN) */
  linear2_output?: TensorData;
  /** Output after dropout (first dropout in sublayer) */
  dropout_output?: TensorData;
  /** Additional dropout output (for FFN with two dropouts) */
  dropout1_output?: TensorData;
  /** Additional dropout output (for FFN with two dropouts) */
  dropout2_output?: TensorData;
  /** Output after residual connection (x + Sublayer(x)) */
  residual_output: TensorData;
}

/**
 * Complete encoder layer computation results.
 */
export interface EncoderLayerResult {
  /** Whether the computation was successful */
  success: boolean;
  /** Layer input tensor */
  input: TensorData;
  /** Layer configuration */
  config: LayerConfig;
  /** Self-attention sublayer results */
  sublayer1: SublayerComputation;
  /** Feed-forward sublayer results */
  sublayer2: SublayerComputation;
  /** Layer output tensor */
  output: TensorData;
  /** Error message if computation failed */
  error?: string;
}

/**
 * Complete decoder layer computation results.
 */
export interface DecoderLayerResult {
  /** Whether the computation was successful */
  success: boolean;
  /** Layer input tensor (decoder input/target) */
  input: TensorData;
  /** Encoder output (memory) for cross-attention */
  encoder_output: TensorData;
  /** Layer configuration */
  config: LayerConfig;
  /** Masked self-attention sublayer results */
  sublayer1: SublayerComputation;
  /** Cross-attention sublayer results */
  sublayer2: SublayerComputation;
  /** Feed-forward sublayer results */
  sublayer3: SublayerComputation;
  /** Layer output tensor */
  output: TensorData;
  /** Error message if computation failed */
  error?: string;
}

/**
 * Layer configuration parameters.
 */
export interface LayerConfig {
  /** Model dimension */
  d_model: number;
  /** Number of attention heads */
  nhead: number;
  /** Feed-forward network dimension */
  dim_feedforward: number;
  /** Dropout rate */
  dropout: number;
  /** Activation function ('relu' or 'gelu') */
  activation: string;
  /** Whether to use pre-layer normalization */
  norm_first: boolean;
  /** Layer normalization epsilon */
  layer_norm_eps: number;
  /** Whether batch dimension comes first */
  batch_first: boolean;
}

/**
 * Input data for layer computation.
 */
export interface LayerInput {
  /** Input tensor data */
  data: number[][];
  /** Shape of the input tensor */
  shape: number[];
  /** Data type */
  dtype: string;
}

/**
 * Request for encoder layer computation.
 */
export interface EncoderLayerRequest {
  /** Transformer configuration */
  config: LayerConfig;
  /** Layer input data */
  input_data: LayerInput;
  /** Source attention mask (optional) */
  src_mask?: number[][];
  /** Whether to use causal attention */
  is_causal: boolean;
}

/**
 * Request for decoder layer computation.
 */
export interface DecoderLayerRequest {
  /** Transformer configuration */
  config: LayerConfig;
  /** Decoder input data (target) */
  input_data: LayerInput;
  /** Encoder output (memory) */
  encoder_output: LayerInput;
  /** Target attention mask (optional) */
  tgt_mask?: number[][];
  /** Memory attention mask (optional) */
  memory_mask?: number[][];
}

/**
 * Options for layer computation.
 */
export interface LayerComputeOptions {
  /** Whether to return attention weights */
  return_attention_weights?: boolean;
  /** Whether to return layer parameters */
  return_parameters?: boolean;
  /** Batch size for computation */
  batch_size?: number;
}

/**
 * Statistics about layer computation.
 */
export interface LayerStatistics {
  /** Total number of parameters */
  num_parameters: number;
  /** Approximate FLOPs for forward pass */
  flops: number;
  /** Approximate memory usage in MB */
  memory_mb: number;
  /** Parameters per sublayer/component */
  sublayer_breakdown: {
    self_attention?: number;
    cross_attention?: number;
    feed_forward?: number;
    layer_norm?: number;
    [key: string]: number | undefined;
  };
}

/**
 * Response from layer statistics endpoint.
 */
export interface LayerStatisticsResponse {
  /** Total number of parameters */
  num_parameters: number;
  /** Approximate FLOPs for forward pass */
  flops: number;
  /** Approximate memory usage in MB */
  memory_mb: number;
  /** Parameters per sublayer/component */
  sublayer_breakdown: {
    self_attention?: number;
    cross_attention?: number;
    feed_forward?: number;
    layer_norm?: number;
    [key: string]: number | undefined;
  };
}

/**
 * Layer computation step for visualization navigation.
 */
export interface LayerStep {
  /** Step identifier */
  id: string;
  /** Step name (e.g., 'input', 'norm1', 'attention', etc.) */
  name: string;
  /** Step display title */
  title: string;
  /** Step description */
  description: string;
  /** Step index in the sequence */
  index: number;
  /** Whether this step is currently active */
  isActive: boolean;
  /** Whether this step has been completed */
  isCompleted: boolean;
  /** Associated sublayer (0, 1, or 2) */
  sublayer?: number;
}

/**
 * Computation stage within a sublayer.
 */
export type ComputationStage =
  | 'input'
  | 'norm'
  | 'attention'
  | 'linear1'
  | 'activation'
  | 'linear2'
  | 'dropout'
  | 'residual'
  | 'output';

/**
 * Data flow node for visualization.
 */
export interface DataFlowNode {
  /** Node identifier */
  id: string;
  /** Node label */
  label: string;
  /** Node type */
  type: 'operation' | 'data' | 'output';
  /** X position */
  x: number;
  /** Y position */
  y: number;
  /** Associated tensor shape */
  shape?: number[];
  /** Associated computation stage */
  stage?: ComputationStage;
}

/**
 * Data flow edge for visualization.
 */
export interface DataFlowEdge {
  /** Edge identifier */
  id: string;
  /** Source node ID */
  source: string;
  /** Target node ID */
  target: string;
  /** Edge label */
  label?: string;
  /** Whether this is a residual connection */
  isResidual?: boolean;
}

/**
 * Complete data flow diagram.
 */
export interface DataFlowDiagram {
  /** All nodes in the flow */
  nodes: DataFlowNode[];
  /** All edges in the flow */
  edges: DataFlowEdge[];
  /** Currently highlighted step */
  highlightStep?: string;
}

/**
 * Tensor statistics for visualization.
 */
export interface TensorStatistics {
  /** Minimum value */
  min: number;
  /** Maximum value */
  max: number;
  /** Mean value */
  mean: number;
  /** Standard deviation */
  std: number;
  /** Median value */
  median: number;
  /** 25th percentile */
  q25: number;
  /** 75th percentile */
  q75: number;
}

/**
 * Comparison between two tensors.
 */
export interface TensorComparison {
  /** Before tensor */
  before: TensorData;
  /** After tensor */
  after: TensorData;
  /** Operation performed */
  operation: string;
  /** Statistics before */
  beforeStats: TensorStatistics;
  /** Statistics after */
  afterStats: TensorStatistics;
  /** Difference statistics */
  differenceStats?: TensorStatistics;
}

/**
 * Layer visualization state.
 */
export interface LayerVisualizationState {
  /** Currently selected layer type */
  layerType: LayerType;
  /** Currently selected layer index */
  layerIndex: number;
  /** Current step in the visualization */
  currentStep: number;
  /** Whether to show tensor values */
  showValues: boolean;
  /** Whether to show tensor shapes */
  showShapes: boolean;
  /** Whether to show statistics */
  showStatistics: boolean;
  /** Whether to animate transitions */
  animateTransitions: boolean;
  /** Computation result */
  computationResult: EncoderLayerResult | DecoderLayerResult | null;
  /** Whether computation is in progress */
  isComputing: boolean;
  /** Computation error */
  computationError: string | null;
  /** Playback state */
  isPlaying: boolean;
  /** Playback speed (ms per step) */
  playbackSpeed: number;
}

/**
 * Layer information.
 */
export interface LayerInfo {
  /** Layer type */
  type: LayerType;
  /** Layer index */
  index: number;
  /** Layer configuration */
  config: LayerConfig;
  /** Number of sublayers */
  numSublayers: number;
  /** Sublayer names */
  sublayerNames: string[];
}

/**
 * Layer type information.
 */
export interface LayerTypeInfo {
  /** Layer type */
  type: LayerType;
  /** Display name */
  name: string;
  /** Description */
  description: string;
  /** Sublayers in this layer type */
  sublayers: string[];
  /** Architecture type (post-norm or pre-norm) */
  architecture: string;
}

/**
 * Response from layer types endpoint.
 */
export interface LayerTypesResponse {
  /** Whether request was successful */
  success: boolean;
  /** Available layer types */
  layer_types: LayerTypeInfo[];
}

/**
 * Progress information for layer computation.
 */
export interface LayerComputationProgress {
  /** Current step being computed */
  currentStep: number;
  /** Total number of steps */
  totalSteps: number;
  /** Current sublayer being computed */
  currentSublayer: number;
  /** Percentage complete */
  percentage: number;
}

/**
 * Error information for layer computation.
 */
export interface LayerComputationError {
  /** Error code */
  code: string;
  /** Error message */
  message: string;
  /** Stack trace (if available) */
  stackTrace?: string;
  /** Step where error occurred */
  errorStep?: string;
}
