/**
 * Training visualization types
 */

// Training metrics
export interface TrainingMetrics {
  epoch: number;
  step: number;
  loss: number;
  accuracy: number;
  learningRate: number;
  gradientNorm: number;
  validationLoss?: number;
  validationAccuracy?: number;
  timestamp: number;
}

// Training configuration
export interface TrainingConfig {
  epochs: number;
  batchSize: number;
  learningRate: number;
  optimizer: OptimizerType;
  scheduler: LRSchedulerType;
  earlyStopping: boolean;
  patience: number;
  checkpointInterval: number;
}

// Optimizer types
export type OptimizerType =
  | 'adam'
  | 'adamw'
  | 'sgd'
  | 'rmsprop'
  | 'adagrad';

// Learning rate scheduler types
export type LRSchedulerType =
  | 'constant'
  | 'step'
  | 'cosine'
  | 'exponential'
  | 'warmup'
  | 'plateau';

// Training phase
export type TrainingPhase = 'idle' | 'running' | 'paused' | 'completed' | 'error';

// Training state
export interface TrainingState {
  phase: TrainingPhase;
  currentEpoch: number;
  currentStep: number;
  totalSteps: number;
  metrics: TrainingMetrics[];
  config: TrainingConfig;
  startTime?: number;
  endTime?: number;
  error?: string;
}

// Layer statistics for training
export interface LayerTrainingStats {
  layerName: string;
  layerType: string;
  weightMean: number;
  weightStd: number;
  gradientMean: number;
  gradientStd: number;
  updateMean: number;
  updateStd: number;
  deadNeurons: number;
  saturationRate: number;
}

// Training event
export interface TrainingEvent {
  type: 'epoch_start' | 'epoch_end' | 'checkpoint' | 'early_stopping' | 'error';
  epoch: number;
  message: string;
  timestamp: number;
  details?: Record<string, unknown>;
}

// Loss curve data
export interface LossCurveData {
  trainLoss: Array<{ epoch: number; value: number }>;
  validationLoss?: Array<{ epoch: number; value: number }>;
  smoothedLoss?: Array<{ epoch: number; value: number }>;
}

// Learning rate schedule
export interface LRSchedulePoint {
  step: number;
  learningRate: number;
  reason: string;
}
