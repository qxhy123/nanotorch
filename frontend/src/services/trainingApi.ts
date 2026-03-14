/**
 * Training API service
 */

import type {
  TrainingMetrics,
  TrainingConfig,
  TrainingState,
  LayerTrainingStats,
  TrainingEvent,
  LossCurveData,
  LRSchedulePoint,
} from '../types/training';

// Mock configuration
const DEFAULT_CONFIG: TrainingConfig = {
  epochs: 10,
  batchSize: 32,
  learningRate: 0.001,
  optimizer: 'adam',
  scheduler: 'cosine',
  earlyStopping: true,
  patience: 3,
  checkpointInterval: 1,
};

// Generate mock training metrics
export function generateMockTrainingMetrics(
  epochs: number = 10,
  stepsPerEpoch: number = 100
): TrainingMetrics[] {
  const metrics: TrainingMetrics[] = [];
  let loss = 2.5;
  let accuracy = 0.1;

  for (let epoch = 0; epoch < epochs; epoch++) {
    // Simulate learning with some noise
    const epochLossDecay = Math.exp(-epoch * 0.3);
    const epochAccuracyGrowth = 1 - Math.exp(-epoch * 0.4);

    for (let step = 0; step < stepsPerEpoch; step++) {
      const progress = (epoch * stepsPerEpoch + step) / (epochs * stepsPerEpoch);

      // Add per-step noise
      const stepNoise = (Math.random() - 0.5) * 0.1;
      loss = 0.2 + 2.3 * epochLossDecay + stepNoise * (1 - progress);
      accuracy = 0.95 - 0.85 * epochAccuracyGrowth + stepNoise * 0.1 * (1 - progress);

      // Clamp values
      loss = Math.max(0.01, Math.min(5, loss));
      accuracy = Math.max(0, Math.min(1, accuracy));

      const validationLoss = loss * (1 + (Math.random() - 0.5) * 0.1);
      const validationAccuracy = accuracy * (1 - (Math.random() - 0.5) * 0.05);

      metrics.push({
        epoch: epoch + 1,
        step: step + 1,
        loss,
        accuracy,
        learningRate: 0.001 * (1 - progress * 0.9),
        gradientNorm: Math.random() * 2 + 0.5,
        validationLoss,
        validationAccuracy: Math.max(0, Math.min(1, validationAccuracy)),
        timestamp: Date.now() - (epochs - epoch) * 100000 + step * 1000,
      });
    }
  }

  return metrics;
}

// Generate mock layer training statistics
export function generateMockLayerStats(): LayerTrainingStats[] {
  const layers = [
    'embedding', 'encoder_1', 'encoder_2', 'encoder_3',
    'encoder_4', 'encoder_5', 'encoder_6',
    'decoder_1', 'decoder_2', 'decoder_3',
    'decoder_4', 'decoder_5', 'decoder_6', 'output'
  ];

  return layers.map((name, index) => {
    const layerType = name.includes('encoder')
      ? 'encoder'
      : name.includes('decoder')
      ? 'decoder'
      : name === 'embedding'
      ? 'embedding'
      : 'output';

    const deadNeuronRate = Math.max(0, 0.05 - index * 0.003);
    const saturationRate = Math.min(0.95, 0.3 + index * 0.05);

    return {
      layerName: name,
      layerType,
      weightMean: (Math.random() - 0.5) * 0.2,
      weightStd: 0.1 + Math.random() * 0.1,
      gradientMean: (Math.random() - 0.5) * 0.01,
      gradientStd: 0.01 + Math.random() * 0.02,
      updateMean: (Math.random() - 0.5) * 0.001,
      updateStd: 0.0001 + Math.random() * 0.0002,
      deadNeurons: Math.floor(deadNeuronRate * 512),
      saturationRate: saturationRate + (Math.random() - 0.5) * 0.1,
    };
  });
}

// Generate mock training events
export function generateMockTrainingEvents(epochs: number): TrainingEvent[] {
  const events: TrainingEvent[] = [];

  for (let epoch = 1; epoch <= epochs; epoch++) {
    events.push({
      type: 'epoch_start',
      epoch,
      message: `Starting epoch ${epoch}`,
      timestamp: Date.now() - (epochs - epoch) * 100000,
    });

    events.push({
      type: 'epoch_end',
      epoch,
      message: `Completed epoch ${epoch}`,
      timestamp: Date.now() - (epochs - epoch) * 100000 + 50000,
      details: {
        loss: 2.5 * Math.exp(-epoch * 0.3),
        accuracy: 0.1 + 0.85 * (1 - Math.exp(-epoch * 0.4)),
      },
    });

    if (epoch % 3 === 0) {
      events.push({
        type: 'checkpoint',
        epoch,
        message: `Model checkpoint saved at epoch ${epoch}`,
        timestamp: Date.now() - (epochs - epoch) * 100000 + 55000,
      });
    }
  }

  events.push({
    type: 'early_stopping',
    epoch: epochs,
    message: 'Training completed successfully',
    timestamp: Date.now(),
  });

  return events;
}

// Generate learning rate schedule
export function generateLRSchedule(
  totalSteps: number,
  config: TrainingConfig
): LRSchedulePoint[] {
  const schedule: LRSchedulePoint[] = [];
  const { learningRate, scheduler } = config;

  for (let step = 0; step <= totalSteps; step += Math.floor(totalSteps / 20)) {
    let lr = learningRate;
    let reason = 'Initial learning rate';

    switch (scheduler) {
      case 'constant':
        lr = learningRate;
        reason = 'Constant learning rate';
        break;
      case 'step': {
        const dropPeriod = Math.floor(totalSteps / 3);
        const drops = Math.floor(step / dropPeriod);
        lr = learningRate * Math.pow(0.5, drops);
        reason = `Step decay: ${drops} drop(s)`;
        break;
      }
      case 'cosine':
        lr = learningRate * 0.5 * (1 + Math.cos((Math.PI * step) / totalSteps));
        reason = 'Cosine annealing';
        break;
      case 'exponential':
        lr = learningRate * Math.pow(0.95, step / 100);
        reason = 'Exponential decay';
        break;
      case 'warmup': {
        const warmupSteps = Math.floor(totalSteps * 0.1);
        if (step < warmupSteps) {
          lr = learningRate * (step / warmupSteps);
          reason = 'Warmup phase';
        } else {
          lr = learningRate * 0.5 * (1 + Math.cos((Math.PI * (step - warmupSteps)) / (totalSteps - warmupSteps)));
          reason = 'Cosine decay after warmup';
        }
        break;
      }
    }

    schedule.push({ step, learningRate: lr, reason });
  }

  return schedule;
}

// Calculate loss curve data
export function calculateLossCurve(metrics: TrainingMetrics[]): LossCurveData {
  const epochMap = new Map<number, { sumLoss: number; sumValLoss: number; count: number }>();

  metrics.forEach((m) => {
    if (!epochMap.has(m.epoch)) {
      epochMap.set(m.epoch, { sumLoss: 0, sumValLoss: 0, count: 0 });
    }
    const entry = epochMap.get(m.epoch)!;
    entry.sumLoss += m.loss;
    if (m.validationLoss) entry.sumValLoss += m.validationLoss;
    entry.count += 1;
  });

  const trainLoss = Array.from(epochMap.entries())
    .map(([epoch, data]) => ({
      epoch,
      value: data.sumLoss / data.count,
    }))
    .sort((a, b) => a.epoch - b.epoch);

  const validationLoss = Array.from(epochMap.entries())
    .filter(([, data]) => data.sumValLoss > 0)
    .map(([epoch, data]) => ({
      epoch,
      value: data.sumValLoss / data.count,
    }))
    .sort((a, b) => a.epoch - b.epoch);

  // Calculate smoothed loss (exponential moving average)
  const smoothedLoss: Array<{ epoch: number; value: number }> = [];
  trainLoss.forEach((point, index) => {
    if (index === 0) {
      smoothedLoss.push(point);
    } else {
      const alpha = 0.3;
      const prevValue = smoothedLoss[index - 1].value;
      smoothedLoss.push({
        epoch: point.epoch,
        value: alpha * point.value + (1 - alpha) * prevValue,
      });
    }
  });

  return { trainLoss, validationLoss, smoothedLoss };
}

// Get training state
export function getTrainingState(): TrainingState {
  const metrics = generateMockTrainingMetrics(10, 100);
  const config = DEFAULT_CONFIG;

  return {
    phase: 'completed',
    currentEpoch: 10,
    currentStep: 1000,
    totalSteps: 1000,
    metrics,
    config,
    startTime: Date.now() - 1000000,
    endTime: Date.now(),
  };
}

// Get layer statistics
export function getLayerStatistics(): LayerTrainingStats[] {
  return generateMockLayerStats();
}

// Get training events
export function getTrainingEvents(epochs: number = 10): TrainingEvent[] {
  return generateMockTrainingEvents(epochs);
}

// Get learning rate schedule
export function getLearningRateSchedule(
  totalSteps: number,
  config: TrainingConfig = DEFAULT_CONFIG
): LRSchedulePoint[] {
  return generateLRSchedule(totalSteps, config);
}
