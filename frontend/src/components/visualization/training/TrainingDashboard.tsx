/**
 * TrainingDashboard Component
 *
 * Comprehensive training dashboard showing:
 * - Loss curves (train/validation with smoothing)
 * - Accuracy progress
 * - Learning rate schedule
 * - Layer-wise statistics
 * - Training events log
 * - Real-time metrics
 */

import React, { useState, useEffect, useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../ui/card';
import { Badge } from '../../ui/badge';
import { Button } from '../../ui/button';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Play,
  Pause,
  RotateCcw,
  Settings,
  TrendingUp,
  TrendingDown,
  Activity,
  Zap,
  Clock,
  CheckCircle,
  XCircle,
  FileText,
} from 'lucide-react';
import type {
  TrainingState,
  LayerTrainingStats,
  TrainingEvent,
  TrainingPhase,
} from '../../../types/training';
import {
  getLayerStatistics,
  getLearningRateSchedule,
  calculateLossCurve,
} from '../../../services/trainingApi';

interface TrainingDashboardProps {
  className?: string;
}

export const TrainingDashboard: React.FC<TrainingDashboardProps> = ({
  className = '',
}) => {
  // State
  const [trainingState, setTrainingState] = useState<TrainingState | null>(null);
  const [layerStats, setLayerStats] = useState<LayerTrainingStats[]>([]);
  const [events, setEvents] = useState<TrainingEvent[]>([]);
  const [selectedTab, setSelectedTab] = useState<'overview' | 'layers' | 'logs'>('overview');
  const [loading, setLoading] = useState(true);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingSpeed, setTrainingSpeed] = useState(100);

  // Load initial data
  useEffect(() => {
    const loadInitialData = async () => {
      setLoading(true);
      try {
        await new Promise(resolve => setTimeout(resolve, 500));

        const config = {
          epochs: 10,
          batchSize: 32,
          learningRate: 0.001,
          optimizer: 'adam' as const,
          scheduler: 'cosine' as const,
          earlyStopping: true,
          patience: 3,
          checkpointInterval: 1,
        };

        const stepsPerEpoch = 100;
        const totalSteps = config.epochs * stepsPerEpoch;

        // Initialize training state
        setTrainingState({
          phase: 'idle',
          currentEpoch: 0,
          currentStep: 0,
          totalSteps,
          metrics: [],
          config,
          startTime: undefined,
          endTime: undefined,
        });

        setLayerStats(getLayerStatistics());
        setEvents([]);
      } catch (error) {
        console.error('Failed to load training dashboard:', error);
      } finally {
        setLoading(false);
      }
    };

    loadInitialData();
  }, []);

  // Training simulation
  useEffect(() => {
    if (!isTraining || !trainingState) return;

    const stepsPerSecond = 1000 / trainingSpeed;
    const interval = setInterval(() => {
      setTrainingState((prev) => {
        if (!prev || prev.phase === 'completed') return prev;

        const newStep = prev.currentStep + 1;
        const newEpoch = Math.floor(newStep / 100) + 1;
        const isCompleted = newStep >= prev.totalSteps;

        // Generate new metric for this step
        const progress = newStep / prev.totalSteps;
        const baseLoss = 0.2 + 2.3 * Math.exp(-newEpoch * 0.3);
        const noise = (Math.random() - 0.5) * 0.1 * (1 - progress);
        const loss = Math.max(0.01, Math.min(5, baseLoss + noise));

        const baseAccuracy = 0.95 - 0.85 * Math.exp(-newEpoch * 0.4);
        const accuracyNoise = noise * 0.1;
        const accuracy = Math.max(0, Math.min(1, baseAccuracy + accuracyNoise));

        const newMetric: typeof prev.metrics[0] = {
          epoch: newEpoch,
          step: (newStep % 100) + 1,
          loss,
          accuracy,
          learningRate: 0.001 * (1 - progress * 0.9),
          gradientNorm: Math.random() * 2 + 0.5,
          validationLoss: loss * (1 + (Math.random() - 0.5) * 0.1),
          validationAccuracy: Math.max(0, Math.min(1, accuracy * (1 - (Math.random() - 0.5) * 0.05))),
          timestamp: Date.now(),
        };

        // Add events for epoch changes
        if (isCompleted) {
          setEvents((prev) => [
            ...prev,
            {
              type: 'epoch_end' as const,
              epoch: newEpoch,
              message: `Completed epoch ${newEpoch}`,
              timestamp: Date.now(),
              details: { loss, accuracy },
            },
            {
              type: 'early_stopping' as const,
              epoch: newEpoch,
              message: 'Training completed successfully!',
              timestamp: Date.now(),
            },
          ]);
        }

        return {
          ...prev,
          phase: isCompleted ? 'completed' : 'running',
          currentEpoch: newEpoch,
          currentStep: newStep,
          metrics: [...prev.metrics, newMetric],
          startTime: prev.startTime || Date.now(),
          endTime: isCompleted ? Date.now() : undefined,
        };
      });
    }, stepsPerSecond);

    return () => clearInterval(interval);
  }, [isTraining, trainingSpeed, trainingState, events]);

  // Start/Reset handlers
  const handleStartTraining = () => {
    if (trainingState?.phase === 'completed') {
      // Reset and start
      setTrainingState({
        ...trainingState,
        phase: 'running',
        currentEpoch: 0,
        currentStep: 0,
        metrics: [],
        startTime: Date.now(),
        endTime: undefined,
      });
      setEvents([]);
    }
    setIsTraining(true);
  };

  const handlePauseTraining = () => {
    setIsTraining(false);
    setTrainingState((prev) =>
      prev ? { ...prev, phase: 'paused' } : null
    );
  };

  const handleResetTraining = () => {
    setIsTraining(false);
    if (trainingState) {
      setTrainingState({
        ...trainingState,
        phase: 'idle',
        currentEpoch: 0,
        currentStep: 0,
        metrics: [],
        startTime: undefined,
        endTime: undefined,
      });
      setEvents([]);
    }
  };

  // Calculated data
  const lossCurveData = useMemo(() => {
    if (!trainingState) return null;
    return calculateLossCurve(trainingState.metrics);
  }, [trainingState]);

  const lrScheduleData = useMemo(() => {
    if (!trainingState) return [];
    return getLearningRateSchedule(trainingState.totalSteps, trainingState.config);
  }, [trainingState]);

  const latestMetrics = useMemo(() => {
    if (!trainingState || trainingState.metrics.length === 0) return null;
    return trainingState.metrics[trainingState.metrics.length - 1];
  }, [trainingState]);

  // Phase badge color
  const getPhaseColor = (phase: TrainingPhase) => {
    switch (phase) {
      case 'running':
        return 'bg-blue-500';
      case 'paused':
        return 'bg-yellow-500';
      case 'completed':
        return 'bg-green-500';
      case 'error':
        return 'bg-red-500';
      default:
        return 'bg-gray-500';
    }
  };

  // Format duration
  const formatDuration = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (hours > 0) {
      return `${hours}h ${minutes % 60}m ${seconds % 60}s`;
    } else if (minutes > 0) {
      return `${minutes}m ${seconds % 60}s`;
    }
    return `${seconds}s`;
  };

  // Event icon
  const getEventIcon = (type: TrainingEvent['type']) => {
    switch (type) {
      case 'epoch_start':
        return <Play className="h-4 w-4 text-blue-500" />;
      case 'epoch_end':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'checkpoint':
        return <FileText className="h-4 w-4 text-purple-500" />;
      case 'early_stopping':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'error':
        return <XCircle className="h-4 w-4 text-red-500" />;
    }
  };

  if (loading) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center py-12">
          <div className="text-center">
            <Activity className="h-12 w-12 mx-auto mb-4 animate-pulse text-primary" />
            <p className="text-muted-foreground">Loading training dashboard...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!trainingState) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center py-12">
          <div className="text-center">
            <Activity className="h-12 w-12 mx-auto mb-4 animate-pulse text-primary" />
            <p className="text-muted-foreground">Initializing training dashboard...</p>
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
              <Activity className="h-5 w-5 text-primary" />
              Training Dashboard
            </CardTitle>
            <CardDescription>
              Real-time training metrics and analysis
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Badge className={`${getPhaseColor(trainingState.phase)} text-white`}>
              {trainingState.phase}
            </Badge>

            {trainingState.phase === 'idle' || trainingState.phase === 'paused' ? (
              <Button variant="default" size="sm" onClick={handleStartTraining}>
                <Play className="h-4 w-4 mr-1" />
                Start Training
              </Button>
            ) : trainingState.phase === 'running' ? (
              <>
                <Button variant="outline" size="sm" onClick={handlePauseTraining}>
                  <Pause className="h-4 w-4 mr-1" />
                  Pause
                </Button>
                <Button variant="outline" size="sm" onClick={handleResetTraining}>
                  <RotateCcw className="h-4 w-4 mr-1" />
                  Reset
                </Button>
              </>
            ) : (
              <Button variant="default" size="sm" onClick={handleStartTraining}>
                <RotateCcw className="h-4 w-4 mr-1" />
                Restart
              </Button>
            )}

            {trainingState.phase === 'running' && (
              <select
                value={trainingSpeed}
                onChange={(e) => setTrainingSpeed(Number(e.target.value))}
                className="text-xs border rounded px-2 py-1 bg-background"
              >
                <option value={200}>Slow</option>
                <option value={100}>Normal</option>
                <option value={50}>Fast</option>
                <option value={10}>Very Fast</option>
              </select>
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Quick Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <motion.div
            className="p-4 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-lg"
            whileHover={{ scale: 1.02 }}
          >
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xs text-blue-600 dark:text-blue-400">Current Loss</div>
                <div className="text-2xl font-bold text-blue-700 dark:text-blue-300">
                  {latestMetrics ? latestMetrics.loss.toFixed(4) : 'N/A'}
                </div>
              </div>
              <TrendingDown className="h-8 w-8 text-blue-500 opacity-50" />
            </div>
          </motion.div>

          <motion.div
            className="p-4 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-lg"
            whileHover={{ scale: 1.02 }}
          >
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xs text-green-600 dark:text-green-400">Accuracy</div>
                <div className="text-2xl font-bold text-green-700 dark:text-green-300">
                  {latestMetrics
                    ? `${(latestMetrics.accuracy * 100).toFixed(1)}%`
                    : 'N/A'}
                </div>
              </div>
              <TrendingUp className="h-8 w-8 text-green-500 opacity-50" />
            </div>
          </motion.div>

          <motion.div
            className="p-4 bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 rounded-lg"
            whileHover={{ scale: 1.02 }}
          >
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xs text-purple-600 dark:text-purple-400">
                  Learning Rate
                </div>
                <div className="text-2xl font-bold text-purple-700 dark:text-purple-300">
                  {latestMetrics
                    ? latestMetrics.learningRate.toExponential(1)
                    : 'N/A'}
                </div>
              </div>
              <Zap className="h-8 w-8 text-purple-500 opacity-50" />
            </div>
          </motion.div>

          <motion.div
            className="p-4 bg-gradient-to-br from-orange-50 to-orange-100 dark:from-orange-900/20 dark:to-orange-800/20 rounded-lg"
            whileHover={{ scale: 1.02 }}
          >
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xs text-orange-600 dark:text-orange-400">
                  Gradient Norm
                </div>
                <div className="text-2xl font-bold text-orange-700 dark:text-orange-300">
                  {latestMetrics
                    ? latestMetrics.gradientNorm.toFixed(3)
                    : 'N/A'}
                </div>
              </div>
              <Activity className="h-8 w-8 text-orange-500 opacity-50" />
            </div>
          </motion.div>
        </div>

        {/* Progress */}
        <div className="p-4 bg-muted rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium">
              Epoch {trainingState.currentEpoch} / {trainingState.config.epochs}
            </span>
            <span className="text-xs text-muted-foreground">
              Step {trainingState.currentStep} / {trainingState.totalSteps}
            </span>
          </div>
          <div className="relative h-3 bg-muted-foreground/20 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500"
              initial={{ width: 0 }}
              animate={{
                width: `${
                  (trainingState.currentStep / trainingState.totalSteps) * 100
                }%`,
              }}
              transition={{ duration: 0.5 }}
            />
          </div>
          {trainingState.startTime && (
            <div className="flex items-center gap-2 mt-2 text-xs text-muted-foreground">
              <Clock className="h-3 w-3" />
              <span>
                Elapsed:{' '}
                {formatDuration(
                  (trainingState.endTime || Date.now()) - trainingState.startTime
                )}
              </span>
            </div>
          )}
        </div>

        {/* Tabs */}
        <div className="flex gap-2 border-b">
          <Button
            variant={selectedTab === 'overview' ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setSelectedTab('overview')}
          >
            Overview
          </Button>
          <Button
            variant={selectedTab === 'layers' ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setSelectedTab('layers')}
          >
            Layers
          </Button>
          <Button
            variant={selectedTab === 'logs' ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setSelectedTab('logs')}
          >
            Event Log
          </Button>
        </div>

        <AnimatePresence mode="wait">
          {/* Overview Tab */}
          {selectedTab === 'overview' && (
            <motion.div
              key="overview"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="space-y-6"
            >
              {/* Loss Curve */}
              {lossCurveData && (
                <div className="space-y-2">
                  <h3 className="text-sm font-medium">Loss Curve</h3>
                  <ResponsiveContainer width="100%" height={250}>
                    <AreaChart data={lossCurveData.trainLoss}>
                      <defs>
                        <linearGradient id="lossGradient" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                      <XAxis
                        dataKey="epoch"
                        label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }}
                        className="text-xs"
                      />
                      <YAxis
                        label={{ value: 'Loss', angle: -90, position: 'insideLeft' }}
                        className="text-xs"
                      />
                      <Tooltip />
                      <Legend />
                      <Area
                        type="monotone"
                        dataKey="value"
                        stroke="#3b82f6"
                        fill="url(#lossGradient)"
                        name="Train Loss"
                      />
                      {lossCurveData.validationLoss && (
                        <Line
                          type="monotone"
                          dataKey="value"
                          data={lossCurveData.validationLoss}
                          stroke="#ef4444"
                          strokeWidth={2}
                          name="Val Loss"
                          dot={false}
                        />
                      )}
                      {lossCurveData.smoothedLoss && (
                        <Line
                          type="monotone"
                          dataKey="value"
                          data={lossCurveData.smoothedLoss}
                          stroke="#8b5cf6"
                          strokeWidth={2}
                          strokeDasharray="5 5"
                          name="Smoothed"
                          dot={false}
                        />
                      )}
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* Learning Rate Schedule */}
              {lrScheduleData.length > 0 && (
                <div className="space-y-2">
                  <h3 className="text-sm font-medium">Learning Rate Schedule</h3>
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={lrScheduleData}>
                      <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                      <XAxis
                        dataKey="step"
                        label={{ value: 'Step', position: 'insideBottom', offset: -5 }}
                        className="text-xs"
                      />
                      <YAxis
                        label={{ value: 'LR', angle: -90, position: 'insideLeft' }}
                        className="text-xs"
                      />
                      <Tooltip
                        content={({ active, payload }) => {
                          if (active && payload && payload.length) {
                            return (
                              <div className="bg-background border rounded-lg p-2 shadow-lg">
                                <p className="text-xs font-medium">
                                  Step: {payload[0].payload.step}
                                </p>
                                <p className="text-xs text-muted-foreground">
                                  LR: {payload[0].payload.learningRate.toExponential(2)}
                                </p>
                                <p className="text-xs text-muted-foreground">
                                  {payload[0].payload.reason}
                                </p>
                              </div>
                            );
                          }
                          return null;
                        }}
                      />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="learningRate"
                        stroke="#8b5cf6"
                        strokeWidth={2}
                        name="Learning Rate"
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </motion.div>
          )}

          {/* Layers Tab */}
          {selectedTab === 'layers' && (
            <motion.div
              key="layers"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="space-y-4"
            >
              <div className="rounded-lg border overflow-hidden">
                <table className="w-full">
                  <thead className="bg-muted">
                    <tr className="text-left">
                      <th className="p-3 text-xs font-medium">Layer</th>
                      <th className="p-3 text-xs font-medium">Type</th>
                      <th className="p-3 text-xs font-medium">Weight μ/σ</th>
                      <th className="p-3 text-xs font-medium">Grad μ/σ</th>
                      <th className="p-3 text-xs font-medium">Dead Neurons</th>
                      <th className="p-3 text-xs font-medium">Saturation</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y">
                    {layerStats.map((layer, index) => (
                      <motion.tr
                        key={layer.layerName}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.05 }}
                        className="hover:bg-muted/50"
                      >
                        <td className="p-3 text-sm font-medium">{layer.layerName}</td>
                        <td className="p-3">
                          <Badge variant="secondary" className="text-xs">
                            {layer.layerType}
                          </Badge>
                        </td>
                        <td className="p-3 text-xs font-mono">
                          {layer.weightMean.toFixed(3)} / {layer.weightStd.toFixed(3)}
                        </td>
                        <td className="p-3 text-xs font-mono">
                          {layer.gradientMean.toFixed(3)} / {layer.gradientStd.toFixed(3)}
                        </td>
                        <td className="p-3 text-xs">
                          <Badge
                            variant={layer.deadNeurons > 10 ? 'destructive' : 'secondary'}
                            className="text-xs"
                          >
                            {layer.deadNeurons}
                          </Badge>
                        </td>
                        <td className="p-3 text-xs">
                          <div className="flex items-center gap-2">
                            <div className="w-16 h-2 bg-muted rounded-full overflow-hidden">
                              <div
                                className="h-full bg-primary"
                                style={{ width: `${layer.saturationRate * 100}%` }}
                              />
                            </div>
                            <span className="text-xs">
                              {(layer.saturationRate * 100).toFixed(0)}%
                            </span>
                          </div>
                        </td>
                      </motion.tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </motion.div>
          )}

          {/* Logs Tab */}
          {selectedTab === 'logs' && (
            <motion.div
              key="logs"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="space-y-2"
            >
              <div className="max-h-96 overflow-y-auto space-y-2">
                {events.map((event, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className="flex items-start gap-3 p-3 bg-muted rounded-lg"
                  >
                    <div className="mt-0.5">{getEventIcon(event.type)}</div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium">Epoch {event.epoch}</span>
                        <span className="text-xs text-muted-foreground">
                          {new Date(event.timestamp).toLocaleTimeString()}
                        </span>
                      </div>
                      <p className="text-sm text-muted-foreground">{event.message}</p>
                      {event.details && (
                        <div className="mt-1 text-xs text-muted-foreground">
                          {Object.entries(event.details).map(([key, value]) => (
                            <span key={key} className="mr-3">
                              {key}: {String(value)}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Configuration */}
        <div className="p-4 bg-muted rounded-lg">
          <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
            <Settings className="h-4 w-4" />
            Configuration
          </h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
            <div>
              <span className="text-muted-foreground">Optimizer:</span>{' '}
              <span className="font-medium">{trainingState.config.optimizer}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Scheduler:</span>{' '}
              <span className="font-medium">{trainingState.config.scheduler}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Batch Size:</span>{' '}
              <span className="font-medium">{trainingState.config.batchSize}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Initial LR:</span>{' '}
              <span className="font-medium">{trainingState.config.learningRate}</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
