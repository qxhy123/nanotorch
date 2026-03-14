/**
 * AutoRegressiveGeneration Component
 *
 * Visualizes the decoder's autoregressive token-by-token generation process
 * with step-by-step animation and probability distribution at each step.
 */

import React, { useState, useCallback, useEffect, useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../ui/card';
import { Badge } from '../../ui/badge';
import { Button } from '../../ui/button';
import { Slider } from '../../ui/slider';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  StepForward,
  RotateCw,
  Zap,
  Clock,
  TrendingUp,
  Info,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { inferenceDemoApi } from '../../../services/inferenceDemoApi';
import type {
  GenerationStep,
  SamplingOptions,
  SamplingStrategy,
} from '../../../types/inference';

interface AutoRegressiveGenerationProps {
  prompt?: string;
  maxLength?: number;
  className?: string;
}

export const AutoRegressiveGeneration: React.FC<AutoRegressiveGenerationProps> = ({
  prompt = 'The future of AI',
  maxLength = 10,
  className = '',
}) => {
  // State
  const [steps, setSteps] = useState<GenerationStep[]>([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1000);
  const [samplingOptions, setSamplingOptions] = useState<Partial<SamplingOptions>>({
    strategy: 'greedy',
    temperature: 1.0,
    topK: 50,
    topP: 0.9,
  });
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [loading, setLoading] = useState(false);
  const [showDistribution, setShowDistribution] = useState(true);

  // Load/Generate steps
  const loadSteps = useCallback(async () => {
    setLoading(true);
    try {
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 500));
      const mockSteps = inferenceDemoApi.generateAutoregressiveSteps(
        prompt,
        maxLength,
        samplingOptions.strategy || 'greedy'
      );
      setSteps(mockSteps);
      setCurrentStep(0);
    } catch (error) {
      console.error('Failed to load generation steps:', error);
    } finally {
      setLoading(false);
    }
  }, [prompt, maxLength, samplingOptions.strategy]);

  // Initial load
  useEffect(() => {
    loadSteps();
  }, [loadSteps]);

  // Auto-play logic
  useEffect(() => {
    if (isPlaying && currentStep < steps.length - 1) {
      const timer = setTimeout(() => {
        setCurrentStep(prev => prev + 1);
      }, playbackSpeed);
      return () => clearTimeout(timer);
    } else if (isPlaying && currentStep >= steps.length - 1) {
      setIsPlaying(false);
    }
  }, [isPlaying, currentStep, steps.length, playbackSpeed]);

  // Control handlers
  const handlePlayPause = () => {
    if (currentStep >= steps.length - 1) {
      setCurrentStep(0);
    }
    setIsPlaying(!isPlaying);
  };

  const handleReset = () => {
    setIsPlaying(false);
    setCurrentStep(0);
  };

  const handleStepForward = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(prev => prev + 1);
    }
  };

  const handleStepBack = () => {
    if (currentStep > 0) {
      setCurrentStep(prev => prev - 1);
    }
  };

  const handleSkipToEnd = () => {
    setIsPlaying(false);
    setCurrentStep(steps.length - 1);
  };

  const handleSkipToStart = () => {
    setIsPlaying(false);
    setCurrentStep(0);
  };

  // Current step data
  const currentStepData = useMemo(() => {
    if (steps.length === 0 || currentStep >= steps.length) return null;
    return steps[currentStep];
  }, [steps, currentStep]);

  // Progress percentage
  const progress = useMemo(() => {
    if (steps.length === 0) return 0;
    return ((currentStep + 1) / steps.length) * 100;
  }, [currentStep, steps.length]);

  // Get tokens for display
  const displayTokens = useMemo(() => {
    if (!currentStepData) return [];
    const words = currentStepData.context.split(' ');
    return words.map((word, idx) => ({
      word,
      isNew: idx === words.length - (currentStepData.stepIndex > 0 ? currentStepData.generatedToken.token.split(' ').length : 1),
    }));
  }, [currentStepData]);

  if (loading) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center py-12">
          <RotateCw className="h-8 w-8 animate-spin text-primary" />
        </CardContent>
      </Card>
    );
  }

  if (steps.length === 0) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center py-12">
          <div className="text-center text-gray-500">
            <Zap className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p className="text-lg font-medium">No generation data available</p>
            <Button onClick={loadSteps} className="mt-4">
              Generate
            </Button>
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
              <Zap className="h-5 w-5 text-primary" />
              Autoregressive Generation
            </CardTitle>
            <CardDescription>
              Watch the decoder generate tokens step by step
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={loadSteps}>
              <RotateCw className="h-4 w-4 mr-1" />
              Regenerate
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Progress Bar */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600 dark:text-gray-400">
              Step {currentStep + 1} of {steps.length}
            </span>
            <span className="font-medium">{progress.toFixed(0)}%</span>
          </div>
          <div className="relative h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500"
              initial={{ width: 0 }}
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
        </div>

        {/* Playback Controls */}
        <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={handleSkipToStart} disabled={currentStep === 0}>
              <SkipBack className="h-4 w-4" />
            </Button>
            <Button variant="outline" size="sm" onClick={handleStepBack} disabled={currentStep === 0}>
              <StepForward className="h-4 w-4 rotate-180" />
            </Button>
            <Button variant="default" size="sm" onClick={handlePlayPause}>
              {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            </Button>
            <Button variant="outline" size="sm" onClick={handleStepForward} disabled={currentStep >= steps.length - 1}>
              <StepForward className="h-4 w-4" />
            </Button>
            <Button variant="outline" size="sm" onClick={handleSkipToEnd} disabled={currentStep >= steps.length - 1}>
              <SkipForward className="h-4 w-4" />
            </Button>
            <Button variant="outline" size="sm" onClick={handleReset}>
              <RotateCw className="h-4 w-4" />
            </Button>
          </div>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4 text-gray-500" />
              <span className="text-sm text-gray-600 dark:text-gray-400">
                Speed: {playbackSpeed}ms
              </span>
            </div>
            <select
              value={playbackSpeed}
              onChange={(e) => setPlaybackSpeed(Number(e.target.value))}
              className="text-xs border rounded px-2 py-1 bg-white dark:bg-gray-700"
            >
              <option value={500}>Fast (0.5s)</option>
              <option value={1000}>Normal (1s)</option>
              <option value={2000}>Slow (2s)</option>
              <option value={3000}>Very Slow (3s)</option>
            </select>
          </div>
        </div>

        {/* Generated Text Display */}
        {currentStepData && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-medium">Generated Sequence</h3>
              <Badge variant="secondary">
                {displayTokens.length} tokens
              </Badge>
            </div>

            <div className="p-4 bg-white dark:bg-gray-900 rounded-lg border">
              <AnimatePresence mode="wait">
                <motion.div
                  key={currentStep}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.3 }}
                  className="flex flex-wrap gap-2"
                >
                  {displayTokens.map((token, idx) => (
                    <motion.span
                      key={idx}
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: idx * 0.02 }}
                      className={`inline-block px-2 py-1 rounded text-sm ${
                        token.isNew
                          ? 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-200 font-medium'
                          : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300'
                      }`}
                    >
                      {token.word}
                    </motion.span>
                  ))}
                </motion.div>
              </AnimatePresence>
            </div>
          </div>
        )}

        {/* Current Token Info */}
        {currentStepData && currentStepData.stepIndex > 0 && (
          <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
            <h4 className="text-sm font-medium mb-3 text-blue-800 dark:text-blue-200">
              Latest Generated Token
            </h4>
            <div className="flex items-center gap-4">
              <div className="flex-1">
                <div className="text-2xl font-bold text-blue-900 dark:text-blue-100">
                  {currentStepData.generatedToken.token}
                </div>
                <div className="text-xs text-blue-700 dark:text-blue-300 mt-1">
                  Token ID: {currentStepData.generatedToken.tokenId}
                </div>
              </div>
              <div className="text-right">
                <div className="text-sm text-blue-700 dark:text-blue-300">
                  Probability:
                </div>
                <div className="text-xl font-bold text-blue-900 dark:text-blue-100">
                  {(currentStepData.generatedToken.probability * 100).toFixed(1)}%
                </div>
              </div>
              <div className="text-right">
                <div className="text-sm text-blue-700 dark:text-blue-300">
                  Time:
                </div>
                <div className="text-xl font-bold text-blue-900 dark:text-blue-100 flex items-center justify-end gap-1">
                  <Clock className="h-4 w-4" />
                  {currentStepData.timeTaken.toFixed(0)}ms
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Probability Distribution */}
        {showDistribution && currentStepData && currentStepData.distribution.tokens.length > 0 && (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-medium">Top Candidates at This Step</h3>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowDistribution(!showDistribution)}
              >
                {showDistribution ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
              </Button>
            </div>

            <div className="space-y-2">
              {currentStepData.distribution.topKTokens.slice(0, 5).map((token, idx) => (
                <motion.div
                  key={token.tokenId}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.05 }}
                  className={`flex items-center gap-3 p-2 rounded-lg ${
                    idx === 0
                      ? 'bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800'
                      : 'bg-gray-50 dark:bg-gray-800'
                  }`}
                >
                  <div className="w-8 text-center text-sm font-mono text-gray-500">
                    #{token.rank}
                  </div>
                  <div className="flex-1 font-mono text-sm">
                    {token.token}
                  </div>
                  <div className="w-32">
                    <div className="relative h-4 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${token.probability * 100}%` }}
                        transition={{ duration: 0.5 }}
                        className="h-full bg-gradient-to-r from-blue-500 to-purple-500"
                      />
                    </div>
                  </div>
                  <div className="text-sm font-medium w-16 text-right">
                    {(token.probability * 100).toFixed(1)}%
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        )}

        {/* Sampling Options */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium flex items-center gap-2">
              <TrendingUp className="h-4 w-4" />
              Sampling Options
            </h3>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowAdvanced(!showAdvanced)}
            >
              {showAdvanced ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
            </Button>
          </div>

          <div className="grid grid-cols-3 gap-6">
            {/* Strategy */}
            <div className="space-y-2">
              <label className="text-xs text-gray-600 dark:text-gray-400">Strategy</label>
              <select
                value={samplingOptions.strategy}
                onChange={(e) => setSamplingOptions({ ...samplingOptions, strategy: e.target.value as SamplingStrategy })}
                className="w-full text-sm border rounded px-2 py-1.5 bg-white dark:bg-gray-700"
              >
                <option value="greedy">Greedy</option>
                <option value="multinomial">Multinomial</option>
                <option value="top-k">Top-K</option>
                <option value="top-p">Top-P (Nucleus)</option>
              </select>
            </div>

            {/* Temperature */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-xs text-gray-600 dark:text-gray-400">
                  <Zap className="h-3 w-3 inline mr-1" />
                  Temperature
                </label>
                <Badge variant="outline">{samplingOptions.temperature?.toFixed(2)}</Badge>
              </div>
              <Slider
                value={[samplingOptions.temperature || 1.0]}
                onValueChange={([value]) => setSamplingOptions({ ...samplingOptions, temperature: value })}
                min={0.1}
                max={2}
                step={0.1}
                className="w-full"
              />
            </div>

            {/* Top-K */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-xs text-gray-600 dark:text-gray-400">Top-K</label>
                <Badge variant="outline">{samplingOptions.topK}</Badge>
              </div>
              <Slider
                value={[samplingOptions.topK || 50]}
                onValueChange={([value]) => setSamplingOptions({ ...samplingOptions, topK: value })}
                min={1}
                max={100}
                step={5}
                className="w-full"
              />
            </div>
          </div>
        </div>

        {/* Info Box */}
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
          <div className="flex items-start gap-3">
            <Info className="h-5 w-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
            <div className="text-sm text-blue-900 dark:text-blue-100">
              <h4 className="font-medium mb-1">Understanding Autoregressive Generation</h4>
              <p className="text-blue-800 dark:text-blue-200">
                Transformers generate text one token at a time, using previously generated tokens
                as context for the next prediction. Each prediction is based on all previous tokens,
                making the process sequential and autoregressive.
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
