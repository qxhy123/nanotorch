/**
 * SamplingStrategyComparison Component
 *
 * Compares different sampling strategies (greedy, multinomial, top-k, top-p)
 * showing how different parameters affect the output.
 */

import React, { useState, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../ui/card';
import { Badge } from '../../ui/badge';
import { Button } from '../../ui/button';
import { Slider } from '../../ui/slider';
import { motion, AnimatePresence } from 'framer-motion';
import {
  GitCompare,
  Play,
  Settings,
  Zap,
  TrendingUp,
  Info,
  RefreshCw,
} from 'lucide-react';
import type {
  StrategyComparison,
  SamplingStrategy,
  SamplingOptions,
} from '../../../types/inference';

interface SamplingStrategyComparisonProps {
  sequence?: string;
  className?: string;
}

const STRATEGIES: { id: SamplingStrategy; name: string; description: string; icon: string }[] = [
  { id: 'greedy', name: 'Greedy', description: 'Always pick the highest probability token', icon: '🎯' },
  { id: 'multinomial', name: 'Multinomial', description: 'Sample from the full distribution', icon: '🎲' },
  { id: 'top-k', name: 'Top-K', description: 'Sample from top K most likely tokens', icon: '📊' },
  { id: 'top-p', name: 'Top-P (Nucleus)', description: 'Sample from smallest set covering P% mass', icon: '🔬' },
];

export const SamplingStrategyComparison: React.FC<SamplingStrategyComparisonProps> = ({
  sequence = 'The future of AI',
  className = '',
}) => {
  const [comparisons, setComparisons] = useState<StrategyComparison[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedStrategies, setSelectedStrategies] = useState<SamplingStrategy[]>(['greedy', 'top-k', 'top-p']);
  const [options, setOptions] = useState<Partial<SamplingOptions>>({
    temperature: 1.0,
    topK: 50,
    topP: 0.9,
    beamWidth: 5,
  });

  // Run comparison
  const runComparison = useCallback(async () => {
    setLoading(true);
    try {
      // Generate mock comparison data for now
      const mockComparisons: StrategyComparison[] = selectedStrategies.map(strategy => {
        const strategyOptions = { ...options, strategy } as SamplingOptions;

        // Generate mock result based on strategy
        let result = sequence;
        let timeMs = Math.random() * 200 + 100;

        switch (strategy) {
          case 'greedy':
            result = 'The future of AI is bright and promising';
            break;
          case 'multinomial':
            result = 'The future of AI holds endless possibilities';
            break;
          case 'top-k':
            result = 'The future of AI will transform society';
            break;
          case 'top-p':
            result = 'The future of AI looks incredibly exciting';
            break;
        }

        return {
          strategy,
          options: strategyOptions,
          result,
          steps: [], // Mock steps
          timeMs,
        };
      });

      setComparisons(mockComparisons);
    } catch (error) {
      console.error('Failed to run comparison:', error);
    } finally {
      setLoading(false);
    }
  }, [selectedStrategies, options, sequence]);

  // Initial run
  React.useEffect(() => {
    runComparison();
  }, []);

  // Toggle strategy selection
  const toggleStrategy = useCallback((strategy: SamplingStrategy) => {
    setSelectedStrategies(prev => {
      const isSelected = prev.includes(strategy);
      if (isSelected && prev.length > 1) {
        return prev.filter(s => s !== strategy);
      } else if (!isSelected) {
        return [...prev, strategy];
      }
      return prev;
    });
  }, []);

  // Update options
  const updateOptions = useCallback((updates: Partial<SamplingOptions>) => {
    setOptions(prev => ({ ...prev, ...updates }));
  }, []);

  // Get strategy info
  const getStrategyInfo = useCallback((strategy: SamplingStrategy) => {
    return STRATEGIES.find(s => s.id === strategy) || STRATEGIES[0];
  }, []);

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <GitCompare className="h-5 w-5 text-primary" />
              Sampling Strategy Comparison
            </CardTitle>
            <CardDescription>
              Compare different sampling strategies and their effects
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={runComparison} disabled={loading}>
              {loading ? <RefreshCw className="h-4 w-4 mr-1 animate-spin" /> : <Play className="h-4 w-4 mr-1" />}
              Run Comparison
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Strategy Selector */}
        <div className="space-y-3">
          <h3 className="text-sm font-medium flex items-center gap-2">
            <Settings className="h-4 w-4" />
            Select Strategies to Compare
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {STRATEGIES.map(strategy => (
              <button
                key={strategy.id}
                onClick={() => toggleStrategy(strategy.id)}
                className={`p-3 rounded-lg border-2 transition-all ${
                  selectedStrategies.includes(strategy.id)
                    ? 'border-primary bg-primary/5'
                    : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
                }`}
              >
                <div className="text-2xl mb-1">{strategy.icon}</div>
                <div className="text-sm font-medium">{strategy.name}</div>
                <div className="text-xs text-gray-500 mt-1">{strategy.description}</div>
              </button>
            ))}
          </div>
        </div>

        {/* Parameter Controls */}
        <div className="space-y-4">
          <h3 className="text-sm font-medium flex items-center gap-2">
            <Zap className="h-4 w-4" />
            Parameters
          </h3>

          <div className="grid grid-cols-3 gap-6">
            {/* Temperature */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-xs text-gray-600 dark:text-gray-400">
                  Temperature
                </label>
                <Badge variant="outline">{options.temperature?.toFixed(2)}</Badge>
              </div>
              <Slider
                value={[options.temperature || 1.0]}
                onValueChange={([value]) => updateOptions({ temperature: value })}
                min={0.1}
                max={2}
                step={0.1}
                className="w-full"
              />
              <p className="text-xs text-gray-500">
                Lower = more focused, Higher = more diverse
              </p>
            </div>

            {/* Top-K */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-xs text-gray-600 dark:text-gray-400">
                  Top-K
                </label>
                <Badge variant="outline">{options.topK}</Badge>
              </div>
              <Slider
                value={[options.topK || 50]}
                onValueChange={([value]) => updateOptions({ topK: value })}
                min={1}
                max={100}
                step={5}
                className="w-full"
              />
              <p className="text-xs text-gray-500">
                Number of top tokens to consider
              </p>
            </div>

            {/* Top-P */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-xs text-gray-600 dark:text-gray-400">
                  Top-P
                </label>
                <Badge variant="outline">{((options.topP || 0.9) * 100).toFixed(0)}%</Badge>
              </div>
              <Slider
                value={[options.topP || 0.9]}
                onValueChange={([value]) => updateOptions({ topP: value })}
                min={0}
                max={1}
                step={0.05}
                className="w-full"
              />
              <p className="text-xs text-gray-500">
                Cumulative probability threshold
              </p>
            </div>
          </div>
        </div>

        {/* Comparison Results */}
        <AnimatePresence mode="wait">
          {comparisons.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-4"
            >
              <h3 className="text-sm font-medium flex items-center gap-2">
                <TrendingUp className="h-4 w-4" />
                Results
              </h3>

              <div className="space-y-3">
                {comparisons.map((comparison) => {
                  const info = getStrategyInfo(comparison.strategy);
                  return (
                    <motion.div
                      key={comparison.strategy}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg"
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <span className="text-2xl">{info.icon}</span>
                          <div>
                            <h4 className="font-medium">{info.name}</h4>
                            <p className="text-xs text-gray-500">{info.description}</p>
                          </div>
                        </div>
                        <Badge variant="secondary">
                          {comparison.timeMs.toFixed(0)}ms
                        </Badge>
                      </div>

                      <div className="p-3 bg-white dark:bg-gray-900 rounded border">
                        <p className="font-mono text-sm">{comparison.result}</p>
                      </div>

                      {/* Strategy-specific parameters */}
                      <div className="flex gap-2 mt-2">
                        {comparison.strategy === 'multinomial' && (
                          <Badge variant="outline" className="text-xs">
                            T = {comparison.options.temperature.toFixed(2)}
                          </Badge>
                        )}
                        {comparison.strategy === 'top-k' && (
                          <Badge variant="outline" className="text-xs">
                            K = {comparison.options.topK}
                          </Badge>
                        )}
                        {comparison.strategy === 'top-p' && (
                          <Badge variant="outline" className="text-xs">
                            P = {(comparison.options.topP * 100).toFixed(0)}%
                          </Badge>
                        )}
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Info Box */}
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
          <div className="flex items-start gap-3">
            <Info className="h-5 w-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
            <div className="text-sm text-blue-900 dark:text-blue-100">
              <h4 className="font-medium mb-1">About Sampling Strategies</h4>
              <ul className="space-y-1 text-blue-800 dark:text-blue-200">
                <li><strong>Greedy:</strong> Always selects the most likely token - deterministic but can be repetitive</li>
                <li><strong>Multinomial:</strong> Samples from full distribution - more diverse but less focused</li>
                <li><strong>Top-K:</strong> Limits sampling to K most likely tokens - balances focus and diversity</li>
                <li><strong>Top-P:</strong> Uses smallest set covering P% probability - adaptive vocabulary size</li>
              </ul>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
