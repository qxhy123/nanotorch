/**
 * ProbabilityDistribution Component
 *
 * Visualizes softmax output probability distribution across vocabulary.
 * Shows top-k highlighting, probability threshold filtering, and comparison
 * between different sampling strategies.
 */

import React, { useState, useMemo, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../ui/card';
import { Badge } from '../../ui/badge';
import { Button } from '../../ui/button';
import { Slider } from '../../ui/slider';
import { motion, AnimatePresence } from 'framer-motion';
import {
  BarChart3,
  Settings,
  Zap,
  Info,
  ChevronDown,
  ChevronUp,
  RefreshCw,
} from 'lucide-react';
import { inferenceApi } from '../../../services/inferenceApi';
import type {
  ProbabilityDistributionData,
  ProbabilityVizState,
} from '../../../types/inference';

interface ProbabilityDistributionProps {
  sequence?: string;
  className?: string;
}

export const ProbabilityDistribution: React.FC<ProbabilityDistributionProps> = ({
  sequence = 'The future of AI',
  className = '',
}) => {
  // State
  const [data, setData] = useState<ProbabilityDistributionData | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedPosition, setSelectedPosition] = useState(0);
  const [vizState, setVizState] = useState<ProbabilityVizState>({
    selectedPosition: 0,
    showTopK: 20,
    showTopP: 0.9,
    temperature: 1.0,
    highlightTokens: [],
    showLogProbs: false,
    sortBy: 'probability',
  });
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Load data
  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      // Use mock data for now - replace with real API call when backend is ready
      const mockData = inferenceApi.generateMockProbabilityDistribution(sequence);
      setData(mockData);
      if (mockData.distributions.length > 0) {
        setSelectedPosition(0);
      }
    } catch (error) {
      console.error('Failed to load probability distribution:', error);
    } finally {
      setLoading(false);
    }
  }, [sequence]);

  // Initial load
  React.useEffect(() => {
    loadData();
  }, [loadData]);

  // Get current distribution
  const currentDistribution = useMemo(() => {
    if (!data || selectedPosition >= data.distributions.length) return null;
    return data.distributions[selectedPosition];
  }, [data, selectedPosition]);

  // Filter and sort tokens based on viz state
  const filteredTokens = useMemo(() => {
    if (!currentDistribution) return [];

    let tokens = [...currentDistribution.tokens];

    // Apply top-k filter
    if (vizState.showTopK > 0) {
      tokens = tokens.slice(0, vizState.showTopK);
    }

    // Apply top-p filter
    if (vizState.showTopP > 0 && vizState.showTopP < 1) {
      let cumulative = 0;
      tokens = tokens.filter(t => {
        cumulative += t.probability;
        return cumulative <= vizState.showTopP;
      });
    }

    // Sort
    switch (vizState.sortBy) {
      case 'token':
        tokens.sort((a, b) => a.token.localeCompare(b.token));
        break;
      case 'id':
        tokens.sort((a, b) => a.tokenId - b.tokenId);
        break;
      case 'probability':
      default:
        // Already sorted by probability (descending)
        break;
    }

    return tokens;
  }, [currentDistribution, vizState]);

  // Calculate statistics
  const statistics = useMemo(() => {
    if (!currentDistribution) return null;
    const tokens = currentDistribution.tokens;

    const entropy = currentDistribution.entropy;
    const perplexity = Math.pow(2, entropy);

    const maxProb = tokens[0]?.probability || 0;
    const minProb = tokens[tokens.length - 1]?.probability || 0;
    const avgProb = tokens.reduce((sum, t) => sum + t.probability, 0) / tokens.length;

    // Top-k coverage
    const topKCov = new Map<number, number>();
    [1, 5, 10, 50, 100].forEach(k => {
      const topK = tokens.slice(0, Math.min(k, tokens.length));
      topKCov.set(k, topK.reduce((s, t) => s + t.probability, 0));
    });

    return {
      entropy,
      perplexity,
      maxProb,
      minProb,
      avgProb,
      topKCov,
    };
  }, [currentDistribution]);

  // Update state handler
  const updateVizState = useCallback((updates: Partial<ProbabilityVizState>) => {
    setVizState(prev => ({ ...prev, ...updates }));
  }, []);

  if (loading) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center py-12">
          <RefreshCw className="h-8 w-8 animate-spin text-primary" />
        </CardContent>
      </Card>
    );
  }

  if (!data || !currentDistribution) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center py-12">
          <div className="text-center text-gray-500">
            <BarChart3 className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p className="text-lg font-medium">No probability data available</p>
            <p className="text-sm">Generate some text to see probability distributions</p>
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
              <BarChart3 className="h-5 w-5 text-primary" />
              Probability Distribution
            </CardTitle>
            <CardDescription>
              Softmax output probabilities at each position
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={loadData}>
              <RefreshCw className="h-4 w-4 mr-1" />
              Refresh
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Position Selector */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium">Position</label>
            <Badge variant="secondary">
              {selectedPosition + 1} / {data.distributions.length}
            </Badge>
          </div>
          <div className="flex gap-2 flex-wrap">
            {data.tokens.map((token, idx) => (
              <Button
                key={idx}
                variant={selectedPosition === idx ? 'default' : 'outline'}
                size="sm"
                onClick={() => setSelectedPosition(idx)}
                className="min-w-[60px]"
              >
                {token || `<EOS>`}
              </Button>
            ))}
          </div>
        </div>

        {/* Statistics Cards */}
        {statistics && (
          <div className="grid grid-cols-4 gap-4">
            <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <div className="text-xs text-gray-600 dark:text-gray-400">Entropy</div>
              <div className="text-lg font-bold text-blue-700 dark:text-blue-300">
                {statistics.entropy.toFixed(3)} bits
              </div>
            </div>
            <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
              <div className="text-xs text-gray-600 dark:text-gray-400">Perplexity</div>
              <div className="text-lg font-bold text-purple-700 dark:text-purple-300">
                {statistics.perplexity.toFixed(1)}
              </div>
            </div>
            <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
              <div className="text-xs text-gray-600 dark:text-gray-400">Max Prob</div>
              <div className="text-lg font-bold text-green-700 dark:text-green-300">
                {(statistics.maxProb * 100).toFixed(1)}%
              </div>
            </div>
            <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
              <div className="text-xs text-gray-600 dark:text-gray-400">Avg Prob</div>
              <div className="text-lg font-bold text-orange-700 dark:text-orange-300">
                {(statistics.avgProb * 100).toFixed(2)}%
              </div>
            </div>
          </div>
        )}

        {/* Controls */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium flex items-center gap-2">
              <Settings className="h-4 w-4" />
              Visualization Controls
            </h3>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowAdvanced(!showAdvanced)}
            >
              {showAdvanced ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
            </Button>
          </div>

          <div className="grid grid-cols-2 gap-6">
            {/* Top-K Filter */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-xs text-gray-600 dark:text-gray-400">
                  Top-K Tokens
                </label>
                <Badge variant="outline">{vizState.showTopK}</Badge>
              </div>
              <Slider
                value={[vizState.showTopK]}
                onValueChange={([value]) => updateVizState({ showTopK: value })}
                min={5}
                max={100}
                step={5}
                className="w-full"
              />
            </div>

            {/* Top-P Filter */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-xs text-gray-600 dark:text-gray-400">
                  Top-P (Nucleus)
                </label>
                <Badge variant="outline">{(vizState.showTopP * 100).toFixed(0)}%</Badge>
              </div>
              <Slider
                value={[vizState.showTopP]}
                onValueChange={([value]) => updateVizState({ showTopP: value })}
                min={0}
                max={1}
                step={0.05}
                className="w-full"
              />
            </div>

            {/* Temperature */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-xs text-gray-600 dark:text-gray-400">
                  <Zap className="h-3 w-3 inline mr-1" />
                  Temperature
                </label>
                <Badge variant="outline">{vizState.temperature.toFixed(2)}</Badge>
              </div>
              <Slider
                value={[vizState.temperature]}
                onValueChange={([value]) => updateVizState({ temperature: value })}
                min={0.1}
                max={2}
                step={0.1}
                className="w-full"
              />
            </div>

            {/* Sort By */}
            <div className="space-y-2">
              <label className="text-xs text-gray-600 dark:text-gray-400">
                Sort By
              </label>
              <div className="flex gap-2">
                {(['probability', 'token', 'id'] as const).map(mode => (
                  <Button
                    key={mode}
                    variant={vizState.sortBy === mode ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => updateVizState({ sortBy: mode })}
                    className="flex-1"
                  >
                    {mode === 'probability' ? 'Prob' : mode}
                  </Button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Probability Bar Chart */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium">Top Tokens</h3>
            <Badge variant="secondary">
              Showing {filteredTokens.length} tokens
            </Badge>
          </div>

          <div className="space-y-2 max-h-[400px] overflow-y-auto">
            <AnimatePresence mode="popLayout">
              {filteredTokens.map((token, idx) => (
                <motion.div
                  key={token.tokenId}
                  layout
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  transition={{ delay: idx * 0.02 }}
                  className={`flex items-center gap-3 p-2 rounded-lg transition-colors ${
                    idx === 0 ? 'bg-green-50 dark:bg-green-900/20' : 'hover:bg-gray-50 dark:hover:bg-gray-800'
                  }`}
                >
                  {/* Rank */}
                  <div className="w-8 text-center text-sm font-mono text-gray-500">
                    #{token.rank}
                  </div>

                  {/* Token */}
                  <div className="flex-1 font-mono text-sm">
                    {token.token}
                  </div>

                  {/* Token ID */}
                  <div className="text-xs text-gray-500">
                    ID: {token.tokenId}
                  </div>

                  {/* Probability Bar */}
                  <div className="w-48">
                    <div className="relative h-6 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${token.probability * 100}%` }}
                        transition={{ duration: 0.5 }}
                        className="h-full bg-gradient-to-r from-blue-500 to-purple-500"
                      />
                      <span className="absolute inset-0 flex items-center justify-center text-xs font-medium">
                        {(token.probability * 100).toFixed(2)}%
                      </span>
                    </div>
                  </div>

                  {/* Log Probability */}
                  {vizState.showLogProbs && (
                    <div className="text-xs text-gray-500 w-20 text-right">
                      {token.logProbability.toFixed(3)}
                    </div>
                  )}
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>

        {/* Top-K Coverage */}
        {statistics && statistics.topKCov.size > 0 && (
          <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
            <h4 className="text-sm font-medium mb-3">Top-K Coverage</h4>
            <div className="grid grid-cols-5 gap-4">
              {Array.from(statistics.topKCov.entries()).map(([k, coverage]) => (
                <div key={k} className="text-center">
                  <div className="text-xs text-gray-600 dark:text-gray-400">Top-{k}</div>
                  <div className="text-lg font-bold text-blue-600 dark:text-blue-400">
                    {(coverage * 100).toFixed(1)}%
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Info Box */}
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
          <div className="flex items-start gap-3">
            <Info className="h-5 w-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
            <div className="text-sm text-blue-900 dark:text-blue-100">
              <h4 className="font-medium mb-1">Understanding Probability Distributions</h4>
              <p className="text-blue-800 dark:text-blue-200">
                The softmax output shows the model's predicted probability for each vocabulary token.
                Higher values indicate more likely next tokens. Entropy measures uncertainty - lower entropy
                means the model is more confident in its prediction.
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
