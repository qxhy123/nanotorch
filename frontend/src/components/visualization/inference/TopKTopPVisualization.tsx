/**
 * TopKTopPVisualization Component
 *
 * Interactive demonstration of sampling strategies:
 * - Top-K sampling visualization
 * - Top-p (nucleus) sampling visualization
 * - Side-by-side comparison
 * - Interactive K and p value controls
 * - Animated sampling process
 * - Temperature effects
 * - Greedy vs sampling comparison
 */

import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../ui/card';
import { Button } from '../../ui/button';
import { Slider } from '../../ui/slider';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import {
  TrendingUp,
  Zap,
  Settings,
  Play,
  RotateCcw,
  Filter,
  PieChart,
  Sparkles,
} from 'lucide-react';

// Token probability data
interface TokenProb {
  token: string;
  probability: number;
  logProb: number;
  cumulative: number;
}

// Generate mock vocabulary probabilities
function generateVocabularyProbs(
  temperature: number = 1.0
): TokenProb[] {
  // Start with a base distribution (power law)
  const tokens = [
    'the', 'a', 'an', 'is', 'are', 'was', 'were',
    'cat', 'dog', 'bird', 'fish', 'tree', 'house',
    'quick', 'brown', 'lazy', 'jumped', 'ran', 'walked',
    'over', 'under', 'through', 'around', 'between',
    'happy', 'sad', 'angry', 'excited', 'calm',
    'very', 'quite', 'rather', 'somewhat', 'extremely',
    'beautiful', 'ugly', 'big', 'small', 'huge', 'tiny',
    'red', 'blue', 'green', 'yellow', 'purple'
  ];

  // Generate probabilities using Zipf-like distribution with temperature
  let rawProbs = tokens.map((token, idx) => {
    const baseRank = idx + 1;
    const tempAdjustedRank = Math.pow(baseRank, 1 / temperature);
    return {
      token,
      probability: 1 / tempAdjustedRank,
      logProb: -Math.log(tempAdjustedRank),
    };
  });

  // Normalize
  const total = rawProbs.reduce((sum, t) => sum + t.probability, 0);
  rawProbs = rawProbs.map((t) => ({
    ...t,
    probability: t.probability / total,
  }));

  // Sort by probability
  rawProbs.sort((a, b) => b.probability - a.probability);

  // Calculate cumulative probability
  let cumulative = 0;
  return rawProbs.map((t) => {
    cumulative += t.probability;
    return { ...t, cumulative };
  });
}

// Apply Top-K filtering
function applyTopK(probs: TokenProb[], k: number): TokenProb[] {
  return probs.slice(0, Math.min(k, probs.length));
}

// Apply Top-p (nucleus) filtering
function applyTopP(probs: TokenProb[], p: number): TokenProb[] {
  const filtered: TokenProb[] = [];
  for (const prob of probs) {
    filtered.push(prob);
    if (prob.cumulative >= p) break;
  }
  return filtered;
}

// Sample from filtered probabilities
function sampleToken(probs: TokenProb[]): { token: string; probability: number } | null {
  if (probs.length === 0) return null;

  // Renormalize filtered probabilities
  const total = probs.reduce((sum, t) => sum + t.probability, 0);
  const normalizedProbs = probs.map((t) => ({
    ...t,
    probability: t.probability / total,
  }));

  // Sample
  const rand = Math.random();
  let cumulative = 0;
  for (const prob of normalizedProbs) {
    cumulative += prob.probability;
    if (rand <= cumulative) {
      return { token: prob.token, probability: prob.probability };
    }
  }

  return { token: normalizedProbs[0].token, probability: normalizedProbs[0].probability };
}

interface TopKTopPVisualizationProps {
  className?: string;
}

export const TopKTopPVisualization: React.FC<TopKTopPVisualizationProps> = ({
  className = '',
}) => {
  const [temperature, setTemperature] = useState(1.0);
  const [topK, setTopK] = useState(10);
  const [topP, setTopP] = useState(0.9);
  const [isAnimating, setIsAnimating] = useState(false);
  const [animationStep, setAnimationStep] = useState(0);
  const [sampledToken, setSampledToken] = useState<{ token: string; strategy: string } | null>(null);

  // Generate base probabilities
  const baseProbs = useMemo(() => generateVocabularyProbs(temperature), [temperature]);

  // Apply Top-K
  const topKProbs = useMemo(() => {
    const filtered = applyTopK(baseProbs, topK);
    return filtered.map((p, idx) => ({ ...p, originalIndex: idx }));
  }, [baseProbs, topK]);

  // Apply Top-p
  const topPProbs = useMemo(() => {
    const filtered = applyTopP(baseProbs, topP);
    return filtered.map((p, idx) => ({ ...p, originalIndex: idx }));
  }, [baseProbs, topP]);

  // Calculate statistics
  const stats = useMemo(() => {
    return {
      totalTokens: baseProbs.length,
      topKTokens: topKProbs.length,
      topPTokens: topPProbs.length,
      topKProbability: topKProbs.reduce((sum, t) => sum + t.probability, 0),
      topPProbability: topPProbs.reduce((sum, t) => sum + t.probability, 0),
      entropy:
        -baseProbs.reduce((sum, t) => sum + t.probability * Math.log2(t.probability), 0),
    };
  }, [baseProbs, topKProbs, topPProbs]);

  // Run sampling animation
  const runAnimation = () => {
    setIsAnimating(true);
    setSampledToken(null);
    setAnimationStep(0);

    const steps = [
      () => setAnimationStep(1),
      () => setAnimationStep(2),
      () => {
        const kSample = sampleToken(topKProbs);
        const pSample = sampleToken(topPProbs);
        setSampledToken({
          token: kSample?.token || pSample?.token || 'N/A',
          strategy: kSample?.token === pSample?.token ? 'both' : kSample?.token || 'top-p',
        });
        setAnimationStep(3);
      },
      () => {
        setIsAnimating(false);
        setAnimationStep(0);
      },
    ];

    let delay = 0;
    steps.forEach((step) => {
      setTimeout(step, delay);
      delay += 800;
    });
  };

  // Reset
  const reset = () => {
    setIsAnimating(false);
    setAnimationStep(0);
    setSampledToken(null);
  };

  // Prepare chart data
  const topKChartData = topKProbs.slice(0, 15).map((p) => ({
    token: p.token,
    probability: (p.probability * 100).toFixed(2),
    filtered: true,
  }));

  const topPChartData = topPProbs.slice(0, 15).map((p) => ({
    token: p.token,
    probability: (p.probability * 100).toFixed(2),
    filtered: true,
  }));

  const baseChartData = baseProbs.slice(0, 15).map((p) => ({
    token: p.token,
    probability: (p.probability * 100).toFixed(2),
    filtered: false,
  }));

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Filter className="h-5 w-5 text-primary" />
              Top-K & Top-p Sampling Visualization
            </CardTitle>
            <CardDescription>
              Interactive demonstration of nucleus sampling strategies
            </CardDescription>
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={runAnimation}
              disabled={isAnimating}
            >
              <Play className="h-4 w-4 mr-1" />
              Sample
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={reset}
              disabled={isAnimating}
            >
              <RotateCcw className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Parameters Control */}
        <div className="space-y-4 p-4 bg-muted rounded-lg">
          <h3 className="text-sm font-medium flex items-center gap-2">
            <Settings className="h-4 w-4" />
            Sampling Parameters
          </h3>

          {/* Temperature */}
          <div className="space-y-2">
            <div className="flex items-center gap-4">
              <label className="text-sm font-medium min-w-[100px]">Temperature:</label>
              <Slider
                value={[temperature]}
                onValueChange={([v]) => setTemperature(v)}
                min={0.1}
                max={2.0}
                step={0.1}
                className="flex-1"
              />
              <span className="text-sm font-mono w-12 text-right">{temperature.toFixed(1)}</span>
            </div>
            <p className="text-xs text-muted-foreground">
              Lower temperature = more focused, Higher temperature = more diverse
            </p>
          </div>

          {/* Top-K */}
          <div className="space-y-2">
            <div className="flex items-center gap-4">
              <label className="text-sm font-medium min-w-[100px]">Top-K:</label>
              <Slider
                value={[topK]}
                onValueChange={([v]) => setTopK(v)}
                min={1}
                max={50}
                step={1}
                className="flex-1"
              />
              <span className="text-sm font-mono w-12 text-right">{topK}</span>
            </div>
            <p className="text-xs text-muted-foreground">
              Sample from the K most likely tokens
            </p>
          </div>

          {/* Top-p */}
          <div className="space-y-2">
            <div className="flex items-center gap-4">
              <label className="text-sm font-medium min-w-[100px]">Top-p:</label>
              <Slider
                value={[topP]}
                onValueChange={([v]) => setTopP(v)}
                min={0.1}
                max={1.0}
                step={0.05}
                className="flex-1"
              />
              <span className="text-sm font-mono w-12 text-right">{topP.toFixed(2)}</span>
            </div>
            <p className="text-xs text-muted-foreground">
              Sample from smallest set of tokens with cumulative probability ≥ p
            </p>
          </div>
        </div>

        {/* Statistics Cards */}
        <div className="grid md:grid-cols-4 gap-4">
          <div className="p-4 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-lg">
            <div className="text-xs text-blue-600 dark:text-blue-400 mb-1">Total Tokens</div>
            <div className="text-2xl font-bold">{stats.totalTokens}</div>
          </div>

          <div className="p-4 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-lg">
            <div className="text-xs text-green-600 dark:text-green-400 mb-1">Top-K Selected</div>
            <div className="text-2xl font-bold">{stats.topKTokens}</div>
            <div className="text-xs text-muted-foreground mt-1">
              {stats.topKProbability.toFixed(1)}% prob mass
            </div>
          </div>

          <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 rounded-lg">
            <div className="text-xs text-purple-600 dark:text-purple-400 mb-1">Top-p Selected</div>
            <div className="text-2xl font-bold">{stats.topPTokens}</div>
            <div className="text-xs text-muted-foreground mt-1">
              {stats.topPProbability.toFixed(1)}% prob mass
            </div>
          </div>

          <div className="p-4 bg-gradient-to-br from-orange-50 to-orange-100 dark:from-orange-900/20 dark:to-orange-800/20 rounded-lg">
            <div className="text-xs text-orange-600 dark:text-orange-400 mb-1">Entropy</div>
            <div className="text-2xl font-bold">{stats.entropy.toFixed(2)}</div>
            <div className="text-xs text-muted-foreground mt-1">
              bits
            </div>
          </div>
        </div>

        {/* Sampling Animation */}
        {isAnimating && (
          <div className="p-4 bg-gradient-to-br from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
            <div className="flex items-center gap-3">
              <Sparkles className="h-5 w-5 text-yellow-600 dark:text-yellow-400 animate-pulse" />
              <div className="flex-1">
                <div className="text-sm font-medium text-yellow-800 dark:text-yellow-200">
                  {animationStep === 0 && 'Initializing sampling...'}
                  {animationStep === 1 && 'Filtering candidates...'}
                  {animationStep === 2 && 'Sampling from filtered distribution...'}
                  {animationStep === 3 && `Sampled token: "${sampledToken?.token}"`}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Comparison Charts */}
        <div className="grid md:grid-cols-2 gap-6">
          {/* Top-K Chart */}
          <div className="space-y-3">
            <h3 className="text-sm font-medium flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-green-500" />
              Top-K Sampling (K={topK})
            </h3>
            <div className="bg-background rounded-lg p-4 border">
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={topKChartData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis type="number" className="text-xs" />
                  <YAxis
                    type="category"
                    dataKey="token"
                    width={60}
                    className="text-xs"
                  />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        return (
                          <div className="bg-background border rounded-lg p-2 shadow-lg">
                            <p className="text-xs font-medium">{payload[0].payload.token}</p>
                            <p className="text-xs">
                              Probability: {payload[0].payload.probability}%
                            </p>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Bar dataKey="probability" fill="#22c55e">
                    {topKChartData.map((_, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={
                          animationStep >= 2 && sampledToken?.strategy !== 'top-p' && index === 0
                            ? '#f97316'
                            : '#22c55e'
                        }
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="text-xs text-muted-foreground text-center">
              Shows top {topKProbs.length} tokens by probability
            </div>
          </div>

          {/* Top-p Chart */}
          <div className="space-y-3">
            <h3 className="text-sm font-medium flex items-center gap-2">
              <PieChart className="h-4 w-4 text-purple-500" />
              Top-p Sampling (p={topP.toFixed(2)})
            </h3>
            <div className="bg-background rounded-lg p-4 border">
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={topPChartData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis type="number" className="text-xs" />
                  <YAxis
                    type="category"
                    dataKey="token"
                    width={60}
                    className="text-xs"
                  />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        return (
                          <div className="bg-background border rounded-lg p-2 shadow-lg">
                            <p className="text-xs font-medium">{payload[0].payload.token}</p>
                            <p className="text-xs">
                              Probability: {payload[0].payload.probability}%
                            </p>
                            <p className="text-xs text-muted-foreground">
                              Cumulative: {(baseProbs.find((b) => b.token === payload[0].payload.token)?.cumulative! * 100).toFixed(1)}%
                            </p>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Bar dataKey="probability" fill="#a855f7">
                    {topPChartData.map((_, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={
                          animationStep >= 2 && sampledToken?.strategy !== 'top-k' && index === 0
                            ? '#f97316'
                            : '#a855f7'
                        }
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="text-xs text-muted-foreground text-center">
              Shows {topPProbs.length} tokens with cumulative probability ≥ {(topP * 100).toFixed(0)}%
            </div>
          </div>
        </div>

        {/* Full Distribution Comparison */}
        <div className="space-y-3">
          <h3 className="text-sm font-medium">Full Distribution Comparison</h3>
          <div className="bg-background rounded-lg p-4 border">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={baseChartData}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis
                  dataKey="token"
                  angle={-45}
                  textAnchor="end"
                  height={80}
                  className="text-xs"
                />
                <YAxis className="text-xs" />
                <Tooltip />
                <Legend />
                <Bar
                  dataKey="probability"
                  fill="#3b82f6"
                  name="All Tokens (%)"
                  opacity={0.5}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Explanation */}
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
          <h3 className="text-sm font-medium mb-2 flex items-center gap-2">
            <Zap className="h-4 w-4 text-blue-600 dark:text-blue-400" />
            Understanding Sampling Strategies
          </h3>
          <div className="grid md:grid-cols-2 gap-4 text-xs text-blue-800 dark:text-blue-200">
            <div>
              <div className="font-medium mb-1">Top-K Sampling</div>
              <ul className="space-y-1 text-blue-700 dark:text-blue-300">
                <li>• Select K most likely tokens</li>
                <li>• Resample probabilities among K tokens</li>
                <li>• Fixed vocabulary size</li>
                <li>• Good for: Controlling diversity</li>
              </ul>
            </div>
            <div>
              <div className="font-medium mb-1">Top-p (Nucleus) Sampling</div>
              <ul className="space-y-1 text-blue-700 dark:text-blue-300">
                <li>• Select tokens until cumulative prob ≥ p</li>
                <li>• Dynamic vocabulary size</li>
                <li>• Adapts to distribution shape</li>
                <li>• Good for: Natural text generation</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Key Differences */}
        <div className="grid md:grid-cols-3 gap-4">
          <div className="p-4 bg-muted rounded-lg">
            <h4 className="text-sm font-medium mb-2">Greedy Sampling</h4>
            <p className="text-xs text-muted-foreground">
              Always pick the most likely token. Deterministic but can lead to repetitive text.
            </p>
          </div>
          <div className="p-4 bg-muted rounded-lg">
            <h4 className="text-sm font-medium mb-2">Top-K Sampling</h4>
            <p className="text-xs text-muted-foreground">
              Limits to K tokens, then samples. More predictable than Top-p, good for debugging.
            </p>
          </div>
          <div className="p-4 bg-muted rounded-lg">
            <h4 className="text-sm font-medium mb-2">Top-p Sampling</h4>
            <p className="text-xs text-muted-foreground">
              Adapts to probability mass. More natural, most commonly used in practice.
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
