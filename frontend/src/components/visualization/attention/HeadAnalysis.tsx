/**
 * HeadAnalysis Component
 *
 * Analyzes behavior of individual attention heads showing which heads learn
 * positional patterns, syntactic patterns, or semantic patterns.
 * Includes head importance ranking and similarity analysis.
 */

import React, { useState, useCallback, useMemo, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../ui/card';
import { Badge } from '../../ui/badge';
import { Button } from '../../ui/button';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Brain,
  Target,
  Eye,
  GitCompare,
  Info,
  BarChart3,
  RefreshCw,
} from 'lucide-react';

// Types
interface AttentionPattern {
  type: 'positional' | 'syntactic' | 'semantic' | 'random';
  confidence: number;
  description: string;
}

interface HeadStatistics {
  headIndex: number;
  meanAttention: number;
  stdAttention: number;
  entropy: number;
  sparsity: number;
  dominantPosition: number;
  pattern: AttentionPattern;
  importance: number;
}

interface HeadSimilarity {
  head1: number;
  head2: number;
  similarity: number;
}

interface HeadAnalysisProps {
  numHeads?: number;
  className?: string;
}

// Mock data generator
function generateMockHeadAnalysis(numHeads: number = 8): HeadStatistics[] {
  const patterns: AttentionPattern[] = [
    { type: 'positional', confidence: 0.92, description: 'Strong local attention pattern' },
    { type: 'positional', confidence: 0.88, description: 'Attends to previous positions' },
    { type: 'syntactic', confidence: 0.85, description: 'Subject-verb relation tracking' },
    { type: 'semantic', confidence: 0.78, description: 'Coreference resolution' },
    { type: 'semantic', confidence: 0.82, description: 'Semantic similarity' },
    { type: 'syntactic', confidence: 0.75, description: 'Object relation tracking' },
    { type: 'positional', confidence: 0.65, description: 'Distant positional bias' },
    { type: 'random', confidence: 0.45, description: 'No clear pattern (under-trained)' },
  ];

  return Array.from({ length: numHeads }, (_, i) => ({
    headIndex: i,
    meanAttention: 0.1 + Math.random() * 0.05,
    stdAttention: 0.08 + Math.random() * 0.06,
    entropy: 2.5 + Math.random() * 1.5,
    sparsity: 0.4 + Math.random() * 0.4,
    dominantPosition: Math.floor(Math.random() * 10),
    pattern: patterns[i] || patterns[0],
    importance: 1 - (i * 0.08) - Math.random() * 0.1,
  })).sort((a, b) => b.importance - a.importance);
}

function generateMockSimilarityMatrix(numHeads: number): HeadSimilarity[] {
  const similarities: HeadSimilarity[] = [];
  for (let i = 0; i < numHeads; i++) {
    for (let j = i + 1; j < numHeads; j++) {
      similarities.push({
        head1: i,
        head2: j,
        similarity: Math.random() * 0.8,
      });
    }
  }
  return similarities;
}

const PATTERN_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  positional: {
    bg: 'bg-blue-50 dark:bg-blue-900/20',
    text: 'text-blue-800 dark:text-blue-200',
    border: 'border-blue-200 dark:border-blue-800',
  },
  syntactic: {
    bg: 'bg-green-50 dark:bg-green-900/20',
    text: 'text-green-800 dark:text-green-200',
    border: 'border-green-200 dark:border-green-800',
  },
  semantic: {
    bg: 'bg-purple-50 dark:bg-purple-900/20',
    text: 'text-purple-800 dark:text-purple-200',
    border: 'border-purple-200 dark:border-purple-800',
  },
  random: {
    bg: 'bg-gray-50 dark:bg-gray-800',
    text: 'text-gray-600 dark:text-gray-400',
    border: 'border-gray-200 dark:border-gray-700',
  },
};

export const HeadAnalysis: React.FC<HeadAnalysisProps> = ({
  numHeads = 8,
  className = '',
}) => {
  // State
  const [headStats, setHeadStats] = useState<HeadStatistics[]>([]);
  const [similarities, setSimilarities] = useState<HeadSimilarity[]>([]);
  const [selectedHead, setSelectedHead] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [viewMode, setViewMode] = useState<'overview' | 'comparison' | 'details'>('overview');
  const [sortBy, setSortBy] = useState<'importance' | 'entropy' | 'sparsity'>('importance');

  // Load data
  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 500));
      const stats = generateMockHeadAnalysis(numHeads);
      const simMatrix = generateMockSimilarityMatrix(numHeads);
      setHeadStats(stats);
      setSimilarities(simMatrix);
    } catch (error) {
      console.error('Failed to load head analysis:', error);
    } finally {
      setLoading(false);
    }
  }, [numHeads]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Sort head statistics
  const sortedStats = useMemo(() => {
    return [...headStats].sort((a, b) => {
      switch (sortBy) {
        case 'importance':
          return b.importance - a.importance;
        case 'entropy':
          return b.entropy - a.entropy;
        case 'sparsity':
          return b.sparsity - a.sparsity;
        default:
          return 0;
      }
    });
  }, [headStats, sortBy]);

  // Get similarity for a pair of heads
  const getSimilarity = useCallback((head1: number, head2: number): number => {
    return similarities.find(s =>
      (s.head1 === head1 && s.head2 === head2) ||
      (s.head1 === head2 && s.head2 === head1)
    )?.similarity || 0;
  }, [similarities]);

  // Pattern distribution
  const patternDistribution = useMemo(() => {
    const dist: Record<string, number> = { positional: 0, syntactic: 0, semantic: 0, random: 0 };
    headStats.forEach(stat => {
      dist[stat.pattern.type]++;
    });
    return dist;
  }, [headStats]);

  if (loading) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center py-12">
          <RefreshCw className="h-8 w-8 animate-spin text-primary" />
        </CardContent>
      </Card>
    );
  }

  if (headStats.length === 0) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center py-12">
          <div className="text-center text-gray-500">
            <Brain className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p className="text-lg font-medium">No head analysis data available</p>
            <Button onClick={loadData} className="mt-4">
              Load Analysis
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
              <Brain className="h-5 w-5 text-primary" />
              Multi-Head Attention Analysis
            </CardTitle>
            <CardDescription>
              Understanding what each attention head learns
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
        {/* View Mode Selector */}
        <div className="flex gap-2">
          <Button
            variant={viewMode === 'overview' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setViewMode('overview')}
          >
            <BarChart3 className="h-4 w-4 mr-1" />
            Overview
          </Button>
          <Button
            variant={viewMode === 'comparison' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setViewMode('comparison')}
          >
            <GitCompare className="h-4 w-4 mr-1" />
            Comparison
          </Button>
          <Button
            variant={viewMode === 'details' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setViewMode('details')}
          >
            <Eye className="h-4 w-4 mr-1" />
            Details
          </Button>
        </div>

        {/* Pattern Distribution Summary */}
        <div className="grid grid-cols-4 gap-4">
          <div className={`p-3 rounded-lg border ${PATTERN_COLORS.positional.bg} ${PATTERN_COLORS.positional.border}`}>
            <div className="text-xs text-gray-600 dark:text-gray-400">Positional</div>
            <div className="text-2xl font-bold">{patternDistribution.positional}</div>
            <div className="text-xs text-gray-500">heads</div>
          </div>
          <div className={`p-3 rounded-lg border ${PATTERN_COLORS.syntactic.bg} ${PATTERN_COLORS.syntactic.border}`}>
            <div className="text-xs text-gray-600 dark:text-gray-400">Syntactic</div>
            <div className="text-2xl font-bold">{patternDistribution.syntactic}</div>
            <div className="text-xs text-gray-500">heads</div>
          </div>
          <div className={`p-3 rounded-lg border ${PATTERN_COLORS.semantic.bg} ${PATTERN_COLORS.semantic.border}`}>
            <div className="text-xs text-gray-600 dark:text-gray-400">Semantic</div>
            <div className="text-2xl font-bold">{patternDistribution.semantic}</div>
            <div className="text-xs text-gray-500">heads</div>
          </div>
          <div className={`p-3 rounded-lg border ${PATTERN_COLORS.random.bg} ${PATTERN_COLORS.random.border}`}>
            <div className="text-xs text-gray-600 dark:text-gray-400">Random</div>
            <div className="text-2xl font-bold">{patternDistribution.random}</div>
            <div className="text-xs text-gray-500">heads</div>
          </div>
        </div>

        {/* Sort Controls */}
        <div className="flex items-center gap-4">
          <span className="text-sm text-gray-600 dark:text-gray-400">Sort by:</span>
          {(['importance', 'entropy', 'sparsity'] as const).map(mode => (
            <Button
              key={mode}
              variant={sortBy === mode ? 'default' : 'outline'}
              size="sm"
              onClick={() => setSortBy(mode)}
            >
              {mode === 'importance' ? 'Importance' : mode}
            </Button>
          ))}
        </div>

        {viewMode === 'overview' && (
          <>
            {/* Head Ranking Cards */}
            <div className="space-y-3">
              <h3 className="text-sm font-medium">Head Ranking by {sortBy}</h3>
              <div className="grid gap-3">
                <AnimatePresence>
                  {sortedStats.map((stat, index) => {
                    const colors = PATTERN_COLORS[stat.pattern.type];
                    return (
                      <motion.div
                        key={stat.headIndex}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        transition={{ delay: index * 0.05 }}
                        className={`p-4 rounded-lg border cursor-pointer transition-all ${
                          selectedHead === stat.headIndex
                            ? 'ring-2 ring-primary ' + colors.bg
                            : colors.bg + ' ' + colors.border
                        }`}
                        onClick={() => setSelectedHead(selectedHead === stat.headIndex ? null : stat.headIndex)}
                      >
                        <div className="flex items-center gap-4">
                          {/* Rank Badge */}
                          <div className="w-8 h-8 flex items-center justify-center rounded-full bg-primary text-white font-bold text-sm">
                            {index + 1}
                          </div>

                          {/* Head Info */}
                          <div className="flex-1">
                            <div className="flex items-center gap-2 mb-1">
                              <h4 className="font-medium">Head {stat.headIndex}</h4>
                              <Badge className={colors.text + ' text-xs'} variant="outline">
                                {stat.pattern.type}
                              </Badge>
                            </div>
                            <p className="text-sm text-gray-600 dark:text-gray-400">
                              {stat.pattern.description}
                            </p>
                          </div>

                          {/* Statistics */}
                          <div className="grid grid-cols-3 gap-4 text-right">
                            <div>
                              <div className="text-xs text-gray-500">Importance</div>
                              <div className="font-bold">{(stat.importance * 100).toFixed(1)}%</div>
                            </div>
                            <div>
                              <div className="text-xs text-gray-500">Entropy</div>
                              <div className="font-bold">{stat.entropy.toFixed(2)}</div>
                            </div>
                            <div>
                              <div className="text-xs text-gray-500">Sparsity</div>
                              <div className="font-bold">{(stat.sparsity * 100).toFixed(0)}%</div>
                            </div>
                          </div>
                        </div>

                        {/* Expanded Details */}
                        <AnimatePresence>
                          {selectedHead === stat.headIndex && (
                            <motion.div
                              initial={{ height: 0, opacity: 0 }}
                              animate={{ height: 'auto', opacity: 1 }}
                              exit={{ height: 0, opacity: 0 }}
                              className="mt-4 pt-4 border-t"
                            >
                              <div className="grid grid-cols-2 gap-4 text-sm">
                                <div>
                                  <span className="text-gray-600 dark:text-gray-400">Mean Attention:</span>
                                  <span className="ml-2 font-medium">{stat.meanAttention.toFixed(4)}</span>
                                </div>
                                <div>
                                  <span className="text-gray-600 dark:text-gray-400">Std Attention:</span>
                                  <span className="ml-2 font-medium">{stat.stdAttention.toFixed(4)}</span>
                                </div>
                                <div>
                                  <span className="text-gray-600 dark:text-gray-400">Confidence:</span>
                                  <span className="ml-2 font-medium">{(stat.pattern.confidence * 100).toFixed(1)}%</span>
                                </div>
                                <div>
                                  <span className="text-gray-600 dark:text-gray-400">Dominant Position:</span>
                                  <span className="ml-2 font-medium">{stat.dominantPosition}</span>
                                </div>
                              </div>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </motion.div>
                    );
                  })}
                </AnimatePresence>
              </div>
            </div>
          </>
        )}

        {viewMode === 'comparison' && (
          <>
            {/* Similarity Matrix */}
            <div className="space-y-4">
              <h3 className="text-sm font-medium">Head Similarity Matrix</h3>
              <div className="overflow-x-auto">
                <div className="inline-block min-w-full">
                  <div className="grid gap-1" style={{ gridTemplateColumns: `auto repeat(${numHeads}, 1fr)` }}>
                    {/* Header row */}
                    <div></div>
                    {Array.from({ length: numHeads }, (_, i) => (
                      <div key={`header-${i}`} className="text-xs text-center font-medium p-2">
                        H{i}
                      </div>
                    ))}

                    {/* Matrix rows */}
                    {Array.from({ length: numHeads }, (_, i) => (
                      <React.Fragment key={`row-${i}`}>
                        <div className="text-xs font-medium p-2 text-right">H{i}</div>
                        {Array.from({ length: numHeads }, (_, j) => {
                          const isDiagonal = i === j;
                          const similarity = isDiagonal ? 1 : getSimilarity(i, j);
                          return (
                            <div
                              key={`${i}-${j}`}
                              className={`aspect-square flex items-center justify-center text-xs font-mono rounded ${
                                isDiagonal
                                  ? 'bg-gray-800 text-white'
                                  : similarity > 0.7
                                  ? 'bg-purple-500 text-white'
                                  : similarity > 0.4
                                  ? 'bg-purple-200 dark:bg-purple-800'
                                  : 'bg-gray-100 dark:bg-gray-800'
                              }`}
                              title={`H${i} vs H${j}: ${(similarity * 100).toFixed(1)}%`}
                            >
                              {isDiagonal ? '1.0' : similarity.toFixed(2)}
                            </div>
                          );
                        })}
                      </React.Fragment>
                    ))}
                  </div>
                </div>
              </div>

              {/* Similarity scale */}
              <div className="flex items-center gap-2 text-xs">
                <span>Low</span>
                <div className="flex-1 h-2 rounded bg-gradient-to-r from-gray-100 via-purple-200 to-purple-500 dark:from-gray-800 dark:via-purple-800 dark:to-purple-500" />
                <span>High</span>
              </div>
            </div>

            {/* Most Similar Pairs */}
            <div className="space-y-3">
              <h3 className="text-sm font-medium">Most Similar Head Pairs</h3>
              <div className="space-y-2">
                {similarities
                  .sort((a, b) => b.similarity - a.similarity)
                  .slice(0, 5)
                  .map((sim, idx) => (
                    <motion.div
                      key={`${sim.head1}-${sim.head2}`}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: idx * 0.05 }}
                      className="flex items-center gap-4 p-3 bg-gray-50 dark:bg-gray-800 rounded-lg"
                    >
                      <div className="flex items-center gap-2">
                        <Badge variant="outline">H{sim.head1}</Badge>
                        <span className="text-gray-400">↔</span>
                        <Badge variant="outline">H{sim.head2}</Badge>
                      </div>
                      <div className="flex-1">
                        <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${sim.similarity * 100}%` }}
                            transition={{ duration: 0.5 }}
                            className="h-full bg-purple-500"
                          />
                        </div>
                      </div>
                      <div className="text-sm font-medium w-16 text-right">
                        {(sim.similarity * 100).toFixed(0)}%
                      </div>
                    </motion.div>
                  ))}
              </div>
            </div>
          </>
        )}

        {viewMode === 'details' && selectedHead !== null && (
          <>
            {/* Detailed Head Analysis */}
            <div className="space-y-4">
              <h3 className="text-sm font-medium">Head {selectedHead} Detailed Analysis</h3>

              {(() => {
                const head = headStats.find(h => h.headIndex === selectedHead);
                if (!head) return null;
                const colors = PATTERN_COLORS[head.pattern.type];

                return (
                  <div className={`p-4 rounded-lg border ${colors.bg} ${colors.border}`}>
                    <div className="grid grid-cols-2 gap-6">
                      {/* Pattern Info */}
                      <div>
                        <h4 className="font-medium mb-3 flex items-center gap-2">
                          <Target className="h-4 w-4" />
                          Learned Pattern
                        </h4>
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <span className="text-gray-600 dark:text-gray-400">Type:</span>
                            <Badge className={colors.text}>{head.pattern.type}</Badge>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600 dark:text-gray-400">Confidence:</span>
                            <span className="font-medium">{(head.pattern.confidence * 100).toFixed(1)}%</span>
                          </div>
                          <p className="text-sm text-gray-700 dark:text-gray-300 mt-2">
                            {head.pattern.description}
                          </p>
                        </div>
                      </div>

                      {/* Attention Statistics */}
                      <div>
                        <h4 className="font-medium mb-3 flex items-center gap-2">
                          <BarChart3 className="h-4 w-4" />
                          Attention Statistics
                        </h4>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-gray-600 dark:text-gray-400">Mean:</span>
                            <span className="font-mono">{head.meanAttention.toFixed(4)}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600 dark:text-gray-400">Std:</span>
                            <span className="font-mono">{head.stdAttention.toFixed(4)}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600 dark:text-gray-400">Entropy:</span>
                            <span className="font-mono">{head.entropy.toFixed(3)}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600 dark:text-gray-400">Sparsity:</span>
                            <span className="font-mono">{(head.sparsity * 100).toFixed(1)}%</span>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Interpretation */}
                    <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                      <h4 className="font-medium mb-2 flex items-center gap-2">
                        <Info className="h-4 w-4" />
                        Interpretation
                      </h4>
                      <div className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                        <p>
                          {head.pattern.type === 'positional' && (
                            <>This head focuses on positional relationships, attending to tokens at specific relative positions.</>
                          )}
                          {head.pattern.type === 'syntactic' && (
                            <>This head tracks syntactic relationships, such as subject-verb or object relations.</>
                          )}
                          {head.pattern.type === 'semantic' && (
                            <>This head captures semantic relationships, attending to tokens with similar meanings or coreferences.</>
                          )}
                          {head.pattern.type === 'random' && (
                            <>This head shows no clear pattern, possibly indicating it's under-trained or redundant.</>
                          )}
                        </p>
                      </div>
                    </div>
                  </div>
                );
              })()}
            </div>
          </>
        )}

        {/* Info Box */}
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
          <div className="flex items-start gap-3">
            <Info className="h-5 w-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
            <div className="text-sm text-blue-900 dark:text-blue-100">
              <h4 className="font-medium mb-1">Understanding Attention Heads</h4>
              <p className="text-blue-800 dark:text-blue-200">
                Multi-head attention allows the model to attend to different representation subspaces
                at different positions simultaneously. Each head can learn different patterns:
                <strong> Positional</strong> (local/distant attention), <strong>Syntactic</strong> (grammar relations),
                or <strong>Semantic</strong> (meaning/coreference).
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
