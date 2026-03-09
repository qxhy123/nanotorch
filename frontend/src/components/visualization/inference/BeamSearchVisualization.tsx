/**
 * BeamSearchVisualization Component
 *
 * Interactive demonstration of Beam Search decoding:
 * - Step-by-step beam expansion visualization
 * - Multiple beam path tracking
 * - Pruning of low-probability candidates
 * - Interactive beam size control
 * - Animated search process
 * - Final path highlighting
 * - Score comparison across beams
 */

import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../ui/card';
import { Button } from '../../ui/button';
import { Slider } from '../../ui/slider';
import {
  ChevronRight,
  GitBranch,
  Play,
  RotateCcw,
  Settings,
  TrendingUp,
  Zap,
  Award,
  Scan,
} from 'lucide-react';

// Token and path types
interface BeamToken {
  token: string;
  logProb: number;
  cumulativeLogProb: number;
}

interface BeamPath {
  id: string;
  tokens: BeamToken[];
  score: number;
  isActive: boolean;
  isPruned: boolean;
  isFinal: boolean;
}

// Generate mock next token probabilities
function generateNextTokenProbs(context: string[]): Array<{token: string, logProb: number}> {
  const candidates = [
    { token: 'the', logProb: -0.8 },
    { token: 'a', logProb: -1.2 },
    { token: 'an', logProb: -1.5 },
    { token: 'is', logProb: -0.9 },
    { token: 'quick', logProb: -2.1 },
    { token: 'brown', logProb: -2.3 },
    { token: 'fox', logProb: -1.8 },
    { token: 'jumps', logProb: -1.6 },
    { token: 'over', logProb: -1.4 },
    { token: 'dog', logProb: -1.7 },
  ];

  // Shuffle based on context to create variety
  context.reduce((hash, t) => hash + t.length, 0);
  return candidates
    .map(c => ({ ...c, logProb: c.logProb + (Math.random() - 0.5) * 0.5 }))
    .sort((a, b) => a.logProb - b.logProb)
    .slice(0, 8);
}

// Expand beams to next step
function expandBeams(
  currentPaths: BeamPath[],
  beamSize: number,
  maxLength: number
): { paths: BeamPath[]; step: number } {
  if (currentPaths.length === 0 || currentPaths.every(p => p.isFinal)) {
    return { paths: currentPaths, step: 0 };
  }

  const step = currentPaths[0].tokens.length;
  if (step >= maxLength) {
    return {
      paths: currentPaths.map(p => ({ ...p, isActive: false, isFinal: true })),
      step: 0,
    };
  }

  // Expand each active path
  const expandedPaths: BeamPath[] = [];
  for (const path of currentPaths) {
    if (!path.isActive) {
      expandedPaths.push(path);
      continue;
    }

    // Get last token for context
    const context = path.tokens.map(t => t.token);
    const nextTokens = generateNextTokenProbs(context);

    // Create new paths for each next token
    for (const nextToken of nextTokens.slice(0, 3)) {
      const newPath: BeamPath = {
        id: `${path.id}-${nextToken.token}-${expandedPaths.length}`,
        tokens: [
          ...path.tokens,
          {
            ...nextToken,
            cumulativeLogProb: path.score + nextToken.logProb,
          },
        ],
        score: path.score + nextToken.logProb,
        isActive: true,
        isPruned: false,
        isFinal: false,
      };
      expandedPaths.push(newPath);
    }
  }

  // Keep only top beamSize paths
  expandedPaths.sort((a, b) => b.score - a.score);
  const keptPaths = expandedPaths.slice(0, beamSize);
  const prunedPaths = expandedPaths.slice(beamSize).map(p => ({ ...p, isPruned: true, isActive: false }));

  // Mark paths that hit EOS or max length as final
  const finalPaths = [...keptPaths, ...prunedPaths].map(p => {
    const lastToken = p.tokens[p.tokens.length - 1];
    if (lastToken?.token === '</s>' || p.tokens.length >= maxLength) {
      return { ...p, isActive: false, isFinal: true };
    }
    return p;
  });

  return { paths: finalPaths, step: step + 1 };
}

interface BeamSearchVisualizationProps {
  className?: string;
}

export const BeamSearchVisualization: React.FC<BeamSearchVisualizationProps> = ({
  className = '',
}) => {
  const [beamSize, setBeamSize] = useState(3);
  const [maxLength, setMaxLength] = useState(8);
  const [isAnimating, setIsAnimating] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [beamPaths, setBeamPaths] = useState<BeamPath[]>([]);
  const [completedPaths, setCompletedPaths] = useState<BeamPath[]>([]);

  // Initialize search
  const initializeSearch = () => {
    const startPaths: BeamPath[] = [
      {
        id: 'start',
        tokens: [{ token: '<s>', logProb: 0, cumulativeLogProb: 0 }],
        score: 0,
        isActive: true,
        isPruned: false,
        isFinal: false,
      },
    ];
    setBeamPaths(startPaths);
    setCurrentStep(0);
    setCompletedPaths([]);
  };

  // Run beam search animation
  const runAnimation = () => {
    initializeSearch();
    setIsAnimating(true);

    let step = 0;
    const maxSteps = maxLength;

    const animate = () => {
      if (step >= maxSteps) {
        setIsAnimating(false);
        return;
      }

      setBeamPaths(prevPaths => {
        const { paths } = expandBeams(prevPaths, beamSize, maxSteps);

        // Check if all paths are complete
        if (paths.every(p => p.isFinal)) {
          setIsAnimating(false);
          setCompletedPaths(paths);
        }

        return paths;
      });

      setCurrentStep(step + 1);
      step++;

      if (step < maxSteps) {
        setTimeout(animate, 1500);
      } else {
        setIsAnimating(false);
      }
    };

    setTimeout(animate, 500);
  };

  // Reset
  const reset = () => {
    setIsAnimating(false);
    setCurrentStep(0);
    setBeamPaths([]);
    setCompletedPaths([]);
  };

  // Calculate statistics
  const stats = useMemo(() => {
    const activePaths = beamPaths.filter(p => p.isActive);
    const prunedPaths = beamPaths.filter(p => p.isPruned);
    const finalPaths = beamPaths.filter(p => p.isFinal);

    const allPaths = [...beamPaths, ...completedPaths];
    const bestPath = allPaths.length > 0 ? allPaths.reduce((best, p) =>
      p.score > best.score ? p : best) : null;

    return {
      totalPaths: beamPaths.length,
      activeCount: activePaths.length,
      prunedCount: prunedPaths.length,
      finalCount: finalPaths.length,
      bestScore: bestPath?.score ?? -Infinity,
      bestSequence: bestPath?.tokens.map(t => t.token).join(' ') ?? '',
    };
  }, [beamPaths, completedPaths]);

  // Prepare visualization data
  const activeBeams = useMemo(() => {
    return beamPaths
      .filter(p => !p.isPruned)
      .sort((a, b) => b.score - a.score)
      .slice(0, beamSize);
  }, [beamPaths, beamSize]);

  const prunedBeams = useMemo(() => {
    return beamPaths.filter(p => p.isPruned);
  }, [beamPaths]);

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <GitBranch className="h-5 w-5 text-primary" />
              Beam Search Visualization
            </CardTitle>
            <CardDescription>
              Explore how beam search maintains multiple candidate sequences
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
              Run Search
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
            Search Parameters
          </h3>

          {/* Beam Size */}
          <div className="space-y-2">
            <div className="flex items-center gap-4">
              <label className="text-sm font-medium min-w-[100px]">Beam Size:</label>
              <Slider
                value={[beamSize]}
                onValueChange={([v]) => setBeamSize(v)}
                min={1}
                max={5}
                step={1}
                className="flex-1"
              />
              <span className="text-sm font-mono w-12 text-right">{beamSize}</span>
            </div>
            <p className="text-xs text-muted-foreground">
              Number of parallel sequences to maintain
            </p>
          </div>

          {/* Max Length */}
          <div className="space-y-2">
            <div className="flex items-center gap-4">
              <label className="text-sm font-medium min-w-[100px]">Max Length:</label>
              <Slider
                value={[maxLength]}
                onValueChange={([v]) => setMaxLength(v)}
                min={4}
                max={15}
                step={1}
                className="flex-1"
              />
              <span className="text-sm font-mono w-12 text-right">{maxLength}</span>
            </div>
            <p className="text-xs text-muted-foreground">
              Maximum sequence length to generate
            </p>
          </div>
        </div>

        {/* Statistics Cards */}
        <div className="grid md:grid-cols-5 gap-4">
          <div className="p-4 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-lg">
            <div className="text-xs text-blue-600 dark:text-blue-400 mb-1">Step</div>
            <div className="text-2xl font-bold">{currentStep}</div>
            <div className="text-xs text-muted-foreground">/ {maxLength}</div>
          </div>

          <div className="p-4 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-lg">
            <div className="text-xs text-green-600 dark:text-green-400 mb-1">Active</div>
            <div className="text-2xl font-bold">{stats.activeCount}</div>
            <div className="text-xs text-muted-foreground">beams</div>
          </div>

          <div className="p-4 bg-gradient-to-br from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-800/20 rounded-lg">
            <div className="text-xs text-red-600 dark:text-red-400 mb-1">Pruned</div>
            <div className="text-2xl font-bold">{stats.prunedCount}</div>
            <div className="text-xs text-muted-foreground">candidates</div>
          </div>

          <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 rounded-lg">
            <div className="text-xs text-purple-600 dark:text-purple-400 mb-1">Best Score</div>
            <div className="text-2xl font-bold">{stats.bestScore.toFixed(2)}</div>
            <div className="text-xs text-muted-foreground">log prob</div>
          </div>

          <div className="p-4 bg-gradient-to-br from-orange-50 to-orange-100 dark:from-orange-900/20 dark:to-orange-800/20 rounded-lg">
            <div className="text-xs text-orange-600 dark:text-orange-400 mb-1">Completed</div>
            <div className="text-2xl font-bold">{stats.finalCount}</div>
            <div className="text-xs text-muted-foreground">sequences</div>
          </div>
        </div>

        {/* Animation Status */}
        {isAnimating && (
          <div className="p-4 bg-gradient-to-br from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
            <div className="flex items-center gap-3">
              <Scan className="h-5 w-5 text-yellow-600 dark:text-yellow-400 animate-pulse" />
              <div className="flex-1">
                <div className="text-sm font-medium text-yellow-800 dark:text-yellow-200">
                  Running beam search... Step {currentStep} of {maxLength}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Active Beams Visualization */}
        {activeBeams.length > 0 && (
          <div className="space-y-3">
            <h3 className="text-sm font-medium flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-green-500" />
              Active Beams (Top {activeBeams.length})
            </h3>
            <div className="space-y-2">
              {activeBeams.map((path, idx) => (
                <div
                  key={path.id}
                  className={`p-4 rounded-lg border-2 transition-all ${
                    idx === 0
                      ? 'border-green-500 bg-green-50 dark:bg-green-900/20'
                      : 'border-border bg-background'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      {idx === 0 && <Award className="h-4 w-4 text-green-500" />}
                      <span className="text-sm font-medium">Beam {idx + 1}</span>
                    </div>
                    <div className="text-sm font-mono">
                      Score: {path.score.toFixed(2)}
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {path.tokens.map((token, tokenIdx) => (
                      <span
                        key={tokenIdx}
                        className={`px-2 py-1 rounded text-xs font-mono ${
                          tokenIdx === path.tokens.length - 1 && isAnimating
                            ? 'bg-primary text-primary-foreground animate-pulse'
                            : 'bg-muted'
                        }`}
                      >
                        {token.token}
                      </span>
                    ))}
                  </div>
                  <div className="mt-2 text-xs text-muted-foreground">
                    Length: {path.tokens.length} tokens
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Pruned Beams */}
        {prunedBeams.length > 0 && (
          <details className="space-y-3">
            <summary className="flex items-center gap-2 cursor-pointer text-sm font-medium hover:text-primary">
              <ChevronRight className="h-4 w-4" />
              Pruned Candidates ({prunedBeams.length})
            </summary>
            <div className="pl-6 space-y-2">
              {prunedBeams.map((path, idx) => (
                <div
                  key={path.id}
                  className="p-3 rounded-lg border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/20 opacity-60"
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs font-medium">Pruned {idx + 1}</span>
                    <span className="text-xs font-mono">{path.score.toFixed(2)}</span>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {path.tokens.map((token, tokenIdx) => (
                      <span
                        key={tokenIdx}
                        className="px-2 py-1 rounded text-xs font-mono bg-muted"
                      >
                        {token.token}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </details>
        )}

        {/* Best Sequence */}
        {stats.bestSequence && stats.bestSequence !== '<s>' && (
          <div className="p-4 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg border border-green-200 dark:border-green-800">
            <h3 className="text-sm font-medium mb-2 flex items-center gap-2">
              <Zap className="h-4 w-4 text-green-600 dark:text-green-400" />
              Best Sequence Found
            </h3>
            <div className="p-3 bg-background rounded-lg">
              <code className="text-sm">{stats.bestSequence}</code>
            </div>
            <div className="mt-2 text-xs text-muted-foreground">
              Log probability: {stats.bestScore.toFixed(2)} | Length: {stats.bestSequence.split(' ').length} tokens
            </div>
          </div>
        )}

        {/* Explanation */}
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
          <h3 className="text-sm font-medium mb-2 flex items-center gap-2">
            <Zap className="h-4 w-4 text-blue-600 dark:text-blue-400" />
            Understanding Beam Search
          </h3>
          <div className="grid md:grid-cols-2 gap-4 text-xs text-blue-800 dark:text-blue-200">
            <div>
              <div className="font-medium mb-1">How It Works</div>
              <ul className="space-y-1 text-blue-700 dark:text-blue-300">
                <li>• Maintains top-K sequences at each step</li>
                <li>• Expands each sequence with next tokens</li>
                <li>• Keeps only best K sequences (pruning)</li>
                <li>• Returns highest-scoring complete sequence</li>
              </ul>
            </div>
            <div>
              <div className="font-medium mb-1">Key Parameters</div>
              <ul className="space-y-1 text-blue-700 dark:text-blue-300">
                <li>• <strong>Beam Size:</strong> Larger = better quality, slower</li>
                <li>• <strong>Length Penalty:</strong> Prevents short sequences</li>
                <li>• <strong>Early Stopping:</strong> Stop when all beams end</li>
                <li>• Balance: Quality vs computation trade-off</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Comparison with Greedy Search */}
        <div className="grid md:grid-cols-3 gap-4">
          <div className="p-4 bg-muted rounded-lg">
            <h4 className="text-sm font-medium mb-2">Greedy Search</h4>
            <p className="text-xs text-muted-foreground">
              Always picks the most likely token. Fast but can miss better sequences.
            </p>
            <div className="mt-2 text-xs text-muted-foreground">
              Beam Size: 1
            </div>
          </div>
          <div className="p-4 bg-muted rounded-lg border-2 border-primary">
            <h4 className="text-sm font-medium mb-2">Beam Search</h4>
            <p className="text-xs text-muted-foreground">
              Keeps multiple candidates. Better quality, more computation.
            </p>
            <div className="mt-2 text-xs text-muted-foreground">
              Current Beam Size: {beamSize}
            </div>
          </div>
          <div className="p-4 bg-muted rounded-lg">
            <h4 className="text-sm font-medium mb-2">Sampling</h4>
            <p className="text-xs text-muted-foreground">
              Random sampling from distribution. Most diverse but less controlled.
            </p>
            <div className="mt-2 text-xs text-muted-foreground">
              Temperature-based
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
