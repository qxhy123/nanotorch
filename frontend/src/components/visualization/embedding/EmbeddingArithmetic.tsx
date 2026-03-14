/**
 * EmbeddingArithmetic Component
 *
 * Interactive demonstration of semantic arithmetic with word embeddings:
 * - Classic word2vec style analogies (king - man + woman = queen)
 * - Interactive word selection and operations
 * - Vector arithmetic visualization
 * - Cosine similarity calculations
 * - Nearest neighbor finding
 * - 2D/3D embedding space visualization
 * - Pre-loaded example analogies
 */

import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../ui/card';
import { Button } from '../../ui/button';
import { motion } from 'framer-motion';
import {
  Calculator,
  Equal,
  RotateCcw,
  Sparkles,
  Zap,
  BookOpen,
  TrendingUp,
} from 'lucide-react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

// Pre-defined vocabulary with simulated embeddings
const VOCABULARY = [
  // Royalty/Gender
  { word: 'king', category: 'royalty', embedding: [0.8, 0.9, 0.3, 0.7, 0.6] },
  { word: 'queen', category: 'royalty', embedding: [0.85, 0.85, 0.8, 0.75, 0.65] },
  { word: 'prince', category: 'royalty', embedding: [0.6, 0.7, 0.2, 0.5, 0.4] },
  { word: 'princess', category: 'royalty', embedding: [0.65, 0.65, 0.7, 0.55, 0.45] },
  { word: 'man', category: 'gender', embedding: [0.5, 0.6, 0.1, 0.4, 0.3] },
  { word: 'woman', category: 'gender', embedding: [0.55, 0.55, 0.6, 0.45, 0.35] },
  { word: 'boy', category: 'gender', embedding: [0.3, 0.4, 0.05, 0.25, 0.15] },
  { word: 'girl', category: 'gender', embedding: [0.35, 0.35, 0.55, 0.3, 0.2] },

  // Countries/Capitals
  { word: 'france', category: 'country', embedding: [0.2, 0.3, 0.4, 0.8, 0.7] },
  { word: 'paris', category: 'capital', embedding: [0.25, 0.35, 0.45, 0.85, 0.75] },
  { word: 'germany', category: 'country', embedding: [0.15, 0.25, 0.35, 0.7, 0.6] },
  { word: 'berlin', category: 'capital', embedding: [0.2, 0.3, 0.4, 0.75, 0.65] },
  { word: 'italy', category: 'country', embedding: [0.1, 0.2, 0.3, 0.6, 0.5] },
  { word: 'rome', category: 'capital', embedding: [0.15, 0.25, 0.35, 0.65, 0.55] },
  { word: 'spain', category: 'country', embedding: [0.05, 0.15, 0.25, 0.5, 0.4] },
  { word: 'madrid', category: 'capital', embedding: [0.1, 0.2, 0.3, 0.55, 0.45] },

  // Verbs (tense)
  { word: 'walk', category: 'verb', embedding: [-0.3, -0.2, 0.1, 0.2, 0.3] },
  { word: 'walking', category: 'verb', embedding: [-0.25, -0.15, 0.15, 0.25, 0.35] },
  { word: 'walked', category: 'verb', embedding: [-0.35, -0.25, 0.05, 0.15, 0.25] },
  { word: 'run', category: 'verb', embedding: [-0.4, -0.3, 0.0, 0.1, 0.2] },
  { word: 'running', category: 'verb', embedding: [-0.35, -0.25, 0.1, 0.15, 0.25] },
  { word: 'ran', category: 'verb', embedding: [-0.45, -0.35, -0.05, 0.05, 0.15] },

  // Animals
  { word: 'dog', category: 'animal', embedding: [-0.1, 0.0, -0.2, -0.1, -0.3] },
  { word: 'cat', category: 'animal', embedding: [-0.05, -0.05, -0.15, -0.05, -0.25] },
  { word: 'bird', category: 'animal', embedding: [0.0, -0.1, -0.1, 0.0, -0.2] },
  { word: 'fish', category: 'animal', embedding: [-0.15, -0.1, -0.25, -0.15, -0.35] },
];

// Pre-loaded example analogies
const EXAMPLE_ANALOGIES = [
  {
    name: 'Classic Gender',
    expression: 'king - man + woman',
    expected: 'queen',
    description: 'Royalty gender analogy',
  },
  {
    name: 'Country-Capital',
    expression: 'paris - france + germany',
    expected: 'berlin',
    description: 'Geographic analogy',
  },
  {
    name: 'Verb Tense',
    expression: 'walking - walk + run',
    expected: 'running',
    description: 'Present continuous tense',
  },
  {
    name: 'Comparative',
    expression: 'queen - king + princess',
    expected: 'princess',
    description: 'Female royalty analogy',
  },
];

// Vector operations
function addVectors(a: number[], b: number[]): number[] {
  return a.map((val, i) => val + b[i]);
}

function subtractVectors(a: number[], b: number[]): number[] {
  return a.map((val, i) => val - b[i]);
}

// Cosine similarity
function cosineSimilarity(a: number[], b: number[]): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Find nearest neighbors
function findNearestNeighbors(
  target: number[],
  vocabulary: typeof VOCABULARY,
  exclude: string[] = [],
  topK: number = 5
): Array<{ word: string; similarity: number; category: string }> {
  const similarities = vocabulary
    .filter((item) => !exclude.includes(item.word))
    .map((item) => ({
      word: item.word,
      similarity: cosineSimilarity(target, item.embedding),
      category: item.category,
    }))
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, topK);

  return similarities;
}

// Project 5D embedding to 2D using PCA-like approach
function projectTo2D(embedding: number[]): { x: number; y: number } {
  // Simple projection: first two dimensions
  return {
    x: embedding[0] * 100,
    y: embedding[1] * 100,
  };
}

interface EmbeddingArithmeticProps {
  className?: string;
}

export const EmbeddingArithmetic: React.FC<EmbeddingArithmeticProps> = ({
  className = '',
}) => {
  // State
  const [operation, setOperation] = useState<string>('king - man + woman');
  const [resultVector, setResultVector] = useState<number[] | null>(null);
  const [neighbors, setNeighbors] = useState<Array<{ word: string; similarity: number; category: string }> | null>(null);

  // Parse and evaluate operation
  const evaluateOperation = useMemo(() => {
    try {
      const tokens = operation.toLowerCase().split(/\s+/);
      const result: number[] = [0, 0, 0, 0, 0];
      const excludeWords: string[] = [];
      let currentOp = '+';

      for (const token of tokens) {
        const word = VOCABULARY.find((v) => v.word === token.replace(/[+-]/g, ''));
        if (!word) continue;

        // Check for operator prefix
        if (token.startsWith('-')) {
          currentOp = '-';
        } else if (token.startsWith('+')) {
          currentOp = '+';
        }

        if (currentOp === '+') {
          addVectors(result, word.embedding).forEach((v, i) => (result[i] = v));
        } else {
          subtractVectors(result, word.embedding).forEach((v, i) => (result[i] = v));
        }

        excludeWords.push(word.word);
      }

      return {
        vector: result,
        excludeWords,
        valid: true,
      };
    } catch {
      return {
        vector: null,
        excludeWords: [],
        valid: false,
      };
    }
  }, [operation]);

  // Calculate result
  const calculateResult = () => {
    if (!evaluateOperation.valid || !evaluateOperation.vector) return;

    setResultVector(evaluateOperation.vector);
    const nearest = findNearestNeighbors(
      evaluateOperation.vector,
      VOCABULARY,
      evaluateOperation.excludeWords
    );
    setNeighbors(nearest);
  };

  // Load example
  const loadExample = (expression: string) => {
    setOperation(expression);
    setResultVector(null);
    setNeighbors(null);
  };

  // Reset
  const reset = () => {
    setOperation('');
    setResultVector(null);
    setNeighbors(null);
  };

  // Prepare visualization data
  const visualizationData = useMemo(() => {
    const data: Array<{ x: number; y: number; word: string; category: string; isResult: boolean }> = [];

    // Add vocabulary words
    VOCABULARY.forEach((item) => {
      const pos = projectTo2D(item.embedding);
      data.push({
        ...pos,
        word: item.word,
        category: item.category,
        isResult: false,
      });
    });

    // Add result vector if available
    if (resultVector) {
      const pos = projectTo2D(resultVector);
      data.push({
        ...pos,
        word: 'Result',
        category: 'result',
        isResult: true,
      });
    }

    return data;
  }, [resultVector]);

  // Category colors
  const getCategoryColor = (category: string): string => {
    const colors: Record<string, string> = {
      royalty: '#ef4444',
      gender: '#f97316',
      country: '#22c55e',
      capital: '#10b981',
      verb: '#3b82f6',
      animal: '#8b5cf6',
      result: '#ec4899',
    };
    return colors[category] || '#6b7280';
  };

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Calculator className="h-5 w-5 text-primary" />
              Embedding Arithmetic
            </CardTitle>
            <CardDescription>
              Explore semantic relationships through vector arithmetic
            </CardDescription>
          </div>
          <Button variant="outline" size="sm" onClick={reset}>
            <RotateCcw className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Example Analogies */}
        <div className="space-y-3">
          <h3 className="text-sm font-medium flex items-center gap-2">
            <BookOpen className="h-4 w-4" />
            Classic Analogies
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {EXAMPLE_ANALOGIES.map((analogy) => (
              <motion.div
                key={analogy.name}
                className="p-3 bg-muted rounded-lg cursor-pointer hover:bg-accent transition-colors"
                whileHover={{ scale: 1.02 }}
                onClick={() => loadExample(analogy.expression)}
              >
                <div className="text-xs font-medium mb-1">{analogy.name}</div>
                <div className="text-xs text-muted-foreground font-mono">{analogy.expression}</div>
                <div className="text-xs text-primary mt-1">→ {analogy.expected}</div>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Operation Input */}
        <div className="space-y-3 p-4 bg-muted rounded-lg">
          <h3 className="text-sm font-medium flex items-center gap-2">
            <Sparkles className="h-4 w-4" />
            Create Your Own
          </h3>

          <div className="flex flex-wrap items-center gap-2">
            <input
              type="text"
              value={operation}
              onChange={(e) => setOperation(e.target.value)}
              placeholder="e.g., king - man + woman"
              className="flex-1 min-w-[200px] px-3 py-2 rounded-md border border-input bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring"
            />
            <Button onClick={calculateResult} disabled={!evaluateOperation.valid}>
              <Equal className="h-4 w-4 mr-2" />
              Calculate
            </Button>
          </div>

          <div className="text-xs text-muted-foreground">
            Available words: {VOCABULARY.map((v) => v.word).sort().join(', ')}
          </div>

          {/* Operation Breakdown */}
          {operation && (
            <div className="flex items-center gap-2 flex-wrap text-sm">
              {operation.split(/\s+/).map((token, idx) => {
                const word = token.replace(/[+-]/g, '');
                const vocabItem = VOCABULARY.find((v) => v.word === word);
                const operator = token.match(/[+-]/)?.[0];

                return (
                  <React.Fragment key={idx}>
                    {idx > 0 && (
                      <span className="font-mono text-muted-foreground">
                        {operator || '+'}
                      </span>
                    )}
                    {vocabItem ? (
                      <span className="px-2 py-1 bg-primary/10 rounded text-primary font-medium">
                        {vocabItem.word}
                      </span>
                    ) : (
                      <span className="px-2 py-1 bg-destructive/10 rounded text-destructive">
                        {word || token}
                      </span>
                    )}
                  </React.Fragment>
                );
              })}
            </div>
          )}
        </div>

        {/* Result */}
        {resultVector && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-4 p-4 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-lg"
          >
            <div className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-blue-600 dark:text-blue-400" />
              <h3 className="text-sm font-medium">Result</h3>
            </div>

            {/* Nearest Neighbors */}
            {neighbors && neighbors.length > 0 && (
              <div className="space-y-2">
                <div className="text-xs text-blue-700 dark:text-blue-300">Nearest neighbors:</div>
                <div className="space-y-1">
                  {neighbors.map((neighbor, idx) => (
                    <motion.div
                      key={neighbor.word}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: idx * 0.1 }}
                      className="flex items-center gap-3 p-2 bg-background/50 rounded"
                    >
                      <div className="flex-shrink-0 w-6 h-6 flex items-center justify-center bg-primary text-primary-foreground rounded-full text-xs font-bold">
                        {idx + 1}
                      </div>
                      <div className="flex-1 font-medium">{neighbor.word}</div>
                      <div className="flex items-center gap-2">
                        <div className="text-xs text-muted-foreground">{neighbor.category}</div>
                        <div className="text-sm font-mono text-primary">
                          {(neighbor.similarity * 100).toFixed(1)}%
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>
            )}

            {/* Similarity Bar */}
            {neighbors && neighbors.length > 0 && (
              <div className="space-y-1">
                <div className="text-xs text-blue-700 dark:text-blue-300">Similarity distribution:</div>
                <div className="h-2 bg-background/50 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${neighbors[0].similarity * 100}%` }}
                    transition={{ duration: 0.5 }}
                    className="h-full bg-gradient-to-r from-blue-500 to-blue-600"
                  />
                </div>
                <div className="text-xs text-muted-foreground">
                  Best match: {neighbors[0].word} ({(neighbors[0].similarity * 100).toFixed(1)}% similarity)
                </div>
              </div>
            )}
          </motion.div>
        )}

        {/* 2D Visualization */}
        {(
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-medium flex items-center gap-2">
                <Zap className="h-4 w-4" />
                Embedding Space (2D Projection)
              </h3>
            </div>

            <div className="bg-background rounded-lg p-4 border">
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart data={visualizationData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis
                    type="number"
                    dataKey="x"
                    label={{ value: 'Dimension 1', position: 'insideBottom', offset: -5 }}
                    className="text-xs"
                  />
                  <YAxis
                    type="number"
                    dataKey="y"
                    label={{ value: 'Dimension 2', angle: -90, position: 'insideLeft' }}
                    className="text-xs"
                  />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const data = payload[0].payload;
                        return (
                          <div className="bg-background border rounded-lg p-2 shadow-lg">
                            <div className="font-medium">{data.word}</div>
                            <div className="text-xs text-muted-foreground">{data.category}</div>
                            {!data.isResult && (
                              <div className="text-xs text-muted-foreground">
                                [{data.x.toFixed(1)}, {data.y.toFixed(1)}]
                              </div>
                            )}
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Scatter dataKey="y" fill="#3b82f6">
                    {visualizationData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={getCategoryColor(entry.category)} />
                    ))}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
            </div>

            {/* Legend */}
            <div className="flex flex-wrap gap-3 text-xs">
              {Array.from(new Set(VOCABULARY.map((v) => v.category))).map((category) => (
                <div key={category} className="flex items-center gap-1">
                  <div
                    className="w-3 h-3 rounded-sm"
                    style={{ backgroundColor: getCategoryColor(category) }}
                  />
                  <span className="capitalize">{category}</span>
                </div>
              ))}
              {resultVector && (
                <div className="flex items-center gap-1">
                  <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: getCategoryColor('result') }} />
                  <span>Result</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* How It Works */}
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
          <h3 className="text-sm font-medium mb-2 flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-blue-600 dark:text-blue-400" />
            How Embedding Arithmetic Works
          </h3>
          <ul className="space-y-1 text-xs text-blue-800 dark:text-blue-200">
            <li>• Words are represented as vectors in a high-dimensional space</li>
            <li>• Similar words have vectors that point in similar directions</li>
            <li>• Semantic relationships are captured as vector differences</li>
            <li>• king - man ≈ queen - woman (gender relationship)</li>
            <li>• paris - france ≈ berlin - germany (capital relationship)</li>
            <li>• The result vector is closest to the semantic "answer"</li>
          </ul>
        </div>

        {/* Key Concepts */}
        <div className="grid md:grid-cols-2 gap-4">
          <div className="p-4 bg-muted rounded-lg">
            <h4 className="text-sm font-medium mb-2">Vector Addition</h4>
            <p className="text-xs text-muted-foreground">
              Combining meanings: "royal" + "female" points toward "queen"
            </p>
          </div>
          <div className="p-4 bg-muted rounded-lg">
            <h4 className="text-sm font-medium mb-2">Vector Subtraction</h4>
            <p className="text-xs text-muted-foreground">
              Removing attributes: "king" - "male" ≈ "monarch"
            </p>
          </div>
          <div className="p-4 bg-muted rounded-lg">
            <h4 className="text-sm font-medium mb-2">Cosine Similarity</h4>
            <p className="text-xs text-muted-foreground">
              Measures angle between vectors (ignores magnitude)
            </p>
          </div>
          <div className="p-4 bg-muted rounded-lg">
            <h4 className="text-sm font-medium mb-2">Analogies</h4>
            <p className="text-xs text-muted-foreground">
              A is to B as C is to ?: vec(A) - vec(B) + vec(C)
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
