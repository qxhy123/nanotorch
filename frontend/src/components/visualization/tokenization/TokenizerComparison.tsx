/**
 * TokenizerComparison Component
 *
 * Compares different tokenizers side-by-side:
 * - Character-level
 * - Word-level
 * - BPE
 *
 * Shows:
 * - Number of tokens produced
 * - Token sequences
 * - OOV (out-of-vocabulary) counts
 * - Compression ratios
 */

import React, { useCallback, useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../ui/card';
import { Badge } from '../../ui/badge';
import { Button } from '../../ui/button';
import { useTransformerStore } from '../../../stores/transformerStore';
import { tokenizerApi } from '../../../services/tokenizerApi';
import type { TokenType, ComparisonResult } from '../../../types/tokenizer';
import { Loader2, GitCompare, ArrowRight, AlertCircle } from 'lucide-react';

interface TokenizerComparisonProps {
  className?: string;
}

const TOKENIZER_COLORS: Record<TokenType, string> = {
  char: 'bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-200 border-blue-200 dark:border-blue-800',
  word: 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-200 border-green-200 dark:border-green-800',
  bpe: 'bg-purple-100 dark:bg-purple-900/30 text-purple-800 dark:text-purple-200 border-purple-200 dark:border-purple-800',
};

const TOKENIZER_ICONS: Record<TokenType, string> = {
  char: 'C',
  word: 'W',
  bpe: 'B',
};

export const TokenizerComparison: React.FC<TokenizerComparisonProps> = ({ className = '' }) => {
  const { inputText, tokenizerVocabSize, tokenizerNumMerges } = useTransformerStore();

  const [isComparing, setIsComparing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [comparisons, setComparisons] = useState<ComparisonResult[]>([]);
  const [selectedTokenizer, setSelectedTokenizer] = useState<TokenType | null>(null);

  // Perform comparison
  const performComparison = useCallback(async () => {
    if (!inputText.trim()) {
      setComparisons([]);
      return;
    }

    setIsComparing(true);
    setError(null);

    try {
      const response = await tokenizerApi.compare({
        text: inputText,
        vocabSize: tokenizerVocabSize,
        numMerges: tokenizerNumMerges,
      });

      if (response.success) {
        setComparisons(response.comparisons);
      } else {
        setError(response.error || 'Comparison failed');
        setComparisons([]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      setComparisons([]);
    } finally {
      setIsComparing(false);
    }
  }, [inputText, tokenizerVocabSize, tokenizerNumMerges]);

  // Auto-compare when input or settings change
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      performComparison();
    }, 500);

    return () => clearTimeout(timeoutId);
  }, [performComparison]);

  // Calculate compression ratio (characters per token)
  const getCompressionRatio = (result: ComparisonResult) => {
    if (result.numTokens === 0) return 0;
    return (inputText.length / result.numTokens).toFixed(2);
  };

  // Get the best tokenizer for a metric
  const getBestFor = (metric: 'numTokens' | 'oovCount') => {
    if (comparisons.length === 0) return null;

    const compare = (a: ComparisonResult, b: ComparisonResult) => {
      if (metric === 'numTokens') {
        // Fewer tokens is better for efficiency
        return a.numTokens < b.numTokens ? a : b;
      } else {
        // Fewer OOV is better
        return a.oovCount < b.oovCount ? a : b;
      }
    };

    return comparisons.reduce(compare, comparisons[0]);
  };

  const bestForTokens = getBestFor('numTokens');
  const bestForOOV = getBestFor('oovCount');

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <GitCompare className="h-5 w-5" />
              Tokenizer Comparison
            </CardTitle>
            <CardDescription>
              Compare different tokenization methods on the same text
            </CardDescription>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={performComparison}
            disabled={isComparing}
          >
            <Loader2 className={`h-4 w-4 mr-1 ${isComparing ? 'animate-spin' : ''}`} />
            Compare
          </Button>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Error Display */}
        {error && (
          <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
            <p className="text-sm text-red-800 dark:text-red-200">{error}</p>
          </div>
        )}

        {/* Loading State */}
        {isComparing && (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
            <span className="ml-2 text-sm text-gray-600 dark:text-gray-400">Comparing tokenizers...</span>
          </div>
        )}

        {/* Comparison Results */}
        {comparisons.length > 0 && !isComparing && (
          <>
            {/* Summary Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {comparisons.map((result) => (
                <div
                  key={result.tokenizerType}
                  className={`
                    p-4 rounded-lg border-2 transition-all cursor-pointer
                    ${selectedTokenizer === result.tokenizerType
                      ? TOKENIZER_COLORS[result.tokenizerType].replace('bg-', 'border-').replace('dark:bg-', 'dark:border-')
                      : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                    }
                  `}
                  onClick={() => setSelectedTokenizer(
                    selectedTokenizer === result.tokenizerType ? null : result.tokenizerType
                  )}
                >
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <div className={`
                        w-8 h-8 rounded-full flex items-center justify-center font-bold
                        ${TOKENIZER_COLORS[result.tokenizerType]}
                      `}>
                        {TOKENIZER_ICONS[result.tokenizerType]}
                      </div>
                      <span className="font-semibold capitalize">{result.tokenizerType}</span>
                    </div>
                    {bestForTokens?.tokenizerType === result.tokenizerType && (
                      <Badge variant="secondary" className="text-xs">
                        Most Efficient
                      </Badge>
                    )}
                  </div>

                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Tokens:</span>
                      <span className="font-mono font-semibold">{result.numTokens}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Compression:</span>
                      <span className="font-mono">{getCompressionRatio(result)}x</span>
                    </div>
                    {result.oovCount > 0 && (
                      <div className="flex justify-between">
                        <span className="text-gray-600 dark:text-gray-400 flex items-center gap-1">
                          <AlertCircle className="h-3 w-3" />
                          OOV:
                        </span>
                        <span className="font-mono text-red-600 dark:text-red-400">{result.oovCount}</span>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>

            {/* Detailed Token Comparison */}
            {selectedTokenizer && (
              <div className="space-y-4">
                <h3 className="font-medium">Detailed Token Sequence</h3>

                {comparisons
                  .filter((c) => c.tokenizerType === selectedTokenizer)
                  .map((result) => (
                    <div key={result.tokenizerType} className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-600 dark:text-gray-400 capitalize">
                          {result.tokenizerType} Tokenizer
                        </span>
                        <Badge variant="outline">{result.numTokens} tokens</Badge>
                      </div>

                      {/* Token Sequence */}
                      <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                        <div className="flex flex-wrap gap-1">
                          {result.tokens.map((token, index) => (
                            <span
                              key={index}
                              className={`
                                inline-flex items-center px-2 py-1 rounded text-sm font-mono
                                ${token.startsWith('<') && token.endsWith('>')
                                  ? 'bg-purple-100 dark:bg-purple-900/30 text-purple-800 dark:text-purple-200'
                                  : 'bg-white dark:bg-gray-700'
                                }
                                border border-gray-200 dark:border-gray-600
                              `}
                              title={`ID: ${result.tokenIds[index]}`}
                            >
                              {token}
                            </span>
                          ))}
                        </div>
                      </div>

                      {/* Token IDs */}
                      <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                        <div className="text-xs text-gray-500 dark:text-gray-400 mb-2">
                          Token IDs
                        </div>
                        <div className="flex flex-wrap gap-1">
                          {result.tokenIds.map((tokenId, index) => (
                            <span
                              key={index}
                              className="inline-flex items-center px-2 py-1 rounded text-xs font-mono bg-blue-50 dark:bg-blue-900/20 text-blue-800 dark:text-blue-200 border border-blue-200 dark:border-blue-800"
                            >
                              {tokenId}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  ))}
              </div>
            )}

            {/* Side-by-Side Comparison */}
            <div className="space-y-4">
              <h3 className="font-medium">Side-by-Side Comparison</h3>

              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2 px-4">Metric</th>
                      {comparisons.map((result) => (
                        <th
                          key={result.tokenizerType}
                          className="text-center py-2 px-4 capitalize"
                        >
                          <div className="flex items-center justify-center gap-2">
                            <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs ${TOKENIZER_COLORS[result.tokenizerType]}`}>
                              {TOKENIZER_ICONS[result.tokenizerType]}
                            </span>
                            {result.tokenizerType}
                          </div>
                        </th>
                      ))}
                      <th className="text-center py-2 px-4">Best</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b">
                      <td className="py-2 px-4 font-medium">Number of Tokens</td>
                      {comparisons.map((result) => (
                        <td key={result.tokenizerType} className="text-center py-2 px-4">
                          <span className={`font-mono ${bestForTokens?.tokenizerType === result.tokenizerType ? 'font-bold text-green-600 dark:text-green-400' : ''}`}>
                            {result.numTokens}
                          </span>
                        </td>
                      ))}
                      <td className="text-center py-2 px-4">
                        {bestForTokens && (
                          <Badge variant="secondary" className="capitalize">
                            {bestForTokens.tokenizerType}
                          </Badge>
                        )}
                      </td>
                    </tr>
                    <tr className="border-b">
                      <td className="py-2 px-4 font-medium">Avg. Characters per Token</td>
                      {comparisons.map((result) => (
                        <td key={result.tokenizerType} className="text-center py-2 px-4">
                          <span className="font-mono">
                            {getCompressionRatio(result)}
                          </span>
                        </td>
                      ))}
                      <td className="text-center py-2 px-4">
                        <ArrowRight className="h-4 w-4 mx-auto text-gray-400" />
                      </td>
                    </tr>
                    <tr className="border-b">
                      <td className="py-2 px-4 font-medium">OOV Tokens</td>
                      {comparisons.map((result) => (
                        <td key={result.tokenizerType} className="text-center py-2 px-4">
                          {result.oovCount > 0 ? (
                            <span className={`font-mono ${bestForOOV?.tokenizerType === result.tokenizerType ? 'font-bold text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                              {result.oovCount}
                            </span>
                          ) : (
                            <span className="text-green-600 dark:text-green-400">✓</span>
                          )}
                        </td>
                      ))}
                      <td className="text-center py-2 px-4">
                        {bestForOOV?.oovCount === 0 ? (
                          <span className="text-green-600 dark:text-green-400">All</span>
                        ) : bestForOOV && (
                          <Badge variant="secondary" className="capitalize">
                            {bestForOOV.tokenizerType}
                          </Badge>
                        )}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            {/* Explanation */}
            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
              <h4 className="font-medium mb-2">Understanding the Results</h4>
              <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                <li><strong>Character:</strong> Splits each character separately. Results in the longest sequences but handles any text.</li>
                <li><strong>Word:</strong> Splits into words using regex. Shorter sequences but may have unknown words.</li>
                <li><strong>BPE:</strong> Learns common subword patterns. Balances sequence length with vocabulary coverage.</li>
              </ul>
            </div>
          </>
        )}

        {/* Empty State */}
        {comparisons.length === 0 && !isComparing && !error && (
          <div className="text-center py-8 text-gray-500 dark:text-gray-400">
            <GitCompare className="h-12 w-12 mx-auto mb-2 opacity-50" />
            <p>Enter text above to compare tokenization methods</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
};