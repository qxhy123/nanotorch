/**
 * VocabularyBrowser Component
 *
 * Displays the vocabulary with:
 * - Search and filter functionality
 * - Token table with ID, text, frequency
 * - Frequency distribution visualization
 * - Token type filtering
 */

import React, { useCallback, useEffect, useState, useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../ui/card';
import { Badge } from '../../ui/badge';
import { Button } from '../../ui/button';
import { Label } from '../../ui/label';
import { Input } from '../../ui/input';
import { Select } from '../../ui/select';
import { useTransformerStore } from '../../../stores/transformerStore';
import { tokenizerApi } from '../../../services/tokenizerApi';
import type { TokenData } from '../../../types/tokenizer';
import { Loader2, Search, BookOpen, BarChart3, Download } from 'lucide-react';

interface VocabularyBrowserProps {
  className?: string;
}

const SORT_OPTIONS = [
  { value: 'id', label: 'Token ID' },
  { value: 'frequency', label: 'Frequency' },
  { value: 'text', label: 'Text (A-Z)' },
];

const FILTER_OPTIONS = [
  { value: 'all', label: 'All Tokens' },
  { value: 'regular', label: 'Regular Only' },
  { value: 'special', label: 'Special Only' },
];

export const VocabularyBrowser: React.FC<VocabularyBrowserProps> = ({ className = '' }) => {
  const {
    tokenizerType,
    tokenizerVocabSize,
    vocabularyData,
    setVocabularyData,
  } = useTransformerStore();

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<'id' | 'frequency' | 'text'>('id');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc');
  const [filterType, setFilterType] = useState<'all' | 'regular' | 'special'>('all');
  const [minFrequency] = useState<number>(0);
  const [maxFrequency, setMaxFrequency] = useState<number>(100000);
  const [selectedToken, setSelectedToken] = useState<TokenData | null>(null);

  // Load vocabulary
  const loadVocabulary = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await tokenizerApi.getVocabulary({
        tokenizerType,
        vocabSize: tokenizerVocabSize,
      });

      if (response.success) {
        setVocabularyData({
          size: response.vocabSize,
          tokens: response.tokens.map((t) => ({
            id: t.id,
            text: t.text,
            frequency: t.frequency,
            isSpecial: t.isSpecial,
          })),
          tokenizerType: response.tokenizerType,
          specialTokens: response.specialTokens,
        });

        // Set max frequency from the response
        if (response.tokens.length > 0) {
          const maxFreq = Math.max(...response.tokens.map((t) => t.frequency));
          setMaxFrequency(maxFreq);
        }
      } else {
        setError(response.error || 'Failed to load vocabulary');
        setVocabularyData(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      setVocabularyData(null);
    } finally {
      setIsLoading(false);
    }
  }, [tokenizerType, tokenizerVocabSize, setVocabularyData]);

  // Load vocabulary when tokenizer type or vocab size changes
  useEffect(() => {
    loadVocabulary();
  }, [loadVocabulary]);

  // Filter and sort tokens
  const filteredTokens = useMemo(() => {
    if (!vocabularyData) return [];

    let tokens = [...vocabularyData.tokens];

    // Apply search filter
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      tokens = tokens.filter((t) =>
        t.text.toLowerCase().includes(query) ||
        t.id.toString().includes(query)
      );
    }

    // Apply type filter
    if (filterType === 'regular') {
      tokens = tokens.filter((t) => !t.isSpecial);
    } else if (filterType === 'special') {
      tokens = tokens.filter((t) => t.isSpecial);
    }

    // Apply frequency filter
    tokens = tokens.filter(
      (t) => t.frequency >= minFrequency && t.frequency <= maxFrequency
    );

    // Sort
    tokens.sort((a, b) => {
      let comparison = 0;
      switch (sortBy) {
        case 'id':
          comparison = a.id - b.id;
          break;
        case 'frequency':
          comparison = a.frequency - b.frequency;
          break;
        case 'text':
          comparison = a.text.localeCompare(b.text);
          break;
      }
      return sortOrder === 'asc' ? comparison : -comparison;
    });

    return tokens;
  }, [vocabularyData, searchQuery, sortBy, sortOrder, filterType, minFrequency, maxFrequency]);

  // Calculate frequency distribution data for the chart
  const frequencyDistribution = useMemo(() => {
    if (!vocabularyData) return [];

    // Group tokens by frequency ranges
    const ranges = [
      { label: '1', min: 1, max: 1 },
      { label: '2-5', min: 2, max: 5 },
      { label: '6-10', min: 6, max: 10 },
      { label: '11-50', min: 11, max: 50 },
      { label: '51-100', min: 51, max: 100 },
      { label: '101-500', min: 101, max: 500 },
      { label: '501-1000', min: 501, max: 1000 },
      { label: '1000+', min: 1001, max: Infinity },
    ];

    return ranges.map((range) => ({
      ...range,
      count: vocabularyData.tokens.filter(
        (t) => t.frequency >= range.min && t.frequency <= range.max
      ).length,
    }));
  }, [vocabularyData]);

  // Handle token click
  const handleTokenClick = (token: TokenData) => {
    setSelectedToken(token);
  };

  // Export vocabulary as JSON
  const exportVocabulary = () => {
    if (!vocabularyData) return;

    const data = {
      tokenizerType: vocabularyData.tokenizerType,
      vocabSize: vocabularyData.size,
      tokens: filteredTokens,
      exportedAt: new Date().toISOString(),
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `vocabulary-${vocabularyData.tokenizerType}-${vocabularyData.size}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <BookOpen className="h-5 w-5" />
              Vocabulary Browser
            </CardTitle>
            <CardDescription>
              Explore {vocabularyData?.size || 0} tokens in the vocabulary
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={loadVocabulary} disabled={isLoading}>
              <Loader2 className={`h-4 w-4 mr-1 ${isLoading ? 'animate-spin' : ''}`} />
              Reload
            </Button>
            <Button variant="outline" size="sm" onClick={exportVocabulary} disabled={!vocabularyData}>
              <Download className="h-4 w-4 mr-1" />
              Export
            </Button>
          </div>
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
        {isLoading && (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
            <span className="ml-2 text-sm text-gray-600 dark:text-gray-400">Loading vocabulary...</span>
          </div>
        )}

        {/* Vocabulary Content */}
        {vocabularyData && !isLoading && (
          <>
            {/* Filters */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {/* Search */}
              <div className="space-y-2">
                <Label>Search</Label>
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                  <Input
                    placeholder="Search tokens..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-10"
                  />
                </div>
              </div>

              {/* Sort By */}
              <div className="space-y-2">
                <Label>Sort By</Label>
                <Select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value as 'id' | 'frequency' | 'text')}
                  className="w-full"
                >
                  {SORT_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </Select>
              </div>

              {/* Filter Type */}
              <div className="space-y-2">
                <Label>Filter</Label>
                <Select
                  value={filterType}
                  onChange={(e) => setFilterType(e.target.value as 'all' | 'regular' | 'special')}
                  className="w-full"
                >
                  {FILTER_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </Select>
              </div>

              {/* Sort Order Toggle */}
              <div className="space-y-2">
                <Label>Order</Label>
                <Button
                  variant="outline"
                  className="w-full justify-start"
                  onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
                >
                  {sortOrder === 'asc' ? 'Ascending' : 'Descending'}
                </Button>
              </div>
            </div>

            {/* Results Summary */}
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-600 dark:text-gray-400">
                Showing {filteredTokens.length} of {vocabularyData.size} tokens
              </span>
              <Badge variant="outline" className="capitalize">
                {vocabularyData.tokenizerType}
              </Badge>
            </div>

            {/* Frequency Distribution */}
            <div className="space-y-2">
              <Label className="flex items-center gap-2">
                <BarChart3 className="h-4 w-4" />
                Frequency Distribution
              </Label>
              <div className="flex items-end gap-1 h-32 bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
                {frequencyDistribution.map((range) => {
                  const maxCount = Math.max(...frequencyDistribution.map((r) => r.count));
                  const height = maxCount > 0 ? (range.count / maxCount) * 100 : 0;
                  return (
                    <div key={range.label} className="flex-1 flex flex-col items-center">
                      <div
                        className="w-full bg-blue-500 dark:bg-blue-600 rounded-t hover:bg-blue-600 dark:hover:bg-blue-500 transition-colors cursor-pointer"
                        style={{ height: `${height}%`, minHeight: range.count > 0 ? '4px' : '0' }}
                        title={`${range.label}: ${range.count} tokens`}
                      />
                      <span className="text-xs text-gray-600 dark:text-gray-400 mt-1 truncate w-full text-center">
                        {range.label}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Token Table */}
            <div className="border rounded-lg overflow-hidden">
              <div className="bg-gray-50 dark:bg-gray-800 px-4 py-2 grid grid-cols-12 gap-4 font-medium text-sm">
                <div className="col-span-1">ID</div>
                <div className="col-span-4">Token</div>
                <div className="col-span-3">Frequency</div>
                <div className="col-span-4">Type</div>
              </div>
              <div className="max-h-96 overflow-y-auto">
                {filteredTokens.length > 0 ? (
                  filteredTokens.map((token) => (
                    <button
                      key={token.id}
                      onClick={() => handleTokenClick(token)}
                      className="w-full px-4 py-3 grid grid-cols-12 gap-4 text-sm hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors border-t"
                    >
                      <div className="col-span-1 font-mono text-gray-600 dark:text-gray-400">
                        {token.id}
                      </div>
                      <div className="col-span-4 font-mono truncate">
                        {token.text}
                      </div>
                      <div className="col-span-3">
                        <Badge variant="outline">{token.frequency}</Badge>
                      </div>
                      <div className="col-span-4">
                        {token.isSpecial ? (
                          <Badge className="bg-purple-100 dark:bg-purple-900/30 text-purple-800 dark:text-purple-200">
                            Special
                          </Badge>
                        ) : (
                          <Badge variant="secondary">Regular</Badge>
                        )}
                      </div>
                    </button>
                  ))
                ) : (
                  <div className="px-4 py-8 text-center text-gray-500 dark:text-gray-400">
                    No tokens match your filters
                  </div>
                )}
              </div>
            </div>

            {/* Selected Token Details */}
            {selectedToken && (
              <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="font-medium">Token Details</h4>
                  <Button variant="ghost" size="sm" onClick={() => setSelectedToken(null)}>
                    Close
                  </Button>
                </div>
                <dl className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <dt className="text-gray-600 dark:text-gray-400">Token ID</dt>
                    <dd className="font-mono text-lg">{selectedToken.id}</dd>
                  </div>
                  <div>
                    <dt className="text-gray-600 dark:text-gray-400">Text</dt>
                    <dd className="font-mono text-lg">{selectedToken.text}</dd>
                  </div>
                  <div>
                    <dt className="text-gray-600 dark:text-gray-400">Frequency</dt>
                    <dd className="text-lg">{selectedToken.frequency}</dd>
                  </div>
                  <div>
                    <dt className="text-gray-600 dark:text-gray-400">Type</dt>
                    <dd>
                      {selectedToken.isSpecial ? (
                        <Badge className="bg-purple-100 dark:bg-purple-900/30 text-purple-800 dark:text-purple-200">
                          Special Token
                        </Badge>
                      ) : (
                        <Badge variant="secondary">Regular Token</Badge>
                      )}
                    </dd>
                  </div>
                </dl>
              </div>
            )}
          </>
        )}

        {/* Empty State */}
        {!vocabularyData && !isLoading && !error && (
          <div className="text-center py-8 text-gray-500 dark:text-gray-400">
            <BookOpen className="h-12 w-12 mx-auto mb-2 opacity-50" />
            <p>Select a tokenizer and vocabulary size to load the vocabulary</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
};