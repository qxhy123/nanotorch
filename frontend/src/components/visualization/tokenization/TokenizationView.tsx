/**
 * TokenizationView Component
 *
 * Displays the tokenization of input text, showing:
 * - The original text with token highlighting
 * - The list of tokens with their IDs and metadata
 * - Controls for tokenizer type selection
 */

import React, { useCallback, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../ui/card';
import { Badge } from '../../ui/badge';
import { Button } from '../../ui/button';
import { Label } from '../../ui/label';
import { Select } from '../../ui/select';
import { Slider } from '../../ui/slider';
import { useTransformerStore } from '../../../stores/transformerStore';
import { tokenizerApi } from '../../../services/tokenizerApi';
import type { TokenType } from '../../../types/tokenizer';
import { Loader2, Hash, Settings, RefreshCw } from 'lucide-react';

interface TokenizationViewProps {
  className?: string;
}

const TOKENIZER_TYPES: Array<{ value: TokenType; label: string; description: string }> = [
  { value: 'char', label: 'Character', description: 'Split text into individual characters' },
  { value: 'word', label: 'Word', description: 'Split text into words using regex' },
  { value: 'bpe', label: 'BPE', description: 'Byte Pair Encoding - learn subword merges' },
];

export const TokenizationView: React.FC<TokenizationViewProps> = ({ className = '' }) => {
  const {
    inputText,
    tokenizerType,
    tokenizerVocabSize,
    tokenizerNumMerges,
    tokenizationResult,
    isTokenizing,
    tokenizationError,
    setTokenizerType,
    setTokenizerVocabSize,
    setTokenizerNumMerges,
    setTokenizationResult,
    setIsTokenizing,
    setTokenizationError,
  } = useTransformerStore();

  const [showSettings, setShowSettings] = React.useState(false);
  const [selectedTokenIndex, setSelectedTokenIndex] = React.useState<number | null>(null);

  // Perform tokenization
  const performTokenization = useCallback(async () => {
    if (!inputText.trim()) {
      setTokenizationResult(null);
      return;
    }

    setIsTokenizing(true);
    setTokenizationError(null);

    try {
      const response = await tokenizerApi.tokenize({
        text: inputText,
        tokenizerType,
        vocabSize: tokenizerVocabSize,
        numMerges: tokenizerNumMerges,
      });

      if (response.success) {
        setTokenizationResult({
          tokenIds: response.tokenIds,
          tokens: response.tokens,
          tokenDetails: response.tokenDetails,
          vocabularySummary: response.vocabularySummary,
          tokenizerType: response.tokenizerType,
        });
      } else {
        setTokenizationError(response.error || 'Tokenization failed');
        setTokenizationResult(null);
      }
    } catch (error) {
      setTokenizationError(error instanceof Error ? error.message : 'Unknown error');
      setTokenizationResult(null);
    } finally {
      setIsTokenizing(false);
    }
  }, [inputText, tokenizerType, tokenizerVocabSize, tokenizerNumMerges, setIsTokenizing, setTokenizationResult, setTokenizationError]);

  // Auto-tokenize when input or settings change
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      performTokenization();
    }, 500);

    return () => clearTimeout(timeoutId);
  }, [performTokenization]);

  // Handle tokenizer type change
  const handleTokenizerTypeChange = (value: TokenType) => {
    setTokenizerType(value);
  };

  // Get background color for a token based on its properties
  const getTokenColor = (token: any, index: number) => {
    if (token.isSpecial) {
      return 'bg-purple-100 dark:bg-purple-900/30 text-purple-800 dark:text-purple-200';
    }
    if (token.startPosition === -1) {
      return 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400';
    }
    if (selectedTokenIndex === index) {
      return 'bg-blue-200 dark:bg-blue-900/50 text-blue-900 dark:text-blue-100';
    }
    return 'bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700';
  };

  // Render a single token
  const renderToken = (token: any, index: number) => {
    const colorClass = getTokenColor(token, index);
    const displayText = token.text || '<unk>';

    return (
      <button
        key={index}
        onClick={() => setSelectedTokenIndex(selectedTokenIndex === index ? null : index)}
        className={`
          inline-flex items-center gap-1 px-2 py-1 m-0.5 rounded border transition-all
          ${colorClass}
          border-gray-200 dark:border-gray-700
        `}
        title={`ID: ${token.tokenId}, Freq: ${token.frequency}`}
      >
        <span className="text-xs font-mono opacity-60">#{token.tokenId}</span>
        <span className="font-medium">{displayText}</span>
        {token.frequency !== undefined && (
          <span className="text-xs opacity-50">({token.frequency})</span>
        )}
      </button>
    );
  };

  // Render text with highlighted tokens
  const renderHighlightedText = () => {
    if (!tokenizationResult || !tokenizationResult.tokenDetails) {
      return <p className="text-gray-500 dark:text-gray-400">{inputText || 'Enter text to tokenize'}</p>;
    }

    const tokens = tokenizationResult.tokenDetails;
    let lastIndex = 0;
    const elements: React.ReactNode[] = [];

    tokens.forEach((token, index) => {
      // Add text before this token
      if (token.startPosition !== null && token.startPosition !== undefined && token.startPosition > lastIndex) {
        elements.push(
          <span key={`text-${index}`}>
            {inputText.substring(lastIndex, token.startPosition)}
          </span>
        );
      }

      // Add the highlighted token
      const isSelected = selectedTokenIndex === index;
      elements.push(
        <mark
          key={`token-${index}`}
          onClick={() => setSelectedTokenIndex(isSelected ? null : index)}
          className={`
            cursor-pointer px-1 rounded transition-all
            ${token.isSpecial
              ? 'bg-purple-200 dark:bg-purple-900/50 text-purple-900 dark:text-purple-100'
              : 'bg-blue-100 dark:bg-blue-900/30 text-blue-900 dark:text-blue-100'
            }
            ${isSelected ? 'ring-2 ring-blue-500' : ''}
          `}
          title={`ID: ${token.tokenId}, Text: "${token.text}"`}
        >
          {token.startPosition !== null ? inputText.substring(token.startPosition, token.endPosition ?? token.startPosition) : token.text}
        </mark>
      );

      lastIndex = token.endPosition !== null ? token.endPosition : lastIndex;
    });

    // Add remaining text
    if (lastIndex < inputText.length) {
      elements.push(
        <span key="text-end">{inputText.substring(lastIndex)}</span>
      );
    }

    return <p className="text-sm leading-relaxed">{elements}</p>;
  };

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Hash className="h-5 w-5" />
              Tokenization
            </CardTitle>
            <CardDescription>
              See how text is converted into tokens for the transformer
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowSettings(!showSettings)}
            >
              <Settings className="h-4 w-4 mr-1" />
              Settings
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={performTokenization}
              disabled={isTokenizing}
            >
              <RefreshCw className={`h-4 w-4 mr-1 ${isTokenizing ? 'animate-spin' : ''}`} />
              Tokenize
            </Button>
          </div>
        </div>

        {/* Settings Panel */}
        {showSettings && (
          <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg space-y-4">
            {/* Tokenizer Type Selection */}
            <div className="space-y-2">
              <Label>Tokenizer Type</Label>
              <Select
                value={tokenizerType}
                onChange={(e) => handleTokenizerTypeChange(e.target.value as TokenType)}
                className="w-full"
              >
                {TOKENIZER_TYPES.map((type) => (
                  <option key={type.value} value={type.value}>
                    {type.label} - {type.description}
                  </option>
                ))}
              </Select>
            </div>

            {/* Vocabulary Size */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label>Vocabulary Size</Label>
                <Badge variant="outline">{tokenizerVocabSize}</Badge>
              </div>
              <Slider
                value={[tokenizerVocabSize]}
                onValueChange={(value) => setTokenizerVocabSize(value[0])}
                min={100}
                max={100000}
                step={100}
                className="w-full"
              />
            </div>

            {/* Number of Merges (only for BPE) */}
            {tokenizerType === 'bpe' && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label>Number of Merges</Label>
                  <Badge variant="outline">{tokenizerNumMerges}</Badge>
                </div>
                <Slider
                  value={[tokenizerNumMerges]}
                  onValueChange={(value) => setTokenizerNumMerges(value[0])}
                  min={10}
                  max={10000}
                  step={10}
                  className="w-full"
                />
              </div>
            )}
          </div>
        )}
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Error Display */}
        {tokenizationError && (
          <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
            <p className="text-sm text-red-800 dark:text-red-200">{tokenizationError}</p>
          </div>
        )}

        {/* Loading State */}
        {isTokenizing && (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
            <span className="ml-2 text-sm text-gray-600 dark:text-gray-400">Tokenizing...</span>
          </div>
        )}

        {/* Results */}
        {tokenizationResult && !isTokenizing && (
          <>
            {/* Summary Statistics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                  {tokenizationResult.tokenIds.length}
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-400">Total Tokens</div>
              </div>
              <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                  {tokenizationResult.vocabularySummary.vocabSize}
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-400">Vocabulary Size</div>
              </div>
              <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                  {tokenizationResult.tokenDetails.filter((t: any) => t.isSpecial).length}
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-400">Special Tokens</div>
              </div>
              <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
                <div className="text-2xl font-bold text-orange-600 dark:text-orange-400 capitalize">
                  {tokenizationResult.tokenizerType}
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-400">Tokenizer Type</div>
              </div>
            </div>

            {/* Highlighted Text */}
            <div className="space-y-2">
              <Label>Text with Token Highlights</Label>
              <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                {renderHighlightedText()}
              </div>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Click on highlighted tokens to see details
              </p>
            </div>

            {/* Token List */}
            <div className="space-y-2">
              <Label>Token Sequence</Label>
              <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg flex flex-wrap">
                {tokenizationResult.tokenDetails.map((token, index) => renderToken(token, index))}
              </div>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Tokens are wrapped with &lt;sos&gt; (start) and &lt;eos&gt; (end) special tokens
              </p>
            </div>

            {/* Selected Token Details */}
            {selectedTokenIndex !== null && tokenizationResult.tokenDetails[selectedTokenIndex] && (
              <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
                <h4 className="font-medium mb-2">Token Details</h4>
                <dl className="grid grid-cols-2 gap-2 text-sm">
                  <div>
                    <dt className="text-gray-600 dark:text-gray-400">ID</dt>
                    <dd className="font-mono">{tokenizationResult.tokenDetails[selectedTokenIndex].tokenId}</dd>
                  </div>
                  <div>
                    <dt className="text-gray-600 dark:text-gray-400">Text</dt>
                    <dd className="font-mono">{tokenizationResult.tokenDetails[selectedTokenIndex].text}</dd>
                  </div>
                  <div>
                    <dt className="text-gray-600 dark:text-gray-400">Frequency</dt>
                    <dd>{tokenizationResult.tokenDetails[selectedTokenIndex].frequency}</dd>
                  </div>
                  <div>
                    <dt className="text-gray-600 dark:text-gray-400">Type</dt>
                    <dd>{tokenizationResult.tokenDetails[selectedTokenIndex].isSpecial ? 'Special' : 'Regular'}</dd>
                  </div>
                  {tokenizationResult.tokenDetails[selectedTokenIndex].startPosition !== -1 && (
                    <>
                      <div>
                        <dt className="text-gray-600 dark:text-gray-400">Position</dt>
                        <dd>
                          {tokenizationResult.tokenDetails[selectedTokenIndex].startPosition} - {tokenizationResult.tokenDetails[selectedTokenIndex].endPosition}
                        </dd>
                      </div>
                    </>
                  )}
                </dl>
              </div>
            )}
          </>
        )}

        {/* Empty State */}
        {!tokenizationResult && !isTokenizing && !tokenizationError && (
          <div className="text-center py-8 text-gray-500 dark:text-gray-400">
            <Hash className="h-12 w-12 mx-auto mb-2 opacity-50" />
            <p>Enter text above to see how it gets tokenized</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
};