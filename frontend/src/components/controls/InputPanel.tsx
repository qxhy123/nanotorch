import React, { useEffect, useMemo } from 'react';
import { useTransformerStore } from '../../stores/transformerStore';
import { DisclosureSection } from '../layout/DisclosureSection';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Label } from '../ui/label';
import { Textarea } from '../ui/textarea';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Play } from 'lucide-react';
import { transformerApi } from '../../services/api';
import type { TransformerInput } from '../../types/transformer';
import {
  buildTokenizerTrainingTexts,
  getEffectiveTokenizerVocabSize,
  tokenizeForTransformer,
} from '../../services/transformerTokenizer';

export const InputPanel: React.FC = () => {
  const inputText = useTransformerStore((state) => state.inputText);
  const targetText = useTransformerStore((state) => state.targetText);
  const config = useTransformerStore((state) => state.config);
  const isLoading = useTransformerStore((state) => state.isLoading);
  const error = useTransformerStore((state) => state.error);
  const tokenizerType = useTransformerStore((state) => state.tokenizerType);
  const tokenizerVocabSize = useTransformerStore((state) => state.tokenizerVocabSize);
  const tokenizerNumMerges = useTransformerStore((state) => state.tokenizerNumMerges);
  const tokenizationResult = useTransformerStore((state) => state.tokenizationResult);
  const isTokenizing = useTransformerStore((state) => state.isTokenizing);

  const setInputText = useTransformerStore((state) => state.setInputText);
  const setTargetText = useTransformerStore((state) => state.setTargetText);
  const setTokens = useTransformerStore((state) => state.setTokens);
  const setTargetTokens = useTransformerStore((state) => state.setTargetTokens);
  const setOutput = useTransformerStore((state) => state.setOutput);
  const setLoading = useTransformerStore((state) => state.setLoading);
  const setError = useTransformerStore((state) => state.setError);
  const setIsTokenizing = useTransformerStore((state) => state.setIsTokenizing);
  const setTokenizationResult = useTransformerStore((state) => state.setTokenizationResult);
  const setTokenizationError = useTransformerStore((state) => state.setTokenizationError);

  const trainingTexts = useMemo(
    () => buildTokenizerTrainingTexts(inputText, targetText),
    [inputText, targetText]
  );
  const effectiveTokenizerVocabSize = useMemo(
    () => getEffectiveTokenizerVocabSize(tokenizerVocabSize, config.vocab_size),
    [tokenizerVocabSize, config.vocab_size]
  );
  const previewTokenIds = tokenizationResult?.tokenIds.slice(0, config.max_seq_len) ?? [];
  const previewTokens = tokenizationResult?.tokens.slice(0, config.max_seq_len) ?? [];
  const totalTokenCount = tokenizationResult?.tokenIds.length ?? 0;
  const isInputTruncated = totalTokenCount > config.max_seq_len;

  useEffect(() => {
    if (!inputText.trim()) {
      setTokenizationResult(null);
      setTokenizationError(null);
      return;
    }

    let cancelled = false;
    const timeoutId = window.setTimeout(async () => {
      setIsTokenizing(true);

      try {
        const result = await tokenizeForTransformer({
          text: inputText,
          tokenizerType,
          tokenizerVocabSize,
          tokenizerNumMerges,
          modelVocabSize: config.vocab_size,
          maxSeqLen: config.max_seq_len,
          trainingTexts,
        });

        if (!cancelled) {
          setTokenizationResult(result.tokenization);
        }
      } catch (tokenizeError: unknown) {
        if (!cancelled) {
          const message = tokenizeError instanceof Error
            ? tokenizeError.message
            : 'Failed to tokenize input';
          setTokenizationError(message);
        }
      } finally {
        if (!cancelled) {
          setIsTokenizing(false);
        }
      }
    }, 300);

    return () => {
      cancelled = true;
      window.clearTimeout(timeoutId);
    };
  }, [
    inputText,
    tokenizerType,
    tokenizerVocabSize,
    tokenizerNumMerges,
    config.vocab_size,
    config.max_seq_len,
    trainingTexts,
    setIsTokenizing,
    setTokenizationResult,
    setTokenizationError,
  ]);

  const handleRun = async () => {
    if (!inputText.trim()) {
      setError('Please enter some input text');
      return;
    }

    setLoading(true);
    setError(null);

    await new Promise((resolve) => setTimeout(resolve, 100));

    try {
      const sourceTokenization = await tokenizeForTransformer({
        text: inputText,
        tokenizerType,
        tokenizerVocabSize,
        tokenizerNumMerges,
        modelVocabSize: config.vocab_size,
        maxSeqLen: config.max_seq_len,
        trainingTexts,
      });

      setTokenizationResult(sourceTokenization.tokenization);
      setTokens(sourceTokenization.truncatedTokenIds);

      const inputData: TransformerInput = {
        text: inputText,
        tokens: sourceTokenization.truncatedTokenIds,
      };

      if (config.num_decoder_layers > 0) {
        if (targetText.trim()) {
          const targetTokenization = await tokenizeForTransformer({
            text: targetText,
            tokenizerType,
            tokenizerVocabSize,
            tokenizerNumMerges,
            modelVocabSize: config.vocab_size,
            maxSeqLen: config.max_seq_len,
            trainingTexts,
          });

          inputData.targetText = targetText;
          inputData.targetTokens = targetTokenization.truncatedTokenIds;
          setTargetTokens(targetTokenization.truncatedTokenIds);
        } else {
          setTargetTokens(sourceTokenization.truncatedTokenIds);
        }
      } else {
        setTargetTokens([]);
      }

      const result = await transformerApi.forward(config, inputData, {
        returnAttention: true,
        returnAllLayers: false,
        returnEmbeddings: true,
      });

      if (result.success && result.data) {
        setOutput(result);
      } else {
        setError(result.error || 'Failed to run forward pass');
      }
    } catch (runError: unknown) {
      const message = runError instanceof Error ? runError.message : 'Failed to run forward pass';
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  const handleQuickExample = (text: string) => {
    setInputText(text);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Input</CardTitle>
        <CardDescription>
          Enter text to process through the Transformer using the shared tokenizer pipeline.
          Higher disclosure levels unlock token previews and decoder-side controls.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="input-text">Input Text</Label>
          <Textarea
            id="input-text"
            placeholder="Enter text here..."
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            rows={4}
            className="resize-none"
          />
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <span>Length: {Array.from(inputText).length} characters</span>
            {isInputTruncated && (
              <Badge variant="destructive">
                Will truncate to {config.max_seq_len} tokens
              </Badge>
            )}
          </div>
        </div>

        {(previewTokenIds.length > 0 || isTokenizing) && (
          <DisclosureSection
            level="intermediate"
            title="Runtime token preview"
            description="Token IDs and tokenizer-cap details unlock at the Intermediate level."
          >
            <div className="space-y-2">
              <Label>Runtime Tokens</Label>
              <div className="p-3 bg-muted rounded-md space-y-2">
                <div className="flex flex-wrap items-center gap-2 text-sm">
                  <Badge variant="secondary">Tokenizer: {tokenizerType}</Badge>
                  <Badge variant="secondary">
                    Effective vocab: {effectiveTokenizerVocabSize.toLocaleString()}
                  </Badge>
                  {tokenizerVocabSize > config.vocab_size && (
                    <Badge variant="outline">
                      Capped by model vocab ({config.vocab_size.toLocaleString()})
                    </Badge>
                  )}
                  {isTokenizing && (
                    <Badge variant="outline">Updating preview...</Badge>
                  )}
                </div>
                <div className="text-xs text-muted-foreground">
                  This preview uses the same tokenizer API and vocab cap as the actual forward pass.
                </div>
                <div className="mt-2">
                  <div className="text-xs text-muted-foreground mb-1">
                    Tokenization ({previewTokenIds.length} tokens used
                    {totalTokenCount > 0 ? ` / ${totalTokenCount} total` : ''}):
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {previewTokenIds.slice(0, 24).map((tokenId, idx) => (
                      <div key={`${tokenId}-${idx}`} className="flex max-w-[6rem] flex-col items-center">
                        <div className="min-h-8 min-w-8 max-w-[6rem] truncate rounded border bg-background px-2 py-1 text-xs font-mono">
                          {previewTokens[idx] ?? `<${idx}>`}
                        </div>
                        <div className="text-[10px] text-muted-foreground font-mono">
                          {tokenId}
                        </div>
                      </div>
                    ))}
                    {previewTokenIds.length > 24 && (
                      <div className="flex items-center text-xs text-muted-foreground">
                        +{previewTokenIds.length - 24} more
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </DisclosureSection>
        )}

        <DisclosureSection
          level="detailed"
          title="Optional decoder input"
          description="Separate decoder-side text input unlocks at the Detailed level."
        >
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="target-text">Decoder Input (Optional)</Label>
              <div className="flex gap-2">
                {config.num_decoder_layers > 0 && (
                  <Badge variant="secondary">Decoder enabled</Badge>
                )}
                <Badge variant="outline">Length: {Array.from(targetText).length}</Badge>
              </div>
            </div>
            <Textarea
              id="target-text"
              placeholder={
                config.num_decoder_layers > 0
                  ? 'Optional: Provide target sequence for decoder visualization. Leave empty to reuse the source sequence.'
                  : 'Decoder input only available when decoder layers are configured'
              }
              value={targetText}
              onChange={(e) => setTargetText(e.target.value)}
              rows={2}
              className="resize-none"
              disabled={config.num_decoder_layers === 0}
            />
            {config.num_decoder_layers > 0 && (
              <p className="text-xs text-muted-foreground">
                If left empty, the source runtime tokens are reused as the decoder input.
              </p>
            )}
          </div>
        </DisclosureSection>

        {config.num_decoder_layers > 0 && isLoading && (
          <div className="space-y-2">
            <Label>Decoder Output</Label>
            <div className="p-3 bg-muted rounded-md space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Status:</span>
                <Badge variant="secondary">Processing...</Badge>
              </div>
            </div>
          </div>
        )}

        {config.num_decoder_layers === 0 && isLoading && (
          <div className="space-y-2">
            <Label>Encoder Output</Label>
            <div className="p-3 bg-muted rounded-md space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Status:</span>
                <Badge variant="secondary">Processing...</Badge>
              </div>
            </div>
          </div>
        )}

        <div className="space-y-2">
          <Label>Quick Examples</Label>
          <div className="flex flex-wrap gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => handleQuickExample('Hello world')}
              disabled={isLoading}
            >
              Hello world
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => handleQuickExample('The quick brown fox jumps over the lazy dog.')}
              disabled={isLoading}
            >
              Quick brown fox
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => handleQuickExample('Machine learning is fascinating.')}
              disabled={isLoading}
            >
              ML example
            </Button>
          </div>
        </div>

        <Button
          onClick={handleRun}
          disabled={isLoading || !inputText.trim()}
          loading={isLoading}
          loadingText="Processing..."
          className="w-full"
          size="lg"
        >
          <Play className="mr-2 h-4 w-4" />
          Run Forward Pass
        </Button>

        {error && (
          <div className="p-3 rounded-md bg-destructive/10 text-destructive text-sm">
            {error}
          </div>
        )}
      </CardContent>
    </Card>
  );
};
