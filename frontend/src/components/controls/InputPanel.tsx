import React from 'react';
import { useTransformerStore } from '../../stores/transformerStore';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Label } from '../ui/label';
import { Textarea } from '../ui/textarea';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Play } from 'lucide-react';
import { transformerApi } from '../../services/api';
import { Latex } from '../ui/Latex';

export const InputPanel: React.FC = () => {
  // 使用 selector 来确保状态正确更新
  const inputText = useTransformerStore((state) => state.inputText);
  const targetText = useTransformerStore((state) => state.targetText);
  const isLoading = useTransformerStore((state) => state.isLoading);
  const error = useTransformerStore((state) => state.error);
  const config = useTransformerStore((state) => state.config);

  const setInputText = useTransformerStore((state) => state.setInputText);
  const setTargetText = useTransformerStore((state) => state.setTargetText);
  const setTargetTokens = useTransformerStore((state) => state.setTargetTokens);
  const setOutput = useTransformerStore((state) => state.setOutput);
  const setLoading = useTransformerStore((state) => state.setLoading);
  const setError = useTransformerStore((state) => state.setError);

  // 实时 tokenization
  const currentTokens = React.useMemo(() => {
    if (!inputText) return [];
    return Array.from(inputText).map((char) =>
      (char.charCodeAt(0) % config.vocab_size)
    );
  }, [inputText, config.vocab_size]);

  const handleRun = async () => {
    if (!inputText.trim()) {
      setError('Please enter some input text');
      return;
    }

    setLoading(true);
    setError(null);

    // 强制延迟以确保 UI 更新
    await new Promise(resolve => setTimeout(resolve, 100));

    try {
      // Simple tokenization: character to token ID mapping
      const tokenIds = currentTokens;

      // Prepare input data
      const inputData: any = {
        text: inputText,
        tokens: tokenIds,
      };

      // Add target text if decoder layers are configured and target text is provided
      if (config.num_decoder_layers > 0 && targetText.trim()) {
        const targetTokenIds = Array.from(targetText).map((char) =>
          (char.charCodeAt(0) % config.vocab_size)
        );
        inputData.targetText = targetText;
        inputData.targetTokens = targetTokenIds;
        setTargetTokens(targetTokenIds);
      }

      const result = await transformerApi.forward(
        config,
        inputData,
        {
          returnAttention: true,
          returnAllLayers: true,
          returnEmbeddings: true,
        }
      );

      if (result.success && result.data) {
        setOutput(result);
      } else {
        setError(result.error || 'Failed to run forward pass');
      }
    } catch (err: any) {
      setError(err.message || 'Failed to run forward pass');
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
        <CardDescription>Enter text to process through the Transformer</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Input Text */}
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
            <span>Length: {inputText.length} characters</span>
            {inputText.length > config.max_seq_len && (
              <Badge variant="destructive">Exceeds max length</Badge>
            )}
          </div>
        </div>

        {/* Token IDs Visualization */}
        {currentTokens.length > 0 && (
          <div className="space-y-2">
            <Label>Token IDs</Label>
            <div className="p-3 bg-muted rounded-md space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Vocabulary Size:</span>
                <Badge variant="secondary">{config.vocab_size.toLocaleString()}</Badge>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Embedding Layer:</span>
                <Badge variant="secondary">({config.vocab_size.toLocaleString()}, {config.d_model})</Badge>
              </div>
              <div className="text-xs text-muted-foreground">
                Each character is mapped to a token ID using: <Latex>{`\\text{charCodeAt}(0) \\bmod \\text{vocab\\_size}`}</Latex>
              </div>
              <div className="mt-2">
                <div className="text-xs text-muted-foreground mb-1">
                  Tokenization ({currentTokens.length} tokens):
                </div>
                <div className="flex flex-wrap gap-1">
                  {currentTokens.slice(0, 32).map((tokenId, idx) => {
                    const char = inputText[idx];
                    return (
                      <div key={idx} className="flex flex-col items-center">
                        <div className="w-8 h-8 flex items-center justify-center bg-background border rounded text-xs font-mono">
                          {char}
                        </div>
                        <div className="text-[10px] text-muted-foreground font-mono">
                          {tokenId}
                        </div>
                      </div>
                    );
                  })}
                  {currentTokens.length > 32 && (
                    <div className="flex items-center text-xs text-muted-foreground">
                      +{currentTokens.length - 32} more
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Decoder Input (Optional) */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label htmlFor="target-text">Decoder Input (Optional)</Label>
            <div className="flex gap-2">
              {config.num_decoder_layers > 0 && (
                <Badge variant="secondary">Decoder enabled</Badge>
              )}
              <Badge variant="outline">Length: {targetText.length}</Badge>
            </div>
          </div>
          <Textarea
            id="target-text"
            placeholder={
              config.num_decoder_layers > 0
                ? "Optional: Provide target sequence for decoder (for visualization). Leave empty to use source as target."
                : "Decoder input only available when decoder layers are configured"
            }
            value={targetText}
            onChange={(e) => setTargetText(e.target.value)}
            rows={2}
            className="resize-none"
            disabled={config.num_decoder_layers === 0}
          />
          {config.num_decoder_layers > 0 && (
            <p className="text-xs text-muted-foreground">
              Optional: For seq2seq visualization. If empty, the source sequence will be used as decoder input.
            </p>
          )}
        </div>

        {/* Decoder Output */}
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

        {/* Encoder Output (for encoder-only mode) */}
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

        {/* Quick Examples */}
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

        {/* Run Button */}
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

        {/* Error Display */}
        {error && (
          <div className="p-3 rounded-md bg-destructive/10 text-destructive text-sm">
            {error}
          </div>
        )}
      </CardContent>
    </Card>
  );
};
