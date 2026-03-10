/**
 * Text Encoder (CLIP) Visualization
 *
 * Shows how text prompts are tokenized and encoded into embeddings
 */

import { useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Type, Sparkles, AlertCircle, BookOpen } from 'lucide-react';
import { InlineMath } from 'react-katex';
import 'katex/dist/katex.min.css';

// Simple tokenizer simulation
const tokenizeText = (text: string): Array<{ token: string; id: number; isSpecial: boolean }> => {
  const tokens = text.toLowerCase().split(/\s+/).filter(t => t.length > 0);
  const result: Array<{ token: string; id: number; isSpecial: boolean }> = [
    { token: '<BOS>', id: 49406, isSpecial: true },
    ...tokens.map((token, i) => ({ token, id: i + 1000, isSpecial: false })),
    { token: '<EOS>', id: 49407, isSpecial: true },
  ];
  return result.slice(0, 77); // Max 77 tokens
};

// Generate pseudo-embeddings
const generateEmbeddings = (tokens: Array<{ token: string; id: number; isSpecial: boolean }>): number[][] => {
  return tokens.map(t => {
    const base = t.id * 12345.67 % 100;
    return Array.from({ length: 768 }, (_, i) => Math.sin(base + i * 0.1) * (0.5 + Math.random() * 0.5));
  });
};

// Project to 2D
const projectTo2D = (embeddings: number[][]): Array<{ x: number; y: number; token: string; color: string }> => {
  return embeddings.map((emb, i) => ({
    x: emb[0] * 10,
    y: emb[1] * 10,
    token: tokenizeText('test token')[i]?.token || '',
    color: `hsl(${(emb[0] + 1) * 180}, 70%, 50%)`
  }));
};

const PROMPT_TEMPLATES = [
  { text: 'a beautiful landscape with mountains', cn: '美丽的山景' },
  { text: 'a cute cat sitting on a table', cn: '一只可爱的猫坐在桌子上' },
  { text: 'futuristic city at night', cn: '未来夜景城市' },
  { text: 'oil painting of a flower garden', cn: '花卉花园油画' },
];

export const TextEncoderView = () => {
  
  const [prompt, setPrompt] = useState('a beautiful landscape with mountains');
  const [negativePrompt, setNegativePrompt] = useState('blurry, low quality');
  const [selectedToken, setSelectedToken] = useState<number | null>(null);

  const tokens = useMemo(() => tokenizeText(prompt), [prompt]);
  const embeddings = useMemo(() => generateEmbeddings(tokens), [tokens]);
  const projection2D = useMemo(() => projectTo2D(embeddings), [embeddings]);

  const tokenCount = tokens.length;
  const maxTokens = 77;
  const remainingTokens = maxTokens - tokenCount;

  const getAttentionWeight = (index: number): number => {
    // Simulate attention weights
    const importance = [0.1, 0.3, 0.8, 0.7, 0.9, 0.6, 0.4, 0.2];
    return importance[index % importance.length];
  };

  const getTokenSuggestions = (): Array<{ type: string; message: string }> => {
    const suggestions = [];
    if (prompt.length < 10) {
      suggestions.push({ type: 'warning', message: 'Prompt is too short, add more details' });
    }
    if (!prompt.includes('high quality') && !prompt.includes('detailed')) {
      suggestions.push({ type: 'info', message: 'Consider adding quality boosters like "high quality"' });
    }
    if (tokens.length >= 70) {
      suggestions.push({ type: 'warning', message: 'Approaching token limit, may be truncated' });
    }
    return suggestions;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center py-8 bg-gradient-to-br from-primary/10 to-primary/5 rounded-lg">
        <h1 className="text-3xl font-bold mb-2">
          {'Text Encoder (CLIP) Visualization'}
        </h1>
        <p className="text-muted-foreground">
          {'Explore how text prompts are tokenized and encoded into embeddings'
          }
        </p>
      </div>

      {/* Prompt Editor */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Type className="h-5 w-5" />
            {'Prompt Editor'}
          </CardTitle>
          <CardDescription>
            {'Enter prompts to see tokenization and embeddings'}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Main Prompt */}
          <div className="space-y-2">
            <label className="text-sm font-medium">
              {'Main Prompt'}
            </label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="w-full min-h-[100px] p-3 rounded-lg border bg-background resize-none"
              placeholder={'Enter text describing the image...'}
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>{tokenCount} / {maxTokens} tokens</span>
              <span>{remainingTokens} {'remaining'}</span>
            </div>
          </div>

          {/* Negative Prompt */}
          <div className="space-y-2">
            <label className="text-sm font-medium">
              {'Negative Prompt'}
            </label>
            <textarea
              value={negativePrompt}
              onChange={(e) => setNegativePrompt(e.target.value)}
              className="w-full min-h-[60px] p-3 rounded-lg border bg-background resize-none"
              placeholder={'Enter things to avoid...'}
            />
          </div>

          {/* Templates */}
          <div className="space-y-2">
            <label className="text-sm font-medium">
              {'Quick Templates'}
            </label>
            <div className="flex flex-wrap gap-2">
              {PROMPT_TEMPLATES.map((template, i) => (
                <Button
                  key={i}
                  variant="outline"
                  size="sm"
                  onClick={() => setPrompt(template.text)}
                >
                  {template.text}
                </Button>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Token Visualization */}
      <Card>
        <CardHeader>
          <CardTitle>
            {'Token Breakdown'}
          </CardTitle>
          <CardDescription>
            {'CLIP tokenizer converts text to token sequence'}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {tokens.map((token, index) => (
              <button
                key={index}
                className={`
                  px-3 py-2 rounded-lg border-2 transition-all
                  ${token.isSpecial
                    ? 'bg-purple-500/20 border-purple-500'
                    : 'bg-blue-500/20 border-blue-500'
                  }
                  ${selectedToken === index ? 'ring-2 ring-primary scale-105' : ''}
                  hover:shadow-md
                `}
                onClick={() => setSelectedToken(index)}
              >
                <div className="text-xs font-mono text-muted-foreground">ID: {token.id}</div>
                <div className="font-medium">{token.token}</div>
                <div className="text-xs text-muted-foreground">
                  <InlineMath style={{ color: 'inherit' }}>{"\\omega"}</InlineMath>: {getAttentionWeight(index).toFixed(2)}
                </div>
              </button>
            ))}
          </div>

          {/* Token Info */}
          {selectedToken !== null && (
            <div className="mt-4 p-4 bg-muted rounded-lg">
              <h4 className="text-sm font-medium mb-2">
                {'Token Details'}
              </h4>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-muted-foreground">{'Token'}: </span>
                  <span className="font-mono">{tokens[selectedToken]?.token}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">{'ID'}: </span>
                  <span className="font-mono">{tokens[selectedToken]?.id}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">{'Type'}: </span>
                  <span>{tokens[selectedToken]?.isSpecial ? ('Special') : ('Regular')}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">{'Attention'}: </span>
                  <span>{getAttentionWeight(selectedToken).toFixed(3)}</span>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Embedding Visualization */}
      <Card>
        <CardHeader>
          <CardTitle>
            {'Text Embeddings'}
          </CardTitle>
          <CardDescription>
            2D projection of <InlineMath style={{ color: 'inherit' }}>{"768\\text{-dim}"}</InlineMath> embedding vectors
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-64 relative bg-muted rounded-lg overflow-hidden">
            <svg viewBox="-10 -10 20 20" className="w-full h-full">
              {/* Grid */}
              <g opacity="0.1">
                {Array.from({ length: 21 }, (_, i) => (
                  <line key={`v-${i}`} x1={-10 + i} y1="-10" x2={-10 + i} y2="10" stroke="currentColor" />
                ))}
                {Array.from({ length: 21 }, (_, i) => (
                  <line key={`h-${i}`} x1="-10" y1={-10 + i} x2="10" y2={-10 + i} stroke="currentColor" />
                ))}
              </g>

              {/* Points */}
              {projection2D.map((point, i) => (
                <g key={i}>
                  <circle
                    cx={point.x}
                    cy={point.y}
                    r="0.3"
                    fill={tokens[i]?.isSpecial ? '#a855f7' : point.color}
                    className="cursor-pointer hover:r-0.5 transition-all"
                    onClick={() => setSelectedToken(i)}
                  />
                  {selectedToken === i && (
                    <text
                      x={point.x}
                      y={point.y - 0.5}
                      fontSize="0.5"
                      textAnchor="middle"
                      fill="currentColor"
                    >
                      {tokens[i]?.token}
                    </text>
                  )}
                </g>
              ))}
            </svg>
          </div>

          {/* Legend */}
          <div className="flex justify-center gap-6 mt-4 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-purple-500" />
              <span>{'Special Token'}</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-blue-500" />
              <span>{'Regular Token'}</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Suggestions */}
      {getTokenSuggestions().length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Sparkles className="h-5 w-5" />
              {'Optimization Suggestions'}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {getTokenSuggestions().map((suggestion, i) => (
                <div key={i} className={`
                  flex items-start gap-3 p-3 rounded-lg
                  ${suggestion.type === 'warning' ? 'bg-yellow-500/10' : 'bg-blue-500/10'}
                `}>
                  {suggestion.type === 'warning' ? (
                    <AlertCircle className="h-5 w-5 text-yellow-500 flex-shrink-0 mt-0.5" />
                  ) : (
                    <BookOpen className="h-5 w-5 text-blue-500 flex-shrink-0 mt-0.5" />
                  )}
                  <span className="text-sm">{suggestion.message}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Learn More */}
      <Card>
        <CardHeader>
          <CardTitle>
            {'About CLIP Text Encoder'}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-sm">
          <p className="text-muted-foreground">
            {'The CLIP text encoder converts text prompts into 768-dimensional embedding vectors, which are used to guide the image generation process. Similar semantic concepts produce similar embeddings, allowing the model to understand text-image relationships.'
            }
          </p>
          <div className="p-4 bg-muted rounded-lg">
            <h4 className="font-medium mb-2">
              {'Key Parameters'}
            </h4>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>{'Model'}: CLIP ViT-L/14</div>
              <div>{'Embedding Dim'}: <InlineMath style={{ color: 'inherit' }}>{"768"}</InlineMath></div>
              <div>{'Max Tokens'}: <InlineMath style={{ color: 'inherit' }}>{"77"}</InlineMath></div>
              <div>{'Vocab Size'}: <InlineMath style={{ color: 'inherit' }}>{"49408"}</InlineMath></div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
