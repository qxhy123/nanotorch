/**
 * Generation Playground
 *
 * Interactive playground with parameter tuning and generation simulation
 */

import { useState, useRef, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Slider } from '../components/ui/slider';
import { Settings, Play, RotateCcw, Download, Clock } from 'lucide-react';
import { InlineMath } from 'react-katex';
import 'katex/dist/katex.min.css';

interface GenerationParams {
  prompt: string;
  negative: string;
  steps: number;
  cfg: number;
  seed: number;
  width: number;
  height: number;
}

const PROMPT_TEMPLATES = [
  'a beautiful landscape with mountains',
  'a cute cat sitting on a table',
  'futuristic city at night',
  'oil painting of a flower garden',
];

export const PlaygroundView = () => {
  const [params, setParams] = useState<GenerationParams>({
    prompt: 'a beautiful landscape with mountains',
    negative: 'blurry, low quality',
    steps: 25,
    cfg: 7.5,
    seed: 42,
    width: 512,
    height: 512
  });
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [history, setHistory] = useState<Array<GenerationParams & { id: number }>>([]);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const startGeneration = () => {
    if (isGenerating) return; // Prevent multiple concurrent generations

    setIsGenerating(true);
    setProgress(0);

    intervalRef.current = setInterval(() => {
      setProgress(p => {
        if (p >= 100) {
          if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
          }
          setIsGenerating(false);
          const newId = Date.now() + Math.random(); // Add random for uniqueness
          setHistory(prev => [{ ...params, id: newId }, ...prev].slice(0, 10));
          return 100;
        }
        return p + 5;
      });
    }, 100);
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  const randomizeSeed = () => {
    setParams(prev => ({ ...prev, seed: Math.floor(Math.random() * 1000000) }));
  };

  const setPreset = (preset: 'fast' | 'quality' | 'balanced') => {
    const presets = {
      fast: { steps: 15, cfg: 6 },
      quality: { steps: 50, cfg: 8 },
      balanced: { steps: 25, cfg: 7.5 }
  };
    setParams(prev => ({ ...prev, ...presets[preset] }));
  };

  const generatePlaceholder = () => {
    // Generate a pseudo image based on params
    const colors = params.prompt.split('').map((c) =>
      `hsl(${(c.charCodeAt(0) * 17 + params.seed) % 360}, 70%, 60%)`
    );
    return colors;
  };

  const placeholderColors = generatePlaceholder();

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center py-8 bg-gradient-to-br from-primary/10 to-primary/5 rounded-lg">
        <h1 className="text-3xl font-bold mb-2">
          {'Generation Playground'}
        </h1>
        <p className="text-muted-foreground">
          {'Real-time parameter tuning and generation simulation'
          }
        </p>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Parameters Panel */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Settings className="h-5 w-5" />
              {'Parameters'}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Prompts */}
            <div className="space-y-3">
              <div className="space-y-2">
                <label className="text-sm font-medium">
                  {'Prompt'}
                </label>
                <textarea
                  value={params.prompt}
                  onChange={(e) => setParams(prev => ({ ...prev, prompt: e.target.value }))}
                  className="w-full min-h-[80px] p-3 rounded-lg border bg-background resize-none text-sm"
                  placeholder={'Describe the image you want to generate...'}
                />
                <div className="flex flex-wrap gap-2">
                  {PROMPT_TEMPLATES.map(template => (
                    <button
                      key={template}
                      className="text-xs px-2 py-1 bg-muted rounded hover:bg-muted/80 transition-colors"
                      onClick={() => setParams(prev => ({ ...prev, prompt: template }))}
                    >
                      {template}
                    </button>
                  ))}
                </div>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">
                  {'Negative Prompt'}
                </label>
                <textarea
                  value={params.negative}
                  onChange={(e) => setParams(prev => ({ ...prev, negative: e.target.value }))}
                  className="w-full min-h-[60px] p-3 rounded-lg border bg-background resize-none text-sm"
                  placeholder={'Describe things to avoid...'}
                />
              </div>
            </div>

            {/* Quick Presets */}
            <div className="space-y-2">
              <label className="text-sm font-medium">
                {'Quick Presets'}
              </label>
              <div className="flex gap-2">
                <Button variant="outline" size="sm" onClick={() => setPreset('fast')}>
                  {'Fast'}
                </Button>
                <Button variant="outline" size="sm" onClick={() => setPreset('balanced')}>
                  {'Balanced'}
                </Button>
                <Button variant="outline" size="sm" onClick={() => setPreset('quality')}>
                  {'Quality'}
                </Button>
              </div>
            </div>

            {/* Sliders */}
            <div className="space-y-4">
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">
                    {'Steps'}
                  </span>
                  <span className="font-mono">{params.steps}</span>
                </div>
                <Slider
                  value={params.steps}
                  onValueChange={(v) => setParams(prev => ({ ...prev, steps: v }))}
                  min={10}
                  max={100}
                  step={5}
                  className="w-full"
                />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">
                    <InlineMath style={{ color: 'inherit' }}>{"\\text{CFG} \\text{ Scale}"}</InlineMath>
                  </span>
                  <span className="font-mono">{params.cfg.toFixed(1)}</span>
                </div>
                <Slider
                  value={params.cfg}
                  onValueChange={(v) => setParams(prev => ({ ...prev, cfg: v }))}
                  min={1}
                  max={20}
                  step={0.5}
                  className="w-full"
                />
              </div>
            </div>

            {/* Seed */}
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">
                  {'Seed'}
                </span>
                <div className="flex items-center gap-2">
                  <span className="font-mono">{params.seed}</span>
                  <Button variant="ghost" size="sm" onClick={randomizeSeed}>
                    <RotateCcw className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </div>

            {/* Generate Button */}
            <Button
              className="w-full"
              size="lg"
              onClick={startGeneration}
              disabled={isGenerating}
            >
              {isGenerating ? (
                <>
                  <Clock className="h-4 w-4 mr-2 animate-spin" />
                  {'Generating...'} ({progress}%)
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 mr-2" />
                  {'Generate'}
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Result Panel */}
        <Card>
          <CardHeader>
            <CardTitle>
              {'Generated Result'}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="aspect-square bg-muted rounded-lg overflow-hidden relative">
              {/* Generated Image Placeholder */}
              <svg viewBox="0 0 64 64" className="w-full h-full">
                {Array.from({ length: 64 }).map((_, i) => {
                  const x = i % 8;
                  const y = Math.floor(i / 8);
                  return (
                    <rect
                      key={i}
                      x={x * 8}
                      y={y * 8}
                      width="8"
                      height="8"
                      fill={placeholderColors[i % placeholderColors.length]}
                      opacity={isGenerating ? 0.5 + (progress / 200) : 1}
                    />
                  );
                })}
              </svg>

              {/* Progress Overlay */}
              {isGenerating && (
                <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                  <div className="text-center">
                    <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-2" />
                    <div className="text-white text-sm">{progress}%</div>
                  </div>
                </div>
              )}
            </div>

            {/* Actions */}
            <div className="mt-4 flex gap-2">
              <Button variant="outline" size="sm" className="flex-1">
                <Download className="h-4 w-4 mr-2" />
                {'Download'}
              </Button>
              <Button variant="outline" size="sm" className="flex-1">
                {'Copy Params'}
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* History */}
      {history.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>
              {'Generation History'}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-5 md:grid-cols-10 gap-2">
              {history.map((item) => (
                <button
                  key={item.id}
                  className="aspect-square bg-muted rounded-lg overflow-hidden hover:ring-2 hover:ring-primary transition-all"
                  onClick={() => setParams(item)}
                  title={item.prompt}
                >
                  <svg viewBox="0 0 4 4" className="w-full h-full">
                    {Array.from({ length: 4 }).map((_, i) => {
                      const x = i % 2;
                      const y = Math.floor(i / 2);
                      const color = `hsl(${(item.prompt.charCodeAt(i % item.prompt.length) * 17 + item.seed) % 360}, 70%, 60%)`;
                      return (
                        <rect
                          key={i}
                          x={x * 2}
                          y={y * 2}
                          width="2"
                          height="2"
                          fill={color}
                        />
                      );
                    })}
                  </svg>
                </button>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Learn More */}
      <Card>
        <CardHeader>
          <CardTitle>
            {'About Playground'}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-sm">
          <p className="text-muted-foreground">
            {'This playground simulates the parameter tuning and generation process of Stable Diffusion. In a real application, these parameters would be sent to a diffusion model backend via API for actual image generation.'
            }
          </p>
          <div className="p-4 bg-muted rounded-lg">
            <h4 className="font-medium mb-2">
              {'Parameter Guide'}
            </h4>
            <ul className="space-y-2 text-xs">
              <li>• <strong>{'Steps'}:</strong> {'More steps = higher quality but slower'}</li>
              <li>• <strong><InlineMath style={{ color: 'inherit' }}>{"\\text{CFG} \\text{ Scale}"}</InlineMath>:</strong> {'Control adherence to prompt'}</li>
              <li>• <strong>{'Seed'}:</strong> {'Same seed produces same result'}</li>
            </ul>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
