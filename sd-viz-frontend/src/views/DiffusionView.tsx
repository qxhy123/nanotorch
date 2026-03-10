/**
 * Diffusion Process Visualization
 *
 * Shows both forward (adding noise) and reverse (denoising) diffusion processes
 */

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Slider } from '../components/ui/slider';
import { Play, Pause, SkipBack, SkipForward, TrendingUp, TrendingDown } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { InlineMath } from 'react-katex';
import 'katex/dist/katex.min.css';

// Sample data
const diffusionData = {
  forward: {
    timesteps: [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000],
    beta: [0.0001, 0.00015, 0.0002, 0.00025, 0.0003, 0.00035, 0.0004, 0.00045, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.0012, 0.0015, 0.0018, 0.002, 0.003, 0.005, 0.02],
    snr: [70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0, -5, -10, -15, -20, -25, -30]
  },
  reverse: {
    steps: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30],
    denoising: [0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95, 97, 99]
  }
};

type ProcessMode = 'forward' | 'reverse';

export const DiffusionView = () => {
  const [mode, setMode] = useState<ProcessMode>('forward');
  const [timestep, setTimestep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(50);

  // Auto-play animation
  useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      setTimestep(prev => {
        const maxStep = mode === 'forward' ? 1000 : 50;
        if (prev >= maxStep) return 0;
        return prev + (mode === 'forward' ? 10 : 1);
      });
    }, speed);

    return () => clearInterval(interval);
  }, [isPlaying, mode, speed]);

  const currentSNR = mode === 'forward'
    ? (() => {
        const idx = Math.min(Math.floor(timestep / 50), 20);
        return diffusionData.forward.snr[idx];
      })()
    : (() => {
        const idx = Math.min(Math.floor(timestep / 3), 14);
        return -30 + (diffusionData.reverse.denoising[idx] * 0.6);
      })();

  const getNoiseLevel = (t: number) => {
    if (mode === 'forward') {
      return Math.min(t / 1000, 1);
    }
    return Math.max(1 - (t / 50), 0);
  };

  const getProcessDescription = () => {
    if (mode === 'forward') {
      return 'Forward diffusion: Gradually add Gaussian noise to the image until it becomes pure random noise.';
    }
    return 'Reverse diffusion: Start from random noise and gradually denoise to generate a clear image.';
  };

  // Prepare chart data
  const chartData = diffusionData.forward.timesteps.map((t, i) => ({
    timestep: t,
    beta: diffusionData.forward.beta[i] * 1000,
    snr: diffusionData.forward.snr[i],
  }));

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center py-8 bg-gradient-to-br from-primary/10 to-primary/5 rounded-lg">
        <h1 className="text-3xl font-bold mb-2">
          Diffusion Process Visualization
        </h1>
        <p className="text-muted-foreground">
          {getProcessDescription()}
        </p>
      </div>

      {/* Mode Switcher */}
      <div className="flex justify-center gap-4">
        <Button
          variant={mode === 'forward' ? 'default' : 'outline'}
          onClick={() => { setMode('forward'); setTimestep(0); setIsPlaying(false); }}
          className="gap-2"
        >
          <TrendingUp className="h-4 w-4" />
          Forward Process
        </Button>
        <Button
          variant={mode === 'reverse' ? 'default' : 'outline'}
          onClick={() => { setMode('reverse'); setTimestep(0); setIsPlaying(false); }}
          className="gap-2"
        >
          <TrendingDown className="h-4 w-4" />
          Reverse Process
        </Button>
      </div>

      {/* Main Visualization */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Image Display */}
        <Card>
          <CardHeader>
            <CardTitle>
              {mode === 'forward'
                ? 'Adding Noise'
                : 'Denoising Process'
              }
            </CardTitle>
            <CardDescription>
              {mode === 'forward'
                ? `Timestep: ${timestep}/1000`
                : `Step: ${timestep}/50`
              }
            </CardDescription>
          </CardHeader>
          <CardContent>
            {/* Simulated Image Display */}
            <div className="aspect-square relative rounded-lg overflow-hidden bg-muted">
              {/* Grid representing image pixels */}
              <div
                className="absolute inset-0 grid grid-cols-8 grid-rows-8 gap-0.5 p-2"
                style={{
                  filter: `grayscale(${getNoiseLevel(timestep) * 100}%)`,
                }}
              >
                {Array.from({ length: 64 }).map((_, i) => (
                  <div
                    key={i}
                    className="rounded-sm"
                    style={{
                      backgroundColor: `hsl(${(i * 5 + timestep * 0.1) % 360}, 70%, ${50 + getNoiseLevel(timestep) * 30}%)`,
                      opacity: 0.8 + Math.random() * getNoiseLevel(timestep) * 0.2,
                    }}
                  />
                ))}
              </div>

              {/* Timestep Label */}
              <div className="absolute bottom-4 left-4 bg-black/70 text-white px-3 py-1 rounded-md text-sm">
                {mode === 'forward' ? 't = ' : 'Step = '}{timestep}
              </div>

              {/* SNR Label */}
              <div className="absolute top-4 right-4 bg-black/70 text-white px-3 py-1 rounded-md text-sm">
                SNR: {currentSNR.toFixed(1)} dB
              </div>
            </div>

            {/* Progress Bar */}
            <div className="mt-4">
              <div className="flex justify-between text-sm text-muted-foreground mb-2">
                <span>{mode === 'forward' ? 'Clean' : 'Noise'}</span>
                <span>{mode === 'forward' ? 'Noise' : 'Clean'}</span>
              </div>
              <div className="h-2 bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary transition-all duration-100"
                  style={{ width: `${getNoiseLevel(timestep) * 100}%` }}
                />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Controls */}
        <Card>
          <CardHeader>
            <CardTitle>
              Controls
            </CardTitle>
            <CardDescription>
              Control timestep and playback
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Play/Pause */}
            <div className="flex justify-center gap-4">
              <Button
                variant="outline"
                size="icon"
                onClick={() => setTimestep(0)}
                disabled={timestep === 0}
              >
                <SkipBack className="h-4 w-4" />
              </Button>
              <Button
                size="icon"
                onClick={() => setIsPlaying(!isPlaying)}
                className="w-16"
              >
                {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              </Button>
              <Button
                variant="outline"
                size="icon"
                onClick={() => setTimestep(mode === 'forward' ? 1000 : 50)}
                disabled={timestep >= (mode === 'forward' ? 1000 : 50)}
              >
                <SkipForward className="h-4 w-4" />
              </Button>
            </div>

            {/* Timestep Slider */}
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">
                  {mode === 'forward' ? 'Timestep' : 'Step'}
                </span>
                <span className="font-mono">{timestep}</span>
              </div>
              <Slider
                value={timestep}
                onValueChange={(v) => setTimestep(v)}
                min={0}
                max={mode === 'forward' ? 1000 : 50}
                step={mode === 'forward' ? 10 : 1}
                className="w-full"
              />
            </div>

            {/* Speed Control */}
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">
                  Speed
                </span>
                <span className="font-mono">{speed}ms</span>
              </div>
              <Slider
                value={speed}
                onValueChange={(v) => setSpeed(v)}
                min={10}
                max={200}
                step={10}
                className="w-full"
              />
            </div>

            {/* Quick Presets */}
            <div className="space-y-2">
              <span className="text-sm text-muted-foreground">
                Quick Jump
              </span>
              <div className="flex flex-wrap gap-2">
                {mode === 'forward' ? [0, 250, 500, 750, 1000].map(t => (
                  <Button
                    key={t}
                    variant={timestep === t ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setTimestep(t)}
                  >
                    {t}
                  </Button>
                )) : [0, 10, 20, 30, 40, 50].map(t => (
                  <Button
                    key={t}
                    variant={timestep === t ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setTimestep(t)}
                  >
                    {t}
                  </Button>
                ))}
              </div>
            </div>

            {/* Statistics */}
            <div className="p-4 bg-muted rounded-lg space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">
                  Noise Level
                </span>
                <span className="font-mono">{(getNoiseLevel(timestep) * 100).toFixed(0)}%</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">
                  SNR
                </span>
                <span className="font-mono">{currentSNR.toFixed(1)} dB</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Noise Schedule Chart */}
      <Card>
        <CardHeader>
          <CardTitle>
            Noise Schedule
          </CardTitle>
          <CardDescription>
            <InlineMath style={{ color: 'inherit' }}>{"\\beta"}</InlineMath> noise schedule and Signal-to-Noise Ratio over timesteps
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis
                  dataKey="timestep"
                  tick={{ fill: 'hsl(var(--muted-foreground))' }}
                />
                <YAxis
                  yAxisId="beta"
                  tick={{ fill: 'hsl(var(--muted-foreground))' }}
                />
                <YAxis
                  yAxisId="snr"
                  orientation="right"
                  tick={{ fill: 'hsl(var(--muted-foreground))' }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'hsl(var(--card))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '8px',
                  }}
                />
                <Area
                  yAxisId="beta"
                  type="monotone"
                  dataKey="beta"
                  stroke="hsl(var(--primary))"
                  fill="hsl(var(--primary))"
                  fillOpacity={0.3}
                  name="Beta"
                />
                <Area
                  yAxisId="snr"
                  type="monotone"
                  dataKey="snr"
                  stroke="hsl(var(--accent))"
                  fill="hsl(var(--accent))"
                  fillOpacity={0.3}
                  name="SNR"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Learn More */}
      <Card>
        <CardHeader>
          <CardTitle>
            How It Works
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-sm">
          <p className="text-muted-foreground">
            Diffusion models generate images by learning to gradually denoise. The forward process adds noise to images until they become pure random noise; the reverse process starts from random noise and uses a trained neural network to predict and remove noise step by step, ultimately generating clear images.
          </p>
          <div className="p-4 bg-muted rounded-lg">
            <h4 className="font-medium mb-2">
              Key Equations
            </h4>
            <div className="space-y-2 font-mono text-xs">
              <div>
                Forward: <InlineMath style={{ color: 'inherit' }}>{"q(x_t|x_{t-1}) = \\mathcal{N}(x_t; \\sqrt{1-\\beta_t}x_{t-1}, \\beta_t I)"}</InlineMath>
              </div>
              <div>
                Reverse: <InlineMath style={{ color: 'inherit' }}>{"p_\\theta(x_{t-1}|x_t) = \\mathcal{N}(x_{t-1}; \\mu_\\theta(x_t,t), \\Sigma_\\theta(x_t,t))"}</InlineMath>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
