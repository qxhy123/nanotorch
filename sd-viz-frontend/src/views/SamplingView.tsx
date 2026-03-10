/**
 * Sampling Methods Comparison
 *
 * Compare different sampling algorithms like DDPM, DDIM, Euler, etc.
 */

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/card';
import { Slider } from '../components/ui/slider';
import { Clock, Award, CheckCircle } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';
import { InlineMath } from 'react-katex';
import 'katex/dist/katex.min.css';

interface SamplingMethod {
  id: string;
  name: string;  description: string;  minSteps: number;
  recommendedSteps: number;
  speed: 'fast' | 'medium' | 'slow';
  quality: number;
  deterministic: boolean;
}

const samplingMethods: SamplingMethod[] = [
  {
    id: 'ddim',
    name: 'DDIM',
    description: 'Deterministic sampling with faster convergence',
    minSteps: 10,
    recommendedSteps: 20,
    speed: 'fast',
    quality: 8,
    deterministic: true
  },
  {
    id: 'euler',
    name: 'Euler',
    description: 'Simple and efficient first-order sampler',
    minSteps: 10,
    recommendedSteps: 30,
    speed: 'fast',
    quality: 7,
    deterministic: false
  },
  {
    id: 'euler-a',
    name: 'Euler a',
    description: 'Euler with ancestor sampling (ancestral)',
    minSteps: 10,
    recommendedSteps: 25,
    speed: 'fast',
    quality: 8,
    deterministic: false
  },
  {
    id: 'dpm-solver',
    name: 'DPM-Solver++',
    description: 'Advanced ODE solver with excellent quality',
    minSteps: 10,
    recommendedSteps: 20,
    speed: 'fast',
    quality: 9,
    deterministic: true
  },
  {
    id: 'heun',
    name: 'Heun',
    description: 'Second-order solver with better accuracy',
    minSteps: 10,
    recommendedSteps: 25,
    speed: 'medium',
    quality: 8,
    deterministic: false
  },
  {
    id: 'lms',
    name: 'LMS',
    description: 'Linear multistep method',
    minSteps: 10,
    recommendedSteps: 40,
    speed: 'medium',
    quality: 7,
    deterministic: false
  },
];

const qualityData = [
  { method: 'DDIM', steps: 10, fid: 25, clip: 0.28 },
  { method: 'DDIM', steps: 20, fid: 18, clip: 0.32 },
  { method: 'DDIM', steps: 50, fid: 12, clip: 0.35 },
  { method: 'Euler', steps: 10, fid: 28, clip: 0.26 },
  { method: 'Euler', steps: 20, fid: 20, clip: 0.30 },
  { method: 'Euler', steps: 50, fid: 14, clip: 0.34 },
  { method: 'DPM++', steps: 10, fid: 22, clip: 0.30 },
  { method: 'DPM++', steps: 20, fid: 14, clip: 0.36 },
  { method: 'DPM++', steps: 50, fid: 10, clip: 0.38 },
];

const speedData = [
  { steps: 10, DDIM: 1.2, Euler: 1.0, 'DPM++': 1.1, Heun: 1.5 },
  { steps: 20, DDIM: 2.0, Euler: 1.8, 'DPM++': 1.9, Heun: 2.8 },
  { steps: 30, DDIM: 2.8, Euler: 2.5, 'DPM++': 2.7, Heun: 4.0 },
  { steps: 50, DDIM: 4.5, Euler: 4.0, 'DPM++': 4.2, Heun: 6.5 },
];

export const SamplingView = () => {
  
  const [selectedMethods, setSelectedMethods] = useState<string[]>(['ddim', 'dpm-solver']);
  const [stepCount, setStepCount] = useState(20);
  const [cfgScale, setCfgScale] = useState(7.5);

  const toggleMethod = (id: string) => {
    setSelectedMethods(prev =>
      prev.includes(id)
        ? prev.filter(m => m !== id)
        : prev.length < 4
        ? [...prev, id]
        : prev
    );
  };

  const getSpeedColor = (speed: string) => {
    switch (speed) {
      case 'fast': return 'text-green-500';
      case 'medium': return 'text-yellow-500';
      case 'slow': return 'text-red-500';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center py-8 bg-gradient-to-br from-primary/10 to-primary/5 rounded-lg">
        <h1 className="text-3xl font-bold mb-2">
          {'Sampling Methods Comparison'}
        </h1>
        <p className="text-muted-foreground">
          {'Compare quality and speed of different sampling algorithms'
          }
        </p>
      </div>

      {/* Method Selector */}
      <Card>
        <CardHeader>
          <CardTitle>
            {'Select Sampling Methods'}
          </CardTitle>
          <CardDescription>
            {'Select up to 4 methods to compare'}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {samplingMethods.map(method => {
              const isSelected = selectedMethods.includes(method.id);
              const canSelect = selectedMethods.length < 4 || isSelected;
              
              return (
                <button
                  key={method.id}
                  disabled={!canSelect}
                  className={`
                    p-4 rounded-lg border-2 text-left transition-all relative
                    ${isSelected
                      ? 'border-primary bg-primary/10 shadow-lg ring-2 ring-primary/20'
                      : canSelect
                        ? 'border-border bg-background hover:bg-accent hover:border-primary/50 hover:shadow-md cursor-pointer'
                        : 'border-border bg-muted/50 opacity-50 cursor-not-allowed'
                    }
                  `}
                  onClick={() => toggleMethod(method.id)}
                >
                  {isSelected && (
                    <div className="absolute top-2 right-2">
                      <CheckCircle className="h-5 w-5 text-primary" />
                    </div>
                  )}
                  <div className="flex items-start justify-between mb-2">
                    <h3 className="font-medium">
                      {method.name}
                    </h3>
                    <div className={`text-sm ${getSpeedColor(method.speed)}`}>
                      {method.speed}
                    </div>
                  </div>
                  <p className="text-sm text-muted-foreground mb-3">
                    {method.description}
                  </p>
                  <div className="flex gap-2 text-xs">
                    <span className="px-2 py-1 bg-muted rounded">
                      {'Quality'}: {method.quality}/10
                    </span>
                    <span className="px-2 py-1 bg-muted rounded">
                      {method.deterministic ? ('Deterministic') : ('Stochastic')}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
          <div className="mt-4 text-sm text-muted-foreground text-center">
            {selectedMethods.length} / 4 methods selected
          </div>
        </CardContent>
      </Card>

      {/* Parameters */}
      <Card>
        <CardHeader>
          <CardTitle>
            {'Parameters'}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">
                {'Sampling Steps'}
              </span>
              <span className="font-mono">{stepCount}</span>
            </div>
            <Slider
              value={stepCount}
              onValueChange={(v) => setStepCount(v)}
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
              <span className="font-mono">{cfgScale.toFixed(1)}</span>
            </div>
            <Slider
              value={cfgScale}
              onValueChange={(v) => setCfgScale(v)}
              min={1}
              max={20}
              step={0.5}
              className="w-full"
            />
          </div>
        </CardContent>
      </Card>

      {/* Quality Comparison */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Award className="h-5 w-5" />
            {'Quality Comparison'}
          </CardTitle>
          <CardDescription>
            <InlineMath style={{ color: 'inherit' }}>{"\\text{FID}"}</InlineMath> score (lower is better) and <InlineMath style={{ color: 'inherit' }}>{"\\text{CLIP}"}</InlineMath> score (higher is better)
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={qualityData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis
                  dataKey="steps"
                  tick={{ fill: 'hsl(var(--muted-foreground))' }}
                  label={{ value: 'Steps', position: 'insideBottom', offset: -5 }}
                />
                <YAxis
                  yAxisId="fid"
                  orientation="left"
                  tick={{ fill: 'hsl(var(--muted-foreground))' }}
                  label={{ value: 'FID', angle: -90, position: 'insideLeft' }}
                />
                <YAxis
                  yAxisId="clip"
                  orientation="right"
                  tick={{ fill: 'hsl(var(--muted-foreground))' }}
                  label={{ value: 'CLIP', angle: 90, position: 'insideRight' }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'hsl(var(--card))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '8px'
  }}
                />
                <Bar yAxisId="fid" dataKey="fid" fill="hsl(var(--primary))" name="FID" />
                <Bar yAxisId="clip" dataKey="clip" fill="hsl(var(--accent))" name="CLIP" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Speed Comparison */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Clock className="h-5 w-5" />
            {'Speed Comparison'}
          </CardTitle>
          <CardDescription>
            {'Generation time in seconds (lower is better)'}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={speedData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis
                  dataKey="steps"
                  tick={{ fill: 'hsl(var(--muted-foreground))' }}
                  label={{ value: 'Steps', position: 'insideBottom', offset: -5 }}
                />
                <YAxis
                  tick={{ fill: 'hsl(var(--muted-foreground))' }}
                  label={{ value: 'Time (s)', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'hsl(var(--card))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '8px'
  }}
                />
                <Line type="monotone" dataKey="DDIM" stroke="hsl(var(--primary))" strokeWidth={2} />
                <Line type="monotone" dataKey="Euler" stroke="hsl(var(--accent))" strokeWidth={2} />
                <Line type="monotone" dataKey="DPM++" stroke="hsl(var(--destructive))" strokeWidth={2} />
                <Line type="monotone" dataKey="Heun" stroke="hsl(280, 70%, 50%)" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Method Details */}
      {selectedMethods.length > 0 && (
        <div className="grid md:grid-cols-2 gap-4">
          {samplingMethods
            .filter(m => selectedMethods.includes(m.id))
            .map(method => (
              <Card key={method.id}>
                <CardHeader>
                  <CardTitle>
                    {method.name}
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2 text-sm">
                  <p className="text-muted-foreground">
                    {method.description}
                  </p>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <span className="text-muted-foreground">{'Recommended'}: </span>
                      <span>{method.recommendedSteps}</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">{'Quality'}: </span>
                      <span>{method.quality}/10</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">{'Speed'}: </span>
                      <span className={getSpeedColor(method.speed)}>{method.speed}</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">{'Deterministic'}: </span>
                      <span>{method.deterministic ? ('Yes') : ('No')}</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
        </div>
      )}

      {/* Learn More */}
      <Card>
        <CardHeader>
          <CardTitle>
            {'About Sampling Methods'}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-sm">
          <p className="text-muted-foreground">
            {'Sampling methods determine the specific path from noise to image. Different algorithms offer different trade-offs between speed, quality, and determinism. DDIM and DPM-Solver++ represent modern fast samplers, while Euler and Heun are classic numerical methods.'
            }
          </p>
          <div className="p-4 bg-muted rounded-lg">
            <h4 className="font-medium mb-2">
              {'Recommendations'}
            </h4>
            <ul className="space-y-2 text-xs">
              <li>• <strong>{'Fast Generation'}:</strong> DPM-Solver++ (20 steps)</li>
              <li>• <strong>{'Best Quality'}:</strong> DDIM (50 steps)</li>
              <li>• <strong>{'Reproducible'}:</strong> DDIM or DPM-Solver++ (deterministic)</li>
              <li>• <strong>{'Creative Exploration'}:</strong> Euler a (stochastic)</li>
            </ul>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
