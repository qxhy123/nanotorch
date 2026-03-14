import React, { useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../ui/card';
import { Badge } from '../../ui/badge';
import { useTransformerStore } from '../../../stores/transformerStore';
import { Latex } from '../../ui/Latex';

interface PositionalEncodingProps {
  className?: string;
}

export const PositionalEncoding: React.FC<PositionalEncodingProps> = ({ className }) => {
  const { config, visualizationState } = useTransformerStore();
  const { colorScheme } = visualizationState;

  const positionalEncodings = useMemo(() => {
    const { max_seq_len, d_model } = config;
    const encodings: number[][] = [];

    for (let pos = 0; pos < Math.min(max_seq_len, 32); pos++) {
      const encoding: number[] = [];
      for (let i = 0; i < Math.min(d_model, 64); i += 2) {
        const angle = pos / Math.pow(10000, (2 * i) / d_model);
        encoding.push(Math.sin(angle));
        if (i + 1 < d_model) {
          encoding.push(Math.cos(angle));
        }
      }
      encodings.push(encoding);
    }

    return encodings;
  }, [config]);

  const waveform = useMemo(() => {
    const data: { pos: number; even: number; odd: number }[] = [];
    for (let pos = 0; pos < 64; pos++) {
      const evenDim = 0;
      const oddDim = 1;
      const angle = pos / Math.pow(10000, (2 * evenDim) / config.d_model);
      const even = Math.sin(angle);
      const odd = Math.cos(pos / Math.pow(10000, (2 * oddDim) / config.d_model));
      data.push({ pos, even, odd });
    }
    return data;
  }, [config]);

  const getColor = (value: number): string => {
    const normalized = (value + 1) / 2; // Normalize from [-1, 1] to [0, 1]

    switch (colorScheme) {
      case 'blues': {
        const blue = Math.round(255 * (1 - normalized));
        return `rgb(${Math.round(200 * normalized)}, ${Math.round(200 * normalized)}, ${255 - blue})`;
      }
      case 'reds': {
        const red = Math.round(255 * (1 - normalized));
        return `rgb(${255 - red}, ${Math.round(200 * normalized)}, ${Math.round(200 * normalized)})`;
      }
      case 'viridis':
        if (normalized < 0.25) {
          return `rgb(68, 1, ${Math.round(84 + normalized * 4 * 100)})`;
        } else if (normalized < 0.5) {
          return `rgb(${Math.round(33 + (normalized - 0.25) * 4 * 200)}, ${Math.round(144 + (normalized - 0.25) * 4 * 50)}, 140)`;
        } else if (normalized < 0.75) {
          return `rgb(${Math.round(233 - (normalized - 0.5) * 4 * 100)}, ${Math.round(196 - (normalized - 0.5) * 4 * 50)}, ${Math.round(106 + (normalized - 0.5) * 4 * 80)})`;
        } else {
          return `rgb(${Math.round(253 - (normalized - 0.75) * 4 * 100)}, ${Math.round(231 - (normalized - 0.75) * 4 * 100)}, ${Math.round(37 + (normalized - 0.75) * 4 * 40)})`;
        }
      default:
        return `rgb(${Math.round(255 * (1 - normalized))}, ${Math.round(255 * (1 - normalized))}, ${Math.round(255 * (1 - normalized))})`;
    }
  };

  const stats = useMemo(() => {
    const allValues = positionalEncodings.flat();
    const min = Math.min(...allValues);
    const max = Math.max(...allValues);
    const mean = allValues.reduce((a, b) => a + b, 0) / allValues.length;

    return { min, max, mean };
  }, [positionalEncodings]);

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Positional Encoding</CardTitle>
            <CardDescription>Sinusoidal position encoding</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Statistics */}
        <div className="flex gap-4 text-sm">
          <Badge variant="secondary">Range: [{stats.min.toFixed(2)}, {stats.max.toFixed(2)}]</Badge>
          <Badge variant="secondary">Mean: {stats.mean.toFixed(4)}</Badge>
        </div>

        {/* Formula */}
        <div className="p-3 bg-muted rounded-md text-sm">
          <div className="text-xs text-muted-foreground mb-2">Formulas:</div>
          <div className="space-y-1">
            <Latex>{`PE(pos, 2i) = \\sin\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)`}</Latex>
            <Latex>{`PE(pos, 2i+1) = \\cos\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)`}</Latex>
          </div>
        </div>

        {/* 2D Heatmap */}
        <div className="space-y-2">
          <h4 className="text-sm font-medium">2D Encoding Matrix</h4>
          <div className="overflow-x-auto">
            <div className="inline-block min-w-full">
              {positionalEncodings.map((encoding, posIdx) => (
                <div key={posIdx} className="flex items-center gap-2 mb-1">
                  <div className="w-8 text-xs text-muted-foreground shrink-0">
                    {posIdx}
                  </div>
                  <div className="flex flex-nowrap">
                    {encoding.map((value, dimIdx) => (
                      <div
                        key={dimIdx}
                        className="w-3 h-6 border-r border-r-background/20"
                        style={{
                          backgroundColor: getColor(value),
                        }}
                        title={`Pos ${posIdx}, Dim ${dimIdx}: ${value.toFixed(4)}`}
                      />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground w-8">Dim</span>
            <div className="flex-1 flex gap-1">
              {Array.from({ length: 16 }, (_, i) => (
                <div key={i} className="w-3 text-xs text-center text-muted-foreground">
                  {i * 4}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Waveform Visualization */}
        <div className="space-y-2">
          <h4 className="text-sm font-medium">Waveform (Dimensions 0 and 1)</h4>
          <svg viewBox="0 0 400 100" className="w-full h-24">
            {/* Grid lines */}
            {[0, 0.5, 1].map((y) => (
              <line
                key={y}
                x1="0"
                y1={y * 100}
                x2="400"
                y2={y * 100}
                stroke="currentColor"
                strokeOpacity="0.1"
              />
            ))}

            {/* Even dimension (sin) */}
            <polyline
              fill="none"
              stroke="rgb(59, 130, 246)"
              strokeWidth="2"
              points={waveform.map((d, i) => {
                const x = (i / waveform.length) * 400;
                const y = ((d.even + 1) / 2) * 100;
                return `${x},${y}`;
              }).join(' ')}
            />

            {/* Odd dimension (cos) */}
            <polyline
              fill="none"
              stroke="rgb(239, 68, 68)"
              strokeWidth="2"
              points={waveform.map((d, i) => {
                const x = (i / waveform.length) * 400;
                const y = ((d.odd + 1) / 2) * 100;
                return `${x},${y}`;
              }).join(' ')}
            />
          </svg>
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>sin (dim 0) - blue</span>
            <span>cos (dim 1) - red</span>
          </div>
        </div>

        {/* Color Scale Legend */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">-1</span>
          <div className="flex-1 h-3 rounded" style={{
            background: `linear-gradient(to right, ${getColor(-1)}, ${getColor(0)}, ${getColor(1)})`
          }} />
          <span className="text-xs text-muted-foreground">+1</span>
        </div>
      </CardContent>
    </Card>
  );
};
