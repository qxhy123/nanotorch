/**
 * Latent Space Exploration
 *
 * Explore VAE latent space with 3D visualization and interpolation
 */

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/card';
import { Slider } from '../components/ui/slider';
import { Box, ArrowRight, ScanLine } from 'lucide-react';
import { InlineMath } from 'react-katex';
import 'katex/dist/katex.min.css';

// Generate pseudo-latent vectors
const generateLatent = (seed: number): number[] => {
  return Array.from({ length: 4 * 8 * 8 }, (_, i) =>
    Math.sin(seed + i * 0.1) * Math.cos(seed * 0.5 + i * 0.05)
  );
};

// Project to 3D
const projectTo3D = (latent: number[]): { x: number; y: number; z: number } => {
  return {
    x: latent[0] * 10,
    y: latent[1] * 10,
    z: latent[2] * 10
  };
};

// Generate interpolation frames
const interpolateFrames = (
  from: number[],
  to: number[],
  steps: number
): number[][] => {
  const frames: number[][] = [];
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    frames.push(from.map((v, j) => v * (1 - t) + to[j] * t));
  }
  return frames;
};

export const LatentSpaceView = () => {
  
  const [mode, setMode] = useState<'vae' | 'interpolation' | 'noise'>('vae');
  const [interpolationPos, setInterpolationPos] = useState(50);

  // Sample latent points
  const latentPoints = Array.from({ length: 100 }, (_, i) => ({
    id: i,
    latent: generateLatent(i),
    position: projectTo3D(generateLatent(i)),
    category: i % 5
  }));

  // Interpolation between two points
  const fromLatent = generateLatent(1);
  const toLatent = generateLatent(50);
  const interpolated = interpolateFrames(
    fromLatent,
    toLatent,
    10
  )[Math.floor(interpolationPos / 10)];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center py-8 bg-gradient-to-br from-primary/10 to-primary/5 rounded-lg">
        <h1 className="text-3xl font-bold mb-2">
          {'Latent Space Exploration'}
        </h1>
        <p className="text-muted-foreground">
          {'Explore VAE latent space and understand how images are represented in compressed space'
          }
        </p>
      </div>

      {/* Mode Selector */}
      <div className="flex justify-center gap-4">
        <button
          className={`px-6 py-3 rounded-lg border-2 transition-all cursor-pointer ${
            mode === 'vae'
              ? 'border-primary bg-primary/10 shadow-lg ring-2 ring-primary/20'
              : 'border-border bg-background hover:bg-accent hover:border-primary/50 hover:shadow-md'
          }`}
          onClick={() => setMode('vae')}
        >
          <Box className="h-5 w-5 mx-auto mb-2" />
          <div className="text-sm font-medium">
            {'VAE Encode/Decode'}
          </div>
        </button>
        <button
          className={`px-6 py-3 rounded-lg border-2 transition-all cursor-pointer ${
            mode === 'interpolation'
              ? 'border-primary bg-primary/10 shadow-lg ring-2 ring-primary/20'
              : 'border-border bg-background hover:bg-accent hover:border-primary/50 hover:shadow-md'
          }`}
          onClick={() => setMode('interpolation')}
        >
          <ArrowRight className="h-5 w-5 mx-auto mb-2" />
          <div className="text-sm font-medium">
            {'Vector Interpolation'}
          </div>
        </button>
        <button
          className={`px-6 py-3 rounded-lg border-2 transition-all cursor-pointer ${
            mode === 'noise'
              ? 'border-primary bg-primary/10 shadow-lg ring-2 ring-primary/20'
              : 'border-border bg-background hover:bg-accent hover:border-primary/50 hover:shadow-md'
          }`}
          onClick={() => setMode('noise')}
        >
          <ScanLine className="h-5 w-5 mx-auto mb-2" />
          <div className="text-sm font-medium">
            {'Noise Exploration'}
          </div>
        </button>
      </div>

      {mode === 'vae' && (
        <>
          {/* VAE Visualization */}
          <Card>
            <CardHeader>
              <CardTitle>
                {'VAE Compression Visualization'}
              </CardTitle>
              <CardDescription>
                Image compression from <InlineMath style={{ color: 'inherit' }}>{"512 \\times 512 \\times 3"}</InlineMath> to <InlineMath style={{ color: 'inherit' }}>{"4 \\times 64 \\times 64"}</InlineMath> latent representation
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-3 gap-6 items-center">
                {/* Original */}
                <div className="text-center">
                  <div className="aspect-square bg-muted rounded-lg mb-3 overflow-hidden">
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
                            fill={`hsl(${(x * 20 + y * 10) % 360}, 70%, 60%)`}
                          />
                        );
                      })}
                    </svg>
                  </div>
                  <div className="text-sm">
                    <div className="font-medium">{'Original'}</div>
                    <div className="text-xs text-muted-foreground font-mono">
                      <InlineMath style={{ color: 'inherit' }}>{"512 \\times 512 \\times 3 = 786,432"}</InlineMath>
                    </div>
                  </div>
                </div>

                {/* Arrow */}
                <div className="text-center">
                  <div className="text-4xl text-primary">→</div>
                  <div className="text-sm text-muted-foreground mt-2">
                    {'Compress 8x'}
                  </div>
                </div>

                {/* Latent */}
                <div className="text-center">
                  <div className="aspect-square bg-muted rounded-lg mb-3 overflow-hidden">
                    <svg viewBox="0 0 64 64" className="w-full h-full">
                      {Array.from({ length: 64 }).map((_, i) => {
                        const x = i % 8;
                        const y = Math.floor(i / 8);
                        const val = (Math.sin(x * 0.5) * Math.cos(y * 0.5) + 1) / 2;
                        return (
                          <rect
                            key={i}
                            x={x * 8}
                            y={y * 8}
                            width="8"
                            height="8"
                            fill={`hsl(240, 70%, ${20 + val * 60}%)`}
                          />
                        );
                      })}
                    </svg>
                  </div>
                  <div className="text-sm">
                    <div className="font-medium">{'Latent'}</div>
                    <div className="text-xs text-muted-foreground font-mono">
                      <InlineMath style={{ color: 'inherit' }}>{"4 \\times 64 \\times 64 = 16,384"}</InlineMath>
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-4 p-4 bg-muted rounded-lg">
                <div className="text-sm font-medium mb-2">
                  {'Compression Ratio'}
                </div>
                <div className="text-2xl font-bold">
                  <InlineMath style={{ color: 'inherit' }}>{"48\\times"}</InlineMath> {'smaller'}
                </div>
                <div className="text-xs text-muted-foreground mt-1">
                  {'Latent space preserves semantics, discards details'
                  }
                </div>
              </div>
            </CardContent>
          </Card>

          {/* 3D Scatter */}
          <Card>
            <CardHeader>
              <CardTitle>
                {'Latent Space 3D Scatter'}
              </CardTitle>
              <CardDescription>
                {'PCA-projected latent vector distribution'}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="aspect-square relative bg-muted rounded-lg overflow-hidden">
                <svg viewBox="-10 -10 20 20" className="w-full h-full">
                  {/* Grid */}
                  <g opacity="0.1">
                    {Array.from({ length: 21 }, (_, i) => (
                      <line
                        key={`v-${i}`}
                        x1={-10 + i}
                        y1="-10"
                        x2={-10 + i}
                        y2="10"
                        stroke="currentColor"
                      />
                    ))}
                    {Array.from({ length: 21 }, (_, i) => (
                      <line
                        key={`h-${i}`}
                        x1="-10"
                        y1={-10 + i}
                        x2="10"
                        y2={-10 + i}
                        stroke="currentColor"
                      />
                    ))}
                  </g>

                  {/* Points */}
                  {latentPoints.map((point) => (
                    <circle
                      key={point.id}
                      cx={point.position.x}
                      cy={point.position.y}
                      r="0.3"
                      fill={`hsl(${point.category * 72}, 70%, 50%)`}
                      className="hover:r-0.5 transition-all"
                    />
                  ))}
                </svg>
              </div>
            </CardContent>
          </Card>
        </>
      )}

      {mode === 'interpolation' && (
        <>
          {/* Vector Interpolation */}
          <Card>
            <CardHeader>
              <CardTitle>
                {'Vector Interpolation'}
              </CardTitle>
              <CardDescription>
                {'Smooth transition between two latent vectors'
                }
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-3 gap-6 items-center">
                {/* From */}
                <div className="text-center">
                  <div className="aspect-square bg-muted rounded-lg mb-3 overflow-hidden">
                    <svg viewBox="0 0 64 64" className="w-full h-full">
                      {Array.from({ length: 64 }).map((_, i) => {
                        const x = i % 8;
                        const y = Math.floor(i / 8);
                        const val = (Math.sin(fromLatent[i] * Math.PI) + 1) / 2;
                        return (
                          <rect
                            key={i}
                            x={x * 8}
                            y={y * 8}
                            width="8"
                            height="8"
                            fill={`hsl(200, 70%, ${20 + val * 60}%)`}
                          />
                        );
                      })}
                    </svg>
                  </div>
                  <div className="text-sm font-medium">
                    {'From'}
                  </div>
                </div>

                {/* Interpolated */}
                <div className="text-center">
                  <div className="aspect-square bg-muted rounded-lg mb-3 overflow-hidden ring-2 ring-primary">
                    <svg viewBox="0 0 64 64" className="w-full h-full">
                      {Array.from({ length: 64 }).map((_, i) => {
                        const x = i % 8;
                        const y = Math.floor(i / 8);
                        const val = (Math.sin(interpolated[i] * Math.PI) + 1) / 2;
                        return (
                          <rect
                            key={i}
                            x={x * 8}
                            y={y * 8}
                            width="8"
                            height="8"
                            fill={`hsl(280, 70%, ${20 + val * 60}%)`}
                          />
                        );
                      })}
                    </svg>
                  </div>
                  <div className="text-sm font-medium">
                    {'Interpolated'} ({interpolationPos}%)
                  </div>
                </div>

                {/* To */}
                <div className="text-center">
                  <div className="aspect-square bg-muted rounded-lg mb-3 overflow-hidden">
                    <svg viewBox="0 0 64 64" className="w-full h-full">
                      {Array.from({ length: 64 }).map((_, i) => {
                        const x = i % 8;
                        const y = Math.floor(i / 8);
                        const val = (Math.sin(toLatent[i] * Math.PI) + 1) / 2;
                        return (
                          <rect
                            key={i}
                            x={x * 8}
                            y={y * 8}
                            width="8"
                            height="8"
                            fill={`hsl(340, 70%, ${20 + val * 60}%)`}
                          />
                        );
                      })}
                    </svg>
                  </div>
                  <div className="text-sm font-medium">
                    {'To'}
                  </div>
                </div>
              </div>

              {/* Slider */}
              <div className="mt-6">
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-muted-foreground">
                    {'Interpolation Position'}
                  </span>
                  <span className="font-mono">{interpolationPos}%</span>
                </div>
                <Slider
                  value={interpolationPos}
                  onValueChange={(v) => setInterpolationPos(v)}
                  min={0}
                  max={100}
                  step={1}
                  className="w-full"
                />
              </div>
            </CardContent>
          </Card>
        </>
      )}

      {mode === 'noise' && (
        <>
          {/* Noise Exploration */}
          <Card>
            <CardHeader>
              <CardTitle>
                {'Noise Exploration'}
              </CardTitle>
              <CardDescription>
                {'Understand the role of noise in diffusion process'
                }
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-4 gap-4">
                {Array.from({ length: 4 }, (_, i) => (
                  <div key={i} className="text-center">
                    <div className="aspect-square bg-muted rounded-lg mb-2 overflow-hidden">
                      <svg viewBox="0 0 64 64" className="w-full h-full">
                        {Array.from({ length: 256 }).map((_, j) => {
                          const x = j % 16;
                          const y = Math.floor(j / 16);
                          const noise = Math.random();
                          return (
                            <rect
                              key={j}
                              x={x * 4}
                              y={y * 4}
                              width="4"
                              height="4"
                              fill={`hsl(0, 0%, ${noise * (50 + i * 10)}%)`}
                            />
                          );
                        })}
                      </svg>
                    </div>
                    <div className="text-xs text-muted-foreground">
                      Sample {i + 1}
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-4 p-4 bg-muted rounded-lg">
                <h4 className="text-sm font-medium mb-2">
                  {'Key Concepts'}
                </h4>
                <ul className="space-y-2 text-xs">
                  <li>• {'Initial noise is Gaussian distributed'}</li>
                  <li>• {'Denoise progressively at each step'}</li>
                  <li>• {'Same seed produces same result'}</li>
                  <li>• {'Noise prediction is deterministic'}</li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </>
      )}

      {/* Learn More */}
      <Card>
        <CardHeader>
          <CardTitle>
            {'About Latent Space'}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-sm">
          <p className="text-muted-foreground">
            {'Latent space is a compressed representation of images. VAE compresses'} <InlineMath style={{ color: 'inherit' }}>{"512 \\times 512"}</InlineMath> {'images to'} <InlineMath style={{ color: 'inherit' }}>{"4 \\times 64 \\times 64"}</InlineMath> {'latent space. While high-frequency details are lost, semantic information is preserved. This allows diffusion models to work in a much smaller space, greatly improving computational efficiency.'
            }
          </p>
          <div className="p-4 bg-muted rounded-lg">
            <h4 className="font-medium mb-2">
              {'VAE Parameters'}
            </h4>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>{'Compression Factor'}: <InlineMath style={{ color: 'inherit' }}>{"8\\times"}</InlineMath></div>
              <div>{'Latent Dim'}: <InlineMath style={{ color: 'inherit' }}>{"4"}</InlineMath> channels</div>
              <div>{'Spatial Size'}: <InlineMath style={{ color: 'inherit' }}>{"64 \\times 64"}</InlineMath></div>
              <div>{'Parameters'}: <InlineMath style={{ color: 'inherit' }}>{"83 \\text{M}"}</InlineMath></div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
