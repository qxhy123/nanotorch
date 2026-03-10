/**
 * Controlled Generation Visualization
 *
 * Explore ControlNet, T2I-Adapter and other controlled generation methods
 */

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/card';
import { Slider } from '../components/ui/slider';
import { Layers, Image as ImageIcon, Workflow } from 'lucide-react';
import { InlineMath } from 'react-katex';
import 'katex/dist/katex.min.css';

type ControlType = 'canny' | 'depth' | 'pose' | 'seg' | 'scribble';
type ControlMethod = 'controlnet' | 't2i-adapter' | 'ip-adapter';

interface ControlInput {
  type: ControlType;
  name: string;  description: string;  icon: React.ReactNode;
}

const controlTypes: ControlInput[] = [
  {
    type: 'canny',
    name: 'Canny Edge',
    description: 'Detect edges in input image',
    icon: <Layers className="h-5 w-5" />
  },
  {
    type: 'depth',
    name: 'Depth Map',
    description: 'Use depth information from input',
    icon: <Layers className="h-5 w-5" />
  },
  {
    type: 'pose',
    name: 'Pose',
    description: 'Control using human pose keypoints',
    icon: <Layers className="h-5 w-5" />
  },
  {
    type: 'seg',
    name: 'Segmentation',
    description: 'Semantic segmentation map',
    icon: <Layers className="h-5 w-5" />
  },
  {
    type: 'scribble',
    name: 'Scribble',
    description: 'Hand-drawn sketches',
    icon: <Layers className="h-5 w-5" />
  },
];

const controlMethods = [
  {
    id: 'controlnet',
    name: 'ControlNet',
    description: 'Zero-convolution architecture for precise control'
  },
  {
    id: 't2i-adapter',
    name: 'T2I-Adapter',
    description: 'Lightweight adapter for style control'
  },
  {
    id: 'ip-adapter',
    name: 'IP-Adapter',
    description: 'Image prompt adapter'
  },
];

export const ControlView = () => {
  
  const [selectedControl, setSelectedControl] = useState<ControlType>('canny');
  const [selectedMethod, setSelectedMethod] = useState<ControlMethod>('controlnet');
  const [controlWeight, setControlWeight] = useState(1.0);
  const [guidanceScale, setGuidanceScale] = useState(7.5);

  const selectedControlData = controlTypes.find(c => c.type === selectedControl);
  const selectedMethodData = controlMethods.find(m => m.id === selectedMethod);

  const generateControlImage = (type: ControlType) => {
    // Generate pseudo control image
    const patterns = {
      canny: Array.from({ length: 256 }, (_, i) => {
        const x = i % 16;
        const y = Math.floor(i / 16);
        const edge = Math.abs(Math.sin(x * 0.5) - Math.cos(y * 0.5)) < 0.3 ? 1 : 0;
        return edge;
      }),
      depth: Array.from({ length: 256 }, (_, i) => {
        const x = i % 16;
        const y = Math.floor(i / 16);
        return Math.sin(x * 0.3) * Math.cos(y * 0.3);
      }),
      pose: Array.from({ length: 256 }, () => Math.random() > 0.9 ? 1 : 0),
      seg: Array.from({ length: 256 }, (_, i) => (i % 16) % 3),
      scribble: Array.from({ length: 256 }, () => Math.random() > 0.85 ? 1 : 0)
  };
    return patterns[type];
  };

  const controlPattern = generateControlImage(selectedControl);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center py-8 bg-gradient-to-br from-primary/10 to-primary/5 rounded-lg">
        <h1 className="text-3xl font-bold mb-2">
          {'Controlled Generation Visualization'}
        </h1>
        <p className="text-muted-foreground">
          {'Explore ControlNet, T2I-Adapter and other controlled generation methods'
          }
        </p>
      </div>

      {/* Control Type Selector */}
      <Card>
        <CardHeader>
          <CardTitle>
            {'Control Type'}
          </CardTitle>
          <CardDescription>
            {'Select the type of control condition'}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {controlTypes.map(control => (
              <button
                key={control.type}
                className={`
                  p-4 rounded-lg border-2 transition-all cursor-pointer
                  ${selectedControl === control.type
                    ? 'border-primary bg-primary/10 shadow-lg ring-2 ring-primary/20'
                    : 'border-border bg-background hover:bg-accent hover:border-primary/50 hover:shadow-md'
                  }
                `}
                onClick={() => setSelectedControl(control.type)}
              >
                <div className="flex justify-center mb-2 text-primary">
                  {control.icon}
                </div>
                <div className="text-sm font-medium">
                  {control.name}
                </div>
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Control Method Selector */}
      <Card>
        <CardHeader>
          <CardTitle>
            {'Control Method'}
          </CardTitle>
          <CardDescription>
            {'Select control architecture'}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-3 gap-4">
            {controlMethods.map(method => (
              <button
                key={method.id}
                className={`
                  p-4 rounded-lg border-2 text-left transition-all cursor-pointer
                  ${selectedMethod === method.id
                    ? 'border-primary bg-primary/10 shadow-lg ring-2 ring-primary/20'
                    : 'border-border bg-background hover:bg-accent hover:border-primary/50 hover:shadow-md'
                  }
                `}
                onClick={() => setSelectedMethod(method.id as ControlMethod)}
              >
                <div className="font-medium mb-2">
                  {method.name}
                </div>
                <p className="text-sm text-muted-foreground">
                  {method.description}
                </p>
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Control Visualization */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Input Control */}
        <Card>
          <CardHeader>
            <CardTitle>
              {'Input Control'}
            </CardTitle>
            <CardDescription>
              {selectedControlData?.name}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="aspect-square bg-muted rounded-lg overflow-hidden">
              <svg viewBox="0 0 16 16" className="w-full h-full">
                {controlPattern.map((val, i) => {
                  const x = i % 16;
                  const y = Math.floor(i / 16);
                  const color = selectedControl === 'seg'
                    ? `hsl(${val * 120}, 70%, 50%)`
                    : `hsl(0, 0%, ${val * 100}%)`;
                  return (
                    <rect
                      key={i}
                      x={x}
                      y={y}
                      width="1"
                      height="1"
                      fill={color}
                    />
                  );
                })}
              </svg>
            </div>
          </CardContent>
        </Card>

        {/* Generated Result */}
        <Card>
          <CardHeader>
            <CardTitle>
              {'Generated Result'}
            </CardTitle>
            <CardDescription>
              {'Generated with control'}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="aspect-square bg-muted rounded-lg overflow-hidden relative">
              <svg viewBox="0 0 64 64" className="w-full h-full opacity-50">
                {controlPattern.map((val, i) => {
                  const x = (i % 16) * 4;
                  const y = Math.floor(i / 16) * 4;
                  const color = `hsl(${(x + y) % 360}, 70%, 60%)`;
                  return (
                    <rect
                      key={i}
                      x={x}
                      y={y}
                      width="4"
                      height="4"
                      fill={color}
                      opacity={val * 0.5 + 0.2}
                    />
                  );
                })}
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="bg-black/70 text-white px-4 py-2 rounded text-sm">
                  {'Preview'}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

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
                {'Control Weight'}
              </span>
              <span className="font-mono">{controlWeight.toFixed(2)}</span>
            </div>
            <Slider
              value={controlWeight}
              onValueChange={(v) => setControlWeight(v)}
              min={0}
              max={2}
              step={0.05}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>0.0</span>
              <span>{'Default'}: 1.0</span>
              <span>2.0</span>
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">
                <InlineMath style={{ color: 'inherit' }}>{"\\text{Guidance} \\text{ Scale}"}</InlineMath>
              </span>
              <span className="font-mono">{guidanceScale.toFixed(1)}</span>
            </div>
            <Slider
              value={guidanceScale}
              onValueChange={(v) => setGuidanceScale(v)}
              min={1}
              max={20}
              step={0.5}
              className="w-full"
            />
          </div>
        </CardContent>
      </Card>

      {/* Architecture Diagram */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Workflow className="h-5 w-5" />
            {'Control Architecture'}
          </CardTitle>
          <CardDescription>
            {selectedMethodData?.name}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col md:flex-row items-center justify-center gap-4 py-8">
            {/* Input */}
            <div className="text-center">
              <div className="w-24 h-24 bg-muted rounded-lg flex items-center justify-center mb-2">
                <ImageIcon className="h-8 w-8 text-muted-foreground" />
              </div>
              <div className="text-sm">{'Input'}</div>
            </div>

            <div className="text-4xl text-primary">→</div>

            {/* Control Extract */}
            <div className="text-center">
              <div className="w-24 h-24 bg-muted rounded-lg flex items-center justify-center mb-2 border-2 border-dashed border-primary">
                <Layers className="h-8 w-8 text-primary" />
              </div>
              <div className="text-sm">{'Control'}</div>
            </div>

            <div className="text-4xl text-primary">+</div>

            {/* SD */}
            <div className="text-center">
              <div className="w-32 h-24 bg-primary/20 rounded-lg flex items-center justify-center mb-2">
                <span className="font-bold text-primary">UNet</span>
              </div>
              <div className="text-sm">{'Denoising'}</div>
            </div>

            <div className="text-4xl text-primary">→</div>

            {/* Output */}
            <div className="text-center">
              <div className="w-24 h-24 bg-gradient-to-br from-primary/20 to-accent/20 rounded-lg flex items-center justify-center mb-2">
                <ImageIcon className="h-8 w-8 text-primary" />
              </div>
              <div className="text-sm">{'Output'}</div>
            </div>
          </div>

          <div className="mt-4 p-4 bg-muted rounded-lg">
            <h4 className="text-sm font-medium mb-2">
              {'How It Works'}
            </h4>
            <p className="text-sm text-muted-foreground">
              {selectedMethod === 'controlnet'
                ? 'ControlNet injects control conditions into each UNet downsampling block via zero-convolution layers for precise spatial control.'
                : selectedMethod === 't2i-adapter'
                ? 'T2I-Adapter uses lightweight adapter layers to inject style information during inference.'
                : 'IP-Adapter enables cross-attention control via image prompts, allowing reference image guidance.'
              }
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Learn More */}
      <Card>
        <CardHeader>
          <CardTitle>
            {'About Controlled Generation'}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-sm">
          <p className="text-muted-foreground">
            {'Controlled generation methods allow users to precisely control the structure and layout of generated images through additional conditional inputs (edge maps, depth maps, poses, etc.). ControlNet is the most popular control method, T2I-Adapter provides a lighter alternative, and IP-Adapter supports image prompt control.'
            }
          </p>
          <div className="p-4 bg-muted rounded-lg">
            <h4 className="font-medium mb-2">
              {'Control Type Comparison'}
            </h4>
            <div className="grid md:grid-cols-2 gap-2 text-xs">
              <div><strong>Canny:</strong> {'Edge contours'}</div>
              <div><strong>Depth:</strong> {'Spatial depth'}</div>
              <div><strong>Pose:</strong> {'Human pose'}</div>
              <div><strong>Seg:</strong> {'Semantic regions'}</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
