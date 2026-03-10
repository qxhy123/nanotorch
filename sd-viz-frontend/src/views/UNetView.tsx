/**
 * UNet Architecture Visualization
 *
 * Shows the 3D UNet architecture with interactive blocks and skip connections
 */

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { ChevronDown, ChevronUp, ChevronRight, Layers, Box, Zap, Eye } from 'lucide-react';
import { InlineMath } from 'react-katex';
import 'katex/dist/katex.min.css';

interface UNetLayer {
  id: string;
  name: string;  type: 'down' | 'mid' | 'up';
  channels: number;
  spatialSize: number;
  params: string;
}

interface BlockDetail {
  layer: UNetLayer;
  inputShape: string;
  outputShape: string;
  description: string;}

const unetLayers: UNetLayer[] = [
  // Input
  { id: 'in', name: 'Input Latents', type: 'down', channels: 4, spatialSize: 64, params: '-' },
  // Downsampling blocks
  { id: 'd1', name: 'Down Block 1', type: 'down', channels: 320, spatialSize: 64, params: '15.8M' },
  { id: 'd2', name: 'Down Block 2', type: 'down', channels: 640, spatialSize: 32, params: '62.6M' },
  { id: 'd3', name: 'Down Block 3', type: 'down', channels: 1280, spatialSize: 16, params: '249.4M' },
  // Middle block
  { id: 'mid', name: 'Middle Block', type: 'mid', channels: 1280, spatialSize: 8, params: '189.2M' },
  // Upsampling blocks
  { id: 'u3', name: 'Up Block 3', type: 'up', channels: 1280, spatialSize: 16, params: '499.0M' },
  { id: 'u2', name: 'Up Block 2', type: 'up', channels: 640, spatialSize: 32, params: '125.2M' },
  { id: 'u1', name: 'Up Block 1', type: 'up', channels: 320, spatialSize: 64, params: '31.6M' },
  // Output
  { id: 'out', name: 'Output Noise', type: 'up', channels: 4, spatialSize: 64, params: '-' },
];

const getBlockDetail = (layer: UNetLayer): BlockDetail => ({
  layer,
  inputShape: `[2, ${layer.type === 'down' && layer.id !== 'in' ? layer.channels / 2 : layer.channels === 4 ? 4 : layer.channels}, ${layer.spatialSize}, ${layer.spatialSize}]`,
  outputShape: `[2, ${layer.channels}, ${layer.spatialSize}, ${layer.spatialSize}]`,
  description: layer.type === 'mid'
    ? 'Processes features at the lowest resolution with self-attention'
    : layer.type === 'down'
    ? 'Downsamples spatial resolution while increasing channels'
    : 'Upsamples spatial resolution with skip connections',
});

export const UNetView = () => {
  
  const [selectedLayer, setSelectedLayer] = useState<string | null>(null);
  const [hoveredLayer, setHoveredLayer] = useState<string | null>(null);

  const selectedBlock = selectedLayer
    ? getBlockDetail(unetLayers.find(l => l.id === selectedLayer)!)
    : null;

  const getBlockColor = (layer: UNetLayer) => {
    if (layer.type === 'mid') return 'bg-purple-500/20 border-purple-500';
    if (layer.type === 'down') return 'bg-blue-500/20 border-blue-500';
    return 'bg-green-500/20 border-green-500';
  };

  const getBlockIcon = (layer: UNetLayer) => {
    if (layer.type === 'mid') return <Zap className="h-4 w-4" />;
    if (layer.type === 'down') return <ChevronDown className="h-4 w-4" />;
    return <ChevronUp className="h-4 w-4" />;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center py-8 bg-gradient-to-br from-primary/10 to-primary/5 rounded-lg">
        <h1 className="text-3xl font-bold mb-2">
          {'UNet Architecture Visualization'}
        </h1>
        <p className="text-muted-foreground">
          {'Explore the 3D architecture of Stable Diffusion UNet with downsample, middle, and upsample paths'
          }
        </p>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-6">
            <div className="text-center">
              <Layers className="h-8 w-8 mx-auto mb-2 text-primary" />
              <p className="text-2xl font-bold">23</p>
              <p className="text-xs text-muted-foreground">
                {'ResNet Blocks'}
              </p>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-center">
              <Eye className="h-8 w-8 mx-auto mb-2 text-primary" />
              <p className="text-2xl font-bold">32</p>
              <p className="text-xs text-muted-foreground">
                {'Attention Heads'}
              </p>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-center">
              <Box className="h-8 w-8 mx-auto mb-2 text-primary" />
              <p className="text-2xl font-bold"><InlineMath style={{ color: 'inherit' }}>{"860 \\text{M}"}</InlineMath></p>
              <p className="text-xs text-muted-foreground">
                {'Parameters'}
              </p>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-center">
              <ChevronRight className="h-8 w-8 mx-auto mb-2 text-primary" />
              <p className="text-2xl font-bold">3</p>
              <p className="text-xs text-muted-foreground">
                {'Resolution Levels'}
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* 3D Architecture Diagram */}
      <Card>
        <CardHeader>
          <CardTitle>
            {'Architecture Diagram'}
          </CardTitle>
          <CardDescription>
            {'Click blocks for details'}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col md:flex-row gap-8">
            {/* Left side - Downsampling */}
            <div className="flex-1 space-y-3">
              <h3 className="text-sm font-medium text-muted-foreground mb-4">
                {'Downsampling Path'}
              </h3>
              {unetLayers.filter(l => l.type === 'down' || l.id === 'in').map(layer => (
                <div
                  key={layer.id}
                  className={`
                    relative p-4 rounded-lg border-2 cursor-pointer transition-all
                    ${getBlockColor(layer)}
                    ${hoveredLayer === layer.id ? 'scale-105 shadow-lg' : ''}
                    ${selectedLayer === layer.id ? 'ring-2 ring-primary' : ''}
                  `}
                  onClick={() => setSelectedLayer(layer.id)}
                  onMouseEnter={() => setHoveredLayer(layer.id)}
                  onMouseLeave={() => setHoveredLayer(null)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      {getBlockIcon(layer)}
                      <div>
                        <div className="font-medium">
                          {layer.name}
                        </div>
                        <div className="text-xs text-muted-foreground font-mono">
                          Ch: {layer.channels} | Size: <InlineMath style={{ color: 'inherit' }}>{`${layer.spatialSize} \\times ${layer.spatialSize}`}</InlineMath>
                        </div>
                      </div>
                    </div>
                    <ChevronRight className="h-4 w-4 text-muted-foreground" />
                  </div>

                  {/* Skip connection indicator */}
                  {layer.type === 'down' && layer.id !== 'in' && (
                    <div className="absolute -right-4 top-1/2 -translate-y-1/2 w-8 h-0.5 bg-yellow-500/50" />
                  )}
                </div>
              ))}
            </div>

            {/* Middle */}
            <div className="flex items-center justify-center">
              <div
                className={`
                  p-6 rounded-lg border-2 cursor-pointer transition-all
                  ${getBlockColor(unetLayers.find(l => l.type === 'mid')!)}
                  ${hoveredLayer === 'mid' ? 'scale-105 shadow-lg' : ''}
                  ${selectedLayer === 'mid' ? 'ring-2 ring-primary' : ''}
                `}
                onClick={() => setSelectedLayer('mid')}
                onMouseEnter={() => setHoveredLayer('mid')}
                onMouseLeave={() => setHoveredLayer(null)}
              >
                <Zap className="h-6 w-6 mb-2 mx-auto" />
                <div className="text-center">
                  <div className="font-medium text-sm">
                    {'Middle Block'}
                  </div>
                  <div className="text-xs text-muted-foreground font-mono">
                    1280 ch
                  </div>
                </div>
              </div>
            </div>

            {/* Right side - Upsampling */}
            <div className="flex-1 space-y-3">
              <h3 className="text-sm font-medium text-muted-foreground mb-4">
                {'Upsampling Path'}
              </h3>
              {unetLayers.filter(l => l.type === 'up' || l.id === 'out').map(layer => (
                <div
                  key={layer.id}
                  className={`
                    relative p-4 rounded-lg border-2 cursor-pointer transition-all
                    ${getBlockColor(layer)}
                    ${hoveredLayer === layer.id ? 'scale-105 shadow-lg' : ''}
                    ${selectedLayer === layer.id ? 'ring-2 ring-primary' : ''}
                  `}
                  onClick={() => setSelectedLayer(layer.id)}
                  onMouseEnter={() => setHoveredLayer(layer.id)}
                  onMouseLeave={() => setHoveredLayer(null)}
                >
                  <div className="flex items-center justify-between">
                    <ChevronRight className="h-4 w-4 text-muted-foreground" />
                    <div className="flex items-center gap-3 flex-1 justify-end">
                      <div className="text-right">
                        <div className="font-medium">
                          {layer.name}
                        </div>
                        <div className="text-xs text-muted-foreground font-mono">
                          Ch: {layer.channels} | Size: <InlineMath style={{ color: 'inherit' }}>{`${layer.spatialSize} \\times ${layer.spatialSize}`}</InlineMath>
                        </div>
                      </div>
                      {getBlockIcon(layer)}
                    </div>
                  </div>

                  {/* Skip connection indicator */}
                  {layer.type === 'up' && layer.id !== 'out' && (
                    <div className="absolute -left-4 top-1/2 -translate-y-1/2 w-8 h-0.5 bg-yellow-500/50" />
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Legend */}
          <div className="flex justify-center gap-6 mt-6 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-blue-500/20 border-2 border-blue-500 rounded" />
              <span>{'Downsample'}</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-purple-500/20 border-2 border-purple-500 rounded" />
              <span>{'Middle'}</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-green-500/20 border-2 border-green-500 rounded" />
              <span>{'Upsample'}</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-8 h-0.5 bg-yellow-500/50" />
              <span>{'Skip Connection'}</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Block Detail */}
      {selectedBlock && (
        <Card className="border-2 border-primary">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>
                  {selectedBlock.layer.name}
                </CardTitle>
                <CardDescription>
                  {'Detailed parameters and operations'}
                </CardDescription>
              </div>
              <Button variant="ghost" size="sm" onClick={() => setSelectedLayer(null)}>
                ×
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Description */}
            <p className="text-sm text-muted-foreground">
              {selectedBlock.description}
            </p>

            {/* Tensor Shapes */}
            <div className="grid md:grid-cols-2 gap-4">
              <div className="p-4 bg-muted rounded-lg">
                <div className="text-xs text-muted-foreground mb-1">
                  {'Input Shape'}
                </div>
                <div className="font-mono text-sm">{selectedBlock.inputShape}</div>
              </div>
              <div className="p-4 bg-muted rounded-lg">
                <div className="text-xs text-muted-foreground mb-1">
                  {'Output Shape'}
                </div>
                <div className="font-mono text-sm">{selectedBlock.outputShape}</div>
              </div>
            </div>

            {/* Operations */}
            <div className="p-4 bg-muted rounded-lg">
              <h4 className="text-sm font-medium mb-3">
                {'Operations'}
              </h4>
              <div className="space-y-2 text-sm">
                {selectedBlock.layer.type === 'mid' ? (
                  <>
                    <div className="flex items-center gap-2">
                      <span className="text-primary">1.</span>
                      <span>GroupNorm(32) → SiLU → Conv2D</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-primary">2.</span>
                      <span>Self-Attention (8 heads)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-primary">3.</span>
                      <span>GroupNorm(32) → SiLU → Conv2D</span>
                    </div>
                  </>
                ) : selectedBlock.layer.type === 'down' ? (
                  <>
                    <div className="flex items-center gap-2">
                      <span className="text-primary">1.</span>
                      <span>GroupNorm(32) → SiLU → Conv2D (downsample)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-primary">2.</span>
                      <span>Spatial Self-Attention (8 heads)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-primary">3.</span>
                      <span>GroupNorm(32) → SiLU → Conv2D</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-primary">4.</span>
                      <span>Residual connection</span>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="flex items-center gap-2">
                      <span className="text-primary">1.</span>
                      <span>Upsample (nearest neighbor)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-primary">2.</span>
                      <span>Concat with skip connection</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-primary">3.</span>
                      <span>GroupNorm(32) → SiLU → Conv2D</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-primary">4.</span>
                      <span>Spatial Self-Attention (8 heads)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-primary">5.</span>
                      <span>GroupNorm(32) → SiLU → Conv2D</span>
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* Parameters */}
            {selectedBlock.layer.params !== '-' && (
              <div className="p-4 bg-muted rounded-lg">
                <h4 className="text-sm font-medium mb-3">
                  {'Parameters'}
                </h4>
                <div className="text-2xl font-bold">{selectedBlock.layer.params}</div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Learn More */}
      <Card>
        <CardHeader>
          <CardTitle>
            {'Architecture Overview'}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-sm">
          <p className="text-muted-foreground">
            {'UNet is the core denoising network of Stable Diffusion. It adopts a U-shaped architecture: the left side progressively downsamples to extract features, while the right side recovers high-resolution details through upsampling and skip connections. The middle block processes global features at the lowest resolution.'
            }
          </p>
          <div className="p-4 bg-muted rounded-lg">
            <h4 className="font-medium mb-2">
              {'Key Features'}
            </h4>
            <ul className="space-y-2 text-xs">
              <li>• {'Skip connections preserve spatial details'}</li>
              <li>• {'Spatial self-attention captures long-range dependencies'}</li>
              <li>• {'GroupNorm stabilizes training'}</li>
              <li>• {'8 attention heads provide multi-scale representations'}</li>
            </ul>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
