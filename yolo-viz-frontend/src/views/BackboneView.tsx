/**
 * Backbone View - CSPDarknet/DarkNet visualization
 */

import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Layers, ArrowDown, Info } from 'lucide-react';
import { BlockMath, InlineMath } from 'react-katex';
import 'katex/dist/katex.min.css';

type BackboneType = 'cspdarknet53' | 'darknet53';

interface LayerBlock {
  id: string;
  name: string;
  type: string;
  inputSize: number;
  outputSize: number;
  channels: number;
  repeat: number;
  description: string;
}

const cspdarknetLayers: LayerBlock[] = [
  { id: 'stem', name: 'Stem', type: 'Conv', inputSize: 640, outputSize: 320, channels: 64, repeat: 1, description: 'Initial convolution with stride 2' },
  { id: 'stage1', name: 'Stage 1', type: 'CSP', inputSize: 320, outputSize: 160, channels: 128, repeat: 3, description: 'CSP block with 3 residual units' },
  { id: 'stage2', name: 'Stage 2', type: 'CSP', inputSize: 160, outputSize: 80, channels: 256, repeat: 9, description: 'CSP block with 9 residual units (P3 output)' },
  { id: 'stage3', name: 'Stage 3', type: 'CSP', inputSize: 80, outputSize: 40, channels: 512, repeat: 9, description: 'CSP block with 9 residual units (P4 output)' },
  { id: 'stage4', name: 'Stage 4', type: 'CSP', inputSize: 40, outputSize: 20, channels: 1024, repeat: 3, description: 'CSP block with 3 residual units (P5 output)' },
  { id: 'sppf', name: 'SPPF', type: 'SPPF', inputSize: 20, outputSize: 20, channels: 1024, repeat: 1, description: 'Spatial Pyramid Pooling Fast' },
];

const darknetLayers: LayerBlock[] = [
  { id: 'conv1', name: 'Conv1', type: 'Conv', inputSize: 640, outputSize: 320, channels: 32, repeat: 1, description: 'Initial 3x3 convolution' },
  { id: 'conv2', name: 'Conv2', type: 'Conv+Res', inputSize: 320, outputSize: 160, channels: 64, repeat: 1, description: 'Downsample + 1 residual block' },
  { id: 'conv3', name: 'Conv3', type: 'Conv+Res', inputSize: 160, outputSize: 80, channels: 128, repeat: 2, description: 'Downsample + 2 residual blocks (P3)' },
  { id: 'conv4', name: 'Conv4', type: 'Conv+Res', inputSize: 80, outputSize: 40, channels: 256, repeat: 8, description: 'Downsample + 8 residual blocks (P4)' },
  { id: 'conv5', name: 'Conv5', type: 'Conv+Res', inputSize: 40, outputSize: 20, channels: 512, repeat: 8, description: 'Downsample + 8 residual blocks (P5)' },
  { id: 'conv6', name: 'Conv6', type: 'Conv+Res', inputSize: 20, outputSize: 10, channels: 1024, repeat: 4, description: 'Downsample + 4 residual blocks' },
];

const backboneInfo = {
  cspdarknet53: {
    name: 'CSPDarknet53',
    description: 'Cross Stage Partial network that splits feature maps and uses partial transitions for better gradient flow.',
    params: '27.6M',
    flops: '65.7 GFLOPs',
  },
  darknet53: {
    name: 'Darknet53',
    description: 'Original backbone with 53 convolutional layers using residual connections.',
    params: '41.6M',
    flops: '91.5 GFLOPs',
  },
};

export const BackboneView: React.FC = () => {
  const [selectedBackbone, setSelectedBackbone] = useState<BackboneType>('cspdarknet53');
  const [selectedLayer, setSelectedLayer] = useState<string | null>(null);

  const layers = selectedBackbone === 'cspdarknet53' ? cspdarknetLayers : darknetLayers;
  const info = backboneInfo[selectedBackbone];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Layers className="h-8 w-8 text-primary" />
        <div>
          <h1 className="text-3xl font-bold">Backbone Network</h1>
          <p className="text-muted-foreground">Feature extraction from input images</p>
        </div>
      </div>

      {/* Backbone Selector */}
      <Card>
        <CardHeader>
          <CardTitle>Select Backbone</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2">
            <Button
              variant={selectedBackbone === 'cspdarknet53' ? 'default' : 'outline'}
              onClick={() => setSelectedBackbone('cspdarknet53')}
            >
              CSPDarknet53
            </Button>
            <Button
              variant={selectedBackbone === 'darknet53' ? 'default' : 'outline'}
              onClick={() => setSelectedBackbone('darknet53')}
            >
              Darknet53
            </Button>
          </div>
          <div className="mt-4 p-4 bg-muted rounded-lg">
            <h3 className="font-semibold">{info.name}</h3>
            <p className="text-sm text-muted-foreground mt-1">{info.description}</p>
            <div className="flex gap-4 mt-2 text-sm">
              <span>Params: <strong>{info.params}</strong></span>
              <span>FLOPs: <strong>{info.flops}</strong></span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Layer Diagram */}
      <div className="grid lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Layer Architecture</CardTitle>
            <CardDescription>Click on a layer to see details</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {/* Input */}
              <div className="bg-gradient-to-r from-blue-500 to-cyan-500 text-white rounded-lg p-3 text-center">
                <div className="font-medium">Input Image</div>
                <div className="text-sm opacity-80">640 x 640 x 3</div>
              </div>

              <ArrowDown className="mx-auto h-6 w-6 text-muted-foreground" />

              {/* Layers */}
              {layers.map((layer, index) => (
                <React.Fragment key={layer.id}>
                  <div
                    className={`rounded-lg p-3 cursor-pointer transition-all ${
                      selectedLayer === layer.id
                        ? 'bg-primary text-primary-foreground ring-2 ring-primary'
                        : 'bg-muted hover:bg-muted/80'
                    }`}
                    onClick={() => setSelectedLayer(layer.id === selectedLayer ? null : layer.id)}
                  >
                    <div className="flex justify-between items-center">
                      <div>
                        <span className="font-medium">{layer.name}</span>
                        <span className="text-xs ml-2 opacity-70">({layer.type})</span>
                      </div>
                      <div className="text-sm">
                        {layer.outputSize}x{layer.outputSize}x{layer.channels}
                      </div>
                    </div>
                    {layer.repeat > 1 && (
                      <div className="text-xs opacity-70 mt-1">
                        Repeat: {layer.repeat}x
                      </div>
                    )}
                  </div>
                  {index < layers.length - 1 && (
                    <ArrowDown className="mx-auto h-4 w-4 text-muted-foreground" />
                  )}
                </React.Fragment>
              ))}

              <ArrowDown className="mx-auto h-6 w-6 text-muted-foreground" />

              {/* Output */}
              <div className="bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg p-3 text-center">
                <div className="font-medium">Feature Maps</div>
                <div className="text-sm opacity-80">P3, P4, P5</div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Layer Details & Formulas */}
        <div className="space-y-6">
          {/* Selected Layer Details */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Info className="h-5 w-5" />
                Layer Details
              </CardTitle>
            </CardHeader>
            <CardContent>
              {selectedLayer ? (
                <div className="space-y-4">
                  {(() => {
                    const layer = layers.find(l => l.id === selectedLayer);
                    if (!layer) return null;
                    return (
                      <>
                        <div>
                          <h3 className="font-semibold text-lg">{layer.name}</h3>
                          <p className="text-sm text-muted-foreground">{layer.description}</p>
                        </div>
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div className="p-3 bg-muted rounded-lg">
                            <div className="text-muted-foreground">Input Size</div>
                            <div className="font-medium">{layer.inputSize} x {layer.inputSize}</div>
                          </div>
                          <div className="p-3 bg-muted rounded-lg">
                            <div className="text-muted-foreground">Output Size</div>
                            <div className="font-medium">{layer.outputSize} x {layer.outputSize}</div>
                          </div>
                          <div className="p-3 bg-muted rounded-lg">
                            <div className="text-muted-foreground">Channels</div>
                            <div className="font-medium">{layer.channels}</div>
                          </div>
                          <div className="p-3 bg-muted rounded-lg">
                            <div className="text-muted-foreground">Type</div>
                            <div className="font-medium">{layer.type}</div>
                          </div>
                        </div>
                      </>
                    );
                  })()}
                </div>
              ) : (
                <p className="text-muted-foreground text-center py-8">
                  Click on a layer to see details
                </p>
              )}
            </CardContent>
          </Card>

          {/* CSP Block Formula */}
          {selectedBackbone === 'cspdarknet53' && (
            <Card>
              <CardHeader>
                <CardTitle>CSP Block Formula</CardTitle>
                <CardDescription>Cross Stage Partial connections</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="bg-muted p-4 rounded-lg overflow-x-auto">
                  <BlockMath math="Y = \text{Concat}(f(X_1), X_2)" />
                </div>
                <div className="text-sm space-y-2">
                  <p>Where:</p>
                  <ul className="list-disc list-inside space-y-1 text-muted-foreground">
                    <li><InlineMath math="X_1, X_2" /> = Split input features (partial transition)</li>
                    <li><InlineMath math="f(\cdot)" /> = Dense block with residual connections</li>
                    <li><InlineMath math="Y" /> = Concatenated output features</li>
                  </ul>
                </div>
                <div className="p-3 bg-primary/5 rounded-lg text-sm">
                  <strong>Advantage:</strong> Reduces computation while maintaining accuracy by processing only part of the feature map through dense connections.
                </div>
              </CardContent>
            </Card>
          )}

          {/* Feature Map Sizes */}
          <Card>
            <CardHeader>
              <CardTitle>Feature Map Progression</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-end justify-between gap-2 h-32">
                {[640, 320, 160, 80, 40, 20].map((size, i) => (
                  <div key={size} className="flex flex-col items-center flex-1">
                    <div
                      className={`w-full bg-gradient-to-t from-primary to-primary/50 rounded-t transition-all`}
                      style={{ height: `${(size / 640) * 100}%` }}
                    />
                    <span className="text-xs mt-1">{size}</span>
                    {i > 0 && i < 5 && (
                      <span className="text-xs text-muted-foreground">/{Math.pow(2, i)}</span>
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};
