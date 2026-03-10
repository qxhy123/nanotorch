/**
 * Head View - Detection head visualization
 */

import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Box, Layers, Target, ArrowRight } from 'lucide-react';
import { BlockMath, InlineMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import type { HeadType } from '../types';

interface DetectionScale {
  name: string;
  gridSize: number;
  stride: number;
  anchors: number;
  objectScale: string;
}

const detectionScales: DetectionScale[] = [
  { name: 'Large', gridSize: 20, stride: 32, anchors: 3, objectScale: 'Large objects' },
  { name: 'Medium', gridSize: 40, stride: 16, anchors: 3, objectScale: 'Medium objects' },
  { name: 'Small', gridSize: 80, stride: 8, anchors: 3, objectScale: 'Small objects' },
];

const headInfo: Record<HeadType, { name: string; description: string; versions: string }> = {
  'anchor-based': {
    name: 'Anchor-Based Detection',
    description: 'Uses predefined anchor boxes as reference for bounding box regression. Each grid cell predicts offsets relative to anchors.',
    versions: 'YOLOv2, v3, v4, v5, v7',
  },
  'anchor-free': {
    name: 'Anchor-Free Detection',
    description: 'Directly predicts bounding box coordinates without anchor priors. Uses point-based regression.',
    versions: 'YOLOv1 (grid), v6, v8',
  },
};

export const HeadView: React.FC = () => {
  const [selectedHead, setSelectedHead] = useState<HeadType>('anchor-free');

  const info = headInfo[selectedHead];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Box className="h-8 w-8 text-primary" />
        <div>
          <h1 className="text-3xl font-bold">Detection Head</h1>
          <p className="text-muted-foreground">Bounding box and class prediction</p>
        </div>
      </div>

      {/* Head Type Selector */}
      <Card>
        <CardHeader>
          <CardTitle>Select Detection Head Type</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2">
            <Button
              variant={selectedHead === 'anchor-based' ? 'default' : 'outline'}
              onClick={() => setSelectedHead('anchor-based')}
            >
              Anchor-Based
            </Button>
            <Button
              variant={selectedHead === 'anchor-free' ? 'default' : 'outline'}
              onClick={() => setSelectedHead('anchor-free')}
            >
              Anchor-Free
            </Button>
          </div>
          <div className="mt-4 p-4 bg-muted rounded-lg">
            <h3 className="font-semibold">{info.name}</h3>
            <p className="text-sm text-muted-foreground mt-1">{info.description}</p>
            <p className="text-xs text-muted-foreground mt-2">Used in: {info.versions}</p>
          </div>
        </CardContent>
      </Card>

      {/* Main Content */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* Output Tensor Visualization */}
        <Card>
          <CardHeader>
            <CardTitle>Output Tensor Structure</CardTitle>
            <CardDescription>Per-scale prediction tensor</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {/* 3D Tensor Visualization */}
              <div className="relative bg-muted rounded-lg p-6">
                <div className="flex items-center justify-center gap-4">
                  {/* Batch */}
                  <div className="text-center">
                    <div className="w-16 h-32 bg-blue-500/30 border-2 border-blue-500 rounded flex items-center justify-center">
                      <span className="text-xs font-medium">B</span>
                    </div>
                    <span className="text-xs mt-1">Batch</span>
                  </div>

                  <span className="text-xl">×</span>

                  {/* Grid */}
                  <div className="text-center">
                    <div className="w-32 h-32 bg-green-500/30 border-2 border-green-500 rounded grid grid-cols-4 gap-0.5 p-1">
                      {Array.from({ length: 16 }).map((_, i) => (
                        <div key={i} className="bg-green-500/50 rounded-sm" />
                      ))}
                    </div>
                    <span className="text-xs mt-1">H × W</span>
                  </div>

                  <span className="text-xl">×</span>

                  {/* Predictions */}
                  <div className="text-center">
                    <div className="w-24 h-32 bg-purple-500/30 border-2 border-purple-500 rounded flex flex-col justify-center p-2 text-xs">
                      <div className="bg-purple-500/50 rounded p-1 mb-1">x, y</div>
                      <div className="bg-purple-500/50 rounded p-1 mb-1">w, h</div>
                      <div className="bg-purple-500/50 rounded p-1 mb-1">conf</div>
                      <div className="bg-purple-500/50 rounded p-1">cls[C]</div>
                    </div>
                    <span className="text-xs mt-1">{selectedHead === 'anchor-based' ? 'A×(5+C)' : '5+C'}</span>
                  </div>
                </div>
              </div>

              {/* Formula */}
              <div className="bg-muted p-4 rounded-lg overflow-x-auto">
                {selectedHead === 'anchor-based' ? (
                  <BlockMath math="\text{Output Shape} = [B, H, W, A \times (5 + C)]" />
                ) : (
                  <BlockMath math="\text{Output Shape} = [B, H, W, 4 + 1 + C]" />
                )}
              </div>

              <div className="grid grid-cols-2 gap-2 text-sm">
                <div className="p-2 bg-blue-500/10 rounded"><InlineMath math="B" /> = Batch size</div>
                <div className="p-2 bg-green-500/10 rounded"><InlineMath math="H, W" /> = Grid dimensions</div>
                {selectedHead === 'anchor-based' && (
                  <div className="p-2 bg-yellow-500/10 rounded"><InlineMath math="A" /> = Anchors per cell</div>
                )}
                <div className="p-2 bg-purple-500/10 rounded"><InlineMath math="C" /> = Number of classes</div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Multi-scale Heads */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Layers className="h-5 w-5" />
              Multi-Scale Detection
            </CardTitle>
            <CardDescription>Three detection heads at different scales</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {detectionScales.map((scale) => (
                <div key={scale.name} className="flex items-center gap-4 p-3 bg-muted rounded-lg">
                  <div className="flex-shrink-0 w-16 h-16 bg-primary/20 rounded flex items-center justify-center">
                    <div
                      className="bg-primary/50 rounded"
                      style={{
                        width: `${(scale.gridSize / 80) * 50}px`,
                        height: `${(scale.gridSize / 80) * 50}px`,
                      }}
                    />
                  </div>
                  <div className="flex-1">
                    <div className="font-medium">{scale.name} Scale Head</div>
                    <div className="text-sm text-muted-foreground">
                      Grid: {scale.gridSize}×{scale.gridSize} | Stride: {scale.stride}
                    </div>
                    <div className="text-xs text-muted-foreground mt-1">
                      → {scale.objectScale}
                    </div>
                  </div>
                  {selectedHead === 'anchor-based' && (
                    <div className="text-sm">
                      <div className="font-medium">{scale.anchors}</div>
                      <div className="text-xs text-muted-foreground">anchors</div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Box Decoding Formulas */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Box Coordinate Decoding</CardTitle>
            <CardDescription>Converting network outputs to bounding box coordinates</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-2 gap-6">
              {/* Anchor-based */}
              <div className={`p-4 rounded-lg ${selectedHead === 'anchor-based' ? 'bg-primary/10 border-2 border-primary' : 'bg-muted'}`}>
                <h4 className="font-semibold mb-3 flex items-center gap-2">
                  <Target className="h-4 w-4" />
                  Anchor-Based Decoding
                </h4>
                <div className="space-y-3 text-sm">
                  <div className="flex items-center gap-2">
                    <InlineMath math="b_x = \sigma(t_x) + c_x" />
                    <ArrowRight className="h-3 w-3 text-muted-foreground" />
                    <span className="text-muted-foreground">Center X</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <InlineMath math="b_y = \sigma(t_y) + c_y" />
                    <ArrowRight className="h-3 w-3 text-muted-foreground" />
                    <span className="text-muted-foreground">Center Y</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <InlineMath math="b_w = p_w \cdot e^{t_w}" />
                    <ArrowRight className="h-3 w-3 text-muted-foreground" />
                    <span className="text-muted-foreground">Width</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <InlineMath math="b_h = p_h \cdot e^{t_h}" />
                    <ArrowRight className="h-3 w-3 text-muted-foreground" />
                    <span className="text-muted-foreground">Height</span>
                  </div>
                </div>
                <div className="mt-3 text-xs text-muted-foreground">
                  <p><InlineMath math="c_x, c_y" /> = grid cell offset</p>
                  <p><InlineMath math="p_w, p_h" /> = anchor dimensions</p>
                </div>
              </div>

              {/* Anchor-free */}
              <div className={`p-4 rounded-lg ${selectedHead === 'anchor-free' ? 'bg-primary/10 border-2 border-primary' : 'bg-muted'}`}>
                <h4 className="font-semibold mb-3 flex items-center gap-2">
                  <Box className="h-4 w-4" />
                  Anchor-Free Decoding
                </h4>
                <div className="space-y-3 text-sm">
                  <div className="flex items-center gap-2">
                    <InlineMath math="b_x = (c_x + 0.5 + t_x) \cdot s" />
                    <ArrowRight className="h-3 w-3 text-muted-foreground" />
                    <span className="text-muted-foreground">Center X</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <InlineMath math="b_y = (c_y + 0.5 + t_y) \cdot s" />
                    <ArrowRight className="h-3 w-3 text-muted-foreground" />
                    <span className="text-muted-foreground">Center Y</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <InlineMath math="b_w = e^{t_w} \cdot s" />
                    <ArrowRight className="h-3 w-3 text-muted-foreground" />
                    <span className="text-muted-foreground">Width</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <InlineMath math="b_h = e^{t_h} \cdot s" />
                    <ArrowRight className="h-3 w-3 text-muted-foreground" />
                    <span className="text-muted-foreground">Height</span>
                  </div>
                </div>
                <div className="mt-3 text-xs text-muted-foreground">
                  <p><InlineMath math="s" /> = stride of detection head</p>
                  <p>Direct regression without anchor priors</p>
                </div>
              </div>
            </div>

            {/* Confidence & Classification */}
            <div className="mt-6 p-4 bg-muted rounded-lg">
              <h4 className="font-semibold mb-3">Confidence & Classification</h4>
              <div className="grid md:grid-cols-2 gap-4 text-sm">
                <div>
                  <InlineMath math="\text{Objectness} = \sigma(t_{obj})" />
                  <p className="text-muted-foreground mt-1">Probability that an object exists</p>
                </div>
                <div>
                  <InlineMath math="\text{Class} = \text{softmax}(t_{cls})" />
                  <p className="text-muted-foreground mt-1">Probability distribution over classes</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};
