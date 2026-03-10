/**
 * Neck View - FPN/PANet/BiFPN visualization
 */

import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { GitMerge, ArrowUp, ArrowDown, ArrowRight, Plus } from 'lucide-react';
import { BlockMath, InlineMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import type { NeckType } from '../types';

interface FeatureLevel {
  name: string;
  size: number;
  channels: number;
  scale: number;
}

const featureLevels: FeatureLevel[] = [
  { name: 'P3', size: 80, channels: 256, scale: 8 },
  { name: 'P4', size: 40, channels: 256, scale: 16 },
  { name: 'P5', size: 20, channels: 256, scale: 32 },
];

const neckInfo: Record<NeckType, { name: string; description: string; papers: string }> = {
  fpn: {
    name: 'Feature Pyramid Network (FPN)',
    description: 'Top-down pathway with lateral connections to build high-level semantic feature maps at all scales.',
    papers: 'Lin et al., 2017',
  },
  panet: {
    name: 'Path Aggregation Network (PANet)',
    description: 'Enhances FPN with bottom-up path augmentation for better information flow.',
    papers: 'Liu et al., 2018',
  },
  bifpn: {
    name: 'Bi-directional FPN (BiFPN)',
    description: 'Weighted bi-directional feature pyramid with learned feature fusion weights.',
    papers: 'Tan et al., 2020',
  },
};

export const NeckView: React.FC = () => {
  const [selectedNeck, setSelectedNeck] = useState<NeckType>('panet');

  const info = neckInfo[selectedNeck];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <GitMerge className="h-8 w-8 text-primary" />
        <div>
          <h1 className="text-3xl font-bold">Neck (Feature Pyramid)</h1>
          <p className="text-muted-foreground">Multi-scale feature fusion network</p>
        </div>
      </div>

      {/* Neck Selector */}
      <Card>
        <CardHeader>
          <CardTitle>Select Neck Architecture</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            <Button
              variant={selectedNeck === 'fpn' ? 'default' : 'outline'}
              onClick={() => setSelectedNeck('fpn')}
            >
              FPN
            </Button>
            <Button
              variant={selectedNeck === 'panet' ? 'default' : 'outline'}
              onClick={() => setSelectedNeck('panet')}
            >
              PANet
            </Button>
            <Button
              variant={selectedNeck === 'bifpn' ? 'default' : 'outline'}
              onClick={() => setSelectedNeck('bifpn')}
            >
              BiFPN
            </Button>
          </div>
          <div className="mt-4 p-4 bg-muted rounded-lg">
            <h3 className="font-semibold">{info.name}</h3>
            <p className="text-sm text-muted-foreground mt-1">{info.description}</p>
            <p className="text-xs text-muted-foreground mt-2">Paper: {info.papers}</p>
          </div>
        </CardContent>
      </Card>

      {/* Visualization */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* Pyramid Diagram */}
        <Card>
          <CardHeader>
            <CardTitle>Feature Pyramid Structure</CardTitle>
            <CardDescription>Multi-scale feature aggregation</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="relative p-8">
              {/* Backbone Features (Left) */}
              <div className="absolute left-4 top-0 bottom-0 flex flex-col justify-around">
                <div className="text-xs text-muted-foreground">Backbone</div>
                {featureLevels.map((level) => (
                  <div
                    key={`backbone-${level.name}`}
                    className="bg-blue-500/20 border-2 border-blue-500 rounded p-2 text-center"
                    style={{ width: `${level.size}px`, minWidth: '60px' }}
                  >
                    <div className="text-xs font-medium">C{level.name.slice(1)}</div>
                    <div className="text-xs text-muted-foreground">{level.size}x{level.size}</div>
                  </div>
                ))}
              </div>

              {/* Output Features (Right) */}
              <div className="absolute right-4 top-0 bottom-0 flex flex-col justify-around">
                <div className="text-xs text-muted-foreground">Output</div>
                {featureLevels.map((level) => (
                  <div
                    key={`output-${level.name}`}
                    className="bg-purple-500/20 border-2 border-purple-500 rounded p-2 text-center"
                    style={{ width: `${level.size}px`, minWidth: '60px' }}
                  >
                    <div className="text-xs font-medium">{level.name}</div>
                    <div className="text-xs text-muted-foreground">{level.size}x{level.size}</div>
                  </div>
                ))}
              </div>

              {/* Center - Connection Visualization */}
              <div className="flex flex-col items-center justify-center h-64 mx-24">
                {selectedNeck === 'fpn' && (
                  <div className="space-y-4 text-center">
                    <div className="text-sm font-medium">Top-Down Pathway</div>
                    <ArrowDown className="mx-auto h-8 w-8 text-primary" />
                    <div className="flex items-center gap-2 justify-center">
                      <div className="w-2 h-2 bg-blue-500 rounded-full" />
                      <Plus className="h-4 w-4" />
                      <div className="w-2 h-2 bg-purple-500 rounded-full" />
                    </div>
                    <div className="text-xs text-muted-foreground">Upsample + Lateral</div>
                  </div>
                )}
                {selectedNeck === 'panet' && (
                  <div className="space-y-2 text-center">
                    <div className="text-sm font-medium">Bi-directional</div>
                    <div className="flex flex-col items-center">
                      <ArrowDown className="h-6 w-6 text-blue-500" />
                      <span className="text-xs">Top-Down</span>
                    </div>
                    <div className="flex flex-col items-center mt-4">
                      <ArrowUp className="h-6 w-6 text-green-500" />
                      <span className="text-xs">Bottom-Up</span>
                    </div>
                  </div>
                )}
                {selectedNeck === 'bifpn' && (
                  <div className="space-y-2 text-center">
                    <div className="text-sm font-medium">Weighted Fusion</div>
                    <div className="flex gap-4">
                      <div className="flex flex-col items-center">
                        <ArrowDown className="h-5 w-5 text-blue-500" />
                        <ArrowUp className="h-5 w-5 text-green-500" />
                      </div>
                    </div>
                    <div className="text-xs text-muted-foreground mt-2">
                      <InlineMath math="w_i \cdot F_i" />
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Scale Legend */}
            <div className="flex justify-center gap-4 mt-4 text-sm">
              {featureLevels.map((level) => (
                <div key={level.name} className="flex items-center gap-1">
                  <div className="w-3 h-3 rounded bg-primary/50" />
                  <span>{level.name}: 1/{level.scale}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Formulas and Details */}
        <div className="space-y-6">
          {/* Feature Fusion Formula */}
          <Card>
            <CardHeader>
              <CardTitle>Feature Fusion Formula</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {selectedNeck === 'fpn' && (
                <>
                  <div className="bg-muted p-4 rounded-lg overflow-x-auto">
                    <BlockMath math="P_i = \text{Conv}(\text{Upsample}(P_{i+1}) + \text{Lateral}(C_i))" />
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Top-down pathway combines upsampled high-level features with lateral connections from backbone.
                  </p>
                </>
              )}
              {selectedNeck === 'panet' && (
                <>
                  <div className="bg-muted p-4 rounded-lg overflow-x-auto space-y-2">
                    <BlockMath math="P_i^{td} = \text{Conv}(\text{Upsample}(P_{i+1}^{td}) + C_i)" />
                    <BlockMath math="N_i = \text{Conv}(\text{Downsample}(N_{i-1}) + P_i^{td})" />
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Adds bottom-up path augmentation after the standard top-down pathway.
                  </p>
                </>
              )}
              {selectedNeck === 'bifpn' && (
                <>
                  <div className="bg-muted p-4 rounded-lg overflow-x-auto">
                    <BlockMath math="P_i^{out} = \text{Conv}\left(\frac{w_1 \cdot P_i^{in} + w_2 \cdot P_i^{td}}{w_1 + w_2 + \epsilon}\right)" />
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Uses learned weights for feature fusion with fast normalized fusion.
                  </p>
                </>
              )}
            </CardContent>
          </Card>

          {/* Feature Levels Table */}
          <Card>
            <CardHeader>
              <CardTitle>Feature Levels</CardTitle>
            </CardHeader>
            <CardContent>
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left py-2">Level</th>
                    <th className="text-left py-2">Size</th>
                    <th className="text-left py-2">Stride</th>
                    <th className="text-left py-2">Object Scale</th>
                  </tr>
                </thead>
                <tbody>
                  {featureLevels.map((level) => (
                    <tr key={level.name} className="border-b">
                      <td className="py-2 font-medium">{level.name}</td>
                      <td className="py-2">{level.size}x{level.size}</td>
                      <td className="py-2">{level.scale}</td>
                      <td className="py-2">
                        {level.scale === 8 ? 'Small' : level.scale === 16 ? 'Medium' : 'Large'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </CardContent>
          </Card>

          {/* Why Multi-scale */}
          <Card className="bg-primary/5 border-primary/20">
            <CardHeader>
              <CardTitle className="text-lg">Why Multi-Scale?</CardTitle>
            </CardHeader>
            <CardContent className="text-sm space-y-2">
              <div className="flex items-start gap-2">
                <ArrowRight className="h-4 w-4 mt-0.5 text-primary shrink-0" />
                <span><strong>P3 (80x80):</strong> Detects small objects with fine spatial resolution</span>
              </div>
              <div className="flex items-start gap-2">
                <ArrowRight className="h-4 w-4 mt-0.5 text-primary shrink-0" />
                <span><strong>P4 (40x40):</strong> Detects medium objects with balanced features</span>
              </div>
              <div className="flex items-start gap-2">
                <ArrowRight className="h-4 w-4 mt-0.5 text-primary shrink-0" />
                <span><strong>P5 (20x20):</strong> Detects large objects with rich semantic information</span>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};
