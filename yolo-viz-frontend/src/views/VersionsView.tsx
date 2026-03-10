/**
 * Versions View - YOLO version comparison and timeline
 */

import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../components/ui/card';
import { GitBranch, Check, ArrowRight } from 'lucide-react';
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
  Legend,
  Tooltip,
} from 'recharts';
import type { YOLOVersion, YOLOVersionInfo } from '../types';

const versions: YOLOVersionInfo[] = [
  {
    version: 'v1',
    year: 2015,
    backbone: 'Custom (24 Conv)',
    mAP: 63.4,
    fps: 45,
    params: '45M',
    anchorType: 'anchor-free',
    features: ['Single scale', 'Grid-based detection', '7x7 grid'],
  },
  {
    version: 'v2',
    year: 2016,
    backbone: 'Darknet-19',
    mAP: 78.6,
    fps: 40,
    params: '50M',
    anchorType: 'anchor-based',
    features: ['Anchor boxes', 'Batch normalization', 'Multi-scale training'],
  },
  {
    version: 'v3',
    year: 2018,
    backbone: 'Darknet-53',
    mAP: 33.0,
    fps: 30,
    params: '62M',
    anchorType: 'anchor-based',
    features: ['Multi-scale detection', 'FPN neck', 'Residual connections'],
  },
  {
    version: 'v4',
    year: 2020,
    backbone: 'CSPDarknet53',
    mAP: 43.5,
    fps: 65,
    params: '64M',
    anchorType: 'anchor-based',
    features: ['CSP backbone', 'PANet', 'Mosaic augmentation', 'Self-adversarial training'],
  },
  {
    version: 'v5',
    year: 2020,
    backbone: 'CSPDarknet',
    mAP: 50.7,
    fps: 140,
    params: '7-86M',
    anchorType: 'anchor-based',
    features: ['Model scaling', 'Auto-anchor', 'Multiple model sizes (n/s/m/l/x)'],
  },
  {
    version: 'v6',
    year: 2022,
    backbone: 'EfficientRep',
    mAP: 52.5,
    fps: 180,
    params: '5-140M',
    anchorType: 'anchor-free',
    features: ['Reparameterization', 'Decoupled head', 'TAL assigner'],
  },
  {
    version: 'v7',
    year: 2022,
    backbone: 'E-ELAN',
    mAP: 56.8,
    fps: 120,
    params: '36M',
    anchorType: 'anchor-based',
    features: ['Extended ELAN', 'Compound scaling', 'Auxiliary head'],
  },
  {
    version: 'v8',
    year: 2023,
    backbone: 'CSPDarknet (C2f)',
    mAP: 53.9,
    fps: 280,
    params: '3-68M',
    anchorType: 'anchor-free',
    features: ['C2f module', 'Anchor-free head', 'Unified API', 'SOTA performance'],
  },
];

const radarData = [
  { metric: 'mAP', v5: 50.7, v7: 56.8, v8: 53.9, fullMark: 60 },
  { metric: 'Speed', v5: 70, v7: 60, v8: 100, fullMark: 100 },
  { metric: 'Params', v5: 30, v7: 36, v8: 25, fullMark: 100 },
  { metric: 'Ease of Use', v5: 90, v7: 70, v8: 95, fullMark: 100 },
  { metric: 'Accuracy', v5: 85, v7: 95, v8: 90, fullMark: 100 },
];

export const VersionsView: React.FC = () => {
  const [selectedVersions, setSelectedVersions] = useState<YOLOVersion[]>(['v5', 'v7', 'v8']);

  const toggleVersion = (version: YOLOVersion) => {
    if (selectedVersions.includes(version)) {
      if (selectedVersions.length > 1) {
        setSelectedVersions(selectedVersions.filter(v => v !== version));
      }
    } else {
      setSelectedVersions([...selectedVersions, version]);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <GitBranch className="h-8 w-8 text-primary" />
        <div>
          <h1 className="text-3xl font-bold">YOLO Versions</h1>
          <p className="text-muted-foreground">Evolution of the YOLO architecture (2015-2023)</p>
        </div>
      </div>

      {/* Timeline */}
      <Card>
        <CardHeader>
          <CardTitle>Timeline</CardTitle>
          <CardDescription>Click to select versions for comparison</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="relative">
            {/* Timeline line */}
            <div className="absolute top-8 left-0 right-0 h-1 bg-muted" />

            {/* Version markers */}
            <div className="flex justify-between relative">
              {versions.map((v) => {
                const isSelected = selectedVersions.includes(v.version);
                return (
                  <div
                    key={v.version}
                    className="flex flex-col items-center cursor-pointer group"
                    onClick={() => toggleVersion(v.version)}
                  >
                    <div
                      className={`w-16 h-16 rounded-full flex items-center justify-center text-sm font-bold transition-all ${
                        isSelected
                          ? 'bg-primary text-primary-foreground scale-110'
                          : 'bg-muted text-muted-foreground hover:bg-accent'
                      }`}
                    >
                      {v.version.toUpperCase()}
                    </div>
                    <div className="mt-2 text-center">
                      <div className="text-xs font-medium">{v.year}</div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Comparison Table */}
      <Card>
        <CardHeader>
          <CardTitle>Version Comparison</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="text-left py-3 px-4">Version</th>
                  <th className="text-left py-3 px-4">Year</th>
                  <th className="text-left py-3 px-4">Backbone</th>
                  <th className="text-left py-3 px-4">Anchor Type</th>
                  <th className="text-right py-3 px-4">mAP</th>
                  <th className="text-right py-3 px-4">FPS</th>
                  <th className="text-right py-3 px-4">Params</th>
                </tr>
              </thead>
              <tbody>
                {versions
                  .filter(v => selectedVersions.includes(v.version))
                  .map((v) => (
                    <tr key={v.version} className="border-b hover:bg-muted/50">
                      <td className="py-3 px-4">
                        <span className="font-bold text-primary">YOLO{v.version}</span>
                      </td>
                      <td className="py-3 px-4">{v.year}</td>
                      <td className="py-3 px-4">{v.backbone}</td>
                      <td className="py-3 px-4">
                        <span className={`px-2 py-1 rounded text-xs ${
                          v.anchorType === 'anchor-free'
                            ? 'bg-green-500/10 text-green-600'
                            : 'bg-blue-500/10 text-blue-600'
                        }`}>
                          {v.anchorType}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-right font-medium">{v.mAP}%</td>
                      <td className="py-3 px-4 text-right font-medium">{v.fps}</td>
                      <td className="py-3 px-4 text-right">{v.params}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Radar Chart & Features */}
      <div className="grid lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Performance Comparison</CardTitle>
            <CardDescription>YOLOv5 vs v7 vs v8</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={radarData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="metric" />
                <PolarRadiusAxis angle={30} domain={[0, 100]} />
                <Radar name="YOLOv5" dataKey="v5" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
                <Radar name="YOLOv7" dataKey="v7" stroke="#22c55e" fill="#22c55e" fillOpacity={0.3} />
                <Radar name="YOLOv8" dataKey="v8" stroke="#ef4444" fill="#ef4444" fillOpacity={0.3} />
                <Legend />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Key Features by Version</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4 max-h-[300px] overflow-y-auto">
              {versions
                .filter(v => selectedVersions.includes(v.version))
                .map((v) => (
                  <div key={v.version} className="p-3 bg-muted rounded-lg">
                    <div className="font-semibold text-primary mb-2">YOLO{v.version}</div>
                    <ul className="space-y-1">
                      {v.features.map((feature, i) => (
                        <li key={i} className="flex items-center gap-2 text-sm">
                          <Check className="h-4 w-4 text-green-500 shrink-0" />
                          {feature}
                        </li>
                      ))}
                    </ul>
                  </div>
                ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Evolution Highlights */}
      <Card>
        <CardHeader>
          <CardTitle>Evolution Highlights</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="p-4 bg-blue-500/10 rounded-lg border border-blue-500/20">
              <h4 className="font-semibold text-blue-600 dark:text-blue-400">v1 → v2</h4>
              <p className="text-sm text-muted-foreground mt-1">
                Introduced anchor boxes, enabling better bounding box predictions
              </p>
            </div>
            <div className="p-4 bg-green-500/10 rounded-lg border border-green-500/20">
              <h4 className="font-semibold text-green-600 dark:text-green-400">v3 → v4</h4>
              <p className="text-sm text-muted-foreground mt-1">
                CSP backbone and PANet for improved feature aggregation
              </p>
            </div>
            <div className="p-4 bg-yellow-500/10 rounded-lg border border-yellow-500/20">
              <h4 className="font-semibold text-yellow-600 dark:text-yellow-400">v5 → v6</h4>
              <p className="text-sm text-muted-foreground mt-1">
                Return to anchor-free with reparameterization techniques
              </p>
            </div>
            <div className="p-4 bg-red-500/10 rounded-lg border border-red-500/20">
              <h4 className="font-semibold text-red-600 dark:text-red-400">v7 → v8</h4>
              <p className="text-sm text-muted-foreground mt-1">
                C2f module, decoupled head, unified API for all tasks
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Recommendation */}
      <Card className="bg-primary/5 border-primary/20">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <ArrowRight className="h-5 w-5" />
            Which Version to Use?
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-3 gap-4 text-sm">
            <div>
              <h4 className="font-semibold">For Production</h4>
              <p className="text-muted-foreground">YOLOv8 - Best balance of speed, accuracy, and ease of use</p>
            </div>
            <div>
              <h4 className="font-semibold">For Maximum Accuracy</h4>
              <p className="text-muted-foreground">YOLOv7 - Highest mAP on standard benchmarks</p>
            </div>
            <div>
              <h4 className="font-semibold">For Legacy Projects</h4>
              <p className="text-muted-foreground">YOLOv5 - Mature ecosystem and extensive documentation</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
