/**
 * GradientFlow Component
 *
 * Visualization showing how gradients flow through the network during backpropagation.
 * Includes gradient magnitude heatmaps, vanishing/exploding gradient detection,
 * and gradient flow comparison between architectures.
 */

import React, { useState, useCallback, useEffect, useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../ui/card';
import { Badge } from '../../ui/badge';
import { Button } from '../../ui/button';
import { Slider } from '../../ui/slider';
import { motion, AnimatePresence } from 'framer-motion';
import {
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Activity,
  Layers,
  GitCompare,
  Info,
  Zap,
  RefreshCw,
  type LucideIcon,
} from 'lucide-react';

// Types
interface LayerGradientData {
  layerName: string;
  layerIndex: number;
  gradientNorm: number;
  meanGradient: number;
  maxGradient: number;
  minGradient: number;
  stdGradient: number;
  gradients: number[];
  parameterCount: number;
}

interface GradientIssue {
  type: 'vanishing' | 'exploding' | 'healthy' | 'unstable';
  layerIndex: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  threshold: number;
}

interface ArchitectureComparison {
  architecture: 'pre-norm' | 'post-norm' | 'resnet' | 'dense';
  gradientNorms: number[];
  meanFlow: number;
  stability: number;
  color: string;
}

interface GradientFlowProps {
  numLayers?: number;
  className?: string;
}

// Mock gradient data generator
function generateMockGradients(
  numLayers: number,
  architecture: 'pre-norm' | 'post-norm' = 'pre-norm'
): LayerGradientData[] {
  const layers: LayerGradientData[] = [];

  for (let i = 0; i < numLayers; i++) {
    // Gradients tend to be smaller in deeper layers (vanishing)
    // Pre-norm architecture helps mitigate this
    const depthFactor = architecture === 'pre-norm' ? 0.85 : 0.7;
    const baseNorm = 1.0 * Math.pow(depthFactor, i);

    // Add some noise
    const noise = () => (Math.random() - 0.5) * 0.3;

    const gradients: number[] = [];
    for (let j = 0; j < 100; j++) {
      gradients.push(baseNorm + noise());
    }

    const mean = gradients.reduce((sum, g) => sum + g, 0) / gradients.length;
    const variance = gradients.reduce((sum, g) => sum + Math.pow(g - mean, 2), 0) / gradients.length;
    const max = Math.max(...gradients);
    const min = Math.min(...gradients);

    layers.push({
      layerName: i === 0 ? 'Output Embedding' : `Encoder Layer ${i}`,
      layerIndex: i,
      gradientNorm: baseNorm,
      meanGradient: mean,
      maxGradient: max,
      minGradient: min,
      stdGradient: Math.sqrt(variance),
      gradients,
      parameterCount: Math.floor(100000 + Math.random() * 500000),
    });
  }

  return layers;
}

function detectGradientIssues(gradients: LayerGradientData[]): GradientIssue[] {
  const issues: GradientIssue[] = [];

  gradients.forEach((layer) => {
    // Vanishing gradient: norm < 0.01
    if (layer.gradientNorm < 0.01) {
      issues.push({
        type: 'vanishing',
        layerIndex: layer.layerIndex,
        severity: layer.gradientNorm < 0.001 ? 'critical' : layer.gradientNorm < 0.005 ? 'high' : 'medium',
        description: `Gradient norm is very small (${layer.gradientNorm.toExponential(2)})`,
        threshold: 0.01,
      });
    }
    // Exploding gradient: norm > 10
    else if (layer.gradientNorm > 10) {
      issues.push({
        type: 'exploding',
        layerIndex: layer.layerIndex,
        severity: layer.gradientNorm > 100 ? 'critical' : layer.gradientNorm > 50 ? 'high' : 'medium',
        description: `Gradient norm is very large (${layer.gradientNorm.toFixed(2)})`,
        threshold: 10,
      });
    }
    // Unstable: high variance relative to mean
    else if (layer.stdGradient / Math.abs(layer.meanGradient) > 5) {
      issues.push({
        type: 'unstable',
        layerIndex: layer.layerIndex,
        severity: 'medium',
        description: `High variance in gradients (std/mean = ${(layer.stdGradient / Math.abs(layer.meanGradient)).toFixed(2)})`,
        threshold: 5,
      });
    }
    // Healthy
    else {
      issues.push({
        type: 'healthy',
        layerIndex: layer.layerIndex,
        severity: 'low',
        description: 'Gradient flow is normal',
        threshold: 0,
      });
    }
  });

  return issues;
}

function generateComparisonData(numLayers: number): ArchitectureComparison[] {
  return [
    {
      architecture: 'pre-norm',
      gradientNorms: Array.from({ length: numLayers }, (_, i) => 1.0 * Math.pow(0.85, i)),
      meanFlow: 0.92,
      stability: 0.95,
      color: '#22c55e',
    },
    {
      architecture: 'post-norm',
      gradientNorms: Array.from({ length: numLayers }, (_, i) => 1.0 * Math.pow(0.7, i)),
      meanFlow: 0.78,
      stability: 0.82,
      color: '#3b82f6',
    },
    {
      architecture: 'resnet',
      gradientNorms: Array.from({ length: numLayers }, (_, i) => 1.0 * Math.pow(0.9, i)),
      meanFlow: 0.88,
      stability: 0.90,
      color: '#f59e0b',
    },
    {
      architecture: 'dense',
      gradientNorms: Array.from({ length: numLayers }, (_, i) => 1.0 * Math.pow(0.6, i)),
      meanFlow: 0.70,
      stability: 0.75,
      color: '#ef4444',
    },
  ];
}

const ISSUE_COLORS: Record<string, { bg: string; text: string; border: string; icon: LucideIcon }> = {
  vanishing: {
    bg: 'bg-yellow-50 dark:bg-yellow-900/20',
    text: 'text-yellow-800 dark:text-yellow-200',
    border: 'border-yellow-200 dark:border-yellow-800',
    icon: AlertTriangle,
  },
  exploding: {
    bg: 'bg-red-50 dark:bg-red-900/20',
    text: 'text-red-800 dark:text-red-200',
    border: 'border-red-200 dark:border-red-800',
    icon: XCircle,
  },
  healthy: {
    bg: 'bg-green-50 dark:bg-green-900/20',
    text: 'text-green-800 dark:text-green-200',
    border: 'border-green-200 dark:border-green-800',
    icon: CheckCircle,
  },
  unstable: {
    bg: 'bg-orange-50 dark:bg-orange-900/20',
    text: 'text-orange-800 dark:text-orange-200',
    border: 'border-orange-200 dark:border-orange-800',
    icon: Zap,
  },
};

export const GradientFlow: React.FC<GradientFlowProps> = ({
  numLayers = 12,
  className = '',
}) => {
  // State
  const [gradientData, setGradientData] = useState<LayerGradientData[]>([]);
  const [issues, setIssues] = useState<GradientIssue[]>([]);
  const [comparisonData, setComparisonData] = useState<ArchitectureComparison[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedArchitecture, setSelectedArchitecture] = useState<'pre-norm' | 'post-norm'>('pre-norm');
  const [viewMode, setViewMode] = useState<'flow' | 'layers' | 'comparison'>('flow');
  const [vanishingThreshold, setVanishingThreshold] = useState(0.01);
  const [explodingThreshold, setExplodingThreshold] = useState(10);

  // Load data
  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 500));

      // Generate gradient data
      const gradients = generateMockGradients(numLayers, selectedArchitecture);
      setGradientData(gradients);

      // Detect issues
      const detectedIssues = detectGradientIssues(gradients);
      setIssues(detectedIssues);

      // Generate comparison data
      const comparison = generateComparisonData(numLayers);
      setComparisonData(comparison);
    } catch (error) {
      console.error('Failed to load gradient data:', error);
    } finally {
      setLoading(false);
    }
  }, [numLayers, selectedArchitecture]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Overall health summary
  const healthSummary = useMemo(() => {
    const critical = issues.filter(i => i.severity === 'critical').length;
    const high = issues.filter(i => i.severity === 'high').length;
    const medium = issues.filter(i => i.severity === 'medium').length;
    const healthy = issues.filter(i => i.type === 'healthy').length;

    return { critical, high, medium, healthy };
  }, [issues]);

  if (loading) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center py-12">
          <RefreshCw className="h-8 w-8 animate-spin text-primary" />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <TrendingDown className="h-5 w-5 text-primary" />
              Gradient Flow Visualization
            </CardTitle>
            <CardDescription>
              Monitor gradient flow during backpropagation
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={loadData}>
              <RefreshCw className="h-4 w-4 mr-1" />
              Refresh
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Health Summary */}
        <div className="grid grid-cols-4 gap-4">
          <div className={`p-3 rounded-lg border ${ISSUE_COLORS.healthy.bg} ${ISSUE_COLORS.healthy.border}`}>
            <div className="flex items-center gap-2 mb-1">
              <CheckCircle className={`h-4 w-4 ${ISSUE_COLORS.healthy.text}`} />
              <div className="text-xs text-gray-600 dark:text-gray-400">Healthy</div>
            </div>
            <div className="text-lg font-bold">{healthSummary.healthy}</div>
          </div>
          <div className={`p-3 rounded-lg border ${ISSUE_COLORS.unstable.bg} ${ISSUE_COLORS.unstable.border}`}>
            <div className="flex items-center gap-2 mb-1">
              <Zap className={`h-4 w-4 ${ISSUE_COLORS.unstable.text}`} />
              <div className="text-xs text-gray-600 dark:text-gray-400">Unstable</div>
            </div>
            <div className="text-lg font-bold">{healthSummary.medium}</div>
          </div>
          <div className={`p-3 rounded-lg border ${ISSUE_COLORS.vanishing.bg} ${ISSUE_COLORS.vanishing.border}`}>
            <div className="flex items-center gap-2 mb-1">
              <AlertTriangle className={`h-4 w-4 ${ISSUE_COLORS.vanishing.text}`} />
              <div className="text-xs text-gray-600 dark:text-gray-400">Vanishing</div>
            </div>
            <div className="text-lg font-bold">{healthSummary.high}</div>
          </div>
          <div className={`p-3 rounded-lg border ${ISSUE_COLORS.exploding.bg} ${ISSUE_COLORS.exploding.border}`}>
            <div className="flex items-center gap-2 mb-1">
              <XCircle className={`h-4 w-4 ${ISSUE_COLORS.exploding.text}`} />
              <div className="text-xs text-gray-600 dark:text-gray-400">Exploding</div>
            </div>
            <div className="text-lg font-bold">{healthSummary.critical}</div>
          </div>
        </div>

        {/* View Mode Selector */}
        <div className="flex gap-2">
          <Button
            variant={viewMode === 'flow' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setViewMode('flow')}
          >
            <Activity className="h-4 w-4 mr-1" />
            Flow View
          </Button>
          <Button
            variant={viewMode === 'layers' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setViewMode('layers')}
          >
            <Layers className="h-4 w-4 mr-1" />
            Layer Details
          </Button>
          <Button
            variant={viewMode === 'comparison' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setViewMode('comparison')}
          >
            <GitCompare className="h-4 w-4 mr-1" />
            Architecture Compare
          </Button>
        </div>

        {/* Architecture Selector */}
        <div className="flex items-center gap-4">
          <span className="text-sm text-gray-600 dark:text-gray-400">Architecture:</span>
          <div className="flex gap-2">
            <Button
              variant={selectedArchitecture === 'pre-norm' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setSelectedArchitecture('pre-norm')}
            >
              Pre-Norm
            </Button>
            <Button
              variant={selectedArchitecture === 'post-norm' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setSelectedArchitecture('post-norm')}
            >
              Post-Norm
            </Button>
          </div>
        </div>

        {viewMode === 'flow' && (
          <>
            {/* Gradient Flow Chart */}
            <div className="space-y-4">
              <h3 className="text-sm font-medium">Gradient Norm Flow Through Layers</h3>

              <div className="relative h-64 bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
                {/* Y-axis */}
                <div className="absolute left-4 top-4 bottom-12 w-12 flex flex-col justify-between text-xs text-gray-500">
                  <span>10</span>
                  <span>1</span>
                  <span>0.1</span>
                  <span>0.01</span>
                </div>

                {/* Chart area */}
                <div className="ml-16 h-full">
                  <svg width="100%" height="100%" viewBox="0 0 400 200" preserveAspectRatio="none">
                    {/* Grid lines */}
                    {[0.01, 0.1, 1, 10].map((val, idx) => {
                      const y = 200 - (Math.log10(val) + 2) / 4 * 200;
                      return (
                        <line
                          key={idx}
                          x1="0" y1={y} x2="400" y2={y}
                          stroke="currentColor"
                          strokeWidth="0.5"
                          className="text-gray-300 dark:text-gray-600"
                        />
                      );
                    })}

                    {/* Gradient flow line */}
                    <motion.polyline
                      initial={{ pathLength: 0 }}
                      animate={{ pathLength: 1 }}
                      transition={{ duration: 1 }}
                      fill="none"
                      stroke="#3b82f6"
                      strokeWidth="2"
                      points={gradientData.map((d, i) => {
                        const x = (i / (gradientData.length - 1)) * 400;
                        const y = Math.max(0, Math.min(200, 200 - (Math.log10(Math.max(d.gradientNorm, 0.001)) + 2) / 4 * 200));
                    return `${x},${y}`;
                  }).join(' ')}
                    />

                    {/* Layer markers */}
                    {gradientData.map((d, i) => {
                      const x = (i / (gradientData.length - 1)) * 400;
                      const y = Math.max(0, Math.min(200, 200 - (Math.log10(Math.max(d.gradientNorm, 0.001)) + 2) / 4 * 200));
                      const issue = issues.find(i => i.layerIndex === d.layerIndex);
                      const color = issue ? ISSUE_COLORS[issue.type].text.replace('text-', '') : '#3b82f6';

                      return (
                        <g key={i}>
                          <circle cx={x} cy={y} r="4" fill={color} />
                          <text
                            x={x}
                            y={y - 10}
                            textAnchor="middle"
                            className="text-xs fill-gray-600 dark:fill-gray-400"
                          >
                            {d.layerIndex}
                          </text>
                        </g>
                      );
                    })}
                  </svg>

                  {/* X-axis */}
                  <div className="absolute left-16 right-4 bottom-4 flex justify-between text-xs text-gray-500">
                    <span>Output</span>
                    <span>Layer 6</span>
                    <span>Layer 12</span>
                    <span>Input</span>
                  </div>
                </div>
              </div>

              {/* Threshold Controls */}
              <div className="grid grid-cols-2 gap-6">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <label className="text-xs text-gray-600 dark:text-gray-400">
                      Vanishing Threshold
                    </label>
                    <Badge variant="outline">{vanishingThreshold.toExponential(0)}</Badge>
                  </div>
                  <Slider
                    value={[Math.log10(vanishingThreshold)]}
                    onValueChange={([val]) => setVanishingThreshold(Math.pow(10, val))}
                    min={-4}
                    max={0}
                    step={0.1}
                    className="w-full"
                  />
                </div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <label className="text-xs text-gray-600 dark:text-gray-400">
                      Exploding Threshold
                    </label>
                    <Badge variant="outline">{explodingThreshold.toFixed(0)}</Badge>
                  </div>
                  <Slider
                    value={[Math.log10(explodingThreshold)]}
                    onValueChange={([val]) => setExplodingThreshold(Math.pow(10, val))}
                    min={0}
                    max={2}
                    step={0.1}
                    className="w-full"
                  />
                </div>
              </div>
            </div>
          </>
        )}

        {viewMode === 'layers' && (
          <>
            {/* Layer-wise Details */}
            <div className="space-y-3">
              <h3 className="text-sm font-medium">Layer-wise Gradient Statistics</h3>

              <div className="space-y-2">
                <AnimatePresence>
                  {gradientData.map((layer, idx) => {
                    const issue = issues.find(i => i.layerIndex === layer.layerIndex);
                    const colors = issue ? ISSUE_COLORS[issue.type] : ISSUE_COLORS.healthy;
                    const Icon = colors.icon;

                    return (
                      <motion.div
                        key={layer.layerIndex}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 20 }}
                        transition={{ delay: idx * 0.03 }}
                        className={`p-4 rounded-lg border ${colors.bg} ${colors.border}`}
                      >
                        <div className="flex items-center gap-4">
                          {/* Layer Info */}
                          <div className="flex-1">
                            <div className="flex items-center gap-2 mb-1">
                              <h4 className="font-medium">{layer.layerName}</h4>
                              <Icon className={`h-4 w-4 ${colors.text}`} />
                            </div>
                            {issue && issue.type !== 'healthy' && (
                              <p className="text-sm text-gray-700 dark:text-gray-300">{issue.description}</p>
                            )}
                          </div>

                          {/* Statistics */}
                          <div className="grid grid-cols-4 gap-4 text-right text-sm">
                            <div>
                              <div className="text-xs text-gray-500">Norm</div>
                              <div className="font-mono font-medium">{layer.gradientNorm.toExponential(2)}</div>
                            </div>
                            <div>
                              <div className="text-xs text-gray-500">Mean</div>
                              <div className="font-mono">{layer.meanGradient.toFixed(4)}</div>
                            </div>
                            <div>
                              <div className="text-xs text-gray-500">Std</div>
                              <div className="font-mono">{layer.stdGradient.toFixed(4)}</div>
                            </div>
                            <div>
                              <div className="text-xs text-gray-500">Params</div>
                              <div className="font-mono">{(layer.parameterCount / 1000).toFixed(0)}K</div>
                            </div>
                          </div>
                        </div>

                        {/* Mini gradient bar */}
                        <div className="mt-3 h-2 bg-gray-200 dark:bg-gray-700 rounded overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-blue-500 via-purple-500 to-red-500"
                            style={{ width: `${Math.min(layer.gradientNorm * 50, 100)}%` }}
                          />
                        </div>
                      </motion.div>
                    );
                  })}
                </AnimatePresence>
              </div>
            </div>
          </>
        )}

        {viewMode === 'comparison' && (
          <>
            {/* Architecture Comparison */}
            <div className="space-y-4">
              <h3 className="text-sm font-medium">Gradient Flow by Architecture</h3>

              <div className="space-y-3">
                {comparisonData.map((comp, idx) => (
                  <motion.div
                    key={comp.architecture}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: idx * 0.1 }}
                    className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg"
                  >
                    <div className="flex items-center gap-4 mb-3">
                      <div
                        className="w-4 h-4 rounded-full"
                        style={{ backgroundColor: comp.color }}
                      />
                      <h4 className="font-medium capitalize">{comp.architecture}</h4>
                      <Badge variant="outline">Stability: {(comp.stability * 100).toFixed(0)}%</Badge>
                      <Badge variant="outline">Mean Flow: {(comp.meanFlow * 100).toFixed(0)}%</Badge>
                    </div>

                    {/* Gradient flow line */}
                    <div className="relative h-8 bg-gray-200 dark:bg-gray-700 rounded overflow-hidden">
                      <svg width="100%" height="100%" viewBox="0 0 400 32" preserveAspectRatio="none">
                        <polyline
                          fill="none"
                          stroke={comp.color}
                          strokeWidth="2"
                          points={comp.gradientNorms.map((norm, i) => {
                            const x = (i / (comp.gradientNorms.length - 1)) * 400;
                            const y = 32 - Math.min(norm * 30, 32);
                            return `${x},${y}`;
                          }).join(' ')}
                        />
                      </svg>
                    </div>
                  </motion.div>
                ))}
              </div>

              {/* Comparison insights */}
              <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
                <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
                  <Info className="h-4 w-4" />
                  Architecture Insights
                </h4>
                <ul className="space-y-1 text-sm text-blue-800 dark:text-blue-200">
                  <li>• <strong>Pre-Norm:</strong> Better gradient flow, more stable training</li>
                  <li>• <strong>Post-Norm:</strong> Prone to vanishing gradients in deep networks</li>
                  <li>• <strong>ResNet:</strong> Residual connections help maintain gradient flow</li>
                  <li>• <strong>Dense:</strong> Gradient issues more pronounced without skip connections</li>
                </ul>
              </div>
            </div>
          </>
        )}

        {/* Info Box */}
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
          <div className="flex items-start gap-3">
            <Info className="h-5 w-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
            <div className="text-sm text-blue-900 dark:text-blue-100">
              <h4 className="font-medium mb-1">Understanding Gradient Flow</h4>
              <p className="text-blue-800 dark:text-blue-200">
                During backpropagation, gradients flow backward through the network.
                <strong> Vanishing gradients</strong> occur when gradients become extremely small in early layers,
                preventing learning. <strong> Exploding gradients</strong> occur when gradients grow exponentially,
                causing instability. Pre-Norm architecture and residual connections help maintain healthy gradient flow.
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
