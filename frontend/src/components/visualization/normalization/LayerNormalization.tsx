import React, { useMemo, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../ui/card';
import { Badge } from '../../ui/badge';
import { Slider } from '../../ui/slider';
import { useTransformerStore } from '../../../stores/transformerStore';
import { Activity } from 'lucide-react';
import { Latex } from '../../ui/Latex';

interface LayerNormalizationProps {
  className?: string;
}

const pseudoRandom = (seed: number): number => {
  const value = Math.sin(seed * 12.9898 + 78.233) * 43758.5453;
  return value - Math.floor(value);
};

export const LayerNormalization: React.FC<LayerNormalizationProps> = ({ className }) => {
  const { config } = useTransformerStore();
  const [testMean, setTestMean] = useState(0);
  const [testStd, setTestStd] = useState(2);

  // Generate sample data for visualization
  const sampleData = useMemo(() => {
    const data: number[] = [];
    for (let i = 0; i < 100; i++) {
      const u1 = Math.max(pseudoRandom(i * 2 + 1), Number.EPSILON);
      const u2 = pseudoRandom(i * 2 + 2);
      const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      data.push(z * testStd + testMean);
    }
    return data;
  }, [testMean, testStd]);

  // Calculate statistics before normalization
  const beforeStats = useMemo(() => {
    const mean = sampleData.reduce((a, b) => a + b, 0) / sampleData.length;
    const variance = sampleData.reduce((a, b) => a + (b - mean) ** 2, 0) / sampleData.length;
    const std = Math.sqrt(variance);
    return { mean, std, variance, min: Math.min(...sampleData), max: Math.max(...sampleData) };
  }, [sampleData]);

  // Calculate statistics after normalization
  const afterStats = useMemo(() => {
    const eps = config.layer_norm_eps;
    const normalized = sampleData.map(x => (x - beforeStats.mean) / (Math.sqrt(beforeStats.variance || 1) + eps));
    const mean = normalized.reduce((a, b) => a + b, 0) / normalized.length;
    const variance = normalized.reduce((a, b) => a + (b - mean) ** 2, 0) / normalized.length;
    const std = Math.sqrt(variance);
    return { mean, std, min: Math.min(...normalized), max: Math.max(...normalized) };
  }, [sampleData, beforeStats, config.layer_norm_eps]);

  // Calculate after affine transformation
  const afterAffineStats = useMemo(() => {
    const gamma = 1.2;
    const beta = 0.5;
    const eps = config.layer_norm_eps;
    const transformed = sampleData.map(x =>
      gamma * ((x - beforeStats.mean) / (Math.sqrt(beforeStats.variance || 1) + eps)) + beta
    );
    const mean = transformed.reduce((a, b) => a + b, 0) / transformed.length;
    const variance = transformed.reduce((a, b) => a + (b - mean) ** 2, 0) / transformed.length;
    const std = Math.sqrt(variance);
    return { mean, std, min: Math.min(...transformed), max: Math.max(...transformed), gamma, beta };
  }, [sampleData, beforeStats, config.layer_norm_eps]);

  // Histogram bins
  const createHistogram = (data: number[], min: number, max: number) => {
    const binCount = 30;
    const binWidth = (max - min) / binCount;
    const bins = new Array(binCount).fill(0);
    data.forEach(value => {
      const binIndex = Math.min(Math.floor((value - min) / binWidth), binCount - 1);
      bins[binIndex] = bins[binIndex] + 1;
    });
    return bins;
  };

  const beforeHistogram = useMemo(
    () => createHistogram(sampleData, beforeStats.min, beforeStats.max),
    [sampleData, beforeStats]
  );

  const afterHistogram = useMemo(
    () => {
      const eps = config.layer_norm_eps;
      const normalized = sampleData.map(x => (x - beforeStats.mean) / (Math.sqrt(beforeStats.variance || 1) + eps));
      return createHistogram(normalized, afterStats.min, afterStats.max);
    },
    [sampleData, beforeStats, afterStats, config.layer_norm_eps]
  );

  const afterAffineHistogram = useMemo(
    () => {
      const eps = config.layer_norm_eps;
      const gamma = afterAffineStats.gamma;
      const beta = afterAffineStats.beta;
      const transformed = sampleData.map(x =>
        gamma * ((x - beforeStats.mean) / (Math.sqrt(beforeStats.variance || 1) + eps)) + beta
      );
      return createHistogram(transformed, afterAffineStats.min, afterAffineStats.max);
    },
    [sampleData, beforeStats, afterAffineStats, config.layer_norm_eps]
  );

  const renderHistogram = (
    bins: number[],
    min: number,
    max: number,
    color: string
  ) => {
    const maxCount = Math.max(...bins);
    const binWidth = (max - min) / bins.length;

    return (
      <div className="flex items-end gap-0.5 h-24">
        {bins.map((count, i) => {
          const height = (count / maxCount) * 100;
          const binStart = min + i * binWidth;
          const binEnd = binStart + binWidth;
          return (
            <div
              key={i}
              className="flex-1 rounded-t transition-all hover:opacity-80"
              style={{
                height: `${height}%`,
                backgroundColor: color,
                opacity: 0.7,
              }}
              title={`[${binStart.toFixed(2)}, ${binEnd.toFixed(2)}]: ${count} values`}
            />
          );
        })}
      </div>
    );
  };

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Layer Normalization</CardTitle>
            <CardDescription>ε = {config.layer_norm_eps.toExponential()}</CardDescription>
          </div>
          <Activity className="h-5 w-5 text-muted-foreground" />
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Formula */}
        <div className="p-4 bg-muted rounded-lg space-y-2 text-sm">
          <Latex display>{`\\text{LayerNorm}(x) = \\gamma \\cdot \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta`}</Latex>
          <div className="text-xs text-muted-foreground">
            Where <Latex>{`\\mu = \\text{mean}(x), \\sigma^2 = \\text{variance}(x), \\gamma = \\text{scale}, \\beta = \\text{shift}`}</Latex>
          </div>
        </div>

        {/* Interactive Controls */}
        <div className="space-y-4 p-4 bg-muted/50 rounded-lg">
          <h4 className="text-sm font-medium">Test Data Controls</h4>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm">Mean: {testMean.toFixed(2)}</span>
              <span className="text-xs text-muted-foreground">Original distribution mean</span>
            </div>
            <Slider
              min={-5}
              max={5}
              step={0.1}
              value={[testMean]}
              onValueChange={([v]) => setTestMean(v)}
            />
          </div>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm">Std Dev: {testStd.toFixed(2)}</span>
              <span className="text-xs text-muted-foreground">Original distribution std</span>
            </div>
            <Slider
              min={0.5}
              max={5}
              step={0.1}
              value={[testStd]}
              onValueChange={([v]) => setTestStd(v)}
            />
          </div>
        </div>

        {/* Before Normalization */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-medium">Before Normalization</h4>
            <div className="flex gap-2">
              <Badge variant="secondary">μ: {beforeStats.mean.toFixed(3)}</Badge>
              <Badge variant="secondary">σ: {beforeStats.std.toFixed(3)}</Badge>
            </div>
          </div>
          <div className="p-4 bg-muted/50 rounded-lg">
            {renderHistogram(beforeHistogram, beforeStats.min, beforeStats.max, 'rgb(239, 68, 68)')}
            <div className="flex justify-between text-xs text-muted-foreground mt-2">
              <span>{beforeStats.min.toFixed(2)}</span>
              <span>{beforeStats.max.toFixed(2)}</span>
            </div>
          </div>
        </div>

        {/* After Normalization */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-medium">After Normalization</h4>
            <div className="flex gap-2">
              <Badge variant="secondary">μ: {afterStats.mean.toFixed(3)}</Badge>
              <Badge variant="secondary">σ: {afterStats.std.toFixed(3)}</Badge>
            </div>
          </div>
          <div className="p-4 bg-muted/50 rounded-lg">
            {renderHistogram(afterHistogram, afterStats.min, afterStats.max, 'rgb(59, 130, 246)')}
            <div className="flex justify-between text-xs text-muted-foreground mt-2">
              <span>{afterStats.min.toFixed(2)}</span>
              <span>{afterStats.max.toFixed(2)}</span>
            </div>
          </div>
        </div>

        {/* After Affine Transformation */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-medium">
              After Affine (γ={afterAffineStats.gamma.toFixed(2)}, β={afterAffineStats.beta.toFixed(2)})
            </h4>
            <div className="flex gap-2">
              <Badge variant="secondary">μ: {afterAffineStats.mean.toFixed(3)}</Badge>
              <Badge variant="secondary">σ: {afterAffineStats.std.toFixed(3)}</Badge>
            </div>
          </div>
          <div className="p-4 bg-muted/50 rounded-lg">
            {renderHistogram(afterAffineHistogram, afterAffineStats.min, afterAffineStats.max, 'rgb(34, 197, 94)')}
            <div className="flex justify-between text-xs text-muted-foreground mt-2">
              <span>{afterAffineStats.min.toFixed(2)}</span>
              <span>{afterAffineStats.max.toFixed(2)}</span>
            </div>
          </div>
        </div>

        {/* Key Points */}
        <div className="p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
          <h4 className="text-sm font-medium mb-2">Key Points</h4>
          <ul className="text-xs text-muted-foreground space-y-1">
            <li>• Normalization centers data around 0 with unit variance</li>
            <li>• γ (gamma) scales the distribution</li>
            <li>• β (beta) shifts the distribution</li>
            <li>• ε prevents division by zero</li>
            <li>• Applied per feature (last dimension)</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  );
};
