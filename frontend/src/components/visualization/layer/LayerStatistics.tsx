/**
 * LayerStatistics Component
 *
 * Displays statistics about the layer computation.
 */

import React from 'react';
import { Card, CardContent } from '../../ui/card';
import { Badge } from '../../ui/badge';
import { BarChart3, Cpu, Zap, HardDrive } from 'lucide-react';
import type { LayerType, LayerConfig } from '../../../types/layer';

interface LayerStatisticsProps {
  layerType: LayerType;
  config: LayerConfig;
}

export const LayerStatistics: React.FC<LayerStatisticsProps> = ({
  layerType,
  config,
}) => {
  // Calculate parameters with safe defaults
  const d_model = config.d_model ?? 512;
  const dim_feedforward = config.dim_feedforward ?? 2048;

  // Self-attention parameters
  const attnParams = 4 * d_model * d_model + 4 * d_model;

  // Feed-forward parameters
  const ffnParams = d_model * dim_feedforward + dim_feedforward +
                  dim_feedforward * d_model + d_model;

  // Layer norm parameters
  const numNorms = layerType === 'decoder' ? 3 : 2;
  const normParams = numNorms * (2 * d_model);

  // Total parameters
  let totalParams = attnParams + ffnParams + normParams;

  // Add cross-attention for decoder
  if (layerType === 'decoder') {
    totalParams += attnParams;
  }

  // Estimate FLOPs (rough approximation)
  const seqLen = 64; // Assume sequence length
  let flops = 2 * seqLen * seqLen * d_model; // Self-attention
  flops += 2 * seqLen * d_model * dim_feedforward; // FFN

  if (layerType === 'decoder') {
    flops += 2 * seqLen * seqLen * d_model; // Cross-attention
  }

  // Estimate memory
  const memoryMB = (totalParams * 4) / (1024 * 1024);

  // Safe formatting function
  const formatNumber = (val: number) => {
    return isNaN(val) || !isFinite(val) ? 'N/A' : val.toLocaleString();
  };

  const formatMemory = (val: number) => {
    return isNaN(val) || !isFinite(val) ? 'N/A' : `${val.toFixed(2)} MB`;
  };

  return (
    <Card>
      <CardContent className="pt-6">
        <div className="flex items-center gap-2 mb-4">
          <BarChart3 className="h-4 w-4 text-gray-500" />
          <h3 className="font-medium text-sm">Layer Statistics</h3>
        </div>

        <div className="grid grid-cols-3 gap-4">
          {/* Parameters */}
          <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
            <div className="flex items-center gap-2 mb-1">
              <Cpu className="h-4 w-4 text-blue-600 dark:text-blue-400" />
              <div className="text-xs text-gray-600 dark:text-gray-400">Parameters</div>
            </div>
            <div className="text-lg font-bold text-blue-700 dark:text-blue-300">
              {formatNumber(totalParams)}
            </div>
          </div>

          {/* FLOPs */}
          <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
            <div className="flex items-center gap-2 mb-1">
              <Zap className="h-4 w-4 text-purple-600 dark:text-purple-400" />
              <div className="text-xs text-gray-600 dark:text-gray-400">FLOPs</div>
            </div>
            <div className="text-lg font-bold text-purple-700 dark:text-purple-300">
              {formatNumber(flops)}
            </div>
          </div>

          {/* Memory */}
          <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
            <div className="flex items-center gap-2 mb-1">
              <HardDrive className="h-4 w-4 text-green-600 dark:text-green-400" />
              <div className="text-xs text-gray-600 dark:text-gray-400">Memory</div>
            </div>
            <div className="text-lg font-bold text-green-700 dark:text-green-300">
              {formatMemory(memoryMB)}
            </div>
          </div>
        </div>

        {/* Breakdown */}
        <div className="mt-4 pt-4 border-t">
          <h4 className="text-xs font-medium mb-2 text-gray-600 dark:text-gray-400">Parameter Breakdown</h4>
          <div className="space-y-1 text-xs">
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Self-Attention:</span>
              <Badge variant="secondary">{formatNumber(attnParams)}</Badge>
            </div>
            {layerType === 'decoder' && (
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">Cross-Attention:</span>
                <Badge variant="secondary">{formatNumber(attnParams)}</Badge>
              </div>
            )}
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Feed-Forward:</span>
              <Badge variant="secondary">{formatNumber(ffnParams)}</Badge>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Layer Norm:</span>
              <Badge variant="secondary">{formatNumber(normParams)}</Badge>
            </div>
            <div className="flex justify-between font-medium pt-1 border-t">
              <span>Total:</span>
              <Badge variant="default">{formatNumber(totalParams)}</Badge>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
