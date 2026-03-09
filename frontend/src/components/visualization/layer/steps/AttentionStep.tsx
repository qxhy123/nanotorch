/**
 * AttentionStep Component
 *
 * Displays attention computation visualization.
 */

import React from 'react';
import { Badge } from '../../../ui/badge';
import { layerApi } from '../../../../services/layerApi';
import type { AttentionComputation, LayerConfig } from '../../../../types/layer';

interface AttentionStepProps {
  attentionData?: AttentionComputation;
  config: LayerConfig;
}

export const AttentionStep: React.FC<AttentionStepProps> = ({
  attentionData,
  config,
}) => {
  if (!attentionData || !attentionData.output) {
    return <div className="text-gray-500">No attention data available</div>;
  }

  const outputFlat = layerApi.flattenTensorData(attentionData.output.data);
  const stats = layerApi.calculateTensorStats(outputFlat);

  // Safe formatting function
  const safeToFixed = (val: number, digits: number = 4) => {
    return isNaN(val) ? 'N/A' : val.toFixed(digits);
  };

  // Safe config values
  const nhead = config.nhead ?? 8;
  const dModel = config.d_model ?? 512;

  return (
    <div className="space-y-4">
      {/* Configuration */}
      <div className="grid grid-cols-3 gap-4">
        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
          <div className="text-xs text-gray-600 dark:text-gray-400">Heads</div>
          <div className="text-lg font-bold">{nhead}</div>
        </div>
        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
          <div className="text-xs text-gray-600 dark:text-gray-400">d_model</div>
          <div className="text-lg font-bold">{dModel}</div>
        </div>
        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
          <div className="text-xs text-gray-600 dark:text-gray-400">Head Dim</div>
          <div className="text-lg font-bold">{Math.floor(dModel / nhead)}</div>
        </div>
      </div>

      {/* Output Statistics */}
      <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
        <h4 className="text-sm font-medium mb-3">Attention Output Statistics</h4>
        <div className="grid grid-cols-4 gap-4">
          <div>
            <div className="text-xs text-gray-600 dark:text-gray-400">Min</div>
            <div className="font-mono text-sm">{safeToFixed(stats.min)}</div>
          </div>
          <div>
            <div className="text-xs text-gray-600 dark:text-gray-400">Max</div>
            <div className="font-mono text-sm">{safeToFixed(stats.max)}</div>
          </div>
          <div>
            <div className="text-xs text-gray-600 dark:text-gray-400">Mean</div>
            <div className="font-mono text-sm">{safeToFixed(stats.mean)}</div>
          </div>
          <div>
            <div className="text-xs text-gray-600 dark:text-gray-400">Std</div>
            <div className="font-mono text-sm">{safeToFixed(stats.std)}</div>
          </div>
        </div>
      </div>

      {/* Computation Steps */}
      <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
        <h4 className="text-sm font-medium mb-3">Computation Steps</h4>
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-sm">
            <Badge variant="secondary">1</Badge>
            <span>Project input to Q, K, V matrices</span>
            <span className="text-gray-500">(4 × d_model² params)</span>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <Badge variant="secondary">2</Badge>
            <span>Compute attention scores: Q · K^T</span>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <Badge variant="secondary">3</Badge>
            <span>Scale by 1/√d_k</span>
            <span className="text-gray-500">(d_k = {config.d_model / config.nhead})</span>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <Badge variant="secondary">4</Badge>
            <span>Apply softmax to get attention weights</span>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <Badge variant="secondary">5</Badge>
            <span>Weighted sum of values</span>
          </div>
        </div>
      </div>

      {/* Additional Attention Data */}
      {attentionData.weights && (
        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h4 className="text-sm font-medium mb-3">Attention Weights</h4>
          <div className="text-sm text-gray-600 dark:text-gray-400">
            Attention weights shape: [{attentionData.weights.shape.join(', ')}]
          </div>
        </div>
      )}
    </div>
  );
};
