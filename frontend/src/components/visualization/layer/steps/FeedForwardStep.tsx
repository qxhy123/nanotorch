/**
 * FeedForwardStep Component
 *
 * Displays feed-forward network computation visualization.
 */

import React from 'react';
import { Badge } from '../../../ui/badge';
import { layerApi } from '../../../../services/layerApi';
import type { SublayerComputation, LayerConfig, TensorData } from '../../../../types/layer';

interface FeedForwardStepProps {
  ffnData: SublayerComputation;
  config: LayerConfig;
}

export const FeedForwardStep: React.FC<FeedForwardStepProps> = ({
  ffnData,
  config,
}) => {
  const linear1Output = ffnData.linear1_output;
  const activationOutput = ffnData.activation_output;
  const linear2Output = ffnData.linear2_output;

  const getStats = (data?: TensorData) => {
    if (!data) return null;
    const flat = layerApi.flattenTensorData(data.data);
    if (!flat || flat.length === 0) return null;
    const stats = layerApi.calculateTensorStats(flat);
    // Check for NaN values
    if (isNaN(stats.min) || isNaN(stats.max) || isNaN(stats.mean) || isNaN(stats.std)) {
      return null;
    }
    return stats;
  };

  const linear1Stats = getStats(linear1Output);
  const activationStats = getStats(activationOutput);
  const linear2Stats = getStats(linear2Output);

  // Safely get config values with defaults
  const dModel = config.d_model ?? 512;
  const dimFeedforward = config.dim_feedforward ?? 2048;
  const activation = config.activation ?? 'relu';

  const safeToFixed = (val: number, digits: number = 3) => {
    return isNaN(val) ? 'N/A' : val.toFixed(digits);
  };

  const safeToLocaleString = (val: number) => {
    return isNaN(val) ? 'N/A' : val.toLocaleString();
  };

  return (
    <div className="space-y-4">
      {/* Architecture */}
      <div className="flex items-center justify-center gap-2 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
        <div className="text-center">
          <div className="text-xs text-gray-500">Input</div>
          <div className="font-mono text-sm">{dModel}d</div>
        </div>
        <div className="text-gray-400">→</div>
        <div className="text-center">
          <div className="text-xs text-blue-600 dark:text-blue-400">Linear1</div>
          <div className="font-mono text-xs">({dModel}, {dimFeedforward})</div>
        </div>
        <div className="text-gray-400">→</div>
        <div className="text-center">
          <div className="text-xs text-orange-600 dark:text-orange-400">{activation}</div>
          <div className="font-mono text-sm">{dimFeedforward}d</div>
        </div>
        <div className="text-gray-400">→</div>
        <div className="text-center">
          <div className="text-xs text-green-600 dark:text-green-400">Linear2</div>
          <div className="font-mono text-xs">({dimFeedforward}, {dModel})</div>
        </div>
        <div className="text-gray-400">→</div>
        <div className="text-center">
          <div className="text-xs text-gray-500">Output</div>
          <div className="font-mono text-sm">{dModel}d</div>
        </div>
      </div>

      {/* Step by Step */}
      <div className="space-y-3">
        {/* Linear1 */}
        {linear1Output && linear1Stats && (
          <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
            <h4 className="text-sm font-medium mb-2 text-blue-800 dark:text-blue-200">Linear1: Expansion</h4>
            <div className="grid grid-cols-4 gap-4 text-sm">
              <div>
                <div className="text-xs text-gray-600 dark:text-gray-400">Min</div>
                <div className="font-mono">{safeToFixed(linear1Stats.min)}</div>
              </div>
              <div>
                <div className="text-xs text-gray-600 dark:text-gray-400">Max</div>
                <div className="font-mono">{safeToFixed(linear1Stats.max)}</div>
              </div>
              <div>
                <div className="text-xs text-gray-600 dark:text-gray-400">Mean</div>
                <div className="font-mono">{safeToFixed(linear1Stats.mean)}</div>
              </div>
              <div>
                <div className="text-xs text-gray-600 dark:text-gray-400">Std</div>
                <div className="font-mono">{safeToFixed(linear1Stats.std)}</div>
              </div>
            </div>
            <div className="mt-2 text-xs text-gray-600 dark:text-gray-400">
              {dModel} → {dimFeedforward} dimensions
            </div>
          </div>
        )}

        {/* Activation */}
        {activationOutput && activationStats && (
          <div className="p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
            <h4 className="text-sm font-medium mb-2 text-orange-800 dark:text-orange-200">
              Activation: {activation.toUpperCase()}
            </h4>
            <div className="grid grid-cols-4 gap-4 text-sm">
              <div>
                <div className="text-xs text-gray-600 dark:text-gray-400">Min</div>
                <div className="font-mono">{safeToFixed(activationStats.min)}</div>
              </div>
              <div>
                <div className="text-xs text-gray-600 dark:text-gray-400">Max</div>
                <div className="font-mono">{safeToFixed(activationStats.max)}</div>
              </div>
              <div>
                <div className="text-xs text-gray-600 dark:text-gray-400">Mean</div>
                <div className="font-mono">{safeToFixed(activationStats.mean)}</div>
              </div>
              <div>
                <div className="text-xs text-gray-600 dark:text-gray-400">Std</div>
                <div className="font-mono">{safeToFixed(activationStats.std)}</div>
              </div>
            </div>
            <div className="mt-2 text-xs text-gray-600 dark:text-gray-400">
              Negative values zeroed out (ReLU) or smoothed (GELU)
            </div>
          </div>
        )}

        {/* Linear2 */}
        {linear2Output && linear2Stats && (
          <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
            <h4 className="text-sm font-medium mb-2 text-green-800 dark:text-green-200">Linear2: Projection</h4>
            <div className="grid grid-cols-4 gap-4 text-sm">
              <div>
                <div className="text-xs text-gray-600 dark:text-gray-400">Min</div>
                <div className="font-mono">{safeToFixed(linear2Stats.min)}</div>
              </div>
              <div>
                <div className="text-xs text-gray-600 dark:text-gray-400">Max</div>
                <div className="font-mono">{safeToFixed(linear2Stats.max)}</div>
              </div>
              <div>
                <div className="text-xs text-gray-600 dark:text-gray-400">Mean</div>
                <div className="font-mono">{safeToFixed(linear2Stats.mean)}</div>
              </div>
              <div>
                <div className="text-xs text-gray-600 dark:text-gray-400">Std</div>
                <div className="font-mono">{safeToFixed(linear2Stats.std)}</div>
              </div>
            </div>
            <div className="mt-2 text-xs text-gray-600 dark:text-gray-400">
              {dimFeedforward} → {dModel} dimensions
            </div>
          </div>
        )}
      </div>

      {/* Parameter Count */}
      <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
        <h4 className="text-sm font-medium mb-2">Parameters</h4>
        <div className="space-y-1 text-sm">
          <div className="flex justify-between">
            <span>W₁ (d_model → d_ff):</span>
            <Badge variant="secondary">{safeToLocaleString(dModel * dimFeedforward)}</Badge>
          </div>
          <div className="flex justify-between">
            <span>b₁:</span>
            <Badge variant="secondary">{safeToLocaleString(dimFeedforward)}</Badge>
          </div>
          <div className="flex justify-between">
            <span>W₂ (d_ff → d_model):</span>
            <Badge variant="secondary">{safeToLocaleString(dimFeedforward * dModel)}</Badge>
          </div>
          <div className="flex justify-between">
            <span>b₂:</span>
            <Badge variant="secondary">{safeToLocaleString(dModel)}</Badge>
          </div>
          <div className="border-t pt-1 mt-1 flex justify-between font-medium">
            <span>Total:</span>
            <Badge variant="default">
              {safeToLocaleString(2 * dModel * dimFeedforward + dModel + dimFeedforward)}
            </Badge>
          </div>
        </div>
      </div>
    </div>
  );
};
