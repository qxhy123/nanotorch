/**
 * LayerNormStep Component
 *
 * Displays layer normalization before/after comparison.
 */

import React from 'react';
import { layerApi } from '../../../../services/layerApi';
import type { TensorData, LayerConfig } from '../../../../types/layer';

interface LayerNormStepProps {
  normInput?: TensorData;
  normOutput?: TensorData;
  config: LayerConfig;
}

export const LayerNormStep: React.FC<LayerNormStepProps> = ({
  normInput,
  normOutput,
  config,
}) => {
  if (!normInput || !normOutput) {
    return <div className="text-gray-500">No layer norm data available</div>;
  }

  const inputFlat = layerApi.flattenTensorData(normInput.data);
  const outputFlat = layerApi.flattenTensorData(normOutput.data);
  const inputStats = layerApi.calculateTensorStats(inputFlat);
  const outputStats = layerApi.calculateTensorStats(outputFlat);

  // Safe formatting function
  const safeToFixed = (val: number, digits: number = 6) => {
    return isNaN(val) ? 'N/A' : val.toFixed(digits);
  };

  return (
    <div className="space-y-4">
      {/* Comparison Cards */}
      <div className="grid grid-cols-2 gap-4">
        {/* Before */}
        <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
          <h4 className="text-sm font-medium mb-3 text-red-800 dark:text-red-200">Before LayerNorm</h4>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Shape:</span>
              <span className="font-mono">[{normInput.shape.join(', ')}]</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Mean:</span>
              <span className="font-mono">{safeToFixed(inputStats.mean)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Std:</span>
              <span className="font-mono">{safeToFixed(inputStats.std)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Range:</span>
              <span className="font-mono">[{safeToFixed(inputStats.min, 3)}, {safeToFixed(inputStats.max, 3)}]</span>
            </div>
          </div>
        </div>

        {/* After */}
        <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
          <h4 className="text-sm font-medium mb-3 text-green-800 dark:text-green-200">After LayerNorm</h4>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Shape:</span>
              <span className="font-mono">[{normOutput.shape.join(', ')}]</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Mean:</span>
              <span className="font-mono">{safeToFixed(outputStats.mean)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Std:</span>
              <span className="font-mono">{safeToFixed(outputStats.std)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Range:</span>
              <span className="font-mono">[{safeToFixed(outputStats.min, 3)}, {safeToFixed(outputStats.max, 3)}]</span>
            </div>
          </div>
        </div>
      </div>

      {/* Key Insight */}
      <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
        <h4 className="text-sm font-medium mb-2 text-blue-800 dark:text-blue-200">Key Insight</h4>
        <p className="text-sm text-gray-700 dark:text-gray-300">
          LayerNorm normalizes each sample to have zero mean and unit variance.
          Notice how the mean is closer to 0 and std is closer to 1 after normalization.
          ε = {config.layer_norm_eps?.toExponential() ?? (1e-5).toExponential()} is used to prevent division by zero.
        </p>
      </div>
    </div>
  );
};
