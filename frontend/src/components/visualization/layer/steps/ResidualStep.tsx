/**
 * ResidualStep Component
 *
 * Displays residual connection visualization.
 */

import React from 'react';
import { Plus } from 'lucide-react';
import { layerApi } from '../../../../services/layerApi';
import type { SublayerComputation } from '../../../../types/layer';

interface ResidualStepProps {
  residualData: SublayerComputation;
  stepIndex: number;
}

export const ResidualStep: React.FC<ResidualStepProps> = ({
  residualData,
  stepIndex,
}) => {
  const residualOutput = residualData.residual_output;

  // Safe stats calculation
  const getStats = () => {
    if (!residualOutput) return null;
    const flat = layerApi.flattenTensorData(residualOutput.data);
    if (!flat || flat.length === 0) return null;
    const stats = layerApi.calculateTensorStats(flat);
    // Check for NaN values
    if (isNaN(stats.min) || isNaN(stats.max) || isNaN(stats.mean) || isNaN(stats.std)) {
      return null;
    }
    return stats;
  };

  const stats = getStats();

  // Try to get the input to the residual (x) and the sublayer output
  // For this visualization, we'll show the concept
  const sublayerType = stepIndex === 0 ? 'Self-Attention' : stepIndex === 1 ? 'Cross-Attention' : 'Feed-Forward';

  // Safe formatting function
  const safeToFixed = (val: number, digits: number = 4) => {
    return isNaN(val) ? 'N/A' : val.toFixed(digits);
  };

  return (
    <div className="space-y-4">
      {/* Residual Formula */}
      <div className="p-4 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
        <h4 className="text-sm font-medium mb-2">Residual Connection</h4>
        <div className="flex items-center justify-center gap-4 text-lg font-mono">
          <span className="text-blue-600 dark:text-blue-400">x</span>
          <Plus className="w-5 h-5 text-gray-500" />
          <span className="text-purple-600 dark:text-purple-400">Sublayer(x)</span>
          <span className="text-gray-600 dark:text-gray-400">=</span>
          <span className="text-green-600 dark:text-green-400">Output</span>
        </div>
        <p className="text-xs text-gray-600 dark:text-gray-400 mt-2 text-center">
          Adds the original input back to the sublayer output
        </p>
      </div>

      {/* Sublayer Info */}
      <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
        <h4 className="text-sm font-medium mb-2">Sublayer: {sublayerType}</h4>
        <div className="text-sm text-gray-600 dark:text-gray-400">
          This residual connection allows gradients to flow directly through the network,
          helping with training deep networks.
        </div>
      </div>

      {/* Output Statistics */}
      {residualOutput && stats && (
        <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
          <h4 className="text-sm font-medium mb-3 text-green-800 dark:text-green-200">Residual Output</h4>
          <div className="grid grid-cols-4 gap-4 text-sm">
            <div>
              <div className="text-xs text-green-600 dark:text-green-400">Min</div>
              <div className="font-mono">{safeToFixed(stats.min)}</div>
            </div>
            <div>
              <div className="text-xs text-green-600 dark:text-green-400">Max</div>
              <div className="font-mono">{safeToFixed(stats.max)}</div>
            </div>
            <div>
              <div className="text-xs text-green-600 dark:text-green-400">Mean</div>
              <div className="font-mono">{safeToFixed(stats.mean)}</div>
            </div>
            <div>
              <div className="text-xs text-green-600 dark:text-green-400">Std</div>
              <div className="font-mono">{safeToFixed(stats.std)}</div>
            </div>
          </div>
          <div className="mt-2 text-xs text-green-700 dark:text-green-300">
            Shape: [{residualOutput.shape.join(', ')}]
          </div>
        </div>
      )}

      {/* Key Benefits */}
      <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
        <h4 className="text-sm font-medium mb-2 text-blue-800 dark:text-blue-200">Key Benefits</h4>
        <ul className="text-xs text-gray-700 dark:text-gray-300 space-y-1">
          <li>• Enables gradient flow through deep networks</li>
          <li>• Helps prevent vanishing gradient problem</li>
          <li>• Allows network to learn identity functions easily</li>
          <li>• Improves training stability and convergence</li>
        </ul>
      </div>
    </div>
  );
};
