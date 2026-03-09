/**
 * InputStep Component
 *
 * Displays the input tensor to the layer.
 */

import React from 'react';
import { Badge } from '../../../ui/badge';
import { layerApi } from '../../../../services/layerApi';
import type { TensorData, LayerConfig } from '../../../../types/layer';

interface InputStepProps {
  data: TensorData;
  config: LayerConfig;
}

export const InputStep: React.FC<InputStepProps> = ({ data, config }) => {
  const flatData = layerApi.flattenTensorData(data.data);
  const stats = layerApi.calculateTensorStats(flatData);

  // Safe formatting function
  const safeToFixed = (val: number, digits: number = 4) => {
    return isNaN(val) ? 'N/A' : val.toFixed(digits);
  };

  return (
    <div className="space-y-4">
      {/* Tensor Info */}
      <div className="grid grid-cols-3 gap-4">
        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
          <div className="text-xs text-gray-600 dark:text-gray-400">Shape</div>
          <div className="font-mono text-sm">[{data.shape.join(', ')}]</div>
        </div>
        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
          <div className="text-xs text-gray-600 dark:text-gray-400">Data Type</div>
          <div className="text-sm">{data.dtype}</div>
        </div>
        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
          <div className="text-xs text-gray-600 dark:text-gray-400">Elements</div>
          <div className="text-sm">{flatData.length.toLocaleString()}</div>
        </div>
      </div>

      {/* Statistics */}
      <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
        <h4 className="text-sm font-medium mb-3">Tensor Statistics</h4>
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

      {/* Configuration */}
      <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
        <h4 className="text-sm font-medium mb-3">Layer Configuration</h4>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-600 dark:text-gray-400">d_model: </span>
            <Badge variant="secondary">{config.d_model}</Badge>
          </div>
          <div>
            <span className="text-gray-600 dark:text-gray-400">nhead: </span>
            <Badge variant="secondary">{config.nhead}</Badge>
          </div>
          <div>
            <span className="text-gray-600 dark:text-gray-400">dim_feedforward: </span>
            <Badge variant="secondary">{config.dim_feedforward}</Badge>
          </div>
          <div>
            <span className="text-gray-600 dark:text-gray-400">dropout: </span>
            <Badge variant="secondary">{config.dropout}</Badge>
          </div>
        </div>
      </div>

      {/* Data Preview */}
      <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
        <h4 className="text-sm font-medium mb-3">Data Preview (First 5 rows)</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b">
                {Array.from({ length: Math.min(10, data.shape[data.shape.length - 1] || 0) }, (_, i) => (
                  <th key={i} className="px-2 py-1 text-left font-mono text-xs">
                    d{i}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(() => {
                // Get the first dimension for rows
                const numElements = data.shape[0] || 0;
                const rowsToShow = Math.min(5, numElements);

                // Handle different data structures
                let displayData: any[][] = [];

                if (data.shape.length === 3) {
                  // 3D tensor [batch, seq, d_model]
                  const batchData = data.data as any[][];
                  if (Array.isArray(batchData) && batchData.length > 0) {
                    displayData = batchData[0] as any[][];
                  }
                } else if (data.shape.length === 2) {
                  // 2D tensor [seq, d_model]
                  displayData = data.data as any[][];
                }

                return displayData.slice(0, rowsToShow).map((row: any[], i: number) => (
                  <tr key={i} className="border-b">
                    {row.slice(0, 10).map((val: any, j: number) => {
                      const numVal = typeof val === 'number' ? val : parseFloat(val);
                      const displayVal = isNaN(numVal) ? 'N/A' : numVal.toFixed(3);
                      return (
                        <td key={j} className="px-2 py-1 font-mono text-xs">
                          {displayVal}
                        </td>
                      );
                    })}
                  </tr>
                ));
              })()}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
