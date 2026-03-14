/**
 * OutputStep Component
 *
 * Displays the final output tensor from the layer.
 */

import React from 'react';
import { CheckCircle } from 'lucide-react';
import { layerApi } from '../../../../services/layerApi';
import type { TensorData, LayerConfig } from '../../../../types/layer';

interface OutputStepProps {
  data: TensorData;
  config: LayerConfig;
}

const getPreviewRows = (tensor: TensorData): number[][] => {
  if (tensor.shape.length === 3 && Array.isArray(tensor.data) && Array.isArray(tensor.data[0])) {
    const batch = tensor.data as number[][][];
    return Array.isArray(batch[0]) ? batch[0] : [];
  }

  if (tensor.shape.length === 2 && Array.isArray(tensor.data)) {
    return tensor.data as number[][];
  }

  return [];
};

export const OutputStep: React.FC<OutputStepProps> = ({ data, config }) => {
  const flatData = layerApi.flattenTensorData(data.data);
  const stats = layerApi.calculateTensorStats(flatData);
  const displayData = getPreviewRows(data);

  // Safe formatting function
  const safeToFixed = (val: number, digits: number = 4) => {
    return isNaN(val) ? 'N/A' : val.toFixed(digits);
  };

  return (
    <div className="space-y-4">
      {/* Success Indicator */}
      <div className="flex items-center justify-center p-6 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
        <CheckCircle className="h-12 w-12 text-green-600 dark:text-green-400 mr-4" />
        <div>
          <h3 className="text-lg font-bold text-green-800 dark:text-green-200">Layer Computation Complete</h3>
          <p className="text-sm text-green-600 dark:text-green-400">Output ready for next layer</p>
        </div>
      </div>

      {/* Output Info */}
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
        <h4 className="text-sm font-medium mb-3">Output Statistics</h4>
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

      {/* Summary */}
      <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
        <h4 className="text-sm font-medium mb-3">Computation Summary</h4>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-600 dark:text-gray-400">Layer processed:</span>
            <span className="font-medium">{config.batch_first ? 'Batch-First' : 'Seq-First'}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600 dark:text-gray-400">Normalization:</span>
            <span className="font-medium">{config.norm_first ? 'Pre-Norm' : 'Post-Norm'}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600 dark:text-gray-400">Dropout rate:</span>
            <span className="font-medium">{(config.dropout * 100).toFixed(1)}%</span>
          </div>
        </div>
      </div>

      {/* Data Preview */}
      {Array.isArray(data.data) && data.data.length > 0 && (
        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h4 className="text-sm font-medium mb-3">Output Preview (First 3 rows)</h4>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  {Array.from({ length: Math.min(8, data.shape[1] || 0) }, (_, i) => (
                    <th key={i} className="px-2 py-1 text-left font-mono text-xs">
                      d{i}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(() => {
                  return displayData.slice(0, 3).map((row, i) => (
                    <tr key={i} className="border-b">
                      {row.slice(0, 8).map((val, j) => {
                        const displayVal = Number.isFinite(val) ? val.toFixed(4) : 'N/A';
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
      )}
    </div>
  );
};
