/**
 * Layer API service.
 *
 * This module provides methods to interact with the layer computation API endpoints,
 * including encoder and decoder layer computation with intermediate results.
 */

import api from './api';
import type {
  EncoderLayerResult,
  DecoderLayerResult,
  LayerConfig,
  LayerStatistics,
  LayerTypeInfo,
} from '../types/layer';

type SerializableValue =
  | boolean
  | number
  | string
  | null
  | undefined
  | SerializableValue[]
  | { [key: string]: SerializableValue };

type NestedNumberArray = number | NestedNumberArray[];

const isRecord = (value: SerializableValue): value is { [key: string]: SerializableValue } =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const hasOwn = (obj: object, key: string): boolean =>
  Object.prototype.hasOwnProperty.call(obj, key);

/**
 * Convert camelCase to snake_case for backend requests.
 */
function toSnakeCase(obj: SerializableValue): SerializableValue {
  if (obj === null || typeof obj !== 'object') {
    return obj;
  }

  if (Array.isArray(obj)) {
    return obj.map(toSnakeCase);
  }

  const result: { [key: string]: SerializableValue } = {};
  for (const key in obj) {
    if (hasOwn(obj, key)) {
      const snakeKey = key.replace(/[A-Z]/g, (letter) => `_${letter.toLowerCase()}`);
      result[snakeKey] = toSnakeCase(obj[key]);
    }
  }
  return result;
}

/**
 * Convert snake_case to camelCase for backend responses.
 */
function toCamelCase(obj: SerializableValue): SerializableValue {
  if (obj === null || typeof obj !== 'object') {
    return obj;
  }

  if (Array.isArray(obj)) {
    return obj.map(toCamelCase);
  }

  const result: { [key: string]: SerializableValue } = {};
  for (const key in obj) {
    if (hasOwn(obj, key)) {
      // Handle special case for is_special -> isSpecial
      if (key === 'is_special') {
        result.isSpecial = toCamelCase(obj[key]);
      }
      // Handle special case for sublayer1 -> sublayer1, etc.
      else if (key.startsWith('sublayer')) {
        result[key] = toCamelCase(obj[key]);
      }
      // General snake_case to camelCase conversion
      else if (key.includes('_')) {
        const camelKey = key.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());
        result[camelKey] = toCamelCase(obj[key]);
      }
      // Already camelCase or no underscores
      else {
        result[key] = toCamelCase(obj[key]);
      }
    }
  }
  return result;
}

/**
 * Layer API client.
 */
export const layerApi = {
  /**
   * Compute encoder layer with intermediate results.
   *
   * @param config - Transformer configuration
   * @param inputData - Layer input data (2D array: [seq_len, d_model])
   * @param options - Optional computation options
   * @returns Encoder layer computation results
   */
  computeEncoderLayer: async (
    config: LayerConfig,
    inputData: number[][],
    options?: {
      srcMask?: number[][];
      isCausal?: boolean;
    }
  ): Promise<EncoderLayerResult> => {
    const requestBody = {
      config: toSnakeCase(config as unknown as SerializableValue),
      input_data: {
        data: inputData,
        shape: [inputData.length, inputData[0]?.length || 0],
        dtype: 'float32',
      },
      src_mask: options?.srcMask || null,
      is_causal: options?.isCausal || false,
    };

    const response = await api.post('/api/v1/layer/encoder/compute', requestBody);
    return toCamelCase(response.data as SerializableValue) as unknown as EncoderLayerResult;
  },

  /**
   * Compute decoder layer with intermediate results.
   *
   * @param config - Transformer configuration
   * @param inputData - Decoder input data (target)
   * @param encoderOutput - Encoder output (memory)
   * @param options - Optional computation options
   * @returns Decoder layer computation results
   */
  computeDecoderLayer: async (
    config: LayerConfig,
    inputData: number[][],
    encoderOutput: number[][],
    options?: {
      tgtMask?: number[][];
      memoryMask?: number[][];
    }
  ): Promise<DecoderLayerResult> => {
    const requestBody = {
      config: toSnakeCase(config as unknown as SerializableValue),
      input_data: {
        data: inputData,
        shape: [inputData.length, inputData[0]?.length || 0],
        dtype: 'float32',
      },
      encoder_output: {
        data: encoderOutput,
        shape: [encoderOutput.length, encoderOutput[0]?.length || 0],
        dtype: 'float32',
      },
      tgt_mask: options?.tgtMask || null,
      memory_mask: options?.memoryMask || null,
    };

    const response = await api.post('/api/v1/layer/decoder/compute', requestBody);
    return toCamelCase(response.data as SerializableValue) as unknown as DecoderLayerResult;
  },

  /**
   * Get statistics for an encoder layer.
   *
   * @param config - Transformer configuration
   * @returns Layer statistics
   */
  getEncoderStatistics: async (config: LayerConfig): Promise<LayerStatistics> => {
    const response = await api.post('/api/v1/layer/encoder/statistics', toSnakeCase(config as unknown as SerializableValue));
    return toCamelCase(response.data as SerializableValue) as unknown as LayerStatistics;
  },

  /**
   * Get statistics for a decoder layer.
   *
   * @param config - Transformer configuration
   * @returns Layer statistics
   */
  getDecoderStatistics: async (config: LayerConfig): Promise<LayerStatistics> => {
    const response = await api.post('/api/v1/layer/decoder/statistics', toSnakeCase(config as unknown as SerializableValue));
    return toCamelCase(response.data as SerializableValue) as unknown as LayerStatistics;
  },

  /**
   * Get available layer types with descriptions.
   *
   * @returns Layer types information
   */
  getLayerTypes: async (): Promise<LayerTypeInfo[]> => {
    const response = await api.get('/api/v1/layer/types');
    const responseData = response.data as SerializableValue;
    const layerTypes = isRecord(responseData)
      ? responseData.layer_types
      : null;
    return Array.isArray(layerTypes) ? (layerTypes as unknown as LayerTypeInfo[]) : [];
  },

  /**
   * Calculate tensor statistics from data.
   *
   * @param data - Tensor data (flattened array)
   * @returns Tensor statistics
   */
  calculateTensorStats: (data: number[]): {
    min: number;
    max: number;
    mean: number;
    std: number;
    median: number;
    q25: number;
    q75: number;
  } => {
    if (data.length === 0) {
      return { min: 0, max: 0, mean: 0, std: 0, median: 0, q25: 0, q75: 0 };
    }

    // Sort for percentiles
    const sorted = [...data].sort((a, b) => a - b);
    const n = sorted.length;

    // Calculate mean
    const mean = data.reduce((sum, val) => sum + val, 0) / data.length;

    // Calculate standard deviation
    const variance = data.reduce((sum, val) => sum + (val - mean) ** 2, 0) / data.length;
    const std = Math.sqrt(variance);

    // Calculate percentiles
    const q25Index = Math.floor(n * 0.25);
    const medianIndex = Math.floor(n * 0.5);
    const q75Index = Math.floor(n * 0.75);

    return {
      min: sorted[0],
      max: sorted[n - 1],
      mean,
      std,
      median: sorted[medianIndex],
      q25: sorted[q25Index],
      q75: sorted[q75Index],
    };
  },

  /**
   * Flatten nested tensor data for statistics calculation.
   *
   * @param data - Nested tensor data
   * @returns Flattened array
   */
  flattenTensorData: (data: NestedNumberArray): number[] => {
    if (Array.isArray(data)) {
      if (data.length === 0) return [];
      if (typeof data[0] === 'number') {
        return data as number[];
      }
      // Recursively flatten nested arrays
      return data.flatMap((item) => layerApi.flattenTensorData(item));
    }
    return [];
  },

  /**
   * Create sample input data for layer computation.
   *
   * @param seqLen - Sequence length
   * @param dModel - Model dimension
   * @param seed - Optional seed for reproducibility
   * @returns Sample input data
   */
  createSampleInput: (
    seqLen: number,
    dModel: number,
    seed?: number
  ): number[][] => {
    // Seeded random number generator
    const random = seed
      ? (() => {
          let s = seed;
          return () => {
            s = Math.sin(s) * 10000;
            return s - Math.floor(s);
          };
        })()
      : Math.random;

    // Create sample data with some structure
    const data: number[][] = [];
    for (let i = 0; i < seqLen; i++) {
      const row: number[] = [];
      for (let j = 0; j < dModel; j++) {
        // Create some variation across positions and dimensions
        const posFactor = i / seqLen;
        const dimFactor = j / dModel;
        row.push(random() * 2 - 1 + posFactor * 0.5 + dimFactor * 0.3);
      }
      data.push(row);
    }
    return data;
  },

  /**
   * Compare two tensors and generate comparison data.
   *
   * @param before - Before tensor
   * @param after - After tensor
   * @param operation - Operation performed
   * @returns Tensor comparison
   */
  compareTensors: (
    before: number[] | number[][],
    after: number[] | number[][],
    operation: string
  ) => {
    const beforeFlat = layerApi.flattenTensorData(before);
    const afterFlat = layerApi.flattenTensorData(after);

    const beforeStats = layerApi.calculateTensorStats(beforeFlat);
    const afterStats = layerApi.calculateTensorStats(afterFlat);

    // Calculate difference statistics
    const differences = beforeFlat.map((b, i) => afterFlat[i] - b);
    const diffStats = layerApi.calculateTensorStats(differences);

    return {
      before: {
        shape: Array.isArray(before[0]) ? [before.length, before[0].length] : [before.length],
        data: beforeFlat,
        dtype: 'float32',
      },
      after: {
        shape: Array.isArray(after[0]) ? [after.length, after[0].length] : [after.length],
        data: afterFlat,
        dtype: 'float32',
      },
      operation,
      beforeStats,
      afterStats,
      differenceStats: diffStats,
    };
  },
};

export default layerApi;
