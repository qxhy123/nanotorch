import axios from 'axios';
import type {
  AttentionData,
  TransformerConfig,
  TransformerInput,
  TransformerOutput,
  TransformerForwardOptions,
} from '../types/transformer';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

type JsonRecord = Record<string, unknown>;

function isJsonRecord(value: unknown): value is JsonRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * Convert snake_case to camelCase.
 */
function toCamelCase(value: unknown): unknown {
  if (value === null || typeof value !== 'object') {
    return value;
  }

  if (Array.isArray(value)) {
    return value.map(toCamelCase);
  }

  if (!isJsonRecord(value)) {
    return value;
  }

  const result: JsonRecord = {};

  for (const [key, entry] of Object.entries(value)) {
    const camelKey = key.replace(/_([a-z])/g, (_, letter: string) => letter.toUpperCase());
    result[camelKey] = toCamelCase(entry);
  }

  return result;
}

function buildInputPayload(input: TransformerInput): JsonRecord {
  const payload: JsonRecord = {
    text: input.text,
  };

  if (input.tokens) {
    payload.tokens = input.tokens;
  }

  if (input.targetText) {
    payload.target_text = input.targetText;
  }

  if (input.targetTokens) {
    payload.target_tokens = input.targetTokens;
  }

  if (input.src) {
    payload.src = input.src;
  }

  if (input.tgt) {
    payload.tgt = input.tgt;
  }

  return payload;
}

function buildForwardOptionsPayload(options?: TransformerForwardOptions): JsonRecord {
  const resolvedOptions: Required<TransformerForwardOptions> = {
    returnAttention: options?.returnAttention ?? true,
    returnAllLayers: options?.returnAllLayers ?? false,
    returnEmbeddings: options?.returnEmbeddings ?? true,
  };

  return {
    return_attention: resolvedOptions.returnAttention,
    return_all_layers: resolvedOptions.returnAllLayers,
    return_embeddings: resolvedOptions.returnEmbeddings,
  };
}

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000,
});

api.interceptors.request.use(
  (config) => config,
  (error) => Promise.reject(error)
);

api.interceptors.response.use(
  (response) => {
    response.data = toCamelCase(response.data);
    return response;
  },
  (error) => Promise.reject(error)
);

export const transformerApi = {
  /**
   * Health check.
   */
  healthCheck: async (): Promise<unknown> => {
    const response = await api.get('/health');
    return response.data;
  },

  /**
   * Forward pass through transformer.
   */
  forward: async (
    config: TransformerConfig,
    input: TransformerInput,
    options?: TransformerForwardOptions
  ): Promise<TransformerOutput> => {
    const response = await api.post('/api/v1/transformer/forward', {
      config,
      input_data: buildInputPayload(input),
      options: buildForwardOptionsPayload(options),
    });
    return response.data as TransformerOutput;
  },

  /**
   * Get attention weights only.
   */
  getAttention: async (
    config: TransformerConfig,
    input: TransformerInput,
    layerIndex: number
  ): Promise<{ success: boolean; data: AttentionData }> => {
    const response = await api.post(
      '/api/v1/transformer/attention',
      {
        config,
        input_data: buildInputPayload(input),
      },
      {
        params: { layer_index: layerIndex },
      }
    );
    return response.data as { success: boolean; data: AttentionData };
  },

  /**
   * Get embeddings.
   */
  getEmbeddings: async (
    config: TransformerConfig,
    input: TransformerInput
  ): Promise<unknown> => {
    const response = await api.post('/api/v1/transformer/embeddings', {
      config,
      input_data: buildInputPayload(input),
    });
    return response.data;
  },

  /**
   * Get positional encodings.
   */
  getPositionalEncodings: async (
    seqLen: number,
    dModel: number
  ): Promise<unknown> => {
    const response = await api.get('/api/v1/transformer/positional-encoding', {
      params: { seq_len: seqLen, d_model: dModel },
    });
    return response.data;
  },

  /**
   * Validate configuration.
   */
  validateConfig: async (config: TransformerConfig): Promise<{
    valid: boolean;
    errors?: string[];
  }> => {
    const response = await api.post('/api/v1/transformer/validate-config', config);
    return response.data as { valid: boolean; errors?: string[] };
  },
};

export default api;
