import axios from 'axios';
import type {
  TransformerConfig,
  TransformerInput,
  TransformerOutput,
  TransformerForwardOptions,
} from '../types/transformer';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * Convert snake_case to camelCase
 */
function toCamelCase(obj: any): any {
  if (obj === null || typeof obj !== 'object') {
    return obj;
  }

  if (Array.isArray(obj)) {
    return obj.map(toCamelCase);
  }

  const result: any = {};
  for (const key in obj) {
    if (obj.hasOwnProperty(key)) {
      const camelKey = key.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());
      result[camelKey] = toCamelCase(obj[key]);
    }
  }
  return result;
}

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 seconds timeout
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    // Convert snake_case to camelCase for frontend
    response.data = toCamelCase(response.data);
    return response;
  },
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export const transformerApi = {
  /**
   * Health check
   */
  healthCheck: async () => {
    const response = await api.get('/health');
    return response.data;
  },

  /**
   * Forward pass through transformer
   */
  forward: async (
    config: TransformerConfig,
    input: TransformerInput,
    options?: TransformerForwardOptions
  ): Promise<TransformerOutput> => {
    // Convert camelCase to snake_case for backend
    const input_data: any = {
      text: input.text,
      tokens: input.tokens,
    };

    if (input.targetText) {
      input_data.target_text = input.targetText;
    }

    if (input.targetTokens) {
      input_data.target_tokens = input.targetTokens;
    }

    if (input.src) {
      input_data.src = input.src;
    }

    if (input.tgt) {
      input_data.tgt = input.tgt;
    }

    const response = await api.post('/api/v1/transformer/forward', {
      config,
      input_data,  // Changed from 'input' to 'input_data' to match backend
      options: options || {
        returnAttention: true,
        returnAllLayers: true,
        returnEmbeddings: true,
      },
    });
    return response.data;
  },

  /**
   * Get attention weights only
   */
  getAttention: async (
    config: TransformerConfig,
    input: TransformerInput,
    layerIndex: number
  ): Promise<any> => {
    const response = await api.post('/api/v1/transformer/attention', {
      config,
      input_data: input,  // Changed to match backend
      layer_index: layerIndex,  // Changed to snake_case
    });
    return response.data;
  },

  /**
   * Get embeddings
   */
  getEmbeddings: async (
    config: TransformerConfig,
    input: TransformerInput
  ): Promise<any> => {
    const response = await api.post('/api/v1/transformer/embeddings', {
      config,
      input_data: input,  // Changed to match backend
    });
    return response.data;
  },

  /**
   * Get layer output
   */
  getLayerOutput: async (
    config: TransformerConfig,
    input: TransformerInput,
    layerName: string
  ): Promise<any> => {
    const response = await api.post('/api/v1/transformer/layer-output', {
      config,
      input_data: input,  // Changed to match backend
      layer_name: layerName,  // Changed to snake_case
    });
    return response.data;
  },

  /**
   * Get positional encodings
   */
  getPositionalEncodings: async (
    seqLen: number,
    dModel: number
  ): Promise<any> => {
    const response = await api.get('/api/v1/transformer/positional-encoding', {
      params: { seq_len: seqLen, d_model: dModel },
    });
    return response.data;
  },

  /**
   * Validate configuration
   */
  validateConfig: async (config: TransformerConfig): Promise<{
    valid: boolean;
    errors?: string[];
  }> => {
    const response = await api.post('/api/v1/transformer/validate-config', config);
    return response.data;
  },
};

export default api;
