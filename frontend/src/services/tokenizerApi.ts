/**
 * Tokenizer API service.
 *
 * This module provides methods to interact with the tokenizer API endpoints,
 * including tokenization, vocabulary retrieval, decoding, and comparison.
 */

import api from './api';
import type {
  TokenType,
  TokenizeRequest,
  TokenizeResponse,
  VocabularyRequest,
  VocabularyResponse,
  DecodeRequest,
  DecodeResponse,
  CompareRequest,
  CompareResponse,
  TokenizerTypeInfo,
} from '../types/tokenizer';

/**
 * Convert camelCase to snake_case for backend requests.
 */
function toSnakeCase(obj: any): any {
  if (obj === null || typeof obj !== 'object') {
    return obj;
  }

  if (Array.isArray(obj)) {
    return obj.map(toSnakeCase);
  }

  const result: any = {};
  for (const key in obj) {
    if (obj.hasOwnProperty(key)) {
      const snakeKey = key.replace(/[A-Z]/g, (letter) => `_${letter.toLowerCase()}`);
      result[snakeKey] = toSnakeCase(obj[key]);
    }
  }
  return result;
}

/**
 * Convert snake_case to camelCase for backend responses.
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
      // Handle special case for is_special -> isSpecial
      if (key === 'is_special') {
        result.isSpecial = toCamelCase(obj[key]);
      }
      // Handle special case for token_id -> tokenId
      else if (key === 'token_id') {
        result.tokenId = toCamelCase(obj[key]);
      }
      // Handle special case for start_position -> startPosition
      else if (key === 'start_position') {
        result.startPosition = toCamelCase(obj[key]);
      }
      // Handle special case for end_position -> endPosition
      else if (key === 'end_position') {
        result.endPosition = toCamelCase(obj[key]);
      }
      // Handle special case for vocab_size -> vocabSize
      else if (key === 'vocab_size') {
        result.vocabSize = toCamelCase(obj[key]);
      }
      // Handle special case for special_tokens -> specialTokens
      else if (key === 'special_tokens') {
        result.specialTokens = toCamelCase(obj[key]);
      }
      // Handle special case for tokenizer_type -> tokenizerType
      else if (key === 'tokenizer_type') {
        result.tokenizerType = toCamelCase(obj[key]);
      }
      // Handle special case for token_ids -> tokenIds
      else if (key === 'token_ids') {
        result.tokenIds = toCamelCase(obj[key]);
      }
      // Handle special case for token_details -> tokenDetails
      else if (key === 'token_details') {
        result.tokenDetails = toCamelCase(obj[key]);
      }
      // Handle special case for vocabulary_summary -> vocabularySummary
      else if (key === 'vocabulary_summary') {
        result.vocabularySummary = toCamelCase(obj[key]);
      }
      // Handle special case for training_texts -> trainingTexts
      else if (key === 'training_texts') {
        result.trainingTexts = toCamelCase(obj[key]);
      }
      // Handle special case for num_merges -> numMerges
      else if (key === 'num_merges') {
        result.numMerges = toCamelCase(obj[key]);
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
 * Tokenizer API client.
 */
export const tokenizerApi = {
  /**
   * Get available tokenizer types with descriptions.
   */
  getTokenizerTypes: async (): Promise<{ success: boolean; tokenizerTypes: TokenizerTypeInfo[] }> => {
    const response = await api.get('/api/v1/tokenizer/types');
    return response.data;
  },

  /**
   * Tokenize text using the specified tokenizer.
   *
   * @param request - Tokenization request with text, tokenizer type, and options
   * @returns Tokenization response with token IDs, tokens, and metadata
   */
  tokenize: async (request: TokenizeRequest): Promise<TokenizeResponse> => {
    const snakeCaseRequest = toSnakeCase(request);
    const response = await api.post('/api/v1/tokenizer/tokenize', snakeCaseRequest);
    return toCamelCase(response.data) as TokenizeResponse;
  },

  /**
   * Get the vocabulary for a tokenizer.
   *
   * @param request - Vocabulary request with tokenizer type and options
   * @returns Vocabulary response with all tokens and their information
   */
  getVocabulary: async (request: VocabularyRequest): Promise<VocabularyResponse> => {
    const snakeCaseRequest = toSnakeCase(request);
    const response = await api.post('/api/v1/tokenizer/vocabulary', snakeCaseRequest);
    return toCamelCase(response.data) as VocabularyResponse;
  },

  /**
   * Get detailed information about a specific token.
   *
   * @param tokenId - Token ID to look up
   * @param tokenizerType - Type of tokenizer
   * @param vocabSize - Maximum vocabulary size
   * @param trainingTexts - Optional training texts
   * @returns Token detail response
   */
  getTokenDetail: async (
    tokenId: number,
    tokenizerType: TokenType,
    vocabSize: number = 10000,
    trainingTexts?: string[]
  ): Promise<{ success: boolean; tokenInfo: any; tokenizerType: TokenType; error?: string }> => {
    const response = await api.post('/api/v1/tokenizer/token-detail', {
      token_id: tokenId,
      tokenizer_type: tokenizerType,
      vocab_size: vocabSize,
      training_texts: trainingTexts,
    });
    return response.data;
  },

  /**
   * Decode token IDs back into text.
   *
   * @param request - Decode request with token IDs and tokenizer type
   * @returns Decoded text
   */
  decode: async (request: DecodeRequest): Promise<DecodeResponse> => {
    const snakeCaseRequest = toSnakeCase(request);
    const response = await api.post('/api/v1/tokenizer/decode', snakeCaseRequest);
    return toCamelCase(response.data) as DecodeResponse;
  },

  /**
   * Compare different tokenizers on the same text.
   *
   * @param request - Comparison request with text and options
   * @returns Comparison results for each tokenizer type
   */
  compare: async (request: CompareRequest): Promise<CompareResponse> => {
    const snakeCaseRequest = toSnakeCase(request);
    const response = await api.post('/api/v1/tokenizer/compare', snakeCaseRequest);
    return toCamelCase(response.data) as CompareResponse;
  },
};

export default tokenizerApi;