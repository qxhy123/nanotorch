import { tokenizerApi } from './tokenizerApi';
import type {
  TokenInfo,
  TokenType,
  TokenizationResult,
} from '../types/tokenizer';

export interface TransformerTokenizationRequest {
  text: string;
  tokenizerType: TokenType;
  tokenizerVocabSize: number;
  tokenizerNumMerges: number;
  modelVocabSize: number;
  maxSeqLen?: number;
  trainingTexts?: string[];
}

export interface TransformerTokenizationResult {
  tokenization: TokenizationResult;
  effectiveVocabSize: number;
  truncatedTokenIds: number[];
  truncatedTokens: string[];
  truncatedTokenDetails: TokenInfo[];
  wasTruncated: boolean;
}

export function buildTokenizerTrainingTexts(...texts: Array<string | undefined>): string[] | undefined {
  const uniqueTexts = texts
    .map((text) => text?.trim())
    .filter((text): text is string => Boolean(text));

  return uniqueTexts.length > 0 ? Array.from(new Set(uniqueTexts)) : undefined;
}

export function getEffectiveTokenizerVocabSize(
  tokenizerVocabSize: number,
  modelVocabSize: number
): number {
  return Math.max(100, Math.min(tokenizerVocabSize, modelVocabSize));
}

function truncateTokenizationResult(
  result: TokenizationResult,
  maxSeqLen: number
): Pick<
  TransformerTokenizationResult,
  'truncatedTokenIds' | 'truncatedTokens' | 'truncatedTokenDetails' | 'wasTruncated'
> {
  const truncatedTokenIds = result.tokenIds.slice(0, maxSeqLen);
  const truncatedTokens = result.tokens.slice(0, maxSeqLen);
  const truncatedTokenDetails = result.tokenDetails.slice(0, maxSeqLen);

  return {
    truncatedTokenIds,
    truncatedTokens,
    truncatedTokenDetails,
    wasTruncated: result.tokenIds.length > maxSeqLen,
  };
}

export async function tokenizeForTransformer(
  request: TransformerTokenizationRequest
): Promise<TransformerTokenizationResult> {
  const effectiveVocabSize = getEffectiveTokenizerVocabSize(
    request.tokenizerVocabSize,
    request.modelVocabSize
  );

  const response = await tokenizerApi.tokenize({
    text: request.text,
    tokenizerType: request.tokenizerType,
    vocabSize: effectiveVocabSize,
    numMerges: request.tokenizerNumMerges,
    trainingTexts: request.trainingTexts,
  });

  if (!response.success) {
    throw new Error(response.error || 'Tokenization failed');
  }

  const tokenization: TokenizationResult = {
    tokenIds: response.tokenIds,
    tokens: response.tokens,
    tokenDetails: response.tokenDetails,
    vocabularySummary: response.vocabularySummary,
    tokenizerType: response.tokenizerType,
  };

  const maxSeqLen = request.maxSeqLen ?? tokenization.tokenIds.length;

  return {
    tokenization,
    effectiveVocabSize,
    ...truncateTokenizationResult(tokenization, maxSeqLen),
  };
}
