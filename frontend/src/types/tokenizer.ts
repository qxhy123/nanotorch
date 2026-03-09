/**
 * TypeScript types for tokenizer functionality.
 *
 * This module defines all types related to tokenization, vocabulary management,
 * and tokenizer configuration.
 */

/**
 * Supported tokenizer types.
 * - char: Character-level tokenization (splits text into individual characters)
 * - word: Word-level tokenization (splits text into words using regex)
 * - bpe: Byte Pair Encoding (subword tokenization with learned merge operations)
 */
export type TokenType = 'char' | 'word' | 'bpe';

/**
 * Display information for tokenizer types.
 */
export interface TokenizerTypeInfo {
  type: TokenType;
  name: string;
  description: string;
}

/**
 * Position information for a token within the original text.
 * Used to highlight which part of the text corresponds to which token.
 */
export interface TokenPosition {
  startPosition: number;
  endPosition: number;
}

/**
 * Detailed information about a single token.
 */
export interface TokenData {
  /** Unique identifier for this token in the vocabulary */
  id: number;
  /** The text string that this token represents */
  text: string;
  /** Frequency of this token in the training corpus */
  frequency: number;
  /** Whether this is a special token (e.g., <pad>, <unk>, <sos>, <eos>) */
  isSpecial: boolean;
  /** Starting position in the original text (if applicable) */
  startPosition?: number | null;
  /** Ending position in the original text (if applicable) */
  endPosition?: number | null;
  /** Embedding vector for this token (if available) */
  embedding?: number[];
}

/**
 * Token information returned from the tokenize endpoint.
 */
export interface TokenInfo {
  tokenId: number;
  text: string;
  frequency: number;
  isSpecial: boolean;
  is_special: boolean;
  startPosition: number | null;
  endPosition: number | null;
  start_position: number | null;
  end_position: number | null;
}

/**
 * Summary statistics about the vocabulary.
 */
export interface VocabularySummary {
  /** Total number of tokens in the vocabulary */
  vocabSize: number;
  /** Special token mappings */
  specialTokens: SpecialTokens;
  /** Most frequent tokens in the vocabulary */
  mostCommon: Array<{
    id: number;
    text: string;
    frequency: number;
  }>;
}

/**
 * Special token IDs used by the tokenizer.
 */
export interface SpecialTokens {
  /** Padding token ID */
  pad: number;
  /** Unknown token ID (for out-of-vocabulary tokens) */
  unk: number;
  /** Start of sequence token ID */
  sos: number;
  /** End of sequence token ID */
  eos: number;
}

/**
 * Complete vocabulary data structure.
 */
export interface VocabularyData {
  /** Total size of the vocabulary */
  size: number;
  /** All tokens in the vocabulary */
  tokens: TokenData[];
  /** Type of tokenizer used */
  tokenizerType: TokenType;
  /** Special token mappings */
  specialTokens: SpecialTokens;
}

/**
 * Result of tokenizing a text.
 */
export interface TokenizationResult {
  /** List of token IDs */
  tokenIds: number[];
  /** List of token strings (in the same order as token IDs) */
  tokens: string[];
  /** Detailed information about each token */
  tokenDetails: TokenInfo[];
  /** Summary of the vocabulary used */
  vocabularySummary: VocabularySummary;
  /** Type of tokenizer that was used */
  tokenizerType: TokenType;
}

/**
 * Request to tokenize text.
 */
export interface TokenizeRequest {
  /** Text to tokenize */
  text: string;
  /** Type of tokenizer to use */
  tokenizerType: TokenType;
  /** Maximum vocabulary size */
  vocabSize?: number;
  /** Number of BPE merge operations (only for BPE tokenizer) */
  numMerges?: number;
  /** Optional training texts (if not provided, uses default corpus) */
  trainingTexts?: string[];
}

/**
 * Response from the tokenize endpoint.
 */
export interface TokenizeResponse {
  /** Whether the tokenization was successful */
  success: boolean;
  /** List of token IDs */
  tokenIds: number[];
  /** List of token strings */
  tokens: string[];
  /** Detailed token information */
  tokenDetails: TokenInfo[];
  /** Vocabulary summary */
  vocabularySummary: VocabularySummary;
  /** Type of tokenizer used */
  tokenizerType: TokenType;
  /** Error message if failed */
  error?: string;
}

/**
 * Request to get vocabulary.
 */
export interface VocabularyRequest {
  /** Type of tokenizer */
  tokenizerType: TokenType;
  /** Maximum vocabulary size */
  vocabSize?: number;
  /** Optional training texts */
  trainingTexts?: string[];
}

/**
 * Response from the vocabulary endpoint.
 */
export interface VocabularyResponse {
  /** Whether the request was successful */
  success: boolean;
  /** Total vocabulary size */
  vocabSize: number;
  /** All tokens in the vocabulary */
  tokens: Array<{
    id: number;
    text: string;
    frequency: number;
    isSpecial: boolean;
  }>;
  /** Special token mappings */
  specialTokens: SpecialTokens;
  /** Type of tokenizer */
  tokenizerType: TokenType;
  /** Error message if failed */
  error?: string;
}

/**
 * Request to decode token IDs.
 */
export interface DecodeRequest {
  /** Token IDs to decode */
  tokenIds: number[];
  /** Type of tokenizer */
  tokenizerType: TokenType;
  /** Maximum vocabulary size */
  vocabSize?: number;
  /** Optional training texts */
  trainingTexts?: string[];
}

/**
 * Response from the decode endpoint.
 */
export interface DecodeResponse {
  /** Whether decoding was successful */
  success: boolean;
  /** Decoded text */
  text: string;
  /** Type of tokenizer used */
  tokenizer_type: TokenType;
  /** Error message if failed */
  error?: string;
}

/**
 * Request to compare tokenizers.
 */
export interface CompareRequest {
  /** Text to tokenize */
  text: string;
  /** Maximum vocabulary size for all tokenizers */
  vocabSize?: number;
  /** Number of BPE merge operations */
  numMerges?: number;
  /** Optional training texts */
  trainingTexts?: string[];
}

/**
 * Result from comparing different tokenizers.
 */
export interface ComparisonResult {
  /** Type of tokenizer */
  tokenizerType: TokenType;
  /** Number of tokens produced */
  numTokens: number;
  /** Token strings */
  tokens: string[];
  /** Token IDs */
  tokenIds: number[];
  /** Number of out-of-vocabulary tokens */
  oovCount: number;
}

/**
 * Response from the compare endpoint.
 */
export interface CompareResponse {
  /** Whether comparison was successful */
  success: boolean;
  /** Comparison results for each tokenizer */
  comparisons: ComparisonResult[];
  /** Error message if failed */
  error?: string;
}

/**
 * Tokenizer state in the global store.
 */
export interface TokenizerState {
  /** Currently selected tokenizer type */
  tokenizerType: TokenType;
  /** Maximum vocabulary size */
  vocabSize: number;
  /** Number of BPE merges (for BPE tokenizer) */
  numMerges: number;
  /** Current vocabulary data */
  vocabularyData: VocabularyData | null;
  /** Latest tokenization result */
  tokenizationResult: TokenizationResult | null;
  /** Whether a tokenization is in progress */
  isLoading: boolean;
  /** Error message if tokenization failed */
  error: string | null;
}

/**
 * Tokenizer configuration options.
 */
export interface TokenizerConfig {
  /** Type of tokenizer */
  type: TokenType;
  /** Maximum vocabulary size */
  vocabSize: number;
  /** Number of BPE merges (only for BPE) */
  numMerges?: number;
  /** Whether to use pretrained vocabulary */
  usePretrained?: boolean;
}

/**
 * Token embedding data for visualization.
 */
export interface TokenEmbeddingData {
  /** Token ID */
  tokenId: number;
  /** Token text */
  text: string;
  /** Embedding vector */
  embedding: number[];
  /** 2D projection coordinates (for visualization) */
  projection?: {
    x: number;
    y: number;
  };
}

/**
 * Frequency distribution data for visualization.
 */
export interface FrequencyDistributionData {
  /** Token texts */
  labels: string[];
  /** Frequency values */
  frequencies: number[];
  /** Token IDs */
  tokenIds: number[];
}

/**
 * Search/filter options for vocabulary browser.
 */
export interface VocabularySearchOptions {
  /** Search query (matches token text) */
  query: string;
  /** Minimum frequency */
  minFrequency?: number;
  /** Maximum frequency */
  maxFrequency?: number;
  /** Whether to include special tokens */
  includeSpecialTokens: boolean;
  /** Sort order */
  sortBy: 'id' | 'frequency' | 'text';
  /** Sort direction */
  sortOrder: 'asc' | 'desc';
  /** Maximum number of results */
  limit?: number;
  /** Offset for pagination */
  offset?: number;
}

/**
 * Filtered vocabulary results.
 */
export interface FilteredVocabularyResult {
  /** Total number of matching tokens */
  total: number;
  /** Filtered tokens */
  tokens: TokenData[];
  /** Whether there are more results */
  hasMore: boolean;
}

/**
 * Tokenizer comparison metrics.
 */
export interface ComparisonMetrics {
  /** Average token sequence length */
  avgSequenceLength: number;
  /** Average vocabulary size used */
  avgVocabSize: number;
  /** Average OOV rate */
  avgOovRate: number;
  /** Compression ratio (tokens per character) */
  compressionRatio: number;
}

/**
 * Tokenizer health/availability information.
 */
export interface TokenizerHealthInfo {
  /** Whether the tokenizer is available */
  available: boolean;
  /** Supported tokenizer types */
  supportedTypes: TokenType[];
  /** Default tokenizer type */
  defaultType: TokenType;
  /** Maximum allowed vocabulary size */
  maxVocabSize: number;
  /** Minimum allowed vocabulary size */
  minVocabSize: number;
}

/**
 * Training corpus statistics.
 */
export interface CorpusStatistics {
  /** Total number of texts */
  textCount: number;
  /** Total number of characters */
  charCount: number;
  /** Estimated number of words */
  wordCount: number;
  /** Most frequent characters */
  topChars: Array<{ char: string; count: number }>;
  /** Most frequent words */
  topWords: Array<{ word: string; count: number }>;
}

/**
 * Event types for tokenizer-related events.
 */
export type TokenizerEventType =
  | 'tokenizer-type-changed'
  | 'vocabulary-loaded'
  | 'tokenization-complete'
  | 'tokenization-error'
  | 'tokenizer-trained';

/**
 * Tokenizer event payload.
 */
export interface TokenizerEvent {
  type: TokenizerEventType;
  payload: {
    tokenizerType?: TokenType;
    timestamp: number;
    data?: unknown;
  };
}

/**
 * Token color mapping for visualization.
 * Maps token types or properties to colors.
 */
export interface TokenColorScheme {
  /** Color for special tokens */
  special: string;
  /** Color for regular tokens */
  regular: string;
  /** Color for unknown/out-of-vocabulary tokens */
  unknown: string;
  /** Color for highlighted tokens */
  highlighted: string;
  /** Background color for tokens */
  background: string;
}

/**
 * Token visualization options.
 */
export interface TokenVisualizationOptions {
  /** Whether to show token IDs */
  showIds: boolean;
  /** Whether to show token frequencies */
  showFrequencies: boolean;
  /** Whether to show special tokens */
  showSpecialTokens: boolean;
  /** Whether to show position information */
  showPositions: boolean;
  /** Whether to show embeddings */
  showEmbeddings: boolean;
  /** Color scheme to use */
  colorScheme: TokenColorScheme;
  /** Maximum number of tokens to display */
  maxTokens?: number;
}