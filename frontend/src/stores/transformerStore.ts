import { create } from 'zustand';
import type {
  TransformerConfig,
  TransformerOutput,
  LayerOutputData,
  AttentionData,
  ComponentVisualizationState,
  AnimationState,
  TutorialState,
  AnimationTimeline,
  AttentionStage,
} from '../types/transformer';
import type {
  TokenType,
  VocabularyData,
  TokenizationResult,
} from '../types/tokenizer';

const DEFAULT_CONFIG: TransformerConfig = {
  d_model: 512,
  nhead: 8,
  num_encoder_layers: 6,
  num_decoder_layers: 6,
  dim_feedforward: 2048,
  dropout: 0.1,
  activation: 'relu',
  max_seq_len: 128,
  vocab_size: 10000,
  layer_norm_eps: 1e-5,
  batch_first: true,
  norm_first: false,
};

const DEFAULT_VISUALIZATION_STATE: ComponentVisualizationState = {
  selectedLayer: 0,
  selectedHead: 0,
  selectedComponent: 'encoder',
  showValues: false,
  colorScheme: 'blues',
};

const DEFAULT_ANIMATION_STATE: AnimationState = {
  isPlaying: false,
  currentStep: 0,
  speed: 1000,
};

const DEFAULT_TUTORIAL_STATE: TutorialState = {
  activeTutorial: null,
  currentStep: 0,
  isTutorialActive: false,
  completedTutorials: [],
  skippedTutorials: [],
};

const DEFAULT_ATTENTION_STAGE: AttentionStage = 'softmax';

// Tokenizer defaults
const DEFAULT_TOKENIZER_TYPE: TokenType = 'char';
const DEFAULT_VOCAB_SIZE = 10000;
const DEFAULT_NUM_MERGES = 1000;

interface TransformerState {
  // Model configuration
  config: TransformerConfig;

  // Input data
  inputText: string;
  targetText: string;
  tokens: number[];
  targetTokens: number[];

  // Computation results
  output: TransformerOutput | null;
  embeddings: LayerOutputData | null;
  attentionWeights: AttentionData[] | null;
  layerOutputs: LayerOutputData[] | null;

  // UI state
  visualizationState: ComponentVisualizationState;
  animationState: AnimationState;
  isLoading: boolean;
  error: string | null;

  // New states for Transformer Visualization 2.0
  tutorialState: TutorialState;
  animationTimelines: Record<string, AnimationTimeline>;
  activeTimeline: string | null;
  attentionComputationStage: AttentionStage;

  // Tokenizer states
  tokenizerType: TokenType;
  tokenizerVocabSize: number;
  tokenizerNumMerges: number;
  vocabularyData: VocabularyData | null;
  tokenizationResult: TokenizationResult | null;
  isTokenizing: boolean;
  tokenizationError: string | null;

  // Actions
  setConfig: (config: Partial<TransformerConfig>) => void;
  setInputText: (text: string) => void;
  setTargetText: (text: string) => void;
  setTokens: (tokens: number[]) => void;
  setTargetTokens: (tokens: number[]) => void;
  setOutput: (output: TransformerOutput | null) => void;
  setSelectedLayer: (layer: number) => void;
  setSelectedHead: (head: number) => void;
  setSelectedComponent: (component: string) => void;
  setShowValues: (show: boolean) => void;
  setColorScheme: (scheme: 'viridis' | 'plasma' | 'inferno' | 'blues' | 'reds') => void;
  setIsPlaying: (playing: boolean) => void;
  setCurrentStep: (step: number) => void;
  setAnimationSpeed: (speed: number) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  reset: () => void;

  // New actions for Transformer Visualization 2.0
  setActiveTutorial: (tutorialId: string | null) => void;
  setTutorialStep: (step: number) => void;
  setIsTutorialActive: (active: boolean) => void;
  completeTutorial: (tutorialId: string) => void;
  skipTutorial: (tutorialId: string) => void;
  registerTimeline: (timeline: AnimationTimeline) => void;
  setActiveTimeline: (timelineId: string | null) => void;
  updateTimelineProgress: (timelineId: string, progress: number) => void;
  setAttentionComputationStage: (stage: AttentionStage) => void;

  // Tokenizer actions
  setTokenizerType: (type: TokenType) => void;
  setTokenizerVocabSize: (size: number) => void;
  setTokenizerNumMerges: (merges: number) => void;
  setVocabularyData: (data: VocabularyData | null) => void;
  setTokenizationResult: (result: TokenizationResult | null) => void;
  setIsTokenizing: (isTokenizing: boolean) => void;
  setTokenizationError: (error: string | null) => void;
  resetTokenizer: () => void;
}

export const useTransformerStore = create<TransformerState>((set) => ({
  // Initial state
  config: DEFAULT_CONFIG,
  inputText: 'The quick brown fox jumps over the lazy dog.',
  targetText: '',
  tokens: [],
  targetTokens: [],
  output: null,
  embeddings: null,
  attentionWeights: null,
  layerOutputs: null,
  visualizationState: DEFAULT_VISUALIZATION_STATE,
  animationState: DEFAULT_ANIMATION_STATE,
  isLoading: false,
  error: null,

  // New states for Transformer Visualization 2.0
  tutorialState: DEFAULT_TUTORIAL_STATE,
  animationTimelines: {},
  activeTimeline: null,
  attentionComputationStage: DEFAULT_ATTENTION_STAGE,

  // Tokenizer states
  tokenizerType: DEFAULT_TOKENIZER_TYPE,
  tokenizerVocabSize: DEFAULT_VOCAB_SIZE,
  tokenizerNumMerges: DEFAULT_NUM_MERGES,
  vocabularyData: null,
  tokenizationResult: null,
  isTokenizing: false,
  tokenizationError: null,

  // Actions
  setConfig: (newConfig) =>
    set((state) => {
      const mergedConfig = { ...state.config, ...newConfig };

      // Auto-adjust selectedHead if nhead changed
      let newSelectedHead = state.visualizationState.selectedHead;
      if (newConfig.nhead !== undefined && newSelectedHead >= newConfig.nhead) {
        newSelectedHead = Math.max(0, newConfig.nhead - 1);
      }

      // Auto-adjust selectedLayer if num_encoder_layers changed
      let newSelectedLayer = state.visualizationState.selectedLayer;
      if (newConfig.num_encoder_layers !== undefined && newSelectedLayer >= newConfig.num_encoder_layers) {
        newSelectedLayer = Math.max(0, newConfig.num_encoder_layers - 1);
      }

      return {
        config: mergedConfig,
        visualizationState: {
          ...state.visualizationState,
          selectedHead: newSelectedHead,
          selectedLayer: newSelectedLayer,
        },
      };
    }),

  setInputText: (text) =>
    set({ inputText: text, error: null }),

  setTargetText: (text) =>
    set({ targetText: text }),

  setTokens: (tokens) =>
    set({ tokens }),

  setTargetTokens: (tokens) =>
    set({ targetTokens: tokens }),

  setOutput: (output) => {
    const data = output?.data as (
      TransformerOutput['data'] & {
        attention_weights?: AttentionData[];
        layer_outputs?: LayerOutputData[];
      }
    ) | undefined;

    set({
      output,
      embeddings: data?.embeddings ? {
        layerName: 'embeddings',
        layerType: 'embedding',
        inputShape: [],
        outputShape: data.embeddings.combined.shape,
        output: data.embeddings.combined,
      } : null,
      attentionWeights: data?.attentionWeights || data?.attention_weights || null,
      layerOutputs: data?.layerOutputs || data?.layer_outputs || null,
    });
  },

  setSelectedLayer: (layer) =>
    set((state) => ({
      visualizationState: {
        ...state.visualizationState,
        selectedLayer: Math.max(0, Math.min(layer, state.config.num_encoder_layers - 1)),
      },
    })),

  setSelectedHead: (head) =>
    set((state) => ({
      visualizationState: {
        ...state.visualizationState,
        selectedHead: Math.max(0, Math.min(head, state.config.nhead - 1)),
      },
    })),

  setSelectedComponent: (component) =>
    set((state) => ({
      visualizationState: {
        ...state.visualizationState,
        selectedComponent: component,
      },
    })),

  setShowValues: (show) =>
    set((state) => ({
      visualizationState: {
        ...state.visualizationState,
        showValues: show,
      },
    })),

  setColorScheme: (scheme) =>
    set((state) => ({
      visualizationState: {
        ...state.visualizationState,
        colorScheme: scheme,
      },
    })),

  setIsPlaying: (playing) =>
    set((state) => ({
      animationState: {
        ...state.animationState,
        isPlaying: playing,
      },
    })),

  setCurrentStep: (step) =>
    set((state) => ({
      animationState: {
        ...state.animationState,
        currentStep: Math.max(0, step),
      },
    })),

  setAnimationSpeed: (speed) =>
    set((state) => ({
      animationState: {
        ...state.animationState,
        speed: Math.max(100, speed),
      },
    })),

  setLoading: (loading) => {
    set({ isLoading: loading, error: null });
  },

  setError: (error) =>
    set({ error }),

  reset: () =>
    set({
      config: DEFAULT_CONFIG,
      inputText: 'The quick brown fox jumps over the lazy dog.',
      targetText: '',
      tokens: [],
      targetTokens: [],
      output: null,
      embeddings: null,
      attentionWeights: null,
      layerOutputs: null,
      visualizationState: DEFAULT_VISUALIZATION_STATE,
      animationState: DEFAULT_ANIMATION_STATE,
      isLoading: false,
      error: null,
      tutorialState: DEFAULT_TUTORIAL_STATE,
      animationTimelines: {},
      activeTimeline: null,
      attentionComputationStage: DEFAULT_ATTENTION_STAGE,
      tokenizerType: DEFAULT_TOKENIZER_TYPE,
      tokenizerVocabSize: DEFAULT_VOCAB_SIZE,
      tokenizerNumMerges: DEFAULT_NUM_MERGES,
      vocabularyData: null,
      tokenizationResult: null,
      isTokenizing: false,
      tokenizationError: null,
    }),

  // New actions for Transformer Visualization 2.0

  setActiveTutorial: (tutorialId) =>
    set((state) => ({
      tutorialState: {
        ...state.tutorialState,
        activeTutorial: tutorialId,
        currentStep: tutorialId ? 0 : state.tutorialState.currentStep,
      },
    })),

  setTutorialStep: (step) =>
    set((state) => ({
      tutorialState: {
        ...state.tutorialState,
        currentStep: Math.max(0, step),
      },
    })),

  setIsTutorialActive: (active) =>
    set((state) => ({
      tutorialState: {
        ...state.tutorialState,
        isTutorialActive: active,
      },
    })),

  completeTutorial: (tutorialId) =>
    set((state) => ({
      tutorialState: {
        ...state.tutorialState,
        completedTutorials: state.tutorialState.completedTutorials.includes(tutorialId)
          ? state.tutorialState.completedTutorials
          : [...state.tutorialState.completedTutorials, tutorialId],
        activeTutorial: null,
        isTutorialActive: false,
      },
    })),

  skipTutorial: (tutorialId) =>
    set((state) => ({
      tutorialState: {
        ...state.tutorialState,
        skippedTutorials: state.tutorialState.skippedTutorials.includes(tutorialId)
          ? state.tutorialState.skippedTutorials
          : [...state.tutorialState.skippedTutorials, tutorialId],
        activeTutorial: null,
        isTutorialActive: false,
      },
    })),

  registerTimeline: (timeline) =>
    set((state) => ({
      animationTimelines: {
        ...state.animationTimelines,
        [timeline.id]: timeline,
      },
    })),

  setActiveTimeline: (timelineId) =>
    set({ activeTimeline: timelineId }),

  updateTimelineProgress: (timelineId, progress) =>
    set((state) => ({
      animationTimelines: {
        ...state.animationTimelines,
        [timelineId]: state.animationTimelines[timelineId]
          ? { ...state.animationTimelines[timelineId], progress }
          : state.animationTimelines[timelineId],
      },
    })),

  setAttentionComputationStage: (stage) =>
    set({ attentionComputationStage: stage }),

  // Tokenizer actions

  setTokenizerType: (type) =>
    set({ tokenizerType: type, vocabularyData: null, tokenizationResult: null, tokenizationError: null }),

  setTokenizerVocabSize: (size) =>
    set({ tokenizerVocabSize: Math.max(100, Math.min(100000, size)) }),

  setTokenizerNumMerges: (merges) =>
    set({ tokenizerNumMerges: Math.max(10, Math.min(10000, merges)) }),

  setVocabularyData: (data) =>
    set({ vocabularyData: data }),

  setTokenizationResult: (result) =>
    set({ tokenizationResult: result }),

  setIsTokenizing: (isTokenizing) =>
    set({ isTokenizing, tokenizationError: isTokenizing ? null : undefined }),

  setTokenizationError: (error) =>
    set({ tokenizationError: error, isTokenizing: false }),

  resetTokenizer: () =>
    set({
      tokenizerType: DEFAULT_TOKENIZER_TYPE,
      tokenizerVocabSize: DEFAULT_VOCAB_SIZE,
      tokenizerNumMerges: DEFAULT_NUM_MERGES,
      vocabularyData: null,
      tokenizationResult: null,
      isTokenizing: false,
      tokenizationError: null,
    }),
}));
