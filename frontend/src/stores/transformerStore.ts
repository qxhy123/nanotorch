import { create } from 'zustand';
import type {
  TransformerConfig,
  LayerOutputData,
  AttentionData,
  ComponentVisualizationState,
  AnimationState,
} from '../types/transformer';

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

interface TransformerState {
  // Model configuration
  config: TransformerConfig;

  // Input data
  inputText: string;
  targetText: string;
  tokens: number[];
  targetTokens: number[];

  // Computation results
  output: any; // TransformerOutput | null
  embeddings: LayerOutputData | null;
  attentionWeights: AttentionData[] | null;
  layerOutputs: LayerOutputData[] | null;

  // UI state
  visualizationState: ComponentVisualizationState;
  animationState: AnimationState;
  isLoading: boolean;
  error: string | null;

  // Actions
  setConfig: (config: Partial<TransformerConfig>) => void;
  setInputText: (text: string) => void;
  setTargetText: (text: string) => void;
  setTokens: (tokens: number[]) => void;
  setTargetTokens: (tokens: number[]) => void;
  setOutput: (output: any) => void;
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
    set({
      output,
      embeddings: output.data?.embeddings ? {
        layerName: 'embeddings',
        layerType: 'embedding',
        inputShape: [],
        outputShape: output.data.embeddings.combined.shape,
        output: output.data.embeddings.combined,
      } : null,
      // Backend returns snake_case, so we need to check for attention_weights
      attentionWeights: output.data?.attention_weights || output.data?.attentionWeights || null,
      // Backend returns snake_case
      layerOutputs: output.data?.layer_outputs || output.data?.layerOutputs || null,
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
    }),
}));
