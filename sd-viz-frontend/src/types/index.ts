/**
 * Global type definitions for SD-Viz
 */

// Stable Diffusion Components
export interface SDComponent {
  id: string;
  name: string;
  nameCN: string;
  type: 'text-encoder' | 'unet' | 'vae';
  description: string;
  descriptionCN: string;
  parameters: Record<string, number | string>;
  inputShape?: number[];
  outputShape?: number[];
}

// Diffusion Process
export interface DiffusionStep {
  timestep: number;
  latent: number[];        // [4, 64, 64] flattened
  noise: number[];         // predicted/actual noise
  denoised: number[];      // denoised latent
  alpha_bar: number;       // cumulative noise schedule
  snr: number;             // signal-to-noise ratio
}

export interface DiffusionProcess {
  id: string;
  prompt: string;
  seed: number;
  steps: DiffusionStep[];
  finalImage: string;      // base64
}

// UNet Architecture
export interface UNetLayer {
  name: string;
  type: 'resnet' | 'attention' | 'downsample' | 'upsample' | 'middle';
  inputShape: [number, number, number, number];  // [B, C, H, W]
  outputShape: [number, number, number, number];
  channels: number;
  parameters: number;
  activation?: string;
  hasAttention: boolean;
}

// Attention
export interface AttentionData {
  timestep: number;
  layer: number;        // 0-15
  head: number;         // 0-7
  type: 'self' | 'cross';
  attentionMap: number[][];  // [4096, 4096] for self, [4096, 77] for cross
  tokens: string[];
  spatialSize: [number, number];  // [64, 64]
}

// Text Encoder
export interface TokenData {
  id: number;
  text: string;
  embedding: number[];   // [768]
  attentionWeight: number;
}

// Sampling
export interface SamplingMethod {
  id: string;
  name: string;
  displayName: string;
  displayNameCN: string;
  description: string;
  descriptionCN: string;
  minSteps: number;
  recommendedSteps: number;
  deterministic: boolean;
  speed: 'fast' | 'medium' | 'slow';
  quality: number;  // 1-10
}

export interface SamplingResult {
  method: string;
  steps: number;
  time: number;        // milliseconds
  fid?: number;
  clipScore?: number;
  image: string;
}

// Latent Space
export interface LatentVector {
  id: string;
  prompt: string;
  latent: number[];     // [4, 64, 64]
  projection2D: [number, number];  // PCA/t-SNE
  category?: string;
}

// Control
export interface ControlInput {
  type: 'depth' | 'canny' | 'pose' | 'seg' | 'scribble';
  image: string;  // base64
  weight: number;
  mode: 'controlnet' | 't2i-adapter' | 'ip-adapter';
}

// Generation
export interface GenerationConfig {
  prompt: string;
  negativePrompt?: string;
  width: 256 | 512 | 768 | 1024;
  height: 256 | 512 | 768 | 1024;
  steps: number;
  cfgScale: number;
  seed: number;
  sampler: SamplingMethod['id'];
  scheduler: 'euler' | 'ddim' | 'dpmsolver' | 'lms';
}

// UI State
export interface AppState {
  // Navigation
  currentRoute: string;
  language: 'en' | 'cn';

  // Theme
  theme: 'light' | 'dark';

  // Diffusion
  timestep: number;
  isPlaying: boolean;

  // Generation
  config: Partial<GenerationConfig>;
  isGenerating: boolean;
  generationProgress: number;

  // Tutorial
  completedLessons: string[];
  currentLesson?: string;
}

// Utility types
export type Nullable<T> = T | null;
export type Optional<T> = T | undefined;
export type Dict<T> = Record<string, T>;
