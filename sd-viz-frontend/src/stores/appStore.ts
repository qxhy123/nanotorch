/**
 * Global application state management with Zustand
 */

import { create } from 'zustand';
import type { AppState, GenerationConfig } from '../types';

interface AppStore extends AppState {
  // Actions
  setCurrentRoute: (route: string) => void;
  setTheme: (theme: 'light' | 'dark') => void;

  // Diffusion actions
  setTimestep: (t: number) => void;
  setIsPlaying: (playing: boolean) => void;

  // Generation actions
  setConfig: (config: Partial<GenerationConfig>) => void;
  setIsGenerating: (generating: boolean) => void;
  setGenerationProgress: (progress: number) => void;

  // Tutorial actions
  completeLesson: (lessonId: string) => void;
  setCurrentLesson: (lessonId: string) => void;
}

export const useAppStore = create<AppStore>((set) => ({
  // Initial state
  currentRoute: '/',
  language: 'en',
  theme: 'light',

  // Diffusion state
  timestep: 0,
  isPlaying: false,

  // Generation state
  config: {
    prompt: '',
    negativePrompt: '',
    width: 512,
    height: 512,
    steps: 50,
    cfgScale: 7.5,
    seed: 42,
    sampler: 'euler',
    scheduler: 'euler',
  },
  isGenerating: false,
  generationProgress: 0,

  // Tutorial state
  completedLessons: [],
  currentLesson: undefined,

  // Actions
  setCurrentRoute: (route) => set({ currentRoute: route }),
  setTheme: (theme) => set({ theme }),

  setTimestep: (t) => set({ timestep: t }),
  setIsPlaying: (playing) => set({ isPlaying: playing }),

  setConfig: (config) => set((state) => ({
    config: { ...state.config, ...config }
  })),
  setIsGenerating: (generating) => set({ isGenerating: generating }),
  setGenerationProgress: (progress) => set({ generationProgress: progress }),

  completeLesson: (lessonId) => set((state) => ({
    completedLessons: [...state.completedLessons, lessonId]
  })),
  setCurrentLesson: (lessonId) => set({ currentLesson: lessonId }),
}));
