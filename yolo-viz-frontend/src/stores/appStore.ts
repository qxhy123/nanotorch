/**
 * Global application state management with Zustand
 */

import { create } from 'zustand';
import type { AppState, YOLOVersion, NeckType, HeadType, BoundingBox } from '../types';

interface AppStore extends AppState {
  // Navigation actions
  setCurrentRoute: (route: string) => void;
  setTheme: (theme: 'light' | 'dark') => void;

  // NMS actions
  setNmsStep: (step: number) => void;
  setIsPlaying: (playing: boolean) => void;
  setConfidenceThreshold: (threshold: number) => void;
  setIouThreshold: (threshold: number) => void;

  // Detection actions
  setSelectedVersion: (version: YOLOVersion) => void;
  setDetectionResults: (results: BoundingBox[]) => void;

  // Architecture actions
  setSelectedBackbone: (backbone: 'cspdarknet53' | 'darknet53') => void;
  setSelectedNeck: (neck: NeckType) => void;
  setSelectedHead: (head: HeadType) => void;
}

export const useAppStore = create<AppStore>((set) => ({
  // Initial state
  currentRoute: '/',
  theme: 'light',

  // NMS state
  nmsStep: 0,
  isPlaying: false,
  confidenceThreshold: 0.5,
  iouThreshold: 0.45,

  // Detection state
  selectedVersion: 'v8',
  detectionResults: [],

  // Backbone state
  selectedBackbone: 'cspdarknet53',

  // Neck state
  selectedNeck: 'panet',

  // Head state
  selectedHead: 'anchor-free',

  // Actions
  setCurrentRoute: (route) => set({ currentRoute: route }),
  setTheme: (theme) => set({ theme }),

  setNmsStep: (step) => set({ nmsStep: step }),
  setIsPlaying: (playing) => set({ isPlaying: playing }),
  setConfidenceThreshold: (threshold) => set({ confidenceThreshold: threshold }),
  setIouThreshold: (threshold) => set({ iouThreshold: threshold }),

  setSelectedVersion: (version) => set({ selectedVersion: version }),
  setDetectionResults: (results) => set({ detectionResults: results }),

  setSelectedBackbone: (backbone) => set({ selectedBackbone: backbone }),
  setSelectedNeck: (neck) => set({ selectedNeck: neck }),
  setSelectedHead: (head) => set({ selectedHead: head }),
}));
