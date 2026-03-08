import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import type { Tutorial, TutorialStep, TutorialState } from '../../types/transformer';
import { useTransformerStore } from '../../stores/transformerStore';

/**
 * Tutorial Context
 */
interface TutorialContextType {
  state: TutorialState;
  tutorials: Record<string, Tutorial>;
  activeTutorial: Tutorial | null;
  currentStepData: TutorialStep | null;
  startTutorial: (tutorialId: string) => void;
  endTutorial: (completed?: boolean) => void;
  nextStep: () => void;
  previousStep: () => void;
  goToStep: (stepIndex: number) => void;
  skipTutorial: () => void;
  resetProgress: () => void;
}

const TutorialContext = createContext<TutorialContextType | undefined>(undefined);

/**
 * Tutorial Provider Props
 */
interface TutorialProviderProps {
  children: React.ReactNode;
  tutorials: Tutorial[];
  autoStart?: boolean;
  autoStartTutorialId?: string;
  persistProgress?: boolean;
  onTutorialComplete?: (tutorialId: string) => void;
  onStepChange?: (tutorialId: string, stepIndex: number) => void;
}

/**
 * TutorialProvider Component
 *
 * Manages tutorial state and provides tutorial functionality.
 * Features:
 * - Tutorial registration and management
 * - Step navigation
 * - Progress tracking
 * - Completion tracking
 * - Keyboard shortcuts
 */
export const TutorialProvider: React.FC<TutorialProviderProps> = ({
  children,
  tutorials,
  autoStart = false,
  autoStartTutorialId,
  onTutorialComplete,
  onStepChange,
}) => {
  const {
    tutorialState,
    setActiveTutorial,
    setTutorialStep,
    setIsTutorialActive,
    completeTutorial,
    skipTutorial: storeSkipTutorial,
  } = useTransformerStore();

  const [tutorialsMap] = useState<Record<string, Tutorial>>(
    Object.fromEntries(tutorials.map((t) => [t.id, t]))
  );

  // Auto-start tutorial if requested
  useEffect(() => {
    if (autoStart && autoStartTutorialId && !tutorialState.activeTutorial) {
      const canStart = !tutorialState.completedTutorials.includes(autoStartTutorialId) ||
                       !tutorialState.skippedTutorials.includes(autoStartTutorialId);
      if (canStart) {
        startTutorial(autoStartTutorialId);
      }
    }
  }, []);

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!tutorialState.isTutorialActive) return;

      switch (e.key) {
        case 'Escape':
          endTutorial(false);
          break;
        case 'ArrowRight':
          nextStep();
          break;
        case 'ArrowLeft':
          previousStep();
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [tutorialState.isTutorialActive, tutorialState.activeTutorial, tutorialState.currentStep]);

  const activeTutorial = tutorialState.activeTutorial
    ? tutorialsMap[tutorialState.activeTutorial]
    : null;

  const currentStepData = activeTutorial?.steps[tutorialState.currentStep] || null;

  const startTutorial = useCallback((tutorialId: string) => {
    const tutorial = tutorialsMap[tutorialId];
    if (!tutorial) {
      console.warn(`Tutorial "${tutorialId}" not found`);
      return;
    }

    setActiveTutorial(tutorialId);
    setTutorialStep(0);
    setIsTutorialActive(true);
  }, [tutorialsMap, setActiveTutorial, setTutorialStep, setIsTutorialActive]);

  const endTutorial = useCallback((completed = false) => {
    if (tutorialState.activeTutorial) {
      if (completed) {
        completeTutorial(tutorialState.activeTutorial);
        onTutorialComplete?.(tutorialState.activeTutorial);
      } else {
        storeSkipTutorial(tutorialState.activeTutorial);
      }
    }
    setIsTutorialActive(false);
    setActiveTutorial(null);
    setTutorialStep(0);
  }, [tutorialState.activeTutorial, completeTutorial, storeSkipTutorial, setIsTutorialActive, setActiveTutorial, setTutorialStep, onTutorialComplete]);

  const nextStep = useCallback(() => {
    if (!activeTutorial) return;

    const currentStep = tutorialState.currentStep;
    const nextIndex = currentStep + 1;

    if (nextIndex < activeTutorial.steps.length) {
      setTutorialStep(nextIndex);
      onStepChange?.(tutorialState.activeTutorial!, nextIndex);

      // Execute step action if exists
      const step = activeTutorial.steps[nextIndex];
      step?.onNext?.();
    } else {
      // Tutorial completed
      endTutorial(true);
    }
  }, [activeTutorial, tutorialState.currentStep, tutorialState.activeTutorial, setTutorialStep, onStepChange, endTutorial]);

  const previousStep = useCallback(() => {
    if (!activeTutorial || tutorialState.currentStep <= 0) return;

    const prevIndex = tutorialState.currentStep - 1;
    setTutorialStep(prevIndex);
    onStepChange?.(tutorialState.activeTutorial!, prevIndex);

    // Execute step action if exists
    const step = activeTutorial.steps[prevIndex];
    step?.onPrev?.();
  }, [activeTutorial, tutorialState.currentStep, tutorialState.activeTutorial, setTutorialStep, onStepChange]);

  const goToStep = useCallback((stepIndex: number) => {
    if (!activeTutorial) return;

    if (stepIndex >= 0 && stepIndex < activeTutorial.steps.length) {
      setTutorialStep(stepIndex);
      onStepChange?.(tutorialState.activeTutorial!, stepIndex);
    }
  }, [activeTutorial, tutorialState.activeTutorial, setTutorialStep, onStepChange]);

  const skipTutorial = useCallback(() => {
    endTutorial(false);
  }, [endTutorial]);

  const resetProgress = useCallback(() => {
    // This would clear all completed/skipped tutorials
    // Implementation depends on persistence strategy
    setIsTutorialActive(false);
    setActiveTutorial(null);
    setTutorialStep(0);
  }, [setIsTutorialActive, setActiveTutorial, setTutorialStep]);

  const value: TutorialContextType = {
    state: tutorialState,
    tutorials: tutorialsMap,
    activeTutorial,
    currentStepData,
    startTutorial,
    endTutorial,
    nextStep,
    previousStep,
    goToStep,
    skipTutorial,
    resetProgress,
  };

  return (
    <TutorialContext.Provider value={value}>
      {children}
    </TutorialContext.Provider>
  );
};

/**
 * Hook to use tutorial context
 */
export const useTutorial = (): TutorialContextType => {
  const context = useContext(TutorialContext);
  if (!context) {
    throw new Error('useTutorial must be used within a TutorialProvider');
  }
  return context;
};

/**
 * Hook to check if an element is targeted by the current tutorial step
 */
export const useTutorialTarget = (elementId: string): boolean => {
  const { currentStepData } = useTutorial();

  if (!currentStepData?.target) return false;

  // Check if current step targets this element
  const target = currentStepData.target;
  const targetMatch = target === `#${elementId}` || target === `.${elementId}`;
  const highlightMatch = currentStepData.highlightElements?.some(selector =>
    selector === `#${elementId}` || selector === `.${elementId}`
  ) ?? false;

  return targetMatch || highlightMatch;
};

/**
 * Hook to get tutorial highlight styles
 */
export const useTutorialHighlight = () => {
  const { currentStepData, state } = useTutorial();
  const isHighlighted = state.isTutorialActive && currentStepData !== null;

  return {
    isHighlighted,
    targetSelector: currentStepData?.target,
    highlightSelectors: currentStepData?.highlightElements,
    shouldHighlight: (elementId: string) => {
      if (!isHighlighted) return false;
      return useTutorialTarget(elementId);
    },
  };
};
