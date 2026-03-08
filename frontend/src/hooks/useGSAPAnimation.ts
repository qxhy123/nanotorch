import { useEffect, useRef, useCallback, useState } from 'react';
import { gsap } from 'gsap';
import { useTransformerStore } from '../stores/transformerStore';
import type { AnimationTimeline } from '../types/transformer';
import { useAttentionStages } from './useAttentionStages';

/**
 * Options for creating a GSAP animation timeline
 */
interface CreateTimelineOptions {
  id: string;
  label: string;
  autoPlay?: boolean;
  repeat?: number;
  yoyo?: boolean;
  onUpdate?: (progress: number) => void;
  onComplete?: () => void;
}

/**
 * Hook for managing GSAP animations
 *
 * Provides:
 * - Timeline creation and management
 * - Playback controls
 * - Progress tracking
 * - Timeline registration with global store
 */
export const useGSAPAnimation = () => {
  const timelineRef = useRef<gsap.core.Timeline | null>(null);
  const { animationTimelines, registerTimeline, setActiveTimeline, updateTimelineProgress } = useTransformerStore();
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);

  /**
   * Register timeline with global store
   */
  useEffect(() => {
    if (timelineRef.current) {
      const timelineData: AnimationTimeline = {
        id: timelineRef.current.data?.id || 'default',
        label: timelineRef.current.data?.label || 'Animation',
        duration: timelineRef.current.duration() * 1000,
        progress,
        isPlaying,
        steps: [],
      };
      registerTimeline(timelineData);
    }
  }, [progress, isPlaying, registerTimeline]);

  /**
   * Create a new GSAP timeline
   */
  const createTimeline = useCallback((options: CreateTimelineOptions) => {
    // Clean up existing timeline
    if (timelineRef.current) {
      timelineRef.current.kill();
    }

    const tl = gsap.timeline({
      id: options.id,
      paused: !options.autoPlay,
      repeat: options.repeat,
      yoyo: options.yoyo,
      onUpdate: () => {
        const prog = tl.progress();
        setProgress(prog);
        updateTimelineProgress(options.id, prog);
        options.onUpdate?.(prog);
      },
      onComplete: () => {
        setIsPlaying(false);
        options.onComplete?.();
      },
      onReverseComplete: () => {
        setIsPlaying(false);
      },
    });

    // Store metadata
    tl.data = { id: options.id, label: options.label };

    timelineRef.current = tl;
    setActiveTimeline(options.id);

    return tl;
  }, [setActiveTimeline, updateTimelineProgress]);

  /**
   * Play the timeline
   */
  const play = useCallback(() => {
    if (timelineRef.current) {
      timelineRef.current.play();
      setIsPlaying(true);
    }
  }, []);

  /**
   * Pause the timeline
   */
  const pause = useCallback(() => {
    if (timelineRef.current) {
      timelineRef.current.pause();
      setIsPlaying(false);
    }
  }, []);

  /**
   * Reverse the timeline
   */
  const reverse = useCallback(() => {
    if (timelineRef.current) {
      timelineRef.current.reverse();
      setIsPlaying(true);
    }
  }, []);

  /**
   * Restart the timeline
   */
  const restart = useCallback(() => {
    if (timelineRef.current) {
      timelineRef.current.restart();
      setIsPlaying(true);
    }
  }, []);

  /**
   * Seek to a specific progress (0-1)
   */
  const seek = useCallback((progressValue: number) => {
    if (timelineRef.current) {
      const clampedProgress = Math.max(0, Math.min(1, progressValue));
      timelineRef.current.progress(clampedProgress);
    }
  }, []);

  /**
   * Seek to a specific time
   */
  const seekToTime = useCallback((time: number) => {
    if (timelineRef.current) {
      timelineRef.current.time(time);
    }
  }, []);

  /**
   * Get current timeline duration in seconds
   */
  const duration = useCallback(() => {
    return timelineRef.current?.duration() || 0;
  }, []);

  /**
   * Get current timeline time in seconds
   */
  const currentTime = useCallback(() => {
    return timelineRef.current?.time() || 0;
  }, []);

  /**
   * Clean up timeline on unmount
   */
  useEffect(() => {
    return () => {
      if (timelineRef.current) {
        timelineRef.current.kill();
      }
    };
  }, []);

  return {
    // Timeline
    timeline: timelineRef.current,
    createTimeline,

    // Playback controls
    play,
    pause,
    reverse,
    restart,
    seek,
    seekToTime,

    // State
    isPlaying,
    progress,
    duration: duration(),
    currentTime: currentTime(),

    // Store
    animationTimelines,
  };
};

/**
 * Hook for animating a DOM element with GSAP
 */
export const useGSAPElement = <T extends HTMLElement>() => {
  const elementRef = useRef<T>(null);
  const animationRef = useRef<gsap.core.Tween | null>(null);

  /**
   * Animate the element
   */
  const animate = useCallback((props: gsap.TweenVars, duration: number = 1) => {
    if (elementRef.current) {
      if (animationRef.current) {
        animationRef.current.kill();
      }
      animationRef.current = gsap.to(elementRef.current, {
        ...props,
        duration,
      });
      return animationRef.current;
    }
    return null;
  }, []);

  /**
   * Animate from current state
   */
  const animateFrom = useCallback((props: gsap.TweenVars, duration: number = 1) => {
    if (elementRef.current) {
      if (animationRef.current) {
        animationRef.current.kill();
      }
      animationRef.current = gsap.from(elementRef.current, {
        ...props,
        duration,
      });
      return animationRef.current;
    }
    return null;
  }, []);

  /**
   * Animate from and to states
   */
  const animateFromTo = useCallback((
    fromProps: gsap.TweenVars,
    toProps: gsap.TweenVars,
    duration: number = 1
  ) => {
    if (elementRef.current) {
      if (animationRef.current) {
        animationRef.current.kill();
      }
      animationRef.current = gsap.fromTo(elementRef.current, fromProps, {
        ...toProps,
        duration,
      });
      return animationRef.current;
    }
    return null;
  }, []);

  /**
   * Kill the animation
   */
  const kill = useCallback(() => {
    if (animationRef.current) {
      animationRef.current.kill();
      animationRef.current = null;
    }
  }, []);

  /**
   * Clean up on unmount
   */
  useEffect(() => {
    return () => {
      kill();
    };
  }, [kill]);

  return {
    ref: elementRef,
    animate,
    animateFrom,
    animateFromTo,
    kill,
    animation: animationRef.current,
  };
};

/**
 * Hook for managing attention stage animations
 */
export const useAttentionStageAnimation = () => {
  const { createTimeline, play, pause, seek, progress, isPlaying } = useGSAPAnimation();
  const { allStages } = useAttentionStages();

  /**
   * Create stage transition timeline
   */
  const createStageTimeline = useCallback(() => {
    const tl = createTimeline({
      id: 'attention-stages',
      label: 'Attention Stages',
      autoPlay: false,
    });

    // Add animations for each stage transition
    allStages.forEach((_stage, _index) => {
      tl.to({}, {
        duration: 0.5,
        onStart: () => {
          // Trigger stage enter animation
        },
      });
    });

    return tl;
  }, [createTimeline, allStages]);

  /**
   * Animate to specific stage
   */
  const animateToStage = useCallback((stageIndex: number) => {
    const totalStages = allStages.length;
    const targetProgress = stageIndex / (totalStages - 1);
    seek(targetProgress);
  }, [allStages.length, seek]);

  return {
    createStageTimeline,
    animateToStage,
    play,
    pause,
    progress,
    isPlaying,
  };
};
