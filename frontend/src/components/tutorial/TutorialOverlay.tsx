import React, { useEffect, useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, ChevronRight, ChevronLeft, SkipForward } from 'lucide-react';
import { useTutorial } from './TutorialProvider';
import { Button } from '../ui/button';
import { Card } from '../ui/card';
import { Badge } from '../ui/badge';

/**
 * TutorialOverlay Component
 *
 * Displays the tutorial UI overlay with step information and navigation.
 * Features:
 * - Auto-positioning tooltip
 * - Highlight target elements
 * - Background dimming
 * - Keyboard navigation
 * - Progress indicator
 */
export const TutorialOverlay: React.FC = () => {
  const {
    state,
    activeTutorial,
    currentStepData,
    nextStep,
    previousStep,
    skipTutorial,
    endTutorial,
  } = useTutorial();

  const [position, setPosition] = useState({ top: 0, left: 0, width: 0, height: 0 });
  const [targetElement, setTargetElement] = useState<HTMLElement | null>(null);
  const overlayRef = useRef<HTMLDivElement>(null);

  // Find and position relative to target element
  useEffect(() => {
    if (!state.isTutorialActive || !currentStepData?.target) {
      setTargetElement(null);
      return;
    }

    const findTarget = () => {
      try {
        const element = document.querySelector(currentStepData.target!) as HTMLElement;
        setTargetElement(element);

        if (element) {
          const rect = element.getBoundingClientRect();
          const scrollX = window.scrollX || window.pageXOffset;
          const scrollY = window.scrollY || window.pageYOffset;

          setPosition({
            top: rect.top + scrollY,
            left: rect.left + scrollX,
            width: rect.width,
            height: rect.height,
          });
        }
      } catch (error) {
        console.warn('Could not find tutorial target:', currentStepData.target);
        setTargetElement(null);
      }
    };

    findTarget();

    // Recalculate on resize
    const handleResize = () => findTarget();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [state.isTutorialActive, currentStepData?.target]);

  // Handle escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && state.isTutorialActive) {
        endTutorial(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [state.isTutorialActive, endTutorial]);

  if (!state.isTutorialActive || !activeTutorial || !currentStepData) {
    return null;
  }

  const currentStep = state.currentStep;
  const totalSteps = activeTutorial.steps.length;
  const progress = ((currentStep + 1) / totalSteps) * 100;

  // Position for the tutorial card
  const getCardPosition = () => {
    if (!targetElement) {
      return {
        position: 'fixed' as const,
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
      };
    }

    const cardWidth = 400; // Approximate card width
    const cardHeight = 300; // Approximate max card height
    const padding = 20;

    const viewportHeight = window.innerHeight;
    const viewportWidth = window.innerWidth;

    let top = position.top;
    let left = position.left;

    // Determine best position based on available space
    const spaceAbove = position.top;
    const spaceBelow = viewportHeight - (position.top + position.height);
    const spaceLeft = position.left;
    const spaceRight = viewportWidth - (position.left + position.width);

    // Prefer right side, then left, then below, then above
    if (spaceRight >= cardWidth + padding) {
      // Position to the right
      left = position.left + position.width + padding;
      top = Math.max(padding, Math.min(top, viewportHeight - cardHeight - padding));
    } else if (spaceLeft >= cardWidth + padding) {
      // Position to the left
      left = position.left - cardWidth - padding;
      top = Math.max(padding, Math.min(top, viewportHeight - cardHeight - padding));
    } else if (spaceBelow >= cardHeight + padding) {
      // Position below
      top = position.top + position.height + padding;
      left = Math.max(padding, Math.min(
        left + position.width / 2 - cardWidth / 2,
        viewportWidth - cardWidth - padding
      ));
    } else if (spaceAbove >= cardHeight + padding) {
      // Position above
      top = position.top - cardHeight - padding;
      left = Math.max(padding, Math.min(
        left + position.width / 2 - cardWidth / 2,
        viewportWidth - cardWidth - padding
      ));
    } else {
      // Center on screen if no good position
      return {
        position: 'fixed' as const,
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
      };
    }

    return {
      position: 'absolute' as const,
      top: `${top}px`,
      left: `${left}px`,
      maxWidth: `${Math.min(cardWidth, viewportWidth - padding * 2)}px`,
    };
  };

  const cardStyle = getCardPosition();

  return (
    <>
      {/* Backdrop */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/50 z-40"
        onClick={() => currentStepData.dismissOnAction && nextStep()}
      />

      {/* Highlight ring around target */}
      <AnimatePresence>
        {targetElement && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            className="pointer-events-none fixed z-40 border-4 border-blue-500 rounded-lg shadow-2xl"
            style={{
              top: position.top,
              left: position.left,
              width: position.width,
              height: position.height,
              transition: 'all 0.3s ease',
            }}
          />
        )}
      </AnimatePresence>

      {/* Tutorial Card */}
      <motion.div
        ref={overlayRef}
        initial={{ opacity: 0, y: 20, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: 20, scale: 0.95 }}
        transition={{ type: 'spring', damping: 25, stiffness: 300 }}
        className="z-50"
        style={cardStyle}
      >
        <Card className="shadow-2xl border-2 border-blue-500">
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b bg-gradient-to-r from-blue-500 to-purple-500">
            <div className="flex items-center gap-2">
              <span className="text-white font-semibold">{activeTutorial.title}</span>
              <Badge variant="secondary" className="bg-white/20 text-white border-white/30">
                Step {currentStep + 1} of {totalSteps}
              </Badge>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => endTutorial(false)}
              className="text-white hover:bg-white/20"
            >
              <X className="w-4 h-4" />
            </Button>
          </div>

          {/* Progress Bar */}
          <div className="h-1 bg-gray-200">
            <motion.div
              className="h-full bg-gradient-to-r from-blue-500 to-purple-500"
              initial={{ width: 0 }}
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>

          {/* Content */}
          <div className="p-6 space-y-4">
            <h3 className="text-xl font-bold text-gray-800">
              {currentStepData.title}
            </h3>

            <div className="text-gray-600 prose prose-sm max-w-none">
              {typeof currentStepData.content === 'string' ? (
                <p>{currentStepData.content}</p>
              ) : (
                currentStepData.content
              )}
            </div>

            {/* Estimated time remaining */}
            {activeTutorial.estimatedTime && (
              <div className="text-sm text-gray-500">
                Estimated time: ~{Math.round(
                  (activeTutorial.estimatedTime * (totalSteps - currentStep - 1)) / totalSteps
                )} min remaining
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="flex items-center justify-between p-4 bg-gray-50 border-t">
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={previousStep}
                disabled={currentStep === 0}
              >
                <ChevronLeft className="w-4 h-4 mr-1" />
                Previous
              </Button>

              {currentStepData.action && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    // Execute the step action
                    if (currentStepData.action?.selector) {
                      const element = document.querySelector(currentStepData.action.selector);
                      if (element instanceof HTMLElement) {
                        element.click();
                      }
                    }
                    if (currentStepData.dismissOnAction) {
                      nextStep();
                    }
                  }}
                >
                  {currentStepData.action.type === 'click' ? 'Click Target' : 'Continue'}
                </Button>
              )}

              <Button
                size="sm"
                onClick={nextStep}
                className="bg-gradient-to-r from-blue-500 to-purple-500"
              >
                {currentStep === totalSteps - 1 ? 'Finish' : 'Next'}
                <ChevronRight className="w-4 h-4 ml-1" />
              </Button>
            </div>

            <Button
              variant="ghost"
              size="sm"
              onClick={skipTutorial}
              className="text-gray-500 hover:text-gray-700"
            >
              <SkipForward className="w-4 h-4 mr-1" />
              Skip Tutorial
            </Button>
          </div>
        </Card>
      </motion.div>

      {/* Keyboard hints */}
      <div className="fixed bottom-4 left-1/2 transform -translate-x-1/2 z-50">
        <div className="flex gap-2 px-4 py-2 bg-black/80 text-white text-xs rounded-full">
          <span className="flex items-center gap-1">
            <kbd className="px-1.5 py-0.5 bg-white/20 rounded">Esc</kbd>
            Close
          </span>
          <span className="flex items-center gap-1">
            <kbd className="px-1.5 py-0.5 bg-white/20 rounded">←</kbd>
            <kbd className="px-1.5 py-0.5 bg-white/20 rounded">→</kbd>
            Navigate
          </span>
        </div>
      </div>
    </>
  );
};

/**
 * TutorialTooltip Component
 *
 * A smaller, simpler tooltip for quick tips
 */
export const TutorialTooltip: React.FC<{
  content: string;
  target: string;
  position?: 'top' | 'bottom' | 'left' | 'right';
}> = ({ content, target, position = 'bottom' }) => {
  const [isVisible, setIsVisible] = useState(false);
  const [coords, setCoords] = useState({ top: 0, left: 0 });

  useEffect(() => {
    const element = document.querySelector(target) as HTMLElement;
    if (!element) return;

    const handleMouseEnter = () => setIsVisible(true);
    const handleMouseLeave = () => setIsVisible(false);

    element.addEventListener('mouseenter', handleMouseEnter);
    element.addEventListener('mouseleave', handleMouseLeave);

    return () => {
      element.removeEventListener('mouseenter', handleMouseEnter);
      element.removeEventListener('mouseleave', handleMouseLeave);
    };
  }, [target]);

  useEffect(() => {
    const element = document.querySelector(target) as HTMLElement;
    if (!element) return;

    const rect = element.getBoundingClientRect();
    const scrollX = window.scrollX || window.pageXOffset;
    const scrollY = window.scrollY || window.pageYOffset;

    const tooltipCoords = {
      top: position === 'bottom' ? rect.bottom + scrollY + 10 :
             position === 'top' ? rect.top + scrollY - 50 :
             rect.top + scrollY + rect.height / 2 - 20,
      left: position === 'right' ? rect.right + scrollX + 10 :
              position === 'left' ? rect.left + scrollX - 150 :
              rect.left + scrollX + rect.width / 2 - 75,
    };

    setCoords(tooltipCoords);
  }, [target, position]);

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.9 }}
          className="fixed z-50 px-4 py-2 bg-gray-900 text-white text-sm rounded-lg shadow-xl max-w-xs"
          style={{
            top: `${coords.top}px`,
            left: `${coords.left}px`,
          }}
        >
          {content}
          {/* Arrow */}
          <div
            className="absolute w-0 h-0 border-l-4 border-r-4 border-b-4 border-transparent border-b-gray-900"
            style={{
              top: position === 'bottom' ? '-8px' : 'auto',
              bottom: position === 'top' ? '-8px' : 'auto',
              left: position === 'left' ? 'auto' : 'calc(50% - 8px)',
              right: position === 'left' ? '-8px' : 'auto',
              transform: position === 'left' ? 'rotate(90deg)' :
                      position === 'right' ? 'rotate(-90deg)' : 'none',
            }}
          />
        </motion.div>
      )}
    </AnimatePresence>
  );
};
