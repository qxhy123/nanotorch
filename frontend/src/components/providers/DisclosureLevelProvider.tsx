import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import type { DisclosureLevel, DisclosureLevelConfig } from '../../types/transformer';

/**
 * Disclosure Level Context
 */
interface DisclosureLevelContextType {
  level: DisclosureLevel;
  config: DisclosureLevelConfig;
  setLevel: (level: DisclosureLevel) => void;
  canShow: (requiredLevel: DisclosureLevel) => boolean;
  getOpacity: (elementLevel: DisclosureLevel) => number;
  resetLevel: () => void;
}

const DisclosureLevelContext = createContext<DisclosureLevelContextType | undefined>(undefined);

/**
 * Disclosure level configurations
 */
const DISCLOSURE_CONFIGS: Record<DisclosureLevel, DisclosureLevelConfig> = {
  overview: {
    level: 'overview',
    showMath: false,
    showImplementation: false,
    showAllParameters: false,
    interactiveElements: ['basic-controls', 'stage-navigation'],
  },
  intermediate: {
    level: 'intermediate',
    showMath: true,
    showImplementation: false,
    showAllParameters: false,
    interactiveElements: ['basic-controls', 'stage-navigation', 'matrix-interaction'],
  },
  detailed: {
    level: 'detailed',
    showMath: true,
    showImplementation: true,
    showAllParameters: false,
    interactiveElements: ['basic-controls', 'stage-navigation', 'matrix-interaction', 'parameter-adjustment'],
  },
  math: {
    level: 'math',
    showMath: true,
    showImplementation: true,
    showAllParameters: true,
    interactiveElements: ['basic-controls', 'stage-navigation', 'matrix-interaction', 'parameter-adjustment', 'formula-manipulation'],
  },
};

/**
 * Disclosure Level Provider Props
 */
interface DisclosureLevelProviderProps {
  children: React.ReactNode;
  defaultLevel?: DisclosureLevel;
  persistKey?: string; // LocalStorage key for persistence
  onLevelChange?: (level: DisclosureLevel) => void;
}

/**
 * DisclosureLevelProvider Component
 *
 * Provides disclosure level context to the application.
 * Features:
 * - Global disclosure level management
 * - LocalStorage persistence
 * - Level change callbacks
 */
export const DisclosureLevelProvider: React.FC<DisclosureLevelProviderProps> = ({
  children,
  defaultLevel = 'intermediate',
  persistKey = 'disclosure-level',
  onLevelChange,
}) => {
  const [level, setLevelState] = useState<DisclosureLevel>(() => {
    // Try to load from localStorage first
    if (persistKey && typeof window !== 'undefined') {
      const saved = localStorage.getItem(persistKey);
      if (saved && isValidDisclosureLevel(saved)) {
        return saved as DisclosureLevel;
      }
    }
    // Fall back to default
    return defaultLevel;
  });

  // Persist to localStorage
  useEffect(() => {
    if (persistKey && typeof window !== 'undefined') {
      localStorage.setItem(persistKey, level);
    }
  }, [level, persistKey]);

  const setLevel = useCallback((newLevel: DisclosureLevel) => {
    setLevelState(newLevel);
    onLevelChange?.(newLevel);
  }, [onLevelChange]);

  const canShow = useCallback((requiredLevel: DisclosureLevel): boolean => {
    const levels: DisclosureLevel[] = ['overview', 'intermediate', 'detailed', 'math'];
    return levels.indexOf(level) >= levels.indexOf(requiredLevel);
  }, [level]);

  const getOpacity = useCallback((elementLevel: DisclosureLevel): number => {
    const levels: DisclosureLevel[] = ['overview', 'intermediate', 'detailed', 'math'];
    const elementIndex = levels.indexOf(elementLevel);
    const currentIndex = levels.indexOf(level);

    if (currentIndex < elementIndex) {
      // Element is at higher disclosure level - make it semi-transparent
      return 0.3;
    }
    return 1;
  }, [level]);

  const resetLevel = useCallback(() => {
    setLevel(defaultLevel);
  }, [setLevel, defaultLevel]);

  const config = DISCLOSURE_CONFIGS[level];

  const value: DisclosureLevelContextType = {
    level,
    config,
    setLevel,
    canShow,
    getOpacity,
    resetLevel,
  };

  return (
    <DisclosureLevelContext.Provider value={value}>
      {children}
    </DisclosureLevelContext.Provider>
  );
};

/**
 * Hook to use disclosure level context
 */
export const useDisclosureLevel = (): DisclosureLevelContextType => {
  const context = useContext(DisclosureLevelContext);
  if (!context) {
    throw new Error('useDisclosureLevel must be used within a DisclosureLevelProvider');
  }
  return context;
};

/**
 * Hook to check if an element should be visible at current disclosure level
 */
export const useDisclosureVisible = (
  requiredLevel: DisclosureLevel
): { visible: boolean; opacity: number } => {
  const { canShow, getOpacity } = useDisclosureLevel();
  return {
    visible: canShow(requiredLevel),
    opacity: getOpacity(requiredLevel),
  };
};

/**
 * Component that conditionally renders based on disclosure level
 */
export const DisclosureGate: React.FC<{
  level: DisclosureLevel;
  children: React.ReactNode;
  fallback?: React.ReactNode;
  fade?: boolean; // Fade out instead of hiding
}> = ({ level, children, fallback = null, fade = false }) => {
  const { visible, opacity } = useDisclosureVisible(level);

  if (fade) {
    return (
      <div style={{ opacity }}>
        {children}
      </div>
    );
  }

  return visible ? <>{children}</> : <>{fallback}</>;
};

/**
 * Utility function to validate disclosure level
 */
function isValidDisclosureLevel(value: string): value is DisclosureLevel {
  return ['overview', 'intermediate', 'detailed', 'math'].includes(value);
}

/**
 * Disclosure level descriptions for UI
 */
export const DISCLOSURE_LEVEL_DESCRIPTIONS: Record<DisclosureLevel, {
  title: string;
  description: string;
  icon: string;
}> = {
  overview: {
    title: 'Overview',
    description: 'Shows the main components and high-level flow. Perfect for getting familiar with the architecture.',
    icon: '🔭',
  },
  intermediate: {
    title: 'Intermediate',
    description: 'Includes mathematical formulas and detailed explanations. Good for understanding the computations.',
    icon: '📊',
  },
  detailed: {
    title: 'Detailed',
    description: 'Shows implementation details and all intermediate computation steps. For deep learning practitioners.',
    icon: '🔬',
  },
  math: {
    title: 'Math',
    description: 'Includes rigorous mathematical derivations and all parameters. For researchers and advanced users.',
    icon: '📐',
  },
};

/**
 * DisclosureLevelSelector Component
 *
 * UI component for selecting disclosure level
 */
export const DisclosureLevelSelector: React.FC<{
  className?: string;
  showDescriptions?: boolean;
}> = ({ className = '', showDescriptions = false }) => {
  const { level, setLevel } = useDisclosureLevel();

  const levels: DisclosureLevel[] = ['overview', 'intermediate', 'detailed', 'math'];

  return (
    <div className={`disclosure-level-selector ${className}`}>
      {levels.map((l) => (
        <button
          key={l}
          onClick={() => setLevel(l)}
          className={`
            px-4 py-2 rounded-lg font-medium transition-all
            ${level === l
              ? 'bg-blue-500 text-white shadow-lg'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }
          `}
        >
          <span className="mr-2">{DISCLOSURE_LEVEL_DESCRIPTIONS[l].icon}</span>
          {DISCLOSURE_LEVEL_DESCRIPTIONS[l].title}
        </button>
      ))}
      {showDescriptions && (
        <p className="mt-2 text-sm text-gray-600">
          {DISCLOSURE_LEVEL_DESCRIPTIONS[level].description}
        </p>
      )}
    </div>
  );
};
