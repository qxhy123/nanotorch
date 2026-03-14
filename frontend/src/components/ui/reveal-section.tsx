import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronRight } from 'lucide-react';
import type { DisclosureLevel } from '../../types/transformer';
import { useDisclosureLevel } from '../providers';

interface RevealSectionProps {
  /**
   * Unique identifier for the section
   */
  id: string;

  /**
   * Title of the section
   */
  title: string;

  /**
   * Content to reveal
   */
  children: React.ReactNode;

  /**
   * Minimum disclosure level required to show this section
   */
  minLevel?: DisclosureLevel;

  /**
   * Whether the section is expanded by default
   */
  defaultExpanded?: boolean;

  /**
   * Whether to show the expand/collapse button
   */
  showToggle?: boolean;

  /**
   * Whether the section is disabled (always collapsed or shown based on disclosure level)
   */
  disabled?: boolean;

  /**
   * Icon to display before the title
   */
  icon?: React.ReactNode;

  /**
   * Additional CSS classes
   */
  className?: string;

  /**
   * Badge to display next to the title
   */
  badge?: string | React.ReactNode;

  /**
   * Custom header content
   */
  headerContent?: React.ReactNode;

  /**
   * Callback when section is expanded/collapsed
   */
  onToggle?: (isExpanded: boolean) => void;
}

/**
 * RevealSection Component
 *
 * A progressive disclosure component that allows users to expand/collapse content.
 * Features:
 * - Framer Motion animations
 * - Keyboard navigation (Enter/Space to toggle)
 * - ARIA accessibility attributes
 * - Disclosure level awareness
 * - Customizable styling
 */
export const RevealSection: React.FC<RevealSectionProps> = ({
  id,
  title,
  children,
  minLevel = 'overview',
  defaultExpanded = false,
  showToggle = true,
  disabled = false,
  icon,
  className = '',
  badge,
  headerContent,
  onToggle,
}) => {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);
  const { canShow, getOpacity } = useDisclosureLevel();
  const visible = canShow(minLevel);

  const handleToggle = () => {
    if (disabled || !showToggle) return;
    const newState = !isExpanded;
    setIsExpanded(newState);
    onToggle?.(newState);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleToggle();
    }
  };

  if (!visible) {
    return null;
  }

  const opacity = getOpacity(minLevel);

  return (
    <motion.div
      id={id}
      className={`reveal-section border rounded-lg overflow-hidden ${className}`}
      style={{ opacity }}
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity, y: 0 }}
      transition={{ duration: 0.2 }}
    >
      {/* Header */}
      <div
        className={`
          flex items-center gap-3 p-4 cursor-pointer
          ${showToggle && !disabled ? 'hover:bg-gray-50' : ''}
          ${disabled ? 'cursor-default' : ''}
          transition-colors
        `}
        onClick={handleToggle}
        onKeyDown={handleKeyDown}
        role="button"
        tabIndex={showToggle && !disabled ? 0 : -1}
        aria-expanded={isExpanded}
        aria-controls={`${id}-content`}
      >
        {/* Toggle Icon */}
        {showToggle && !disabled && (
          <motion.div
            animate={{ rotate: isExpanded ? 90 : 0 }}
            transition={{ duration: 0.2 }}
            className="text-gray-500"
          >
            <ChevronRight className="w-5 h-5" />
          </motion.div>
        )}

        {/* Custom Icon */}
        {icon && (
          <div className="text-gray-600">
            {icon}
          </div>
        )}

        {/* Title */}
        <h3 className="flex-1 font-semibold text-gray-800">
          {title}
        </h3>

        {/* Badge */}
        {badge && (
          <span className="px-2 py-1 text-xs bg-blue-100 text-blue-700 rounded-full">
            {badge}
          </span>
        )}

        {/* Custom Header Content */}
        {headerContent}
      </div>

      {/* Content */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            id={`${id}-content`}
            className="overflow-hidden"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{
              height: { duration: 0.3, ease: 'easeInOut' },
              opacity: { duration: 0.2 },
            }}
          >
            <div className="p-4 pt-0">
              {children}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

/**
 * DisclosureBadge Component
 *
 * Displays the disclosure level as a colored badge
 */
export const DisclosureBadge: React.FC<{
  level: DisclosureLevel;
  className?: string;
}> = ({ level, className = '' }) => {
  const colors: Record<DisclosureLevel, { bg: string; text: string; label: string }> = {
    overview: { bg: 'bg-green-100', text: 'text-green-700', label: 'Overview' },
    intermediate: { bg: 'bg-blue-100', text: 'text-blue-700', label: 'Intermediate' },
    detailed: { bg: 'bg-purple-100', text: 'text-purple-700', label: 'Detailed' },
    math: { bg: 'bg-pink-100', text: 'text-pink-700', label: 'Math' },
  };

  const { bg, text, label } = colors[level];

  return (
    <span className={`px-2 py-1 text-xs rounded-full ${bg} ${text} ${className}`}>
      {label}
    </span>
  );
};

/**
 * DisclosureControl Component
 *
 * Allows users to change the disclosure level
 */
export const DisclosureControl: React.FC<{
  level: DisclosureLevel;
  onChange: (level: DisclosureLevel) => void;
  disabled?: boolean;
}> = ({ level, onChange, disabled = false }) => {
  const levels: DisclosureLevel[] = ['overview', 'intermediate', 'detailed', 'math'];
  const colors: Record<DisclosureLevel, string> = {
    overview: '#22c55e',
    intermediate: '#3b82f6',
    detailed: '#8b5cf6',
    math: '#ec4899',
  };

  return (
    <div className="flex items-center gap-2">
      <span className="text-sm font-medium text-gray-600">Detail Level:</span>
      <div className="flex gap-1">
        {levels.map((l) => (
          <button
            key={l}
            onClick={() => onChange(l)}
            disabled={disabled}
            className={`
              px-3 py-1 text-xs font-medium rounded-full transition-all
              ${level === l
                ? 'text-white shadow-md'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }
              ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
            `}
            style={level === l ? { backgroundColor: colors[l] } : {}}
          >
            {l.charAt(0).toUpperCase() + l.slice(1)}
          </button>
        ))}
      </div>
    </div>
  );
};

/**
 * StackedRevealSections Component
 *
 * Manages multiple reveal sections with accordion behavior
 */
export const StackedRevealSections: React.FC<{
  sections: Array<{
    id: string;
    title: string;
    content: React.ReactNode;
    minLevel?: DisclosureLevel;
    icon?: React.ReactNode;
    badge?: string | React.ReactNode;
  }>;
  allowMultipleOpen?: boolean;
  defaultOpen?: string | string[];
}> = ({ sections, allowMultipleOpen = false, defaultOpen = [] }) => {
  const [openSections, setOpenSections] = useState<Set<string>>(
    new Set(Array.isArray(defaultOpen) ? defaultOpen : [defaultOpen].filter(Boolean))
  );

  const toggleSection = (id: string) => {
    setOpenSections((prev) => {
      const newSet = new Set(prev);
      if (allowMultipleOpen) {
        if (newSet.has(id)) {
          newSet.delete(id);
        } else {
          newSet.add(id);
        }
      } else {
        // Accordion mode - close all others
        newSet.clear();
        newSet.add(id);
      }
      return newSet;
    });
  };

  return (
    <div className="space-y-2">
      {sections.map((section) => (
        <RevealSection
          key={section.id}
          id={section.id}
          title={section.title}
          minLevel={section.minLevel}
          defaultExpanded={openSections.has(section.id)}
          onToggle={(isExpanded) => {
            if (isExpanded && !allowMultipleOpen) {
              toggleSection(section.id);
            } else if (!isExpanded && openSections.has(section.id)) {
              toggleSection(section.id);
            }
          }}
          icon={section.icon}
          badge={section.badge}
        >
          {section.content}
        </RevealSection>
      ))}
    </div>
  );
};
