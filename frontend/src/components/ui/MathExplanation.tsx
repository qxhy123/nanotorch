import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { BookOpen, TrendingUp } from 'lucide-react';
import { Latex } from './Latex';
import type { MathExplanation, MathExplanationLevel } from '../../types/transformer';
import { Card } from './card';

interface MathExplanationProps {
  explanations: Record<MathExplanationLevel, MathExplanation>;
  defaultLevel?: MathExplanationLevel;
  onLevelChange?: (level: MathExplanationLevel) => void;
  showTabs?: boolean;
}

/**
 * MathExplanation Component
 *
 * Displays mathematical explanations at different levels of rigor.
 * Features:
 * - Three levels: Intuitive, Formal, Rigorous
 * - Tab-based navigation
 * - Interactive examples
 * - Visual aids
 */
export const MathExplanationComponent: React.FC<MathExplanationProps> = ({
  explanations,
  defaultLevel = 'intuitive',
  onLevelChange,
  showTabs = true,
}) => {
  const [currentLevel, setCurrentLevel] = useState<MathExplanationLevel>(defaultLevel);

  const levels: MathExplanationLevel[] = ['intuitive', 'formal', 'rigorous'];

  const levelIcons: Record<MathExplanationLevel, React.ReactNode> = {
    intuitive: <BookOpen className="w-4 h-4" />,
    formal: <BookOpen className="w-4 h-4" />,
    rigorous: <TrendingUp className="w-4 h-4" />,
  };

  const levelColors: Record<MathExplanationLevel, string> = {
    intuitive: 'bg-green-500',
    formal: 'bg-blue-500',
    rigorous: 'bg-purple-500',
  };

  const handleLevelChange = (level: MathExplanationLevel) => {
    setCurrentLevel(level);
    onLevelChange?.(level);
  };

  return (
    <Card className="math-explanation overflow-hidden">
      {/* Level Tabs */}
      {showTabs && (
        <div className="flex border-b">
          {levels.map((level) => (
            <button
              key={level}
              onClick={() => handleLevelChange(level)}
              className={`
                flex-1 flex items-center justify-center gap-2 px-4 py-3 font-medium transition-all
                ${currentLevel === level
                  ? 'text-white border-b-2'
                  : 'text-gray-600 hover:bg-gray-50'
                }
              `}
              style={{
                backgroundColor: currentLevel === level ? levelColors[level] : undefined,
                borderColor: currentLevel === level ? levelColors[level] : undefined,
              }}
            >
              {levelIcons[level]}
              <span className="capitalize">{level}</span>
            </button>
          ))}
        </div>
      )}

      {/* Content */}
      <AnimatePresence mode="wait">
        <motion.div
          key={currentLevel}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.2 }}
          className="p-6"
        >
          {(() => {
            const explanation = explanations[currentLevel];
            return (
              <div className="space-y-4">
                {/* Title */}
                <h3 className="text-xl font-bold text-gray-800">{explanation.title}</h3>

                {/* Formula */}
                {explanation.formula && (
                  <div className="formula-container p-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg border border-blue-200">
                    <Latex display={true}>{explanation.formula as string}</Latex>
                  </div>
                )}

                {/* Content */}
                <div className="prose prose-sm max-w-none">
                  {typeof explanation.content === 'string' ? (
                    <p className="text-gray-700">{explanation.content}</p>
                  ) : (
                    explanation.content
                  )}
                </div>

                {/* Visual Aid */}
                {explanation.visualAid && (
                  <div className="visual-aid mt-4">
                    {explanation.visualAid}
                  </div>
                )}

                {/* Examples */}
                {explanation.examples && explanation.examples.length > 0 && (
                  <div className="examples mt-6">
                    <h4 className="text-sm font-semibold mb-3">Examples:</h4>
                    <div className="space-y-3">
                      {explanation.examples.map((example, index) => (
                        <div
                          key={index}
                          className="p-3 bg-gray-50 rounded-lg border border-gray-200"
                        >
                          <p className="text-sm font-medium text-gray-800 mb-2">
                            {example.description}
                          </p>
                          {example.steps && (
                            <ol className="text-xs text-gray-600 space-y-1 ml-4">
                              {example.steps.map((step, i) => (
                                <li key={i}>{step}</li>
                              ))}
                            </ol>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            );
          })()}
        </motion.div>
      </AnimatePresence>
    </Card>
  );
};

/**
 * MathExplanationAccordion Component
 *
 * Accordion-style math explanation with collapsible levels
 */
export const MathExplanationAccordion: React.FC<{
  explanations: Record<MathExplanationLevel, MathExplanation>;
}> = ({ explanations }) => {
  const [openLevel, setOpenLevel] = useState<MathExplanationLevel | null>(null);

  return (
    <div className="space-y-2">
      {Object.entries(explanations).map(([level, explanation]) => (
        <motion.div
          key={level}
          className="border rounded-lg overflow-hidden"
          initial={false}
          animate={{
            height: openLevel === level ? 'auto' : 'auto',
          }}
        >
          <button
            onClick={() => setOpenLevel(openLevel === level ? null : level as MathExplanationLevel)}
            className="w-full flex items-center gap-3 px-4 py-3 bg-white hover:bg-gray-50 transition-colors"
          >
            <motion.div
              animate={{ rotate: openLevel === level ? 90 : 0 }}
              transition={{ duration: 0.2 }}
            >
              <svg className="w-5 h-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </motion.div>
            <span className="font-medium capitalize">{level}</span>
          </button>

          <AnimatePresence>
            {openLevel === level && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.3 }}
                className="overflow-hidden"
              >
                <div className="p-4 bg-gray-50">
                  <h4 className="font-semibold mb-2">{explanation.title}</h4>
                  {explanation.formula && (
                    <div className="mb-3">
                      <Latex display={true}>{explanation.formula}</Latex>
                    </div>
                  )}
                  <p className="text-sm text-gray-700">{explanation.content}</p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      ))}
    </div>
  );
};

/**
 * Preset math explanations for common Transformer operations
 */
const ATTENTION_EXPLANATIONS: Record<string, Record<MathExplanationLevel, MathExplanation>> = {
  scaled_dot_product: {
    intuitive: {
      level: 'intuitive',
      title: 'Intuitive Explanation',
      content: 'Imagine you\'re at a party trying to decide who to talk to. You "query" each person to see if they share your interests. The "scaled dot product" is like measuring how similar your interests are to theirs, but we divide by a number to keep the scores reasonable.',
      formula: '\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V',
    },
    formal: {
      level: 'formal',
      title: 'Formal Definition',
      content: 'The scaled dot-product attention computes the compatibility between queries and keys. The scaling factor 1/√dₖ prevents the dot products from growing too large in magnitude, which would push the softmax function into regions with extremely small gradients.',
      formula: '\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V',
      examples: [
        {
          description: 'For a single attention head with dₖ = 64:',
          input: { Q: [1, 2, 3], K: [4, 5, 6] },
          output: { attention_weights: [0.1, 0.3, 0.6] },
          steps: [
            'Compute Q·K^T: dot product of query and key',
            'Scale by 1/√64 = 1/8',
            'Apply softmax to get attention weights',
          ],
        },
      ],
    },
    rigorous: {
      level: 'rigorous',
      title: 'Mathematical Derivation',
      content: 'Given query matrix Q ∈ ℝⁿˣᵈᵏ, key matrix K ∈ ℝⁿˣᵈᵏ, and value matrix V ∈ ℝⁿˣᵈᵛ, the attention mechanism computes:',
      formula: '\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V',
    },
  },
};

/**
 * Helper component for displaying preset explanations
 */
export const PresetMathExplanation: React.FC<{
  topic: keyof typeof ATTENTION_EXPLANATIONS;
  defaultLevel?: MathExplanationLevel;
}> = ({ topic, defaultLevel }) => {
  return (
    <MathExplanationComponent
      explanations={ATTENTION_EXPLANATIONS[topic]}
      defaultLevel={defaultLevel}
    />
  );
};
