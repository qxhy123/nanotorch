import React, { useState, useMemo } from 'react';
import katex from 'katex';
import type { ColorCodedMathProps } from '../../types/transformer';
import { useSemanticColors } from '../../hooks/useSemanticColors';

/**
 * ColorCodedMath Component
 *
 * Renders LaTeX formulas with color-coded variables.
 * Features:
 * - KaTeX rendering with custom colors
 * - Interactive variable highlighting
 * - Hover effects
 * - Variable click callbacks
 */
export const ColorCodedMath: React.FC<ColorCodedMathProps> = ({
  latex,
  colorMap = {},
  highlightVariables = [],
  interactive = false,
}) => {
  const { getColors } = useSemanticColors();
  const [hoveredVariable] = useState<string | null>(null);

  // Extract variables from the LaTeX expression
  const variables = useMemo(() => {
    const vars: string[] = [];
    const regex = /\\([a-zA-Z]+)|([a-zA-Z])\b/g;
    let match;
    while ((match = regex.exec(latex)) !== null) {
      const varName = match[1] || match[2];
      if (varName && !vars.includes(varName)) {
        vars.push(varName);
      }
    }
    return vars;
  }, [latex]);

  // Apply color coding to LaTeX
  const coloredLatex = useMemo(() => {
    let colored = latex;

    // Sort variables by length (longer first) to avoid partial replacements
    const sortedVars = [...variables].sort((a, b) => b.length - a.length);

    sortedVars.forEach((variable) => {
      const color = colorMap[variable] || getVariableColor(variable);
      const isHighlighted = highlightVariables.includes(variable) || hoveredVariable === variable;

      if (isHighlighted) {
        // Add textcolor command
        const escapedVar = variable.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const regex = new RegExp(`(?<!\\\\\\\\)${escapedVar}`, 'g');
        colored = colored.replace(regex, `\\\\textcolor{${color}}{${variable}}`);
      }
    });

    return colored;
  }, [latex, variables, colorMap, highlightVariables, hoveredVariable]);

  // Render LaTeX
  const renderLatex = (latexStr: string, displayMode: boolean = false): string => {
    try {
      return katex.renderToString(latexStr, {
        displayMode,
        throwOnError: false,
        trust: true,
        strict: false,
      });
    } catch (error) {
      console.error('KaTeX render error:', error);
      return latexStr;
    }
  };

  const getVariableColor = (variable: string): string => {
    // Map common variables to semantic colors
    const semanticMap: Record<string, string> = {
      'Q': getColors('query').primary,
      'K': getColors('key').primary,
      'V': getColors('value').primary,
      'X': getColors('embedding').primary,
      'W': getColors('mlp').primary,
      'A': getColors('attention').primary,
      'q': getColors('query').light,
      'k': getColors('key').light,
      'v': getColors('value').light,
      'x': getColors('embedding').light,
    };

    return semanticMap[variable] || '#6366f1';
  };

  return (
    <div className="color-coded-math-container">
      {interactive ? (
        <div
          dangerouslySetInnerHTML={{
            __html: renderLatex(coloredLatex, false),
          }}
          className="inline"
        />
      ) : (
        <span
          dangerouslySetInnerHTML={{
            __html: renderLatex(coloredLatex, false),
          }}
        />
      )}
    </div>
  );
};

/**
 * ColorCodedMathDisplay Component
 *
 * Full display mode version for standalone formulas
 */
export const ColorCodedMathDisplay: React.FC<
  ColorCodedMathProps & {
    description?: string;
    variableLegend?: Record<string, { color: string; description: string }>;
  }
> = ({
  latex,
  colorMap = {},
  highlightVariables = [],
  interactive = false,
  onVariableClick,
  description,
  variableLegend,
}) => {
  const { getColors } = useSemanticColors();

  // Default color mapping
  const defaultColorMap: Record<string, string> = {
    'Q': getColors('query').primary,
    'K': getColors('key').primary,
    'V': getColors('value').primary,
    'X': getColors('embedding').primary,
    'W_Q': getColors('query').secondary,
    'W_K': getColors('key').secondary,
    'W_V': getColors('value').secondary,
    'd_k': getColors('attention').primary,
    'd_model': getColors('embedding').secondary,
  };

  const mergedColorMap = { ...defaultColorMap, ...colorMap };

  return (
    <div className="color-coded-math-display p-6 bg-gradient-to-br from-blue-50 to-purple-50 rounded-lg border border-blue-200">
      <div className="flex justify-center items-center mb-4">
        <ColorCodedMath
          latex={latex}
          colorMap={mergedColorMap}
          highlightVariables={highlightVariables}
          interactive={interactive}
          onVariableClick={onVariableClick}
        />
      </div>

      {description && (
        <p className="text-sm text-gray-600 text-center mb-4">{description}</p>
      )}

      {variableLegend && (
        <div className="variable-legend mt-4 pt-4 border-t border-blue-200">
          <h4 className="text-sm font-medium mb-2">Variables:</h4>
          <div className="flex flex-wrap gap-2">
            {Object.entries(variableLegend).map(([variable, info]) => {
              const color = mergedColorMap[variable] || info.color;
              return (
                <div key={variable} className="flex items-center gap-1">
                  <span
                    className="w-3 h-3 rounded"
                    style={{ backgroundColor: color }}
                  />
                  <span className="text-xs font-mono">{variable}</span>
                  <span className="text-xs text-gray-500">- {info.description}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

/**
 * MathVariableHighlight Component
 *
 * Shows a formula with all variables highlighted on hover
 */
export const MathVariableHighlight: React.FC<{
  latex: string;
  variableDescriptions: Record<string, string>;
}> = ({ latex, variableDescriptions }) => {
  const [highlightedVar, setHighlightedVar] = useState<string | null>(null);
  const { getColors } = useSemanticColors();

  const colors: Record<string, string> = {
    'Q': getColors('query').primary,
    'K': getColors('key').primary,
    'V': getColors('value').primary,
    'X': getColors('embedding').primary,
  };

  const latexWithColors = useMemo(() => {
    let result = latex;
    Object.entries(variableDescriptions).forEach(([variable, _]) => {
      const color = colors[variable] || '#6366f1';
      if (highlightedVar === variable) {
        result = result.replace(
          new RegExp(`\\b${variable}\\b`, 'g'),
          `\\\\textcolor{${color}}{\\\\underline{${variable}}}`
        );
      }
    });
    return result;
  }, [latex, variableDescriptions, highlightedVar, colors]);

  return (
    <div className="math-variable-highlight">
      <div
        className="formula-container p-4 bg-white rounded-lg border"
        dangerouslySetInnerHTML={{
          __html: katex.renderToString(latexWithColors, {
            displayMode: true,
            throwOnError: false,
          }),
        }}
      />
      <div className="variable-list mt-4 grid grid-cols-2 gap-2">
        {Object.entries(variableDescriptions).map(([variable, description]) => (
          <button
            key={variable}
            className={`
              px-3 py-2 rounded-lg text-left text-sm transition-all
              ${highlightedVar === variable
                ? 'bg-blue-100 border-2 border-blue-400'
                : 'bg-gray-50 border border-gray-200 hover:bg-gray-100'
              }
            `}
            onClick={() => setHighlightedVar(highlightedVar === variable ? null : variable)}
            onMouseEnter={() => setHighlightedVar(variable)}
            onMouseLeave={() => setHighlightedVar(null)}
          >
            <span className="font-mono font-bold">{variable}</span>
            <span className="ml-2 text-gray-600">{description}</span>
          </button>
        ))}
      </div>
    </div>
  );
};
