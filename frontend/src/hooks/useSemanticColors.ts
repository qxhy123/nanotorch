import { useCallback } from 'react';
import type {
  DisclosureLevel,
} from '../types/transformer';
import {
  QUERY_COLORS,
  KEY_COLORS,
  VALUE_COLORS,
  ATTENTION_COLORS,
  EMBEDDING_COLORS,
  MLP_COLORS,
  NORMALIZATION_COLORS,
  RESIDUAL_COLORS,
  OUTPUT_COLORS,
  getSemanticColors,
  getScaleColor,
  hexToRgba,
  getContrastColor,
} from '../theme/colors';

/**
 * Semantic color component types
 */
export type SemanticColorType =
  | 'query'
  | 'key'
  | 'value'
  | 'attention'
  | 'embedding'
  | 'mlp'
  | 'normalization'
  | 'residual'
  | 'output';

/**
 * Color scheme types for heatmaps
 */
export type ColorScheme = 'viridis' | 'plasma' | 'inferno' | 'blues' | 'reds';

/**
 * Hook for accessing semantic colors throughout components
 *
 * Provides:
 * - Semantic color palettes by component type
 * - Disclosure level based color intensity
 * - Color utilities
 * - Contrast calculations
 */
export const useSemanticColors = () => {
  /**
   * Get semantic colors for a specific component type
   */
  const getColors = useCallback((type: SemanticColorType) => {
    return getSemanticColors(type);
  }, []);

  /**
   * Get color with opacity based on disclosure level
   */
  const getColorWithOpacity = useCallback((
    colorHex: string,
    disclosureLevel: DisclosureLevel,
    baseOpacity: number = 1
  ): string => {
    const opacityMap: Record<DisclosureLevel, number> = {
      overview: 0.6,
      intermediate: 0.8,
      detailed: 0.9,
      math: 1,
    };

    const levelOpacity = opacityMap[disclosureLevel];
    return hexToRgba(colorHex, baseOpacity * levelOpacity);
  }, []);

  /**
   * Get QKV color combination
   */
  const getQKVColors = useCallback(() => {
    return {
      query: QUERY_COLORS,
      key: KEY_COLORS,
      value: VALUE_COLORS,
      gradient: [
        QUERY_COLORS.primary,
        KEY_COLORS.primary,
        VALUE_COLORS.primary,
      ],
    };
  }, []);

  /**
   * Get attention stage colors
   */
  const getAttentionStageColors = useCallback(() => {
    return {
      queries: QUERY_COLORS.primary,
      keys: KEY_COLORS.primary,
      values: VALUE_COLORS.primary,
      dotProduct: ATTENTION_COLORS.primary,
      scaled: ATTENTION_COLORS.secondary,
      masked: ATTENTION_COLORS.light,
      softmax: ATTENTION_COLORS.primary,
      weightedSum: VALUE_COLORS.primary,
    };
  }, []);

  /**
   * Get heatmap color scale
   */
  const getHeatmapColor = useCallback((
    value: number,
    scheme: ColorScheme = 'viridis'
  ): string => {
    return getScaleColor(scheme, value);
  }, []);

  /**
   * Get color for heatmap based on value
   */
  const getHeatmapColorScale = useCallback((
    values: number[],
    scheme: ColorScheme = 'viridis'
  ): string[] => {
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min;

    return values.map(v => {
      const normalized = range > 0 ? (v - min) / range : 0.5;
      return getScaleColor(scheme, normalized);
    });
  }, []);

  /**
   * Get contrasting text color for background
   */
  const getTextColor = useCallback((backgroundColor: string): string => {
    return getContrastColor(backgroundColor);
  }, []);

  /**
   * Get component-specific styling
   */
  const getComponentStyle = useCallback((
    type: SemanticColorType,
    options: {
      disclosureLevel?: DisclosureLevel;
      opacity?: number;
      variant?: 'primary' | 'secondary' | 'light' | 'dark';
    } = {}
  ) => {
    const colors = getColors(type);
    const { disclosureLevel = 'intermediate', opacity = 1, variant = 'primary' } = options;

    const color = colors[variant];
    const colorWithOpacity = getColorWithOpacity(color, disclosureLevel, opacity);

    return {
      backgroundColor: colorWithOpacity,
      borderColor: color,
      color: getTextColor(color),
      // Add hover state
      hoverColor: hexToRgba(color, opacity * 0.8),
      // Add active state
      activeColor: hexToRgba(color, opacity * 0.6),
    };
  }, [getColors, getColorWithOpacity, getTextColor]);

  return {
    // Color getters
    getColors,
    getColorWithOpacity,
    getQKVColors,
    getAttentionStageColors,
    getHeatmapColor,
    getHeatmapColorScale,
    getTextColor,
    getComponentStyle,

    // Direct color access
    query: QUERY_COLORS,
    key: KEY_COLORS,
    value: VALUE_COLORS,
    attention: ATTENTION_COLORS,
    embedding: EMBEDDING_COLORS,
    mlp: MLP_COLORS,
    normalization: NORMALIZATION_COLORS,
    residual: RESIDUAL_COLORS,
    output: OUTPUT_COLORS,
  };
};

/**
 * Extended hook with disclosure level awareness
 */
export const useSemanticColorsWithDisclosure = (disclosureLevel: DisclosureLevel) => {
  const baseColors = useSemanticColors();

  /**
   * Get colors adjusted for disclosure level
   */
  const getDisclosureColors = useCallback((type: SemanticColorType) => {
    const colors = baseColors.getColors(type);
    const intensityMap: Record<DisclosureLevel, 'light' | 'primary' | 'secondary' | 'dark'> = {
      overview: 'light',
      intermediate: 'primary',
      detailed: 'secondary',
      math: 'dark',
    };

    const variant = intensityMap[disclosureLevel];
    return {
      ...colors,
      active: colors[variant],
    };
  }, [disclosureLevel, baseColors]);

  /**
   * Check if element should be visible at current disclosure level
   */
  const isVisible = useCallback((
    elementLevel: DisclosureLevel,
    currentLevel: DisclosureLevel = disclosureLevel
  ): boolean => {
    const levels: DisclosureLevel[] = ['overview', 'intermediate', 'detailed', 'math'];
    return levels.indexOf(currentLevel) >= levels.indexOf(elementLevel);
  }, [disclosureLevel]);

  /**
   * Get opacity based on disclosure level difference
   */
  const getOpacity = useCallback((
    elementLevel: DisclosureLevel,
    currentLevel: DisclosureLevel = disclosureLevel
  ): number => {
    const levels: DisclosureLevel[] = ['overview', 'intermediate', 'detailed', 'math'];
    const elementIndex = levels.indexOf(elementLevel);
    const currentIndex = levels.indexOf(currentLevel);

    if (currentIndex < elementIndex) {
      // Element is at higher disclosure level - make it semi-transparent
      return 0.3;
    }
    return 1;
  }, [disclosureLevel]);

  return {
    ...baseColors,
    getDisclosureColors,
    isVisible,
    getOpacity,
    disclosureLevel,
  };
};

/**
 * Hook for matrix visualization colors
 */
export const useMatrixColors = () => {
  const { getHeatmapColor } = useSemanticColors();

  /**
   * Get color for a cell in a matrix
   */
  const getCellColor = useCallback((
    value: number,
    min: number,
    max: number,
    scheme: ColorScheme = 'viridis'
  ): string => {
    const normalized = max > min ? (value - min) / (max - min) : 0.5;
    return getHeatmapColor(normalized, scheme);
  }, [getHeatmapColor]);

  /**
   * Get colors for entire matrix row
   */
  const getRowColors = useCallback((
    row: number[],
    min: number,
    max: number,
    scheme: ColorScheme = 'viridis'
  ): string[] => {
    return row.map(v => getCellColor(v, min, max, scheme));
  }, [getCellColor]);

  /**
   * Get colors for entire matrix
   */
  const getMatrixColors = useCallback((
    matrix: number[][],
    min: number,
    max: number,
    scheme: ColorScheme = 'viridis'
  ): string[][] => {
    return matrix.map(row => getRowColors(row, min, max, scheme));
  }, [getRowColors]);

  /**
   * Get highlighted cell colors (for attention visualization)
   */
  const getHighlightedCellColors = useCallback((
    matrix: number[][],
    highlightIndices: Array<[number, number]>,
    scheme: ColorScheme = 'viridis'
  ): { colors: string[][], highlighted: string[][] } => {
    const flat = matrix.flat();
    const min = Math.min(...flat);
    const max = Math.max(...flat);

    const highlightSet = new Set(highlightIndices.map(([r, c]) => `${r},${c}`));

    const colors: string[][] = [];
    const highlighted: string[][] = [];

    matrix.forEach((row, r) => {
      colors[r] = [];
      highlighted[r] = [];
      row.forEach((val, c) => {
        const baseColor = getCellColor(val, min, max, scheme);
        colors[r][c] = baseColor;
        highlighted[r][c] = highlightSet.has(`${r},${c}`)
          ? baseColor
          : 'transparent';
      });
    });

    return { colors, highlighted };
  }, [getCellColor]);

  return {
    getCellColor,
    getRowColors,
    getMatrixColors,
    getHighlightedCellColors,
  };
};

/**
 * Hook for QKV visualization colors
 */
export const useQKVColors = () => {
  const { getQKVColors } = useSemanticColors();

  /**
   * Get color for Q, K, or V
   */
  const getQKorVColor = useCallback((type: 'query' | 'key' | 'value') => {
    const qkv = getQKVColors();
    return qkv[type].primary;
  }, [getQKVColors]);

  /**
   * Get QKV gradient for visualization
   */
  const getQKVGradient = useCallback(() => {
    const qkv = getQKVColors();
    return `linear-gradient(90deg, ${qkv.gradient.join(', ')})`;
  }, [getQKVColors]);

  /**
   * Get color for attention head (alternating Q/K/V colors)
   */
  const getHeadColor = useCallback((headIndex: number) => {
    const qkv = getQKVColors();
    const colors = [
      qkv.query.primary,
      qkv.key.primary,
      qkv.value.primary,
    ];
    return colors[headIndex % colors.length];
  }, [getQKVColors]);

  return {
    getQKorVColor,
    getQKVGradient,
    getHeadColor,
  };
};
