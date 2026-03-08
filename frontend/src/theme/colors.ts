/**
 * Semantic color system for Transformer visualization
 *
 * This color system is inspired by Transformer Explainer and provides
 * consistent, meaningful colors across all visualizations.
 */

// ============================================
// SEMANTIC COLORS
// ============================================

/**
 * Query (Q) matrices - Blue theme
 * Represents the "query" vectors that seek information
 */
export const QUERY_COLORS = {
  primary: '#3b82f6',      // bright blue
  secondary: '#2563eb',    // darker blue
  light: '#93c5fd',        // light blue
  dark: '#1e40af',         // very dark blue
  gradient: ['#3b82f6', '#60a5fa', '#93c5fd'],
  rgb: { r: 59, g: 130, b: 246 },
} as const;

/**
 * Key (K) matrices - Red theme
 * Represents the "key" vectors that are queried against
 */
export const KEY_COLORS = {
  primary: '#ef4444',      // bright red
  secondary: '#dc2626',    // darker red
  light: '#fca5a5',        // light red
  dark: '#991b1b',         // very dark red
  gradient: ['#ef4444', '#f87171', '#fca5a5'],
  rgb: { r: 239, g: 68, b: 68 },
} as const;

/**
 * Value (V) matrices - Green theme
 * Represents the "value" vectors that contain the actual information
 */
export const VALUE_COLORS = {
  primary: '#22c55e',      // bright green
  secondary: '#16a34a',    // darker green
  light: '#86efac',        // light green
  dark: '#166534',         // very dark green
  gradient: ['#22c55e', '#4ade80', '#86efac'],
  rgb: { r: 34, g: 197, b: 94 },
} as const;

/**
 * Attention weights - Pink/Magenta theme
 * Represents attention weights and connections
 */
export const ATTENTION_COLORS = {
  primary: '#ec4899',      // bright pink
  secondary: '#db2777',    // darker pink
  light: '#f9a8d4',        // light pink
  dark: '#9d174d',         // very dark pink
  gradient: ['#ec4899', '#f472b6', '#f9a8d4'],
  rgb: { r: 236, g: 72, b: 153 },
} as const;

/**
 * Embedding vectors - Purple theme
 * Represents token embeddings and positional encodings
 */
export const EMBEDDING_COLORS = {
  primary: '#8b5cf6',      // bright purple
  secondary: '#7c3aed',    // darker purple
  light: '#c4b5fd',        // light purple
  dark: '#5b21b6',         // very dark purple
  gradient: ['#8b5cf6', '#a78bfa', '#c4b5fd'],
  rgb: { r: 139, g: 92, b: 246 },
} as const;

/**
 * MLP/FeedForward - Cyan theme
 * Represents feed-forward network computations
 */
export const MLP_COLORS = {
  primary: '#06b6d4',      // bright cyan
  secondary: '#0891b2',    // darker cyan
  light: '#67e8f9',        // light cyan
  dark: '#164e63',         // very dark cyan
  gradient: ['#06b6d4', '#22d3ee', '#67e8f9'],
  rgb: { r: 6, g: 182, b: 212 },
} as const;

/**
 * Layer Normalization - Orange theme
 * Represents normalization operations
 */
export const NORMALIZATION_COLORS = {
  primary: '#f59e0b',      // bright orange
  secondary: '#d97706',    // darker orange
  light: '#fbbf24',        // light orange
  dark: '#b45309',         // very dark orange
  gradient: ['#f59e0b', '#fbbf24', '#fcd34d'],
  rgb: { r: 245, g: 158, b: 11 },
} as const;

/**
 * Residual connections - Gray theme
 * Represents skip connections
 */
export const RESIDUAL_COLORS = {
  primary: '#6b7280',      // medium gray
  secondary: '#4b5563',    // darker gray
  light: '#d1d5db',        // light gray
  dark: '#1f2937',         // very dark gray
  gradient: ['#6b7280', '#9ca3af', '#d1d5db'],
  rgb: { r: 107, g: 114, b: 128 },
} as const;

/**
 * Output - Indigo theme
 * Represents final output tokens
 */
export const OUTPUT_COLORS = {
  primary: '#6366f1',      // bright indigo
  secondary: '#4f46e5',    // darker indigo
  light: '#a5b4fc',        // light indigo
  dark: '#3730a3',         // very dark indigo
  gradient: ['#6366f1', '#818cf8', '#a5b4fc'],
  rgb: { r: 99, g: 102, b: 241 },
} as const;

// ============================================
// COLOR PALETTES BY COMPONENT TYPE
// ============================================

/**
 * Get the semantic color palette for a given component type
 */
export const getSemanticColors = (type: 'query' | 'key' | 'value' | 'attention' | 'embedding' | 'mlp' | 'normalization' | 'residual' | 'output') => {
  switch (type) {
    case 'query':
      return QUERY_COLORS;
    case 'key':
      return KEY_COLORS;
    case 'value':
      return VALUE_COLORS;
    case 'attention':
      return ATTENTION_COLORS;
    case 'embedding':
      return EMBEDDING_COLORS;
    case 'mlp':
      return MLP_COLORS;
    case 'normalization':
      return NORMALIZATION_COLORS;
    case 'residual':
      return RESIDUAL_COLORS;
    case 'output':
      return OUTPUT_COLORS;
    default:
      return ATTENTION_COLORS;
  }
};

// ============================================
// ATTENTION MATRIX HEATMAP COLORS
// ============================================

/**
 * Color scales for attention matrix heatmaps
 */
export const HEATMAP_COLOR_SCALES = {
  viridis: [
    '#440154', '#482878', '#3e4989', '#31688e', '#26828e',
    '#1f9e89', '#35b779', '#6dcd59', '#b4de2c', '#fde725'
  ],
  plasma: [
    '#0d0887', '#46039f', '#7201a8', '#9c179e', '#bd3786',
    '#d8576b', '#ed7953', '#fb9f3a', '#fdca26', '#f0f921'
  ],
  inferno: [
    '#000004', '#1b0c42', '#4a0c6b', '#781c6d', '#a52c60',
    '#cf4446', '#ed6925', '#fb9b06', '#f7d13d', '#fcffa4'
  ],
  blues: [
    '#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6',
    '#4292c6', '#2171b5', '#08519c', '#08306b'
  ],
  reds: [
    '#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a',
    '#ef3b2c', '#cb181d', '#a50f15', '#67000d'
  ],
} as const;

// ============================================
// UTILITY FUNCTIONS
// ============================================

/**
 * Convert hex color to RGB object
 */
export const hexToRgb = (hex: string): { r: number; g: number; b: number } => {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? {
    r: parseInt(result[1], 16),
    g: parseInt(result[2], 16),
    b: parseInt(result[3], 16)
  } : { r: 0, g: 0, b: 0 };
};

/**
 * Convert RGB object to CSS rgba string
 */
export const rgbToRgba = (rgb: { r: number; g: number; b: number }, alpha: number = 1): string => {
  return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${alpha})`;
};

/**
 * Convert hex to CSS rgba string
 */
export const hexToRgba = (hex: string, alpha: number = 1): string => {
  const rgb = hexToRgb(hex);
  return rgbToRgba(rgb, alpha);
};

/**
 * Interpolate between two colors
 */
export const interpolateColor = (
  color1: string,
  color2: string,
  factor: number
): string => {
  const rgb1 = hexToRgb(color1);
  const rgb2 = hexToRgb(color2);

  const r = Math.round(rgb1.r + factor * (rgb2.r - rgb1.r));
  const g = Math.round(rgb1.g + factor * (rgb2.g - rgb1.g));
  const b = Math.round(rgb1.b + factor * (rgb2.b - rgb1.b));

  return `rgb(${r}, ${g}, ${b})`;
};

/**
 * Get color from a color scale based on a value (0-1)
 */
export const getScaleColor = (
  scale: keyof typeof HEATMAP_COLOR_SCALES,
  value: number
): string => {
  const colors = HEATMAP_COLOR_SCALES[scale];
  const clampedValue = Math.max(0, Math.min(1, value));
  const index = clampedValue * (colors.length - 1);
  const lowerIndex = Math.floor(index);
  const upperIndex = Math.ceil(index);
  const factor = index - lowerIndex;

  if (lowerIndex === upperIndex) {
    return colors[lowerIndex];
  }

  return interpolateColor(colors[lowerIndex], colors[upperIndex], factor);
};

/**
 * Create a gradient string for CSS
 */
export const createGradient = (
  colors: string[],
  direction: 'horizontal' | 'vertical' | 'diagonal' = 'horizontal'
): string => {
  const angle = direction === 'horizontal' ? '90deg' :
                direction === 'vertical' ? '180deg' : '135deg';
  return `linear-gradient(${angle}, ${colors.join(', ')})`;
};

/**
 * Get contrasting text color (black or white) based on background
 */
export const getContrastColor = (hexColor: string): string => {
  const rgb = hexToRgb(hexColor);
  const luminance = (0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b) / 255;
  return luminance > 0.5 ? '#000000' : '#ffffff';
};

// ============================================
// COLOR COMBINATIONS FOR QKV VISUALIZATION
// ============================================

/**
 * Combined QKV color palette for multi-component visualizations
 */
export const QKV_COLORS = {
  query: QUERY_COLORS,
  key: KEY_COLORS,
  value: VALUE_COLORS,
  gradient: [
    QUERY_COLORS.primary,
    KEY_COLORS.primary,
    VALUE_COLORS.primary
  ],
} as const;

/**
 * Attention computation stage colors
 */
export const ATTENTION_STAGE_COLORS = {
  queries: QUERY_COLORS.primary,
  keys: KEY_COLORS.primary,
  values: VALUE_COLORS.primary,
  dot_product: ATTENTION_COLORS.primary,
  scaled: ATTENTION_COLORS.secondary,
  masked: ATTENTION_COLORS.light,
  softmax: ATTENTION_COLORS.primary,
  weighted_sum: VALUE_COLORS.primary,
} as const;

// ============================================
// CSS CUSTOM PROPERTIES EXPORT
// ============================================

/**
 * CSS custom properties for use in styled-components or CSS files
 */
export const CSS_COLORS = `
  /* Query Colors */
  --color-query-primary: ${QUERY_COLORS.primary};
  --color-query-secondary: ${QUERY_COLORS.secondary};
  --color-query-light: ${QUERY_COLORS.light};
  --color-query-dark: ${QUERY_COLORS.dark};

  /* Key Colors */
  --color-key-primary: ${KEY_COLORS.primary};
  --color-key-secondary: ${KEY_COLORS.secondary};
  --color-key-light: ${KEY_COLORS.light};
  --color-key-dark: ${KEY_COLORS.dark};

  /* Value Colors */
  --color-value-primary: ${VALUE_COLORS.primary};
  --color-value-secondary: ${VALUE_COLORS.secondary};
  --color-value-light: ${VALUE_COLORS.light};
  --color-value-dark: ${VALUE_COLORS.dark};

  /* Attention Colors */
  --color-attention-primary: ${ATTENTION_COLORS.primary};
  --color-attention-secondary: ${ATTENTION_COLORS.secondary};
  --color-attention-light: ${ATTENTION_COLORS.light};
  --color-attention-dark: ${ATTENTION_COLORS.dark};

  /* Embedding Colors */
  --color-embedding-primary: ${EMBEDDING_COLORS.primary};
  --color-embedding-secondary: ${EMBEDDING_COLORS.secondary};
  --color-embedding-light: ${EMBEDDING_COLORS.light};
  --color-embedding-dark: ${EMBEDDING_COLORS.dark};

  /* MLP Colors */
  --color-mlp-primary: ${MLP_COLORS.primary};
  --color-mlp-secondary: ${MLP_COLORS.secondary};
  --color-mlp-light: ${MLP_COLORS.light};
  --color-mlp-dark: ${MLP_COLORS.dark};

  /* Normalization Colors */
  --color-normalization-primary: ${NORMALIZATION_COLORS.primary};
  --color-normalization-secondary: ${NORMALIZATION_COLORS.secondary};
  --color-normalization-light: ${NORMALIZATION_COLORS.light};
  --color-normalization-dark: ${NORMALIZATION_COLORS.dark};

  /* Residual Colors */
  --color-residual-primary: ${RESIDUAL_COLORS.primary};
  --color-residual-secondary: ${RESIDUAL_COLORS.secondary};
  --color-residual-light: ${RESIDUAL_COLORS.light};
  --color-residual-dark: ${RESIDUAL_COLORS.dark};

  /* Output Colors */
  --color-output-primary: ${OUTPUT_COLORS.primary};
  --color-output-secondary: ${OUTPUT_COLORS.secondary};
  --color-output-light: ${OUTPUT_COLORS.light};
  --color-output-dark: ${OUTPUT_COLORS.dark};
` as const;
