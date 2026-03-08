/**
 * Color utility functions for Transformer visualization
 */

import type { DisclosureLevel } from '../types/transformer';

/**
 * Color scheme types for heatmaps
 */
export type ColorScheme = 'viridis' | 'plasma' | 'inferno' | 'blues' | 'reds';

/**
 * Viridis color scale
 */
export const VIRIDIS_SCALE = [
  '#440154', '#482878', '#3e4989', '#31688e', '#26828e',
  '#1f9e89', '#35b779', '#6dcd59', '#b4de2c', '#fde725'
];

/**
 * Plasma color scale
 */
export const PLASMA_SCALE = [
  '#0d0887', '#46039f', '#7201a8', '#9c179e', '#bd3786',
  '#d8576b', '#ed7953', '#fb9f3a', '#fdca26', '#f0f921'
];

/**
 * Inferno color scale
 */
export const INFERNO_SCALE = [
  '#000004', '#1b0c42', '#4a0c6b', '#781c6d', '#a52c60',
  '#cf4446', '#ed6925', '#fb9b06', '#f7d13d', '#fcffa4'
];

/**
 * Blues color scale
 */
export const BLUES_SCALE = [
  '#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6',
  '#4292c6', '#2171b5', '#08519c', '#08306b'
];

/**
 * Reds color scale
 */
export const REDS_SCALE = [
  '#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a',
  '#ef3b2c', '#cb181d', '#a50f15', '#67000d'
];

/**
 * All color scales
 */
export const COLOR_SCALES: Record<ColorScheme, string[]> = {
  viridis: VIRIDIS_SCALE,
  plasma: PLASMA_SCALE,
  inferno: INFERNO_SCALE,
  blues: BLUES_SCALE,
  reds: REDS_SCALE,
};

/**
 * Convert hex color to RGB object
 */
export function hexToRgb(hex: string): { r: number; g: number; b: number } {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? {
    r: parseInt(result[1], 16),
    g: parseInt(result[2], 16),
    b: parseInt(result[3], 16)
  } : { r: 0, g: 0, b: 0 };
}

/**
 * Convert RGB object to CSS rgba string
 */
export function rgbToRgba(rgb: { r: number; g: number; b: number }, alpha: number = 1): string {
  return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${alpha})`;
}

/**
 * Convert hex to CSS rgba string
 */
export function hexToRgba(hex: string, alpha: number = 1): string {
  const rgb = hexToRgb(hex);
  return rgbToRgba(rgb, alpha);
}

/**
 * Interpolate between two colors
 */
export function interpolateColor(
  color1: string,
  color2: string,
  factor: number
): string {
  const rgb1 = hexToRgb(color1);
  const rgb2 = hexToRgb(color2);

  const r = Math.round(rgb1.r + factor * (rgb2.r - rgb1.r));
  const g = Math.round(rgb1.g + factor * (rgb2.g - rgb1.g));
  const b = Math.round(rgb1.b + factor * (rgb2.b - rgb1.b));

  return `rgb(${r}, ${g}, ${b})`;
}

/**
 * Get color from a color scale based on a value (0-1)
 */
export function getScaleColor(
  scale: ColorScheme,
  value: number
): string {
  const colors = COLOR_SCALES[scale];
  const clampedValue = Math.max(0, Math.min(1, value));
  const index = clampedValue * (colors.length - 1);
  const lowerIndex = Math.floor(index);
  const upperIndex = Math.ceil(index);
  const factor = index - lowerIndex;

  if (lowerIndex === upperIndex) {
    return colors[lowerIndex];
  }

  return interpolateColor(colors[lowerIndex], colors[upperIndex], factor);
}

/**
 * Create a gradient string for CSS
 */
export function createGradient(
  colors: string[],
  direction: 'horizontal' | 'vertical' | 'diagonal' = 'horizontal'
): string {
  const angle = direction === 'horizontal' ? '90deg' :
                direction === 'vertical' ? '180deg' : '135deg';
  return `linear-gradient(${angle}, ${colors.join(', ')})`;
}

/**
 * Get contrasting text color (black or white) based on background
 */
export function getContrastColor(hexColor: string): string {
  const rgb = hexToRgb(hexColor);
  const luminance = (0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b) / 255;
  return luminance > 0.5 ? '#000000' : '#ffffff';
}

/**
 * Calculate perceived brightness of a color
 */
export function getBrightness(hexColor: string): number {
  const rgb = hexToRgb(hexColor);
  return (rgb.r * 299 + rgb.g * 587 + rgb.b * 114) / 1000;
}

/**
 * Darken a color by a percentage
 */
export function darkenColor(hexColor: string, percent: number): string {
  const rgb = hexToRgb(hexColor);
  const factor = 1 - percent / 100;

  const r = Math.round(rgb.r * factor);
  const g = Math.round(rgb.g * factor);
  const b = Math.round(rgb.b * factor);

  return `rgb(${r}, ${g}, ${b})`;
}

/**
 * Lighten a color by a percentage
 */
export function lightenColor(hexColor: string, percent: number): string {
  const rgb = hexToRgb(hexColor);
  const factor = 1 + percent / 100;

  const r = Math.min(255, Math.round(rgb.r * factor));
  const g = Math.min(255, Math.round(rgb.g * factor));
  const b = Math.min(255, Math.round(rgb.b * factor));

  return `rgb(${r}, ${g}, ${b})`;
}

/**
 * Blend two colors
 */
export function blendColors(color1: string, color2: string, ratio: number): string {
  const rgb1 = hexToRgb(color1);
  const rgb2 = hexToRgb(color2);
  const factor = Math.max(0, Math.min(1, ratio));

  const r = Math.round(rgb1.r + factor * (rgb2.r - rgb1.r));
  const g = Math.round(rgb1.g + factor * (rgb2.g - rgb1.g));
  const b = Math.round(rgb1.b + factor * (rgb2.b - rgb1.b));

  return `rgb(${r}, ${g}, ${b})`;
}

/**
 * Get opacity-adjusted color based on disclosure level
 */
export function getDisclosureOpacity(
  baseOpacity: number,
  elementLevel: DisclosureLevel,
  currentLevel: DisclosureLevel
): number {
  const levels: DisclosureLevel[] = ['overview', 'intermediate', 'detailed', 'math'];
  const elementIndex = levels.indexOf(elementLevel);
  const currentIndex = levels.indexOf(currentLevel);

  if (currentIndex < elementIndex) {
    // Element is at higher disclosure level - reduce opacity
    return baseOpacity * 0.3;
  }
  return baseOpacity;
}

/**
 * Generate a color palette from a base color
 */
export function generateColorPalette(
  baseColor: string,
  steps: number = 5
): string[] {
  const palette: string[] = [];
  const rgb = hexToRgb(baseColor);

  for (let i = 0; i < steps; i++) {
    const factor = i / (steps - 1);
    const r = Math.round(rgb.r * (1 - factor * 0.5));
    const g = Math.round(rgb.g * (1 - factor * 0.5));
    const b = Math.round(rgb.b * (1 - factor * 0.5));
    palette.push(`rgb(${r}, ${g}, ${b})`);
  }

  return palette;
}

/**
 * Get color for a value in a range
 */
export function getValueColor(
  value: number,
  min: number,
  max: number,
  scheme: ColorScheme = 'viridis'
): string {
  if (max === min) return getScaleColor(scheme, 0.5);
  const normalized = (value - min) / (max - min);
  return getScaleColor(scheme, Math.max(0, Math.min(1, normalized)));
}

/**
 * Convert HSL to RGB
 */
export function hslToRgb(h: number, s: number, l: number): string {
  let r: number, g: number, b: number;

  if (s === 0) {
    r = g = b = l;
  } else {
    const hue2rgb = (p: number, q: number, t: number) => {
      if (t < 0) t += 1;
      if (t > 1) t -= 1;
      if (t < 1/6) return p + (q - p) * 6 * t;
      if (t < 1/2) return q;
      if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
      return p;
    };

    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hue2rgb(p, q, h + 1/3);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1/3);
  }

  return `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;
}

/**
 * Create rainbow color palette
 */
export function createRainbowPalette(steps: number): string[] {
  const palette: string[] = [];
  for (let i = 0; i < steps; i++) {
    const h = i / steps;
    palette.push(hslToRgb(h, 0.7, 0.5));
  }
  return palette;
}
