/**
 * Cache utility for memoizing computations
 *
 * Provides LRU (Least Recently Used) cache implementation
 * for expensive computations like tensor operations, color calculations, etc.
 */

import type { CacheEntry } from '../types/transformer';

/**
 * Default cache configuration
 */
const DEFAULT_CACHE_CONFIG = {
  maxSize: 100,
  defaultTTL: 5 * 60 * 1000, // 5 minutes
};

/**
 * Cache configuration options
 */
export interface CacheConfig {
  maxSize: number;
  defaultTTL: number;
}

/**
 * LRU Cache implementation
 */
export class LRUCache<K, V> {
  private cache: Map<K, CacheEntry<V>>;
  private maxSize: number;
  private defaultTTL: number;
  private hits: number = 0;
  private misses: number = 0;

  constructor(config: Partial<CacheConfig> = {}) {
    const { maxSize, defaultTTL } = { ...DEFAULT_CACHE_CONFIG, ...config };
    this.cache = new Map();
    this.maxSize = maxSize;
    this.defaultTTL = defaultTTL;
  }

  /**
   * Check if entry is expired
   */
  private isExpired(entry: CacheEntry<V>, ttl?: number): boolean {
    const now = Date.now();
    const entryTTL = ttl || this.defaultTTL;
    return now - entry.timestamp > entryTTL;
  }

  /**
   * Evict least recently used entry
   */
  private evict(): void {
    if (this.cache.size >= this.maxSize) {
      // Get first (least recently used) entry
      const firstKey = this.cache.keys().next().value;
      if (firstKey !== undefined) {
        this.cache.delete(firstKey);
      }
    }
  }

  /**
   * Move entry to end (mark as recently used)
   */
  private markAsUsed(key: K): void {
    const entry = this.cache.get(key);
    if (entry) {
      this.cache.delete(key);
      this.cache.set(key, entry);
    }
  }

  /**
   * Get value from cache
   */
  get(key: K, ttl?: number): V | null {
    const entry = this.cache.get(key);

    if (!entry) {
      this.misses++;
      return null;
    }

    if (this.isExpired(entry, ttl)) {
      this.cache.delete(key);
      this.misses++;
      return null;
    }

    this.hits++;
    entry.hits++;
    this.markAsUsed(key);
    return entry.data;
  }

  /**
   * Set value in cache
   */
  set(key: K, value: V, size: number = 1): void {
    this.evict();

    const entry: CacheEntry<V> = {
      data: value,
      timestamp: Date.now(),
      hits: 0,
      size,
    };

    this.cache.set(key, entry);
  }

  /**
   * Get or compute value
   */
  getOrCompute(key: K, compute: () => V, ttl?: number): V {
    const cached = this.get(key, ttl);
    if (cached !== null) {
      return cached;
    }

    const value = compute();
    this.set(key, value);
    return value;
  }

  /**
   * Check if key exists and is not expired
   */
  has(key: K, ttl?: number): boolean {
    const entry = this.cache.get(key);
    return entry !== undefined && !this.isExpired(entry, ttl);
  }

  /**
   * Delete entry from cache
   */
  delete(key: K): boolean {
    return this.cache.delete(key);
  }

  /**
   * Clear all entries
   */
  clear(): void {
    this.cache.clear();
    this.hits = 0;
    this.misses = 0;
  }

  /**
   * Get cache statistics
   */
  getStats(): {
    size: number;
    hits: number;
    misses: number;
    hitRate: number;
  } {
    const total = this.hits + this.misses;
    return {
      size: this.cache.size,
      hits: this.hits,
      misses: this.misses,
      hitRate: total > 0 ? this.hits / total : 0,
    };
  }

  /**
   * Clean expired entries
   */
  clean(ttl?: number): number {
    let cleaned = 0;
    for (const [key, entry] of this.cache.entries()) {
      if (this.isExpired(entry, ttl)) {
        this.cache.delete(key);
        cleaned++;
      }
    }
    return cleaned;
  }

  /**
   * Get all keys
   */
  keys(): K[] {
    return Array.from(this.cache.keys());
  }

  /**
   * Get all values
   */
  values(): V[] {
    return Array.from(this.cache.values()).map(entry => entry.data);
  }
}

/**
 * Default cache instance
 */
const defaultCache = new LRUCache<string, unknown>();

/**
 * Memoize a function with caching
 */
export function memoize<Args extends unknown[], Return>(
  fn: (...args: Args) => Return,
  options: Partial<CacheConfig> & {
    keyFn?: (...args: Args) => string;
    ttl?: number;
  } = {}
): (...args: Args) => Return {
  const cache = new LRUCache<string, Return>(options);
  const { keyFn, ttl } = options;

  return (...args: Args): Return => {
    const key = keyFn ? keyFn(...args) : JSON.stringify(args);
    return cache.getOrCompute(key, () => fn(...args), ttl);
  };
}

/**
 * Create a cache key from multiple values
 */
export function createCacheKey(...parts: unknown[]): string {
  return parts
    .map(part => {
      if (typeof part === 'object' && part !== null) {
        return JSON.stringify(part);
      }
      return String(part);
    })
    .join(':');
}

/**
 * Memoized tensor operations cache
 */
export const tensorCache = new LRUCache<string, number[][]>({
  maxSize: 50,
  defaultTTL: 10 * 60 * 1000, // 10 minutes
});

/**
 * Memoized color calculations cache
 */
export const colorCache = new LRUCache<string, string>({
  maxSize: 1000,
  defaultTTL: 15 * 60 * 1000, // 15 minutes
});

/**
 * Memoized heatmap cache for attention matrices
 */
export const heatmapCache = new LRUCache<string, string[][]>({
  maxSize: 100,
  defaultTTL: 5 * 60 * 1000, // 5 minutes
});

/**
 * Cache utilities for common operations
 */
export const cacheUtils = {
  /**
   * Clear all caches
   */
  clearAll(): void {
    defaultCache.clear();
    tensorCache.clear();
    colorCache.clear();
    heatmapCache.clear();
  },

  /**
   * Get statistics for all caches
   */
  getAllStats(): Record<string, ReturnType<LRUCache<unknown, unknown>['getStats']>> {
    return {
      default: defaultCache.getStats(),
      tensor: tensorCache.getStats(),
      color: colorCache.getStats(),
      heatmap: heatmapCache.getStats(),
    };
  },

  /**
   * Clean expired entries in all caches
   */
  cleanAll(): number {
    return (
      defaultCache.clean() +
      tensorCache.clean() +
      colorCache.clean() +
      heatmapCache.clean()
    );
  },
};

/**
 * Hook for using cache in components
 */
export function useCache<K extends string, V>(
  cache: LRUCache<K, V>,
  ttl?: number
): {
  get: (key: K) => V | null;
  set: (key: K, value: V) => void;
  has: (key: K) => boolean;
  delete: (key: K) => boolean;
  clear: () => void;
  stats: ReturnType<LRUCache<K, V>['getStats']>;
} {
  return {
    get: (key: K) => cache.get(key, ttl),
    set: (key: K, value: V) => cache.set(key, value),
    has: (key: K) => cache.has(key, ttl),
    delete: (key: K) => cache.delete(key),
    clear: () => cache.clear(),
    stats: cache.getStats(),
  };
}
