import type {
  GenerationStep,
  ProbabilityDistributionData,
  SamplingOptions,
  SamplingStrategy,
  StrategyComparison,
  TokenProbability,
} from '../types/inference';

function hashString(value: string): number {
  let hash = 2166136261;

  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }

  return hash >>> 0;
}

function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;

  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function buildTokenProbability(
  tokenId: number,
  token: string,
  probability: number,
  rank: number
): TokenProbability {
  return {
    tokenId,
    token,
    probability,
    logProbability: Math.log(probability + 1e-10),
    rank,
  };
}

function normalizeProbabilities(tokens: TokenProbability[]): TokenProbability[] {
  const total = tokens.reduce((sum, token) => sum + token.probability, 0);

  return tokens
    .map((token) => ({
      ...token,
      probability: token.probability / total,
      logProbability: Math.log(token.probability / total + 1e-10),
    }))
    .sort((left, right) => right.probability - left.probability)
    .map((token, index) => ({
      ...token,
      rank: index + 1,
    }));
}

export function generateProbabilityDistribution(
  sequence: string = 'The future of AI',
  vocabSize: number = 10000
): ProbabilityDistributionData {
  const tokens = sequence.split(' ').filter(Boolean);
  const effectiveTokens = tokens.length > 0 ? tokens : ['<empty>'];
  const seed = hashString(`distribution:${sequence}:${vocabSize}`);
  const rng = createSeededRandom(seed);
  const distributions = [];
  const steps: GenerationStep[] = [];

  for (let position = 0; position < effectiveTokens.length; position += 1) {
    const actualToken = effectiveTokens[position];
    const actualTokenIndex = (position * 17 + 11) % 40;
    const candidates: TokenProbability[] = [];

    for (let tokenIndex = 0; tokenIndex < Math.min(vocabSize, 100); tokenIndex += 1) {
      const isActualToken = tokenIndex === actualTokenIndex;
      const baseProbability = isActualToken
        ? 0.25 + rng() * 0.08
        : 0.002 + rng() * 0.012;

      candidates.push(
        buildTokenProbability(
          tokenIndex,
          isActualToken ? actualToken : `token_${position}_${tokenIndex}`,
          baseProbability,
          tokenIndex + 1
        )
      );
    }

    const normalized = normalizeProbabilities(candidates);
    const entropy = -normalized.reduce(
      (sum, token) => sum + token.probability * Math.log2(token.probability + 1e-10),
      0
    );

    const distribution = {
      position,
      tokens: normalized,
      entropy,
      topToken: normalized[0],
      topKTokens: normalized.slice(0, 10),
      cumulativeProbability: normalized
        .slice(0, 10)
        .reduce((sum, token) => sum + token.probability, 0),
    };

    distributions.push(distribution);
    steps.push({
      stepIndex: position,
      position,
      generatedToken: {
        tokenId: normalized[0].tokenId,
        token: normalized[0].token,
        probability: normalized[0].probability,
        chosenStrategy: 'greedy',
        alternatives: normalized.slice(0, 5),
      },
      distribution,
      context: effectiveTokens.slice(0, position + 1).join(' '),
      timeTaken: 55 + rng() * 85,
    });
  }

  return {
    sequence,
    tokens: effectiveTokens,
    distributions,
    samplingOptions: {
      strategy: 'greedy',
      temperature: 1.0,
      topK: 50,
      topP: 0.9,
      beamWidth: 5,
    },
    generatedSequence: sequence,
    steps,
  };
}

export function generateAutoregressiveSteps(
  prompt: string,
  maxLength: number = 10,
  strategy: SamplingStrategy = 'greedy'
): GenerationStep[] {
  const promptTokens = prompt.split(' ').filter(Boolean);
  const seed = hashString(`autoregressive:${prompt}:${maxLength}:${strategy}`);
  const rng = createSeededRandom(seed);
  const continuations = [
    'is',
    'will be',
    'has become',
    'represents',
    'brings',
    'offers',
    'creates',
    'transforms',
  ];
  const nextWords = [
    'a new era of',
    'incredible opportunities for',
    'groundbreaking advances in',
    'unprecedented possibilities in',
    'remarkable solutions for',
  ];
  const steps: GenerationStep[] = [];
  let currentSequence = [...promptTokens];

  steps.push({
    stepIndex: 0,
    position: Math.max(0, promptTokens.length - 1),
    generatedToken: {
      tokenId: -1,
      token: '<START>',
      probability: 1.0,
      chosenStrategy: strategy,
      alternatives: [],
    },
    distribution: {
      position: Math.max(0, promptTokens.length - 1),
      tokens: [],
      entropy: 0,
      topToken: buildTokenProbability(-1, '<START>', 1.0, 1),
      topKTokens: [],
      cumulativeProbability: 1.0,
    },
    context: prompt,
    timeTaken: 0,
  });

  for (let stepIndex = 0; stepIndex < maxLength; stepIndex += 1) {
    const continuation = continuations[Math.floor(rng() * continuations.length)];
    const nextWord = nextWords[Math.floor(rng() * nextWords.length)];
    const generatedText = stepIndex < 3 ? continuation : `${continuation} ${nextWord}`;
    const probabilities = normalizeProbabilities([
      buildTokenProbability(stepIndex * 10, generatedText, 0.28 + rng() * 0.08, 1),
      buildTokenProbability(stepIndex * 10 + 1, 'might be', 0.14 + rng() * 0.03, 2),
      buildTokenProbability(stepIndex * 10 + 2, 'could be', 0.11 + rng() * 0.03, 3),
      buildTokenProbability(stepIndex * 10 + 3, 'should be', 0.08 + rng() * 0.02, 4),
      buildTokenProbability(stepIndex * 10 + 4, 'would be', 0.06 + rng() * 0.02, 5),
      buildTokenProbability(stepIndex * 10 + 5, 'can be', 0.05 + rng() * 0.02, 6),
    ]);

    currentSequence = [...currentSequence, ...generatedText.split(' ')];
    const distribution = {
      position: currentSequence.length - 1,
      tokens: probabilities,
      entropy: -probabilities.reduce(
        (sum, token) => sum + token.probability * Math.log2(token.probability + 1e-10),
        0
      ),
      topToken: probabilities[0],
      topKTokens: probabilities.slice(0, 6),
      cumulativeProbability: probabilities.reduce((sum, token) => sum + token.probability, 0),
    };

    steps.push({
      stepIndex: stepIndex + 1,
      position: currentSequence.length - 1,
      generatedToken: {
        tokenId: probabilities[0].tokenId,
        token: probabilities[0].token,
        probability: probabilities[0].probability,
        chosenStrategy: strategy,
        alternatives: probabilities.slice(0, 5),
      },
      distribution,
      context: currentSequence.join(' '),
      timeTaken: 65 + rng() * 95,
    });

    if (currentSequence.length > 20) {
      break;
    }
  }

  return steps;
}

export function generateStrategyComparisons(
  sequence: string,
  selectedStrategies: SamplingStrategy[],
  options: Partial<SamplingOptions>
): StrategyComparison[] {
  const seed = hashString(`comparison:${sequence}:${selectedStrategies.join(',')}`);
  const rng = createSeededRandom(seed);
  const resultMap: Record<SamplingStrategy, string> = {
    greedy: 'The future of AI is bright and promising',
    multinomial: 'The future of AI holds surprising possibilities',
    'top-k': 'The future of AI will transform society',
    'top-p': 'The future of AI looks incredibly exciting',
    'beam-search': 'The future of AI can be explored through multiple strong candidates',
  };

  return selectedStrategies.map((strategy, index) => ({
    strategy,
    options: {
      strategy,
      temperature: options.temperature ?? 1.0,
      topK: options.topK ?? 50,
      topP: options.topP ?? 0.9,
      beamWidth: options.beamWidth ?? 5,
    },
    result: resultMap[strategy],
    steps: generateAutoregressiveSteps(sequence, 4 + (index % 2), strategy),
    timeMs: 95 + rng() * 110 + index * 12,
  }));
}

export const inferenceDemoApi = {
  generateProbabilityDistribution,
  generateAutoregressiveSteps,
  generateStrategyComparisons,
};
