/**
 * ArchitectureComparison Component
 *
 * Interactive tool for comparing different Transformer architectures:
 * - Compare Original Transformer, BERT, GPT, T5, BART, etc.
 * - Side-by-side architectural differences visualization
 * - Key parameters comparison
 * - Interactive component highlighting
 * - Performance metrics and benchmarks
 * - Use case recommendations
 */

import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../ui/card';
import { Button } from '../../ui/button';
import { Select } from '../../ui/select';
import {
  Layers,
  Zap,
  TrendingUp,
  Code,
  Check,
  X,
} from 'lucide-react';

// Architecture types
type ArchitectureType =
  | 'original'
  | 'bert'
  | 'gpt'
  | 't5'
  | 'bart'
  | 'roberta'
  | 'distilbert'
  | 'electra';

// Architecture definitions
const ARCHITECTURES: Record<
  ArchitectureType,
  {
    name: string;
    fullName: string;
    description: string;
    year: number;
    type: 'encoder-decoder' | 'encoder-only' | 'decoder-only';
    keyFeatures: string[];
    strengths: string[];
    weaknesses: string[];
    useCases: string[];
    baseConfig: {
      layers: number;
      heads: number;
      dModel: number;
      dFF: number;
      vocabSize: number;
      maxPositions: number;
      parameters: string;
    };
    performance: {
      glueScore: string;
      squadScore: string;
      trainingTime: string;
      inferenceSpeed: string;
    };
  }
> = {
  original: {
    name: 'Transformer',
    fullName: 'Original Transformer',
    description: 'The original "Attention Is All You Need" architecture with encoder-decoder structure',
    year: 2017,
    type: 'encoder-decoder',
    keyFeatures: [
      'Multi-head self-attention',
      'Positional encoding',
      'Encoder-decoder architecture',
      'Scaled dot-product attention',
      'Residual connections',
    ],
    strengths: [
      'Parallel processing',
      'Strong performance on seq2seq tasks',
      'Handles long-range dependencies',
      'Foundation for all modern Transformers',
    ],
    weaknesses: [
      'Computationally expensive',
      'Large model size',
      'Requires large datasets',
      'Not optimized for specific tasks',
    ],
    useCases: [
      'Machine translation',
      'Text summarization',
      'Question answering',
      'Text generation',
    ],
    baseConfig: {
      layers: 6,
      heads: 8,
      dModel: 512,
      dFF: 2048,
      vocabSize: 37000,
      maxPositions: 512,
      parameters: '65M',
    },
    performance: {
      glueScore: 'N/A',
      squadScore: 'N/A',
      trainingTime: 'Baseline',
      inferenceSpeed: 'Baseline',
    },
  },
  bert: {
    name: 'BERT',
    fullName: 'Bidirectional Encoder Representations from Transformers',
    description: 'Encoder-only architecture pre-trained with masked language modeling and next sentence prediction',
    year: 2018,
    type: 'encoder-only',
    keyFeatures: [
      'Bidirectional context',
      'Masked Language Modeling (MLM)',
      'Next Sentence Prediction (NSP)',
      '[CLS] and [SEP] special tokens',
      'Segment embeddings',
    ],
    strengths: [
      'Deep bidirectional understanding',
      'Excellent for NLU tasks',
      'Strong transfer learning',
      'State-of-the-art on many benchmarks',
    ],
    weaknesses: [
      'Not designed for generation',
      'Fixed-length inputs (512 tokens)',
      'Masking during pre-training',
      'Heavy computational cost',
    ],
    useCases: [
      'Text classification',
      'Named entity recognition',
      'Question answering',
      'Sentiment analysis',
    ],
    baseConfig: {
      layers: 12,
      heads: 12,
      dModel: 768,
      dFF: 3072,
      vocabSize: 30522,
      maxPositions: 512,
      parameters: '110M',
    },
    performance: {
      glueScore: '80.5',
      squadScore: '93.2',
      trainingTime: '4 days',
      inferenceSpeed: 'Medium',
    },
  },
  gpt: {
    name: 'GPT',
    fullName: 'Generative Pre-trained Transformer',
    description: 'Decoder-only architecture with causal attention for autoregressive text generation',
    year: 2018,
    type: 'decoder-only',
    keyFeatures: [
      'Autoregressive generation',
      'Causal attention mask',
      'Language modeling objective',
      'Left-to-right processing',
      'Zero-shot capabilities',
    ],
    strengths: [
      'Excellent text generation',
      'Strong zero-shot performance',
      'Scalable architecture',
      'Efficient inference',
    ],
    weaknesses: [
      'Unidirectional context',
      'Limited understanding tasks',
      'No bidirectional context',
      'Requires large compute',
    ],
    useCases: [
      'Text generation',
      'Story writing',
      'Code generation',
      'Chatbots',
    ],
    baseConfig: {
      layers: 12,
      heads: 12,
      dModel: 768,
      dFF: 3072,
      vocabSize: 50257,
      maxPositions: 1024,
      parameters: '117M',
    },
    performance: {
      glueScore: '72.0',
      squadScore: 'N/A',
      trainingTime: '8 days',
      inferenceSpeed: 'Fast',
    },
  },
  t5: {
    name: 'T5',
    fullName: 'Text-to-Text Transfer Transformer',
    description: 'Encoder-decoder architecture that frames all NLP tasks as text-to-text problems',
    year: 2019,
    type: 'encoder-decoder',
    keyFeatures: [
      'Text-to-text framework',
      'Unified task format',
      'Span corruption objective',
      'Various model sizes',
      'Multilingual support',
    ],
    strengths: [
      'Unified approach to all tasks',
      'Flexible architecture',
      'Strong performance',
      'Easy to adapt',
    ],
    weaknesses: [
      'Requires task formatting',
      'Slower than decoder-only',
      'More complex architecture',
      'Higher memory usage',
    ],
    useCases: [
      'Translation',
      'Summarization',
      'Classification',
      'Question answering',
    ],
    baseConfig: {
      layers: 12,
      heads: 12,
      dModel: 768,
      dFF: 3072,
      vocabSize: 32128,
      maxPositions: 512,
      parameters: '220M',
    },
    performance: {
      glueScore: '82.0',
      squadScore: '91.5',
      trainingTime: '6 days',
      inferenceSpeed: 'Medium',
    },
  },
  bart: {
    name: 'BART',
    fullName: 'Bidirectional and Auto-Regressive Transformers',
    description: 'Denoising autoencoder with bidirectional encoder and autoregressive decoder',
    year: 2019,
    type: 'encoder-decoder',
    keyFeatures: [
      'Denoising objective',
      'Bidirectional encoder',
      'Autoregressive decoder',
      'Pre-training with corruption',
      'Strong for generation',
    ],
    strengths: [
      'Excellent for summarization',
      'Strong generation quality',
      'Bidirectional understanding',
      'Handles long sequences',
    ],
    weaknesses: [
      'Complex architecture',
      'Slower than encoder-only',
      'High memory usage',
      'Long training time',
    ],
    useCases: [
      'Text summarization',
      'Machine translation',
      'Text generation',
      'Dialog systems',
    ],
    baseConfig: {
      layers: 12,
      heads: 16,
      dModel: 1024,
      dFF: 4096,
      vocabSize: 50265,
      maxPositions: 1024,
      parameters: '400M',
    },
    performance: {
      glueScore: 'N/A',
      squadScore: 'N/A',
      trainingTime: '5 days',
      inferenceSpeed: 'Medium',
    },
  },
  roberta: {
    name: 'RoBERTa',
    fullName: 'Robustly Optimized BERT Approach',
    description: 'Optimized BERT with dynamic masking, larger batches, and longer training',
    year: 2019,
    type: 'encoder-only',
    keyFeatures: [
      'Dynamic masking',
      'Larger batch sizes',
      'Longer training',
      'No NSP objective',
      'Larger byte-level BPE',
    ],
    strengths: [
      'Better than BERT on most tasks',
      'More robust training',
      'Simpler pre-training',
      'Strong performance',
    ],
    weaknesses: [
      'Higher computational cost',
      'Longer training time',
      'Larger memory footprint',
      'Still not for generation',
    ],
    useCases: [
      'Text classification',
      'Question answering',
      'NLU tasks',
      'Benchmarking',
    ],
    baseConfig: {
      layers: 12,
      heads: 12,
      dModel: 768,
      dFF: 3072,
      vocabSize: 50265,
      maxPositions: 512,
      parameters: '125M',
    },
    performance: {
      glueScore: '88.5',
      squadScore: '94.6',
      trainingTime: '5 days',
      inferenceSpeed: 'Medium',
    },
  },
  distilbert: {
    name: 'DistilBERT',
    fullName: 'Distilled BERT',
    description: 'Smaller, faster, cheaper version of BERT using knowledge distillation',
    year: 2019,
    type: 'encoder-only',
    keyFeatures: [
      'Knowledge distillation',
      '40% fewer parameters',
      '60% faster inference',
      'Retains 97% performance',
      'Single encoder layer removed',
    ],
    strengths: [
      'Much faster inference',
      'Smaller model size',
      'Good performance',
      'Lower computational cost',
    ],
    weaknesses: [
      'Lower accuracy than BERT',
      'Still encoder-only',
      'Not for generation',
      'Distillation complexity',
    ],
    useCases: [
      'Edge deployment',
      'Real-time applications',
      'Resource-constrained systems',
      'Mobile applications',
    ],
    baseConfig: {
      layers: 6,
      heads: 12,
      dModel: 768,
      dFF: 3072,
      vocabSize: 30522,
      maxPositions: 512,
      parameters: '66M',
    },
    performance: {
      glueScore: '77.5',
      squadScore: '90.0',
      trainingTime: '3 days',
      inferenceSpeed: 'Fast',
    },
  },
  electra: {
    name: 'ELECTRA',
    fullName: 'Efficiently Learning an Encoder that Classifies Token Replacements Accurately',
    description: 'Pre-training method using replaced token detection instead of masked language modeling',
    year: 2020,
    type: 'encoder-only',
    keyFeatures: [
      'Replaced token detection',
      'Discriminator architecture',
      'More efficient training',
      'Sample-efficient',
      'Generator-discriminator',
    ],
    strengths: [
      'Very efficient training',
      'Strong performance',
      'Less compute than BERT',
      'Sample efficient',
      'Scalable',
    ],
    weaknesses: [
      'Complex two-model setup',
      'Not as widely adopted',
      'Generation focus limited',
      'Training complexity',
    ],
    useCases: [
      'Text classification',
      'Question answering',
      'NLU tasks',
      'Resource-efficient training',
    ],
    baseConfig: {
      layers: 12,
      heads: 12,
      dModel: 768,
      dFF: 3072,
      vocabSize: 30522,
      maxPositions: 512,
      parameters: '14M (discriminator only)',
    },
    performance: {
      glueScore: '84.0',
      squadScore: '92.0',
      trainingTime: '4 days',
      inferenceSpeed: 'Fast',
    },
  },
};

interface ArchitectureComparisonProps {
  className?: string;
}

export const ArchitectureComparison: React.FC<ArchitectureComparisonProps> = ({
  className = '',
}) => {
  const [selectedArch1, setSelectedArch1] = useState<ArchitectureType>('original');
  const [selectedArch2, setSelectedArch2] = useState<ArchitectureType>('bert');
  const [activeTab, setActiveTab] = useState<'overview' | 'config' | 'performance' | 'components'>('overview');

  const arch1 = ARCHITECTURES[selectedArch1];
  const arch2 = ARCHITECTURES[selectedArch2];

  // Comparison metrics
  const comparisonMetrics = useMemo(() => {
    return [
      {
        label: 'Parameters',
        arch1: arch1.baseConfig.parameters,
        arch2: arch2.baseConfig.parameters,
        better: 'smaller',
      },
      {
        label: 'Layers',
        arch1: arch1.baseConfig.layers,
        arch2: arch2.baseConfig.layers,
        better: 'optimal',
      },
      {
        label: 'Attention Heads',
        arch1: arch1.baseConfig.heads,
        arch2: arch2.baseConfig.heads,
        better: 'optimal',
      },
      {
        label: 'Model Dimension',
        arch1: arch1.baseConfig.dModel,
        arch2: arch2.baseConfig.dModel,
        better: 'optimal',
      },
      {
        label: 'FFN Dimension',
        arch1: arch1.baseConfig.dFF,
        arch2: arch2.baseConfig.dFF,
        better: 'optimal',
      },
      {
        label: 'Max Sequence Length',
        arch1: arch1.baseConfig.maxPositions,
        arch2: arch2.baseConfig.maxPositions,
        better: 'larger',
      },
    ];
  }, [arch1, arch2]);

  // Render architecture block diagram
  const renderArchitectureDiagram = (arch: ArchitectureType) => {
    const config = ARCHITECTURES[arch];

    return (
      <div className="flex flex-col items-center gap-2 p-4 bg-muted rounded-lg">
        <div className="text-sm font-medium">{config.name}</div>

        {config.type === 'encoder-only' && (
          <>
            <div className="w-32 h-8 bg-blue-500 rounded flex items-center justify-center text-white text-xs">
              Encoder x{config.baseConfig.layers}
            </div>
            <div className="w-1 h-4 bg-blue-300"></div>
            <div className="w-32 h-8 bg-green-500 rounded flex items-center justify-center text-white text-xs">
              Output
            </div>
          </>
        )}

        {config.type === 'decoder-only' && (
          <>
            <div className="w-32 h-8 bg-purple-500 rounded flex items-center justify-center text-white text-xs">
              Decoder x{config.baseConfig.layers}
            </div>
            <div className="w-1 h-4 bg-purple-300"></div>
            <div className="w-32 h-8 bg-green-500 rounded flex items-center justify-center text-white text-xs">
              Output
            </div>
          </>
        )}

        {config.type === 'encoder-decoder' && (
          <>
            <div className="w-32 h-8 bg-blue-500 rounded flex items-center justify-center text-white text-xs">
              Encoder x{Math.floor(config.baseConfig.layers / 2)}
            </div>
            <div className="w-1 h-4 bg-blue-300"></div>
            <div className="w-32 h-8 bg-purple-500 rounded flex items-center justify-center text-white text-xs">
              Decoder x{Math.floor(config.baseConfig.layers / 2)}
            </div>
            <div className="w-1 h-4 bg-purple-300"></div>
            <div className="w-32 h-8 bg-green-500 rounded flex items-center justify-center text-white text-xs">
              Output
            </div>
          </>
        )}
      </div>
    );
  };

  // Get compatibility status
  const getCompatibilityStatus = (arch1: ArchitectureType, arch2: ArchitectureType) => {
    const type1 = ARCHITECTURES[arch1].type;
    const type2 = ARCHITECTURES[arch2].type;

    if (type1 === type2) {
      return {
        status: 'compatible',
        message: `${ARCHITECTURES[arch1].name} and ${ARCHITECTURES[arch2].name} have the same architecture type (${type1})`,
      };
    }

    return {
      status: 'different',
      message: `${ARCHITECTURES[arch1].name} is ${type1} while ${ARCHITECTURES[arch2].name} is ${type2}`,
    };
  };

  const compatibility = getCompatibilityStatus(selectedArch1, selectedArch2);

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Layers className="h-5 w-5 text-primary" />
          Architecture Comparison Tool
        </CardTitle>
        <CardDescription>
          Compare different Transformer architectures side by side
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Architecture Selection */}
        <div className="grid md:grid-cols-2 gap-6">
          <div className="space-y-2">
            <label className="text-sm font-medium">First Architecture</label>
            <Select
              value={selectedArch1}
              onChange={(e) => setSelectedArch1(e.target.value as ArchitectureType)}
              className="w-full"
            >
              {Object.entries(ARCHITECTURES).map(([key, arch]) => (
                <option key={key} value={key}>
                  {arch.name} - {arch.fullName}
                </option>
              ))}
            </Select>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Second Architecture</label>
            <Select
              value={selectedArch2}
              onChange={(e) => setSelectedArch2(e.target.value as ArchitectureType)}
              className="w-full"
            >
              {Object.entries(ARCHITECTURES).map(([key, arch]) => (
                <option key={key} value={key}>
                  {arch.name} - {arch.fullName}
                </option>
              ))}
            </Select>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex flex-wrap gap-2 border-b">
          <Button
            variant={activeTab === 'overview' ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setActiveTab('overview')}
          >
            Overview
          </Button>
          <Button
            variant={activeTab === 'config' ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setActiveTab('config')}
          >
            Configuration
          </Button>
          <Button
            variant={activeTab === 'performance' ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setActiveTab('performance')}
          >
            Performance
          </Button>
          <Button
            variant={activeTab === 'components' ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setActiveTab('components')}
          >
            Components
          </Button>
        </div>

        {/* Overview Tab */}
        {activeTab === 'overview' && (
          <div className="space-y-4">
            {/* Architecture Diagrams */}
            <div className="grid md:grid-cols-2 gap-6">
              {renderArchitectureDiagram(selectedArch1)}
              {renderArchitectureDiagram(selectedArch2)}
            </div>

            {/* Description Cards */}
            <div className="grid md:grid-cols-2 gap-4">
              <div className="p-4 bg-muted rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-medium">{arch1.fullName}</h3>
                  <span className="text-xs text-muted-foreground">{arch1.year}</span>
                </div>
                <p className="text-sm text-muted-foreground">{arch1.description}</p>
                <div className="mt-3">
                  <span className="text-xs px-2 py-1 bg-primary/10 rounded-full text-primary">
                    {arch1.type}
                  </span>
                </div>
              </div>

              <div className="p-4 bg-muted rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-medium">{arch2.fullName}</h3>
                  <span className="text-xs text-muted-foreground">{arch2.year}</span>
                </div>
                <p className="text-sm text-muted-foreground">{arch2.description}</p>
                <div className="mt-3">
                  <span className="text-xs px-2 py-1 bg-primary/10 rounded-full text-primary">
                    {arch2.type}
                  </span>
                </div>
              </div>
            </div>

            {/* Key Features Comparison */}
            <div className="grid md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <h4 className="text-sm font-medium flex items-center gap-2">
                  <Zap className="h-4 w-4" />
                  {arch1.name} Key Features
                </h4>
                <ul className="space-y-1">
                  {arch1.keyFeatures.map((feature, idx) => (
                    <li key={idx} className="text-xs flex items-start gap-2">
                      <Check className="h-3 w-3 text-green-500 mt-0.5 flex-shrink-0" />
                      <span>{feature}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <div className="space-y-2">
                <h4 className="text-sm font-medium flex items-center gap-2">
                  <Zap className="h-4 w-4" />
                  {arch2.name} Key Features
                </h4>
                <ul className="space-y-1">
                  {arch2.keyFeatures.map((feature, idx) => (
                    <li key={idx} className="text-xs flex items-start gap-2">
                      <Check className="h-3 w-3 text-green-500 mt-0.5 flex-shrink-0" />
                      <span>{feature}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}

        {/* Configuration Tab */}
        {activeTab === 'config' && (
          <div className="space-y-4">
            <h3 className="text-sm font-medium">Configuration Comparison</h3>
            <div className="space-y-3">
              {comparisonMetrics.map((metric) => (
                <div key={metric.label} className="grid grid-cols-3 gap-4 items-center p-3 bg-muted rounded-lg">
                  <div className="text-sm font-medium">{metric.label}</div>
                  <div className="text-center font-mono text-sm">{metric.arch1}</div>
                  <div className="text-center font-mono text-sm">{metric.arch2}</div>
                </div>
              ))}
            </div>

            {/* Architecture Type Notice */}
            <div className={`p-4 rounded-lg border ${
              compatibility.status === 'compatible'
                ? 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800'
                : 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800'
            }`}>
              <div className="text-sm font-medium mb-1">
                {compatibility.status === 'compatible' ? 'Compatible Architecture Types' : 'Different Architecture Types'}
              </div>
              <div className="text-xs text-muted-foreground">{compatibility.message}</div>
            </div>
          </div>
        )}

        {/* Performance Tab */}
        {activeTab === 'performance' && (
          <div className="space-y-4">
            <h3 className="text-sm font-medium">Performance Metrics</h3>
            <div className="grid md:grid-cols-2 gap-4">
              {/* Arch 1 Performance */}
              <div className="space-y-3">
                <h4 className="text-sm font-medium">{arch1.name}</h4>
                <div className="space-y-2">
                  <div className="flex justify-between items-center p-2 bg-muted rounded">
                    <span className="text-xs">GLUE Score</span>
                    <span className="text-sm font-mono font-medium">{arch1.performance.glueScore}</span>
                  </div>
                  <div className="flex justify-between items-center p-2 bg-muted rounded">
                    <span className="text-xs">SQuAD Score</span>
                    <span className="text-sm font-mono font-medium">{arch1.performance.squadScore}</span>
                  </div>
                  <div className="flex justify-between items-center p-2 bg-muted rounded">
                    <span className="text-xs">Training Time</span>
                    <span className="text-sm font-mono font-medium">{arch1.performance.trainingTime}</span>
                  </div>
                  <div className="flex justify-between items-center p-2 bg-muted rounded">
                    <span className="text-xs">Inference Speed</span>
                    <span className="text-sm font-mono font-medium">{arch1.performance.inferenceSpeed}</span>
                  </div>
                </div>
              </div>

              {/* Arch 2 Performance */}
              <div className="space-y-3">
                <h4 className="text-sm font-medium">{arch2.name}</h4>
                <div className="space-y-2">
                  <div className="flex justify-between items-center p-2 bg-muted rounded">
                    <span className="text-xs">GLUE Score</span>
                    <span className="text-sm font-mono font-medium">{arch2.performance.glueScore}</span>
                  </div>
                  <div className="flex justify-between items-center p-2 bg-muted rounded">
                    <span className="text-xs">SQuAD Score</span>
                    <span className="text-sm font-mono font-medium">{arch2.performance.squadScore}</span>
                  </div>
                  <div className="flex justify-between items-center p-2 bg-muted rounded">
                    <span className="text-xs">Training Time</span>
                    <span className="text-sm font-mono font-medium">{arch2.performance.trainingTime}</span>
                  </div>
                  <div className="flex justify-between items-center p-2 bg-muted rounded">
                    <span className="text-xs">Inference Speed</span>
                    <span className="text-sm font-mono font-medium">{arch2.performance.inferenceSpeed}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Components Tab */}
        {activeTab === 'components' && (
          <div className="space-y-4">
            {/* Strengths */}
            <div className="grid md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <h4 className="text-sm font-medium flex items-center gap-2 text-green-600 dark:text-green-400">
                  <TrendingUp className="h-4 w-4" />
                  {arch1.name} Strengths
                </h4>
                <ul className="space-y-1">
                  {arch1.strengths.map((strength, idx) => (
                    <li key={idx} className="text-xs flex items-start gap-2">
                      <Check className="h-3 w-3 text-green-500 mt-0.5 flex-shrink-0" />
                      <span>{strength}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <div className="space-y-2">
                <h4 className="text-sm font-medium flex items-center gap-2 text-green-600 dark:text-green-400">
                  <TrendingUp className="h-4 w-4" />
                  {arch2.name} Strengths
                </h4>
                <ul className="space-y-1">
                  {arch2.strengths.map((strength, idx) => (
                    <li key={idx} className="text-xs flex items-start gap-2">
                      <Check className="h-3 w-3 text-green-500 mt-0.5 flex-shrink-0" />
                      <span>{strength}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            {/* Weaknesses */}
            <div className="grid md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <h4 className="text-sm font-medium flex items-center gap-2 text-red-600 dark:text-red-400">
                  <X className="h-4 w-4" />
                  {arch1.name} Weaknesses
                </h4>
                <ul className="space-y-1">
                  {arch1.weaknesses.map((weakness, idx) => (
                    <li key={idx} className="text-xs flex items-start gap-2">
                      <X className="h-3 w-3 text-red-500 mt-0.5 flex-shrink-0" />
                      <span>{weakness}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <div className="space-y-2">
                <h4 className="text-sm font-medium flex items-center gap-2 text-red-600 dark:text-red-400">
                  <X className="h-4 w-4" />
                  {arch2.name} Weaknesses
                </h4>
                <ul className="space-y-1">
                  {arch2.weaknesses.map((weakness, idx) => (
                    <li key={idx} className="text-xs flex items-start gap-2">
                      <X className="h-3 w-3 text-red-500 mt-0.5 flex-shrink-0" />
                      <span>{weakness}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            {/* Use Cases */}
            <div className="grid md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <h4 className="text-sm font-medium flex items-center gap-2">
                  <Code className="h-4 w-4" />
                  {arch1.name} Use Cases
                </h4>
                <div className="flex flex-wrap gap-1">
                  {arch1.useCases.map((useCase, idx) => (
                    <span
                      key={idx}
                      className="text-xs px-2 py-1 bg-primary/10 text-primary rounded-full"
                    >
                      {useCase}
                    </span>
                  ))}
                </div>
              </div>

              <div className="space-y-2">
                <h4 className="text-sm font-medium flex items-center gap-2">
                  <Code className="h-4 w-4" />
                  {arch2.name} Use Cases
                </h4>
                <div className="flex flex-wrap gap-1">
                  {arch2.useCases.map((useCase, idx) => (
                    <span
                      key={idx}
                      className="text-xs px-2 py-1 bg-primary/10 text-primary rounded-full"
                    >
                      {useCase}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};
