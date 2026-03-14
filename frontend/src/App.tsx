import { Suspense, lazy, useEffect, useRef, useState } from 'react';
import { useTransformerStore } from './stores/transformerStore';
import { InputPanel } from './components/controls/InputPanel';
import { ParameterPanel } from './components/controls/ParameterPanel';
import { TransformerFlow } from './components/visualization/transformer/TransformerFlow';
import {
  TutorialProvider,
  TutorialOverlay,
  useTutorial,
} from './components/tutorial';
import {
  DisclosureLevelProvider,
  DisclosureLevelSelector,
  useDisclosureLevel,
} from './components/providers';
import { DisclosureLevelSummary, DisclosureSection } from './components/layout/DisclosureSection';
import { allTutorials } from './tutorials';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card';
import { Latex } from './components/ui/Latex';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Button } from './components/ui/button';
import type { DisclosureLevel, TransformerConfig } from './types/transformer';
import {
  Brain,
  Settings,
  Sliders,
  Zap,
  Box,
  BookOpen,
  Network,
  Hash,
  TrendingUp,
  Activity,
  X,
} from 'lucide-react';

const StructureTab = lazy(() => import('./tabs/StructureTab'));
const EmbeddingsTab = lazy(() => import('./tabs/EmbeddingsTab'));
const AttentionTab = lazy(() => import('./tabs/AttentionTab'));
const StagedTab = lazy(() => import('./tabs/StagedTab'));
const LayersTab = lazy(() => import('./tabs/LayersTab'));
const SankeyTab = lazy(() => import('./tabs/SankeyTab'));
const TokenizationTab = lazy(() => import('./tabs/TokenizationTab'));
const InferenceTab = lazy(() => import('./tabs/InferenceTab'));
const TrainingTab = lazy(() => import('./tabs/TrainingTab'));

const TUTORIAL_TARGET_TO_TAB: Record<string, string> = {
  '#input-panel': 'overview',
  '#embedding-visualization': 'embeddings',
  '#embedding-math': 'embeddings',
  '#positional-encoding-visualization': 'embeddings',
  '#positional-math': 'embeddings',
  '#attention-visualization': 'attention',
  '#multi-head-attention': 'attention',
  '#attention-formula': 'attention',
  '#multi-head-explanation': 'attention',
  '#staged-attention-view': 'staged',
  '#mask-visualization': 'attention',
  '#feed-forward-visualization': 'layers',
  '#cross-attention-viz': 'layers',
  '#layer-stack-visualization': 'layers',
  '#layer-norm-math': 'layers',
  '#embedding-reference': 'embeddings',
  '#layer-reference': 'layers',
  '#sankey-diagram': 'sankey',
};

function TabFallback() {
  return (
    <div className="rounded-lg border bg-card p-8 text-center text-sm text-muted-foreground">
      Loading visualization...
    </div>
  );
}

function OverviewRoadmapCard() {
  return (
    <Card id="overview-roadmap">
      <CardHeader>
        <CardTitle>What to explore next</CardTitle>
        <CardDescription>
          Use the tabs as a guided path from high-level structure to math-heavy internals.
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <div className="rounded-lg border bg-muted/30 p-4">
          <div className="text-sm font-medium">Structure + Sankey</div>
          <p className="mt-2 text-sm text-muted-foreground">
            Start with the full architecture layout and flow map to orient yourself.
          </p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-4">
          <div className="text-sm font-medium">Embeddings + Attention</div>
          <p className="mt-2 text-sm text-muted-foreground">
            Then see how token vectors are built and how relationships are computed.
          </p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-4">
          <div className="text-sm font-medium">Layers + Inference</div>
          <p className="mt-2 text-sm text-muted-foreground">
            Move into stacked blocks and decoding strategies once the core flow feels familiar.
          </p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-4">
          <div className="text-sm font-medium">Training + Tokenization</div>
          <p className="mt-2 text-sm text-muted-foreground">
            Finish with optimization diagnostics and the front-end tokenizer pipeline.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

function OverviewDiagnosticsCard({ config }: { config: TransformerConfig }) {
  const headDim = config.d_model / config.nhead;
  const totalBlocks = config.num_encoder_layers + config.num_decoder_layers;
  const embeddingParams = config.vocab_size * config.d_model;
  const residualWidth = config.max_seq_len * config.d_model;

  return (
    <Card id="overview-diagnostics">
      <CardHeader>
        <CardTitle>Configuration Diagnostics</CardTitle>
        <CardDescription>
          A quick read of the main scale knobs behind the current live demo.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-3 md:grid-cols-4">
          <div className="rounded-lg border bg-muted/30 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Head dim</div>
            <div className="mt-2 text-2xl font-semibold">{headDim}</div>
          </div>
          <div className="rounded-lg border bg-muted/30 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Total blocks</div>
            <div className="mt-2 text-2xl font-semibold">{totalBlocks}</div>
          </div>
          <div className="rounded-lg border bg-muted/30 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Embedding params</div>
            <div className="mt-2 text-2xl font-semibold">{embeddingParams.toLocaleString()}</div>
          </div>
          <div className="rounded-lg border bg-muted/30 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Residual width</div>
            <div className="mt-2 text-2xl font-semibold">{residualWidth.toLocaleString()}</div>
          </div>
        </div>
        <div className="rounded-lg border bg-background p-4 text-sm text-muted-foreground">
          With <code>d_model = {config.d_model}</code> and <code>nhead = {config.nhead}</code>,
          each head sees <code>d_k = {headDim}</code> features. The model currently uses{' '}
          {config.num_encoder_layers} encoder layers and {config.num_decoder_layers} decoder layers.
        </div>
      </CardContent>
    </Card>
  );
}

function OverviewMathReference({ config }: { config: TransformerConfig }) {
  const attentionProjectionParams = 4 * config.d_model * config.d_model;
  const ffnParams = (2 * config.d_model * config.dim_feedforward)
    + config.dim_feedforward
    + config.d_model;

  return (
    <Card id="overview-reference">
      <CardHeader>
        <CardTitle>Model Summary Reference</CardTitle>
        <CardDescription>
          Core formulas and approximate block-level parameter counts for the active config.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          <div className="rounded-lg border bg-muted/30 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Attention proj/block</div>
            <div className="mt-2 text-2xl font-semibold">{attentionProjectionParams.toLocaleString()}</div>
          </div>
          <div className="rounded-lg border bg-muted/30 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">FFN params/block</div>
            <div className="mt-2 text-2xl font-semibold">{ffnParams.toLocaleString()}</div>
          </div>
          <div className="rounded-lg border bg-muted/30 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Context window</div>
            <div className="mt-2 text-2xl font-semibold">{config.max_seq_len}</div>
          </div>
          <div className="rounded-lg border bg-muted/30 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Vocabulary size</div>
            <div className="mt-2 text-2xl font-semibold">{config.vocab_size.toLocaleString()}</div>
          </div>
        </div>

        <div className="grid gap-4 lg:grid-cols-2">
          <div className="rounded-lg border bg-background p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Attention</div>
            <Latex display>
              {'\\operatorname{Attention}(Q, K, V) = \\operatorname{softmax}\\left(\\frac{QK^{\\top}}{\\sqrt{d_k}}\\right)V'}
            </Latex>
          </div>
          <div className="rounded-lg border bg-background p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Feed-forward</div>
            <Latex display>{'\\operatorname{FFN}(x) = W_2\\,\\phi(W_1x + b_1) + b_2'}</Latex>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function AppContent() {
  const [activeTab, setActiveTab] = useState('overview');
  const [showMobileConfig, setShowMobileConfig] = useState(false);
  const config = useTransformerStore((state) => state.config);
  const { startTutorial, currentStepData, state: tutorialState } = useTutorial();
  const { level, setLevel } = useDisclosureLevel();
  const previousTutorialDisclosureRef = useRef<DisclosureLevel | null>(null);
  const tutorialTab = currentStepData?.target
    ? TUTORIAL_TARGET_TO_TAB[currentStepData.target]
    : undefined;
  const visibleTab = tutorialTab || activeTab;

  useEffect(() => {
    if (tutorialState.isTutorialActive) {
      if (previousTutorialDisclosureRef.current === null) {
        previousTutorialDisclosureRef.current = level;
      }
      return;
    }

    if (previousTutorialDisclosureRef.current !== null) {
      const previousLevel = previousTutorialDisclosureRef.current;
      previousTutorialDisclosureRef.current = null;
      if (level !== previousLevel) {
        setLevel(previousLevel);
      }
    }
  }, [tutorialState.isTutorialActive, level, setLevel]);

  useEffect(() => {
    const requiredLevel = currentStepData?.requiredDisclosureLevel;
    if (!tutorialState.isTutorialActive || !requiredLevel || level === requiredLevel) {
      return;
    }

    setLevel(requiredLevel);
  }, [
    tutorialState.isTutorialActive,
    currentStepData?.requiredDisclosureLevel,
    level,
    setLevel,
  ]);

  const handleStartTutorial = () => {
    const tutorialId = allTutorials[0]?.id;
    if (tutorialId) {
      startTutorial(tutorialId);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Brain className="h-8 w-8 text-primary" />
              <div>
                <h1 className="text-2xl font-bold">Transformer Visualization</h1>
                <p className="text-sm text-muted-foreground">
                  Interactive exploration of Transformer architecture
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowMobileConfig(true)}
                className="gap-2 lg:hidden"
              >
                <Settings className="h-4 w-4" />
                Config
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={handleStartTutorial}
                className="gap-2"
              >
                <BookOpen className="h-4 w-4" />
                Tutorial
              </Button>
              <a
                href="https://github.com/qxhy123/nanotorch"
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-muted-foreground hover:text-foreground"
              >
                nanotorch
              </a>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-6">
        <Tabs value={visibleTab} onValueChange={setActiveTab}>
          <div className="space-y-4">
            <TabsList className="grid w-full grid-cols-10 lg:w-auto lg:inline-grid">
              <TabsTrigger value="overview" className="gap-2">
                <Sliders className="h-4 w-4" />
                Overview
              </TabsTrigger>
              <TabsTrigger value="structure" className="gap-2">
                <Box className="h-4 w-4" />
                Structure
              </TabsTrigger>
              <TabsTrigger value="embeddings" className="gap-2">
                <Zap className="h-4 w-4" />
                Embeddings
              </TabsTrigger>
              <TabsTrigger value="attention" className="gap-2">
                <Brain className="h-4 w-4" />
                Attention
              </TabsTrigger>
              <TabsTrigger value="staged" className="gap-2 view-mode-staged">
                <Settings className="h-4 w-4" />
                Staged View
              </TabsTrigger>
              <TabsTrigger value="layers" className="gap-2">
                <Settings className="h-4 w-4" />
                Layers
              </TabsTrigger>
              <TabsTrigger value="sankey" className="gap-2">
                <Network className="h-4 w-4" />
                Data Flow
              </TabsTrigger>
              <TabsTrigger value="tokenization" className="gap-2">
                <Hash className="h-4 w-4" />
                Tokenization
              </TabsTrigger>
              <TabsTrigger value="inference" className="gap-2">
                <TrendingUp className="h-4 w-4" />
                Inference
              </TabsTrigger>
              <TabsTrigger value="training" className="gap-2">
                <Activity className="h-4 w-4" />
                Training
              </TabsTrigger>
            </TabsList>

            <div
              id="disclosure-level-selector"
              className="rounded-lg border bg-card px-4 py-3"
            >
              <div className="mb-2 text-sm font-medium">Disclosure Level</div>
              <DisclosureLevelSelector
                className="flex flex-wrap gap-2"
                showDescriptions
              />
            </div>
          </div>

          <TabsContent value="overview" className="space-y-6 mt-6">
            <DisclosureLevelSummary title="Overview learning path" />

            <div className="grid lg:grid-cols-3 gap-6">
              <div id="input-panel" className="lg:col-span-1">
                <InputPanel />
              </div>

              <div className="lg:col-span-2">
                <TransformerFlow />
              </div>
            </div>

            <div className="grid md:grid-cols-4 gap-4">
              <div className="p-4 bg-muted rounded-lg">
                <div className="text-sm text-muted-foreground">Model Dimension</div>
                <div className="text-2xl font-bold">{config.d_model}</div>
              </div>
              <div className="p-4 bg-muted rounded-lg">
                <div className="text-sm text-muted-foreground">Attention Heads</div>
                <div className="text-2xl font-bold">{config.nhead}</div>
              </div>
              <div className="p-4 bg-muted rounded-lg">
                <div className="text-sm text-muted-foreground">Encoder Layers</div>
                <div className="text-2xl font-bold">{config.num_encoder_layers}</div>
              </div>
              <div className="p-4 bg-muted rounded-lg">
                <div className="text-sm text-muted-foreground">FFN Dimension</div>
                <div className="text-2xl font-bold">{config.dim_feedforward}</div>
              </div>
            </div>

            <DisclosureSection
              level="intermediate"
              title="How to navigate the demo"
              description="A guided reading path across the visualization tabs unlocks at the Intermediate level."
            >
              <OverviewRoadmapCard />
            </DisclosureSection>

            <DisclosureSection
              level="detailed"
              title="Configuration diagnostics"
              description="Derived scale metrics unlock at the Detailed level."
            >
              <OverviewDiagnosticsCard config={config} />
            </DisclosureSection>

            <DisclosureSection
              level="math"
              title="Model summary reference"
              description="Core equations and per-block parameter approximations unlock at the Math level."
            >
              <OverviewMathReference config={config} />
            </DisclosureSection>
          </TabsContent>

          <TabsContent value="structure" className="mt-6">
            <Suspense fallback={<TabFallback />}>
              <StructureTab />
            </Suspense>
          </TabsContent>

          <TabsContent value="embeddings" className="mt-6">
            <Suspense fallback={<TabFallback />}>
              <EmbeddingsTab />
            </Suspense>
          </TabsContent>

          <TabsContent value="attention" className="mt-6">
            <Suspense fallback={<TabFallback />}>
              <AttentionTab />
            </Suspense>
          </TabsContent>

          <TabsContent value="staged" className="mt-6">
            <Suspense fallback={<TabFallback />}>
              <StagedTab />
            </Suspense>
          </TabsContent>

          <TabsContent value="layers" className="mt-6">
            <Suspense fallback={<TabFallback />}>
              <LayersTab />
            </Suspense>
          </TabsContent>

          <TabsContent value="sankey" className="mt-6">
            <Suspense fallback={<TabFallback />}>
              <SankeyTab />
            </Suspense>
          </TabsContent>

          <TabsContent value="tokenization" className="mt-6">
            <Suspense fallback={<TabFallback />}>
              <TokenizationTab />
            </Suspense>
          </TabsContent>

          <TabsContent value="inference" className="mt-6">
            <Suspense fallback={<TabFallback />}>
              <InferenceTab />
            </Suspense>
          </TabsContent>

          <TabsContent value="training" className="mt-6">
            <Suspense fallback={<TabFallback />}>
              <TrainingTab />
            </Suspense>
          </TabsContent>
        </Tabs>

        <div className="fixed right-4 top-24 bottom-4 w-80 overflow-y-auto hidden lg:block">
          <ParameterPanel />
        </div>

        {showMobileConfig && (
          <div className="fixed inset-0 z-50 bg-black/50 lg:hidden">
            <div className="absolute inset-x-4 top-4 bottom-4 flex flex-col rounded-xl bg-background shadow-2xl">
              <div className="flex items-center justify-between border-b px-4 py-3">
                <div>
                  <p className="font-medium">Model Configuration</p>
                  <p className="text-sm text-muted-foreground">
                    Adjust the live demo model on mobile.
                  </p>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setShowMobileConfig(false)}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
              <div className="flex-1 overflow-y-auto p-4">
                <ParameterPanel />
              </div>
            </div>
          </div>
        )}
      </main>

      <footer className="border-t mt-12">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between text-sm text-muted-foreground">
            <p>Built with nanotorch, React, and TypeScript</p>
            <p>© 2026 Transformer Visualization</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

function App() {
  return (
    <DisclosureLevelProvider>
      <TutorialProvider tutorials={allTutorials}>
        <AppContent />
        <TutorialOverlay />
      </TutorialProvider>
    </DisclosureLevelProvider>
  );
}

export default App;
