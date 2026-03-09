import { useState } from 'react';
import { useTransformerStore } from './stores/transformerStore';
import { InputPanel } from './components/controls/InputPanel';
import { ParameterPanel } from './components/controls/ParameterPanel';
import { TokenEmbedding } from './components/visualization/embedding/TokenEmbedding';
import { PositionalEncoding } from './components/visualization/embedding/PositionalEncoding';
import { AttentionMatrix } from './components/visualization/attention/AttentionMatrix';
import { MultiHeadAttention } from './components/visualization/attention/MultiHeadAttention';
import { StagedAttentionVisualization } from './components/visualization/attention/StagedAttentionVisualization';
import { FeedForward } from './components/visualization/feedforward/FeedForward';
import { LayerNormalization } from './components/visualization/normalization/LayerNormalization';
import { TransformerFlow } from './components/visualization/transformer/TransformerFlow';
import { TransformerStructure3D } from './components/visualization/transformer/TransformerStructure3D';
import { TransformerSankey, useTransformerSankeyData } from './components/visualization/transformer/TransformerSankey';
import { FlowDirectionGraph, useQKVFlowData } from './components/visualization/shared/FlowDirectionGraph';
import { DisclosureLevelProvider, DisclosureLevelSelector } from './components/providers/DisclosureLevelProvider';
import { TutorialProvider, TutorialOverlay } from './components/tutorial';
import { allTutorials } from './tutorials';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Button } from './components/ui/button';
import { Brain, Settings, Sliders, Zap, Box, BookOpen, Network } from 'lucide-react';

function AppContent() {
  const [activeTab, setActiveTab] = useState('overview');
  const [showTutorial, setShowTutorial] = useState(false);
  const config = useTransformerStore((state) => state.config);

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
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
                onClick={() => setShowTutorial(true)}
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

      {/* Main Content */}
      <main className="container mx-auto px-4 py-6">
        {/* Disclosure Level Selector */}
        <div className="mb-4 flex justify-end">
          <DisclosureLevelSelector showDescriptions={false} />
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-7 lg:w-auto lg:inline-grid">
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
            <TabsTrigger value="staged" className="gap-2">
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
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6 mt-6">
            <div className="grid lg:grid-cols-3 gap-6">
              {/* Input Panel */}
              <div className="lg:col-span-1">
                <InputPanel />
              </div>

              {/* Transformer Flow */}
              <div className="lg:col-span-2">
                <TransformerFlow />
              </div>
            </div>

            {/* Quick Stats */}
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
          </TabsContent>

          {/* Structure Tab */}
          <TabsContent value="structure" className="space-y-6 mt-6">
            <TransformerStructure3D />
          </TabsContent>

          {/* Embeddings Tab */}
          <TabsContent value="embeddings" className="space-y-6 mt-6">
            <div className="grid lg:grid-cols-2 gap-6">
              <TokenEmbedding />
              <PositionalEncoding />
            </div>
          </TabsContent>

          {/* Attention Tab */}
          <TabsContent value="attention" className="space-y-6 mt-6">
            <div className="grid lg:grid-cols-2 gap-6">
              <AttentionMatrix />
              <MultiHeadAttention />
            </div>
          </TabsContent>

          {/* Layers Tab */}
          <TabsContent value="layers" className="space-y-6 mt-6">
            <div className="grid lg:grid-cols-2 gap-6">
              <FeedForward />
              <LayerNormalization />
            </div>
          </TabsContent>

          {/* Flow Tab */}
          <TabsContent value="flow" className="space-y-6 mt-6">
            <TransformerFlow />
          </TabsContent>

          {/* Staged Attention Tab */}
          <TabsContent value="staged" className="space-y-6 mt-6">
            <StagedAttentionTabContent />
          </TabsContent>

          {/* Architecture Flow Tab */}
          <TabsContent value="sankey" className="space-y-6 mt-6">
            <SankeyVisualization config={config} />
          </TabsContent>
        </Tabs>

        {/* Settings Panel (Fixed on right for larger screens) */}
        <div className="fixed right-4 top-24 bottom-4 w-80 overflow-y-auto hidden lg:block">
          <ParameterPanel />
        </div>
      </main>

      {/* Tutorial Overlay */}
      {showTutorial && (
        <TutorialProvider tutorials={allTutorials}>
          <TutorialOverlay />
        </TutorialProvider>
      )}

      {/* Footer */}
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

// Staged Attention Tab Content Component
function StagedAttentionTabContent() {
  const attentionWeights = useTransformerStore((state) => state.attentionWeights);
  const tokens = useTransformerStore((state) => state.tokens);
  const inputText = useTransformerStore((state) => state.inputText);

  if (!attentionWeights || attentionWeights.length === 0) {
    return (
      <div className="text-center text-gray-500 py-12">
        <Brain className="h-16 w-16 mx-auto mb-4 opacity-50" />
        <p className="text-lg font-medium">No attention data available</p>
        <p className="text-sm">Enter some text and run the model to see the staged attention visualization.</p>
      </div>
    );
  }

  const tokenList = tokens.length > 0
    ? tokens.map(t => `Token_${t}`)
    : inputText.split(' ').map((t, i) => t || `<${i}>`);

  // Generate flow graph data for QKV visualization
  const { nodes: flowNodes, links: flowLinks } = useQKVFlowData(attentionWeights[0], tokenList);

  return (
    <div className="space-y-6">
      {/* QKV Flow Direction Graph */}
      {flowNodes.length > 0 && (
        <FlowDirectionGraph
          nodes={flowNodes}
          links={flowLinks}
          title="QKV Attention Flow"
          description="Visualizes the flow of Query, Key, and Value through the attention mechanism"
        />
      )}

      {/* Staged Attention Visualization */}
      <StagedAttentionVisualization
        attentionData={attentionWeights[0]}
        tokens={tokenList}
      />
    </div>
  );
}

// Sankey Visualization Component
function SankeyVisualization({ config }: { config: any }) {
  const sankeyData = useTransformerSankeyData(null, config);

  if (!sankeyData || !sankeyData.nodes || sankeyData.nodes.length === 0) {
    return (
      <div className="text-center text-gray-500 py-12">
        <Network className="h-16 w-16 mx-auto mb-4 opacity-50" />
        <p className="text-lg font-medium">No Sankey data available</p>
        <p className="text-sm">Configure the Transformer to see the data flow visualization.</p>
      </div>
    );
  }

  return <TransformerSankey data={sankeyData} />;
}

// Wrap app with providers
function App() {
  return (
    <TutorialProvider tutorials={allTutorials}>
      <DisclosureLevelProvider defaultLevel="intermediate">
        <AppContent />
      </DisclosureLevelProvider>
    </TutorialProvider>
  );
}

export default App;
