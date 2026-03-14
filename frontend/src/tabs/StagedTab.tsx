import { Brain } from 'lucide-react';
import { DisclosureLevelSummary, DisclosureSection } from '../components/layout/DisclosureSection';
import { useTransformerStore } from '../stores/transformerStore';
import { StagedAttentionVisualization } from '../components/visualization/attention/StagedAttentionVisualization';
import {
  FlowDirectionGraph,
  useQKVFlowData,
} from '../components/visualization/shared/FlowDirectionGraph';
import { VisualizationNotice } from '../components/layout/VisualizationNotice';
import { Latex } from '../components/ui/Latex';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';

function StagedAttentionReference() {
  const config = useTransformerStore((state) => state.config);
  const tokens = useTransformerStore((state) => state.tokens);
  const inputText = useTransformerStore((state) => state.inputText);
  const selectedHead = useTransformerStore((state) => state.visualizationState.selectedHead);

  const tokenCount = tokens.length > 0 ? tokens.length : inputText.split(' ').filter(Boolean).length;
  const attentionCells = tokenCount > 0 ? tokenCount * tokenCount : 0;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Staged Attention Reference</CardTitle>
        <CardDescription>
          Current head selection, token count, and the equations behind the staged walkthrough.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-3 md:grid-cols-4">
          <div className="rounded-lg border bg-muted/40 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Selected head</div>
            <div className="mt-2 text-2xl font-semibold">{selectedHead + 1}</div>
          </div>
          <div className="rounded-lg border bg-muted/40 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Token count</div>
            <div className="mt-2 text-2xl font-semibold">{tokenCount}</div>
          </div>
          <div className="rounded-lg border bg-muted/40 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Attention cells</div>
            <div className="mt-2 text-2xl font-semibold">{attentionCells.toLocaleString()}</div>
          </div>
          <div className="rounded-lg border bg-muted/40 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Head width</div>
            <div className="mt-2 text-2xl font-semibold">{config.d_model / config.nhead}</div>
          </div>
        </div>

        <div className="grid gap-4 lg:grid-cols-2">
          <div className="rounded-lg border bg-background p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Score computation</div>
            <Latex display>{'S = \\frac{QK^{\\top}}{\\sqrt{d_k}}'}</Latex>
          </div>
          <div className="rounded-lg border bg-background p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Weighted aggregation</div>
            <Latex display>{'\\operatorname{head}_h = \\operatorname{softmax}(S_h)V_h'}</Latex>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function StagedAttentionTabContent() {
  const attentionWeights = useTransformerStore((state) => state.attentionWeights);
  const tokens = useTransformerStore((state) => state.tokens);
  const inputText = useTransformerStore((state) => state.inputText);

  const primaryAttention = attentionWeights?.[0] ?? null;
  const tokenList = tokens.length > 0
    ? tokens.map((tokenId) => `Token_${tokenId}`)
    : inputText
        .split(' ')
        .map((token, index) => token || `<${index}>`);
  const { nodes: flowNodes, links: flowLinks } = useQKVFlowData(primaryAttention, tokenList);

  if (!primaryAttention) {
    return (
      <div className="text-center text-gray-500 py-12">
        <Brain className="h-16 w-16 mx-auto mb-4 opacity-50" />
        <p className="text-lg font-medium">No attention data available</p>
        <p className="text-sm">
          Enter some text and run the model to see the staged attention visualization.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <DisclosureSection
        id="staged-attention-view"
        level="intermediate"
        title="Staged attention walkthrough"
        description="The step-by-step replay of the attention pipeline unlocks at the Intermediate level."
      >
        <StagedAttentionVisualization
          attentionData={primaryAttention}
          tokens={tokenList}
        />
      </DisclosureSection>

      {flowNodes.length > 0 && (
        <DisclosureSection
          level="detailed"
          title="QKV flow graph"
          description="The explicit node-link graph of query, key, and value flow unlocks at the Detailed level."
        >
          <FlowDirectionGraph
            nodes={flowNodes}
            links={flowLinks}
            title="QKV Attention Flow"
            description="Visualizes the flow of query, key, and value through the attention mechanism."
          />
        </DisclosureSection>
      )}
    </div>
  );
}

export default function StagedTab() {
  return (
    <div className="space-y-6">
      <DisclosureLevelSummary title="Step-by-step attention path" />

      <VisualizationNotice title="Step-by-step view uses demo projections">
        This walkthrough replays deterministic simulated Q/K/V data so the staged attention
        explanation stays stable from run to run.
      </VisualizationNotice>
      <StagedAttentionTabContent />
      <DisclosureSection
        level="math"
        title="Staged attention reference"
        description="Per-head equations and current tensor sizes unlock at the Math level."
      >
        <StagedAttentionReference />
      </DisclosureSection>
    </div>
  );
}
