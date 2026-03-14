import { Network } from 'lucide-react';
import { DisclosureLevelSummary, DisclosureSection } from '../components/layout/DisclosureSection';
import { useTransformerStore } from '../stores/transformerStore';
import {
  TransformerSankey,
  useTransformerSankeyData,
} from '../components/visualization/transformer/TransformerSankey';
import { VisualizationNotice } from '../components/layout/VisualizationNotice';
import { Latex } from '../components/ui/Latex';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';

function SankeyFlowReference() {
  const config = useTransformerStore((state) => state.config);
  const totalBlocks = config.num_encoder_layers + config.num_decoder_layers;
  const attentionBranches = config.nhead * (config.num_encoder_layers + (2 * config.num_decoder_layers));

  return (
    <Card>
      <CardHeader>
        <CardTitle>Data Flow Reference</CardTitle>
        <CardDescription>
          How the Sankey diagram maps stack depth, head fan-out, and residual flow width.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-3 md:grid-cols-4">
          <div className="rounded-lg border bg-muted/40 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Total blocks</div>
            <div className="mt-2 text-2xl font-semibold">{totalBlocks}</div>
          </div>
          <div className="rounded-lg border bg-muted/40 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Head branches</div>
            <div className="mt-2 text-2xl font-semibold">{attentionBranches.toLocaleString()}</div>
          </div>
          <div className="rounded-lg border bg-muted/40 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Residual width</div>
            <div className="mt-2 text-2xl font-semibold">{config.d_model}</div>
          </div>
          <div className="rounded-lg border bg-muted/40 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Context length</div>
            <div className="mt-2 text-2xl font-semibold">{config.max_seq_len}</div>
          </div>
        </div>

        <div className="grid gap-4 lg:grid-cols-2">
          <div className="rounded-lg border bg-background p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Flow abstraction</div>
            <Latex display>{'\\text{tokens} \\rightarrow \\text{embeddings} \\rightarrow \\text{stacked blocks} \\rightarrow \\text{logits}'}</Latex>
          </div>
          <div className="rounded-lg border bg-background p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Attention fan-out</div>
            <Latex display>{'\\text{branches} \\approx n_{head} \\times (L_{enc} + 2L_{dec})'}</Latex>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function SankeyVisualization() {
  const config = useTransformerStore((state) => state.config);
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

export default function SankeyTab() {
  return (
    <div className="space-y-6">
      <DisclosureLevelSummary title="Data flow reading path" />

      <VisualizationNotice title="Architecture flow, not runtime tracing">
        The Sankey diagram is derived from the current Transformer configuration to explain
        data flow structure; it is not a per-request tensor trace.
      </VisualizationNotice>
      <div id="sankey-diagram">
        <SankeyVisualization />
      </div>
      <DisclosureSection
        level="detailed"
        title="How to read the Sankey"
        description="Flow-reading guidance unlocks at the Detailed level."
      >
        <Card>
          <CardHeader>
            <CardTitle>Reading Guide</CardTitle>
            <CardDescription>
              Use width as a proxy for tensor width and branch count as a proxy for parallel heads.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-3 md:grid-cols-3">
            <div className="rounded-lg border bg-muted/30 p-4 text-sm text-muted-foreground">
              Wider links represent larger intermediate representations or fan-out through the stack.
            </div>
            <div className="rounded-lg border bg-muted/30 p-4 text-sm text-muted-foreground">
              Encoder flow stays self-contained, while decoder flow splits into self-attention and cross-attention paths.
            </div>
            <div className="rounded-lg border bg-muted/30 p-4 text-sm text-muted-foreground">
              The diagram explains structure, not per-token runtime activations from a live request.
            </div>
          </CardContent>
        </Card>
      </DisclosureSection>
      <DisclosureSection
        level="math"
        title="Flow accounting reference"
        description="Stack-size and branch-count formulas unlock at the Math level."
      >
        <SankeyFlowReference />
      </DisclosureSection>
    </div>
  );
}
