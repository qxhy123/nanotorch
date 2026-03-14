import { DisclosureLevelSummary, DisclosureSection } from '../components/layout/DisclosureSection';
import { Latex } from '../components/ui/Latex';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { AttentionMatrix } from '../components/visualization/attention/AttentionMatrix';
import { MultiHeadAttention } from '../components/visualization/attention/MultiHeadAttention';
import { HeadAnalysis } from '../components/visualization/attention/HeadAnalysis';
import { QKVDecomposition } from '../components/visualization/attention/QKVDecomposition';
import {
  ScaledDotProductVisualization,
  AttentionPlayground,
} from '../components/visualization/attention';
import { VisualizationNotice } from '../components/layout/VisualizationNotice';
import { useTransformerStore } from '../stores/transformerStore';

function AttentionMathReference() {
  const config = useTransformerStore((state) => state.config);
  const headDim = config.d_model / config.nhead;
  const qkvProjectionParams = 3 * config.d_model * config.d_model;
  const outputProjectionParams = config.d_model * config.d_model;
  const attentionCells = config.nhead * config.max_seq_len * config.max_seq_len;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Attention Math Reference</CardTitle>
        <CardDescription>
          Parameter counts, tensor shapes, and quadratic attention cost for the current setup.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-3 md:grid-cols-4">
          <div className="rounded-lg border bg-muted/40 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Head dim</div>
            <div className="mt-2 text-2xl font-semibold">{headDim}</div>
          </div>
          <div className="rounded-lg border bg-muted/40 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">QKV params</div>
            <div className="mt-2 text-2xl font-semibold">{qkvProjectionParams.toLocaleString()}</div>
          </div>
          <div className="rounded-lg border bg-muted/40 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Output proj params</div>
            <div className="mt-2 text-2xl font-semibold">{outputProjectionParams.toLocaleString()}</div>
          </div>
          <div className="rounded-lg border bg-muted/40 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Score cells</div>
            <div className="mt-2 text-2xl font-semibold">{attentionCells.toLocaleString()}</div>
          </div>
        </div>

        <div className="grid gap-4 lg:grid-cols-2">
          <div className="rounded-lg border bg-background p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">
              Core attention equation
            </div>
            <Latex display>
              {'\\operatorname{Attention}(Q, K, V) = \\operatorname{softmax}\\left(\\frac{QK^{\\top}}{\\sqrt{d_k}}\\right)V'}
            </Latex>
          </div>
          <div className="rounded-lg border bg-background p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">
              Multi-head shape
            </div>
            <Latex display>
              {'Q, K, V \\in \\mathbb{R}^{n_{head} \\times seq \\times d_k}'}
            </Latex>
            <Latex display>
              {'A \\in \\mathbb{R}^{n_{head} \\times seq \\times seq}'}
            </Latex>
          </div>
        </div>

        <div className="rounded-lg border bg-muted/20 p-4 text-sm text-muted-foreground">
          With `d_model = {config.d_model}`, `nhead = {config.nhead}`, and
          `max_seq_len = {config.max_seq_len}`, attention memory scales with
          `O(nhead * seq_len^2)`.
        </div>
      </CardContent>
    </Card>
  );
}

export default function AttentionTab() {
  return (
    <div className="space-y-6">
      <DisclosureLevelSummary title="Attention learning path" />

      <VisualizationNotice title="Deterministic simulated attention internals">
        The live forward pass comes from the backend model, but the Q/K/V projections and
        per-head attention matrices in this tab are deterministic stand-ins until the API
        exposes real layer internals.
      </VisualizationNotice>

      <div className="grid lg:grid-cols-2 gap-6">
        <div id="attention-visualization">
          <AttentionMatrix />
        </div>
        <div id="multi-head-attention">
          <div id="multi-head-explanation">
            <MultiHeadAttention />
          </div>
        </div>
      </div>

      <DisclosureSection
        level="detailed"
        title="Head analysis"
        description="Per-head pattern diagnostics unlock at the Detailed level."
      >
        <HeadAnalysis />
      </DisclosureSection>

      <DisclosureSection
        level="detailed"
        title="QKV decomposition"
        description="Projection-by-projection breakdown unlocks at the Detailed level."
      >
        <QKVDecomposition />
      </DisclosureSection>

      <DisclosureSection
        id="attention-formula"
        level="intermediate"
        title="Scaled dot-product walkthrough"
        description="Step-by-step matrix math unlocks at the Intermediate level."
      >
        <ScaledDotProductVisualization />
      </DisclosureSection>

      <DisclosureSection
        id="mask-visualization"
        level="intermediate"
        title="Attention masking playground"
        description="Interactive masking and score manipulation unlock at the Intermediate level."
      >
        <AttentionPlayground />
      </DisclosureSection>

      <DisclosureSection
        level="math"
        title="Attention math reference"
        description="Full tensor shapes, parameter counts, and scaling laws unlock at the Math level."
      >
        <AttentionMathReference />
      </DisclosureSection>
    </div>
  );
}
