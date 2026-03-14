import { Latex } from '../components/ui/Latex';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { DisclosureLevelSummary, DisclosureSection } from '../components/layout/DisclosureSection';
import { TokenEmbedding } from '../components/visualization/embedding/TokenEmbedding';
import { PositionalEncoding } from '../components/visualization/embedding/PositionalEncoding';
import { PositionEncodingExplorer } from '../components/visualization/embedding/PositionEncodingExplorer';
import { EmbeddingArithmetic } from '../components/visualization/embedding/EmbeddingArithmetic';
import { useTransformerStore } from '../stores/transformerStore';

function EmbeddingMathReference() {
  const config = useTransformerStore((state) => state.config);
  const embeddingParams = config.vocab_size * config.d_model;
  const positionalSlots = config.max_seq_len * config.d_model;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Embedding Math Reference</CardTitle>
        <CardDescription>
          Full lookup, shape, and positional encoding details for the current demo config.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-3 md:grid-cols-3">
          <div className="rounded-lg border bg-muted/40 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">
              Embedding table params
            </div>
            <div className="mt-2 text-2xl font-semibold">{embeddingParams.toLocaleString()}</div>
          </div>
          <div className="rounded-lg border bg-muted/40 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">
              Positional slots
            </div>
            <div className="mt-2 text-2xl font-semibold">{positionalSlots.toLocaleString()}</div>
          </div>
          <div className="rounded-lg border bg-muted/40 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">
              Context window
            </div>
            <div className="mt-2 text-2xl font-semibold">{config.max_seq_len}</div>
          </div>
        </div>

        <div className="grid gap-4 lg:grid-cols-2">
          <div className="rounded-lg border bg-background p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">
              Token lookup
            </div>
            <Latex display>{'x_i = E[t_i],\\quad E \\in \\mathbb{R}^{|V| \\times d_{model}}'}</Latex>
          </div>
          <div className="rounded-lg border bg-background p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">
              Sinusoidal position
            </div>
            <Latex display>
              {'PE(pos, 2i) = \\sin\\left(pos / 10000^{2i / d_{model}}\\right)'}
            </Latex>
            <Latex display>
              {'PE(pos, 2i + 1) = \\cos\\left(pos / 10000^{2i / d_{model}}\\right)'}
            </Latex>
          </div>
        </div>

        <div className="rounded-lg border bg-muted/20 p-4 text-sm text-muted-foreground">
          Current demo tensor shapes are driven by `vocab_size = {config.vocab_size}`,
          `d_model = {config.d_model}`, and `max_seq_len = {config.max_seq_len}`.
        </div>
      </CardContent>
    </Card>
  );
}

export default function EmbeddingsTab() {
  return (
    <div className="space-y-6">
      <DisclosureLevelSummary title="Embeddings learning path" />

      <div className="grid lg:grid-cols-2 gap-6">
        <div id="embedding-visualization" className="token-heatmap">
          <TokenEmbedding />
        </div>
        <div id="positional-encoding-visualization" className="positional-heatmap">
          <PositionalEncoding />
        </div>
      </div>

      <DisclosureSection
        id="positional-math"
        level="intermediate"
        title="Positional encoding explorer"
        description="Waveforms, heatmaps, and encoding family comparisons unlock at the Intermediate level."
      >
        <PositionEncodingExplorer />
      </DisclosureSection>

      <DisclosureSection
        id="embedding-math"
        level="detailed"
        title="Embedding arithmetic"
        description="Semantic vector arithmetic and analogy search unlock at the Detailed level."
      >
        <EmbeddingArithmetic />
      </DisclosureSection>

      <DisclosureSection
        id="embedding-reference"
        level="math"
        title="Embedding math reference"
        description="Full formulas, parameter counts, and tensor shapes unlock at the Math level."
      >
        <EmbeddingMathReference />
      </DisclosureSection>
    </div>
  );
}
