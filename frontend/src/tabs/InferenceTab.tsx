import { DisclosureLevelSummary, DisclosureSection } from '../components/layout/DisclosureSection';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Latex } from '../components/ui/Latex';
import {
  ProbabilityDistribution,
  SamplingStrategyComparison,
  AutoRegressiveGeneration,
  TopKTopPVisualization,
  BeamSearchVisualization,
} from '../components/visualization/inference';
import { VisualizationNotice } from '../components/layout/VisualizationNotice';
import { useTransformerStore } from '../stores/transformerStore';

function InferenceMathReference() {
  const config = useTransformerStore((state) => state.config);
  const candidateTokens = config.vocab_size;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Inference Reference</CardTitle>
        <CardDescription>
          Decoding formulas and candidate-set scale for the current demo configuration.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-3 md:grid-cols-4">
          <div className="rounded-lg border bg-muted/40 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Vocabulary size</div>
            <div className="mt-2 text-2xl font-semibold">{candidateTokens.toLocaleString()}</div>
          </div>
          <div className="rounded-lg border bg-muted/40 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Context window</div>
            <div className="mt-2 text-2xl font-semibold">{config.max_seq_len}</div>
          </div>
          <div className="rounded-lg border bg-muted/40 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Model width</div>
            <div className="mt-2 text-2xl font-semibold">{config.d_model}</div>
          </div>
          <div className="rounded-lg border bg-muted/40 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Per-step search space</div>
            <div className="mt-2 text-2xl font-semibold">{candidateTokens.toLocaleString()}</div>
          </div>
        </div>

        <div className="grid gap-4 lg:grid-cols-2">
          <div className="rounded-lg border bg-background p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Sampling distribution</div>
            <Latex display>{'p(x_t \\mid x_{<t}) = \\operatorname{softmax}(z_t / T)'}</Latex>
          </div>
          <div className="rounded-lg border bg-background p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Beam search objective</div>
            <Latex display>{'\\hat{y} = \\arg\\max_y \\sum_t \\log p(y_t \\mid y_{<t}, x)'}</Latex>
          </div>
        </div>

        <div className="rounded-lg border bg-muted/20 p-4 text-sm text-muted-foreground">
          Every decoding step starts from the full vocabulary and then pruning methods such as top-k,
          top-p, or beam search reduce the candidate set before selecting the next token.
        </div>
      </CardContent>
    </Card>
  );
}

export default function InferenceTab() {
  return (
    <div className="space-y-6">
      <DisclosureLevelSummary title="Decoding learning path" />

      <VisualizationNotice title="Inference tab is a demo sandbox">
        These probability and decoding panels currently use deterministic local demo services and
        do not call a live inference backend.
      </VisualizationNotice>
      <AutoRegressiveGeneration />
      <DisclosureSection
        level="intermediate"
        title="Probability and sampling views"
        description="Distribution-level decoding intuition unlocks at the Intermediate level."
      >
        <div className="grid lg:grid-cols-2 gap-6">
          <ProbabilityDistribution />
          <SamplingStrategyComparison />
        </div>
      </DisclosureSection>
      <DisclosureSection
        level="detailed"
        title="Top-k / top-p controls"
        description="Candidate-pruning controls unlock at the Detailed level."
      >
        <TopKTopPVisualization />
      </DisclosureSection>
      <DisclosureSection
        level="detailed"
        title="Beam search exploration"
        description="Structured multi-path decoding unlocks at the Detailed level."
      >
        <BeamSearchVisualization />
      </DisclosureSection>
      <DisclosureSection
        level="math"
        title="Inference reference"
        description="Decoding equations and search-space accounting unlock at the Math level."
      >
        <InferenceMathReference />
      </DisclosureSection>
    </div>
  );
}
