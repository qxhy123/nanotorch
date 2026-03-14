import { DisclosureLevelSummary, DisclosureSection } from '../components/layout/DisclosureSection';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Latex } from '../components/ui/Latex';
import {
  GradientFlow,
  TrainingDashboard,
  LossCurve,
  WeightDistribution,
  ModelProfiler,
  TensorShapeTracker,
} from '../components/visualization/training';
import { VisualizationNotice } from '../components/layout/VisualizationNotice';
import { useTransformerStore } from '../stores/transformerStore';

type LossPoint = {
  epoch: number;
  value: number;
};

function buildDeterministicLossSeries(
  base: number,
  decay: number,
  floor: number,
  wobble: number,
  phase: number
): LossPoint[] {
  return Array.from({ length: 50 }, (_, index) => {
    const epoch = index + 1;
    const trend = base * Math.exp(-index * decay) + floor;
    const seasonal =
      Math.sin(index * 0.55 + phase) * wobble +
      Math.cos(index * 0.18 + phase) * wobble * 0.4;

    return {
      epoch,
      value: Number((trend + seasonal).toFixed(4)),
    };
  });
}

const TRAIN_LOSS = buildDeterministicLossSeries(2.5, 0.05, 0.2, 0.05, 0.3);
const VALIDATION_LOSS = buildDeterministicLossSeries(2.5, 0.05, 0.35, 0.07, 1.1);

function TrainingMathReference() {
  const config = useTransformerStore((state) => state.config);
  const totalBlocks = config.num_encoder_layers + config.num_decoder_layers;
  const expansionRatio = (config.dim_feedforward / config.d_model).toFixed(1);
  const residualWidth = config.max_seq_len * config.d_model;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Training Reference</CardTitle>
        <CardDescription>
          Optimization equations and model-shape facts that drive the training visualizations.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-3 md:grid-cols-4">
          <div className="rounded-lg border bg-muted/40 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Total blocks</div>
            <div className="mt-2 text-2xl font-semibold">{totalBlocks}</div>
          </div>
          <div className="rounded-lg border bg-muted/40 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Residual width</div>
            <div className="mt-2 text-2xl font-semibold">{residualWidth.toLocaleString()}</div>
          </div>
          <div className="rounded-lg border bg-muted/40 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">FFN expansion</div>
            <div className="mt-2 text-2xl font-semibold">{expansionRatio}x</div>
          </div>
          <div className="rounded-lg border bg-muted/40 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Layer norm eps</div>
            <div className="mt-2 text-2xl font-semibold">{config.layer_norm_eps}</div>
          </div>
        </div>

        <div className="grid gap-4 lg:grid-cols-2">
          <div className="rounded-lg border bg-background p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Gradient descent</div>
            <Latex display>{'\\theta_{t+1} = \\theta_t - \\eta \\nabla_\\theta \\mathcal{L}(\\theta_t)'}</Latex>
          </div>
          <div className="rounded-lg border bg-background p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Cross-entropy</div>
            <Latex display>{'\\mathcal{L} = -\\sum_i y_i \\log \\hat{y}_i'}</Latex>
          </div>
        </div>

        <div className="rounded-lg border bg-muted/20 p-4 text-sm text-muted-foreground">
          Tensor-shape tracking is especially useful because each block preserves the residual stream
          width while activations temporarily expand to `dim_feedforward = {config.dim_feedforward}`.
        </div>
      </CardContent>
    </Card>
  );
}

export default function TrainingTab() {
  return (
    <div className="space-y-6">
      <DisclosureLevelSummary title="Training diagnostics path" />

      <VisualizationNotice title="Training dashboards are illustrative">
        The charts in this tab are educational visualizations and deterministic sample series,
        not metrics streamed from a live training loop.
      </VisualizationNotice>
      <TrainingDashboard />
      <DisclosureSection
        level="intermediate"
        title="Loss curves"
        description="Training and validation loss trends unlock at the Intermediate level."
      >
        <LossCurve
          trainLoss={TRAIN_LOSS}
          validationLoss={VALIDATION_LOSS}
        />
      </DisclosureSection>
      <DisclosureSection
        level="detailed"
        title="Gradient diagnostics"
        description="Gradient flow and weight distribution views unlock at the Detailed level."
      >
        <div className="space-y-6">
          <GradientFlow />
          <WeightDistribution />
        </div>
      </DisclosureSection>
      <DisclosureSection
        level="math"
        title="Profiler and tensor tracker"
        description="Shape-level and cost-level diagnostics unlock at the Math level."
      >
        <div className="space-y-6">
          <ModelProfiler />
          <TensorShapeTracker />
        </div>
      </DisclosureSection>
      <DisclosureSection
        level="math"
        title="Training reference"
        description="Optimization equations and model-shape facts unlock at the Math level."
      >
        <TrainingMathReference />
      </DisclosureSection>
    </div>
  );
}
