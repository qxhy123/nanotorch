import { DisclosureLevelSummary, DisclosureSection } from '../components/layout/DisclosureSection';
import { Latex } from '../components/ui/Latex';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { useTransformerStore } from '../stores/transformerStore';
import { FeedForward, FFNActivationMap } from '../components/visualization/feedforward';
import { LayerNormalization } from '../components/visualization/normalization/LayerNormalization';
import { LayerVisualization } from '../components/visualization/layer/LayerVisualization';
import { DropoutMaskVisualization } from '../components/visualization/layers/DropoutMaskVisualization';
import { ResidualConnection } from '../components/visualization/layers/ResidualConnection';
import { VisualizationNotice } from '../components/layout/VisualizationNotice';

function LayersMathReference() {
  const config = useTransformerStore((state) => state.config);
  const ffnParams = (2 * config.d_model * config.dim_feedforward)
    + config.dim_feedforward
    + config.d_model;
  const layerNormParams = 2 * config.d_model;
  const residualWidth = config.max_seq_len * config.d_model;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Layer Math Reference</CardTitle>
        <CardDescription>
          Feed-forward width, LayerNorm parameters, and residual tensor sizes for this model.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-3 md:grid-cols-4">
          <div className="rounded-lg border bg-muted/40 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">FFN params</div>
            <div className="mt-2 text-2xl font-semibold">{ffnParams.toLocaleString()}</div>
          </div>
          <div className="rounded-lg border bg-muted/40 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">LayerNorm params</div>
            <div className="mt-2 text-2xl font-semibold">{layerNormParams.toLocaleString()}</div>
          </div>
          <div className="rounded-lg border bg-muted/40 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Residual width</div>
            <div className="mt-2 text-2xl font-semibold">{residualWidth.toLocaleString()}</div>
          </div>
          <div className="rounded-lg border bg-muted/40 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Stack depth</div>
            <div className="mt-2 text-2xl font-semibold">
              {config.num_encoder_layers + config.num_decoder_layers}
            </div>
          </div>
        </div>

        <div className="grid gap-4 lg:grid-cols-2">
          <div className="rounded-lg border bg-background p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">
              LayerNorm
            </div>
            <Latex display>
              {'\\operatorname{LayerNorm}(x) = \\gamma \\odot \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\varepsilon}} + \\beta'}
            </Latex>
          </div>
          <div className="rounded-lg border bg-background p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">
              Feed-forward block
            </div>
            <Latex display>{'\\operatorname{FFN}(x) = W_2\\,\\phi(W_1 x + b_1) + b_2'}</Latex>
          </div>
        </div>

        <div className="rounded-lg border bg-muted/20 p-4 text-sm text-muted-foreground">
          Each transformer block keeps the residual stream shape at
          `[seq_len, d_model] = [{config.max_seq_len}, {config.d_model}]`, while the FFN widens to
          `dim_feedforward = {config.dim_feedforward}` before projecting back down.
        </div>
      </CardContent>
    </Card>
  );
}

export default function LayersTab() {
  const config = useTransformerStore((state) => state.config);

  return (
    <div className="space-y-6">
      <DisclosureLevelSummary title="Transformer block learning path" />

      <VisualizationNotice title="Layer cards are explanatory demos">
        Per-layer backend capture is not exposed yet, so these encoder/decoder panels render
        deterministic educational walkthroughs instead of live `layer_outputs`.
      </VisualizationNotice>
      <div id="layer-stack-visualization" className="grid lg:grid-cols-2 gap-6">
        <LayerVisualization layerType="encoder" layerIndex={0} />
        {config.num_decoder_layers > 0 && (
          <div id="cross-attention-viz">
            <LayerVisualization layerType="decoder" layerIndex={0} />
          </div>
        )}
      </div>

      <DisclosureSection
        level="intermediate"
        title="Core layer internals"
        description="Feed-forward and normalization walkthroughs unlock at the Intermediate level."
        focusSelectors={['#feed-forward-visualization', '#layer-norm-math']}
      >
        <div className="grid lg:grid-cols-2 gap-6">
          <div id="feed-forward-visualization">
            <FeedForward />
          </div>
          <div id="layer-norm-math">
            <LayerNormalization />
          </div>
        </div>
      </DisclosureSection>

      <DisclosureSection
        level="detailed"
        title="FFN activation map"
        description="Neuron activation breakdown unlocks at the Detailed level."
      >
        <FFNActivationMap />
      </DisclosureSection>

      <DisclosureSection
        level="detailed"
        title="Dropout mask visualization"
        description="Stochastic masking internals unlock at the Detailed level."
      >
        <DropoutMaskVisualization />
      </DisclosureSection>

      <DisclosureSection
        level="detailed"
        title="Residual connection walkthrough"
        description="Residual stream composition unlocks at the Detailed level."
      >
        <ResidualConnection />
      </DisclosureSection>

      <DisclosureSection
        id="layer-reference"
        level="math"
        title="Layer math reference"
        description="Formula-level block accounting unlocks at the Math level."
      >
        <LayersMathReference />
      </DisclosureSection>
    </div>
  );
}
