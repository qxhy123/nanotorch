import { lazy, Suspense, useState } from 'react';
import { Box, Cpu, Layers, LoaderCircle } from 'lucide-react';
import { DisclosureLevelSummary, DisclosureSection } from '../components/layout/DisclosureSection';
import { VisualizationNotice } from '../components/layout/VisualizationNotice';
import { Badge } from '../components/ui/badge';
import { Button } from '../components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Latex } from '../components/ui/Latex';
import { useTransformerStore } from '../stores/transformerStore';

const LazyTransformerStructure3D = lazy(async () => {
  const module = await import('../components/visualization/transformer/TransformerStructure3D');
  return { default: module.TransformerStructure3D };
});

const LazyArchitectureComparison = lazy(async () => {
  const module = await import('../components/visualization/architecture/ArchitectureComparison');
  return { default: module.ArchitectureComparison };
});

function SectionFallback({
  title,
  description,
}: {
  title: string;
  description: string;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <LoaderCircle className="h-4 w-4 animate-spin text-primary" />
          {title}
        </CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid gap-3 sm:grid-cols-3">
          <div className="h-24 animate-pulse rounded-lg bg-muted" />
          <div className="h-24 animate-pulse rounded-lg bg-muted" />
          <div className="h-24 animate-pulse rounded-lg bg-muted" />
        </div>
      </CardContent>
    </Card>
  );
}

function StructureMathReference() {
  const config = useTransformerStore((state) => state.config);
  const totalBlocks = config.num_encoder_layers + config.num_decoder_layers;
  const attentionModules = config.num_encoder_layers + (2 * config.num_decoder_layers);
  const residualAdds = (2 * config.num_encoder_layers) + (3 * config.num_decoder_layers);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Architecture Reference</CardTitle>
        <CardDescription>
          Stack depth and block composition for the current transformer configuration.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-3 md:grid-cols-4">
          <div className="rounded-lg border bg-muted/40 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Model width</div>
            <div className="mt-2 text-2xl font-semibold">{config.d_model}</div>
          </div>
          <div className="rounded-lg border bg-muted/40 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Total blocks</div>
            <div className="mt-2 text-2xl font-semibold">{totalBlocks}</div>
          </div>
          <div className="rounded-lg border bg-muted/40 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Attention modules</div>
            <div className="mt-2 text-2xl font-semibold">{attentionModules}</div>
          </div>
          <div className="rounded-lg border bg-muted/40 p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Residual adds</div>
            <div className="mt-2 text-2xl font-semibold">{residualAdds}</div>
          </div>
        </div>

        <div className="grid gap-4 lg:grid-cols-2">
          <div className="rounded-lg border bg-background p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Encoder block</div>
            <Latex display>{'x_{l+1} = \\operatorname{LN}(x_l + \\operatorname{FFN}(\\operatorname{MHA}(x_l)))'}</Latex>
          </div>
          <div className="rounded-lg border bg-background p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Decoder block</div>
            <Latex display>{'y_{l+1} = \\operatorname{LN}(y_l + \\operatorname{CrossAttn}(\\operatorname{SelfAttn}(y_l), x))'}</Latex>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default function StructureTab() {
  const [show3D, setShow3D] = useState(false);

  return (
    <div className="space-y-6">
      <DisclosureLevelSummary title="Architecture exploration path" />

      <VisualizationNotice title="3D explorer now loads on demand">
        The WebGL scene is isolated from the rest of the Structure tab so lighter devices
        can read the architecture view first and only fetch the 3D runtime when needed.
      </VisualizationNotice>

      <DisclosureSection
        level="detailed"
        title="Interactive 3D explorer"
        description="The immersive node-by-node architecture view unlocks at the Detailed level."
      >
        {show3D ? (
          <div className="space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border bg-muted/30 px-4 py-3">
              <div className="flex items-center gap-3">
                <div className="rounded-full bg-primary/10 p-2 text-primary">
                  <Box className="h-4 w-4" />
                </div>
                <div>
                  <p className="text-sm font-medium">Interactive 3D explorer</p>
                  <p className="text-xs text-muted-foreground">
                    Loaded separately so it does not slow down the whole Structure tab.
                  </p>
                </div>
              </div>
              <Button variant="outline" size="sm" onClick={() => setShow3D(false)}>
                Hide 3D View
              </Button>
            </div>

            <Suspense
              fallback={
                <SectionFallback
                  title="Loading 3D architecture"
                  description="Preparing the WebGL scene and transformer graph."
                />
              }
            >
              <LazyTransformerStructure3D />
            </Suspense>
          </div>
        ) : (
          <Card className="border-dashed">
            <CardHeader>
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <Box className="h-5 w-5 text-primary" />
                    Deferred 3D Architecture View
                  </CardTitle>
                  <CardDescription>
                    Open the WebGL explorer only when you want the immersive structure view.
                  </CardDescription>
                </div>
                <Badge variant="secondary" className="shrink-0">
                  On-demand load
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-3 md:grid-cols-3">
                <div className="rounded-lg border bg-background p-4">
                  <div className="mb-2 flex items-center gap-2 text-sm font-medium">
                    <Cpu className="h-4 w-4 text-primary" />
                    Lower idle GPU cost
                  </div>
                  <p className="text-sm text-muted-foreground">
                    We keep the tab lightweight until the 3D canvas is actually needed.
                  </p>
                </div>
                <div className="rounded-lg border bg-background p-4">
                  <div className="mb-2 flex items-center gap-2 text-sm font-medium">
                    <Layers className="h-4 w-4 text-primary" />
                    Faster first interaction
                  </div>
                  <p className="text-sm text-muted-foreground">
                    The architecture comparison can stay responsive without waiting for Three.js.
                  </p>
                </div>
                <div className="rounded-lg border bg-background p-4">
                  <div className="mb-2 flex items-center gap-2 text-sm font-medium">
                    <Box className="h-4 w-4 text-primary" />
                    Same 3D tools
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Camera presets, node details, and the scene explorer are still available after load.
                  </p>
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-3">
                <Button onClick={() => setShow3D(true)}>Load 3D Explorer</Button>
                <span className="text-xs text-muted-foreground">
                  Best when you want to inspect node-by-node structure rather than browse the overview.
                </span>
              </div>
            </CardContent>
          </Card>
        )}
      </DisclosureSection>

      <Suspense
        fallback={
          <SectionFallback
            title="Loading architecture comparison"
            description="Preparing the side-by-side architecture analysis."
          />
        }
      >
        <LazyArchitectureComparison />
      </Suspense>

      <DisclosureSection
        level="math"
        title="Architecture reference"
        description="Block accounting and stack formulas unlock at the Math level."
      >
        <StructureMathReference />
      </DisclosureSection>
    </div>
  );
}
