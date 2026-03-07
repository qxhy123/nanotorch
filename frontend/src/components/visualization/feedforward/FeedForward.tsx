import React, { useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../ui/card';
import { Badge } from '../../ui/badge';
import { useTransformerStore } from '../../../stores/transformerStore';
import { Zap } from 'lucide-react';
import { Latex } from '../../ui/Latex';

interface FeedForwardProps {
  className?: string;
}

export const FeedForward: React.FC<FeedForwardProps> = ({ className }) => {
  const { config, visualizationState } = useTransformerStore();
  const { selectedLayer } = visualizationState;

  const expansionRatio = useMemo(() => {
    return config.dim_feedforward / config.d_model;
  }, [config]);

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Feed Forward Network</CardTitle>
            <CardDescription>
              Layer {selectedLayer + 1} | Expansion: {expansionRatio}x
            </CardDescription>
          </div>
          <Zap className="h-5 w-5 text-muted-foreground" />
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Architecture Diagram */}
        <div className="space-y-2">
          <h4 className="text-sm font-medium">Network Architecture</h4>
          <div className="flex items-center justify-center gap-2 p-4 bg-muted rounded-lg overflow-x-auto">
            {/* Input */}
            <div className="flex flex-col items-center gap-2 shrink-0">
              <div className="w-20 h-16 bg-primary/20 rounded-lg flex items-center justify-center">
                <span className="text-sm font-medium">Input</span>
              </div>
              <span className="text-xs text-muted-foreground">
                {config.d_model}d
              </span>
            </div>

            <div className="text-muted-foreground shrink-0">→</div>

            {/* Linear 1 */}
            <div className="flex flex-col items-center gap-2 shrink-0">
              <div className="w-24 h-16 bg-blue-500/20 rounded-lg flex items-center justify-center">
                <span className="text-sm font-medium">Linear 1</span>
              </div>
              <span className="text-xs text-muted-foreground">
                ({config.d_model}, {config.dim_feedforward})
              </span>
            </div>

            <div className="text-muted-foreground shrink-0">→</div>

            {/* Activation */}
            <div className="flex flex-col items-center gap-2 shrink-0">
              <div className="w-20 h-16 bg-orange-500/20 rounded-lg flex items-center justify-center">
                <span className="text-sm font-medium">{config.activation.toUpperCase()}</span>
              </div>
              <span className="text-xs text-muted-foreground">
                {config.dim_feedforward}d
              </span>
            </div>

            <div className="text-muted-foreground shrink-0">→</div>

            {/* Linear 2 */}
            <div className="flex flex-col items-center gap-2 shrink-0">
              <div className="w-24 h-16 bg-green-500/20 rounded-lg flex items-center justify-center">
                <span className="text-sm font-medium">Linear 2</span>
              </div>
              <span className="text-xs text-muted-foreground">
                ({config.dim_feedforward}, {config.d_model})
              </span>
            </div>

            <div className="text-muted-foreground shrink-0">→</div>

            {/* Output */}
            <div className="flex flex-col items-center gap-2 shrink-0">
              <div className="w-20 h-16 bg-primary/20 rounded-lg flex items-center justify-center">
                <span className="text-sm font-medium">Output</span>
              </div>
              <span className="text-xs text-muted-foreground">
                {config.d_model}d
              </span>
            </div>
          </div>
        </div>

        {/* Parameters */}
        <div className="space-y-2">
          <h4 className="text-sm font-medium">Layer Parameters</h4>
          <div className="grid grid-cols-2 gap-4">
            <div className="p-3 bg-muted/50 rounded-lg">
              <div className="text-xs text-muted-foreground">Input Dimension</div>
              <div className="text-xl font-bold">{config.d_model}</div>
            </div>
            <div className="p-3 bg-muted/50 rounded-lg">
              <div className="text-xs text-muted-foreground">Hidden Dimension</div>
              <div className="text-xl font-bold">{config.dim_feedforward}</div>
            </div>
            <div className="p-3 bg-muted/50 rounded-lg">
              <div className="text-xs text-muted-foreground">Output Dimension</div>
              <div className="text-xl font-bold">{config.d_model}</div>
            </div>
            <div className="p-3 bg-muted/50 rounded-lg">
              <div className="text-xs text-muted-foreground">Activation</div>
              <div className="text-xl font-bold">{config.activation.toUpperCase()}</div>
            </div>
          </div>
        </div>

        {/* Formula */}
        <div className="space-y-2">
          <h4 className="text-sm font-medium">Computation</h4>
          <div className="p-4 bg-muted rounded-lg space-y-3 text-sm">
            <Latex>{`FFN(x) = \\text{Linear}_2(\\text{Activation}(\\text{Linear}_1(x)))`}</Latex>
            <div className="text-xs text-muted-foreground">
              <Latex>{`= \\max(0, xW_1 + b_1)W_2 + b_2 \\quad \\text{(ReLU)}`}</Latex>
            </div>
            <div className="text-xs text-muted-foreground">
              <Latex>{`= 0.5xW_2(1 + \\tanh(\\sqrt{2/\\pi}(xW_1 + b_1)(0.44xW_1 + 0.44b_1 + 1))) + b_2 \\quad \\text{(GELU approximation)}`}</Latex>
            </div>
          </div>
        </div>

        {/* Parameter Count */}
        <div className="space-y-2">
          <h4 className="text-sm font-medium">Parameter Count</h4>
          <div className="p-4 bg-muted rounded-lg space-y-2">
            <div className="flex justify-between text-sm">
              <span>W<sub>1</sub> weight matrix:</span>
              <Badge variant="secondary">{(config.d_model * config.dim_feedforward).toLocaleString()}</Badge>
            </div>
            <div className="flex justify-between text-sm">
              <span>b<sub>1</sub> bias:</span>
              <Badge variant="secondary">{config.dim_feedforward.toLocaleString()}</Badge>
            </div>
            <div className="flex justify-between text-sm">
              <span>W<sub>2</sub> weight matrix:</span>
              <Badge variant="secondary">{(config.dim_feedforward * config.d_model).toLocaleString()}</Badge>
            </div>
            <div className="flex justify-between text-sm">
              <span>b<sub>2</sub> bias:</span>
              <Badge variant="secondary">{config.d_model.toLocaleString()}</Badge>
            </div>
            <div className="border-t pt-2 mt-2 flex justify-between text-sm font-medium">
              <span>Total:</span>
              <Badge variant="default">
                {(2 * config.d_model * config.dim_feedforward + config.d_model + config.dim_feedforward).toLocaleString()}
              </Badge>
            </div>
          </div>
        </div>

        {/* Activation Function Visualization */}
        <div className="space-y-2">
          <h4 className="text-sm font-medium">Activation Function</h4>
          <div className="p-4 bg-muted rounded-lg">
            {config.activation === 'relu' ? (
              <>
                <div className="font-medium mb-2">ReLU (Rectified Linear Unit)</div>
                <div className="text-sm text-muted-foreground mb-3">
                  <Latex>{`\\text{ReLU}(x) = \\max(0, x)`}</Latex>
                </div>
                <svg viewBox="-3 -1.5 6 3" className="w-full h-24">
                  <line x1="-2.5" y1="0" x2="2.5" y2="0" stroke="currentColor" strokeOpacity="0.2" />
                  <line x1="0" y1="-1.2" x2="0" y2="1.2" stroke="currentColor" strokeOpacity="0.2" />
                  <polyline
                    fill="none"
                    stroke="rgb(59, 130, 246)"
                    strokeWidth="0.1"
                    points="-2.5,0 0,0 0,1.2 2.5,1.2"
                  />
                </svg>
              </>
            ) : (
              <>
                <div className="font-medium mb-2">GELU (Gaussian Error Linear Unit)</div>
                <div className="text-sm text-muted-foreground mb-3">
                  <Latex>{`\\text{GELU}(x) = x \\cdot \\Phi(x) \\approx 0.5x\\left(1 + \\tanh\\left(\\sqrt{\\frac{2}{\\pi}}(x + 0.044715x^3)\\right)\\right)`}</Latex>
                </div>
                <svg viewBox="-3 -1.5 6 3" className="w-full h-24">
                  <line x1="-2.5" y1="0" x2="2.5" y2="0" stroke="currentColor" strokeOpacity="0.2" />
                  <line x1="0" y1="-1.2" x2="0" y2="1.2" stroke="currentColor" strokeOpacity="0.2" />
                  <path
                    fill="none"
                    stroke="rgb(59, 130, 246)"
                    strokeWidth="0.1"
                    d="M -2.5,-0.01 Q -1,-0.01 0,0 T 2.5,1.2"
                  />
                </svg>
              </>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
