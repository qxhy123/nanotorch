import React from 'react';
import { useTransformerStore } from '../../stores/transformerStore';
import { DisclosureSection } from '../layout/DisclosureSection';
import { useDisclosureLevel } from '../providers';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Label } from '../ui/label';
import { Input } from '../ui/input';
import { Slider } from '../ui/slider';
import { Select } from '../ui/select';
import { Button } from '../ui/button';
import { RotateCcw } from 'lucide-react';

export const ParameterPanel: React.FC = () => {
  const { config, setConfig, reset } = useTransformerStore();
  const { level } = useDisclosureLevel();
  const [isResetting, setIsResetting] = React.useState(false);
  const headDim = config.d_model / config.nhead;
  const expansionRatio = (config.dim_feedforward / config.d_model).toFixed(1);

  const handleReset = () => {
    setIsResetting(true);
    reset();
    setTimeout(() => setIsResetting(false), 500);
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Model Configuration</CardTitle>
            <CardDescription>
              Adjust Transformer model parameters. Current level: {level}. Higher disclosure
              levels unlock capacity, regularization, and numerical controls.
            </CardDescription>
          </div>
          <Button variant="outline" size="icon" onClick={handleReset} loading={isResetting}>
            <RotateCcw className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* d_model */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label htmlFor="d_model">Model Dimension (d_model)</Label>
            <span className="text-sm text-muted-foreground">{config.d_model}</span>
          </div>
          <Slider
            id="d_model"
            min={128}
            max={1024}
            step={64}
            value={[config.d_model]}
            onValueChange={([value]) => setConfig({ d_model: value })}
          />
        </div>

        {/* nhead */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label htmlFor="nhead">Number of Heads (nhead)</Label>
            <span className="text-sm text-muted-foreground">{config.nhead}</span>
          </div>
          <Slider
            id="nhead"
            min={1}
            max={16}
            step={1}
            value={[config.nhead]}
            onValueChange={([value]) => {
              if (config.d_model % value === 0) {
                setConfig({ nhead: value });
              }
            }}
          />
          <p className="text-xs text-muted-foreground">
            Must divide d_model evenly ({config.d_model} % {config.nhead} = {config.d_model % config.nhead})
          </p>
        </div>

        {/* num_encoder_layers */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label htmlFor="num_encoder_layers">Encoder Layers</Label>
            <span className="text-sm text-muted-foreground">{config.num_encoder_layers}</span>
          </div>
          <Slider
            id="num_encoder_layers"
            min={1}
            max={12}
            step={1}
            value={[config.num_encoder_layers]}
            onValueChange={([value]) => setConfig({ num_encoder_layers: value })}
          />
        </div>

        {/* num_decoder_layers */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label htmlFor="num_decoder_layers">Decoder Layers</Label>
            <span className="text-sm text-muted-foreground">{config.num_decoder_layers}</span>
          </div>
          <Slider
            id="num_decoder_layers"
            min={0}
            max={12}
            step={1}
            value={[config.num_decoder_layers]}
            onValueChange={([value]) => setConfig({ num_decoder_layers: value })}
          />
        </div>

        <DisclosureSection
          level="intermediate"
          title="Sequence and capacity controls"
          description="Sequence length, vocabulary size, and FFN width unlock at the Intermediate level."
        >
          <div className="space-y-6">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="dim_feedforward">Feedforward Dimension</Label>
                <span className="text-sm text-muted-foreground">{config.dim_feedforward}</span>
              </div>
              <Slider
                id="dim_feedforward"
                min={512}
                max={4096}
                step={256}
                value={[config.dim_feedforward]}
                onValueChange={([value]) => setConfig({ dim_feedforward: value })}
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="max_seq_len">Max Sequence Length</Label>
                <span className="text-sm text-muted-foreground">{config.max_seq_len}</span>
              </div>
              <Slider
                id="max_seq_len"
                min={64}
                max={512}
                step={64}
                value={[config.max_seq_len]}
                onValueChange={([value]) => setConfig({ max_seq_len: value })}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="vocab_size">Vocabulary Size</Label>
              <Input
                id="vocab_size"
                type="number"
                min={100}
                max={100000}
                value={config.vocab_size}
                onChange={(e) => setConfig({ vocab_size: parseInt(e.target.value) || 10000 })}
              />
            </div>
          </div>
        </DisclosureSection>

        <DisclosureSection
          level="detailed"
          title="Regularization and activations"
          description="Dropout and activation-function controls unlock at the Detailed level."
        >
          <div className="space-y-6">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="dropout">Dropout Rate</Label>
                <span className="text-sm text-muted-foreground">{config.dropout}</span>
              </div>
              <Slider
                id="dropout"
                min={0}
                max={0.5}
                step={0.05}
                value={[config.dropout]}
                onValueChange={([value]) => setConfig({ dropout: value })}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="activation">Activation Function</Label>
              <Select
                id="activation"
                value={config.activation}
                onChange={(e) => setConfig({ activation: e.target.value as 'relu' | 'gelu' })}
              >
                <option value="relu">ReLU</option>
                <option value="gelu">GELU</option>
              </Select>
            </div>
          </div>
        </DisclosureSection>

        <DisclosureSection
          level="math"
          title="Numerical and execution settings"
          description="LayerNorm epsilon and layout-related flags unlock at the Math level."
        >
          <div className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="layer_norm_eps">Layer Norm Epsilon</Label>
              <Input
                id="layer_norm_eps"
                type="number"
                step="1e-6"
                value={config.layer_norm_eps}
                onChange={(e) => setConfig({ layer_norm_eps: parseFloat(e.target.value) || 1e-5 })}
              />
            </div>

            <div className="flex items-center justify-between">
              <Label htmlFor="batch_first">Batch First</Label>
              <input
                id="batch_first"
                type="checkbox"
                checked={config.batch_first}
                onChange={(e) => setConfig({ batch_first: e.target.checked })}
                className="h-4 w-4"
              />
            </div>

            <div className="flex items-center justify-between">
              <Label htmlFor="norm_first">Norm First (Pre-LN)</Label>
              <input
                id="norm_first"
                type="checkbox"
                checked={config.norm_first}
                onChange={(e) => setConfig({ norm_first: e.target.checked })}
                className="h-4 w-4"
              />
            </div>

            <div className="rounded-lg border bg-muted/30 p-4 text-sm text-muted-foreground">
              Each head currently sees {headDim} features, and the FFN expands the residual stream
              by about {expansionRatio}x before projecting back to `d_model`.
            </div>
          </div>
        </DisclosureSection>
      </CardContent>
    </Card>
  );
};
