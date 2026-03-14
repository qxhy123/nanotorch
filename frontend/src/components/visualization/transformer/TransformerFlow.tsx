import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../ui/card';
import { Badge } from '../../ui/badge';
import { Button } from '../../ui/button';
import { useTransformerStore } from '../../../stores/transformerStore';
import { Play, Pause, SkipBack, SkipForward, Info, ChevronDown, ChevronUp, Zap } from 'lucide-react';
import { Latex } from '../../ui/Latex';

interface TransformerFlowProps {
  className?: string;
}

type FlowStep = {
  id: string;
  name: string;
  description: string;
  type: 'input' | 'embedding' | 'positional' | 'encoder' | 'decoder' | 'output';
  details?: React.ReactNode;
};

const FLOW_STEPS: FlowStep[] = [
  {
    id: 'input',
    name: 'Input',
    description: 'Tokenized text input',
    type: 'input',
    details: (
      <div className="space-y-2 text-sm">
        <div className="flex items-center justify-between p-2 bg-background rounded">
          <span className="text-muted-foreground">Input tokens:</span>
          <Badge variant="secondary">[batch, seq_len]</Badge>
        </div>
        <p className="text-muted-foreground">Raw token IDs from tokenizer</p>
      </div>
    )
  },
  {
    id: 'embedding',
    name: 'Token Embedding',
    description: 'Convert tokens to dense vectors',
    type: 'embedding',
    details: (
      <div className="space-y-2 text-sm">
        <div className="flex items-center justify-between p-2 bg-background rounded">
          <span className="text-muted-foreground">Output shape:</span>
          <Badge variant="secondary">[batch, seq_len, d_model]</Badge>
        </div>
        <div className="text-xs text-muted-foreground bg-background p-2 rounded">
          Each token ID → <Latex>{`d_{model}`}</Latex>-dim vector
        </div>
      </div>
    )
  },
  {
    id: 'positional',
    name: 'Positional Encoding',
    description: 'Add position information using sin/cos',
    type: 'positional',
    details: (
      <div className="space-y-2 text-sm">
        <div className="bg-background p-2 rounded">
          <Latex>{`PE(pos, 2i) = \\sin\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)`}</Latex>
        </div>
        <div className="bg-background p-2 rounded">
          <Latex>{`PE(pos, 2i+1) = \\cos\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)`}</Latex>
        </div>
        <p className="text-muted-foreground">Adds position info to embeddings</p>
      </div>
    )
  },
  {
    id: 'encoder',
    name: 'Encoder Layers',
    description: 'Process through self-attention & feed-forward',
    type: 'encoder',
    details: (
      <div className="space-y-2 text-sm">
        <div className="text-xs font-mono bg-background p-2 rounded space-y-1">
          <div>for each encoder layer:</div>
          <div className="pl-4">→ Multi-Head Self-Attention</div>
          <div className="pl-4">→ Add & Norm</div>
          <div className="pl-4">→ Feed-Forward Network</div>
          <div className="pl-4">→ Add & Norm</div>
        </div>
      </div>
    )
  },
  {
    id: 'decoder',
    name: 'Decoder Layers',
    description: 'Generate with masked, cross attention & FFN',
    type: 'decoder',
    details: (
      <div className="space-y-2 text-sm">
        <div className="text-xs font-mono bg-background p-2 rounded space-y-1">
          <div>for each decoder layer:</div>
          <div className="pl-4">→ Masked Multi-Head Self-Attn</div>
          <div className="pl-4">→ Add & Norm</div>
          <div className="pl-4">→ Cross-Attention (encoder output)</div>
          <div className="pl-4">→ Add & Norm</div>
          <div className="pl-4">→ Feed-Forward Network</div>
          <div className="pl-4">→ Add & Norm</div>
        </div>
      </div>
    )
  },
  {
    id: 'output',
    name: 'Output',
    description: 'Final predictions and probabilities',
    type: 'output',
    details: (
      <div className="space-y-2 text-sm">
        <div className="flex items-center justify-between p-2 bg-background rounded">
          <span className="text-muted-foreground">Output shape:</span>
          <Badge variant="secondary">[batch, seq_len, vocab_size]</Badge>
        </div>
        <p className="text-muted-foreground">Logits → Softmax → Probabilities</p>
      </div>
    )
  },
];

// 数据流动画组件
const DataFlowParticle: React.FC<{ active: boolean; delay: number }> = ({ active, delay }) => {
  const [position, setPosition] = useState(0);

  useEffect(() => {
    if (!active) {
      const frameId = requestAnimationFrame(() => {
        setPosition(0);
      });
      return () => cancelAnimationFrame(frameId);
    }

    const timeout = setTimeout(() => {
      setPosition(1);
    }, delay);

    return () => clearTimeout(timeout);
  }, [active, delay]);

  if (!active) return null;

  return (
    <div
      className="absolute top-1/2 -translate-y-1/2 w-2 h-2 bg-primary rounded-full shadow-lg transition-all duration-700 ease-in-out"
      style={{
        left: `${position * 100}%`,
        opacity: position > 0 && position < 1 ? 1 : 0,
      }}
    />
  );
};

export const TransformerFlow: React.FC<TransformerFlowProps> = ({ className }) => {
  const { config, output, animationState, setIsPlaying, setCurrentStep, setAnimationSpeed } = useTransformerStore();
  const { isPlaying, speed } = animationState;
  const [selectedStep, setSelectedStep] = useState(0);
  const [expandedDetails, setExpandedDetails] = useState(true);
  const [hoveredStep, setHoveredStep] = useState<number | null>(null);
  const [showKeyHint, setShowKeyHint] = useState(true);
  const containerRef = useRef<HTMLDivElement>(null);

  // 获取当前步骤的实际数据
  const getStepData = useCallback((stepId: string) => {
    if (!output?.data) return null;

    switch (stepId) {
      case 'embedding':
      case 'positional':
        return output.data.embeddings;
      case 'encoder':
        return output.data.layerOutputs;
      case 'decoder':
        return output.data.layerOutputs;
      default:
        return null;
    }
  }, [output]);

  // 动画播放逻辑
  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (isPlaying) {
      interval = setInterval(() => {
        setSelectedStep((prev) => {
          const next = (prev + 1) % FLOW_STEPS.length;
          setCurrentStep(next);
          return next;
        });
      }, speed);
    }
    return () => clearInterval(interval);
  }, [isPlaying, speed, setCurrentStep]);

  const handleStepClick = useCallback((index: number) => {
    setSelectedStep(index);
    setCurrentStep(index);
  }, [setCurrentStep]);

  // 键盘快捷键
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // 忽略在输入框中的按键
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      switch (e.key) {
        case 'ArrowLeft':
          e.preventDefault();
          setSelectedStep((prev) => {
            const next = Math.max(0, prev - 1);
            setCurrentStep(next);
            return next;
          });
          break;
        case 'ArrowRight':
          e.preventDefault();
          setSelectedStep((prev) => {
            const next = Math.min(FLOW_STEPS.length - 1, prev + 1);
            setCurrentStep(next);
            return next;
          });
          break;
        case ' ':
          e.preventDefault();
          setIsPlaying(!isPlaying);
          setShowKeyHint(false);
          break;
        case 'Home':
          e.preventDefault();
          handleStepClick(0);
          break;
        case 'End':
          e.preventDefault();
          handleStepClick(FLOW_STEPS.length - 1);
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleStepClick, isPlaying, setIsPlaying, setCurrentStep]);

  const getStepColor = (type: FlowStep['type'], isSelected: boolean, isHovered: boolean) => {
    const colors = {
      input: 'bg-gray-500 border-gray-500',
      embedding: 'bg-blue-500 border-blue-500',
      positional: 'bg-purple-500 border-purple-500',
      encoder: 'bg-green-500 border-green-500',
      decoder: 'bg-orange-500 border-orange-500',
      output: 'bg-red-500 border-red-500',
    };

    if (isSelected) {
      return `ring-4 ring-opacity-50 ${colors[type].replace('bg-', 'ring-')} ring-offset-2`;
    }
    if (isHovered) {
      return `${colors[type]} bg-opacity-20 scale-105`;
    }
    return 'bg-muted border-border hover:bg-muted/80';
  };

  const getStepIcon = (type: FlowStep['type']) => {
    const icons = {
      input: '📝',
      embedding: '🔤',
      positional: '📍',
      encoder: '⬆️',
      decoder: '⬇️',
      output: '📊',
    };
    return icons[type];
  };

  const currentStepData = getStepData(FLOW_STEPS[selectedStep].id);

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Zap className="h-5 w-5 text-primary" />
              Transformer Flow
            </CardTitle>
            <CardDescription>Interactive pipeline visualization</CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="icon"
              onClick={() => handleStepClick(0)}
              className="transition-all duration-200"
              disabled={isPlaying}
              title="First step (Home)"
            >
              <SkipBack className="h-4 w-4" />
            </Button>
            <Button
              variant={isPlaying ? 'destructive' : 'default'}
              size="icon"
              onClick={() => {
                setIsPlaying(!isPlaying);
                setShowKeyHint(false);
              }}
              className="transition-all duration-200"
              title={isPlaying ? 'Pause (Space)' : 'Play (Space)'}
            >
              {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            </Button>
            <Button
              variant="outline"
              size="icon"
              onClick={() => handleStepClick(FLOW_STEPS.length - 1)}
              className="transition-all duration-200"
              disabled={isPlaying}
              title="Last step (End)"
            >
              <SkipForward className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* 键盘快捷键提示 */}
        {showKeyHint && (
          <div className="mt-2 flex items-center gap-3 text-xs text-muted-foreground bg-muted/50 p-2 rounded">
            <kbd className="px-2 py-1 bg-background border rounded">←</kbd>
            <kbd className="px-2 py-1 bg-background border rounded">→</kbd>
            <span>navigate</span>
            <kbd className="px-2 py-1 bg-background border rounded ml-2">Space</kbd>
            <span>play/pause</span>
            <button
              onClick={() => setShowKeyHint(false)}
              className="ml-auto text-muted-foreground hover:text-foreground"
            >
              ×
            </button>
          </div>
        )}
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Flow Diagram */}
        <div ref={containerRef} className="relative">
          <div className="flex items-center justify-between gap-2 overflow-x-auto pb-4">
            {FLOW_STEPS.map((step, index) => (
              <React.Fragment key={step.id}>
                <div
                  className={`relative flex flex-col items-center gap-2 p-4 rounded-lg border-2 cursor-pointer transition-all duration-300 min-w-[120px] ${
                    selectedStep === index
                      ? getStepColor(step.type, true, false)
                      : getStepColor(step.type, false, hoveredStep === index)
                  }`}
                  onClick={() => handleStepClick(index)}
                  onMouseEnter={() => setHoveredStep(index)}
                  onMouseLeave={() => setHoveredStep(null)}
                >
                  {/* 激活指示器 */}
                  {selectedStep === index && (
                    <div className="absolute -top-1 -right-1 w-3 h-3 bg-primary rounded-full animate-ping" />
                  )}
                  {selectedStep === index && (
                    <div className="absolute -top-1 -right-1 w-3 h-3 bg-primary rounded-full" />
                  )}

                  <div className="text-2xl transition-transform duration-300">
                    {getStepIcon(step.type)}
                  </div>
                  <div className="text-sm font-medium text-center">{step.name}</div>
                  <Badge
                    variant={selectedStep === index ? 'default' : 'secondary'}
                    className="text-xs transition-all duration-300"
                  >
                    {index + 1}
                  </Badge>

                  {/* 数据指示器 */}
                  {getStepData(step.id) && (
                    <div className="absolute -bottom-1 -left-1 w-4 h-4 bg-green-500 rounded-full border-2 border-background" title="Has data" />
                  )}
                </div>

                {/* 连接线和数据流动画 */}
                {index < FLOW_STEPS.length - 1 && (
                  <div className="relative flex items-center justify-center shrink-0 w-12">
                    <div
                      className={`h-0.5 transition-all duration-500 ${
                        selectedStep > index ? 'bg-primary' : 'bg-muted-foreground'
                      }`}
                    />
                    <div
                      className={`w-2 h-2 rounded-full transition-all duration-500 ${
                        selectedStep > index ? 'bg-primary scale-125' : 'bg-muted-foreground'
                      }`}
                    />
                    {/* 数据流动画 */}
                    <DataFlowParticle active={selectedStep > index} delay={0} />
                  </div>
                )}
              </React.Fragment>
            ))}
          </div>
        </div>

        {/* 步骤详情 */}
        <div className="space-y-3">
          {/* 详情切换按钮 */}
          <button
            onClick={() => setExpandedDetails(!expandedDetails)}
            className="flex items-center gap-2 text-sm font-medium hover:text-primary transition-colors"
          >
            <Info className="h-4 w-4" />
            <span>Step Details</span>
            {expandedDetails ? (
              <ChevronUp className="h-4 w-4" />
            ) : (
              <ChevronDown className="h-4 w-4" />
            )}
          </button>

          {expandedDetails && (
            <div className="p-6 bg-muted/50 rounded-lg space-y-4 animate-in fade-in slide-in-from-top-2 duration-300">
              {/* 步骤标题和描述 */}
              <div className="flex items-start gap-4">
                <div className="text-4xl animate-bounce duration-1000">
                  {getStepIcon(FLOW_STEPS[selectedStep].type)}
                </div>
                <div className="flex-1">
                  <h3 className="text-lg font-semibold mb-1 flex items-center gap-2">
                    {FLOW_STEPS[selectedStep].name}
                    {getStepData(FLOW_STEPS[selectedStep].id) && (
                      <Badge variant="outline" className="text-xs">Data Available</Badge>
                    )}
                  </h3>
                  <p className="text-sm text-muted-foreground">
                    {FLOW_STEPS[selectedStep].description}
                  </p>
                </div>
              </div>

              {/* 配置信息 */}
              {FLOW_STEPS[selectedStep].id === 'embedding' && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm p-2 bg-background rounded">
                    <span className="font-medium">Embedding Dim:</span>
                    <Badge variant="secondary">{config.d_model}</Badge>
                  </div>
                  <div className="flex items-center justify-between text-sm p-2 bg-background rounded">
                    <span className="font-medium">Vocab Size:</span>
                    <Badge variant="secondary">{config.vocab_size.toLocaleString()}</Badge>
                  </div>
                  <div className="text-xs text-muted-foreground p-2 bg-background rounded">
                    <Latex>{`\\text{Shape: } [batch, seq\\_len, {d_{model}}]`}</Latex>
                  </div>
                </div>
              )}

              {FLOW_STEPS[selectedStep].id === 'encoder' && (
                <div className="grid grid-cols-2 gap-2">
                  <div className="flex items-center justify-between text-sm p-2 bg-background rounded">
                    <span className="font-medium">Layers:</span>
                    <Badge variant="secondary">{config.num_encoder_layers}</Badge>
                  </div>
                  <div className="flex items-center justify-between text-sm p-2 bg-background rounded">
                    <span className="font-medium">Heads:</span>
                    <Badge variant="secondary">{config.nhead}</Badge>
                  </div>
                  <div className="flex items-center justify-between text-sm p-2 bg-background rounded">
                    <span className="font-medium">FFN Dim:</span>
                    <Badge variant="secondary">{config.dim_feedforward}</Badge>
                  </div>
                  <div className="flex items-center justify-between text-sm p-2 bg-background rounded">
                    <span className="font-medium">Dropout:</span>
                    <Badge variant="secondary">{config.dropout}</Badge>
                  </div>
                </div>
              )}

              {FLOW_STEPS[selectedStep].id === 'decoder' && (
                <div className="grid grid-cols-2 gap-2">
                  <div className="flex items-center justify-between text-sm p-2 bg-background rounded">
                    <span className="font-medium">Layers:</span>
                    <Badge variant="secondary">{config.num_decoder_layers}</Badge>
                  </div>
                  <div className="flex items-center justify-between text-sm p-2 bg-background rounded">
                    <span className="font-medium">Heads:</span>
                    <Badge variant="secondary">{config.nhead}</Badge>
                  </div>
                  <div className="flex items-center justify-between text-sm p-2 bg-background rounded">
                    <span className="font-medium">FFN Dim:</span>
                    <Badge variant="secondary">{config.dim_feedforward}</Badge>
                  </div>
                  <div className="flex items-center justify-between text-sm p-2 bg-background rounded col-span-2">
                    <span className="font-medium">Attention Types:</span>
                    <div className="flex gap-1">
                      <Badge variant="outline">Masked Self</Badge>
                      <Badge variant="outline">Cross</Badge>
                    </div>
                  </div>
                </div>
              )}

              {/* 自定义详情 */}
              {FLOW_STEPS[selectedStep].details}

              {/* 实际数据展示 */}
              {currentStepData && (
                <div className="pt-2 border-t">
                  <div className="text-xs font-medium text-muted-foreground mb-2">Live Data</div>
                  <div className="text-xs font-mono bg-background p-2 rounded">
                    {Array.isArray(currentStepData)
                      ? `${currentStepData.length} layers available`
                      : 'Data loaded - Check visualization panels for details'}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* 进度和速度控制 */}
        <div className="space-y-4">
          {/* 可拖拽进度条 */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">Progress</span>
              <span className="text-muted-foreground">
                Step {selectedStep + 1} of {FLOW_STEPS.length}
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={FLOW_STEPS.length - 1}
              value={selectedStep}
              onChange={(e) => handleStepClick(parseInt(e.target.value))}
              disabled={isPlaying}
              className="w-full h-2 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary disabled:cursor-not-allowed disabled:opacity-50"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              {FLOW_STEPS.map((step, i) => (
                <span
                  key={step.id}
                  className={`cursor-pointer hover:text-foreground transition-colors ${selectedStep === i ? 'text-primary font-medium' : ''}`}
                  onClick={() => !isPlaying && handleStepClick(i)}
                >
                  {i + 1}
                </span>
              ))}
            </div>
          </div>

          {/* 动画速度控制 */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">Animation Speed</span>
              <Badge variant="secondary">{speed}ms</Badge>
            </div>
            <input
              type="range"
              min={100}
              max={2000}
              step={100}
              value={speed}
              onChange={(e) => setAnimationSpeed(parseInt(e.target.value))}
              className="w-full h-2 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Fast (100ms)</span>
              <span>Slow (2s)</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
