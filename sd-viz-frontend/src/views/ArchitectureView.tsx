/**
 * Architecture Overview View
 *
 * Shows the Stable Diffusion architecture with three main components:
 * - Text Encoder (CLIP)
 * - UNet (denoising network)
 * - VAE (Variational Autoencoder)
 */

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/card';
import { Button } from '../components/ui/button';
import {
  Brain,
  Box,
  Image as ImageIcon,
  ChevronRight,
  ExternalLink,
  Layers,
  MemoryStick,
} from 'lucide-react';
import { InlineMath } from 'react-katex';
import 'katex/dist/katex.min.css';

interface SDComponent {
  id: string;
  name: string;
  type: 'text-encoder' | 'unet' | 'vae';
  description: string;
  icon: React.ReactNode;
  parameters: Record<string, string | React.ReactNode>;
  specs: {
    params?: string;
    flops?: string;
    memory?: string;
  };
}

const components: SDComponent[] = [
  {
    id: 'text-encoder',
    name: 'Text Encoder (CLIP)',
    type: 'text-encoder',
    description: 'Encodes text prompts into embeddings that guide the image generation',
    icon: <Brain className="h-6 w-6" />,
    parameters: {
      'Model': 'CLIP ViT-L/14',
      'Embedding Dim': <InlineMath style={{ color: 'inherit' }}>{"768"}</InlineMath>,
      'Max Tokens': <InlineMath style={{ color: 'inherit' }}>{"77"}</InlineMath>,
    },
    specs: {
      params: '123M',
      memory: '~2GB',
    },
  },
  {
    id: 'unet',
    name: 'UNet',
    type: 'unet',
    description: 'Progressively removes noise from the latent representation',
    icon: <Box className="h-6 w-6" />,
    parameters: {
      'Channels': '320 → 1280 → 320',
      'Layers': 'Down × 4, Middle, Up × 4',
      'Heads': '8 per attention',
    },
    specs: {
      params: '860M',
      flops: '~37 GFLOPs',
      memory: '~3GB',
    },
  },
  {
    id: 'vae',
    name: 'VAE',
    type: 'vae',
    description: 'Compresses images to latent space and decodes back',
    icon: <ImageIcon className="h-6 w-6" />,
    parameters: {
      'Compression': <InlineMath style={{ color: 'inherit' }}>{"8\\times (f=8)"}</InlineMath>,
      'Latent Dim': <InlineMath style={{ color: 'inherit' }}>{"4 \\times 64 \\times 64"}</InlineMath>,
      'Output': <InlineMath style={{ color: 'inherit' }}>{"512 \\times 512 \\times 3"}</InlineMath>,
    },
    specs: {
      params: '83M',
      memory: '~1GB',
    },
  },
];

const flowSteps = [
  {
    id: 'prompt',
    label: 'Text Prompt',
    description: '"a beautiful landscape with mountains"',
  },
  {
    id: 'embeddings',
    label: 'Text Embeddings',
    description: '[77 × 768] vector',
  },
  {
    id: 'latent-noise',
    label: 'Random Latent',
    description: '[4, 64, 64] Gaussian noise',
  },
  {
    id: 'denoised',
    label: 'Denoised Latent',
    description: '[4, 64, 64] clean latent',
  },
  {
    id: 'image',
    label: 'Generated Image',
    description: '[512, 512, 3] RGB image',
  },
];

export const ArchitectureView: React.FC = () => {
  const [selectedComponent, setSelectedComponent] = useState<string | null>(null);
  const [activeStep, setActiveStep] = useState<number>(0);

  const selectedComp = components.find((c) => c.id === selectedComponent);

  return (
    <div className="space-y-6">
      {/* Hero Section */}
      <div className="text-center py-12 bg-gradient-to-br from-primary/10 to-primary/5 rounded-lg">
        <h1 className="text-4xl font-bold mb-2">
          Stable Diffusion Architecture
        </h1>
        <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
          Explore the three core components of Stable Diffusion and how they work together
        </p>
      </div>

      {/* Quick Stats */}
      <div className="grid md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Total Parameters</p>
                <p className="text-2xl font-bold"><InlineMath style={{ color: 'inherit' }}>{"860 \\text{M}"}</InlineMath></p>
              </div>
              <MemoryStick className="h-8 w-8 text-muted-foreground opacity-50" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Components</p>
                <p className="text-2xl font-bold">3</p>
              </div>
              <Layers className="h-8 w-8 text-muted-foreground opacity-50" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Latent Dim</p>
                <p className="text-2xl font-bold"><InlineMath style={{ color: 'inherit' }}>{"4 \\times 64 \\times 64"}</InlineMath></p>
              </div>
              <Box className="h-8 w-8 text-muted-foreground opacity-50" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Output</p>
                <p className="text-2xl font-bold"><InlineMath style={{ color: 'inherit' }}>{"512 \\times 512"}</InlineMath></p>
              </div>
              <ImageIcon className="h-8 w-8 text-muted-foreground opacity-50" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Components Grid */}
      <div className="grid md:grid-cols-3 gap-6">
        {components.map((component) => (
          <Card
            key={component.id}
            className={`cursor-pointer transition-all hover:shadow-lg ${
              selectedComponent === component.id ? 'ring-2 ring-primary' : ''
            }`}
            onClick={() => setSelectedComponent(component.id)}
          >
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="flex-shrink-0 p-3 bg-primary/10 rounded-lg text-primary">
                  {component.icon}
                </div>
                <div>
                  <CardTitle className="text-lg">
                    {component.name}
                  </CardTitle>
                  <CardDescription className="text-xs">
                    {component.type.toUpperCase()}
                  </CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground mb-4">
                {component.description}
              </p>
              <div className="space-y-2 text-xs">
                {Object.entries(component.parameters).map(([key, value]) => (
                  <div key={key} className="flex justify-between">
                    <span className="text-muted-foreground">{key}:</span>
                    <span className="font-mono">{value}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Data Flow */}
      <Card>
        <CardHeader>
          <CardTitle>
            Data Flow
          </CardTitle>
          <CardDescription>
            Complete flow from text prompt to generated image
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap items-center justify-center gap-4 py-8">
            {flowSteps.map((step, index) => (
              <React.Fragment key={step.id}>
                <div
                  className={`flex flex-col items-center p-4 rounded-lg transition-all cursor-pointer ${
                    activeStep === index ? 'bg-primary text-primary-foreground scale-105' : 'bg-muted hover:bg-muted/80'
                  }`}
                  onClick={() => setActiveStep(index)}
                >
                  <div className="text-sm font-medium mb-1">
                    {step.label}
                  </div>
                  <div className="text-xs opacity-80">{step.description}</div>
                </div>
                {index < flowSteps.length - 1 && (
                  <ChevronRight className="h-5 w-5 text-muted-foreground" />
                )}
              </React.Fragment>
            ))}
          </div>

          {/* Step Details */}
          <div className="mt-6 p-4 bg-muted rounded-lg">
            <h4 className="text-sm font-medium mb-2">
              Step Details
            </h4>
            <div className="space-y-2 text-sm">
              <div className="flex items-start gap-2">
                <span className="text-muted-foreground">{activeStep + 1}.</span>
                <span>{flowSteps[activeStep].description}</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Component Detail Modal (simplified) */}
      {selectedComp && (
        <Card className="border-2 border-primary">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-primary/10 rounded-lg text-primary">
                  {selectedComp.icon}
                </div>
                <div>
                  <CardTitle>
                    {selectedComp.name}
                  </CardTitle>
                  <CardDescription>{selectedComp.type.toUpperCase()}</CardDescription>
                </div>
              </div>
              <Button variant="ghost" size="sm" onClick={() => setSelectedComponent(null)}>
                ×
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-sm">
              {selectedComp.description}
            </p>

            {/* Specs */}
            {selectedComp.specs.params && (
              <div className="grid grid-cols-3 gap-4">
                <div className="p-3 bg-muted rounded-lg">
                  <div className="text-xs text-muted-foreground mb-1">Parameters</div>
                  <div className="text-lg font-bold">{selectedComp.specs.params}</div>
                </div>
                {selectedComp.specs.flops && (
                  <div className="p-3 bg-muted rounded-lg">
                    <div className="text-xs text-muted-foreground mb-1">FLOPs</div>
                    <div className="text-lg font-bold">{selectedComp.specs.flops}</div>
                  </div>
                )}
                {selectedComp.specs.memory && (
                  <div className="p-3 bg-muted rounded-lg">
                    <div className="text-xs text-muted-foreground mb-1">Memory</div>
                    <div className="text-lg font-bold">{selectedComp.specs.memory}</div>
                  </div>
                )}
              </div>
            )}

            {/* Parameters */}
            <div className="p-4 bg-muted rounded-lg">
              <h4 className="text-sm font-medium mb-3">Parameters</h4>
              <div className="space-y-2 text-sm">
                {Object.entries(selectedComp.parameters).map(([key, value]) => (
                  <div key={key} className="flex justify-between border-b border-border/50 pb-2">
                    <span className="text-muted-foreground">{key}</span>
                    <span className="font-mono">{value}</span>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Learn More */}
      <Card>
        <CardHeader>
          <CardTitle>
            Learn More
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3 text-sm">
            <p className="text-muted-foreground">
              Stable Diffusion is a latent diffusion model that generates images by progressively denoising in a compressed latent space.
            </p>
            <div className="flex gap-2">
              <a
                href="https://arxiv.org/abs/2112.10752"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none ring-offset-background border border-input bg-background hover:bg-accent hover:text-accent-foreground h-10 px-4 py-2"
              >
                <ExternalLink className="h-4 w-4 mr-2" />
                Paper
              </a>
              <a
                href="https://github.com/Stability-AI/generative-models"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none ring-offset-background border border-input bg-background hover:bg-accent hover:text-accent-foreground h-10 px-4 py-2"
              >
                <ExternalLink className="h-4 w-4 mr-2" />
                Code
              </a>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
