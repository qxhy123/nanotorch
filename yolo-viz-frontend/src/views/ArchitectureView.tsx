/**
 * Architecture View - YOLO overall architecture visualization
 */

import React from 'react';
import { Link } from 'react-router-dom';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Layers, GitMerge, Box, ArrowRight, Target, Zap, Settings, BarChart3, Sparkles } from 'lucide-react';
import { BlockMath, InlineMath } from 'react-katex';
import 'katex/dist/katex.min.css';

const stats = [
  { label: 'Total Parameters', value: '3.2M - 68M', icon: Settings },
  { label: 'Components', value: '3 Main', icon: Box },
  { label: 'Input Size', value: '640x640', icon: Target },
  { label: 'mAP@0.5', value: '53.9%', icon: BarChart3 },
];

const components = [
  {
    name: 'Backbone',
    icon: Layers,
    path: '/backbone',
    description: 'Feature extraction network (CSPDarknet)',
    details: 'Extracts hierarchical features from input images at multiple scales',
    color: 'from-blue-500 to-cyan-500',
  },
  {
    name: 'Neck',
    icon: GitMerge,
    path: '/neck',
    description: 'Feature pyramid network (FPN/PANet)',
    details: 'Combines multi-scale features for better object detection',
    color: 'from-purple-500 to-pink-500',
  },
  {
    name: 'Head',
    icon: Box,
    path: '/head',
    description: 'Detection head (Decoupled/Anchor-free)',
    details: 'Predicts bounding boxes, objectness, and class probabilities',
    color: 'from-orange-500 to-red-500',
  },
];

const dataFlow = [
  { label: 'Input Image', size: '640x640x3' },
  { label: 'Backbone', size: 'Multi-scale features' },
  { label: 'Neck', size: 'P3, P4, P5' },
  { label: 'Head', size: 'Predictions' },
  { label: 'Detections', size: 'Boxes + Classes' },
];

export const ArchitectureView: React.FC = () => {
  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-primary/10 via-primary/5 to-background border p-8 md:p-12">
        <div className="relative z-10">
          <div className="flex items-center gap-3 mb-4">
            <Target className="h-10 w-10 text-primary" />
            <h1 className="text-4xl font-bold">YOLO Architecture</h1>
          </div>
          <p className="text-lg text-muted-foreground max-w-2xl mb-6">
            You Only Look Once - Real-time object detection system that processes images in a single forward pass.
            Explore the architecture components and understand how YOLO achieves state-of-the-art speed and accuracy.
          </p>
          <div className="flex flex-wrap gap-3">
            <Link to="/playground">
              <Button size="lg">
                <Sparkles className="mr-2 h-5 w-5" />
                Try Interactive Demo
              </Button>
            </Link>
            <Link to="/versions">
              <Button variant="outline" size="lg">
                Compare Versions
              </Button>
            </Link>
          </div>
        </div>
        <div className="absolute -right-20 -top-20 h-64 w-64 rounded-full bg-primary/10 blur-3xl" />
        <div className="absolute -bottom-20 -left-20 h-64 w-64 rounded-full bg-primary/5 blur-3xl" />
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {stats.map((stat) => {
          const Icon = stat.icon;
          return (
            <Card key={stat.label} className="text-center p-4">
              <Icon className="h-6 w-6 mx-auto text-primary mb-2" />
              <div className="text-2xl font-bold">{stat.value}</div>
              <div className="text-sm text-muted-foreground">{stat.label}</div>
            </Card>
          );
        })}
      </div>

      {/* Main Components */}
      <div>
        <h2 className="text-2xl font-bold mb-4">Main Components</h2>
        <div className="grid md:grid-cols-3 gap-6">
          {components.map((component) => {
            const Icon = component.icon;
            return (
              <Link key={component.name} to={component.path}>
                <Card className="h-full hover:shadow-lg transition-shadow cursor-pointer group">
                  <CardHeader>
                    <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${component.color} flex items-center justify-center mb-2`}>
                      <Icon className="h-6 w-6 text-white" />
                    </div>
                    <CardTitle className="flex items-center gap-2">
                      {component.name}
                      <ArrowRight className="h-4 w-4 opacity-0 group-hover:opacity-100 transition-opacity" />
                    </CardTitle>
                    <CardDescription>{component.description}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">{component.details}</p>
                  </CardContent>
                </Card>
              </Link>
            );
          })}
        </div>
      </div>

      {/* Data Flow Diagram */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5 text-primary" />
            Data Flow Pipeline
          </CardTitle>
          <CardDescription>
            Single forward pass through the network for real-time detection
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap items-center justify-center gap-2 py-6">
            {dataFlow.map((step, index) => (
              <React.Fragment key={step.label}>
                <div className="flex flex-col items-center p-4 bg-muted rounded-lg min-w-[120px]">
                  <span className="font-semibold text-sm">{step.label}</span>
                  <span className="text-xs text-muted-foreground mt-1">{step.size}</span>
                </div>
                {index < dataFlow.length - 1 && (
                  <ArrowRight className="h-5 w-5 text-muted-foreground shrink-0" />
                )}
              </React.Fragment>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Output Format */}
      <Card>
        <CardHeader>
          <CardTitle>Output Tensor Format</CardTitle>
          <CardDescription>
            YOLO outputs predictions in a structured tensor format
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="bg-muted p-4 rounded-lg overflow-x-auto">
            <BlockMath math="\text{Output} = [B, S^2, A \times (5 + C)]" />
          </div>
          <div className="grid md:grid-cols-2 gap-4 text-sm">
            <div className="space-y-2">
              <p><InlineMath math="B" /> = Batch size</p>
              <p><InlineMath math="S" /> = Grid size (e.g., 20, 40, 80)</p>
              <p><InlineMath math="A" /> = Number of anchors per cell</p>
            </div>
            <div className="space-y-2">
              <p><InlineMath math="5" /> = Box parameters (<InlineMath math="x, y, w, h, conf" />)</p>
              <p><InlineMath math="C" /> = Number of classes (e.g., 80 for COCO)</p>
            </div>
          </div>
          <div className="mt-4 p-4 bg-primary/5 rounded-lg">
            <p className="text-sm font-medium mb-2">Box Coordinates Decoding:</p>
            <div className="space-y-2 text-sm">
              <div className="flex items-center gap-2">
                <InlineMath math="b_x = \sigma(t_x) + c_x" />
                <span className="text-muted-foreground">Center x</span>
              </div>
              <div className="flex items-center gap-2">
                <InlineMath math="b_y = \sigma(t_y) + c_y" />
                <span className="text-muted-foreground">Center y</span>
              </div>
              <div className="flex items-center gap-2">
                <InlineMath math="b_w = p_w \cdot e^{t_w}" />
                <span className="text-muted-foreground">Width</span>
              </div>
              <div className="flex items-center gap-2">
                <InlineMath math="b_h = p_h \cdot e^{t_h}" />
                <span className="text-muted-foreground">Height</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Quick Links */}
      <div className="grid md:grid-cols-3 gap-4">
        <Link to="/nms">
          <Card className="p-4 hover:bg-accent transition-colors cursor-pointer">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-green-500/10 flex items-center justify-center">
                <Target className="h-5 w-5 text-green-500" />
              </div>
              <div>
                <p className="font-medium">Non-Maximum Suppression</p>
                <p className="text-sm text-muted-foreground">Step-by-step NMS animation</p>
              </div>
            </div>
          </Card>
        </Link>
        <Link to="/loss">
          <Card className="p-4 hover:bg-accent transition-colors cursor-pointer">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-red-500/10 flex items-center justify-center">
                <BarChart3 className="h-5 w-5 text-red-500" />
              </div>
              <div>
                <p className="font-medium">Loss Functions</p>
                <p className="text-sm text-muted-foreground">IoU, GIoU, DIoU, CIoU</p>
              </div>
            </div>
          </Card>
        </Link>
        <Link to="/anchors">
          <Card className="p-4 hover:bg-accent transition-colors cursor-pointer">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-blue-500/10 flex items-center justify-center">
                <Box className="h-5 w-5 text-blue-500" />
              </div>
              <div>
                <p className="font-medium">Anchor Boxes</p>
                <p className="text-sm text-muted-foreground">Grid cells & anchor matching</p>
              </div>
            </div>
          </Card>
        </Link>
      </div>
    </div>
  );
};

