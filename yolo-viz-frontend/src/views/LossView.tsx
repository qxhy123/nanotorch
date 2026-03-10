/**
 * Loss View - Loss function comparison and visualization
 */

import React, { useState, useMemo } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Slider } from '../components/ui/slider';
import { TrendingDown, Move, Info } from 'lucide-react';
import { BlockMath, InlineMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid } from 'recharts';
import type { LossType } from '../types';

const lossInfo: Record<LossType, { name: string; description: string; formula: string; advantage: string }> = {
  iou: {
    name: 'IoU Loss',
    description: 'Basic Intersection over Union loss',
    formula: 'L_{IoU} = 1 - IoU',
    advantage: 'Scale-invariant, directly optimizes the evaluation metric',
  },
  giou: {
    name: 'GIoU Loss',
    description: 'Generalized IoU - handles non-overlapping boxes',
    formula: 'L_{GIoU} = 1 - IoU + \\frac{|C \\setminus (A \\cup B)|}{|C|}',
    advantage: 'Provides gradient for non-overlapping boxes',
  },
  diou: {
    name: 'DIoU Loss',
    description: 'Distance IoU - considers center distance',
    formula: 'L_{DIoU} = 1 - IoU + \\frac{\\rho^2(b, b^{gt})}{c^2}',
    advantage: 'Faster convergence by optimizing center distance',
  },
  ciou: {
    name: 'CIoU Loss',
    description: 'Complete IoU - considers overlap, center, and aspect ratio',
    formula: 'L_{CIoU} = 1 - IoU + \\frac{\\rho^2(b, b^{gt})}{c^2} + \\alpha v',
    advantage: 'Most comprehensive, best regression accuracy',
  },
};

const CANVAS_SIZE = 300;
const GT_BOX = { x: 100, y: 100, w: 100, h: 80 };

export const LossView: React.FC = () => {
  const [selectedLoss, setSelectedLoss] = useState<LossType>('ciou');
  const [predBox, setPredBox] = useState({ x: 120, y: 90, w: 90, h: 90 });

  // Calculate losses
  const losses = useMemo(() => {
    const gtBox = GT_BOX;

    // Calculate intersection
    const x1 = Math.max(gtBox.x, predBox.x);
    const y1 = Math.max(gtBox.y, predBox.y);
    const x2 = Math.min(gtBox.x + gtBox.w, predBox.x + predBox.w);
    const y2 = Math.min(gtBox.y + gtBox.h, predBox.y + predBox.h);

    const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const areaGT = gtBox.w * gtBox.h;
    const areaPred = predBox.w * predBox.h;
    const union = areaGT + areaPred - intersection;

    const iou = union > 0 ? intersection / union : 0;

    // GIoU: enclosing box
    const encX1 = Math.min(gtBox.x, predBox.x);
    const encY1 = Math.min(gtBox.y, predBox.y);
    const encX2 = Math.max(gtBox.x + gtBox.w, predBox.x + predBox.w);
    const encY2 = Math.max(gtBox.y + gtBox.h, predBox.y + predBox.h);
    const encArea = (encX2 - encX1) * (encY2 - encY1);
    const giou = iou - (encArea - union) / encArea;

    // DIoU: center distance
    const gtCenterX = gtBox.x + gtBox.w / 2;
    const gtCenterY = gtBox.y + gtBox.h / 2;
    const predCenterX = predBox.x + predBox.w / 2;
    const predCenterY = predBox.y + predBox.h / 2;
    const centerDist = Math.sqrt(Math.pow(gtCenterX - predCenterX, 2) + Math.pow(gtCenterY - predCenterY, 2));
    const diagonalDist = Math.sqrt(Math.pow(encX2 - encX1, 2) + Math.pow(encY2 - encY1, 2));
    const diou = iou - Math.pow(centerDist / diagonalDist, 2);

    // CIoU: aspect ratio
    const v = (4 / Math.pow(Math.PI, 2)) * Math.pow(
      Math.atan(gtBox.w / gtBox.h) - Math.atan(predBox.w / predBox.h),
      2
    );
    const alpha = v / (1 - iou + v + 1e-7);
    const ciou = diou - alpha * v;

    return {
      iou: { value: iou, loss: 1 - iou },
      giou: { value: giou, loss: 1 - giou },
      diou: { value: diou, loss: 1 - diou },
      ciou: { value: ciou, loss: 1 - ciou },
      metrics: { intersection, union, centerDist, encArea },
    };
  }, [predBox]);

  const lossChartData = [
    { name: 'IoU', loss: losses.iou.loss, fill: '#3b82f6' },
    { name: 'GIoU', loss: losses.giou.loss, fill: '#22c55e' },
    { name: 'DIoU', loss: losses.diou.loss, fill: '#f59e0b' },
    { name: 'CIoU', loss: losses.ciou.loss, fill: '#ef4444' },
  ];

  const lossComponentData = [
    { name: 'Box Loss', value: 40, color: '#3b82f6' },
    { name: 'Objectness', value: 35, color: '#22c55e' },
    { name: 'Class Loss', value: 25, color: '#f59e0b' },
  ];

  const info = lossInfo[selectedLoss];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <TrendingDown className="h-8 w-8 text-primary" />
        <div>
          <h1 className="text-3xl font-bold">Loss Functions</h1>
          <p className="text-muted-foreground">IoU family loss comparison</p>
        </div>
      </div>

      {/* Loss Type Selector */}
      <Card>
        <CardHeader>
          <CardTitle>Select Loss Function</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2 mb-4">
            {(Object.keys(lossInfo) as LossType[]).map((loss) => (
              <Button
                key={loss}
                variant={selectedLoss === loss ? 'default' : 'outline'}
                onClick={() => setSelectedLoss(loss)}
              >
                {lossInfo[loss].name}
              </Button>
            ))}
          </div>
          <div className="p-4 bg-muted rounded-lg">
            <h3 className="font-semibold">{info.name}</h3>
            <p className="text-sm text-muted-foreground mt-1">{info.description}</p>
            <div className="mt-3 p-3 bg-background rounded overflow-x-auto">
              <BlockMath math={info.formula} />
            </div>
            <p className="text-xs text-primary mt-2">{info.advantage}</p>
          </div>
        </CardContent>
      </Card>

      {/* Interactive Box Visualization */}
      <div className="grid lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Move className="h-5 w-5" />
              Drag to Move Predicted Box
            </CardTitle>
            <CardDescription>Adjust the predicted box to see loss changes</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {/* Sliders for box position */}
              <div className="grid grid-cols-2 gap-4">
                <Slider
                  value={predBox.x}
                  onValueChange={(v) => setPredBox(p => ({ ...p, x: v }))}
                  min={0}
                  max={200}
                  step={5}
                  label="X Position"
                />
                <Slider
                  value={predBox.y}
                  onValueChange={(v) => setPredBox(p => ({ ...p, y: v }))}
                  min={0}
                  max={200}
                  step={5}
                  label="Y Position"
                />
                <Slider
                  value={predBox.w}
                  onValueChange={(v) => setPredBox(p => ({ ...p, w: v }))}
                  min={20}
                  max={200}
                  step={5}
                  label="Width"
                />
                <Slider
                  value={predBox.h}
                  onValueChange={(v) => setPredBox(p => ({ ...p, h: v }))}
                  min={20}
                  max={200}
                  step={5}
                  label="Height"
                />
              </div>

              {/* Box visualization */}
              <div className="relative bg-muted rounded-lg" style={{ height: CANVAS_SIZE }}>
                <svg className="w-full h-full" viewBox={`0 0 ${CANVAS_SIZE} ${CANVAS_SIZE}`}>
                  {/* Enclosing box (for GIoU) */}
                  {selectedLoss !== 'iou' && (
                    <rect
                      x={Math.min(GT_BOX.x, predBox.x)}
                      y={Math.min(GT_BOX.y, predBox.y)}
                      width={Math.max(GT_BOX.x + GT_BOX.w, predBox.x + predBox.w) - Math.min(GT_BOX.x, predBox.x)}
                      height={Math.max(GT_BOX.y + GT_BOX.h, predBox.y + predBox.h) - Math.min(GT_BOX.y, predBox.y)}
                      fill="rgba(100, 100, 100, 0.1)"
                      stroke="gray"
                      strokeDasharray="4,4"
                      strokeWidth={1}
                    />
                  )}

                  {/* Ground truth box */}
                  <rect
                    x={GT_BOX.x}
                    y={GT_BOX.y}
                    width={GT_BOX.w}
                    height={GT_BOX.h}
                    fill="rgba(34, 197, 94, 0.2)"
                    stroke="rgb(34, 197, 94)"
                    strokeWidth={2}
                  />
                  <text x={GT_BOX.x + 4} y={GT_BOX.y + 14} fontSize={10} fill="rgb(34, 197, 94)">GT</text>

                  {/* Predicted box */}
                  <rect
                    x={predBox.x}
                    y={predBox.y}
                    width={predBox.w}
                    height={predBox.h}
                    fill="rgba(59, 130, 246, 0.2)"
                    stroke="rgb(59, 130, 246)"
                    strokeWidth={2}
                  />
                  <text x={predBox.x + 4} y={predBox.y + 14} fontSize={10} fill="rgb(59, 130, 246)">Pred</text>

                  {/* Center distance line (for DIoU/CIoU) */}
                  {(selectedLoss === 'diou' || selectedLoss === 'ciou') && (
                    <>
                      <line
                        x1={GT_BOX.x + GT_BOX.w / 2}
                        y1={GT_BOX.y + GT_BOX.h / 2}
                        x2={predBox.x + predBox.w / 2}
                        y2={predBox.y + predBox.h / 2}
                        stroke="rgb(239, 68, 68)"
                        strokeWidth={2}
                        strokeDasharray="4,4"
                      />
                      <circle cx={GT_BOX.x + GT_BOX.w / 2} cy={GT_BOX.y + GT_BOX.h / 2} r={3} fill="rgb(34, 197, 94)" />
                      <circle cx={predBox.x + predBox.w / 2} cy={predBox.y + predBox.h / 2} r={3} fill="rgb(59, 130, 246)" />
                    </>
                  )}
                </svg>

                {/* Legend */}
                <div className="absolute bottom-2 right-2 bg-background/90 backdrop-blur p-2 rounded text-xs space-y-1">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 border-2 border-green-500 bg-green-500/20" />
                    <span>Ground Truth</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 border-2 border-blue-500 bg-blue-500/20" />
                    <span>Predicted</span>
                  </div>
                  {selectedLoss !== 'iou' && (
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 border border-gray-400 border-dashed" />
                      <span>Enclosing Box</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Loss Values */}
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Info className="h-5 w-5" />
                Loss Values
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {(Object.keys(lossInfo) as LossType[]).map((loss) => {
                  const value = losses[loss];
                  const isSelected = selectedLoss === loss;
                  return (
                    <div
                      key={loss}
                      className={`p-3 rounded-lg ${isSelected ? 'bg-primary/10 border-2 border-primary' : 'bg-muted'}`}
                    >
                      <div className="flex justify-between items-center">
                        <span className="font-medium">{lossInfo[loss].name}</span>
                        <span className="text-lg font-bold">{value.loss.toFixed(4)}</span>
                      </div>
                      <div className="w-full bg-background rounded-full h-2 mt-2">
                        <div
                          className="h-2 rounded-full transition-all"
                          style={{
                            width: `${Math.max(0, Math.min(100, value.loss * 100))}%`,
                            backgroundColor: isSelected ? 'hsl(var(--primary))' : 'gray',
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>

          {/* Metrics */}
          <Card>
            <CardHeader>
              <CardTitle>Metrics</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="p-2 bg-muted rounded">
                  <span className="text-muted-foreground">IoU:</span>
                  <span className="font-medium ml-2">{losses.iou.value.toFixed(4)}</span>
                </div>
                <div className="p-2 bg-muted rounded">
                  <span className="text-muted-foreground">Center Dist:</span>
                  <span className="font-medium ml-2">{losses.metrics.centerDist.toFixed(1)}px</span>
                </div>
                <div className="p-2 bg-muted rounded">
                  <span className="text-muted-foreground">Intersection:</span>
                  <span className="font-medium ml-2">{losses.metrics.intersection.toFixed(0)}px²</span>
                </div>
                <div className="p-2 bg-muted rounded">
                  <span className="text-muted-foreground">Union:</span>
                  <span className="font-medium ml-2">{losses.metrics.union.toFixed(0)}px²</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Loss Comparison Chart */}
      <div className="grid lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Loss Comparison</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={lossChartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis domain={[0, 2]} />
                <Tooltip />
                <Bar dataKey="loss" fill="hsl(var(--primary))">
                  {lossChartData.map((entry, index) => (
                    <Cell key={index} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>YOLO Total Loss Components</CardTitle>
            <CardDescription>
              <InlineMath math="L = \lambda_1 L_{box} + \lambda_2 L_{obj} + \lambda_3 L_{cls}" />
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie
                  data={lossComponentData}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  label={({ name, percent }) => `${name}: ${((percent ?? 0) * 100).toFixed(0)}%`}
                >
                  {lossComponentData.map((entry, index) => (
                    <Cell key={index} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Formula Summary */}
      <Card>
        <CardHeader>
          <CardTitle>IoU Family Formulas</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="p-4 bg-muted rounded-lg">
              <h4 className="font-semibold mb-2">IoU</h4>
              <BlockMath math="\text{IoU} = \frac{|A \cap B|}{|A \cup B|}" />
            </div>
            <div className="p-4 bg-muted rounded-lg">
              <h4 className="font-semibold mb-2">GIoU</h4>
              <BlockMath math="\text{GIoU} = \text{IoU} - \frac{|C \setminus (A \cup B)|}{|C|}" />
            </div>
            <div className="p-4 bg-muted rounded-lg">
              <h4 className="font-semibold mb-2">DIoU</h4>
              <BlockMath math="\text{DIoU} = \text{IoU} - \frac{\rho^2(b, b^{gt})}{c^2}" />
            </div>
            <div className="p-4 bg-muted rounded-lg">
              <h4 className="font-semibold mb-2">CIoU</h4>
              <BlockMath math="\text{CIoU} = \text{DIoU} - \alpha v" />
              <p className="text-xs text-muted-foreground mt-1">
                where <InlineMath math="v = \frac{4}{\pi^2}(\arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h})^2" />
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
