/**
 * NMS View - Non-Maximum Suppression step-by-step visualization
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Slider } from '../components/ui/slider';
import { Filter, Play, Pause, SkipForward, SkipBack, RotateCcw, Info } from 'lucide-react';
import { BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import type { BoundingBox, NMSStep } from '../types';

// Sample detection boxes before NMS
const initialBoxes: BoundingBox[] = [
  { x: 100, y: 80, width: 120, height: 150, confidence: 0.95, classId: 0, className: 'person' },
  { x: 110, y: 90, width: 115, height: 145, confidence: 0.88, classId: 0, className: 'person' },
  { x: 105, y: 85, width: 118, height: 148, confidence: 0.82, classId: 0, className: 'person' },
  { x: 300, y: 100, width: 80, height: 100, confidence: 0.91, classId: 1, className: 'car' },
  { x: 310, y: 105, width: 75, height: 95, confidence: 0.75, classId: 1, className: 'car' },
  { x: 450, y: 200, width: 60, height: 80, confidence: 0.65, classId: 0, className: 'person' },
  { x: 455, y: 205, width: 58, height: 78, confidence: 0.55, classId: 0, className: 'person' },
];

const boxColors: Record<number, string> = {
  0: 'rgb(59, 130, 246)', // person - blue
  1: 'rgb(34, 197, 94)',  // car - green
};

const calculateIoU = (box1: BoundingBox, box2: BoundingBox): number => {
  const x1 = Math.max(box1.x, box2.x);
  const y1 = Math.max(box1.y, box2.y);
  const x2 = Math.min(box1.x + box1.width, box2.x + box2.width);
  const y2 = Math.min(box1.y + box1.height, box2.y + box2.height);

  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const area1 = box1.width * box1.height;
  const area2 = box2.width * box2.height;
  const union = area1 + area2 - intersection;

  return union > 0 ? intersection / union : 0;
};

const generateNMSSteps = (boxes: BoundingBox[], confThreshold: number, iouThreshold: number): NMSStep[] => {
  const steps: NMSStep[] = [];

  // Filter by confidence first
  const filteredBoxes = boxes.filter(b => b.confidence >= confThreshold);
  const sortedIndices = filteredBoxes
    .map((_, i) => i)
    .sort((a, b) => filteredBoxes[b].confidence - filteredBoxes[a].confidence);

  steps.push({
    step: 0,
    description: `Filter boxes by confidence ≥ ${confThreshold.toFixed(2)} and sort by confidence`,
    boxes: filteredBoxes,
    selectedIndices: [],
    suppressedIndices: [],
  });

  const selected: number[] = [];
  const suppressed = new Set<number>();

  for (let i = 0; i < sortedIndices.length; i++) {
    const currentIdx = sortedIndices[i];

    if (suppressed.has(currentIdx)) continue;

    // Add step for selecting current box
    selected.push(currentIdx);
    steps.push({
      step: steps.length,
      description: `Select box ${currentIdx + 1} (confidence: ${filteredBoxes[currentIdx].confidence.toFixed(2)})`,
      boxes: filteredBoxes,
      selectedIndices: [...selected],
      suppressedIndices: Array.from(suppressed),
      currentBox: currentIdx,
    });

    // Compare with remaining boxes
    for (let j = i + 1; j < sortedIndices.length; j++) {
      const compareIdx = sortedIndices[j];
      if (suppressed.has(compareIdx)) continue;

      // Only compare same class
      if (filteredBoxes[currentIdx].classId !== filteredBoxes[compareIdx].classId) continue;

      const iou = calculateIoU(filteredBoxes[currentIdx], filteredBoxes[compareIdx]);

      if (iou >= iouThreshold) {
        suppressed.add(compareIdx);
        steps.push({
          step: steps.length,
          description: `IoU(${currentIdx + 1}, ${compareIdx + 1}) = ${iou.toFixed(3)} ≥ ${iouThreshold.toFixed(2)} → Suppress box ${compareIdx + 1}`,
          boxes: filteredBoxes,
          selectedIndices: [...selected],
          suppressedIndices: Array.from(suppressed),
          currentBox: currentIdx,
          comparedBox: compareIdx,
          iou,
        });
      } else {
        steps.push({
          step: steps.length,
          description: `IoU(${currentIdx + 1}, ${compareIdx + 1}) = ${iou.toFixed(3)} < ${iouThreshold.toFixed(2)} → Keep box ${compareIdx + 1}`,
          boxes: filteredBoxes,
          selectedIndices: [...selected],
          suppressedIndices: Array.from(suppressed),
          currentBox: currentIdx,
          comparedBox: compareIdx,
          iou,
        });
      }
    }
  }

  steps.push({
    step: steps.length,
    description: `NMS complete! ${selected.length} boxes remaining`,
    boxes: filteredBoxes,
    selectedIndices: selected,
    suppressedIndices: Array.from(suppressed),
  });

  return steps;
};

export const NMSView: React.FC = () => {
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);
  const [iouThreshold, setIouThreshold] = useState(0.45);
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  // Generate NMS steps using useMemo
  const nmsSteps = useMemo(() => {
    return generateNMSSteps(initialBoxes, confidenceThreshold, iouThreshold);
  }, [confidenceThreshold, iouThreshold]);

  // Reset step when nmsSteps changes (via key mechanism in handlers)
  const handleConfidenceChange = useCallback((value: number) => {
    setConfidenceThreshold(value);
    setCurrentStep(0);
    setIsPlaying(false);
  }, []);

  const handleIouChange = useCallback((value: number) => {
    setIouThreshold(value);
    setCurrentStep(0);
    setIsPlaying(false);
  }, []);

  // Auto-play with interval
  useEffect(() => {
    if (!isPlaying) return;

    if (currentStep >= nmsSteps.length - 1) {
      // Use a microtask to avoid the sync setState warning
      queueMicrotask(() => setIsPlaying(false));
      return;
    }

    const timer = setTimeout(() => {
      setCurrentStep(s => s + 1);
    }, 1000);

    return () => clearTimeout(timer);
  }, [isPlaying, currentStep, nmsSteps.length]);

  const currentStepData = nmsSteps[currentStep] || { boxes: [], selectedIndices: [], suppressedIndices: [], description: '' };

  const handleReset = useCallback(() => {
    setCurrentStep(0);
    setIsPlaying(false);
  }, []);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Filter className="h-8 w-8 text-primary" />
        <div>
          <h1 className="text-3xl font-bold">Non-Maximum Suppression</h1>
          <p className="text-muted-foreground">Step-by-step NMS visualization</p>
        </div>
      </div>

      {/* Controls */}
      <Card>
        <CardHeader>
          <CardTitle>NMS Parameters</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid md:grid-cols-2 gap-6">
            <Slider
              value={confidenceThreshold}
              onValueChange={handleConfidenceChange}
              min={0}
              max={1}
              step={0.05}
              label="Confidence Threshold"
            />
            <Slider
              value={iouThreshold}
              onValueChange={handleIouChange}
              min={0}
              max={1}
              step={0.05}
              label="IoU Threshold"
            />
          </div>
        </CardContent>
      </Card>

      {/* Main Visualization */}
      <div className="grid lg:grid-cols-3 gap-6">
        {/* Box Visualization */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Detection Boxes</CardTitle>
            <CardDescription>
              Step {currentStep + 1} of {nmsSteps.length}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="relative bg-muted rounded-lg overflow-hidden" style={{ height: '400px' }}>
              <svg className="w-full h-full" viewBox="0 0 600 400">
                {/* Draw all boxes */}
                {currentStepData.boxes.map((box, i) => {
                  const isSelected = currentStepData.selectedIndices.includes(i);
                  const isSuppressed = currentStepData.suppressedIndices.includes(i);
                  const isCurrent = currentStepData.currentBox === i;
                  const isCompared = currentStepData.comparedBox === i;

                  let strokeColor = boxColors[box.classId] || 'gray';
                  let fillOpacity = 0.1;
                  let strokeWidth = 2;
                  let strokeDasharray = 'none';

                  if (isSuppressed) {
                    strokeColor = 'gray';
                    strokeDasharray = '4,4';
                    fillOpacity = 0.05;
                  } else if (isSelected) {
                    strokeWidth = 3;
                    fillOpacity = 0.2;
                  }

                  if (isCurrent) {
                    strokeColor = 'rgb(234, 179, 8)'; // yellow
                    strokeWidth = 4;
                  }

                  if (isCompared) {
                    strokeColor = 'rgb(236, 72, 153)'; // pink
                    strokeWidth = 3;
                  }

                  return (
                    <g key={i}>
                      <rect
                        x={box.x}
                        y={box.y}
                        width={box.width}
                        height={box.height}
                        fill={strokeColor}
                        fillOpacity={fillOpacity}
                        stroke={strokeColor}
                        strokeWidth={strokeWidth}
                        strokeDasharray={strokeDasharray}
                      />
                      <text
                        x={box.x + 4}
                        y={box.y + 14}
                        fontSize={12}
                        fill={strokeColor}
                        fontWeight="bold"
                      >
                        {i + 1}
                      </text>
                      <text
                        x={box.x + 4}
                        y={box.y + 26}
                        fontSize={10}
                        fill={strokeColor}
                      >
                        {box.confidence.toFixed(2)}
                      </text>
                    </g>
                  );
                })}

                {/* IoU visualization */}
                {currentStepData.currentBox !== undefined && currentStepData.comparedBox !== undefined && (
                  <line
                    x1={currentStepData.boxes[currentStepData.currentBox].x + currentStepData.boxes[currentStepData.currentBox].width / 2}
                    y1={currentStepData.boxes[currentStepData.currentBox].y + currentStepData.boxes[currentStepData.currentBox].height / 2}
                    x2={currentStepData.boxes[currentStepData.comparedBox].x + currentStepData.boxes[currentStepData.comparedBox].width / 2}
                    y2={currentStepData.boxes[currentStepData.comparedBox].y + currentStepData.boxes[currentStepData.comparedBox].height / 2}
                    stroke="rgba(255,255,255,0.5)"
                    strokeWidth={2}
                    strokeDasharray="4,4"
                  />
                )}
              </svg>

              {/* Legend */}
              <div className="absolute bottom-2 right-2 bg-background/90 backdrop-blur p-2 rounded text-xs space-y-1">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 border-2 border-yellow-500" />
                  <span>Current</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 border-2 border-pink-500" />
                  <span>Comparing</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 border-2 border-gray-400 border-dashed" />
                  <span>Suppressed</span>
                </div>
              </div>
            </div>

            {/* Playback controls */}
            <div className="flex items-center justify-center gap-2 mt-4">
              <Button variant="outline" size="icon" onClick={handleReset}>
                <RotateCcw className="h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                size="icon"
                onClick={() => setCurrentStep(s => Math.max(0, s - 1))}
                disabled={currentStep === 0}
              >
                <SkipBack className="h-4 w-4" />
              </Button>
              <Button
                size="icon"
                onClick={() => setIsPlaying(!isPlaying)}
                disabled={currentStep >= nmsSteps.length - 1}
              >
                {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              </Button>
              <Button
                variant="outline"
                size="icon"
                onClick={() => setCurrentStep(s => Math.min(nmsSteps.length - 1, s + 1))}
                disabled={currentStep >= nmsSteps.length - 1}
              >
                <SkipForward className="h-4 w-4" />
              </Button>
            </div>

            {/* Progress bar */}
            <div className="w-full bg-muted rounded-full h-2 mt-2">
              <div
                className="bg-primary h-2 rounded-full transition-all"
                style={{ width: `${((currentStep + 1) / nmsSteps.length) * 100}%` }}
              />
            </div>
          </CardContent>
        </Card>

        {/* Step Info */}
        <div className="space-y-6">
          {/* Current Step Description */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Info className="h-5 w-5" />
                Current Step
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm">{currentStepData.description}</p>
              {currentStepData.iou !== undefined && (
                <div className="mt-3 p-3 bg-muted rounded-lg">
                  <div className="text-2xl font-bold text-center">
                    IoU = {currentStepData.iou.toFixed(3)}
                  </div>
                  <div className="text-center text-sm text-muted-foreground mt-1">
                    {currentStepData.iou >= iouThreshold ? (
                      <span className="text-red-500">≥ {iouThreshold.toFixed(2)} (Suppress)</span>
                    ) : (
                      <span className="text-green-500">&lt; {iouThreshold.toFixed(2)} (Keep)</span>
                    )}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Box Status */}
          <Card>
            <CardHeader>
              <CardTitle>Box Status</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>Total boxes:</span>
                  <span className="font-medium">{initialBoxes.length}</span>
                </div>
                <div className="flex justify-between">
                  <span>After confidence filter:</span>
                  <span className="font-medium">{currentStepData.boxes.length}</span>
                </div>
                <div className="flex justify-between text-green-600">
                  <span>Selected:</span>
                  <span className="font-medium">{currentStepData.selectedIndices.length}</span>
                </div>
                <div className="flex justify-between text-red-600">
                  <span>Suppressed:</span>
                  <span className="font-medium">{currentStepData.suppressedIndices.length}</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* NMS Formula */}
      <Card>
        <CardHeader>
          <CardTitle>NMS Algorithm</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="bg-muted p-4 rounded-lg overflow-x-auto">
            <BlockMath math="\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}" />
          </div>
          <div className="grid md:grid-cols-4 gap-4 text-sm">
            <div className="p-3 bg-muted rounded-lg">
              <strong>1. Filter</strong>
              <p className="text-muted-foreground">Remove boxes with confidence &lt; threshold</p>
            </div>
            <div className="p-3 bg-muted rounded-lg">
              <strong>2. Sort</strong>
              <p className="text-muted-foreground">Sort remaining boxes by confidence (descending)</p>
            </div>
            <div className="p-3 bg-muted rounded-lg">
              <strong>3. Select</strong>
              <p className="text-muted-foreground">Pick highest confidence box</p>
            </div>
            <div className="p-3 bg-muted rounded-lg">
              <strong>4. Suppress</strong>
              <p className="text-muted-foreground">Remove boxes with IoU ≥ threshold, repeat</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
