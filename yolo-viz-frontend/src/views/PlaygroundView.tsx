/**
 * Playground View - Interactive detection demo
 */

import React, { useState, useCallback } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Slider } from '../components/ui/slider';
import { Sparkles, Image, Play, RefreshCw, Settings } from 'lucide-react';
import type { YOLOVersion, BoundingBox } from '../types';

// Sample images for detection
const sampleImages = [
  { id: 'street', name: 'Street Scene', description: 'Urban traffic with cars and pedestrians' },
  { id: 'office', name: 'Office', description: 'Indoor scene with people and objects' },
  { id: 'park', name: 'Park', description: 'Outdoor scene with people and animals' },
  { id: 'kitchen', name: 'Kitchen', description: 'Indoor scene with appliances and utensils' },
];

// Simulated detection results
const generateDetections = (imageId: string, version: YOLOVersion, confThreshold: number): BoundingBox[] => {
  const baseDetections: Record<string, BoundingBox[]> = {
    street: [
      { x: 50, y: 120, width: 100, height: 60, confidence: 0.95, classId: 2, className: 'car' },
      { x: 200, y: 100, width: 80, height: 150, confidence: 0.88, classId: 0, className: 'person' },
      { x: 320, y: 110, width: 90, height: 55, confidence: 0.82, classId: 2, className: 'car' },
      { x: 400, y: 130, width: 70, height: 130, confidence: 0.76, classId: 0, className: 'person' },
      { x: 480, y: 140, width: 60, height: 50, confidence: 0.65, classId: 1, className: 'bicycle' },
    ],
    office: [
      { x: 80, y: 50, width: 100, height: 180, confidence: 0.92, classId: 0, className: 'person' },
      { x: 220, y: 80, width: 90, height: 170, confidence: 0.89, classId: 0, className: 'person' },
      { x: 350, y: 150, width: 80, height: 60, confidence: 0.78, classId: 56, className: 'chair' },
      { x: 150, y: 200, width: 120, height: 40, confidence: 0.72, classId: 60, className: 'laptop' },
    ],
    park: [
      { x: 100, y: 80, width: 60, height: 140, confidence: 0.91, classId: 0, className: 'person' },
      { x: 200, y: 150, width: 80, height: 60, confidence: 0.85, classId: 16, className: 'dog' },
      { x: 350, y: 100, width: 70, height: 150, confidence: 0.79, classId: 0, className: 'person' },
      { x: 450, y: 180, width: 50, height: 40, confidence: 0.68, classId: 32, className: 'bench' },
    ],
    kitchen: [
      { x: 50, y: 100, width: 120, height: 150, confidence: 0.94, classId: 68, className: 'refrigerator' },
      { x: 200, y: 120, width: 100, height: 80, confidence: 0.87, classId: 69, className: 'oven' },
      { x: 350, y: 150, width: 60, height: 40, confidence: 0.75, classId: 41, className: 'cup' },
      { x: 420, y: 140, width: 80, height: 100, confidence: 0.71, classId: 44, className: 'bottle' },
    ],
  };

  // Add some version-specific variance
  const versionMultiplier: Record<YOLOVersion, number> = {
    v1: 0.85, v2: 0.88, v3: 0.90, v4: 0.93, v5: 0.95, v6: 0.96, v7: 0.98, v8: 1.0,
  };

  const detections = (baseDetections[imageId] || baseDetections.street).map(box => ({
    ...box,
    confidence: Math.min(0.99, box.confidence * versionMultiplier[version]),
  }));

  return detections.filter(d => d.confidence >= confThreshold);
};

const classColors: Record<number, string> = {
  0: '#3b82f6',  // person - blue
  1: '#22c55e',  // bicycle - green
  2: '#ef4444',  // car - red
  16: '#f59e0b', // dog - yellow
  32: '#8b5cf6', // bench - purple
  41: '#ec4899', // cup - pink
  44: '#06b6d4', // bottle - cyan
  56: '#6366f1', // chair - indigo
  60: '#14b8a6', // laptop - teal
  68: '#f97316', // refrigerator - orange
  69: '#84cc16', // oven - lime
};

export const PlaygroundView: React.FC = () => {
  const [selectedImage, setSelectedImage] = useState('street');
  const [selectedVersion, setSelectedVersion] = useState<YOLOVersion>('v8');
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);
  const [iouThreshold, setIouThreshold] = useState(0.45);
  const [detections, setDetections] = useState<BoundingBox[]>([]);
  const [isDetecting, setIsDetecting] = useState(false);
  const [processingTime, setProcessingTime] = useState<number | null>(null);

  const runDetection = useCallback(() => {
    setIsDetecting(true);
    setDetections([]);

    // Simulate processing time based on version
    const baseTime: Record<YOLOVersion, number> = {
      v1: 22, v2: 25, v3: 33, v4: 15, v5: 7, v6: 5.5, v7: 8, v8: 3.5,
    };

    const time = baseTime[selectedVersion] + Math.random() * 2;

    setTimeout(() => {
      const results = generateDetections(selectedImage, selectedVersion, confidenceThreshold);
      setDetections(results);
      setProcessingTime(time);
      setIsDetecting(false);
    }, 500);
  }, [selectedImage, selectedVersion, confidenceThreshold]);

  const versions: YOLOVersion[] = ['v5', 'v6', 'v7', 'v8'];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Sparkles className="h-8 w-8 text-primary" />
        <div>
          <h1 className="text-3xl font-bold">Detection Playground</h1>
          <p className="text-muted-foreground">Interactive object detection demo</p>
        </div>
      </div>

      {/* Controls */}
      <div className="grid lg:grid-cols-3 gap-6">
        {/* Image Selection */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Image className="h-5 w-5" />
              Sample Image
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-2">
              {sampleImages.map((img) => (
                <Button
                  key={img.id}
                  variant={selectedImage === img.id ? 'default' : 'outline'}
                  className="h-auto py-3 flex flex-col"
                  onClick={() => setSelectedImage(img.id)}
                >
                  <span className="font-medium">{img.name}</span>
                  <span className="text-xs opacity-70">{img.description}</span>
                </Button>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Model Selection */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Settings className="h-5 w-5" />
              Model Version
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-2 mb-4">
              {versions.map((v) => (
                <Button
                  key={v}
                  variant={selectedVersion === v ? 'default' : 'outline'}
                  onClick={() => setSelectedVersion(v)}
                >
                  YOLO{v}
                </Button>
              ))}
            </div>
            <div className="space-y-4">
              <Slider
                value={confidenceThreshold}
                onValueChange={setConfidenceThreshold}
                min={0}
                max={1}
                step={0.05}
                label="Confidence"
              />
              <Slider
                value={iouThreshold}
                onValueChange={setIouThreshold}
                min={0}
                max={1}
                step={0.05}
                label="IoU Threshold"
              />
            </div>
          </CardContent>
        </Card>

        {/* Actions & Stats */}
        <Card>
          <CardHeader>
            <CardTitle>Detection</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <Button
              className="w-full"
              size="lg"
              onClick={runDetection}
              disabled={isDetecting}
            >
              {isDetecting ? (
                <>
                  <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                  Detecting...
                </>
              ) : (
                <>
                  <Play className="mr-2 h-4 w-4" />
                  Run Detection
                </>
              )}
            </Button>

            {processingTime !== null && (
              <div className="p-4 bg-muted rounded-lg space-y-2">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Processing Time:</span>
                  <span className="font-medium">{processingTime.toFixed(1)} ms</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Objects Detected:</span>
                  <span className="font-medium">{detections.length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">FPS (estimated):</span>
                  <span className="font-medium">{(1000 / processingTime).toFixed(0)}</span>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Detection Visualization */}
      <Card>
        <CardHeader>
          <CardTitle>Detection Results</CardTitle>
          <CardDescription>
            {selectedImage} scene with YOLO{selectedVersion} (conf ≥ {confidenceThreshold.toFixed(2)})
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="relative bg-gradient-to-br from-muted to-muted/50 rounded-lg overflow-hidden" style={{ height: '400px' }}>
            {/* Placeholder background */}
            <div className="absolute inset-0 flex items-center justify-center text-muted-foreground">
              <div className="text-center">
                <Image className="h-12 w-12 mx-auto mb-2 opacity-50" />
                <p className="text-sm">Sample: {sampleImages.find(i => i.id === selectedImage)?.name}</p>
              </div>
            </div>

            {/* Detection boxes */}
            <svg className="absolute inset-0 w-full h-full" viewBox="0 0 600 400">
              {detections.map((box, i) => {
                const color = classColors[box.classId] || '#888';
                return (
                  <g key={i}>
                    <rect
                      x={box.x}
                      y={box.y}
                      width={box.width}
                      height={box.height}
                      fill={`${color}20`}
                      stroke={color}
                      strokeWidth={2}
                    />
                    <rect
                      x={box.x}
                      y={box.y - 20}
                      width={Math.max(box.className.length * 8 + 40, 80)}
                      height={20}
                      fill={color}
                    />
                    <text
                      x={box.x + 4}
                      y={box.y - 6}
                      fontSize={12}
                      fill="white"
                      fontWeight="bold"
                    >
                      {box.className} {(box.confidence * 100).toFixed(0)}%
                    </text>
                  </g>
                );
              })}
            </svg>

            {/* Empty state */}
            {detections.length === 0 && !isDetecting && (
              <div className="absolute inset-0 flex items-center justify-center">
                <p className="text-muted-foreground text-sm">Click "Run Detection" to see results</p>
              </div>
            )}

            {/* Loading state */}
            {isDetecting && (
              <div className="absolute inset-0 flex items-center justify-center bg-background/50 backdrop-blur-sm">
                <div className="text-center">
                  <RefreshCw className="h-8 w-8 mx-auto animate-spin text-primary" />
                  <p className="mt-2">Running YOLO{selectedVersion}...</p>
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Detection List */}
      {detections.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Detected Objects</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-3">
              {detections.map((box, i) => {
                const color = classColors[box.classId] || '#888';
                return (
                  <div
                    key={i}
                    className="p-3 rounded-lg border"
                    style={{ borderColor: color }}
                  >
                    <div className="flex items-center gap-2">
                      <div
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: color }}
                      />
                      <span className="font-medium capitalize">{box.className}</span>
                    </div>
                    <div className="text-sm text-muted-foreground mt-1">
                      Confidence: {(box.confidence * 100).toFixed(1)}%
                    </div>
                    <div className="text-xs text-muted-foreground">
                      Box: ({box.x}, {box.y}) {box.width}x{box.height}
                    </div>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Tips */}
      <Card className="bg-primary/5 border-primary/20">
        <CardContent className="pt-6">
          <div className="grid md:grid-cols-3 gap-4 text-sm">
            <div>
              <h4 className="font-semibold">Confidence Threshold</h4>
              <p className="text-muted-foreground">Higher values = fewer but more confident detections</p>
            </div>
            <div>
              <h4 className="font-semibold">IoU Threshold</h4>
              <p className="text-muted-foreground">Lower values = more aggressive NMS, fewer overlapping boxes</p>
            </div>
            <div>
              <h4 className="font-semibold">Model Version</h4>
              <p className="text-muted-foreground">Newer versions are faster and more accurate</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
