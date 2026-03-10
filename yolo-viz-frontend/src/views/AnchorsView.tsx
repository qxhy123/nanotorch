/**
 * Anchors View - Interactive anchor box and grid visualization
 */

import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Slider } from '../components/ui/slider';
import { Grid3X3, Target, Info, Eye } from 'lucide-react';
import { BlockMath, InlineMath } from 'react-katex';
import 'katex/dist/katex.min.css';

interface AnchorSet {
  scale: number;
  gridSize: number;
  anchors: Array<{ width: number; height: number }>;
  color: string;
}

const anchorSets: AnchorSet[] = [
  {
    scale: 8,
    gridSize: 80,
    anchors: [
      { width: 10, height: 13 },
      { width: 16, height: 30 },
      { width: 33, height: 23 },
    ],
    color: 'rgb(59, 130, 246)', // blue
  },
  {
    scale: 16,
    gridSize: 40,
    anchors: [
      { width: 30, height: 61 },
      { width: 62, height: 45 },
      { width: 59, height: 119 },
    ],
    color: 'rgb(34, 197, 94)', // green
  },
  {
    scale: 32,
    gridSize: 20,
    anchors: [
      { width: 116, height: 90 },
      { width: 156, height: 198 },
      { width: 373, height: 326 },
    ],
    color: 'rgb(239, 68, 68)', // red
  },
];

const GRID_DISPLAY_SIZE = 10; // Display 10x10 grid for visualization

export const AnchorsView: React.FC = () => {
  const [selectedScale, setSelectedScale] = useState<number>(16);
  const [selectedCell, setSelectedCell] = useState<{ row: number; col: number } | null>(null);
  const [iouThreshold, setIouThreshold] = useState(0.5);
  const [showAnchors, setShowAnchors] = useState(true);

  const currentAnchorSet = anchorSets.find(a => a.scale === selectedScale) || anchorSets[1];

  // Sample ground truth box for demonstration
  const groundTruthBox = { x: 3.5, y: 4.2, w: 2.8, h: 1.9 };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Grid3X3 className="h-8 w-8 text-primary" />
        <div>
          <h1 className="text-3xl font-bold">Anchor Boxes & Grid</h1>
          <p className="text-muted-foreground">Understanding anchor-based detection</p>
        </div>
      </div>

      {/* Scale Selector */}
      <Card>
        <CardHeader>
          <CardTitle>Select Detection Scale</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2 mb-4">
            {anchorSets.map((set) => (
              <Button
                key={set.scale}
                variant={selectedScale === set.scale ? 'default' : 'outline'}
                onClick={() => setSelectedScale(set.scale)}
                style={{
                  borderColor: set.color,
                  backgroundColor: selectedScale === set.scale ? set.color : undefined,
                }}
              >
                Stride {set.scale} ({set.gridSize}×{set.gridSize})
              </Button>
            ))}
          </div>
          <div className="flex items-center gap-4">
            <Button
              variant={showAnchors ? 'default' : 'outline'}
              size="sm"
              onClick={() => setShowAnchors(!showAnchors)}
            >
              <Eye className="h-4 w-4 mr-2" />
              {showAnchors ? 'Hide' : 'Show'} Anchors
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Main Content */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* Grid Visualization */}
        <Card>
          <CardHeader>
            <CardTitle>Grid Cells & Anchors</CardTitle>
            <CardDescription>Click on a cell to see its anchors</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="relative aspect-square bg-muted rounded-lg overflow-hidden">
              {/* Grid */}
              <svg className="w-full h-full" viewBox="0 0 100 100">
                {/* Grid lines */}
                {Array.from({ length: GRID_DISPLAY_SIZE + 1 }).map((_, i) => (
                  <React.Fragment key={i}>
                    <line
                      x1={i * (100 / GRID_DISPLAY_SIZE)}
                      y1={0}
                      x2={i * (100 / GRID_DISPLAY_SIZE)}
                      y2={100}
                      stroke="currentColor"
                      strokeOpacity={0.2}
                      strokeWidth={0.3}
                    />
                    <line
                      x1={0}
                      y1={i * (100 / GRID_DISPLAY_SIZE)}
                      x2={100}
                      y2={i * (100 / GRID_DISPLAY_SIZE)}
                      stroke="currentColor"
                      strokeOpacity={0.2}
                      strokeWidth={0.3}
                    />
                  </React.Fragment>
                ))}

                {/* Ground Truth Box */}
                <rect
                  x={groundTruthBox.x * (100 / GRID_DISPLAY_SIZE)}
                  y={groundTruthBox.y * (100 / GRID_DISPLAY_SIZE)}
                  width={groundTruthBox.w * (100 / GRID_DISPLAY_SIZE)}
                  height={groundTruthBox.h * (100 / GRID_DISPLAY_SIZE)}
                  fill="rgba(255, 215, 0, 0.3)"
                  stroke="gold"
                  strokeWidth={0.5}
                />
                <text
                  x={groundTruthBox.x * (100 / GRID_DISPLAY_SIZE) + 1}
                  y={groundTruthBox.y * (100 / GRID_DISPLAY_SIZE) + 3}
                  fontSize={3}
                  fill="gold"
                >
                  GT Box
                </text>

                {/* Clickable cells */}
                {Array.from({ length: GRID_DISPLAY_SIZE }).map((_, row) =>
                  Array.from({ length: GRID_DISPLAY_SIZE }).map((_, col) => (
                    <rect
                      key={`${row}-${col}`}
                      x={col * (100 / GRID_DISPLAY_SIZE)}
                      y={row * (100 / GRID_DISPLAY_SIZE)}
                      width={100 / GRID_DISPLAY_SIZE}
                      height={100 / GRID_DISPLAY_SIZE}
                      fill={selectedCell?.row === row && selectedCell?.col === col ? 'rgba(139, 92, 246, 0.3)' : 'transparent'}
                      stroke={selectedCell?.row === row && selectedCell?.col === col ? 'rgb(139, 92, 246)' : 'transparent'}
                      strokeWidth={0.3}
                      className="cursor-pointer hover:fill-primary/10"
                      onClick={() => setSelectedCell({ row, col })}
                    />
                  ))
                )}

                {/* Anchors for selected cell */}
                {showAnchors && selectedCell && currentAnchorSet.anchors.map((anchor, i) => {
                  const cellSize = 100 / GRID_DISPLAY_SIZE;
                  const centerX = (selectedCell.col + 0.5) * cellSize;
                  const centerY = (selectedCell.row + 0.5) * cellSize;
                  // Scale anchors for visualization (divide by grid actual size, multiply by display)
                  const anchorW = (anchor.width / currentAnchorSet.gridSize) * 100;
                  const anchorH = (anchor.height / currentAnchorSet.gridSize) * 100;

                  return (
                    <rect
                      key={i}
                      x={centerX - anchorW / 2}
                      y={centerY - anchorH / 2}
                      width={anchorW}
                      height={anchorH}
                      fill="transparent"
                      stroke={currentAnchorSet.color}
                      strokeWidth={0.4}
                      strokeDasharray={i === 0 ? '0' : i === 1 ? '2,1' : '1,1'}
                      opacity={0.8}
                    />
                  );
                })}

                {/* Cell center point */}
                {selectedCell && (
                  <circle
                    cx={(selectedCell.col + 0.5) * (100 / GRID_DISPLAY_SIZE)}
                    cy={(selectedCell.row + 0.5) * (100 / GRID_DISPLAY_SIZE)}
                    r={1}
                    fill={currentAnchorSet.color}
                  />
                )}
              </svg>

              {/* Legend */}
              <div className="absolute bottom-2 right-2 bg-background/90 backdrop-blur p-2 rounded text-xs space-y-1">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-yellow-400/30 border border-yellow-400" />
                  <span>Ground Truth</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3" style={{ border: `2px solid ${currentAnchorSet.color}` }} />
                  <span>Anchor</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Anchor Details & IoU */}
        <div className="space-y-6">
          {/* Current Scale Anchors */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="h-5 w-5" />
                Anchor Sizes (Stride {selectedScale})
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {currentAnchorSet.anchors.map((anchor, i) => (
                  <div key={i} className="flex items-center gap-4 p-3 bg-muted rounded-lg">
                    <div
                      className="w-12 h-12 border-2 flex items-center justify-center text-xs"
                      style={{
                        borderColor: currentAnchorSet.color,
                        width: `${Math.min(48, anchor.width / 4)}px`,
                        height: `${Math.min(48, anchor.height / 4)}px`,
                      }}
                    >
                      {i + 1}
                    </div>
                    <div>
                      <div className="font-medium">Anchor {i + 1}</div>
                      <div className="text-sm text-muted-foreground">
                        {anchor.width} × {anchor.height} pixels
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Aspect ratio: {(anchor.width / anchor.height).toFixed(2)}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* IoU Threshold */}
          <Card>
            <CardHeader>
              <CardTitle>IoU Matching Threshold</CardTitle>
              <CardDescription>Minimum IoU to assign anchor to GT box</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Slider
                value={iouThreshold}
                onValueChange={setIouThreshold}
                min={0}
                max={1}
                step={0.05}
                label="IoU Threshold"
              />
              <div className="p-4 bg-muted rounded-lg">
                <div className="text-sm">
                  <p><strong>Positive sample:</strong> IoU ≥ {iouThreshold.toFixed(2)}</p>
                  <p><strong>Negative sample:</strong> IoU &lt; {(iouThreshold * 0.6).toFixed(2)}</p>
                  <p><strong>Ignored:</strong> {(iouThreshold * 0.6).toFixed(2)} ≤ IoU &lt; {iouThreshold.toFixed(2)}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Selected Cell Info */}
          {selectedCell && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Info className="h-5 w-5" />
                  Cell ({selectedCell.row}, {selectedCell.col})
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-sm space-y-2">
                  <p>Center: ({((selectedCell.col + 0.5) / GRID_DISPLAY_SIZE * 640).toFixed(0)}, {((selectedCell.row + 0.5) / GRID_DISPLAY_SIZE * 640).toFixed(0)}) px</p>
                  <p>Grid offset: (<InlineMath math={`c_x=${selectedCell.col}`} />, <InlineMath math={`c_y=${selectedCell.row}`} />)</p>
                  <p>Responsible for detecting objects whose center falls in this cell.</p>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      {/* Anchor Assignment Formula */}
      <Card>
        <CardHeader>
          <CardTitle>Anchor Assignment</CardTitle>
          <CardDescription>How anchors are matched to ground truth boxes</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="bg-muted p-4 rounded-lg overflow-x-auto">
            <BlockMath math="\text{IoU}(A, G) = \frac{|A \cap G|}{|A \cup G|} = \frac{\text{Intersection Area}}{\text{Union Area}}" />
          </div>
          <div className="grid md:grid-cols-3 gap-4 text-sm">
            <div className="p-4 bg-green-500/10 rounded-lg border border-green-500/30">
              <h4 className="font-semibold text-green-600 dark:text-green-400">Positive Sample</h4>
              <p className="mt-1">IoU ≥ threshold</p>
              <p className="text-muted-foreground">Anchor learns to predict this GT box</p>
            </div>
            <div className="p-4 bg-red-500/10 rounded-lg border border-red-500/30">
              <h4 className="font-semibold text-red-600 dark:text-red-400">Negative Sample</h4>
              <p className="mt-1">IoU &lt; low threshold</p>
              <p className="text-muted-foreground">Anchor learns "no object" prediction</p>
            </div>
            <div className="p-4 bg-yellow-500/10 rounded-lg border border-yellow-500/30">
              <h4 className="font-semibold text-yellow-600 dark:text-yellow-400">Ignored</h4>
              <p className="mt-1">IoU in between</p>
              <p className="text-muted-foreground">Not used for training</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
