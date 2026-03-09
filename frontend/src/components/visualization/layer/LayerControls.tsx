/**
 * LayerControls Component
 *
 * Control panel for layer visualization with display options.
 */

import React from 'react';
import { Card, CardContent } from '../../ui/card';
import { Label } from '../../ui/label';
import { Button } from '../../ui/button';
import { Settings, Eye, BarChart3 } from 'lucide-react';

interface LayerControlsProps {
  showValues: boolean;
  showShapes: boolean;
  showStatistics: boolean;
  onToggleValues: () => void;
  onToggleShapes: () => void;
  onToggleStatistics: () => void;
}

export const LayerControls: React.FC<LayerControlsProps> = ({
  showValues,
  showShapes,
  showStatistics,
  onToggleValues,
  onToggleShapes,
  onToggleStatistics,
}) => {
  return (
    <Card>
      <CardContent className="pt-6">
        <div className="flex items-center gap-2 mb-4">
          <Settings className="h-4 w-4 text-gray-500" />
          <h3 className="font-medium text-sm">Display Options</h3>
        </div>

        <div className="space-y-4">
          {/* Show Values */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Eye className="h-4 w-4 text-gray-500" />
              <Label className="text-sm">Show Values</Label>
            </div>
            <Button
              variant={showValues ? "default" : "outline"}
              size="sm"
              onClick={onToggleValues}
            >
              {showValues ? 'On' : 'Off'}
            </Button>
          </div>

          {/* Show Shapes */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4 text-gray-500" />
              <Label className="text-sm">Show Shapes</Label>
            </div>
            <Button
              variant={showShapes ? "default" : "outline"}
              size="sm"
              onClick={onToggleShapes}
            >
              {showShapes ? 'On' : 'Off'}
            </Button>
          </div>

          {/* Show Statistics */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4 text-gray-500" />
              <Label className="text-sm">Show Statistics</Label>
            </div>
            <Button
              variant={showStatistics ? "default" : "outline"}
              size="sm"
              onClick={onToggleStatistics}
            >
              {showStatistics ? 'On' : 'Off'}
            </Button>
          </div>
        </div>

        <div className="mt-4 pt-4 border-t text-xs text-gray-500">
          Customize what information is displayed in the visualization
        </div>
      </CardContent>
    </Card>
  );
};
