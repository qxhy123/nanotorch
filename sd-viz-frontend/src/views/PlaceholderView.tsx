/**
 * Placeholder view for unimplemented modules
 */

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';

interface PlaceholderViewProps {
  title: string;
  description: string;
  icon: React.ReactNode;
}

export const PlaceholderView: React.FC<PlaceholderViewProps> = ({
  title,
  description,
  icon,
}) => {
  
  return (
    <div className="space-y-6">
      <div className="text-center py-12">
        <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-primary/10 mb-4">
          {icon}
        </div>
        <h1 className="text-3xl font-bold mb-2">
          {title}
        </h1>
        <p className="text-muted-foreground max-w-md mx-auto">
          {description}
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Coming Soon</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            This visualization is under development. Check back soon!
          </p>
        </CardContent>
      </Card>
    </div>
  );
};
