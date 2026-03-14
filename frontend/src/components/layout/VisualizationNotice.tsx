import type { ReactNode } from 'react';
import { Info } from 'lucide-react';

export function VisualizationNotice({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-950">
      <div className="flex items-start gap-3">
        <Info className="mt-0.5 h-4 w-4 shrink-0 text-amber-700" />
        <div>
          <p className="font-medium">{title}</p>
          <p className="mt-1 text-amber-800">{children}</p>
        </div>
      </div>
    </div>
  );
}
