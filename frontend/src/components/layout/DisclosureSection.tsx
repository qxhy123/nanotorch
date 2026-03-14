import { useEffect, useMemo, useRef } from 'react';
import type { ReactNode } from 'react';
import { Lock, Sparkles } from 'lucide-react';

import { DISCLOSURE_LEVEL_DESCRIPTIONS, useDisclosureLevel } from '../providers';
import { useTutorial } from '../tutorial';
import type { DisclosureLevel } from '../../types/transformer';
import { cn } from '../../lib/utils';
import { Button } from '../ui/button';
import { Card, CardContent } from '../ui/card';

function DisclosureLevelChip({
  level,
  active,
}: {
  level: DisclosureLevel;
  active: boolean;
}) {
  const descriptor = DISCLOSURE_LEVEL_DESCRIPTIONS[level];

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs font-medium',
        active
          ? 'border-primary/20 bg-primary/10 text-primary'
          : 'border-border bg-background text-muted-foreground'
      )}
    >
      <span>{descriptor.icon}</span>
      <span>{descriptor.title}</span>
    </span>
  );
}

export function DisclosureLevelSummary({
  title,
  className,
}: {
  title: string;
  className?: string;
}) {
  const { level, config } = useDisclosureLevel();
  const descriptor = DISCLOSURE_LEVEL_DESCRIPTIONS[level];

  const capabilities = [
    { label: 'Core visuals', active: true },
    { label: 'Math explanations', active: config.showMath },
    { label: 'Implementation details', active: config.showImplementation },
    { label: 'Full parameter reference', active: config.showAllParameters },
  ];

  return (
    <Card className={cn('border-primary/10 bg-gradient-to-r from-primary/5 via-background to-background', className)}>
      <CardContent className="flex flex-col gap-4 p-4">
        <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-primary" />
              <p className="text-sm font-semibold">{title}</p>
            </div>
            <p className="text-sm text-muted-foreground">{descriptor.description}</p>
          </div>
          <DisclosureLevelChip level={level} active />
        </div>

        <div className="flex flex-wrap gap-2">
          {capabilities.map((item) => (
            <span
              key={item.label}
              className={cn(
                'inline-flex items-center gap-2 rounded-full px-2.5 py-1 text-xs font-medium',
                item.active
                  ? 'bg-emerald-100 text-emerald-800'
                  : 'bg-slate-100 text-slate-600'
              )}
            >
              <span>{item.active ? 'Unlocked' : 'Locked'}</span>
              <span>{item.label}</span>
            </span>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

export function DisclosureSection({
  id,
  level,
  title,
  description,
  children,
  className,
  focusSelectors = [],
}: {
  id?: string;
  level: DisclosureLevel;
  title: string;
  description: string;
  children: ReactNode;
  className?: string;
  focusSelectors?: string[];
}) {
  const { canShow, setLevel } = useDisclosureLevel();
  const { state: tutorialState, currentStepData } = useTutorial();
  const containerRef = useRef<HTMLDivElement | null>(null);
  const visible = canShow(level);
  const descriptor = DISCLOSURE_LEVEL_DESCRIPTIONS[level];
  const watchedSelectors = useMemo(() => {
    const selectors: string[] = [...focusSelectors];
    if (id) {
      selectors.push(`#${id}`, `.${id}`);
    }
    return selectors;
  }, [focusSelectors, id]);

  const selectorMatched = useMemo(() => {
    if (!tutorialState.isTutorialActive || !currentStepData) {
      return false;
    }

    const targetSelectors = [
      currentStepData.target,
      ...(currentStepData.highlightElements ?? []),
    ].filter(Boolean) as string[];

    if (watchedSelectors.some((selector) => targetSelectors.includes(selector))) {
      return true;
    }

    return false;
  }, [currentStepData, tutorialState.isTutorialActive, watchedSelectors]);

  const isTutorialFocused = selectorMatched;

  useEffect(() => {
    if (!visible || !isTutorialFocused || !containerRef.current) {
      return;
    }

    const node = containerRef.current;
    const frameId = requestAnimationFrame(() => {
      node.scrollIntoView({
        behavior: 'smooth',
        block: 'center',
        inline: 'nearest',
      });
    });

    return () => cancelAnimationFrame(frameId);
  }, [isTutorialFocused, visible]);

  if (visible) {
    return (
      <div
        ref={containerRef}
        id={id}
        className={cn(
          className,
          isTutorialFocused && 'scroll-mt-24 rounded-xl ring-2 ring-primary/50 ring-offset-4 ring-offset-background transition-shadow'
        )}
      >
        {isTutorialFocused && (
          <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-primary/20 bg-primary/10 px-3 py-1 text-xs font-medium text-primary">
            <Sparkles className="h-3.5 w-3.5" />
            Tutorial focus
          </div>
        )}
        {children}
      </div>
    );
  }

  return (
    <Card
      ref={containerRef}
      id={id}
      className={cn(
        'border-dashed border-border/80 bg-muted/30 shadow-none',
        isTutorialFocused && 'scroll-mt-24 ring-2 ring-primary/50 ring-offset-4 ring-offset-background',
        className
      )}
    >
      <CardContent className="flex flex-col gap-4 p-5">
        <div className="flex items-start gap-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full border bg-background text-muted-foreground">
            <Lock className="h-4 w-4" />
          </div>
          <div className="space-y-1">
            <p className="text-sm font-semibold">{title}</p>
            <p className="text-sm text-muted-foreground">{description}</p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          <DisclosureLevelChip level={level} active={false} />
          <span>Raise the disclosure level to unlock this section.</span>
          {isTutorialFocused && <span className="text-primary">Tutorial is pointing here now.</span>}
        </div>

        <div>
          <Button variant="outline" size="sm" onClick={() => setLevel(level)}>
            Switch to {descriptor.title}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
