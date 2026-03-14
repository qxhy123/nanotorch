import type { ReactNode } from 'react';

import { Latex } from '../ui/Latex';

function renderInline(text: string): ReactNode[] {
  const parts: ReactNode[] = [];
  const pattern = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let lastIndex = 0;

  for (const match of text.matchAll(pattern)) {
    const [token] = match;
    const index = match.index ?? 0;

    if (index > lastIndex) {
      parts.push(text.slice(lastIndex, index));
    }

    if (token.startsWith('**') && token.endsWith('**')) {
      parts.push(
        <strong key={`${index}-bold`} className="font-semibold text-foreground">
          {token.slice(2, -2)}
        </strong>
      );
    } else if (token.startsWith('`') && token.endsWith('`')) {
      parts.push(
        <code
          key={`${index}-code`}
          className="rounded bg-muted px-1.5 py-0.5 font-mono text-[0.9em] text-foreground"
        >
          {token.slice(1, -1)}
        </code>
      );
    }

    lastIndex = index + token.length;
  }

  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return parts;
}

function extractDisplayMath(line: string): string | null {
  const trimmed = line.trim();
  if (trimmed.startsWith('$$') && trimmed.endsWith('$$') && trimmed.length > 4) {
    return trimmed.slice(2, -2).trim();
  }
  return null;
}

function renderParagraphBlock(block: string, key: string) {
  const lines = block
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);

  const segments: Array<{ type: 'paragraph' | 'math'; value: string }> = [];
  let paragraphBuffer: string[] = [];

  const flushParagraph = () => {
    if (paragraphBuffer.length === 0) {
      return;
    }
    segments.push({
      type: 'paragraph',
      value: paragraphBuffer.join(' '),
    });
    paragraphBuffer = [];
  };

  lines.forEach((line) => {
    const formula = extractDisplayMath(line);
    if (formula) {
      flushParagraph();
      segments.push({ type: 'math', value: formula });
      return;
    }
    paragraphBuffer.push(line);
  });

  flushParagraph();

  return (
    <div key={key} className="space-y-3">
      {segments.map((segment, index) => (
        segment.type === 'math' ? (
          <div key={`${key}-math-${index}`} className="rounded-lg border bg-muted/30 px-3 py-2">
            <Latex display>{segment.value}</Latex>
          </div>
        ) : (
          <p key={`${key}-paragraph-${index}`} className="leading-6 text-muted-foreground">
            {renderInline(segment.value)}
          </p>
        )
      ))}
    </div>
  );
}

export function TutorialRichContent({ content }: { content: string }) {
  const blocks = content
    .trim()
    .split(/\n\s*\n/)
    .map((block) => block.trim())
    .filter(Boolean);

  return (
    <div className="space-y-4">
      {blocks.map((block, blockIndex) => {
        const lines = block
          .split('\n')
          .map((line) => line.trim())
          .filter(Boolean);

        const unorderedItems = lines
          .map((line) => line.match(/^(?:[-*]|\u2022)\s+(.*)$/)?.[1] ?? null);
        const orderedItems = lines
          .map((line) => line.match(/^\d+\.\s+(.*)$/)?.[1] ?? null);

        if (lines.length > 0 && unorderedItems.every(Boolean)) {
          return (
            <ul key={`block-${blockIndex}`} className="ml-5 list-disc space-y-2 text-muted-foreground">
              {unorderedItems.map((item, itemIndex) => (
                <li key={`block-${blockIndex}-item-${itemIndex}`}>{renderInline(item ?? '')}</li>
              ))}
            </ul>
          );
        }

        if (lines.length > 0 && orderedItems.every(Boolean)) {
          return (
            <ol key={`block-${blockIndex}`} className="ml-5 list-decimal space-y-2 text-muted-foreground">
              {orderedItems.map((item, itemIndex) => (
                <li key={`block-${blockIndex}-item-${itemIndex}`}>{renderInline(item ?? '')}</li>
              ))}
            </ol>
          );
        }

        return renderParagraphBlock(block, `block-${blockIndex}`);
      })}
    </div>
  );
}
