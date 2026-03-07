import React, { useEffect, useRef } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

interface LatexProps {
  children: string | React.ReactNode;
  display?: boolean;
  className?: string;
}

export const Latex: React.FC<LatexProps> = ({ children, display = false, className = '' }) => {
  const containerRef = useRef<HTMLSpanElement>(null);
  const latex = typeof children === 'string' ? children : '';

  useEffect(() => {
    if (containerRef.current && latex) {
      try {
        katex.render(latex, containerRef.current, {
          displayMode: display,
          throwOnError: false,
        });
      } catch (error) {
        console.error('KaTeX render error:', error);
        if (containerRef.current) {
          containerRef.current.textContent = latex;
        }
      }
    }
  }, [latex, display]);

  return (
    <span ref={containerRef} className={className} style={display ? { display: 'block' } : {}} />
  );
};
