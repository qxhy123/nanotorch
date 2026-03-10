declare module 'react-katex' {
  import { Component, CSSProperties, ReactNode } from 'react';

  export interface KatexPropsBase {
    errorColor?: string;
    renderError?: (error: Error) => ReactNode;
    style?: CSSProperties;
  }

  export interface KatexPropsWithMath extends KatexPropsBase {
    math: string;
    children?: never;
  }

  export interface KatexPropsWithChildren extends KatexPropsBase {
    math?: never;
    children: string;
  }

  export type KatexProps = KatexPropsWithMath | KatexPropsWithChildren;

  export class InlineMath extends Component<KatexProps> {}
  export class BlockMath extends Component<KatexProps> {}
}
