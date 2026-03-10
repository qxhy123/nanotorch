/**
 * YOLO Visualization Types
 */

// YOLO version types
export type YOLOVersion = 'v1' | 'v2' | 'v3' | 'v4' | 'v5' | 'v6' | 'v7' | 'v8';

// Bounding box representation
export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  classId: number;
  className: string;
}

// Backbone layer types
export type LayerType = 'conv' | 'csp' | 'sppf' | 'c2f' | 'pool' | 'residual';

export interface BackboneLayer {
  id: string;
  name: string;
  type: LayerType;
  inputChannels: number;
  outputChannels: number;
  kernelSize: number;
  stride: number;
  inputSize: number;
  outputSize: number;
}

// Neck types
export type NeckType = 'fpn' | 'panet' | 'bifpn';

export interface NeckFeature {
  level: string; // P3, P4, P5, etc.
  size: number;
  channels: number;
}

// Detection head types
export type HeadType = 'anchor-based' | 'anchor-free';

export interface DetectionHead {
  scale: number; // 8, 16, 32
  gridSize: number;
  anchors: number;
  classes: number;
}

// Anchor box representation
export interface Anchor {
  width: number;
  height: number;
  scale: number;
}

// NMS step for animation
export interface NMSStep {
  step: number;
  description: string;
  boxes: BoundingBox[];
  selectedIndices: number[];
  suppressedIndices: number[];
  currentBox?: number;
  comparedBox?: number;
  iou?: number;
}

// Loss types
export type LossType = 'iou' | 'giou' | 'diou' | 'ciou';

export interface LossValue {
  type: LossType;
  value: number;
  formula: string;
}

// YOLO version information
export interface YOLOVersionInfo {
  version: YOLOVersion;
  year: number;
  backbone: string;
  mAP: number;
  fps: number;
  params: string;
  anchorType: HeadType;
  features: string[];
}

// Detection result
export interface DetectionResult {
  boxes: BoundingBox[];
  processingTime: number;
  modelVersion: YOLOVersion;
}

// App state
export interface AppState {
  currentRoute: string;
  theme: 'light' | 'dark';

  // NMS state
  nmsStep: number;
  isPlaying: boolean;
  confidenceThreshold: number;
  iouThreshold: number;

  // Detection state
  selectedVersion: YOLOVersion;
  detectionResults: BoundingBox[];

  // Backbone state
  selectedBackbone: 'cspdarknet53' | 'darknet53';

  // Neck state
  selectedNeck: NeckType;

  // Head state
  selectedHead: HeadType;
}

// Grid cell for anchor visualization
export interface GridCell {
  row: number;
  col: number;
  anchors: Anchor[];
  hasObject: boolean;
  responsibleAnchor?: number;
}
