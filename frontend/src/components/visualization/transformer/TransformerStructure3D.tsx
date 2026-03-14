import React, { memo, useMemo, useRef, useState, useCallback, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../ui/card';
import { Badge } from '../../ui/badge';
import { Button } from '../../ui/button';
import { useTransformerStore } from '../../../stores/transformerStore';
import { Box, Info, RefreshCw, Camera, Home, RotateCw, ArrowUpDown, Move3D } from 'lucide-react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import type { ThreeEvent } from '@react-three/fiber';
import { hierarchy } from 'd3-hierarchy';
import type { HierarchyNode, HierarchyPointNode } from 'd3-hierarchy';
import { BoxGeometry, BufferAttribute, Vector3 } from 'three';
import type { TransformerConfig } from '../../../types/transformer';
import { OrbitControls as OrbitControlsImpl } from 'three/examples/jsm/controls/OrbitControls.js';

// Camera preset positions
const CAMERA_PRESETS: Record<string, { position: [number, number, number]; target: [number, number, number]; name: string }> = {
  default: { position: [20, 15, 20] as [number, number, number], target: [0, 5, 0] as [number, number, number], name: 'Default' },
  front: { position: [0, 10, 35] as [number, number, number], target: [0, 5, 0] as [number, number, number], name: 'Front' },
  side: { position: [35, 10, 0] as [number, number, number], target: [0, 5, 0] as [number, number, number], name: 'Side' },
  top: { position: [0, 40, 0.1] as [number, number, number], target: [0, 0, 0] as [number, number, number], name: 'Top' },
  isometric: { position: [25, 25, 25] as [number, number, number], target: [0, 5, 0] as [number, number, number], name: 'Isometric' },
  diagonal: { position: [-20, 15, 20] as [number, number, number], target: [0, 5, 0] as [number, number, number], name: 'Diagonal' },
};

interface TreeNode {
  id: string;
  name: string;
  type: string;
  description: string;
  children?: TreeNode[];
  config?: Record<string, boolean | number | string | undefined>;
}

interface TreeNodeWithPosition extends HierarchyPointNode<TreeNode> {
  x: number;
  y: number;
  z: number;
}

// Natural color palette - inspired by nature and scientific visualization
const COLOR_PALETTE: Record<string, { primary: string; secondary: string; glow: string }> = {
  root: { primary: '#6366f1', secondary: '#4338ca', glow: '#818cf8' },        // Indigo
  input: { primary: '#64748b', secondary: '#475569', glow: '#94a3b8' },        // Slate
  embedding: { primary: '#0ea5e9', secondary: '#0284c7', glow: '#38bdf8' },     // Sky Blue
  positional: { primary: '#8b5cf6', secondary: '#7c3aed', glow: '#a78bfa' },    // Violet
  encoder: { primary: '#10b981', secondary: '#059669', glow: '#34d399' },       // Emerald
  encoder_layer: { primary: '#14b8a6', secondary: '#0d9488', glow: '#2dd4bf' }, // Teal
  decoder: { primary: '#f59e0b', secondary: '#d97706', glow: '#fbbf24' },       // Amber
  decoder_layer: { primary: '#f97316', secondary: '#ea580c', glow: '#fb923c' }, // Orange
  norm: { primary: '#ec4899', secondary: '#db2777', glow: '#f472b6' },          // Pink
  attention: { primary: '#22c55e', secondary: '#16a34a', glow: '#4ade80' },      // Green
  feedforward: { primary: '#a855f7', secondary: '#9333ea', glow: '#c084fc' },    // Purple
  output: { primary: '#ef4444', secondary: '#dc2626', glow: '#f87171' },         // Red
};

const getNodeColor = (type: string) => COLOR_PALETTE[type] || COLOR_PALETTE.input;

const getNodeDimensions = (type: string): [number, number, number] => {
  // Return [width, height, depth] for box
  switch (type) {
    case 'root':
      return [3, 1, 3];
    case 'input':
    case 'output':
      return [1.5, 0.6, 1.5];
    case 'encoder':
    case 'decoder':
      return [2, 0.8, 2];
    case 'encoder_layer':
    case 'decoder_layer':
      return [1.8, 0.7, 1.8];
    case 'embedding':
      return [1.4, 0.6, 1.4];
    case 'positional':
      return [1.2, 0.6, 1.2];
    case 'attention':
      return [1.3, 0.5, 1.3];
    case 'feedforward':
      return [1.2, 0.5, 1.2];
    case 'norm':
      return [0.8, 0.4, 0.8];
    default:
      return [1, 0.5, 1];
  }
};

const BOX_GEOMETRIES: Record<string, any> = Object.fromEntries(
  Object.keys(COLOR_PALETTE).map((type) => [type, new BoxGeometry(...getNodeDimensions(type))])
);
const FALLBACK_BOX_GEOMETRY = new BoxGeometry(...getNodeDimensions('input'));

const getBoxGeometry = (type: string) => BOX_GEOMETRIES[type] || FALLBACK_BOX_GEOMETRY;

const getTransformerTree = (config: TransformerConfig): TreeNode => {
  const { d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, activation, vocab_size } = config;

  const tree: TreeNode = {
    id: 'root',
    name: 'Transformer',
    type: 'root',
    description: 'Complete Transformer Architecture',
    children: [
      {
        id: 'input',
        name: 'Input',
        type: 'input',
        description: 'Token indices from tokenizer',
      },
      {
        id: 'embedding-stage',
        name: 'Embedding Stage',
        type: 'embedding',
        description: 'Token and Positional Encoding',
        children: [
          {
            id: 'token-embedding',
            name: 'Token Embedding',
            type: 'embedding',
            description: 'Convert tokens to dense vectors',
            config: { vocab_size, d_model },
          },
          {
            id: 'positional-encoding',
            name: 'Positional Encoding',
            type: 'positional',
            description: 'Add position information',
            config: { d_model },
          },
        ],
      },
      {
        id: 'encoder',
        name: 'Encoder Stack',
        type: 'encoder',
        description: `${num_encoder_layers} encoder layers`,
        config: { layers: num_encoder_layers, d_model, nhead, dim_feedforward },
        children: Array.from({ length: Math.min(num_encoder_layers, 6) }, (_, i) => ({
          id: `encoder-layer-${i}`,
          name: `Encoder L${i + 1}`,
          type: 'encoder_layer',
          description: `Self-attention + FFN`,
          config: { layer: i + 1, d_model, nhead, dim_feedforward, activation },
          children: [
            {
              id: `encoder-${i}-norm1`,
              name: 'LayerNorm',
              type: 'norm',
              description: 'Pre-attention norm',
            },
            {
              id: `encoder-${i}-attention`,
              name: 'Multi-Head Attn',
              type: 'attention',
              description: `${nhead} heads`,
              config: { nhead, head_dim: Math.floor(d_model / nhead) },
            },
            {
              id: `encoder-${i}-norm2`,
              name: 'LayerNorm',
              type: 'norm',
              description: 'Post-attention norm',
            },
            {
              id: `encoder-${i}-ffn`,
              name: 'FeedForward',
              type: 'feedforward',
              description: `FFN(${dim_feedforward})`,
              config: { hidden: dim_feedforward, activation },
            },
          ],
        })),
      },
    ],
  };

  // Add decoder if present
  if (num_decoder_layers > 0) {
    tree.children!.push({
      id: 'decoder',
      name: 'Decoder Stack',
      type: 'decoder',
      description: `${num_decoder_layers} decoder layers`,
      config: { layers: num_decoder_layers, d_model, nhead, dim_feedforward },
      children: Array.from({ length: Math.min(num_decoder_layers, 6) }, (_, i) => ({
        id: `decoder-layer-${i}`,
        name: `Decoder L${i + 1}`,
        type: 'decoder_layer',
        description: 'Masked + Cross + FFN',
        config: { layer: i + 1, d_model, nhead, dim_feedforward, activation },
        children: [
          {
            id: `decoder-${i}-norm1`,
            name: 'LayerNorm',
            type: 'norm',
            description: 'Pre-norm',
          },
          {
            id: `decoder-${i}-masked-attn`,
            name: 'Masked Self-Attn',
            type: 'attention',
            description: 'Causal mask',
            config: { nhead, masked: true },
          },
          {
            id: `decoder-${i}-cross-attn`,
            name: 'Cross-Attention',
            type: 'attention',
            description: 'Attend encoder',
            config: { nhead, cross: true },
          },
          {
            id: `decoder-${i}-norm2`,
            name: 'LayerNorm',
            type: 'norm',
            description: 'Post-norm',
          },
          {
            id: `decoder-${i}-ffn`,
            name: 'FeedForward',
            type: 'feedforward',
            description: `FFN(${dim_feedforward})`,
            config: { hidden: dim_feedforward, activation },
          },
        ],
      })),
    });
  }

  tree.children!.push({
    id: 'output',
    name: 'Output',
    type: 'output',
    description: 'Project to vocabulary',
    config: { vocab_size },
  });

  return tree;
};

// 3D Box Node Component with improved visuals
interface BoxNode3DProps {
  node: TreeNodeWithPosition;
  isSelected: boolean;
  onClick: (node: TreeNodeWithPosition) => void;
}

const BoxNode3D = memo(function BoxNode3D({
  node,
  isSelected,
  onClick,
}: BoxNode3DProps) {
  const data = node.data;
  const colors = getNodeColor(data.type);
  const dimensions = getNodeDimensions(data.type);

  return (
    <group position={[node.x, node.y, node.z]}>
      {isSelected && (
        <mesh>
          <boxGeometry args={[dimensions[0] * 1.16, dimensions[1] * 1.16, dimensions[2] * 1.16]} />
          <meshBasicMaterial
            color={colors.glow}
            transparent
            opacity={0.12}
          />
        </mesh>
      )}

        <mesh
        onClick={(event: ThreeEvent<MouseEvent>) => {
          event.stopPropagation();
          onClick(node);
        }}
      >
        <primitive object={getBoxGeometry(data.type)} attach="geometry" />
        <meshStandardMaterial
          color={colors.primary}
          roughness={0.35}
          metalness={0.35}
          emissive={colors.secondary}
          emissiveIntensity={isSelected ? 0.38 : 0.12}
        />
      </mesh>

      <lineSegments>
        <edgesGeometry args={[getBoxGeometry(data.type)]} />
        <lineBasicMaterial color={colors.glow} opacity={0.7} transparent />
      </lineSegments>
    </group>
  );
});

// 3D Scene Component
interface Scene3DProps {
  treeLayout: TreeNodeWithPosition[];
  selectedNode: TreeNodeWithPosition | null;
  onNodeClick: (node: TreeNodeWithPosition) => void;
  cameraPreset: string;
}

// Camera Controller Component
const CameraController: React.FC<{ preset: string }> = ({ preset }) => {
  const { camera, gl, invalidate } = useThree();
  const controlsRef = useRef<any>(null);
  const animationRef = useRef<number | null>(null);

  useEffect(() => {
    const controls = new OrbitControlsImpl(camera, gl.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 5;
    controls.maxDistance = 80;
    controls.enablePan = true;
    controls.panSpeed = 1;
    controls.minPolarAngle = 0;
    controls.maxPolarAngle = Math.PI;
    controls.enableZoom = true;
    controls.zoomSpeed = 1;
    controls.autoRotateSpeed = 0.3;

    const handleChange = () => invalidate();
    controls.addEventListener('change', handleChange);
    controlsRef.current = controls;

    return () => {
      controls.removeEventListener('change', handleChange);
      controls.dispose();
      controlsRef.current = null;
    };
  }, [camera, gl, invalidate]);

  useFrame(() => {
    if (controlsRef.current) {
      controlsRef.current.update();
    }
  });

  const animateCamera = useCallback((targetPosition: [number, number, number], targetLookAt: [number, number, number]) => {
    // Cancel any existing animation
    if (animationRef.current !== null) {
      cancelAnimationFrame(animationRef.current);
    }

    const startPosition = camera.position.clone();
    const endPosition = new Vector3(...targetPosition);
    const startTarget = controlsRef.current?.target.clone() || new Vector3(0, 5, 0);
    const endTarget = new Vector3(...targetLookAt);

    let progress = 0;
    const duration = 1000; // ms
    const startTime = Date.now();

    const animate = () => {
      const elapsed = Date.now() - startTime;
      progress = Math.min(elapsed / duration, 1);

      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3);

      camera.position.lerpVectors(startPosition, endPosition, eased);

      if (controlsRef.current) {
        const currentTarget = new Vector3();
        currentTarget.lerpVectors(startTarget, endTarget, eased);
        controlsRef.current.target.copy(currentTarget);
        controlsRef.current.update();
      }
      invalidate();

      if (progress < 1) {
        animationRef.current = requestAnimationFrame(animate);
      } else {
        // Ensure final position is exactly set
        camera.position.copy(endPosition);
        if (controlsRef.current) {
          controlsRef.current.target.copy(endTarget);
          controlsRef.current.update();
        }
        invalidate();
        animationRef.current = null;
      }
    };

    animate();
  }, [camera, invalidate]);

  // Handle preset changes
  useEffect(() => {
    const config = CAMERA_PRESETS[preset as keyof typeof CAMERA_PRESETS];
    if (controlsRef.current) {
      controlsRef.current.autoRotate = preset === 'auto';
    }
    if (config) {
      animateCamera(config.position, config.target);
    }

    // Cleanup on unmount
    return () => {
      if (animationRef.current !== null) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [preset, animateCamera]);

  return null;
};

const Scene3D = memo(function Scene3D({
  treeLayout,
  selectedNode,
  onNodeClick,
  cameraPreset,
}: Scene3DProps) {
  // Calculate connections with curves
  const connections = useMemo(() => {
    const lines: Array<{
      start: [number, number, number];
      end: [number, number, number];
      color: string;
    }> = [];

    treeLayout.forEach((node) => {
      if (node.children && node.children.length > 0) {
        const colors = getNodeColor(node.data.type);
        node.children.forEach((child) => {
          lines.push({
            start: [node.x, node.y, node.z] as [number, number, number],
            end: [child.x, child.y, child.z] as [number, number, number],
            color: colors.glow,
          });
        });
      }
    });

    return lines;
  }, [treeLayout]);

  return (
    <>
      <ambientLight intensity={0.4} />
      <directionalLight position={[10, 10, 5]} intensity={0.75} />
      <directionalLight position={[-10, -10, -5]} intensity={0.25} />
      <pointLight position={[0, 10, 0]} intensity={0.35} color="#6366f1" />

      <CameraController preset={cameraPreset} />

      <gridHelper args={[100, 100, '#1e293b', '#0f172a']} position={[0, -10, 0]} />
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -10.1, 0]}>
        <planeGeometry args={[100, 100]} />
        <meshStandardMaterial color="#0f172a" roughness={0.9} />
      </mesh>

      {treeLayout.map((node) => (
        <BoxNode3D
          key={node.data.id}
          node={node}
          isSelected={selectedNode?.data.id === node.data.id}
          onClick={onNodeClick}
        />
      ))}

      {connections.map((line, i) => (
        <group key={`connection-${i}`}>
          <Line
            points={[line.start, line.end]}
            color={line.color}
            opacity={0.3}
            transparent
            lineWidth={2}
          />
          <mesh position={line.start}>
            <sphereGeometry args={[0.1, 8, 8]} />
            <meshBasicMaterial color={line.color} opacity={0.5} transparent />
          </mesh>
          <mesh position={line.end}>
            <sphereGeometry args={[0.1, 8, 8]} />
            <meshBasicMaterial color={line.color} opacity={0.5} transparent />
          </mesh>
        </group>
      ))}
    </>
  );
});

// Custom Line component that works with R3F
interface LineProps {
  points: [number, number, number][];
  color: string;
  opacity?: number;
  transparent?: boolean;
  lineWidth?: number;
}

const Line: React.FC<LineProps> = ({ points, color, opacity = 1, transparent = false, lineWidth = 1 }) => {
  const pointsRef = useRef<any>(null);

  useEffect(() => {
    if (!pointsRef.current) {
      return;
    }

    const positions = new Float32Array(points.flat());
    pointsRef.current.setAttribute('position', new BufferAttribute(positions, 3));
  }, [points]);

  return (
    <line>
      <bufferGeometry ref={pointsRef} />
      <lineBasicMaterial
        color={color}
        opacity={opacity}
        transparent={transparent}
        linewidth={lineWidth}
      />
    </line>
  );
};

// WebGL Context Handler Component
const WebGLContextHandler: React.FC<{
  onError: (error: Error) => void;
  children: React.ReactNode;
}> = ({ onError, children }) => {
  const { gl } = useThree();

  useEffect(() => {
    const canvas = gl.domElement;

    const handleContextLost = (event: Event) => {
      event.preventDefault();
      console.error('WebGL context lost');
      onError(new Error('WebGL context was lost'));
    };

    const handleContextRestored = () => {
      console.info('WebGL context restored');
      // Force a re-render by invalidating the renderer
      gl.setSize(gl.domElement.width, gl.domElement.height);
    };

    canvas.addEventListener('webglcontextlost', handleContextLost);
    canvas.addEventListener('webglcontextrestored', handleContextRestored);

    return () => {
      canvas.removeEventListener('webglcontextlost', handleContextLost);
      canvas.removeEventListener('webglcontextrestored', handleContextRestored);
    };
  }, [gl, onError]);

  return <>{children}</>;
};

export const TransformerStructure3D: React.FC<{ className?: string }> = ({ className }) => {
  const { config } = useTransformerStore();
  const [selectedNode, setSelectedNode] = useState<TreeNodeWithPosition | null>(null);
  const [showHelp, setShowHelp] = useState(true);
  const [cameraPreset, setCameraPreset] = useState('default');
  const [webGLError, setWebGLError] = useState<string | null>(null);
  const [canvasKey, setCanvasKey] = useState(0);

  // Generate improved 3D tree layout
  const treeLayout = useMemo(() => {
    const treeData = getTransformerTree(config);
    const root = hierarchy(treeData);

    // Collect all nodes with their depth
    const nodesWithDepth: Array<{ node: HierarchyNode<TreeNode>; depth: number }> = [];
    root.descendants().forEach(node => {
      nodesWithDepth.push({ node, depth: node.depth });
    });

    // Group nodes by depth
    const nodesByDepth: Map<number, HierarchyNode<TreeNode>[]> = new Map();
    nodesWithDepth.forEach(({ node, depth }) => {
      if (!nodesByDepth.has(depth)) {
        nodesByDepth.set(depth, []);
      }
      nodesByDepth.get(depth)!.push(node);
    });

    // Create 3D positions
    const nodes3D: TreeNodeWithPosition[] = [];
    const layerSpacing = 8;
    const nodeSpacing = 5;

    // Sort nodes within each depth level for better organization
    const sortNodes = (nodes: HierarchyNode<TreeNode>[]) => {
      return nodes.sort((a, b) => {
        // Keep Input and Output at the edges
        if (a.data.type === 'input') return -1;
        if (b.data.type === 'input') return 1;
        if (a.data.type === 'output') return 1;
        if (b.data.type === 'output') return -1;

        // Then sort by type group
        const typeOrder = ['root', 'embedding', 'encoder', 'encoder_layer', 'decoder', 'decoder_layer', 'attention', 'feedforward', 'norm', 'positional'];
        const aIndex = typeOrder.indexOf(a.data.type);
        const bIndex = typeOrder.indexOf(b.data.type);

        if (aIndex !== -1 && bIndex !== -1) return aIndex - bIndex;
        if (aIndex !== -1) return -1;
        if (bIndex !== -1) return 1;

        return a.data.id.localeCompare(b.data.id);
      });
    };

    nodesByDepth.forEach((nodes, depth) => {
      const sortedNodes = sortNodes(nodes);

      sortedNodes.forEach((node, index) => {
        // Calculate X position - spread nodes evenly
        const totalWidth = (sortedNodes.length - 1) * nodeSpacing;
        const x = -totalWidth / 2 + index * nodeSpacing;

        // Calculate Y position - layers go downward
        const y = 20 - depth * layerSpacing;

        // Calculate Z position - give each depth a unique Z offset
        let z = 0;
        if (depth === 0) {
          z = 0;
        } else if (depth === 1) {
          z = -6;
        } else if (depth === 2) {
          z = 6;
        } else if (depth === 3) {
          z = -12;
        } else if (depth === 4) {
          z = 12;
        } else {
          z = (depth % 2 === 0 ? 1 : -1) * (6 + Math.floor(depth / 2) * 6);
        }

        (node as TreeNodeWithPosition).x = x;
        (node as TreeNodeWithPosition).y = y;
        (node as TreeNodeWithPosition).z = z;
        nodes3D.push(node as TreeNodeWithPosition);
      });
    });

    return nodes3D;
  }, [config]);

  const handleNodeClick = useCallback((node: TreeNodeWithPosition) => {
    setSelectedNode(selectedNode?.data.id === node.data.id ? null : node);
    setShowHelp(false);
  }, [selectedNode]);

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Box className="h-5 w-5 text-primary" />
              3D Transformer Architecture
            </CardTitle>
            <CardDescription>
              Interactive 3D visualization with cubic nodes. Labels are moved into the detail panel
              to keep the scene lighter on low-power devices.
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={() => setCameraPreset('default')}>
              <Home className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setCameraPreset(cameraPreset === 'auto' ? 'default' : 'auto')}
            >
              <RefreshCw className={`h-4 w-4 ${cameraPreset === 'auto' ? 'animate-spin' : ''}`} />
            </Button>
            <Button variant="outline" size="icon" onClick={() => setShowHelp(!showHelp)}>
              <Info className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Help panel */}
        {showHelp && (
          <div className="p-4 bg-gradient-to-r from-slate-900 to-slate-800 rounded-lg text-xs space-y-2 border border-slate-600">
            <div className="font-semibold text-white">🎮 Controls</div>
            <div className="grid grid-cols-2 gap-2 text-slate-300">
              <div>🖱️ <span className="text-sky-400 font-medium">Left Drag</span> - Rotate</div>
              <div>🖱️ <span className="text-sky-400 font-medium">Right Drag</span> - Pan</div>
              <div>🖱️ <span className="text-sky-400 font-medium">Scroll</span> - Zoom</div>
              <div>🖱️ <span className="text-sky-400 font-medium">Click Node</span> - Details</div>
            </div>
          </div>
        )}

        {/* 3D Canvas */}
        <div className="relative w-full h-[550px] rounded-lg overflow-hidden border border-slate-800">
          {webGLError ? (
            // WebGL Error Fallback
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-900 text-white p-8">
              <div className="text-center space-y-4 max-w-md">
                <div className="text-6xl">⚠️</div>
                <h3 className="text-xl font-bold text-red-400">WebGL Error</h3>
                <p className="text-sm text-slate-300">{webGLError}</p>
                <div className="text-xs text-slate-400 space-y-2">
                  <p>This may be caused by:</p>
                  <ul className="list-disc list-inside space-y-1">
                    <li>Too many WebGL contexts open</li>
                    <li>GPU driver issues</li>
                    <li>Browser hardware acceleration disabled</li>
                  </ul>
                </div>
                <div className="flex gap-2 justify-center">
                  <Button
                    onClick={() => {
                      setWebGLError(null);
                      setCanvasKey(prev => prev + 1);
                    }}
                    className="bg-blue-600 hover:bg-blue-700"
                  >
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Retry
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => window.location.reload()}
                    className="border-slate-600 text-white"
                  >
                    Reload Page
                  </Button>
                </div>
              </div>
            </div>
          ) : (
            <Canvas
              key={canvasKey}
              dpr={[1, 1.5]}
              frameloop={cameraPreset === 'auto' ? 'always' : 'demand'}
              camera={{ position: [20, 15, 20], fov: 45 }}
              shadows={false}
              gl={{
                antialias: true,
                alpha: true,
                powerPreference: 'high-performance',
                failIfMajorPerformanceCaveat: false,
              }}
              onError={(error) => {
                console.error('Canvas error:', error);
                setWebGLError('Failed to initialize WebGL. See console for details.');
              }}
              onCreated={({ gl }) => {
                gl.setClearColor(0x0f172a, 1);
              }}
            >
              <WebGLContextHandler onError={(error) => setWebGLError(error.message)}>
                <></>
              </WebGLContextHandler>
              <color attach="background" args={['#0f172a']} />
              <fog attach="fog" args={['#0f172a', 30, 80]} />
              <Scene3D
                treeLayout={treeLayout}
                selectedNode={selectedNode}
                onNodeClick={handleNodeClick}
                cameraPreset={cameraPreset}
              />
            </Canvas>
          )}

          {/* Camera control overlay and other UI - only show when no error */}
          {!webGLError && (
            <>
              {/* Camera control overlay */}
              <div className="absolute top-4 left-4 flex flex-col gap-2">
            <div className="bg-slate-900/90 backdrop-blur-sm rounded-lg p-2 border border-slate-600 shadow-xl">
              <div className="text-[10px] text-slate-300 mb-2 px-1 font-medium">Camera Views</div>
              <div className="grid grid-cols-3 gap-1">
                <Button
                  variant="secondary"
                  size="sm"
                  className="h-7 text-[10px] bg-slate-800 hover:bg-slate-700 text-white border border-slate-600"
                  onClick={() => setCameraPreset('front')}
                >
                  <ArrowUpDown className="h-3 w-3 mr-1" />
                  Front
                </Button>
                <Button
                  variant="secondary"
                  size="sm"
                  className="h-7 text-[10px] bg-slate-800 hover:bg-slate-700 text-white border border-slate-600"
                  onClick={() => setCameraPreset('side')}
                >
                  <Move3D className="h-3 w-3 mr-1" />
                  Side
                </Button>
                <Button
                  variant="secondary"
                  size="sm"
                  className="h-7 text-[10px] bg-slate-800 hover:bg-slate-700 text-white border border-slate-600"
                  onClick={() => setCameraPreset('top')}
                >
                  <Camera className="h-3 w-3 mr-1" />
                  Top
                </Button>
                <Button
                  variant="secondary"
                  size="sm"
                  className="h-7 text-[10px] col-span-2 bg-slate-800 hover:bg-slate-700 text-white border border-slate-600"
                  onClick={() => setCameraPreset('isometric')}
                >
                  <RotateCw className="h-3 w-3 mr-1" />
                  Isometric
                </Button>
                <Button
                  variant="secondary"
                  size="sm"
                  className="h-7 text-[10px] bg-slate-800 hover:bg-slate-700 text-white border border-slate-600"
                  onClick={() => setCameraPreset('diagonal')}
                >
                  <Home className="h-3 w-3 mr-1" />
                  Reset
                </Button>
              </div>
            </div>
          </div>

          {/* Stats overlay */}
          <div className="absolute bottom-4 left-4 flex gap-2">
            <Badge variant="secondary" className="bg-slate-900/80 text-white border-slate-700">
              {treeLayout.length} nodes
            </Badge>
            <Badge variant="secondary" className="bg-slate-900/80 text-white border-slate-700">
              {config.num_encoder_layers + config.num_decoder_layers} layers
            </Badge>
            <Badge variant="secondary" className="bg-slate-900/80 text-white border-slate-700">
              {CAMERA_PRESETS[cameraPreset as keyof typeof CAMERA_PRESETS]?.name || 'Custom'}
            </Badge>
          </div>

          {/* Type legend */}
          <div className="absolute top-4 right-4 space-y-1">
            <div className="bg-slate-900/80 backdrop-blur-sm rounded-lg p-2 border border-slate-600">
              {Object.entries(COLOR_PALETTE).slice(0, 6).map(([type, colors]) => (
                <div key={type} className="flex items-center gap-2 text-[10px] mb-1 last:mb-0">
                  <div
                    className="w-3 h-3 rounded-sm border border-white/30 shadow-sm"
                    style={{ backgroundColor: colors.primary }}
                  />
                  <span className="text-white capitalize font-medium">
                    {type.replace('_', ' ')}
                  </span>
                </div>
              ))}
            </div>
          </div>
            </>
          )}
        </div>

        {/* Selected node details */}
        {selectedNode && (
          <div className="p-4 bg-gradient-to-r from-slate-900 to-slate-800 rounded-lg border border-slate-600 space-y-3 animate-in fade-in slide-in-from-top-2">
            <div className="flex items-center justify-between">
              <h4 className="font-semibold text-white flex items-center gap-2">
                <div
                  className="w-4 h-4 rounded-sm"
                  style={{ backgroundColor: getNodeColor(selectedNode.data.type).primary }}
                />
                {selectedNode.data.name}
              </h4>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => setSelectedNode(null)}
                className="bg-slate-800 hover:bg-slate-700 text-white border border-slate-600"
              >
                Clear
              </Button>
            </div>
            <p className="text-sm text-slate-300">{selectedNode.data.description}</p>
            {selectedNode.data.config && (
              <div className="grid grid-cols-2 gap-2 text-xs">
                {Object.entries(selectedNode.data.config).map(([key, value]) => (
                  <div key={key} className="flex items-center justify-between p-2 bg-slate-950/50 rounded border border-slate-700">
                    <span className="text-slate-400">{key}:</span>
                    <Badge variant="secondary" className="bg-slate-800 text-slate-200">
                      {String(value)}
                    </Badge>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Full color legend */}
        <div className="grid grid-cols-4 gap-2 text-xs">
          {Object.entries(COLOR_PALETTE).map(([type, colors]) => (
            <div key={type} className="flex items-center gap-2 p-2 bg-slate-900/50 rounded border border-slate-700">
              <div
                className="w-4 h-4 rounded-sm border border-white/20"
                style={{
                  backgroundColor: colors.primary,
                  boxShadow: `0 0 8px ${colors.glow}60`,
                }}
              />
              <span className="text-slate-200 capitalize font-medium">{type.replace('_', ' ')}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};
