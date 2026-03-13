# YOLO-Viz Frontend

YOLO Object Detection Visualization Platform - React 19 + TypeScript + Vite

This is a subproject of [NanoTorch](../). See the main [CLAUDE.md](../CLAUDE.md) for overall project context.

## Project-Specific Compact Instructions

When compacting during YOLO-Viz development:

### Preserve
- **Current view being worked on**: Which of the 9 views is being modified
- **Navigation issues**: Route changes, MainLayout problems
- **Visualization bugs**: KaTeX rendering, Recharts issues, 3D visualizations
- **Type errors**: TypeScript compilation errors
- **Build/dev server issues**: Vite configuration problems

### Can Summarize
- General UI/UX discussions
- Styling tweaks (unless they fix a bug)

## Tech Stack

- **Framework**: React 19.2.0
- **Language**: TypeScript 5.9.3
- **Build Tool**: Vite 7.3.1
- **Styling**: TailwindCSS v4.2.1
- **State**: Zustand 5.0.11
- **Routing**: React Router 7.13.1
- **Charts**: Recharts 3.8.0
- **Math**: KaTeX 0.16.38 + react-katex 3.1.0
- **Icons**: Lucide React 0.577.0

## Available Scripts

```bash
npm run dev      # Start dev server
npm run build    # Build for production
npm run lint     # Run ESLint
npm run preview  # Preview production build
```

## View Routes

| Route | Component | Description |
|-------|-----------|-------------|
| `/` | ArchitectureView | YOLO overview |
| `/backbone` | BackboneView | CSPDarknet/DarkNet53 |
| `/neck` | NeckView | FPN/PANet/BiFPN |
| `/head` | HeadView | Detection head |
| `/anchors` | AnchorsView | Grid + anchors |
| `/nms` | NMSView | NMS animation |
| `/loss` | LossView | Loss functions |
| `/versions` | VersionsView | Version comparison |
| `/playground` | PlaygroundView | Interactive demo |
