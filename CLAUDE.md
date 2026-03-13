# NanoTorch Project

A PyTorch-like deep learning framework from scratch, with web-based visualization platforms.

## Project Structure

```
nanotorch/
├── nanotorch/           # Core Python library (autograd, nn, optim, etc.)
├── examples/            # Usage examples
├── tests/               # Unit tests
├── benchmarks/          # Performance benchmarks
├── frontend/            # Original React visualization (DNN/Transformer)
├── sd-viz-frontend/     # Stable Diffusion visualization (React 19)
├── yolo-viz-frontend/   # YOLO visualization (React 19)
├── backend/             # Visualization backend server
└── docs/                # Documentation
```

## Compact Instructions

When compacting conversation history, please:

### What MUST be Preserved
- **Active development context**: Current file being edited, current task/bug being fixed
- **Code changes made**: All file modifications, their purpose, and location
- **Key decisions**: Architectural decisions, trade-offs discussed, API design choices
- **Test failures/errors**: Current error messages, stack traces, and debugging context
- **Implementation progress**: Which phase of implementation we're in (e.g., "Phase 2 of 4")

### What Can Be Summarized
- General discussions about framework design (keep 1-2 key points)
- Code explorations that didn't lead to changes
- File reads that were informational only
- Successful build/test confirmations (just note "build passed")

### Summary Format
- Limit summaries to 300-500 words
- Use bullet points for multiple items
- Include file paths with line numbers for key changes (e.g., `src/nn/linear.py:45`)
- Keep the most recent error/stack trace if debugging

### Examples

**Preserve this:**
```
Error: RuntimeError in nanotorch/nn/conv.py:142
Shape mismatch: expected (B, 64, 28, 28), got (B, 32, 28, 28)
Caused by incorrect out_channels calculation in Conv2d forward pass
```

**Summarize this:**
```
Discussed autograd implementation details. Decided to use PyTorch-compatible
tensor storage format for better interoperability.
```

## Development Guidelines

### Python Code Style
- Follow PEP 8
- Use type hints (mypy compatible)
- Docstrings follow Google style
- Maximum line length: 100

### Frontend (React/TypeScript)
- Use functional components with hooks
- TailwindCSS for styling
- Recharts for data visualization
- KaTeX for math formulas

### Testing
- Use pytest for Python tests
- Each module should have >80% coverage
- Run `pytest tests/` before committing

## Quick Start

```bash
# Install dependencies
pip install -e .

# Run tests
pytest tests/

# Start visualization backend
./start-backend.sh

# Start frontend
cd frontend && npm run dev
# or for SD-Viz:
cd sd-viz-frontend && npm run dev
# or for YOLO-Viz:
cd yolo-viz-frontend && npm run dev
```

## Key Module Reference

### nanotorch/autograd
- `Tensor`: Core tensor class with autograd
- `Function`: Base class for differentiable operations
- `backward()`: Compute gradients

### nanotorch/nn
- `Module`: Base class for neural network layers
- `Linear`, `Conv2d`, `MaxPool2d`: Common layers
- `Sequential`: Layer container

### nanotorch/optim
- `SGD`, `Adam`: Optimizers
- `Optimizer`: Base optimizer class

### Frontend Views (yolo-viz-frontend)
- ArchitectureView: YOLO overview
- BackboneView: CSPDarknet visualization
- NeckView: FPN/PANet pyramids
- HeadView: Detection head structure
- AnchorsView: Interactive grid + anchors
- NMSView: Step-by-step NMS animation
- LossView: IoU/GIoU/DIoU/CIoU comparison
- VersionsView: YOLO v1-v8 timeline
- PlaygroundView: Interactive detection demo
