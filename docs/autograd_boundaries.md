# Autograd Boundaries

This note records the remaining `.data` and NumPy-heavy paths after the
Autograd Unification work was completed.

## What Is Done

The differentiable training graph is unified around `Function.apply(...)` and the
shared engine in `nanotorch/autograd.py`.

Completed areas include:

- Core tensor ops and reductions
- Embedding and embedding bag backward paths
- High-level model graph assembly in detection, recommendation, RNN, attention,
  and Stable Diffusion modules
- Removal of legacy `_op` / `_backward_operation(...)` dispatch

## Remaining Intentional Categories

These remaining `.data` usages are not treated as autograd-unification bugs.

### 1. Buffer Or State Mutation

These update non-differentiable buffers and should stay outside the autograd graph.

Representative files:

- `nanotorch/nn/normalization.py`
- `nanotorch/nn/batchnorm.py`
- optimizer implementations under `nanotorch/optim/`
- gradient clipping helpers in `nanotorch/utils.py`

### 2. Low-Level Kernel Implementations

These paths implement numeric kernels directly and naturally work on raw arrays.

Representative files:

- `nanotorch/nn/conv.py`
- some custom backward kernels in `nanotorch/autograd.py`

### 3. Detached Evaluation Or Inference Aggregation

These aggregate already-computed results and do not need gradient tracking.

Representative files:

- `nanotorch/nn/metrics.py`
- decode / scheduler plumbing in detection and Stable Diffusion inference code

### 4. Initialization-Time Parameter Writes

These are one-time setup operations, not training-graph operations.

Representative files:

- `nanotorch/nn/embedding.py` (`padding_idx` zeroing)
- selected initialization helpers in `nanotorch/utils.py`

### 5. Detection Loss Taxonomy

Detection losses currently fall into two groups:

- Full or semi-full losses that still use NumPy-heavy target assignment / decoding
  logic internally, especially older YOLO variants (`yolo_v1` through `yolo_v8`).
- Lightweight placeholder or demo losses that intentionally use detached MSE-style
  calculations for simplified examples and tests (`YOLOv9Loss`, `YOLOv10Loss`,
  `YOLOv11Loss`, and various `*LossSimple` classes).

These are implementation-scope decisions, not evidence that the core autograd
mechanism is still split.

## Practical Rule

When evaluating a remaining `.data` usage, classify it as a real problem only if it:

- sits on a user-visible differentiable forward path,
- severs gradient flow between trainable parameters and a loss, and
- is not just mutating buffers, parameters, optimizer state, or evaluation caches.
