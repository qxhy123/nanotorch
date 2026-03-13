# nanotorch Design Documentation

This document describes the current architecture of nanotorch after the autograd unification work. It focuses on the Python package and separately calls out the repository-level visualization app.

## Project Goals

nanotorch is built around a few stable goals:

1. Educational clarity over framework-level complexity.
2. PyTorch-like APIs where they help learning.
3. NumPy as the default execution substrate, with optional CuPy-backed CUDA paths.
4. Shared autograd mechanics instead of per-file backward entry points.
5. Practical documentation that matches the codebase as it exists today.

## Architecture Overview

The repository has two major layers:

```text
nanotorch/
├── nanotorch/          # Python package
│   ├── tensor.py       # Tensor object and user-facing tensor methods
│   ├── autograd.py     # Function abstraction and shared backward engine
│   ├── device.py       # CPU/CUDA device helpers
│   ├── backend/        # NumPy / CuPy backend abstractions
│   ├── nn/             # Layers, losses, attention, normalization, sequence models
│   ├── optim/          # Optimizers and learning-rate schedulers
│   ├── data/           # Dataset, samplers, DataLoader
│   ├── transforms/     # Preprocessing and augmentation helpers
│   └── tokenizer/      # Character, word, and BPE tokenizers
├── docs/               # Design notes, API docs, tutorials
├── tests/              # Validation and regression coverage
├── frontend/           # Visualization frontend application
└── backend/            # Visualization backend application
```

The Python package is the published library. The repository-level `frontend/` and `backend/` directories provide the Transformer visualization app and are not part of the package metadata in `pyproject.toml`.

## Tensor Model

`Tensor` is the central data structure. In the current implementation it stores:

- `data`: NumPy array by default, or a CuPy array when running on CUDA
- `requires_grad`: whether this tensor participates in gradient tracking
- `grad`: accumulated gradient, stored as another `Tensor`
- `_parents`: the arguments captured from the operation that produced this tensor
- `_ctx`: a `FunctionContext` instance with saved tensors and values for backward
- `_fn`: the `Function` subclass responsible for the operation
- `_device`: a `Device` object describing CPU or CUDA placement

This is a change from the older string-dispatch design. The current graph records executable function metadata directly instead of symbolic operation names.

### Forward construction

Most differentiable operations are defined through `Function` subclasses in `nanotorch.autograd`:

```python
class SomeOp(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return Tensor(...)

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.get_saved_tensors()
        return grad_x, grad_y
```

`Function.apply(...)` is the shared entry point. It:

1. Creates a fresh `FunctionContext`.
2. Runs `forward(...)`.
3. Attaches `_ctx`, `_fn`, and `_parents` to tensor outputs.
4. Leaves non-tensor outputs detached from the gradient graph.

`Tensor.backward(...)` is a thin wrapper around `nanotorch.autograd.backward(...)`, so there is one public backward traversal path.

## Autograd Engine

The shared backward engine in `nanotorch.autograd` performs reverse-mode autodiff in four steps:

1. Build a topological ordering by traversing `Tensor._parents`.
2. Seed the root gradient with ones when the caller does not provide one.
3. Walk the graph in reverse topological order.
4. Call each node's `Function.backward(...)` and accumulate parent gradients.

### Gradient accumulation and broadcasting

Gradient accumulation is centralized in helper functions such as `accumulate_grad(...)`, `accumulate_grad_batch(...)`, and `reduce_gradient_numpy(...)`.

That centralization matters for consistency:

- broadcasting reductions are handled in one place
- gradients are accumulated into existing `parent.grad` tensors when shapes already match
- operation implementations can return raw arrays or tensors and let the engine normalize them

This is the main reason the current design is easier to extend than the previous hand-dispatched backward model.

### Context handling

`FunctionContext` provides two storage channels:

- `save_for_backward(...)` for tensors needed during backward
- `save_value(...)` for lightweight metadata such as shapes, axes, or flags

Using explicit context objects keeps forward/backward coupling local to each operation and avoids global autograd state.

## Intentional Autograd Boundaries

Autograd is now unified around `Function.apply(...)` plus the shared backward engine, but the codebase still contains deliberate raw-array and `.data` boundaries.

These boundaries are expected in several places:

- optimizer state updates and parameter writes
- low-level numeric kernels that naturally operate on raw arrays
- detached evaluation and metrics paths
- initialization-time mutations
- selected detection and demo losses that intentionally stay NumPy-heavy

Those cases do not imply that the core autograd mechanism is split. The classification rules and examples live in [`docs/autograd_boundaries.md`](./autograd_boundaries.md), and that document should be treated as the source of truth for remaining boundary audits.

## Modules, State, and Training Workflow

`nanotorch.nn.Module` provides the model-building surface:

- parameter and submodule registration
- `parameters()` traversal
- `state_dict()` / `load_state_dict(...)`
- `train()` / `eval()` mode switches
- composition helpers such as `Sequential`

The current `nn` package covers the core educational path plus several extended families:

- feed-forward layers, activations, and losses
- convolutions, pooling, dropout, and normalization
- attention, embeddings, and Transformer blocks
- RNN / LSTM / GRU implementations
- recommendation-oriented layers and metrics

The public API intentionally exposes more breadth than an introductory tensor toy project, but the maturity of each subsystem is not identical.

## Optimizers, Data, and Utilities

`nanotorch.optim` contains the standard training loop tools:

- optimizers: `SGD`, `Adam`, `AdamW`, `RMSprop`, `Adagrad`
- schedulers: step, milestone, exponential, cosine, linear, plateau, and warmup variants

`nanotorch.data` and related helpers provide the supporting pipeline pieces:

- `Dataset`, `TensorDataset`, `Subset`
- `SequentialSampler`, `RandomSampler`, `BatchSampler`
- `DataLoader` and collate helpers
- preprocessing and augmentation utilities under `transforms`
- text preprocessing utilities under `tokenizer`

## Device and Backend Story

nanotorch is CPU-first. NumPy is the default backend and the best-covered path.

The package also includes:

- `Device` abstractions for CPU and CUDA selection
- backend interfaces under `nanotorch.backend`
- optional CuPy-backed execution when CuPy is installed and the subsystem supports it

This is more accurate than saying the project has "no GPU support", but it is still not equivalent to full PyTorch CUDA parity. Coverage remains uneven across modules and tests.

## Visualization App Boundary

The repository-level visualization app is intentionally documented as a companion system rather than part of the package core.

- `frontend/` contains the interactive UI
- `backend/` contains the FastAPI service layer
- both depend on the Python package for model behavior and tensor data

This separation keeps the package metadata focused on the library while still documenting the broader educational workflow available in the repository.

## Limitations and Future Direction

Current limitations:

1. Educational simplicity still wins over full PyTorch feature parity.
2. CPU code paths are the most reliable; CUDA/CuPy support is optional and partial.
3. Advanced and experimental subsystems vary more in maturity than the tensor/autograd/nn/optim core.
4. Performance optimization is secondary to readability in many implementations.

Likely future direction:

1. Expand operation coverage without re-fragmenting autograd.
2. Improve CUDA coverage and backend consistency.
3. Keep repository documentation aligned as experimental subsystems evolve.
4. Add more gradient checks and subsystem-specific regression tests.

## Testing Strategy

The core test strategy is:

1. Unit tests for tensor, autograd, `nn`, and optimizer behavior.
2. Regression tests for previously broken gradient flows.
3. Smoke tests for importability and representative training-loop paths.
4. Targeted tests for experimental subsystems where behavior is still evolving.

## Contributing

When changing implementation details, update the code and the design docs together. In particular:

- keep architecture descriptions aligned with `tensor.py` and `autograd.py`
- do not reintroduce stale string-dispatch documentation
- document intentional raw-array boundaries instead of treating every `.data` use as an autograd bug
- prefer examples that use currently exported, importable symbols
