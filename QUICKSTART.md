# Transformer Visualization Quick Start

English quick start for the repository's visualization app. For Chinese instructions, see [`QUICKSTART_CN.md`](./QUICKSTART_CN.md).

This guide is for the repository-level frontend/backend visualization system. It is separate from the published `nanotorch` Python package metadata and requires both services to run for the full experience.

## What this app includes

- Frontend UI for Transformer structure, embeddings, attention, layer flow, tokenization, inference, and training-oriented views
- FastAPI backend that exposes Transformer, tokenizer, and layer-analysis endpoints backed by nanotorch components
- Shared Python library code from the `nanotorch` package

## Prerequisites

- Python `3.8+`
- Node.js `18+` and `npm`
- A local checkout of this repository

## Setup

### 1. Python environment

From the repository root:

```bash
uv venv
source .venv/bin/activate
uv sync
python -m pip install -r backend/requirements.txt
```

`uv sync` installs the library dependencies from `pyproject.toml`. The backend service also needs the FastAPI stack from `backend/requirements.txt`.

### 2. Frontend dependencies

```bash
cd frontend
npm install
cd ..
```

## Start the app

### Recommended: start services manually

Backend:

```bash
cd backend
PYTHONPATH="$(pwd)/..:$(pwd)" python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Frontend, in a second terminal:

```bash
cd frontend
npm run dev
```

Then open:

- Frontend app: `http://localhost:5173`
- Backend health: `http://localhost:8000/health`
- Backend docs: `http://localhost:8000/docs`

### Optional helper scripts

The repository also includes `./start-backend.sh` and `./start-frontend.sh`. They are convenience wrappers for this checkout; if you move the repository, verify their hard-coded paths before using them.

## Quick validation

### Smoke check the backend

```bash
curl http://localhost:8000/health
```

Expected result: a JSON health payload that includes the backend status and nanotorch version.

### Run the API smoke test script

From the repository root:

```bash
python test_api.py
```

## Key API surfaces

The backend currently exposes three main groups of endpoints:

- Transformer endpoints under `/api/v1/transformer`
- Tokenizer endpoints under `/api/v1/tokenizer`
- Layer analysis endpoints under `/api/v1/layer`

Common examples:

```bash
curl -X POST http://localhost:8000/api/v1/transformer/forward \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "d_model": 128,
      "nhead": 8,
      "num_encoder_layers": 2,
      "num_decoder_layers": 0,
      "dim_feedforward": 256,
      "dropout": 0.1,
      "activation": "relu",
      "max_seq_len": 64,
      "vocab_size": 1000,
      "layer_norm_eps": 1e-5,
      "batch_first": true,
      "norm_first": false
    },
    "input_data": {
      "text": "Hello world"
    },
    "options": {
      "return_attention": true,
      "return_all_layers": true,
      "return_embeddings": true
    }
  }'
```

```bash
curl -X POST http://localhost:8000/api/v1/tokenizer/tokenize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "hello nanotorch",
    "tokenizer_type": "char"
  }'
```

## Important notes

- The Transformer routes expect `input_data`, not `input`.
- The frontend talks to the backend at `http://localhost:8000` by default.
- Full visualization features require both the frontend dev server and the backend API.
- This quick start documents the visualization app only; package usage examples stay in [`README.md`](./README.md).

## Troubleshooting

### Backend import errors

Make sure the environment is activated and `backend/requirements.txt` is installed. If you start the backend manually, keep `PYTHONPATH` pointing at both the repo root and `backend/`.

### Frontend cannot reach the backend

- Confirm the backend is running: `curl http://localhost:8000/health`
- Confirm the frontend dev server is running on `http://localhost:5173`
- Check frontend environment variables if you changed the API base URL

### API request validation fails

Start with the interactive schema at `http://localhost:8000/docs` and confirm field names such as `input_data`, `config`, and `options`.

## Related docs

- [`README.md`](./README.md): package overview and library usage
- [`README_VISUALIZATION.md`](./README_VISUALIZATION.md): visualization architecture and feature inventory
- [`QUICKSTART_CN.md`](./QUICKSTART_CN.md): Chinese version of this quick start
