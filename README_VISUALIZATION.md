# Transformer Visualization Project

A comprehensive React + TypeScript frontend for visualizing Transformer model computations, built on top of the nanotorch library.

## Project Structure

```
nanotorch/
├── frontend/                    # React + TypeScript frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── ui/              # shadcn/ui base components
│   │   │   ├── visualization/   # Visualization components
│   │   │   │   ├── embedding/   # Token/Positional Encoding
│   │   │   │   ├── attention/   # Multi-Head Attention
│   │   │   │   ├── feedforward/ # FFN
│   │   │   │   ├── normalization/ # LayerNorm
│   │   │   │   └── transformer/ # Complete flow
│   │   │   ├── controls/        # Parameter/Input controls
│   │   │   └── layout/          # Layout components
│   │   ├── stores/              # Zustand stores
│   │   ├── services/            # API services
│   │   ├── types/               # TypeScript types
│   │   └── utils/               # Utility functions
│   └── package.json
│
├── backend/                     # FastAPI backend
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   ├── api/routes/          # API routes
│   │   ├── core/                # Transformer wrapper
│   │   ├── models/              # Pydantic models
│   │   └── utils/               # Utilities
│   └── requirements.txt
│
└── nanotorch/                   # Core library (existing)
    └── nn/
        ├── transformer.py       # Transformer implementation
        └── attention.py         # Attention implementation
```

## Tech Stack

### Frontend
- **React 18.3** + **TypeScript 5.3**
- **Vite 5.0** as build tool
- **TailwindCSS v3** for styling
- **shadcn/ui** for base UI components
- **Zustand** for state management
- **Axios** for API requests

### Backend
- **FastAPI** for REST API
- **uvicorn** as ASGI server
- **Pydantic** for data validation
- **nanotorch** for Transformer implementation

## Features Implemented

### Visualization Components

1. **Token Embedding**
   - Heatmap visualization of embedding vectors
   - Per-token dimension analysis
   - Statistical information (min, max, mean)

2. **Positional Encoding**
   - 2D sinusoidal encoding matrix
   - Waveform visualization
   - Formula explanation

3. **Attention Matrix**
   - Interactive attention weight heatmaps
   - Multi-head selection
   - Color scheme options (viridis, plasma, blues, reds, inferno)
   - Per-cell value inspection

4. **Multi-Head Attention**
   - Architecture diagram
   - Computation steps explanation
   - Head configuration details

5. **Feed Forward Network**
   - Network architecture visualization
   - Activation function comparison (ReLU/GELU)
   - Parameter count calculation

6. **Layer Normalization**
   - Before/after comparison
   - Interactive data generation
   - Gamma/Beta parameter effects

7. **Transformer Flow**
   - Complete pipeline visualization
   - Step-by-step animation
   - Interactive navigation

### Control Components

1. **Input Panel**
   - Text input for processing
   - Quick example selection
   - Token display
   - Error handling

2. **Parameter Panel**
   - Model configuration (d_model, nhead, layers, etc.)
   - Real-time validation
   - Reset functionality

## Running the Project

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Access at: http://localhost:5173

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API documentation at: http://localhost:8000/docs

## API Endpoints

- `GET /health` - Health check
- `POST /api/v1/transformer/forward` - Run forward pass
- `POST /api/v1/transformer/attention` - Get attention weights
- `POST /api/v1/transformer/embeddings` - Get embeddings
- `GET /api/v1/transformer/positional-encoding` - Get positional encodings
- `POST /api/v1/transformer/validate-config` - Validate configuration

## Development Notes

- The frontend uses shadcn/ui components styled with TailwindCSS
- State management via Zustand with a centralized transformerStore
- API communication via Axios with interceptors for logging
- TypeScript for type safety throughout

## Future Enhancements

- Real-time attention weight animation
- Export visualization as PNG/SVG
- More detailed layer-by-layer breakdown
- Support for decoder-side visualization
- Performance optimization for large sequences
