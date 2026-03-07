# Transformer Visualization Frontend

React + TypeScript frontend for visualizing Transformer model computations.

## Tech Stack

- **React 18.3** with TypeScript 5.3
- **Vite 5.0** for build tooling
- **TailwindCSS** for styling
- **Zustand** for state management
- **D3.js** / **React Flow** for visualizations
- **Axios** for API requests

## Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

## API Configuration

The API URL is configured via environment variable:
- Create a `.env` file with `VITE_API_BASE_URL=http://localhost:8000`

## Project Structure

```
src/
├── components/
│   ├── ui/                    # shadcn/ui base components
│   ├── visualization/         # Visualization components
│   │   ├── embedding/         # Token/Positional Encoding
│   │   ├── attention/         # Multi-Head Attention
│   │   ├── feedforward/       # FFN
│   │   ├── normalization/     # LayerNorm
│   │   └── transformer/       # Complete flow
│   ├── controls/              # Parameter/Input controls
│   └── layout/                # Layout components
├── stores/                    # Zustand stores
├── services/                  # API services
├── types/                     # TypeScript types
└── utils/                     # Utility functions
```
