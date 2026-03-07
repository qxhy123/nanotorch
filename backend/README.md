# Transformer Visualization Backend

FastAPI backend for Transformer model visualization.

## Tech Stack

- **FastAPI** - Web framework
- **uvicorn** - ASGI server
- **Pydantic** - Data validation
- **nanotorch** - Transformer implementation

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Project Structure

```
backend/
├── app/
│   ├── main.py                 # FastAPI application
│   ├── api/
│   │   └── routes/
│   │       └── transformer.py  # Transformer endpoints
│   ├── core/
│   │   └── transformer_wrapper.py  # nanotorch wrapper
│   ├── models/
│   │   └── schemas.py          # Pydantic models
│   └── utils/
│       └── tensor_serialization.py  # Tensor utilities
└── requirements.txt
```

## Environment

The backend uses nanotorch from the parent directory.
Make sure you're in the nanotorch project root.
