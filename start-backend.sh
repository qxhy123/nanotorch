#!/bin/bash
# Start backend server

cd /Users/yangyang/ai_projs/nanotorch/backend
export PYTHONPATH=/Users/yangyang/ai_projs/nanotorch:/Users/yangyang/ai_projs/nanotorch/backend

echo "Starting Transformer Visualization Backend..."
echo "API will be available at: http://localhost:8000"
echo "API docs at: http://localhost:8000/docs"
echo ""

python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
