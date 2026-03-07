#!/bin/bash
# Test script for Transformer Visualization

echo "======================================"
echo "Transformer Visualization - Test Script"
echo "======================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test Backend
echo -e "${YELLOW}[1/4] Testing Backend...${NC}"
cd /Users/yangyang/ai_projs/nanotorch/backend

# Check nanotorch import
echo "  Testing nanotorch import..."
python -c "
import sys
sys.path.insert(0, '/Users/yangyang/ai_projs/nanotorch')
from nanotorch.nn.transformer import Transformer
print('  ✓ nanotorch import OK')
" 2>/dev/null

if [ $? -eq 0 ]; then
    echo -e "${GREEN}  ✓ Backend import test PASSED${NC}"
else
    echo -e "${RED}  ✗ Backend import test FAILED${NC}"
    exit 1
fi

# Test TransformerWrapper
echo "  Testing TransformerWrapper..."
python -c "
import sys
sys.path.insert(0, '/Users/yangyang/ai_projs/nanotorch')
sys.path.insert(0, '/Users/yangyang/ai_projs/nanotorch/backend')
from app.core.transformer_wrapper import TransformerWrapper, NANOTORCH_AVAILABLE
if NANOTORCH_AVAILABLE:
    wrapper = TransformerWrapper(d_model=512, nhead=8, num_encoder_layers=2, num_decoder_layers=0)
    print('  ✓ TransformerWrapper created')
else:
    raise Exception('nanotorch not available')
" 2>/dev/null

if [ $? -eq 0 ]; then
    echo -e "${GREEN}  ✓ TransformerWrapper test PASSED${NC}"
else
    echo -e "${RED}  ✗ TransformerWrapper test FAILED${NC}"
    exit 1
fi

# Test Frontend Build
echo ""
echo -e "${YELLOW}[2/4] Testing Frontend Build...${NC}"
cd /Users/yangyang/ai_projs/nanotorch/frontend

npm run build > /tmp/frontend_build.log 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}  ✓ Frontend build PASSED${NC}"
    echo "  Build output:"
    tail -3 /tmp/frontend_build.log | sed 's/^/    /'
else
    echo -e "${RED}  ✗ Frontend build FAILED${NC}"
    tail -10 /tmp/frontend_build.log | sed 's/^/    /'
    exit 1
fi

# Summary
echo ""
echo "======================================"
echo -e "${GREEN}All tests PASSED!${NC}"
echo "======================================"
echo ""
echo "To run the application:"
echo ""
echo "1. Backend (in one terminal):"
echo "   cd /Users/yangyang/ai_projs/nanotorch/backend"
echo "   PYTHONPATH=/Users/yangyang/ai_projs/nanotorch:/Users/yangyang/ai_projs/nanotorch/backend python -m uvicorn app.main:app --reload"
echo ""
echo "2. Frontend (in another terminal):"
echo "   cd /Users/yangyang/ai_projs/nanotorch/frontend"
echo "   npm run dev"
echo ""
echo "Then open http://localhost:5173 in your browser"
echo ""
