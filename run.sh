#!/bin/bash

# run.sh — Start both Backend (FastAPI) and Frontend (Streamlit)

# Colors for logging
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}⚖️  Starting Consumer Law Agentic RAG system...${NC}"

# 1. Activate virtual environment
if [ -d "venv" ]; then
    echo -e "${GREEN}✓ Activating virtual environment...${NC}"
    source venv/bin/activate
else
    echo -e "${BLUE}⚠️  venv not found. Running with system python...${NC}"
fi

# 2. Function to kill background processes on exit
cleanup() {
    echo -e "\n${BLUE}🛑 Stopping services...${NC}"
    kill $BACKEND_PID $FRONTEND_PID
    exit
}
trap cleanup SIGINT

# 3. Start Backend (FastAPI)
echo -e "${GREEN}🚀 Starting Backend (FastAPI) on port 8000...${NC}"
python -m uvicorn app:app --reload --port 8000 > backend.log 2>&1 &
BACKEND_PID=$!

# 4. Wait for backend to be ready
echo -e "${BLUE}⏳ Waiting for backend to initialize...${NC}"
until $(curl --output /dev/null --silent --fail http://localhost:8000/health); do
    printf '.'
    sleep 1
done
echo -e "\n${GREEN}✓ Backend is UP!${NC}"

# 5. Start Frontend (Streamlit)
echo -e "${GREEN}🎨 Starting Frontend (Streamlit) on port 8501...${NC}"
streamlit run frontend/streamlit_app.py --server.port 8501
FRONTEND_PID=$!

# Wait for processes
wait $BACKEND_PID $FRONTEND_PID
