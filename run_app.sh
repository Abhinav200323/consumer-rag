#!/bin/bash

# run_app.sh — Start Backend (FastAPI) and React Frontend (Lex Assist)

# Colors for logging
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}⚖️  Starting Lex Assist System...${NC}"

# 1. Activate virtual environment and install dependencies
if [ -d "venv" ]; then
    echo -e "${GREEN}✓ Activating virtual environment...${NC}"
    source venv/bin/activate
    echo -e "${GREEN}✓ Installing Python dependencies...${NC}"
    pip install -r requirements.txt
else
    echo -e "${RED}❌ venv not found. Please create it first using 'python -m venv venv'${NC}"
    exit 1
fi

# 2. Cleanup function
cleanup() {
    echo -e "\n${BLUE}🛑 Stopping services...${NC}"
    kill $BACKEND_PID
    # Find and kill the npm/vite process
    lsof -ti:5173 | xargs kill -9 2>/dev/null
    exit
}
trap cleanup SIGINT

# 3. Start Backend (FastAPI)
echo -e "${GREEN}🚀 Starting Backend (FastAPI) on port 8000...${NC}"
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
BACKEND_PID=$!

# 4. Wait for backend to be ready
echo -e "${BLUE}⏳ Waiting for backend to initialize...${NC}"
MAX_RETRIES=30
COUNT=0
until $(curl --output /dev/null --silent --fail http://localhost:8000/health); do
    printf '.'
    sleep 1
    COUNT=$((COUNT+1))
    if [ $COUNT -ge $MAX_RETRIES ]; then
        echo -e "\n${RED}❌ Backend failed to start. Check backend.log for details.${NC}"
        kill $BACKEND_PID
        exit 1
    fi
done
echo -e "\n${GREEN}✓ Backend is UP!${NC}"

# 5. Start Frontend (React/Vite)
echo -e "${GREEN}🎨 Starting Frontend (Lex Assist) on port 5173...${NC}"
cd lex-assist
if [ ! -d "node_modules" ]; then
    echo -e "${BLUE}📦 Installing node_modules...${NC}"
    npm install
fi
npm run dev
