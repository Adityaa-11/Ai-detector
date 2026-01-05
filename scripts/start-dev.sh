#!/bin/bash

# Start both frontend and backend for development

echo "🚀 Starting AI Detection Development Environment"
echo "================================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is required but not installed."
    exit 1
fi

# Check if node/pnpm is available
if ! command -v pnpm &> /dev/null; then
    echo "❌ pnpm is required but not installed."
    echo "   Install with: npm install -g pnpm"
    exit 1
fi

# Start backend in background
echo ""
echo "📦 Starting Python backend on port 8000..."
cd backend

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "   Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv and install deps
source venv/bin/activate
pip install -q -r requirements.txt

# Start backend
uvicorn api:app --reload --port 8000 &
BACKEND_PID=$!

cd ..

# Give backend time to start
sleep 3

# Start frontend
echo ""
echo "🌐 Starting Next.js frontend on port 3000..."
pnpm dev &
FRONTEND_PID=$!

echo ""
echo "================================================"
echo "✅ Development servers started!"
echo ""
echo "   Frontend: http://localhost:3000"
echo "   Backend:  http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both servers"
echo "================================================"

# Wait for Ctrl+C
trap "echo ''; echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" SIGINT
wait

