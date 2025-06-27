#!/bin/bash

echo "🚀 Starting AI Agent Platform services..."

# Set environment variables for Cloud Run
export PYTHONUNBUFFERED=1
export ENVIRONMENT=${ENVIRONMENT:-production}
export PORT=${PORT:-8080}

# Start health check server in background
echo "📊 Starting health server on port $PORT..."
python deployment/health_server.py &
HEALTH_PID=$!

# Give health server a moment to start
sleep 2

# Start main Slack bot application
echo "🤖 Starting Slack bot with AI agents..."
python start_bot.py &
BOT_PID=$!

# Function to cleanup on exit
cleanup() {
    echo "🛑 Shutting down services..."
    kill $HEALTH_PID 2>/dev/null
    kill $BOT_PID 2>/dev/null
    exit 0
}

# Handle termination signals
trap cleanup SIGTERM SIGINT

# Wait for either process to exit
wait $HEALTH_PID $BOT_PID