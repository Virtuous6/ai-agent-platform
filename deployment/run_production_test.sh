#!/bin/bash

# 🎯 AI Agent Platform - Production Test Runner
# Sets up environment and executes the $2.50 comprehensive test

echo "🚀 AI AGENT PLATFORM - PRODUCTION TEST SETUP"
echo "=============================================="

# Check for required environment variables
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ ERROR: OPENAI_API_KEY not set"
    echo "Please set your OpenAI API key:"
    echo "export OPENAI_API_KEY='your-api-key-here'"
    exit 1
fi

if [ -z "$SUPABASE_URL" ]; then
    echo "❌ ERROR: SUPABASE_URL not set"
    echo "Please set your Supabase URL:"
    echo "export SUPABASE_URL='https://your-project.supabase.co'"
    exit 1
fi

if [ -z "$SUPABASE_KEY" ]; then
    echo "❌ ERROR: SUPABASE_KEY not set"  
    echo "Please set your Supabase key:"
    echo "export SUPABASE_KEY='your-supabase-key'"
    exit 1
fi

echo "✅ Environment variables validated"

# Set test-specific environment variables
export MAX_GOAL_COST="${MAX_GOAL_COST:-2.50}"
export COST_ALERT_THRESHOLD="${COST_ALERT_THRESHOLD:-2.00}"
export AUTO_STOP_AT_BUDGET="${AUTO_STOP_AT_BUDGET:-true}"

echo "💰 Test Budget: $${MAX_GOAL_COST}"
echo "⚠️  Alert Threshold: $${COST_ALERT_THRESHOLD}"
echo "🛑 Auto-stop: ${AUTO_STOP_AT_BUDGET}"
echo ""

# Run the production test
echo "🎯 STARTING COMPREHENSIVE PRODUCTION TEST..."
echo "Expected duration: 10-15 minutes"
echo "Real OpenAI costs will be incurred!"
echo ""

read -p "Continue with production test? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Production test cancelled"
    exit 0
fi

echo "🚀 Executing production test..."
python3 production_test.py

echo ""
echo "🎯 Production test completed"
echo "Check the logs above for detailed results" 