#!/bin/bash

# Cloud Run Deployment Script
# Usage: ./deploy.sh PROJECT_ID

PROJECT_ID=$1

if [ -z "$PROJECT_ID" ]; then
    echo "Usage: ./deploy.sh PROJECT_ID"
    exit 1
fi

echo "🚀 Deploying AI Agent Platform to Google Cloud Run"
echo "Project: $PROJECT_ID"

# Set project
gcloud config set project $PROJECT_ID

# Enable APIs
echo "📡 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build and deploy
echo "🔨 Building and deploying..."
gcloud builds submit --config cloudbuild.yaml

# Get service URL
SERVICE_URL=$(gcloud run services describe ai-agent-platform --region=us-central1 --format="value(status.url)")

echo "✅ Deployment complete!"
echo "🌐 Service URL: $SERVICE_URL"
echo "📊 Monitor: https://console.cloud.google.com/run"
echo ""
echo "🧪 To stress test:"
echo "python deployment/stress_test.py $SERVICE_URL"
