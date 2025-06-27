#!/usr/bin/env python3
"""
Google Cloud Run Deployment Script for AI Agent Platform
Optimized for serverless auto-scaling and stress testing
"""

import os
import json
import yaml
import subprocess
from pathlib import Path

def create_cloudbuild_config():
    """Create Cloud Build configuration for CI/CD."""
    cloudbuild_config = {
        "steps": [
            {
                "name": "gcr.io/cloud-builders/docker",
                "args": [
                    "build",
                    "-t", "gcr.io/$PROJECT_ID/ai-agent-platform:$COMMIT_SHA",
                    "-t", "gcr.io/$PROJECT_ID/ai-agent-platform:latest",
                    "."
                ]
            },
            {
                "name": "gcr.io/cloud-builders/docker", 
                "args": ["push", "gcr.io/$PROJECT_ID/ai-agent-platform:$COMMIT_SHA"]
            },
            {
                "name": "gcr.io/cloud-builders/docker",
                "args": ["push", "gcr.io/$PROJECT_ID/ai-agent-platform:latest"]
            },
            {
                "name": "gcr.io/google.com/cloudsdktool/cloud-sdk",
                "entrypoint": "gcloud",
                "args": [
                    "run", "deploy", "ai-agent-platform",
                    "--image", "gcr.io/$PROJECT_ID/ai-agent-platform:$COMMIT_SHA",
                    "--region", "us-central1",
                    "--platform", "managed",
                    "--allow-unauthenticated"
                ]
            }
        ],
        "images": [
            "gcr.io/$PROJECT_ID/ai-agent-platform:$COMMIT_SHA",
            "gcr.io/$PROJECT_ID/ai-agent-platform:latest"
        ]
    }
    
    with open("cloudbuild.yaml", "w") as f:
        yaml.dump(cloudbuild_config, f, default_flow_style=False)
    
    print("✅ Created cloudbuild.yaml for CI/CD")

def create_cloud_run_dockerfile():
    """Create optimized Dockerfile for Cloud Run."""
    dockerfile_content = """# Use the official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8080

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    git \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add Flask for health checks
RUN pip install flask psutil

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE $PORT

# Health check endpoint
COPY deployment/health_server.py .

# Start script that runs multiple services
COPY deployment/start_services.sh .
RUN chmod +x deployment/start_services.sh

# Use start script as entrypoint
CMD ["./deployment/start_services.sh"]"""
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    print("✅ Created Cloud Run optimized Dockerfile")

def create_cloud_run_service_yaml():
    """Create Cloud Run service configuration."""
    service_config = {
        "apiVersion": "serving.knative.dev/v1",
        "kind": "Service",
        "metadata": {
            "name": "ai-agent-platform",
            "annotations": {
                "run.googleapis.com/ingress": "all",
                "run.googleapis.com/execution-environment": "gen2"
            }
        },
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                        "autoscaling.knative.dev/maxScale": "100",
                        "autoscaling.knative.dev/minScale": "0",
                        "run.googleapis.com/cpu-throttling": "false",
                        "run.googleapis.com/memory": "2Gi",
                        "run.googleapis.com/cpu": "2",
                        "run.googleapis.com/timeout": "900s"
                    }
                },
                "spec": {
                    "containerConcurrency": 80,
                    "containers": [
                        {
                            "image": "gcr.io/PROJECT_ID/ai-agent-platform:latest",
                            "ports": [
                                {
                                    "containerPort": 8080
                                }
                            ],
                            "env": [
                                {
                                    "name": "ENVIRONMENT",
                                    "value": "production"
                                },
                                {
                                    "name": "LOG_LEVEL", 
                                    "value": "INFO"
                                },
                                {
                                    "name": "ENABLE_PERFORMANCE_TRACKING",
                                    "value": "true"
                                },
                                {
                                    "name": "MAX_CONCURRENT_AGENTS",
                                    "value": "100"
                                }
                            ],
                            "resources": {
                                "limits": {
                                    "cpu": "2",
                                    "memory": "2Gi"
                                }
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 8080
                                },
                                "initialDelaySeconds": 60,
                                "periodSeconds": 30
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/health", 
                                    "port": 8080
                                },
                                "initialDelaySeconds": 10,
                                "periodSeconds": 5
                            }
                        }
                    ]
                }
            }
        }
    }
    
    with open("cloud-run-service.yaml", "w") as f:
        yaml.dump(service_config, f, default_flow_style=False)
    
    print("✅ Created Cloud Run service configuration")

def create_start_services_script():
    """Create startup script for multiple services."""
    start_script = """#!/bin/bash

# Start health check server in background
python health_server.py &

# Start main application
exec python start_bot.py"""
    
    Path("deployment").mkdir(exist_ok=True)
    with open("deployment/start_services.sh", "w") as f:
        f.write(start_script)
    
    print("✅ Created multi-service startup script")

def create_stress_test_script():
    """Create stress testing script for Cloud Run."""
    stress_test_script = """#!/usr/bin/env python3
\"\"\"
Cloud Run Stress Test Script
Tests auto-scaling capabilities of the AI agent platform
\"\"\"

import asyncio
import aiohttp
import time
import json
from concurrent.futures import ThreadPoolExecutor

class CloudRunStressTester:
    def __init__(self, service_url: str):
        self.service_url = service_url.rstrip('/')
        self.results = []
    
    async def send_request(self, session, request_id):
        \"\"\"Send a single test request.\"\"\"
        start_time = time.time()
        try:
            async with session.post(
                f"{self.service_url}/slack/events",
                json={
                    "type": "event_callback",
                    "event": {
                        "type": "app_mention",
                        "text": f"<@bot> stress test request {request_id}",
                        "user": "stress_test_user",
                        "channel": "stress_test"
                    }
                },
                timeout=30
            ) as response:
                end_time = time.time()
                
                result = {
                    "request_id": request_id,
                    "status_code": response.status,
                    "response_time": end_time - start_time,
                    "success": response.status == 200
                }
                
                self.results.append(result)
                return result
                
        except Exception as e:
            end_time = time.time()
            result = {
                "request_id": request_id,
                "status_code": 0,
                "response_time": end_time - start_time,
                "success": False,
                "error": str(e)
            }
            self.results.append(result)
            return result
    
    async def run_stress_test(self, total_requests=1000, concurrent_requests=50):
        \"\"\"Run the stress test.\"\"\"
        print(f"🚀 Starting stress test against {self.service_url}")
        print(f"📊 Total requests: {total_requests}")
        print(f"⚡ Concurrent: {concurrent_requests}")
        
        async with aiohttp.ClientSession() as session:
            # Create semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(concurrent_requests)
            
            async def limited_request(request_id):
                async with semaphore:
                    return await self.send_request(session, request_id)
            
            # Send all requests
            tasks = [limited_request(i) for i in range(total_requests)]
            await asyncio.gather(*tasks)
        
        self.analyze_results()
    
    def analyze_results(self):
        \"\"\"Analyze stress test results.\"\"\"
        if not self.results:
            print("❌ No results to analyze")
            return
        
        successful = [r for r in self.results if r["success"]]
        failed = [r for r in self.results if not r["success"]]
        
        response_times = [r["response_time"] for r in successful]
        
        print("\\n📈 STRESS TEST RESULTS:")
        print(f"Total Requests: {len(self.results)}")
        print(f"Successful: {len(successful)} ({len(successful)/len(self.results)*100:.1f}%)")
        print(f"Failed: {len(failed)} ({len(failed)/len(self.results)*100:.1f}%)")
        
        if response_times:
            print(f"\\n⏱️  RESPONSE TIMES:")
            print(f"Average: {sum(response_times)/len(response_times):.2f}s")
            print(f"Min: {min(response_times):.2f}s")
            print(f"Max: {max(response_times):.2f}s")
        
        # Save detailed results
        with open("stress_test_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        print("\\n💾 Detailed results saved to stress_test_results.json")

async def main():
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python stress_test.py <cloud-run-service-url>")
        sys.exit(1)
    
    service_url = sys.argv[1]
    tester = CloudRunStressTester(service_url)
    
    # Run stress test
    await tester.run_stress_test(
        total_requests=500,
        concurrent_requests=25
    )

if __name__ == "__main__":
    asyncio.run(main())"""
    
    with open("deployment/stress_test.py", "w") as f:
        f.write(stress_test_script)
    
    print("✅ Created Cloud Run stress test script")

def create_deploy_script():
    """Create deployment script."""
    deploy_script = """#!/bin/bash

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
"""
    
    with open("deployment/deploy.sh", "w") as f:
        f.write(deploy_script)
    
    # Make executable
    os.chmod("deployment/deploy.sh", 0o755)
    print("✅ Created deployment script")

def display_setup_instructions():
    """Display setup instructions."""
    print("""
🔧 GOOGLE CLOUD RUN SETUP:

1. PREREQUISITES:
   • Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install
   • Create GCP project: https://console.cloud.google.com/
   • Enable billing on the project

2. AUTHENTICATION:
   gcloud auth login
   gcloud auth configure-docker

3. SET ENVIRONMENT VARIABLES:
   • Go to Cloud Run console
   • Set these environment variables:
     - SLACK_BOT_TOKEN
     - SLACK_SIGNING_SECRET
     - SLACK_APP_TOKEN
     - SUPABASE_URL
     - SUPABASE_KEY
     - OPENAI_API_KEY

4. DEPLOY:
   ./deployment/deploy.sh YOUR_PROJECT_ID

5. STRESS TEST:
   python deployment/stress_test.py https://YOUR_SERVICE_URL

📊 CLOUD RUN ADVANTAGES FOR STRESS TESTING:
• Auto-scales 0→100+ instances in seconds
• Pay only for actual usage (great for LLM costs)
• Built-in load balancing
• Real-time monitoring in GCP console
• Traffic splitting for A/B testing agents
• Global deployment options
""")

def main():
    """Main deployment setup for Google Cloud Run."""
    print("☁️  Google Cloud Run Deployment Setup")
    print("=" * 50)
    
    # Create deployment directory
    Path("deployment").mkdir(exist_ok=True)
    
    # Create all deployment files
    create_cloud_run_dockerfile()
    create_cloudbuild_config()
    create_cloud_run_service_yaml()
    create_start_services_script()
    create_stress_test_script()
    create_deploy_script()
    
    print("\n✅ DEPLOYMENT FILES CREATED:")
    print("• Dockerfile - Optimized for Cloud Run")
    print("• cloudbuild.yaml - CI/CD pipeline")
    print("• cloud-run-service.yaml - Service configuration")
    print("• deployment/start_services.sh - Multi-service startup")
    print("• deployment/stress_test.py - Stress testing tool")
    print("• deployment/deploy.sh - One-click deployment")
    
    display_setup_instructions()

if __name__ == "__main__":
    main() 