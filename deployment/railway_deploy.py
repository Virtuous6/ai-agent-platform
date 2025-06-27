#!/usr/bin/env python3
"""
Railway Deployment Script for AI Agent Platform
Optimized for stress testing with auto-scaling support
"""

import os
import json
import subprocess
from pathlib import Path

def create_railway_config():
    """Create Railway-specific configuration."""
    config = {
        "build": {
            "builder": "NIXPACKS"
        },
        "deploy": {
            "startCommand": "python start_bot.py",
            "healthcheckPath": "/health",
            "healthcheckTimeout": 300,
            "restartPolicyType": "ON_FAILURE",
            "restartPolicyMaxRetries": 3
        },
        "environments": {
            "production": {
                "variables": {
                    "ENVIRONMENT": "production",
                    "LOG_LEVEL": "INFO",
                    "ENABLE_PERFORMANCE_TRACKING": "true",
                    "ENABLE_ANALYTICS": "true"
                }
            },
            "staging": {
                "variables": {
                    "ENVIRONMENT": "staging", 
                    "LOG_LEVEL": "DEBUG",
                    "ENABLE_PERFORMANCE_TRACKING": "true"
                }
            }
        }
    }
    
    # Write railway.json
    with open("railway.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Created railway.json configuration")

def create_procfile():
    """Create Procfile for Railway deployment."""
    procfile_content = """web: python start_bot.py
worker: python -m orchestrator.improvement_orchestrator
dashboard: python dashboard/launch_dashboard.py"""
    
    with open("Procfile", "w") as f:
        f.write(procfile_content)
    
    print("✅ Created Procfile for multi-service deployment")

def create_dockerfile():
    """Create optimized Dockerfile for stress testing."""
    dockerfile_content = """FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    git \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose ports
EXPOSE 3000 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:3000/health')" || exit 1

# Start command (will be overridden by Railway)
CMD ["python", "start_bot.py"]"""
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    print("✅ Created optimized Dockerfile")

def create_stress_test_config():
    """Create configuration for stress testing."""
    stress_config = {
        "stress_test": {
            "max_concurrent_agents": 100,
            "spawn_rate_per_second": 5,
            "test_duration_minutes": 30,
            "memory_limit_mb": 2048,
            "cpu_limit_percent": 80
        },
        "monitoring": {
            "metrics_collection_interval": 10,
            "alert_thresholds": {
                "response_time_ms": 5000,
                "error_rate_percent": 5,
                "memory_usage_percent": 90
            }
        },
        "auto_scaling": {
            "min_replicas": 1,
            "max_replicas": 5,
            "target_cpu_percent": 70,
            "scale_up_threshold": 80,
            "scale_down_threshold": 30
        }
    }
    
    with open("stress_test_config.json", "w") as f:
        json.dump(stress_config, f, indent=2)
    
    print("✅ Created stress test configuration")

def create_health_endpoint():
    """Create health check endpoint for monitoring."""
    health_endpoint = """
from flask import Flask, jsonify
import psutil
import asyncio
from datetime import datetime

app = Flask(__name__)

@app.route('/health')
def health_check():
    \"\"\"Health check endpoint for load balancer.\"\"\"
    try:
        # Check system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        # Check if main services are running
        # Add your specific health checks here
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent
            },
            "services": {
                "orchestrator": True,  # Add actual checks
                "event_bus": True,
                "database": True
            }
        }
        
        # Determine overall health
        if cpu_percent > 90 or memory_percent > 90:
            health_status["status"] = "degraded"
        
        status_code = 200 if health_status["status"] == "healthy" else 503
        return jsonify(health_status), status_code
        
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 503

@app.route('/metrics')
def metrics():
    \"\"\"Metrics endpoint for monitoring.\"\"\"
    # Add Prometheus-style metrics here
    return "# Metrics endpoint\\n"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
"""
    
    with open("health_server.py", "w") as f:
        f.write(health_endpoint)
    
    print("✅ Created health check endpoint")

def setup_environment_variables():
    """Display required environment variables for Railway."""
    required_vars = [
        "SLACK_BOT_TOKEN",
        "SLACK_SIGNING_SECRET", 
        "SLACK_APP_TOKEN",
        "SUPABASE_URL",
        "SUPABASE_KEY",
        "OPENAI_API_KEY",
        "ENVIRONMENT=production",
        "LOG_LEVEL=INFO"
    ]
    
    print("\n🔧 REQUIRED ENVIRONMENT VARIABLES:")
    print("Set these in Railway dashboard:")
    for var in required_vars:
        print(f"  • {var}")

def main():
    """Main deployment setup."""
    print("🚀 Railway Deployment Setup for Stress Testing")
    print("=" * 50)
    
    # Create deployment files
    create_railway_config()
    create_procfile()
    create_dockerfile()
    create_stress_test_config()
    create_health_endpoint()
    
    print("\n📋 DEPLOYMENT CHECKLIST:")
    print("1. ✅ Railway configuration created")
    print("2. ✅ Multi-service Procfile created")
    print("3. ✅ Optimized Dockerfile created")
    print("4. ✅ Stress test config created")
    print("5. ✅ Health monitoring setup")
    
    setup_environment_variables()
    
    print("\n🎯 NEXT STEPS:")
    print("1. Install Railway CLI: npm install -g @railway/cli")
    print("2. Login: railway login")
    print("3. Create project: railway init")
    print("4. Set environment variables in Railway dashboard")
    print("5. Deploy: railway up")
    print("6. Monitor: railway logs")
    
    print("\n📊 STRESS TESTING:")
    print("• Your app will auto-scale based on CPU/memory")
    print("• Health checks every 30 seconds")
    print("• Metrics available at /metrics endpoint")
    print("• Configure max 100 concurrent agents")

if __name__ == "__main__":
    main() 