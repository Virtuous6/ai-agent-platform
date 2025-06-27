#!/usr/bin/env python3
"""
Health Check Server for Google Cloud Run
Provides health and metrics endpoints for load balancer monitoring
"""

import os
import sys
import json
import asyncio
import threading
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flask import Flask, jsonify

# Import your platform components (with fallbacks)
try:
    import psutil
except ImportError:
    psutil = None

app = Flask(__name__)

# Global health state
health_state = {
    "status": "starting",
    "timestamp": datetime.utcnow().isoformat(),
    "uptime_seconds": 0,
    "checks": {}
}

start_time = time.time()

def update_health_state():
    """Update the global health state."""
    global health_state, start_time
    
    current_time = time.time()
    uptime = current_time - start_time
    
    checks = {}
    overall_healthy = True
    
    # System resource checks
    if psutil:
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            checks["system"] = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "healthy": cpu_percent < 90 and memory.percent < 90 and disk.percent < 90
            }
            
            if not checks["system"]["healthy"]:
                overall_healthy = False
                
        except Exception as e:
            checks["system"] = {"error": str(e), "healthy": False}
            overall_healthy = False
    else:
        checks["system"] = {"status": "psutil not available", "healthy": True}
    
    # Check if main components can be imported
    try:
        from orchestrator.agent_orchestrator import AgentOrchestrator
        checks["orchestrator"] = {"healthy": True, "status": "importable"}
    except Exception as e:
        checks["orchestrator"] = {"healthy": False, "error": str(e)}
        # Don't fail health check for import errors in cloud environment
    
    try:
        from events.event_bus import EventBus
        checks["event_bus"] = {"healthy": True, "status": "importable"}
    except Exception as e:
        checks["event_bus"] = {"healthy": False, "error": str(e)}
        # Don't fail health check for import errors in cloud environment
    
    try:
        from database.supabase_logger import SupabaseLogger
        checks["database"] = {"healthy": True, "status": "importable"}
    except Exception as e:
        checks["database"] = {"healthy": False, "error": str(e)}
        # Don't fail health check for import errors in cloud environment
    
    # Determine overall status
    if uptime < 60:  # Starting up
        status = "starting"
    elif overall_healthy:
        status = "healthy"
    else:
        status = "degraded"
    
    health_state = {
        "status": status,
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": int(uptime),
        "checks": checks,
        "environment": {
            "python_version": sys.version,
            "platform": sys.platform,
            "environment": os.getenv("ENVIRONMENT", "development")
        }
    }

@app.route('/health')
def health_check():
    """Health check endpoint for Cloud Run load balancer."""
    update_health_state()
    
    status_code = 200
    if health_state["status"] == "starting":
        status_code = 503  # Service Unavailable during startup
    elif health_state["status"] == "degraded":
        status_code = 503  # Service Unavailable when degraded
    
    return jsonify(health_state), status_code

@app.route('/health/live')
def liveness_probe():
    """Liveness probe - simple check that service is running."""
    return jsonify({
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": int(time.time() - start_time)
    }), 200

@app.route('/health/ready')
def readiness_probe():
    """Readiness probe - check if service is ready to accept traffic."""
    update_health_state()
    
    # Service is ready if it's healthy or just starting (but not degraded)
    ready = health_state["status"] in ["healthy", "starting"]
    
    status_code = 200 if ready else 503
    
    return jsonify({
        "status": "ready" if ready else "not_ready",
        "timestamp": datetime.utcnow().isoformat(),
        "health_status": health_state["status"]
    }), status_code

@app.route('/metrics')
def metrics():
    """Metrics endpoint for monitoring (Prometheus format)."""
    update_health_state()
    
    metrics_data = []
    
    # Add basic metrics
    metrics_data.append(f'# HELP ai_agent_uptime_seconds Total uptime in seconds')
    metrics_data.append(f'# TYPE ai_agent_uptime_seconds counter')
    metrics_data.append(f'ai_agent_uptime_seconds {health_state["uptime_seconds"]}')
    
    # Add system metrics if available
    if "system" in health_state["checks"] and "cpu_percent" in health_state["checks"]["system"]:
        system = health_state["checks"]["system"]
        
        metrics_data.append(f'# HELP ai_agent_cpu_percent CPU usage percentage')
        metrics_data.append(f'# TYPE ai_agent_cpu_percent gauge')
        metrics_data.append(f'ai_agent_cpu_percent {system["cpu_percent"]}')
        
        metrics_data.append(f'# HELP ai_agent_memory_percent Memory usage percentage')
        metrics_data.append(f'# TYPE ai_agent_memory_percent gauge')
        metrics_data.append(f'ai_agent_memory_percent {system["memory_percent"]}')
    
    # Add health status as metric
    health_status_value = 1 if health_state["status"] == "healthy" else 0
    metrics_data.append(f'# HELP ai_agent_healthy Health status (1=healthy, 0=unhealthy)')
    metrics_data.append(f'# TYPE ai_agent_healthy gauge')
    metrics_data.append(f'ai_agent_healthy {health_status_value}')
    
    return '\n'.join(metrics_data) + '\n', 200, {'Content-Type': 'text/plain'}

@app.route('/status')
def status():
    """Detailed status endpoint for debugging."""
    update_health_state()
    
    detailed_status = {
        **health_state,
        "request_info": {
            "method": "GET",
            "path": "/status",
            "user_agent": request.headers.get("User-Agent", ""),
            "remote_addr": request.remote_addr
        },
        "environment_vars": {
            "PORT": os.getenv("PORT", "Not set"),
            "ENVIRONMENT": os.getenv("ENVIRONMENT", "Not set"),
            "LOG_LEVEL": os.getenv("LOG_LEVEL", "Not set")
        }
    }
    
    return jsonify(detailed_status), 200

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "error": "Not found",
        "message": "The requested endpoint does not exist",
        "available_endpoints": [
            "/health",
            "/health/live", 
            "/health/ready",
            "/metrics",
            "/status"
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred",
        "timestamp": datetime.utcnow().isoformat()
    }), 500

def run_health_server():
    """Run the health check server."""
    port = int(os.getenv("PORT", 8080))
    host = "0.0.0.0"
    
    print(f"🏥 Starting health check server on {host}:{port}")
    print(f"📊 Health endpoint: http://{host}:{port}/health")
    print(f"📈 Metrics endpoint: http://{host}:{port}/metrics")
    
    try:
        app.run(
            host=host,
            port=port,
            debug=False,
            threaded=True,
            use_reloader=False
        )
    except Exception as e:
        print(f"❌ Failed to start health server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    run_health_server() 