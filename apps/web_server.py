#!/usr/bin/env python3
"""
Web Server for AI Agent Platform on Cloud Run
Provides web endpoints for the Slack bot and health monitoring
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from flask import Flask, request, jsonify
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global health state
start_time = datetime.utcnow()

@app.route('/', methods=['GET'])
def root():
    """Root endpoint."""
    return jsonify({
        "service": "AI Agent Platform",
        "status": "running",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": [
            "/health",
            "/slack/events",
            "/metrics",
            "/status"
        ]
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for Cloud Run."""
    try:
        # Basic health checks
        uptime = (datetime.utcnow() - start_time).total_seconds()
        
        health_data = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": int(uptime),
            "service": "ai-agent-platform",
            "version": "1.0.0"
        }
        
        # Add system metrics if available
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            health_data["system"] = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "available_memory_mb": memory.available // 1024 // 1024
            }
            
            # Consider unhealthy if resources are very high
            if cpu_percent > 95 or memory.percent > 95:
                health_data["status"] = "degraded"
                
        except Exception as e:
            logger.warning(f"Could not get system metrics: {e}")
        
        status_code = 200 if health_data["status"] == "healthy" else 503
        return jsonify(health_data), status_code
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 503

@app.route('/slack/events', methods=['POST'])
def slack_events():
    """Handle Slack events (for stress testing)."""
    try:
        data = request.get_json()
        
        # Handle URL verification challenge
        if data and data.get('type') == 'url_verification':
            return jsonify({"challenge": data.get('challenge')})
        
        # Log the event for stress testing analysis
        logger.info(f"Received Slack event: {data.get('type', 'unknown')}")
        
        # Simple response for stress testing
        response = {
            "status": "processed",
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": data.get('type', 'unknown') if data else 'no_data',
            "agent_response": "Hello! AI Agent Platform is running on Cloud Run."
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error processing Slack event: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus-style metrics endpoint."""
    try:
        uptime = (datetime.utcnow() - start_time).total_seconds()
        
        metrics_lines = [
            "# HELP ai_agent_uptime_seconds Total uptime in seconds",
            "# TYPE ai_agent_uptime_seconds counter",
            f"ai_agent_uptime_seconds {int(uptime)}",
            "",
            "# HELP ai_agent_requests_total Total number of requests",
            "# TYPE ai_agent_requests_total counter",
            "ai_agent_requests_total 1",  # This would be tracked properly in a real implementation
            "",
            "# HELP ai_agent_healthy Health status (1=healthy, 0=unhealthy)",
            "# TYPE ai_agent_healthy gauge",
            "ai_agent_healthy 1"
        ]
        
        # Add system metrics if available
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            metrics_lines.extend([
                "",
                "# HELP ai_agent_cpu_percent CPU usage percentage",
                "# TYPE ai_agent_cpu_percent gauge",
                f"ai_agent_cpu_percent {cpu_percent}",
                "",
                "# HELP ai_agent_memory_percent Memory usage percentage", 
                "# TYPE ai_agent_memory_percent gauge",
                f"ai_agent_memory_percent {memory.percent}"
            ])
        except Exception:
            pass
        
        return '\n'.join(metrics_lines) + '\n', 200, {'Content-Type': 'text/plain'}
        
    except Exception as e:
        logger.error(f"Error generating metrics: {e}")
        return "# Error generating metrics\n", 500, {'Content-Type': 'text/plain'}

@app.route('/status', methods=['GET'])
def status():
    """Detailed status endpoint."""
    try:
        uptime = (datetime.utcnow() - start_time).total_seconds()
        
        status_data = {
            "service": "AI Agent Platform",
            "status": "running",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": int(uptime),
            "environment": {
                "python_version": sys.version,
                "platform": sys.platform,
                "environment": os.getenv("ENVIRONMENT", "production"),
                "port": os.getenv("PORT", "8080")
            }
        }
        
        # Add system info if available
        try:
            import platform
            status_data["system"] = {
                "cpu_count": os.cpu_count(),
                "platform": platform.platform(),
                "python_implementation": platform.python_implementation()
            }
            
            # Add resource usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            status_data["resources"] = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "total_memory_mb": memory.total // 1024 // 1024,
                "available_memory_mb": memory.available // 1024 // 1024
            }
        except Exception as e:
            status_data["system_error"] = str(e)
        
        return jsonify(status_data), 200
        
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "error": "Not found",
        "message": "The requested endpoint does not exist",
        "available_endpoints": ["/", "/health", "/slack/events", "/metrics", "/status"],
        "timestamp": datetime.utcnow().isoformat()
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred",
        "timestamp": datetime.utcnow().isoformat()
    }), 500

if __name__ == '__main__':
    # Get port from environment (Cloud Run sets this)
    port = int(os.getenv('PORT', 8080))
    host = '0.0.0.0'
    
    logger.info(f"🚀 Starting AI Agent Platform web server")
    logger.info(f"🌐 Listening on {host}:{port}")
    logger.info(f"📊 Health endpoint: http://{host}:{port}/health")
    logger.info(f"📈 Metrics endpoint: http://{host}:{port}/metrics")
    logger.info(f"🤖 Slack events: http://{host}:{port}/slack/events")
    
    try:
        app.run(
            host=host,
            port=port,
            debug=False,
            threaded=True
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1) 