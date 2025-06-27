#!/usr/bin/env python3
"""
Web Dashboard for AI Agent Platform

Provides a beautiful web interface for monitoring the self-improving AI system
with real-time data from Supabase.
"""

import os
import sys
import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flask import Flask, render_template, jsonify, request
from dashboard.components.supabase_data_viewer import SupabaseDataViewer
from database.supabase_logger import SupabaseLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, 
            template_folder='../dashboard/templates',
            static_folder='../dashboard/static')

# Global components
db_logger = None
supabase_viewer = None

def initialize_components():
    """Initialize dashboard components."""
    global db_logger, supabase_viewer
    
    try:
        db_logger = SupabaseLogger()
        supabase_viewer = SupabaseDataViewer(db_logger=db_logger)
        logger.info("Dashboard components initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        return False

@app.route('/')
def dashboard_home():
    """Main dashboard page."""
    return render_template('dashboard.html')

# Async route wrapper
def async_route(f):
    """Decorator to handle async routes in Flask."""
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(f(*args, **kwargs))
        finally:
            loop.close()
    wrapper.__name__ = f.__name__
    return wrapper

@app.route('/api/overview')
@async_route
async def api_overview():
    """Get overview data for the dashboard."""
    try:
        if not supabase_viewer:
            return jsonify({"error": "Dashboard not initialized"}), 500
        
        # Get cost data
        cost_data = await supabase_viewer.get_cost_analytics_data()
        
        # Get basic counts
        messages_data = await supabase_viewer.get_messages_data(5)
        workflows_data = await supabase_viewer.get_workflow_runs_data(5)
        mcp_data = await supabase_viewer.get_mcp_connections_data()
        conversations_data = await supabase_viewer.get_conversations_data(5)
        
        overview = {
            "system_health": {
                "overall_score": 0.85,
                "status": "healthy"
            },
            "costs": {
                "today_cost": cost_data.get("today_cost", 0.0),
                "weekly_cost": cost_data.get("weekly_cost", 0.0),
                "monthly_projection": cost_data.get("monthly_projection", 0.0),
                "efficiency_score": cost_data.get("efficiency_score", 0.78)
            },
            "activity": {
                "total_messages": messages_data.get("total_count", 0),
                "active_workflows": len([w for w in workflows_data.get("workflows", []) if w.get("status") == "running"]),
                "active_mcps": mcp_data.get("active_count", 0),
                "active_conversations": conversations_data.get("active_count", 0)
            },
            "recent_activity": {
                "messages": messages_data.get("messages", [])[:5],
                "workflows": workflows_data.get("workflows", [])[:3]
            }
        }
        
        return jsonify(overview)
        
    except Exception as e:
        logger.error(f"Error getting overview data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/messages')
@async_route
async def api_messages():
    """Get messages data."""
    try:
        if not supabase_viewer:
            return jsonify({"error": "Dashboard not initialized"}), 500
        
        limit = request.args.get('limit', 20, type=int)
        data = await supabase_viewer.get_messages_data(limit)
        return jsonify(data)
        
    except Exception as e:
        logger.error(f"Error getting messages data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/workflows')
@async_route
async def api_workflows():
    """Get workflow runs data."""
    try:
        if not supabase_viewer:
            return jsonify({"error": "Dashboard not initialized"}), 500
        
        limit = request.args.get('limit', 15, type=int)
        data = await supabase_viewer.get_workflow_runs_data(limit)
        return jsonify(data)
        
    except Exception as e:
        logger.error(f"Error getting workflows data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/mcp-connections')
@async_route
async def api_mcp_connections():
    """Get MCP connections data."""
    try:
        if not supabase_viewer:
            return jsonify({"error": "Dashboard not initialized"}), 500
        
        data = await supabase_viewer.get_mcp_connections_data()
        return jsonify(data)
        
    except Exception as e:
        logger.error(f"Error getting MCP connections data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/conversations')
@async_route
async def api_conversations():
    """Get conversations data."""
    try:
        if not supabase_viewer:
            return jsonify({"error": "Dashboard not initialized"}), 500
        
        limit = request.args.get('limit', 15, type=int)
        data = await supabase_viewer.get_conversations_data(limit)
        return jsonify(data)
        
    except Exception as e:
        logger.error(f"Error getting conversations data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/costs')
@async_route
async def api_costs():
    """Get cost analytics data."""
    try:
        if not supabase_viewer:
            return jsonify({"error": "Dashboard not initialized"}), 500
        
        data = await supabase_viewer.get_cost_analytics_data()
        return jsonify(data)
        
    except Exception as e:
        logger.error(f"Error getting cost data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/mcp-types')
@async_route
async def api_mcp_types():
    """Get available MCP types for creation."""
    try:
        if not supabase_viewer:
            return jsonify({"error": "Dashboard not initialized"}), 500
        
        types = await supabase_viewer.get_available_mcp_types()
        return jsonify({"mcp_types": types})
        
    except Exception as e:
        logger.error(f"Error getting MCP types: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/mcp-connections', methods=['POST'])
@async_route
async def api_create_mcp_connection():
    """Create a new MCP connection."""
    try:
        if not supabase_viewer:
            return jsonify({"error": "Dashboard not initialized"}), 500
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        result = await supabase_viewer.add_mcp_connection(data)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error creating MCP connection: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    try:
        status = "healthy" if supabase_viewer else "degraded"
        return jsonify({
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
            "service": "ai-agent-web-dashboard",
            "supabase_connected": supabase_viewer is not None
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 503

if __name__ == '__main__':
    print("🚀 AI Agent Platform - Web Dashboard")
    print("=" * 50)
    print()
    
    # Initialize components
    print("🔧 Initializing dashboard components...")
    if initialize_components():
        print("✅ Components initialized successfully!")
    else:
        print("⚠️  Some components failed to initialize")
        print("   Dashboard will work with limited functionality")
    
    print()
    print("🌐 Starting web dashboard server...")
    
    # Get port from environment
    port = int(os.getenv('PORT', 5000))
    host = '0.0.0.0' if os.getenv('ENVIRONMENT') == 'production' else '127.0.0.1'
    
    print(f"📊 Dashboard URL: http://{host}:{port}")
    print(f"🏥 Health check: http://{host}:{port}/health")
    print()
    print("🎮 Features:")
    print("  • Real-time Supabase data")
    print("  • Messages, workflows, MCP connections")
    print("  • Cost analytics and monitoring")
    print("  • Interactive MCP management")
    print()
    
    try:
        app.run(
            host=host,
            port=port,
            debug=False,
            threaded=True
        )
    except Exception as e:
        logger.error(f"Failed to start web dashboard: {e}")
        sys.exit(1) 