"""
Terminal-style Dashboard for AI Agent Platform

Provides a retro terminal interface for monitoring the self-improving AI system.
Features ASCII art, color coding, and real-time data updates.
"""

import asyncio
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
import json
import logging

try:
    import curses
    import threading
    CURSES_AVAILABLE = True
except ImportError:
    CURSES_AVAILABLE = False

from .styles.colors import TerminalColors
from .styles.ascii_art import ASCII_ART
from .utils.real_time_updater import RealTimeUpdater
from .utils.data_formatter import DataFormatter

logger = logging.getLogger(__name__)

class TerminalDashboard:
    """
    Retro terminal-style dashboard for monitoring AI agent platform.
    
    Features:
    - Real-time system monitoring
    - ASCII art headers
    - Color-coded status indicators
    - Modular component display
    - Keyboard navigation
    """
    
    def __init__(self, db_logger=None, orchestrator=None, event_bus=None):
        """Initialize the terminal dashboard."""
        self.db_logger = db_logger
        self.orchestrator = orchestrator
        self.event_bus = event_bus
        
        # Terminal state
        self.screen = None
        self.running = False
        self.current_view = "overview"
        self.refresh_rate = 2.0  # seconds
        
        # Dashboard components
        self.components = {}
        self.data_cache = {}
        self.last_update = {}
        
        # Initialize components
        self._initialize_components()
        
        # Real-time data updater
        self.updater = RealTimeUpdater(
            db_logger=db_logger,
            orchestrator=orchestrator,
            event_bus=event_bus
        )
        
        # Terminal styling
        self.colors = TerminalColors()
        self.formatter = DataFormatter()
        
        logger.info("Terminal Dashboard initialized")
    
    def _initialize_components(self):
        """Initialize dashboard components."""
        try:
            from .components.system_health import SystemHealthMonitor
            from .components.agent_monitor import AgentMonitor
            from .components.cost_analytics import CostAnalytics
            from .components.event_stream import EventStreamViewer
            from .components.logs_viewer import LogsViewer
            
            self.components = {
                "health": SystemHealthMonitor(
                    db_logger=self.db_logger,
                    orchestrator=self.orchestrator
                ),
                "agents": AgentMonitor(
                    db_logger=self.db_logger,
                    orchestrator=self.orchestrator
                ),
                "costs": CostAnalytics(
                    db_logger=self.db_logger
                ),
                "events": EventStreamViewer(
                    event_bus=self.event_bus
                ),
                "logs": LogsViewer(
                    db_logger=self.db_logger
                )
            }
            
        except ImportError as e:
            logger.warning(f"Could not initialize all components: {e}")
            self.components = {}
    
    def start(self):
        """Start the terminal dashboard."""
        if not CURSES_AVAILABLE:
            logger.error("Curses not available - falling back to simple output")
            asyncio.run(self._simple_dashboard())
            return
        
        try:
            # Initialize curses
            curses.wrapper(self._main_loop)
        except KeyboardInterrupt:
            logger.info("Dashboard stopped by user")
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
        finally:
            self.stop()
    
    def _main_loop(self, stdscr):
        """Main dashboard loop with curses."""
        self.screen = stdscr
        self.running = True
        
        # Configure curses
        curses.curs_set(0)  # Hide cursor
        stdscr.timeout(100)  # Non-blocking getch
        
        # Initialize colors if supported
        if curses.has_colors():
            curses.start_color()
            self._init_color_pairs()
        
        # Start data update task
        update_task = threading.Thread(target=self._update_data_loop, daemon=True)
        update_task.start()
        
        while self.running:
            try:
                # Clear screen
                stdscr.clear()
                
                # Draw dashboard
                self._draw_header(stdscr)
                self._draw_navigation(stdscr)
                self._draw_current_view(stdscr)
                self._draw_status_bar(stdscr)
                
                # Refresh screen
                stdscr.refresh()
                
                # Handle input
                key = stdscr.getch()
                self._handle_input(key)
                
                # Sleep to control refresh rate
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                break
    
    def _init_color_pairs(self):
        """Initialize color pairs for terminal."""
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)   # Success/Good
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)     # Error/Critical
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # Warning
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)    # Info/Headers
        curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK) # Highlights
        curses.init_pair(6, curses.COLOR_WHITE, curses.COLOR_BLACK)   # Normal text
    
    def _draw_header(self, stdscr):
        """Draw ASCII art header."""
        header_lines = ASCII_ART["header"].split('\n')
        
        for i, line in enumerate(header_lines):
            if i < curses.LINES - 1:
                stdscr.addstr(i, 0, line[:curses.COLS-1], curses.color_pair(4))
        
        # Add timestamp line with proper spacing
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status_line = f"AI AGENT PLATFORM DASHBOARD v1.0 | {timestamp}"
        
        # Position the status line below the header with spacing
        status_y = len(header_lines) + 1
        if status_y < curses.LINES:
            stdscr.addstr(status_y, 0, status_line[:curses.COLS-1], curses.color_pair(6))
    
    def _draw_navigation(self, stdscr):
        """Draw navigation menu."""
        nav_y = 12  # Increased spacing below header
        nav_items = [
            ("1", "Overview", "overview"),
            ("2", "Agents", "agents"), 
            ("3", "Costs", "costs"),
            ("4", "Events", "events"),
            ("5", "Logs", "logs"),
            ("Q", "Quit", "quit")
        ]
        
        nav_line = "  ".join([
            f"[{key}] {name}" + ("*" if view == self.current_view else "")
            for key, name, view in nav_items
        ])
        
        if nav_y < curses.LINES - 1:
            stdscr.addstr(nav_y, 0, nav_line[:curses.COLS-1], curses.color_pair(5))
    
    def _draw_current_view(self, stdscr):
        """Draw the current dashboard view."""
        start_y = 14  # Adjusted for new navigation position
        max_lines = curses.LINES - start_y - 3
        
        if self.current_view == "overview":
            self._draw_overview(stdscr, start_y, max_lines)
        elif self.current_view == "agents":
            self._draw_agents_view(stdscr, start_y, max_lines)
        elif self.current_view == "costs":
            self._draw_costs_view(stdscr, start_y, max_lines)
        elif self.current_view == "events":
            self._draw_events_view(stdscr, start_y, max_lines)
        elif self.current_view == "logs":
            self._draw_logs_view(stdscr, start_y, max_lines)
    
    def _draw_overview(self, stdscr, start_y, max_lines):
        """Draw system overview."""
        data = self.data_cache.get("overview", {})
        
        lines = [
            "╔════════════════════════════════════════════════════════════════════╗",
            "║                           SYSTEM OVERVIEW                         ║",
            "╠════════════════════════════════════════════════════════════════════╣",
        ]
        
        # System Health
        health_score = data.get("system_health", {}).get("overall_score", 0.0)
        health_status = self._get_status_indicator(health_score)
        lines.append(f"║ System Health:     {health_status} {health_score:.1%}                           ║")
        
        # Active Agents
        agents_data = data.get("agent_ecosystem", {})
        active_count = agents_data.get("active_count", 0)
        total_configs = agents_data.get("total_configurations", 0)
        lines.append(f"║ Active Agents:     {active_count:3d} / {total_configs:4d}                     ║")
        
        # Cost Efficiency
        cost_efficiency = data.get("system_health", {}).get("cost_efficiency", 0.0)
        cost_status = self._get_status_indicator(cost_efficiency)
        lines.append(f"║ Cost Efficiency:   {cost_status} {cost_efficiency:.1%}                           ║")
        
        # Recent Improvements
        improvements = data.get("improvement_status", {})
        roi = improvements.get("roi_last_30_days", 0.0)
        lines.append(f"║ ROI (30 days):     {roi:.2f}x                                  ║")
        
        lines.extend([
            "╠════════════════════════════════════════════════════════════════════╣",
            "║                            RECENT ACTIVITY                        ║",
            "╚════════════════════════════════════════════════════════════════════╝"
        ])
        
        # Recent events
        events = data.get("recent_events", [])
        for event in events[:5]:  # Show last 5 events
            timestamp = event.get("timestamp", "")[:19]  # Remove microseconds
            event_type = event.get("type", "unknown")
            lines.append(f"  {timestamp} - {event_type}")
        
        # Draw lines
        for i, line in enumerate(lines[:max_lines]):
            if start_y + i < curses.LINES - 1:
                stdscr.addstr(start_y + i, 0, line[:curses.COLS-1], curses.color_pair(6))
    
    def _draw_agents_view(self, stdscr, start_y, max_lines):
        """Draw agents monitoring view."""
        data = self.data_cache.get("agents", {})
        
        lines = [
            "╔════════════════════════════════════════════════════════════════════╗",
            "║                         AGENT MONITORING                          ║",
            "╠════════════════════════════════════════════════════════════════════╣",
            "║ ID               │ Type        │ Status │ Success │ Cost/Req      ║",
            "╠════════════════════════════════════════════════════════════════════╣"
        ]
        
        agents = data.get("active_agents", [])
        for agent in agents[:max_lines-6]:
            agent_id = agent.get("agent_id", "unknown")[:15]
            agent_type = agent.get("specialty", "general")[:10]
            status = "ACTIVE" if agent.get("is_active") else "IDLE"
            success_rate = agent.get("success_rate", 0.0)
            cost_per_req = agent.get("avg_cost_per_request", 0.0)
            
            status_color = curses.color_pair(1) if status == "ACTIVE" else curses.color_pair(6)
            
            line = f"║ {agent_id:<15} │ {agent_type:<10} │ {status:<6} │ {success_rate:>6.1%} │ ${cost_per_req:>10.4f} ║"
            lines.append(line)
        
        lines.append("╚════════════════════════════════════════════════════════════════════╝")
        
        # Draw lines
        for i, line in enumerate(lines[:max_lines]):
            if start_y + i < curses.LINES - 1:
                color = curses.color_pair(6)
                if "ACTIVE" in line:
                    color = curses.color_pair(1)
                stdscr.addstr(start_y + i, 0, line[:curses.COLS-1], color)
    
    def _draw_costs_view(self, stdscr, start_y, max_lines):
        """Draw cost analytics view."""
        data = self.data_cache.get("costs", {})
        
        lines = [
            "╔════════════════════════════════════════════════════════════════════╗",
            "║                         COST ANALYTICS                            ║",
            "╠════════════════════════════════════════════════════════════════════╣"
        ]
        
        # Cost summary
        daily_cost = data.get("daily_cost", 0.0)
        monthly_projection = daily_cost * 30
        efficiency_score = data.get("efficiency_score", 0.0)
        
        lines.extend([
            f"║ Today's Cost:      ${daily_cost:>8.2f}                              ║",
            f"║ Monthly Proj:      ${monthly_projection:>8.2f}                              ║",
            f"║ Efficiency:        {efficiency_score:>7.1%}                               ║",
            "╠════════════════════════════════════════════════════════════════════╣",
            "║                      OPTIMIZATION OPPORTUNITIES                   ║",
            "╠════════════════════════════════════════════════════════════════════╣"
        ])
        
        # Cost optimizations
        optimizations = data.get("optimizations", [])
        for opt in optimizations[:5]:
            title = opt.get("title", "Unknown")[:40]
            savings = opt.get("potential_savings", 0.0)
            lines.append(f"║ {title:<40} | ${savings:>6.2f} ║")
        
        lines.append("╚════════════════════════════════════════════════════════════════════╝")
        
        # Draw lines
        for i, line in enumerate(lines[:max_lines]):
            if start_y + i < curses.LINES - 1:
                stdscr.addstr(start_y + i, 0, line[:curses.COLS-1], curses.color_pair(6))
    
    def _draw_events_view(self, stdscr, start_y, max_lines):
        """Draw events stream view."""
        data = self.data_cache.get("events", {})
        
        lines = [
            "╔════════════════════════════════════════════════════════════════════╗",
            "║                          EVENT STREAM                             ║",
            "╠════════════════════════════════════════════════════════════════════╣"
        ]
        
        events = data.get("recent_events", [])
        for event in events[:max_lines-4]:
            timestamp = event.get("timestamp", "")[:19]
            event_type = event.get("type", "unknown")
            source = event.get("source", "system")
            
            # Color code by event type
            color = curses.color_pair(6)
            if "error" in event_type.lower() or "failed" in event_type.lower():
                color = curses.color_pair(2)
            elif "success" in event_type.lower() or "completed" in event_type.lower():
                color = curses.color_pair(1)
            elif "warning" in event_type.lower():
                color = curses.color_pair(3)
            
            line = f"║ {timestamp} │ {event_type:<20} │ {source:<10} ║"
            lines.append((line, color))
        
        lines.append(("╚════════════════════════════════════════════════════════════════════╝", curses.color_pair(6)))
        
        # Draw lines
        for i, line_data in enumerate(lines[:max_lines]):
            if start_y + i < curses.LINES - 1:
                if isinstance(line_data, tuple):
                    line, color = line_data
                else:
                    line, color = line_data, curses.color_pair(6)
                stdscr.addstr(start_y + i, 0, line[:curses.COLS-1], color)
    
    def _draw_logs_view(self, stdscr, start_y, max_lines):
        """Draw real-time logs view."""
        data = self.data_cache.get("logs", {})
        
        lines = [
            "╔════════════════════════════════════════════════════════════════════╗",
            "║                         REAL-TIME LOGS                            ║",
            "╠════════════════════════════════════════════════════════════════════╣"
        ]
        
        logs = data.get("recent_logs", [])
        for log in logs[:max_lines-4]:
            timestamp = log.get("timestamp", "")[:19]
            level = log.get("level", "INFO")
            message = log.get("message", "")[:45]
            
            # Color code by log level
            color = curses.color_pair(6)
            if level == "ERROR":
                color = curses.color_pair(2)
            elif level == "WARNING":
                color = curses.color_pair(3)
            elif level == "INFO":
                color = curses.color_pair(1)
            
            line = f"║ {timestamp} [{level:>5}] {message:<45} ║"
            lines.append((line, color))
        
        lines.append(("╚════════════════════════════════════════════════════════════════════╝", curses.color_pair(6)))
        
        # Draw lines
        for i, line_data in enumerate(lines[:max_lines]):
            if start_y + i < curses.LINES - 1:
                if isinstance(line_data, tuple):
                    line, color = line_data
                else:
                    line, color = line_data, curses.color_pair(6)
                stdscr.addstr(start_y + i, 0, line[:curses.COLS-1], color)
    
    def _draw_status_bar(self, stdscr):
        """Draw bottom status bar."""
        status_y = curses.LINES - 2
        
        refresh_time = datetime.now().strftime("%H:%M:%S")
        update_status = "LIVE" if self.updater.is_running() else "PAUSED"
        
        status_line = f"Last Update: {refresh_time} | Status: {update_status} | Press 'h' for help"
        
        stdscr.addstr(status_y, 0, status_line[:curses.COLS-1], curses.color_pair(4))
    
    def _handle_input(self, key):
        """Handle keyboard input."""
        if key == ord('q') or key == ord('Q'):
            self.running = False
        elif key == ord('1'):
            self.current_view = "overview"
        elif key == ord('2'):
            self.current_view = "agents"
        elif key == ord('3'):
            self.current_view = "costs"
        elif key == ord('4'):
            self.current_view = "events"
        elif key == ord('5'):
            self.current_view = "logs"
        elif key == ord('r') or key == ord('R'):
            # Force refresh
            asyncio.create_task(self._refresh_data())
    
    def _get_status_indicator(self, value: float) -> str:
        """Get color-coded status indicator."""
        if value >= 0.8:
            return "●"  # Green - Good
        elif value >= 0.6:
            return "◐"  # Yellow - Warning  
        else:
            return "●"  # Red - Critical
    
    def _update_data_loop(self):
        """Background thread for updating data."""
        while self.running:
            try:
                # Update data cache
                asyncio.run(self._refresh_data())
                time.sleep(self.refresh_rate)
            except Exception as e:
                logger.error(f"Error updating data: {e}")
                time.sleep(5)
    
    async def _refresh_data(self):
        """Refresh all dashboard data."""
        try:
            # Get data from updater
            if self.updater:
                self.data_cache = await self.updater.get_all_data()
            
        except Exception as e:
            logger.error(f"Error refreshing data: {e}")
    
    async def _simple_dashboard(self):
        """Simple text-based dashboard fallback."""
        logger.info("Starting simple dashboard (curses not available)")
        
        while True:
            try:
                # Clear screen
                os.system('clear' if os.name == 'posix' else 'cls')
                
                # Print header
                print(ASCII_ART["simple_header"])
                print("="*70)
                
                # Get data
                data = await self.updater.get_all_data() if self.updater else {}
                
                # Print system overview
                health = data.get("overview", {}).get("system_health", {})
                print(f"System Health: {health.get('overall_score', 0.0):.1%}")
                
                agents = data.get("overview", {}).get("agent_ecosystem", {})
                print(f"Active Agents: {agents.get('active_count', 0)}")
                
                print("\nPress Ctrl+C to exit...")
                
                await asyncio.sleep(self.refresh_rate)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Simple dashboard error: {e}")
                await asyncio.sleep(5)
    
    def stop(self):
        """Stop the dashboard."""
        self.running = False
        if self.updater:
            self.updater.stop()
    
    async def get_api_data(self) -> Dict[str, Any]:
        """Get dashboard data in API format for AI consumption."""
        return await self.updater.get_all_data() if self.updater else {} 