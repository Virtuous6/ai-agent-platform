"""
Data Formatter for Dashboard

Utilities for formatting data for display in the terminal dashboard.
Handles metrics formatting, table creation, and data visualization.
"""

from datetime import datetime, timedelta
from typing import Any, List, Dict, Optional, Union
import json

class DataFormatter:
    """Utilities for formatting dashboard data."""
    
    def __init__(self):
        """Initialize the data formatter."""
        pass
    
    def format_percentage(self, value: float, decimals: int = 1) -> str:
        """Format a decimal as percentage."""
        return f"{value:.{decimals}%}"
    
    def format_currency(self, amount: float, decimals: int = 2) -> str:
        """Format currency values."""
        return f"${amount:.{decimals}f}"
    
    def format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 1:
            return f"{seconds*1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    def format_bytes(self, bytes_value: int) -> str:
        """Format byte values."""
        if bytes_value < 1024:
            return f"{bytes_value}B"
        elif bytes_value < 1024 * 1024:
            return f"{bytes_value/1024:.1f}KB"
        elif bytes_value < 1024 * 1024 * 1024:
            return f"{bytes_value/(1024*1024):.1f}MB"
        else:
            return f"{bytes_value/(1024*1024*1024):.1f}GB"
    
    def format_number(self, value: Union[int, float], compact: bool = False) -> str:
        """Format numbers with optional compact notation."""
        if compact:
            if abs(value) >= 1_000_000:
                return f"{value/1_000_000:.1f}M"
            elif abs(value) >= 1_000:
                return f"{value/1_000:.1f}K"
        
        if isinstance(value, float):
            return f"{value:.2f}"
        else:
            return str(value)
    
    def format_timestamp(self, timestamp: str, format_type: str = "short") -> str:
        """Format timestamps for display."""
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            if format_type == "short":
                return dt.strftime("%H:%M:%S")
            elif format_type == "date":
                return dt.strftime("%Y-%m-%d")
            elif format_type == "full":
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                return dt.strftime("%H:%M:%S")
                
        except (ValueError, AttributeError):
            return timestamp[:19] if len(timestamp) >= 19 else timestamp
    
    def format_relative_time(self, timestamp: str) -> str:
        """Format timestamp as relative time (e.g., '2 minutes ago')."""
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            now = datetime.utcnow()
            diff = now - dt.replace(tzinfo=None)
            
            if diff.total_seconds() < 60:
                return f"{int(diff.total_seconds())}s ago"
            elif diff.total_seconds() < 3600:
                return f"{int(diff.total_seconds()/60)}m ago"
            elif diff.total_seconds() < 86400:
                return f"{int(diff.total_seconds()/3600)}h ago"
            else:
                return f"{int(diff.total_seconds()/86400)}d ago"
                
        except (ValueError, AttributeError):
            return "unknown"
    
    def create_progress_bar(self, value: float, width: int = 20, filled_char: str = "█", empty_char: str = "░") -> str:
        """Create a text progress bar."""
        filled = int(value * width)
        empty = width - filled
        return f"[{filled_char * filled}{empty_char * empty}]"
    
    def create_sparkline(self, values: List[float], width: int = 10) -> str:
        """Create a sparkline chart."""
        if not values or len(values) < 2:
            return "No data"
        
        # Normalize values to 0-7 range for spark characters
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val if max_val != min_val else 1
        
        spark_chars = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
        
        sparkline = ""
        for value in values[-width:]:  # Take last 'width' values
            normalized = (value - min_val) / range_val
            char_index = min(int(normalized * (len(spark_chars) - 1)), len(spark_chars) - 1)
            sparkline += spark_chars[char_index]
        
        return sparkline
    
    def format_table_row(self, columns: List[str], widths: List[int], align: str = "left") -> str:
        """Format a table row with specified column widths."""
        formatted_cols = []
        
        for i, (col, width) in enumerate(zip(columns, widths)):
            col_str = str(col)[:width]  # Truncate if too long
            
            if align == "right":
                formatted_cols.append(col_str.rjust(width))
            elif align == "center":
                formatted_cols.append(col_str.center(width))
            else:  # left
                formatted_cols.append(col_str.ljust(width))
        
        return " │ ".join(formatted_cols)
    
    def format_status_indicator(self, value: float, thresholds: Dict[str, float] = None) -> str:
        """Format status indicator based on value."""
        if thresholds is None:
            thresholds = {"excellent": 0.9, "good": 0.7, "warning": 0.5, "critical": 0.1}
        
        if value >= thresholds["excellent"]:
            return "●●●●●"
        elif value >= thresholds["good"]:
            return "●●●●○"
        elif value >= thresholds["warning"]:
            return "●●●○○"
        elif value >= thresholds["critical"]:
            return "●●○○○"
        else:
            return "●○○○○"
    
    def format_trend_indicator(self, values: List[float]) -> str:
        """Format trend indicator based on recent values."""
        if len(values) < 2:
            return "→"
        
        recent = values[-3:] if len(values) >= 3 else values
        
        if len(recent) == 1:
            return "→"
        
        # Calculate trend
        first = recent[0]
        last = recent[-1]
        
        if last > first * 1.1:  # 10% increase
            return "↗"
        elif last < first * 0.9:  # 10% decrease
            return "↘"
        else:
            return "→"
    
    def format_health_summary(self, health_data: Dict[str, Any]) -> Dict[str, str]:
        """Format system health data for display."""
        formatted = {}
        
        for key, value in health_data.items():
            if isinstance(value, float) and 0 <= value <= 1:
                # Assume it's a score between 0 and 1
                formatted[key] = f"{value:.1%}"
            elif isinstance(value, (int, float)):
                formatted[key] = self.format_number(value)
            else:
                formatted[key] = str(value)
        
        return formatted
    
    def format_cost_summary(self, cost_data: Dict[str, Any]) -> Dict[str, str]:
        """Format cost data for display."""
        formatted = {}
        
        for key, value in cost_data.items():
            if "cost" in key.lower() or "savings" in key.lower() or key == "daily_cost":
                if isinstance(value, (int, float)):
                    formatted[key] = self.format_currency(value)
                else:
                    formatted[key] = str(value)
            elif "efficiency" in key.lower() or "rate" in key.lower():
                if isinstance(value, float) and 0 <= value <= 1:
                    formatted[key] = self.format_percentage(value)
                else:
                    formatted[key] = str(value)
            else:
                formatted[key] = str(value)
        
        return formatted
    
    def format_agent_summary(self, agent_data: Dict[str, Any]) -> Dict[str, str]:
        """Format agent data for display."""
        formatted = {}
        
        for key, value in agent_data.items():
            if "rate" in key.lower() and isinstance(value, float) and 0 <= value <= 1:
                formatted[key] = self.format_percentage(value)
            elif "cost" in key.lower() and isinstance(value, (int, float)):
                formatted[key] = self.format_currency(value)
            elif "time" in key.lower() and isinstance(value, (int, float)):
                formatted[key] = self.format_duration(value)
            elif isinstance(value, (int, float)):
                formatted[key] = self.format_number(value)
            else:
                formatted[key] = str(value)
        
        return formatted
    
    def truncate_text(self, text: str, max_length: int, suffix: str = "...") -> str:
        """Truncate text to fit within specified length."""
        if len(text) <= max_length:
            return text
        
        if len(suffix) >= max_length:
            return text[:max_length]
        
        return text[:max_length - len(suffix)] + suffix
    
    def format_json_compact(self, data: Any, max_length: int = 50) -> str:
        """Format JSON data in a compact way for display."""
        try:
            json_str = json.dumps(data, separators=(',', ':'))
            return self.truncate_text(json_str, max_length)
        except (TypeError, ValueError):
            return self.truncate_text(str(data), max_length)
    
    def format_list_compact(self, items: List[Any], max_items: int = 3, max_length: int = 50) -> str:
        """Format a list in a compact way for display."""
        if not items:
            return "[]"
        
        # Take first max_items
        display_items = items[:max_items]
        item_strs = [str(item) for item in display_items]
        
        # Add ellipsis if there are more items
        if len(items) > max_items:
            item_strs.append(f"... +{len(items) - max_items}")
        
        result = ", ".join(item_strs)
        return self.truncate_text(result, max_length)
    
    def create_ascii_table(self, headers: List[str], rows: List[List[str]], max_width: int = 70) -> List[str]:
        """Create ASCII table with borders."""
        if not headers or not rows:
            return []
        
        # Calculate column widths
        col_count = len(headers)
        col_widths = [len(header) for header in headers]
        
        for row in rows:
            for i, cell in enumerate(row[:col_count]):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Adjust widths to fit max_width
        total_width = sum(col_widths) + (col_count - 1) * 3 + 4  # borders and separators
        if total_width > max_width:
            # Proportionally reduce column widths
            scale_factor = (max_width - (col_count - 1) * 3 - 4) / sum(col_widths)
            col_widths = [max(8, int(w * scale_factor)) for w in col_widths]
        
        lines = []
        
        # Top border
        top_border = "╔" + "╦".join("═" * w for w in col_widths) + "╗"
        lines.append(top_border)
        
        # Header
        header_row = "║" + "║".join(h.center(w) for h, w in zip(headers, col_widths)) + "║"
        lines.append(header_row)
        
        # Header separator
        header_sep = "╠" + "╬".join("═" * w for w in col_widths) + "╣"
        lines.append(header_sep)
        
        # Data rows
        for row in rows:
            row_cells = []
            for i, (cell, width) in enumerate(zip(row[:col_count], col_widths)):
                cell_str = str(cell)
                if len(cell_str) > width:
                    cell_str = cell_str[:width-3] + "..."
                row_cells.append(cell_str.ljust(width))
            
            data_row = "║" + "║".join(row_cells) + "║"
            lines.append(data_row)
        
        # Bottom border
        bottom_border = "╚" + "╩".join("═" * w for w in col_widths) + "╝"
        lines.append(bottom_border)
        
        return lines 